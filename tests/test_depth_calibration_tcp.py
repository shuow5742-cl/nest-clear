"""160 多点会话，通过本机 Modbus TCP 寄存器驱动物理模型。"""

import csv
from dataclasses import replace
import json
from types import SimpleNamespace
import time

import cv2
import numpy as np
import pytest
from autoweaver.motion_policy.world_board import WorldBoard
from autoweaver.worker import AsyncRequestWorker
from autoweaver.worker.clock import BTClock

from nest_clear.comm import Handshake, PLCClient
from nest_clear.config import load_config
from nest_clear.depthcal.layout import film_layout
from nest_clear.flow.batch import build_batch
from nest_clear.flow.context import RequestTracker
from nest_clear.flow.modes import Mode
from nest_clear.paths import live_image_file
from nest_clear.record.events import EventRecorder
from nest_clear.tooloffset.commands import send
from nest_clear.vision.geometry import Calibration
from nest_clear.workers.plc import PlcWorker
from sim.fake_plc import FakePLC


class FilmCamera(AsyncRequestWorker):
    name = 'camera'

    def __init__(self):
        super().__init__()
        self.layout, self.primary, self.extra = film_layout()
        self.origin = np.array([150., 115.])
        self.actual = (*tuple(self.origin + self.layout[self.primary[0]] + [1., -1.]), -5.)
        self.blank = False
        self.anomaly = False
        self.offcenter = False
        self.exposure_us = 6000.
        self.cached = None
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.markers = {label: cv2.rotate(cv2.aruco.generateImageMarker(dictionary, int(label[1:]), 80),
                                         cv2.ROTATE_90_COUNTERCLOCKWISE) for label in self.layout}

    def focus(self, label):
        return -5.037 + .0005 * self.layout[label][0] + (.05 if self.anomaly and label == self.primary[-1] else 0.)

    def execute(self, payload):
        if self.blank:
            return np.full((420, 420), 220, np.uint8), {}
        if self.cached and self.cached[0] == self.actual and self.cached[1] == self.exposure_us:
            return self.cached[2], {'exposure_us': self.exposure_us}
        x, y, z = self.actual
        label = min(self.layout, key=lambda key: np.linalg.norm(self.origin + self.layout[key] - (x, y)))
        dx, dy = self.origin + self.layout[label] - (x, y)
        u, v = (260, 210) if self.offcenter else (int(round(210 + 20 * dx)), int(round(210 - 20 * dy)))
        image = np.full((420, 420), 225, np.uint8)
        cv2.circle(image, (u, v), 37, 18, 6, cv2.LINE_AA)
        cv2.line(image, (u - 90, v), (u + 90, v), 18, 2, cv2.LINE_AA)
        cv2.line(image, (u, v - 90), (u, v + 90), 18, 2, cv2.LINE_AA)
        if 0 <= u - 180 <= 340 and 0 <= v - 180 <= 340:
            image[v - 180:v - 100, u - 180:u - 100] = self.markers[label]
        frame = cv2.GaussianBlur(image, (0, 0), max(.001, 12 * abs(z - self.focus(label))))
        self.cached = (self.actual, self.exposure_us, frame)
        return frame, {'exposure_us': self.exposure_us}


@pytest.fixture
def tcp_rig(tmp_path):
    camera = FilmCamera()
    cfg = load_config('config/nest_clear.yaml')
    settings = {**cfg.depth_calibration, 'camera_z_min_mm': -6.3, 'coarse_range_mm': .4,
                'coarse_step_mm': .10, 'fine_range_mm': .18, 'fine_step_mm': .03, 'overshoot_mm': .1,
                'settle_window_s': .0001, 'frames_per_step': 1}
    cfg = replace(cfg, depth_calibration=settings, travel=((80, 220), (0, 230), (-15, 0)),
                  tool_offset={**cfg.tool_offset, 'card_thickness_mm': 0.},
                  camera_calibration={**cfg.camera_calibration, 'board_thickness_mm': 0.})
    cal = Calibration(np.array([[2000., 0., 210], [0., 2000., 210], [0., 0., 1.]]),
                      np.zeros(5), np.diag([1., -1., -1.]), np.zeros(3), -105)
    events = EventRecorder(tmp_path, run=True)
    with FakePLC() as plc:
        io = PLCClient('127.0.0.1', plc.port, 1, plc.contract)
        board = WorldBoard()
        clock = BTClock(board)
        for worker in (PlcWorker(Handshake(io, plc.contract, events=events)), camera):
            clock.attach_worker(worker)
        batch = build_batch(board, cfg, [], RequestTracker(0), events, calibration=cal)
        handle = clock.submit(batch)
        writes = []

        def exchange(xyz=None, func=160, tool_id=65535):
            if xyz is not None:
                camera.actual = tuple(xyz)
            plc.set_actual_xyz(camera.actual)
            task = plc.request(func, tool_id=tool_id)
            first_write = len(plc.writes)
            deadline = time.monotonic() + 6
            while batch.blackboard.read('committed_func') is None or events.request_pending or plc.io.read('result_id') != task:
                clock.tick_once()
                assert handle.result is None
                if time.monotonic() > deadline:
                    pytest.fail(f'未收到 Task_ID={task} 回应')
                time.sleep(.00015)
            code = plc.io.read('reply_func')
            writes.append(plc.writes[first_write:])
            addresses = [address for address, _ in writes[-1]]
            assert addresses == ([40942, 40934, 40933] if code == 160 else [40934, 40933])
            assert writes[-1][-1] == (40933, [task])
            return code, plc.io.read_real32_block(40942, 3) if code == 160 else None

        try:
            yield SimpleNamespace(exchange=exchange, camera=camera, cfg=cfg, events=events,
                                  batch=batch, writes=writes, root=tmp_path)
        finally:
            clock.shutdown()
            io.close()


@pytest.mark.parametrize('anomaly', [False, True])
def test_160_multisite(tcp_rig, anomaly):
    rig = tcp_rig
    rig.camera.anomaly = anomaly
    coordinates = None
    session = None
    for _ in range(600):
        code, coordinates = rig.exchange(coordinates)
        if code == 161:
            break
        assert code == 160, rig.events.status.data['depthcal']
        assert coordinates[2] >= rig.cfg.depth_calibration['camera_z_min_mm']
        session = rig.batch.blackboard.read('depth_session')
    else:
        pytest.fail('未完成多点会话')
    assert rig.batch.blackboard.read('mode') == Mode.IDLE
    assert len(session.results) == (9 if anomaly else 5)
    assert session.feasibility['met']
    assert len(session.requests) == len(session.replies) == len(rig.writes)
    assert session.replies[-1] == {'task_id': len(rig.writes), 'func': 161, 'coordinates': {}}
    for result in session.results:
        assert len(result['fits']) == 3
        assert abs(result['z_peak'] - rig.camera.focus(result['label'])) < .01
    if anomaly:
        assert session.initial_plane['max_residual'] > .02
        assert session.plane['after_extra'].startswith('加测后残差')
    else:
        assert session.plane['tilt_mm_per_100mm'] == pytest.approx(.05, abs=.004)
    # 核对 runs/<日期>/depthcal/<会话>/ 全部记录文件与曲线状态
    session_dir = session.record.directory
    assert session_dir.is_dir() and session_dir.parent.name == 'depthcal'
    with (session_dir / 'curves.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(session.samples) > 0
    assert all(r['direction'] == 'down' for r in rows)
    assert all(r[m] for r in rows for m in ('tenengrad', 'laplacian', 'brenner'))
    assert len(list((session_dir / 'roi').glob('*.png'))) == len(rows)
    for result in session.results:
        assert (session_dir / f"{result['label']}_center.png").is_file()
        assert (session_dir / f"{result['label']}_peak.png").is_file()
    fits_doc = json.loads((session_dir / 'fits.json').read_text(encoding='utf-8'))
    assert len(fits_doc['points']) == len(session.results)
    assert fits_doc['feasibility']['met']
    plane_doc = json.loads((session_dir / 'plane.json').read_text(encoding='utf-8'))
    assert plane_doc['final']['max_residual'] == pytest.approx(session.plane['max_residual'])
    rec_doc = json.loads((session_dir / 'session_record.json').read_text(encoding='utf-8'))
    assert rec_doc['outcome'] == 'done' and rec_doc['exposure_us'] == 6000.0
    assert len(rec_doc['requests']) == len(rec_doc['replies']) == len(rig.writes)
    plc_lines = [json.loads(line) for line in (session_dir / 'plc.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(plc_lines) == 2 * len(rig.writes)
    assert live_image_file(rig.root, 'depthcal_roi').is_file()
    status_depth = rig.events.status.data['depthcal']
    assert status_depth['phase'] == 'done' and len(status_depth['curves']) == len(session.results)
    print(f'anomaly={anomaly} requests={len(rig.writes)} tilt={session.plane["tilt_mm_per_100mm"]:.6f}'
          f' initial={session.initial_plane["max_residual"]:.6f} final={session.plane["max_residual"]:.6f}')


@pytest.mark.parametrize(('case', 'reason'), [
    ('missing_limit', 'DEPTH-Z-MIN'), ('blank', 'DEPTH-TARGET'), ('boundary', 'DEPTH-PEAK-BOUNDARY'),
    ('below_limit', 'DEPTH-Z-LIMIT'), ('wrong_request', 'DEPTH-MODE'),
    ('center_limit', 'DEPTH-CENTER'), ('exposure_change', 'DEPTH-EXPOSURE'),
    ('abort', 'DEPTH-ABORTED'), ('arrival_mismatch', 'DEPTH-ARRIVAL'),
    (170, 'DEPTH-UNIMPLEMENTED'), (171, 'DEPTH-UNIMPLEMENTED'),
    (180, 'DEPTH-UNIMPLEMENTED'), (181, 'DEPTH-UNIMPLEMENTED'),
    (190, 'DEPTH-UNIMPLEMENTED'), (191, 'DEPTH-UNIMPLEMENTED'),
])
def test_160_error(tcp_rig, case, reason):
    rig = tcp_rig
    kwargs = {}
    if case == 'missing_limit':
        rig.cfg.depth_calibration['camera_z_min_mm'] = None
    elif case == 'blank':
        rig.camera.blank = True
    elif case == 'center_limit':
        rig.camera.offcenter = True
    elif case in ('below_limit', 'wrong_request', 'arrival_mismatch', 'abort', 'exposure_change'):
        code, target = rig.exchange()
        assert code == 160
        if case == 'below_limit':
            kwargs = {'xyz': (*target[:2], -6.5)}
        elif case == 'wrong_request':
            kwargs = {'func': 100, 'xyz': target}
        elif case == 'arrival_mismatch':
            kwargs = {'xyz': (target[0] + 0.5, target[1], target[2])}
        elif case == 'abort':
            send(rig.root, 'depthcal_abort')
            kwargs = {'xyz': target}
        elif case == 'exposure_change':
            rig.camera.exposure_us = 8000.
            kwargs = {'xyz': target}
    elif case == 'boundary':
        rig.cfg.depth_calibration['coarse_range_mm'] = .025
        rig.cfg.depth_calibration['coarse_step_mm'] = .01
    elif isinstance(case, int):
        kwargs = {'func': case}
    for _ in range(15):
        code, target = rig.exchange(**kwargs)
        if code == 999:
            break
        assert case in ('boundary', 'center_limit') and code == 160
        kwargs = {'xyz': target}
    assert code == 999 and rig.batch.blackboard.read('mode') == Mode.IDLE
    assert rig.events.status.data['depthcal']['reason_code'] == reason
    if case not in ('missing_limit', 170, 171, 180, 181, 190, 191):
        rec_path, = (rig.root / rig.events.clock().date().isoformat() / 'depthcal').glob('*/session_record.json')
        rec = json.loads(rec_path.read_text(encoding='utf-8'))
        assert rec['outcome'] == 'failed' and rec['diagnostic']['reason_code'] == reason
