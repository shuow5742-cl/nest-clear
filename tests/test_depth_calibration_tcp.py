"""160 多点会话，通过本机 Modbus TCP 寄存器驱动物理模型。"""

import csv
from dataclasses import replace
import json
import math
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
        frame = cv2.GaussianBlur(image, (0, 0), math.hypot(.4, 1.5 * (z - self.focus(label))))
        self.cached = (self.actual, self.exposure_us, frame)
        return frame, {'exposure_us': self.exposure_us}


class ScriptedPlcWorker(PlcWorker):
    def __init__(self, handshake, plc_sim):
        super().__init__(handshake)
        self.plc_sim = plc_sim
        self.on_actual_read = None

    def execute(self, payload):
        if payload['operation'] == 'actual' and self.on_actual_read is not None:
            self.plc_sim.set_actual_xyz(self.on_actual_read())
        return super().execute(payload)


@pytest.fixture
def tcp_rig(tmp_path):
    camera = FilmCamera()
    cfg = load_config('config/nest_clear.yaml')
    settings = {**cfg.depth_calibration, 'camera_z_min_mm': -7.0,
                'coarse_step_mm': .50, 'overshoot_mm': .15,
                'settle_window_s': .0001, 'arrival_timeout_s': .05, 'frames_per_step': 1,
                'travel': ((80, 220), (0, 230), (-15, 0))}  # 仿真场景需要更宽的下针深度标定行程
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
        plc_worker = ScriptedPlcWorker(Handshake(io, plc.contract, events=events), plc)
        for worker in (plc_worker, camera):
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
                                  batch=batch, writes=writes, root=tmp_path, plc=plc, plc_worker=plc_worker)
        finally:
            clock.shutdown()
            io.close()


def test_160_arrival_waits_for_old_pose_and_002mm_fine_step(tcp_rig):
    """测试 1 & 2：旧位置/中途值滑动窗口等待；0.02mm 细扫步移动期间不得提前接受。"""
    rig = tcp_rig
    rig.cfg.depth_calibration['arrival_timeout_s'] = 1.0

    # 第 1 次 160 居中完成，拿到下一个预期目标 target1
    code, target1 = rig.exchange()
    assert code == 160
    session = rig.batch.blackboard.read('depth_session')

    # 测试 1：先停在旧位置/中途位置 6 次，再到达 target1 并稳定 6 次
    old_pose = (target1[0] + 0.8, target1[1] - 0.6, target1[2] + 0.5)
    mid_pose = (target1[0] + 0.1, target1[1] - 0.1, target1[2] + 0.1)
    seq1 = [old_pose] * 6 + [mid_pose] * 2 + [target1] * 10
    idx1 = [0]

    def read_seq1():
        i = min(idx1[0], len(seq1) - 1)
        idx1[0] += 1
        return seq1[i]

    rig.plc_worker.on_actual_read = read_seq1
    code, target2 = rig.exchange(target1)
    assert code == 160
    assert session.actual == pytest.approx(target1, abs=1e-4)
    assert session.all_arrivals[-1]['status'] == 'ok'
    assert session.all_arrivals[-1]['samples'] >= 14
    assert session.arrival_summary['total_anomalies'] == 0

    # 测试 2（回归）：模拟细扫 0.02mm 步距期间，前 8 个采样点每步仅走 0.002mm（窗口极差 0.010 <= 0.02mm），
    # 但尚未到位（均值偏差 0.011~0.015 > 0.01mm），不得提前接受；到达目标并稳定后才接受。
    step_target = (target2[0], target2[1], target2[2] - 0.02)
    z0 = target2[2]
    moving_window = [(step_target[0], step_target[1], z0 - 0.002 * k) for k in range(8)]
    settled_window = [step_target] * 10
    seq2 = moving_window + settled_window
    idx2 = [0]

    def read_seq2():
        i = min(idx2[0], len(seq2) - 1)
        idx2[0] += 1
        return seq2[i]

    rig.plc_worker.on_actual_read = read_seq2
    # 手动把 expected 设成细扫下行 0.02mm 目标，验证不会在移动期间提前接受
    session.expected = step_target
    code, _ = rig.exchange(step_target)
    assert code == 160
    last = session.all_arrivals[-1]
    assert last['status'] == 'ok'
    assert last['samples'] >= 9
    assert abs(last['actual'][2] - step_target[2]) <= 0.01
    assert session.arrival_summary['total_anomalies'] == 0
    rig.plc_worker.on_actual_read = None


def test_160_arrival_offset_003mm_records_and_completes_161(tcp_rig):
    """测试 3：稳定但偏 +0.03mm → 超时后继续并最终回 161；arrivals.csv / 事件 / arrival_summary 均记录。"""
    from nest_clear.ui.model import classify_event

    rig = tcp_rig
    rig.cfg.depth_calibration['arrival_timeout_s'] = 0.01
    target_label = rig.camera.primary[0]
    target = None
    session = None
    injected = [False]

    for _ in range(1200):
        rig.plc_worker.on_actual_read = None
        actual_to_send = target
        if session is not None and session.label == target_label and session.phase == 'peak' and target is not None and not injected[0]:
            # 在首点峰位留档时制造稳定但 Z 偏 +0.03 mm 的到位超差
            biased = (target[0], target[1], target[2] + 0.03)
            actual_to_send = biased
            injected[0] = True
            rig.plc_worker.on_actual_read = lambda b=biased: b
        code, target = rig.exchange(actual_to_send)
        if code == 161:
            break
        assert code == 160
        session = rig.batch.blackboard.read('depth_session')
    else:
        pytest.fail('未完成多点会话')

    assert injected[0]
    assert session.feasibility['met']
    session_dir = session.record.directory

    # 1) arrivals.csv 校验
    with (session_dir / 'arrivals.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(session.all_arrivals) > 0
    warn_rows = [r for r in rows if r['label'] == target_label and r['phase'] == 'peak']
    assert len(warn_rows) == 1
    wrow = warn_rows[0]
    assert wrow['status'] == '超差'
    assert wrow['max_delta_axis'] == 'Z'
    assert float(wrow['max_abs_delta_mm']) == pytest.approx(0.03, abs=1e-4)

    # 2) events.jsonl 校验 depthcal_arrival_warning
    event_rows = [
        json.loads(line)
        for line in (rig.events.run_directory / 'events.jsonl').read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    warn_events = [e for e in event_rows if e.get('event') == 'depthcal_arrival_warning']
    assert len(warn_events) >= 1
    we = next(e for e in warn_events if e.get('label') == target_label and e.get('phase') == 'peak')
    assert we['type'] == '超差'
    assert we['delta'][2] == pytest.approx(0.03, abs=1e-4)
    assert classify_event(we)['color'] == 'yellow'

    # 3) fits.json（results）逐点 arrival 与顶层 arrival_summary 校验
    fits_doc = json.loads((session_dir / 'fits.json').read_text(encoding='utf-8'))
    target_pt = next(p for p in fits_doc['points'] if p['label'] == target_label)
    assert target_pt['arrival']['anomalies'] >= 1
    assert 'peak' in target_pt['arrival']['phases']
    assert target_pt['arrival']['max_delta_axis'] == 'Z'
    assert target_pt['arrival']['max_delta_mm'] == pytest.approx(0.03, abs=1e-4)

    summary = fits_doc['arrival_summary']
    assert target_label in summary['anomaly_points']
    assert summary['total_anomalies'] >= 1
    assert summary['avg_elapsed_s'] >= 0 and summary['max_elapsed_s'] >= 0


def test_160_arrival_always_jittering_continues_as_unstable(tcp_rig):
    """测试 4：一直抖动不稳 → 超时后继续，记“未稳定”/“超差+未稳定”，不回 999。"""
    rig = tcp_rig
    rig.cfg.depth_calibration['arrival_timeout_s'] = 0.015

    # 4a) 首次 160（尚无 expected）一直抖动 ±0.03mm -> 不回 999，记“未稳定”
    base0 = rig.camera.actual
    tick0 = [0]

    def jitter0():
        tick0[0] += 1
        d = 0.03 if tick0[0] % 2 else -0.03
        return (base0[0], base0[1], base0[2] + d)

    rig.plc_worker.on_actual_read = jitter0
    code, target1 = rig.exchange()
    assert code == 160
    session = rig.batch.blackboard.read('depth_session')
    assert session.all_arrivals[0]['status'] == '未稳定'
    assert session.all_arrivals[0]['unstable'] is True

    # 4b) 有 expected 后一直围绕目标抖动 ±0.03mm -> 记“未稳定”（若偏 0.03mm+抖动则“超差+未稳定”），仍回 160
    tick1 = [0]

    def jitter_around_target():
        tick1[0] += 1
        d = 0.03 if tick1[0] % 2 else -0.03
        return (target1[0], target1[1], target1[2] + d)

    rig.plc_worker.on_actual_read = jitter_around_target
    code, target2 = rig.exchange(target1)
    assert code == 160
    assert session.all_arrivals[1]['status'] == '未稳定'

    tick2 = [0]

    def jitter_and_offset():
        tick2[0] += 1
        d = 0.03 if tick2[0] % 2 else -0.03
        return (target2[0], target2[1], target2[2] + 0.03 + d)

    rig.plc_worker.on_actual_read = jitter_and_offset
    code, _ = rig.exchange(target2)
    assert code == 160
    assert session.all_arrivals[2]['status'] == '超差+未稳定'
    assert session.arrival_summary['total_anomalies'] == 3
    assert session.arrival_summary['unstable_count'] == 3
    assert session.arrival_summary['over_tolerance_count'] == 1
    rig.plc_worker.on_actual_read = None


@pytest.mark.parametrize('anomaly', [False, True])
def test_160_multisite(tcp_rig, anomaly):
    rig = tcp_rig
    rig.camera.anomaly = anomaly
    coordinates = None
    session = None
    for _ in range(1800):
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
    assert session.results[0]['fine_plan']['mode'] == 'auto'
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
        reader = csv.DictReader(f)
        assert tuple(reader.fieldnames) == ('label', 'pass', 'direction', 'z_cmd', 'z_actual', 'tenengrad', 'laplacian', 'brenner')
        rows = list(reader)
    assert len(rows) == len(session.samples) > 0
    assert all(r['direction'] == 'down' for r in rows)
    assert all(r[m] for r in rows for m in ('tenengrad', 'laplacian', 'brenner'))
    assert len(list((session_dir / 'roi').glob('*.png'))) == len(rows)
    for result in session.results:
        assert (session_dir / f"{result['label']}_center.png").is_file()
        assert (session_dir / f"{result['label']}_peak.png").is_file()
    fits_doc = json.loads((session_dir / 'fits.json').read_text(encoding='utf-8'))
    assert fits_doc['dof_ratio'] == pytest.approx(0.8)
    assert len(fits_doc['points']) == len(session.results)
    assert fits_doc['feasibility']['met']
    for pt in fits_doc['points']:
        assert pt['dof_mm'] > 0 and pt['dof_range_mm'] >= 0 and pt['dof_uncovered'] == 0
        assert pt['coarse_fit']['dof_ratio'] == pytest.approx(0.8)
        for fit_item in pt['fits']:
            assert fit_item['dof_covered'] is True
            assert fit_item['dof_ratio'] == pytest.approx(0.8)
            assert fit_item['dof_mm'] > 0 and fit_item['dof_gauss_mm'] > 0
            assert fit_item['dof_z_lower'] < fit_item['z_peak'] < fit_item['dof_z_upper']
    plane_doc = json.loads((session_dir / 'plane.json').read_text(encoding='utf-8'))
    assert plane_doc['final']['max_residual'] == pytest.approx(session.plane['max_residual'])
    rec_doc = json.loads((session_dir / 'session_record.json').read_text(encoding='utf-8'))
    assert rec_doc['outcome'] == 'done' and rec_doc['exposure_us'] == 6000.0
    assert rec_doc['dof_summary'] == fits_doc['dof_summary']
    assert rec_doc['dof_summary']['dof_ratio'] == pytest.approx(0.8)
    assert rec_doc['dof_summary']['covered_points'] == len(session.results)
    assert rec_doc['dof_summary']['uncovered_fits'] == 0
    assert rec_doc['dof_summary']['min_mm'] <= rec_doc['dof_summary']['mean_mm'] <= rec_doc['dof_summary']['max_mm']
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
    ('abort', 'DEPTH-ABORTED'),
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
    elif case in ('below_limit', 'wrong_request', 'abort', 'exposure_change'):
        code, target = rig.exchange()
        assert code == 160
        if case == 'below_limit':
            kwargs = {'xyz': (*target[:2], rig.cfg.depth_calibration['camera_z_min_mm'] - 0.2)}
        elif case == 'wrong_request':
            kwargs = {'func': 100, 'xyz': target}
        elif case == 'abort':
            send(rig.root, 'depthcal_abort')
            kwargs = {'xyz': target}
        elif case == 'exposure_change':
            rig.camera.exposure_us = 8000.
            kwargs = {'xyz': target}
    elif case == 'boundary':
        rig.cfg.depth_calibration['camera_z_min_mm'] = -4.5
        rig.camera.actual = (*rig.camera.actual[:2], -4.2)
    elif isinstance(case, int):
        kwargs = {'func': case}
    for _ in range(30):
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


def test_160_independent_depth_travel_tcp(tmp_path):
    """全局行程外但深度行程内（X=40, Y=180）时 160 正常下发；超出深度行程则回 999 且诊断含轴与范围。"""
    camera = FilmCamera()
    # T07 相对中心为 (-45, +105)，origin=(85, 75) 使 T07 位于 (40, 180)
    camera.origin = np.array([85.0, 75.0])
    camera.actual = (40.0, 180.0, -5.0)
    cfg = load_config('config/nest_clear.yaml')
    cfg = replace(
        cfg,
        depth_calibration={**cfg.depth_calibration, 'settle_window_s': .0001, 'frames_per_step': 1},
        tool_offset={**cfg.tool_offset, 'card_thickness_mm': 0.},
        camera_calibration={**cfg.camera_calibration, 'board_thickness_mm': 0.},
    )
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
        clock.submit(batch)

        def send_160(xyz):
            camera.actual = tuple(xyz)
            plc.set_actual_xyz(camera.actual)
            task = plc.request(160, tool_id=65535)
            deadline = time.monotonic() + 6
            while batch.blackboard.read('committed_func') is None or events.request_pending or plc.io.read('result_id') != task:
                clock.tick_once()
                if time.monotonic() > deadline:
                    pytest.fail(f'未收到 Task_ID={task} 回应')
                time.sleep(.00015)
            code = plc.io.read('reply_func')
            coords = plc.io.read_real32_block(40942, 3) if code == 160 else None
            return code, coords

        try:
            # 1) (40, 180, -5) 在全局行程 (80..180, -10..150) 之外，但在 depth_calibration.travel (32..158.6, -15..201) 之内
            code, target = send_160((40.0, 180.0, -5.0))
            assert code == 160 and target is not None
            assert 32.0 <= target[0] <= 158.6 and -15.0 <= target[1] <= 201.0

            # 2) 超出 depth_calibration.travel 上限（X=165 > 158.6）→ 999 且诊断含轴、目标值、范围和行程来源
            code, _ = send_160((165.0, 180.0, -5.0))
            assert code == 999 and batch.blackboard.read('mode') == Mode.IDLE
            diag = events.status.data['depthcal']
            assert diag['reason_code'] == 'POSITION-TRAVEL'
            assert 'X' in diag['what'] and '165.000' in diag['what']
            assert '[32.0, 158.6]' in diag['what']
            assert '下针深度标定行程（depth_calibration.travel）' in diag['what']
        finally:
            clock.shutdown()
            io.close()
