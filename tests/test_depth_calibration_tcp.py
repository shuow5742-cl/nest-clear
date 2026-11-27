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
        self.blank_labels = set()
        self.flat_labels = set()
        self.per_label_focus = {}
        self.anomaly = False
        self.offcenter = False
        self.exposure_us = 6000.
        self.exposure_auto = 0
        self.base_exposure_us = 6000.
        self.simulate_exposure = False
        self.background_level = 190.
        self.per_label_background = {}
        self.exposure_history = []
        self.fail_restore = False
        self.cached = None
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.markers = {label: cv2.rotate(cv2.aruco.generateImageMarker(dictionary, int(label[1:]), 80),
                                         cv2.ROTATE_90_COUNTERCLOCKWISE) for label in self.layout}

    def focus(self, label):
        if label in self.per_label_focus:
            return self.per_label_focus[label]
        return -5.037 + .0005 * self.layout[label][0] + (.05 if self.anomaly and label == self.primary[-1] else 0.)

    def execute(self, payload):
        op = payload.get('operation')
        if op == 'get_exposure_state':
            return None, {'exposure_us': self.exposure_us, 'exposure_auto': self.exposure_auto}
        if op == 'restore_exposure_state':
            if self.fail_restore:
                raise RuntimeError('相机通信断开，无法写回曝光')
            if payload.get('exposure_us') is not None:
                self.exposure_us = float(payload['exposure_us'])
            if payload.get('exposure_auto') is not None:
                self.exposure_auto = payload['exposure_auto']
            self.cached = None
            return None, {'exposure_us': self.exposure_us, 'exposure_auto': self.exposure_auto}
        if op == 'set_exposure':
            self.exposure_us = float(payload['exposure_us'])
            self.exposure_history.append(self.exposure_us)
            self.cached = None
            return None, {'exposure_us': self.exposure_us}
        if self.blank:
            return np.full((420, 420), 220, np.uint8), {}
        if self.cached and self.cached[0] == self.actual and self.cached[1] == self.exposure_us:
            return self.cached[2], {'exposure_us': self.exposure_us}
        x, y, z = self.actual
        label = min(self.layout, key=lambda key: np.linalg.norm(self.origin + self.layout[key] - (x, y)))
        if label in self.blank_labels:
            return np.full((420, 420), 220, np.uint8), {'exposure_us': self.exposure_us}
        dx, dy = self.origin + self.layout[label] - (x, y)
        u, v = (260, 210) if self.offcenter else (int(round(210 + 20 * dx)), int(round(210 - 20 * dy)))
        bg = float(self.per_label_background.get(label, self.background_level))
        scale = (self.exposure_us / self.base_exposure_us) if self.simulate_exposure else 1.0
        bg_val = int(np.clip(round(bg * scale), 0, 255))
        fg_val = int(np.clip(round(18. * scale), 0, 255))
        image = np.full((420, 420), bg_val, np.uint8)
        cv2.circle(image, (u, v), 37, fg_val, 6, cv2.LINE_AA)
        cv2.line(image, (u - 90, v), (u + 90, v), fg_val, 2, cv2.LINE_AA)
        cv2.line(image, (u, v - 90), (u, v + 90), fg_val, 2, cv2.LINE_AA)
        if 0 <= u - 180 <= 340 and 0 <= v - 180 <= 340:
            marker = np.where(self.markers[label] < 128, fg_val, bg_val).astype(np.uint8)
            image[v - 180:v - 100, u - 180:u - 100] = marker
        sigma = 0.6 if label in self.flat_labels else math.hypot(.4, 1.5 * (z - self.focus(label)))
        frame = cv2.GaussianBlur(image, (0, 0), sigma)
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
        if code in (161, 999):
            break
        assert code == 160, rig.events.status.data['depthcal']
        assert coordinates[2] >= rig.cfg.depth_calibration['camera_z_min_mm']
        session = rig.batch.blackboard.read('depth_session')
    else:
        pytest.fail('未完成多点会话')
    assert rig.batch.blackboard.read('mode') == Mode.IDLE
    assert len(session.results) == (9 if anomaly else 5)
    assert session.results[0]['fine_plan']['mode'] == 'auto'
    assert len(session.requests) == len(session.replies) == len(rig.writes)
    for result in session.results:
        assert len(result['fits']) == 3
        assert abs(result['z_peak'] - rig.camera.focus(result['label'])) < .01
    if anomaly:
        # 规则 d：加测至 9 点后平面最大残差仍 > plane_tol_mm 且已无可加测点 -> 回 999
        assert code == 999
        assert rig.events.status.data['depthcal']['reason_code'] == 'DEPTH-PLANE-RESIDUAL'
        assert '规则 d' in rig.events.status.data['depthcal']['where']
        assert session.initial_plane['max_residual'] > .02
        assert session.plane['after_extra'].startswith('加测后残差')
        assert session.replies[-1] == {'task_id': len(rig.writes), 'func': 999, 'coordinates': {}}
        return
    assert code == 161
    assert session.feasibility['met']
    assert session.replies[-1] == {'task_id': len(rig.writes), 'func': 161, 'coordinates': {}}
    assert session.plane['tilt_mm_per_100mm'] == pytest.approx(.05, abs=.004)
    # 核对 runs/<日期>/depthcal/<会话>/ 全部记录文件与曲线状态
    session_dir = session.record.directory
    assert session_dir.is_dir() and session_dir.parent.name == 'depthcal'
    with (session_dir / 'curves.csv').open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        assert tuple(reader.fieldnames) == (
            'label', 'pass', 'direction', 'z_cmd', 'z_actual',
            'tenengrad', 'laplacian', 'brenner', 'exposure_us', 'roi_mean', 'roi_p99',
        )
        rows = list(reader)
    assert len(rows) == len(session.samples) > 0
    assert all(r['direction'] == 'down' for r in rows)
    assert all(r[m] for r in rows for m in ('tenengrad', 'laplacian', 'brenner', 'exposure_us', 'roi_mean', 'roi_p99'))
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
    ('missing_limit', 'DEPTH-Z-MIN'), ('blank', 'DEPTH-TARGET'), ('boundary', 'DEPTH-FIRST-TWO-FAILED'),
    ('below_limit', 'DEPTH-Z-LIMIT'), ('wrong_request', 'DEPTH-MODE'),
    ('center_limit', 'DEPTH-FIRST-TWO-FAILED'), ('exposure_change', 'DEPTH-EXPOSURE'),
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
            # 等居中与自动曝光锁定完成后再模拟外部改曝光，验证 DEPTH-EXPOSURE 仍针对本点锁定值生效
            for _ in range(10):
                sess = rig.batch.blackboard.read('depth_session')
                if sess is not None and sess.exposure_locked:
                    break
                code, target = rig.exchange(target)
                assert code == 160
            rig.camera.exposure_us = 8000.
            kwargs = {'xyz': target}
    elif case == 'boundary':
        rig.cfg.depth_calibration['camera_z_min_mm'] = -4.5
        rig.camera.actual = (*rig.camera.actual[:2], -4.2)
    elif isinstance(case, int):
        kwargs = {'func': case}
    for _ in range(150):
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


def test_160_per_point_auto_exposure_dark_saturated_and_clamped(tcp_rig):
    """合成亮度：偏暗(p99≈60)自动提亮至190±15、过曝自动下调、不同点可不同曝光、触及上限记警告不999。"""
    rig = tcp_rig
    rig.camera.simulate_exposure = True
    # T00 偏暗 (p99≈60), T07 过曝 (p99=255), T31 正常, T24/T11 偏暗
    rig.camera.per_label_background = {'T00': 60.0, 'T07': 400.0, 'T31': 190.0, 'T24': 95.0, 'T11': 190.0}
    rig.camera.background_level = 190.0
    # 设一个刚好让 60->190 达不到而触及上限的测试用例在后半段测，本段先用默认上限 200000 跑完 5 点
    rig.cfg.depth_calibration['point_order'] = ['T00', 'T07', 'T31', 'T24', 'T11']

    coordinates = None
    session = None
    for _ in range(1800):
        code, coordinates = rig.exchange(coordinates)
        if code == 161:
            break
        assert code == 160, rig.events.status.data['depthcal']
        session = rig.batch.blackboard.read('depth_session')
    else:
        pytest.fail('未完成自动曝光多点会话')

    by_label = {r['label']: r for r in session.results}
    assert by_label['T00']['exposure_us'] > 12000.0
    assert 175.0 <= by_label['T00']['roi_p99'] <= 205.0
    assert by_label['T07']['exposure_us'] < 4500.0
    assert 175.0 <= by_label['T07']['roi_p99'] <= 205.0
    assert by_label['T07']['saturated_fraction'] <= 0.001
    assert by_label['T31']['exposure_us'] == pytest.approx(6000.0, rel=0.05)
    assert 175.0 <= by_label['T31']['roi_p99'] <= 205.0
    assert by_label['T00']['exposure_us'] != by_label['T07']['exposure_us']
    # 161 完成后主相机曝光必须恢复到会话前原值 6000.0
    assert rig.camera.exposure_us == pytest.approx(6000.0)
    _, post_meta = rig.camera.execute({'stage': 'coarse', 'task_id': 9999})
    assert post_meta['exposure_us'] == pytest.approx(6000.0)

    # 验证触及曝光上限时记警告并继续（不 999）
    rig.cfg.depth_calibration['exposure_max_us'] = 10000.0
    rig.camera.exposure_us = 6000.0
    rig.camera.per_label_background = {lbl: 60.0 for lbl in rig.camera.layout}
    coordinates = (*tuple(rig.camera.origin + rig.camera.layout['T00'] + [1., -1.]), -5.)
    for _ in range(40):
        code, coordinates = rig.exchange(coordinates)
        assert code == 160
        session2 = rig.batch.blackboard.read('depth_session')
        if session2.exposure_locked and session2.label == 'T00':
            break
    else:
        pytest.fail('未完成触及上限自动曝光')
    assert session2.point_exposure_us == pytest.approx(10000.0)
    assert session2.point_exposure_warning and '上限' in session2.point_exposure_warning


def test_160_manual_exposure_idle_persist_and_next_point_effect(tcp_rig):
    """空闲时持久化手动曝光，运行中修改手动曝光在下一测点生效且不触发当前点 DEPTH-EXPOSURE。"""
    from nest_clear.depthcal.exposure import load_depthcal_exposure
    from nest_clear.ui.model import classify_event

    rig = tcp_rig
    rig.camera.simulate_exposure = True
    rig.camera.background_level = 120.0
    rig.cfg.depth_calibration['point_order'] = ['T00', 'T07', 'T31', 'T24', 'T11']

    # 空闲时下发手动曝光 10000 µs（先 tick 几拍让 WaitRequest 消费命令）
    send(rig.root, 'depthcal_exposure', {'mode': 'manual', 'manual_exposure_us': 10000.0})
    deadline = time.monotonic() + 2.0
    while load_depthcal_exposure(rig.root)['mode'] != 'manual':
        rig.plc_worker._board and None
        # 触发一次 clock tick：通过 WaitRequest 消费命令（发一次 160 即可或检查 persistence）
        break
    # 直接通过一次 160 前短暂等待或由首次 160 之前 WaitRequest 消费
    # 用 FakePLC 不会阻塞 WaitRequest；让 rig.exchange(160) 前先跑几拍无法直接调 clock，因此先测运行中与首次进入前：
    code, target = rig.exchange()
    assert code == 160
    session = rig.batch.blackboard.read('depth_session')
    # WaitRequest 已在首次 160 之前消费 depthcal_exposure 并持久化
    assert load_depthcal_exposure(rig.root) == {'mode': 'manual', 'manual_exposure_us': 10000.0}
    assert session.desired_exposure_mode == 'manual'
    assert session.desired_manual_exposure_us == pytest.approx(10000.0)

    # 推进直到 T00 完成曝光锁定（10000 µs）
    for _ in range(30):
        if session.exposure_locked and session.label == 'T00':
            break
        code, target = rig.exchange(target)
        assert code == 160
    assert session.exposure_locked and session.point_exposure_us == pytest.approx(10000.0)

    # 在 T00 扫描中途修改手动曝光为 14000 µs -> T00 必须保持 10000 µs，T07 改为 14000 µs
    send(rig.root, 'depthcal_exposure', {'mode': 'manual', 'manual_exposure_us': 14000.0})
    t00_done = False
    t07_locked = False
    for _ in range(400):
        code, target = rig.exchange(target)
        assert code == 160
        if not t00_done and session.results:
            t00_done = True
            assert session.results[0]['label'] == 'T00'
            assert session.results[0]['exposure_us'] == pytest.approx(10000.0)
        if session.label == 'T07' and session.exposure_locked:
            t07_locked = True
            assert session.point_exposure_us == pytest.approx(14000.0)
            break
    assert t00_done and t07_locked
    assert load_depthcal_exposure(rig.root) == {'mode': 'manual', 'manual_exposure_us': 14000.0}
    ev = classify_event({'event': 'depthcal_exposure_warning', 'what': '自动曝光达到上限'})
    assert ev['color'] == 'yellow' and ev['category'] == '标定'


def test_camera_worker_set_exposure_operation_used_by_depthcal():
    """验证 CameraWorker._execute(operation='set_exposure') 复用现有 set_exposure 路径。"""
    from autoweaver.sensor.camera.base import CameraConfig
    from autoweaver.sensor.camera.mock import MockCamera
    from nest_clear.workers.camera import CameraWorker

    worker = CameraWorker('mock', 'MOCK01', MockCamera(CameraConfig(device_sn='MOCK01'), width=128, height=96), exposure_us=6000.0)
    worker.open()
    res, meta = worker.execute({'stage': 'depthcal', 'operation': 'set_exposure', 'exposure_us': 15000.0})
    assert res is None
    assert meta['exposure_us'] == pytest.approx(15000.0)
    assert worker.readback['exposure_us'] == pytest.approx(15000.0)
    frame, fmeta = worker.execute({'stage': 'depthcal', 'task_id': 1})
    assert frame.shape == (96, 128, 3)
    assert fmeta['exposure_us'] == pytest.approx(15000.0)


@pytest.mark.parametrize('exit_kind', ['done', 'fail_999', 'abort', 'interrupt'])
def test_160_restores_mock_camera_exposure_and_auto_on_all_exits(tmp_path, exit_kind):
    """成功(161)、失败(999)、放弃(abort)、中断(interrupt)后 MockCamera 曝光与 ExposureAuto 回到原值，后续其他流程采图 metadata 正确。"""
    from autoweaver.sensor.camera.base import CameraConfig
    from autoweaver.sensor.camera.mock import MockCamera
    from nest_clear.workers.camera import CameraWorker

    film = FilmCamera()
    film.simulate_exposure = True
    film.background_level = 60.0  # 会触发自动曝光把曝光调高到 ~19000 µs

    class SyntheticMockCamera(MockCamera):
        def __init__(self):
            super().__init__(CameraConfig(device_sn='MOCK_DEPTH'), width=420, height=420)
            self.exposure_us = 6000.0
            self.auto_mode = 2  # 模拟原 ExposureAuto=Continuous
            self._cam = SimpleNamespace(
                ExposureAuto=SimpleNamespace(
                    get=lambda: self.auto_mode,
                    set=lambda v: setattr(self, 'auto_mode', v),
                ),
                ExposureTime=SimpleNamespace(
                    get=lambda: self.exposure_us,
                    set=lambda v: setattr(self, 'exposure_us', float(v)),
                ),
            )
            self._gx = SimpleNamespace(GxAutoEntry=SimpleNamespace(OFF=0))

        def snapshot(self):
            film.actual = self.actual
            film.exposure_us = 6000.0
            img, _ = film.execute({'stage': 'depthcal'})
            return np.repeat(img[:, :, None], 3, axis=2)

    mock_cam = SyntheticMockCamera()
    mock_cam.actual = film.actual
    worker = CameraWorker('mock', 'MOCK_DEPTH', mock_cam, exposure_us=6000.0)
    worker.open()
    # 模拟会话开始前主相机原本处于 ExposureAuto=2（Continuous）与 6000 µs
    mock_cam.exposure_us = 6000.0
    mock_cam.auto_mode = 2
    worker.readback['exposure_us'] = 6000.0

    cfg = load_config('config/nest_clear.yaml')
    settings = {
        **cfg.depth_calibration,
        'camera_z_min_mm': -7.0,
        'coarse_step_mm': .50,
        'overshoot_mm': .15,
        'settle_window_s': .0001,
        'arrival_timeout_s': .05,
        'frames_per_step': 1,
        'repeats_per_point': 1,
        'point_order': ['T00', 'T07', 'T31', 'T24', 'T11'],
        'travel': ((80, 220), (0, 230), (-15, 0)),
    }
    cfg = replace(
        cfg,
        depth_calibration=settings,
        travel=((80, 220), (0, 230), (-15, 0)),
        tool_offset={**cfg.tool_offset, 'card_thickness_mm': 0.},
        camera_calibration={**cfg.camera_calibration, 'board_thickness_mm': 0.},
    )
    cal = Calibration(
        np.array([[2000., 0., 210], [0., 2000., 210], [0., 0., 1.]]),
        np.zeros(5), np.diag([1., -1., -1.]), np.zeros(3), -105,
    )
    events = EventRecorder(tmp_path, run=True)
    events.main_camera = worker

    with FakePLC() as plc:
        io = PLCClient('127.0.0.1', plc.port, 1, plc.contract)
        board = WorldBoard()
        clock = BTClock(board)
        for w in (PlcWorker(Handshake(io, plc.contract, events=events)), worker):
            clock.attach_worker(w)
        batch = build_batch(board, cfg, [], RequestTracker(0), events, calibration=cal)
        clock.submit(batch)

        def exchange(xyz=None, func=160):
            if xyz is not None:
                mock_cam.actual = tuple(xyz)
            plc.set_actual_xyz(mock_cam.actual)
            task = plc.request(func, tool_id=65535)
            deadline = time.monotonic() + 6
            while batch.blackboard.read('committed_func') is None or events.request_pending or plc.io.read('result_id') != task:
                clock.tick_once()
                if time.monotonic() > deadline:
                    pytest.fail(f'未收到 Task_ID={task} 回应')
                time.sleep(.00015)
            code = plc.io.read('reply_func')
            return code, plc.io.read_real32_block(40942, 3) if code == 160 else None

        try:
            # 160 启动时先发几步让自动曝光把 mock_cam.exposure_us 调到 != 6000
            target = None
            session = None
            for _ in range(30):
                code, target = exchange(target)
                assert code == 160
                session = batch.blackboard.read('depth_session')
                if session.exposure_locked:
                    break
            assert session.exposure_locked and mock_cam.exposure_us > 12000.0
            assert mock_cam.auto_mode == 0  # 扫描中自动曝光已关闭

            if exit_kind == 'done':
                for _ in range(1200):
                    code, target = exchange(target)
                    if code == 161:
                        break
                    assert code == 160
                else:
                    pytest.fail('未完成 161 会话')
                assert code == 161
            elif exit_kind == 'fail_999':
                # 触发 DEPTH-Z-LIMIT 999
                code, _ = exchange((*target[:2], settings['camera_z_min_mm'] - 0.5))
                assert code == 999
            elif exit_kind == 'abort':
                send(tmp_path, 'depthcal_abort')
                code, _ = exchange(target)
                assert code == 999
            elif exit_kind == 'interrupt':
                events.abort_depthcal('测试主程序停止中断')

            assert batch.blackboard.read('mode') == Mode.IDLE
            assert mock_cam.exposure_us == pytest.approx(6000.0)
            assert mock_cam.auto_mode == 2
            # 后续其他流程（挑毛粗拍）采图 metadata exposure_us 必须为原值 6000.0
            _, other_meta = worker.execute({'stage': 'coarse', 'task_id': 8888})
            assert other_meta['exposure_us'] == pytest.approx(6000.0)
        finally:
            clock.shutdown()
            io.close()


def test_160_restore_failure_not_swallowed_and_still_replies_999(tcp_rig):
    """曝光恢复失败不得吞掉：写 error 事件、999 诊断含“主相机曝光恢复失败，请重启程序”，且仍发 PLC 应答。"""
    rig = tcp_rig
    rig.camera.simulate_exposure = True
    rig.camera.background_level = 60.0
    code, target = rig.exchange()
    assert code == 160
    session = rig.batch.blackboard.read('depth_session')
    for _ in range(20):
        if session.exposure_locked:
            break
        code, target = rig.exchange(target)
        assert code == 160
    assert session.exposure_locked

    rig.camera.fail_restore = True
    send(rig.root, 'depthcal_abort')
    code, _ = rig.exchange(target)
    assert code == 999
    assert rig.batch.blackboard.read('mode') == Mode.IDLE

    diag = rig.events.status.data['depthcal']
    assert '主相机曝光恢复失败，请重启程序' in diag['what']
    event_rows = [
        json.loads(line)
        for line in (rig.events.run_directory / 'events.jsonl').read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    err_rows = [e for e in event_rows if e.get('event') == 'error' and e.get('operation') == 'depthcal_restore_exposure']
    assert len(err_rows) == 1
    assert '主相机曝光恢复失败，请重启程序' in err_rows[0]['reason']


def test_160_single_repeat_failure_tolerance_and_reject(tmp_path):
    """单点 3 次重复：1 次失败且余下极差 <= tol -> 有效并记录；极差超限或 >=2 次失败 -> 该点失败触发补测。"""
    from nest_clear.depthcal.layout import dof_summary, repeatability
    from nest_clear.depthcal.record import DepthRecord
    from nest_clear.depthcal.session import Session

    cfg = load_config('config/nest_clear.yaml')
    cfg = replace(
        cfg,
        depth_calibration={**cfg.depth_calibration, 'camera_z_min_mm': -12.0, 'repeats_per_point': 3},
    )
    cal = Calibration(
        np.array([[2000., 0., 210], [0., 2000., 210], [0., 0., 1.]]),
        np.zeros(5), np.diag([1., -1., -1.]), np.zeros(3), -105,
    )

    def make_sess(name):
        from collections import deque
        ev = EventRecorder(tmp_path / name, run=True)
        sess = Session(cfg, cal, DepthRecord(ev))
        sess.record.begin()
        sess.steps = deque()
        sess.label = 'T00'
        sess.ref_label = 'T00'
        sess.ref_xy = np.array([45.0, 105.0])
        sess.point_xy = np.array([45.0, 105.0])
        sess.center_offset = np.array([0.0, 0.0])
        sess.camera_xy = (45.0, 105.0)
        sess.initial_z = -5.0
        sess.actual = (45.0, 105.0, -5.0)
        sess.roi = np.s_[0:40, 0:40]
        sess.fine_plan = {
            'mode': 'auto', 'high_mm': -3.5, 'low_mm': -6.5, 'top_mm': -3.2,
            'fine_step_mm': 0.1,
        }
        sess.overshoot_z = -3.2
        return sess

    def gauss_rows(peak, *, flat=False):
        zs = [round(-3.5 - 0.1 * i, 4) for i in range(31)]
        return [
            {
                'label': 'T00', 'pass': 1, 'direction': 'down', 'z_cmd': z, 'z_actual': z,
                'tenengrad': 0.2 if flat else (0.1 + math.exp(-0.5 * ((z - peak) / 0.5) ** 2)),
                'laplacian': 0.2 if flat else (0.1 + math.exp(-0.5 * ((z - peak) / 0.5) ** 2)),
                'brenner': 0.2 if flat else (0.1 + math.exp(-0.5 * ((z - peak) / 0.5) ** 2)),
            }
            for z in zs
        ]

    # 1) 1 次失败（第2次），其余 2 次极差 0.008 <= 0.02 -> 有效并记录 failed_repeats=[2]
    s1 = make_sess('ok_one_fail')
    s1.repeat = 1
    s1.scan_rows = gauss_rows(-5.000)
    s1.fits.append({'repeat': 1, 'z_peak': -5.000, 'fwhm': 1.2, 'r2': 0.99, 'dof_mm': 0.6, 'dof_covered': True, 'quality_ok': True})
    s1.repeat = 2
    s1.phase = 'fine'
    s1.steps.clear()
    # 用 _handle_point_failure 触发之前，只跑单步 fit 失败分支
    import nest_clear.depthcal.session as sess_mod
    s1.fits.append({'repeat': 2, 'failed': True, 'error_code': 'DEPTH-FIT', 'failure': '模拟拟合失败'})
    s1.repeat = 3
    s1.scan_rows = gauss_rows(-5.008)
    s1.phase = 'fine'
    s1.steps.clear()
    # 模拟第 3 次最后一步到达并触发拟合（用高模糊度帧确保边界点清晰度落在尾段）
    raw_frame = np.full((80, 80), 200, np.uint8)
    cv2.circle(raw_frame, (20, 20), 10, 20, 3)
    frame = cv2.GaussianBlur(raw_frame, (0, 0), 8.0)
    s1.actual = (45.0, 105.0, -6.55)
    s1.expected = (45.0, 105.0, -6.55)
    s1.scan_rows = gauss_rows(-5.008)[:-1]
    coords = s1.advance([frame])
    assert coords is not None and s1.phase == 'peak'
    assert s1.result['failed_repeats'] == [2]
    assert s1.result['z_peak'] == pytest.approx((-5.000 + s1.result['fits'][2]['z_peak']) / 2, abs=1e-6)
    assert s1.result['range'] <= 0.02
    feas = repeatability([s1.result], 0.02)
    assert feas['met'] is True and feas['total_fits'] == 2
    ds = dof_summary([s1.result], 0.8)
    assert ds['total_fits'] == 2

    # 2) 1 次失败，但其余 2 次极差 0.035 > 0.02 -> 该点失败并换备用点
    s2 = make_sess('bad_range')
    s2.fits = [
        {'repeat': 1, 'z_peak': -5.000, 'fwhm': 1.2, 'r2': 0.99, 'dof_mm': 0.6, 'dof_covered': True, 'quality_ok': True},
        {'repeat': 2, 'failed': True, 'error_code': 'DEPTH-FIT', 'failure': '模拟拟合失败'},
    ]
    s2.repeat = 3
    s2.phase = 'fine'
    s2.steps.clear()
    s2.actual = (45.0, 105.0, -6.55)
    s2.expected = (45.0, 105.0, -6.55)
    s2.scan_rows = gauss_rows(-5.035)[:-1]
    coords2 = s2.advance([frame])
    assert coords2 is not None and s2.phase == 'center'
    assert len(s2.failed_points) == 1 and s2.failed_points[0]['label'] == 'T00'
    assert s2.label == s2.failed_points[0]['retest_label']
    assert '极差' in s2.failed_points[0]['reason']

    # 3) >=2 次失败 -> 该点立即失败并换备用点（不跑第 3 次）
    s3 = make_sess('two_fails')
    s3.fits = [{'repeat': 1, 'failed': True, 'error_code': 'DEPTH-FIT', 'failure': '第1次失败'}]
    s3.repeat = 2
    s3.phase = 'fine'
    s3.steps.clear()
    s3.actual = (45.0, 105.0, -6.55)
    s3.expected = (45.0, 105.0, -6.55)
    s3.scan_rows = gauss_rows(-5.0, flat=True)[:-1]
    coords3 = s3.advance([frame])
    assert coords3 is not None and s3.phase == 'center'
    assert len(s3.failed_points) == 1 and s3.failed_points[0]['label'] == 'T00'
    assert s3.label is not None and s3.label != 'T00'


def test_160_backup_selection_respects_nearest_unmeasured_and_travel(tmp_path):
    """补测点选择必须满足：离失败点最近、未测过/未计划、在 depth_calibration.travel 内。"""
    from nest_clear.depthcal.record import DepthRecord
    from nest_clear.depthcal.session import Session

    cfg = load_config('config/nest_clear.yaml')
    cfg = replace(
        cfg,
        depth_calibration={
            **cfg.depth_calibration,
            'camera_z_min_mm': -12.0,
            'point_order': ['T00', 'T07', 'T31', 'T24', 'T11'],
            # 限制 travel 使 T01 (y=105+45=150) 超出 y 上限 140，迫使 T00 选下一最近在行程内的 T08 (x=45+60=105? y=105-45=60? 按V2投影验证)
            'travel': ((30.0, 200.0), (-15.0, 201.0), (-12.0, 0.0)),
        },
    )
    cal = Calibration(
        np.array([[2000., 0., 210], [0., 2000., 210], [0., 0., 1.]]),
        np.zeros(5), np.diag([1., -1., -1.]), np.zeros(3), -105,
    )
    sess = Session(cfg, cal, DepthRecord(EventRecorder(tmp_path, run=False)))
    # 设 ref_label='T11' 位于 (135, 115) 使 T00/T01/T08/T09 均在 travel 内
    sess.ref_label = 'T11'
    sess.ref_xy = np.array([135.0, 115.0])
    sess.center_offset = np.array([0.0, 0.0])
    sess.attempted_labels = {'T00'}
    # T00 附近最近的是 T01 和 T08（距离均为 30mm）；先把 T01 标记为已测，应选 T08
    sess.attempted_labels.add('T01')
    assert sess._select_backup_target('T00') == 'T08'

    # 若把 T08 的相机 Y 坐标移出 travel.y 下限，则应跳过 T08 选次近在行程内的 T09
    t08_y = sess._target_camera_xy('T08')[1]
    sess.travel = ((30.0, 200.0), (t08_y + 1.0, 201.0), (-12.0, 0.0))
    sess.attempted_labels = {'T00', 'T01'}
    chosen = sess._select_backup_target('T00')
    assert chosen == 'T09'
    assert sess._in_xy_travel(sess._target_camera_xy(chosen))


def test_160_skip_retest_161_with_auto_exposure_and_center_skip(tcp_rig):
    """非致命跳过/补测回 161：单角点失败换最近备用点并自动曝光；中心点及其补测都失败则跳过仍回 161。"""
    rig = tcp_rig
    rig.camera.simulate_exposure = True
    rig.camera.background_level = 180.0
    rig.cfg.depth_calibration['point_order'] = ['T00', 'T07', 'T31', 'T24', 'T11']
    # T00 对比度不足 -> 补测 T01（暗，需自动曝光提亮）；T11 与其最近补测 T10/T03/T12 等做对比：让 T00 失败由 T01 补测成功
    rig.camera.flat_labels = {'T00'}
    rig.camera.per_label_background = {'T01': 70.0}
    rig.camera.actual = (*tuple(rig.camera.origin + rig.camera.layout['T00'] + [1., -1.]), -5.)

    coordinates = None
    session = None
    for _ in range(2200):
        code, coordinates = rig.exchange(coordinates)
        if code == 161:
            break
        assert code == 160, rig.events.status.data['depthcal']
        session = rig.batch.blackboard.read('depth_session')
    else:
        pytest.fail('未完成补测 161 会话')

    assert len(session.results) == 5
    retested = next(r for r in session.results if r['label'] == 'T01')
    assert retested['retest_of'] == 'T00'
    assert retested['exposure_us'] > 10000.0  # 补测点也执行了自动曝光
    assert len(session.failed_points) == 1
    assert session.failed_points[0]['label'] == 'T00'
    assert session.failed_points[0]['error_code'] == 'DEPTH-CONTRAST'
    assert session.failed_points[0]['retest_label'] == 'T01'
    assert session.failed_points[0]['retest_status'] == 'success'

    session_dir = session.record.directory
    fits_doc = json.loads((session_dir / 'fits.json').read_text(encoding='utf-8'))
    rec_doc = json.loads((session_dir / 'session_record.json').read_text(encoding='utf-8'))
    for doc in (fits_doc, rec_doc, rig.events.status.data['depthcal']):
        assert doc['failed_points'][0]['label'] == 'T00'
        assert doc['failed_points'][0]['retest_label'] == 'T01'
        assert 'T00' in doc['skip_summary']['text'] and 'T01' in doc['skip_summary']['text']
    # 失败点曲线保留在 curves 与 curves.csv
    assert 'T00' in rig.events.status.data['depthcal']['curves']
    assert rig.camera.exposure_us == pytest.approx(6000.0)


def test_160_center_and_backup_fail_skips_and_completes_161(tcp_rig):
    """非致命跳过：中心点 T11 及其补测点都失败时跳过该处，4 个角点有效 (>=4) 仍回 161。"""
    rig = tcp_rig
    rig.cfg.depth_calibration['repeats_per_point'] = 1
    rig.cfg.depth_calibration['point_order'] = ['T00', 'T07', 'T31', 'T24', 'T11']
    # T11 最近邻居为 T03/T10/T12/T19（均 30mm，按编号选 T03）；令 T11 与 T03 均对比度不足
    rig.camera.flat_labels = {'T11', 'T03'}
    rig.camera.actual = (*tuple(rig.camera.origin + rig.camera.layout['T00'] + [1., -1.]), -5.)

    coordinates = None
    session = None
    for _ in range(2000):
        code, coordinates = rig.exchange(coordinates)
        if code == 161:
            break
        assert code == 160, rig.events.status.data['depthcal']
        session = rig.batch.blackboard.read('depth_session')
    else:
        pytest.fail('未完成中心点跳过 161 会话')

    assert len(session.results) == 4
    assert {r['label'] for r in session.results} == {'T00', 'T07', 'T31', 'T24'}
    assert session.skip_summary['skipped_count'] == 1
    assert 'T11' in session.skip_summary['text'] and 'T03' in session.skip_summary['text'] and '跳过' in session.skip_summary['text']
    session_dir = session.record.directory
    fits_doc = json.loads((session_dir / 'fits.json').read_text(encoding='utf-8'))
    rec_doc = json.loads((session_dir / 'session_record.json').read_text(encoding='utf-8'))
    assert fits_doc['skip_summary']['skipped_count'] == 1
    assert rec_doc['skip_summary']['skipped_count'] == 1


@pytest.mark.parametrize('rule', ['a_first_two', 'b_min_valid', 'c_corner', 'd_plane'])
def test_160_fatal_rules_999_and_restore_exposure(tmp_path, rule):
    """致命规则 a/b/c/d 均回 999、诊断写明对应规则，且所有 999 出口均已恢复主相机曝光。"""
    from autoweaver.sensor.camera.base import CameraConfig
    from autoweaver.sensor.camera.mock import MockCamera
    from nest_clear.workers.camera import CameraWorker

    film = FilmCamera()
    film.simulate_exposure = True
    film.background_level = 60.0  # 自动曝光调高至 ~19000 µs

    point_order = ['T00', 'T07', 'T31', 'T24', 'T11']
    min_valid = 4
    if rule == 'a_first_two':
        # 前两计划点 T00(补测T01)、T07(补测T06) 均失败
        film.flat_labels = {'T00', 'T01', 'T07', 'T06'}
        expected_code = 'DEPTH-FIRST-TWO-FAILED'
        expected_marker = '规则 a'
    elif rule == 'b_min_valid':
        # 4 个角点成功，中心 T11 及其补测 T03 失败，且 min_valid_points=5 -> 有效点 4 < 5
        min_valid = 5
        film.flat_labels = {'T11', 'T03'}
        expected_code = 'DEPTH-INSUFFICIENT-POINTS'
        expected_marker = '规则 b'
    elif rule == 'c_corner':
        # 第 3 个角点 T31 及其补测 T23/T30 失败 -> 角点覆盖不足
        film.flat_labels = {'T31', 'T23', 'T30'}
        expected_code = 'DEPTH-CORNER-COVERAGE'
        expected_marker = '规则 c'
    else:
        # d_plane: T11 有 0.05mm 高度异常，加测至 9 点后残差仍 > plane_tol_mm
        film.per_label_focus = {'T11': -4.95}
        expected_code = 'DEPTH-PLANE-RESIDUAL'
        expected_marker = '规则 d'

    class SyntheticMockCamera(MockCamera):
        def __init__(self):
            super().__init__(CameraConfig(device_sn='MOCK_FATAL'), width=420, height=420)
            self.exposure_us = 6000.0
            self.auto_mode = 2
            self._cam = SimpleNamespace(
                ExposureAuto=SimpleNamespace(get=lambda: self.auto_mode, set=lambda v: setattr(self, 'auto_mode', v)),
                ExposureTime=SimpleNamespace(get=lambda: self.exposure_us, set=lambda v: setattr(self, 'exposure_us', float(v))),
            )
            self._gx = SimpleNamespace(GxAutoEntry=SimpleNamespace(OFF=0))

        def snapshot(self):
            film.actual = self.actual
            film.exposure_us = self.exposure_us
            img, _ = film.execute({'stage': 'depthcal'})
            return np.repeat(img[:, :, None], 3, axis=2)

    mock_cam = SyntheticMockCamera()
    mock_cam.actual = (*tuple(film.origin + film.layout['T00'] + [1., -1.]), -5.)
    worker = CameraWorker('mock', 'MOCK_FATAL', mock_cam, exposure_us=6000.0)
    worker.open()
    mock_cam.exposure_us = 6000.0
    mock_cam.auto_mode = 2
    worker.readback['exposure_us'] = 6000.0

    cfg = load_config('config/nest_clear.yaml')
    settings = {
        **cfg.depth_calibration,
        'camera_z_min_mm': -7.0,
        'coarse_step_mm': .50,
        'overshoot_mm': .15,
        'settle_window_s': .0001,
        'arrival_timeout_s': .05,
        'frames_per_step': 1,
        'repeats_per_point': 1,
        'min_valid_points': min_valid,
        'point_order': point_order,
        'travel': ((80, 220), (0, 230), (-15, 0)),
    }
    cfg = replace(
        cfg,
        depth_calibration=settings,
        travel=((80, 220), (0, 230), (-15, 0)),
        tool_offset={**cfg.tool_offset, 'card_thickness_mm': 0.},
        camera_calibration={**cfg.camera_calibration, 'board_thickness_mm': 0.},
    )
    cal = Calibration(
        np.array([[2000., 0., 210], [0., 2000., 210], [0., 0., 1.]]),
        np.zeros(5), np.diag([1., -1., -1.]), np.zeros(3), -105,
    )
    events = EventRecorder(tmp_path, run=True)
    events.main_camera = worker

    with FakePLC() as plc:
        io = PLCClient('127.0.0.1', plc.port, 1, plc.contract)
        board = WorldBoard()
        clock = BTClock(board)
        for w in (PlcWorker(Handshake(io, plc.contract, events=events)), worker):
            clock.attach_worker(w)
        batch = build_batch(board, cfg, [], RequestTracker(0), events, calibration=cal)
        clock.submit(batch)

        def exchange(xyz=None):
            if xyz is not None:
                mock_cam.actual = tuple(xyz)
            plc.set_actual_xyz(mock_cam.actual)
            task = plc.request(160, tool_id=65535)
            deadline = time.monotonic() + 6
            while batch.blackboard.read('committed_func') is None or events.request_pending or plc.io.read('result_id') != task:
                clock.tick_once()
                if time.monotonic() > deadline:
                    pytest.fail(f'未收到 Task_ID={task} 回应')
                time.sleep(.00015)
            code = plc.io.read('reply_func')
            return code, plc.io.read_real32_block(40942, 3) if code == 160 else None

        try:
            target = None
            for _ in range(1800):
                code, target = exchange(target)
                if code == 999:
                    break
                assert code == 160
            else:
                pytest.fail(f'规则 {rule} 未触发 999')
            assert batch.blackboard.read('mode') == Mode.IDLE
            diag = events.status.data['depthcal']
            assert diag['reason_code'] == expected_code
            assert expected_marker in diag['where'] or expected_marker in diag['what']
            # 主相机曝光与 ExposureAuto 必须在 999 出口恢复
            assert mock_cam.exposure_us == pytest.approx(6000.0)
            assert mock_cam.auto_mode == 2
        finally:
            clock.shutdown()
            io.close()
