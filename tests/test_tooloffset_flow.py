"""工具偏移新协议骨架与挑取精度测试共享夹具。"""
from dataclasses import replace
from types import SimpleNamespace
import json
import math
import time

import cv2
import numpy as np
import pytest
from autoweaver.motion_policy.world_board import WorldBoard
from autoweaver.worker import AsyncRequestWorker
from autoweaver.worker.clock import BTClock

from nest_clear.config import load_config
from nest_clear.flow.batch import build_batch
from nest_clear.flow.context import RequestTracker
from nest_clear.flow.modes import Mode
from nest_clear.record.events import EventRecorder
from nest_clear.tooloffset.commands import send
from nest_clear.tooloffset.session import nominal_targets
from nest_clear.tooltip.store import initialize_tools
from nest_clear.vision.geometry import Calibration
from nest_clear.workers.tooloffset import ToolOffsetWorker


class PLC(AsyncRequestWorker):
    name = 'plc'

    def __init__(self):
        super().__init__()
        self.request = (0, 0, None)
        self.actual = (130., 70., -5.)
        self.final_coords = {1: (130., 70., -5.), 2: (130., 70., -5.)}
        self.replies = {}
        self.jitter = 0.
        self.reads = 0

    def execute(self, payload):
        op = payload['operation']
        if op == 'request':
            return self.request
        if op == 'actual':
            self.reads += 1
            delta = self.jitter * (1 if self.reads % 2 else -1)
            return tuple(x + delta for x in self.actual)
        if op == 'calibration_final':
            return self.final_coords[payload['tool']]
        if op == 'reply':
            self.replies[payload['task_id']] = (payload['func'], payload['coordinates'])
            self.diagnostic = payload['diagnostic']


class Camera(AsyncRequestWorker):
    """兼容 pick_accuracy 测试使用的合成菲林图相机。"""
    name = 'camera'

    def __init__(self, plc):
        super().__init__()
        self.plc = plc
        self.scale = 1.005
        self.blank = False

    def execute(self, payload):
        image = np.full((3000, 4000), 255, np.uint8)
        if self.blank:
            return image, {}
        angle = .02
        mat = self.scale * np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        origin = np.array([130., 70.])
        pose = self.plc.actual

        def pixel(p):
            world = origin + mat @ np.asarray(p)
            return np.array([2000 + 20 * (world[0] - pose[0]), 1500 - 20 * (world[1] - pose[1])])

        d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        for i, pt in nominal_targets().items():
            c = pixel(pt)
            if not (-200 < c[0] < 4200 and -200 < c[1] < 3200):
                continue
            ic = tuple(np.round(c).astype(int))
            cv2.circle(image, ic, 37, 0, 6, cv2.LINE_AA)
            for axis in ([5, 0], [0, 5]):
                p1, p2 = pixel(pt - np.asarray(axis)), pixel(pt + np.asarray(axis))
                cv2.line(image, tuple(np.round(p1).astype(int)), tuple(np.round(p2).astype(int)), 0, 2, cv2.LINE_AA)
            marker = cv2.aruco.generateImageMarker(d, i, 100)
            dest = np.float32([pixel(pt + [5, 9]), pixel(pt + [9, 9]), pixel(pt + [9, 5]), pixel(pt + [5, 5])])
            transform = cv2.getPerspectiveTransform(np.float32([[0, 0], [99, 0], [99, 99], [0, 99]]), dest)
            warped = cv2.warpPerspective(marker, transform, (4000, 3000), borderValue=255)
            image = np.minimum(image, warped)
        return image, {}


@pytest.fixture
def rig(tmp_path, request):
    cfg = load_config('config/nest_clear.yaml')
    directory = tmp_path / 'calibration'
    directory.mkdir()
    initialize_tools(directory / 'tools.yaml')
    camera_path = directory / 'camera.yaml'
    camera_state = getattr(request, 'param', 'ready')
    if camera_state != 'missing':
        camera_path.write_text('test')
    cfg = replace(
        cfg,
        tools=((0., 0., 0.), (0., 0., 0.)),
        tool_calibration=replace(cfg.tool_calibration, directory=str(directory)),
        travel=((80, 180), (-10, 150), (-15, 0)),
        camera_calibration={**cfg.camera_calibration, 'start_xyz': [130., 70., -5.], 'board_thickness_mm': 0.},
        tool_offset={**cfg.tool_offset, 'settle_window_s': .001},
    )
    cal = Calibration(np.array([[2000., 0, 2000], [0, 2000., 1500], [0, 0, 1]]), np.zeros(5), np.diag([1., -1., -1.]), np.zeros(3), -105., placeholder=camera_state == 'placeholder')
    events = EventRecorder(tmp_path, run=True)
    board = WorldBoard()
    plc = PLC()
    clock = BTClock(board)
    camera = Camera(plc)
    for w in (plc, camera, ToolOffsetWorker()):
        clock.attach_worker(w)
    batch = build_batch(board, cfg, [], RequestTracker(0), events, calibration=cal, calibration_path=camera_path)
    handle = clock.submit(batch)

    def until(predicate, timeout=15):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            clock.tick_once()
            assert handle.result is None
            if predicate():
                return
            time.sleep(.001)
        raise AssertionError('等待超时')

    def submit_request(task, code, tool_id=1):
        plc.request = (task, code, tool_id)
        until(lambda: task in plc.replies)
        return plc.replies[task]

    yield SimpleNamespace(plc=plc, camera=camera, batch=batch, events=events, until=until, request=submit_request, root=tmp_path, cfg=cfg)
    clock.shutdown()


def test_removed_95_and_out_of_session_requests(rig):
    assert rig.request(1, 95, 1) == (999, {'pc_tool_id': 1})
    assert rig.request(2, 100, 1) == (999, {'pc_tool_id': 1})
    assert rig.request(3, 110, 1) == (999, {'pc_tool_id': 1})
    assert rig.request(4, 130, 1) == (999, {'pc_tool_id': 1})
    rig.batch.blackboard.write('mode', Mode.PICKING, 'test')
    assert rig.request(5, 90, 1) == (999, {'pc_tool_id': 1})
    assert rig.request(6, 100, 1) == (999, {'pc_tool_id': 1})
    assert rig.request(7, 110, 1) == (999, {'pc_tool_id': 1})
    assert rig.request(8, 100, 2) == (999, {'pc_tool_id': 2})


@pytest.mark.parametrize(('rig', 'reason_code'), [
    ('missing', 'CALIBRATION-MISSING'),
    ('placeholder', 'CALIBRATION-PLACEHOLDER'),
], indirect=['rig'])
def test_unusable_camera_calibration_rejects_90_with_99(rig, reason_code):
    assert rig.request(1, 90, 1) == (99, {'pc_tool_id': 1})
    assert rig.plc.diagnostic['reason_code'] == reason_code
    assert rig.batch.blackboard.read('tooloffset_session') is None
    assert rig.batch.blackboard.read('mode') == Mode.IDLE
    state = rig.events.status.data['tooloffset']
    assert state['phase'] == 'failed' and state['failure_code'] == reason_code
    record_path, = (rig.events.run_directory / 'tooloffset').glob('*/session_record.json')
    record = json.loads(record_path.read_text(encoding='utf-8'))
    assert record['outcome'] == 'failed' and record['reason'] == rig.plc.diagnostic['what']
    assert record['requests'] == [{'task_id': 1, 'func': 90, 'plc_tool_id': 1}]
    assert record['replies'] == [{'task_id': 1, 'reply_func': 99, 'coordinates': {}, 'reason': record['reason']}]


def test_90_while_baseline_is_active_replies_99(rig):
    rig.batch.blackboard.write('mode', Mode.BASELINE, 'test')
    assert rig.request(1, 90, 1) == (99, {'pc_tool_id': 1})
    assert rig.plc.diagnostic['reason_code'] == 'TOOLOFFSET-BUSY'
    assert rig.batch.blackboard.read('mode') == Mode.BASELINE


def test_preview_release_failure_replies_99(rig):
    def fail_release():
        raise OSError('主相机预览释放失败')

    rig.events.stop_main_preview = fail_release
    assert rig.request(1, 90, 1) == (99, {'pc_tool_id': 1})
    assert rig.plc.diagnostic['what'] == '主相机预览释放失败'
    assert rig.batch.blackboard.read('tooloffset_session') is None
