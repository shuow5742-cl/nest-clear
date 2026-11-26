"""不使用 TCP 的工具偏移协议和合成卡片测试。"""
from dataclasses import replace
from types import SimpleNamespace
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
from nest_clear.record.events import EventRecorder
from nest_clear.tooloffset.commands import send
from nest_clear.tooloffset.session import nominal_targets
from nest_clear.tooltip.store import initialize_tools, load_tools
from nest_clear.vision.geometry import Calibration
from nest_clear.workers.tooloffset import ToolOffsetWorker


class PLC(AsyncRequestWorker):
    name = 'plc'
    def __init__(self):
        super().__init__(); self.request = (0,0,None); self.actual = (130.,70.,-5.); self.replies = {}; self.jitter = 0.; self.reads = 0
    def execute(self, payload):
        if payload['operation'] == 'request': return self.request
        if payload['operation'] == 'actual':
            self.reads += 1
            delta = self.jitter * (1 if self.reads % 2 else -1)
            return tuple(x + delta for x in self.actual)
        if payload['operation'] == 'reply': self.replies[payload['task_id']] = (payload['func'], payload['coordinates'])


class Camera(AsyncRequestWorker):
    name = 'camera'
    def __init__(self, plc): super().__init__(); self.plc = plc; self.scale = 1.005; self.blank = False
    def execute(self, payload):
        image = np.full((3000, 4000), 255, np.uint8)
        if self.blank: return image, {}
        angle = .02
        mat = self.scale * np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
        origin = np.array([130.,70.]); pose = self.plc.actual
        def pixel(p):
            world = origin + mat @ np.asarray(p)
            return np.array([2000 + 20*(world[0]-pose[0]), 1500 - 20*(world[1]-pose[1])])
        d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        for i, pt in nominal_targets().items():
            c = pixel(pt)
            if not (-200 < c[0] < 4200 and -200 < c[1] < 3200): continue
            ic = tuple(np.round(c).astype(int))
            cv2.circle(image, ic, 37, 0, 6, cv2.LINE_AA)
            for axis in ([5,0],[0,5]):
                p1, p2 = pixel(pt-np.asarray(axis)), pixel(pt+np.asarray(axis))
                cv2.line(image, tuple(np.round(p1).astype(int)), tuple(np.round(p2).astype(int)), 0, 2, cv2.LINE_AA)
            marker = cv2.aruco.generateImageMarker(d, i, 100)
            dest = np.float32([pixel(pt + [5,9]), pixel(pt + [9,9]), pixel(pt + [9,5]), pixel(pt + [5,5])])
            transform = cv2.getPerspectiveTransform(np.float32([[0,0],[99,0],[99,99],[0,99]]), dest)
            warped = cv2.warpPerspective(marker, transform, (4000,3000), borderValue=255)
            image = np.minimum(image, warped)
        return image, {}


def test_normal_real_tree(tmp_path):
    cfg = load_config('config/nest_clear.yaml')
    directory = tmp_path / 'calibration'; directory.mkdir()
    initialize_tools(directory / 'tools.yaml')
    from nest_clear.tooltip.store import save_tools
    from nest_clear.tooloffset.store import save_result
    save_tools(directory / 'tools.yaml', 2, [7.,8.,9.], source='fixture')
    save_result(directory / 'tool_offset.yaml', 2, {'offset_xy_mm':[7.,8.], 'n':7})
    save_result(directory / 'tool_offset.yaml', 1, {'offset_xy_mm':[1.,2.], 'n':6})
    old_result = (directory / 'tool_offset.yaml').read_bytes()
    camera_path = directory / 'camera.yaml'; camera_path.write_text('test')
    cfg = replace(cfg, tools=((0., 0., 0.), (0., 0., 0.)), tool_calibration=replace(cfg.tool_calibration, directory=str(directory)),
                  travel=((80,180),(-10,150),(-15,0)),
                  camera_calibration={**cfg.camera_calibration, 'start_xyz':[130.,70.,-5.], 'board_thickness_mm':0.},
                  tool_offset={**cfg.tool_offset, 'settle_window_s':.005, 'min_points':5})
    cal = Calibration(np.array([[2000.,0,2000],[0,2000.,1500],[0,0,1]]), np.zeros(5), np.diag([1.,-1.,-1.]), np.zeros(3), -105.)
    events = EventRecorder(tmp_path, run=True); board = WorldBoard(); plc = PLC(); clock = BTClock(board)
    for w in (plc, Camera(plc), ToolOffsetWorker()): clock.attach_worker(w)
    batch = build_batch(board, cfg, [], RequestTracker(0), events, calibration=cal, calibration_path=camera_path)
    handle = clock.submit(batch)
    def until(predicate, timeout=30):
        deadline = time.monotonic()+timeout
        while time.monotonic()<deadline:
            clock.tick_once()
            assert handle.result is None
            if predicate(): return
            time.sleep(.001)
        raise AssertionError('等待超时')
    try:
        plc.request=(1,90,1)
        until(lambda: batch.blackboard.read('tooloffset_session',None) is not None)
        send(tmp_path,'select_tool_and_start',{'tool':1,'mode':'center'})
        until(lambda: 1 in plc.replies)
        assert plc.replies[1] == (90, {'scan_xyz':(130.,70.,-5.), 'pc_tool_id':1})
        task = 1
        while True:
            task += 1; plc.request=(task,95,1)
            until(lambda: task in plc.replies or (events.status and events.status.data['tooloffset'].get('phase') in ('tool','confirm')))
            if task not in plc.replies:
                state = events.status.data['tooloffset']; engine = batch.blackboard.read('tooloffset_session')['engine']
                if state['phase'] == 'tool':
                    nominal = nominal_targets()[engine.current]
                    angle = .02
                    matrix = 1.005 * np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
                    target = np.array([130.,70.]) + matrix @ nominal
                    plc.actual = (target[0]+2.2,target[1]-1.3,-5.)
                    send(tmp_path,'tool_in_place')
                    until(lambda: task in plc.replies or events.status.data['tooloffset'].get('phase') == 'confirm')
                if task not in plc.replies:
                    send(tmp_path,'confirm_overwrite',{'accept':True})
                    until(lambda: task in plc.replies)
            code, coords = plc.replies[task]
            assert coords["pc_tool_id"] == 1
            assert code in (91,95), events.status.data['tooloffset']
            if code == 91: break
            task += 1; plc.request=(task,90,1)
            until(lambda: task in plc.replies)
            assert plc.replies[task][0] == 90
            plc.actual = plc.replies[task][1]['scan_xyz']
        assert list(events.run_directory.glob('tooloffset/*/*_raw.png'))
        result = load_tools(directory / 'tools.yaml')['tools']['tool1']['offset_xyz']
        assert result[:2] == pytest.approx([2.2,-1.3], abs=.01)
        import yaml
        result_file = yaml.safe_load((directory / 'tool_offset.yaml').read_text())
        assert result_file['tool2']['offset_xy_mm'] == [7.,8.]
        assert load_tools(directory / 'tools.yaml')['tools']['tool2']['offset_xyz'] == [7.,8.,9.]
        assert result_file['tool1']['baseline_stale'] is True
        assert any(p.read_bytes() == old_result for p in (directory / 'history').glob('tool_offset_*.yaml'))
    finally:
        clock.shutdown()


@pytest.fixture
def rig(tmp_path):
    cfg = load_config('config/nest_clear.yaml')
    directory = tmp_path / 'calibration'; directory.mkdir()
    initialize_tools(directory / 'tools.yaml')
    camera_path = directory / 'camera.yaml'; camera_path.write_text('test')
    cfg = replace(cfg, tools=((0., 0., 0.), (0., 0., 0.)), tool_calibration=replace(cfg.tool_calibration, directory=str(directory)),
                  travel=((80,180),(-10,150),(-15,0)),
                  camera_calibration={**cfg.camera_calibration, 'start_xyz':[130.,70.,-5.], 'board_thickness_mm':0.},
                  tool_offset={**cfg.tool_offset, 'settle_window_s':.001, 'min_points':5})
    cal = Calibration(np.array([[2000.,0,2000],[0,2000.,1500],[0,0,1]]), np.zeros(5), np.diag([1.,-1.,-1.]), np.zeros(3), -105.)
    events = EventRecorder(tmp_path, run=True); board = WorldBoard(); plc = PLC(); clock = BTClock(board)
    camera = Camera(plc)
    for w in (plc, camera, ToolOffsetWorker()): clock.attach_worker(w)
    batch = build_batch(board, cfg, [], RequestTracker(0), events, calibration=cal, calibration_path=camera_path)
    handle = clock.submit(batch)
    def until(predicate, timeout=15):
        deadline = time.monotonic()+timeout
        while time.monotonic()<deadline:
            clock.tick_once()
            assert handle.result is None
            if predicate(): return
            time.sleep(.001)
        raise AssertionError('等待超时')
    def request(task, code, tool_id=1):
        plc.request = (task,code,tool_id); until(lambda: task in plc.replies); return plc.replies[task]
    def begin(task=1):
        plc.request=(task,90,1)
        until(lambda: events.status.data['tooloffset'].get('phase') == 'select')
        send(tmp_path,'select_tool_and_start',{'tool':1,'mode':'center'})
        until(lambda: task in plc.replies)
        return plc.replies[task]
    yield SimpleNamespace(plc=plc, camera=camera, batch=batch, events=events, until=until, request=request, begin=begin, root=tmp_path, cfg=cfg)
    clock.shutdown()


def test_no_session_and_wrong_order(rig):
    assert rig.request(1,95)[0] == 999
    assert rig.begin(2)[0] == 90
    assert rig.request(3,90)[0] == 999
    assert rig.batch.blackboard.read('tooloffset_session') is None


def test_arrival_failure_and_abort(rig):
    assert rig.begin()[0] == 90
    rig.plc.actual = (140.,70.,-5.)
    assert rig.request(2,95)[0] == 999
    rig.plc.actual = (130.,70.,-5.)
    rig.plc.request = (3,90,1)
    rig.until(lambda: rig.events.status.data['tooloffset'].get('phase') == 'select')
    send(rig.root,'abort')
    rig.until(lambda: 3 in rig.plc.replies)
    assert rig.plc.replies[3] == (92, {"pc_tool_id": 1})


def test_no_start_and_picking(rig):
    object.__setattr__(rig.cfg, 'camera_calibration', {**rig.cfg.camera_calibration, 'start_xyz':None})
    assert rig.request(1,90)[0] == 999
    from nest_clear.flow.modes import Mode
    rig.batch.blackboard.write('mode', Mode.PICKING, 'test')
    assert rig.request(2,90)[0] == 999


def test_jitter_stable_and_unstable_tool_wait(rig):
    rig.cfg.tool_offset['jitter_tol_mm'] = .02
    assert rig.begin()[0] == 90
    rig.plc.jitter = .008
    assert rig.request(2,95)[0] == 95
    code, coords = rig.request(3,90)
    assert code == 90
    rig.plc.actual = coords['scan_xyz']
    rig.plc.request=(4,95,1)
    rig.until(lambda: rig.events.status.data['tooloffset'].get('phase') == 'tool')
    engine = rig.batch.blackboard.read('tooloffset_session')['engine']
    target = engine.measured[engine.current]
    rig.plc.actual = (target[0]+2.2,target[1]-1.3,-5.)
    rig.plc.jitter = .05
    send(rig.root,'tool_in_place')
    rig.until(lambda: '抖动过大' in rig.events.status.data['tooloffset'].get('message',''))
    assert 4 not in rig.plc.replies and not engine.samples
    rig.plc.jitter = .008
    send(rig.root,'tool_in_place')
    rig.until(lambda: 4 in rig.plc.replies)
    assert rig.plc.replies[4][0] == 95
    assert len(engine.samples) == 1


@pytest.mark.parametrize('bad', ['blank','scale'])
def test_business_fail_overview(rig, bad):
    if bad == 'blank': rig.camera.blank = True
    else: rig.camera.scale = 1.05
    assert rig.begin()[0] == 90
    assert rig.request(2,95)[0] == 92
    assert rig.batch.blackboard.read('tooloffset_session') is None


def test_center_edge_and_refuse_save(rig):
    object.__setattr__(rig.cfg, 'travel', ((-500,500),(-500,500),(-15,0)))
    rig.plc.request=(1,90,1)
    rig.until(lambda: rig.events.status.data['tooloffset'].get('phase') == 'select')
    send(rig.root,'select_tool_and_start',{'tool':1,'mode':'center_edge'})
    rig.until(lambda: 1 in rig.plc.replies)
    task, presses, edge_photos = 1, 0, 0
    while True:
        task += 1; rig.plc.request=(task,95,1)
        rig.until(lambda: task in rig.plc.replies or rig.events.status.data['tooloffset'].get('phase') in ('tool','confirm'))
        if task not in rig.plc.replies:
            state = rig.events.status.data['tooloffset']; engine = rig.batch.blackboard.read('tooloffset_session')['engine']
            if state['phase'] == 'tool':
                p = engine.measured[engine.current]
                rig.plc.actual = (p[0]+2.2,p[1]-1.3,-5.)
                presses += 1; send(rig.root,'tool_in_place')
                rig.until(lambda: task in rig.plc.replies or rig.events.status.data['tooloffset'].get('phase') == 'confirm')
            if task not in rig.plc.replies:
                assert rig.events.status.data['tooloffset']['phase'] == 'confirm'
                assert len(engine.edges) == 4
                send(rig.root,'confirm_overwrite',{'accept':False})
                rig.until(lambda: task in rig.plc.replies)
                assert rig.plc.replies[task][0] == 92
                assert not (rig.root / 'calibration' / 'tool_offset.yaml').exists()
                break
        assert rig.plc.replies[task][0] == 95
        engine = rig.batch.blackboard.read('tooloffset_session')['engine']
        if engine.phase == 'edge': edge_photos += 1
        task += 1
        code, coords = rig.request(task,90)
        assert code == 90
        rig.plc.actual = coords['scan_xyz']
    assert presses == 5 and edge_photos == 4


def test_ignored_command_acknowledged_and_no_reply(rig):
    rig.plc.request = (1,90,1)
    rig.until(lambda: rig.events.status.data['tooloffset'].get('phase') == 'select')
    seq = send(rig.root, 'tool_in_place')
    rig.until(lambda: rig.events.status.data['command_seq'] == seq)
    assert 1 not in rig.plc.replies
    assert rig.batch.blackboard.read('tooloffset_session')['tool'] is None
    send(rig.root,'select_tool_and_start',{'tool':2,'mode':'center'})
    rig.until(lambda: 1 in rig.plc.replies)
    assert rig.plc.replies[1][0] == 90


def test_missing_camera_calibration_and_abort_between_requests(rig):
    camera_path = rig.root / 'calibration' / 'camera.yaml'
    camera_path.unlink()
    assert rig.request(1,90)[0] == 999
    camera_path.write_text('test')
    assert rig.begin(2)[0] == 90
    assert rig.request(3,95)[0] == 95
    send(rig.root, 'abort')
    rig.until(lambda: rig.batch.blackboard.read('tooloffset_session',None) is None)
    assert 4 not in rig.plc.replies


@pytest.mark.parametrize("code", [100, 110])
def test_new_offset_flow_placeholder(rig, code):
    assert rig.request(1, code, 1) == (999, {"pc_tool_id": 1})
    assert "未收到 90 却收到工具偏移标定请求" in rig.events.status.data["tooltip"]["message"]
    assert rig.begin(2) == (90, {"scan_xyz": (130., 70., -5.), "pc_tool_id": 1})
    assert rig.request(3, code, 1) == (131, {"pc_tool_id": 1})
    assert rig.events.status.data["tooloffset"]["message"] == "工具偏移标定新流程尚未实现"
    assert rig.request(4, 95, 2) == (999, {"pc_tool_id": 2})
    assert "40905 从 1 变为 2" in rig.events.status.data["tooloffset"]["message"]
