"""通信联调假世界与真实相机、视觉 worker 的无 TCP 验证。"""
from collections import Counter
from dataclasses import replace
from pathlib import Path
import yaml
import pytest

from nest_clear.commtest import World, VisionAdapter, coordinates, branches
from nest_clear.config import load_config
from nest_clear.flow.work_cycle import WorkCycle
from nest_clear.guard import GuardError
from nest_clear.plan.scan_path import build_scan_path, guard_position
from nest_clear.vision.detector import StubDetector
from nest_clear.vision.geometry import load_calibration
from nest_clear.workers.camera import CameraWorker
from nest_clear.workers.vision import VisionWorker


class Events:
    def __init__(self): self.rows = []
    def emit(self, name, **fields): self.rows.append((name, fields))


def drive(scenario, broken=False):
    config = load_config('config/commtest.yaml')
    config = replace(config, commtest={'scenario': scenario, 'seed': 26})
    path = build_scan_path(config.scan, config.travel)
    camera = CameraWorker('mock', 'test', image_size=(640,480))
    camera.open()
    calibration = load_calibration('config/calibration/camera.yaml')
    detector = StubDetector()
    if broken:
        def fail(image): raise RuntimeError('detector failed')
        detector.detect = fail
    real = VisionWorker(detector, calibration, config.vision.dedup_radius_mm, config.vision.max_targets, config.vision)
    events = Events()
    adapter = VisionAdapter(real, World(config, path), events)
    cycle = WorkCycle(path, *coordinates(config), config.vision)
    reply = cycle.start(); replies = [reply]; coords = []; captures = 0
    try:
        for task in range(1, 500):
            if reply.scan_xyz: guard_position(reply.scan_xyz, config.travel, 'test'); coords.append(reply.scan_xyz)
            if reply.tool1_xyz:
                for point in (reply.tool1_xyz, reply.tool2_xyz): guard_position(point, config.travel, 'test')
                assert reply.tool2_xyz == pytest.approx((reply.tool1_xyz[0]+2, reply.tool1_xyz[1]+2, reply.tool1_xyz[2]))
                coords.extend((reply.tool1_xyz, reply.tool2_xyz))
            if reply.func_code in (9, 10): break
            if cycle.state.value == 'WAIT_PICK':
                cycle.mark_reply(task, task, {}); cycle.mark_done(task, task); reply = cycle.on_picked()
            else:
                stage = 'coarse' if cycle.state.value == 'WAIT_COARSE' else 'fine'
                image, _ = camera.execute({'stage':stage, 'task_id':task}); captures += 1
                result = adapter.execute({'image':image, 'pose':reply.scan_xyz, 'work_action':stage, 'cycle':cycle, 'scan':path[cycle.view_index], 'scan_path':path, 'capture_id':str(captures)})
                adapter.drain()
                reply = cycle.on_coarse(path[cycle.view_index], result['observations'], str(captures)) if stage == 'coarse' else cycle.on_fine(reply.scan_xyz, result['fine_result'])
            replies.append(reply)
        else: pytest.fail('流程未结束')
    finally: camera.on_stop()
    assert len(events.rows) == captures
    return Counter(r.func_code for r in replies), coords, events.rows


@pytest.mark.parametrize('scenario,terminal', [('normal',9),('tool_change',10),('fine_fail',9),('mixed',9)])
def test_tray(scenario, terminal):
    counts, coords, rows = drive(scenario)
    assert counts[terminal] == 1 and counts[999] == 0
    assert all(row['status'] in ('ok','empty') for _, row in rows)
    if scenario == 'fine_fail': assert counts[1] > 36
    assert coords == drive(scenario)[1]


def test_detector_exception_is_recorded_and_tray_continues():
    counts, _, rows = drive('normal', broken=True)
    assert counts[9] == 1 and counts[999] == 0
    assert all(row['status'] == 'error' and row['exception_type'] == 'RuntimeError' and 'detector failed' in row['traceback'] for _, row in rows)


def test_config_guards(tmp_path):
    data = yaml.safe_load(Path('config/commtest.yaml').read_text())
    data['travel']['x'] = [79,180]
    path = tmp_path / 'test.yaml'; path.write_text(yaml.safe_dump(data))
    with pytest.raises(GuardError): load_config(path)
    data = yaml.safe_load(Path('config/nest_clear.yaml').read_text()); data['camera']['sn'] = 'auto'; path.write_text(yaml.safe_dump(data))
    with pytest.raises(GuardError): load_config(path)


def test_real_tree_no_tcp_unsupported_and_camera_failure(tmp_path):
    import json
    import time
    from autoweaver.motion_policy.world_board import WorldBoard
    from autoweaver.worker.clock import BTClock
    from test_flow_recovery_no_tcp import MemoryPLC
    from nest_clear.flow.batch import build_batch
    from nest_clear.flow.context import RequestTracker
    from nest_clear.record.events import EventRecorder
    config = load_config('config/commtest.yaml')
    path = build_scan_path(config.scan, config.travel)
    cal = load_calibration('config/calibration/camera.yaml')
    board = WorldBoard(); clock = BTClock(board); plc = MemoryPLC()
    camera = CameraWorker('mock', 'test', image_size=(640,480)); camera.open()
    def fail_capture(): raise RuntimeError('mock camera failure')
    camera.camera.snapshot_with_meta = fail_capture
    events = EventRecorder(tmp_path, run=True); events.run_mode = 'commtest'
    vision = VisionAdapter(VisionWorker(StubDetector(), cal, 1, 10, config.vision), World(config, path), events)
    for worker in (plc, camera, vision): clock.attach_worker(worker)
    handle = clock.submit(build_batch(board, config, path, RequestTracker(0), events, branches=branches(), calibration=cal))
    def request(task, func):
        plc.request = (task, func)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            clock.tick_once()
            if task in plc.replies: return plc.replies[task]
            assert handle.result is None
            time.sleep(.001)
        pytest.fail('应答超时')
    try:
        assert request(1, 80)[0] == 999
        assert request(2, 100)[0] == 999
        code, data = request(3, 1); assert code == 1
        plc.actual = data['scan_xyz']
        assert request(4, 20)[0] == 999
    finally: clock.shutdown()
    rows = [json.loads(line) for line in (events.run_directory / 'events.jsonl').read_text().splitlines()]
    replies = [row for row in rows if row['event'] == 'reply']
    assert [row['reason_code'] for row in replies[:2]] == ['COMMTEST-UNSUPPORTED'] * 2
    assert all(row.get('commtest') for row in replies)
    assert any('mock camera failure' in str(row) for row in rows if row['event'] == 'error')


@pytest.mark.parametrize('scenario,terminal', [('normal',9),('tool_change',10),('fine_fail',9),('mixed',9)])
def test_real_tree_whole_tray_no_tcp(tmp_path, scenario, terminal):
    import time
    from autoweaver.motion_policy.world_board import WorldBoard
    from autoweaver.worker.clock import BTClock
    from test_flow_recovery_no_tcp import MemoryPLC
    from nest_clear.flow.batch import build_batch
    from nest_clear.flow.context import RequestTracker
    from nest_clear.record.events import EventRecorder
    config = load_config('config/commtest.yaml')
    config = replace(config, commtest={'scenario':scenario, 'seed':26})
    path = build_scan_path(config.scan, config.travel)
    cal = load_calibration('config/calibration/camera.yaml')
    events = EventRecorder(tmp_path); events.run_mode = 'commtest'
    board = WorldBoard(); clock = BTClock(board); plc = MemoryPLC()
    camera = CameraWorker('mock', 'test', image_size=(64,48)); camera.open()
    vision = VisionAdapter(VisionWorker(StubDetector(), cal, 1, 10, config.vision), World(config, path), events)
    for worker in (plc,camera,vision): clock.attach_worker(worker)
    handle = clock.submit(build_batch(board, config, path, RequestTracker(0), events, branches=branches(), calibration=cal))
    task = 0; func = 1; counts = Counter()
    try:
        while task < 200:
            task += 1; plc.request = (task,func); deadline = time.monotonic() + 5
            while task not in plc.replies:
                clock.tick_once()
                assert handle.result is None
                assert time.monotonic() < deadline
                time.sleep(.0005)
            reply, data = plc.replies[task]; counts[reply] += 1
            if reply == 1:
                plc.actual = data['scan_xyz']; guard_position(plc.actual, config.travel, 'scan'); func = 20
            elif reply == 3:
                a,b = data['pick1_xyz'],data['pick2_xyz']
                guard_position(a, config.travel, 'tool'); guard_position(b, config.travel, 'tool')
                assert b == pytest.approx((a[0]+2,a[1]+2,a[2])); func = 60
            else:
                assert reply == terminal
                break
        else: pytest.fail('未结束')
    finally: clock.shutdown()
    assert counts[terminal] == 1 and counts[999] == 0
    if scenario == 'fine_fail': assert counts[1] > 36


def test_auto_camera_requires_exactly_one(monkeypatch):
    from nest_clear.commtest import resolve_camera
    from nest_clear.tools import list_cameras
    for devices in ([], [{'sn':'a'}, {'sn':'b'}], [{'model_name':'missing'}]):
        monkeypatch.setattr(list_cameras, 'list_cameras', lambda: devices)
        with pytest.raises(GuardError, match='恰好一台'): resolve_camera()
    monkeypatch.setattr(list_cameras, 'list_cameras', lambda: [{'sn':'real-sn'}])
    assert resolve_camera() == 'real-sn'


def test_archive_falls_back_to_coordinates_when_projection_fails():
    import numpy as np
    from types import SimpleNamespace
    from nest_clear.commtest import ArchiveAdapter
    class Archive:
        def rendered(self, image, *args, **kwargs): self.image = image
    base = Archive(); archive = ArchiveAdapter(base)
    cal = load_calibration('config/calibration/camera.yaml')
    cycle = SimpleNamespace(state=SimpleNamespace(value='WAIT_PICK'))
    archive.remember('capture', np.zeros((200,400,3), dtype=np.uint8), [], (100.,0.,0.), (0.,0.,999999.), cal, cycle, 'fine')
    archive.rendered(np.zeros((200,400,3), dtype=np.uint8), 1, 1, 'raw', capture_id='capture')
    assert np.any(base.image[:, :, 0] > 0)
