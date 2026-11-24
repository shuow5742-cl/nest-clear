import json
import os
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest
from autoweaver.motion_policy.world_board import WorldBoard
from autoweaver.worker.clock import BTClock

from nest_clear.config import load_config
from nest_clear.flow.batch import build_batch
from nest_clear.flow.context import RequestTracker
from nest_clear.flow.leaves import Instant
from nest_clear.flow.modes import Mode
from nest_clear.preview.main_host import MainPreview
from nest_clear.record.events import EventRecorder
from nest_clear.supervisor.runner import Supervisor
from nest_clear.workers.camera import CameraWorker
from test_flow_recovery_no_tcp import MemoryPLC
from test_preview_control import PLC


def wait_for(check, seconds=8):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            result = check()
            if result:
                return result
        except (OSError, ValueError):
            pass
        time.sleep(.02)
    raise AssertionError('timed out waiting for preview')


def write_command(root, seq, on=True, mode='preview'):
    path = root / 'state' / 'ui_command.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'seq': seq, 'preview': on, 'mode': mode}))


def read(path):
    return json.loads(path.read_text())


@pytest.fixture
def rig(tmp_path):
    cfg = load_config('config/nest_clear.yaml')
    cfg = replace(cfg, record_directory=str(tmp_path), camera_sn='mock', camera_image_size=(320, 240),
                  vision=replace(cfg.vision, camera_backend='mock'),
                  white_balance_override_path=str(tmp_path / 'white_balance.yaml'))
    worker = CameraWorker('mock', 'mock', exposure_us=cfg.exposure_us, gain_db=cfg.gain_db,
                          balance_ratio=cfg.balance_ratio, image_size=cfg.camera_image_size)
    worker.open()
    events = EventRecorder(tmp_path, 'main')
    rows = []
    old_emit = events.emit
    def emit(event, **fields):
        rows.append((event, fields)); old_emit(event, **fields)
    events.emit = emit
    sup = Supervisor(cfg, PLC())
    sup.child = SimpleNamespace(pid=os.getpid(), poll=lambda: None)
    sup._publish()
    host = MainPreview(cfg, worker, events)
    events.stop_main_preview = host.stop
    yield SimpleNamespace(cfg=cfg, worker=worker, events=events, rows=rows, sup=sup, host=host, root=tmp_path)
    host.stop()
    worker.camera.close()


def test_running_child_preview_status_matches_standalone(rig, monkeypatch):
    supervisor_events = []
    monkeypatch.setattr(rig.sup.events, 'emit', lambda name, **fields: supervisor_events.append(name))
    def cannot_spawn(*args, **kwargs):
        raise AssertionError('supervisor must not open a camera while the child is running')
    monkeypatch.setattr('nest_clear.supervisor.runner.multiprocessing.get_context', cannot_spawn)
    write_command(rig.root, 1)
    rig.sup._preview_tick()
    assert rig.sup.preview_process is None
    assert 'preview_rejected' not in supervisor_events
    rig.host.poll(Mode.IDLE)
    status = wait_for(lambda: (data if (data := read(rig.root / 'state' / 'preview_status.json')).get('frames', 0) >= 2 else None))
    assert {'sharpness', 'corners', 'board_offset_mm', 'mode', 'frames', 'fps', 'last_frame_time', 'capture_to_publish_s', 'error', 'actual_xyz'} <= status.keys()
    assert (rig.root / 'live' / 'preview_latest.jpg').exists()
    rig.sup._publish()
    parent = read(rig.root / 'supervisor.json')
    assert parent['preview']['on'] and parent['preview']['mode'] == 'preview' and parent['preview']['seq'] == 1
    rig.host.stop()
    rig.sup._publish()
    assert not read(rig.root / 'supervisor.json')['preview']['on']


def test_eligibility_rejects_picking_but_manual_allows(rig):
    write_command(rig.root, 1)
    rig.host.poll(Mode.PICKING)
    state = read(rig.root / 'state' / 'main_preview_state.json')
    assert not state['on'] and '挑毛中' in state['error']
    rig.sup.word = 1
    rig.sup._publish()
    write_command(rig.root, 2)
    rig.host.poll(Mode.PICKING)
    wait_for(lambda: read(rig.root / 'state' / 'preview_status.json').get('frames'))
    assert read(rig.root / 'state' / 'main_preview_state.json')['on']


def test_request_one_closes_preview_and_restores_settings_before_reply(rig, monkeypatch):
    from nest_clear.preview.white_balance import write_ratios
    original_exposure = rig.worker.readback['exposure_us']
    original_ratios = list(rig.worker.readback['balance_ratio'])
    write_command(rig.root, 1, mode='focus')
    rig.host.poll(Mode.IDLE)
    wait_for(lambda: read(rig.root / 'state' / 'preview_status.json').get('frames'))
    action = rig.root / 'state' / 'focus_action.json'
    action.write_text(json.dumps({'seq': 10, 'action': 'set_exposure', 'exposure_us': 12000}))
    wait_for(lambda: read(rig.root / 'state' / 'preview_status.json').get('temporary_exposure', {}).get('state') == 'done')
    def white_balance(worker, settings, calibration, override_path, root, publish):
        write_ratios(worker, [1.1, 1.0, 1.2])
        publish({'state': 'success', 'time': 'test'})
    monkeypatch.setattr('nest_clear.preview.white_balance.run_white_balance', white_balance)
    action.write_text(json.dumps({'seq': 11, 'action': 'white_balance'}))
    wait_for(lambda: read(rig.root / 'state' / 'preview_status.json').get('white_balance', {}).get('state') == 'success')
    assert rig.worker.readback['balance_ratio'] == [1.1, 1.0, 1.2]
    assert rig.worker.readback['exposure_us'] == 12000
    class Reply(Instant):
        def __init__(self): super().__init__('测试应答')
        def perform(self):
            assert rig.host.thread is None
            assert rig.worker.readback['exposure_us'] == original_exposure
            assert rig.worker.readback['balance_ratio'] == original_ratios
            image, _ = rig.worker.execute({'stage': 'coarse'})
            assert image.size
            self._blackboard.write('reply_func', 1, self.name)
            self._blackboard.write('reply_coordinates', {}, self.name)
            self._blackboard.write('reply_tool', None, self.name)
    board = WorldBoard(); clock = BTClock(board); plc = MemoryPLC(); clock.attach_worker(plc)
    batch = build_batch(board, rig.cfg, [], RequestTracker(0), rig.events,
                        branches={1: (frozenset({Mode.IDLE}), lambda ctx: Reply())})
    handle = clock.submit(batch); plc.request = (1, 1)
    try:
        deadline = time.monotonic() + 8
        while 1 not in plc.replies and time.monotonic() < deadline:
            clock.tick_once(); time.sleep(.002)
            assert handle.result is None
        assert plc.replies[1][0] == 1
        close = [d for event, d in rig.rows if event == 'preview_auto_close']
        assert any(d['camera'] == 'main' and d['func'] == 1 and d['task_id'] == 1 for d in close)
    finally:
        clock.shutdown()


def test_child_start_handoff_accepts_new_command_only(rig, monkeypatch):
    # Sequence 1 belonged to the previous owner; sequence 2 arrived while the
    # child was initializing and must still be handled once initialization ends.
    handoff = rig.root / 'state' / 'main_preview_handoff.json'
    handoff.write_text(json.dumps({'pid': os.getppid(), 'seq': 1}))
    write_command(rig.root, 2)
    host = MainPreview(rig.cfg, rig.worker, rig.events)
    try:
        assert host.seq == 1
        host.poll(Mode.IDLE)
        wait_for(lambda: read(rig.root / 'state' / 'preview_status.json').get('frames'))
        thread = host.thread
        host.poll(Mode.IDLE)
        assert host.thread is thread
    finally:
        host.stop()
