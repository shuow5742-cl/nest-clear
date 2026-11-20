import json
import time
from types import SimpleNamespace

from autoweaver.motion_policy.world_board import WorldBoard
from nest_clear.paths import ui_command_file
from nest_clear.preview.controller import PreviewController
from nest_clear.record.events import EventRecorder
from nest_clear.ui.model import preview_controls
from test_camera_calibration_flow import rig


def command(root, value, seq):
    path = ui_command_file(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'preview': value, 'seq': seq, 'time': '2026-01-01T00:00:00'}))


def controller(root):
    events = EventRecorder(root, run=True)
    return PreviewController(root, WorldBoard(), events, {}), events


def test_commands_and_camcal_rejection(tmp_path):
    p, events = controller(tmp_path)
    command(tmp_path, True, 1); p.request_active = True; p.tick()
    assert json.loads(events.status.path.read_text())['preview']['seq'] == 1
    assert p.state['on'] and p.state['seq'] == 1
    command(tmp_path, False, 2); p.poll_command()
    assert not p.state['on'] and p.state['seq'] == 2
    events.active_camcal = {'shots': 1}
    command(tmp_path, True, 3); p.poll_command()
    assert not p.state['on'] and p.state['seq'] == 3
    assert p.state['blocked_reason'] == '相机标定进行中'


def test_busy_request_never_submits(tmp_path):
    p, _ = controller(tmp_path)
    p.state['on'] = True
    p.request_active = True
    p.tick()
    assert p.capture_id is None


def test_camcal_request_switches_off_and_still_replies(rig):
    p = PreviewController(rig.events.directory, rig.clock._board, rig.events, {}, camera=rig.camera)
    command(rig.events.directory, True, 1); p.poll_command()
    assert p.state['on']
    assert rig.request(1, 80)[0] == 80
    assert not p.state['on']
    events = [json.loads(line) for line in (rig.events.run_directory / 'events.jsonl').read_text().splitlines()]
    assert any(row['event'] == 'preview_off' and row['source'] == 'camcal_start' for row in events)


def test_ui_controls():
    assert not preview_controls({}, None)['enabled']
    assert not preview_controls({}, 3.1)['enabled']
    assert preview_controls({}, 1)['enabled']
    assert not preview_controls({'camcal': {'event': 'camcal_starting'}}, 1)['enabled']
    assert not preview_controls({'camcal': {'event': 'camcal_step'}}, 1)['enabled']
    assert preview_controls({'camcal': {'event': 'camcal_done'}}, 1)['enabled']


def test_preview_does_not_consume_camera_sequence():
    import numpy as np
    from nest_clear.workers.camera import CameraWorker
    camera = SimpleNamespace(snapshot=lambda: np.zeros((20, 20), dtype=np.uint8))
    worker = CameraWorker('mock', 'test', camera=camera)
    _, before = worker.execute({'stage': 'normal'})
    _, preview = worker.execute({'stage': 'preview'})
    _, after = worker.execute({'stage': 'normal'})
    assert preview['capture_seq'] is None
    assert after['capture_seq'] == before['capture_seq'] + 1


def test_worker_preview_writes_frame_and_status(tmp_path):
    import numpy as np
    from autoweaver.worker.clock import BTClock
    from nest_clear.workers.camera import CameraWorker
    from test_camera_calibration_flow import PLC
    device = SimpleNamespace(snapshot=lambda: np.zeros((120, 160), dtype=np.uint8), close=lambda: None)
    camera = CameraWorker('mock', 'test', camera=device)
    board, events = WorldBoard(), EventRecorder(tmp_path, run=True)
    clock = BTClock(board)
    clock.attach_worker(camera); clock.attach_worker(PLC())
    p = PreviewController(tmp_path, board, events, {}, interval=.5, camera=camera)
    command(tmp_path, True, 1)
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not p.state['frames']:
            clock.tick_once(); p.tick(); time.sleep(.01)
        assert p.state['frames'] > 0
        assert (tmp_path / 'live' / 'preview_latest.jpg').is_file()
        assert events.seq_capture == 0
    finally:
        p.close(); clock.shutdown()
