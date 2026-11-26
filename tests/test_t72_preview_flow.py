import json
import time
from types import SimpleNamespace

import pytest
from autoweaver.motion_policy.nodes.node import Status
from autoweaver.motion_policy.world_board import WorldBoard
from autoweaver.worker.clock import BTClock

from nest_clear.flow.leaves import CallWorker, Instant, WaitRequest
from nest_clear.flow.modes import Mode
from nest_clear.flow.batch import build_batch
from nest_clear.flow.context import RequestTracker
from nest_clear.record.events import EventRecorder
from test_flow_recovery_no_tcp import MemoryPLC
from test_hair_integration import Board


class PreparationStub:
    def __init__(self):
        self.enabled = {'front': False, 'side': False}
        self.busy = False
        self.commands = []

    @property
    def active(self):
        return any(self.enabled.values())

    def command(self, name, params):
        self.commands.append((name, params))
        if name == 'tooltip_preview':
            self.enabled[params['camera']] = params['on']

    def stop_previews(self, cameras):
        stopped = []
        for camera in sorted(cameras):
            if self.enabled[camera]:
                self.enabled[camera] = False
                stopped.append(camera)
        return stopped


@pytest.mark.parametrize('mode,accepted', [(Mode.IDLE, True), (Mode.PICKING, True), (Mode.BASELINE, False)])
def test_command_uses_per_camera_eligibility(tmp_path, monkeypatch, mode, accepted):
    (tmp_path / 'supervisor.json').write_text(json.dumps({'pid': 1, 'child_pid': 2, 'word': 0}))
    rows = []
    prep = PreparationStub()
    events = SimpleNamespace(directory=tmp_path, tooltip_preparation=prep, baseline_session=None,
                             emit=lambda name, **data: rows.append((name, data)))
    reader = SimpleNamespace(last_seq=1, poll=lambda allowed: ('tooltip_preview', {'camera': 'front', 'on': True}))
    monkeypatch.setattr('nest_clear.flow.tool_offset.reader', lambda ctx: reader)
    monkeypatch.setattr(CallWorker, 'on_running', lambda self: Status.RUNNING)
    node = WaitRequest(None, SimpleNamespace(), events)
    node._blackboard = Board(mode=mode)
    assert node.on_running() == Status.RUNNING
    assert node._blackboard.read('mode') == mode
    assert bool(prep.commands) == accepted
    if not accepted:
        assert any('对刀基准建立中' in data.get('message', '') for _, data in rows)


def test_baseline_start_stops_previews_before_session(monkeypatch, tmp_path):
    rows = []; prep = PreparationStub(); prep.enabled.update(front=True, side=True)
    events = SimpleNamespace(directory=tmp_path, tooltip_preparation=prep, baseline_session=None,
                             emit=lambda event, **data: rows.append((event, data)))
    def start():
        assert not prep.active
        return SimpleNamespace(closed=False)
    events.start_baseline = start
    reader = SimpleNamespace(last_seq=1, poll=lambda allowed: ('baseline_start', {}))
    monkeypatch.setattr('nest_clear.flow.tool_offset.reader', lambda ctx: reader)
    monkeypatch.setattr(CallWorker, 'on_running', lambda self: Status.RUNNING)
    node = WaitRequest(None, SimpleNamespace(), events); node._blackboard = Board(mode=Mode.IDLE)
    assert node.on_running() == Status.RUNNING
    assert node._blackboard.read('mode') == Mode.BASELINE
    auto = [data for event, data in rows if event == 'preview_auto_close']
    assert {data['camera'] for data in auto} == {'front', 'side'}
    assert all(data['mode'] == Mode.BASELINE.value and data['reason'] == 'mode_entry' for data in auto)


def test_tool_mode_entry_stops_used_cameras():
    from nest_clear.flow.calibration import change_mode
    rows = []; prep = PreparationStub(); prep.enabled.update(front=True, side=True)
    events = SimpleNamespace(tooltip_preparation=prep,
                             emit=lambda event, **data: rows.append((event, data)))
    board = Board(mode=Mode.IDLE, task_id=7, request_func=100, next_index=0)
    change_mode(board, SimpleNamespace(events=events), Mode.CALIBRATING_1, 'test')
    assert not prep.active and board.read('mode') == Mode.CALIBRATING_1
    auto = [data for event, data in rows if event == 'preview_auto_close']
    assert {data['camera'] for data in auto} == {'front', 'side'}
    assert all(data['mode'] == Mode.CALIBRATING_1.value and data['task_id'] == 7 and data['func'] == 100 for data in auto)


def test_stop_waits_until_camera_thread_releases_device(tmp_path):
    import threading
    from nest_clear.baseline.preparation import Preparation
    rows = []
    events = SimpleNamespace(directory=tmp_path, emit=lambda event, **data: rows.append((event, data)))
    prep = Preparation(events)
    released = threading.Event(); stop = threading.Event()
    def acquisition():
        stop.wait()
        time.sleep(.02)
        released.set()
    thread = threading.Thread(target=acquisition); thread.start()
    prep.enabled['front'] = True
    prep.state['front']['on'] = True
    prep.stops['front'] = stop
    prep.threads['front'] = thread
    assert prep.stop_previews({'front'}) == ['front']
    assert released.is_set() and not thread.is_alive() and not prep.enabled['front']


def test_paused_main_routes_preview_but_preserves_other_commands(tmp_path):
    from nest_clear.main import Application
    from nest_clear.tooloffset.commands import send
    (tmp_path / 'supervisor.json').write_text(json.dumps({'pid': 1, 'child_pid': 2, 'word': 0}))
    prep = PreparationStub(); rows = []
    events = SimpleNamespace(directory=tmp_path, tooltip_preparation=prep, baseline_session=None,
                             request_pending=False, emit=lambda event, **data: rows.append((event, data)))
    app = Application.__new__(Application); app.events = events
    app.handle = SimpleNamespace(batch=SimpleNamespace(blackboard=Board(mode=Mode.PICKING)))
    gate = SimpleNamespace(beat=lambda: None, paused=lambda: True)
    seq = send(tmp_path, 'baseline_start')
    app.step(gate)
    assert events.command_reader.last_seq < seq
    send(tmp_path, 'tooltip_preview', {'camera': 'side', 'on': True})
    app.step(gate)
    assert prep.enabled['side']


@pytest.mark.parametrize('pending', [False, True])
def test_preview_does_not_reopen_calibration_worker_device(tmp_path, pending):
    from nest_clear.flow.leaves import handle_preview_command
    (tmp_path / 'supervisor.json').write_text(json.dumps({'pid': 1, 'child_pid': 2, 'word': 1}))
    prep = PreparationStub(); calls = []
    prep.worker = SimpleNamespace(opened=True, on_stop=lambda: calls.append('closed'))
    rows = []
    events = SimpleNamespace(directory=tmp_path, tooltip_preparation=prep, baseline_session=None,
                             request_pending=pending, emit=lambda event, **data: rows.append((event, data)))
    handle_preview_command(events, Mode.CALIBRATING_1, 'tooltip_preview', {'camera': 'front', 'on': True})
    assert calls == ([] if pending else ['closed'])
    assert prep.enabled['front'] is not pending
    if pending:
        assert any('应答后重试' in data.get('message', '') for _, data in rows)


@pytest.mark.parametrize('func,mode,closes', [(100, Mode.CALIBRATING_1, True), (1, Mode.IDLE, False)])
def test_request_closes_only_used_cameras_and_still_replies(tmp_path, func, mode, closes):
    events = EventRecorder(tmp_path, 'main')
    prep = PreparationStub()
    prep.enabled.update(front=True, side=True)
    events.tooltip_preparation = prep
    rows = []
    emit = events.emit
    def record(event, **data):
        rows.append((event, data))
        emit(event, **data)
    events.emit = record

    class Reply(Instant):
        def __init__(self): super().__init__('测试应答')
        def perform(self):
            assert prep.active != closes
            self._blackboard.write('reply_func', func, self.name)
            self._blackboard.write('reply_coordinates', {}, self.name)
            self._blackboard.write('reply_tool', None, self.name)

    board = WorldBoard(); clock = BTClock(board); plc = MemoryPLC(); clock.attach_worker(plc)
    batch = build_batch(board, None, [], RequestTracker(0), events,
                        branches={func: (frozenset({mode}), lambda ctx: Reply())})
    handle = clock.submit(batch)
    batch.blackboard.write('mode', mode, '测试')
    plc.request = (1, func, 2 if func == 100 else None)
    try:
        deadline = time.monotonic() + 5
        while 1 not in plc.replies and time.monotonic() < deadline:
            clock.tick_once()
            assert handle.result is None
            time.sleep(.001)
        assert plc.replies[1][0] == func
        auto = [data for event, data in rows if event == 'preview_auto_close']
        assert {item['camera'] for item in auto} == ({'front', 'side'} if closes else set())
        assert all(item['mode'] == mode.value and item['task_id'] == 1 and item['func'] == func and item['reason'] == 'plc_request' for item in auto)
    finally:
        clock.shutdown()
