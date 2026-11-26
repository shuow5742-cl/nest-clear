import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from nest_clear.baseline.runout import RunoutController, evaluate_frames
from nest_clear.tooltip.runout import RunoutSession
from nest_clear.preview.control import close_previews_for


class Detector:
    def detect(self, frame):
        u, v, status, rms = frame
        return SimpleNamespace(point_uv=(u, v), coarse=bool(status), diagnostic='末端对比度不足' if status else '', axis_rms_px=rms)


def test_quality_and_median(monkeypatch):
    monkeypatch.setattr('nest_clear.baseline.runout.make_tip_detector', lambda config: Detector())
    good = [(0, 0, 0, .1), (.1, 0, 0, .1), (.1, 0, 0, .1), (.1, 0, 0, .1), (.4, 0, 0, .1)]
    session = RunoutSession()
    _, result = evaluate_frames({'front': good, 'side': good}, {}, session)
    assert not result['rejected']
    assert session.to_record()['presses'][0]['front_median_uv'][0] == .1
    bad = list(good); bad[1] = (0, 5, 1, .1)
    _, result = evaluate_frames({'front': good, 'side': bad}, {}, session)
    assert '侧面第 2 帧' in result['reason'] and 'v 标准差' in result['reason']
    bad[1] = (0, 0, 0, 3)
    _, result = evaluate_frames({'front': good, 'side': bad}, {}, session)
    assert '侧面第 2 帧工具在动' in result['reason']
    assert len(session.to_record()['presses']) == 1


def test_record_and_plc_interruption(tmp_path, monkeypatch):
    events = SimpleNamespace(directory=tmp_path, rows=[])
    events.emit = lambda event, **fields: events.rows.append((event, fields))
    frame = np.zeros((10, 10), dtype=np.uint8)
    prep = SimpleNamespace(events=events, lock=threading.RLock(), enabled={'front': True, 'side': True},
                           streams={'front': 1, 'side': 1}, state={n: {'detection_status': '精定位'} for n in ('front', 'side')},
                           config=None, ui_config={}, capture_after=lambda *args: [SimpleNamespace(frame=frame)] * 5)
    prep.runout = RunoutController(prep)
    prep.stop_previews = lambda cameras: list(cameras)
    monkeypatch.setattr('nest_clear.baseline.runout.detect_frames', lambda frames, configs, *args: ({'front': [[0, 0]]*5, 'side': [[0, 0]]*5}, None))
    prep.runout.command('tooltip_runout_start', 1)
    prep.runout.command('tooltip_runout_press', 2)
    prep.runout.thread.join(3)
    assert prep.runout.state['reply_seq'] == 2
    events.tooltip_preparation = prep
    close_previews_for(events, {'front', 'side'}, 'plc_request', func=100)
    path = prep.runout.directory
    assert len(json.loads((path/'summary.json').read_text())['presses']) == 1
    assert json.loads((path/'session.jsonl').read_text())['result']['count'] == 1
    assert (path/'001_front.png').exists() and (path/'001_side.png').exists()
    assert prep.runout.state['message'] == '被 PLC 请求中断'
    assert {event for event, _ in events.rows} >= {'tooltip_runout_start', 'tooltip_runout_press', 'tooltip_runout_end'}


def test_plc_interrupt_does_not_wait_for_capture(tmp_path, monkeypatch):
    events = SimpleNamespace(directory=tmp_path, rows=[])
    events.emit = lambda event, **fields: events.rows.append((event, fields))
    entered = threading.Event()
    frame = np.zeros((10, 10), dtype=np.uint8)

    def capture(*args):
        entered.set()
        time.sleep(3)
        return [SimpleNamespace(frame=frame)] * 5

    prep = SimpleNamespace(events=events, lock=threading.RLock(), enabled={'front': True, 'side': True},
                           streams={'front': 1, 'side': 1}, state={name: {'detection_status': '精定位'} for name in ('front', 'side')},
                           config=None, ui_config={}, capture_after=capture)
    prep.runout = RunoutController(prep)
    stopped = []
    prep.stop_previews = lambda cameras: stopped.extend(cameras) or list(cameras)
    events.tooltip_preparation = prep
    monkeypatch.setattr('nest_clear.baseline.runout.detect_frames', lambda frames, configs, *args: ({'front': [[0, 0]]*5, 'side': [[0, 0]]*5}, None))
    additions = []
    monkeypatch.setattr('nest_clear.baseline.runout.add_coordinates', lambda *args: additions.append(args))
    prep.runout.command('tooltip_runout_start', 1)
    prep.runout.session.add_press([[0, 0]] * 5, [[0, 0]] * 5, np.eye(2) * .0135, np.eye(2) * .0135)
    prep.runout.command('tooltip_runout_press', 2)
    assert entered.wait(1)
    directory = prep.runout.directory
    worker = prep.runout.thread
    start = time.monotonic()
    close_previews_for(events, {'front', 'side'}, 'plc_request', func=100)
    assert time.monotonic() - start < .2
    assert set(stopped) == {'front', 'side'}
    assert len(json.loads((directory / 'summary.json').read_text())['presses']) == 1
    assert prep.runout.state['message'] == '被 PLC 请求中断'
    assert prep.runout.state['reply_seq'] == 2
    assert any(event == 'tooltip_runout_end' for event, _ in events.rows)
    assert worker.is_alive()
    worker.join(4)
    assert not worker.is_alive()
    assert additions == []
    record = json.loads((directory / 'session.jsonl').read_text())
    assert record['interrupted'] is True and record['result']['rejected']
    assert len(json.loads((directory / 'summary.json').read_text())['presses']) == 1
    assert prep.runout.state['message'] == '被 PLC 请求中断'


def test_interrupt_during_calculation_keeps_new_session_untouched(tmp_path, monkeypatch):
    from nest_clear.baseline.runout import add_coordinates
    events = SimpleNamespace(directory=tmp_path, rows=[])
    events.emit = lambda event, **fields: events.rows.append((event, fields))
    entered, release = threading.Event(), threading.Event()
    frame = np.zeros((10, 10), dtype=np.uint8)
    prep = SimpleNamespace(events=events, lock=threading.RLock(), enabled={'front': True, 'side': True},
                           streams={'front': 1, 'side': 1}, state={name: {'detection_status': '精定位'} for name in ('front', 'side')},
                           config=None, ui_config={}, capture_after=lambda *args: [SimpleNamespace(frame=frame)] * 5)
    prep.runout = RunoutController(prep)
    monkeypatch.setattr('nest_clear.baseline.runout.detect_frames', lambda *args: ({'front': [[0, 0]]*5, 'side': [[0, 0]]*5}, None))

    def calculate(*args):
        entered.set()
        assert release.wait(3)
        return add_coordinates(*args)

    monkeypatch.setattr('nest_clear.baseline.runout.add_coordinates', calculate)
    prep.runout.command('tooltip_runout_start', 1)
    prep.runout.command('tooltip_runout_press', 2)
    assert entered.wait(1)
    directory, worker = prep.runout.directory, prep.runout.thread
    start = time.monotonic()
    prep.runout.interrupt('被 PLC 请求中断')
    assert time.monotonic() - start < .2
    assert json.loads((directory / 'summary.json').read_text())['presses'] == []
    prep.runout.command('tooltip_runout_start', 3)
    new_session = prep.runout.session
    release.set()
    worker.join(4)
    assert not worker.is_alive()
    assert json.loads((directory / 'session.jsonl').read_text())['interrupted'] is True
    assert prep.runout.session is new_session
    assert new_session.to_record()['presses'] == []
    assert prep.runout.state['active'] and prep.runout.state['reply_seq'] == 3

@pytest.mark.parametrize('host', ['supervisor', 'main'])
def test_both_hosts_route_commands(host, monkeypatch):
    from nest_clear.flow.leaves import PREVIEW_COMMANDS, handle_preview_command
    from nest_clear.flow.modes import Mode
    seen = []
    prep = SimpleNamespace(worker=None, command=lambda name, params: seen.append((name, params)))
    events = SimpleNamespace(tooltip_preparation=prep, emit=lambda *a, **k: None, baseline_session=None)
    monkeypatch.setattr('nest_clear.flow.leaves.preview_command_allowed', lambda *args: (True, None))
    for name in ('tooltip_runout_start', 'tooltip_runout_press', 'tooltip_runout_end'):
        assert name in PREVIEW_COMMANDS
        handle_preview_command(events, Mode.IDLE, name, {'_seq': 17})
    assert [x[0] for x in seen] == ['tooltip_runout_start', 'tooltip_runout_press', 'tooltip_runout_end']
