"""相机标定末帧、异常及超时都必须结束 PLC 握手。"""
import time

import pytest

from test_camera_calibration_flow import rig, rows, reply_row, FakeSession


def test_final_failure_reply(rig, monkeypatch):
    original = FakeSession.add
    def add(self, image, actual):
        if self.samples:
            step = original(self, image, actual)
            step.kind = 'fail'
            step.code = 'CAMCAL-RMS'
            step.reason = '整体重投影误差超过阈值'
            step.detail = {'per_image_rms': [1.0663] * 53, 'repeat_pairs': [[1, 2]] * 20}
            return step
        return original(self, image, actual)
    monkeypatch.setattr(FakeSession, 'add', add)
    assert rig.request(1, 80)[0] == 80
    rig.plc.actual = (1., 0., 0.)
    assert rig.request(2, 80)[0] == 82
    assert reply_row(rig, 2)['func'] == 82
    assert any(r['event'] == 'camcal_failed' and r['code'] == 'CAMCAL-RMS' for r in rows(rig))


def test_worker_exception_reply(rig, monkeypatch):
    def broken(*args):
        raise RuntimeError('worker exception')
    monkeypatch.setattr(FakeSession, 'add', broken)
    assert rig.request(1, 80)[0] == 82
    assert 'worker exception' in reply_row(rig, 1)['what']
    assert any(r['event'] == 'camcal_failed' for r in rows(rig))


def test_worker_timeout_reply(rig, monkeypatch):
    original = FakeSession.add
    def slow(*args):
        time.sleep(.2)
        return original(*args)
    monkeypatch.setattr(FakeSession, 'add', slow)
    rig.cfg.camera_calibration['solve_timeout_s'] = .03
    assert rig.request(1, 80)[0] == 82
    assert reply_row(rig, 1)['reason_code'] == 'CAMCAL-TIMEOUT'
    assert any(r['event'] == 'camcal_timeout' and r['code'] == 'CAMCAL-TIMEOUT' for r in rows(rig))
    assert any(r['event'] == 'camcal_failed' for r in rows(rig))
    time.sleep(.25)


def test_pause_during_final_worker_still_replies(rig, monkeypatch):
    import threading
    from nest_clear.main import Application
    from test_main_gate import Gate

    entered, release = threading.Event(), threading.Event()
    original = FakeSession.add
    def blocked(self, image, actual):
        if self.samples:
            entered.set()
            assert release.wait(2)
            step = original(self, image, actual)
            step.kind, step.code, step.reason = 'fail', 'CAMCAL-RMS', 'RMS 超限'
            return step
        return original(self, image, actual)
    monkeypatch.setattr(FakeSession, 'add', blocked)
    assert rig.request(1, 80)[0] == 80
    reply_row(rig, 1)
    rig.plc.actual = (1., 0., 0.)
    rig.plc.request = (2, 80)
    app = Application.__new__(Application)
    app.clock, app.events = rig.clock, rig.events
    app.handle = type('Handle', (), {'result': None})()
    gate = Gate()
    gate.value = False
    deadline = time.monotonic() + 2
    while not entered.is_set() and time.monotonic() < deadline:
        app.step(gate)
        time.sleep(.001)
    assert entered.is_set()
    gate.value = True
    release.set()
    deadline = time.monotonic() + 2
    while 2 not in rig.plc.replies and time.monotonic() < deadline:
        app.step(gate)
        time.sleep(.001)
    assert rig.plc.replies[2][0] == 82
    while rig.events.request_pending and time.monotonic() < deadline:
        app.step(gate)
        time.sleep(.001)
    assert not rig.events.request_pending
    assert any(r['event'] == 'camcal_failed' for r in rows(rig))
    ticks = rig.clock.tick_id
    app.step(gate)
    assert rig.clock.tick_id == ticks
