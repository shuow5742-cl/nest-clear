"""PLC 主动放弃与父程序代答之间的边界，无需 TCP。"""
from dataclasses import replace

import pytest

from nest_clear.comm import load_contract
from nest_clear.config import load_config
from nest_clear.supervisor.runner import Supervisor
from nest_clear.workers.plc import PlcWorker
from nest_clear.comm.handshake import Handshake


class MemoryPLC:
    def __init__(self):
        self.contract = load_contract('config/plc_contract.yaml')
        self.values = {'status': 0, 'task_id': 11, 'request_func': 20, 'result_id': 0, 'reply_func': 0}
        self.writes = []
        self.fail_reply = False

    def read(self, name):
        return self.values.get(name, 0)

    def write(self, name, value):
        self.values[name] = value
        self.writes.append((name, value))

    def read_u16(self, address):
        name = next(k for k, v in self.contract.registers.items() if v.address == address)
        return self.read(name)

    def write_u16(self, address, value):
        name = next(k for k, v in self.contract.registers.items() if v.address == address)
        if name == 'reply_func' and self.fail_reply:
            raise TimeoutError('断线')
        self.write(name, value)

    def close(self):
        pass


class Child:
    pid = 123
    def poll(self): return None
    def terminate(self): pass
    def wait(self, timeout=None): return 0


@pytest.fixture
def setup(tmp_path):
    plc = MemoryPLC()
    sup = Supervisor(replace(load_config('config/nest_clear.yaml'), record_directory=str(tmp_path)), plc)
    rows = []
    sup.events.listener = rows.append
    return plc, sup, rows


@pytest.mark.parametrize('status', [1 << 7, 1 << 6])
def test_plc_stop_abandons_without_reply(setup, monkeypatch, status):
    plc, sup, rows = setup
    sup.child = Child()
    monkeypatch.setattr(sup, '_start', lambda: None)
    plc.values['status'] = status
    sup.tick()
    assert not any(k in ('result_id', 'reply_func') for k, _ in plc.writes)
    assert any(r['event'] == 'request_abandoned' and r['task_id'] == 11 and r['request_func'] == 20 for r in rows)


def test_parent_exit_replies(setup, monkeypatch):
    plc, sup, rows = setup
    sup.child = Child()
    monkeypatch.setattr(sup, 'tick', lambda: (_ for _ in ()).throw(KeyboardInterrupt()))
    sup.run()
    assert plc.values['reply_func'] == 999
    assert plc.values['result_id'] == 11
    assert any(r['event'] == 'supervisor_reply_999' for r in rows)
    assert sup._pending_reply is None


@pytest.mark.parametrize('changed', ['initializing', 'estop', 'cleared', 'new_task'])
def test_pending_reply_abandoned_on_plc_change(setup, monkeypatch, changed):
    plc, sup, rows = setup
    plc.fail_reply = True
    sup._reply_after_failure('子进程异常')
    assert sup._pending_reply['task_id'] == 11
    plc.fail_reply = False
    if changed == 'initializing': plc.values['status'] = 128
    elif changed == 'estop': plc.values['status'] = 64
    elif changed == 'cleared': plc.values['task_id'] = 0
    else: plc.values.update(task_id=12, request_func=60)
    monkeypatch.setattr(sup, '_start', lambda: None)
    sup.tick()
    assert sup._pending_reply is None
    assert not any(k in ('result_id', 'reply_func') for k, _ in plc.writes)
    assert any(r['event'] == 'request_abandoned' and r['task_id'] == 11 and r['request_func'] == 20 for r in rows)


@pytest.mark.parametrize('status', [64, 128])
def test_startup_does_not_reply_during_plc_stop(setup, status):
    plc, sup, rows = setup
    plc.values['status'] = status
    assert PlcWorker(Handshake(plc, plc.contract, events=sup.events)).initialize() == 11
    assert not any(k in ('result_id', 'reply_func') for k, _ in plc.writes)
    assert any(r['event'] == 'request_abandoned' and r['task_id'] == 11 for r in rows)


def test_unidentified_pending_reply_cannot_be_sent_to_new_task(setup, monkeypatch):
    plc, sup, rows = setup
    original_read = plc.read
    def disconnected(name):
        raise TimeoutError('断线，无法确认原任务')
    monkeypatch.setattr(plc, 'read', disconnected)
    sup._reply_after_failure('子进程异常')
    assert sup._pending_reply['unidentified']
    monkeypatch.setattr(plc, 'read', original_read)
    plc.values['task_id'] = 12
    sup.tick()
    assert sup._pending_reply is None
    assert not any(k in ('result_id', 'reply_func') for k, _ in plc.writes)
    assert any(r['event'] == 'request_abandoned' and r['task_id'] is None for r in rows)


def test_startup_zero_baseline_clears_reply_area(setup):
    plc, sup, rows = setup
    plc.values.update(task_id=0, result_id=11, reply_func=9, status=128)
    assert PlcWorker(Handshake(plc, plc.contract, events=sup.events)).initialize() == 0
    assert plc.writes == [('result_id', 0), ('reply_func', 0)]


def test_startup_already_answered_preserves_reply_area(setup):
    plc, sup, rows = setup
    plc.values.update(task_id=11, result_id=11, reply_func=9)
    assert PlcWorker(Handshake(plc, plc.contract, events=sup.events)).initialize() == 11
    assert plc.values['reply_func'] == 9
    assert plc.writes == []


def test_startup_unanswered_request_gets_error_reply(setup):
    plc, sup, rows = setup
    plc.values.update(task_id=11, result_id=7, reply_func=1, status=0)
    assert PlcWorker(Handshake(plc, plc.contract, events=sup.events)).initialize() == 11
    assert plc.writes == [('reply_func', 999), ('result_id', 11)]
