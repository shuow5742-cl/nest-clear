import time
import numpy as np
import pytest
from nest_clear.guard import GuardError
from nest_clear.workers.camera import CameraWorker
from nest_clear.comm import Handshake, load_contract
from nest_clear.comm.codec import encode_real32

class Camera:
    def __init__(self, result): self.result = result
    def open(self): return True
    def close(self): pass
    def snapshot(self):
        if isinstance(self.result, Exception): raise self.result
        if self.result == 'slow' if isinstance(self.result, str) else False:
            time.sleep(.2)
            return np.zeros((3, 4), dtype=np.uint8)
        return self.result

@pytest.mark.parametrize('value', [RuntimeError('断线'), None, np.zeros((0, 4), dtype=np.uint8), np.zeros((2, 2), dtype=np.float32), 'slow'])
def test_capture_guard(value):
    worker = CameraWorker('mock', 'SN-123', Camera(value), timeout_s=.01)
    with pytest.raises(GuardError) as caught:
        worker.execute({'stage': 'coarse', 'task_id': 7})
    assert 'SN-123' in str(caught.value) and '粗拍' in str(caught.value)

def test_size_and_metadata():
    worker = CameraWorker('mock', 'S', Camera(np.zeros((3, 4), dtype=np.uint8)))
    worker.size = (5, 3)
    with pytest.raises(GuardError, match='尺寸'): worker.execute({'stage':'fine'})
    worker.size = (4, 3)
    _, meta = worker.execute({'stage':'fine'})
    assert meta['capture_seq'] and meta['t_after']['monotonic'] >= meta['t_before']['monotonic']

class Registers:
    def __init__(self): self.words = {}; self.writes = []; self.corrupt = False
    def write_u16(self, address, value): self.words[address] = value; self.writes.append(address)
    def write_real32_block(self, address, values):
        for i, word in enumerate(sum((encode_real32(v) for v in values), [])): self.words[address+i] = word
    def read_u16(self, address): return self.words.get(address, 0) + self.corrupt

def test_readback_does_not_commit():
    contract = load_contract('config/plc_contract.yaml'); io = Registers(); io.corrupt = True
    with pytest.raises(GuardError, match='COMM-READBACK'):
        Handshake(io, contract).write_reply(4, 1, {'scan_xyz': [1, 2, 3]})
    assert contract['result_id'].address not in io.writes

def test_task_double_read_retries():
    contract = load_contract('config/plc_contract.yaml')
    class IO:
        values = iter([1, 20, 2, 2, 60, 2])
        def read_u16(self, address): return next(self.values)
    assert Handshake(IO(), contract).read_request() == (2, 60)

def test_exposure_readback_mismatch():
    class BadCamera(Camera):
        @property
        def exposure_us(self): return 9
        @exposure_us.setter
        def exposure_us(self, value): pass
    worker = CameraWorker('mock', 'S-EXPOSURE', BadCamera(None), exposure_us=12)
    with pytest.raises(GuardError, match='期望 12，实际 9') as error:
        worker.open()
    assert 'S-EXPOSURE' in str(error.value)


def test_startup_config_failure_is_logged(monkeypatch, tmp_path):
    import sys
    from nest_clear import main
    monkeypatch.setattr(main, 'ROOT', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['nest-clear', '--config', str(tmp_path / 'missing.yaml')])
    with pytest.raises(SystemExit): main.main()
    text = (tmp_path / 'runs/startup_error.log').read_text()
    assert all(field in text for field in ('CONFIG-LOAD', '发生了什么', '位置', '排查建议'))


@pytest.mark.parametrize('message,chinese', [
    ('Failed to capture image', '取图超时'),
    ('Frame capture failed: incomplete frame', '残帧'),
    ('Failed to convert image to BGR', '转换失败'),
    ('Failed to get numpy array', '转换失败'),
])
def test_sdk_runtime_error_is_translated(message, chinese):
    worker = CameraWorker('mock', 'S', Camera(RuntimeError(message)))
    with pytest.raises(GuardError, match=chinese):
        worker.execute({'stage': 'fine', 'task_id': 12})


def test_image_size_guards_and_mock():
    class SizedCamera(Camera):
        def get_frame_size(self): return (4, 3)
    with pytest.raises(GuardError, match='相机尺寸不符'):
        CameraWorker('mock', 'S', SizedCamera(None), image_size=(5, 3)).open()
    with pytest.raises(GuardError, match='标定主点'):
        CameraWorker('mock', 'S', SizedCamera(None), image_size=(4, 3), principal=(4, 1)).open()
    worker = CameraWorker('mock', 'S', SizedCamera(np.zeros((2, 4), np.uint8)), image_size=(4, 3))
    worker.open()
    with pytest.raises(GuardError, match='图像尺寸不符'):
        worker.execute({'stage': 'coarse', 'task_id': 1})
    mock = CameraWorker('mock', 'S', image_size=(40, 30), principal=(10, 10))
    mock.open()
    image, metadata = mock.execute({'stage': 'coarse', 'task_id': 2})
    assert image.shape[:2] == (30, 40)
    assert metadata['frame_id'] is None and metadata['device_timestamp'] is None
    assert metadata['frame_metadata_reason'] == '框架 DahengCamera 未暴露，待 Auto_Weaver 扩展'
    mock.on_stop()


def test_plc_stall_injected_clock(monkeypatch):
    from types import SimpleNamespace
    from autoweaver.motion_policy.nodes.node import Status
    from nest_clear.flow.leaves import WaitRequest, CallWorker
    class Board:
        def __init__(self): self.data = {'poll_result': (7, 20)}
        def read(self, key, *args): return self.data.get(key)
        def write(self, key, value, *args): self.data[key] = value
    class Events:
        def __init__(self): self.rows = []
        def emit(self, event, **fields): self.rows.append((event, fields))
    now = [1.]
    board, events = Board(), Events()
    wait = WaitRequest(board, SimpleNamespace(observed=7, processed=7), 5, events, clock=lambda: now[0])
    wait.set_blackboard(board)
    monkeypatch.setattr(CallWorker, 'on_start', lambda self: Status.RUNNING)
    monkeypatch.setattr(CallWorker, 'on_running', lambda self: Status.SUCCESS)
    wait.on_start()
    now[0] = 121.
    wait.on_running()
    assert events.rows[0][0] == 'plc_stall'
    assert events.rows[0][1]['waited_s'] == 120 and events.rows[0][1]['last_task_id'] == 7


def test_heartbeat_failure_and_restoration(tmp_path, monkeypatch):
    import json
    from dataclasses import replace
    from nest_clear.config import load_config
    from nest_clear.supervisor.runner import Supervisor
    class PLC:
        broken = True
        def read(self, key): return 0
        def write(self, key, value):
            if key == 'heartbeat' and self.broken: raise OSError('断线')
    now = [10.]
    monkeypatch.setattr('nest_clear.supervisor.runner.time.monotonic', lambda: now[0])
    plc = PLC()
    supervisor = Supervisor(replace(load_config('config/nest_clear.yaml'), record_directory=str(tmp_path)), plc)
    supervisor.tick()
    plc.broken = False
    now[0] = 13.
    supervisor.tick()
    rows = [json.loads(s) for s in next(tmp_path.glob('*/events.jsonl')).read_text().splitlines()]
    assert any(r['event'] == 'heartbeat_failed' and '断线' in r['reason'] for r in rows)
    assert any(r['event'] == 'heartbeat_restored' and r['duration_s'] == 3. for r in rows)


@pytest.mark.parametrize('fail', [False, True])
def test_application_close_clears_ready(fail, caplog):
    from types import SimpleNamespace
    from datetime import datetime
    from nest_clear.main import Application
    writes, events = [], []
    class IO:
        def write(self, name, value):
            writes.append((name, value))
            if fail: raise OSError('写入失败')
        def close(self): pass
    app = Application.__new__(Application)
    app.clock, app.io = None, IO()
    app.events = SimpleNamespace(emit=lambda event, **kw: events.append((event, kw)), clock=datetime.now, monotonic=lambda: 1.)
    app.summary = SimpleNamespace(finish=lambda *args: None)
    app.archive = SimpleNamespace(close=lambda: None)
    app.close()
    assert writes == [('algorithm_ready', 0)]
    if fail:
        assert '40932' in caplog.text and any(e == 'error' and kw['operation'] == 'clear_algorithm_ready' for e, kw in events)


@pytest.mark.parametrize('value', [[0, 3036], [4024, -1], [1.5, 2], [True, 2], [5], '4024x3036'])
def test_invalid_camera_image_size(tmp_path, value):
    import yaml
    from nest_clear.config import load_config
    data = yaml.safe_load(open('config/nest_clear.yaml', encoding='utf-8'))
    data['camera']['image_size'] = value
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(GuardError, match='camera.image_size'):
        load_config(path, require_tools=False)
