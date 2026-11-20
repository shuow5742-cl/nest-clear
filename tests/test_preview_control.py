import json
import time
from dataclasses import replace
from types import SimpleNamespace

from nest_clear.config import load_config
from nest_clear.paths import ui_command_file
from nest_clear.preview.process import atomic_write
from nest_clear.supervisor.runner import Supervisor
from nest_clear.ui.model import preview_controls


class PLC:
    word = 0
    disconnected = False
    def read(self, name):
        if self.disconnected:
            raise OSError('断线')
        return {'status': self.word, 'algorithm_ready': 0, 'actual_x': 1., 'actual_y': 2., 'actual_z': 3.}.get(name, 0)
    def write(self, name, value):
        pass
    def close(self):
        pass


def command(root, value, seq):
    path = ui_command_file(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'preview': value, 'seq': seq}))


def test_ui_controls():
    assert not preview_controls({}, None)['enabled']
    assert preview_controls({'pid': 1}, 1)['enabled']
    assert not preview_controls({'pid': 1, 'child_pid': 2}, 1)['enabled']
    assert '挑毛主流程' in preview_controls({'pid': 1, 'child_state': '启动中'}, 1)['reason']
    assert not preview_controls({'pid': 1}, 4)['enabled']


def test_preview_process_and_xyz(tmp_path):
    cfg = load_config('config/nest_clear.yaml')
    cfg = replace(cfg, record_directory=str(tmp_path), camera_sn='mock', camera_image_size=(320, 240), vision=replace(cfg.vision, camera_backend='mock'))
    plc = PLC()
    sup = Supervisor(cfg, plc)
    command(tmp_path, True, 1)
    try:
        sup.tick(); sup._publish()
        assert sup.preview_process is not None
        deadline = time.monotonic() + 10
        status = tmp_path / 'state' / 'preview_status.json'
        while time.monotonic() < deadline:
            sup.tick(); sup._publish()
            if status.exists() and json.loads(status.read_text()).get('frames', 0) >= 3:
                break
            time.sleep(.05)
        data = json.loads(status.read_text())
        assert data.get('frames', 0) >= 3, data
        assert (tmp_path / 'live' / 'preview_latest.jpg').is_file()
        assert sup.preview_xyz == [1., 2., 3.]
        plc.disconnected = True
        sup._preview_xyz_next = 0
        sup.tick()
        assert sup.preview_xyz is None
        command(tmp_path, False, 2)
        process = sup.preview_process
        sup.tick()
        assert sup.preview_process is None and not process.is_alive()
    finally:
        sup._stop_preview('test')


def test_stop_before_child_and_reject(tmp_path, monkeypatch):
    cfg = replace(load_config('config/nest_clear.yaml'), record_directory=str(tmp_path))
    sup = Supervisor(cfg, PLC())
    order = []
    class Process:
        pid = 4
        def is_alive(self): return not getattr(self, 'stopped', False)
        def terminate(self): order.append('terminate'); self.stopped = True
        def join(self, timeout=None): order.append('join')
    sup.preview_process = Process()
    monkeypatch.setattr(sup, '_clear_ready', lambda: order.append('ready') or True)
    monkeypatch.setattr('nest_clear.supervisor.runner.subprocess.Popen', lambda *a, **k: order.append('start') or SimpleNamespace(pid=5, poll=lambda: None))
    sup.plc.word = 128
    sup.tick()
    assert order[:4] == ['terminate', 'join', 'ready', 'start']
    command(tmp_path, True, 1)
    sup.tick()
    assert sup.preview_process is None


def test_atomic_write(tmp_path, monkeypatch):
    import nest_clear.preview.process as process
    target = tmp_path / 'frame.jpg'
    target.write_bytes(b'old')
    original = process.os.replace
    def replace_file(src, dst):
        assert target.read_bytes() == b'old'
        assert src.read_bytes() == b'new'
        original(src, dst)
    monkeypatch.setattr(process.os, 'replace', replace_file)
    atomic_write(target, b'new')
    assert target.read_bytes() == b'new'


def test_supervisor_exit_stops_preview(tmp_path, monkeypatch):
    cfg = replace(load_config('config/nest_clear.yaml'), record_directory=str(tmp_path))
    cfg = replace(cfg, ui=replace(cfg.ui, enabled=False))
    sup = Supervisor(cfg, PLC())
    stopped = []
    monkeypatch.setattr(sup, '_stop_preview', lambda reason: stopped.append(reason) or True)
    monkeypatch.setattr(sup, '_stop', lambda reason: False)
    monkeypatch.setattr(sup, '_reply_after_failure', lambda *args, **kwargs: None)
    monkeypatch.setattr('nest_clear.supervisor.runner.time.sleep', lambda _: (_ for _ in ()).throw(KeyboardInterrupt()))
    sup.run()
    assert stopped == ['supervisor_exit']


def test_preview_closes_camera(tmp_path, monkeypatch):
    import numpy as np
    import nest_clear.preview.process as preview
    cfg = replace(load_config('config/nest_clear.yaml'), record_directory=str(tmp_path))
    closed = []
    class Stop:
        stopped = False
        def set(self): self.stopped = True
        def is_set(self): return self.stopped
        def wait(self, delay): self.stopped = True
    class Worker:
        def __init__(self, *args, **kwargs):
            self.camera = SimpleNamespace(close=lambda: closed.append(True))
        def open(self): pass
        def _execute(self, payload): return np.zeros((120, 160, 3), dtype=np.uint8), {}
    class Thread:
        def __init__(self, target, args, daemon): self.target, self.args = target, args
        def start(self): self.target(*self.args)
        def is_alive(self): return False
    monkeypatch.setattr(preview, 'threading', SimpleNamespace(Event=Stop, Thread=Thread))
    monkeypatch.setattr(preview.signal, 'signal', lambda *args: None)
    monkeypatch.setattr(preview, 'CameraWorker', Worker)
    monkeypatch.setattr(preview, 'analyze', lambda image, config: (image, {}))
    preview.run_preview(cfg)
    assert closed == [True]
    assert json.loads((tmp_path / 'state/preview_status.json').read_text())['frames'] == 1
