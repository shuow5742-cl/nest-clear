import json
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from nest_clear.config import load_config
from nest_clear.supervisor.runner import Supervisor
from nest_clear.tooloffset.commands import send, startup_reader
from nest_clear.ui.model import supervisor_tooltip_main
from test_preview_control import PLC


def wait_for(check):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        try:
            value = check()
            if value:
                return value
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(.02)
    raise AssertionError('timed out')


@pytest.fixture
def rig(tmp_path, monkeypatch):
    cfg = replace(load_config('config/nest_clear.yaml'), record_directory=str(tmp_path))
    devices = ['/dev/v4l/by-path/pci-usb-front-video-index0', '/dev/v4l/by-path/pci-usb-side-video-index0', '/dev/v4l/by-path/pci-usb-new-video-index0']
    history = []
    class Camera:
        def __init__(self, name, device):
            self.name = name
            self.config = {'device': device, 'settings_path': str(tmp_path / 'cameras.yaml')}
            self.device = None
            self.exposure = 100
        def open(self):
            assert self.device is None
            self.device = object(); history.append(('open', self.name))
        def close(self):
            if self.device is not None:
                history.append(('close', self.name))
                self.device = None
        def capture(self):
            time.sleep(.01)
            return np.full((80, 100), 210, np.uint8)
        def get_exposure(self): return self.exposure
        def set_exposure(self, value): self.exposure = value
        def settings_snapshot(self): return {'device': self.config['device'], 'controls': {'exposure': self.exposure}}
    class Worker:
        def __init__(self, *args, **kwargs):
            self.opened = False
            self.cameras = {name: Camera(name, device) for name, device in zip(('front', 'side'), devices)}
    monkeypatch.setattr('nest_clear.supervisor.tooltip_host.CalibCameraWorker', Worker)
    monkeypatch.setattr('nest_clear.baseline.preparation.enumerate_capture_devices', lambda: devices)
    monkeypatch.setattr('nest_clear.baseline.preparation.auto_exposure', lambda *a, **kw: {'exposure': 100, 'p99': 210, 'saturated_fraction': 0, 'contrast': 100})
    sup = Supervisor(cfg, PLC()); sup._publish()
    yield SimpleNamespace(sup=sup, root=tmp_path, devices=devices, history=history)
    sup._stop_tooltip()


def read(path):
    return json.loads(path.read_text())


def issue(rig, name, params):
    seq = send(rig.root, name, params)
    rig.sup._tooltip_tick()
    return seq


def test_supervisor_preview_and_camera_assign_swap(rig):
    seq = issue(rig, 'tooltip_preview', {'camera': 'front', 'on': True})
    wait_for(lambda: read(rig.root / 'state/tooltip_preview.json')['cameras'].get('front'))
    assert (rig.root / 'live/baseline_front.jpg').exists()
    state = read(rig.root / 'state/supervisor_tooltip_status.json')
    assert state['command_seq'] == seq and state['baseline']['preview']['front']['on']
    assert state['baseline']['detected_cameras'] == rig.devices
    issue(rig, 'tooltip_preview', {'camera': 'front', 'on': False})
    issue(rig, 'tooltip_assign', {'camera': 'front', 'device': rig.devices[2]})
    wait_for(lambda: not rig.sup.tooltip_host.tooltip_preparation.busy)
    issue(rig, 'tooltip_swap', {})
    wait_for(lambda: not rig.sup.tooltip_host.tooltip_preparation.busy)
    state = read(rig.root / 'state/supervisor_tooltip_status.json')
    assert state['baseline']['preview']['side']['device'] == rig.devices[2]


def test_start_joins_preview_before_child_and_preserves_command_boundary(rig, monkeypatch):
    seq = issue(rig, 'tooltip_preview', {'camera': 'front', 'on': True})
    wait_for(lambda: ('open', 'front') in rig.history)
    monkeypatch.setattr(rig.sup, '_clear_ready', lambda: True)
    def spawn(*args, **kwargs):
        assert ('close', 'front') in rig.history
        rig.history.append(('child', 'start'))
        return SimpleNamespace(pid=42, poll=lambda: None)
    monkeypatch.setattr('nest_clear.supervisor.runner.subprocess.Popen', spawn)
    rig.sup.plc.word = 1 << 7
    rig.sup.tick()
    assert rig.history.index(('close', 'front')) < rig.history.index(('child', 'start'))
    assert rig.sup.tooltip_host is None
    handoff = read(rig.root / 'state/tooltip_handoff.json')
    assert handoff['seq'] == seq
    monkeypatch.setattr('nest_clear.tooloffset.commands.os.getppid', lambda: handoff['pid'])
    reader, transferred = startup_reader(rig.root, rig.sup.events)
    assert transferred
    assert reader.poll({'tooltip_preview'}) is None
    send(rig.root, 'tooltip_preview', {'camera': 'side', 'on': True})
    rig.sup._tooltip_tick()
    assert rig.sup.tooltip_host is None  # Child owns all tooltip cameras.
    assert reader.poll({'tooltip_preview'})[1]['camera'] == 'side'


def test_supervisor_status_overlay_only_without_child():
    main = {'mode': '挑毛中', 'baseline': {'phase': 'live'}, 'command_seq': 3}
    host = {'pid': 5, 'baseline': {'phase': 'prepare', 'preview': {'front': {'on': True}}}, 'command_seq': 8}
    parent = {'pid': 5, 'child_pid': None}
    result = supervisor_tooltip_main(main, parent, host)
    assert result['mode'] == '空闲' and result['command_seq'] == 8
    assert result['baseline']['preview']['front']['on']
    assert supervisor_tooltip_main(main, {**parent, 'child_pid': 42}, host) is main
    assert not supervisor_tooltip_main(main, parent, {**host, 'pid': 6})['baseline']['preview']


def test_supervisor_routes_retest_and_peak_reset(rig, monkeypatch):
    issue(rig, 'tooltip_preview', {'camera': 'front', 'on': True})
    wait_for(lambda: ('open', 'front') in rig.history)
    prep = rig.sup.tooltip_host.tooltip_preparation
    original = prep.command
    commands = []
    def track(name, params):
        commands.append((name, params.get('camera')))
        return original(name, params)
    monkeypatch.setattr(prep, 'command', track)
    issue(rig, 'tooltip_exposure_retest', {'camera': 'front'})
    issue(rig, 'tooltip_peak_reset', {'camera': 'front'})
    assert ('tooltip_exposure_retest', 'front') in commands
    assert ('tooltip_peak_reset', 'front') in commands


def test_startup_reader_ignores_old_command_without_valid_handoff(rig):
    seq = send(rig.root, 'tooltip_preview', {'camera': 'front', 'on': True})
    path = rig.root / 'state/tooltip_handoff.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'pid': -1, 'seq': 0}))
    reader, transferred = startup_reader(rig.root, rig.sup.events)
    assert not transferred and reader.last_seq == seq
    assert reader.poll({'tooltip_preview'}) is None


def test_failed_camera_release_does_not_prevent_child_start(rig, monkeypatch):
    issue(rig, 'tooltip_preview', {'camera': 'front', 'on': True})
    wait_for(lambda: ('open', 'front') in rig.history)
    host = rig.sup.tooltip_host
    events = []
    monkeypatch.setattr(rig.sup.events, 'emit', lambda name, **fields: events.append((name, fields)))
    camera = host.worker.cameras['front']
    original_close = camera.close
    monkeypatch.setattr(camera, 'close', lambda: None)
    spawned = []
    monkeypatch.setattr('nest_clear.supervisor.runner.subprocess.Popen', lambda *a, **kw: spawned.append(True) or SimpleNamespace(pid=42, poll=lambda: None))
    rig.sup._start()
    assert spawned and rig.sup.child.pid == 42
    assert rig.sup._tooltip_release_thread.is_alive()
    assert any(name == 'error' and fields.get('code') == 'TOOLTIP-RELEASE-TIMEOUT' and 'front' in fields['camera'] and fields['duration_s'] >= 1 and fields.get('hint') for name, fields in events)
    original_close()
    rig.sup._tooltip_release_thread.join(3)
    assert not rig.sup._tooltip_release_thread.is_alive()
    assert any(name == 'tooltip_release_complete' for name, fields in events)


def test_child_exit_does_not_replay_its_last_command(rig, monkeypatch):
    rig.sup.child = SimpleNamespace(pid=42, poll=lambda: 0)
    monkeypatch.setattr(rig.sup, '_reply_after_failure', lambda reason: None)
    seq = send(rig.root, 'tooltip_preview', {'camera': 'front', 'on': True})
    assert not rig.sup._running()
    rig.sup._tooltip_tick()
    assert rig.sup.tooltip_host is None and not rig.history
    state = read(rig.root / 'state/supervisor_tooltip_status.json')
    assert state['command_seq'] == seq and not state['baseline']['preview']


def test_camera_exclusive_lock_and_release(tmp_path):
    from nest_clear.devices.calib_cameras import CalibCamera
    config = {'backend': 'mock', '_ownership_dir': str(tmp_path), 'device': 'same-camera',
              'settings_path': str(tmp_path / 'missing.yaml')}
    first = CalibCamera('front', config, lambda: [0, 0, 0])
    second = CalibCamera('front', config, lambda: [0, 0, 0])
    first.open()
    try:
        with pytest.raises(Exception, match='仍被父程序预览占用'):
            second.open()
    finally:
        first.close()
    second.open()
    second.close()


def test_camera_ownership_is_shared_across_processes(tmp_path):
    import subprocess
    import sys
    from nest_clear.devices.calib_cameras import CalibCamera
    config = {'backend': 'mock', '_ownership_dir': str(tmp_path), 'device': 'physical-device',
              'settings_path': str(tmp_path / 'missing.yaml')}
    owner = CalibCamera('front', config, lambda: [0, 0, 0])
    script = '''import json,sys
from nest_clear.devices.calib_cameras import CalibCamera
camera = CalibCamera('front', json.loads(sys.argv[1]), lambda: [0,0,0])
try:
    camera.open()
except Exception as exc:
    print(exc)
    sys.exit(7)
camera.close()
'''
    owner.open()
    try:
        blocked = subprocess.run([sys.executable, '-c', script, json.dumps(config)], capture_output=True, text=True, timeout=10)
        assert blocked.returncode == 7 and '仍被父程序预览占用' in blocked.stdout
    finally:
        owner.close()
    released = subprocess.run([sys.executable, '-c', script, json.dumps(config)], capture_output=True, text=True, timeout=10)
    assert released.returncode == 0, released.stderr


def test_preview_reports_camera_held_by_other_process(tmp_path, monkeypatch):
    from nest_clear.baseline.preparation import Preparation
    from nest_clear.devices.calib_cameras import CalibCamera
    from nest_clear.record.events import EventRecorder
    from nest_clear.workers.calib_camera import CalibCameraWorker
    monkeypatch.setattr('nest_clear.baseline.preparation.enumerate_capture_devices', lambda: [])
    settings = {'front': {'backend': 'mock', 'device': 'shared-front', 'settings_path': str(tmp_path / 'cameras.yaml')},
                'side': {'backend': 'mock', 'device': 'other-side', 'settings_path': str(tmp_path / 'cameras.yaml')}}
    events = EventRecorder(tmp_path, 'test')
    worker = CalibCameraWorker(settings, events=events)
    owner = CalibCamera('front', worker.cameras['front'].config, lambda: [0, 0, 0])
    owner.open()
    prep = Preparation(events, worker)
    try:
        prep.command('tooltip_preview', {'camera': 'front', 'on': True})
        wait_for(lambda: prep.state['front'].get('error'))
        assert '仍被父程序预览占用' in prep.state['front']['error']
    finally:
        prep.stop_previews(('front', 'side'))
        prep.close()
        owner.close()
