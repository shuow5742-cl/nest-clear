"""Automatic role selection uses capture capabilities and USB sysfs identity."""
from types import SimpleNamespace
import pytest
from nest_clear.devices import v4l2_enum


@pytest.fixture
def usb_devices(tmp_path, monkeypatch):
    bypath = tmp_path / 'by-path'; bypath.mkdir()
    dev = tmp_path / 'dev'; dev.mkdir()
    sysclass = tmp_path / 'sys' / 'class'; sysclass.mkdir(parents=True)
    sysdevices = tmp_path / 'sys' / 'devices'; sysdevices.mkdir()
    monkeypatch.setattr(v4l2_enum, '_is_capture_device', lambda path: True)
    def add(port, vendor='05a3', product='9230', *, index=None):
        index = len(list(dev.iterdir())) if index is None else index
        node = dev / f'video{index}'; node.touch()
        usb = sysdevices / f'usb{index}'; usb.mkdir()
        (usb / 'idVendor').write_text(vendor); (usb / 'idProduct').write_text(product)
        interface = usb / 'interface'; interface.mkdir()
        entry = sysclass / node.name; entry.mkdir()
        (entry / 'device').symlink_to(interface, target_is_directory=True)
        path = bypath / f'pci-0000:00:14.0-usb-0:{port}:1.0-video-index0'
        path.symlink_to(node)
        return str(path)
    def resolve():
        return v4l2_enum.resolve_tooltip_roles(directory=bypath, sys_class=sysclass)
    return add, resolve, bypath, sysclass


@pytest.mark.parametrize('small,large', [('5', '6'), ('9', '10'), ('1.2', '3')])
def test_numeric_order_and_filter(usb_devices, small, large):
    add, resolve, _, _ = usb_devices
    side = add(large); front = add(small); add('20', vendor='1234')
    assert resolve() == {'front': front, 'side': side}


@pytest.mark.parametrize('port', ['5', '6'])
def test_one_camera_never_assigned(usb_devices, port):
    add, resolve, _, _ = usb_devices
    add(port)
    with pytest.raises(ValueError, match=f'只检测到 1 台对刀相机（usb-0:{port}），需两台都接上才能区分正/侧'):
        resolve()


def test_zero_and_three_cameras(usb_devices):
    add, resolve, _, _ = usb_devices
    with pytest.raises(ValueError, match='未检测到对刀相机'): resolve()
    for port in ('10', '5', '9'): add(port)
    with pytest.raises(ValueError, match='usb-0:5、usb-0:9、usb-0:10'): resolve()


def test_usb_dedup_and_custom_ids(usb_devices):
    add, resolve, bypath, sysclass = usb_devices
    a = add('5', vendor='abcd', product='0001'); b = add('6', vendor='abcd', product='0001')
    (bypath / 'a-usbv2-0:5:1.0-video-index0').symlink_to(__import__('os').path.realpath(a))
    assert v4l2_enum.resolve_tooltip_roles('abcd', '0001', directory=bypath, sys_class=sysclass) == {'front': a, 'side': b}
    with pytest.raises(ValueError, match='未检测到对刀相机'): resolve()


def test_worker_records_role_changes(tmp_path, monkeypatch):
    from nest_clear.workers.calib_camera import CalibCameraWorker
    rows = []
    events = SimpleNamespace(directory=tmp_path, emit=lambda event, **fields: rows.append((event, fields)))
    paths = lambda a, b: {'front': f'/dev/pci-usb-0:{a}:1.0-video-index0', 'side': f'/dev/pci-usb-0:{b}:1.0-video-index0'}
    current = [paths(5, 6)]
    monkeypatch.setattr('nest_clear.workers.calib_camera.resolve_tooltip_roles', lambda *args: current[0])
    worker = CalibCameraWorker({n: {'backend': 'uvc'} for n in ('front', 'side')}, events=events)
    worker.resolve_roles(); worker.resolve_roles()
    assert [row for row in rows if row[0] == 'tooltip_roles'][-1][1] == {'front': 'usb-0:5', 'side': 'usb-0:6'}
    assert not any(row[0] == 'warning' for row in rows)
    current[0] = paths(6, 10); worker.resolve_roles()
    assert any('插口角色已变化' in row[1]['reason'] for row in rows if row[0] == 'warning')
    rows.clear(); current[0] = paths(1, 2)
    restarted = CalibCameraWorker({n: {'backend': 'uvc'} for n in ('front', 'side')}, events=events)
    restarted.resolve_roles()
    assert any('插口角色已变化' in row[1]['reason'] for row in rows if row[0] == 'warning')


def test_uvc_open_ignores_legacy_assignments(tmp_path, monkeypatch):
    import cv2
    import numpy as np
    from nest_clear.devices.calib_cameras import CalibCamera
    from nest_clear.devices.tooltip_camera_store import save_assignments
    a = tmp_path / 'camera-a'; b = tmp_path / 'camera-b'; a.touch(); b.touch()
    settings = tmp_path / 'tooltip_cameras.yaml'
    save_assignments(settings, {'front': '/dev/v4l/by-path/old-video-index0', 'side': None})
    monkeypatch.setattr('nest_clear.devices.calib_cameras.resolve_tooltip_roles', lambda *args: {'front': str(a), 'side': str(b)})
    monkeypatch.setattr('nest_clear.devices.calib_cameras._is_capture_device', lambda path: True)
    opened = []
    class Device:
        def __init__(self, path, backend): opened.append(path)
        def isOpened(self): return True
        def set(self, *args): return True
        def read(self): return True, np.zeros((2, 2), dtype=np.uint8)
        def get(self, prop):
            return {cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*'YUYV'), cv2.CAP_PROP_FRAME_WIDTH: 1920, cv2.CAP_PROP_FRAME_HEIGHT: 1080}[prop]
        def release(self): pass
    class Controls:
        def __init__(self, path): pass
        def apply(self, settings, settle): return settings
        def close(self): pass
    monkeypatch.setattr(cv2, 'VideoCapture', Device)
    monkeypatch.setattr('nest_clear.devices.calib_cameras.V4L2Controls', Controls)
    camera = CalibCamera('front', {'backend': 'uvc', 'device': 'old', 'settings_path': str(settings)})
    camera.open(); camera.close()
    assert opened == [str(a)]


def test_missing_node_and_legacy_devices_ignored(tmp_path, usb_devices):
    import yaml
    from nest_clear.devices.tooltip_camera_store import load_settings
    add, resolve, _, _ = usb_devices
    path = add('5')
    __import__('pathlib').Path(path).unlink()
    with pytest.raises(ValueError, match='未检测到对刀相机'):
        resolve()
    saved = tmp_path / 'tooltip_cameras.yaml'
    saved.write_text(yaml.safe_dump({'devices': 'obsolete', 'cameras': {}}))
    assert load_settings(saved)['devices'] == 'obsolete'


def test_preview_reports_missing_pair_for_both_roles(tmp_path):
    from nest_clear.baseline.preparation import Preparation
    from nest_clear.ui.model import baseline_preview_display
    class Events:
        directory = tmp_path
        def __init__(self): self.rows = []
        def emit(self, event, **fields): self.rows.append((event, fields))
    reason = '只检测到 1 台对刀相机（usb-0:6），需两台都接上才能区分正/侧'
    class Worker:
        cameras = {n: SimpleNamespace(config={'backend': 'uvc'}) for n in ('front', 'side')}
        def resolve_roles(self): raise ValueError(reason)
    events = Events(); prep = Preparation(events, Worker())
    prep._discover()
    for name in ('front', 'side'):
        assert prep.state[name]['device'] is None
        assert baseline_preview_display(prep.state[name])['message'] == reason
    assert any((row[1].get('preview') or {}).get('front', {}).get('error') == reason for row in events.rows)
    prep.close()


def test_legacy_commands_ack_and_reject_in_any_mode(tmp_path):
    from nest_clear.tooloffset.commands import CommandReader, send
    rows = []
    events = SimpleNamespace(emit=lambda event, **fields: rows.append((event, fields)))
    reader = CommandReader(tmp_path, events)
    for name in ('tooltip_assign', 'tooltip_swap'):
        seq = send(tmp_path, name, {})
        assert reader.poll(set(), defer_unallowed=True) is None
        assert reader.last_seq == seq
        assert rows[-2][0] == 'tooloffset_command' and not rows[-2][1]['accepted']
        assert rows[-1][1]['message'] == '对刀相机已按插口固定，不可在界面修改'


def test_configurable_usb_ids(tmp_path):
    import yaml
    from pathlib import Path
    from nest_clear.config import load_config
    data = yaml.safe_load(Path('config/nest_clear.yaml').read_text())
    data['tool_calibration']['usb_vendor_id'] = 'ABCD'
    data['tool_calibration']['usb_product_id'] = '1234'
    path = tmp_path / 'config.yaml'; path.write_text(yaml.safe_dump(data))
    config = load_config(path)
    for camera in config.tool_calibration.cameras.values():
        assert camera['usb_vendor_id'] == 'abcd' and camera['usb_product_id'] == '1234'
    data['tool_calibration']['usb_vendor_id'] = 'not-hex'
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(Exception, match='usb_vendor_id'): load_config(path)
