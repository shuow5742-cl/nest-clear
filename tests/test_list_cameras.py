"""大恒相机枚举命令不要求测试环境安装 SDK。"""
import sys
from types import SimpleNamespace

from nest_clear.tools import list_cameras


def test_enumerates_camera_details(monkeypatch, capsys):
    manager = SimpleNamespace(update_device_list=lambda: (1, [{"sn": "SN123", "model_name": "M", "device_class": "USB"}]))
    monkeypatch.setitem(sys.modules, "gxipy", SimpleNamespace(DeviceManager=lambda: manager))
    assert list_cameras.main() == 0
    output = capsys.readouterr().out
    assert all(value in output for value in ("SN123", "M", "USB"))


def test_missing_sdk_is_clear(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "gxipy", None)
    assert list_cameras.main() == 1
    assert "gxipy" in capsys.readouterr().err


def test_resolve_unique_camera(monkeypatch):
    import pytest
    from nest_clear.devices.camera_select import resolve_camera
    from nest_clear.guard import GuardError
    from nest_clear.tools import list_cameras as module
    monkeypatch.setattr(module, 'list_cameras', lambda: [{'sn': 'serial-1'}])
    assert resolve_camera() == 'serial-1'
    for devices in ([], [{'sn': 'a'}, {'sn': 'b'}], [{'sn': ''}]):
        monkeypatch.setattr(module, 'list_cameras', lambda: devices)
        with pytest.raises(GuardError, match='恰好一台'):
            resolve_camera()
