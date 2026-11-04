"""验证双相机 Worker 及主程序的设备预检和释放。"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from nest_clear.config import load_config
from nest_clear.guard import GuardError
from nest_clear.main import Application
from nest_clear.workers.calib_camera import CalibCameraWorker
from sim.fake_plc import FakePLC

ROOT = Path(__file__).resolve().parents[1]


def test_capture_and_mock_offset() -> None:
    """两视图均返回图像，注入针尖偏差后两视图的针尖随之移动。"""
    configs = {name: {"backend": "mock"} for name in ("front", "side")}
    position = [0, 0, 0]
    original = CalibCameraWorker(configs, lambda: position)
    shifted = CalibCameraWorker(configs, lambda: position, [1, 2, 0])
    original.open()
    shifted.open()
    try:
        before = original.execute({})
        after = shifted.execute({})
        assert set(before) == {"front", "side"}
        for name, shift in (("front", 10), ("side", 20)):
            assert before[name].shape == (480, 640)
            assert np.where(after[name] == 0)[1].mean() - np.where(before[name] == 0)[1].mean() == shift
        position[2] = 1
        assert np.where(original.execute({})["front"] == 0)[0].mean() > np.where(before["front"] == 0)[0].mean()
    finally:
        original.on_stop()
        shifted.on_stop()


def test_capture_failure_identifies_camera(monkeypatch: pytest.MonkeyPatch) -> None:
    """单台采图失败时保留视图、后端和现场排查建议。"""
    worker = CalibCameraWorker({name: {"backend": "mock"} for name in ("front", "side")})
    worker.open()
    monkeypatch.setattr(worker.cameras["side"], "position", lambda: 1 / 0)
    try:
        with pytest.raises(GuardError) as error:
            worker.execute({})
        assert all(text in str(error.value) for text in ("side", "mock", "检查"))
    finally:
        worker.on_stop()


def test_open_failure_releases_first_camera(monkeypatch: pytest.MonkeyPatch) -> None:
    """侧面相机预检失败时释放已打开的正面设备。"""
    devices = []

    class Device:
        """模拟按顺序打开成功及失败的 UVC 设备。"""

        def __init__(self, index: int) -> None:
            """记录设备索引及释放状态。"""
            self.index = index
            self.released = False
            devices.append(self)

        def isOpened(self) -> bool:
            """仅正面设备可打开。"""
            return self.index == 0

        def release(self) -> None:
            """记录设备已释放。"""
            self.released = True

    monkeypatch.setattr("cv2.VideoCapture", Device)
    worker = CalibCameraWorker({name: {"backend": "uvc", "device": index} for index, name in enumerate(("front", "side"))})
    with pytest.raises(GuardError) as error:
        worker.open()
    assert all(text in str(error.value) for text in ("side", "uvc", "检查"))
    assert all(device.released for device in devices)


def test_application_attaches_and_closes_calibration_cameras(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """启用时预检并装配两台模拟相机，退出时关闭它们。"""
    closed = []
    from nest_clear.devices.calib_cameras import CalibCamera

    original_close = CalibCamera.close

    def close(camera: CalibCamera) -> None:
        """记录关闭动作并保留真实清理逻辑。"""
        closed.append(camera.name)
        original_close(camera)

    monkeypatch.setattr(CalibCamera, "close", close)
    with FakePLC() as plc:
        config = load_config(ROOT / "config/nest_clear.yaml")
        config = replace(
            config,
            plc=replace(config.plc, ip=plc.host, port=plc.port),
            vision=replace(config.vision, camera_backend="mock"),
            tool_calibration=replace(config.tool_calibration, cameras={name: {"backend": "mock"} for name in ("front", "side")}),
            record_directory=str(tmp_path),
        )
        app = Application(config, ROOT / "config/nest_clear.yaml")
        assert app.calib_camera is not None
        app.close()
        assert sorted(closed) == ["front", "side"]


@pytest.mark.parametrize("enabled", [True, False])
def test_application_camera_preflight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    """启用时预检失败并报错；关闭时不创建或打开对刀设备。"""
    calls = []

    def fail_open(self: object) -> None:
        """模拟指定设备无法打开并记录预检调用。"""
        calls.append(self)
        raise GuardError("TEST", "side 打开失败", "后端 uvc", "检查连接")

    monkeypatch.setattr("nest_clear.devices.calib_cameras.CalibCamera.open", fail_open)
    with FakePLC() as plc:
        config = load_config(ROOT / "config/nest_clear.yaml")
        config = replace(
            config,
            plc=replace(config.plc, ip=plc.host, port=plc.port),
            vision=replace(config.vision, camera_backend="mock"),
            tool_calibration=replace(config.tool_calibration, enabled=enabled),
            record_directory=str(tmp_path),
        )
        if enabled:
            with pytest.raises(GuardError, match="打开失败"):
                Application(config, ROOT / "config/nest_clear.yaml")
            assert len(calls) == 1
        else:
            app = Application(config, ROOT / "config/nest_clear.yaml")
            try:
                assert app.calib_camera is None
                assert not calls
            finally:
                app.close()
