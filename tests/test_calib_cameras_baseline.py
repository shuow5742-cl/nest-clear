"""相机故障守卫及不可识别剪影测试。"""

import pytest

from nest_clear.devices.calib_cameras import CalibCamera
from nest_clear.guard import GuardError
from nest_clear.tooltip import SilhouetteTipDetector


def test_uvc_open_failure(monkeypatch) -> None:
    """模拟 UVC 打开失败，检查身份与现场排查建议。"""

    class ClosedDevice:
        """无硬件时模拟已关闭的 UVC 设备。"""

        def isOpened(self) -> bool:
            """返回设备无法打开，触发守卫。"""
            return False

        def release(self) -> None:
            """允许失败分支正常释放资源。"""
            return None

    monkeypatch.setattr("cv2.VideoCapture", lambda device: ClosedDevice())
    with pytest.raises(GuardError) as error:
        CalibCamera("front", {"backend": "uvc", "device": 0}).open()
    assert "front" in str(error.value) and "uvc" in str(error.value) and "检查" in str(error.value)


def test_daheng_open_failure(monkeypatch) -> None:
    """模拟大恒 SDK 打开失败，不依赖现场 SDK 或相机。"""

    def fail_open(self) -> bool:
        """模拟相机打开失败，验证外层诊断。"""
        return False

    monkeypatch.setattr("autoweaver.sensor.camera.daheng.DahengCamera.open", fail_open)
    with pytest.raises(GuardError) as error:
        CalibCamera("side", {"backend": "daheng", "sn": "test"}).open()
    assert "side" in str(error.value) and "daheng" in str(error.value) and "检查" in str(error.value)


def test_mock_open_failure() -> None:
    """mock 缺失坐标来源时，报告视图、后端及建议。"""
    with pytest.raises(GuardError) as error:
        CalibCamera("front", {"backend": "mock"}).open()
    assert "front" in str(error.value) and "mock" in str(error.value) and "检查" in str(error.value)


def test_mock_undetectable_tip() -> None:
    """注入认不出针尖的全亮画面，检测器应返回无目标。"""
    camera = CalibCamera("front", {"backend": "mock", "undetectable": True}, lambda: [0, 0, 0])
    camera.open()
    try:
        result = SilhouetteTipDetector().detect(camera.capture())
        assert result.point_uv is None
        assert result.diagnostic
    finally:
        camera.close()
