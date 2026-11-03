"""包装框架 Mock 或按序列号打开的大恒相机。"""
from typing import Any

from autoweaver.sensor.camera.base import CameraConfig
from autoweaver.sensor.camera.mock import MockCamera
from autoweaver.sensor.camera.daheng import DahengCamera
from nest_clear.guard import GuardError
from .async_worker import AsyncWorker


class CameraWorker(AsyncWorker):
    """以异步 note 执行相机采图，避免 SDK 阻塞 tick。"""
    def __init__(self, backend: str, sn: str, camera: Any = None) -> None:
        """保存相机参数；允许测试注入兼容相机。"""
        super().__init__("camera")
        try:
            self.camera = camera or (MockCamera(CameraConfig(device_sn=sn)) if backend == "mock" else DahengCamera(CameraConfig(device_sn=sn, trigger_mode=True)))
        except Exception as exc:
            raise GuardError("CAMERA-CREATE", f"相机创建失败：{exc}", f"相机 SN={sn}", "检查大恒 SDK、相机模式和序列号。") from exc
        self.sn = sn

    def open(self) -> None:
        """启动预检时打开相机，失败时提供现场排查信息。"""
        try:
            if not self.camera.open():
                raise OSError("相机返回打开失败")
        except Exception as exc:
            raise GuardError("CAMERA-OPEN", f"打开失败：{exc}", f"相机 SN={self.sn}", "检查相机供电、连接、SDK 和序列号。") from exc

    def execute(self, payload: dict[str, Any]) -> Any:
        """在后台线程采集单张图像。"""
        return self.camera.snapshot()

    def on_stop(self) -> None:
        """关闭设备，供时钟退出时释放相机。"""
        self.camera.close()
