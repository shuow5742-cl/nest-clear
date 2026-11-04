"""在后台线程统一采集正面和侧面的对刀相机图像。"""

from typing import Any, Callable

import numpy as np
from autoweaver.worker import AsyncRequestWorker

from nest_clear.devices.calib_cameras import CalibCamera


class CalibCameraWorker(AsyncRequestWorker):
    """持有两台设备，供后续对刀流程通过异步请求采图。"""

    name = "calib_camera"

    def __init__(
        self,
        cameras: dict[str, dict[str, Any]],
        position: Callable[[], object] | None = None,
        tip_offset: object | None = None,
    ) -> None:
        """构造两视图；模拟参数可注入以支持闭环仿真。"""
        super().__init__()
        self.cameras = {}
        for name in ("front", "side"):
            config = dict(cameras[name])
            if config.get("backend") == "mock":
                config.setdefault("target_uv", [320, 240])
                config.setdefault("p0", [0, 0, 0])
                config.setdefault("jacobian", [[10, 0, 0], [0, 0, 10]] if name == "front" else [[0, 10, 0], [0, 0, 10]])
                if tip_offset is not None:
                    config["tip_offset"] = tip_offset
            self.cameras[name] = CalibCamera(name, config, position or (lambda: [0, 0, 0]))

    def open(self) -> None:
        """预检时打开两台相机；任何失败都释放已打开的设备。"""
        try:
            for camera in self.cameras.values():
                camera.open()
        except BaseException:
            self.on_stop()
            raise

    def execute(self, payload: dict[str, Any]) -> dict[str, np.ndarray]:
        """一次请求依次取得两视图，交给后续对刀算法。"""
        return {name: camera.capture() for name, camera in self.cameras.items()}

    def on_stop(self) -> None:
        """关闭两台相机，兼容预检失败后的重复清理。"""
        try:
            self.cameras["front"].close()
        finally:
            self.cameras["side"].close()
