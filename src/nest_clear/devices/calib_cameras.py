"""两台对刀相机统一采集接口及模拟、大恒和 UVC 后端。"""

from typing import Any, Callable
import cv2
import numpy as np

from nest_clear.guard import GuardError


class CalibCamera:
    """封装单视图设备，图像统一输出为灰度 uint8。"""

    def __init__(self, name: str, config: dict, position: Callable[[], object] | None = None) -> None:
        """保存视图配置以及模拟时的平台坐标来源。"""
        self.name = name
        self.config = config
        self.backend = config.get("backend")
        self.position = position
        self.device: Any = None
        self.rng = np.random.default_rng(config.get("seed", 0))

    def open(self) -> None:
        """打开指定设备，失败时报告相机身份和现场检查项。"""
        try:
            if self.backend == "mock":
                if self.position is None:
                    raise ValueError("mock 需要坐标来源")
            elif self.backend == "uvc":
                self.device = cv2.VideoCapture(self.config["device"])
                if not self.device.isOpened():
                    raise OSError("VideoCapture 无法打开")
            elif self.backend == "daheng":
                from autoweaver.sensor.camera.base import CameraConfig
                from autoweaver.sensor.camera.daheng import DahengCamera

                self.device = DahengCamera(CameraConfig(device_sn=self.config["sn"], trigger_mode=True))
                if not self.device.open():
                    raise OSError("大恒相机返回打开失败")
            else:
                raise ValueError(f"未知后端 {self.backend!r}")
        except Exception as exc:
            self.close()
            raise GuardError(
                "CALIB-CAMERA-OPEN",
                f"{self.name} 相机打开失败：{exc}",
                f"标定相机 {self.name}，后端 {self.backend}",
                "检查供电、连线、设备索引或 SN、访问权限及大恒 SDK。",
            ) from exc

    def capture(self) -> np.ndarray:
        """采集一张原始灰度图；模拟剪影随平台位姿移动。"""
        try:
            if self.backend == "mock":
                size = self.config.get("size", [480, 640])
                image = np.full(size, 255, dtype=np.uint8)
                if not self.config.get("undetectable", False):
                    position = np.asarray(self.position(), dtype=float)
                    point = np.asarray(self.config["target_uv"], dtype=float) + np.asarray(
                        self.config["jacobian"], dtype=float
                    ) @ (
                        position
                        - np.asarray(self.config["p0"], dtype=float)
                        + np.asarray(self.config.get("tip_offset", [0, 0, 0]), dtype=float)
                    )
                    point += self.rng.normal(0, self.config.get("noise_px", 0), 2)
                    x, y = np.rint(point).astype(int)
                    cv2.line(image, (x, y - 35), (x, y), 0, 3)
                return image
            if self.backend == "uvc":
                ok, image = self.device.read()
                if not ok:
                    raise OSError("读取帧失败")
            else:
                image = self.device.snapshot()
            image = np.asarray(image)
            if image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if image.ndim != 2 or image.dtype != np.uint8 or not image.size:
                raise ValueError("图像不是有效的 8 位灰度图")
            return image
        except Exception as exc:
            raise GuardError(
                "CALIB-CAMERA-CAPTURE",
                f"{self.name} 采图失败：{exc}",
                f"后端 {self.backend}",
                "检查相机连接、触发模式和图像格式。",
            ) from exc

    def close(self) -> None:
        """释放相机资源。"""
        if self.device is not None:
            if self.backend == "uvc":
                self.device.release()
            else:
                self.device.close()
            self.device = None
