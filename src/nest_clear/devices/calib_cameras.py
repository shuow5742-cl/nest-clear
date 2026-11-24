"""两台对刀相机统一采集接口及模拟、大恒和 UVC 后端。"""

from typing import Any, Callable
import errno
import fcntl
import os
import struct
from pathlib import Path
from .v4l2_controls import V4L2Controls, DEFAULTS
from .tooltip_camera_store import load_settings
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
        self.controls = None
        self.actual_settings = {}
        self.exposure_status = "未测曝光"

    def open(self) -> None:
        """打开指定设备，失败时报告相机身份和现场检查项。"""
        try:
            if self.backend == "mock":
                if self.position is None:
                    raise ValueError("mock 需要坐标来源")
            elif self.backend == "uvc":
                device = self.config["device"]
                label = "主" if self.name == "front" else "辅助"
                if isinstance(device, str) and device.startswith("待接入"):
                    raise ValueError(f"{label}对刀相机未配置")
                path = f"/dev/video{device}" if isinstance(device, int) else device
                if not os.path.exists(path):
                    raise FileNotFoundError(f"未找到{label}对刀相机：{path}，检查线缆与插口")
                fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
                try:
                    cap = bytearray(104)
                    fcntl.ioctl(fd, 0x80685600, cap, True)
                    capabilities, device_caps = struct.unpack_from('II', cap, 84)
                    effective = device_caps if capabilities & 0x80000000 else capabilities
                    if not effective & (0x1 | 0x1000): raise ValueError(f"{path} 不是视频 capture 节点（可能是元数据节点）")
                finally: os.close(fd)
                self.device = cv2.VideoCapture(device, cv2.CAP_V4L2)
                if not self.device.isOpened(): raise OSError("VideoCapture 无法打开")
                settings = {**DEFAULTS, **self.config.get("camera_controls", {})}
                self.device.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
                self.device.set(cv2.CAP_PROP_FRAME_WIDTH, settings['width'])
                self.device.set(cv2.CAP_PROP_FRAME_HEIGHT, settings['height'])
                ok, _ = self.device.read()
                if not ok: raise OSError("启动取流失败；检查 USB 带宽和连接")
                actual_format = int(self.device.get(cv2.CAP_PROP_FOURCC))
                width, height = int(self.device.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.device.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if (width, height, actual_format) != (settings['width'], settings['height'], cv2.VideoWriter_fourcc(*'YUYV')):
                    raise ValueError(f"格式回读不一致：期望 YUYV {settings['width']}x{settings['height']}，实际 fourcc={actual_format} {width}x{height}")
                store = self.config.get('settings_path', 'config/calibration/tooltip_cameras.yaml')
                saved = load_settings(store)['cameras'].get(self.name, {})
                if saved.get('measured') and saved.get('device') == device and type(saved.get('exposure')) is int:
                    settings['exposure'] = saved['exposure']; self.exposure_status = '已测曝光'
                self.controls = V4L2Controls(path)
                def settle():
                    for _ in range(3): self.capture()
                actual = self.controls.apply(settings, settle)
                self.actual_settings = {'device': device, 'format': 'YUYV', 'width': width, 'height': height, 'controls': actual, 'exposure_status': self.exposure_status}
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
            if (isinstance(exc, OSError) and exc.errno == errno.ENOSPC) or "No space left on device" in str(exc):
                exc = OSError("USB 带宽不足：两台对刀相机须插在不同 USB 控制器上")
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

    def set_exposure(self, value: int) -> None:
        if self.controls is None: raise RuntimeError("相机控制未打开")
        self.controls.set('exposure', value)
        actual = self.controls.get('exposure')
        if actual != value: raise ValueError(f"exposure 期望 {value}，实际 {actual}")
        self.actual_settings['controls']['exposure'] = actual

    def get_exposure(self) -> int:
        if self.controls is None: raise RuntimeError("相机控制未打开")
        return self.controls.get('exposure')

    def settings_snapshot(self) -> dict:
        from copy import deepcopy
        return deepcopy(self.actual_settings)

    def close(self) -> None:
        """释放相机资源。"""
        if self.controls is not None:
            self.controls.close()
            self.controls = None
        if self.device is not None:
            if self.backend == "uvc":
                self.device.release()
            else:
                self.device.close()
            self.device = None
