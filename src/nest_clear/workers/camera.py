"""相机采图守卫及可追溯采图信息。"""
from datetime import datetime, timezone
import itertools
import logging
import queue
import threading
import time
from typing import Any

import numpy as np
from autoweaver.sensor.camera.base import CameraConfig
from autoweaver.sensor.camera.mock import MockCamera
from autoweaver.sensor.camera.daheng import DahengCamera
from autoweaver.worker import AsyncRequestWorker
from nest_clear.guard import GuardError

logger = logging.getLogger(__name__)
_sequence = itertools.count(1)


class CameraWorker(AsyncRequestWorker):
    name = "camera"

    def __init__(self, backend: str, sn: str, camera: Any = None, *, timeout_s: float = 3.0,
                 exposure_us: float | None = None, gain_db: float | None = None,
                 balance_ratio: tuple[float, ...] | None = None, principal: tuple[float, float] | None = None, role: str = "主相机", image_size: tuple[int, int] | None = None) -> None:
        super().__init__()
        self.sn, self.backend, self.timeout_s = sn, backend, timeout_s
        self.settings = {"exposure_us": exposure_us, "gain_db": gain_db, "balance_ratio": balance_ratio}
        self.principal = principal
        self.role = role
        self.needs_reopen = False
        self.size = None
        self.expected_size = image_size
        try:
            self.camera = camera or (MockCamera(CameraConfig(device_sn=sn), width=(image_size or (1280, 960))[0], height=(image_size or (1280, 960))[1]) if backend == "mock" else DahengCamera(CameraConfig(device_sn=sn, trigger_mode=True)))
        except Exception as exc:
            raise GuardError("CAMERA-CREATE", f"相机创建失败：{exc}", f"相机 SN={sn}", "检查大恒 SDK 和序列号。") from exc

    def open(self) -> None:
        try:
            if self.backend == "daheng" and hasattr(self.camera, "_gx"):
                _, devices = self.camera._gx.DeviceManager().update_device_list()
                serials = [str(d.get("sn", d.get("serial_number", "未知"))) for d in devices]
                logger.info("发现相机 SN：%s", serials)
                if self.sn not in serials:
                    raise ValueError(f"期望 SN={self.sn}，实际发现 SN={serials}")
            if self.backend == "daheng" and hasattr(self.camera, "_configure_trigger"):
                # 只配置触发；框架默认会改自动曝光、增益和白平衡。
                self.camera._configure_camera = self.camera._configure_trigger
            if not self.camera.open():
                raise OSError("相机返回打开失败")
            if self.expected_size is not None and not hasattr(self.camera, "get_frame_size"):
                raise ValueError("相机未提供实际图像宽高，无法核对 image_size")
            if self.principal and self.expected_size and not (0 <= self.principal[0] < self.expected_size[0] and 0 <= self.principal[1] < self.expected_size[1]):
                raise ValueError(f"标定主点 {self.principal} 不在配置图像尺寸 {self.expected_size} 内")
            if hasattr(self.camera, "get_frame_size"):
                self.size = self.camera.get_frame_size()
                if self.expected_size is not None and tuple(self.size) != tuple(self.expected_size):
                    raise ValueError(f"相机尺寸不符：配置 {self.expected_size}，实际 {self.size}")
                if self.principal and not (0 <= self.principal[0] < self.size[0] and 0 <= self.principal[1] < self.size[1]):
                    raise ValueError(f"标定主点 {self.principal} 不在相机尺寸 {self.size} 内")
            device = getattr(self.camera, "_cam", None)
            for key, feature in (("exposure_us", "ExposureTime"), ("gain_db", "Gain")):
                wanted = self.settings[key]
                if wanted is None:
                    logger.info("相机 SN=%s %s 未设置", self.sn, key)
                    continue
                if device is not None:
                    auto = getattr(device, "ExposureAuto" if key == "exposure_us" else "GainAuto", None)
                    if auto is not None:
                        auto.set(self.camera._gx.GxAutoEntry.OFF)
                    node = getattr(device, feature)
                    node.set(wanted)
                    actual = node.get()
                else:
                    setattr(self.camera, key, wanted)
                    actual = getattr(self.camera, key)
                if abs(actual - wanted) > max(0.01, abs(wanted) * 0.001):
                    raise ValueError(f"{key} 期望 {wanted}，实际 {actual}")
            if self.settings["balance_ratio"] is not None:
                if device is None:
                    self.camera.balance_ratio = self.settings["balance_ratio"]
                    actual = self.camera.balance_ratio
                else:
                    device.BalanceWhiteAuto.set(self.camera._gx.GxAutoEntry.OFF)
                    actual = []
                    for channel, wanted in zip((self.camera._gx.GxBalanceRatioSelectorEntry.RED, self.camera._gx.GxBalanceRatioSelectorEntry.GREEN, self.camera._gx.GxBalanceRatioSelectorEntry.BLUE), self.settings["balance_ratio"]):
                        device.BalanceRatioSelector.set(channel)
                        device.BalanceRatio.set(wanted)
                        actual.append(device.BalanceRatio.get())
                if any(abs(a-b) > 0.003 for a, b in zip(actual, self.settings["balance_ratio"])):
                    raise ValueError(f"balance_ratio 期望 {self.settings['balance_ratio']}，实际 {actual}")
            else:
                logger.info("相机 SN=%s balance_ratio 未设置", self.sn)
        except Exception as exc:
            raise GuardError("CAMERA-OPEN", f"打开或参数回读失败：{exc}", f"相机 SN={self.sn}", "检查供电、连接、SDK、序列号及相机参数。") from exc

    def execute(self, payload: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        stage = {"coarse": "粗拍", "fine": "精定位"}.get(payload.get("stage"), payload.get("stage", "未知阶段"))
        where = f"相机角色={self.role}，SN={self.sn}，阶段={stage}，Task_ID={payload.get('task_id')}"
        before = {"monotonic": time.monotonic(), "wall": datetime.now(timezone.utc).isoformat()}
        result: queue.Queue = queue.Queue(maxsize=1)
        def capture() -> None:
            try:
                result.put((True, self.camera.snapshot()))
            except Exception as exc:
                result.put((False, exc))
        try:
            if self.needs_reopen:
                raise RuntimeError("上次采图超时，相机需要重开")
            threading.Thread(target=capture, daemon=True).start()
            # gxipy DataStream.get_image() 默认 timeout=1000ms，超时返回 None；
            # Auto_Weaver 的 DahengCamera.snapshot 抛 RuntimeError("Failed to capture image")。
            # capture_timeout_s 是 SDK 超时之外的外层兜底，防止底层调用卡死。
            try:
                ok, image = result.get(timeout=self.timeout_s)
            except queue.Empty:
                self.needs_reopen = True
                raise TimeoutError(f"采图超过 {self.timeout_s} 秒，相机需要重开")
            if not ok:
                if isinstance(image, RuntimeError):
                    messages = {"Failed to capture image": "取图超时", "Frame capture failed: incomplete frame": "残帧", "Failed to convert image to BGR": "转换失败", "Failed to get numpy array": "转换失败"}
                    if str(image) in messages:
                        raise RuntimeError(f"{messages[str(image)]}：{image}") from image
                raise image
            if not isinstance(image, np.ndarray) or image.size == 0:
                raise ValueError("图像为 None、空图或非数组")
            if image.ndim not in (2, 3) or image.dtype != np.uint8:
                raise ValueError(f"图像格式异常：shape={image.shape}，dtype={image.dtype}，期望 uint8 灰度或彩色")
            h, w = image.shape[:2]
            if self.size is not None and (w, h) != tuple(self.size):
                raise ValueError(f"图像尺寸不符：期望 {self.size}，实际 {(w, h)}")
            if self.principal and not (0 <= self.principal[0] < w and 0 <= self.principal[1] < h):
                raise ValueError(f"标定主点 {self.principal} 不在图像尺寸 {(w, h)} 内")
            device = getattr(self.camera, "_cam", None)
            def read(feature: str) -> Any:
                try:
                    return getattr(device, feature).get() if device else getattr(self.camera, {"ExposureTime": "exposure_us", "Gain": "gain_db"}[feature], None)
                except Exception:
                    return None
            return image, {"capture_seq": next(_sequence), "t_before": before,
                           "t_after": {"monotonic": time.monotonic(), "wall": datetime.now(timezone.utc).isoformat()},
                           "frame_id": None, "device_timestamp": None, "frame_metadata_reason": "框架 DahengCamera 未暴露，待 Auto_Weaver 扩展",
                           "exposure_us": read("ExposureTime"), "gain_db": read("Gain"), "settings": self.settings}
        except Exception as exc:
            raise GuardError("CAMERA-CAPTURE", f"采图失败：{exc}", where, "检查相机连接、触发、图像尺寸和标定；超时后重开相机。") from exc

    def on_stop(self) -> None:
        if not self.needs_reopen:
            self.camera.close()
