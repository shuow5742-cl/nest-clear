"""相机采图守卫及可追溯采图信息。"""
from dataclasses import asdict
import itertools
import logging
import queue
import threading
import time
from typing import Any

import numpy as np
from autoweaver.sensor.camera.base import CameraConfig, FrameMeta
from autoweaver.sensor.camera.mock import MockCamera
from autoweaver.sensor.camera.daheng import DahengCamera
from autoweaver.worker import AsyncRequestWorker
from nest_clear.guard import GuardError
from nest_clear.devices.gx_enum import enum_value

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
        self.readback = {"exposure_us": None, "gain_db": None, "balance_ratio": None}
        self.model = None
        self.principal = principal
        self.role = role
        self.needs_reopen = False
        self.capture_lock = threading.Lock()
        self.size = None
        self.expected_size = image_size
        try:
            self.camera = camera or (MockCamera(CameraConfig(device_sn=sn), width=(image_size or (1280, 960))[0], height=(image_size or (1280, 960))[1]) if backend == "mock" else DahengCamera(CameraConfig(device_sn=sn, trigger_mode=True)))
            self._simulate_exposure = isinstance(self.camera, MockCamera)
            self._mock_base_exposure = exposure_us or 10000.0
        except Exception as exc:
            raise GuardError("CAMERA-CREATE", f"相机创建失败：{exc}", f"相机 SN={sn}", "检查大恒 SDK 和序列号。") from exc

    def open(self) -> None:
        try:
            if self.backend == "daheng" and hasattr(self.camera, "_gx"):
                _, devices = self.camera._gx.DeviceManager().update_device_list()
                match = next((d for d in devices if str(d.get("sn", d.get("serial_number", ""))) == self.sn), {})
                self.model = match.get("model_name", match.get("model"))
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
                self.readback[key] = actual
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
                self.readback["balance_ratio"] = list(actual)
            else:
                logger.info("相机 SN=%s balance_ratio 未设置", self.sn)
            for key, feature in (("exposure_us", "ExposureTime"), ("gain_db", "Gain")):
                if self.readback[key] is None:
                    try:
                        self.readback[key] = getattr(device, feature).get() if device else getattr(self.camera, key, None)
                    except Exception:
                        logger.warning("相机 %s %s 回读不可用", self.sn, key)
            if self._simulate_exposure and self.readback['exposure_us'] is None:
                self.camera.exposure_us = self._mock_base_exposure
                self.readback['exposure_us'] = self._mock_base_exposure
            if self.readback["balance_ratio"] is None and device is None:
                self.readback["balance_ratio"] = getattr(self.camera, "balance_ratio", None)
            elif self.readback["balance_ratio"] is None:
                try:
                    selector = device.BalanceRatioSelector
                    previous = enum_value(selector.get())
                    channels = self.camera._gx.GxBalanceRatioSelectorEntry
                    try:
                        values = []
                        for channel in (channels.RED, channels.GREEN, channels.BLUE):
                            selector.set(channel)
                            values.append(device.BalanceRatio.get())
                        self.readback["balance_ratio"] = values
                    finally:
                        selector.set(previous)
                except Exception:
                    logger.warning("相机 %s 白平衡回读不可用", self.sn)
        except Exception as exc:
            raise GuardError("CAMERA-OPEN", f"打开或参数回读失败：{exc}", f"相机 SN={self.sn}", "检查供电、连接、SDK、序列号及相机参数。") from exc

    def exposure_auto_mode(self):
        device = getattr(self.camera, '_cam', None)
        node = getattr(device, 'ExposureAuto', None) if device is not None else None
        return enum_value(node.get()) if node is not None else None

    def restore_exposure_auto_mode(self, value):
        device = getattr(self.camera, '_cam', None)
        node = getattr(device, 'ExposureAuto', None) if device is not None else None
        if node is not None and value is not None:
            node.set(enum_value(value))

    def _set_exposure_locked(self, exposure_us: float) -> float:
        """在已持有 capture_lock 时设置曝光并回读，失败时抛错。"""
        if exposure_us is None or not np.isfinite(exposure_us) or exposure_us <= 0:
            raise ValueError('曝光必须为正的有限数')
        device = getattr(self.camera, '_cam', None)
        if device is not None:
            auto = getattr(device, 'ExposureAuto', None)
            if auto is not None:
                auto.set(self.camera._gx.GxAutoEntry.OFF)
            node = device.ExposureTime
            node.set(float(exposure_us))
            actual = float(node.get())
        else:
            if hasattr(self.camera, 'set_exposure_time'):
                self.camera.set_exposure_time(float(exposure_us))
            self.camera.exposure_us = float(exposure_us)
            actual = float(self.camera.exposure_us)
        if abs(actual - exposure_us) > max(0.01, abs(exposure_us) * .001):
            raise ValueError(f'曝光设置期望 {exposure_us}，实际 {actual}')
        self.readback['exposure_us'] = actual
        return actual

    def set_exposure(self, exposure_us: float) -> float:
        """在无并发采图时设置曝光并回读，失败时抛错。"""
        with self.capture_lock:
            return self._set_exposure_locked(exposure_us)

    def execute(self, payload: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        with self.capture_lock:
            return self._execute(payload)

    def _execute(self, payload: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        stage = {"coarse": "粗拍", "fine": "精定位"}.get(payload.get("stage"), payload.get("stage", "未知阶段"))
        where = f"相机角色={self.role}，SN={self.sn}，阶段={stage}，Task_ID={payload.get('task_id')}"
        result: queue.Queue = queue.Queue(maxsize=1)
        def capture() -> None:
            try:
                if hasattr(self.camera, "snapshot_with_meta"):
                    frame = self.camera.snapshot_with_meta()
                else:
                    # 兼容只实现 snapshot 的现场适配器及对刀模拟相机。
                    wall_before, mono_before = time.time(), time.monotonic()
                    image = self.camera.snapshot()
                    mono_after, wall_after = time.monotonic(), time.time()
                    frame = (image, FrameMeta(pc_time_before=wall_before, pc_time_after=wall_after,
                                              pc_mono_before=mono_before, pc_mono_after=mono_after))
                result.put((True, frame))
            except Exception as exc:
                result.put((False, exc))
        try:
            if self.needs_reopen:
                raise RuntimeError("上次采图超时，相机需要重开")
            if payload.get("operation") == "set_exposure":
                actual = self._set_exposure_locked(float(payload["exposure_us"]))
                return None, {"exposure_us": actual, "settings": self.settings}
            if payload.get("stage") == "depthcal":
                device = getattr(self.camera, "_cam", None)
                if device is not None:
                    auto = self.exposure_auto_mode()
                    if auto is not None and auto != enum_value(self.camera._gx.GxAutoEntry.OFF):
                        raise GuardError("DEPTH-EXPOSURE", "下针深度标定要求关闭自动曝光", where,
                                         "设置主相机固定曝光后重新发起 160。")
                    exposure = float(device.ExposureTime.get())
                    if not np.isfinite(exposure) or exposure <= 0:
                        raise ValueError("主相机固定曝光回读无效")
            threading.Thread(target=capture, daemon=True).start()
            # gxipy DataStream.get_image() 默认 timeout=1000ms，超时返回 None；
            # Auto_Weaver 的 DahengCamera.snapshot 抛 RuntimeError("Failed to capture image")。
            # capture_timeout_s 是 SDK 超时之外的外层兜底，防止底层调用卡死。
            try:
                ok, captured = result.get(timeout=self.timeout_s)
            except queue.Empty:
                self.needs_reopen = True
                raise TimeoutError(f"采图超过 {self.timeout_s} 秒，相机需要重开")
            if not ok:
                if isinstance(captured, RuntimeError):
                    messages = {"Failed to capture image": "取图超时", "Frame capture failed: incomplete frame": "残帧", "Failed to convert image to BGR": "转换失败", "Failed to get numpy array": "转换失败"}
                    if str(captured) in messages:
                        raise RuntimeError(f"{messages[str(captured)]}：{captured}") from captured
                raise captured
            image, frame_meta = captured
            if not isinstance(image, np.ndarray) or image.size == 0:
                raise ValueError("图像为 None、空图或非数组")
            if image.ndim not in (2, 3) or image.dtype != np.uint8:
                raise ValueError(f"图像格式异常：shape={image.shape}，dtype={image.dtype}，期望 uint8 灰度或彩色")
            if self._simulate_exposure:
                actual_exposure = getattr(self.camera, 'exposure_us', None) or self._mock_base_exposure
                image = np.clip(image.astype(np.float32) * (actual_exposure / self._mock_base_exposure), 0, 255).astype(np.uint8)
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
            metadata = asdict(frame_meta)
            # 基类只提供时间；图像尺寸可从实际帧可靠获得。
            if metadata["width"] is None:
                metadata["width"] = w
            if metadata["height"] is None:
                metadata["height"] = h
            for key, feature in (("exposure_us", "ExposureTime"), ("gain_db", "Gain")):
                if metadata[key] is None:
                    metadata[key] = read(feature)
            return image, {"capture_seq": None if payload.get("stage") == "preview" else next(_sequence), **metadata, "settings": self.settings}
        except GuardError:
            raise
        except Exception as exc:
            raise GuardError("CAMERA-CAPTURE", f"采图失败：{exc}", where, "检查相机连接、触发、图像尺寸和标定；超时后重开相机。") from exc

    def on_stop(self) -> None:
        if not self.needs_reopen:
            self.camera.close()
