"""在后台线程统一采集正面和侧面的对刀相机图像。"""

import json
import logging
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Callable

from nest_clear.devices.v4l2_enum import resolve_tooltip_roles, usb_port

import cv2
import numpy as np
from autoweaver.worker import AsyncRequestWorker
from autoweaver.sensor.camera.base import CameraBase

from nest_clear.devices.calib_cameras import CalibCamera
from nest_clear.guard import GuardError
from nest_clear.workers.camera import CameraWorker
from nest_clear.record.images import tray_number


class CaptureImages(dict):
    """图片与已经归档的路径，保持原有按视图访问接口。"""
    def __init__(self) -> None:
        super().__init__()
        self.raw_paths: dict[str, str | None] = {}
        self.frames: dict[str, list[np.ndarray]] = {}
        self.capture_times: dict[str, list[float]] = {}


class CalibCameraWorker(AsyncRequestWorker):
    """持有两台设备，供后续对刀流程通过异步请求采图。"""

    name = "calib_camera"

    def __init__(
        self,
        cameras: dict[str, dict[str, Any]],
        position: Callable[[], object] | None = None,
        tip_offset: object | None = None,
        timeout_s: float = 3.0,
        events: Any = None,
    ) -> None:
        """构造两视图；模拟参数可注入以支持闭环仿真。"""
        super().__init__()
        self.cameras = {}
        self.timeout_s = timeout_s
        self.events = events
        self.blocked = set()
        self.capture_done = {}
        self.opened = False
        self._roles_lock = threading.RLock()
        self._last_roles = None
        for name in ("front", "side"):
            config = dict(cameras[name])
            if events is not None and getattr(events, 'directory', None) is not None:
                config['_ownership_dir'] = str(events.directory)
            if config.get("backend") == "uvc":
                config['_role_resolver'] = self.resolve_roles
            if config.get("backend") == "mock":
                config.setdefault("target_uv", [320, 240])
                config.setdefault("p0", [0, 0, 0])
                config.setdefault("jacobian", [[10, 0, 0], [0, 0, 10]] if name == "front" else [[0, 10, 0], [0, 0, 10]])
                if tip_offset is not None:
                    config["tip_offset"] = tip_offset
            self.cameras[name] = CalibCamera(name, config, position or (lambda: [0, 0, 0]))

    def resolve_roles(self):
        with self._roles_lock:
            cfg = self.cameras['front'].config
            roles = resolve_tooltip_roles(cfg.get('usb_vendor_id', '05a3'), cfg.get('usb_product_id', '9230'))
            ports = {name: usb_port(path)[1] for name, path in roles.items()}
            previous = self._last_roles
            directory = getattr(self.events, 'directory', None)
            state_path = Path(directory) / 'state' / 'tooltip_roles.json' if directory else None
            if previous is None and state_path:
                try: previous = json.loads(state_path.read_text(encoding='utf-8'))
                except (OSError, ValueError): pass
            if previous and previous != ports:
                message = f'对刀相机插口角色已变化：{previous} → {ports}'
                logging.getLogger(__name__).warning(message)
                if self.events: self.events.emit('warning', reason=message)
            if self.events: self.events.emit('tooltip_roles', **ports)
            self._last_roles = ports
            if state_path:
                state_path.parent.mkdir(parents=True, exist_ok=True)
                temp_name = None
                try:
                    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=state_path.parent,
                                                     prefix='.tooltip_roles_', delete=False) as stream:
                        temp_name = stream.name
                        json.dump(ports, stream, ensure_ascii=False)
                    os.replace(temp_name, state_path)
                finally:
                    if temp_name and os.path.exists(temp_name): os.unlink(temp_name)
            return roles

    def _camera_error(self, exc: Exception) -> GuardError:
        detail = str(exc).replace("\n", "；")
        message = f"对刀相机未接入或打不开：{detail}；请接好正面/侧面相机后重试"
        if self.events:
            self.events.emit("warning", reason=message)
        return GuardError("CALIB-CAMERA-UNAVAILABLE", message, "对刀相机", "请检查相机连接。")

    def open(self) -> None:
        """预检时打开两台相机；任何失败都释放已打开的设备。"""
        if self.opened:
            return
        if any(getattr(camera, 'backend', None) == 'uvc' for camera in self.cameras.values()):
            try:
                roles = self.resolve_roles()
            except Exception as exc:
                self.on_stop()
                raise self._camera_error(exc) from exc
            if any(getattr(camera, 'backend', None) == 'uvc' and camera.device is not None and
                   camera.actual_settings.get('device') != roles[name]
                   for name, camera in self.cameras.items()):
                self.on_stop()
        for name in list(self.blocked):
            if self.capture_done[name].is_set():
                self.cameras[name].close()
                self.blocked.remove(name)
        if self.blocked:
            raise self._camera_error(RuntimeError('上次采图超时，底层调用尚未返回，请稍后重试'))
        errors = []
        for camera in self.cameras.values():
            if camera.device is not None:
                continue
            try:
                camera.open()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise self._camera_error(RuntimeError("；".join(str(error) for error in errors))) from errors[0]
        if any(getattr(camera, 'backend', None) == 'uvc' and
               camera.actual_settings.get('device') != roles[name]
               for name, camera in self.cameras.items()):
            self.on_stop()
            raise self._camera_error(RuntimeError('对刀相机插口在打开期间发生变化，请重试'))
        self.opened = True

    def execute(self, payload: dict[str, Any]) -> CaptureImages:
        """取得两视图；五帧请求只返回在该请求之后发起的新取图。"""
        count = payload.get("frames_per_view", 1)
        requested = payload.get("capture_after")
        self.open()
        result = CaptureImages()
        for name, camera in self.cameras.items():
            done = threading.Event()
            self.capture_done[name] = done

            class Adapter:
                def __init__(self, camera, done):
                    self.camera = camera
                    self.done = done

                def snapshot(self):
                    try:
                        return self.camera.capture()
                    finally:
                        self.done.set()
                snapshot_with_meta = CameraBase.snapshot_with_meta
            worker = CameraWorker("mock", camera.config.get("sn", f"{camera.backend}:{camera.config.get('device', name)}"), Adapter(camera, done), timeout_s=self.timeout_s, role=f"对刀 {name}")
            worker.needs_reopen = name in self.blocked
            try:
                stage = "preview" if payload.get("stage") == "baseline_preview" else "对刀"
                capture_payload = {"stage": stage, "task_id": payload.get("task_id")}
                frame_index = 0
                if requested is not None and time.monotonic() <= requested:
                    raise ValueError("采图请求时间尚未到达")
                if count > 1 and camera.backend == "uvc":
                    # The exclusive worker has no continuous preview stream to
                    # drain V4L2. Discard the queued buffers and one in-flight
                    # exposure before collecting count distinct reads.
                    buffers = int(camera.device.get(cv2.CAP_PROP_BUFFERSIZE)) or 4
                    for _ in range(buffers + 1):
                        done.clear()
                        worker.execute({"stage": "preview", "task_id": payload.get("task_id")})
                frames, starts = [], []
                for frame_index in range(1, count + 1):
                    done.clear()
                    started = time.monotonic()
                    image, frame_meta = worker.execute(capture_payload)
                    frames.append(image)
                    starts.append(started)
                    if frame_index == 1:
                        meta = frame_meta
                result[name] = frames[0]
                result.frames[name] = frames
                result.capture_times[name] = starts
                if str(payload.get("stage", "")).startswith("baseline_"):
                    continue
                if self.events and self.events.run_directory is not None:
                    sequence, capture_id = self.events.next_capture()
                    archive = getattr(self.events, "archive", None)
                    tray = self.events.tray_id or "none"
                    iteration = payload.get("iteration", 1)
                    kind, shot, view = f"calib_{name}", f"i{iteration:02d}", payload.get("view", 0)
                    stem = f"{sequence:06d}_{kind}_t{tray_number(tray)}_v{view:02d}_{shot}"
                    base = self.events.run_directory.relative_to(self.events.directory) / "images"
                    enabled = archive is not None and archive.enabled
                    meta["capture_seq"] = sequence
                    self.events.emit("capture", capture_id=capture_id, xid=self.events.xid, kind=kind, tray=tray,
                                     view=view, shot=shot, task_id=payload.get("task_id"),
                                     request_func=payload.get("request_func"), actual_xyz=payload.get("actual"),
                                     target_xyz=payload.get("target"), raw_path=str(base / f"{stem}_raw.png") if enabled else None,
                                     ovl_path=str(base / f"{stem}_ovl.jpg") if enabled else None,
                                     queued_wall=self.events.clock().isoformat(), **meta)
                    if archive:
                        try:
                            result.raw_paths[name] = archive.capture_raw(result[name], sequence=sequence,
                                capture_id=capture_id, kind=kind, tray=tray, view=view, shot=shot,
                                task_id=payload.get("task_id"))
                        except Exception:
                            import logging
                            logging.getLogger(__name__).exception("对刀原图归档失败")
                elif self.events:
                    self.events.emit("capture", task_id=payload.get("task_id"), stage="对刀", role=name, **meta)
            except Exception as exc:
                if worker.needs_reopen:
                    self.blocked.add(name)
                self.on_stop()
                label = "正面" if name == "front" else "侧面"
                detail = f"{label}第 {frame_index} 帧" if frame_index else f"{label}取流"
                raise self._camera_error(RuntimeError(f"{detail}：{exc}")) from exc
            finally:
                if worker.needs_reopen:
                    self.blocked.add(name)
        return result

    def on_stop(self) -> None:
        """关闭两台相机，兼容预检失败后的重复清理。"""
        for name, camera in self.cameras.items():
            if name not in self.blocked:
                camera.close()
        self.opened = False
