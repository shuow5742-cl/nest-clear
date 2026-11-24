"""在后台线程统一采集正面和侧面的对刀相机图像。"""

from typing import Any, Callable
import threading

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
        for name in ("front", "side"):
            config = dict(cameras[name])
            if config.get("backend") == "mock":
                config.setdefault("target_uv", [320, 240])
                config.setdefault("p0", [0, 0, 0])
                config.setdefault("jacobian", [[10, 0, 0], [0, 0, 10]] if name == "front" else [[0, 10, 0], [0, 0, 10]])
                if tip_offset is not None:
                    config["tip_offset"] = tip_offset
            self.cameras[name] = CalibCamera(name, config, position or (lambda: [0, 0, 0]))

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
        self.opened = True

    def execute(self, payload: dict[str, Any]) -> dict[str, np.ndarray]:
        """一次请求依次取得两视图，交给后续对刀算法。"""
        self.open()
        result = CaptureImages()
        for name, camera in self.cameras.items():
            done = threading.Event()
            self.capture_done[name] = done
            class Adapter:
                def snapshot(self):
                    try:
                        return camera.capture()
                    finally:
                        done.set()
                snapshot_with_meta = CameraBase.snapshot_with_meta
            worker = CameraWorker("mock", camera.config.get("sn", f"{camera.backend}:{camera.config.get('device', name)}"), Adapter(), timeout_s=self.timeout_s, role=f"对刀 {name}")
            worker.needs_reopen = name in self.blocked
            try:
                stage = "preview" if payload.get("stage") == "baseline_preview" else "对刀"
                result[name], meta = worker.execute({"stage": stage, "task_id": payload.get("task_id")})
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
                raise self._camera_error(exc) from exc
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
