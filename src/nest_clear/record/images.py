"""后台归档原图与检测叠加图，并清理过期图片。"""

from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
import logging
import os
from queue import Empty, Full, Queue
from pathlib import Path
import shutil
import threading
from typing import Any

import cv2
import numpy as np

from nest_clear.paths import daily_file
from .events import EventRecorder

logger = logging.getLogger("nest_clear.record.images")


def record_image_error(events: EventRecorder, image_name: str, exc: Exception,
                       task_id: int | None = None) -> None:
    """记录画面异常；事件记录失败也不能阻断业务应答。"""
    reason = f"{image_name}失败：{type(exc).__name__}: {exc}"
    logger.error("%s，Task_ID=%s", reason, task_id)
    try:
        events.emit("error", operation="image", image=image_name, task_id=task_id, reason=reason)
    except Exception:
        logger.exception("图像错误事件记录失败：%s", image_name)


class ImageArchive:
    """限制待写图片总数，防止磁盘慢时占满内存。"""

    def __init__(
        self,
        events: EventRecorder,
        enabled: bool,
        image_format: str,
        quality: int,
        retention_days: int,
        queue_size: int = 32,
    ) -> None:
        """清理过期图片，队列容量包括正在写入的图片。"""
        self.events = events
        self.enabled = enabled
        self.image_format = image_format
        self.quality = quality
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.slots = threading.BoundedSemaphore(queue_size)
        self.preview_queue: Queue[np.ndarray | None] = Queue(maxsize=1)
        self.preview_thread: threading.Thread | None = None
        self.preview_lock = threading.Lock()
        removed = []
        cutoff = date.today() - timedelta(days=retention_days)
        for directory in events.directory.glob("????-??-??"):
            try:
                day = date.fromisoformat(directory.name)
                images = directory / "images"
                if day < cutoff and images.is_dir() and not images.is_symlink():
                    shutil.rmtree(images)
                    removed.append(directory.name)
            except (ValueError, OSError) as exc:
                logger.warning("清理图像失败：%s", exc)
        events.emit("images_cleanup", removed=removed, retention_days=retention_days)

    def raw(self, image: Any, scan: int, task_id: int, stage: str | None = None) -> str | None:
        """拍照成功后立即提交原图并返回相对路径。"""
        stem = f"{self.events.tray_id}_{scan:03d}_{stage}_{self.events.clock().strftime('%H%M%S_%f')}" if stage else None
        return self._submit(image, None, scan, task_id, None, stem)

    def overlay(
        self,
        image: Any,
        targets: list[Any],
        scan: int,
        task_id: int,
        raw_path: str | None,
    ) -> str | None:
        """检测成功后另行提交叠加图，与原图使用同一文件名前缀。"""
        return self._submit(image, targets, scan, task_id, raw_path)

    def rendered(self, image: np.ndarray, scan: int, task_id: int, raw_path: str | None,
                 preview: bool = False) -> None:
        """异步归档画面及最新 JPEG，预览交给独立线程且只保留最新帧。"""
        self._submit(image, [], scan, task_id, raw_path)
        self.executor.submit(self._write_live, image.copy())
        if preview:
            self._offer_preview(image)

    def _write_live(self, image: np.ndarray) -> None:
        """在归档线程原子替换最新画面，失败时只记录警告。"""
        try:
            path = daily_file(self.events.directory, "live", self.events.clock().date()) / "latest.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            ok, data = cv2.imencode(".jpg", image)
            if not ok:
                raise OSError("JPEG 编码失败")
            temp = path.with_suffix(".tmp")
            temp.write_bytes(data.tobytes())
            os.replace(temp, path)
        except Exception as exc:
            record_image_error(self.events, "最新画面", exc)

    def _offer_preview(self, image: np.ndarray) -> None:
        """无等待地替换待预览帧，窗口卡顿时丢弃旧帧而不阻塞检测。"""
        with self.preview_lock:
            if self.preview_thread is None:
                self.preview_thread = threading.Thread(target=self._preview_loop, name="vision-preview", daemon=True)
                self.preview_thread.start()
            try:
                self.preview_queue.put_nowait(image.copy())
            except Full:
                try:
                    self.preview_queue.get_nowait()
                except Empty:
                    pass
                try:
                    self.preview_queue.put_nowait(image.copy())
                except Full:
                    pass

    def _preview_loop(self) -> None:
        """只在独立守护线程调用 OpenCV 窗口，失败不影响检测及图片归档。"""
        try:
            if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
                raise RuntimeError("没有可用的图形显示环境")
            while True:
                image = self.preview_queue.get()
                if image is None:
                    break
                cv2.imshow("nest-clear", image)
                cv2.waitKey(1)
        except Exception as exc:
            logger.warning("预览窗口失败：%s", exc)
        finally:
            try:
                cv2.destroyWindow("nest-clear")
            except Exception:
                pass

    def calibration(self, image: Any, tool: int, task_id: int, iteration: int,
                    view: str, point: tuple[float, float] | None = None,
                    target: tuple[float, float] | None = None, raw_path: str | None = None) -> str | None:
        """提交对刀原图或针尖与目标像素叠加图，沿用同一限流队列。"""
        stem = f"calib_{tool}_{task_id}_{iteration}_{view}_{self.events.clock().strftime('%H%M%S_%f')}"
        overlay = None if point is None else [(point, target)]
        return self._submit(image, overlay, iteration, task_id, raw_path, stem)

    def _submit(
        self,
        image: Any,
        targets: list[Any] | None,
        scan: int,
        task_id: int,
        raw_path: str | None,
        stem: str | None = None,
    ) -> str | None:
        """无等待地申请图片槽位，队列满时记录丢弃。"""
        if not self.enabled:
            return None
        now = self.events.clock()
        kind = "raw" if targets is None else "overlay"
        if raw_path:
            relative = Path(raw_path.replace(f"_raw.{self.image_format}", f"_{kind}.{self.image_format}"))
        else:
            stem = stem or f"{self.events.tray_id}_{scan:03d}_{now.strftime('%H%M%S_%f')}"
            path = (
                daily_file(self.events.directory, "images", now.date()) / f"{stem}_{kind}.{self.image_format}"
            )
            relative = path.relative_to(self.events.directory)
        if not self.slots.acquire(blocking=False):
            logger.warning("图片队列已满，丢弃%s：扫描位 %s，Task_ID=%s", kind, scan, task_id)
            self.events.emit(
                "image_dropped",
                kind=kind,
                path=str(relative),
                scan_index=scan,
                task_id=task_id,
            )
            return None
        try:
            future = self.executor.submit(
                self._write,
                self.events.directory / relative,
                image.copy(),
                None if targets is None else list(targets),
            )
            future.add_done_callback(lambda future: self.slots.release())
        except Exception as exc:
            self.slots.release()
            record_image_error(self.events, str(relative), exc, task_id)
            self.events.emit("image_dropped", kind=kind, path=str(relative), scan_index=scan, task_id=task_id)
            return None
        self.events.emit("image_submitted", kind=kind, path=str(relative), scan_index=scan, task_id=task_id)
        return str(relative)

    def _write(self, path: Path, image: Any, targets: list[Any] | None) -> None:
        """压缩写原图或带序号、分数的叠加图，失败只告警。"""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if targets is not None:
                if image.ndim == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                for index, target in enumerate(targets, 1):
                    if isinstance(target, tuple):
                        point_uv, target_uv = target
                        point = tuple(round(x) for x in point_uv)
                        destination = tuple(round(x) for x in target_uv)
                        cv2.circle(image, point, 5, (0, 0, 255), 2)
                        cv2.circle(image, destination, 5, (0, 255, 0), 2)
                        cv2.line(image, point, destination, (255, 0, 0), 2)
                        text = f"du={target_uv[0] - point_uv[0]:.1f} dv={target_uv[1] - point_uv[1]:.1f} px"
                        cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        continue
                    pixel = target.pixel
                    point = (round(pixel.u), round(pixel.v))
                    cv2.circle(image, point, 5, (0, 0, 255), 2)
                    cv2.putText(
                        image,
                        f"{index} {pixel.score:.2f}",
                        point,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                    )
            if self.image_format == "png":
                params = [cv2.IMWRITE_PNG_COMPRESSION, self.quality]
            else:
                params = [cv2.IMWRITE_JPEG_QUALITY, self.quality]
            if not cv2.imwrite(str(path), image, params):
                raise OSError("图像编码或写入失败")
        except Exception as exc:
            record_image_error(self.events, str(path), exc)

    def close(self) -> None:
        """等待已提交的图像写完。"""
        self.executor.shutdown(wait=True)
        if self.preview_thread is not None and self.preview_thread.is_alive():
            try:
                self.preview_queue.put_nowait(None)
            except Full:
                try:
                    self.preview_queue.get_nowait()
                    self.preview_queue.put_nowait(None)
                except (Empty, Full):
                    pass
            # GUI 后端可能卡住；守护线程不能阻止主程序退出。
            self.preview_thread.join(timeout=0.1)
