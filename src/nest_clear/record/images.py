"""后台归档原图与检测叠加图，并清理过期图片。"""

from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
import logging
from pathlib import Path
import shutil
import threading
from typing import Any

import cv2

from nest_clear.paths import daily_file
from .events import EventRecorder

logger = logging.getLogger("nest_clear.record.images")


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

    def raw(self, image: Any, scan: int, task_id: int) -> str | None:
        """拍照成功后立即提交原图并返回相对路径。"""
        return self._submit(image, None, scan, task_id, None)

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
            logger.warning("提交图像存档失败：%s", exc)
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
            logger.warning("图像存档失败 %s：%s", path, exc)

    def close(self) -> None:
        """等待已提交的图像写完。"""
        self.executor.shutdown(wait=True)
