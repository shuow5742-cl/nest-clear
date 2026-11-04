"""从明亮背景的针尖剪影中检测针尖像素。"""

from dataclasses import dataclass
from typing import Protocol
import cv2
import numpy as np

from .core import _error


@dataclass(frozen=True)
class TipDetection:
    """Detected image point, detector confidence and human-readable diagnostic."""

    point_uv: tuple[float, float] | None  # image uv in pixels, or None
    confidence: float  # detector-specific score, not necessarily a probability
    diagnostic: str = ""  # explanatory text


class TipDetector(Protocol):
    """Interface for extracting a tool tip from an image."""

    def detect(self, image: np.ndarray) -> TipDetection:
        """处理针尖标定数据，供对刀流程调用。
           Return pixel position or None with confidence and diagnostics."""
        ...


@dataclass(frozen=True)
class SilhouetteTipDetector:
    """Find the directional extreme of the largest dark grayscale component.

    Direction is image-frame (u, v); threshold is an 8-bit intensity and
    min_area is in pixels. Invalid configuration raises GuardError.
    """

    direction: tuple[float, float] = (0.0, 1.0)  # image-frame direction
    threshold: float = 127  # maximum dark intensity, inclusive
    min_area: int = 5  # pixels

    def __post_init__(self) -> None:
        """校验针尖方向与分割参数，供检测使用。"""
        try:
            d = np.asarray(self.direction, dtype=float)
        except (TypeError, ValueError) as exc:
            raise _error("direction must be a numeric vector", locals()) from exc
        if d.shape != (2,) or not np.all(np.isfinite(d)) or np.linalg.norm(d) == 0:
            raise _error("direction must be a finite nonzero (u, v) vector", locals())
        if not np.isfinite(self.threshold) or not 0 <= self.threshold <= 255:
            raise _error("threshold must be in [0, 255]", locals())
        if type(self.min_area) is not int or self.min_area < 1:
            raise _error("min_area must be a positive integer", locals())
        object.__setattr__(self, "direction", tuple(d / np.linalg.norm(d)))

    def detect(self, image: np.ndarray) -> TipDetection:
        """处理针尖标定数据，供对刀流程调用。
           Locate an extreme in a uint8 grayscale image, or return no point.

        Tied extreme pixels are averaged. Confidence is 1 or 0 and does not
        estimate probability. Invalid image shape or dtype raises GuardError.
        """
        image = np.asarray(image)
        if image.ndim != 2 or image.size == 0 or image.dtype != np.uint8:
            raise _error("image must be a nonempty uint8 grayscale array", locals())
        mask = (image <= self.threshold).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count <= 1:
            return TipDetection(None, 0.0, "no dark component")
        index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < self.min_area:
            return TipDetection(None, 0.0, "largest component is too small")
        y, x = np.nonzero(labels == index)
        points = np.column_stack((x, y))
        projection = points @ np.asarray(self.direction)
        # 平端可能有多个极值像素，取均值作为针尖中心。
        tip = points[np.isclose(projection, projection.max(), rtol=0, atol=1e-9)].mean(axis=0)
        return TipDetection(tuple(tip), 1.0, f"component area: {area} pixels")
