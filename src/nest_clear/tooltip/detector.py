"""从明亮背景的针尖剪影中检测针尖像素。"""

from dataclasses import dataclass
from typing import Protocol
import cv2
import numpy as np

from .core import _error

SILHOUETTE_VERSION = "silhouette-1"


@dataclass(frozen=True)
class TipDetection:
    """Detected image point, detector confidence and human-readable diagnostic."""

    point_uv: tuple[float, float] | None  # image uv in pixels, or None
    confidence: float  # detector-specific score, not necessarily a probability
    diagnostic: str = ""  # explanatory text
    focused: bool = True
    edge_width_px: float | None = None
    width_px: float | None = None
    coarse: bool = False
    axis_rms_px: float | None = None


class TipDetector(Protocol):
    """Interface for extracting a tool tip from an image."""

    def detect(self, image: np.ndarray) -> TipDetection:
        """处理针尖标定数据，供对刀流程调用。
           Return pixel position or None with confidence and diagnostics."""
        ...


@dataclass(frozen=True)
class SilhouetteTipDetector:
    """Measure a subpixel tip and edge focus from the largest dark component.

    Direction is image-frame (u, v); threshold is an 8-bit intensity and
    min_area is in pixels. Invalid configuration raises GuardError.
    """

    direction: tuple[float, float] = (0.0, 1.0)  # image-frame direction
    threshold: float = 127  # maximum dark intensity, inclusive
    min_area: int = 5  # pixels
    probe_offset_px: int = 7
    max_edge_width_px: float = 6.0

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
        if type(self.probe_offset_px) is not int or self.probe_offset_px <= 0:
            raise _error(f"probe_offset_px 必须为正整数，实际为 {self.probe_offset_px!r}；请检查配置", locals())
        if isinstance(self.max_edge_width_px, bool) or not isinstance(self.max_edge_width_px, (int, float)) or not np.isfinite(self.max_edge_width_px) or self.max_edge_width_px <= 0:
            raise _error(f"max_edge_width_px 必须为有限正数，实际为 {self.max_edge_width_px!r}；请检查配置", locals())
        object.__setattr__(self, "direction", tuple(d / np.linalg.norm(d)))

    def detect(self, image: np.ndarray) -> TipDetection:
        """检测亚像素针尖与清晰度；无效图像触发 GuardError。

        置信度仅表示是否取得位置，并非概率。
        """
        image = np.asarray(image)
        if image.ndim != 2 or image.size == 0 or image.dtype != np.uint8:
            raise _error("image must be a nonempty uint8 grayscale array", locals())
        mask = (image <= self.threshold).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count <= 1:
            return TipDetection(None, 0.0, "no dark component", False)
        h = image.shape[0]
        candidates = [
            i for i in range(1, count)
            if not (
                self.direction[1] > 0.5
                and int(stats[i, cv2.CC_STAT_TOP]) > 2
                and int(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]) >= h - 1
            )
        ]
        if not candidates:
            return TipDetection(None, 0.0, "no dark component", False)
        index = max(candidates, key=lambda i: int(stats[i, cv2.CC_STAT_AREA]))
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < self.min_area:
            return TipDetection(None, 0.0, "largest component is too small", False)
        y, x = np.nonzero(labels == index)
        points = np.column_stack((x, y))
        projection = points @ np.asarray(self.direction)
        d = np.asarray(self.direction)
        n = np.array([-d[1], d[0]])
        end = projection.max()
        tip_pixels = points[projection >= end - 0.5]
        center = float(np.mean(tip_pixels @ n))
        span = float(np.hypot(*image.shape))
        step = 0.1

        def profile(origin, axis, lo, hi):
            t = np.arange(lo, hi + step / 2, step)
            xy = origin[None, :] + t[:, None] * axis
            if np.any(xy[:, 0] < 0) or np.any(xy[:, 0] > image.shape[1] - 1) or np.any(xy[:, 1] < 0) or np.any(xy[:, 1] > image.shape[0] - 1):
                return None, None
            vals = cv2.remap(image.astype(np.float32), xy[:, 0].astype(np.float32).reshape(1, -1), xy[:, 1].astype(np.float32).reshape(1, -1), cv2.INTER_LINEAR).ravel()
            return t, vals

        def crossing(t, vals, level, rising):
            indices = np.flatnonzero((vals[:-1] <= level) & (vals[1:] > level) if rising else (vals[:-1] > level) & (vals[1:] <= level))
            if not len(indices):
                return None
            i = indices[-1] if rising else indices[0]
            return float(t[i] + (level - vals[i]) * (t[i + 1] - t[i]) / (vals[i + 1] - vals[i]))

        def fallback(reason):
            # 末端 20% 的区域给出可用于大步纠偏的保守估计。
            selected = points[projection >= end - 0.2 * (end - projection.min())]
            return TipDetection(tuple(map(float, selected.mean(axis=0))), 1.0, reason + "；使用末端20%质心作为退化粗位置", False, None, None, True)

        # 初始截面以分割极值回退，随后在截面中心线上精确求末端。
        for _ in range(2):
            origin = d * (end - self.probe_offset_px) + n * center
            # 将采样限制在图像内，截面两端必须有背景。
            ts = np.arange(-span, span + step, step)
            xy = origin + ts[:, None] * n
            valid = (xy[:, 0] >= 0) & (xy[:, 0] <= image.shape[1]-1) & (xy[:, 1] >= 0) & (xy[:, 1] <= image.shape[0]-1)
            ts = ts[valid]
            if len(ts) < 3:
                return TipDetection(None, 0.0, "宽度截面越出图像", False)
            t, vals = profile(origin, n, ts[0] + step, ts[-1] - step)
            if t is None or len(t) < 3:
                return TipDetection(None, 0.0, "宽度截面越出图像", False)
            dark = np.flatnonzero(vals <= self.threshold)
            if not len(dark) or dark[0] == 0 or dark[-1] == len(vals)-1:
                return TipDetection(None, 0.0, "宽度截面缺少两条阈值边缘", False)
            left = crossing(t[:dark[0]+2], vals[:dark[0]+2], self.threshold, False)
            right = crossing(t[dark[-1]-1:], vals[dark[-1]-1:], self.threshold, True)
            if left is None or right is None:
                return TipDetection(None, 0.0, "宽度截面缺少两条阈值边缘", False)
            center += (left + right) / 2
            width = right - left
            origin_tip = n * center
            ttip, vtip = profile(origin_tip, d, end - self.probe_offset_px, end + min(20, self.probe_offset_px * 2))
            if ttip is None:
                return TipDetection(None, 0.0, "针尖中心线越出图像", False)
            refined = crossing(ttip, vtip, self.threshold, True)
            if refined is None:
                return fallback("针尖中心线没有阈值穿越")
            end = refined

        # 两端背景取各自边缘外侧的中位数，针体取截面中心附近中位数；
        # 分别在每条边寻找 10%/90% 灰度穿越，取两侧较大值。
        body = float(np.median(vals[(t >= left + width * .35) & (t <= right - width * .35)]))
        # 窄针尖截面两侧的模糊边缘可能重叠，导致中心灰度高于真实针体，
        # 以该灰度归一化会低估边缘宽度。用连通域灰度中位数估计针体，
        # 若截面中心保留的暗度不足整体针体对比度的 85%，不判为清晰。
        component_body = float(np.median(image[labels == index]))
        background_samples = np.concatenate((vals[t < left - 2], vals[t > right + 2]))
        if len(background_samples) >= 6:
            background = float(np.median(background_samples))
            if background > component_body + 1 and background - body < .85 * (background - component_body):
                point = d * end + n * center
                return TipDetection(tuple(map(float, point)), 1.0,
                                    "截面过窄或边缘重叠，针体中心灰度不稳定，无法可靠测量清晰度",
                                    False, None, width, False)
        widths = []
        for edge, side in ((left, -1), (right, 1)):
            outside = vals[t < left - 2] if side < 0 else vals[t > right + 2]
            if len(outside) < 3:
                return fallback("截面背景不足，无法测量边缘清晰度")
            background = float(np.median(outside))
            if background <= body + 1:
                return fallback("截面灰度对比不足，无法测量边缘清晰度")
            subset = t < (left + right) / 2 if side < 0 else t > (left + right) / 2
            et, ev = t[subset], vals[subset]
            low = crossing(et, ev, body + .1 * (background-body), side > 0)
            high = crossing(et, ev, body + .9 * (background-body), side > 0)
            if low is None or high is None:
                return fallback("边缘过渡不完整，无法测量清晰度")
            widths.append(abs(high-low))
        edge_width = max(widths)
        # Measure shaft straightness in axial bins; exclude the ends of the
        # component, where round caps and the tip can change the profile.
        bins = np.rint(projection - projection.min()).astype(int)
        rows = np.arange(bins.max() + 1)
        lo = np.full(len(rows), np.inf)
        hi = np.full(len(rows), -np.inf)
        lateral = points @ n
        np.minimum.at(lo, bins, lateral)
        np.maximum.at(hi, bins, lateral)
        body_rows = (rows >= rows[-1]*.25) & (rows <= rows[-1]*.75) & np.isfinite(lo)
        rms = None
        if np.count_nonzero(body_rows) >= 3:
            centers = (lo[body_rows] + hi[body_rows]) / 2
            fit_rows = rows[body_rows]
            slope, intercept = np.polyfit(fit_rows, centers, 1)
            rms = float(np.sqrt(np.mean((centers - (slope*fit_rows + intercept))**2)))
        point = d * end + n * center
        return TipDetection(tuple(map(float, point)), 1.0, f"component area: {area} pixels", edge_width <= self.max_edge_width_px, edge_width, width, False, rms)
