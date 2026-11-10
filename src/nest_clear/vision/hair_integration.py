"""将毛候选投影到工作平面、应用物理门并绘制操作画面。"""
from dataclasses import dataclass
import math

import cv2
import numpy as np

from .detector import ModuleTarget, PixelTarget
from .geometry import Calibration, pixel_to_module
from .hair import HairCandidate, HairDetectConfig


@dataclass
class Observation:
    """关联一个候选的扎点、身份、物理尺寸和当前帧状态。"""

    target: ModuleTarget  # 扎点像素及其模组 XYZ，供工具补偿后下发
    identity: tuple[float, float, float]  # bbox 中心投影后的模组 XYZ，供跨帧匹配
    candidate: HairCandidate  # 原始轮廓、暗度和像素尺寸
    length: float  # 局部尺度近似的长度，mm
    area: float  # 投影轮廓的面积，mm²；stub 用分数作为排序键
    reason: str | None  # 物理门拒绝原因；通过时为 None
    status: str = "eligible"  # eligible、rejected、failed 或 attempted


def measure(c: HairCandidate, pose: tuple[float, float, float], calibration: Calibration,
            cfg: HairDetectConfig) -> Observation:
    """投影轮廓求面积；长度用中心水平、垂直局部尺度均值近似。"""
    def project(point: tuple[float, float]) -> tuple[float, float, float]:
        """用拍照时实时模组坐标把一个像素投影到工作平面。"""
        return pixel_to_module(*point, pose, calibration)

    center = project(c.identity_uv)
    x = project((c.identity_uv[0] + 1, c.identity_uv[1]))
    y = project((c.identity_uv[0], c.identity_uv[1] + 1))
    scale = (math.dist(center[:2], x[:2]) + math.dist(center[:2], y[:2])) / 2
    points = [project(p) for p in c.polygon]
    area = abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:] + points[:1]))) / 2
    length = c.length_px * scale
    width = area / length if length > 0 else 0
    reason = ("short" if length < cfg.min_length_mm else
              "small" if area < cfg.min_area_mm2 else
              "thin" if width < cfg.min_width_mm else
              "pale" if (c.gray_ratio is not None and c.dark_frac is not None and
                         (c.gray_ratio > cfg.max_gray_ratio or c.dark_frac < cfg.min_dark_frac)) else None)
    pixel = PixelTarget(*c.pick_uv, 1.0, c)
    return Observation(ModuleTarget(pixel, project(c.pick_uv)), center, c, length, area, reason,
                       "rejected" if reason else "eligible")


# OpenCV BGR：绿=要挑，黄=物理门拒绝，红=挑取失败；青=曾判成功但又出现（仍跳过并告警）。
COLORS = {"eligible": (0, 255, 0), "rejected": (0, 255, 255),
          "failed": (0, 0, 255), "attempted": (255, 255, 0)}
LABELS = {"eligible": "PICK", "rejected": "skip", "failed": "FAILED", "attempted": "REAPPEARED"}


def render(image: np.ndarray, observations: list[Observation], selected: Observation | None = None) -> np.ndarray:
    """按画面尺寸缩放标注，显示每个候选和本次选中目标的扎点。"""
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    factor = max(0.5, max(image.shape[:2]) / 1280.0)
    font_scale = 0.5 * factor
    number_scale = font_scale * 2.0
    thickness = max(1, round(factor))
    line_width = max(1, round(2 * factor))
    for observation in observations:
        candidate = observation.candidate
        color = COLORS[observation.status]
        poly = np.asarray(candidate.polygon, dtype=np.int32)
        if len(poly) >= 3:
            layer = image.copy()
            cv2.fillPoly(layer, [poly], color)
            image = cv2.addWeighted(layer, 0.2, image, 0.8, 0)
        x1, y1, x2, y2 = map(round, candidate.bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, line_width)
        label = LABELS[observation.status]
        if observation.reason and observation.status == "rejected":
            label += f" {observation.reason}"
        baseline = max(round(28 * factor), y1 - round(6 * factor))
        left = x1
        # 与一代相同，单独放大长度数字，便于在整幅图上辨认。
        for text, scale, width in ((f"{label} L=", font_scale, thickness),
                                   (f"{observation.length:.2f}", number_scale, line_width),
                                   ("mm", font_scale, thickness)):
            cv2.putText(image, text, (left, baseline), cv2.FONT_HERSHEY_SIMPLEX, scale, color, width, cv2.LINE_AA)
            left += cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, width)[0][0]
        if observation is selected:
            point = tuple(map(round, candidate.pick_uv))
            cv2.drawMarker(image, point, (0, 0, 255), cv2.MARKER_CROSS, max(7, round(18 * factor)), line_width)
            cv2.circle(image, point, max(3, round(6 * factor)), (0, 0, 255), line_width)
    return image
