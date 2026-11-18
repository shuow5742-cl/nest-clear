"""精定位坐标及工作面框内的候选挑选。"""
from dataclasses import dataclass, replace
import math
from typing import Callable, Literal

import cv2
import numpy as np

from .geometry import Calibration, pixel_to_module
from .hair import HairCandidate, HairDetectConfig
from .hair.core import _hair_pick_point


@dataclass(frozen=True)
class FineResult:
    status: Literal["ok", "fine_empty", "fine_point_outside", "fine_rejected"]
    candidate: HairCandidate | None = None
    pick_uv: tuple[float, float] | None = None
    target_xyz: tuple[float, float, float] | None = None
    offset_mm: float | None = None
    identity: tuple[float, float, float] | None = None
    box_candidate_count: int = 0


def fine_camera_xy(target_xyz: tuple[float, float, float], pose: tuple[float, float, float],
                   calibration: Calibration) -> tuple[float, float, float]:
    """将目标移到标定主点射线下，保持拍照高度。"""
    center = pixel_to_module(calibration.K[0, 2], calibration.K[1, 2], pose, calibration)
    return (pose[0] + target_xyz[0] - center[0],
            pose[1] + target_xyz[1] - center[1], pose[2])


def select_fine_target(image: np.ndarray, candidates: list[HairCandidate],
                       pose: tuple[float, float, float], calibration: Calibration, *,
                       box_half_mm: float, is_attempted: Callable[[HairCandidate], bool],
                       passes_gate: Callable[[HairCandidate], bool],
                       cfg: HairDetectConfig) -> FineResult:
    """按工作面方框裁剪候选，重新取点并按原轮廓物理面积排序。"""
    if not math.isfinite(box_half_mm) or box_half_mm < 0:
        raise ValueError("box_half_mm must be finite and nonnegative")
    center = pixel_to_module(calibration.K[0, 2], calibration.K[1, 2], pose, calibration)
    intersected = False
    eligible = False
    box_candidate_count = 0
    best = None
    h, w = image.shape[:2]
    for c in candidates:
        if len(c.polygon) < 3:
            continue
        polygon = np.asarray(c.polygon, dtype=np.int32)
        x, y, width, height = cv2.boundingRect(polygon)
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x + width), min(h, y + height)
        if x0 >= x1 or y0 >= y1:
            continue
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(mask, [polygon - [x0, y0]], 255)
        ys, xs = np.nonzero(mask)
        # 批量去畸变和求交，与 pixel_to_module 使用相同的射线公式。
        uv = np.column_stack((xs + x0, ys + y0)).astype(np.float64).reshape(-1, 1, 2)
        rays_xy = cv2.undistortPoints(uv, calibration.K, calibration.distortion).reshape(-1, 2)
        rays = np.column_stack((rays_xy, np.ones(len(uv)))) @ calibration.rotation.T
        origin = np.asarray(pose, dtype=float) + calibration.translation
        with np.errstate(divide="ignore", invalid="ignore"):
            distance = (calibration.plane_z - origin[2]) / rays[:, 2]
            xy = origin[:2] + distance[:, None] * rays[:, :2]
        inside = (np.isfinite(xy).all(axis=1) & (distance > 0) &
                  (np.abs(rays[:, 2]) >= 1e-12) &
                  (np.abs(xy[:, 0] - center[0]) <= box_half_mm) &
                  (np.abs(xy[:, 1] - center[1]) <= box_half_mm))
        if not inside.any():
            continue
        box_candidate_count += 1
        intersected = True
        if not passes_gate(c) or is_attempted(c):
            continue
        eligible = True
        mask[ys[~inside], xs[~inside]] = 0
        # 加零边界，避免实心裁剪区域的距离变换没有背景。
        padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        dist = cv2.distanceTransform(padded, cv2.DIST_L2, 5)
        px, py, _, _ = _hair_pick_point(padded, dist, cfg=cfg)
        uv_pick = (float(px + x0 - 1), float(py + y0 - 1))
        xyz = pixel_to_module(*uv_pick, pose, calibration)
        if abs(xyz[0] - center[0]) > box_half_mm or abs(xyz[1] - center[1]) > box_half_mm:
            continue
        points = [pixel_to_module(*p, pose, calibration) for p in c.polygon]
        area = abs(sum(a[0] * b[1] - b[0] * a[1]
                       for a, b in zip(points, points[1:] + points[:1]))) / 2
        original = pixel_to_module(*c.pick_uv, pose, calibration)
        result = FineResult("ok", c, uv_pick, xyz, math.dist(xyz[:2], original[:2]), pixel_to_module(*c.identity_uv, pose, calibration))
        if best is None or area > best[0]:
            best = (area, result)
    if best is not None:
        return replace(best[1], box_candidate_count=box_candidate_count)
    return FineResult("fine_point_outside" if eligible else
                      "fine_rejected" if intersected else "fine_empty", box_candidate_count=box_candidate_count)
