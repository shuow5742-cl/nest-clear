"""通过 ArUco 定位打印靶，拟合圆环并以十字交叉检查。"""
from dataclasses import dataclass
import math

import cv2
import numpy as np

from nest_clear.vision.geometry import pixel_to_module


@dataclass
class Target:
    index: int
    ring_uv: tuple
    cross_uv: tuple
    cross_error_mm: float
    ellipse: tuple
    marker: list

    @property
    def label(self):
        return f'T{self.index:02d}'


def detect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    corners, ids, _ = cv2.aruco.ArucoDetector(dictionary).detectMarkers(gray)
    if ids is None:
        return []
    result = []
    for raw, marker_id in zip(corners, ids.ravel()):
        if not 0 <= marker_id < 32:
            continue
        c = raw.reshape(4, 2).astype(float)
        right = ((c[1] - c[0]) + (c[2] - c[3])) / 8
        down = ((c[3] - c[0]) + (c[2] - c[1])) / 8
        scale = (np.linalg.norm(right) + np.linalg.norm(down)) / 2
        if scale < 2:
            continue
        center = c.mean(axis=0) - 7 * right + 7 * down
        # 将局部靶面变换到每毫米 20 像素、十字水平垂直的坐标系。
        extent, ppmm = 6., 20
        side = int(2 * extent * ppmm + 1)
        basis = np.column_stack((right, down)) / ppmm
        origin = center - basis @ np.array([extent * ppmm, extent * ppmm])
        affine = np.column_stack((basis, origin)).astype(np.float32)
        patch = cv2.warpAffine(gray, affine, (side, side), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=255)
        edges = cv2.Canny(patch, 40, 120)
        yy, xx = np.indices(edges.shape)
        rad = np.hypot(xx - extent * ppmm, yy - extent * ppmm)
        mask = (edges > 0) & (rad > 1.48 * ppmm) & (rad < 2.22 * ppmm)
        # 排除十字与圆环的四个交点。
        mask &= (np.abs(xx - extent * ppmm) > .27 * ppmm) & (np.abs(yy - extent * ppmm) > .27 * ppmm)
        pts = np.column_stack((xx[mask], yy[mask])).astype(np.float32)
        if len(pts) < 30:
            continue
        ellipse = cv2.fitEllipse(pts.reshape(-1, 1, 2))
        (ex, ey), (a, b), angle = ellipse
        if not (3.0 * ppmm < a < 4.7 * ppmm and 3.0 * ppmm < b < 4.7 * ppmm):
            continue
        dark = 255 - patch.astype(float)
        def center_line(horizontal):
            # 十字臂位于圆环之外，取两侧积分以抑制印刷噪声。
            along = np.abs((xx if horizontal else yy) - extent * ppmm)
            across = (yy if horizontal else xx) - extent * ppmm
            values = []
            for delta in range(-int(.6 * ppmm), int(.6 * ppmm) + 1):
                values.append(dark[(along > 2.5 * ppmm) & (along < 4.8 * ppmm) & (across == delta)].mean())
            peak = int(np.argmax(values))
            indices = np.arange(max(0, peak-2), min(len(values), peak+3))
            weights = np.maximum(np.asarray(values)[indices] - np.median(values), 0)
            return (float(np.average(indices, weights=weights)) if weights.sum() else float(peak)) - int(.6 * ppmm)
        cx = extent * ppmm + center_line(False)
        cy = extent * ppmm + center_line(True)
        ring = origin + basis @ np.array([ex, ey])
        cross = origin + basis @ np.array([cx, cy])
        error = math.hypot(ex - cx, ey - cy) / ppmm
        result.append(Target(int(marker_id), tuple(map(float, ring)), tuple(map(float, cross)), float(error),
                             ((float(ex), float(ey)), (float(a), float(b)), float(angle)), c.tolist()))
    return result


def module_points(targets, pose, calibration):
    return {t.index: pixel_to_module(*t.ring_uv, pose, calibration) for t in targets}
