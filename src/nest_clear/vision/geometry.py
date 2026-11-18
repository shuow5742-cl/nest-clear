"""非远心相机的去畸变射线与模组工作平面求交。"""
from dataclasses import dataclass
import logging
from pathlib import Path

import cv2
import numpy as np
import yaml

from nest_clear.guard import GuardError

logger = logging.getLogger("nest_clear.vision.geometry")


@dataclass(frozen=True)
class Calibration:
    """保存相机到模组的固定位姿以及全局工作平面高度。"""
    K: np.ndarray
    distortion: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    plane_z: float
    placeholder: bool = False
    plane_point: np.ndarray | None = None
    plane_normal: np.ndarray | None = None

    def __post_init__(self):
        point = np.asarray([0., 0., self.plane_z] if self.plane_point is None else self.plane_point, dtype=float)
        normal = np.asarray([0., 0., 1.] if self.plane_normal is None else self.plane_normal, dtype=float)
        if point.shape != (3,) or normal.shape != (3,) or not np.isfinite(point).all() or not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-12:
            raise ValueError("工作平面点或法向量无效")
        object.__setattr__(self, "plane_point", point)
        object.__setattr__(self, "plane_normal", normal / np.linalg.norm(normal))


def load_calibration(path: str | Path) -> Calibration:
    """加载并验证标定数据，拒绝形状或旋转矩阵错误。"""
    try:
        with open(path, encoding="utf-8") as file:
            data = yaml.safe_load(file)
        k = np.asarray(data["K"], dtype=float)
        d = np.asarray(data["distortion"], dtype=float)
        r = np.asarray(data["rotation"], dtype=float)
        t = np.asarray(data["translation"], dtype=float)
        z = float(data["plane_z"])
        placeholder = data["placeholder"]
        if k.shape != (3, 3) or d.shape not in ((4,), (5,), (8,), (12,), (14,)) or r.shape != (3, 3) or t.shape != (3,) or type(placeholder) is not bool:
            raise ValueError("矩阵或畸变系数形状错误")
        if not all(np.isfinite(a).all() for a in (k, d, r, t)) or not np.isfinite(z) or k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]) or not np.allclose(r.T @ r, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(r), 1, atol=1e-6):
            raise ValueError("内参、旋转或数值无效")
        result = Calibration(k, d, r, t, z, placeholder, data.get("plane_point"), data.get("plane_normal"))
    except (OSError, yaml.YAMLError, KeyError, TypeError, ValueError) as exc:
        raise GuardError("CALIBRATION-INVALID", f"标定文件无效：{exc}", str(path), "检查内参、畸变、外参及工作平面。") from exc
    if placeholder:
        logger.warning("相机标定为占位值，未标定：%s", path)
    return result


def pixel_to_module(u: float, v: float, pose: tuple[float, float, float], calibration: Calibration) -> tuple[float, float, float]:
    """去畸变后将相机射线转换为模组系，并与固定高度的平面求交。"""
    try:
        if len(pose) != 3 or not np.isfinite([u, v, *pose]).all():
            raise ValueError("像素或实时坐标不是有限数")
        xy = cv2.undistortPoints(np.array([[[u, v]]], dtype=np.float64), calibration.K, calibration.distortion)[0, 0]
        ray = calibration.rotation @ np.array([xy[0], xy[1], 1.0])
        origin = np.asarray(pose, dtype=float) + calibration.translation
        denom = calibration.plane_normal @ ray
        if abs(denom) < 1e-12:
            raise ValueError("射线与工作平面平行")
        distance = (calibration.plane_normal @ (calibration.plane_point - origin)) / denom
        if distance <= 0:
            raise ValueError(f"平面不在相机前方，射线参数 {distance}")
        result = origin + distance * ray
        if not np.isfinite(result).all():
            raise ValueError("交点不是有限数")
        return tuple(float(x) for x in result)
    except (ValueError, TypeError, cv2.error) as exc:
        raise GuardError("GEOMETRY-INTERSECTION", f"像素 {u}, {v} 换算失败：{exc}", f"拍照实时坐标 {pose}", "检查标定、相机方向、工作平面和检测输出。") from exc


def scale_um_per_px(calibration: Calibration, plane_z: float, camera_z: float) -> float:
    """主点处两条正交射线在工作平面的平均物方尺度。"""
    from dataclasses import replace
    import math
    calib = replace(calibration, plane_z=plane_z, plane_point=np.array([0., 0., plane_z]), plane_normal=np.array([0., 0., 1.]))
    u, v = calibration.K[0, 2], calibration.K[1, 2]
    pose = (0., 0., camera_z)
    center = pixel_to_module(u, v, pose, calib)
    sx = 1000 * math.dist(center, pixel_to_module(u + 1, v, pose, calib))
    sy = 1000 * math.dist(center, pixel_to_module(u, v + 1, pose, calib))
    logger.info("主点尺度：u=%.5f、v=%.5f µm/px，方向差=%.5f µm/px", sx, sy, abs(sx-sy))
    return (sx + sy) / 2
