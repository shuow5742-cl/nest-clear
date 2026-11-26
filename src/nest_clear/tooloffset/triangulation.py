"""主相机多视角视线最小二乘三角定位与下一视角规划。"""
from dataclasses import dataclass, field, replace
import math
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares

from nest_clear.vision.geometry import Calibration, pixel_to_module


@dataclass(frozen=True)
class TriangulationConfig:
    """三角定位配置参数。"""

    frames_per_shot: int = 3
    frame_std_max_px: float = 0.3
    baseline_min_mm: float = 20.0
    baseline_max_mm: float = 30.0
    min_views: int = 3
    max_views: int = 5
    max_ray_residual_mm: float = 0.02
    min_baseline_ratio: float = 0.15
    approx_tip_z_mm: float | None = None
    station_bounds_mm: dict[str, list[float]] | None = None
    center_margin_frac: float = 0.10
    position_tolerance_mm: float = 0.1


@dataclass
class TriangulationView:
    """单个视角的拍照坐标、亚像素检测结果与三维视线。"""

    index: int
    pose: tuple[float, float, float]
    uv: tuple[float, float]
    origin: tuple[float, float, float]
    ray: tuple[float, float, float]
    frame_std_uv: tuple[float, float]
    quality: float
    raw_path: str | None = None
    overlay_path: str | None = None


@dataclass
class TriangulationResult:
    """三角定位求解结果与几何质量诊断。"""

    tip_xyz_mm: tuple[float, float, float]
    target_xy_mm: tuple[float, float]
    ray_residuals_mm: list[float]
    max_residual_mm: float
    rms_residual_mm: float
    reproj_errors_px: list[float]
    max_angle_rad: float
    max_baseline_ratio: float
    depths_mm: list[float]
    ok: bool
    reason: str = ""


def ray_from_pixel(u: float, v: float, pose: tuple[float, float, float], calibration: Calibration) -> tuple[np.ndarray, np.ndarray]:
    """根据 camera.yaml 内外参与拍照 PLC 坐标计算模组系下相机光心与单位视线向量。"""
    xy = cv2.undistortPoints(np.array([[[float(u), float(v)]]], dtype=np.float64), calibration.K, calibration.distortion)[0, 0]
    ray = np.asarray(calibration.rotation, dtype=float) @ np.array([xy[0], xy[1], 1.0], dtype=float)
    norm = float(np.linalg.norm(ray))
    if norm < 1e-12 or not np.isfinite(norm):
        raise ValueError("相机视线向量退化")
    origin = np.asarray(pose, dtype=float) + np.asarray(calibration.translation, dtype=float)
    return origin, ray / norm


def project_point(xyz: np.ndarray | tuple[float, float, float], pose: tuple[float, float, float], calibration: Calibration) -> tuple[float, float] | None:
    """将模组系三维点投影回主相机像素坐标；点在相机后方时返回 None。"""
    origin = np.asarray(pose, dtype=float) + np.asarray(calibration.translation, dtype=float)
    pc = (np.asarray(xyz, dtype=float) - origin) @ np.asarray(calibration.rotation, dtype=float)
    if not np.isfinite(pc).all() or pc[2] <= 1e-6:
        return None
    uv, _ = cv2.projectPoints(pc.reshape(1, 1, 3), np.zeros(3), np.zeros(3), calibration.K, calibration.distortion)
    u, v = float(uv[0, 0, 0]), float(uv[0, 0, 1])
    if not (math.isfinite(u) and math.isfinite(v)):
        return None
    return (u, v)


def target_xy_from_tip(tip_xyz: tuple[float, float, float] | np.ndarray) -> tuple[float, float]:
    """将三角定位得到的标定尖三维模组坐标换算为与生产 calculate_pick_coords 一致的目标机台 XY。

    说明：
    生产视觉 `pixel_to_module(u, v, pose, cal)` 使用光心 `origin = pose + cal.translation`
    和视线 `ray = cal.rotation @ [x_n, y_n, 1]` 与工作平面求交，返回目标在 `camera.yaml`
    模组坐标系下的三维坐标 `(X, Y, Z)`；`calculate_pick_coords` 再用 `target[:2] + offset[:2]`
    下发工具坐标（已包含相机倾斜 0.6° 与畸变在射线求交中的影响）。
    由于三角定位直接在同一 `camera.yaml` 模组坐标系中求交得到标定尖三维坐标 `(X_t, Y_t, Z_t)`，
    其与在标定尖所在高度 `Z_t` 平面调用 `pixel_to_module` 的结果一致，因此目标机台 XY 即 `(X_t, Y_t)`。
    """
    return float(tip_xyz[0]), float(tip_xyz[1])


def verify_target_xy_consistency(tip_xyz: tuple[float, float, float], pose: tuple[float, float, float], uv: tuple[float, float], calibration: Calibration) -> float:
    """用标定尖高度处的水平工作面验证 pixel_to_module 与 target_xy_from_tip 的一致性（返回偏差 mm）。"""
    at_tip = replace(
        calibration,
        plane_z=float(tip_xyz[2]),
        plane_point=np.array([0.0, 0.0, float(tip_xyz[2])], dtype=float),
        plane_normal=np.array([0.0, 0.0, 1.0], dtype=float),
    )
    projected = pixel_to_module(uv[0], uv[1], pose, at_tip)
    target = target_xy_from_tip(tip_xyz)
    return float(math.hypot(projected[0] - target[0], projected[1] - target[1]))


def triangulate_rays(views: list[TriangulationView], calibration: Calibration, *,
                     max_residual_mm: float = 0.02, min_baseline_ratio: float = 0.15) -> TriangulationResult:
    """多视角视线最小二乘交点求解并执行重投影优化与质量门限检查。"""
    if len(views) < 2:
        raise ValueError("三角定位至少需要两个视角")
    A = np.zeros((3, 3), dtype=float)
    b = np.zeros(3, dtype=float)
    origins = []
    rays = []
    for view in views:
        o = np.asarray(view.origin, dtype=float)
        d = np.asarray(view.ray, dtype=float)
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ o
        origins.append(o)
        rays.append(d)
    if np.linalg.matrix_rank(A) < 3:
        raise ValueError("视线近似平行，三角定位矩阵降秩")
    x_init = np.linalg.solve(A, b)
    if not np.isfinite(x_init).all():
        raise ValueError("三角定位初始解非有限数")

    # 在三维射线初值上做重投影误差非线性最小二乘精化，使像素噪声下达到最大似然估计
    def reproj_residual(xyz: np.ndarray) -> np.ndarray:
        errs = []
        for view in views:
            proj = project_point(xyz, view.pose, calibration)
            if proj is None:
                errs.extend((1e4, 1e4))
            else:
                errs.extend((proj[0] - view.uv[0], proj[1] - view.uv[1]))
        return np.asarray(errs, dtype=float)

    opt = least_squares(reproj_residual, x_init, method="lm")
    xyz = opt.x if opt.success and np.isfinite(opt.x).all() else x_init

    ray_residuals: list[float] = []
    depths: list[float] = []
    reproj_errors: list[float] = []
    for view, o, d in zip(views, origins, rays):
        diff = xyz - o
        depth = float(np.dot(d, diff))
        depths.append(depth)
        perp = diff - depth * d
        ray_residuals.append(float(np.linalg.norm(perp)))
        proj = project_point(xyz, view.pose, calibration)
        reproj_errors.append(float(math.hypot(proj[0] - view.uv[0], proj[1] - view.uv[1])) if proj is not None else float("inf"))

    max_res = float(max(ray_residuals))
    rms_res = float(math.sqrt(sum(r * r for r in ray_residuals) / len(ray_residuals)))

    max_angle = 0.0
    max_ratio = 0.0
    for i in range(len(views)):
        for j in range(i + 1, len(views)):
            dot = float(np.clip(np.dot(rays[i], rays[j]), -1.0, 1.0))
            angle = float(math.acos(dot))
            if angle > max_angle:
                max_angle = angle
            baseline = float(np.linalg.norm(origins[i] - origins[j]))
            mean_dist = 0.5 * (depths[i] + depths[j])
            if mean_dist > 1e-6:
                ratio = baseline / mean_dist
                if ratio > max_ratio:
                    max_ratio = ratio

    reasons = []
    if any(depth <= 0 for depth in depths):
        reasons.append("三角定位交点位于相机后方")
    if max_res > max_residual_mm:
        reasons.append(f"视线交点最大残差 {max_res:.4f} mm 超过阈值 {max_residual_mm:.4f} mm")
    effective_ratio = min(max_ratio, max_angle)
    if effective_ratio < min_baseline_ratio:
        reasons.append(f"视线夹角/基线比 {effective_ratio:.3f} 小于阈值 {min_baseline_ratio:.3f}")

    tip_tuple = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
    return TriangulationResult(
        tip_xyz_mm=tip_tuple,
        target_xy_mm=target_xy_from_tip(tip_tuple),
        ray_residuals_mm=ray_residuals,
        max_residual_mm=max_res,
        rms_residual_mm=rms_res,
        reproj_errors_px=reproj_errors,
        max_angle_rad=max_angle,
        max_baseline_ratio=float(effective_ratio),
        depths_mm=depths,
        ok=not reasons,
        reason="；".join(reasons),
    )


def _estimate_initial_tip(view1: TriangulationView, calibration: Calibration, cfg: TriangulationConfig) -> np.ndarray:
    """仅有第 1 张视角时，结合参考高度估算标定尖大致三维位置以规划后续视角。"""
    z_hint = float(cfg.approx_tip_z_mm) if cfg.approx_tip_z_mm is not None else float(calibration.plane_z)
    o = np.asarray(view1.origin, dtype=float)
    d = np.asarray(view1.ray, dtype=float)
    if abs(d[2]) > 1e-6:
        s = (z_hint - o[2]) / d[2]
        if s > 10.0:
            return o + s * d
    return o + 110.0 * d


def plan_next_viewpoint(
    first_pose: tuple[float, float, float],
    visited_poses: list[tuple[float, float, float]],
    tip_est_xyz: np.ndarray,
    calibration: Calibration,
    travel: tuple[tuple[float, float], ...],
    image_size: tuple[int, int],
    cfg: TriangulationConfig,
) -> tuple[float, float, float] | None:
    """规划下一拍照坐标：相对第 1 点仅往 X+ 与 ±Y、基线在配置范围内、在行程与可去范围内、预测落入画面中央 80%。"""
    sb = cfg.station_bounds_mm or {}
    x_lo = max(float(travel[0][0]), float(first_pose[0]), float(sb["x"][0]) if sb.get("x") else -math.inf)
    x_hi = min(float(travel[0][1]), float(sb["x"][1]) if sb.get("x") else math.inf)
    y_lo = max(float(travel[1][0]), float(sb["y"][0]) if sb.get("y") else -math.inf)
    y_hi = min(float(travel[1][1]), float(sb["y"][1]) if sb.get("y") else math.inf)
    z = float(first_pose[2])
    z_lo = max(float(travel[2][0]), float(sb["z"][0]) if sb.get("z") else -math.inf)
    z_hi = min(float(travel[2][1]), float(sb["z"][1]) if sb.get("z") else math.inf)
    if not (z_lo <= z <= z_hi) or x_lo > x_hi or y_lo > y_hi:
        return None

    w, h = image_size
    m = cfg.center_margin_frac
    u_min, u_max = m * w, (1.0 - m) * w
    v_min, v_max = m * h, (1.0 - m) * h

    radii = sorted(
        {round(float(r), 3) for r in np.linspace(cfg.baseline_min_mm, cfg.baseline_max_mm, 5)},
        key=lambda r: abs(r - 0.5 * (cfg.baseline_min_mm + cfg.baseline_max_mm)),
    )
    angles_deg = list(range(-90, 91, 10))
    visited_xy = [np.asarray(p[:2], dtype=float) for p in visited_poses]

    best_score = -math.inf
    best_pose: tuple[float, float, float] | None = None
    for r in radii:
        for deg in angles_deg:
            rad = math.radians(deg)
            dx = max(0.0, r * math.cos(rad))
            dy = r * math.sin(rad)
            px = float(first_pose[0] + dx)
            py = float(first_pose[1] + dy)
            if not (x_lo - 1e-9 <= px <= x_hi + 1e-9 and y_lo - 1e-9 <= py <= y_hi + 1e-9):
                continue
            px = float(np.clip(px, x_lo, x_hi))
            py = float(np.clip(py, y_lo, y_hi))
            cand_xy = np.array([px, py], dtype=float)
            actual_r = float(np.linalg.norm(cand_xy - visited_xy[0]))
            if not (cfg.baseline_min_mm - 1e-6 <= actual_r <= cfg.baseline_max_mm + 1e-6):
                continue
            min_dist = min(float(np.linalg.norm(cand_xy - vp)) for vp in visited_xy)
            if min_dist < 5.0:
                continue
            pose = (px, py, z)
            proj = project_point(tip_est_xyz, pose, calibration)
            if proj is None:
                continue
            u, v = proj
            if not (u_min <= u <= u_max and v_min <= v <= v_max):
                continue
            # 鼓励与已有拍照点张成最大面积三角形/多边形，并给 X+ 分量与画面居中适度加权
            margin_px = min((u - u_min) / max(w, 1), (u_max - u) / max(w, 1),
                            (v - v_min) / max(h, 1), (v_max - v) / max(h, 1))
            if len(visited_xy) >= 2:
                pts = np.vstack([*visited_xy, cand_xy])
                centered = pts - pts.mean(axis=0)
                svals = np.linalg.svd(centered, compute_uv=False)
                geom_score = float(svals[1]) if len(svals) > 1 else 0.0
            else:
                geom_score = min_dist + (2.0 if dx > 1.0 else 0.0)
            score = geom_score * 10.0 + min_dist + margin_px * 15.0 + 0.2 * dx
            if score > best_score:
                best_score = score
                best_pose = pose
    return best_pose


class TriangulationSession:
    """管理 90 多视角采图、下一位置选择与三角定位求解状态。"""

    def __init__(self, calibration: Calibration, travel: tuple[tuple[float, float], ...],
                 image_size: tuple[int, int], config: TriangulationConfig | dict | None = None) -> None:
        if isinstance(config, TriangulationConfig):
            self.cfg = config
        elif isinstance(config, dict):
            self.cfg = TriangulationConfig(**{k: v for k, v in config.items() if k in TriangulationConfig.__dataclass_fields__})
        else:
            self.cfg = TriangulationConfig()
        self.calibration = calibration
        self.travel = travel
        self.image_size = image_size
        self.views: list[TriangulationView] = []
        self.target_pose: tuple[float, float, float] | None = None
        self.result: TriangulationResult | None = None
        self.failure: str | None = None

    def add_observation(
        self,
        pose: tuple[float, float, float],
        uv: tuple[float, float],
        *,
        frame_std_uv: tuple[float, float] = (0.0, 0.0),
        quality: float = 1.0,
        raw_path: str | None = None,
        overlay_path: str | None = None,
    ) -> str:
        """加入一个视角观测，返回 'move' / 'done' / 'failed'。"""
        origin, ray = ray_from_pixel(uv[0], uv[1], pose, self.calibration)
        self.views.append(
            TriangulationView(
                index=len(self.views) + 1,
                pose=tuple(float(x) for x in pose),
                uv=(float(uv[0]), float(uv[1])),
                origin=tuple(float(x) for x in origin),
                ray=tuple(float(x) for x in ray),
                frame_std_uv=tuple(float(x) for x in frame_std_uv),
                quality=float(quality),
                raw_path=raw_path,
                overlay_path=overlay_path,
            )
        )
        n = len(self.views)
        if n >= self.cfg.min_views:
            try:
                res = triangulate_rays(
                    self.views,
                    self.calibration,
                    max_residual_mm=self.cfg.max_ray_residual_mm,
                    min_baseline_ratio=self.cfg.min_baseline_ratio,
                )
            except ValueError as exc:
                res = None
                err = str(exc)
            else:
                err = res.reason
            if res is not None and res.ok:
                self.result = res
                self.target_pose = None
                return "done"
            if n >= self.cfg.max_views:
                self.result = res
                self.failure = f"三角定位已采集 {n} 个视角仍不满足质量门限：{err}"
                return "failed"
            tip_est = np.asarray(res.tip_xyz_mm, dtype=float) if res is not None else _estimate_initial_tip(self.views[0], self.calibration, self.cfg)
        elif n == 2:
            try:
                res2 = triangulate_rays(self.views, self.calibration, max_residual_mm=1e6, min_baseline_ratio=0.0)
                tip_est = np.asarray(res2.tip_xyz_mm, dtype=float)
            except ValueError:
                tip_est = _estimate_initial_tip(self.views[0], self.calibration, self.cfg)
        else:
            tip_est = _estimate_initial_tip(self.views[0], self.calibration, self.cfg)

        nxt = plan_next_viewpoint(
            self.views[0].pose,
            [v.pose for v in self.views],
            tip_est,
            self.calibration,
            self.travel,
            self.image_size,
            self.cfg,
        )
        if nxt is None:
            self.failure = "无法规划满足行程、X≥首点且标定尖在画面中央80%的下一拍照坐标"
            return "failed"
        self.target_pose = nxt
        return "move"
