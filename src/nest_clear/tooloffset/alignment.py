"""对刀相机姿态测量（四步探测拟合雅可比）、水平对准解算与防撞间隙预测。"""
from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from nest_clear.plan.scan_path import guard_position
from nest_clear.tooltip.core import fit_view_jacobian


@dataclass(frozen=True)
class AlignConfig:
    """对刀相机探测与对准配置参数。"""

    probe_step_xy_mm: float = 0.2
    probe_step_z_mm: float = 2.0
    align_tol_mm: float = 0.05
    max_align_iterations: int = 10
    max_step_mm: float = 0.5
    max_total_mm: float = 3.0
    min_gap_mm: float = 1.0
    consistency_tolerance_mm: float = 0.05
    position_tolerance_mm: float = 0.1
    fallback_um_per_px: float = 15.2
    max_jacobian_rms_px: float = 2.0
    max_jacobian_condition: float = 300.0
    frame_std_max_px: float = 0.3


def solve_view_delta(
    tool_uv: Sequence[float],
    ref_uv: Sequence[float],
    j_xy: Sequence[float],
    j_z: Sequence[float],
) -> tuple[float, float, float]:
    """求解单视图对准与竖直间隙：r - p = J_xy*Δxy + J_z*Δz。

    返回 ``(delta_xy_mm, delta_z_mm, gap_mm)``；
    ``gap_mm > 0`` 表示工具尖在标定尖上方，``gap_mm <= 0`` 表示工具尖已低于标定尖。
    """
    p = np.asarray(tool_uv, dtype=float)
    r = np.asarray(ref_uv, dtype=float)
    jx = np.asarray(j_xy, dtype=float)
    jz = np.asarray(j_z, dtype=float)
    mat = np.column_stack((jx, jz))
    if abs(float(np.linalg.det(mat))) < 1e-8 or abs(float(jz[1])) < 1e-8:
        raise ValueError("相机雅可比矩阵退化或 Z 轴垂直投影过小")
    d_xy, d_z = np.linalg.solve(mat, r - p)
    # 图像坐标系 +v 朝向标定尖下方，因此与 jz[1] 同号的 Δz 为向标定尖下方移动的距离
    z_down_sign = 1.0 if jz[1] >= 0.0 else -1.0
    gap_mm = float(z_down_sign * d_z)
    return float(d_xy), float(d_z), gap_mm


def view_gap_mm(
    view: str,
    tool_uv: Sequence[float],
    ref_uv: Sequence[float],
    jacobian: np.ndarray | Sequence | None,
    *,
    fallback_um_per_px: float = 15.2,
    move_xyz: Sequence[float] = (0.0, 0.0, 0.0),
) -> float:
    """计算单视图中工具尖与标定尖的当前或移动后预测竖直间隙（mm）。"""
    p = np.asarray(tool_uv, dtype=float)
    r = np.asarray(ref_uv, dtype=float)
    move = np.asarray(move_xyz, dtype=float)
    if jacobian is not None:
        J = np.asarray(jacobian, dtype=float)
        p_after = p + J @ move
        xy_col = 0 if view == "front" else 1
        _, _, gap = solve_view_delta(p_after, r, J[:, xy_col], J[:, 2])
        return gap
    # 尚未测出 Z 雅可比前，使用像素竖直差按名义像元尺度估算
    mm_per_px = float(fallback_um_per_px) / 1000.0
    return float((r[1] - p[1]) * mm_per_px)


def predict_collision_gap_mm(
    tool_points: dict[str, Sequence[float]],
    ref_points: dict[str, Sequence[float]],
    jacobians: dict[str, np.ndarray | Sequence | None],
    move_xyz: Sequence[float] = (0.0, 0.0, 0.0),
    *,
    fallback_um_per_px: float = 15.2,
) -> tuple[float, dict[str, float]]:
    """综合正面与侧面视图预测移动后最小竖直间隙（mm）。"""
    per_view = {
        view: view_gap_mm(
            view,
            tool_points[view],
            ref_points[view],
            jacobians.get(view),
            fallback_um_per_px=fallback_um_per_px,
            move_xyz=move_xyz,
        )
        for view in ("front", "side")
    }
    return float(min(per_view.values())), per_view


def _clamp_relative_to_p0(
    target: tuple[float, float, float],
    p0: tuple[float, float, float],
    max_total_mm: float,
) -> None:
    for axis_name, t, o in zip(("X", "Y", "Z"), target, p0):
        if abs(t - o) > max_total_mm + 1e-9:
            raise ValueError(f"{axis_name} 相对 P0 累计位移 {abs(t - o):.3f} mm 超过上限 {max_total_mm:g} mm")


class ToolAlignmentSession:
    """管理单个工具的 100/110 姿态探测（4 步）、对准收敛与防撞检查。"""

    def __init__(
        self,
        tool: int,
        target_xy_mm: tuple[float, float],
        travel: tuple[tuple[float, float], ...],
        config: AlignConfig | dict | None = None,
    ) -> None:
        if tool not in (1, 2):
            raise ValueError(f"工具号无效：{tool}")
        if isinstance(config, AlignConfig):
            self.cfg = config
        elif isinstance(config, dict):
            self.cfg = AlignConfig(**{k: v for k, v in config.items() if k in AlignConfig.__dataclass_fields__})
        else:
            self.cfg = AlignConfig()
        self.tool = tool
        self.target_xy_mm = (float(target_xy_mm[0]), float(target_xy_mm[1]))
        self.travel = travel

        self.p0: tuple[float, float, float] | None = None
        self.probe_index: int = 0  # 已发出的探测步数 0..4
        self.probe_poses: list[tuple[float, float, float]] = []
        self.probe_tool_uv: list[dict[str, tuple[float, float]]] = []
        self.probe_ref_uv: list[dict[str, tuple[float, float]]] = []
        self.jacobians: dict[str, np.ndarray] = {}
        self.jacobian_fit_info: dict[str, dict] = {}
        self.align_iteration: int = 0
        self.last_target: tuple[float, float, float] | None = None
        self.last_delta_xy: tuple[float, float] | None = None
        self.last_gap_mm: float | None = None
        self.last_predicted_gap_mm: float | None = None
        self.p_final: tuple[float, float, float] | None = None
        self.offset_xy: tuple[float, float] | None = None
        self.history: list[dict] = []

    def _probe_targets(self, p0: tuple[float, float, float]) -> list[tuple[float, float, float]]:
        sxy = float(self.cfg.probe_step_xy_mm)
        sz = float(self.cfg.probe_step_z_mm)
        return [
            (p0[0] + sxy, p0[1], p0[2]),
            (p0[0], p0[1] + sxy, p0[2]),
            (p0[0], p0[1], p0[2] + sz),
            (p0[0], p0[1], p0[2] - sz),
        ]

    def _preliminary_jacobians(self) -> dict[str, np.ndarray]:
        """当尚未完成第 4 步 Z-2 时，用前三步（含 Z+2）估出临时雅可比供防撞检查。"""
        if self.jacobians:
            return self.jacobians
        if len(self.probe_poses) < 4:
            return {}
        out: dict[str, np.ndarray] = {}
        p0 = np.asarray(self.probe_poses[0], dtype=float)
        uv0 = self.probe_tool_uv[0]
        for view in ("front", "side"):
            samples = [
                (
                    np.asarray(pose, dtype=float) - p0,
                    np.asarray(uvs[view], dtype=float) - np.asarray(uv0[view], dtype=float),
                )
                for pose, uvs in zip(self.probe_poses[1:], self.probe_tool_uv[1:])
            ]
            out[view] = fit_view_jacobian(samples).jacobian
        return out

    def _fit_all_jacobians(self) -> None:
        p0 = np.asarray(self.probe_poses[0], dtype=float)
        uv0 = self.probe_tool_uv[0]
        for view in ("front", "side"):
            samples = [
                (
                    np.asarray(pose, dtype=float) - p0,
                    np.asarray(uvs[view], dtype=float) - np.asarray(uv0[view], dtype=float),
                )
                for pose, uvs in zip(self.probe_poses[1:], self.probe_tool_uv[1:])
            ]
            fit = fit_view_jacobian(samples)
            if fit.rms_px > self.cfg.max_jacobian_rms_px:
                raise ValueError(f"{view} 雅可比拟合 RMS {fit.rms_px:.3f} > {self.cfg.max_jacobian_rms_px:g} px")
            if fit.condition_number > self.cfg.max_jacobian_condition:
                raise ValueError(f"{view} 雅可比条件数 {fit.condition_number:.1f} > {self.cfg.max_jacobian_condition:g}")
            self.jacobians[view] = fit.jacobian
            self.jacobian_fit_info[view] = {
                "matrix": fit.jacobian.tolist(),
                "rms_px": float(fit.rms_px),
                "condition_number": float(fit.condition_number),
            }

    def step(
        self,
        actual_xyz: tuple[float, float, float],
        tool_points: dict[str, tuple[float, float]],
        ref_points: dict[str, tuple[float, float]],
    ) -> dict:
        """处理一次 100/110 到位观测，返回 next_phase / target / converged / collision / failed 决策。"""
        actual = tuple(float(v) for v in actual_xyz)
        jacs = self._preliminary_jacobians() if not self.jacobians else self.jacobians
        cur_gap, cur_gaps = predict_collision_gap_mm(
            tool_points,
            ref_points,
            jacs,
            (0.0, 0.0, 0.0),
            fallback_um_per_px=self.cfg.fallback_um_per_px,
        )
        self.last_gap_mm = cur_gap

        if cur_gap < self.cfg.min_gap_mm:
            self.last_predicted_gap_mm = cur_gap
            return {
                "status": "collision",
                "reason": f"会碰到标定尖：当前竖直间隙 {cur_gap:.3f} mm 小于安全下限 {self.cfg.min_gap_mm:g} mm",
                "current_gap_mm": cur_gap,
                "predicted_gap_mm": cur_gap,
                "per_view_gap_mm": cur_gaps,
            }

        # 阶段 1：姿态探测（P0, X+0.2, Y+0.2, Z+2, Z-2）
        if not self.jacobians:
            if self.p0 is None:
                self.p0 = actual
            self.probe_poses.append(actual)
            self.probe_tool_uv.append({k: tuple(map(float, v)) for k, v in tool_points.items()})
            self.probe_ref_uv.append({k: tuple(map(float, v)) for k, v in ref_points.items()})

            if len(self.probe_poses) <= 4:
                targets = self._probe_targets(self.p0)
                nxt = targets[len(self.probe_poses) - 1]
                self.probe_index = len(self.probe_poses)
                try:
                    guard_position(nxt, self.travel, f"姿态测量第 {self.probe_index}/4 步")
                except Exception as exc:
                    return {"status": "failed", "reason": str(exc)}
                # 第 4 步 (Z-2) 前已有 Z+2 数据，可用临时雅可比预测移动后间隙
                move = tuple(nxt[i] - actual[i] for i in range(3))
                jacs_pred = self._preliminary_jacobians()
                pred_gap, _ = predict_collision_gap_mm(
                    tool_points,
                    ref_points,
                    jacs_pred,
                    move,
                    fallback_um_per_px=self.cfg.fallback_um_per_px,
                )
                self.last_predicted_gap_mm = pred_gap
                if jacs_pred and pred_gap < self.cfg.min_gap_mm:
                    return {
                        "status": "collision",
                        "reason": f"会碰到标定尖：执行探测步后预测竖直间隙 {pred_gap:.3f} mm < {self.cfg.min_gap_mm:g} mm",
                        "current_gap_mm": cur_gap,
                        "predicted_gap_mm": pred_gap,
                    }
                self.last_target = nxt
                entry = {
                    "kind": "probe",
                    "probe_step": self.probe_index,
                    "actual_xyz": list(actual),
                    "target_xyz": list(nxt),
                    "current_gap_mm": cur_gap,
                    "predicted_gap_mm": pred_gap if jacs_pred else cur_gap,
                }
                self.history.append(entry)
                return {
                    "status": "move",
                    "phase": "probing",
                    "probe_step": self.probe_index,
                    "target": nxt,
                    "current_gap_mm": cur_gap,
                    "predicted_gap_mm": pred_gap if jacs_pred else cur_gap,
                }

            # 已集齐全部 5 点（P0 + X+0.2 + Y+0.2 + Z+2 + Z-2），拟合相机雅可比
            try:
                self._fit_all_jacobians()
            except ValueError as exc:
                return {"status": "failed", "reason": str(exc)}

        # 阶段 2：用已拟合雅可比解算 Δx / Δy 并判收敛或下发对准移动
        dx, dz_f, gap_f = solve_view_delta(tool_points["front"], ref_points["front"], self.jacobians["front"][:, 0], self.jacobians["front"][:, 2])
        dy, dz_s, gap_s = solve_view_delta(tool_points["side"], ref_points["side"], self.jacobians["side"][:, 1], self.jacobians["side"][:, 2])
        cur_gap = float(min(gap_f, gap_s))
        self.last_delta_xy = (dx, dy)
        self.last_gap_mm = cur_gap

        if cur_gap < self.cfg.min_gap_mm:
            self.last_predicted_gap_mm = cur_gap
            return {
                "status": "collision",
                "reason": f"会碰到标定尖：当前竖直间隙 {cur_gap:.3f} mm 小于安全下限 {self.cfg.min_gap_mm:g} mm",
                "delta_xy_mm": (dx, dy),
                "current_gap_mm": cur_gap,
                "predicted_gap_mm": cur_gap,
            }

        if abs(dx) <= self.cfg.align_tol_mm and abs(dy) <= self.cfg.align_tol_mm:
            self.p_final = actual
            offset_xy_uncorrected = (
                float(actual[0] - self.target_xy_mm[0]),
                float(actual[1] - self.target_xy_mm[1]),
            )
            self.offset_xy = (
                float(actual[0] + dx - self.target_xy_mm[0]),
                float(actual[1] + dy - self.target_xy_mm[1]),
            )
            self.history.append({
                "kind": "converged",
                "align_iteration": self.align_iteration,
                "actual_xyz": list(actual),
                "delta_xy_mm": [dx, dy],
                "current_gap_mm": cur_gap,
                "p_final_xyz": list(self.p_final),
                "offset_xy_uncorrected_mm": list(offset_xy_uncorrected),
                "offset_xy_mm": list(self.offset_xy),
            })
            return {
                "status": "converged",
                "align_iteration": self.align_iteration,
                "delta_xy_mm": (dx, dy),
                "current_gap_mm": cur_gap,
                "p_final": self.p_final,
                "offset_xy_uncorrected": offset_xy_uncorrected,
                "offset_xy": self.offset_xy,
            }

        self.align_iteration += 1
        if self.align_iteration > self.cfg.max_align_iterations:
            return {
                "status": "failed",
                "reason": f"对准超过最大轮次 {self.cfg.max_align_iterations} 仍未收敛（ΔX={dx:+.3f}, ΔY={dy:+.3f} mm）",
                "delta_xy_mm": (dx, dy),
                "current_gap_mm": cur_gap,
            }

        step_x = float(np.clip(dx, -self.cfg.max_step_mm, self.cfg.max_step_mm))
        step_y = float(np.clip(dy, -self.cfg.max_step_mm, self.cfg.max_step_mm))
        nxt = (actual[0] + step_x, actual[1] + step_y, actual[2])

        try:
            assert self.p0 is not None
            _clamp_relative_to_p0(nxt, self.p0, self.cfg.max_total_mm)
            guard_position(nxt, self.travel, f"对准第 {self.align_iteration} 轮")
        except Exception as exc:
            return {
                "status": "failed",
                "reason": str(exc),
                "delta_xy_mm": (dx, dy),
                "current_gap_mm": cur_gap,
            }

        pred_gap, pred_gaps = predict_collision_gap_mm(
            tool_points,
            ref_points,
            self.jacobians,
            (step_x, step_y, 0.0),
            fallback_um_per_px=self.cfg.fallback_um_per_px,
        )
        self.last_predicted_gap_mm = pred_gap
        if pred_gap < self.cfg.min_gap_mm:
            return {
                "status": "collision",
                "reason": f"会碰到标定尖：移动后预测竖直间隙 {pred_gap:.3f} mm 小于安全下限 {self.cfg.min_gap_mm:g} mm",
                "delta_xy_mm": (dx, dy),
                "current_gap_mm": cur_gap,
                "predicted_gap_mm": pred_gap,
                "per_view_gap_mm": pred_gaps,
            }

        self.last_target = nxt
        self.history.append({
            "kind": "align",
            "align_iteration": self.align_iteration,
            "actual_xyz": list(actual),
            "target_xyz": list(nxt),
            "delta_xy_mm": [dx, dy],
            "step_xy_mm": [step_x, step_y],
            "current_gap_mm": cur_gap,
            "predicted_gap_mm": pred_gap,
        })
        return {
            "status": "move",
            "phase": "aligning",
            "align_iteration": self.align_iteration,
            "target": nxt,
            "delta_xy_mm": (dx, dy),
            "current_gap_mm": cur_gap,
            "predicted_gap_mm": pred_gap,
        }
