"""双相机针尖对准的局部线性求解与模组坐标补偿。"""

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
import numpy as np

from nest_clear.guard import GuardError


def _error(message: str, values: dict) -> GuardError:
    """附上实际输入与排查建议，供现场定位标定问题。"""
    actual = {k: repr(v)[:300] for k, v in values.items() if k != "exc"}
    return GuardError("TOOLTIP-INPUT", f"针尖标定参数或数据不符合要求：{message}；实际值：{actual}", "tooltip.core", "检查针尖标定数据、相机视图和参数的形状及单位")


def _array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    """处理针尖标定数据，供对刀流程调用。
       Validate and convert a finite numeric array of the requested shape."""
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in "iuf":
            raise _error(f"{name} must contain numbers", locals())
        a = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise _error(f"{name} must contain numbers", locals()) from exc
    if a.shape != shape or not np.all(np.isfinite(a)):
        raise _error(f"{name} must be finite with shape {shape}", locals())
    return a


def _tuple(value: object, shape: tuple[int, ...], name: str) -> tuple:
    """处理针尖标定数据，供对刀流程调用。
       Validate and freeze a vector or matrix as tuples."""
    a = _array(value, shape, name)
    return tuple(map(tuple, a.tolist())) if len(shape) == 2 else tuple(a.tolist())


@dataclass(frozen=True)
class ViewCalib:
    """Fixed camera name, target pixel and local 2×3 J in pixels/mm.

    J maps platform-frame xyz displacement to image-frame uv displacement.
    Invalid names, shapes or nonfinite values raise GuardError.
    """

    name: str  # camera identifier
    target_uv: tuple[float, float]  # image coordinates, pixels
    jacobian: tuple[tuple[float, float, float], tuple[float, float, float]]  # pixels/mm

    def __post_init__(self) -> None:
        """校验并冻结基准数据，供求解与 YAML 共用。"""
        if not isinstance(self.name, str) or not self.name.strip():
            raise _error("view name must be a non-empty string", locals())
        object.__setattr__(self, "target_uv", _tuple(self.target_uv, (2,), "target_uv"))
        object.__setattr__(self, "jacobian", _tuple(self.jacobian, (2, 3), "jacobian"))


@dataclass(frozen=True)
class ToolTipBaseline:
    """Baseline coordinates and views; malformed fields raise GuardError."""

    tool_name: str  # tool identifier
    p0: tuple[float, float, float]  # platform-frame xyz, mm
    views: tuple[ViewCalib, ...]  # fixed-camera calibrations
    offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)  # platform-frame mm
    timestamp: str = ""  # caller-defined timestamp
    note: str = ""  # caller-defined note

    def __post_init__(self) -> None:
        """校验并冻结基准数据，供求解与 YAML 共用。"""
        if not isinstance(self.tool_name, str) or not self.tool_name.strip():
            raise _error("tool_name must be a non-empty string", locals())
        object.__setattr__(self, "p0", _tuple(self.p0, (3,), "p0"))
        object.__setattr__(self, "offset_xyz", _tuple(self.offset_xyz, (3,), "offset_xyz"))
        try:
            views = tuple(self.views)
        except TypeError as exc:
            raise _error("views 必须是相机标定序列", locals()) from exc
        if not views or any(not isinstance(v, ViewCalib) for v in views):
            raise _error("views must contain at least one ViewCalib", locals())
        if len({v.name for v in views}) != len(views):
            raise _error("duplicate view name", locals())
        object.__setattr__(self, "views", views)
        if not isinstance(self.timestamp, str) or not isinstance(self.note, str):
            raise _error("timestamp and note must be strings", locals())


@dataclass(frozen=True)
class JacobianFit:
    """Fit matrix, per-sample pixel residuals, component RMS and input condition number."""

    jacobian: np.ndarray  # pixels/mm, shape (2, 3)
    residuals_uv: np.ndarray  # measured minus fitted uv, pixels
    rms_px: float  # RMS over all residual components, pixels
    condition_number: float  # dimensionless condition number of sample matrix


def fit_view_jacobian(samples: Sequence) -> JacobianFit:
    """处理针尖标定数据，供对刀流程调用。
       Fit J from (platform xyz mm, image uv pixels) displacement pairs.

    Return a JacobianFit for uv = J @ xyz; insufficient or rank-deficient
    samples and invalid coordinates raise GuardError.
    """
    try:
        samples = list(samples)
    except TypeError as exc:
        raise _error("samples 必须是位移样本序列", locals()) from exc
    if len(samples) < 3:
        raise _error("at least three independent displacement samples are required", locals())
    try:
        if any(len(s) != 2 for s in samples):
            raise _error("each sample must be a pair (xyz, uv)", locals())
        x = np.array([_array(s[0], (3,), "sample xyz") for s in samples])
        y = np.array([_array(s[1], (2,), "sample uv") for s in samples])
    except (IndexError, TypeError) as exc:
        raise _error("each sample must be a pair (xyz, uv)", locals()) from exc
    if np.linalg.matrix_rank(x) < 3:
        raise _error("sample displacements are rank deficient; span all three axes", locals())
    coef = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ coef
    return JacobianFit(
        coef.T, residual, float(np.sqrt(np.mean(residual**2))), float(np.linalg.cond(x))
    )


@dataclass(frozen=True)
class CorrectionResult:
    """Joint correction and per-view residuals and conditional axis diagnostics."""

    delta_xyz: np.ndarray  # platform-frame correction, mm
    residuals_uv: dict[str, np.ndarray]  # predicted minus requested uv, pixels
    residuals_mm: dict[str, np.ndarray]  # pseudoinverse-projected residual, mm
    condition_number: float  # weighted stacked matrix condition number
    axis_estimates: dict[str, dict[str, float]]  # conditional estimates, mm
    axis_disagreement: dict[str, float]  # max minus min per axis, mm


def solve_correction(
    observed: Mapping[str, Sequence[float]],
    baseline: ToolTipBaseline | Sequence[ViewCalib],
    *,
    weights: Mapping[str, float] | None = None,
    primary_view: str | None = None,
) -> CorrectionResult:
    """处理针尖标定数据，供对刀流程调用。
       Solve target - observed = J @ delta for platform-frame delta in mm.

    Observations are uv pixels keyed by view. Positive per-view weights and an
    optional primary view (double weight) affect the least-squares objective.
    Return residuals, condition number and conditional axis estimates. Missing
    views, invalid weights or a rank-deficient system raise GuardError.
    """
    try:
        views = baseline.views if isinstance(baseline, ToolTipBaseline) else tuple(baseline)
    except TypeError as exc:
        raise _error("baseline 必须包含相机标定序列", locals()) from exc
    if (
        not views
        or any(not isinstance(v, ViewCalib) for v in views)
        or len({v.name for v in views}) != len(views)
    ):
        raise _error("views must be nonempty and have unique names", locals())
    names = {v.name for v in views}
    if set(observed) != names:
        raise _error(
            f"observed view mismatch: missing {sorted(names - set(observed))}, "
            f"unknown {sorted(set(observed) - names)}"
        , locals())
    if weights is not None and set(weights) - names:
        raise _error("weights contain unknown views", locals())
    if primary_view is not None and primary_view not in names:
        raise _error("unknown primary_view", locals())
    matrices, targets, scales = [], [], []
    for v in views:
        try:
            w = float((weights or {}).get(v.name, 1.0))
        except (TypeError, ValueError) as exc:
            raise _error(f"weight for {v.name} must be numeric", locals()) from exc
        if not np.isfinite(w) or w <= 0:
            raise _error(f"weight for {v.name} must be positive and finite", locals())
        # 主视图权重加倍；设备正面相机可作为主视图。
        if v.name == primary_view:
            w *= 2
        matrices.append(np.asarray(v.jacobian))
        targets.append(
            np.asarray(v.target_uv) - _array(observed[v.name], (2,), f"observed {v.name}")
        )
        scales.append(np.sqrt(w))
    a = np.vstack([j * w for j, w in zip(matrices, scales)])
    b = np.concatenate([t * w for t, w in zip(targets, scales)])
    if np.linalg.matrix_rank(a) < 3:
        raise _error(
            "view system is underdetermined or rank deficient; three axes must be observable"
        , locals())
    delta = np.linalg.lstsq(a, b, rcond=None)[0]
    residuals = {v.name: j @ delta - t for v, j, t in zip(views, matrices, targets)}
    mm = {v.name: np.linalg.pinv(j) @ residuals[v.name] for v, j in zip(views, matrices)}
    estimates, disagreements = {}, {}
    for axis, label in enumerate("xyz"):
        per_view = {}
        for v, j, t in zip(views, matrices, targets):
            col = j[:, axis]
            if np.linalg.norm(col) > np.linalg.norm(j) * 1e-10:
                # 固定另外两轴的联合解，比较两台相机对 Z 的估计。
                rem = t - j @ delta + col * delta[axis]
                per_view[v.name] = float(col @ rem / (col @ col))
        estimates[label] = per_view
        if len(per_view) >= 2:
            disagreements[label] = max(per_view.values()) - min(per_view.values())
    return CorrectionResult(
        delta, residuals, mm, float(np.linalg.cond(a)), estimates, disagreements
    )


def is_aligned(result: CorrectionResult, tol_mm: float) -> bool:
    """处理针尖标定数据，供对刀流程调用。
       Return whether max(abs(delta_xyz)) <= tol_mm; reject invalid tolerances.

    Per-axis tolerance matches independent platform axis limits; a vector norm
    would reject a correction with each individual axis within that limit.
    Residual consistency is assessed separately by check_quality.
    """
    if not np.isfinite(tol_mm) or tol_mm < 0:
        raise _error("tol_mm must be finite and non-negative", locals())
    return bool(np.max(np.abs(result.delta_xyz)) <= tol_mm)


@dataclass(frozen=True)
class QualityResult:
    """Independent model consistency and conditioning checks."""

    residuals_mm: dict[str, float]  # Euclidean norm of projected residual by view, mm
    condition_number: float  # weighted stacked matrix condition number
    residual_ok: bool  # all view norms are within the requested limit
    condition_ok: bool  # condition number is within the requested limit

    @property
    def ok(self) -> bool:
        """处理针尖标定数据，供对刀流程调用。
           Return whether both quality checks pass."""
        return self.residual_ok and self.condition_ok


def check_quality(
    result: CorrectionResult, max_residual_mm: float, max_condition: float
) -> QualityResult:
    """处理针尖标定数据，供对刀流程调用。
       Check projected residual norms (mm) and condition number independently.

    Return diagnostics without imposing an application policy. Limits must be
    finite and nonnegative, and max_condition must be at least one.
    """
    if not np.isfinite(max_residual_mm) or max_residual_mm < 0:
        raise _error("max_residual_mm must be finite and non-negative", locals())
    if not np.isfinite(max_condition) or max_condition < 1:
        raise _error("max_condition must be finite and at least one", locals())
    residuals = {name: float(np.linalg.norm(value)) for name, value in result.residuals_mm.items()}
    return QualityResult(
        residuals,
        result.condition_number,
        all(value <= max_residual_mm for value in residuals.values()),
        result.condition_number <= max_condition,
    )


@dataclass(frozen=True)
class CompensationResult:
    """Platform displacement and resulting compensation in the platform frame, mm."""

    delta_xyz: np.ndarray  # p_final - p0, mm
    offset_xyz: np.ndarray  # new compensation, mm


def compute_compensation(
    baseline: ToolTipBaseline,
    p_final: Sequence[float],
    *,
    convention: str = "target_plus_compensation",
) -> CompensationResult:
    """处理针尖标定数据，供对刀流程调用。
       Return delta and new offset in mm for the declared command convention.

    For target_plus_compensation add delta; for target_minus_compensation
    subtract delta. Invalid convention or p_final shape/values raise GuardError.
    """
    delta = _array(p_final, (3,), "p_final") - baseline.p0
    # §8：挑取坐标 = 目标 + 补偿，故新补偿 = 基准补偿 + Δ。
    # Δ = 最终模组坐标 - 基准 P0；反号约定仅为兼容原 API。
    if convention == "target_plus_compensation":
        sign = 1
    elif convention == "target_minus_compensation":
        sign = -1
    else:
        raise _error("convention must be target_plus_compensation or target_minus_compensation", locals())
    return CompensationResult(delta, np.asarray(baseline.offset_xyz) + sign * delta)
