"""导出双相机针尖标定的求解、检测和基准文件接口。"""

from .core import (
    ViewCalib,
    ToolTipBaseline,
    JacobianFit,
    CorrectionResult,
    CompensationResult,
    fit_view_jacobian,
    solve_correction,
    is_aligned,
    check_quality,
    QualityResult,
    compute_compensation,
)
from .schema import ToolCalSchemaError, load_baseline, save_baseline
from .detector import TipDetector, TipDetection, SilhouetteTipDetector

__all__ = [
    "ViewCalib",
    "ToolTipBaseline",
    "JacobianFit",
    "CorrectionResult",
    "CompensationResult",
    "fit_view_jacobian",
    "solve_correction",
    "is_aligned",
    "compute_compensation",
    "check_quality",
    "QualityResult",
    "ToolCalSchemaError",
    "load_baseline",
    "save_baseline",
    "TipDetector",
    "TipDetection",
    "SilhouetteTipDetector",
]
