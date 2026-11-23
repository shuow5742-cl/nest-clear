from dataclasses import dataclass

@dataclass(frozen=True)
class CameraCalibConfig:
    solve_timeout_s: float = 60.0  # 异步 worker 应答时限
    start_xyz: tuple[float, float, float] | None = None
    cols: int = 21
    rows: int = 16
    square_mm: float = 5.0
    marker_mm: float = 3.75
    dictionary: str = 'DICT_5X5_250'
    board_thickness_mm: float = 3.0
    xy_limit_mm: float = 25.0
    z_limit_mm: float = 5.0
    grid_half_mm: float = 20.0
    z_outer_mm: float = 5.0
    z_inner_mm: float = 2.5
    position_tolerance_mm: float = 0.05
    min_corner_ratio: float = 0.6
    saturation_level: int = 250
    max_saturated_fraction: float = 0.01
    glare_blob_min_px: int = 2000
    glare_corner_margin_px: int | None = None
    glare_min_lost_corners: int = 2
    max_rms_px: float = 0.25
    max_image_rms_px: float = 0.5
    max_repeat_px: float = 0.5  # 旧配置兼容，不参与判定
    module_repeatability_mm: float = 0.02
    repeat_drift_margin_mm: float = 0.01
    max_axis_scale_error: float = 0.002
    min_coverage: float = 0.9
