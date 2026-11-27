"""加载并严格校验业务配置，为通信和后续流程提供不可变配置。"""

from dataclasses import dataclass, field, fields, replace
import ipaddress
import logging
import math
from pathlib import Path
from typing import Any, NoReturn

import yaml

from .depth_config import DEFAULTS as DEPTH_DEFAULTS, METRICS
from .guard import GuardError
from .vision.hair.config import HairDetectConfig
from .tooltip.store import load_tools

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PLC:
    """保存 PLC 网络参数；本机 IP 仅用于记录，不用于绑定。"""

    ip: str
    local_ip: str
    port: int
    timeout_s: float
    write_readback: bool = True


@dataclass(frozen=True)
class Scan:
    """保存扫描范围；三个角点为 XY，扫描高度由 z 单独指定。"""

    corners: tuple[tuple[float, float], ...]
    step_x: float
    step_y: float
    z: float
    mode: str = "corners"
    origin_xy: tuple[float, float] = (0.0, 0.0)
    size_mm: tuple[float, float] = (182.7, 171.8)
    fov_mm: tuple[float, float] | None = None
    min_overlap_mm: float = 20.0
    x_direction: int = -1
    y_direction: int = 1


@dataclass(frozen=True)
class SupervisorConfig:
    """保存父程序轮询、看门狗和子进程启动参数。"""

    poll_s: float
    watchdog_s: float
    child_command: tuple[str, ...]
    startup_grace_s: float = 60.0


@dataclass(frozen=True)
class UIConfig:
    enabled: bool = True
    fullscreen: bool = False
    refresh_ms: int = 300
    python: str = "/usr/bin/python3"


@dataclass(frozen=True)
class VisionConfig:
    """保存相机模式、标定文件和占位检测器参数。"""

    camera_backend: str
    calibration: str
    pixels: tuple[tuple[float, ...], ...]
    dedup_radius_mm: float
    max_targets: int
    hair_detector: HairDetectConfig = field(default_factory=HairDetectConfig)
    detector: str = "hair"
    allow_placeholder_calibration: bool = False
    review_radius_mm: float = 0.3
    skip_radius_mm: float = 0.6
    max_picks_per_view: int = 30
    preview_window: bool = False
    responsibility_zone: bool = True
    ignore_regions_px: tuple[tuple[float, ...], ...] = ()
    max_fine_targets: int = 10
    fine_box_half_mm: float = 4.0
    cap_ratio: float = 1.2
    tool_check_min_attempts: int = 3
    tool_check_ratio: float = 0.5
    max_coarse_shots_per_view: int = 30
    scale_z_bin_mm: float = 0.1
    pick_region_mm: tuple[tuple[float, float], tuple[float, float]] | None = None


@dataclass(frozen=True)
class ToolCalibrationConfig:
    """双相机、检测器和对刀容差的配置。"""

    directory: str
    cameras: dict[str, dict[str, Any]]
    detection: dict[str, dict[str, Any]]
    position_tolerance_mm: float
    tip_tolerance_mm: float
    consistency_tolerance_mm: float
    max_iterations: int
    max_residual_mm: float
    max_condition: float
    z_disagreement_mm: float
    enabled: bool = True
    quality_hard_fail: bool = True
    unreliable_action: str = "fail"
    click_timeout_s: float = 120.0
    probe_step_mm: float = 0.2
    max_step_mm: float = 0.5
    max_total_mm: float = 2.0
    camera_controls: dict[str, Any] = field(default_factory=dict)
    auto_exposure: dict[str, Any] = field(default_factory=dict)
    exposure_scan: dict[str, Any] = field(default_factory=dict)
    frame_std_max_px: float = 0.3


@dataclass(frozen=True)
class Config:
    """冻结完整业务配置，避免运行时被各组件意外更改。"""

    plc: PLC
    scan: Scan
    arrival_tolerance_mm: float
    pick_depth_z: float
    tools: tuple[tuple[float, ...], ...]
    travel: tuple[tuple[float, ...], ...]
    camera_sn: str
    record_directory: str
    supervisor: SupervisorConfig
    vision: VisionConfig
    tool_calibration: ToolCalibrationConfig
    capture_timeout_s: float = 3.0
    exposure_us: float | None = None
    gain_db: float | None = None
    balance_ratio: tuple[float, ...] | None = None
    image_enabled: bool = True
    image_format: str = "png"
    image_quality: int = 3
    image_retention_days: int = 30
    image_queue_size: int = 32
    camera_image_size: tuple[int, int] = (4024, 3036)
    machine_id: str = "M01"
    camera_calibration: dict[str, Any] = field(default_factory=dict)
    ui: UIConfig = field(default_factory=UIConfig)
    ui_preview: dict[str, Any] = field(default_factory=dict)
    tool_offset: dict[str, Any] = field(default_factory=dict)
    pick_accuracy: dict[str, Any] = field(default_factory=dict)
    exposure_override_path: str = ""
    exposure_source: str = "配置"
    white_balance_override_path: str = ""
    white_balance_source: str = "配置"
    arrival_settle: dict[str, Any] = field(default_factory=lambda: {"sample_interval_s": .05, "settle_samples": 4, "jitter_tol_mm": .02, "settle_timeout_s": 1.0})
    test_mode: dict[str, Any] = field(default_factory=lambda: {"non_pickable": [f"T{i:02d}" for i in range(3, 31, 3)]})
    depth_calibration: dict[str, Any] = field(default_factory=lambda: dict(DEPTH_DEFAULTS))


def _fail(path: str, message: str) -> NoReturn:
    """将校验失败转为统一中文错误，供启动预检展示。"""
    raise GuardError("CONFIG-INVALID", message, f"配置项 {path}", "请检查配置文件中的字段、类型和取值。")


def _mapping(value: Any, path: str, keys: list[str]) -> dict[str, Any]:
    """检查映射的必需字段和未知字段，防止拼写错误被忽略。"""
    if not isinstance(value, dict):
        _fail(path, "应为映射")
    missing = set(keys) - value.keys()
    extra = value.keys() - set(keys)
    if missing or extra:
        _fail(path, f"缺少字段 {sorted(missing)}，未知字段 {sorted(extra)}")
    return value


def _travel_bounds(value: Any, path: str) -> tuple[tuple[float, float], ...]:
    """校验三轴行程映射，确保每轴为 [min, max] 两个有限数且 min < max。"""
    t = _mapping(value, path, ["x", "y", "z"])
    bounds = tuple(_vector(t[axis], f"{path}.{axis}", 2) for axis in "xyz")
    for axis, (lo, hi) in zip("xyz", bounds):
        if lo >= hi:
            _fail(f"{path}.{axis}", "最小值必须小于最大值")
    return bounds


def _depth_config(value, global_travel):
    path = "depth_calibration"
    if not isinstance(value, dict):
        _fail(path, "应为映射")
    settings = _mapping({**DEPTH_DEFAULTS, **value}, path, list(DEPTH_DEFAULTS))
    for key, default in DEPTH_DEFAULTS.items():
        name = f"{path}.{key}"
        item = settings[key]
        if type(default) is float:
            settings[key] = _number(item, name, True)
        if type(default) is int and (type(item) is not int or item < 1):
            _fail(name, "须为正整数")
    # 独立行程：配置 depth_calibration.travel 时校验每轴 [min<max] 与有限数；缺省回退全局 travel。
    if settings["travel"] is None:
        logger.info("配置项 depth_calibration.travel 未配置，下针深度标定将回退使用全局 travel %s", global_travel)
        effective_travel = global_travel
    else:
        effective_travel = _travel_bounds(settings["travel"], f"{path}.travel")
        settings["travel"] = effective_travel
    z_travel = effective_travel[2]
    lower = settings["camera_z_min_mm"]
    if lower is not None:
        lower = _number(lower, f"{path}.camera_z_min_mm")
        if not z_travel[0] <= lower < z_travel[1]:
            _fail(f"{path}.camera_z_min_mm", "须在 Z 行程内且低于行程上限")
        settings["camera_z_min_mm"] = lower
    for key, options in (
        ("fine_range_mode", ("auto", "fixed")),
        ("metric", METRICS),
        ("peak_method", ("gauss", "half_height")),
        ("exposure_mode", ("auto", "manual")),
    ):
        if settings[key] not in options:
            _fail(f"{path}.{key}", f"请选择 {options}")
    if settings["coarse_max_span_mm"] is not None:
        settings["coarse_max_span_mm"] = _number(settings["coarse_max_span_mm"], f"{path}.coarse_max_span_mm", True)
    if settings["manual_exposure_us"] is not None:
        settings["manual_exposure_us"] = _number(settings["manual_exposure_us"], f"{path}.manual_exposure_us", True)
    if settings["exposure_min_us"] >= settings["exposure_max_us"]:
        _fail(f"{path}.exposure_min_us", "必须小于 exposure_max_us")
    if settings["manual_exposure_us"] is not None and not (
        settings["exposure_min_us"] <= settings["manual_exposure_us"] <= settings["exposure_max_us"]
    ):
        _fail(f"{path}.manual_exposure_us", "须在 [exposure_min_us, exposure_max_us] 范围内")
    if not 0 < settings["exposure_target_p99"] < 250:
        _fail(f"{path}.exposure_target_p99", "须在 0 和 250 之间")
    if settings["fine_range_min_mm"] > settings["fine_range_max_mm"]:
        _fail(f"{path}.fine_range_min_mm", "不能大于 fine_range_max_mm")
    if settings["fine_step_min_mm"] > settings["fine_step_max_mm"]:
        _fail(f"{path}.fine_step_min_mm", "不能大于 fine_step_max_mm")
    if not 0 < settings["defocus_ratio"] < 1:
        _fail(f"{path}.defocus_ratio", "须在 0 和 1 之间")
    if not 0 < settings["blur_contrast_ratio"] < 1:
        _fail(f"{path}.blur_contrast_ratio", "须在 0 和 1 之间")
    if settings["min_peak_contrast"] <= 1.0:
        _fail(f"{path}.min_peak_contrast", "须大于 1.0")
    if not 0 < settings["dof_ratio"] < 1:
        _fail(f"{path}.dof_ratio", "须在 0 和 1 之间")
    if settings["settle_samples"] < 2:
        _fail(f"{path}.settle_samples", "须至少为 2")
    if not 5 <= settings["max_points"] <= 9:
        _fail(f"{path}.max_points", "须为 5 至 9")
    if not 3 <= settings["min_valid_points"] <= settings["max_points"]:
        _fail(f"{path}.min_valid_points", "须为 3 至 max_points")
    for key, count in (("point_order", 5), ("extra_points", 4)):
        labels = settings[key]
        if labels is None:
            continue
        if (not isinstance(labels, list) or len(labels) != count
                or any(label not in [f"T{i:02d}" for i in range(32)] for label in labels)
                or len(set(labels)) != count):
            _fail(f"{path}.{key}", f"须提供 {count} 个不同的菲林标签 T00–T31")
    return settings


def _number(value: Any, path: str, positive: bool = False) -> float:
    """验证有限数字和可选的正数约束，供各配置字段共用。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, f"应为有限数字，实际为 {value!r}")
    try:
        result = float(value)
    except OverflowError:
        _fail(path, f"数字超出范围：{value!r}")
    if not math.isfinite(result):
        _fail(path, f"应为有限数字，实际为 {value!r}")
    if positive and result <= 0:
        _fail(path, f"应大于零，实际为 {value!r}")
    return result


def _vector(value: Any, path: str, size: int) -> tuple[float, ...]:
    """验证固定长度数字列表，供角点、补偿和行程使用。"""
    if not isinstance(value, list) or len(value) != size:
        _fail(path, f"应为包含 {size} 个数字的列表")
    return tuple(_number(item, f"{path}[{index}]") for index, item in enumerate(value))


def _string(value: Any, path: str) -> str:
    """拒绝空字符串，避免设备标识和记录路径静默缺失。"""
    if not isinstance(value, str) or not value.strip():
        _fail(path, "应为非空字符串")
    return value


def _ip(value: Any, path: str) -> str:
    """校验 IP 地址，供 PLC 网络参数和记录使用。"""
    text = _string(value, path)
    try:
        ipaddress.ip_address(text)
    except ValueError:
        _fail(path, f"不是有效 IP 地址：{text}")
    return text


def _hair_config(value: Any) -> HairDetectConfig:
    """Validate optional reference-pixel tuning without silently accepting typos."""
    path = "vision.hair_detector"
    if not isinstance(value, dict):
        _fail(path, "应为映射")
    data = dict(value)
    if "um_per_px" in data:
        _fail(path + ".um_per_px", "旧配置键已停用，请改为 expected_um_per_px；实际尺度由标定推算")
    pick = data.pop("pick_point", {})
    if not isinstance(pick, dict):
        _fail(path + ".pick_point", "应为映射")
    extra_pick = set(pick) - {"mode", "thick_pow", "central_pow", "dens_pow"}
    if extra_pick:
        _fail(path + ".pick_point", f"未知字段 {sorted(extra_pick)}")
    for key, val in pick.items():
        target = "pick_point" if key == "mode" else key
        if target in data:
            _fail(path + ".pick_point", f"重复配置 {target}")
        data[target] = val
    defaults = HairDetectConfig()
    allowed = {f.name for f in fields(HairDetectConfig)}
    extra = set(data) - allowed
    if extra:
        _fail(path, f"未知字段 {sorted(extra)}")
    values = {}
    fractions = {"scale_tolerance", "fiber_dens_th", "fiber_mean_dens_min", "contain_overlap", "hair_mid_band", "trace_adapt", "max_gray_ratio", "min_dark_frac"}
    scales = {"line_blur_scale", "seed_grow_scale"}
    nonnegative = {"dens_pow", "central_pow", "thick_pow", "min_width_mm", "max_gray_ratio", "min_dark_frac", "seed_grow_close"}
    gray = {"hair_dark_max", "line_dark_max", "fiber_fine_gray_max", "trace_ridge_max"}
    for name, value in data.items():
        item_path = f"{path}.{name}"
        default = getattr(defaults, name)
        if name == "pick_point":
            if value != "thick_central":
                _fail(item_path, "仅支持 thick_central")
        elif isinstance(default, bool):
            if type(value) is not bool:
                _fail(item_path, "应为布尔值")
        elif isinstance(default, tuple):
            if not isinstance(value, (list, tuple)) or len(value) != 2 or any(type(v) is not int or v < 1 for v in value) or value[0] >= value[1]:
                _fail(item_path, "应为两个递增的正整数")
            value = tuple(value)
        else:
            if isinstance(default, int) and type(value) is not int:
                _fail(item_path, "应为整数")
            number = _number(value, item_path)
            if number < 0 or (number == 0 and name not in nonnegative):
                _fail(item_path, "应为正数" if name not in nonnegative else "应为非负数")
            if name in fractions and number > 1:
                _fail(item_path, "应在 0 到 1 之间")
            if name in scales and number > 1:
                _fail(item_path, "应大于 0 且不超过 1")
            if name in gray and number > 255:
                _fail(item_path, "灰度应在 0 到 255 之间")
            if name.endswith("_k") or name in {"fiber_dens_win_ds", "fiber_close_k_ds"}:
                if number < 3 or number % 2 != 1:
                    _fail(item_path, "核尺寸应为不小于 3 的奇数")
        values[name] = value
    result = HairDetectConfig(**values)
    if result.fiber_fine_gray_max <= result.hair_dark_max:
        _fail(path + ".fiber_fine_gray_max", "必须大于 hair_dark_max，否则取点暗度归一化除数为零或负数")
    ratio = result.reference_um_per_px / result.um_per_px
    if not math.isfinite(ratio) or ratio == 0 or not math.isfinite(ratio * ratio):
        _fail(path, "参考尺度与当前尺度比例超出数值范围")
    return result


def _camera_calibration(data: Any) -> dict[str, Any]:
    """校验共享接口字段；缺省值由运行时 CameraCalibConfig 提供。"""
    integers = {"cols", "rows", "saturation_level", "glare_blob_min_px", "glare_min_lost_corners"}
    fractions = {"min_corner_ratio", "max_saturated_fraction", "min_coverage", "max_axis_scale_error"}
    numbers = {"square_mm", "marker_mm", "board_thickness_mm", "xy_limit_mm", "z_limit_mm",
               "grid_half_mm", "z_outer_mm", "z_inner_mm", "position_tolerance_mm", "max_rms_px",
               "max_image_rms_px", "max_repeat_px", "module_repeatability_mm", "repeat_drift_margin_mm", "solve_timeout_s"} | fractions
    allowed = integers | numbers | {"dictionary", "start_xyz", "glare_corner_margin_px"}
    if not isinstance(data, dict):
        _fail("camera_calibration", "应为映射")
    if set(data) - allowed:
        _fail("camera_calibration", f"未知字段 {sorted(set(data) - allowed)}")
    for key, value in data.items():
        path = f"camera_calibration.{key}"
        if key == "start_xyz":
            if value is not None:
                _vector(value, path, 3)
        elif key == "glare_corner_margin_px":
            if value is not None and (type(value) is not int or value < 0):
                _fail(path, "应为非负整数或 null")
        elif key == "dictionary":
            _string(value, path)
        elif key in integers:
            if type(value) is not int or value < (2 if key in {"cols", "rows"} else 1) or (key == "saturation_level" and value > 255):
                _fail(path, "整数超出允许范围")
        else:
            n = _number(value, path)
            nonnegative = {"board_thickness_mm", "max_saturated_fraction", "max_axis_scale_error", "module_repeatability_mm", "repeat_drift_margin_mm"}
            if n < 0 or (n == 0 and key not in nonnegative) or (key in fractions and n > 1):
                _fail(path, "应为正数" + ("且不超过 1" if key in fractions else ""))
    defaults = {"square_mm": 5.0, "marker_mm": 3.75, "xy_limit_mm": 25.0,
                "z_limit_mm": 5.0, "grid_half_mm": 20.0, "z_outer_mm": 5.0, "z_inner_mm": 2.5}
    v = {**defaults, **data}
    if v["marker_mm"] >= v["square_mm"]:
        _fail("camera_calibration.marker_mm", "必须小于 square_mm")
    if v["grid_half_mm"] > v["xy_limit_mm"] or v["z_outer_mm"] > v["z_limit_mm"] or v["z_inner_mm"] > v["z_outer_mm"]:
        _fail("camera_calibration", "标定网格或 Z 偏移超出相对行程限制")
    return dict(data)


def load_config(path: str | Path, *, require_tools: bool = True) -> Config:
    """读取 YAML 并校验所有字段和行程，供启动预检调用。"""
    try:
        with open(path, encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except (OSError, yaml.YAMLError) as exc:
        raise GuardError(
            "CONFIG-LOAD", f"无法读取配置：{exc}", str(path), "检查文件路径和 YAML 格式。"
        ) from exc
    if isinstance(data, dict):
        if data.get("run_mode") == "commtest":
            _fail("run_mode", "通信联调模式已删除，请改用 config/nest_clear.yaml")
        if data.get("run_mode") == "production":
            data.pop("run_mode")
        data = {"arrival_settle": {}, "depth_calibration": {}, "test_mode": {}, "pick_accuracy": {}, "tool_offset": {}, "camera_calibration": {}, "ui": {}, "ui_preview": {}, **data}
    root = _mapping(
        data,
        "根",
        [
            "test_mode",
            "plc",
            "scan",
            "arrival_tolerance_mm",
            "arrival_settle",
            "pick_depth_z",
            "tool_calibration",
            "camera_calibration",
            "tool_offset",
            "pick_accuracy",
            "depth_calibration",
            "ui_preview",
            "travel",
            "camera",
            "record",
            "supervisor",
            "vision",
            "ui",
        ],
    )
    test_mode = _mapping({"non_pickable": [f"T{i:02d}" for i in range(3, 31, 3)], **root["test_mode"]}, "test_mode", ["non_pickable"]) if isinstance(root["test_mode"], dict) else _fail("test_mode", "应为映射")
    if not isinstance(test_mode["non_pickable"], list) or any(not isinstance(x, str) or len(x) != 3 or x[0] != "T" or not x[1:].isdigit() or not 0 <= int(x[1:]) <= 31 for x in test_mode["non_pickable"]):
        _fail("test_mode.non_pickable", "须为 T00–T31 列表")
    from nest_clear.tooloffset.session import DEFAULTS as offset_defaults
    if not isinstance(root["tool_offset"], dict):
        _fail("tool_offset", "应为映射")
    new_offset_defaults = {
        "ref_tip_top": {
            "bg_kernel_px": 101,
            "min_contrast": 25.0,
            "contrast_ratio": 0.18,
            "min_area_px": 150,
            "max_area_px": 500000,
            "min_apex_angle_deg": 5.0,
            "max_apex_angle_deg": 120.0,
            "max_edge_rms_px": 1.0,
            "max_tip_offset_px": 8.0,
            "tip_skip_px": 4,
            "base_trim_frac": 0.25,
        },
        "triangulation": {
            "frames_per_shot": 3,
            "frame_std_max_px": 0.3,
            "baseline_min_mm": 20.0,
            "baseline_max_mm": 30.0,
            "min_views": 3,
            "max_views": 5,
            "max_ray_residual_mm": 0.02,
            "min_baseline_ratio": 0.15,
            "approx_tip_z_mm": None,
            "station_bounds_mm": None,
            "center_margin_frac": 0.10,
            "position_tolerance_mm": 0.1,
        },
        "ref_tip_side": {
            "min_contrast": 25.0,
            "contrast_ratio": 0.22,
            "min_area_px": 40,
            "min_height_px": 15,
            "min_cone_angle_deg": 4.0,
            "max_edge_rms_px": 1.5,
            "border_margin_px": 2,
        },
        "align": {
            "probe_step_xy_mm": 0.2,
            "probe_step_z_mm": 2.0,
            "align_tol_mm": 0.05,
            "max_align_iterations": 10,
            "max_step_mm": 0.5,
            "max_total_mm": 3.0,
            "min_gap_mm": 1.0,
            "consistency_tolerance_mm": 0.05,
            "position_tolerance_mm": 0.1,
            "fallback_um_per_px": 15.2,
            "max_jacobian_rms_px": 2.0,
            "max_jacobian_condition": 300.0,
            "frame_std_max_px": 0.3,
        },
    }
    offset = _mapping({**offset_defaults, **new_offset_defaults, **root["tool_offset"]}, "tool_offset", list(offset_defaults) + list(new_offset_defaults))
    for key in ("center_tol_mm", "outlier_mm", "se_tol_mm", "edge_warn_mm", "card_thickness_mm", "ring_cross_tol_mm", "settle_window_s", "jitter_tol_mm"):
        offset[key] = _number(offset[key], "tool_offset." + key, True)
    for key in ("min_points", "max_points", "settle_samples"):
        if type(offset[key]) is not int or offset[key] < 2:
            _fail("tool_offset." + key, "须为不小于 2 的整数")
    if offset["max_points"] < offset["min_points"]:
        _fail("tool_offset.max_points", "不能小于 min_points")
    if not isinstance(offset["prior_offset_mm"], (list, tuple)) or len(offset["prior_offset_mm"]) != 2:
        _fail("tool_offset.prior_offset_mm", "须为两个有限数字")
    offset["prior_offset_mm"] = [_number(v, "tool_offset.prior_offset_mm") for v in offset["prior_offset_mm"]]
    for key in ("point_order", "edge_targets"):
        if not isinstance(offset[key], list) or not offset[key] or any(not isinstance(v, str) or len(v) != 3 or v[0] != "T" or not v[1:].isdigit() or int(v[1:]) > 31 for v in offset[key]):
            _fail("tool_offset." + key, "须为 T00–T31 列表")
    top_cfg = _mapping({**new_offset_defaults["ref_tip_top"], **(offset["ref_tip_top"] if isinstance(offset["ref_tip_top"], dict) else _fail("tool_offset.ref_tip_top", "应为映射"))}, "tool_offset.ref_tip_top", list(new_offset_defaults["ref_tip_top"]))
    if type(top_cfg["bg_kernel_px"]) is not int or top_cfg["bg_kernel_px"] < 3 or top_cfg["bg_kernel_px"] % 2 != 1:
        _fail("tool_offset.ref_tip_top.bg_kernel_px", "须为不小于 3 的奇数")
    for key in ("min_area_px", "max_area_px", "tip_skip_px"):
        if type(top_cfg[key]) is not int or top_cfg[key] < 1:
            _fail("tool_offset.ref_tip_top." + key, "须为正整数")
    if top_cfg["max_area_px"] < top_cfg["min_area_px"]:
        _fail("tool_offset.ref_tip_top.max_area_px", "不能小于 min_area_px")
    for key in ("min_contrast", "contrast_ratio", "min_apex_angle_deg", "max_apex_angle_deg", "max_edge_rms_px", "max_tip_offset_px", "base_trim_frac"):
        top_cfg[key] = _number(top_cfg[key], "tool_offset.ref_tip_top." + key, True)
    if not 0 < top_cfg["contrast_ratio"] < 1 or not 0 < top_cfg["base_trim_frac"] < 0.8 or top_cfg["max_apex_angle_deg"] <= top_cfg["min_apex_angle_deg"]:
        _fail("tool_offset.ref_tip_top", "比例或顶角范围无效")
    offset["ref_tip_top"] = top_cfg
    tri_cfg = _mapping({**new_offset_defaults["triangulation"], **(offset["triangulation"] if isinstance(offset["triangulation"], dict) else _fail("tool_offset.triangulation", "应为映射"))}, "tool_offset.triangulation", list(new_offset_defaults["triangulation"]))
    for key in ("frames_per_shot", "min_views", "max_views"):
        if type(tri_cfg[key]) is not int or tri_cfg[key] < 1:
            _fail("tool_offset.triangulation." + key, "须为正整数")
    if tri_cfg["min_views"] < 3 or tri_cfg["max_views"] < tri_cfg["min_views"]:
        _fail("tool_offset.triangulation", "视角数须满足 3 <= min_views <= max_views")
    for key in ("frame_std_max_px", "baseline_min_mm", "baseline_max_mm", "max_ray_residual_mm", "min_baseline_ratio", "center_margin_frac", "position_tolerance_mm"):
        tri_cfg[key] = _number(tri_cfg[key], "tool_offset.triangulation." + key, True)
    if tri_cfg["baseline_max_mm"] < tri_cfg["baseline_min_mm"] or not 0 < tri_cfg["center_margin_frac"] < 0.5:
        _fail("tool_offset.triangulation", "基线范围或画面边距无效")
    if tri_cfg["approx_tip_z_mm"] is not None:
        tri_cfg["approx_tip_z_mm"] = _number(tri_cfg["approx_tip_z_mm"], "tool_offset.triangulation.approx_tip_z_mm")
    if tri_cfg["station_bounds_mm"] is not None:
        sb = tri_cfg["station_bounds_mm"]
        if not isinstance(sb, dict) or not set(sb) <= {"x", "y", "z"}:
            _fail("tool_offset.triangulation.station_bounds_mm", "应为含 x/y/z 键的映射或 null")
        parsed_sb = {}
        for axis in ("x", "y", "z"):
            if axis in sb and sb[axis] is not None:
                vec = _vector(sb[axis], f"tool_offset.triangulation.station_bounds_mm.{axis}", 2)
                if vec[0] >= vec[1]:
                    _fail(f"tool_offset.triangulation.station_bounds_mm.{axis}", "最小值必须小于最大值")
                parsed_sb[axis] = list(vec)
        tri_cfg["station_bounds_mm"] = parsed_sb
    offset["triangulation"] = tri_cfg
    side_cfg = _mapping({**new_offset_defaults["ref_tip_side"], **(offset["ref_tip_side"] if isinstance(offset["ref_tip_side"], dict) else _fail("tool_offset.ref_tip_side", "应为映射"))}, "tool_offset.ref_tip_side", list(new_offset_defaults["ref_tip_side"]))
    for key in ("min_area_px", "min_height_px"):
        if type(side_cfg[key]) is not int or side_cfg[key] < 1:
            _fail("tool_offset.ref_tip_side." + key, "须为正整数")
    if type(side_cfg["border_margin_px"]) is not int or side_cfg["border_margin_px"] < 0:
        _fail("tool_offset.ref_tip_side.border_margin_px", "须为非负整数")
    for key in ("min_contrast", "contrast_ratio", "min_cone_angle_deg", "max_edge_rms_px"):
        side_cfg[key] = _number(side_cfg[key], "tool_offset.ref_tip_side." + key, True)
    if not 0 < side_cfg["contrast_ratio"] < 1:
        _fail("tool_offset.ref_tip_side.contrast_ratio", "应在 0 到 1 之间")
    offset["ref_tip_side"] = side_cfg
    align_cfg = _mapping({**new_offset_defaults["align"], **(offset["align"] if isinstance(offset["align"], dict) else _fail("tool_offset.align", "应为映射"))}, "tool_offset.align", list(new_offset_defaults["align"]))
    if type(align_cfg["max_align_iterations"]) is not int or align_cfg["max_align_iterations"] < 1:
        _fail("tool_offset.align.max_align_iterations", "须为正整数")
    for key in ("probe_step_xy_mm", "probe_step_z_mm", "align_tol_mm", "max_step_mm", "max_total_mm", "min_gap_mm", "consistency_tolerance_mm", "position_tolerance_mm", "fallback_um_per_px", "max_jacobian_rms_px", "max_jacobian_condition", "frame_std_max_px"):
        align_cfg[key] = _number(align_cfg[key], "tool_offset.align." + key, True)
    if align_cfg["probe_step_z_mm"] > 2.0:
        _fail("tool_offset.align.probe_step_z_mm", "Z 向探测步长不得超过 2.0 mm")
    if align_cfg["max_jacobian_condition"] < 1.0:
        _fail("tool_offset.align.max_jacobian_condition", "条件数门限须 >= 1.0")
    offset["align"] = align_cfg
    camcal = _camera_calibration(root["camera_calibration"])
    start = camcal.get("start_xyz")
    plc_data = root["plc"]
    if isinstance(plc_data, dict):
        for key in ("stall_warn_s", "comm_fail_timeout_s"):
            if key in plc_data:
                _fail(f"plc.{key}", "该键已删除：算法端不做超时，超时由 PLC 负责")
        plc_data = {"write_readback": True, **plc_data}
    p = _mapping(plc_data, "plc", ["ip", "local_ip", "port", "timeout_s", "write_readback"])
    port = p["port"]
    if type(port) is not int or not 1 <= port <= 65535:
        _fail("plc.port", "端口应为 1 到 65535 的整数")
    if type(p["write_readback"]) is not bool:
        _fail("plc", "回读开关须为布尔值")
    plc = PLC(
        _ip(p["ip"], "plc.ip"),
        _ip(p["local_ip"], "plc.local_ip"),
        port,
        _number(p["timeout_s"], "plc.timeout_s", True),
        p["write_readback"],
    )
    travel = _travel_bounds(root["travel"], "travel")
    depth_calibration = _depth_config(root["depth_calibration"], travel)
    if start is not None and any(not lo <= value <= hi for value, (lo, hi) in zip(start, travel)):
        _fail("camera_calibration.start_xyz", "必须在 travel 范围内")
    scan_data = {"mode": "corners", "tray": {"origin_xy": [0, 0], "size_mm": [182.7, 171.8], "x_direction": -1, "y_direction": 1}, "fov_mm": None, "min_overlap_mm": 20, **root["scan"]}
    s = _mapping(scan_data, "scan", ["corners", "step_x", "step_y", "z", "mode", "tray", "fov_mm", "min_overlap_mm"])
    if s["mode"] not in ("corners", "tray"):
        _fail("scan.mode", "应为 corners 或 tray")
    tray = _mapping(s["tray"], "scan.tray", ["origin_xy", "size_mm", "x_direction", "y_direction"])
    if any(type(tray[k]) is not int or tray[k] not in (-1, 1) for k in ("x_direction", "y_direction")):
        _fail("scan.tray", "x_direction 和 y_direction 必须为 1 或 -1")
    if not isinstance(s["corners"], list) or len(s["corners"]) != 3:
        _fail("scan.corners", "必须提供三个 XY 角点")
    corners = tuple(_vector(point, f"scan.corners[{index}]", 2) for index, point in enumerate(s["corners"]))
    for index, point in enumerate(corners):
        for axis, value, bounds in zip("xy", point, travel):
            if not bounds[0] <= value <= bounds[1]:
                _fail(f"scan.corners[{index}].{axis}", f"角点 {value} 超出行程 {bounds}")
    z = _number(s["z"], "scan.z")
    depth = _number(root["pick_depth_z"], "pick_depth_z")
    if not travel[2][0] <= z <= travel[2][1]:
        _fail("scan.z", "扫描高度超出 Z 行程")
    if not travel[2][0] <= depth <= travel[2][1]:
        _fail("pick_depth_z", "下针深度超出 Z 行程")
    scan = Scan(
        corners, _number(s["step_x"], "scan.step_x", True), _number(s["step_y"], "scan.step_y", True), z,
        s["mode"], _vector(tray["origin_xy"], "scan.tray.origin_xy", 2),
        _vector(tray["size_mm"], "scan.tray.size_mm", 2),
        None if s["fov_mm"] is None else _vector(s["fov_mm"], "scan.fov_mm", 2),
        _number(s["min_overlap_mm"], "scan.min_overlap_mm"), tray["x_direction"], tray["y_direction"]
    )
    camera = _mapping({"capture_timeout_s": 3.0, "exposure_us": None, "gain_db": None, "balance_ratio": None, "image_size": [4024, 3036], **root["camera"]}, "camera", ["sn", "capture_timeout_s", "exposure_us", "gain_db", "balance_ratio", "image_size"])
    image_size = camera["image_size"]
    if not isinstance(image_size, (list, tuple)) or len(image_size) != 2 or any(type(n) is not int or n <= 0 for n in image_size):
        _fail("camera.image_size", "必须为 [宽, 高] 两个正整数")
    capture_timeout = _number(camera["capture_timeout_s"], "camera.capture_timeout_s", True)
    exposure = None if camera["exposure_us"] is None else _number(camera["exposure_us"], "camera.exposure_us", True)
    gain = None if camera["gain_db"] is None else _number(camera["gain_db"], "camera.gain_db")
    balance = None if camera["balance_ratio"] is None else _vector(camera["balance_ratio"], "camera.balance_ratio", 3)
    record_data = root["record"]
    if isinstance(record_data, dict):
        record_data = {
            "image_enabled": True,
            "image_format": "png",
            "image_quality": 3,
            "image_retention_days": 30,
            "image_queue_size": 32,
            "machine_id": "M01",
            **record_data,
        }
    record = _mapping(
        record_data,
        "record",
        [
            "directory",
            "image_enabled",
            "image_format",
            "image_quality",
            "image_retention_days",
            "image_queue_size",
            "machine_id",
        ],
    )
    if type(record["image_enabled"]) is not bool or record["image_format"] not in ("png", "jpg"):
        _fail("record", "图片开关应为布尔值，格式应为 png 或 jpg")
    limit = 9 if record["image_format"] == "png" else 100
    if (
        type(record["image_quality"]) is not int
        or not 0 <= record["image_quality"] <= limit
        or type(record["image_retention_days"]) is not int
        or record["image_retention_days"] < 0
    ):
        _fail("record", "压缩质量或保留天数无效")
    if type(record["image_queue_size"]) is not int or record["image_queue_size"] < 1:
        _fail("record.image_queue_size", "图片队列容量必须为正整数")
    sup = _mapping(
        root["supervisor"], "supervisor", ["poll_s", "watchdog_s", "child_command", "startup_grace_s"]
    )
    command = sup["child_command"]
    if not isinstance(command, list) or not command:
        _fail("supervisor.child_command", "应为非空命令参数列表")
    supervisor = SupervisorConfig(
        _number(sup["poll_s"], "supervisor.poll_s", True),
        _number(sup["watchdog_s"], "supervisor.watchdog_s", True),
        tuple(_string(arg, "supervisor.child_command") for arg in command),
        _number(sup["startup_grace_s"], "supervisor.startup_grace_s", True),
    )
    vision_data = root["vision"]
    if isinstance(vision_data, dict):
        vision_data = {"pick_region_mm": None, "scale_z_bin_mm": 0.1, "responsibility_zone": True, "ignore_regions_px": [], "hair_detector": {}, "detector": "hair", "allow_placeholder_calibration": False, "review_radius_mm": 0.3, "skip_radius_mm": 0.6, "max_picks_per_view": 30, "preview": {"window": False}, **vision_data}
    v = _mapping(
        vision_data,
        "vision",
        ["camera_backend", "calibration", "pixels", "dedup_radius_mm", "max_targets", "hair_detector", "detector", "allow_placeholder_calibration", "review_radius_mm", "skip_radius_mm", "max_picks_per_view", "preview", "responsibility_zone", "ignore_regions_px", "max_fine_targets", "fine_box_half_mm", "cap_ratio", "tool_check_min_attempts", "tool_check_ratio", "max_coarse_shots_per_view", "scale_z_bin_mm", "pick_region_mm"],
    )
    if v["camera_backend"] not in ("mock", "daheng"):
        _fail("vision.camera_backend", "应为 mock 或 daheng")
    exposure_path = Path(v['calibration'])
    if not exposure_path.is_absolute():
        exposure_path = Path(path).resolve().parent / exposure_path
    from nest_clear.calibration_bootstrap import ensure_calibration_files
    ensure_calibration_files(exposure_path.parent)
    exposure_path = exposure_path.parent / 'camera_exposure.yaml'
    exposure_source = '配置'
    if exposure_path.exists():
        try:
            with exposure_path.open(encoding='utf-8') as f:
                override = yaml.safe_load(f)
            if not isinstance(override, dict):
                raise ValueError('曝光覆盖文件须为映射')
            exposure = _number(override.get('exposure_us'), 'camera_exposure.exposure_us', True)
            exposure_source = '自测 ' + str(override.get('time') or '时间未知')
        except (OSError, yaml.YAMLError, ValueError) as exc:
            _fail('camera_exposure', str(exc))
    wb_path = exposure_path.parent / 'camera_white_balance.yaml'
    wb_source = '配置'
    if wb_path.exists():
        try:
            with wb_path.open(encoding='utf-8') as f:
                override = yaml.safe_load(f)
            if not isinstance(override, dict):
                raise ValueError('白平衡覆盖文件须为映射')
            balance = _vector(override.get('balance_ratio'), 'camera_white_balance.balance_ratio', 3)
            if any(x <= 0 for x in balance):
                raise ValueError('白平衡参数须为正数')
            wb_source = '自测 ' + str(override.get('time') or '时间未知')
        except (OSError, yaml.YAMLError, ValueError) as exc:
            _fail('camera_white_balance', str(exc))
    if not isinstance(v["pixels"], list):
        _fail("vision.pixels", "应为 [u, v, score] 列表")
    pixels = tuple(_vector(item, "vision.pixels", 3) for item in v["pixels"])
    if any(not 0 <= pixel[2] <= 1 for pixel in pixels):
        _fail("vision.pixels", "目标分数必须在 0 到 1 之间")
    radius = _number(v["dedup_radius_mm"], "vision.dedup_radius_mm")
    if radius < 0 or type(v["max_targets"]) is not int or v["max_targets"] < 1:
        _fail("vision", "去重半径应非负，最大目标数应为正整数")
    if v["detector"] not in ("hair", "stub") or type(v["allow_placeholder_calibration"]) is not bool:
        _fail("vision", "检测器或占位标定开关无效")
    preview = _mapping(v["preview"], "vision.preview", ["window"])
    if type(preview["window"]) is not bool or type(v["max_picks_per_view"]) is not int or v["max_picks_per_view"] < 1:
        _fail("vision", "预览开关或每视野上限无效")
    review = _number(v["review_radius_mm"], "vision.review_radius_mm", True)
    skip = _number(v["skip_radius_mm"], "vision.skip_radius_mm", True)
    if type(v["responsibility_zone"]) is not bool or not isinstance(v["ignore_regions_px"], list):
        _fail("vision", "负责区开关须为布尔值，屏蔽区须为矩形列表")
    regions = tuple(_vector(r, f"vision.ignore_regions_px[{i}]", 4) for i, r in enumerate(v["ignore_regions_px"]))
    if any(x1 > x2 or y1 > y2 for x1, y1, x2, y2 in regions):
        _fail("vision.ignore_regions_px", "矩形须按左上、右下顺序填写")
    for key in ("max_fine_targets", "tool_check_min_attempts", "max_coarse_shots_per_view"):
        if type(v[key]) is not int or v[key] < 1:
            _fail(f"vision.{key}", "应为正整数")
    fine_half = _number(v["fine_box_half_mm"], "vision.fine_box_half_mm", True)
    from nest_clear.pickaccuracy.session import DEFAULTS as accuracy_defaults
    if not isinstance(root['pick_accuracy'], dict):
        _fail('pick_accuracy', '应为映射')
    accuracy = _mapping({**accuracy_defaults, **root['pick_accuracy']}, 'pick_accuracy', list(accuracy_defaults))
    for key in accuracy_defaults:
        if key in ('max_points', 'min_points_a', 'min_points_b', 'min_infield_corners', 'z_settle_samples'):
            bounds = {'max_points': (1, 1000), 'min_points_a': (3, 9), 'min_points_b': (3, 16), 'min_infield_corners': (0, 4), 'z_settle_samples': (2, 100)}
            lo, hi = bounds[key]
            if type(accuracy[key]) is not int or not lo <= accuracy[key] <= hi: _fail('pick_accuracy.' + key, f'须为 {lo} 到 {hi} 的整数')
        else:
            accuracy[key] = _number(accuracy[key], 'pick_accuracy.' + key, key not in ('calib_tool_z_mm', 'z_flat_tol_mm'))
            if key == 'z_flat_tol_mm' and accuracy[key] < 0:
                _fail('pick_accuracy.z_flat_tol_mm', '不能为负数')
    if accuracy['center_tol_mm'] > fine_half / 2:
        _fail('pick_accuracy.center_tol_mm', '不能超过精定位框半宽的一半')
    cap_ratio = _number(v["cap_ratio"], "vision.cap_ratio", True)
    tool_ratio = _number(v["tool_check_ratio"], "vision.tool_check_ratio")
    if not 0 <= tool_ratio <= 1:
        _fail("vision.tool_check_ratio", "应在 0 到 1 之间")
    region = None
    if v["pick_region_mm"] is not None:
        region_data = _mapping(v["pick_region_mm"], "vision.pick_region_mm", ["x", "y"])
        region = tuple(_vector(region_data[axis], f"vision.pick_region_mm.{axis}", 2) for axis in ("x", "y"))
        if any(low >= high for low, high in region):
            _fail("vision.pick_region_mm", "矩形每个轴的最小值必须小于最大值")
    vision = VisionConfig(
        v["camera_backend"], _string(v["calibration"], "vision.calibration"), pixels, radius, v["max_targets"],
        _hair_config(v["hair_detector"]), v["detector"], v["allow_placeholder_calibration"],
        review, skip, v["max_picks_per_view"], preview["window"], v["responsibility_zone"], regions,
        v["max_fine_targets"], fine_half, cap_ratio, v["tool_check_min_attempts"], tool_ratio,
        v["max_coarse_shots_per_view"], _number(v["scale_z_bin_mm"], "vision.scale_z_bin_mm", True), region
    )
    if scan.fov_mm is None:
        scan = replace(scan, fov_mm=tuple(vision.hair_detector.expected_um_per_px * n / 1000 for n in (4024, 3036)))
    tc_data = root["tool_calibration"]
    if isinstance(tc_data, dict):
        tc_data = {"enabled": True, "quality_hard_fail": True, "unreliable_action": "fail", "click_timeout_s": 120.0, "probe_step_mm": .2, "max_step_mm": .5, "max_total_mm": 2., "camera_controls": {}, "auto_exposure": {}, "exposure_scan": {}, "frame_std_max_px": .3, "usb_vendor_id": "05a3", "usb_product_id": "9230", **tc_data}
    tc = _mapping(tc_data, "tool_calibration", [
        "directory", "cameras", "detection", "position_tolerance_mm", "tip_tolerance_mm",
        "consistency_tolerance_mm", "max_iterations", "max_residual_mm", "max_condition",
        "z_disagreement_mm", "enabled", "quality_hard_fail", "unreliable_action", "click_timeout_s", "probe_step_mm", "max_step_mm", "max_total_mm", "camera_controls", "auto_exposure", "exposure_scan", "frame_std_max_px", "usb_vendor_id", "usb_product_id",
    ])
    if type(tc["enabled"]) is not bool or type(tc["quality_hard_fail"]) is not bool:
        _fail("tool_calibration.enabled", "应为布尔值")
    directory = _string(tc["directory"], "tool_calibration.directory")
    cameras = _mapping(tc["cameras"], "tool_calibration.cameras", ["front", "side"])
    detection = _mapping(tc["detection"], "tool_calibration.detection", ["front", "side"])
    from .devices.v4l2_controls import DEFAULTS as control_defaults
    from .devices.tooltip_exposure import DEFAULTS as exposure_defaults
    def validate_controls(value, path):
        if not isinstance(value, dict): _fail(path, "应为映射")
        ranges = {'width': (1, 10000), 'height': (1, 10000), 'exposure': (1, 5000), 'auto_exposure': (1, 1), 'dynamic_framerate': (0, 0), 'white_balance_auto': (0, 0), 'white_balance_temperature': (2800, 6500), 'sharpness': (0, 0), 'backlight_compensation': (0, 0), 'gain': (0, 0), 'gamma': (72, 500), 'brightness': (-255, 255), 'contrast': (0, 255), 'saturation': (0, 255), 'hue': (-180, 180), 'power_line_frequency': (1, 1)}
        for key, val in value.items():
            if key not in ranges: _fail(path, f"未知字段 {key}")
            if val is None and control_defaults[key] is None: continue
            low, high = ranges[key]
            if type(val) is not int or not low <= val <= high: _fail(path + '.' + key, f"须为 {low}–{high} 的整数")
        return dict(value)
    common_controls = {**control_defaults, **validate_controls(tc['camera_controls'], 'tool_calibration.camera_controls')}
    ae = tc['auto_exposure']
    if not isinstance(ae, dict) or set(ae) - set(exposure_defaults): _fail('tool_calibration.auto_exposure', '字段无效')
    ae = {**exposure_defaults, **ae}
    for k in ('target_low', 'target_high', 'settle_frames', 'max_iterations', 'min_exposure', 'max_exposure'):
        if type(ae[k]) is not int: _fail('tool_calibration.auto_exposure.' + k, '须为整数')
    if not (0 <= ae['target_low'] < ae['target_high'] < 250 and 0 <= ae['settle_frames'] <= 100 and 1 <= ae['max_iterations'] <= 100 and 1 <= ae['min_exposure'] < ae['max_exposure'] <= 5000): _fail('tool_calibration.auto_exposure', '范围无效')
    for k in ('roi_fraction', 'max_saturated_fraction'):
        if not 0 < _number(ae[k], 'tool_calibration.auto_exposure.' + k) <= 1: _fail('tool_calibration.auto_exposure.' + k, '须在 0 到 1 之间')
    # Default maximum is 10 units (1 ms); each unit is 100 µs.
    exposure_scan_defaults = dict(method='quality_scan', max_exposure=10, contrast_min=40., sharpness_ratio=.9, saturation_max=.05)
    exposure_scan = tc['exposure_scan']
    if not isinstance(exposure_scan, dict) or set(exposure_scan) - set(exposure_scan_defaults): _fail('tool_calibration.exposure_scan', '字段无效')
    exposure_scan = {**exposure_scan_defaults, **exposure_scan}
    if exposure_scan['method'] not in ('quality_scan', 'brightness'): _fail('tool_calibration.exposure_scan.method', '应为 quality_scan 或 brightness')
    if type(exposure_scan['max_exposure']) is not int or not 1 <= exposure_scan['max_exposure'] <= 5000: _fail('tool_calibration.exposure_scan.max_exposure', '须为 1 到 5000 的整数')
    if not 0 <= _number(exposure_scan['contrast_min'], 'tool_calibration.exposure_scan.contrast_min') <= 255: _fail('tool_calibration.exposure_scan.contrast_min', '须在 0 到 255 之间')
    for key in ('sharpness_ratio', 'saturation_max'):
        if not 0 < _number(exposure_scan[key], 'tool_calibration.exposure_scan.' + key) <= 1: _fail('tool_calibration.exposure_scan.' + key, '须在 0 到 1 之间')
    import re
    for key in ('usb_vendor_id', 'usb_product_id'):
        if not isinstance(tc[key], str) or not re.fullmatch(r'[0-9a-fA-F]{4}', tc[key]):
            _fail('tool_calibration.' + key, '应为四位十六进制 USB ID')
    cameras = {name: dict(value) if isinstance(value, dict) else value for name, value in cameras.items()}
    for name in ("front", "side"):
        camera_config = cameras[name]
        if not isinstance(camera_config, dict):
            _fail(f"tool_calibration.cameras.{name}", "应为映射")
        backend = camera_config.get("backend")
        if backend == "uvc":
            _mapping(camera_config, f"tool_calibration.cameras.{name}", ["backend", "device"] + (["camera_controls"] if "camera_controls" in camera_config else []))
            device = camera_config['device']
            if not ((type(device) is int and device >= 0) or (isinstance(device, str) and device.strip())):
                _fail(f"tool_calibration.cameras.{name}.device", "设备须为非负整数或非空路径")
            camera_config['camera_controls'] = {**common_controls, **validate_controls(camera_config.get('camera_controls', {}), f'tool_calibration.cameras.{name}.camera_controls')}
            camera_config['usb_vendor_id'] = tc['usb_vendor_id'].lower()
            camera_config['usb_product_id'] = tc['usb_product_id'].lower()
            camera_config['settings_path'] = str(Path(directory) / 'tooltip_cameras.yaml')
        elif backend == "daheng":
            _mapping(camera_config, f"tool_calibration.cameras.{name}", ["backend", "sn"])
            _string(camera_config["sn"], f"tool_calibration.cameras.{name}.sn")
        elif backend == "mock":
            _mapping(camera_config, f"tool_calibration.cameras.{name}", ["backend"])
        else:
            _fail(f"tool_calibration.cameras.{name}.backend", "应为 mock、uvc 或 daheng")
        det = _mapping(detection[name], f"tool_calibration.detection.{name}", ["direction", "threshold", "min_area"] + [key for key in ("probe_offset_px", "max_edge_width_px", "method", "x_range", "min_height", "max_axis_rms_px", "max_distortion_axis_rms_px", "min_tip_contrast", "max_sequence_std_px") if key in detection[name]])
        if det.get('method', 'trident') not in ('trident', 'silhouette'):
            _fail(f'tool_calibration.detection.{name}.method', '应为 trident 或 silhouette')
        for key in ('max_axis_rms_px', 'max_distortion_axis_rms_px', 'min_tip_contrast', 'max_sequence_std_px'):
            if key in det: _number(det[key], f'tool_calibration.detection.{name}.{key}', True)
        if 'min_height' in det and (type(det['min_height']) is not int or det['min_height'] <= 0):
            _fail(f'tool_calibration.detection.{name}.min_height', '应为正整数')
        if 'x_range' in det and det['x_range'] is not None:
            x_range = _vector(det['x_range'], f'tool_calibration.detection.{name}.x_range', 2)
            if x_range[0] >= x_range[1]: _fail(f'tool_calibration.detection.{name}.x_range', '范围起点必须小于终点')
        if "probe_offset_px" in det and (type(det["probe_offset_px"]) is not int or det["probe_offset_px"] <= 0):
            _fail(f"tool_calibration.detection.{name}.probe_offset_px", f"应为正整数，实际为 {det['probe_offset_px']!r}；请检查配置")
        if "max_edge_width_px" in det:
            _number(det["max_edge_width_px"], f"tool_calibration.detection.{name}.max_edge_width_px", True)
        direction = _vector(det["direction"], f"tool_calibration.detection.{name}.direction", 2)
        if not any(direction):
            _fail(f"tool_calibration.detection.{name}.direction", "方向不能为零")
        threshold = _number(det["threshold"], f"tool_calibration.detection.{name}.threshold")
        if not 0 <= threshold <= 255 or type(det["min_area"]) is not int or det["min_area"] < 1:
            _fail(f"tool_calibration.detection.{name}", "阈值应在 0 到 255 之间，面积应为正整数")
    if type(tc["max_iterations"]) is not int or tc["max_iterations"] < 1:
        _fail("tool_calibration.max_iterations", "应为正整数")
    max_condition = _number(tc["max_condition"], "tool_calibration.max_condition", True)
    if max_condition < 1:
        _fail("tool_calibration.max_condition", "应大于等于 1")
    if tc["unreliable_action"] not in ("fail", "click"):
        _fail("tool_calibration.unreliable_action", "应为 fail 或 click")
    click_timeout = _number(tc["click_timeout_s"], "tool_calibration.click_timeout_s", True)
    tool_calibration = ToolCalibrationConfig(
        directory, cameras, detection,
        _number(tc["position_tolerance_mm"], "tool_calibration.position_tolerance_mm", True),
        _number(tc["tip_tolerance_mm"], "tool_calibration.tip_tolerance_mm", True),
        _number(tc["consistency_tolerance_mm"], "tool_calibration.consistency_tolerance_mm", True),
        tc["max_iterations"],
        _number(tc["max_residual_mm"], "tool_calibration.max_residual_mm", True),
        max_condition,
        _number(tc["z_disagreement_mm"], "tool_calibration.z_disagreement_mm", True),
        tc["enabled"], tc["quality_hard_fail"], tc["unreliable_action"], click_timeout,
        _number(tc["probe_step_mm"], "tool_calibration.probe_step_mm", True),
        _number(tc["max_step_mm"], "tool_calibration.max_step_mm", True),
        _number(tc["max_total_mm"], "tool_calibration.max_total_mm", True),
        common_controls, ae, exposure_scan,
        _number(tc["frame_std_max_px"], "tool_calibration.frame_std_max_px", True),
    )
    ensure_calibration_files(directory)
    tools_path = Path(directory) / "tools.yaml"
    try:
        if not require_tools and not tools_path.exists():
            stored = {key: {"offset_xyz": [0.0, 0.0, 0.0]} for key in ("tool1", "tool2")}
        else:
            stored = load_tools(tools_path)["tools"]
    except GuardError as exc:
        raise GuardError(
            "CONFIG-TOOLS", f"无法加载工具补偿 {tools_path}：{exc}", "启动预检",
            "请用 python -m nest_clear.tools.tooltip_baseline 建立工具基准及补偿。",
        ) from exc
    offsets = tuple(tuple(stored[key]["offset_xyz"]) for key in ("tool1", "tool2"))
    if not isinstance(root["ui"], dict):
        _fail("ui", "应为映射")
    ui_data = _mapping({"enabled": True, "fullscreen": False, "refresh_ms": 300, "python": "/usr/bin/python3", **root["ui"]}, "ui", ["enabled", "fullscreen", "refresh_ms", "python"])
    _string(ui_data["python"], "ui.python")
    if type(ui_data["enabled"]) is not bool or type(ui_data["fullscreen"]) is not bool or type(ui_data["refresh_ms"]) is not int or ui_data["refresh_ms"] < 50:
        _fail("ui", "开关须为布尔值，刷新间隔须为不小于 50 的整数")
    if not isinstance(root["ui_preview"], dict):
        _fail("ui_preview", "应为映射")
    preview_data = _mapping({"interval_s": .15, "width": 960, "jpeg_quality": 75, "analysis_interval_s": 1.0, "exposure_sweep": {}, "white_balance": {}, **root["ui_preview"]}, "ui_preview", ["interval_s", "width", "jpeg_quality", "analysis_interval_s", "exposure_sweep", "white_balance"])
    preview_data["interval_s"] = _number(preview_data["interval_s"], "ui_preview.interval_s", True)
    preview_data["analysis_interval_s"] = _number(preview_data["analysis_interval_s"], "ui_preview.analysis_interval_s", True)
    if type(preview_data["width"]) is not int or preview_data["width"] < 100 or type(preview_data["jpeg_quality"]) is not int or not 1 <= preview_data["jpeg_quality"] <= 100:
        _fail("ui_preview", "width 须不小于 100，jpeg_quality 须在 1 至 100 之间")
    sweep = preview_data['exposure_sweep']
    if not isinstance(sweep, dict): _fail('ui_preview.exposure_sweep', '须为映射')
    sweep = _mapping({'min_us': 500, 'max_us': 30000, 'steps': 10, 'settle_frames': 1, **sweep},
                     'ui_preview.exposure_sweep', ['min_us', 'max_us', 'steps', 'settle_frames'])
    sweep['min_us'] = _number(sweep['min_us'], 'ui_preview.exposure_sweep.min_us', True)
    sweep['max_us'] = _number(sweep['max_us'], 'ui_preview.exposure_sweep.max_us', True)
    if sweep['max_us'] <= sweep['min_us'] or type(sweep['steps']) is not int or not 2 <= sweep['steps'] <= 100 or type(sweep['settle_frames']) is not int or not 0 <= sweep['settle_frames'] <= 20:
        _fail('ui_preview.exposure_sweep', '曝光范围须递增，steps 须为 2–100，settle_frames 须为 0–20')
    preview_data['exposure_sweep'] = sweep
    from nest_clear.preview.white_balance import DEFAULTS
    wb = preview_data['white_balance']
    if not isinstance(wb, dict): _fail('ui_preview.white_balance', '须为映射')
    wb = _mapping({**DEFAULTS, **wb}, 'ui_preview.white_balance', list(DEFAULTS))
    for key in wb:
        wb[key] = _number(wb[key], 'ui_preview.white_balance.' + key, True)
    if not (wb['roi_fraction'] <= 1 and wb['max_saturated_fraction'] <= 1 and wb['min_green'] < wb['max_green'] <= 255 and wb['max_p99'] <= 255):
        _fail('ui_preview.white_balance', '阈值范围无效')
    preview_data['white_balance'] = wb
    settle_defaults = {"sample_interval_s": .05, "settle_samples": 4, "jitter_tol_mm": .02, "settle_timeout_s": 1.0}
    if not isinstance(root["arrival_settle"], dict):
        _fail("arrival_settle", "应为映射")
    settle = _mapping({**settle_defaults, **root["arrival_settle"]}, "arrival_settle", list(settle_defaults))
    for key in ("sample_interval_s", "jitter_tol_mm", "settle_timeout_s"):
        settle[key] = _number(settle[key], "arrival_settle." + key, True)
    if type(settle["settle_samples"]) is not int or not 2 <= settle["settle_samples"] <= 40:
        _fail("arrival_settle.settle_samples", "应为 2 至 40 的整数")
    return Config(
        plc,
        scan,
        _number(root["arrival_tolerance_mm"], "arrival_tolerance_mm", True),
        depth,
        offsets,
        travel,
        _string(camera["sn"], "camera.sn"),
        _string(record["directory"], "record.directory"),
        supervisor,
        vision,
        tool_calibration,
        capture_timeout, exposure, gain, balance,
        record["image_enabled"],
        record["image_format"],
        record["image_quality"],
        record["image_retention_days"],
        record["image_queue_size"],
        tuple(image_size),
        _string(record["machine_id"], "record.machine_id"),
        camcal,
        UIConfig(**ui_data),
        preview_data, offset, accuracy, str(exposure_path), exposure_source, str(wb_path), wb_source, settle, test_mode, depth_calibration,
    )
