"""加载并严格校验业务配置，为通信和后续流程提供不可变配置。"""

from dataclasses import dataclass, field, fields, replace
import ipaddress
import math
from pathlib import Path
from typing import Any, NoReturn

import yaml

from .guard import GuardError
from .vision.hair.config import HairDetectConfig
from .tooltip.store import load_tools


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
    test_mode: dict[str, Any] = field(default_factory=lambda: {"non_pickable": [f"T{i:02d}" for i in range(3, 31, 3)]})


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
        data = {"test_mode": {}, "pick_accuracy": {}, "tool_offset": {}, "camera_calibration": {}, "ui": {}, "ui_preview": {}, **data}
    root = _mapping(
        data,
        "根",
        [
            "test_mode",
            "plc",
            "scan",
            "arrival_tolerance_mm",
            "pick_depth_z",
            "tool_calibration",
            "camera_calibration",
            "tool_offset",
            "pick_accuracy",
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
    offset = _mapping({**offset_defaults, **root["tool_offset"]}, "tool_offset", list(offset_defaults))
    for key in ("center_tol_mm", "outlier_mm", "se_tol_mm", "edge_warn_mm", "card_thickness_mm", "ring_cross_tol_mm", "settle_window_s", "jitter_tol_mm"):
        offset[key] = _number(offset[key], "tool_offset." + key, True)
    for key in ("min_points", "max_points", "settle_samples"):
        if type(offset[key]) is not int or offset[key] < 2:
            _fail("tool_offset." + key, "须为不小于 2 的整数")
    if offset["max_points"] < offset["min_points"]:
        _fail("tool_offset.max_points", "不能小于 min_points")
    for key in ("point_order", "edge_targets"):
        if not isinstance(offset[key], list) or not offset[key] or any(not isinstance(v, str) or len(v) != 3 or v[0] != "T" or not v[1:].isdigit() or int(v[1:]) > 31 for v in offset[key]):
            _fail("tool_offset." + key, "须为 T00–T31 列表")
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
    t = _mapping(root["travel"], "travel", ["x", "y", "z"])
    travel = tuple(_vector(t[axis], f"travel.{axis}", 2) for axis in "xyz")
    for axis, bounds in zip("xyz", travel):
        if bounds[0] >= bounds[1]:
            _fail(f"travel.{axis}", "最小值必须小于最大值")
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
        if key == 'max_points':
            if type(accuracy[key]) is not int or accuracy[key] < 1: _fail('pick_accuracy.max_points', '须为正整数')
        else:
            accuracy[key] = _number(accuracy[key], 'pick_accuracy.' + key, key != 'calib_tool_z_mm')
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
        tc_data = {"enabled": True, "quality_hard_fail": True, "unreliable_action": "fail", "click_timeout_s": 120.0, "probe_step_mm": .2, "max_step_mm": .5, "max_total_mm": 2., **tc_data}
    tc = _mapping(tc_data, "tool_calibration", [
        "directory", "cameras", "detection", "position_tolerance_mm", "tip_tolerance_mm",
        "consistency_tolerance_mm", "max_iterations", "max_residual_mm", "max_condition",
        "z_disagreement_mm", "enabled", "quality_hard_fail", "unreliable_action", "click_timeout_s", "probe_step_mm", "max_step_mm", "max_total_mm",
    ])
    if type(tc["enabled"]) is not bool or type(tc["quality_hard_fail"]) is not bool:
        _fail("tool_calibration.enabled", "应为布尔值")
    directory = _string(tc["directory"], "tool_calibration.directory")
    cameras = _mapping(tc["cameras"], "tool_calibration.cameras", ["front", "side"])
    detection = _mapping(tc["detection"], "tool_calibration.detection", ["front", "side"])
    for name in ("front", "side"):
        camera_config = cameras[name]
        if not isinstance(camera_config, dict):
            _fail(f"tool_calibration.cameras.{name}", "应为映射")
        backend = camera_config.get("backend")
        if backend == "uvc":
            _mapping(camera_config, f"tool_calibration.cameras.{name}", ["backend", "device"])
            if type(camera_config["device"]) is not int or camera_config["device"] < 0:
                _fail(f"tool_calibration.cameras.{name}.device", "设备索引应为非负整数")
        elif backend == "daheng":
            _mapping(camera_config, f"tool_calibration.cameras.{name}", ["backend", "sn"])
            _string(camera_config["sn"], f"tool_calibration.cameras.{name}.sn")
        elif backend == "mock":
            _mapping(camera_config, f"tool_calibration.cameras.{name}", ["backend"])
        else:
            _fail(f"tool_calibration.cameras.{name}.backend", "应为 mock、uvc 或 daheng")
        det = _mapping(detection[name], f"tool_calibration.detection.{name}", ["direction", "threshold", "min_area"] + [key for key in ("probe_offset_px", "max_edge_width_px") if key in detection[name]])
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
        preview_data, offset, accuracy, str(exposure_path), exposure_source, str(wb_path), wb_source, test_mode,
    )
