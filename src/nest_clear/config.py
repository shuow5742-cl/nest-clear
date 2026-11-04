"""加载并严格校验业务配置，为通信和后续流程提供不可变配置。"""

from dataclasses import dataclass
import ipaddress
import math
from pathlib import Path
from typing import Any, NoReturn

import yaml

from .guard import GuardError
from .tooltip.store import load_tools


@dataclass(frozen=True)
class PLC:
    """保存 PLC 网络参数；本机 IP 仅用于记录，不用于绑定。"""

    ip: str
    local_ip: str
    port: int
    timeout_s: float
    comm_fail_timeout_s: float = 10.0


@dataclass(frozen=True)
class Scan:
    """保存扫描范围；三个角点为 XY，扫描高度由 z 单独指定。"""

    corners: tuple[tuple[float, float], ...]
    step_x: float
    step_y: float
    z: float


@dataclass(frozen=True)
class SupervisorConfig:
    """保存父程序轮询、看门狗和子进程启动参数。"""

    poll_s: float
    watchdog_s: float
    child_command: tuple[str, ...]
    startup_grace_s: float = 60.0


@dataclass(frozen=True)
class VisionConfig:
    """保存相机模式、标定文件和占位检测器参数。"""

    camera_backend: str
    calibration: str
    pixels: tuple[tuple[float, ...], ...]
    dedup_radius_mm: float
    max_targets: int


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
    image_enabled: bool = True
    image_format: str = "png"
    image_quality: int = 3
    image_retention_days: int = 30
    image_queue_size: int = 32


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


def load_config(path: str | Path, *, require_tools: bool = True) -> Config:
    """读取 YAML 并校验所有字段和行程，供启动预检调用。"""
    try:
        with open(path, encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except (OSError, yaml.YAMLError) as exc:
        raise GuardError(
            "CONFIG-LOAD", f"无法读取配置：{exc}", str(path), "检查文件路径和 YAML 格式。"
        ) from exc
    root = _mapping(
        data,
        "根",
        [
            "plc",
            "scan",
            "arrival_tolerance_mm",
            "pick_depth_z",
            "tool_calibration",
            "travel",
            "camera",
            "record",
            "supervisor",
            "vision",
        ],
    )
    plc_data = root["plc"]
    if isinstance(plc_data, dict):
        plc_data = {"comm_fail_timeout_s": 10.0, **plc_data}
    p = _mapping(plc_data, "plc", ["ip", "local_ip", "port", "timeout_s", "comm_fail_timeout_s"])
    port = p["port"]
    if type(port) is not int or not 1 <= port <= 65535:
        _fail("plc.port", "端口应为 1 到 65535 的整数")
    plc = PLC(
        _ip(p["ip"], "plc.ip"),
        _ip(p["local_ip"], "plc.local_ip"),
        port,
        _number(p["timeout_s"], "plc.timeout_s", True),
        _number(p["comm_fail_timeout_s"], "plc.comm_fail_timeout_s", True),
    )
    t = _mapping(root["travel"], "travel", ["x", "y", "z"])
    travel = tuple(_vector(t[axis], f"travel.{axis}", 2) for axis in "xyz")
    for axis, bounds in zip("xyz", travel):
        if bounds[0] >= bounds[1]:
            _fail(f"travel.{axis}", "最小值必须小于最大值")
    s = _mapping(root["scan"], "scan", ["corners", "step_x", "step_y", "z"])
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
        corners, _number(s["step_x"], "scan.step_x", True), _number(s["step_y"], "scan.step_y", True), z
    )
    camera = _mapping(root["camera"], "camera", ["sn"])
    record_data = root["record"]
    if isinstance(record_data, dict):
        record_data = {
            "image_enabled": True,
            "image_format": "png",
            "image_quality": 3,
            "image_retention_days": 30,
            "image_queue_size": 32,
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
    v = _mapping(
        root["vision"],
        "vision",
        ["camera_backend", "calibration", "pixels", "dedup_radius_mm", "max_targets"],
    )
    if v["camera_backend"] not in ("mock", "daheng"):
        _fail("vision.camera_backend", "应为 mock 或 daheng")
    if not isinstance(v["pixels"], list):
        _fail("vision.pixels", "应为 [u, v, score] 列表")
    pixels = tuple(_vector(item, "vision.pixels", 3) for item in v["pixels"])
    if any(not 0 <= pixel[2] <= 1 for pixel in pixels):
        _fail("vision.pixels", "目标分数必须在 0 到 1 之间")
    radius = _number(v["dedup_radius_mm"], "vision.dedup_radius_mm")
    if radius < 0 or type(v["max_targets"]) is not int or v["max_targets"] < 1:
        _fail("vision", "去重半径应非负，最大目标数应为正整数")
    vision = VisionConfig(
        v["camera_backend"], _string(v["calibration"], "vision.calibration"), pixels, radius, v["max_targets"]
    )
    tc_data = root["tool_calibration"]
    if isinstance(tc_data, dict) and "enabled" not in tc_data:
        tc_data = {**tc_data, "enabled": True}
    tc = _mapping(tc_data, "tool_calibration", [
        "directory", "cameras", "detection", "position_tolerance_mm", "tip_tolerance_mm",
        "consistency_tolerance_mm", "max_iterations", "max_residual_mm", "max_condition",
        "z_disagreement_mm", "enabled",
    ])
    if type(tc["enabled"]) is not bool:
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
        det = _mapping(detection[name], f"tool_calibration.detection.{name}", ["direction", "threshold", "min_area"])
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
    tool_calibration = ToolCalibrationConfig(
        directory, cameras, detection,
        _number(tc["position_tolerance_mm"], "tool_calibration.position_tolerance_mm", True),
        _number(tc["tip_tolerance_mm"], "tool_calibration.tip_tolerance_mm", True),
        _number(tc["consistency_tolerance_mm"], "tool_calibration.consistency_tolerance_mm", True),
        tc["max_iterations"],
        _number(tc["max_residual_mm"], "tool_calibration.max_residual_mm", True),
        max_condition,
        _number(tc["z_disagreement_mm"], "tool_calibration.z_disagreement_mm", True),
        tc["enabled"],
    )
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
        record["image_enabled"],
        record["image_format"],
        record["image_quality"],
        record["image_retention_days"],
        record["image_queue_size"],
    )
