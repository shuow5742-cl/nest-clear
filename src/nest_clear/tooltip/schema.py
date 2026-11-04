"""读写带版本号的工具针尖基准 YAML。"""

from pathlib import Path
import yaml
from nest_clear.guard import GuardError

from .core import ToolTipBaseline, ViewCalib


class ToolCalSchemaError(GuardError):
    """基准文件格式或读写错误，兼容原公开异常名称。"""

    def __init__(self, message: str) -> None:
        """记录基准文件错误和现场排查建议。"""
        super().__init__("TOOLTIP-SCHEMA", f"基准文件错误：{message}", "tooltip.schema", "检查基准 YAML 路径、字段、数值和文件权限")


class _Loader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _mapping(loader: _Loader, node: yaml.Node) -> dict:
    """处理针尖标定数据，供对刀流程调用。
       Construct a mapping while validating its keys."""
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str):
            raise ToolCalSchemaError("mapping keys must be strings")
        if key in result:
            raise ToolCalSchemaError(f"duplicate key: {key}")
        result[key] = loader.construct_object(value_node)
    return result


_Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _fields(value: object, required: set[str], optional: set[str], context: str) -> None:
    """处理针尖标定数据，供对刀流程调用。
       Validate required and allowed mapping fields."""
    if not isinstance(value, dict):
        raise ToolCalSchemaError(f"{context} must be a mapping")
    missing = required - value.keys()
    unknown = value.keys() - required - optional
    if missing or unknown:
        raise ToolCalSchemaError(
            f"{context}: missing fields {sorted(missing)}; unknown fields {sorted(unknown)}"
        )


def load_baseline(path: str | Path) -> ToolTipBaseline:
    """处理针尖标定数据，供对刀流程调用。
       Read a version 1 baseline; invalid files raise ToolCalSchemaError."""
    try:
        with Path(path).open(encoding="utf-8") as f:
            raw = yaml.load(f, Loader=_Loader)
        _fields(
            raw,
            {"version", "tool_name", "p0", "offset_xyz", "views"},
            {"timestamp", "note"},
            "baseline",
        )
        if type(raw["version"]) is not int or raw["version"] != 1:
            raise ToolCalSchemaError("version must be 1")
        if not isinstance(raw["views"], list):
            raise ToolCalSchemaError("views must be a list")
        views = []
        for i, v in enumerate(raw["views"]):
            _fields(v, {"name", "target_uv", "jacobian"}, set(), f"views[{i}]")
            try:
                views.append(ViewCalib(**v))
            except (GuardError, TypeError) as exc:
                raise ToolCalSchemaError(f"views[{i}]: {exc}") from exc
        return ToolTipBaseline(
            raw["tool_name"],
            raw["p0"],
            views,
            raw["offset_xyz"],
            raw.get("timestamp", ""),
            raw.get("note", ""),
        )
    except (yaml.YAMLError, GuardError, ValueError, TypeError, OSError) as exc:
        raise ToolCalSchemaError(f"{path}: {exc}") from exc


def save_baseline(baseline: ToolTipBaseline, path: str | Path) -> None:
    """处理针尖标定数据，供对刀流程调用。
       Write a baseline as version 1 YAML; filesystem errors raise ToolCalSchemaError."""
    raw = {
        "version": 1,
        "tool_name": baseline.tool_name,
        "p0": list(baseline.p0),
        "offset_xyz": list(baseline.offset_xyz),
        "timestamp": baseline.timestamp,
        "note": baseline.note,
        "views": [
            {
                "name": v.name,
                "target_uv": list(v.target_uv),
                "jacobian": [list(row) for row in v.jacobian],
            }
            for v in baseline.views
        ],
    }
    try:
        with Path(path).open("w", encoding="utf-8") as f:
            yaml.safe_dump(raw, f, allow_unicode=True, sort_keys=False)
    except (OSError, yaml.YAMLError) as exc:
        raise ToolCalSchemaError(f"写入基准失败：路径 {path}，原因 {exc}") from exc
