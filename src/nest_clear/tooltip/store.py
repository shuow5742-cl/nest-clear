"""工具基准和当前补偿的持久化接口。"""

from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import yaml

from . import schema
from .core import ToolTipBaseline
from nest_clear.guard import GuardError


def _error(message: str) -> GuardError:
    """将存储失败转换为可排查的守卫错误。"""
    return GuardError("TOOLTIP-STORE", message, "tooltip.store", "检查文件内容、目录权限及磁盘空间。")


def _read(path: Path) -> object:
    """使用严格 YAML 加载器读取数据。"""
    try:
        with path.open(encoding="utf-8") as stream:
            return yaml.load(stream, Loader=schema._Loader)
    except (OSError, yaml.YAMLError) as exc:
        raise _error(f"读取 {path} 失败：{exc}") from exc


def load_baseline(path: str | Path) -> ToolTipBaseline:
    """读取并严格验证已标定基准。"""
    return schema.load_baseline(path)


def load_tools(path: str | Path) -> dict:
    """读取两把工具的当前补偿，拒绝未知字段和非法数值。"""
    path = Path(path)
    raw = _read(path)
    if (
        not isinstance(raw, dict)
        or set(raw) != {"version", "tools"}
        or type(raw["version"]) is not int
        or raw["version"] != 1
    ):
        raise _error(f"{path}：版本或字段无效")
    tools = raw["tools"]
    if not isinstance(tools, dict) or set(tools) != {"tool1", "tool2"}:
        raise _error(f"{path}：必须包含 tool1 和 tool2")
    for name, item in tools.items():
        if not isinstance(item, dict) or set(item) != {"offset_xyz", "updated_at", "source"}:
            raise _error(f"{path}：{name} 字段无效")
        vector = item["offset_xyz"]
        if (
            not isinstance(vector, list)
            or len(vector) != 3
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in vector)
        ):
            raise _error(f"{path}：{name} offset_xyz 必须是三个有限数字")
        if any(not isinstance(item[key], str) or not item[key].strip() for key in ("updated_at", "source")):
            raise _error(f"{path}：{name} 更新时间和来源不能为空")
    return raw


def _write(
    path: Path, data: dict, tool: int, action: str, old: object, source: str, task_id: int | None
) -> None:
    """同目录原子替换、保留旧版，并追加审计历史。"""
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        now = datetime.now().astimezone()
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix="." + path.name, delete=False
        ) as stream:
            temporary = Path(stream.name)
            yaml.safe_dump(data, stream, allow_unicode=True, sort_keys=False)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            backup = path.with_name(path.name + ".bak." + now.strftime("%Y%m%d%H%M%S%f"))
            shutil.copy2(path, backup)
        os.replace(temporary, path)
        row = {
            "timestamp": now.isoformat(),
            "tool": tool,
            "action": action,
            "old": old,
            "new": data,
            "source": source,
        }
        if task_id is not None:
            row["Task_ID"] = task_id
        with (path.parent / "tools_history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    except (OSError, yaml.YAMLError, TypeError, ValueError) as exc:
        raise _error(f"保存 {path} 失败：{exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_baseline(
    baseline: ToolTipBaseline, path: str | Path, *, tool: int, source: str, task_id: int | None = None
) -> None:
    """保存基准并保留旧文件及审计记录。"""
    if tool not in (1, 2) or baseline.tool_name != f"tool{tool}" or not source:
        raise _error("工具号、基准名称或来源无效")
    path = Path(path)
    old = _read(path) if path.exists() else None
    data = {
        "version": 1,
        "tool_name": baseline.tool_name,
        "p0": list(baseline.p0),
        "offset_xyz": list(baseline.offset_xyz),
        "timestamp": baseline.timestamp,
        "note": baseline.note,
        "views": [
            {
                "name": view.name,
                "target_uv": list(view.target_uv),
                "jacobian": [list(row) for row in view.jacobian],
            }
            for view in baseline.views
        ],
    }
    _write(path, data, tool, "baseline", old, source, task_id)


def initialize_tools(path: str | Path) -> None:
    """首次建立补偿文件；已有文件绝不覆盖，供基准工具使用。"""
    path = Path(path)
    data = {
        "version": 1,
        "tools": {
            name: {"offset_xyz": [0.0, 0.0, 0.0], "updated_at": "占位，未标定", "source": "占位，未标定"}
            for name in ("tool1", "tool2")
        },
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(data, stream, allow_unicode=True, sort_keys=False)
    except OSError as exc:
        raise _error(f"初始化 {path} 失败：{exc}") from exc


def save_tools(
    path: str | Path, tool: int, offset_xyz: object, *, source: str, task_id: int | None = None
) -> None:
    """更新一把工具的补偿，另一把保持不变。"""
    if tool not in (1, 2) or not isinstance(source, str) or not source.strip():
        raise _error("工具号或来源无效")
    path = Path(path)
    data = load_tools(path)
    old = data["tools"][f"tool{tool}"].copy()
    if (
        not isinstance(offset_xyz, (list, tuple))
        or len(offset_xyz) != 3
        or any(type(value) not in (int, float) or not math.isfinite(value) for value in offset_xyz)
    ):
        raise _error("offset_xyz 必须是三个有限数字")
    data["tools"][f"tool{tool}"] = {
        "offset_xyz": list(offset_xyz),
        "updated_at": datetime.now().astimezone().isoformat(),
        "source": source,
    }
    _write(path, data, tool, "offset", old, source, task_id)
