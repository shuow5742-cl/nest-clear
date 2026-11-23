"""工具偏移结果持久化。调用方负责决定何时确认保存。"""
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import shutil
import tempfile

import yaml


def _tool_key(tool):
    if type(tool) is not int or tool not in (1, 2):
        raise ValueError("工具号必须为 1 或 2")
    return f"tool{tool}"


def _read(path):
    path = Path(path)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"工具偏移文件格式无效：{path}")
    return data


def load_offset(path, tool):
    """返回指定工具的 XY 偏移；文件或该工具不存在时返回 None。"""
    key = _tool_key(tool)
    result = _read(path).get(key)
    if result is None:
        return None
    if not isinstance(result, dict):
        raise ValueError(f"{key} 结果格式无效")
    xy = result.get("offset_xy_mm")
    if not isinstance(xy, (list, tuple)) or len(xy) != 2 or any(
        type(v) not in (int, float) or not math.isfinite(v) for v in xy
    ):
        raise ValueError(f"{key} offset_xy_mm 必须是两个有限数字")
    return tuple(float(v) for v in xy)


def load_pick_z(path, tool):
    """读取该工具所有人工对准样本的模组 Z，返回最大值与样本列表。

    缺失或损坏的样本不可用于安全坐标；调用方应拒绝下发。
    """
    key = _tool_key(tool)
    result = _read(path).get(key)
    if not isinstance(result, dict):
        return None
    samples = result.get("samples")
    if not isinstance(samples, list) or not samples:
        return None
    values = []
    for sample in samples:
        xyz = sample.get("tool_xyz_mm") if isinstance(sample, dict) else None
        if (not isinstance(xyz, (list, tuple)) or len(xyz) != 3 or
                type(xyz[2]) not in (int, float) or not math.isfinite(xyz[2])):
            return None
        values.append(float(xyz[2]))
    return max(values), values


def save_result(path, tool, result):
    """备份旧文件并原子替换结果，保留另一把工具。返回备份路径或 None。"""
    key = _tool_key(tool)
    if not isinstance(result, dict):
        raise ValueError("工具偏移结果必须为字典")
    xy = result.get("offset_xy_mm")
    if not isinstance(xy, (list, tuple)) or len(xy) != 2 or any(
        type(v) not in (int, float) or not math.isfinite(v) for v in xy
    ):
        raise ValueError("offset_xy_mm 必须是两个有限数字")
    path = Path(path)
    data = _read(path)
    data[key] = result
    return save_mapping(path, data, 'tool_offset')


def save_mapping(path, data, prefix):
    """备份并原子替换 YAML 映射；供需要人工确认的标定结果共用。"""
    path = Path(path)
    # 序列化在备份及替换之前完成，避免无效数据影响现有文件。
    serialized = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if path.exists():
        history = path.parent / "history"
        history.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=datetime.now(timezone.utc).strftime(prefix + "_%Y%m%dT%H%M%S_%fZ_"), suffix=".yaml", dir=history, delete=False) as f:
            backup = Path(f.name)
        try:
            shutil.copy2(path, backup)
        except Exception:
            backup.unlink(missing_ok=True)
            raise
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix="." + prefix + "_", suffix=".tmp", delete=False) as f:
            temp_path = Path(f.name)
            f.write(serialized)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return backup


def save_confirmed(path, tools_path, tool, result, *, task_id=None):
    """确认后同步结果与现有挑取补偿；同步失败恢复原文件。"""
    from nest_clear.tooltip.store import load_tools, save_tools

    key = _tool_key(tool)
    path, tools_path = Path(path), Path(tools_path)
    current = load_tools(tools_path)["tools"][key]["offset_xyz"]
    xy = result.get("offset_xy_mm") if isinstance(result, dict) else None
    if not isinstance(xy, (list, tuple)) or len(xy) != 2 or any(
        type(v) not in (int, float) or not math.isfinite(v) for v in xy
    ):
        raise ValueError("offset_xy_mm 必须是两个有限数字")
    now = datetime.now().astimezone().isoformat()
    result = dict(result, baseline_stale=True)
    # 两个结果文件无法进行单个文件系统原子提交；失败时恢复第一份文件。
    original = path.read_bytes() if path.exists() else None
    backup = save_result(path, tool, result)
    try:
        history = tools_path.parent / "history"
        history.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=datetime.now(timezone.utc).strftime("tools_%Y%m%dT%H%M%S_%fZ_"), suffix=".yaml", dir=history, delete=False) as f:
            tools_backup = Path(f.name)
        try:
            shutil.copy2(tools_path, tools_backup)
        except Exception:
            tools_backup.unlink(missing_ok=True)
            raise
        xyz = (float(xy[0]), float(xy[1]), current[2])
        save_tools(tools_path, tool, xyz, source=f"工具偏移标定 {now}", task_id=task_id)
    except Exception:
        if original is None:
            path.unlink(missing_ok=True)
        else:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".tool_offset_restore_", delete=False) as f:
                restore = Path(f.name)
                try:
                    f.write(original)
                    f.flush()
                    os.fsync(f.fileno())
                except Exception:
                    restore.unlink(missing_ok=True)
                    raise
            os.replace(restore, path)
        raise
    return backup, tools_backup, xyz
