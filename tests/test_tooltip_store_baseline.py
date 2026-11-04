"""基准存储的原子性、审计及严格读取回归。"""

import json
from pathlib import Path
import shutil

import pytest
import yaml

from nest_clear.guard import GuardError
from nest_clear.tooltip import ToolTipBaseline, ViewCalib
from nest_clear.tooltip import store


@pytest.fixture
def calibration(tmp_path: Path) -> Path:
    """复制占位配置，防止测试污染现场文件。"""
    directory = tmp_path / "calibration"
    shutil.copytree("config/calibration", directory)
    return directory


@pytest.fixture
def baseline() -> ToolTipBaseline:
    """提供可序列化的确定性基准供存储测试复用。"""
    view = ViewCalib("front", [10, 20], [[1, 0, 0], [0, 0, 1]])
    return ToolTipBaseline("tool1", [1, 2, 3], [view], [4, 5, 6])


@pytest.mark.parametrize("kind", ["baseline", "tools"])
def test_atomic_replace_failure_preserves_original(calibration, baseline, monkeypatch, kind) -> None:
    """替换前注入文件系统故障，原文件仍应完整且无成功历史。"""
    path = calibration / ("tool1_baseline.yaml" if kind == "baseline" else "tools.yaml")
    before = path.read_bytes()

    def fail_replace(source, destination) -> None:
        """模拟原子替换失败，不改变目标文件。"""
        raise OSError("注入替换失败")

    monkeypatch.setattr(store.os, "replace", fail_replace)
    with pytest.raises(GuardError, match="注入替换失败"):
        if kind == "baseline":
            store.save_baseline(baseline, path, tool=1, source="test")
        else:
            store.save_tools(path, 1, [4, 5, 6], source="test")
    assert path.read_bytes() == before
    assert not (calibration / "tools_history.jsonl").exists()
    assert not list(calibration.glob("." + path.name + "*"))


@pytest.mark.parametrize("kind", ["baseline", "tools"])
def test_backup_contains_exact_old_file(calibration, baseline, kind) -> None:
    """两种保存均须备份原始文件内容而非新值。"""
    path = calibration / ("tool1_baseline.yaml" if kind == "baseline" else "tools.yaml")
    before = path.read_bytes()
    if kind == "baseline":
        store.save_baseline(baseline, path, tool=1, source="test")
        assert store.load_baseline(path) == baseline
    else:
        store.save_tools(path, 1, [4, 5, 6], source="test")
        assert store.load_tools(path)["tools"]["tool1"]["offset_xyz"] == [4, 5, 6]
    backups = list(calibration.glob(path.name + ".bak.*"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == before
    assert path.read_bytes() != before


def test_history_has_required_fields(calibration, baseline) -> None:
    """每条历史均包含时间、工具、动作、新旧值和来源及可选任务号。"""
    store.save_baseline(baseline, calibration / "tool1_baseline.yaml", tool=1, source="test", task_id=12)
    store.save_tools(calibration / "tools.yaml", 1, [4, 5, 6], source="engineer")
    rows = [
        json.loads(line)
        for line in (calibration / "tools_history.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["action"] for row in rows] == ["baseline", "offset"]
    for row in rows:
        assert {"timestamp", "tool", "action", "old", "new", "source"} <= row.keys()
        assert row["timestamp"] and row["tool"] == 1 and row["old"] is not None and row["new"]
    assert rows[0]["Task_ID"] == 12
    assert "Task_ID" not in rows[1]
    assert rows[1]["source"] == "engineer"


@pytest.mark.parametrize("filename", ["tools.yaml", "tool1_baseline.yaml"])
def test_invalid_yaml_rejected(calibration, filename) -> None:
    """不合法 YAML 不能作为配置静默使用。"""
    path = calibration / filename
    path.write_text("version: [\n", encoding="utf-8")
    with pytest.raises(GuardError):
        (store.load_tools if filename == "tools.yaml" else store.load_baseline)(path)


def test_missing_fields_rejected(calibration) -> None:
    """工具记录缺少来源字段时拒绝读取。"""
    path = calibration / "tools.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    del data["tools"]["tool1"]["source"]
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(GuardError):
        store.load_tools(path)


def test_wrong_vector_shape_rejected(calibration) -> None:
    """补偿不是三维向量时拒绝读取。"""
    path = calibration / "tools.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["tools"]["tool1"]["offset_xyz"] = [1, 2]
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(GuardError):
        store.load_tools(path)


def test_baseline_missing_field_rejected(calibration, baseline) -> None:
    """基准缺字段时通过现有 schema 报告守卫错误。"""
    path = calibration / "tool1_baseline.yaml"
    store.save_baseline(baseline, path, tool=1, source="test")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    del data["p0"]
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(GuardError):
        store.load_baseline(path)


def test_baseline_bad_matrix_shape_rejected(calibration, baseline) -> None:
    """基准雅可比形状错误时拒绝读取。"""
    path = calibration / "tool1_baseline.yaml"
    store.save_baseline(baseline, path, tool=1, source="test")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["views"][0]["jacobian"] = [[1, 2], [3, 4]]
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(GuardError):
        store.load_baseline(path)
