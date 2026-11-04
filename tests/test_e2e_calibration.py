"""真实 TCP 握手下的双相机对刀迭代。"""

from dataclasses import replace
import json
from pathlib import Path
import threading
import time

import pytest

from nest_clear.config import Scan, load_config
from nest_clear.main import Application
from nest_clear.tooltip import ToolTipBaseline, ViewCalib
from nest_clear.tooltip.store import initialize_tools, load_tools, save_baseline, save_tools
from nest_clear.vision.detector import PixelTarget
from sim.fake_plc import FakePLC

ROOT = Path(__file__).resolve().parents[1]
P0 = (10.0, 20.0, 30.0)
VIEWS = (
    ViewCalib("front", (320, 242), ((20, 0, 0), (0, 0, 20))),
    ViewCalib("side", (320, 242), ((0, 20, 0), (0, 0, 20))),
)


@pytest.fixture
def rig(tmp_path: Path):
    """建立可用基准，并让模拟相机根据 PLC 实时坐标生成剪影。"""
    for tool in (1, 2):
        save_baseline(ToolTipBaseline(f"tool{tool}", P0, VIEWS, (1.0, 2.0, 3.0) if tool == 1 else (0.0, 0.0, 0.0)), tmp_path / f"tool{tool}_baseline.yaml", tool=tool, source="test")
    initialize_tools(tmp_path / "tools.yaml")
    save_tools(tmp_path / "tools.yaml", 1, (1.0, 2.0, 3.0), source="test")
    with FakePLC() as plc:
        config = load_config(ROOT / "config/nest_clear.yaml")
        tc = replace(config.tool_calibration, directory=str(tmp_path), cameras={"front": {"backend": "mock"}, "side": {"backend": "mock"}})
        config = replace(config, tools=((1.0, 2.0, 3.0), (0.0, 0.0, 0.0)), plc=replace(config.plc, ip=plc.host, port=plc.port),
                         vision=replace(config.vision, camera_backend="mock"),
                         record_directory=str(tmp_path / "runs"), tool_calibration=tc)
        config = replace(config, scan=Scan(((1.0, 0.0), (0.0, 0.0), (1.0, 0.0)), 1.0, 1.0, 0.0))
        class Detector:
            def __init__(self):
                self.calls = 0
            def detect(self, image):
                self.calls += 1
                return [PixelTarget(640, 480, 0.9), PixelTarget(660, 480, 0.8)] if self.calls == 1 else []
        app = Application(config, ROOT / "config/nest_clear.yaml", detector=Detector())
        for view in VIEWS:
            camera = app.calib_camera.cameras[view.name]
            camera.position = lambda: app.io.read_real32_block(plc.contract["actual_x"].address, 3)
            camera.config.update(target_uv=[320, 240], p0=P0, jacobian=view.jacobian, tip_offset=[0, 0, 0])
        stop = threading.Event()
        errors = []

        def loop():
            """高频运行主循环并保留异常。"""
            try:
                while not stop.is_set():
                    app.step()
                    time.sleep(0.002)
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=loop)
        thread.start()
        yield plc, app, tmp_path
        stop.set()
        thread.join(3)
        app.close()
        assert not errors


def offset(app, xyz):
    """设置模拟针尖相对基准的偏差。"""
    for camera in app.calib_camera.cameras.values():
        camera.config["tip_offset"] = [-x for x in xyz]


def test_direct_and_one_offset(rig):
    """零偏差直接完成；新会话偏移坐标等于 P0 加真实偏差。"""
    plc, app, _ = rig
    assert [r["func_code"] for r in plc.start_tool_calibration(1, P0)] == [111]
    # 完成后仍在对刀模式，模拟 PLC 重启进程由独立测试验证偏移。
    task = plc.request(1)
    assert plc.wait_reply(task)["func_code"] == 999


@pytest.mark.parametrize("tool", [1, 2])
def test_offset_and_wrong_tool(rig, tool):
    """两工具首次偏移、地址及工具号模式检查对称。"""
    plc, app, _ = rig
    offset(app, (0.5, 0.3, 0.2))
    plc.set_actual_xyz(P0)
    task = plc.request(100 if tool == 1 else 101)
    first = plc.wait_reply(task, 5)
    assert first["func_code"] == (100 if tool == 1 else 101)
    xyz = first[f"calibration{tool}_xyz"]
    assert xyz == pytest.approx([10.5, 20.3, 30.2], abs=0.03)
    assert plc.contract[f"calibration{tool}_x"].address in [a for a, _ in plc.writes]
    task = plc.request(120 if tool == 1 else 110)
    assert plc.wait_reply(task, 5)["func_code"] == 999
    task = plc.request(1)
    assert plc.wait_reply(task, 5)["func_code"] == 999
    plc.set_actual_xyz(xyz)
    task = plc.request(110 if tool == 1 else 120)
    assert plc.wait_reply(task, 5)["func_code"] == (111 if tool == 1 else 121)


def test_multiple_offsets(rig):
    """小于位置公差的定位误差迫使多轮收敛。"""
    plc, app, _ = rig
    offset(app, (0.5, 0.3, 0.2))
    records = plc.start_tool_calibration(1, P0, timeout_s=5, position_error=(0.04, 0, 0))
    assert records[0]["func_code"] == 100
    assert all(r["func_code"] == 110 for r in records[1:-1])
    assert len(records) >= 3 and records[-1]["func_code"] == 111


@pytest.mark.parametrize("case", ["initial", "arrival", "iterations", "detect", "placeholder", "disabled"])
def test_failures_recover_to_idle(rig, case):
    """每种业务失败返回 131，随后可正常开始挑毛。"""
    plc, app, path = rig
    offset(app, (0.5, 0, 0))
    actual = P0
    error = (0, 0, 0)
    if case == "initial":
        actual = (11, 20, 30)
    elif case == "arrival":
        error = (0.2, 0, 0)
    elif case == "iterations":
        object.__setattr__(app.config.tool_calibration, "max_iterations", 1)
        error = (0.04, 0, 0)
    elif case == "detect":
        app.calib_camera.cameras["front"].config["undetectable"] = True
    elif case == "placeholder":
        (path / "tool1_baseline.yaml").write_text('status: "占位，未标定"\n', encoding="utf-8")
    elif case == "disabled":
        object.__setattr__(app.config.tool_calibration, "enabled", False)
    records = plc.start_tool_calibration(1, actual, timeout_s=5, position_error=error)
    assert records[-1]["func_code"] == 131
    task = plc.request(1)
    assert plc.wait_reply(task, 5)["func_code"] == 1
    events = [json.loads(line) for line in next((path / "runs").glob("*/events.jsonl")).read_text().splitlines()]
    assert any(event["event"] == "tooltip_failed" for event in events)


@pytest.mark.parametrize("tool", [1, 2])
def test_finish_saves_compensation(rig, tool):
    plc, app, path = rig
    offset(app, (0.5, 0.3, 0.2))
    before = load_tools(path / "tools.yaml")
    records = plc.start_tool_calibration(tool, P0, timeout_s=5, finish=True)
    assert records[-1]["func_code"] == 130
    final = records[0][f"calibration{tool}_xyz"]
    base = (1, 2, 3) if tool == 1 else (0, 0, 0)
    expected = [b + x - p for b, x, p in zip(base, final, P0)]
    assert load_tools(path / "tools.yaml")["tools"][f"tool{tool}"]["offset_xyz"] == pytest.approx(expected)
    assert app.config.tools[tool - 1] == pytest.approx(expected)
    backups = list(path.glob("tools.yaml.bak.*"))
    assert any(load_tools(p) == before for p in backups)
    history = [json.loads(line) for line in (path / "tools_history.jsonl").read_text().splitlines()]
    assert {"Task_ID", "tool", "P0", "P_final", "delta", "old_compensation", "new_compensation"} <= history[-1].keys()
    records = plc.drive_tray(timeout_s=5)
    assert [r["func_code"] for r in records] == [1, 3, 3, 1, 9]
    expected1 = expected if tool == 1 else (1, 2, 3)
    expected2 = expected if tool == 2 else (0, 0, 0)
    assert records[1]["pick1_xyz"] == pytest.approx([1 + expected1[0], expected1[1], expected1[2]])
    assert records[2]["pick2_xyz"] == pytest.approx([3 + expected2[0], expected2[1], expected2[2]])
    assert {40960, 40966} <= {address for address, _ in plc.writes}


def test_finish_inconsistent_and_unfinished(rig):
    plc, app, path = rig
    offset(app, (0.5, 0.3, 0.2))
    before = (path / "tools.yaml").read_bytes()
    old = app.config.tools
    records = plc.start_tool_calibration(1, P0, timeout_s=5, finish=True, final_error=(0.1, 0, 0))
    assert records[-1]["func_code"] == 131
    assert (path / "tools.yaml").read_bytes() == before and app.config.tools == old
    plc.set_actual_xyz(P0)
    assert plc.wait_reply(plc.request(100), 5)["func_code"] == 100
    assert plc.wait_reply(plc.request(130), 5)["func_code"] == 131


def test_finish_without_offset_and_realtime_check(rig):
    plc, app, _ = rig
    records = plc.start_tool_calibration(1, P0, timeout_s=5, finish=True, final_error=(0.02, 0, 0))
    assert records[-1]["func_code"] == 130
    plc.set_actual_xyz(P0)
    assert plc.wait_reply(plc.request(100), 5)["func_code"] == 111
    plc.set_calibration_final(1, P0)
    plc.set_actual_xyz((11, 20, 30))
    assert plc.wait_reply(plc.request(130), 5)["func_code"] == 131


def test_finish_save_failure(rig, monkeypatch):
    plc, app, path = rig
    old = app.config.tools
    before = (path / "tools.yaml").read_bytes()
    def broken(*args, **kwargs):
        raise OSError("injected save failure")
    monkeypatch.setattr("nest_clear.flow.calibration.save_tools", broken)
    records = plc.start_tool_calibration(1, P0, timeout_s=5, finish=True)
    assert records[-1]["func_code"] == 131
    assert app.config.tools == old and (path / "tools.yaml").read_bytes() == before


def test_calibration_record_and_report(rig):
    """成功和识别失败的会话可由图片、摘要及日报完整追溯。"""
    from datetime import date
    from nest_clear.record.report import report

    plc, app, path = rig
    offset(app, (0.5, 0.3, 0.2))
    success = plc.start_tool_calibration(1, P0, timeout_s=5, finish=True)
    assert success[-1]["func_code"] == 130
    plc.set_actual_xyz(P0)
    app.calib_camera.cameras["front"].config["undetectable"] = True
    failure = plc.start_tool_calibration(1, P0, timeout_s=5)
    assert failure[-1]["func_code"] == 131
    assert plc.wait_reply(plc.request(130), 5)["func_code"] == 999
    app.archive.close()
    directory = next((path / "runs").glob("????-??-??"))
    rows = [json.loads(line) for line in (directory / "tooltip_summary.jsonl").read_text().splitlines()]
    good, bad = rows
    assert {"start_time", "end_time", "tool", "iterations", "result", "reason", "delta", "old_compensation", "new_compensation", "task_ids", "images"} <= good.keys()
    assert good["result"] == "成功" and good["iterations"] == 1
    assert good["delta"] == pytest.approx([0.5, 0.3, 0.2], abs=0.03)
    assert good["old_compensation"] == pytest.approx([1, 2, 3])
    assert good["new_compensation"] == pytest.approx([1.5, 2.3, 3.2], abs=0.03)
    assert len(good["task_ids"]) == 3 and len(good["images"]) == 2
    assert bad["result"] == "失败" and "无法识别针尖" in bad["reason"]
    for capture in good["images"]:
        for image_path in (*capture["raw"].values(), *capture["overlay"].values()):
            assert image_path and (path / "runs" / image_path).is_file()
            assert "calib_1_" in image_path
    assert all((path / "runs" / image_path).is_file() for image_path in bad["images"][0]["raw"].values())
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    assert any(row["event"] == "tooltip_detect" and row["images"]["overlay"]["front"] for row in events)
    assert any(row["event"] == "tooltip_failed" and row["images"][0]["raw"]["front"] for row in events)
    text = report(path / "runs", date.fromisoformat(directory.name))
    assert "工具标定：" in text and "迭代 1 次，成功" in text
    assert "需要人看的：" in text and "对刀 131" in text and "无法识别针尖" in text
    assert "模式 空闲 → 对刀中(工具1)" in text
    attention = text.split("需要人看的：", 1)[1]
    assert attention.count("Task_ID=5") == 1
    assert "模式不允许：空闲 时收到 130（对刀完成），须先开始对应工具的对刀会话" in attention
    assert "no dark component" not in text
    assert "front" not in text
    assert text.count("    错误码：TOOLTIP-FAILED；发生了什么：正面相机无法识别针尖：图像中未找到暗色针尖区域") == 2
    assert text.count("    排查建议：检查背光与曝光，以及针尖是否在画面内。") == 2
    assert "位置：对刀" not in text
    failed_event = next(row for row in events if row["event"] == "tooltip_failed")
    assert "位置：对刀" in failed_event["reason"]
    print("\n" + text)


def test_calibration_record_images_disabled(rig):
    """关闭图片归档仍可完成对刀并写摘要。"""
    plc, app, path = rig
    app.archive.enabled = False
    records = plc.start_tool_calibration(1, P0, timeout_s=5, finish=True)
    assert records[-1]["func_code"] == 130
    row = json.loads(next((path / "runs").glob("*/tooltip_summary.jsonl")).read_text())
    assert row["result"] == "成功"
    assert all(value is None for value in row["images"][0]["raw"].values())
    assert all(value is None for value in row["images"][0]["overlay"].values())
