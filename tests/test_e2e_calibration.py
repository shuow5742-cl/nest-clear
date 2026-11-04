"""真实 TCP 握手下的双相机对刀迭代。"""

from dataclasses import replace
import json
from pathlib import Path
import threading
import time

import pytest

from nest_clear.config import load_config
from nest_clear.main import Application
from nest_clear.tooltip import ToolTipBaseline, ViewCalib
from nest_clear.tooltip.store import save_baseline
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
        save_baseline(ToolTipBaseline(f"tool{tool}", P0, VIEWS), tmp_path / f"tool{tool}_baseline.yaml", tool=tool, source="test")
    with FakePLC() as plc:
        config = load_config(ROOT / "config/nest_clear.yaml")
        tc = replace(config.tool_calibration, directory=str(tmp_path), cameras={"front": {"backend": "mock"}, "side": {"backend": "mock"}})
        config = replace(config, plc=replace(config.plc, ip=plc.host, port=plc.port),
                         vision=replace(config.vision, camera_backend="mock"),
                         record_directory=str(tmp_path / "runs"), tool_calibration=tc)
        app = Application(config, ROOT / "config/nest_clear.yaml")
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
