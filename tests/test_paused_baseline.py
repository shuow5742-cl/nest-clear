"""验证 PLC 暂停/停止/手动模式下对刀基准建立仍可执行，且暂停期间不接收新 PLC 请求。"""

from dataclasses import replace
import json
from pathlib import Path
import shutil
import time

import pytest

from nest_clear.config import load_config
from nest_clear.flow.modes import Mode
from nest_clear.main import Application
from nest_clear.paths import main_status_file, supervisor_state_file
from nest_clear.record.events import EventRecorder
from nest_clear.supervisor.gate import SupervisorGate
from nest_clear.tooloffset.commands import send
from nest_clear.tooltip.store import initialize_tools, load_baseline, save_tools
from sim.fake_plc import FakePLC

ROOT = Path(__file__).resolve().parents[1]


def _write_paused(root: Path, paused: bool) -> None:
    path = supervisor_state_file(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"pid": 100, "child_pid": 200, "child_state": "就绪", "paused": paused}),
        encoding="utf-8",
    )


def _read_main_status(root: Path) -> dict:
    return json.loads(main_status_file(root).read_text(encoding="utf-8"))


def _step_until(app: Application, gate: SupervisorGate, condition, timeout: float = 6.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.step(gate)
        if condition():
            return
        time.sleep(0.01)
    pytest.fail("等待条件超时")


@pytest.fixture
def paused_rig(tmp_path: Path):
    cal = tmp_path / "cal"
    shutil.copytree(ROOT / "config/calibration/examples", cal)
    if not (cal / "tools.yaml").exists():
        initialize_tools(cal / "tools.yaml")
    save_tools(cal / "tools.yaml", 1, [1.0, 2.0, 3.0], source="test")
    runs = tmp_path / "runs"
    runs.mkdir()
    _write_paused(runs, True)

    with FakePLC() as plc:
        plc.set_actual_xyz([0.0, 0.0, 0.0])
        base = load_config(ROOT / "config/nest_clear.yaml")
        detection = {
            name: {**base.tool_calibration.detection[name], "method": "silhouette", "threshold": 127, "min_area": 5}
            for name in ("front", "side")
        }
        cameras = {
            name: {
                "backend": "mock",
                "size": [480, 640],
                "target_uv": [320, 240],
                "ref_tip_undetectable": True,
            }
            for name in ("front", "side")
        }
        config = replace(
            base,
            plc=replace(base.plc, ip=plc.host, port=plc.port),
            vision=replace(
                base.vision,
                camera_backend="mock",
                detector="stub",
                pixels=(),
                allow_placeholder_calibration=True,
                calibration=str(cal / "camera.yaml"),
            ),
            tool_calibration=replace(
                base.tool_calibration,
                enabled=True,
                directory=str(cal),
                cameras=cameras,
                detection=detection,
            ),
            tool_offset={**base.tool_offset, "settle_samples": 3, "settle_window_s": 0.02, "jitter_tol_mm": 0.02},
            record_directory=str(runs),
        )
        events = EventRecorder(runs, machine_id=config.machine_id, run=True)
        app = Application(config, ROOT / "config/nest_clear.yaml", events=events)
        gate = SupervisorGate(runs, mark=app.mark)
        try:
            yield plc, app, gate, runs, cal
        finally:
            app.close()


def test_paused_baseline_full_session_and_command_ack(paused_rig) -> None:
    """暂停状态下发 baseline_* 全流程（开始/选工具/测量/保存/确认/结束）均被确认并完成。"""
    plc, app, gate, runs, cal = paused_rig
    assert gate.paused() is True

    seq = send(runs, "baseline_start")
    _step_until(
        app,
        gate,
        lambda: (
            app.events.baseline_session is not None
            and _read_main_status(runs)["command_seq"] >= seq
        ),
    )
    assert app.handle.batch.blackboard.read("mode") == Mode.BASELINE

    seq = send(runs, "baseline_select", {"tool": 1})
    _step_until(
        app,
        gate,
        lambda: (
            app.events.baseline_session.phase == "live"
            and _read_main_status(runs)["command_seq"] >= seq
        ),
    )

    seq = send(runs, "baseline_measure")
    _step_until(
        app,
        gate,
        lambda: (
            app.events.baseline_session.phase == "success"
            and _read_main_status(runs)["command_seq"] >= seq
        ),
    )
    assert app.events.baseline_session.candidate["p0"] == pytest.approx([0.0, 0.0, 0.0])

    seq = send(runs, "baseline_save")
    _step_until(
        app,
        gate,
        lambda: (
            app.events.baseline_session.phase == "confirm"
            and _read_main_status(runs)["command_seq"] >= seq
        ),
    )

    seq = send(runs, "baseline_confirm", {"accept": True})
    _step_until(
        app,
        gate,
        lambda: (
            app.events.baseline_session.phase == "saved"
            and _read_main_status(runs)["command_seq"] >= seq
        ),
    )
    loaded = load_baseline(cal / "tool1_baseline.yaml")
    assert loaded.tool_name == "tool1"
    assert loaded.p0 == pytest.approx([0.0, 0.0, 0.0])

    seq = send(runs, "baseline_end")
    _step_until(
        app,
        gate,
        lambda: (
            app.events.baseline_session is None
            and app.handle.batch.blackboard.read("mode") == Mode.IDLE
            and _read_main_status(runs)["command_seq"] >= seq
        ),
    )


@pytest.mark.parametrize("func,tool_id,expected_reply", [(20, 0, 999), (100, 2, 131)])
def test_paused_ignores_new_plc_requests_even_during_baseline_until_resumed(
    paused_rig, func: int, tool_id: int, expected_reply: int
) -> None:
    """暂停期间 PLC 发新功能码（含 20/100）不被接收；恢复未暂停后照常接收并应答。"""
    plc, app, gate, runs, _ = paused_rig
    seq = send(runs, "baseline_start")
    _step_until(app, gate, lambda: app.events.baseline_session is not None and _read_main_status(runs)["command_seq"] >= seq)

    task = plc.request(func, tool_id=tool_id)
    for _ in range(15):
        app.step(gate)
        time.sleep(0.01)
    assert plc.io.read("result_id") == 0
    assert app.tracker.processed is None
    assert app.events.request_pending is False

    seq = send(runs, "baseline_end")
    _step_until(
        app,
        gate,
        lambda: app.events.baseline_session is None and _read_main_status(runs)["command_seq"] >= seq,
    )
    for _ in range(10):
        app.step(gate)
        time.sleep(0.01)
    assert plc.io.read("result_id") == 0

    _write_paused(runs, False)
    _step_until(app, gate, lambda: plc.io.read("result_id") == task and not app.events.request_pending)
    assert plc.io.read("reply_func") == expected_reply


def test_baseline_session_unaffected_by_mid_session_pause_and_resume(paused_rig) -> None:
    """基准会话中途从非暂停→暂停→恢复→暂停，会话持续推进且不被中断。"""
    plc, app, gate, runs, cal = paused_rig
    _write_paused(runs, False)
    seq = send(runs, "baseline_start")
    _step_until(app, gate, lambda: app.events.baseline_session is not None and _read_main_status(runs)["command_seq"] >= seq)

    # 非暂停期间选工具
    seq = send(runs, "baseline_select", {"tool": 1})
    _step_until(app, gate, lambda: app.events.baseline_session.phase == "live" and _read_main_status(runs)["command_seq"] >= seq)

    # 会话中途切为暂停：测量与采图仍正常完成，同时 PLC 请求不被接收
    _write_paused(runs, True)
    task = plc.request(20)
    seq = send(runs, "baseline_measure")
    _step_until(app, gate, lambda: app.events.baseline_session.phase == "success" and _read_main_status(runs)["command_seq"] >= seq)
    assert plc.io.read("result_id") == 0

    # 恢复为未暂停：保存并结束正常；PLC 在基准期间的请求回 999 也不破坏会话
    _write_paused(runs, False)
    _step_until(app, gate, lambda: plc.io.read("result_id") == task and not app.events.request_pending)
    assert plc.io.read("reply_func") == 999
    assert app.events.baseline_session is not None
    assert app.events.baseline_session.phase == "success"

    # 再次切回暂停：保存与结束仍正常完成
    _write_paused(runs, True)
    seq = send(runs, "baseline_save")
    _step_until(app, gate, lambda: app.events.baseline_session.phase == "confirm" and _read_main_status(runs)["command_seq"] >= seq)
    seq = send(runs, "baseline_confirm", {"accept": True})
    _step_until(app, gate, lambda: app.events.baseline_session.phase == "saved" and _read_main_status(runs)["command_seq"] >= seq)
    assert load_baseline(cal / "tool1_baseline.yaml").tool_name == "tool1"

    seq = send(runs, "baseline_end")
    _step_until(app, gate, lambda: app.events.baseline_session is None and _read_main_status(runs)["command_seq"] >= seq)
    assert app.handle.batch.blackboard.read("mode") == Mode.IDLE
