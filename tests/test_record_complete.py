"""验证跨日事件、盘摘要、存图和真实子进程日报。"""

from datetime import date, datetime, timedelta, timezone
import json
import os
import signal
import threading
from pathlib import Path
import sys
import time

import numpy as np
import pytest
import yaml

from nest_clear.record.events import EventRecorder
from nest_clear.record.images import ImageArchive
from nest_clear.record.report import report
from nest_clear.record.summary import TraySummary
from nest_clear.supervisor.gate import SupervisorGate
from sim.fake_plc import FakePLC

ROOT = Path(__file__).resolve().parents[1]


def test_midnight_and_summary(tmp_path: Path) -> None:
    """跨午夜事件分文件，暂停扣时且正常盘摘要保留在开始日期。"""
    now = [datetime(2026, 1, 1, 23, 59, 59, tzinfo=timezone.utc)]
    mono = [10.0]
    events = EventRecorder(tmp_path, clock=lambda: now[0], monotonic=lambda: mono[0])
    summary = TraySummary(events)
    events.emit("request", task_id=1)
    mono[0] = 12
    events.emit("device_pause_begin")
    now[0] += timedelta(seconds=4)
    mono[0] = 16
    events.emit("device_pause_end")
    events.emit("detection", count=2)
    events.emit("work_cycle", type="fine", reason="ok", fine_xyz=[0, 0, 0])
    events.emit("reply_sent")
    mono[0] = 20
    events.emit("tray_end")
    first = json.loads((tmp_path / "2026-01-01/events.jsonl").read_text().splitlines()[0])
    assert first["tray_id"] == "2026-01-01_0001"
    assert (tmp_path / "2026-01-02/events.jsonl").exists()
    row = json.loads((tmp_path / "2026-01-01/summary.jsonl").read_text())
    assert (row["scans"], row["targets"], row["picks"], row["pause_s"], row["net_s"], row["status"]) == (
        1,
        2,
        1,
        4,
        6,
        "normal",
    )
    events.emit("request", task_id=2)
    summary.finish("interrupted", now[0].isoformat(), 22)
    assert json.loads((tmp_path / "2026-01-02/summary.jsonl").read_text())["status"] == "interrupted"


def test_images_cleanup(tmp_path: Path) -> None:
    """只删除过期图片，原图和叠加图均在后台完成。"""
    old = tmp_path / "2000-01-01"
    (old / "images").mkdir(parents=True)
    (old / "events.jsonl").write_text("保留")
    events = EventRecorder(tmp_path)
    events.tray_id = "001"
    archive = ImageArchive(events, True, "png", 3, 30)
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    raw_path = archive.raw(image, 1, 1)
    archive.overlay(image, [], 1, 1, raw_path)
    archive.close()
    assert not (old / "images").exists()
    assert (old / "events.jsonl").exists()
    assert len(list(tmp_path.glob("*/images/*.png"))) == 2


def test_gate_pause_and_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """真实暂停门的变化进入摘要并能在日报中看到。"""
    now = [10.0]
    monkeypatch.setattr("nest_clear.supervisor.gate.time.monotonic", lambda: now[0])
    events = EventRecorder(tmp_path, monotonic=lambda: now[0])
    TraySummary(events)
    events.emit("request", task_id=1)
    gate = SupervisorGate(tmp_path, mark=events.emit)
    state = tmp_path / "supervisor.json"
    state.write_text('{"paused": true}')
    assert gate.paused()
    now[0] = 15
    state.write_text('{"paused": false}')
    assert not gate.paused()
    now[0] = 20
    events.emit("tray_end")
    text = report(tmp_path, date.today())
    assert "盘数：1" in text and "暂停开始" in text and "净耗时 5.0 秒" in text


def test_real_supervisor_child_tray(tmp_path: Path) -> None:
    """模拟 PLC 驱动真实父子进程完成一盘，并验证日报。"""
    import subprocess

    config = yaml.safe_load((ROOT / "config/nest_clear.yaml").read_text())
    with FakePLC() as plc:
        config["plc"].update(ip=plc.host, port=plc.port, timeout_s=1)
        config["record"]["directory"] = str(tmp_path / "runs")
        config["tool_calibration"]["enabled"] = False
        config["vision"].update(
            camera_backend="mock",
            detector="stub",
            responsibility_zone=False,  # 本测试验证记录，不测试视野分配。
            allow_placeholder_calibration=True,
            calibration=str(ROOT / "config/calibration/camera.yaml"),
            pixels=[[640, 480, 0.9], [660, 480, 0.8]],
        )
        config["scan"].update(mode="corners", corners=[[1, 0], [0, 0], [1, 0]], step_x=1, step_y=1, z=0)
        config["supervisor"].update(
            poll_s=0.02,
            watchdog_s=3,
            startup_grace_s=10,
            child_command=[
                sys.executable,
                "-m",
                "nest_clear.main",
                "--config",
                str(tmp_path / "config.yaml"),
                "--supervised",
            ],
        )
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump(config))
        process = subprocess.Popen(
            [sys.executable, "-m", "nest_clear.supervisor", "--config", str(config_path)],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            plc.set_status(128)
            alive = tmp_path / "runs/main.alive"
            deadline = time.monotonic() + 15
            while not alive.exists() and time.monotonic() < deadline:
                assert process.poll() is None
                time.sleep(0.05)
            assert alive.exists()
            assert plc.drive_tray(timeout_s=8)[-1]["func_code"] == 9
            deadline = time.monotonic() + 5
            while not list((tmp_path / "runs").glob("*/*/summary.jsonl")) and time.monotonic() < deadline:
                time.sleep(0.02)
            # PLC 清除任务编号并重新拉起初始化位，父程序按既有边沿重启主程序。
            state = tmp_path / "runs/supervisor.json"
            events_path = tmp_path / "runs" / date.today().isoformat() / "events.jsonl"
            plc.set_status(0)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if state.exists() and json.loads(state.read_text()).get("word") == 0:
                    break
                time.sleep(0.02)
            else:
                raise AssertionError("父程序未观察到初始化位清零")
            plc.io.write("task_id", 0)
            plc.task_id = 0
            plc.set_status(128)
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                assert process.poll() is None
                events = [json.loads(line) for line in events_path.read_text().splitlines()]
                starts = [row["child_pid"] for row in events if row["event"] == "child_start"]
                if len(starts) >= 2 and plc.io.read("algorithm_ready") == 1:
                    break
                time.sleep(0.05)
            else:
                raise AssertionError("初始化后主程序未重启并就绪")
            assert starts[-1] != starts[0]
            assert plc.io.read("task_id") == 0
            assert plc.io.read("result_id") == 0
            assert plc.io.read("reply_func") == 0
            child = starts[-1]
            first_task = plc.request(1)
            assert plc.wait_reply(first_task, 8)["func_code"] == 1
            plc.set_status(8)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if json.loads(state.read_text()).get("paused"):
                    break
                time.sleep(0.02)
            else:
                raise AssertionError("父程序未发布暂停")
            pending = plc.request(20)
            os.kill(child, signal.SIGKILL)
            assert plc.wait_reply(pending, 8)["func_code"] == 999
            plc.set_status(0)
            time.sleep(0.1)
            plc.set_status(128)
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                rows = [json.loads(line) for path in sorted((tmp_path / "runs").glob("*/*/summary.jsonl"))
                        for line in path.read_text().splitlines()]
                if len(rows) == 2:
                    break
                time.sleep(0.05)
            assert rows[1]["status"] == "interrupted"
            text = report(tmp_path / "runs", date.today())
            assert "盘数：2" in text and "正常结束" in text
            assert "挑取 0 次" in text
            assert "需要人看的：" in text and "代答 999" in text and "中断的盘" in text
            print("\n" + text)
        finally:
            process.send_signal(__import__("signal").SIGINT)
            process.wait(timeout=10)


def test_recover_interrupted_checkpoint(tmp_path: Path) -> None:
    """异常退出后重启时补写一次中断摘要，保留最后已知时间。"""
    mono = [1.0]
    events = EventRecorder(tmp_path, monotonic=lambda: mono[0])
    TraySummary(events)
    events.emit("request", task_id=1)
    mono[0] = 4
    events.emit("work_cycle", type="fine", reason="ok", fine_xyz=[0, 0, 0])
    events.emit("reply_sent")
    TraySummary(EventRecorder(tmp_path, monotonic=lambda: 50.0))
    path = tmp_path / date.today().isoformat() / "summary.jsonl"
    row = json.loads(path.read_text())
    assert row["status"] == "interrupted" and row["duration_s"] == 3 and row["picks"] == 1
    assert not (tmp_path / "tray_active.json").exists()


def test_log_midnight(tmp_path: Path) -> None:
    """日志处理器跨午夜时切换目录。"""
    import logging
    from nest_clear.record.log_setup import DailyHandler

    day = [date(2026, 1, 1)]
    handler = DailyHandler(tmp_path, "main", clock=lambda: day[0])
    record = logging.LogRecord("nest_clear.test", logging.INFO, __file__, 1, "测试", (), None)
    handler.emit(record)
    day[0] = date(2026, 1, 2)
    handler.emit(record)
    handler.close()
    assert (tmp_path / "2026-01-01/main.log").read_text().strip() == "测试"
    assert (tmp_path / "2026-01-02/main.log").read_text().strip() == "测试"


def test_bounded_images_drop_count(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """写图卡住时下一张立即丢弃，事件和盘摘要都记录张数。"""
    events = EventRecorder(tmp_path)
    TraySummary(events)
    events.emit("request", task_id=1)
    archive = ImageArchive(events, True, "png", 3, 30, queue_size=1)
    entered = threading.Event()
    release = threading.Event()

    def blocked(path: Path, image: object, targets: object) -> None:
        """控制后台写入时长，确定队列确实已满。"""
        entered.set()
        release.wait(3)

    archive._write = blocked
    frame = np.zeros((5, 5), dtype=np.uint8)
    try:
        assert archive.raw(frame, 1, 1)
        assert entered.wait(2)
        assert archive.raw(frame, 2, 2) is None
        assert "队列已满" in caplog.text
        events.emit("tray_end")
        summary = json.loads(next(tmp_path.glob("*/summary.jsonl")).read_text())
        assert summary["images_dropped"] == 1
        rows = [json.loads(line) for line in next(tmp_path.glob("*/events.jsonl")).read_text().splitlines()]
        assert any(row["event"] == "image_dropped" and row["task_id"] == 2 for row in rows)
    finally:
        release.set()
        archive.close()


def test_report_groups_and_attention(tmp_path: Path) -> None:
    """日报按日志器及首句分组，列出每种需要人工查看的事项。"""
    day = date(2026, 1, 2)
    directory = tmp_path / day.isoformat()
    directory.mkdir()
    events = [
        {"ts": "2026-01-02T03:04:05+08:00", "event": "device_state", "bits": ["急停中"]},
        {"ts": "2026-01-02T03:04:06+08:00", "event": "watchdog", "reason": "超时"},
        {"ts": "2026-01-02T03:04:07+08:00", "event": "error", "task_id": 1, "reason": "检测失败"},
        {"ts": "2026-01-02T03:04:08+08:00", "event": "reply", "task_id": 1, "func": 999},
    ]
    (directory / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in events))
    (directory / "main.log").write_text(
        "2026-01-02 03:04:05,123 WARNING nest_clear.test 相机断线；检查网络\n"
        "2026-01-02 03:04:06,123 WARNING nest_clear.test 相机断线；重试\n"
        "2026-01-02 03:04:07,123 ERROR nest_clear.test 检测失败。检查设备\n"
    )
    text = report(tmp_path, day)
    assert "nest_clear.test：相机断线（2 条）" in text
    assert "03:04:08 999" in text and "检测失败" in text
    assert "03:04:06 看门狗" in text and "03:04:05 急停中" in text
    assert "2026-01-02T03" not in text


def test_report_mode_rejection_deduplicates_and_translates_older_diagnostics(tmp_path: Path) -> None:
    """拒绝事件及其普通 999 和错误日志只算一项，旧检测记录也能阅读。"""
    day = date(2026, 1, 2)
    directory = tmp_path / day.isoformat()
    directory.mkdir()
    reason = "挑毛进行中收到对刀指令，须先停止挑毛并由 PLC 重新初始化"
    events = [
        {"ts": "2026-01-02T03:04:05", "event": "reply", "tray_id": "盘1", "task_id": 8, "func": 999},
        {"ts": "2026-01-02T03:04:05", "event": "mode_rejected", "tray_id": "盘1", "task_id": 8,
         "mode": "挑毛中", "request_func": 100, "reason": reason},
        {"ts": "2026-01-02T03:04:06", "event": "reply", "tray_id": "盘1", "task_id": 9, "func": 999},
        {"ts": "2026-01-02T03:04:07", "event": "tooltip_failed", "task_id": 10, "tool": 1,
         "reason": "错误码：TOOLTIP-FAILED\n发生了什么：front 相机无法识别针尖：no dark component\n位置：对刀\n排查建议：检查设备。"},
    ]
    (directory / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in events))
    (directory / "main.log").write_text(
        f"2026-01-02 03:04:05,123 ERROR nest_clear.flow Task_ID=8 处理失败：{reason}\n"
    )
    text = report(tmp_path, day)
    attention = text.split("需要人看的：", 1)[1]
    assert attention.count("Task_ID=8") == 1
    assert "模式不允许：挑毛中 时收到 100（工具1开始对刀），须先停止挑毛并由 PLC 重新初始化" in attention
    assert "Task_ID=9" in attention and "999 次数：2" in text
    assert "no dark component" not in text
    assert "    错误码：TOOLTIP-FAILED；发生了什么：正面相机无法识别针尖：图像中未找到暗色针尖区域\n    排查建议：检查背光与曝光" in attention
    assert "位置：对刀" not in attention


def test_report_attention_includes_all_t20_failures(tmp_path):
    day = date(2026, 1, 2)
    events = EventRecorder(tmp_path, clock=lambda: datetime(2026, 1, 2, tzinfo=timezone.utc))
    for event, fields in [
        ('error', {'reason': 'CAMERA-CAPTURE 采图失败'}),
        ('error', {'operation': 'write_readback', 'reason': '回读失败'}),
        ('heartbeat_failed', {'reason': '心跳故障'}),
        ('error', {'operation': 'clear_algorithm_ready', 'reason': '就绪位清零失败'}),
        ('placeholder_calibration', {}),
        ('tooltip_quality_warning', {'reason': '对刀质量超限'}),
        ('error', {'operation': 'image', 'reason': '图片写盘失败'}),
    ]:
        events.emit(event, **fields)
    attention = report(tmp_path, day).split('需要人看的：', 1)[1]
    for text in ('采图失败', '回读失败', '心跳中断', '就绪位清零失败', '占位标定放行中', '对刀质量超限', '图片写盘失败'):
        assert text in attention
