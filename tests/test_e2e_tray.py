"""通过真实 Modbus TCP 模拟 PLC 验证整盘握手及故障。"""
from dataclasses import replace
import json
from pathlib import Path
import threading
import time

import pytest

from nest_clear.config import Scan, load_config
from nest_clear.main import Application
from nest_clear.vision.detector import PixelTarget
from sim.fake_plc import FakePLC

ROOT = Path(__file__).resolve().parents[1]


class SequenceDetector:
    """首张图返回两个目标，后续图为空，用于检验不重复拍照。"""
    def __init__(self) -> None:
        """初始化采图计数。"""
        self.calls = 0

    def detect(self, image: object) -> list[PixelTarget]:
        """依次生成两目标和空目标。"""
        self.calls += 1
        if self.calls == 1:
            return [PixelTarget(640, 480, .9), PixelTarget(660, 480, .8)]
        return []


@pytest.fixture
def running(tmp_path: Path):
    """启动模拟 PLC、Mock 相机和后台主循环。"""
    with FakePLC(wrap=3) as plc:
        config = load_config(ROOT / "config/nest_clear.yaml")
        config = replace(config, plc=replace(config.plc, ip=plc.host, port=plc.port), scan=Scan(((1., 0.), (0., 0.), (1., 0.)), 1., 1., 0.), record_directory=str(tmp_path), tools=((1., 2., 3.), (4., 5., 6.)), vision=replace(config.vision, camera_backend="mock"))
        detector = SequenceDetector()
        app = Application(config, ROOT / "config/nest_clear.yaml", detector=detector)
        stopped = threading.Event()
        errors = []

        def loop() -> None:
            """运行高频测试时钟并收集后台异常。"""
            try:
                while not stopped.is_set():
                    app.step()
                    time.sleep(.002)
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=loop)
        thread.start()
        yield plc, app, detector, tmp_path
        stopped.set()
        thread.join(3)
        app.close()
        assert not errors


def test_full_tray_and_wrap(running) -> None:
    """首位不拍照、同位挑两次，工具交替，末位无目标后结束。"""
    plc, app, detector, directory = running
    records = plc.drive_tray(timeout_s=5)
    assert [item["func_code"] for item in records] == [1, 3, 3, 1, 9]
    assert [item["request_func"] for item in records] == [1, 1, 60, 60, 1]
    assert [item["task_id"] for item in records] == [1, 2, 3, 1, 2]
    assert detector.calls == 2
    assert records[1]["pick1_xyz"] == pytest.approx([2., 2., 3.])
    assert records[2]["pick2_xyz"] == pytest.approx([7., 5., 6.])
    writes = [address for address, _ in plc.writes]
    assert plc.contract["pick1_x"].address in writes
    assert plc.contract["pick2_x"].address in writes
    assert writes.index(plc.contract["pick1_x"].address) < writes.index(plc.contract["pick2_x"].address)
    for name in ("pick1_x", "pick2_x"):
        index = writes.index(plc.contract[name].address)
        assert writes[index:index + 3] == [plc.contract[name].address, plc.contract["reply_func"].address, plc.contract["result_id"].address]
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        events = [json.loads(line) for line in next(directory.glob("*/events.jsonl")).read_text().splitlines()]
        if any(event["event"] == "tray_end" for event in events):
            break
        time.sleep(.005)
    replies = [event for event in events if event["event"] == "reply"]
    assert [event["tool"] for event in replies if event["func"] == 3] == [1, 2]
    assert len([event for event in events if event["event"] == "tray_end"]) == 1


def test_unknown_and_arrival_error(running) -> None:
    """未知请求和扫描到位超差均返回 999，附带原因。"""
    plc, app, detector, directory = running
    task = plc.request(123)
    assert plc.wait_reply(task, 5)["func_code"] == 999
    task = plc.request(1)
    reply = plc.wait_reply(task, 5)
    assert reply["func_code"] == 1
    assert detector.calls == 0
    plc.set_actual_xyz([100, 100, 100])
    task = plc.request(1)
    assert plc.wait_reply(task, 5)["func_code"] == 999
    events = [json.loads(line) for line in next(directory.glob("*/events.jsonl")).read_text().splitlines()]
    assert any(event["event"] == "error" and "未知功能码" in event["reason"] for event in events)
    assert any(event["event"] == "arrival" and not event["ok"] for event in events)


def test_nonzero_baseline_ignored(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """启动时清应答并忽略已有编号，只处理后续变化。"""
    with FakePLC(wrap=3) as plc:
        plc.task_id = 2
        plc.io.write("request_func", 1)
        plc.io.write("task_id", 2)
        plc.io.write("result_id", 2)
        plc.io.write("reply_func", 1)
        config = load_config(ROOT / "config/nest_clear.yaml")
        config = replace(config, plc=replace(config.plc, ip=plc.host, port=plc.port), record_directory=str(tmp_path), vision=replace(config.vision, camera_backend="mock"))
        app = Application(config, ROOT / "config/nest_clear.yaml")
        try:
            assert plc.io.read("result_id") == 0
            assert plc.io.read("reply_func") == 0
            for _ in range(60):
                app.step()
                time.sleep(.002)
            assert plc.io.read("result_id") == 0
            assert "忽略遗留请求" in caplog.text
            task = plc.request(1)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and plc.io.read("result_id") != task:
                app.step()
                time.sleep(.002)
            assert plc.io.read("result_id") == task
            assert plc.io.read("reply_func") == 1
        finally:
            app.close()


def test_detector_failure_returns_999(running) -> None:
    """检测器异常必须通过 999 和事件暴露，不能被完成编号掩盖。"""
    plc, app, detector, directory = running

    def broken(image: object) -> list[PixelTarget]:
        """模拟检测设备失败。"""
        raise RuntimeError("检测器离线")

    detector.detect = broken
    task = plc.request(1)
    reply = plc.wait_reply(task, 5)
    plc.set_actual_xyz(reply["scan_xyz"])
    task = plc.request(1)
    assert plc.wait_reply(task, 5)["func_code"] == 999
    assert "检测器离线" in next(directory.glob("*/events.jsonl")).read_text()


def test_next_tray_starts_with_tool_one(running) -> None:
    """回 9 后重新提交 Batch，新盘工具顺序从一号重新开始。"""
    plc, app, detector, directory = running
    assert plc.drive_tray(timeout_s=5)[-1]["func_code"] == 9
    detector.calls = 0
    records = plc.drive_tray(timeout_s=5)
    assert [record["func_code"] for record in records] == [1, 3, 3, 1, 9]
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        events = [json.loads(line) for line in next(directory.glob("*/events.jsonl")).read_text().splitlines()]
        if sum(event["event"] == "tray_end" for event in events) == 2:
            break
        time.sleep(.005)
    picks = [event["tool"] for event in events if event["event"] == "reply" and event["func"] == 3]
    assert picks == [1, 2, 1, 2]
