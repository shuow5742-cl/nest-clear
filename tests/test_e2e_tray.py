"""通过真实 Modbus TCP 模拟 PLC 验证整盘握手及故障。"""

from collections.abc import Iterator
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
            return [PixelTarget(640, 480, 0.9), PixelTarget(660, 480, 0.8)]
        if self.calls == 2:
            return [PixelTarget(660, 480, 0.8)]
        return []


@pytest.fixture
def running(tmp_path: Path) -> Iterator[tuple[FakePLC, Application, SequenceDetector, Path]]:
    """启动模拟 PLC、Mock 相机和后台主循环。"""
    with FakePLC(wrap=3) as plc:
        config = load_config(ROOT / "config/nest_clear.yaml")
        config = replace(
            config,
            plc=replace(config.plc, ip=plc.host, port=plc.port),
            scan=Scan(((1.0, 0.0), (0.0, 0.0), (1.0, 0.0)), 1.0, 1.0, 0.0),
            record_directory=str(tmp_path),
            tools=((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)),
            vision=replace(config.vision, camera_backend="mock", detector="stub", allow_placeholder_calibration=True,
                           responsibility_zone=False),
            tool_calibration=replace(config.tool_calibration, enabled=False),
        )
        detector = SequenceDetector()
        app = Application(config, ROOT / "config/nest_clear.yaml", detector=detector)
        stopped = threading.Event()
        errors = []

        def loop() -> None:
            """运行高频测试时钟并收集后台异常。"""
            try:
                while not stopped.is_set():
                    app.step()
                    time.sleep(0.002)
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=loop)
        thread.start()
        yield plc, app, detector, tmp_path
        stopped.set()
        thread.join(3)
        app.close()
        assert not errors


def test_full_tray_and_wrap(running, monkeypatch):
    """多视野依次经历三种精定位失败与成功，再检查两工具和提交顺序。"""
    from types import SimpleNamespace
    from nest_clear.vision.fine import FineResult
    from nest_clear.workers.vision import VisionWorker
    plc, app, detector, directory = running
    statuses = iter(("fine_empty", "fine_point_outside", "fine_rejected", "ok"))
    def execute(worker, payload):
        """注入确定的视觉结果，实际握手仍通过 TCP。"""
        cycle = payload["cycle"]
        if payload["work_action"] == "coarse":
            observations = []
            if cycle.view_index == 0 and cycle.attempt_count < 4:
                for x in (10., 20., 30., 40.):
                    observations.append(SimpleNamespace(target=SimpleNamespace(xyz=(x, 0., 0.)), status="eligible", reason=None, area=50 - x))
            return {"observations": observations}
        status = next(statuses)
        return {"fine_result": FineResult(status, target_xyz=(40., 0., 0.) if status == "ok" else None)}
    monkeypatch.setattr("nest_clear.flow.picking.render", lambda image, *args: image)
    monkeypatch.setattr(VisionWorker, "execute", execute)
    records = plc.drive_tray(timeout_s=5)
    assert records[-1]["func_code"] == 9
    assert sum(row["func_code"] == 3 for row in records) == 1
    pick = next(row for row in records if row["func_code"] == 3)
    assert pick["pick1_xyz"] == pytest.approx([41., 2., 3.])
    assert pick["pick2_xyz"] == pytest.approx([44., 5., 6.])
    writes = [address for address, _ in plc.writes]
    i = writes.index(plc.contract["pick1_x"].address)
    assert writes[i:i + 4] == [plc.contract[name].address for name in ("pick1_x", "pick2_x", "reply_func", "result_id")]
    assert plc.wait_reply(plc.request(1), 5)["func_code"] == 999



def test_unknown_and_arrival_error(running: tuple[FakePLC, Application, SequenceDetector, Path]) -> None:
    """未知请求和扫描到位超差均返回 999，附带原因。"""
    plc, app, detector, directory = running
    task = plc.request(123)
    assert plc.wait_reply(task, 5)["func_code"] == 999
    task = plc.request(1)
    reply = plc.wait_reply(task, 5)
    assert reply["func_code"] == 1
    assert detector.calls == 0
    plc.set_actual_xyz([100, 100, 100])
    task = plc.request(20)
    assert plc.wait_reply(task, 5)["func_code"] == 999
    events = [json.loads(line) for line in next(directory.glob("*/events.jsonl")).read_text().splitlines()]
    assert any(event["event"] == "error" and "功能码 123" in event["reason"] for event in events)
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
        config = replace(
            config,
            plc=replace(config.plc, ip=plc.host, port=plc.port),
            record_directory=str(tmp_path),
            vision=replace(config.vision, camera_backend="mock", detector="stub", allow_placeholder_calibration=True),
            tool_calibration=replace(config.tool_calibration, enabled=False),
        )
        app = Application(config, ROOT / "config/nest_clear.yaml")
        try:
            assert plc.io.read("result_id") == 0
            assert plc.io.read("reply_func") == 0
            for _ in range(60):
                app.step()
                time.sleep(0.002)
            assert plc.io.read("result_id") == 0
            assert "忽略遗留请求" in caplog.text
            task = plc.request(1)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and plc.io.read("result_id") != task:
                app.step()
                time.sleep(0.002)
            assert plc.io.read("result_id") == task
            assert plc.io.read("reply_func") == 1
        finally:
            app.close()


def test_detector_failure_returns_999(running: tuple[FakePLC, Application, SequenceDetector, Path]) -> None:
    """检测器异常必须通过 999 和事件暴露，不能被完成编号掩盖。"""
    plc, app, detector, directory = running

    def broken(image: object) -> list[PixelTarget]:
        """模拟检测设备失败。"""
        raise RuntimeError("检测器离线")

    detector.detect = broken
    task = plc.request(1)
    reply = plc.wait_reply(task, 5)
    plc.set_actual_xyz(reply["scan_xyz"])
    task = plc.request(20)
    assert plc.wait_reply(task, 5)["func_code"] == 999
    app.archive.close()
    events = [json.loads(line) for line in next(directory.glob("*/events.jsonl")).read_text().splitlines()]
    error = next(event for event in events if event["event"] == "error" and "检测器离线" in event["reason"])
    assert error["raw_path"] and (directory / error["raw_path"]).exists()
    reply = next(event for event in events if event["event"] == "reply" and event["func"] == 999)
    assert reply["raw_path"] == error["raw_path"]


def test_terminal_rejects_new_tray(running, monkeypatch):
    """一盘结束后保持终态，直到 PLC 重启主程序。"""
    from nest_clear.workers.vision import VisionWorker
    plc, app, detector, directory = running
    monkeypatch.setattr(VisionWorker, "execute", lambda worker, payload: {"observations": []})
    assert plc.drive_tray(timeout_s=5)[-1]["func_code"] == 9
    assert plc.wait_reply(plc.request(1), 5)["func_code"] == 999


@pytest.mark.parametrize('stage', ['coarse', 'fine'])
def test_capture_failure_returns_999_and_context(running, monkeypatch, stage):
    from types import SimpleNamespace
    from nest_clear.workers.vision import VisionWorker
    plc, app, _, directory = running
    def broken():
        raise RuntimeError('Failed to capture image')
    monkeypatch.setattr('nest_clear.flow.picking.render', lambda image, *args: image)
    monkeypatch.setattr(VisionWorker, 'execute', lambda worker, payload: {
        'observations': [SimpleNamespace(target=SimpleNamespace(xyz=(0., 0., 0.)),
                                         status='eligible', reason=None, area=3.)]})
    if stage == 'coarse':
        monkeypatch.setattr(app.camera.camera, 'snapshot', broken)
    start = plc.wait_reply(plc.request(1), 5)
    plc.set_actual_xyz(start['scan_xyz'])
    if stage == 'fine':
        coarse = plc.wait_reply(plc.request(20), 5)
        assert coarse['func_code'] == 1
        plc.set_actual_xyz(coarse['scan_xyz'])
        monkeypatch.setattr(app.camera.camera, 'snapshot', broken)
    task = plc.request(20)
    assert plc.wait_reply(task, 5)['func_code'] == 999
    rows = [json.loads(line) for line in next(directory.glob('*/events.jsonl')).read_text().splitlines()]
    event = next(row for row in rows if row['event'] == 'error' and row.get('task_id') == task)
    assert all(value in event['reason'] for value in (app.config.camera_sn, '粗拍' if stage == 'coarse' else '精定位', str(task), '取图超时'))
