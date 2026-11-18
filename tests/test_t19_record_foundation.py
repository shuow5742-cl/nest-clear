from datetime import date, datetime, timedelta, timezone
import json

import numpy as np

from nest_clear.record.events import EventRecorder
from nest_clear.record.images import ImageArchive
from nest_clear.record.report import _day_rows, report


def test_run_records_cross_midnight_and_archive(tmp_path):
    now = [datetime(2026, 1, 1, 23, 59, 59, tzinfo=timezone.utc)]
    events = EventRecorder(tmp_path, run=True, clock=lambda: now[0])
    events.begin_request(1, 20)
    events.plc("request", task_id=1, request_func=20)
    events.emit("request", task_id=1, func=20)
    archive = ImageArchive(events, True, "png", 3, 30)
    seq, capture_id = events.next_capture()
    events.emit("capture", capture_id=capture_id, task_id=1)
    path = archive.capture_raw(np.zeros((8, 8), np.uint8), sequence=seq,
                               capture_id=capture_id, kind="coarse", tray="1", view=1,
                               shot="c01", task_id=1)
    events.plc("reply", task_id=1, func=1)
    now[0] += timedelta(seconds=2)
    events.emit("reply", task_id=1, func=1)
    events.plc("plc_disconnected", task_id=1, failures=1)
    archive.close()
    assert events.run_id == "20260101_235959"
    assert events.run_directory.parent.name == "2026-01-01"
    assert path.endswith("000001_coarse_t1_v01_c01_raw.png")
    rows = [json.loads(line) for line in (events.run_directory / "events.jsonl").read_text().splitlines()]
    assert any(row["event"] == "image_written" and row["capture_id"] == capture_id for row in rows)
    assert all({"run_id", "machine_id", "wall", "mono", "seq_event"} <= row.keys() for row in rows)
    assert any(row["event"] == "reply" for row in _day_rows(tmp_path, "events.jsonl", date(2026, 1, 2), "wall"))
    assert not any(row["event"] == "reply" for row in _day_rows(tmp_path, "events.jsonl", date(2026, 1, 1), "wall"))
    edges = [json.loads(line) for line in (events.run_directory / "plc.jsonl").read_text().splitlines()]
    assert [row["event"] for row in edges] == ["request", "reply", "plc_disconnected"]
    assert all(row["xid"] == 1 for row in edges)
    assert next(row for row in rows if row["event"] == "request")["xid"] == edges[0]["xid"]


def test_report_collects_multiple_runs_and_cross_day(tmp_path):
    now = [datetime(2026, 1, 1, 23, 59, 59, tzinfo=timezone.utc)]
    first = EventRecorder(tmp_path, run=True, clock=lambda: now[0])
    first.tray_id = "one"
    now[0] += timedelta(seconds=2)
    first.emit("work_cycle", type="coarse", view_index=0)
    now[0] += timedelta(seconds=1)
    second = EventRecorder(tmp_path, run=True, clock=lambda: now[0])
    second.tray_id = "two"
    second.emit("work_cycle", type="coarse", view_index=0)
    text = report(tmp_path, date(2026, 1, 2))
    assert "盘 one" in text and "盘 two" in text
    assert text.count("粗拍 1 次") == 2


def test_startup_metadata_snapshot_and_updates(tmp_path, monkeypatch):
    from pathlib import Path
    from dataclasses import replace
    from nest_clear.config import load_config
    from nest_clear.main import write_meta
    from nest_clear.workers.camera import CameraWorker
    root = Path(__file__).resolve().parents[1]
    config_path = root / "config/nest_clear.yaml"
    config = replace(load_config(config_path), record_directory=str(tmp_path))
    calibration = config_path.parent / config.vision.calibration
    events = EventRecorder(tmp_path, run=True)
    camera = CameraWorker("mock", "SN", image_size=(4024, 3036))
    camera.open()
    monkeypatch.setattr("nest_clear.main.repository_commit", lambda _: "test-sha")
    monkeypatch.setattr("nest_clear.main.repository_dirty", lambda _: False)
    monkeypatch.setattr("nest_clear.main.framework_commit", lambda: "test-framework")
    try:
        write_meta(events, config_path, calibration, config, camera, None, [(0., 0., config.scan.z)])
        meta = json.loads((events.run_directory / "meta.json").read_text())
        assert meta["code"] == {"sha": "test-sha", "dirty": False}
        assert meta["camera"]["width"] == 4024
        assert meta["config"]["content"] == config_path.read_text()
        assert meta["detection_scales"][0]["parameters"]
        assert all(meta[key]["sha256"] for key in ("config", "contract", "calibration", "tools"))
        events.meta_update(task_id=12, tool=1, old=[0, 0, 0], new=[1, 2, 3])
        update = json.loads((events.run_directory / "meta_updates.jsonl").read_text())
        assert update["task_id"] == 12 and update["new"] == [1, 2, 3]
    finally:
        camera.on_stop()


def test_main_capture_records_pose_and_archive(tmp_path):
    from types import SimpleNamespace
    from test_hair_integration import Board
    from nest_clear.flow.leaves import RecordCapture
    from nest_clear.workers.camera import CameraWorker
    events = EventRecorder(tmp_path, run=True)
    events.tray_id = "tray1"
    events.begin_request(3, 20)
    archive = ImageArchive(events, True, "png", 3, 30)
    events.archive = archive
    camera = CameraWorker("mock", "S", image_size=(40, 30))
    camera.open()
    image, metadata = camera.execute({"stage": "coarse", "task_id": 3})
    board = Board(work_cycle=SimpleNamespace(view_index=0, coarse_shots=0, attempt_count=0),
                  work_action="coarse", capture_meta=metadata, image=image, task_id=3,
                  request_func=20, actual=(1., 2., 3.), scan=(1., 2., 3.), next_index=1)
    leaf = RecordCapture(events)
    leaf._blackboard = board
    leaf.perform()
    archive.close()
    camera.on_stop()
    rows = [json.loads(line) for line in (events.run_directory / "events.jsonl").read_text().splitlines()]
    capture = next(row for row in rows if row["event"] == "capture")
    assert capture["actual_xyz"] == [1., 2., 3.] and capture["target_xyz"] == [1., 2., 3.]
    assert capture["capture_seq"] == 1 and capture["shot"] == "c01"
    assert (tmp_path / capture["raw_path"]).exists()
    assert any(row["event"] == "image_written" and row["capture_id"] == capture["capture_id"] for row in rows)


def test_application_enables_run_directory_without_tcp(tmp_path, monkeypatch):
    from dataclasses import replace
    from pathlib import Path
    from nest_clear.config import load_config
    from nest_clear.main import Application
    class IO:
        def __init__(self, *args): pass
        def read_u16(self, address): return 0
        def write_u16(self, address, value): pass
        def write(self, key, value): pass
        def close(self): pass
    monkeypatch.setattr("nest_clear.main.PLCClient", IO)
    monkeypatch.setattr("nest_clear.main.repository_commit", lambda _: "test")
    monkeypatch.setattr("nest_clear.main.repository_dirty", lambda _: False)
    monkeypatch.setattr("nest_clear.main.framework_commit", lambda: "test")
    config_path = Path(__file__).resolve().parents[1] / "config/nest_clear.yaml"
    config = load_config(config_path)
    config = replace(config, record_directory=str(tmp_path),
                     vision=replace(config.vision, camera_backend="mock", detector="stub"),
                     tool_calibration=replace(config.tool_calibration, enabled=False))
    app = Application(config, config_path)
    try:
        assert app.events.run_directory.parent.parent == tmp_path
        assert (app.events.run_directory / "meta.json").exists()
        assert (app.events.run_directory / "events.jsonl").exists()
        assert not list(tmp_path.glob("*/events.jsonl"))
    finally:
        app.close()
