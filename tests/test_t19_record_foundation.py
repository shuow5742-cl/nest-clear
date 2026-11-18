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
