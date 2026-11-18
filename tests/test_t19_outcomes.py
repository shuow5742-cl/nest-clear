"""目标账、结局和状态画面的回归测试，不依赖 PLC 网络。"""
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from nest_clear.flow.work_cycle import WorkCycle
from nest_clear.record.events import EventRecorder
from nest_clear.vision.fine import FineResult
from nest_clear.vision.hair_integration import COLORS, render
from test_hair_integration import candidate, measure, HairDetectConfig, Calibration


def test_outcomes_and_target_identity(tmp_path):
    events = EventRecorder(tmp_path, run=True)
    events.begin_request(1, 20)
    params = SimpleNamespace(skip_radius_mm=.2, max_fine_targets=10, cap_ratio=3,
                             tool_check_min_attempts=99, tool_check_ratio=.5, max_coarse_shots_per_view=30)
    def make():
        return WorkCycle([(0, 0, 0)], lambda xyz, pose: xyz, lambda xyz: (xyz, xyz), params,
                         record=events.emit, tray="2026-01-01_0001")
    def obs(x):
        return SimpleNamespace(target=SimpleNamespace(xyz=(x, 0, 0)), identity=(x, 0, 0),
                               status="eligible", reason=None, area=1)
    def attempt(outcome):
        cycle = make(); cycle.start()
        first = obs(1.)
        cycle.on_coarse((0, 0, 0), [first], "coarse")
        cycle.begin_fine("fine", 1, 20, 1)
        if outcome == "not_sent":
            cycle.on_fine((0, 0, 0), FineResult("fine_empty"))
        else:
            cycle.on_fine((0, 0, 0), FineResult("ok", target_xyz=(1, 0, 0)))
            cycle.mark_reply(1, 1, {"pick1_xyz": (1, 0, 0), "pick2_xyz": (1, 0, 0)})
            if outcome == "pending":
                cycle.pending()
            else:
                cycle.mark_done(2, 2); cycle.on_picked()
                repeat = obs(1.05)
                cycle.on_coarse((0, 0, 0), [repeat] if outcome == "failed" else [], "review")
                if outcome == "failed":
                    assert repeat.target_id == first.target_id and repeat.status == "failed"
        return first.target_id
    for outcome in ("success", "failed", "pending", "not_sent"):
        attempt(outcome)
    rows = [json.loads(line) for line in (events.run_directory / "events.jsonl").read_text().splitlines()]
    assert {row["outcome"] for row in rows if row["event"] == "pick_result"} == {"success", "failed", "pending", "not_sent"}
    assert any(row["event"] == "target_match" and row["capture_id"] == "review" for row in rows)


def test_all_status_boxes_and_pick_crosses():
    calibration = Calibration(np.eye(3), np.zeros(5), np.eye(3), np.zeros(3), 1)
    image = np.zeros((300, 500, 3), np.uint8)
    observations = []
    for index, status in enumerate(("eligible", "rejected", "failed", "fine_failed", "attempted", "other_view")):
        x = 20 + 70*index
        raw = candidate(x=x, w=30, h=30)
        raw = replace(raw, pick_uv=(x+15, 115), identity_uv=(x+15, 115),
                      bbox=(x, 100, x+30, 130), polygon=((x,100),(x+30,100),(x+30,130),(x,130)))
        item = measure(raw, (0, 0, 0), calibration, HairDetectConfig())
        item.status = status
        observations.append(item)
    rendered = render(image, observations)
    for item in observations:
        x, y = map(round, item.candidate.pick_uv)
        assert tuple(rendered[y, x]) == COLORS[item.status]
        left, top = map(round, item.candidate.bbox[:2])
        assert tuple(rendered[top, left]) == COLORS[item.status]


def test_detection_event_contains_candidates(tmp_path):
    from dataclasses import replace
    from nest_clear.config import load_config
    from nest_clear.flow.context import FlowContext, RequestTracker
    from nest_clear.flow.picking import record_detection
    from test_hair_integration import Board
    config = load_config("config/nest_clear.yaml")
    calibration = Calibration(np.eye(3), np.zeros(5), np.eye(3), np.zeros(3), 1)
    events = EventRecorder(tmp_path, run=True)
    events.tray_id = "2026-01-01_0001"
    events.begin_request(5, 20)
    cycle = WorkCycle([(0, 0, 0)], lambda xyz, pose: xyz, lambda xyz: (xyz, xyz), config.vision,
                      record=events.emit, tray=events.tray_id)
    cycle.start()
    obs = measure(candidate(w=10, h=5), (0, 0, 0), calibration, config.vision.hair_detector)
    cycle.on_coarse((0, 0, 0), [obs], "capture-a")
    board = Board(task_id=5, request_func=20, capture_id="capture-a", actual=(0, 0, 0))
    ctx = FlowContext(None, config, [(0, 0, 0)], RequestTracker(0), events, calibration)
    record_detection(ctx, board, cycle, {"observations": [obs], "elapsed_ms": 1.2, "um_per_px": 10}, "coarse", 0)
    rows = [json.loads(line) for line in (events.run_directory / "events.jsonl").read_text().splitlines()]
    detection = next(row for row in rows if row["event"] == "detection")
    assert detection["capture_id"] == "capture-a" and detection["xid"] == 1
    assert {"target_id", "bbox_px", "identity_px", "identity_xyz", "pick_px", "pick_xyz", "length_mm", "area_mm2", "width_mm", "gray_ratio", "dark_frac", "gate_passed", "reject_reason", "responsibility", "status", "rank"} <= detection["candidates"][0].keys()


def test_ambiguous_match_is_logged(tmp_path):
    events = EventRecorder(tmp_path, run=True)
    params = SimpleNamespace(skip_radius_mm=1., max_fine_targets=10, cap_ratio=1,
                             tool_check_min_attempts=99, tool_check_ratio=.5, max_coarse_shots_per_view=30)
    cycle = WorkCycle([(0, 0, 0)], lambda xyz, pose: xyz, lambda xyz: (xyz, xyz), params, record=events.emit)
    cycle.start()
    def obs(x):
        return SimpleNamespace(target=SimpleNamespace(xyz=(x, 0, 0)), identity=(x, 0, 0), status="eligible", reason=None, area=1)
    first, second = obs(0), obs(.5)
    cycle.match_observations([first, second], "first")
    middle = obs(.2)
    cycle.match_observations([middle], "second")
    assert middle.target_id == first.target_id
    rows = [json.loads(line) for line in (events.run_directory / "events.jsonl").read_text().splitlines()]
    ambiguous = next(row for row in rows if row["event"] == "match_ambiguous")
    assert ambiguous["capture_id"] == "second" and len(ambiguous["matches"]) == 2


def test_report_uses_recorded_outcomes_and_attention(tmp_path):
    from datetime import date
    from nest_clear.record.report import report
    day = date(2026, 1, 2)
    folder = tmp_path / day.isoformat()
    folder.mkdir()
    rows = [
        {"event": "capture", "tray_id": "t", "capture_id": "before", "raw_path": "images/before.png"},
        {"event": "capture", "tray_id": "t", "capture_id": "after", "raw_path": "images/after.png"},
        {"event": "work_cycle", "tray_id": "t", "type": "coarse"},
        {"event": "reply_sent", "tray_id": "t"},
        {"event": "reply_sent", "tray_id": "t"},
        {"event": "pick_result", "tray_id": "t", "outcome": "failed", "target_id": "t-1-001", "coarse_capture_id": "before", "review_capture_id": "after"},
        {"event": "pick_result", "tray_id": "t", "outcome": "pending"},
        {"event": "pick_result", "tray_id": "t", "outcome": "not_sent"},
        {"event": "image_dropped", "tray_id": "t"},
        {"event": "match_ambiguous", "tray_id": "t"},
    ]
    (folder / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    text = report(tmp_path, day)
    assert "成功 0、失败 1、待判定 1" in text and "未下发 1" in text
    assert "t-1-001" in text and "before.png" in text and "after.png" in text
    assert "缺图" in text and "待判定超过 20%" in text and "身份匹配歧义" in text


def test_pending_is_recorded_before_shutdown_and_superseded(tmp_path):
    from nest_clear.record.report import _work_stats
    from nest_clear.record.summary import TraySummary
    events = EventRecorder(tmp_path, run=True)
    summary = TraySummary(events)
    events.emit("request", task_id=1)
    events.emit("pick_result", attempt_id="a", outcome="pending")
    events.emit("pick_result", attempt_id="a", outcome="success")
    assert summary.data["pending"] == 0 and summary.data["success"] == 1
    rows = [json.loads(line) for line in (events.run_directory / "events.jsonl").read_text().splitlines()]
    stats = _work_stats(rows)
    assert stats["pending"] == 0 and stats["success"] == 1
