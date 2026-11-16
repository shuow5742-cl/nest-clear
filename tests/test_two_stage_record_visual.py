"""验证两段定位记录和画面，不需要 PLC 网络。"""
import json
from datetime import date

import cv2
import numpy as np

from nest_clear.record.events import EventRecorder
from nest_clear.record.images import ImageArchive
from nest_clear.record.report import report
from nest_clear.vision.fine import FineResult
from nest_clear.vision.geometry import Calibration
from nest_clear.vision.hair_integration import render_fine


def test_report_two_stage_events(tmp_path):
    """下一粗拍结算成功失败，未复拍挑取保持待判定。"""
    day = date(2026, 1, 2)
    directory = tmp_path / day.isoformat()
    directory.mkdir()
    rows = []
    def add(kind, **fields):
        rows.append({"event": kind, "tray_id": "t", **fields})
    for x in (1, 5):
        add("work_cycle", type="fine", reason="ok", view_index=0, coarse_xyz=[x, 0, 0], fine_xyz=[x, 0, 0])
    for reason in ("fine_empty", "fine_point_outside", "fine_rejected"):
        add("work_cycle", type="fine", reason=reason, view_index=0)
    add("work_cycle", type="coarse", view_index=0, red_xyz=[[1, 0, 0]], match_radius_mm=0.5,
        reason="cap", tool_change=True, attempts=3, tool_check_min_attempts=3,
        tool_check_total=2, tool_check_threshold=1)
    add("work_cycle", type="fine", reason="ok", view_index=1, coarse_xyz=[9, 0, 0], fine_xyz=[9, 0, 0])
    add("work_cycle", type="coarse", view_index=2, red_xyz=[], match_radius_mm=0.5, reason="coarse_limit")
    for code in (9, 10, 999):
        add("reply", func=code)
    (directory / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    text = report(tmp_path, day)
    assert "粗拍 2 次，精定位 6 次" in text
    assert "成功 1、失败 1、待判定 1" in text
    assert all(reason in text for reason in ("fine_empty", "fine_point_outside", "fine_rejected", "cap", "coarse_limit"))
    assert "回 9/10/999 次数 1/1/1" in text


def test_fine_render_and_stage_archive(tmp_path):
    """投影框和精拍点使用不同颜色，原图与叠加图文件名含拍照阶段。"""
    calibration = Calibration(np.array([[100., 0, 50], [0, 100., 50], [0, 0, 1]]),
                              np.zeros(5), np.eye(3), np.zeros(3), 100.)
    image = np.zeros((100, 100), np.uint8)
    result = FineResult("ok", pick_uv=(50, 50))
    output = render_fine(image, [], result, (0, 0, 0), calibration, 20)
    assert output.shape == (100, 100, 3)
    assert np.array_equal(output[30, 30], [255, 0, 255])
    assert (output == [255, 255, 0]).all(axis=2).any()
    events = EventRecorder(tmp_path)
    events.tray_id = "t"
    archive = ImageArchive(events, True, "png", 3, 30)
    for stage in ("coarse", "fine"):
        path = archive.raw(image, 1, 1, stage)
        archive.rendered(output, 1, 1, path)
    archive.close()
    paths = list(tmp_path.glob("*/images/*.png"))
    assert len(paths) == 4
    assert sum("coarse" in p.name for p in paths) == 2
    assert sum("fine" in p.name for p in paths) == 2
    assert all(cv2.imread(str(p)).shape[:2] == (100, 100) for p in paths)
