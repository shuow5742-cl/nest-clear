"""T95：两路针尖都最清晰即可建立对刀基准、取消主相机中心约束、峰值样本与靠边保护、人工确认参与就绪及换针对刀目标像素回归。"""
from dataclasses import replace
from datetime import datetime
import csv
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml
from autoweaver.motion_policy.nodes.node import Status

from nest_clear.baseline.focus_rules import (
    BASELINE_RULE,
    EDGE_MARGIN_MESSAGE,
    STAGE_ASCENDING_TEXT,
    STAGE_OVERSHOT_TEXT,
    STAGE_READY_TEXT,
    evaluate_baseline_readiness,
    evaluate_camera_readiness,
)
from nest_clear.baseline.preparation import Preparation, ReadinessTracker, analyze_tip
from nest_clear.baseline.session import BaselineSession, DEFAULTS, settings as baseline_settings
from nest_clear.config import load_config
from nest_clear.flow.calibration import (
    ConfirmCalibration,
    DetectTips,
    SaveCompensation,
    SolveTips,
    StartCalibration,
)
from nest_clear.flow.modes import Mode
from nest_clear.paths import tooltip_preview_file
from nest_clear.tooltip import ToolTipBaseline, ViewCalib
from nest_clear.tooltip.click import ClickResult
from nest_clear.tooltip.focus_metrics import BandGradientTracker
from nest_clear.tooltip.store import initialize_tools, load_baseline, save_baseline, save_tools
from nest_clear.tooltip.trident import TridentResult, TridentTipDetector
from nest_clear.workers.calib_camera import CaptureImages
from test_trident import scene
from test_ui_headless import Style, Widget

ROOT = Path(__file__).resolve().parents[1]


class _Events:
    def __init__(self, root: Path):
        self.directory = root
        self.run_directory = root / "run"
        self.rows = []

    def clock(self):
        return datetime.now().astimezone()

    def emit(self, event, **kw):
        self.rows.append((event, kw))

    def meta_update(self, **kw):
        pass


class _AxisPlc:
    def __init__(self, xyz=(1.0, 2.0, 3.0)):
        self.xyz = list(xyz)

    def execute(self, payload):
        assert payload == {"operation": "actual"}
        return list(self.xyz)


def _wait(predicate, timeout=6.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("等待条件超时")


def _render_needle(center_x: int = 300, sigma: float = 0.0) -> np.ndarray:
    """生成指定横向位置 center_x 和离焦程度 sigma 的标准三棱针图（600x600）。"""
    y, x = np.indices((600, 600))
    image = np.clip(225 + 30 * np.sin(x / 18) * np.sin(y / 17), 0, 255).astype(np.uint8)
    cv2.rectangle(image, (center_x - 50, 0), (center_x + 50, 100), 20, -1)
    cv2.rectangle(image, (center_x - 15, 100), (center_x + 15, 400), 20, -1)
    for du, end in ((-10, 440), (0, 450), (10, 445)):
        cv2.line(image, (center_x + du, 400), (center_x + du, end), 30, 3)
    if sigma > 0.05:
        image = cv2.GaussianBlur(image, (0, 0), sigma)
    return image


def test_settings_includes_t95_focus_readiness_defaults(tmp_path):
    cfg_path = tmp_path / "tooltip_baseline.yaml"
    cfg_path.write_text("baseline_ui:\n  main_tol_mm: 0.15\n", encoding="utf-8")
    loaded = baseline_settings(cfg_path)
    assert loaded["focus_smooth_frames"] == 5
    assert loaded["focus_pass_percent"] == pytest.approx(92.0)
    assert loaded["focus_ready_percent"] == pytest.approx(95.0)
    assert loaded["peak_reset_margin_percent"] == pytest.approx(3.0)
    assert loaded["focus_good_px"] == pytest.approx(2.0)
    assert loaded["focus_warn_px"] == pytest.approx(3.0)
    assert loaded["focus_max_px"] == pytest.approx(3.0)
    assert loaded["focus_ready_frames"] == 5
    assert loaded["peak_min_frames"] == 10
    assert loaded["edge_margin_px"] == pytest.approx(40.0)

    bad_pct = tmp_path / "bad_pct.yaml"
    bad_pct.write_text("baseline_ui:\n  focus_ready_percent: 120\n", encoding="utf-8")
    with pytest.raises(ValueError, match="focus_ready_percent"):
        baseline_settings(bad_pct)

    bad_pass = tmp_path / "bad_pass.yaml"
    bad_pass.write_text("baseline_ui:\n  focus_pass_percent: 96\n  focus_ready_percent: 95\n", encoding="utf-8")
    with pytest.raises(ValueError, match="focus_pass_percent"):
        baseline_settings(bad_pass)

    bad_smooth = tmp_path / "bad_smooth.yaml"
    bad_smooth.write_text("baseline_ui:\n  focus_smooth_frames: 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="focus_smooth_frames"):
        baseline_settings(bad_smooth)

    bad_max_px = tmp_path / "bad_max_px.yaml"
    bad_max_px.write_text("baseline_ui:\n  focus_good_px: 2.5\n  focus_max_px: 2.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="focus_max_px"):
        baseline_settings(bad_max_px)

    bad_frames = tmp_path / "bad_frames.yaml"
    bad_frames.write_text("baseline_ui:\n  peak_min_frames: 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="peak_min_frames"):
        baseline_settings(bad_frames)

    ok_zero_margin = tmp_path / "ok_zero_margin.yaml"
    ok_zero_margin.write_text("baseline_ui:\n  peak_reset_margin_percent: 0\n", encoding="utf-8")
    assert baseline_settings(ok_zero_margin)["peak_reset_margin_percent"] == pytest.approx(0.0)

    for bad_text in (
        "baseline_ui:\n  peak_reset_margin_percent: -0.1\n",
        "baseline_ui:\n  focus_pass_percent: 92\n  peak_reset_margin_percent: 8.0\n",
        "baseline_ui:\n  focus_pass_percent: 92\n  peak_reset_margin_percent: 9.5\n",
        "baseline_ui:\n  peak_reset_margin_percent: true\n",
    ):
        bad_margin = tmp_path / "bad_margin.yaml"
        bad_margin.write_text(bad_text, encoding="utf-8")
        with pytest.raises(ValueError, match="peak_reset_margin_percent"):
            baseline_settings(bad_margin)

    with pytest.raises(ValueError, match="peak_reset_margin_percent"):
        BandGradientTracker(smooth_frames=5, pass_percent=92.0, peak_reset_margin_percent=-0.5)
    with pytest.raises(ValueError, match="peak_reset_margin_percent"):
        BandGradientTracker(smooth_frames=5, pass_percent=92.0, peak_reset_margin_percent=8.0)


def _feed_readiness(rtracker, tracked, status, reason, tip_uv, edge_px, cfg, manual_confirmed=False):
    return rtracker.update(
        "front",
        {
            "detection_status": status,
            "detection_reason": reason,
            "tip_uv": list(tip_uv),
            "image_size": [600, 600],
            "percent": tracked["percent"],
            "peak": tracked["best"],
            "peak_confirmed": tracked.get("peak_confirmed", False),
            "sample_count": tracked["sample_count"],
            "edge_width_px": edge_px,
        },
        cfg,
        manual_confirmed=manual_confirmed,
    )


def test_overshoot_median_peak_reset_and_edge_fallback():
    """越峰确认、5帧中值抗单帧噪声尖峰、更高新峰重置、2.4px黄色可就绪与3.2px禁用的合成清晰度序列测试。"""
    tracker = BandGradientTracker(smooth_frames=5, pass_percent=92.0)
    rtracker = ReadinessTracker()
    cfg = dict(DEFAULTS, peak_min_frames=10, focus_ready_frames=5)

    def step_val(val: float, edge_px: float = 1.6, tip_uv=(300.0, 450.0)):
        tracked = tracker.update(val, tip_uv)
        return rtracker.update(
            "front",
            {
                "detection_status": "精定位",
                "detection_reason": "",
                "tip_uv": list(tip_uv),
                "image_size": [600, 600],
                "percent": tracked["percent"],
                "peak": tracked["best"],
                "peak_confirmed": tracked["peak_confirmed"],
                "sample_count": tracked["sample_count"],
                "edge_width_px": edge_px,
            },
            cfg,
        ), tracked

    # 1. 单调上升且尚未越峰 -> 即使 >10 帧且百分比 100%，依然不就绪，显示“尚未越过最清晰点”
    last_r = None
    for v in [60, 65, 70, 75, 80, 85, 90, 95, 100, 100, 100, 100, 100, 100, 100]:
        last_r, _ = step_val(float(v), edge_px=1.5)
        assert last_r["ready"] is False
    assert last_r["peak_confirmed"] is False
    assert last_r["stage_text"] == STAGE_ASCENDING_TEXT

    # 2. 单帧噪声尖峰（如突然抖到 150）被 5 帧中值过滤：峰值不抬高到 150，也不误触发越峰
    spike_r, spike_t = step_val(150.0, edge_px=1.5)
    assert spike_t["current"] == pytest.approx(100.0)
    assert spike_t["best"] == pytest.approx(100.0)
    assert spike_r["peak_confirmed"] is False

    # 再持续降到 <= 92%（例如连续 5 帧 85.0），中值降到 85.0 -> 越峰确认，阶段变为“已越峰，往回点动到峰值附近”
    for _ in range(4):
        last_r, last_t = step_val(85.0, edge_px=2.2)
    assert last_t["current"] == pytest.approx(85.0)
    assert last_t["percent"] == pytest.approx(85.0)
    assert last_r["peak_confirmed"] is True
    assert last_r["ready"] is False
    assert last_r["stage_text"] == STAGE_OVERSHOT_TEXT

    # 3. 回到 >= 95%（98.0）连续 5 帧，且边缘 2.4px（处于 2.0~3.0 黄色区间但 <= focus_max_px=3.0）-> 就绪并给出偏宽可接受提示
    for _ in range(2):
        # 前两帧受窗口中旧 85 影响需先让中值上到 >=95
        step_val(98.0, edge_px=2.4)
    ready_r = None
    for i in range(5):
        ready_r, _ = step_val(98.0, edge_px=2.4)
        if i < 4:
            assert ready_r["ready"] is False
    assert ready_r["ready"] is True
    assert ready_r["stage_text"] == STAGE_READY_TEXT
    assert ready_r["wide_edge_note"] == "该路最佳边缘 2.4px 偏宽，可接受"

    # 4. 边缘 3.2px (> focus_max_px=3.0) -> 禁止就绪
    too_wide_r, _ = step_val(98.0, edge_px=3.2)
    assert too_wide_r["ready"] is False
    assert "兜底上限 3" in too_wide_r["reason"]

    # 5. 出现新的更高平滑峰（连续 5 帧 120.0）-> peak_confirmed 重置，需再次越峰
    for _ in range(5):
        higher_r, higher_t = step_val(120.0, edge_px=1.5)
    assert higher_t["best"] == pytest.approx(120.0)
    assert higher_r["peak_confirmed"] is False
    assert higher_r["ready"] is False
    assert higher_r["stage_text"] == STAGE_ASCENDING_TEXT

    # 6. 再次越峰到 105.0 (87.5% <= 92%)，再回 118.0 (98.3% >= 95%) 连续 5 帧 -> 再次就绪
    for _ in range(5):
        step_val(105.0, edge_px=1.8)
    for _ in range(7):
        final_r, _ = step_val(118.0, edge_px=1.6)
    assert final_r["ready"] is True
    assert final_r["peak_confirmed"] is True


def test_peak_reset_margin_keeps_confirmation_on_small_gain_and_resets_on_large_gain():
    """越峰后回到峰值附近略超旧峰（1–2%）更新 best 且保留确认并能就绪；高出 >=3% 清除确认需重越峰。"""
    cfg = dict(DEFAULTS, peak_min_frames=10, focus_ready_frames=5, peak_reset_margin_percent=3.0)

    def feed(tracker, rtracker, val: float, edge_px: float = 1.6):
        tracked = tracker.update(val, (300.0, 450.0))
        readiness = rtracker.update(
            "front",
            {
                "detection_status": "精定位",
                "detection_reason": "",
                "tip_uv": [300.0, 450.0],
                "image_size": [600, 600],
                "percent": tracked["percent"],
                "peak": tracked["best"],
                "peak_confirmed": tracked["peak_confirmed"],
                "best_updated": tracked["best_updated"],
                "peak_confirm_kept": tracked["peak_confirm_kept"],
                "last_peak_event": tracked["last_peak_event"],
                "sample_count": tracked["sample_count"],
                "edge_width_px": edge_px,
            },
            cfg,
        )
        return readiness, tracked

    # 1. 建峰 100.0，越峰到 88.0（<=92%），再往回点动到 101.5（+1.5%）和 102.0（+2.0%）
    tracker = BandGradientTracker(smooth_frames=5, pass_percent=92.0, peak_reset_margin_percent=3.0)
    rtracker = ReadinessTracker()
    for _ in range(10):
        feed(tracker, rtracker, 100.0)
    for _ in range(5):
        r, t = feed(tracker, rtracker, 88.0, edge_px=2.2)
    assert r["peak_confirmed"] is True

    # 前两帧受窗口中旧 88 影响，第 3 帧中值跃升到 101.5（+1.5% < 3%）
    for _ in range(2):
        feed(tracker, rtracker, 101.5)
    r1, t1 = feed(tracker, rtracker, 101.5)
    assert t1["best"] == pytest.approx(101.5)
    assert t1["best_updated"] is True
    assert t1["peak_confirm_kept"] is True
    assert t1["peak_event"] == "kept"
    assert t1["peak_confirmed"] is True
    assert r1["peak_confirmed"] is True
    assert "保留越峰确认" in r1["peak_update_note"]

    # 继续微调到 102.0（累计 +2.0% < 3%）并稳定 5 帧 -> 仍保留确认且直接就绪
    r2 = None
    t2 = None
    for _ in range(7):
        r2, t2 = feed(tracker, rtracker, 102.0)
    assert t2["best"] == pytest.approx(102.0)
    assert t2["peak_confirmed"] is True
    assert r2["ready"] is True
    assert r2["stage_text"] == STAGE_READY_TEXT
    assert "保留越峰确认" in r2["peak_update_note"]

    # 2. 若继续爬升至 103.5（相对原已确认峰 100.0 高出 3.5% >= 3%，即使相对 102.0 仅 +1.47%）-> 清除确认，防止阶梯爬坡绕过门限
    for _ in range(3):
        r3, t3 = feed(tracker, rtracker, 103.5)
    assert t3["best"] == pytest.approx(103.5)
    assert t3["best_updated"] is True
    assert t3["peak_confirm_kept"] is False
    assert t3["peak_event"] == "reset"
    assert t3["peak_confirmed"] is False
    assert r3["ready"] is False
    assert r3["stage_text"] == STAGE_ASCENDING_TEXT
    assert "清除确认" in r3["peak_update_note"]

    # 3. 边界测试：直接高出恰好 3.0%（100.0 -> 越峰 88.0 -> 103.0）-> 清除确认，必须重新越峰后才能就绪
    tracker2 = BandGradientTracker(smooth_frames=5, pass_percent=92.0, peak_reset_margin_percent=3.0)
    rtracker2 = ReadinessTracker()
    for _ in range(10):
        feed(tracker2, rtracker2, 100.0)
    for _ in range(5):
        feed(tracker2, rtracker2, 88.0, edge_px=2.2)
    for _ in range(7):
        r_big, t_big = feed(tracker2, rtracker2, 103.0)
    assert t_big["best"] == pytest.approx(103.0)
    assert t_big["peak_confirmed"] is False
    assert r_big["ready"] is False
    assert t_big["last_peak_event"] == "reset"

    # 重新越峰到 94.0（91.26% <= 92%），再回到 102.0（99.0% >= 95%）连续 5 帧 -> 就绪
    for _ in range(5):
        r_over, _ = feed(tracker2, rtracker2, 94.0, edge_px=2.1)
    assert r_over["peak_confirmed"] is True
    for _ in range(7):
        r_done, _ = feed(tracker2, rtracker2, 102.0, edge_px=1.6)
    assert r_done["ready"] is True
    assert r_done["peak_confirmed"] is True
    assert r_done["peak_update_note"] == ""


def test_peak_min_frames_consecutive_edge_margin_and_reset():
    base = scene()
    blur = cv2.GaussianBlur(base, (0, 0), 1.4)
    detector = TridentTipDetector()
    sharp_det = detector.detect(base)
    blur_det = detector.detect(blur)
    assert sharp_det.status == "精定位" and sharp_det.edge_width_px < 2.0

    tracker = BandGradientTracker(smooth_frames=5, pass_percent=92.0)
    rtracker = ReadinessTracker()
    cfg = dict(DEFAULTS)

    # 1. 前 9 帧即使百分比 100%，因样本不足 peak_min_frames=10 且未越峰，不判就绪
    last_r = None
    for _ in range(9):
        status, _, metrics, tracked = analyze_tip(base, sharp_det, tracker, 13.5, 40)
        last_r = _feed_readiness(rtracker, tracked, status, sharp_det.diagnostic, sharp_det.point_uv, metrics["edge_width_px"], cfg)
        assert last_r["ready"] is False
    assert "峰值样本不足" in last_r["short_reason"]

    # 第 10 帧仅样本够了但未越峰 -> 提示尚未越过最清晰点
    status, _, metrics, tracked = analyze_tip(base, sharp_det, tracker, 13.5, 40)
    no_pass = _feed_readiness(rtracker, tracked, status, sharp_det.diagnostic, sharp_det.point_uv, metrics["edge_width_px"], cfg)
    assert no_pass["ready"] is False
    assert "尚未越过最清晰点" in no_pass["stage_text"]

    # 2. 越过峰值（连续几帧模糊图 <=92%），再回到峰值连续 5 帧 -> 就绪
    for _ in range(5):
        status, _, metrics, tracked = analyze_tip(blur, blur_det, tracker, 13.5, 40)
        pass_r = _feed_readiness(rtracker, tracked, status, blur_det.diagnostic, blur_det.point_uv, metrics["edge_width_px"], cfg)
    assert pass_r["peak_confirmed"] is True
    assert pass_r["ready"] is False
    for _ in range(2):
        status, _, metrics, tracked = analyze_tip(base, sharp_det, tracker, 13.5, 40)
        _feed_readiness(rtracker, tracked, status, sharp_det.diagnostic, sharp_det.point_uv, metrics["edge_width_px"], cfg)
    for _ in range(5):
        status, _, metrics, tracked = analyze_tip(base, sharp_det, tracker, 13.5, 40)
        last_r = _feed_readiness(rtracker, tracked, status, sharp_det.diagnostic, sharp_det.point_uv, metrics["edge_width_px"], cfg)
    assert last_r["ready"] is True
    assert last_r["consecutive_ready"] == 5

    # 3. 靠边（< edge_margin_px=40）立即打断就绪并给出移回提示
    edge_r = rtracker.update(
        "front",
        {
            "detection_status": "精定位",
            "detection_reason": "",
            "tip_uv": [25.0, 450.0],
            "image_size": [600, 600],
            "percent": 100.0,
            "peak": tracked["best"],
            "peak_confirmed": True,
            "sample_count": tracked["sample_count"],
            "edge_width_px": metrics["edge_width_px"],
        },
        cfg,
    )
    assert edge_r["ready"] is False
    assert edge_r["edge_near"] is True
    assert EDGE_MARGIN_MESSAGE in edge_r["reason"]

    # 4. 峰值清零后重新累计，不足 10 帧不判就绪
    tracker.reset()
    rtracker.reset()
    for _ in range(6):
        status, _, metrics, tracked = analyze_tip(base, sharp_det, tracker, 13.5, 40)
        after_reset = _feed_readiness(rtracker, tracked, status, sharp_det.diagnostic, sharp_det.point_uv, metrics["edge_width_px"], cfg)
    assert after_reset["sample_count"] == 6
    assert after_reset["peak_confirmed"] is False
    assert after_reset["ready"] is False


def test_simulated_y_front_x_side_sweep_offcenter_ready_and_save(tmp_path, monkeypatch):
    """正面清晰度随 Y、侧面随 X 变化；两路过峰后回峰值就绪；离中心远仍可建立并保存新格式基准。"""
    base = scene()
    cal = tmp_path / "cal"
    cal.mkdir()
    initialize_tools(cal / "tools.yaml")
    save_tools(cal / "tools.yaml", 1, [1.5, -0.5, 2.0], source="test")
    calib_path = cal / "camera.yaml"
    calib_path.write_text("camera: test\n", encoding="utf-8")

    # 写一个旧格式基准，确认新格式保存后旧读取逻辑兼容，且读取旧基准也不受影响
    old_baseline_path = cal / "tool2_baseline.yaml"
    old_baseline_path.write_text(
        yaml.safe_dump({
            "version": 1,
            "tool_name": "tool2",
            "p0": [0, 0, 0],
            "offset_xyz": [0, 0, 0],
            "views": [
                {"name": "front", "target_uv": [300, 452], "jacobian": None},
                {"name": "side", "target_uv": [300, 452], "jacobian": None},
            ],
        }),
        encoding="utf-8",
    )
    assert load_baseline(old_baseline_path).baseline_rule is None

    detection = {
        n: {"method": "trident", "direction": [0, 1], "threshold": 127, "min_area": 5, "max_axis_rms_px": 0.7}
        for n in ("front", "side")
    }
    cfg = SimpleNamespace(
        tool_calibration=SimpleNamespace(directory=str(cal), detection=detection, click_timeout_s=2, frame_std_max_px=0.3),
        tool_offset={"settle_samples": 3, "settle_window_s": 0.02, "jitter_tol_mm": 0.02},
        tools=((1.5, -0.5, 2.0), (0, 0, 0)),
    )

    plc = _AxisPlc((10.0, 20.0, 30.0))

    # 预渲染不同模糊度图像（正面针尖刻意位于 x=150，远离开中心 x=300）
    front_sharp = _render_needle(center_x=150, sigma=0.0)
    front_mid = _render_needle(center_x=150, sigma=0.95)
    side_sharp = _render_needle(center_x=320, sigma=0.0)
    side_mid = _render_needle(center_x=320, sigma=0.95)

    # 模拟点动过程：正面清晰度随 Y 在 y=20.0 达到峰值，侧面清晰度随 X 在 x=10.0 达到峰值
    class _SimCameras:
        def open(self):
            pass

        def on_stop(self):
            pass

        def _img(self, name):
            if name == "front":
                dy = abs(plc.xyz[1] - 20.0)
                return front_sharp if dy < 0.05 else front_mid
            dx = abs(plc.xyz[0] - 10.0)
            return side_sharp if dx < 0.05 else side_mid

        def execute(self, payload):
            images = CaptureImages()
            count = payload.get("frames_per_view", 1)
            for name in ("front", "side"):
                frame = self._img(name)
                images.frames[name] = [frame.copy() for _ in range(count)]
                images.capture_times[name] = [time.monotonic() for _ in range(count)]
                images[name] = images.frames[name][0]
            return images

    # 加快测试采样频率（不改变业务逻辑，只缩短轮询休眠）
    orig_wait = BaselineSession.run
    events = _Events(tmp_path)
    ui_cfg = {**DEFAULTS, "main_target_uv": [300, 452], "main_tol_mm": 0.05}
    session = BaselineSession(cfg, _SimCameras(), plc, events, calib_path, ui_cfg)
    preview_file = tooltip_preview_file(tmp_path)

    def read_preview():
        if not preview_file.exists():
            return {}
        return json.loads(preview_file.read_text(encoding="utf-8")).get("cameras", {})

    try:
        session.submit("baseline_select", {"tool": 1}, 1)
        _wait(lambda: session.phase == "live")

        # 阶段 A：两路都在峰值处累计 >=10 帧建立峰值，此时均未越峰，均不就绪
        _wait(lambda: (read_preview().get("front", {}).get("sample_count", 0) >= 10 and
                       read_preview().get("side", {}).get("sample_count", 0) >= 10), timeout=8.0)
        cams = read_preview()
        assert cams["front"]["ready"] is False and cams["front"]["peak_confirmed"] is False
        assert cams["side"]["ready"] is False and cams["side"]["peak_confirmed"] is False

        # 点动 Y 越过峰值（y=20.0 -> 20.5），正面降到 <=92% 确认峰值，随后回到 y=20.0 连续 5 帧就绪；侧面仍未越峰
        plc.xyz[1] = 20.5
        _wait(lambda: bool(read_preview().get("front", {}).get("peak_confirmed")), timeout=8.0)
        plc.xyz[1] = 20.0
        _wait(lambda: (read_preview().get("front", {}).get("ready", False)
                       and not read_preview().get("side", {}).get("ready", True)), timeout=8.0)
        cams = read_preview()
        overall = evaluate_baseline_readiness(cams["front"], cams["side"], config=ui_cfg)
        assert overall["ready"] is False
        assert "侧面" in overall["banner_text"] and "点动 X" in overall["banner_text"]
        assert "正面" not in overall["disabled_reason"]

        # 阶段 B：点动 X 越过峰值（x=10.0 -> 10.5）使侧面完成越峰确认，随后清零重新验证正面未就绪场景
        plc.xyz[0] = 10.5
        _wait(lambda: bool(read_preview().get("side", {}).get("peak_confirmed")), timeout=8.0)
        plc.xyz[0] = 10.0
        _wait(lambda: (read_preview().get("front", {}).get("ready", False)
                       and read_preview().get("side", {}).get("ready", False)), timeout=8.0)

        # 清零峰值后两路从头累计并只让侧面越峰回峰、让正面停在糊位检查正面提示
        session.submit("tooltip_peak_reset", {"camera": "all"}, 2)
        _wait(lambda: any(ev == "tooltip_peak_reset" and kw.get("camera") == "all" for ev, kw in events.rows))
        _wait(lambda: (read_preview().get("front", {}).get("sample_count", 99) < 10 and
                       not read_preview().get("side", {}).get("peak_confirmed", True)), timeout=8.0)
        plc.xyz = [10.0, 20.0, 30.0]
        _wait(lambda: (read_preview().get("front", {}).get("sample_count", 0) >= 10 and
                       read_preview().get("side", {}).get("sample_count", 0) >= 10), timeout=8.0)
        plc.xyz[0] = 10.5
        _wait(lambda: bool(read_preview().get("side", {}).get("peak_confirmed")), timeout=8.0)
        plc.xyz[0] = 10.0
        _wait(lambda: read_preview().get("side", {}).get("ready", False), timeout=8.0)
        plc.xyz[1] = 20.5
        _wait(lambda: bool(read_preview().get("front", {}).get("peak_confirmed")), timeout=8.0)
        _wait(lambda: (not read_preview().get("front", {}).get("ready", True)
                       and read_preview().get("side", {}).get("ready", False)), timeout=8.0)
        cams = read_preview()
        overall_front_blur = evaluate_baseline_readiness(cams["front"], cams["side"], config=ui_cfg)
        assert overall_front_blur["ready"] is False
        assert "正面" in overall_front_blur["banner_text"] and "点动 Y" in overall_front_blur["banner_text"]

        # 阶段 C：往回点动 Y 回到峰值 y=20.0，连续 5 帧后两路均就绪（且正面针尖离中心 150px ≈ 2.025mm！）
        plc.xyz[1] = 20.0
        _wait(lambda: (read_preview().get("front", {}).get("ready", False)
                       and read_preview().get("side", {}).get("ready", False)), timeout=8.0)
        cams = read_preview()
        overall_ready = evaluate_baseline_readiness(cams["front"], cams["side"], config=ui_cfg)
        assert overall_ready["ready"] is True
        assert overall_ready["banner_text"] == "✔ 两路针尖都最清晰，可作为对刀基准 → 点【对刀基准】"

        # 建立并保存基准（中心不再阻止！）
        session.submit("baseline_measure", {}, 3)
        _wait(lambda: session.phase == "success")
        assert session.candidate["inside"] is False  # 离中心约 2mm，远超 0.05mm，但照样成功
        assert abs(session.candidate["pixels"]["front"][0] - 150.0) < 2.0
        assert session.candidate["p0"] == pytest.approx([10.0, 20.0, 30.0])

        session.submit("baseline_save", {}, 4)
        _wait(lambda: session.phase == "confirm")
        session.submit("baseline_confirm", {"accept": True}, 5)
        _wait(lambda: session.phase == "saved")

        saved_raw = yaml.safe_load((cal / "tool1_baseline.yaml").read_text(encoding="utf-8"))
        assert saved_raw["baseline_rule"] == BASELINE_RULE
        assert saved_raw["focus_criteria"]["focus_smooth_frames"] == 5
        assert saved_raw["focus_criteria"]["focus_pass_percent"] == pytest.approx(92.0)
        assert saved_raw["focus_criteria"]["focus_ready_percent"] == pytest.approx(95.0)
        assert saved_raw["focus_criteria"]["peak_reset_margin_percent"] == pytest.approx(3.0)
        assert saved_raw["focus_criteria"]["focus_max_px"] == pytest.approx(3.0)
        assert saved_raw["focus_criteria"]["peak_min_frames"] == 10
        for v in saved_raw["views"]:
            assert v["focus_percent"] >= 95.0
            assert v["focus_peak"] > 0
            assert v["edge_width_px"] <= 3.0
            assert v["peak_confirmed"] is True

        # 检查 focus_trace.csv 与 baseline 记录在同目录且字段齐全
        trace_path = session.record_dir / "focus_trace.csv"
        assert trace_path.is_file()
        with trace_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) >= 20
        expected_cols = {
            "timestamp", "camera", "raw_sharpness", "smoothed_sharpness",
            "peak", "percent", "edge_width_px", "contrast",
            "axis_rms_px", "detection_status", "peak_confirmed", "peak_stage", "ready",
            "best_updated", "peak_confirm_kept", "peak_event",
        }
        assert set(rows[0].keys()) == expected_cols
        assert {r["camera"] for r in rows} == {"front", "side"}
        assert any(r["peak_confirmed"] == "0" for r in rows)
        assert any(r["peak_confirmed"] == "1" and r["ready"] == "1" for r in rows)
        assert any(r["best_updated"] == "1" for r in rows)

        loaded = load_baseline(cal / "tool1_baseline.yaml")
        assert loaded.baseline_rule == BASELINE_RULE
        assert loaded.views[0].target_uv[0] < 160.0
        assert loaded.views[0].peak_confirmed is True
    finally:
        session.end()
        session.thread.join(5)


def test_manual_view_confirm_participates_in_readiness_when_clarity_met(tmp_path, monkeypatch):
    """某路识别不准时人工点选，按“已人工确认”参与就绪判定，但清晰度条件仍必须满足。"""
    base = scene()
    det = TridentTipDetector()
    sharp_res = det.detect(base)
    coarse_blurry = replace(sharp_res, status="粗定位", diagnostic="针身轴拟合残差过大", axis_rms_px=0.72, sharpness={**sharp_res.sharpness, "edge_width_px": 3.5})
    coarse_sharp = replace(sharp_res, status="粗定位", diagnostic="针身轴拟合残差过大", axis_rms_px=0.72)

    tracker = BandGradientTracker(smooth_frames=3, pass_percent=92.0)
    rtracker = ReadinessTracker()
    cfg = dict(DEFAULTS, peak_min_frames=5, focus_ready_frames=3)

    # 先越峰确认峰值，以便隔离验证人工确认下 3.5px 兜底失败与清晰后成功
    overshoot_det = replace(
        coarse_blurry,
        sharpness={**sharp_res.sharpness, "band_gradient_p995": float(sharp_res.sharpness["band_gradient_p995"]) * 0.75, "edge_width_px": 2.5},
    )
    for _ in range(3):
        analyze_tip(base, sharp_res, tracker, 13.5, 40, manual_confirmed=True, manual_tip=(300.0, 452.0))
    for _ in range(3):
        analyze_tip(base, overshoot_det, tracker, 13.5, 40, manual_confirmed=True, manual_tip=(300.0, 452.0))
    for _ in range(2):
        analyze_tip(base, coarse_blurry, tracker, 13.5, 40, manual_confirmed=True, manual_tip=(300.0, 452.0))

    # 1. 即使已人工确认，若边缘宽度 3.5 > focus_max_px(3.0) 仍不就绪
    for _ in range(3):
        status, _, metrics, tracked = analyze_tip(base, coarse_blurry, tracker, 13.5, 40, manual_confirmed=True, manual_tip=(300.0, 452.0))
        r_blur = rtracker.update(
            "side",
            {
                "detection_status": status,
                "detection_reason": coarse_blurry.diagnostic,
                "tip_uv": [300.0, 452.0],
                "image_size": [600, 600],
                "percent": tracked["percent"],
                "peak": tracked["best"],
                "peak_confirmed": tracked["peak_confirmed"],
                "sample_count": tracked["sample_count"],
                "edge_width_px": metrics["edge_width_px"],
            },
            cfg,
            manual_confirmed=True,
        )
    assert r_blur["ready"] is False
    assert "边缘" in r_blur["reason"]

    # 2. 调清晰至边缘 <= focus_max_px 且百分比 >= 95% 后，连续达标即可就绪
    for _ in range(3):
        status, _, metrics, tracked = analyze_tip(base, coarse_sharp, tracker, 13.5, 40, manual_confirmed=True, manual_tip=(300.0, 452.0))
        r_ok = rtracker.update(
            "side",
            {
                "detection_status": status,
                "detection_reason": coarse_sharp.diagnostic,
                "tip_uv": [300.0, 452.0],
                "image_size": [600, 600],
                "percent": tracked["percent"],
                "peak": tracked["best"],
                "peak_confirmed": tracked["peak_confirmed"],
                "sample_count": tracked["sample_count"],
                "edge_width_px": metrics["edge_width_px"],
            },
            cfg,
            manual_confirmed=True,
        )
    assert r_ok["ready"] is True and r_ok["manual_confirmed"] is True


def test_edge_near_rejected_in_evaluate(tmp_path):
    """针尖距离画面边缘 < edge_margin_px 时 _evaluate 也拒绝建立。"""
    img = scene()
    session = object.__new__(BaselineSession)
    session.config = SimpleNamespace(
        tool_calibration=SimpleNamespace(
            detection={"front": {"method": "trident", "threshold": 127}, "side": {"method": "trident", "threshold": 127}}
        )
    )
    session.cfg = dict(DEFAULTS)
    session.measurement = {
        "views": {
            "front": {"detector": {"method": "trident", "version": "trident-3"}, "accepted": True, "frames": []},
            "side": {"detector": {"method": "trident", "version": "trident-3"}, "accepted": True, "frames": []},
        },
        "rejected": False,
        "reason": "",
    }
    session.events = _Events(tmp_path)
    session.candidate = None
    session.live_snapshots = {}
    session.manual_points = {}
    edge_front = TridentResult(point_uv=(20.0, 300.0), status="精定位", diagnostic="", axis_rms_px=0.2, sharpness={"edge_width_px": 1.4})
    ok_side = TridentResult(point_uv=(300.0, 300.0), status="精定位", diagnostic="", axis_rms_px=0.2, sharpness={"edge_width_px": 1.4})
    with pytest.raises(ValueError, match="针尖太靠近画面边缘"):
        session._evaluate({"front": img, "side": img}, {"front": edge_front, "side": ok_side})


def test_tool_change_100_110_uses_offcenter_baseline_target_uv(tmp_path):
    """换针对刀（100/110 + 40905=2）以基准保存的偏离中心 front 像素为目标，不隐含依赖视野中心。"""
    from nest_clear.devices.calib_cameras import CalibCamera
    from nest_clear.tooloffset.commands import send

    config = load_config(ROOT / "config/nest_clear.yaml")
    for settings in config.tool_calibration.detection.values():
        settings["method"] = "silhouette"
    config = replace(
        config,
        travel=((-100.0, 100.0),) * 3,
        tool_calibration=replace(config.tool_calibration, enabled=True, directory=str(tmp_path), tip_tolerance_mm=0.03),
        tools=((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)),
    )
    initialize_tools(tmp_path / "tools.yaml")
    p0 = (10.0, 20.0, 30.0)
    offcenter_front_uv = (140.0, 182.0)  # 刻意远离开中心 (320, 240)；CalibCamera 线宽 4 使剪影末端 +2px
    side_uv = (410.0, 222.0)
    j_front = ((20.0, 0.0, 0.0), (0.0, 0.0, 20.0))
    j_side = ((0.0, 20.0, 0.0), (0.0, 0.0, 20.0))
    baseline = ToolTipBaseline(
        "tool2",
        p0,
        [ViewCalib("front", offcenter_front_uv, j_front), ViewCalib("side", side_uv, j_side)],
        config.tools[1],
        baseline_rule=BASELINE_RULE,
    )
    save_baseline(baseline, tmp_path / "tool2_baseline.yaml", tool=2, source="test")

    events = _Events(tmp_path)
    ctx = SimpleNamespace(config=config, events=events)

    class Board:
        def __init__(self, **values):
            self.data = values

        def read(self, key, default=None):
            return self.data.get(key, default)

        def write(self, key, value, name):
            self.data[key] = value

    bb = Board(request_func=110, plc_tool_id=2, task_id=1, mode=Mode.IDLE, actual=p0)
    start = StartCalibration(ctx)
    start._blackboard = bb
    start.perform()
    bb.write("mode", Mode.CALIBRATING_2, "test")

    # 模拟新工具装夹偏移 (+0.25, -0.15, 0) mm
    offset = (0.25, -0.15, 0.0)
    cameras = {}
    for name, uv, jac in (("front", (140.0, 180.0), j_front), ("side", (410.0, 220.0), j_side)):
        cam = CalibCamera(
            name,
            {"backend": "mock", "target_uv": list(uv), "p0": p0, "jacobian": jac, "tip_offset": offset},
            lambda: bb.read("actual"),
        )
        cam.open()
        cameras[name] = cam

    def step_once():
        images = CaptureImages()
        for name, camera in cameras.items():
            images.frames[name] = [camera.capture() for _ in range(5)]
            images[name] = images.frames[name][0]
        bb.write("tooltip_images", images, "test")
        bb.read("tooltip_session")["images"].append(
            {"raw": {}, "overlay": {}, "iteration": len(bb.read("tooltip_session")["images"]) + 1}
        )
        det = DetectTips(ctx)
        det._blackboard = bb
        assert det.on_start() == Status.SUCCESS, bb.read("error")
        confirm = ConfirmCalibration(ctx)
        confirm._blackboard = bb
        assert confirm.on_start() == Status.RUNNING, bb.read("error")
        return confirm

    node = step_once()
    assert not node.aligned
    send(tmp_path, "tooltip_confirm_move")
    assert node.on_running() == Status.SUCCESS
    assert bb.read("reply_func") == 110
    target = bb.read("reply_coordinates")["calibration2_xyz"]
    bb.write("actual", target, "test")

    node2 = step_once()
    assert node2.aligned
    send(tmp_path, "tooltip_confirm_result")
    assert node2.on_running() == Status.SUCCESS
    assert bb.read("reply_func") == 121
    bb.write("tooltip_final", bb.read("actual"), "test")
    finish = SaveCompensation(ctx)
    finish._blackboard = bb
    finish.perform()
    assert bb.read("reply_func") == 130
    saved_offset = yaml.safe_load((tmp_path / "tools.yaml").read_text(encoding="utf-8"))["tools"]["tool2"]["offset_xyz"]
    assert saved_offset == pytest.approx([4.0 - 0.25, 5.0 + 0.15, 6.0], abs=0.06)


def test_ui_overall_banner_and_disabled_reason_without_cv_numpy_yaml(tmp_path, monkeypatch):
    """UI 屏蔽 cv2/numpy/yaml，校验顶部总状态、单路状态条、置灰原因、灰色中心参考及全清零。"""
    for module in ("cv2", "numpy", "yaml"):
        monkeypatch.setitem(sys.modules, module, None)
    from nest_clear.ui import app

    for name in ("Tk", "Frame", "Label", "Canvas", "Toplevel", "Text", "Button"):
        monkeypatch.setattr(app.tk, name, Widget)
    for name in ("Notebook", "Treeview", "Progressbar", "Scrollbar", "Button", "Entry", "Combobox", "Spinbox"):
        monkeypatch.setattr(app.ttk, name, Widget)
    monkeypatch.setattr(app.ttk, "Style", Style)
    monkeypatch.setattr(app.tkfont, "families", lambda root: ["Noto Sans Mono CJK SC"])

    class FakeCard(Widget):
        def __init__(self, parent, title, theme):
            super().__init__(parent)
            self.body = Widget(self)
            self.title_label = Widget(self)

    class FakePicture(Widget):
        def __init__(self, parent, path, theme):
            super().__init__(parent)
            self.image = None

        def poll(self):
            pass

    class FakeKeyValueGrid(Widget):
        def __init__(self, parent, keys, theme):
            super().__init__(parent)
            self.values_data = {}

        def set(self, values):
            self.values_data = values

    monkeypatch.setattr(app, "Card", FakeCard)
    monkeypatch.setattr(app, "Picture", FakePicture)
    monkeypatch.setattr(app, "KeyValueGrid", FakeKeyValueGrid)

    now_iso = datetime.now().astimezone().isoformat()
    preview_path = tooltip_preview_file(tmp_path)
    preview_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. 正面 2.4px（黄色区间但相对峰值已越峰达标）仍就绪，侧面未越峰且模糊 -> 按钮置灰 + 写明侧面点动 X + 尚未越过最清晰点
    preview_path.write_text(json.dumps({
        "updated_at": now_iso,
        "cameras": {
            "front": {
                "source": "band_gradient_p995", "band_gradient": 154.0, "peak": 156.0, "percent": 98.7,
                "edge_width_px": 2.40, "edge_width_um": 32.4, "contrast": 220, "axis_rms_px": 0.30,
                "axis_rms_limit": 0.7, "detection_status": "精定位", "tip_uv": [300.0, 500.0],
                "image_size": [1920, 1080], "sample_count": 15, "peak_confirmed": True, "consecutive_ready": 5, "ready": True,
                "history": [82.0, 91.0, 98.7], "updated_at": now_iso,
            },
            "side": {
                "source": "band_gradient_p995", "band_gradient": 96.0, "peak": 150.0, "percent": 64.0,
                "edge_width_px": 3.20, "edge_width_um": 43.2, "contrast": 120, "axis_rms_px": 0.55,
                "axis_rms_limit": 0.7, "detection_status": "精定位", "tip_uv": [960.0, 500.0],
                "image_size": [1920, 1080], "sample_count": 15, "peak_confirmed": False, "consecutive_ready": 0, "ready": False,
                "history": [45.0, 55.0, 64.0], "updated_at": now_iso,
            },
        },
    }), encoding="utf-8")

    ui = app.App(tmp_path, page="对刀基准建立", ignore_before_ns=0)
    ui.update_baseline({
        "mode": "对刀基准建立中",
        "command_seq": 0,
        "baseline": {
            "phase": "live",
            "tool": 1,
            "live_delta": [-3.0, 1.5],
            "live_inside": False,
            "preview": {"front": {"on": False}, "side": {"on": False}},
        },
    })
    assert ui.baseline_measure.options["state"] == "disabled"
    assert "侧面" in ui.baseline_measure_reason.options["text"] and "点动 X" in ui.baseline_measure_reason.options["text"]
    assert "✖ 还需对准" in ui.baseline_overall.options["text"]
    assert "64%" in ui.baseline_overall.options["text"]
    assert ui.baseline_live.options["fg"] == app.MUTED
    # 状态条保持黄色“⚠ 稍虚”颜色逻辑，同时一致显示“✔ 最清晰”与“该路最佳边缘 2.4px 偏宽，可接受”
    assert ui.baseline_views["front"]["focus_badge"].options["text"] == "⚠ 稍虚"
    assert STAGE_READY_TEXT in ui.baseline_views["front"]["focus_summary"].options["text"]
    assert "该路最佳边缘 2.4px 偏宽，可接受" in ui.baseline_views["front"]["focus_summary"].options["text"]
    assert "尚未越过最清晰点" in ui.baseline_views["side"]["hint"].options["text"]
    assert ui.baseline_peak_reset.options["state"] == "normal"

    # 2. 两路都最清晰（即使正面严重偏离中心 [-3.0, 1.5] mm）-> 顶部变绿 ✔，按钮可点
    preview_path.write_text(json.dumps({
        "updated_at": now_iso,
        "cameras": {
            "front": {
                "source": "band_gradient_p995", "band_gradient": 154.0, "peak": 156.0, "percent": 98.7,
                "edge_width_px": 1.42, "edge_width_um": 19.2, "contrast": 220, "axis_rms_px": 0.30,
                "axis_rms_limit": 0.7, "detection_status": "精定位", "tip_uv": [300.0, 500.0],
                "image_size": [1920, 1080], "sample_count": 15, "peak_confirmed": True, "consecutive_ready": 5, "ready": True,
                "history": [82.0, 91.0, 98.7], "updated_at": now_iso,
            },
            "side": {
                "source": "band_gradient_p995", "band_gradient": 147.0, "peak": 150.0, "percent": 98.0,
                "edge_width_px": 1.55, "edge_width_um": 20.9, "contrast": 210, "axis_rms_px": 0.28,
                "axis_rms_limit": 0.7, "detection_status": "精定位", "tip_uv": [960.0, 500.0],
                "image_size": [1920, 1080], "sample_count": 15, "peak_confirmed": True, "consecutive_ready": 5, "ready": True,
                "history": [80.0, 92.0, 98.0], "updated_at": now_iso,
            },
        },
    }), encoding="utf-8")
    ui.update_baseline({
        "mode": "对刀基准建立中",
        "command_seq": 0,
        "baseline": {
            "phase": "live",
            "tool": 1,
            "live_delta": [-3.0, 1.5],
            "live_inside": False,
            "preview": {"front": {"on": False}, "side": {"on": False}},
        },
    })
    assert ui.baseline_measure.options["state"] == "normal"
    assert ui.baseline_measure_reason.options["text"] == ""
    assert ui.baseline_overall.options["text"] == "✔ 两路针尖都最清晰，可作为对刀基准 → 点【对刀基准】"

    # 3. 界面显示“峰值略升保留确认”与“发现更高新峰（≥3%）清除确认”状态
    preview_path.write_text(json.dumps({
        "updated_at": now_iso,
        "cameras": {
            "front": {
                "source": "band_gradient_p995", "band_gradient": 159.1, "peak": 159.1, "percent": 100.0,
                "edge_width_px": 1.40, "edge_width_um": 18.9, "contrast": 225, "axis_rms_px": 0.28,
                "axis_rms_limit": 0.7, "detection_status": "精定位", "tip_uv": [300.0, 500.0],
                "image_size": [1920, 1080], "sample_count": 20, "peak_confirmed": True,
                "best_updated": True, "peak_confirm_kept": True, "last_peak_event": "kept",
                "consecutive_ready": 5, "ready": True, "history": [90.0, 96.0, 100.0], "updated_at": now_iso,
            },
            "side": {
                "source": "band_gradient_p995", "band_gradient": 160.0, "peak": 160.0, "percent": 100.0,
                "edge_width_px": 1.45, "edge_width_um": 19.6, "contrast": 215, "axis_rms_px": 0.30,
                "axis_rms_limit": 0.7, "detection_status": "精定位", "tip_uv": [960.0, 500.0],
                "image_size": [1920, 1080], "sample_count": 22, "peak_confirmed": False,
                "best_updated": True, "peak_confirm_kept": False, "last_peak_event": "reset",
                "consecutive_ready": 0, "ready": False, "history": [90.0, 96.0, 100.0], "updated_at": now_iso,
            },
        },
    }), encoding="utf-8")
    ui.update_baseline({
        "mode": "对刀基准建立中",
        "command_seq": 0,
        "baseline": {
            "phase": "live",
            "tool": 1,
            "live_delta": [0.0, 0.0],
            "live_inside": True,
            "preview": {"front": {"on": False}, "side": {"on": False}},
        },
    })
    assert "峰值略升已更新，保留越峰确认" in ui.baseline_views["front"]["focus_summary"].options["text"]
    assert "发现更高新峰" in ui.baseline_views["side"]["focus_summary"].options["text"]
    assert "已清除确认需重越峰" in ui.baseline_views["side"]["hint"].options["text"]
    assert ui.baseline_measure.options["state"] == "disabled"


def test_session_focus_trace_records_peak_kept_and_reset(tmp_path, monkeypatch):
    """会话实时链路在回到峰值附近高 1.5% 时保留确认并写 focus_trace.csv；高 >=3% 时重置确认。"""
    base_det = TridentTipDetector().detect(scene())
    assert base_det.status == "精定位"
    img = scene()
    session = object.__new__(BaselineSession)
    session.config = SimpleNamespace(
        tool_calibration=SimpleNamespace(detection={"front": {"method": "trident", "threshold": 127}, "side": {"method": "trident", "threshold": 127}})
    )
    session.cfg = dict(DEFAULTS, peak_min_frames=8, focus_ready_frames=3, focus_smooth_frames=3)
    session.events = _Events(tmp_path)
    session.record_dir = tmp_path / "run" / "baseline" / "trace_test"
    session.trackers = {n: BandGradientTracker(smooth_frames=3, pass_percent=92.0, peak_reset_margin_percent=3.0) for n in ("front", "side")}
    session.readiness_trackers = {n: ReadinessTracker() for n in ("front", "side")}
    session.history = {"front": [], "side": []}
    session.manual_points = {}
    session.live_snapshots = {}
    session._live_last_ts = None
    session._read_exposure_ms = lambda name: 10.0
    session._target = lambda image: (300.0, 300.0)

    def push_sharpness(front_val: float, side_val: float):
        dets = {
            "front": replace(base_det, sharpness={**base_det.sharpness, "band_gradient_p995": front_val, "edge_width_px": 1.5}),
            "side": replace(base_det, sharpness={**base_det.sharpness, "band_gradient_p995": side_val, "edge_width_px": 1.5}),
        }
        now = time.monotonic()
        session._publish_live_focus({"front": img, "side": img}, dets, now, now)

    # 建峰 100 -> 越峰 88 -> 正面回 101.5 (+1.5% 保留确认并就绪)，侧面回 104.0 (+4.0% 重置确认)
    for _ in range(8):
        push_sharpness(100.0, 100.0)
    for _ in range(3):
        push_sharpness(88.0, 88.0)
    assert session.live_snapshots["front"]["peak_confirmed"] is True
    assert session.live_snapshots["side"]["peak_confirmed"] is True

    for _ in range(5):
        push_sharpness(101.5, 104.0)

    assert session.live_snapshots["front"]["peak"] == pytest.approx(101.5)
    assert session.live_snapshots["front"]["peak_confirmed"] is True
    assert session.live_snapshots["front"]["ready"] is True
    assert "保留越峰确认" in session.live_snapshots["front"]["peak_update_note"]

    assert session.live_snapshots["side"]["peak"] == pytest.approx(104.0)
    assert session.live_snapshots["side"]["peak_confirmed"] is False
    assert session.live_snapshots["side"]["ready"] is False
    assert "清除确认" in session.live_snapshots["side"]["peak_update_note"]

    trace_path = session.record_dir / "focus_trace.csv"
    with trace_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    front_kept = [r for r in rows if r["camera"] == "front" and r["peak_event"] == "kept"]
    side_reset = [r for r in rows if r["camera"] == "side" and r["peak_event"] == "reset"]
    assert len(front_kept) == 1
    assert front_kept[0]["best_updated"] == "1"
    assert front_kept[0]["peak_confirm_kept"] == "1"
    assert front_kept[0]["peak_confirmed"] == "1"
    assert len(side_reset) == 1
    assert side_reset[0]["best_updated"] == "1"
    assert side_reset[0]["peak_confirm_kept"] == "0"
    assert side_reset[0]["peak_confirmed"] == "0"
