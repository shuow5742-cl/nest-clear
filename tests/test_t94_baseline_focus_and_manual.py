"""T94：对刀基准实时对焦状态显示、粗定位边缘测量、人工确认放行与元信息序列化回归。"""
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

from nest_clear.baseline.preparation import Preparation, analyze_tip
from nest_clear.baseline.session import BaselineSession, DEFAULTS, settings as baseline_settings, summary as baseline_summary
from nest_clear.tooltip.click import ClickResult
from nest_clear.tooltip.focus_metrics import BandGradientTracker
from nest_clear.tooltip.store import initialize_tools, load_baseline, save_tools
from nest_clear.tooltip.trident import TridentResult, TridentTipDetector
from nest_clear.workers.calib_camera import CaptureImages
from test_trident import scene
from test_ui_headless import Style, Widget


def _blurry_side_scene():
    """构造边缘宽度约 3.5px、axis_rms 约 0.72px 的侧面虚焦合成三棱针图。"""
    img = scene()
    # 在针身左侧局部制造约 3.2px 的台阶过渡，使直线拟合残差稳定落在 ~0.72px 附近
    img[220:360, 285:289] = 235
    blurred = cv2.GaussianBlur(img, (0, 0), 1.35)
    return blurred


class _Events:
    def __init__(self, root: Path):
        self.directory = root
        self.run_directory = root / "run"
        self.rows = []

    def clock(self):
        return datetime.now().astimezone()

    def emit(self, event, **kw):
        self.rows.append((event, kw))


class _MemoryPlc:
    def execute(self, payload):
        assert payload == {"operation": "actual"}
        return [1.0, 2.0, 3.0]


def _wait(predicate, timeout=6.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("等待条件超时")


def test_settings_defaults_when_focus_thresholds_omitted(tmp_path):
    cfg_path = tmp_path / "tooltip_baseline.yaml"
    cfg_path.write_text("baseline_ui:\n  main_tol_mm: 0.1\n", encoding="utf-8")
    loaded = baseline_settings(cfg_path)
    assert loaded["focus_good_px"] == pytest.approx(2.0)
    assert loaded["focus_warn_px"] == pytest.approx(3.0)

    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text("baseline_ui:\n  focus_good_px: 3.5\n  focus_warn_px: 2.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="focus_warn_px"):
        baseline_settings(bad_path)


def test_trident_coarse_frame_reports_edge_width_without_entering_peak_tracker():
    front_img = scene()
    side_img = _blurry_side_scene()
    detector = TridentTipDetector()
    front_det = detector.detect(front_img)
    side_det = detector.detect(side_img)
    assert front_det.status == "精定位"
    assert side_det.status == "粗定位"
    assert side_det.edge_width_px is not None and side_det.edge_width_px > 2.5

    tracker = BandGradientTracker()
    status, reason, metrics, tracked = analyze_tip(side_img, side_det, tracker, 13.5, 40)
    assert status == "粗定位" and reason
    assert metrics["edge_width_px"] == pytest.approx(side_det.edge_width_px)
    assert metrics["edge_width_um"] == pytest.approx(side_det.edge_width_px * 13.5)
    assert metrics["contrast"] == pytest.approx(side_det.tip_contrast)
    assert metrics["axis_rms_px"] == pytest.approx(side_det.axis_rms_px)
    assert tracked["best"] is None and tracked["percent"] is None


def test_ui_model_and_headless_display_live_focus_when_preview_off(tmp_path, monkeypatch):
    for module in ("cv2", "numpy", "yaml"):
        monkeypatch.setitem(sys.modules, module, None)
    from nest_clear.paths import tooltip_preview_file
    from nest_clear.ui import app
    from nest_clear.ui.model import baseline_focus_status, baseline_preview_display, read_tooltip_preview

    # 1. 纯模型分类阈值校验：绿 <= 2.0，黄 <= 3.0，红 > 3.0，未检出灰
    assert baseline_focus_status(None)["color_key"] == "gray"
    assert baseline_focus_status({"detection_status": "未识别", "edge_width_px": 1.5})["badge"] == "未检出"
    good = baseline_focus_status({"detection_status": "精定位", "edge_width_px": 1.42, "edge_width_um": 19.2, "contrast": 223, "axis_rms_px": 0.30})
    assert (good["color_key"], good["badge"]) == ("green", "✔ 已对焦")
    assert "1.42 px（19.2 µm）" in good["summary"] and "门限 0.7" in good["summary"] and "精定位" in good["summary"]
    warn = baseline_focus_status({"detection_status": "精定位", "edge_width_px": 2.50, "edge_width_um": 33.8, "contrast": 160, "axis_rms_px": 0.50})
    assert (warn["color_key"], warn["badge"]) == ("yellow", "⚠ 稍虚")
    bad = baseline_focus_status({"detection_status": "粗定位", "edge_width_px": 3.51, "edge_width_um": 47.4, "contrast": 113, "axis_rms_px": 0.72})
    assert (bad["color_key"], bad["badge"]) == ("red", "✖ 未对焦")
    assert "3.51 px（47.4 µm）" in bad["summary"] and "0.72 px（门限 0.7）" in bad["summary"] and "粗定位" in bad["summary"]

    # 2. 会话中 preview.on=False 时，baseline_preview_display(session_live=True) 仍展示实时读数
    off_item = {"on": False}
    assert baseline_preview_display(off_item, live={"edge_width_px": 1.42})["on"] is False
    assert baseline_preview_display(off_item, live={"edge_width_px": 1.42})["metrics"] is None
    in_session = baseline_preview_display(off_item, live={"edge_width_px": 1.42, "history": [90, 96]}, session_live=True)
    assert in_session["no_live_data"] is False
    assert in_session["live"]["edge_width_px"] == 1.42

    # 3. 无头 Tk 界面校验（已屏蔽 cv2/numpy/yaml）
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
    preview_path.write_text(json.dumps({
        "updated_at": now_iso,
        "cameras": {
            "front": {
                "source": "band_gradient_p995", "band_gradient": 154.5, "peak": 158.0, "percent": 97.8,
                "edge_width_px": 1.42, "edge_width_um": 19.2, "contrast": 223, "axis_rms_px": 0.30,
                "axis_rms_limit": 0.7, "detection_status": "精定位", "tip_uv": [300.0, 452.0],
                "delta_uv": [0.0, 0.0], "history": [82.0, 91.0, 97.8], "updated_at": now_iso,
            },
            "side": {
                "source": "band_gradient_p995", "band_gradient": None, "peak": None, "percent": None,
                "edge_width_px": 3.51, "edge_width_um": 47.4, "contrast": 113, "axis_rms_px": 0.72,
                "axis_rms_limit": 0.7, "detection_status": "粗定位", "tip_uv": [300.0, 452.0],
                "delta_uv": None, "history": [], "updated_at": now_iso,
            },
        },
    }), encoding="utf-8")

    assert set(read_tooltip_preview(tmp_path)) == {"front", "side"}
    ui = app.App(tmp_path, page="对刀基准建立", ignore_before_ns=0)
    ui.update_baseline({
        "mode": "对刀基准建立中",
        "command_seq": 0,
        "baseline": {
            "phase": "live",
            "tool": 2,
            "preview": {"front": {"on": False}, "side": {"on": False}},
            "config": {"focus_good_px": 2.0, "focus_warn_px": 3.0},
        },
    })
    front_v = ui.baseline_views["front"]
    side_v = ui.baseline_views["side"]
    assert front_v["focus_badge"].options["text"] == "✔ 已对焦"
    assert "1.42 px（19.2 µm）" in front_v["focus_summary"].options["text"]
    assert "清晰度 98%（相对本次峰值）" in front_v["info"].options["text"]
    assert front_v["reset"].options["state"] == "normal"

    assert side_v["focus_badge"].options["text"] == "✖ 未对焦"
    assert "3.51 px（47.4 µm）" in side_v["focus_summary"].options["text"]
    assert "点动 X" in side_v["info"].options["text"]
    assert "越过最清晰点后会下降" in side_v["hint"].options["text"]
    assert side_v["reset"].options["state"] == "normal"


def test_session_publishes_live_focus_and_manual_confirm_saves_override(tmp_path, monkeypatch):
    front_sharp = scene()
    side_blurry = _blurry_side_scene()
    cal = tmp_path / "cal"
    cal.mkdir()
    initialize_tools(cal / "tools.yaml")
    save_tools(cal / "tools.yaml", 2, [1.0, 2.0, 3.0], source="test")
    calib_path = cal / "camera.yaml"
    calib_path.write_text("camera: test\n", encoding="utf-8")

    detection = {n: {"method": "trident", "direction": [0, 1], "threshold": 127, "min_area": 5, "max_axis_rms_px": 0.7}
                 for n in ("front", "side")}
    cfg = SimpleNamespace(
        tool_calibration=SimpleNamespace(directory=str(cal), detection=detection, click_timeout_s=2, frame_std_max_px=0.3),
        tool_offset={"settle_samples": 3, "settle_window_s": 0.03, "jitter_tol_mm": 0.02},
        tools=((0, 0, 0), (1, 2, 3)),
    )

    class _Cameras:
        def __init__(self):
            self.opened = False

        def open(self):
            self.opened = True

        def on_stop(self):
            self.opened = False

        def execute(self, payload):
            images = CaptureImages()
            count = payload.get("frames_per_view", 1)
            for name, frame in (("front", front_sharp), ("side", side_blurry)):
                images.frames[name] = [frame.copy() for _ in range(count)]
                images.capture_times[name] = [time.monotonic() for _ in range(count)]
                images[name] = images.frames[name][0]
            return images

    events = _Events(tmp_path)
    ui_cfg = {**DEFAULTS, "main_target_uv": [300, 452]}
    session = BaselineSession(
        cfg,
        _Cameras(),
        _MemoryPlc(),
        events,
        calib_path,
        ui_cfg,
        click=lambda image, point, title: ClickResult(tuple(point) if point else (300.0, 452.0), "manual", 8.0, 1.0),
    )

    # 强制侧面检测结果精确对齐典型失败现场值：edge≈3.5px, axis_rms=0.72px, 粗定位
    orig_detect = TridentTipDetector.detect

    def controlled_detect(self, gray):
        res = orig_detect(self, gray)
        if np.array_equal(gray, side_blurry):
            sharpness = dict(res.sharpness or {}, edge_width_px=3.5)
            return replace(
                res,
                status="粗定位",
                diagnostic="针身轴拟合残差过大",
                axis_rms_px=0.72,
                sharpness=sharpness,
            )
        return res

    monkeypatch.setattr(TridentTipDetector, "detect", controlled_detect)

    try:
        session.submit("baseline_select", {"tool": 2}, 1)
        _wait(lambda: session.phase == "live")

        from nest_clear.paths import tooltip_preview_file
        preview_file = tooltip_preview_file(tmp_path)
        _wait(lambda: preview_file.exists() and (json.loads(preview_file.read_text()).get("cameras", {}).get("front", {}).get("percent") is not None))
        doc = json.loads(preview_file.read_text(encoding="utf-8"))
        assert doc["cameras"]["front"]["detection_status"] == "精定位"
        assert doc["cameras"]["front"]["edge_width_px"] < 2.0
        assert doc["cameras"]["front"]["percent"] == pytest.approx(100.0)
        assert doc["cameras"]["side"]["detection_status"] == "粗定位"
        assert doc["cameras"]["side"]["edge_width_px"] == pytest.approx(3.5)
        assert doc["cameras"]["side"]["axis_rms_px"] == pytest.approx(0.72)
        assert doc["cameras"]["side"]["percent"] is None

        # 实时标注图左上角判据统一：侧面 3.5px 标“偏糊”而非“清晰”
        assert (tmp_path / "live" / "baseline_side.jpg").exists()

        # 会话内峰值清零
        session.submit("tooltip_peak_reset", {"camera": "front"}, 2)
        _wait(lambda: any(ev == "tooltip_peak_reset" for ev, _ in events.rows))

        # baseline_measure 保持 0.7 门限不变并输出可操作中文提示
        session.submit("baseline_measure", {}, 3)
        _wait(lambda: session.phase == "fail")
        assert "侧面针身轴拟合残差 0.72px > 0.7（侧面边缘宽 3.5px，偏糊）→ 调侧面相机对焦至绿色后重测，或点【识别不准，手动确认】" == session.message

        # baseline_manual 人工确认不被单帧 axis_rms 0.72 硬拦截，允许保存并记录 manual_override
        session.submit("baseline_manual", {}, 4)
        _wait(lambda: session.phase == "success")
        assert session.candidate["manual_override"] is True
        side_focus = session.candidate["focus"]["side"]
        assert side_focus["manual_override"] is True
        assert side_focus["axis_rms_px"] == pytest.approx(0.72)
        assert side_focus["edge_width_px"] == pytest.approx(3.5)
        assert side_focus["override_note"] == "人工确认（轴残差 0.72>0.7）"
        assert session.candidate["focus"]["front"]["manual_override"] is False

        session.submit("baseline_save", {}, 5)
        _wait(lambda: session.phase == "confirm")
        session.submit("baseline_confirm", {"accept": True}, 6)
        _wait(lambda: session.phase == "saved")

        saved_path = cal / "tool2_baseline.yaml"
        raw = yaml.safe_load(saved_path.read_text(encoding="utf-8"))
        assert raw["manual_override"] is True
        assert "人工确认（轴残差 0.72>0.7）" in raw["note"]
        side_view_raw = next(v for v in raw["views"] if v["name"] == "side")
        assert side_view_raw["manual_override"] is True
        assert side_view_raw["axis_rms_px"] == pytest.approx(0.72)
        assert side_view_raw["edge_width_px"] == pytest.approx(3.5)
        assert side_view_raw["override_note"] == "人工确认（轴残差 0.72>0.7）"

        loaded = load_baseline(saved_path)
        assert loaded.manual_override is True
        assert loaded.views[1].override_note == "人工确认（轴残差 0.72>0.7）"
        info = baseline_summary(saved_path)
        assert info["manual_override"] is True
        assert info["override_notes"] == ["人工确认（轴残差 0.72>0.7）"]
    finally:
        session.end()
        session.thread.join(5)


def test_front_unfocused_without_manual_confirm_still_rejected(tmp_path):
    img = scene()
    session = object.__new__(BaselineSession)
    session.config = SimpleNamespace(
        tool_calibration=SimpleNamespace(
            detection={
                "front": {"method": "trident", "threshold": 127},
                "side": {"method": "trident", "threshold": 127},
            }
        )
    )
    session.cfg = dict(DEFAULTS, main_target_uv=[300, 452])
    session.measurement = {
        "views": {
            "front": {"detector": {"method": "trident", "version": "trident-3"}, "accepted": False, "frames": []},
            "side": {"detector": {"method": "trident", "version": "trident-3"}, "accepted": False, "frames": []},
        },
        "rejected": True,
        "reason": "正面未通过",
    }
    session.events = _Events(tmp_path)
    session.candidate = None
    front_coarse = TridentResult(point_uv=(300.0, 452.0), status="粗定位", diagnostic="针身轴拟合残差过大", axis_rms_px=0.75, sharpness={"edge_width_px": 3.5})
    side_coarse = TridentResult(point_uv=(300.0, 452.0), status="粗定位", diagnostic="针身轴拟合残差过大", axis_rms_px=0.72, sharpness={"edge_width_px": 3.5})
    with pytest.raises(ValueError, match="主对刀相机识别不可靠"):
        session._evaluate(
            {"front": img, "side": img},
            {"front": front_coarse, "side": side_coarse},
            manual={"side": ClickResult((300.0, 452.0), "manual", 8.0, 1.0)},
        )


def test_annotate_uses_edge_width_focus_threshold(tmp_path, monkeypatch):
    from nest_clear.baseline import session as session_mod

    captured_labels = []
    orig_draw = session_mod._draw_labels

    def spy_draw(image_bgr, items):
        captured_labels.extend(item[1] for item in items)
        return orig_draw(image_bgr, items)

    monkeypatch.setattr(session_mod, "_draw_labels", spy_draw)
    obj = object.__new__(BaselineSession)
    obj.config = SimpleNamespace(tool_calibration=SimpleNamespace(detection={"front": {"threshold": 127}, "side": {"threshold": 127}}))
    obj.events = SimpleNamespace(directory=tmp_path)
    obj.cfg = dict(DEFAULTS, main_target_uv=[300, 452])
    obj.candidate = None

    img = scene()
    front_ok = TridentResult(point_uv=(300.0, 452.0), status="精定位", diagnostic="", axis_rms_px=0.3, sharpness={"edge_width_px": 1.42})
    side_blur = TridentResult(point_uv=(300.0, 452.0), status="精定位", diagnostic="", axis_rms_px=0.69, sharpness={"edge_width_px": 3.43})
    obj._annotate({"front": img, "side": img}, {"front": front_ok, "side": side_blur})
    assert any("边缘 1.42px  清晰" in text for text in captured_labels)
    assert any("边缘 3.43px  偏糊" in text for text in captured_labels)
    assert not any("3.43px  清晰" in text for text in captured_labels)
