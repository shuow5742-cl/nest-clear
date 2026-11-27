"""界面驱动的对刀基准会话；设备采图及操作均在独立线程执行。"""

from dataclasses import replace
from datetime import datetime
import copy
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import threading
import time
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

from nest_clear.guard import GuardError
from nest_clear.paths import live_image_file
from nest_clear.tooloffset.store import _read, save_mapping
from nest_clear.tooltip import SilhouetteTipDetector, ToolTipBaseline, ViewCalib
from nest_clear.tooltip.factory import baseline_detector_warning, make_tip_detector
from nest_clear.baseline.preparation import (
    ReadinessTracker,
    analyze_tip,
    build_preview_camera_payload,
    make_tip_tracker,
    write_tooltip_preview_file,
)
from nest_clear.baseline.focus_rules import (
    BASELINE_RULE,
    CAMERA_AXES,
    EDGE_MARGIN_MESSAGE,
    FOCUS_DEFAULTS,
    evaluate_baseline_readiness,
    evaluate_camera_readiness,
    is_near_edge,
)
from nest_clear.baseline.runout import FRAME_COUNT, FRAME_QUALITY_PASSED, measure_frames
from nest_clear.tooltip.trident import TridentResult
from nest_clear.tooltip.click import ClickResult, confirm_tip
from nest_clear.tooltip.schema import load_baseline
from nest_clear.tooltip.store import load_tools

DEFAULTS: dict[str, Any] = {
    "main_target_uv": None,
    "main_tol_mm": 0.05,
    # 11-25 实测：Y 每 0.04mm 侧面平移 2.95px；原估算 15.2。
    "main_um_per_px": 13.5,
    "aux_um_per_px": 13.5,
    "min_focus_contrast": 40,
    **FOCUS_DEFAULTS,
    "runout_max_mm": 0.05,
    # 11-25 评测：静止组中位 0.05px；个别帧最远齿选择翻转可达 1.6px
    "runout_frame_std_max_px": 0.3,
}

_FONT_PATHS = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
)
_FONT_CACHE: dict[int, Any] = {}


def _load_cjk_font(size: int) -> Any:
    """加载并缓存系统中文字体，缺失时返回 None 以便回退英文绘制。"""
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    font = None
    for path in _FONT_PATHS:
        if Path(path).is_file():
            try:
                font = ImageFont.truetype(path, size)
                break
            except OSError:
                continue
    _FONT_CACHE[size] = font
    return font


def _draw_labels(image_bgr: np.ndarray, items: list[tuple[tuple[int, int], str, str, tuple[int, int, int], int]]) -> np.ndarray:
    """在图像上批量绘制中文标签（带描边抗遮挡），无字体时退回 OpenCV 英文。"""
    if not items:
        return image_bgr
    has_font = any(_load_cjk_font(size) is not None for _, _, _, _, size in items)
    if has_font:
        pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        for (x, y), text_cn, text_en, bgr, size in items:
            font = _load_cjk_font(size)
            rgb = (int(bgr[2]), int(bgr[1]), int(bgr[0]))
            if font is not None:
                draw.text((x, y), text_cn, font=font, fill=rgb, stroke_width=2, stroke_fill=(255, 255, 255))
            else:
                draw.text((x, y), text_en, fill=rgb)
        return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    out = image_bgr
    for (x, y), _, text_en, bgr, size in items:
        scale = max(0.42, size / 32.0)
        cv2.putText(out, text_en, (x, y + size), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(out, text_en, (x, y + size), cv2.FONT_HERSHEY_SIMPLEX, scale, bgr, 1, cv2.LINE_AA)
    return out


def format_outside_message(delta: list[float] | tuple[float, float]) -> str:
    """按画面方向描述针尖相对基准中心的位置，避免移动方向歧义。"""
    dx, dy = float(delta[0]), float(delta[1])
    h_text = "居中" if abs(dx) < 5e-4 else f"{'右侧' if dx > 0 else '左侧'} {abs(dx):.3f}mm"
    v_text = "居中" if abs(dy) < 5e-4 else f"{'下方' if dy > 0 else '上方'} {abs(dy):.3f}mm"
    return f"基准标定失败，请将工具尖移动至标定范围（针尖在中心{h_text}、{v_text}，画面方向）"


def settings(path: str | Path) -> dict[str, Any]:
    """读取并校验 config/tooltip_baseline.yaml 中的 baseline_ui 参数。"""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg = {**DEFAULTS, **(data.get("baseline_ui") or {})}
    for key in (
        "main_tol_mm",
        "main_um_per_px",
        "aux_um_per_px",
        "min_focus_contrast",
        "focus_good_px",
        "focus_warn_px",
        "focus_max_px",
        "focus_pass_percent",
        "focus_ready_percent",
        "runout_max_mm",
        "runout_frame_std_max_px",
    ):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} 必须为有限正数")
    if cfg["focus_ready_percent"] > 100:
        raise ValueError("focus_ready_percent 不能大于 100")
    if cfg["focus_pass_percent"] >= cfg["focus_ready_percent"]:
        raise ValueError("focus_pass_percent 必须小于 focus_ready_percent")
    if cfg["focus_warn_px"] < cfg["focus_good_px"]:
        raise ValueError("focus_warn_px 不能小于 focus_good_px")
    if cfg["focus_max_px"] < cfg["focus_good_px"]:
        raise ValueError("focus_max_px 不能小于 focus_good_px")
    for key in ("focus_smooth_frames", "focus_ready_frames", "peak_min_frames"):
        value = cfg[key]
        if type(value) is not int or value < 1:
            raise ValueError(f"{key} 必须为正整数")
    margin = cfg["edge_margin_px"]
    if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not math.isfinite(margin) or margin < 0:
        raise ValueError("edge_margin_px 必须为非负有限数")
    reset_margin = cfg["peak_reset_margin_percent"]
    max_reset_margin = 100.0 - float(cfg["focus_pass_percent"])
    if (
        isinstance(reset_margin, bool)
        or not isinstance(reset_margin, (int, float))
        or not math.isfinite(reset_margin)
        or reset_margin < 0
        or reset_margin >= max_reset_margin
    ):
        raise ValueError(
            f"peak_reset_margin_percent 必须满足 0 <= x < 100 - focus_pass_percent（当前上限 {max_reset_margin:g}）"
        )
    target = cfg["main_target_uv"]
    if target is not None and (
        not isinstance(target, (tuple, list))
        or len(target) != 2
        or any(type(x) not in (int, float) or not math.isfinite(x) for x in target)
    ):
        raise ValueError("main_target_uv 必须为两个有限数字或 null")
    return cfg


def _view_quality_summary(
    view: dict[str, Any] | None,
    detection_item: dict[str, Any] | None,
    detection: Any = None,
) -> tuple[float | None, float | None, float | None]:
    """提取视图的代表性轴拟合残差、边缘宽度与轴残差门限。"""
    cfg = detection_item or {}
    method = cfg.get("method", "trident")
    rms_limit = float(cfg.get("max_axis_rms_px", 0.7)) if method == "trident" else None
    frames = (view or {}).get("frames") or []
    rms_vals = [
        float(f["axis_rms_px"])
        for f in frames
        if isinstance(f.get("axis_rms_px"), (int, float)) and math.isfinite(f["axis_rms_px"])
    ]
    edge_vals = [
        float(f["edge_width_px"])
        for f in frames
        if isinstance(f.get("edge_width_px"), (int, float)) and math.isfinite(f["edge_width_px"])
    ]
    if rms_vals and rms_limit is not None:
        over = [v for v in rms_vals if v > rms_limit]
        axis_rms = float(np.median(over)) if over else float(np.median(rms_vals))
    elif rms_vals:
        axis_rms = float(np.median(rms_vals))
    elif detection is not None and isinstance(getattr(detection, "axis_rms_px", None), (int, float)) and math.isfinite(detection.axis_rms_px):
        axis_rms = float(detection.axis_rms_px)
    else:
        axis_rms = None
    if edge_vals:
        edge_px = float(np.median(edge_vals))
    elif detection is not None and isinstance(getattr(detection, "edge_width_px", None), (int, float)) and math.isfinite(detection.edge_width_px):
        edge_px = float(detection.edge_width_px)
    else:
        edge_px = None
    return axis_rms, edge_px, rms_limit


def _format_measure_failure(measurement: dict[str, Any], detection_cfg: dict[str, Any], ui_cfg: dict[str, Any]) -> str:
    """将测量拒绝原因格式化为可操作的中文提示。"""
    good_px = float(ui_cfg.get("focus_good_px", 2.0))
    warn_px = float(ui_cfg.get("focus_warn_px", 3.0))
    parts: list[str] = []
    for cam, label in (("front", "正面"), ("side", "侧面")):
        view = measurement["views"][cam]
        if view["accepted"]:
            continue
        reasons = view["reasons"]
        if any("工具在动或画面畸变" in r for r in reasons):
            parts.extend(reasons)
        elif any("针身轴拟合残差过大" in r for r in reasons):
            axis_rms, edge_px, rms_limit = _view_quality_summary(view, (detection_cfg or {}).get(cam, {}))
            if axis_rms is not None and rms_limit is not None:
                if edge_px is not None:
                    blur_word = "偏糊" if edge_px > warn_px else ("稍虚" if edge_px > good_px else "对焦合格")
                    edge_desc = f"（{label}边缘宽 {edge_px:.1f}px，{blur_word}）"
                else:
                    edge_desc = ""
                parts.append(
                    f"{label}针身轴拟合残差 {axis_rms:.2f}px > {rms_limit:g}{edge_desc}"
                    f"→ 调{label}相机对焦至绿色后重测，或点【识别不准，手动确认】"
                )
            else:
                parts.extend(reasons)
        else:
            parts.extend(reasons)
    return "；".join(parts) if parts else measurement["reason"]


def baseline_mapping(baseline: ToolTipBaseline) -> dict[str, Any]:
    """将界面建立的基准对象转换为保留雅可比待定标记的 YAML 映射。"""
    views = []
    for view in baseline.views:
        item: dict[str, Any] = {
            "name": view.name,
            "target_uv": list(view.target_uv),
            "jacobian": None,
        }
        if view.width_px is not None:
            item["width_px"] = view.width_px
        if view.click is not None:
            item["click"] = view.click
        if view.manual_override:
            item["manual_override"] = True
        if view.axis_rms_px is not None:
            item["axis_rms_px"] = view.axis_rms_px
        if view.edge_width_px is not None:
            item["edge_width_px"] = view.edge_width_px
        if view.override_note:
            item["override_note"] = view.override_note
        if view.focus_percent is not None:
            item["focus_percent"] = view.focus_percent
        if view.focus_peak is not None:
            item["focus_peak"] = view.focus_peak
        if view.peak_confirmed is not None:
            item["peak_confirmed"] = view.peak_confirmed
        views.append(item)
    return {
        "version": 1,
        "tool_name": baseline.tool_name,
        "p0": list(baseline.p0),
        "offset_xyz": list(baseline.offset_xyz),
        "timestamp": baseline.timestamp,
        "note": baseline.note,
        "jacobian_pending": True,
        **({"detector": baseline.detector} if baseline.detector else {}),
        **({"manual_override": True} if baseline.manual_override else {}),
        **({"baseline_rule": baseline.baseline_rule} if baseline.baseline_rule else {}),
        **({"focus_criteria": dict(baseline.focus_criteria)} if baseline.focus_criteria else {}),
        "views": views,
    }


def _parse_iso(value: Any) -> datetime | None:
    """解析 ISO 时间戳用于新旧比较；格式无效返回 None。"""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        return dt.astimezone()
    return dt


def _is_newer(offset_time: Any, baseline_time: Any) -> bool:
    """判断工具偏移时间是否晚于现有基准时间。"""
    t_off = _parse_iso(offset_time)
    t_base = _parse_iso(baseline_time)
    if t_off is not None and t_base is not None:
        return t_off > t_base
    if offset_time and baseline_time:
        return str(offset_time) > str(baseline_time)
    return False


def summary(path: str | Path, stale: bool = False, detection=None) -> dict[str, Any]:
    """读取单把工具现有基准的简要信息，占位或缺失时标记 missing。"""
    try:
        b = load_baseline(path)
        return {
            "time": b.timestamp,
            "detector": b.detector,
            "detector_warning": baseline_detector_warning(b.detector, detection or {}),
            "p0": list(b.p0),
            "pixels": {v.name: list(v.target_uv) for v in b.views},
            "jacobian_pending": b.jacobian_pending or any(v.jacobian is None for v in b.views),
            "stale": bool(stale),
            "manual_override": b.manual_override,
            "override_notes": [v.override_note for v in b.views if v.override_note],
            "note": b.note,
        }
    except Exception:
        return {"missing": True, "stale": bool(stale)}


def existing_baselines(config: Any, calibration_path: str | Path) -> dict[str, dict[str, Any]]:
    """汇总两把工具的现有基准、tool_offset 时效与 tools.yaml 补偿。"""
    cal_dir = Path(config.tool_calibration.directory)
    offsets_path = Path(calibration_path).parent / "tool_offset.yaml"
    if not offsets_path.exists() and (cal_dir / "tool_offset.yaml").exists():
        offsets_path = cal_dir / "tool_offset.yaml"
    try:
        offsets = _read(offsets_path)
    except Exception:
        offsets = {}
    try:
        tools_data = load_tools(cal_dir / "tools.yaml")["tools"]
    except Exception:
        tools_data = {}
    result: dict[str, dict[str, Any]] = {}
    for n in (1, 2):
        key = f"tool{n}"
        entry = offsets.get(key) if isinstance(offsets.get(key), dict) else {}
        item = summary(cal_dir / f"{key}_baseline.yaml", bool(entry.get("baseline_stale")),
                       getattr(config.tool_calibration, 'detection', {}))
        try:
            from nest_clear.devices.tooltip_camera_store import baseline_differences
            raw = _read(cal_dir / f"{key}_baseline.yaml")
            item['camera_differences'] = baseline_differences(config, raw.get('camera_settings')) if isinstance(raw, dict) else []
        except Exception:
            item['camera_differences'] = []
        item["tool_offset_time"] = entry.get("time")
        item["tool_offset_missing"] = not bool(entry)
        item["offset_newer"] = _is_newer(entry.get("time"), item.get("time"))
        tool_entry = tools_data.get(key) if isinstance(tools_data.get(key), dict) else {}
        if "offset_xyz" in tool_entry:
            item["offset_xyz"] = list(tool_entry["offset_xyz"])
        result[key] = item
    return result


def warn_existing_detectors(events, existing):
    """Publish the same legacy-baseline warning from either camera host."""
    warnings = {name: item['detector_warning'] for name, item in existing.items()
                if item.get('detector_warning')}
    previous = getattr(events, 'baseline_detector_warnings', {})
    for name, warning in warnings.items():
        if previous.get(name) != warning:
            events.emit('warning', tool=int(name[-1]), reason=warning)
    events.baseline_detector_warnings = warnings


def publish_existing_baselines(events, app_config, calibration_path, **state):
    existing = existing_baselines(app_config, calibration_path)
    warn_existing_detectors(events, existing)
    events.emit('baseline_state', existing=existing, **state)


def refresh_existing(ctx: Any) -> None:
    """其他标定更新 tool_offset 后刷新对刀基准页摘要，不影响原流程。"""
    try:
        cal_path = Path(getattr(ctx, "calibration_path", None) or ctx.config.vision.calibration)
        publish_existing_baselines(ctx.events, ctx.config, cal_path)
    except Exception as exc:
        ctx.events.emit("warning", reason=f"刷新对刀基准状态失败：{exc}")


def _click_worker(connection: Any, images: dict[str, np.ndarray], points: dict[str, Any], names: tuple[str, ...], tool: int | None) -> None:
    """在独立进程依次运行 T23 针尖放大点选窗口，避免 Qt 影响主程序。"""
    try:
        results: dict[str, ClickResult] = {}
        for name in names:
            label = "主" if name == "front" else "辅助"
            results[name] = confirm_tip(images[name], points[name], f"工具{tool} {label}对刀相机")
            if results[name].point_uv is None:
                break
        connection.send((True, results))
    except BaseException as exc:
        connection.send((False, str(exc)))
    finally:
        connection.close()


class BaselineSession:
    """主线程提交界面命令，唯一设备线程负责采图、检测、点选、采样与保存。"""

    def __init__(
        self,
        config: Any,
        cameras: Any,
        plc: Any,
        events: Any,
        calibration_path: str | Path,
        ui_config: dict[str, Any],
        *,
        click: Callable[..., ClickResult] = confirm_tip,
    ) -> None:
        """初始化会话并启动后台相机线程；任何异常路径均释放相机。"""
        self.config = config
        self.cameras = cameras
        self.plc = plc
        self.events = events
        self.calibration_path = Path(calibration_path)
        self.cfg = dict(DEFAULTS, **ui_config)
        self.click = click
        self.queue: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue()
        self.phase = "select"
        self.tool: int | None = None
        self.message = "请选择工具"
        self.notice = ""
        self.candidate: dict[str, Any] | None = None
        self.frames: dict[str, np.ndarray] | None = None
        self.measurement: dict[str, Any] | None = None
        self.detections: dict[str, Any] = {}
        self.closed = False
        self.stop = threading.Event()
        self.commands: list[dict[str, Any]] = []
        self.measurements: list[dict[str, Any]] = []
        self.saved_path: str | None = None
        self.shots = 0
        detector_settings = getattr(getattr(config, "tool_calibration", None), "detection", {}) or {}
        self.trackers = {n: make_tip_tracker(detector_settings.get(n, {}), self.cfg) for n in ("front", "side")}
        self.readiness_trackers = {n: ReadinessTracker() for n in ("front", "side")}
        self.history: dict[str, list[tuple[float, float]]] = {"front": [], "side": []}
        self.manual_points: dict[str, list[float]] = {}
        self.manual_clicks: dict[str, dict[str, Any]] = {}
        self.live_snapshots: dict[str, dict[str, Any]] = {}
        self._last_live_emit = 0.0
        self._live_last_ts: float | None = None
        self.start_time = events.clock().isoformat()
        self.directory = events.clock().strftime("%Y%m%d_%H%M%S_%f")
        self.record_dir = events.run_directory / "baseline" / self.directory if events.run_directory else None
        self.publish()
        self.thread = threading.Thread(target=self.run, name="baseline-camera", daemon=True)
        self.thread.start()

    def existing(self) -> dict[str, dict[str, Any]]:
        """返回两把工具现有基准、工具偏移状态及当前补偿。"""
        return existing_baselines(self.config, self.calibration_path)

    def _selection_notice(self, tool: int, existing: dict[str, dict[str, Any]]) -> str:
        """选工具时检查 tool_offset 是否缺失或较新，仅作提示不阻止。"""
        info = existing.get(f"tool{tool}") or {}
        if info.get("tool_offset_missing"):
            return f"提示：工具{tool} 尚无工具偏移标定结果（不阻止建基准）"
        if info.get("offset_newer"):
            return f"提示：工具{tool} 的工具偏移标定结果比现有对刀基准新（不阻止建基准）"
        return ""

    def publish(self, **extra: Any) -> None:
        """发布最新基准会话状态到事件与界面状态文件。"""
        existing = self.existing()
        warn_existing_detectors(self.events, existing)
        data: dict[str, Any] = {
            "phase": self.phase,
            "tool": self.tool,
            "message": self.message,
            "notice": self.notice,
            "candidate": self.candidate,
            "existing": existing,
            "config": self.cfg,
        }
        if self.phase in ("select", "done"):
            data["live_delta"] = None
            data["live_inside"] = False
            data["live_readiness"] = None
        else:
            data["manual_confirmed"] = {n: n in getattr(self, "manual_points", {}) for n in ("front", "side")}
            snapshots = getattr(self, "live_snapshots", None)
            if snapshots:
                data["live_readiness"] = evaluate_baseline_readiness(
                    snapshots.get("front"),
                    snapshots.get("side"),
                    config=self.cfg,
                    manual_confirmed=data["manual_confirmed"],
                )
        if self.tool is not None:
            try:
                data["offset_xyz"] = load_tools(Path(self.config.tool_calibration.directory) / "tools.yaml")["tools"][f"tool{self.tool}"]["offset_xyz"]
            except Exception as exc:
                data["offset_error"] = str(exc)
        else:
            data["offset_xyz"] = None
        data.update(extra)
        self.events.emit("baseline_state", **data)

    def submit(self, command: str, params: dict[str, Any], seq: int) -> None:
        """记录界面命令并投递到后台设备线程；结束命令直接置位停止标志。"""
        self.commands.append({"command": command, "params": params, "seq": seq})
        if command == "baseline_end":
            self.end()
        else:
            self.queue.put((command, params))

    def end(self, reason: str = "用户结束") -> None:
        """请求结束会话并唤醒等待循环以释放相机。"""
        self.message = reason
        self.stop.set()

    def _frame(self, stage: str = "baseline_preview") -> dict[str, np.ndarray]:
        """复用对刀相机 worker 采图，保留超时与阻塞保护。"""
        return self.cameras.execute({"stage": stage})

    def _detect(self, images: dict[str, np.ndarray]) -> dict[str, Any]:
        """实时预览按两路配置各检测一帧。"""
        return {
            name: make_tip_detector(self.config.tool_calibration.detection[name], SilhouetteTipDetector).detect(images[name])
            for name in ("front", "side")
        }

    def _target(self, image: np.ndarray) -> tuple[float, float]:
        """返回主对刀相机容许圆中心像素坐标，默认图像中心。"""
        target = self.cfg["main_target_uv"]
        if target is not None:
            return (float(target[0]), float(target[1]))
        return ((image.shape[1] - 1) / 2.0, (image.shape[0] - 1) / 2.0)

    def _annotate(self, images: dict[str, np.ndarray], detections: dict[str, Any]) -> None:
        """在主/辅助整幅画面及右侧放大小窗绘制针尖、轮廓与清晰度就绪状态（无容许圆红绿判定）。"""
        cfg = getattr(self, "cfg", DEFAULTS)
        snapshots = getattr(self, "live_snapshots", {}) or {}
        manual_pts = getattr(self, "manual_points", {}) or {}
        for name in ("front", "side"):
            raw = images[name]
            base_bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR) if raw.ndim == 2 else raw.copy()
            h, w = base_bgr.shape[:2]
            d = detections.get(name)
            scale_um = float(cfg["main_um_per_px"] if name == "front" else cfg["aux_um_per_px"])
            crop_size = max(60, round(min(w, h) * 0.125))
            if crop_size % 2:
                crop_size += 1
            inset_size = max(280, round(h * 0.625))
            zoom = inset_size / crop_size

            tip_uv = tuple(float(x) for x in d.point_uv) if d and d.point_uv is not None else None
            manual_uv = (
                tuple(float(x) for x in self.candidate["pixels"][name])
                if self.candidate and name in self.candidate.get("click", {}) and self.candidate["pixels"].get(name)
                else (tuple(float(x) for x in manual_pts[name]) if name in manual_pts else None)
            )
            eval_uv = manual_uv if manual_uv is not None else tip_uv
            roi_center = eval_uv if eval_uv is not None else ((w - 1) / 2.0, (h - 1) / 2.0)

            # 放大图取原图加轮廓，避免整幅图线标遮挡细节。
            zoom_src = base_bgr.copy()
            detection_cfg = getattr(getattr(self.config, "tool_calibration", None), "detection", {}) or {}
            threshold = (detection_cfg.get(name) or {}).get("threshold", 127)
            mask = (raw <= threshold).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                cv2.drawContours(base_bgr, [largest], -1, (255, 120, 0), 1)
                cv2.drawContours(zoom_src, [largest], -1, (255, 120, 0), 1)

            labels: list[tuple[tuple[int, int], str, str, tuple[int, int, int], int]] = []
            center_ref = self._target(raw) if name == "front" else None
            if center_ref is not None:
                cx, cy = round(center_ref[0]), round(center_ref[1])
                cv2.drawMarker(base_bgr, (cx, cy), (160, 165, 175), cv2.MARKER_CROSS, 24, 1)

            if tip_uv is not None:
                tip_pt = (round(tip_uv[0]), round(tip_uv[1]))
                cv2.circle(base_bgr, tip_pt, 7, (0, 140, 255), 2, cv2.LINE_AA)
                cv2.drawMarker(base_bgr, tip_pt, (0, 220, 255), cv2.MARKER_CROSS, 24, 2)
            if manual_uv is not None:
                click_pt = (round(manual_uv[0]), round(manual_uv[1]))
                cv2.circle(base_bgr, click_pt, 8, (180, 0, 255), 2, cv2.LINE_AA)
                cv2.drawMarker(base_bgr, click_pt, (255, 0, 180), cv2.MARKER_TILTED_CROSS, 24, 2)
                labels.append(((min(w - 70, click_pt[0] + 10), max(4, click_pt[1] - 20)), "人工点", "MANUAL", (180, 0, 220), 14))

            if name == "front":
                if eval_uv is not None and center_ref is not None:
                    dx, dy = ((eval_uv[i] - center_ref[i]) * scale_um / 1000.0 for i in (0, 1))
                    labels.append((
                        (12, 8),
                        f"中心偏差（参考）横向 {dx:+.3f}mm  竖向 {dy:+.3f}mm",
                        f"Ref center dx {dx:+.3f}mm dy {dy:+.3f}mm",
                        (100, 110, 125),
                        16,
                    ))
                else:
                    labels.append(((12, 8), "等待识别针尖", "Waiting tip", (100, 110, 125), 16))

            snap = snapshots.get(name) or {}
            axis_hint = CAMERA_AXES[name]["jog_hint"]
            pct = snap.get("percent")
            pct_str = f"{pct:.0f}%" if isinstance(pct, (int, float)) and math.isfinite(pct) else "--%"
            readiness = evaluate_camera_readiness(
                name,
                {
                    **snap,
                    "detection_status": snap.get("detection_status")
                    or ("未识别" if not d or d.point_uv is None else "粗定位" if d.coarse else "精定位"),
                    "detection_reason": getattr(d, "diagnostic", "") if d else "",
                    "tip_uv": list(eval_uv) if eval_uv is not None else None,
                    "image_size": [w, h],
                    "edge_width_px": snap.get("edge_width_px") if snap.get("edge_width_px") is not None else (d.edge_width_px if d else None),
                },
                config=cfg,
                manual_confirmed=(name in manual_pts),
            )
            color_map = {"green": (0, 155, 0), "yellow": (0, 145, 220), "red": (0, 0, 220), "gray": (100, 110, 125)}

            if d and (d.point_uv is not None or name in manual_pts):
                edge_val = snap.get("edge_width_px") if snap.get("edge_width_px") is not None else d.edge_width_px
                edge_text = f"{edge_val:.2f}px" if edge_val is not None else "--"
                if edge_val is not None and math.isfinite(edge_val) and edge_val > 0:
                    good_px = float(cfg.get("focus_good_px", 2.0))
                    warn_px = float(cfg.get("focus_warn_px", 3.0))
                    if edge_val <= good_px:
                        focus_cn, focus_en, focus_color = "清晰", "GOOD", (0, 155, 0)
                    elif edge_val <= warn_px:
                        focus_cn, focus_en, focus_color = "稍虚", "WARN", (0, 145, 220)
                    else:
                        focus_cn, focus_en, focus_color = "偏糊", "BLUR", (0, 0, 220)
                else:
                    focus_cn = "清晰" if d.focused else "偏糊"
                    focus_en = "OK" if d.focused else "FAIL"
                    focus_color = (0, 155, 0) if d.focused and not d.coarse else (0, 0, 220)
                if readiness["edge_near"]:
                    focus_color = (0, 0, 220)
                elif readiness["ready"]:
                    focus_color = (0, 155, 0)
                coarse_cn = " 粗定位" if d.coarse else ""
                manual_cn = " 人工确认" if name in manual_pts else ""
                ready_tag = f" [{readiness['stage_text']}]" if not readiness["edge_near"] else ""
                labels.append((
                    (12, 34 if name == "front" else 10),
                    f"清晰度 {pct_str}  边缘 {edge_text}  {focus_cn}{coarse_cn}{manual_cn}{ready_tag}",
                    f"focus {pct_str} edge {edge_text} {focus_en}",
                    focus_color,
                    15,
                ))
                if readiness["edge_near"]:
                    labels.append(((12, 58 if name == "front" else 34), EDGE_MARGIN_MESSAGE, "Near image edge", (0, 0, 220), 14))
                elif readiness.get("wide_edge_note"):
                    labels.append(((12, 58 if name == "front" else 34), readiness["wide_edge_note"], "Wide edge accepted", (0, 145, 220), 14))
                if not readiness["edge_near"] and readiness.get("peak_update_note"):
                    base_y = (58 if name == "front" else 34) + (20 if readiness.get("wide_edge_note") else 0)
                    peak_en = "Peak updated, confirm kept" if readiness.get("peak_confirmed") else "New peak >= margin, re-overshoot"
                    labels.append(((12, base_y), readiness["peak_update_note"], peak_en, (0, 145, 220), 14))
            else:
                labels.append((
                    (12, 34 if name == "front" else 10),
                    f"未检测到针尖（{axis_hint}）",
                    "NO TIP / focus FAIL",
                    (0, 0, 220),
                    15,
                ))

            # 右侧放大小窗：围绕针尖局部放大，展示清晰度与调焦引导。
            matrix = np.array(
                [
                    [zoom, 0.0, inset_size / 2.0 - roi_center[0] * zoom],
                    [0.0, zoom, inset_size / 2.0 - roi_center[1] * zoom],
                ],
                dtype=np.float32,
            )
            zoomed = cv2.warpAffine(zoom_src, matrix, (inset_size, inset_size), flags=cv2.INTER_NEAREST, borderValue=(235, 235, 235))
            if tip_uv is not None:
                zx = (tip_uv[0] - roi_center[0]) * zoom + inset_size / 2.0
                zy = (tip_uv[1] - roi_center[1]) * zoom + inset_size / 2.0
                if 0 <= zx < inset_size and 0 <= zy < inset_size:
                    cv2.circle(zoomed, (round(zx), round(zy)), 6, (0, 0, 0), -1, cv2.LINE_AA)
                    cv2.circle(zoomed, (round(zx), round(zy)), 5, (0, 220, 255), -1, cv2.LINE_AA)
            if manual_uv is not None:
                mx = (manual_uv[0] - roi_center[0]) * zoom + inset_size / 2.0
                my = (manual_uv[1] - roi_center[1]) * zoom + inset_size / 2.0
                if 0 <= mx < inset_size and 0 <= my < inset_size:
                    cv2.circle(zoomed, (round(mx), round(my)), 6, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(zoomed, (round(mx), round(my)), 5, (200, 0, 220), -1, cv2.LINE_AA)

            panel_w = inset_size + 24
            panel_h = max(h, inset_size + 102)
            canvas = np.full((panel_h, w + panel_w, 3), (246, 248, 250), dtype=np.uint8)
            canvas[:h, :w] = base_bgr
            cv2.line(canvas, (w, 0), (w, panel_h - 1), (208, 213, 221), 1)
            ix, iy = w + 12, 34
            canvas[iy:iy + inset_size, ix:ix + inset_size] = zoomed
            cv2.rectangle(canvas, (ix - 1, iy - 1), (ix + inset_size, iy + inset_size), (150, 160, 175), 1)

            title_cn = f"针尖局部放大 {zoom:.1f}× ({crop_size}×{crop_size}px)"
            title_en = f"Zoom {zoom:.1f}x ({crop_size}x{crop_size}px)"
            labels.append(((ix, 8), title_cn, title_en, (30, 41, 59), 15))

            ly = iy + inset_size + 8
            labels.append(((ix, ly), "黄实心点：自动检测  紫实心点：人工确认", "Yellow: Auto  Purple: Manual", (60, 70, 85), 13))
            labels.append(((ix, ly + 22), CAMERA_AXES[name]["geometry_hint"], f"Adjust {CAMERA_AXES[name]['axis']}", (60, 70, 85), 13))
            status_cn = "状态：已最清晰（可作为对刀基准）" if readiness["ready"] else f"状态：{readiness['short_reason']}"
            status_en = "Status: Ready" if readiness["ready"] else f"Status: Adjust {CAMERA_AXES[name]['axis']}"
            labels.append(((ix, ly + 44), status_cn, status_en, color_map.get(readiness["color_key"], (0, 0, 220)), 14))

            annotated = _draw_labels(canvas, labels)
            path = live_image_file(self.events.directory, "baseline_" + name)
            path.parent.mkdir(parents=True, exist_ok=True)
            ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                temporary = path.with_suffix(".tmp")
                temporary.write_bytes(encoded.tobytes())
                os.replace(temporary, path)

    def _archive(self, images: dict[str, np.ndarray]) -> dict[str, str]:
        """存档当前测量帧原图到运行目录 baseline/ 并记录事件。"""
        self.shots += 1
        paths: dict[str, str] = {}
        if self.record_dir is not None:
            self.record_dir.mkdir(parents=True, exist_ok=True)
            for name, image in images.items():
                path = self.record_dir / f"{self.shots:03d}_{name}_raw.png"
                if not cv2.imwrite(str(path), image):
                    raise OSError(f"保存原图失败：{path}")
                paths[name] = str(path)
        self.events.emit("baseline_capture", tool=self.tool, shot=self.shots, paths=paths)
        return paths

    def _stable_position(self) -> list[float]:
        """复用工具偏移标定的抖动采样参数读取稳定 PLC 实时坐标作为 P0。"""
        cfg = self.config.tool_offset
        for _ in range(3):
            samples = []
            for i in range(cfg["settle_samples"]):
                if self.stop.is_set():
                    raise RuntimeError("会话已结束")
                value = self.plc.execute({"operation": "actual"})
                if len(value) != 3 or not all(math.isfinite(x) for x in value):
                    raise ValueError("PLC 实时坐标无效")
                samples.append(value)
                if i + 1 < cfg["settle_samples"]:
                    self.stop.wait(cfg["settle_window_s"] / (cfg["settle_samples"] - 1))
            arr = np.asarray(samples, dtype=float)
            if np.all(np.ptp(arr, axis=0) <= cfg["jitter_tol_mm"]):
                return arr.mean(axis=0).tolist()
        raise ValueError("PLC 实时坐标不稳定，请等待稳定后再按【对刀基准】")

    def _manual(self, images: dict[str, np.ndarray], ds: dict[str, Any], names: tuple[str, ...]) -> dict[str, ClickResult]:
        """调用 T23 放大点选窗口获取人工亚像素针尖位置。"""
        if self.click is not confirm_tip:
            return {view: self.click(images[view], ds[view].point_uv, view) for view in names}
        ctx = mp.get_context("spawn")
        parent, child = ctx.Pipe(False)
        points = {v: ds[v].point_uv for v in names}
        process = ctx.Process(target=_click_worker, args=(child, images, points, names, self.tool), daemon=True)
        process.start()
        child.close()
        deadline = time.monotonic() + self.config.tool_calibration.click_timeout_s
        try:
            while not parent.poll(0.1):
                if self.stop.is_set():
                    raise ValueError("会话已结束")
                if time.monotonic() >= deadline:
                    raise ValueError("人工点选超时")
                if not process.is_alive():
                    raise ValueError("人工点选窗口异常退出")
            ok, result = parent.recv()
            if not ok:
                raise ValueError(result)
            return result
        finally:
            parent.close()
            if process.is_alive():
                process.terminate()
            process.join(timeout=2)

    def _record_measurement(self, mode: str, outcome: str, reason: str | None = None) -> None:
        """记录单次对刀基准测量/点选结果用于会话归档。"""
        entry = {
            "time": self.events.clock().isoformat(),
            "mode": mode,
            "outcome": outcome,
            "reason": reason,
            "candidate": self.candidate,
            "measurement": self.measurement,
        }
        self.measurements.append({**entry, "candidate": copy.deepcopy(self.candidate)})
        self.events.emit("baseline_measure", tool=self.tool, **entry)

    def _evaluate(self, images: dict[str, np.ndarray], ds: dict[str, Any], manual: dict[str, ClickResult] | None = None, aux_only: bool = False) -> None:
        """判定检测/点选、主相机容许圆、清晰度及 P0 稳定性并刷新标注图。"""
        points: dict[str, list[float] | None] = {
            name: list(d.point_uv) if d.point_uv is not None else None
            for name, d in ds.items()
        }
        clicked: dict[str, dict[str, Any]] = {}
        for name, result in (manual or {}).items():
            if isinstance(ds[name], TridentResult) and (
                "工具在动或画面畸变" in ds[name].diagnostic
                or (ds[name].axis_rms_px is not None and ds[name].axis_rms_px > 2.0)
            ):
                raise ValueError(f"{'主对刀' if name == 'front' else '辅助'}相机识别不可靠：{ds[name].diagnostic}")
            if result.point_uv is None:
                raise ValueError("已取消手动点选")
            points[name] = [float(result.point_uv[0]), float(result.point_uv[1])]
            clicked[name] = {"source": result.source, "zoom": result.zoom, "elapsed_s": result.elapsed_s}
        detection_cfg = getattr(getattr(self.config, "tool_calibration", None), "detection", {}) or {}
        measurement_views = (self.measurement or {}).get("views", {})
        focus: dict[str, dict[str, Any]] = {}
        for name, d in ds.items():
            view_info = measurement_views.get(name)
            axis_rms, edge_px, rms_limit = _view_quality_summary(view_info, detection_cfg.get(name, {}), d)
            was_rejected = (
                (view_info is not None and not view_info.get("accepted", True))
                or bool(d.coarse)
                or (not d.focused)
                or (d.point_uv is None)
            )
            is_override = (name in clicked) and was_rejected
            if is_override:
                if axis_rms is not None and rms_limit is not None and axis_rms > rms_limit:
                    override_note = f"人工确认（轴残差 {axis_rms:.2f}>{rms_limit:g}）"
                else:
                    override_note = "人工确认（自动识别未通过）"
            else:
                override_note = None
            resolved_edge = edge_px if edge_px is not None else d.edge_width_px
            snap = getattr(self, "live_snapshots", {}).get(name) or {}
            focus_pct = snap.get("percent")
            focus_peak = snap.get("peak")
            focus[name] = {
                "focused": d.focused,
                "edge_width_px": resolved_edge,
                "width_px": d.width_px,
                "coarse": d.coarse,
                "confidence": d.confidence,
                "diagnostic": d.diagnostic,
                "axis_rms_px": axis_rms,
                "axis_rms_limit": rms_limit,
                "manual_override": is_override,
                "override_note": override_note,
                "focus_percent": float(focus_pct) if isinstance(focus_pct, (int, float)) and math.isfinite(focus_pct) else None,
                "focus_peak": float(focus_peak) if isinstance(focus_peak, (int, float)) and math.isfinite(focus_peak) else None,
                "peak_confirmed": bool(snap["peak_confirmed"]) if snap.get("peak_confirmed") is not None else None,
            }
            if name in clicked:
                clicked[name]["manual_override"] = is_override
                if axis_rms is not None:
                    clicked[name]["axis_rms_px"] = axis_rms
                if resolved_edge is not None:
                    clicked[name]["edge_width_px"] = resolved_edge
                if override_note:
                    clicked[name]["override_note"] = override_note
        self.detections = ds
        if aux_only:
            if not self.candidate or not self.candidate.get("p0"):
                raise ValueError("请先建立主相机基准")
            self.candidate["pixels"]["side"] = points["side"]
            self.candidate["click"]["side"] = clicked["side"]
            self.candidate["focus"]["side"] = focus["side"]
            self.candidate["detector"]["side"] = self.measurement["views"]["side"]["detector"]
            self.candidate["manual_override"] = any(
                self.candidate["focus"][n].get("manual_override") for n in ("front", "side")
            )
            self.candidate["passed"] = True
            self._annotate(images, ds)
            self.phase = "success"
            self.message = "辅助相机基准已重设，请保存"
            if self.measurement["rejected"]:
                note = focus["side"].get("override_note")
                self.message += f"（{note}）" if note else ("。自动识别未通过，已使用人工点选。" + self.measurement["reason"])
            self._record_measurement("aux_manual", "success")
            self.publish()
            return
        center = self._target(images["front"])
        delta = (
            [(points["front"][i] - center[i]) * self.cfg["main_um_per_px"] / 1000.0 for i in (0, 1)]
            if points["front"] is not None
            else None
        )
        distance = math.hypot(*delta) if delta is not None else None
        inside = distance is not None and distance <= self.cfg["main_tol_mm"]
        self.candidate = {
            "pixels": points,
            "center": list(center),
            "delta_mm": delta,
            "distance_mm": distance,
            "inside": inside,
            "passed": False,
            "focus": focus,
            "detector": {name: view["detector"] for name, view in self.measurement["views"].items()},
            "click": clicked,
            "manual_override": any(focus[n].get("manual_override") for n in ("front", "side")),
            "p0": None,
        }
        self._annotate(images, ds)
        mode = "manual" if clicked else "auto"
        margin_px = float(getattr(self, "cfg", DEFAULTS).get("edge_margin_px", 40.0))
        for name in ("front", "side"):
            label = "主对刀相机" if name == "front" else "辅助相机"
            if points[name] is None:
                raise ValueError(f"{label}无法识别针尖，检查背光和遮挡：{ds[name].diagnostic}")
            if ds[name].coarse and name not in clicked:
                raise ValueError(f"{label}识别不可靠，请手动确认：{ds[name].diagnostic}")
            if not ds[name].focused and name not in clicked:
                raise ValueError(f"{label}针尖模糊，检查对焦/背光：{ds[name].diagnostic}")
            img_h, img_w = images[name].shape[:2]
            if is_near_edge(points[name], (img_w, img_h), margin_px):
                raise ValueError(f"{label}{EDGE_MARGIN_MESSAGE}")
        self.candidate["p0"] = self._stable_position()
        self.candidate["passed"] = True
        self._annotate(images, ds)
        self.phase = "success"
        override_notes = [focus[n]["override_note"] for n in ("front", "side") if focus[n].get("override_note")]
        self.message = "基准建立成功（待保存）"
        if override_notes:
            self.message += "：" + "；".join(override_notes)
        elif self.measurement["rejected"]:
            self.message += "。自动识别未通过，已使用人工点选。" + self.measurement["reason"]
        self._record_measurement(mode, "success")
        self.publish()

    def _save(self) -> None:
        """覆盖保存当前工具基准并备份历史，同时清除 tool_offset 的 baseline_stale。"""
        c = self.candidate
        if not c or not c.get("p0") or not c.get("passed"):
            raise ValueError("没有可保存的基准")
        directory = Path(self.config.tool_calibration.directory)
        offset = load_tools(directory / "tools.yaml")["tools"][f"tool{self.tool}"]["offset_xyz"]
        views = []
        override_notes = []
        for name, label in (("front", "正面"), ("side", "侧面")):
            f_info = c["focus"].get(name) or {}
            is_override = bool(f_info.get("manual_override"))
            note_item = f_info.get("override_note")
            if is_override and note_item:
                override_notes.append(f"{label}：{note_item}")
            views.append(
                ViewCalib(
                    name,
                    c["pixels"][name],
                    None,
                    f_info.get("width_px"),
                    True if name in c["click"] else None,
                    manual_override=is_override,
                    axis_rms_px=f_info.get("axis_rms_px") if is_override else None,
                    edge_width_px=f_info.get("edge_width_px"),
                    override_note=note_item if is_override else None,
                    focus_percent=f_info.get("focus_percent"),
                    focus_peak=f_info.get("focus_peak"),
                    peak_confirmed=f_info.get("peak_confirmed"),
                )
            )
        has_override = bool(c.get("manual_override")) or any(v.manual_override for v in views)
        baseline_note = "界面建立（双路最清晰规则），雅可比待测"
        if override_notes:
            baseline_note += "；" + "；".join(override_notes)
        criteria = {
            "focus_smooth_frames": int(self.cfg.get("focus_smooth_frames", 5)),
            "focus_pass_percent": float(self.cfg.get("focus_pass_percent", 92.0)),
            "focus_ready_percent": float(self.cfg.get("focus_ready_percent", 95.0)),
            "peak_reset_margin_percent": float(self.cfg.get("peak_reset_margin_percent", 3.0)),
            "focus_good_px": float(self.cfg.get("focus_good_px", 2.0)),
            "focus_warn_px": float(self.cfg.get("focus_warn_px", 3.0)),
            "focus_max_px": float(self.cfg.get("focus_max_px", 3.0)),
            "focus_ready_frames": int(self.cfg.get("focus_ready_frames", 5)),
            "peak_min_frames": int(self.cfg.get("peak_min_frames", 10)),
            "edge_margin_px": float(self.cfg.get("edge_margin_px", 40.0)),
        }
        baseline = ToolTipBaseline(
            f"tool{self.tool}",
            c["p0"],
            views,
            offset,
            datetime.now().astimezone().isoformat(),
            baseline_note,
            True,
            detector=c["detector"],
            manual_override=has_override,
            baseline_rule=BASELINE_RULE,
            focus_criteria=criteria,
        )
        path = directory / f"tool{self.tool}_baseline.yaml"
        offsets_path = self.calibration_path.parent / "tool_offset.yaml"
        if not offsets_path.exists() and (directory / "tool_offset.yaml").exists():
            offsets_path = directory / "tool_offset.yaml"
        old = path.read_bytes() if path.exists() else None
        mapping = baseline_mapping(baseline)
        if hasattr(self.cameras, 'cameras'):
            mapping['camera_settings'] = {name: camera.settings_snapshot() for name, camera in self.cameras.cameras.items()}
        save_mapping(path, mapping, f"tool{self.tool}_baseline")
        try:
            offsets = _read(offsets_path)
            if isinstance(offsets.get(f"tool{self.tool}"), dict) and offsets[f"tool{self.tool}"].get("baseline_stale"):
                offsets[f"tool{self.tool}"]["baseline_stale"] = False
                save_mapping(offsets_path, offsets, "tool_offset")
        except Exception:
            if old is None:
                path.unlink(missing_ok=True)
            else:
                restore = path.with_suffix(".restore")
                restore.write_bytes(old)
                os.replace(restore, path)
            raise
        self.saved_path = str(path)
        self.events.emit("baseline_saved", tool=self.tool, path=str(path), p0=c["p0"], offset_xyz=list(offset), detector=baseline.detector)
        try:
            from nest_clear.flow.tool_offset import _existing
            digest = hashlib.sha256(self.calibration_path.read_bytes()).hexdigest()
            self.events.emit("tooloffset_state", existing=_existing(self, offsets_path, digest))
        except Exception as exc:
            self.events.emit("baseline_error", reason=f"刷新工具偏移状态失败：{exc}")
        self.phase = "saved"
        self.message = f"请在 PLC 侧记录当前坐标为工具{self.tool}对刀基准位置：{c['p0']}"
        self.publish()

    def _command(self, name: str, params: dict[str, Any]) -> None:
        """分发单条界面命令并在当前阶段校验合法性。"""
        if name == "baseline_select" and self.phase == "select":
            tool = params.get("tool")
            if type(tool) is not int or tool not in (1, 2):
                raise ValueError("请选择工具1或工具2")
            self.cameras.open()
            for cam in ("front", "side"):
                self.trackers[cam].reset()
                self.readiness_trackers[cam].reset()
                self.history[cam] = []
            self.manual_points.clear()
            self.manual_clicks.clear()
            self.live_snapshots.clear()
            self.tool = tool
            self.notice = self._selection_notice(tool, self.existing())
            self.phase = "live"
            self.message = "请点动使两路针尖都最清晰（正面调 Y、侧面调 X）后按【对刀基准】"
            self.events.emit("baseline_select", tool=tool, notice=self.notice)
            self.publish()
        elif name == "tooltip_peak_reset":
            camera = params.get("camera")
            targets = ("front", "side") if camera in (None, "all") else ((camera,) if camera in self.trackers else ())
            if targets:
                for cam in targets:
                    self.trackers[cam].reset()
                    self.readiness_trackers[cam].reset()
                    self.history[cam] = []
                    self.live_snapshots.pop(cam, None)
                self.events.emit("tooltip_peak_reset", camera=camera or "all")
                self.publish()
        elif name == "baseline_manual_view" and self.phase in ("live", "fail", "success", "saved"):
            camera = params.get("camera")
            if camera not in ("front", "side"):
                raise ValueError("请选择正面或侧面相机")
            images = self.frames or self._frame("baseline_preview")
            ds = self._detect(images)
            d = ds[camera]
            if isinstance(d, TridentResult) and (
                "工具在动或画面畸变" in d.diagnostic
                or (d.axis_rms_px is not None and d.axis_rms_px > 2.0)
            ):
                raise ValueError(f"{'主对刀' if camera == 'front' else '辅助'}相机识别不可靠：{d.diagnostic}")
            previous_phase = self.phase
            previous_message = self.message
            manual = self._manual(images, ds, (camera,))
            result = manual.get(camera)
            if result is None or result.point_uv is None:
                self.phase = previous_phase if previous_phase in ("live", "success", "saved") else "live"
                self.message = "已取消手动点选" if previous_phase in ("live", "fail") else previous_message
                self.events.emit("baseline_manual_cancelled", tool=self.tool, command=name, camera=camera)
                self.publish()
                return
            self.manual_points[camera] = [float(result.point_uv[0]), float(result.point_uv[1])]
            self.manual_clicks[camera] = {
                "source": result.source,
                "zoom": result.zoom,
                "elapsed_s": result.elapsed_s,
            }
            self.events.emit("baseline_manual_view", tool=self.tool, camera=camera, point_uv=self.manual_points[camera])
            self.message = f"{'正面' if camera == 'front' else '侧面'}已人工确认针尖，请继续调焦至峰值附近"
            self.publish()
        elif name in ("baseline_measure", "baseline_manual", "baseline_aux") and self.phase in ("live", "fail", "success", "saved"):
            self.measurement = None
            if name == "baseline_aux" and (not self.candidate or not self.candidate.get("p0")):
                raise ValueError("请先建立基准")
            cfg = self.config.tool_calibration
            images = self.cameras.execute({"stage": "baseline_measure", "frames_per_view": FRAME_COUNT,
                                           "capture_after": time.monotonic()})
            self.frames = images
            self._archive(images)
            try:
                ds, self.measurement = measure_frames(
                    images.frames, cfg.detection, cfg.frame_std_max_px, rms_max=2.0,
                    capture_times=images.capture_times)
            finally:
                images.frames.clear()  # Keep the archived first frames for display and manual picks.
            manual_pre: dict[str, ClickResult] = {}
            if self.measurement["rejected"]:
                if name == "baseline_measure":
                    # 若仅因已在实时画面人工点选的相机未通过单帧自动质量，且无运动畸变/五帧漂移，则沿用该路人工点。
                    can_use_pre_manual = True
                    for cam in ("front", "side"):
                        view_info = self.measurement["views"][cam]
                        if view_info["accepted"]:
                            continue
                        reasons = view_info["reasons"]
                        has_hard = any(("工具在动或画面畸变" in r or "标准差" in r) for r in reasons)
                        if has_hard or cam not in self.manual_points:
                            can_use_pre_manual = False
                            break
                        info = self.manual_clicks.get(cam) or {}
                        manual_pre[cam] = ClickResult(
                            tuple(self.manual_points[cam]),
                            str(info.get("source", "manual")),
                            float(info.get("zoom", 8.0)),
                            float(info.get("elapsed_s", 0.0)),
                        )
                    if not can_use_pre_manual or not manual_pre:
                        raise ValueError(_format_measure_failure(self.measurement, cfg.detection, getattr(self, "cfg", DEFAULTS)))
                else:
                    target_views = ("side",) if name == "baseline_aux" else ("front", "side")
                    distortion = [
                        reason
                        for cam in target_views
                        for reason in self.measurement["views"][cam]["reasons"]
                        if "工具在动或画面畸变" in reason
                    ]
                    if distortion:
                        raise ValueError("；".join(distortion))
            for camera, view in self.measurement["views"].items():
                if view["accepted"]:
                    ds[camera] = replace(ds[camera], point_uv=tuple(view["median_uv"]),
                                         diagnostic=FRAME_QUALITY_PASSED)
            manual: dict[str, ClickResult] = dict(manual_pre)
            if name != "baseline_measure":
                names: tuple[str, ...] = ("side",) if name == "baseline_aux" else ("front", "side")
                previous_phase = self.phase
                previous_message = self.message
                manual = self._manual(images, ds, names)
                if any(view not in manual or manual[view].point_uv is None for view in names):
                    if name == "baseline_aux" and self.candidate and self.candidate.get("p0"):
                        self.phase = "success" if previous_phase in ("success", "saved") else previous_phase
                        self.message = "已取消辅助相机手动重设"
                    else:
                        self.phase = previous_phase if previous_phase in ("live", "success", "saved") else "live"
                        self.message = "已取消手动点选" if previous_phase in ("live", "fail") else previous_message
                    self._record_measurement(name, "cancelled")
                    self.events.emit("baseline_manual_cancelled", tool=self.tool, command=name)
                    self.publish()
                    return
            self._evaluate(images, ds, manual, name == "baseline_aux")
        elif name == "baseline_save" and self.phase == "success":
            self.phase = "confirm"
            self.message = "请确认覆盖保存"
            self.publish()
        elif name == "baseline_confirm" and self.phase == "confirm":
            accept = params.get("accept") is True
            self.events.emit("baseline_confirm", tool=self.tool, accept=accept)
            if accept:
                self._save()
            else:
                self.phase = "live"
                self.message = "未保存，请点动后重新建立基准"
                self.candidate = None
                self.measurement = None
                self.publish()
        elif name == "baseline_end":
            self.end()

    def _read_exposure_ms(self, name: str) -> float | None:
        """从已打开相机缓存读取当前曝光毫秒数，无开销且安全降级。"""
        cam = (getattr(self.cameras, "cameras", None) or {}).get(name)
        actual = getattr(cam, "actual_settings", None) or {}
        exp = (actual.get("controls") or {}).get("exposure")
        return float(exp) / 10.0 if isinstance(exp, (int, float)) else None

    def _publish_live_focus(self, images: dict[str, np.ndarray], detections: dict[str, Any], now: float, proc_start: float) -> None:
        """计算两路实时清晰度/边缘宽度/轴残差/就绪度并写入 runs/state/tooltip_preview.json。"""
        directory = getattr(self.events, "directory", None)
        fps = round(1 / (now - self._live_last_ts), 1) if self._live_last_ts is not None and now > self._live_last_ts else None
        self._live_last_ts = now
        detection_cfg = getattr(getattr(self.config, "tool_calibration", None), "detection", {}) or {}
        cameras_payload: dict[str, dict[str, Any]] = {}
        updated_at = self.events.clock().isoformat()
        trace_rows: list[list[Any]] = []
        for name in ("front", "side"):
            frame = images[name]
            det = detections[name]
            det_cfg = detection_cfg.get(name, {})
            scale = float(self.cfg["main_um_per_px"] if name == "front" else self.cfg["aux_um_per_px"])
            is_manual = name in self.manual_points
            manual_tip = tuple(self.manual_points[name]) if is_manual else None
            status, reason, metrics, tracked = analyze_tip(
                frame,
                det,
                self.trackers[name],
                scale,
                self.cfg.get("min_focus_contrast", 40),
                det_cfg.get("threshold", 127),
                manual_confirmed=is_manual,
                manual_tip=manual_tip,
            )
            self.history[name] = [(t, v) for t, v in self.history[name] if now - t <= 30.0]
            if tracked["percent"] is not None:
                self.history[name].append((now, tracked["percent"]))
            tip = manual_tip if (is_manual and manual_tip is not None) else det.point_uv
            center = self._target(images["front"]) if name == "front" else None
            delta_uv = [tip[0] - center[0], tip[1] - center[1]] if name == "front" and tip is not None and center is not None else None
            image_size = (frame.shape[1], frame.shape[0])
            readiness = self.readiness_trackers[name].update(
                name,
                {
                    "detection_status": status,
                    "detection_reason": getattr(det, "diagnostic", ""),
                    "tip_uv": list(tip) if tip is not None else None,
                    "image_size": list(image_size),
                    "percent": tracked["percent"],
                    "peak": tracked["best"],
                    "peak_confirmed": tracked.get("peak_confirmed", False),
                    "best_updated": tracked.get("best_updated", False),
                    "peak_confirm_kept": tracked.get("peak_confirm_kept", False),
                    "last_peak_event": tracked.get("last_peak_event", ""),
                    "sample_count": tracked.get("sample_count", 0),
                    "edge_width_px": metrics["edge_width_px"],
                },
                self.cfg,
                manual_confirmed=is_manual,
            )
            payload = build_preview_camera_payload(
                metrics=metrics,
                tracked=tracked,
                status=status,
                reason=reason,
                detection=det,
                tip=tip,
                delta_uv=delta_uv,
                history=[v for _, v in self.history[name]],
                fps=fps,
                scale=scale,
                detector_cfg=det_cfg,
                ui_config=self.cfg,
                updated_at=updated_at,
                exposure_ms=self._read_exposure_ms(name),
                processing_ms=max(0.0, (time.monotonic() - proc_start) * 1000.0),
                image_size=image_size,
                readiness=readiness,
                manual_confirmed=is_manual,
            )
            self.live_snapshots[name] = payload
            cameras_payload[name] = payload
            raw_sharpness = metrics.get("band_gradient") if metrics.get("band_gradient") is not None else tracked.get("raw")
            trace_rows.append([
                updated_at,
                name,
                f"{float(raw_sharpness):.6g}" if isinstance(raw_sharpness, (int, float)) and math.isfinite(raw_sharpness) else "",
                f"{float(tracked['current']):.6g}" if isinstance(tracked.get("current"), (int, float)) and math.isfinite(tracked["current"]) else "",
                f"{float(tracked['best']):.6g}" if isinstance(tracked.get("best"), (int, float)) and math.isfinite(tracked["best"]) else "",
                f"{float(tracked['percent']):.4f}" if isinstance(tracked.get("percent"), (int, float)) and math.isfinite(tracked["percent"]) else "",
                f"{float(metrics['edge_width_px']):.4f}" if isinstance(metrics.get("edge_width_px"), (int, float)) and math.isfinite(metrics["edge_width_px"]) else "",
                f"{float(metrics['contrast']):.4f}" if isinstance(metrics.get("contrast"), (int, float)) and math.isfinite(metrics["contrast"]) else "",
                f"{float(det.axis_rms_px):.4f}" if isinstance(getattr(det, "axis_rms_px", None), (int, float)) and math.isfinite(det.axis_rms_px) else "",
                status,
                int(bool(readiness.get("peak_confirmed"))),
                readiness.get("stage", ""),
                int(bool(readiness.get("ready"))),
                int(bool(tracked.get("best_updated"))),
                int(bool(tracked.get("peak_confirm_kept"))),
                tracked.get("peak_event", ""),
            ])
        if self.record_dir is not None and trace_rows:
            self._append_focus_trace(trace_rows)
        if directory is not None:
            write_tooltip_preview_file(directory, cameras_payload, updated_at)

    def _append_focus_trace(self, rows: list[list[Any]]) -> None:
        """将实时清晰度与越峰状态追加写入本次基准记录目录下的 focus_trace.csv。"""
        if self.record_dir is None or not rows:
            return
        self.record_dir.mkdir(parents=True, exist_ok=True)
        trace_path = self.record_dir / "focus_trace.csv"
        need_header = (not trace_path.exists()) or trace_path.stat().st_size == 0
        with trace_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            if need_header:
                writer.writerow([
                    "timestamp",
                    "camera",
                    "raw_sharpness",
                    "smoothed_sharpness",
                    "peak",
                    "percent",
                    "edge_width_px",
                    "contrast",
                    "axis_rms_px",
                    "detection_status",
                    "peak_confirmed",
                    "peak_stage",
                    "ready",
                    "best_updated",
                    "peak_confirm_kept",
                    "peak_event",
                ])
            writer.writerows(rows)

    def run(self) -> None:
        """后台循环：处理队列命令，并在等待测量时以 ≥5Hz 刷新实时画面与清晰度。"""
        try:
            preparation = getattr(self.events, 'tooltip_preparation', None)
            if preparation: preparation.await_handoff()
            while not self.stop.is_set():
                loop_start = time.monotonic()
                try:
                    command, params = self.queue.get_nowait()
                    try:
                        self._command(command, params)
                    except Exception as exc:
                        if command == "baseline_aux" and self.candidate and self.candidate.get("p0"):
                            self.phase = "success"
                        else:
                            self.phase = "select" if self.tool is None else "fail"
                        self.message = str(exc)
                        self.events.emit("baseline_error", reason=str(exc), tool=self.tool)
                        if command in ("baseline_measure", "baseline_manual", "baseline_aux"):
                            self._record_measurement(command, "fail", str(exc))
                        self.publish()
                except queue.Empty:
                    pass
                if self.tool is not None and not self.stop.is_set() and self.phase in ("live", "fail"):
                    try:
                        images = self._frame("baseline_preview")
                    except GuardError as exc:
                        if exc.code != "CALIB-CAMERA-UNAVAILABLE":
                            raise
                        self.cameras.on_stop()
                        self.tool = None
                        self.phase = "select"
                        self.message = f"未接入：{exc}"
                        self.events.emit("baseline_error", reason=self.message, tool=None)
                        self.publish()
                        continue
                    self.frames = images
                    proc_start = time.monotonic()
                    self.detections = self._detect(images)
                    now = time.monotonic()
                    try:
                        self._publish_live_focus(images, self.detections, now, proc_start)
                    except Exception:
                        pass
                    if now - self._last_live_emit >= 0.5:
                        self._last_live_emit = now
                        d = self.detections["front"]
                        center = self._target(images["front"])
                        delta = (
                            [(d.point_uv[i] - center[i]) * self.cfg["main_um_per_px"] / 1000.0 for i in (0, 1)]
                            if d.point_uv is not None
                            else None
                        )
                        manual_map = {n: n in self.manual_points for n in ("front", "side")}
                        overall = evaluate_baseline_readiness(
                            self.live_snapshots.get("front"),
                            self.live_snapshots.get("side"),
                            config=self.cfg,
                            manual_confirmed=manual_map,
                        )
                        self.events.emit(
                            "baseline_state",
                            live_delta=delta,
                            live_inside=bool(delta is not None and math.hypot(*delta) <= self.cfg["main_tol_mm"]),
                            manual_confirmed=manual_map,
                            live_readiness=overall,
                        )
                    self._annotate(images, self.detections)
                remaining = 0.2 - (time.monotonic() - loop_start)
                if remaining > 0:
                    self.stop.wait(remaining)
        except BaseException as exc:
            self.message = f"相机会话异常：{exc}"
            self.events.emit("baseline_error", reason=self.message, tool=self.tool)
        finally:
            try:
                directory = getattr(self.events, "directory", None)
                if directory is not None:
                    try:
                        write_tooltip_preview_file(directory, {}, self.events.clock().isoformat())
                    except Exception:
                        pass
                self.cameras.on_stop()
            finally:
                self.phase = "done"
                self.closed = True
                record = {
                    "start_time": self.start_time,
                    "end_time": self.events.clock().isoformat(),
                    "tool": self.tool,
                    "message": self.message,
                    "candidate": self.candidate,
                    "saved_path": self.saved_path,
                    "measurements": self.measurements,
                    "commands": self.commands,
                    "shots": self.shots,
                }
                if self.record_dir is not None:
                    try:
                        self.record_dir.mkdir(parents=True, exist_ok=True)
                        (self.record_dir / "session_record.json").write_text(
                            json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False),
                            encoding="utf-8",
                        )
                    except Exception as exc:
                        self.events.emit("baseline_error", reason=f"会话记录写入失败：{exc}")
                self.events.emit("baseline_finished", **record)
                self.publish()
