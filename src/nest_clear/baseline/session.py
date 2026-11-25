"""界面驱动的对刀基准会话；设备采图及操作均在独立线程执行。"""

from datetime import datetime
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
from nest_clear.tooltip.click import ClickResult, confirm_tip
from nest_clear.tooltip.schema import load_baseline
from nest_clear.tooltip.store import load_tools

DEFAULTS: dict[str, Any] = {
    "main_target_uv": None,
    "main_tol_mm": 0.05,
    "main_um_per_px": 15.2,
    "aux_um_per_px": 15.2,
    "min_focus_contrast": 40,
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
    for key in ("main_tol_mm", "main_um_per_px", "aux_um_per_px", "min_focus_contrast"):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} 必须为有限正数")
    target = cfg["main_target_uv"]
    if target is not None and (
        not isinstance(target, (tuple, list))
        or len(target) != 2
        or any(type(x) not in (int, float) or not math.isfinite(x) for x in target)
    ):
        raise ValueError("main_target_uv 必须为两个有限数字或 null")
    return cfg


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
        views.append(item)
    return {
        "version": 1,
        "tool_name": baseline.tool_name,
        "p0": list(baseline.p0),
        "offset_xyz": list(baseline.offset_xyz),
        "timestamp": baseline.timestamp,
        "note": baseline.note,
        "jacobian_pending": True,
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


def summary(path: str | Path, stale: bool = False) -> dict[str, Any]:
    """读取单把工具现有基准的简要信息，占位或缺失时标记 missing。"""
    try:
        b = load_baseline(path)
        return {
            "time": b.timestamp,
            "p0": list(b.p0),
            "pixels": {v.name: list(v.target_uv) for v in b.views},
            "jacobian_pending": b.jacobian_pending or any(v.jacobian is None for v in b.views),
            "stale": bool(stale),
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
        item = summary(cal_dir / f"{key}_baseline.yaml", bool(entry.get("baseline_stale")))
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


def refresh_existing(ctx: Any) -> None:
    """其他标定更新 tool_offset 后刷新对刀基准页摘要，不影响原流程。"""
    try:
        cal_path = Path(getattr(ctx, "calibration_path", None) or ctx.config.vision.calibration)
        ctx.events.emit("baseline_state", existing=existing_baselines(ctx.config, cal_path))
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
        self.measure_frames: dict[str, np.ndarray] | None = None
        self.detections: dict[str, Any] = {}
        self.closed = False
        self.stop = threading.Event()
        self.commands: list[dict[str, Any]] = []
        self.measurements: list[dict[str, Any]] = []
        self.saved_path: str | None = None
        self.shots = 0
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
        """对两路图像运行现有先粗后细剪影针尖检测。"""
        return {
            name: SilhouetteTipDetector(**self.config.tool_calibration.detection[name]).detect(images[name])
            for name in ("front", "side")
        }

    def _target(self, image: np.ndarray) -> tuple[float, float]:
        """返回主对刀相机容许圆中心像素坐标，默认图像中心。"""
        target = self.cfg["main_target_uv"]
        if target is not None:
            return (float(target[0]), float(target[1]))
        return ((image.shape[1] - 1) / 2.0, (image.shape[0] - 1) / 2.0)

    def _annotate(self, images: dict[str, np.ndarray], detections: dict[str, Any]) -> None:
        """在主/辅助整幅画面及右侧放大小窗绘制基准中心、容许圆、针尖、轮廓及中文状态。"""
        for name in ("front", "side"):
            raw = images[name]
            base_bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR) if raw.ndim == 2 else raw.copy()
            h, w = base_bgr.shape[:2]
            d = detections.get(name)
            scale_um = float(self.cfg["main_um_per_px"] if name == "front" else self.cfg["aux_um_per_px"])
            tol_px = float(self.cfg["main_tol_mm"]) * 1000.0 / scale_um
            crop_size = max(60, int(math.ceil(tol_px * 3.0)))
            if crop_size % 2:
                crop_size += 1
            inset_size = max(280, round(h * 0.625))
            zoom = inset_size / crop_size

            has_target = False
            roi_label_cn = ""
            roi_label_en = ""
            if name == "front":
                roi_center = self._target(raw)
                has_target = True
                roi_label_cn = "基准中心"
                roi_label_en = "Target"
            elif self.candidate and self.candidate.get("passed"):
                pts = self.candidate["pixels"]["side"]
                roi_center = (float(pts[0]), float(pts[1]))
                has_target = True
                roi_label_cn = "辅助基准点（本次建立）"
                roi_label_en = "Aux Target"
            elif d and d.point_uv is not None:
                roi_center = (float(d.point_uv[0]), float(d.point_uv[1]))
            else:
                roi_center = ((w - 1) / 2.0, (h - 1) / 2.0)

            tip_uv = tuple(float(x) for x in d.point_uv) if d and d.point_uv is not None else None
            manual_uv = (
                tuple(float(x) for x in self.candidate["pixels"][name])
                if self.candidate and name in self.candidate.get("click", {})
                else None
            )
            eval_uv = manual_uv if manual_uv is not None else tip_uv
            inside = False
            if has_target and eval_uv is not None:
                inside = math.dist(eval_uv, roi_center) * scale_um / 1000.0 <= float(self.cfg["main_tol_mm"])
            circle_color = (0, 175, 0) if (name == "side" and has_target) or inside else (0, 0, 230)

            # 放大图取原图加轮廓，避免整幅图大十字线遮挡细节。
            zoom_src = base_bgr.copy()
            mask = (raw <= self.config.tool_calibration.detection[name]["threshold"]).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                cv2.drawContours(base_bgr, [largest], -1, (255, 120, 0), 1)
                cv2.drawContours(zoom_src, [largest], -1, (255, 120, 0), 1)

            labels: list[tuple[tuple[int, int], str, str, tuple[int, int, int], int]] = []
            if has_target:
                cx, cy = round(roi_center[0]), round(roi_center[1])
                cross_color = (220, 0, 220)
                cv2.line(base_bgr, (0, cy), (w - 1, cy), cross_color, 1)
                cv2.line(base_bgr, (cx, 0), (cx, h - 1), cross_color, 1)
                half = crop_size // 2
                x0, y0 = max(0, cx - half), max(0, cy - half)
                x1, y1 = min(w - 1, cx + half), min(h - 1, cy + half)
                cv2.rectangle(base_bgr, (x0, y0), (x1, y1), cross_color, 2)
                cv2.circle(base_bgr, (cx, cy), max(2, round(tol_px)), circle_color, 1, cv2.LINE_AA)
                labels.append(((min(w - 90, x0 + 2), max(4, y0 - 22)), roi_label_cn, roi_label_en, cross_color, 15))

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
                if eval_uv is not None:
                    dx, dy = ((eval_uv[i] - roi_center[i]) * scale_um / 1000.0 for i in (0, 1))
                    labels.append(((12, 8), f"横向偏差 {dx:+.3f}mm  竖向偏差 {dy:+.3f}mm", f"dx {dx:+.3f}mm dy {dy:+.3f}mm", circle_color, 18))
                else:
                    labels.append(((12, 8), "等待识别针尖", "Waiting tip", (0, 0, 220), 18))

            if d and d.point_uv is not None:
                edge_text = f"{d.edge_width_px:.2f}px" if d.edge_width_px is not None else "--"
                focus_cn = "清晰" if d.focused else "模糊"
                coarse_cn = " 粗定位" if d.coarse else ""
                focus_color = (0, 155, 0) if d.focused and not d.coarse else (0, 0, 220)
                labels.append((
                    (12, 34 if name == "front" else 10),
                    f"可靠度 {d.confidence:.2f}  边缘 {edge_text}  {focus_cn}{coarse_cn}",
                    f"conf {d.confidence:.2f} edge {edge_text} {'OK' if d.focused else 'FAIL'}",
                    focus_color,
                    15,
                ))
            else:
                labels.append((
                    (12, 34 if name == "front" else 10),
                    "未检测到针尖（模糊或无目标）",
                    "NO TIP / focus FAIL",
                    (0, 0, 220),
                    15,
                ))

            # 右侧放大小窗：裁剪约 60x60 放大至约 300px；辅助相机建立成功前仅放大针尖，不画中心十字或圆圈。
            matrix = np.array(
                [
                    [zoom, 0.0, inset_size / 2.0 - roi_center[0] * zoom],
                    [0.0, zoom, inset_size / 2.0 - roi_center[1] * zoom],
                ],
                dtype=np.float32,
            )
            zoomed = cv2.warpAffine(zoom_src, matrix, (inset_size, inset_size), flags=cv2.INTER_NEAREST, borderValue=(235, 235, 235))
            if has_target:
                ic = inset_size // 2
                cv2.line(zoomed, (0, ic), (inset_size - 1, ic), (220, 0, 220), 1)
                cv2.line(zoomed, (ic, 0), (ic, inset_size - 1), (220, 0, 220), 1)
                zoom_radius = max(3, round(tol_px * zoom))
                cv2.circle(zoomed, (inset_size // 2, inset_size // 2), zoom_radius, circle_color, 2, cv2.LINE_AA)

            out_of_inset = False
            if tip_uv is not None:
                zx = (tip_uv[0] - roi_center[0]) * zoom + inset_size / 2.0
                zy = (tip_uv[1] - roi_center[1]) * zoom + inset_size / 2.0
                if 0 <= zx < inset_size and 0 <= zy < inset_size:
                    cv2.circle(zoomed, (round(zx), round(zy)), 6, (0, 0, 0), -1, cv2.LINE_AA)
                    cv2.circle(zoomed, (round(zx), round(zy)), 5, (0, 220, 255), -1, cv2.LINE_AA)
                else:
                    out_of_inset = True
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

            if has_target:
                title_cn = f"{'基准中心' if name == 'front' else '辅助基准点'}放大 {zoom:.1f}× ({crop_size}×{crop_size}px)"
            else:
                title_cn = f"针尖局部放大 {zoom:.1f}× ({crop_size}×{crop_size}px)"
            title_en = f"Zoom {zoom:.1f}x ({crop_size}x{crop_size}px)"
            labels.append(((ix, 8), title_cn, title_en, (30, 41, 59), 15))

            ly = iy + inset_size + 8
            if has_target:
                labels.append(((ix, ly), f"品红十字：{roi_label_cn}  圆圈：{self.cfg['main_tol_mm']:.2f}mm", f"Cross: {roi_label_en}  Circle: {self.cfg['main_tol_mm']:.2f}mm", (60, 70, 85), 13))
                labels.append(((ix, ly + 22), "黄实心点：自动检测  紫实心点：人工确认", "Yellow: Auto  Purple: Manual", (60, 70, 85), 13))
                if out_of_inset:
                    labels.append(((ix, ly + 44), "状态：针尖超出放大窗范围", "Status: Tip outside zoom", (0, 0, 220), 14))
                elif name == "front":
                    labels.append(((ix, ly + 44), "状态：针尖在容许圆内" if inside else "状态：针尖超出容许圆", "Status: Inside" if inside else "Status: Outside", circle_color, 14))
                else:
                    labels.append(((ix, ly + 44), "状态：辅助基准范围已建立", "Status: Aux baseline set", (0, 155, 0), 14))
            else:
                labels.append(((ix, ly), "黄实心点：自动检测  紫实心点：人工确认", "Yellow: Auto  Purple: Manual", (60, 70, 85), 13))

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
        }
        self.measurements.append(entry)
        self.events.emit("baseline_measure", tool=self.tool, mode=mode, outcome=outcome, reason=reason, candidate=self.candidate)

    def _evaluate(self, images: dict[str, np.ndarray], ds: dict[str, Any], manual: dict[str, ClickResult] | None = None, aux_only: bool = False) -> None:
        """判定检测/点选、主相机容许圆、清晰度及 P0 稳定性并刷新标注图。"""
        points: dict[str, list[float] | None] = {
            name: list(d.point_uv) if d.point_uv is not None else None
            for name, d in ds.items()
        }
        clicked: dict[str, dict[str, Any]] = {}
        for name, result in (manual or {}).items():
            if result.point_uv is None:
                raise ValueError("已取消手动点选")
            points[name] = [float(result.point_uv[0]), float(result.point_uv[1])]
            clicked[name] = {"source": result.source, "zoom": result.zoom, "elapsed_s": result.elapsed_s}
        focus = {
            name: {
                "focused": d.focused,
                "edge_width_px": d.edge_width_px,
                "width_px": d.width_px,
                "coarse": d.coarse,
                "confidence": d.confidence,
                "diagnostic": d.diagnostic,
            }
            for name, d in ds.items()
        }
        self.detections = ds
        if aux_only:
            if not self.candidate or not self.candidate.get("p0"):
                raise ValueError("请先建立主相机基准")
            self.candidate["pixels"]["side"] = points["side"]
            self.candidate["click"]["side"] = clicked["side"]
            self.candidate["focus"]["side"] = focus["side"]
            self.candidate["passed"] = True
            self._annotate(images, ds)
            self.phase = "success"
            self.message = "辅助相机基准已重设，请保存"
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
            "click": clicked,
            "p0": None,
        }
        self._annotate(images, ds)
        mode = "manual" if clicked else "auto"
        for name in ("front", "side"):
            label = "主对刀相机" if name == "front" else "辅助相机"
            if points[name] is None:
                raise ValueError(f"{label}无法识别针尖，检查背光和遮挡")
            if ds[name].coarse and name not in clicked:
                raise ValueError(f"{label}识别不可靠，请手动确认")
            if not ds[name].focused and name not in clicked:
                raise ValueError(f"{label}针尖模糊，检查对焦/背光：{ds[name].diagnostic}")
        if not inside:
            assert delta is not None
            self.phase = "fail"
            self.message = format_outside_message(delta)
            self._record_measurement(mode, "fail", self.message)
            self.publish()
            return
        self.candidate["p0"] = self._stable_position()
        self.candidate["passed"] = True
        self._annotate(images, ds)
        self.phase = "success"
        self.message = "基准建立成功（待保存）"
        self._record_measurement(mode, "success")
        self.publish()

    def _save(self) -> None:
        """覆盖保存当前工具基准并备份历史，同时清除 tool_offset 的 baseline_stale。"""
        c = self.candidate
        if not c or not c.get("p0") or not c.get("inside"):
            raise ValueError("没有可保存的基准")
        directory = Path(self.config.tool_calibration.directory)
        offset = load_tools(directory / "tools.yaml")["tools"][f"tool{self.tool}"]["offset_xyz"]
        views = [
            ViewCalib(
                name,
                c["pixels"][name],
                None,
                c["focus"][name].get("width_px"),
                True if name in c["click"] else None,
            )
            for name in ("front", "side")
        ]
        baseline = ToolTipBaseline(
            f"tool{self.tool}",
            c["p0"],
            views,
            offset,
            datetime.now().astimezone().isoformat(),
            "界面建立，雅可比待测",
            True,
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
        self.events.emit("baseline_saved", tool=self.tool, path=str(path), p0=c["p0"], offset_xyz=list(offset))
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
            self.tool = tool
            self.notice = self._selection_notice(tool, self.existing())
            self.phase = "live"
            self.message = "请点动针尖到主相机中心后按【对刀基准】"
            self.events.emit("baseline_select", tool=tool, notice=self.notice)
            self.publish()
        elif name in ("baseline_measure", "baseline_manual", "baseline_aux") and self.phase in ("live", "fail", "success", "saved"):
            if name == "baseline_aux":
                if not self.candidate or not self.candidate.get("p0"):
                    raise ValueError("请先建立基准")
                images = self.measure_frames if self.measure_frames is not None else self._frame("baseline_measure")
                self._archive(images)
            else:
                images = self._frame("baseline_measure")
                self.frames = images
                self.measure_frames = images
                self._archive(images)
            ds = self._detect(images)
            manual: dict[str, ClickResult] = {}
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
                self.measure_frames = None
                self.publish()
        elif name == "baseline_end":
            self.end()

    def run(self) -> None:
        """后台循环：处理队列命令，并在等待测量时以约 5fps 刷新实时画面。"""
        last_detect = 0.0
        try:
            preparation = getattr(self.events, 'tooltip_preparation', None)
            if preparation: preparation.await_handoff()
            while not self.stop.is_set():
                try:
                    command, params = self.queue.get_nowait()
                    try:
                        self._command(command, params)
                    except Exception as exc:
                        self.phase = "select" if self.tool is None else "fail"
                        self.message = str(exc)
                        self.events.emit("baseline_error", reason=str(exc), tool=self.tool)
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
                    if time.monotonic() - last_detect >= 0.5:
                        self.detections = self._detect(images)
                        last_detect = time.monotonic()
                        d = self.detections["front"]
                        center = self._target(images["front"])
                        delta = (
                            [(d.point_uv[i] - center[i]) * self.cfg["main_um_per_px"] / 1000.0 for i in (0, 1)]
                            if d.point_uv is not None
                            else None
                        )
                        self.events.emit(
                            "baseline_state",
                            live_delta=delta,
                            live_inside=bool(delta is not None and math.hypot(*delta) <= self.cfg["main_tol_mm"]),
                        )
                    self._annotate(images, self.detections)
                self.stop.wait(0.2)
        except BaseException as exc:
            self.message = f"相机会话异常：{exc}"
            self.events.emit("baseline_error", reason=self.message, tool=self.tool)
        finally:
            try:
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
