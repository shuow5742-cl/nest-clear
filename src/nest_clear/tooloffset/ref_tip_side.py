"""对刀相机（front/side）下方标定尖检测与双目标（工具尖+标定尖）叠加诊断。"""
from dataclasses import dataclass
from time import perf_counter
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from nest_clear.baseline.runout import FRAME_COUNT, measure_frames


@dataclass(frozen=True)
class RefTipSideConfig:
    """对刀相机标定尖检测参数；现场实拍图确定后微调。"""

    min_contrast: float = 25.0
    contrast_ratio: float = 0.22
    min_area_px: int = 40
    min_height_px: int = 15
    min_cone_angle_deg: float = 4.0
    max_edge_rms_px: float = 1.5
    border_margin_px: int = 2
    max_tip_offset_px: float = 6.0


@dataclass
class RefTipSideResult:
    """对刀相机单帧标定尖检测结果。"""

    point_uv: tuple[float, float] | None = None
    quality: float = 0.0
    focused: bool = False
    diagnostic: str = ""
    contrast: float | None = None
    edge_rms_px: float | None = None
    cone_angle_deg: float | None = None
    area_px: int | None = None
    flank_segments: tuple[tuple[tuple[float, float], tuple[float, float]], ...] = ()
    elapsed_ms: float = 0.0


def _fit_edge_line(ys: np.ndarray, us: np.ndarray) -> tuple[float, float, float] | None:
    """用 MAD 去外点线性拟合 u = m*y + b。"""
    if len(ys) < 5:
        return None
    y = np.asarray(ys, dtype=float)
    u = np.asarray(us, dtype=float)
    good = np.ones(len(y), dtype=bool)
    for _ in range(3):
        if np.count_nonzero(good) < 5:
            return None
        m, b = np.polyfit(y[good], u[good], 1)
        res = np.abs(u - (m * y + b))
        thr = max(0.4, 3.5 * 1.4826 * float(np.median(res[good])))
        nxt = res <= thr
        if np.array_equal(nxt, good):
            break
        good = nxt
    if np.count_nonzero(good) < 5:
        return None
    m, b = np.polyfit(y[good], u[good], 1)
    rms = float(np.sqrt(np.mean((u[good] - (m * y[good] + b)) ** 2)))
    return float(m), float(b), rms


def _row_crossing(row: np.ndarray, x0: int, x1: int, *, left: bool, min_contrast: float) -> tuple[float, float] | None:
    """在水平灰度行上求单侧亮到暗 50% 亚像素边界。"""
    w = len(row)
    lo = max(0, x0)
    hi = min(w - 1, x1)
    if hi - lo < 6:
        return None
    xs = np.arange(lo, hi + 1, dtype=float)
    vals = row[lo:hi + 1].astype(float)
    if left:
        bg = float(np.median(vals[:5]))
        body = float(np.median(vals[-2:]))
        idx_fn = lambda lvl: np.flatnonzero((vals[:-1] > lvl) & (vals[1:] <= lvl))
    else:
        bg = float(np.median(vals[-5:]))
        body = float(np.median(vals[:2]))
        idx_fn = lambda lvl: np.flatnonzero((vals[:-1] <= lvl) & (vals[1:] > lvl))
    contrast = bg - body
    if contrast < min_contrast:
        return None
    level = body + 0.5 * contrast
    hits = idx_fn(level)
    if not len(hits):
        return None
    mid = 0.5 * (xs[0] + xs[-1])
    j = int(hits[np.argmin(np.abs(xs[hits] - mid))])
    denom = float(vals[j + 1] - vals[j])
    if abs(denom) < 1e-9:
        return None
    frac = float(np.clip((level - vals[j]) / denom, 0.0, 1.0))
    return float(xs[j] + frac), contrast


class RefTipSideDetector:
    """对刀相机标定尖检测器（图像上下翻转后以顶边进入暗色区域定位顶点，再映射回原图坐标）。"""

    def __init__(self, config: RefTipSideConfig | dict | None = None) -> None:
        if isinstance(config, RefTipSideConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = RefTipSideConfig(**{k: v for k, v in config.items() if k in RefTipSideConfig.__dataclass_fields__})
        else:
            self.config = RefTipSideConfig()

    def detect(self, image: np.ndarray) -> RefTipSideResult:
        """检测视野底部朝上的标定锥尖，返回原图亚像素 (u, v) 与诊断指标。"""
        started = perf_counter()

        def done(**kw) -> RefTipSideResult:
            return RefTipSideResult(elapsed_ms=(perf_counter() - started) * 1000.0, **kw)

        arr = np.asarray(image)
        if arr.ndim == 3:
            gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        elif arr.ndim == 2:
            gray = arr
        else:
            raise ValueError("需要二维灰度图或三通道 BGR 图像")
        if gray.dtype != np.uint8 or not gray.size:
            raise ValueError("需要非空 uint8 图像")

        cfg = self.config
        h, w = gray.shape
        # 将标定尖（底部朝上）上下翻转为顶部朝下，天然与顶部进入的工具针杆隔离
        flipped = cv2.flip(gray, 0)
        bg = cv2.dilate(flipped, cv2.getStructuringElement(cv2.MORPH_RECT, (61, 31)))
        diff = bg.astype(np.float32) - flipped.astype(np.float32)
        thr = np.maximum(cfg.min_contrast, cfg.contrast_ratio * bg.astype(np.float32))
        binary = (diff >= thr).astype(np.uint8)
        n_comp, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        choices = [
            idx
            for idx in range(1, n_comp)
            if int(stats[idx, cv2.CC_STAT_TOP]) <= cfg.border_margin_px
            and int(stats[idx, cv2.CC_STAT_TOP] + stats[idx, cv2.CC_STAT_HEIGHT]) < h - 1
            and int(stats[idx, cv2.CC_STAT_AREA]) >= cfg.min_area_px
            and int(stats[idx, cv2.CC_STAT_HEIGHT]) >= cfg.min_height_px
        ]
        if not choices:
            return done(diagnostic="未识别到对刀标定尖：画面底部不存在向上伸入的暗色锥体")
        if len(choices) > 1:
            return done(diagnostic=f"对刀标定尖候选不唯一：底部存在 {len(choices)} 个连通域")

        comp_idx = choices[0]
        area = int(stats[comp_idx, cv2.CC_STAT_AREA])
        comp = labels == comp_idx
        yy, _ = np.nonzero(comp)
        y_top = int(yy.min())
        y_end = int(yy.max())
        length = y_end - y_top
        if length < cfg.min_height_px:
            return done(area_px=area, diagnostic=f"对刀标定尖有效高度不足（{length} px）")

        y_start = y_top + max(1, int(round(0.10 * length)))
        y_stop = y_end - max(3, min(6, int(round(0.20 * length))))
        if y_stop - y_start < 6:
            y_start = y_top + 1
            y_stop = y_end - 1

        rows_l, cols_l, rows_r, cols_r, contrasts = [], [], [], [], []
        step = max(1, (y_stop - y_start) // 35)
        margin = 10
        for y in range(y_start, y_stop + 1, step):
            xs = np.flatnonzero(comp[y])
            if len(xs) < 2:
                continue
            xl0, xr0 = int(xs[0]), int(xs[-1])
            half = max(2, min(6, (xr0 - xl0) // 2))
            row = flipped[y]
            hl = _row_crossing(row, xl0 - margin, xl0 + half, left=True, min_contrast=cfg.min_contrast)
            if hl is not None:
                rows_l.append(y)
                cols_l.append(hl[0])
                contrasts.append(hl[1])
            hr = _row_crossing(row, xr0 - half, xr0 + margin, left=False, min_contrast=cfg.min_contrast)
            if hr is not None:
                rows_r.append(y)
                cols_r.append(hr[0])
                contrasts.append(hr[1])

        fit_l = _fit_edge_line(np.asarray(rows_l), np.asarray(cols_l))
        fit_r = _fit_edge_line(np.asarray(rows_r), np.asarray(cols_r))
        if fit_l is None or fit_r is None:
            return done(area_px=area, diagnostic="对刀标定尖两侧边缘有效采样不足")

        ml, bl, rms_l = fit_l
        mr, br, rms_r = fit_r
        rms = float(math.hypot(rms_l, rms_r) / math.sqrt(2.0))
        angle_deg = float(math.degrees(math.atan(ml) - math.atan(mr)))
        med_contrast = float(np.median(contrasts)) if contrasts else 0.0

        # 锥侧边收敛则精确求两直线交点；若为平行圆柱针杆则回退至沿中心线末端 50% 灰度穿越
        flank_segments: tuple[tuple[tuple[float, float], tuple[float, float]], ...] = ()
        if abs(angle_deg) >= cfg.min_cone_angle_deg:
            if angle_deg <= 0:
                return done(area_px=area, cone_angle_deg=angle_deg, diagnostic=f"对刀标定尖两侧未向尖端收敛（{angle_deg:.1f}°）")
            y_tip_flip = (br - bl) / (ml - mr)
            u_tip = ml * y_tip_flip + bl
            if abs(y_tip_flip - y_end) > cfg.max_tip_offset_px:
                return done(
                    area_px=area,
                    cone_angle_deg=angle_deg,
                    edge_rms_px=rms,
                    contrast=med_contrast,
                    diagnostic=f"对刀标定尖交点与轮廓最远端偏差过大（{y_tip_flip - y_end:+.2f} px）",
                )
            flank_segments = (
                ((ml * y_start + bl, float((h - 1) - y_start)), (u_tip, float((h - 1) - y_tip_flip))),
                ((mr * y_start + br, float((h - 1) - y_start)), (u_tip, float((h - 1) - y_tip_flip))),
            )
        else:
            mc = 0.5 * (ml + mr)
            bc = 0.5 * (bl + br)
            ys_ax = np.arange(max(y_top, y_end - 8), min(h - 1, y_end + 6), dtype=np.float32)
            xs_ax = (mc * ys_ax + bc).astype(np.float32)
            if len(ys_ax) < 6 or xs_ax.min() < 0 or xs_ax.max() > w - 1:
                return done(area_px=area, diagnostic="对刀标定尖平行边缘轴向采样超出视野")
            vals = cv2.remap(flipped, xs_ax.reshape(1, -1), ys_ax.reshape(1, -1), cv2.INTER_LINEAR).ravel().astype(float)
            b_vals = cv2.remap(bg, xs_ax.reshape(1, -1), ys_ax.reshape(1, -1), cv2.INTER_LINEAR).ravel().astype(float)
            body = float(np.median(vals[:max(2, len(vals) // 3)]))
            bg_ax = float(np.median(b_vals[-max(2, len(b_vals) // 3):]))
            c_ax = bg_ax - body
            if c_ax < cfg.min_contrast:
                return done(area_px=area, diagnostic="对刀标定尖末端对比度不足")
            lvl = body + 0.5 * c_ax
            hits = np.flatnonzero((vals[:-1] <= lvl) & (vals[1:] > lvl))
            if not len(hits):
                return done(area_px=area, diagnostic="对刀标定尖末端未检测到灰度穿越")
            j = int(hits[-1])
            denom = float(vals[j + 1] - vals[j])
            frac = float(np.clip((lvl - vals[j]) / denom, 0.0, 1.0)) if abs(denom) > 1e-9 else 0.0
            y_tip_flip = float(ys_ax[j] + frac)
            u_tip = float(mc * y_tip_flip + bc)
            med_contrast = max(med_contrast, c_ax)

        if rms > cfg.max_edge_rms_px:
            return done(
                area_px=area,
                edge_rms_px=rms,
                cone_angle_deg=angle_deg,
                contrast=med_contrast,
                flank_segments=flank_segments,
                diagnostic=f"对刀标定尖侧边残差过大（{rms:.3f} > {cfg.max_edge_rms_px:g} px）",
            )
        if not (0.0 <= u_tip < w and 0.0 <= y_tip_flip < h):
            return done(area_px=area, diagnostic="对刀标定尖顶点超出图像范围")

        v_tip = float((h - 1) - y_tip_flip)
        rms_score = max(0.0, 1.0 - rms / max(cfg.max_edge_rms_px, 1e-6))
        contrast_score = min(1.0, med_contrast / 120.0)
        quality = float(np.clip(0.6 * rms_score + 0.4 * contrast_score, 0.05, 1.0))
        return done(
            point_uv=(float(u_tip), v_tip),
            quality=quality,
            focused=True,
            diagnostic="对刀相机标定尖识别成功",
            contrast=med_contrast,
            edge_rms_px=rms,
            cone_angle_deg=angle_deg,
            area_px=area,
            flank_segments=flank_segments,
        )


def measure_ref_tip_frames(frames: dict, config: RefTipSideConfig | dict | None, *, std_max: float = 0.3) -> tuple[dict, dict]:
    """对正面与侧面各 5 帧图像做标定尖检测、质量门限及中位数汇总。"""
    detector = RefTipSideDetector(config)
    representatives, views, reasons = {}, {}, []
    for camera, label in (("front", "正面"), ("side", "侧面")):
        view_frames = list(frames.get(camera, ()))
        records, view_reasons = [], []
        for idx, frame in enumerate(view_frames, 1):
            res = detector.detect(frame)
            if idx == 1:
                representatives[camera] = res
            reason = "" if res.point_uv is not None and res.focused else (res.diagnostic or "标定尖未识别")
            if reason:
                view_reasons.append(f"{label}标定尖第 {idx} 帧{reason}")
            records.append({
                "index": idx,
                "uv": list(map(float, res.point_uv)) if res.point_uv is not None else None,
                "quality": res.quality,
                "focused": res.focused,
                "edge_rms_px": res.edge_rms_px,
                "cone_angle_deg": res.cone_angle_deg,
                "contrast": res.contrast,
                "diagnostic": res.diagnostic,
                "reason": reason,
            })
        std = median = None
        if len(records) != FRAME_COUNT:
            view_reasons.append(f"{label}标定尖应采集 {FRAME_COUNT} 帧，实际 {len(records)} 帧")
        elif all(row["uv"] is not None for row in records):
            pts = np.asarray([row["uv"] for row in records], dtype=float)
            std = np.std(pts, axis=0).tolist()
            median = np.median(pts, axis=0).tolist()
            for axis_name, spread in zip(("u", "v"), std):
                if spread > std_max:
                    view_reasons.append(f"{label}标定尖五帧 {axis_name} 标准差 {spread:.3f} > {std_max:g} px")
        reasons.extend(view_reasons)
        views[camera] = {
            "frames": records,
            "std_uv": std,
            "median_uv": median,
            "accepted": not view_reasons,
            "reasons": view_reasons,
        }
    return representatives, {"views": views, "std_max_px": std_max, "rejected": bool(reasons), "reason": "；".join(reasons)}


def measure_dual_tips(frames: dict, tool_detection_cfg: dict, ref_cfg: RefTipSideConfig | dict | None, *, std_max: float = 0.3, capture_times: dict | None = None) -> dict:
    """同时检测五帧中位工具尖（Trident）与五帧中位标定尖，任一失败均拒收。"""
    tool_reps, tool_meas = measure_frames(frames, tool_detection_cfg, std_max=std_max, capture_times=capture_times)
    ref_reps, ref_meas = measure_ref_tip_frames(frames, ref_cfg, std_max=std_max)
    reasons = [r for r in (tool_meas.get("reason"), ref_meas.get("reason")) if r]
    tool_pts = {cam: tuple(tool_meas["views"][cam]["median_uv"]) for cam in ("front", "side") if tool_meas["views"][cam]["median_uv"]}
    ref_pts = {cam: tuple(ref_meas["views"][cam]["median_uv"]) for cam in ("front", "side") if ref_meas["views"][cam]["median_uv"]}
    return {
        "ok": not reasons and len(tool_pts) == 2 and len(ref_pts) == 2,
        "reason": "；".join(reasons),
        "tool_points": tool_pts,
        "ref_points": ref_pts,
        "tool_reps": tool_reps,
        "ref_reps": ref_reps,
        "tool_measurement": tool_meas,
        "ref_measurement": ref_meas,
    }


def _load_font(size: int):
    from pathlib import Path

    for p in (
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
    ):
        if p.is_file():
            try:
                return ImageFont.truetype(str(p), size)
            except OSError:
                continue
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def render_tooloffset_side_overlay(
    image: np.ndarray,
    view: str,
    *,
    tool_uv: tuple[float, float] | None = None,
    ref_uv: tuple[float, float] | None = None,
    ref_result: RefTipSideResult | None = None,
    z_dir_uv: tuple[float, float] | None = None,
    delta_mm: float | None = None,
    gap_mm: float | None = None,
    status_text: str = "",
) -> np.ndarray:
    """绘制对刀相机工具尖、标定尖与过工具尖 Z 方向直线叠加图。"""
    arr = np.asarray(image)
    out = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR) if arr.ndim == 2 else arr.copy()
    h, w = out.shape[:2]

    if ref_result is not None:
        for p0, p1 in ref_result.flank_segments:
            cv2.line(out, (int(round(p0[0])), int(round(p0[1]))), (int(round(p1[0])), int(round(p1[1]))), (255, 200, 0), 1, cv2.LINE_AA)

    if tool_uv is not None:
        tx, ty = float(tool_uv[0]), float(tool_uv[1])
        # 画过工具尖沿 Z 方向的像素直线（默认垂直向下）
        dx, dy = z_dir_uv if z_dir_uv is not None else (0.0, 1.0)
        norm = math.hypot(dx, dy)
        if norm > 1e-9:
            ux, uy = dx / norm, dy / norm
            span = max(w, h)
            p_a = (int(round(tx - span * ux)), int(round(ty - span * uy)))
            p_b = (int(round(tx + span * ux)), int(round(ty + span * uy)))
            cv2.line(out, p_a, p_b, (255, 190, 0), 2, cv2.LINE_AA)
        cv2.drawMarker(out, (int(round(tx)), int(round(ty))), (0, 190, 60), cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)

    if ref_uv is not None:
        rx, ry = int(round(ref_uv[0])), int(round(ref_uv[1]))
        cv2.circle(out, (rx, ry), 7, (0, 60, 255), 2, cv2.LINE_AA)
        cv2.drawMarker(out, (rx, ry), (0, 60, 255), cv2.MARKER_TILTED_CROSS, 18, 2, cv2.LINE_AA)
        if tool_uv is not None:
            cv2.line(out, (int(round(tool_uv[0])), int(round(tool_uv[1]))), (rx, ry), (210, 80, 220), 1, cv2.LINE_AA)

    rgb = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(rgb)
    font = _load_font(max(16, round(w * 28 / 640)))
    axis_label = "ΔX" if view == "front" else "ΔY"
    title = "正面视图" if view == "front" else "侧面视图"
    parts = [title]
    if delta_mm is not None:
        parts.append(f"{axis_label}={delta_mm:+.3f}mm")
    if gap_mm is not None:
        parts.append(f"间隙={gap_mm:.2f}mm")
    if status_text:
        parts.append(status_text)
    lines = [parts[0]]
    for part in parts[1:]:
        line = f"{lines[-1]} | {part}"
        if draw.textlength(line, font=font) <= w - 16:
            lines[-1] = line
        else:
            lines.append(part)
    text = "\n".join(lines)
    _, _, _, bottom = draw.multiline_textbbox((8, 4), text, font=font)
    draw.rectangle((0, 0, w, bottom + 4), fill=(255, 255, 255))
    draw.multiline_text((8, 4), text, font=font, fill=(25, 35, 50))
    return cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
