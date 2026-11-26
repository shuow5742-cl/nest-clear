"""主相机斜上方标定锥尖亚像素检测与诊断图渲染。"""
from dataclasses import dataclass
from time import perf_counter
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class RefTipTopConfig:
    """主相机标定尖检测参数；现场实拍图确定后微调。"""

    bg_kernel_px: int = 101
    min_contrast: float = 25.0
    contrast_ratio: float = 0.18
    min_area_px: int = 150
    max_area_px: int = 500000
    min_apex_angle_deg: float = 5.0
    max_apex_angle_deg: float = 120.0
    max_edge_rms_px: float = 1.0
    max_tip_offset_px: float = 8.0
    tip_skip_px: int = 4
    base_trim_frac: float = 0.25


@dataclass
class RefTipTopResult:
    """标定尖单帧检测结果与质量指标。"""

    point_uv: tuple[float, float] | None = None
    quality: float = 0.0
    focused: bool = False
    diagnostic: str = ""
    contrast: float | None = None
    edge_rms_px: float | None = None
    apex_angle_deg: float | None = None
    area_px: int | None = None
    flank_segments: tuple[tuple[tuple[float, float], tuple[float, float]], ...] = ()
    contour: np.ndarray | None = None
    elapsed_ms: float = 0.0


def _fit_flank(s_vals: np.ndarray, w_vals: np.ndarray) -> tuple[float, float, np.ndarray, float] | None:
    """用鲁棒最小二乘拟合单侧锥边 w = m*s + b。"""
    if len(s_vals) < 6:
        return None
    s = np.asarray(s_vals, dtype=float)
    w = np.asarray(w_vals, dtype=float)
    good = np.ones(len(s), dtype=bool)
    for _ in range(3):
        if np.count_nonzero(good) < 6:
            return None
        m, b = np.polyfit(s[good], w[good], 1)
        res = np.abs(w - (m * s + b))
        mad = float(np.median(res[good]))
        thr = max(0.4, 3.5 * 1.4826 * mad)
        new_good = res <= thr
        if np.array_equal(new_good, good):
            break
        good = new_good
    if np.count_nonzero(good) < 6:
        return None
    m, b = np.polyfit(s[good], w[good], 1)
    rms = float(np.sqrt(np.mean((w[good] - (m * s[good] + b)) ** 2)))
    return float(m), float(b), good, rms


def _subpixel_crossing(vals: np.ndarray, t_coords: np.ndarray, *, inner_left: bool, min_contrast: float) -> tuple[float, float] | None:
    """在一维灰度剖面中求局部背景与暗区中点的亚像素穿越位置。"""
    if len(vals) < 12:
        return None
    if inner_left:
        bg = float(np.median(vals[-6:]))
        body = float(np.median(vals[:4]))
        # 寻找从内(暗)到外(亮)的穿越
        rising = True
    else:
        bg = float(np.median(vals[:6]))
        body = float(np.median(vals[-4:]))
        # 寻找从外(亮)到内(暗)的穿越
        rising = False
    contrast = bg - body
    if contrast < min_contrast:
        return None
    level = body + 0.5 * contrast
    if rising:
        idx = np.flatnonzero((vals[:-1] <= level) & (vals[1:] > level))
    else:
        idx = np.flatnonzero((vals[:-1] > level) & (vals[1:] <= level))
    if not len(idx):
        return None
    # 选最接近剖面中段（预期边缘位置）的穿越点
    mid = 0.5 * (t_coords[0] + t_coords[-1])
    j = int(idx[np.argmin(np.abs(t_coords[idx] - mid))])
    denom = float(vals[j + 1] - vals[j])
    if abs(denom) < 1e-9:
        return None
    frac = float(np.clip((level - vals[j]) / denom, 0.0, 1.0))
    return float(t_coords[j] + frac * (t_coords[j + 1] - t_coords[j])), contrast


class RefTipTopDetector:
    """主相机斜视标定锥尖亚像素检测器。"""

    def __init__(self, config: RefTipTopConfig | dict | None = None) -> None:
        if isinstance(config, RefTipTopConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = RefTipTopConfig(**{k: v for k, v in config.items() if k in RefTipTopConfig.__dataclass_fields__})
        else:
            self.config = RefTipTopConfig()

    def detect(self, image: np.ndarray) -> RefTipTopResult:
        """从白/灰背景的斜视锥体图像中定位锥尖顶点并评估质量。"""
        started = perf_counter()

        def done(**kw) -> RefTipTopResult:
            return RefTipTopResult(elapsed_ms=(perf_counter() - started) * 1000.0, **kw)

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
        scale = 4 if min(h, w) >= 400 else 1
        if scale > 1:
            small = cv2.resize(gray, (max(8, w // scale), max(8, h // scale)), interpolation=cv2.INTER_AREA)
            k = max(3, (cfg.bg_kernel_px // scale) | 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            bg_small = cv2.morphologyEx(small, cv2.MORPH_CLOSE, kernel)
            bg = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            k = max(3, cfg.bg_kernel_px | 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

        diff = bg.astype(np.float32) - gray.astype(np.float32)
        thr = np.maximum(cfg.min_contrast, cfg.contrast_ratio * bg.astype(np.float32))
        binary = (diff >= thr).astype(np.uint8)
        n_comp, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        candidates = []
        for idx in range(1, n_comp):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if cfg.min_area_px <= area <= cfg.max_area_px:
                candidates.append((idx, area))
        if not candidates:
            return done(diagnostic="主相机未识别到标定尖：未找到足够对比度的暗色锥体")
        if len(candidates) > 1:
            return done(diagnostic=f"主相机标定尖候选不唯一：检出 {len(candidates)} 个候选区域")

        comp_idx, area = candidates[0]
        bx, by, bw, bh, _ = [int(v) for v in stats[comp_idx]]
        pad = 24
        x0, y0 = max(0, bx - pad), max(0, by - pad)
        x1, y1 = min(w, bx + bw + pad), min(h, by + bh + pad)
        crop_gray = gray[y0:y1, x0:x1].astype(np.float32)
        crop_mask = (labels[y0:y1, x0:x1] == comp_idx).astype(np.uint8)
        contours, _ = cv2.findContours(crop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = (max(contours, key=cv2.contourArea).reshape(-1, 2) + np.array([x0, y0])) if contours else None

        ys, xs = np.nonzero(crop_mask)
        pts = np.column_stack((xs, ys)).astype(float)
        center = pts.mean(axis=0)
        cov = np.cov((pts - center).T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        u1 = eigvecs[:, int(np.argmax(eigvals))]
        u1 = u1 / np.linalg.norm(u1)
        u2 = np.array([-u1[1], u1[0]])

        s_proj = (pts - center) @ u1
        w_proj = (pts - center) @ u2
        s_min, s_max = float(s_proj.min()), float(s_proj.max())
        length = s_max - s_min
        if length < 15.0:
            return done(area_px=area, contour=contour, diagnostic=f"标定尖候选长度过短（{length:.1f} px）")

        # 比较两端靠内侧 5%–25% 区的截面宽度，较窄一端为锥尖方向
        low_band = (s_proj >= s_min + 0.05 * length) & (s_proj <= s_min + 0.25 * length)
        high_band = (s_proj >= s_max - 0.25 * length) & (s_proj <= s_max - 0.05 * length)
        w_low = float(np.ptp(w_proj[low_band])) if np.any(low_band) else float("inf")
        w_high = float(np.ptp(w_proj[high_band])) if np.any(high_band) else float("inf")
        d = u1 if w_high <= w_low else -u1
        n = np.array([-d[1], d[0]])

        s_proj = (pts - center) @ d
        w_proj = (pts - center) @ n
        s_base, s_apex = float(s_proj.min()), float(s_proj.max())
        apex_mask = s_proj >= s_apex - 1.5
        w_apex_est = float(np.median(w_proj[apex_mask])) if np.any(apex_mask) else 0.0

        tip_skip = min(float(cfg.tip_skip_px), 0.15 * length)
        s_start = s_base + cfg.base_trim_frac * length
        s_stop = s_apex - tip_skip
        if s_stop - s_start < 8.0:
            s_start = s_base + 0.15 * length
            s_stop = s_apex - max(1.0, min(2.0, 0.1 * length))
        if s_stop - s_start < 6.0:
            return done(area_px=area, contour=contour, diagnostic="标定尖有效直线侧边长度不足")

        n_scan = int(min(80, max(20, round(s_stop - s_start))))
        scan_s = np.linspace(s_start, s_stop, n_scan)
        # 在裁剪 mask 中按 (s, w) 分桶找每条扫描线左右边界
        s_left, w_left, s_right, w_right, contrasts = [], [], [], [], []
        ch, cw = crop_gray.shape
        margin_t = 12.0
        for s_val in scan_s:
            near = np.abs(s_proj - s_val) <= 1.2
            if np.count_nonzero(near) < 2:
                continue
            ws_near = w_proj[near]
            wl0, wr0 = float(ws_near.min()), float(ws_near.max())
            if wr0 - wl0 < 2.0:
                continue
            half_span = min(8.0, 0.5 * (wr0 - wl0))
            if half_span < 2.5:
                continue
            # 左侧剖面：从外 (+bg) 到内 (中心线)，右侧剖面：从内 (中心线) 到外 (+bg)
            tl = np.arange(wl0 - margin_t, wl0 + half_span + 0.01, 0.5, dtype=np.float32)
            tr = np.arange(wr0 - half_span, wr0 + margin_t + 0.01, 0.5, dtype=np.float32)
            base_pt = center + s_val * d
            xl = (base_pt[0] + tl * n[0]).astype(np.float32).reshape(1, -1)
            yl = (base_pt[1] + tl * n[1]).astype(np.float32).reshape(1, -1)
            xr = (base_pt[0] + tr * n[0]).astype(np.float32).reshape(1, -1)
            yr = (base_pt[1] + tr * n[1]).astype(np.float32).reshape(1, -1)
            if xl.min() >= 0 and xl.max() <= cw - 1 and yl.min() >= 0 and yl.max() <= ch - 1:
                vl = cv2.remap(crop_gray, xl, yl, cv2.INTER_LINEAR).ravel()
                hit = _subpixel_crossing(vl, tl, inner_left=False, min_contrast=cfg.min_contrast)
                if hit is not None:
                    s_left.append(s_val)
                    w_left.append(hit[0])
                    contrasts.append(hit[1])
            if xr.min() >= 0 and xr.max() <= cw - 1 and yr.min() >= 0 and yr.max() <= ch - 1:
                vr = cv2.remap(crop_gray, xr, yr, cv2.INTER_LINEAR).ravel()
                hit = _subpixel_crossing(vr, tr, inner_left=True, min_contrast=cfg.min_contrast)
                if hit is not None:
                    s_right.append(s_val)
                    w_right.append(hit[0])
                    contrasts.append(hit[1])

        fit_l = _fit_flank(np.asarray(s_left), np.asarray(w_left))
        fit_r = _fit_flank(np.asarray(s_right), np.asarray(w_right))
        if fit_l is None or fit_r is None:
            return done(area_px=area, contour=contour, diagnostic="标定尖两侧边缘有效采样不足")

        ml, bl, good_l, rms_l = fit_l
        mr, br, good_r, rms_r = fit_r
        denom = ml - mr
        if abs(denom) < 1e-4:
            return done(area_px=area, contour=contour, diagnostic="标定尖两侧边缘平行，无法求交顶点")
        angle_deg = float(math.degrees(math.atan(ml) - math.atan(mr)))
        if angle_deg <= 0:
            return done(area_px=area, contour=contour, apex_angle_deg=angle_deg, diagnostic=f"标定尖两侧边缘未向顶点收敛（夹角 {angle_deg:.1f}°）")

        s_star = (br - bl) / denom
        w_star = ml * s_star + bl
        rms = float(math.hypot(rms_l, rms_r) / math.sqrt(2.0))
        med_contrast = float(np.median(contrasts)) if contrasts else 0.0

        def to_global(sv: float, wv: float) -> tuple[float, float]:
            pt = center + sv * d + wv * n
            return float(pt[0] + x0), float(pt[1] + y0)

        tip_uv = to_global(s_star, w_star)
        flank_segments = (
            (to_global(s_start, ml * s_start + bl), tip_uv),
            (to_global(s_start, mr * s_start + br), tip_uv),
        )

        reasons = []
        if not (cfg.min_apex_angle_deg <= angle_deg <= cfg.max_apex_angle_deg):
            reasons.append(f"锥顶角 {angle_deg:.1f}° 超出允许范围 [{cfg.min_apex_angle_deg:g}°, {cfg.max_apex_angle_deg:g}°]")
        if rms > cfg.max_edge_rms_px:
            reasons.append(f"侧边直线拟合残差 {rms:.3f} px 超过阈值 {cfg.max_edge_rms_px:g} px")
        if abs(s_star - s_apex) > cfg.max_tip_offset_px or abs(w_star - w_apex_est) > cfg.max_tip_offset_px:
            reasons.append(f"交点与轮廓顶点偏差过大（Δs={s_star - s_apex:.2f}, Δw={w_star - w_apex_est:.2f} px）")
        if med_contrast < cfg.min_contrast:
            reasons.append(f"对比度 {med_contrast:.1f} 低于 {cfg.min_contrast:g}")

        if reasons:
            return done(
                point_uv=None,
                quality=0.0,
                focused=False,
                diagnostic="；".join(reasons),
                contrast=med_contrast,
                edge_rms_px=rms,
                apex_angle_deg=angle_deg,
                area_px=area,
                flank_segments=flank_segments,
                contour=contour,
            )

        rms_score = max(0.0, 1.0 - rms / max(cfg.max_edge_rms_px, 1e-6))
        contrast_score = min(1.0, med_contrast / 120.0)
        quality = float(np.clip(0.6 * rms_score + 0.4 * contrast_score, 0.05, 1.0))
        return done(
            point_uv=tip_uv,
            quality=quality,
            focused=True,
            diagnostic="主相机标定尖识别成功",
            contrast=med_contrast,
            edge_rms_px=rms,
            apex_angle_deg=angle_deg,
            area_px=area,
            flank_segments=flank_segments,
            contour=contour,
        )


def render_ref_tip_top_overlay(image: np.ndarray, result: RefTipTopResult, *, shot_index: int | None = None,
                               principal: tuple[float, float] | None = None, margin_frac: float = 0.10) -> np.ndarray:
    """绘制主相机标定尖识别诊断图（含轮廓、拟合侧边、亚像素交点与局部放大窗）。"""
    arr = np.asarray(image)
    rendered = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR) if arr.ndim == 2 else arr.copy()
    h, w = rendered.shape[:2]

    # 画中央 80% 安全区域虚线框
    x0, x1 = int(round(margin_frac * w)), int(round((1.0 - margin_frac) * w))
    y0, y1 = int(round(margin_frac * h)), int(round((1.0 - margin_frac) * h))
    cv2.rectangle(rendered, (x0, y0), (x1, y1), (200, 160, 40), 2, cv2.LINE_AA)

    if principal is not None:
        cv2.drawMarker(rendered, (int(round(principal[0])), int(round(principal[1]))), (255, 140, 0), cv2.MARKER_CROSS, 45, 2)

    if result.contour is not None and len(result.contour):
        cv2.polylines(rendered, [result.contour.astype(np.int32)], True, (0, 190, 80), 2, cv2.LINE_AA)
    for p_start, p_end in result.flank_segments:
        cv2.line(
            rendered,
            (int(round(p_start[0])), int(round(p_start[1]))),
            (int(round(p_end[0])), int(round(p_end[1]))),
            (255, 220, 0),
            2,
            cv2.LINE_AA,
        )
    if result.point_uv is not None:
        pt = (int(round(result.point_uv[0])), int(round(result.point_uv[1])))
        cv2.circle(rendered, pt, 8, (0, 60, 255), 2, cv2.LINE_AA)
        cv2.drawMarker(rendered, pt, (0, 60, 255), cv2.MARKER_CROSS, 36, 2, cv2.LINE_AA)

        # 右上角局部放大窗
        inset = min(320, max(140, w // 5), h // 3)
        span = 60
        cx, cy = pt
        padded = cv2.copyMakeBorder(rendered, span, span, span, span, cv2.BORDER_CONSTANT, value=(245, 245, 245))
        crop = padded[cy + span - span // 2:cy + span + span // 2, cx + span - span // 2:cx + span + span // 2]
        if crop.size and w >= inset + 16:
            crop = cv2.resize(crop, (inset, inset), interpolation=cv2.INTER_NEAREST)
            rendered[12:12 + inset, w - inset - 12:w - 12] = crop
            cv2.rectangle(rendered, (w - inset - 12, 12), (w - 12, 12 + inset), (120, 120, 120), 2)

    rgb = Image.fromarray(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(rgb)
    font_size = max(20, min(48, w // 70))
    from pathlib import Path

    font = next(
        (ImageFont.truetype(str(p), font_size) for p in (
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        ) if p.is_file()),
        ImageFont.load_default(),
    )
    prefix = f"第 {shot_index} 张  " if shot_index is not None else ""
    if result.point_uv is not None:
        lines = [
            (f"{prefix}标定尖 ({result.point_uv[0]:.2f}, {result.point_uv[1]:.2f}) px", (20, 130, 65)),
            (f"质量 {result.quality:.2f}  RMS {result.edge_rms_px:.3f} px  锥角 {result.apex_angle_deg:.1f}°", (20, 90, 160)),
        ]
    else:
        lines = [(f"{prefix}识别失败：{result.diagnostic}", (180, 35, 24))]
    for i, (text, color) in enumerate(lines):
        draw.text((16, 16 + i * (font_size + 8)), text, font=font, fill=color, stroke_width=2, stroke_fill=(255, 255, 255))
    return cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
