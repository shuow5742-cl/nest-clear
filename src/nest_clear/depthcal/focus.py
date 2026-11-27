"""计算亮度归一化清晰度及两种对焦峰估计。"""

import math

import cv2
import numpy as np

from nest_clear.depth_config import DEFAULTS as DEPTH_DEFAULTS, METRICS
from nest_clear.guard import GuardError


def scores(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = gray.astype(np.float64)
    normalization = float(gray.mean()) ** 2
    if normalization <= 0:
        raise GuardError('DEPTH-DARK', 'ROI 平均灰度为零', '对焦采图', '检查光源和曝光。')
    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    return dict(zip(METRICS, (
        float(np.mean(dx * dx + dy * dy)) / normalization,
        float(cv2.Laplacian(gray, cv2.CV_64F).var()) / normalization,
        float(np.mean(np.square(gray[:, 2:] - gray[:, :-2]))) / normalization,
    )))


def boundary_error(z, index, where):
    if len(z) < 3:
        return GuardError('DEPTH-PEAK-BOUNDARY', '对焦峰不在扫描范围内，Z 扫描不足 3 步',
                          where, '检查安全 Z 下限、起始 Z 与扫描步长。')
    lower = z[index] == min(z)
    direction = 'Z−（向下）' if lower else 'Z+（向上）'
    hint = ('核查工具间隙及安全 Z 下限，再评估是否扩大向下的可用扫描范围。' if lower
            else '在 Z 行程内提高起始 Z 或扩大向上扫描范围。')
    return GuardError('DEPTH-PEAK-BOUNDARY',
                      f'对焦峰不在扫描范围内，最大清晰度在 {direction} 边界 Z={z[index]:.4f} mm',
                      where, hint)


def peak_index(z, values, where):
    index = int(np.argmax(values))
    if len(z) < 3 or index in (0, len(z) - 1):
        raise boundary_error(z, index, where)
    return index


def _threshold_crossings(z, values, index, ratio):
    """在实测峰值两侧找跨过 ratio×峰值 的相邻两点并线性插值，返回 (Z_下, Z_上)。"""
    peak_val = float(values[index])
    threshold = peak_val * float(ratio)
    if peak_val <= 0 or not math.isfinite(threshold):
        return None, None
    crossings = []
    for direction in (-1, 1):
        inner = index
        outer = inner + direction
        while 0 <= outer < len(z) and values[outer] > threshold:
            inner, outer = outer, outer + direction
        if not 0 <= outer < len(z) or values[inner] == values[outer]:
            crossings.append(None)
        else:
            fraction = (threshold - values[inner]) / (values[outer] - values[inner])
            crossings.append(float(z[inner] + fraction * (z[outer] - z[inner])))
    return crossings[0], crossings[1]


def _half_height(z, values, index):
    lo, hi = _threshold_crossings(z, values, index, .5)
    if lo is None or hi is None:
        return None
    return {'z_peak': float((lo + hi) / 2), 'fwhm': float(hi - lo)}


def measure_dof(z, values, index, ratio=DEPTH_DEFAULTS['dof_ratio']):
    """计算清晰度 >= ratio×峰值 的插值景深区间；任一侧未跨过阈值时记为未覆盖。"""
    z_lo, z_hi = _threshold_crossings(z, values, index, ratio)
    covered = z_lo is not None and z_hi is not None
    return {
        'dof_mm': float(z_hi - z_lo) if covered else None,
        'dof_ratio': float(ratio),
        'dof_z_lower': z_lo,
        'dof_z_upper': z_hi,
        'dof_covered': covered,
    }


def fit_peak(rows, metric, method, where, dof_ratio=DEPTH_DEFAULTS['dof_ratio']):
    ordered = sorted(rows, key=lambda row: row['z_actual'])
    z = np.array([row['z_actual'] for row in ordered])
    values = np.array([row[metric] for row in ordered])
    index = peak_index(z, values, where)
    half = _half_height(z, values, index)
    dof = measure_dof(z, values, index, dof_ratio)
    mask = values >= .5 * values[index]
    origin = z[index]
    gauss = None
    if mask.sum() >= 3 and values[index] > 0:
        x = z[mask] - origin
        y = np.log(values[mask])
        quadratic = np.polynomial.polynomial.polyfit(x, y, 2)
        _, b, a = quadratic
        if a < 0:
            peak = float(origin - b / (2 * a))
            if z[0] < peak < z[-1]:
                residual = y - np.polynomial.polynomial.polyval(x, quadratic)
                total = np.sum(np.square(y - y.mean()))
                gauss = {'z_peak': peak, 'fwhm': 2 * math.sqrt(-math.log(2) / a),
                         'dof_gauss_mm': 2 * math.sqrt(-math.log(1 / dof_ratio) / a),
                         'r2': float(1 - np.sum(residual ** 2) / total) if total > 0 else None}
    selected = gauss if method == 'gauss' else half
    if selected is None:
        raise GuardError('DEPTH-FIT', f'{metric} 的 {method} 峰拟合失败', where,
                         '检查细扫曲线是否覆盖峰两侧；可调整细扫范围、步长、曝光或清晰度主指标。')
    left_points = int(np.sum(z[mask] < origin))
    right_points = int(np.sum(z[mask] > origin))
    min_ratio = float(np.min(values[mask]) / values[index]) if mask.any() and values[index] > 0 else None
    issues = []
    if left_points < 5:
        issues.append(f'峰左侧拟合点 {left_points} < 5')
    if right_points < 5:
        issues.append(f'峰右侧拟合点 {right_points} < 5')
    if min_ratio is None or min_ratio > 0.6:
        ratio_text = f'{min_ratio:.2f}' if min_ratio is not None else '—'
        issues.append(f'最低参与点 {ratio_text}×峰值 > 0.60×峰值')
    quality_ok = not issues
    return {
        **selected,
        'r2': gauss['r2'] if gauss else None,
        'gauss': gauss,
        'half_height': half,
        'dof_gauss_mm': gauss['dof_gauss_mm'] if gauss else None,
        **dof,
        'quality_ok': quality_ok,
        'insufficient_points': not quality_ok,
        'left_points': left_points,
        'right_points': right_points,
        'min_ratio': min_ratio,
        'quality_issues': issues,
        'quality_note': '拟合点不足（' + '；'.join(issues) + '）' if issues else '',
    }


def fit_coarse_fwhm(rows, metric, method, where, dof_ratio=DEPTH_DEFAULTS['dof_ratio']):
    """粗扫峰拟合；若窄峰导致半高以上点不足 3 个则回退半高宽法估计 FWHM。"""
    try:
        fit = fit_peak(rows, metric, method, where, dof_ratio=dof_ratio)
    except GuardError as exc:
        ordered = sorted(rows, key=lambda row: row['z_actual'])
        z = np.array([row['z_actual'] for row in ordered])
        values = np.array([row[metric] for row in ordered])
        index = peak_index(z, values, where)
        if method == 'gauss' and int(np.sum(values >= .5 * values[index])) < 3:
            half = _half_height(z, values, index)
            if half and isinstance(half.get('fwhm'), (int, float)) and math.isfinite(half['fwhm']) and half['fwhm'] > 0:
                dof = measure_dof(z, values, index, dof_ratio)
                return {**half, 'r2': None, 'gauss': None, 'half_height': half,
                        'dof_gauss_mm': None, **dof, 'coarse_fallback': 'half_height'}, None
        return None, exc.what
    fwhm = fit.get('fwhm')
    if not isinstance(fwhm, (int, float)) or not math.isfinite(fwhm) or fwhm <= 0:
        return None, '粗扫 FWHM 无效'
    return fit, None


def plan_fine_scan(cfg, coarse_fwhm, coarse_peak, z_min, z_max, fallback_reason=None):
    """根据粗扫 FWHM 与 Z 行程约束计算细扫半宽、步距及截断/回退提示。"""
    configured_mode = cfg['fine_range_mode']
    fwhm_valid = isinstance(coarse_fwhm, (int, float)) and math.isfinite(coarse_fwhm) and coarse_fwhm > 0
    fallback = False
    if configured_mode == 'auto':
        if fwhm_valid:
            raw_half = cfg['fine_range_factor'] * float(coarse_fwhm)
            half_width = min(max(raw_half, cfg['fine_range_min_mm']), cfg['fine_range_max_mm'])
            raw_step = (2.0 * half_width) / cfg['fine_steps_target']
            fine_step = min(max(raw_step, cfg['fine_step_min_mm']), cfg['fine_step_max_mm'])
            actual_mode = 'auto'
            reason = None
        else:
            half_width = float(cfg['fine_range_mm'])
            fine_step = float(cfg['fine_step_mm'])
            actual_mode = 'fixed_fallback'
            fallback = True
            reason = fallback_reason or '粗扫 FWHM 无效'
    else:
        half_width = float(cfg['fine_range_mm'])
        fine_step = float(cfg['fine_step_mm'])
        actual_mode = 'fixed'
        reason = None

    nominal_high = coarse_peak + half_width
    nominal_low = coarse_peak - half_width
    low = max(nominal_low, z_min)
    top = min(nominal_high + cfg['overshoot_mm'], z_max)
    high = min(nominal_high, top - fine_step)
    truncated_low = nominal_low < z_min - 1e-12
    truncated_high = high < nominal_high - 1e-12
    truncated = truncated_low or truncated_high
    lower_half = max(0.0, coarse_peak - low)
    upper_half = max(0.0, high - coarse_peak)
    notes = []
    if truncated_low:
        notes.append(f'下限受 camera_z_min_mm={z_min:.3f}mm 截断（下半宽 {lower_half:.3f} < {half_width:.3f} mm）')
    if truncated_high:
        notes.append(f'上限受 Z 行程 {z_max:.3f}mm 截断（上半宽 {upper_half:.3f} < {half_width:.3f} mm）')
    return {
        'mode': actual_mode,
        'configured_mode': configured_mode,
        'fallback': fallback,
        'fallback_reason': reason,
        'coarse_fwhm_mm': float(coarse_fwhm) if fwhm_valid else None,
        'half_width_mm': float(half_width),
        'effective_half_width_mm': float((high - low) / 2.0),
        'lower_half_mm': float(lower_half),
        'upper_half_mm': float(upper_half),
        'fine_step_mm': float(fine_step),
        'high_mm': float(high),
        'low_mm': float(low),
        'top_mm': float(top),
        'truncated': truncated,
        'truncated_low': truncated_low,
        'truncated_high': truncated_high,
        'truncation_note': '；'.join(notes) if notes else None,
    }
