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


def has_peak_contrast(v_peak, v_low, min_peak_contrast=DEPTH_DEFAULTS['min_peak_contrast']):
    """判断清晰度动态对比度 v_peak / v_low 是否达到最小要求。"""
    if not (math.isfinite(v_peak) and math.isfinite(v_low)) or v_peak <= 0:
        return False
    if v_low <= 0:
        return True
    return (v_peak / v_low) >= float(min_peak_contrast)


def _is_blurred_point(value, v_peak, v_low, blur_contrast_ratio, defocus_ratio):
    """动态范围判据或峰值比值判据任一满足即视为明显模糊。"""
    if v_peak <= 0 or not math.isfinite(value):
        return False
    ratio_ok = (value / v_peak) <= float(defocus_ratio)
    dyn = v_peak - v_low
    contrast_ok = (dyn > 0) and (((value - v_low) / dyn) <= float(blur_contrast_ratio))
    return ratio_ok or contrast_ok


def find_side_blur(
    values,
    side_indices,
    v_peak,
    v_low,
    *,
    blur_contrast_ratio=DEPTH_DEFAULTS['blur_contrast_ratio'],
    defocus_ratio=DEPTH_DEFAULTS['defocus_ratio'],
    min_peak_contrast=DEPTH_DEFAULTS['min_peak_contrast'],
):
    """从峰所在下标起沿单侧外扩，找离开峰后单调下降且连续 2 点明显模糊的下标。"""
    if len(side_indices) < 3 or not has_peak_contrast(v_peak, v_low, min_peak_contrast):
        return None
    tol = 0.02 * max(float(v_peak) - float(v_low), 0.0)
    prev_blurred = False
    for offset in range(1, len(side_indices)):
        idx = side_indices[offset]
        prev_idx = side_indices[offset - 1]
        cur_val = float(values[idx])
        prev_val = float(values[prev_idx])
        # 离开峰后必须呈单调下降趋势，防止噪声/未离峰时误判。
        if cur_val > prev_val + tol or cur_val >= v_peak:
            return None
        cur_blurred = _is_blurred_point(cur_val, v_peak, v_low, blur_contrast_ratio, defocus_ratio)
        if offset >= 2 and prev_blurred and cur_blurred:
            return idx
        prev_blurred = cur_blurred
    return None


def analyze_coarse_scan(
    rows,
    metric,
    cfg,
    where,
    *,
    upper_blurred_hint=False,
    upper_limited_by=None,
    lower_limited_by=None,
):
    """分析向下粗扫曲线：校验对比度与内部峰，定位两侧明显模糊 Z 边界。"""
    ordered = sorted(rows, key=lambda row: row['z_actual'], reverse=True)
    z = [float(row['z_actual']) for row in ordered]
    values = [float(row[metric]) for row in ordered]
    if not values:
        raise boundary_error(z, 0, where)
    v_peak = max(values)
    v_low = min(values)
    if not has_peak_contrast(v_peak, v_low, cfg['min_peak_contrast']):
        raise GuardError(
            'DEPTH-CONTRAST',
            '清晰度变化太小，景深过大或画面太暗/对比度不足',
            where,
            '检查光源、曝光、对焦范围或增大 Z 向扫描行程。',
        )
    idx = peak_index(np.array(z), np.array(values), where)
    blur_ratio = cfg['blur_contrast_ratio']
    defocus = cfg['defocus_ratio']
    min_contrast = cfg['min_peak_contrast']

    upper_indices = list(range(idx, -1, -1))
    upper_blur_idx = find_side_blur(
        values, upper_indices, v_peak, v_low,
        blur_contrast_ratio=blur_ratio, defocus_ratio=defocus, min_peak_contrast=min_contrast,
    )
    if upper_blur_idx is not None:
        blur_high_reached = True
        blur_high_z = z[upper_blur_idx]
    elif upper_blurred_hint:
        blur_high_reached = True
        blur_high_z = z[0]
    else:
        blur_high_reached = False
        blur_high_z = z[0]

    lower_indices = list(range(idx, len(values)))
    lower_blur_idx = find_side_blur(
        values, lower_indices, v_peak, v_low,
        blur_contrast_ratio=blur_ratio, defocus_ratio=defocus, min_peak_contrast=min_contrast,
    )
    if lower_blur_idx is not None:
        blur_low_reached = True
        blur_low_z = z[lower_blur_idx]
    else:
        blur_low_reached = False
        blur_low_z = z[-1]

    limit_notes = []
    if not blur_high_reached:
        if upper_limited_by == 'span':
            limit_notes.append('上侧未达明显模糊（受 coarse_max_span_mm 限制）')
        else:
            limit_notes.append('上侧未达明显模糊（受行程上限限制）')
    if not blur_low_reached:
        if lower_limited_by == 'span':
            limit_notes.append('下侧未达明显模糊（受 coarse_max_span_mm 限制）')
        else:
            limit_notes.append('下侧未达明显模糊（受 Z 下限限制）')

    return {
        'peak_idx': idx,
        'z_peak_coarse': z[idx],
        'v_peak': float(v_peak),
        'v_low': float(v_low),
        'contrast_ratio': float(v_peak / v_low) if v_low > 0 else float('inf'),
        'blur_low_z': float(blur_low_z),
        'blur_high_z': float(blur_high_z),
        'blur_low_reached': blur_low_reached,
        'blur_high_reached': blur_high_reached,
        'limit_notes': limit_notes,
    }


def plan_fine_scan(
    cfg,
    blur_low_z,
    blur_high_z,
    z_min,
    z_max,
    *,
    blur_low_reached=True,
    blur_high_reached=True,
    coarse_peak=None,
    coarse_fwhm=None,
    limit_notes=None,
):
    """在两侧明显模糊点（或受限边界）之间均匀插点规划细测步距与范围。"""
    configured_mode = cfg['fine_range_mode']
    if configured_mode == 'fixed':
        anchor = float(coarse_peak if coarse_peak is not None else 0.5 * (blur_low_z + blur_high_z))
        half_width = float(cfg['fine_range_mm'])
        fine_step = float(cfg['fine_step_mm'])
        nominal_low = anchor - half_width
        nominal_high = anchor + half_width
        actual_mode = 'fixed'
    else:
        nominal_low = float(blur_low_z)
        nominal_high = float(blur_high_z)
        span = max(0.0, nominal_high - nominal_low)
        raw_step = span / cfg['fine_steps_target']
        fine_step = min(max(raw_step, cfg['fine_step_min_mm']), cfg['fine_step_max_mm'])
        half_width = span / 2.0
        actual_mode = 'auto'

    low = max(nominal_low, z_min)
    top = min(nominal_high + cfg['overshoot_mm'], z_max)
    high = min(nominal_high, top - fine_step)
    truncated_low = (nominal_low < z_min - 1e-12) or (not blur_low_reached)
    truncated_high = (high < nominal_high - 1e-12) or (not blur_high_reached)
    truncated = truncated_low or truncated_high
    notes = list(limit_notes or [])
    if nominal_low < z_min - 1e-12:
        notes.append(f'下限受 camera_z_min_mm={z_min:.3f}mm 截断')
    if high < nominal_high - 1e-12 and blur_high_reached:
        notes.append(f'上限受 Z 行程 {z_max:.3f}mm 截断')
    fwhm_valid = isinstance(coarse_fwhm, (int, float)) and math.isfinite(coarse_fwhm) and coarse_fwhm > 0
    return {
        'mode': actual_mode,
        'configured_mode': configured_mode,
        'fallback': False,
        'fallback_reason': None,
        'coarse_fwhm_mm': float(coarse_fwhm) if fwhm_valid else None,
        'half_width_mm': float(half_width),
        'span_mm': float(max(0.0, high - low)),
        'effective_half_width_mm': float(max(0.0, (high - low) / 2.0)),
        'fine_step_mm': float(fine_step),
        'blur_low_z': float(blur_low_z),
        'blur_high_z': float(blur_high_z),
        'blur_low_reached': bool(blur_low_reached),
        'blur_high_reached': bool(blur_high_reached),
        'high_mm': float(high),
        'low_mm': float(low),
        'top_mm': float(top),
        'truncated': truncated,
        'truncated_low': truncated_low,
        'truncated_high': truncated_high,
        'truncation_note': '；'.join(dict.fromkeys(notes)) if notes else None,
    }
