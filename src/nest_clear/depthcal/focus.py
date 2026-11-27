"""计算亮度归一化清晰度及两种对焦峰估计。"""

import math

import cv2
import numpy as np

from nest_clear.depth_config import METRICS
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


def _half_height(z, values, index):
    half = values[index] * .5
    crossings = []
    for direction in (-1, 1):
        inner = index
        outer = inner + direction
        while 0 <= outer < len(z) and values[outer] > half:
            inner, outer = outer, outer + direction
        if not 0 <= outer < len(z) or values[inner] == values[outer]:
            return None
        fraction = (half - values[inner]) / (values[outer] - values[inner])
        crossings.append(z[inner] + fraction * (z[outer] - z[inner]))
    lo, hi = crossings
    return {'z_peak': float((lo + hi) / 2), 'fwhm': float(hi - lo)}


def fit_peak(rows, metric, method, where):
    z = np.array([row['z_actual'] for row in reversed(rows)])
    values = np.array([row[metric] for row in reversed(rows)])
    index = peak_index(z, values, where)
    half = _half_height(z, values, index)
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
                         'r2': float(1 - np.sum(residual ** 2) / total) if total > 0 else None}
    selected = gauss if method == 'gauss' else half
    if selected is None:
        raise GuardError('DEPTH-FIT', f'{metric} 的 {method} 峰拟合失败', where,
                         '检查细扫曲线是否覆盖峰两侧；可调整细扫范围、步长、曝光或清晰度主指标。')
    return {**selected, 'r2': gauss['r2'] if gauss else None, 'gauss': gauss, 'half_height': half}
