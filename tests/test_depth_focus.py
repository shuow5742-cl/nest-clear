"""菲林图案在已知 Z 焦点的合成扫描、多指标拟合与平面判定。"""

import cv2
import numpy as np
import pytest

from nest_clear.depthcal.focus import fit_peak, scores
from nest_clear.depthcal.layout import fit_plane, repeatability
from nest_clear.depthcal.session import descending
from nest_clear.guard import GuardError


def film_ring(size=160, center=(80, 80)):
    image = np.full((size, size), 210, np.float32)
    x, y = center
    cv2.circle(image, center, 37, 20, 6)
    cv2.line(image, (x - 70, y), (x + 70, y), 20, 2)
    cv2.line(image, (x, y - 70), (x, y + 70), 20, 2)
    return image


@pytest.mark.parametrize(('noise', 'tolerance'), [(0., .002), (2., .01)])
@pytest.mark.parametrize('metric', ['tenengrad', 'laplacian', 'brenner'])
def test_synthetic_focus(noise, tolerance, metric):
    image = film_ring()
    focus = -5.137
    rng = np.random.default_rng(88)
    rows = []
    for z in descending(-4.5, -5.7, .02):
        sigma = max(.0001, 6 * abs(z - focus))
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        metrics = [scores(np.clip(blurred * rng.uniform(.8, 1.15) +
                                  rng.normal(0, noise, image.shape), 0, 255))
                   for _ in range(3)]
        rows.append({'z_actual': z, **{k: float(np.median([m[k] for m in metrics])) for k in metrics[0]}})
    fit = fit_peak(rows, metric, 'gauss', '细扫')
    error = abs(fit['z_peak'] - focus)
    half_error = abs(fit['half_height']['z_peak'] - focus)
    print(f'metric={metric} noise={noise} gauss_err={error:.8f} half_err={half_error:.8f}')
    gauss_tol = 0.005 if (metric == 'laplacian' and noise == 0.) else tolerance
    assert error <= gauss_tol
    assert half_error <= tolerance
    assert fit['fwhm'] > 0 and fit['r2'] > .8


def test_boundary_and_plane_errors():
    rows = [{'z_actual': -5.0 - 0.2 * i, 'tenengrad': float(i + 1)} for i in range(5)]
    with pytest.raises(GuardError) as exc_info:
        fit_peak(rows, 'tenengrad', 'gauss', '粗扫')
    assert exc_info.value.code == 'DEPTH-PEAK-BOUNDARY'

    # 平面拟合：倾斜 0.05 mm/100mm
    pts = [
        {'label': 'T01', 'xy': [90.0, 20.0], 'z_peak': -5.00 + 0.0005 * 90.0, 'range': 0.004, 'std': 0.002},
        {'label': 'T05', 'xy': [90.0, 120.0], 'z_peak': -5.00 + 0.0005 * 90.0, 'range': 0.005, 'std': 0.0025},
        {'label': 'T29', 'xy': [170.0, 120.0], 'z_peak': -5.00 + 0.0005 * 170.0, 'range': 0.003, 'std': 0.0015},
        {'label': 'T26', 'xy': [170.0, 20.0], 'z_peak': -5.00 + 0.0005 * 170.0, 'range': 0.004, 'std': 0.002},
        {'label': 'T12', 'xy': [130.0, 70.0], 'z_peak': -5.00 + 0.0005 * 130.0, 'range': 0.004, 'std': 0.002},
    ]
    plane = fit_plane(pts)
    assert plane['tilt_mm_per_100mm'] == pytest.approx(0.05, abs=1e-6)
    assert plane['max_residual'] < 1e-9
    feas = repeatability(pts, 0.02)
    assert feas['met'] and feas['max_range'] == pytest.approx(0.005) and feas['max_std'] == pytest.approx(0.0025)

    # 共线报错
    collinear = [{'label': f'P{i}', 'xy': [100.0 + i, 50.0], 'z_peak': -5.0} for i in range(5)]
    with pytest.raises(GuardError) as exc_info:
        fit_plane(collinear)
    assert exc_info.value.code == 'DEPTH-PLANE'
