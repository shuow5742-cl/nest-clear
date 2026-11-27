"""菲林图案在已知 Z 焦点的合成扫描、多指标拟合与平面判定。"""

import math

import cv2
import numpy as np
import pytest

from nest_clear.config import load_config
from nest_clear.depth_config import DEFAULTS as DEPTH_DEFAULTS
from nest_clear.depthcal.focus import fit_coarse_fwhm, fit_peak, plan_fine_scan, scores
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


def _sim_rows(image, zs, focus, blur_fn, noise=0.0, rng=None):
    rows = []
    for z in zs:
        blurred = cv2.GaussianBlur(image, (0, 0), blur_fn(z - focus))
        if noise == 0.0:
            rows.append({'z_actual': z, **scores(blurred)})
        else:
            ms = [scores(np.clip(blurred * rng.uniform(.8, 1.15) + rng.normal(0, noise, image.shape), 0, 255))
                  for _ in range(3)]
            rows.append({'z_actual': z, **{k: float(np.median([m[k] for m in ms])) for k in ms[0]}})
    return rows


@pytest.mark.parametrize(('label', 'k', 'exp_half_rng', 'exp_step_rng'), [
    ('fwhm_0.5mm', 3.6, (0.30, 0.48), (0.02, 0.025)),
    ('fwhm_2.0mm', 0.9, (1.35, 1.65), (0.065, 0.085)),
    ('fwhm_4.0mm', 0.44, (2.75, 3.00), (0.095, 0.10)),
])
@pytest.mark.parametrize(('noise', 'tol'), [(0.0, 0.002), (2.0, 0.01)])
def test_auto_fine_range_and_accuracy_across_fwhm(label, k, exp_half_rng, exp_step_rng, noise, tol):
    image = film_ring()
    focus = -5.137
    cfg = dict(DEPTH_DEFAULTS)
    rng = np.random.default_rng(88)
    blur_fn = lambda dz: math.hypot(0.4, k * dz)

    coarse = _sim_rows(image, descending(-2.0, -8.0, cfg['coarse_step_mm']), focus, blur_fn, noise, rng)
    coarse_fit, err_msg = fit_coarse_fwhm(coarse, 'tenengrad', 'gauss', '粗扫')
    assert coarse_fit is not None and err_msg is None
    plan = plan_fine_scan(cfg, coarse_fit['fwhm'], coarse_fit['z_peak'], -12.0, 0.0)
    assert plan['mode'] == 'auto' and not plan['fallback'] and not plan['truncated']
    assert exp_half_rng[0] <= plan['half_width_mm'] <= exp_half_rng[1]
    assert exp_step_rng[0] <= plan['fine_step_mm'] <= exp_step_rng[1]

    fine = _sim_rows(image, descending(plan['high_mm'], plan['low_mm'], plan['fine_step_mm']), focus, blur_fn, noise, rng)
    fit = fit_peak(fine, 'tenengrad', 'gauss', '细扫')
    error = abs(fit['z_peak'] - focus)
    print(f'{label} noise={noise}: half={plan["half_width_mm"]:.4f} step={plan["fine_step_mm"]:.4f}'
          f' fwhm={fit["fwhm"]:.4f} err={error:.6f} L={fit["left_points"]} R={fit["right_points"]} min_r={fit["min_ratio"]:.3f}')
    assert error <= tol
    assert fit['quality_ok'] and not fit['insufficient_points']
    assert fit['left_points'] >= 5 and fit['right_points'] >= 5 and fit['min_ratio'] <= 0.6


def test_2mm_fixed_triggers_insufficient_points_while_auto_passes():
    image = film_ring()
    focus = -5.137
    blur_fn = lambda dz: math.hypot(0.4, 0.9 * dz)  # FWHM ≈ 2.0 mm

    # fixed ±0.6 mm: 仅覆盖峰顶平坦段，最低参与点 > 0.6 × 峰值，触发“拟合点不足”
    fixed_cfg = {**DEPTH_DEFAULTS, 'fine_range_mode': 'fixed'}
    fixed_plan = plan_fine_scan(fixed_cfg, 2.0, focus, -12.0, 0.0)
    assert fixed_plan['mode'] == 'fixed' and fixed_plan['half_width_mm'] == pytest.approx(0.6)
    fixed_rows = _sim_rows(image, descending(fixed_plan['high_mm'], fixed_plan['low_mm'], fixed_plan['fine_step_mm']), focus, blur_fn)
    fixed_fit = fit_peak(fixed_rows, 'tenengrad', 'gauss', '细扫')
    assert not fixed_fit['quality_ok'] and fixed_fit['insufficient_points']
    assert '拟合点不足' in fixed_fit['quality_note']
    assert fixed_fit['min_ratio'] > 0.6
    feas_fixed = repeatability([{'label': 'T02', 'xy': [96., 25.], 'fits': [{'repeat': 1, **fixed_fit}],
                                 'z_peak': fixed_fit['z_peak'], 'range': 0.001, 'std': 0.0005}], 0.02)
    assert feas_fixed['insufficient_fits'] == 1 and not feas_fixed['met']

    # auto: 自适应放大范围到 ~1.5mm，覆盖下降段，不触发“拟合点不足”
    auto_cfg = dict(DEPTH_DEFAULTS)
    coarse = _sim_rows(image, descending(-2.0, -8.0, auto_cfg['coarse_step_mm']), focus, blur_fn)
    coarse_fit, _ = fit_coarse_fwhm(coarse, 'tenengrad', 'gauss', '粗扫')
    auto_plan = plan_fine_scan(auto_cfg, coarse_fit['fwhm'], coarse_fit['z_peak'], -12.0, 0.0)
    auto_rows = _sim_rows(image, descending(auto_plan['high_mm'], auto_plan['low_mm'], auto_plan['fine_step_mm']), focus, blur_fn)
    auto_fit = fit_peak(auto_rows, 'tenengrad', 'gauss', '细扫')
    assert auto_fit['quality_ok'] and not auto_fit['insufficient_points']
    assert auto_fit['min_ratio'] <= 0.6
    feas_auto = repeatability([{'label': 'T02', 'xy': [96., 25.], 'fits': [{'repeat': 1, **auto_fit}],
                                'z_peak': auto_fit['z_peak'], 'range': 0.001, 'std': 0.0005}], 0.02)
    assert feas_auto['insufficient_fits'] == 0 and feas_auto['met']


def test_coarse_fallback_and_z_truncation_and_config():
    cfg = dict(DEPTH_DEFAULTS)
    # 粗扫拟合失败 / FWHM 无效 -> 回退 fixed
    bad_coarse = [{'z_actual': -5.0 - 0.2 * i, 'tenengrad': v} for i, v in enumerate((0.95, 0.90, 1.00, 0.92, 0.96))]
    cfit, err_msg = fit_coarse_fwhm(bad_coarse, 'tenengrad', 'gauss', '粗扫')
    assert cfit is None and err_msg is not None
    fb_plan = plan_fine_scan(cfg, None, -5.4, -12.0, 0.0, err_msg)
    assert fb_plan['mode'] == 'fixed_fallback' and fb_plan['fallback']
    assert fb_plan['half_width_mm'] == pytest.approx(cfg['fine_range_mm'])
    assert fb_plan['fine_step_mm'] == pytest.approx(cfg['fine_step_mm'])

    # 下限与上限 Z 约束截断提示
    trunc_low = plan_fine_scan(cfg, 3.0, -11.2, -12.0, 0.0)
    assert trunc_low['truncated'] and trunc_low['truncated_low']
    assert trunc_low['low_mm'] == pytest.approx(-12.0)
    assert 'camera_z_min_mm' in trunc_low['truncation_note']

    trunc_high = plan_fine_scan(cfg, 3.0, -1.0, -12.0, 0.0)
    assert trunc_high['truncated'] and trunc_high['truncated_high']
    assert trunc_high['top_mm'] == pytest.approx(0.0)
    assert 'Z 行程' in trunc_high['truncation_note']

    loaded = load_config('config/nest_clear.yaml')
    assert loaded.depth_calibration['fine_range_mode'] == 'auto'
    assert loaded.depth_calibration['fine_steps_target'] == 40
