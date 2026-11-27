"""菲林图案在已知 Z 焦点的合成扫描、多指标拟合与平面判定。"""

import math

import cv2
import numpy as np
import pytest

from nest_clear.config import load_config
from nest_clear.depth_config import DEFAULTS as DEPTH_DEFAULTS
from nest_clear.depthcal.focus import (
    analyze_coarse_scan, fit_coarse_fwhm, fit_peak, measure_dof, next_auto_exposure,
    plan_fine_scan, roi_brightness, scores,
)
from nest_clear.depthcal.layout import dof_summary, fit_plane, repeatability
from nest_clear.depthcal.session import ascending, descending
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


@pytest.mark.parametrize(('label', 'k', 'exp_step_rng'), [
    ('fwhm_1.0mm', 1.8, (0.03, 0.15)),
    ('fwhm_2.0mm', 0.9, (0.05, 0.20)),
    ('fwhm_4.0mm', 0.44, (0.10, 0.25)),
])
@pytest.mark.parametrize(('noise', 'tol'), [(0.0, 0.01), (2.0, 0.015)])
def test_auto_fine_range_and_accuracy_across_fwhm(label, k, exp_step_rng, noise, tol):
    image = film_ring()
    focus = -5.137
    cfg = dict(DEPTH_DEFAULTS)
    rng = np.random.default_rng(88)
    blur_fn = lambda dz: math.hypot(0.4, k * dz)

    coarse = _sim_rows(image, descending(-0.5, -10.5, cfg['coarse_step_mm']), focus, blur_fn, noise, rng)
    analysis = analyze_coarse_scan(coarse, 'tenengrad', cfg, '粗扫')
    assert analysis['blur_low_reached'] and analysis['blur_high_reached']
    assert analysis['blur_low_z'] < focus < analysis['blur_high_z']
    plan = plan_fine_scan(
        cfg, analysis['blur_low_z'], analysis['blur_high_z'], -12.0, 0.0,
        blur_low_reached=analysis['blur_low_reached'],
        blur_high_reached=analysis['blur_high_reached'],
        coarse_peak=analysis['z_peak_coarse'],
        limit_notes=analysis['limit_notes'],
    )
    assert plan['mode'] == 'auto' and not plan['fallback'] and not plan['truncated']
    assert exp_step_rng[0] <= plan['fine_step_mm'] <= exp_step_rng[1]

    fine = _sim_rows(image, descending(plan['high_mm'], plan['low_mm'], plan['fine_step_mm']), focus, blur_fn, noise, rng)
    fit = fit_peak(fine, 'tenengrad', 'gauss', '细扫')
    error = abs(fit['z_peak'] - focus)
    print(f'{label} noise={noise}: span=[{plan["blur_low_z"]:.2f},{plan["blur_high_z"]:.2f}] step={plan["fine_step_mm"]:.4f}'
          f' fwhm={fit["fwhm"]:.4f} err={error:.6f} L={fit["left_points"]} R={fit["right_points"]} min_r={fit["min_ratio"]:.3f}')
    assert error <= tol
    assert fit['quality_ok'] and not fit['insufficient_points']
    assert fit['left_points'] >= 5 and fit['right_points'] >= 5 and fit['min_ratio'] <= 0.6
    assert fit['dof_covered'] and fit['dof_ratio'] == pytest.approx(0.8)
    assert fit['dof_z_lower'] < fit['z_peak'] < fit['dof_z_upper']
    assert 0 < fit['dof_mm'] < fit['fwhm'] and fit['dof_gauss_mm'] > 0


@pytest.mark.parametrize('fwhm_mm', [0.5, 2.0, 4.0])
def test_synthetic_gaussian_dof_matches_analytical_within_two_steps(fwhm_mm):
    focus = -5.137
    cfg = dict(DEPTH_DEFAULTS)
    sigma = fwhm_mm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    ratio = cfg['dof_ratio']
    analytical_dof = sigma * 2.0 * math.sqrt(2.0 * math.log(1.0 / ratio))

    coarse_rows = [
        {'z_actual': z, 'tenengrad': math.exp(-0.5 * ((z - focus) / sigma) ** 2)}
        for z in descending(-0.5, -10.5, cfg['coarse_step_mm'])
    ]
    analysis = analyze_coarse_scan(coarse_rows, 'tenengrad', cfg, '粗扫')
    coarse_fit, err_msg = fit_coarse_fwhm(coarse_rows, 'tenengrad', 'gauss', '粗扫', dof_ratio=ratio)
    assert coarse_fit is not None and err_msg is None and coarse_fit['dof_covered']

    plan = plan_fine_scan(
        cfg, analysis['blur_low_z'], analysis['blur_high_z'], -12.0, 0.0,
        blur_low_reached=analysis['blur_low_reached'],
        blur_high_reached=analysis['blur_high_reached'],
        coarse_peak=coarse_fit['z_peak'],
        coarse_fwhm=coarse_fit['fwhm'],
    )
    step = plan['fine_step_mm']
    fine_rows = [
        {'z_actual': z, 'tenengrad': math.exp(-0.5 * ((z - focus) / sigma) ** 2)}
        for z in descending(plan['high_mm'], plan['low_mm'], step)
    ]
    fit = fit_peak(fine_rows, 'tenengrad', 'gauss', '细扫', dof_ratio=ratio)
    err_gauss = abs(fit['dof_mm'] - fit['dof_gauss_mm'])
    err_true = abs(fit['dof_mm'] - analytical_dof)
    print(f'FWHM={fwhm_mm}mm step={step:.4f} interp={fit["dof_mm"]:.6f} '
          f'gauss={fit["dof_gauss_mm"]:.6f} analytical={analytical_dof:.6f} err={err_gauss:.6f}')
    assert fit['dof_covered'] is True
    assert err_gauss <= 2.0 * step
    assert err_true <= 2.0 * step


def test_single_side_uncovered_dof_excluded_from_stats():
    focus = -5.0
    sigma = 2.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    # 上侧覆盖到下降段，下侧在 -5.15 处截断（清晰度 > 0.96 > 0.80），因此单侧未跨过 0.8×峰值
    zs_uncovered = descending(-3.5, -5.15, 0.05)
    uncovered_rows = [
        {'z_actual': z, 'tenengrad': math.exp(-0.5 * ((z - focus) / sigma) ** 2)}
        for z in zs_uncovered
    ]
    fit_uncov = fit_peak(uncovered_rows, 'tenengrad', 'gauss', '细扫', dof_ratio=0.8)
    assert fit_uncov['dof_covered'] is False
    assert fit_uncov['dof_mm'] is None
    assert fit_uncov['dof_z_upper'] is not None
    assert fit_uncov['dof_z_lower'] is None
    assert fit_uncov['dof_gauss_mm'] > 0

    # 另一侧完整覆盖的曲线应正常参与统计，未覆盖次不计入
    zs_covered = descending(-3.5, -6.5, 0.05)
    covered_rows = [
        {'z_actual': z, 'tenengrad': math.exp(-0.5 * ((z - focus) / sigma) ** 2)}
        for z in zs_covered
    ]
    fit_cov1 = fit_peak(covered_rows, 'tenengrad', 'gauss', '细扫', dof_ratio=0.8)
    fit_cov2 = fit_peak(
        [{'z_actual': z, 'tenengrad': math.exp(-0.5 * ((z - focus) / (sigma * 1.02)) ** 2)} for z in zs_covered],
        'tenengrad', 'gauss', '细扫', dof_ratio=0.8,
    )
    assert fit_cov1['dof_covered'] and fit_cov2['dof_covered']

    p1_fits = [
        {'repeat': 1, **fit_cov1},
        {'repeat': 2, **fit_uncov},
        {'repeat': 3, **fit_cov2},
    ]
    p1_dofs = [f['dof_mm'] for f in p1_fits if f['dof_covered']]
    p2_fits = [{'repeat': 1, **fit_uncov}, {'repeat': 2, **fit_uncov}]
    results = [
        {'label': 'T02', 'fits': p1_fits,
         'dof_mm': sum(p1_dofs) / len(p1_dofs),
         'dof_range_mm': max(p1_dofs) - min(p1_dofs),
         'dof_uncovered': 1},
        {'label': 'T05', 'fits': p2_fits, 'dof_mm': None, 'dof_range_mm': None, 'dof_uncovered': 2},
    ]
    summary = dof_summary(results, 0.8)
    assert summary['covered_points'] == 1 and summary['total_points'] == 2
    assert summary['covered_fits'] == 2 and summary['total_fits'] == 5
    assert summary['uncovered_fits'] == 3
    assert summary['uncovered_items'] == ['T02#2', 'T05#1', 'T05#2']
    assert summary['mean_mm'] == pytest.approx(sum(p1_dofs) / len(p1_dofs))
    assert summary['min_mm'] == pytest.approx(summary['mean_mm'])
    assert summary['max_mm'] == pytest.approx(summary['mean_mm'])

    # measure_dof 直接调用覆盖边界行为
    z_arr = np.array([-6.0, -5.5, -5.0, -4.5, -4.0])
    v_arr = np.array([0.85, 0.92, 1.0, 0.70, 0.50])
    md = measure_dof(z_arr, v_arr, 2, 0.8)
    assert not md['dof_covered'] and md['dof_mm'] is None and md['dof_z_lower'] is None and md['dof_z_upper'] is not None


def test_2mm_fixed_triggers_insufficient_points_while_auto_passes():
    image = film_ring()
    focus = -5.137
    blur_fn = lambda dz: math.hypot(0.4, 0.9 * dz)  # FWHM ≈ 2.0 mm

    # fixed ±0.6 mm: 仅覆盖峰顶平坦段，最低参与点 > 0.6 × 峰值，触发“拟合点不足”
    fixed_cfg = {**DEPTH_DEFAULTS, 'fine_range_mode': 'fixed'}
    fixed_plan = plan_fine_scan(fixed_cfg, -7.0, -3.0, -12.0, 0.0, coarse_peak=focus)
    assert fixed_plan['mode'] == 'fixed' and fixed_plan['half_width_mm'] == pytest.approx(0.6)
    fixed_rows = _sim_rows(image, descending(fixed_plan['high_mm'], fixed_plan['low_mm'], fixed_plan['fine_step_mm']), focus, blur_fn)
    fixed_fit = fit_peak(fixed_rows, 'tenengrad', 'gauss', '细扫')
    assert not fixed_fit['quality_ok'] and fixed_fit['insufficient_points']
    assert '拟合点不足' in fixed_fit['quality_note']
    assert fixed_fit['min_ratio'] > 0.6
    feas_fixed = repeatability([{'label': 'T02', 'xy': [96., 25.], 'fits': [{'repeat': 1, **fixed_fit}],
                                 'z_peak': fixed_fit['z_peak'], 'range': 0.001, 'std': 0.0005}], 0.02)
    assert feas_fixed['insufficient_fits'] == 1 and not feas_fixed['met']

    # auto: 在两侧明显模糊点之间均匀插点，覆盖下降段，不触发“拟合点不足”
    auto_cfg = dict(DEPTH_DEFAULTS)
    coarse = _sim_rows(image, descending(-0.5, -10.5, auto_cfg['coarse_step_mm']), focus, blur_fn)
    analysis = analyze_coarse_scan(coarse, 'tenengrad', auto_cfg, '粗扫')
    auto_plan = plan_fine_scan(
        auto_cfg, analysis['blur_low_z'], analysis['blur_high_z'], -12.0, 0.0,
        blur_low_reached=analysis['blur_low_reached'], blur_high_reached=analysis['blur_high_reached'],
        coarse_peak=analysis['z_peak_coarse'],
    )
    auto_rows = _sim_rows(image, descending(auto_plan['high_mm'], auto_plan['low_mm'], auto_plan['fine_step_mm']), focus, blur_fn)
    auto_fit = fit_peak(auto_rows, 'tenengrad', 'gauss', '细扫')
    assert auto_fit['quality_ok'] and not auto_fit['insufficient_points']
    assert auto_fit['min_ratio'] <= 0.6
    feas_auto = repeatability([{'label': 'T02', 'xy': [96., 25.], 'fits': [{'repeat': 1, **auto_fit}],
                                'z_peak': auto_fit['z_peak'], 'range': 0.001, 'std': 0.0005}], 0.02)
    assert feas_auto['insufficient_fits'] == 0 and feas_auto['met']


def test_coarse_blur_search_and_limits_and_config():
    cfg = dict(DEPTH_DEFAULTS)
    assert cfg['coarse_step_mm'] == 0.5
    assert cfg['blur_contrast_ratio'] == 0.3
    assert cfg['min_peak_contrast'] == 1.15
    assert cfg['coarse_max_span_mm'] is None
    assert cfg['fine_step_max_mm'] == 0.25

    # 1) 合成高斯峰 + 大底值 + 噪声，覆盖 FWHM = 1 / 3 / 6 mm；峰在中间 / 靠上边界 / 靠下边界
    rng = np.random.default_rng(91)
    for fwhm_mm in (1.0, 3.0, 6.0):
        sigma = fwhm_mm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        for peak_z in (-6.0, -1.2, -10.8):
            zs = list(descending(0.0, -12.0, cfg['coarse_step_mm']))
            rows = [
                {
                    'z_actual': z,
                    'tenengrad': 0.16 + 0.40 * math.exp(-0.5 * ((z - peak_z) / sigma) ** 2) + rng.normal(0, 0.002),
                }
                for z in zs
            ]
            analysis = analyze_coarse_scan(rows, 'tenengrad', cfg, '粗扫')
            plan = plan_fine_scan(
                cfg, analysis['blur_low_z'], analysis['blur_high_z'], -12.0, 0.0,
                blur_low_reached=analysis['blur_low_reached'],
                blur_high_reached=analysis['blur_high_reached'],
                coarse_peak=analysis['z_peak_coarse'],
                limit_notes=analysis['limit_notes'],
            )
            assert not plan['fallback']
            assert cfg['fine_step_min_mm'] <= plan['fine_step_mm'] <= cfg['fine_step_max_mm']
            assert plan['low_mm'] < peak_z < plan['high_mm']
            if peak_z == -6.0 and fwhm_mm <= 3.0:
                assert analysis['blur_low_reached'] and analysis['blur_high_reached']
            if peak_z == -1.2 and fwhm_mm == 6.0:
                assert not analysis['blur_high_reached']
                assert any('上侧未达明显模糊' in n for n in analysis['limit_notes'])
            if peak_z == -10.8 and fwhm_mm == 6.0:
                assert not analysis['blur_low_reached']
                assert any('下侧未达明显模糊' in n for n in analysis['limit_notes'])

    # 2) 对比度不足（类似 11-26 现场 0.1545 -> 0.1631） -> DEPTH-CONTRAST
    flat_rows = [{'z_actual': -3.0 - 0.5 * i, 'tenengrad': 0.1545 + 0.0008 * i} for i in range(13)]
    with pytest.raises(GuardError) as exc_info:
        analyze_coarse_scan(flat_rows, 'tenengrad', cfg, '粗扫')
    assert exc_info.value.code == 'DEPTH-CONTRAST'
    assert '清晰度变化太小' in exc_info.value.what

    # 3) 峰在边界（对比度足够但峰在最高点） -> DEPTH-PEAK-BOUNDARY
    edge_rows = [{'z_actual': -0.5 * i, 'tenengrad': 0.15 + 0.30 * math.exp(-0.5 * (i / 2.0) ** 2)} for i in range(8)]
    with pytest.raises(GuardError) as exc_info:
        analyze_coarse_scan(edge_rows, 'tenengrad', cfg, '粗扫')
    assert exc_info.value.code == 'DEPTH-PEAK-BOUNDARY'

    # 4) 下限与上限 Z 约束截断提示
    trunc_low = plan_fine_scan(cfg, -13.0, -9.0, -12.0, 0.0, blur_low_reached=False, limit_notes=['下侧未达明显模糊（受 Z 下限限制）'])
    assert trunc_low['truncated'] and trunc_low['truncated_low']
    assert trunc_low['low_mm'] == pytest.approx(-12.0)
    assert '下侧未达明显模糊' in trunc_low['truncation_note']

    loaded = load_config('config/nest_clear.yaml')
    assert loaded.depth_calibration['coarse_step_mm'] == 0.5
    assert loaded.depth_calibration['blur_contrast_ratio'] == 0.3
    assert loaded.depth_calibration['min_peak_contrast'] == 1.15
    assert loaded.depth_calibration['coarse_max_span_mm'] is None
    assert loaded.depth_calibration['fine_step_max_mm'] == 0.25
    assert loaded.depth_calibration['fine_range_mode'] == 'auto'
    assert loaded.depth_calibration['fine_steps_target'] == 40
    assert loaded.depth_calibration['dof_ratio'] == 0.8


def test_roi_brightness_and_next_auto_exposure():
    cfg = dict(DEPTH_DEFAULTS)
    dark = np.full((60, 60), 60, np.uint8)
    dark[:5, :5] = 15
    st_dark = roi_brightness(dark)
    assert st_dark['roi_p99'] == pytest.approx(60.0)
    assert st_dark['saturated_fraction'] == 0.0
    nxt, warn = next_auto_exposure(6000.0, st_dark, cfg)
    assert warn is None and nxt == pytest.approx(6000.0 * (190.0 / 60.0))

    ok = np.full((60, 60), 192, np.uint8)
    nxt_ok, warn_ok = next_auto_exposure(19000.0, roi_brightness(ok), cfg)
    assert nxt_ok is None and warn_ok is None

    sat = np.full((60, 60), 255, np.uint8)
    nxt_sat, warn_sat = next_auto_exposure(20000.0, roi_brightness(sat), cfg)
    assert warn_sat is None and nxt_sat < 20000.0

    nxt_max, warn_max = next_auto_exposure(cfg['exposure_max_us'], st_dark, cfg)
    assert nxt_max is None and warn_max is not None and '上限' in warn_max
