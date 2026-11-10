from dataclasses import fields, replace
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from hair_scenes import frame_for
from nest_clear.config import load_config
from nest_clear.guard import GuardError
from nest_clear.vision.hair import HairDetectConfig, detect_candidates, scaled, effective_pixel_values
from nest_clear.vision.hair.scaling import INVENTORY

ROOT = Path(__file__).resolve().parents[1]


def test_inventory_and_scaling():
    reference = HairDetectConfig(um_per_px=6.959)
    assert scaled(reference) is reference
    assert set(INVENTORY) == {f.name for f in fields(HairDetectConfig)}
    half = scaled(replace(reference, um_per_px=reference.reference_um_per_px * 2))
    assert half.bridge_max_hop == 85
    assert half.bridge_flank_w == (8, 12)
    assert half.trace_step == 3
    assert half.fiber_min_area == 375
    assert half.hair_area_norm == 20000.0
    assert half.dark_open_k == 3  # 2.5 rounds to the nearest permitted odd kernel
    assert scaled(replace(reference, um_per_px=reference.reference_um_per_px / 0.8)).dark_open_k == 5  # tie -> larger odd
    assert half.dark_close_k == 11
    assert half.dark_open_k == 3
    assert half.hair_dark_max == reference.hair_dark_max
    assert half.seed_grow_close == 0
    tiny = scaled(replace(reference, um_per_px=reference.reference_um_per_px * 10000))
    assert tiny.dark_close_k == 3
    assert tiny.trace_half_win == 1
    assert tiny.fiber_min_area == 1
    assert tiny.trace_step > 0
    assert effective_pixel_values(replace(reference, um_per_px=13.918))['dark_close_k'] == 11


def test_downsample_units_and_reference_kernels():
    reference = HairDetectConfig(um_per_px=6.959)
    assert scaled(reference) == reference
    quarter = scaled(replace(reference, um_per_px=reference.reference_um_per_px * 4))
    assert (quarter.fiber_ds, quarter.merge_group_scale) == (1, 1)
    assert (quarter.line_blur_scale, quarter.seed_grow_scale) == (1.0, 1.0)
    # 15, 11 and 3 are reference processing-grid pixels at ds=2.
    assert (quarter.fiber_dens_win_ds, quarter.fiber_close_k_ds, quarter.fiber_open_k) == (7, 5, 3)
    assert quarter.line_blur_sigma * quarter.line_blur_scale == pytest.approx(15 * 0.25)
    assert quarter.line_blur_sigma * quarter.seed_grow_scale == pytest.approx(15 * 0.25)
    assert quarter.merge_close_bound_min_px == round(300000 * 0.25 ** 2)
    # k=.75 retains ds=2, so the fiber kernel conversion must include both ds values.
    three_quarters = scaled(replace(reference, um_per_px=reference.reference_um_per_px / .75))
    assert three_quarters.fiber_ds == 2
    assert three_quarters.fiber_dens_win_ds == 11
    assert three_quarters.line_blur_scale == pytest.approx(2 / 3)
    # Non-default reference ds must participate in the conversion too.
    custom = replace(reference, fiber_ds=4, fiber_dens_win_ds=15,
                     seed_grow_close=8, um_per_px=reference.reference_um_per_px * 2)
    actual = scaled(custom)
    assert actual.fiber_ds == 2 and actual.fiber_dens_win_ds == 15
    assert actual.seed_grow_close == 4 and actual.seed_grow_scale == 1
    assert actual.line_blur_sigma * actual.seed_grow_scale == pytest.approx(7.5)
    assert INVENTORY['geom_subsample_target'][1] == 'fixed'
    assert '性能' in INVENTORY['geom_subsample_target'][2]


def test_merge_close_optimization_branches_match():
    from nest_clear.vision.hair.core import _close_local_bounded
    rng = np.random.default_rng(134)
    mask = np.zeros((240, 360), np.uint8)
    for _ in range(20):
        x, y = rng.integers(10, 350), rng.integers(10, 230)
        cv2.circle(mask, (int(x), int(y)), 3, 255, -1)
    cfg = HairDetectConfig()
    direct = _close_local_bounded(mask, 15, cfg=replace(cfg, merge_close_bound_min_px=mask.size))
    bounded = _close_local_bounded(mask, 15, cfg=replace(cfg, merge_close_bound_min_px=1))
    np.testing.assert_array_equal(direct, bounded)


@pytest.mark.parametrize('name', ['big_blob', 'hair_with_tails'])
def test_downsampled_synthetic_thick_target(name):
    cfg = scaled(HairDetectConfig())
    k = cfg.reference_um_per_px / cfg.um_per_px
    image = frame_for(name)
    small = cv2.resize(image, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    candidates = detect_candidates(small, cfg)
    # Both scenes have a dark core over 0.6 mm wide. The detected pick must
    # land in that original core, even after fourfold resolution reduction.
    def in_target(c):
        x, y = (int(round(v / k)) for v in c.pick_uv)
        return 0 <= x < image.shape[1] and 0 <= y < image.shape[0] and image[y, x, 0] < 40
    assert any(in_target(c) for c in candidates)


@pytest.mark.parametrize('filename', ['20260814_144406_352_8.png', '20260805_204006_3_0.png', '20260813_194541_1365_1.png'])
def test_downsampled_real_thick_target(filename):
    path = Path.home() / 'pluck-ds/patches' / filename
    if not path.parent.exists():
        pytest.skip(f'现场图块目录不存在：{path.parent}')
    assert path.is_file()
    image = cv2.imread(str(path))
    golden = json.loads((ROOT / 'tests/data/hair_parity/real' / (path.stem + '.json')).read_text())
    # Reference clumps with mean width >= 0.6 mm: validate that at least one
    # resized-image pick remains inside one of these reference polygons.
    polys = [np.asarray(c['polygon_xy'], np.float32) for c in golden['clumps']
             if c['mask_area'] / max(1, c['length_px']) >= 0.6 / 0.006959 and c['length_px'] > 140]
    assert polys
    cfg = scaled(HairDetectConfig()); k = cfg.reference_um_per_px / cfg.um_per_px
    small = cv2.resize(image, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    candidates = detect_candidates(small, cfg)
    assert any(cv2.pointPolygonTest(poly, (c.pick_uv[0] / k, c.pick_uv[1] / k), False) >= 0
               for poly in polys for c in candidates)


def config_with_hair(tmp_path, hair):
    data = yaml.safe_load((ROOT / 'config/nest_clear.yaml').read_text())
    data['vision']['hair_detector'] = hair
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(data))
    return load_config(path, require_tools=False).vision.hair_detector


def test_load_hair_config(tmp_path):
    config = config_with_hair(tmp_path, {'um_per_px': 27.75, 'pick_point': {'central_pow': 3}})
    assert config.central_pow == 3
    assert config.line_min_area == 150
    assert config.reference_um_per_px == 6.959
    integer_sigma = config_with_hair(tmp_path, {'um_per_px': 27.836, 'line_blur_sigma': 15})
    assert scaled(integer_sigma).line_blur_sigma == pytest.approx(3.75)


@pytest.mark.parametrize('hair', [None, {'um_per_px': 0}, {'reference_um_per_px': -1},
    {'um_per_px': True}, {'dark_open_k': 2}, {'dark_open_k': 4}, {'bridge_max_hop': 1.5},
    {'min_dark_frac': 1.2}, {'pick_point': {'mode': 'legacy'}}, {'pick_point': {'oops': 1}},
    {'bridge_flank_w': [24, 16]}, {'pick_refine_enable': 1}, {'made_up': 1},
    {'fiber_fine_gray_max': 50}, {'central_pow': 2, 'pick_point': {'central_pow': 2}}])
def test_invalid_config(tmp_path, hair):
    with pytest.raises(GuardError, match='vision.hair_detector'):
        config_with_hair(tmp_path, hair)
