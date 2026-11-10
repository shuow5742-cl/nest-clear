import json
from pathlib import Path

import pytest

from nest_clear.vision.hair import HairDetectConfig, detect_candidates, gen1_pixel_gates, scaled

import cv2
from hair_scenes import SCENES, frame_for

DATA = Path(__file__).parent / 'data/hair_parity'


def assert_parity(frame, golden):
    cfg = scaled(HairDetectConfig(um_per_px=6.959))
    candidates = detect_candidates(frame, cfg)
    assert len(candidates) == len(golden['clumps'])
    eligible = [c for c in candidates if gen1_pixel_gates(c, cfg)[0]]
    assert len(eligible) == len(golden['hairs'])
    for c, hair in zip(eligible, golden['hairs']):
        assert c.pick_uv == pytest.approx(hair['px'], abs=1e-6)
        assert c.angle_deg == pytest.approx(hair['pick_angle_deg'], abs=1e-6)
    for c, old, pale in zip(candidates, golden['clumps'], golden['paleness']):
        assert gen1_pixel_gates(c, cfg) == (old['eligible_for_pick'], old['reject_reason'])
        assert c.pick_uv == pytest.approx(old['pick_point_xy'], abs=1e-6)
        assert c.angle_deg == pytest.approx(old['pick_angle_deg'], abs=1e-6)
        assert c.length_px == pytest.approx(old['length_px'], abs=1e-6)
        assert c.area_px == old['mask_area']
        assert c.bbox == pytest.approx(tuple(old['bbox'].values()), abs=1e-6)
        assert c.polygon == tuple(map(tuple, old['polygon_xy']))
        assert c.source == old['pick_method']
        assert (c.gray_ratio, c.dark_frac) == pytest.approx(pale, abs=1e-6)


@pytest.mark.parametrize('name', SCENES)
def test_gen1_parity(name):
    golden = json.loads((DATA / f'{name}.json').read_text())
    assert_parity(frame_for(name), golden)


REAL_GOLDENS = sorted((DATA / 'real').glob('*.json'))


def test_real_golden_manifest():
    assert len(REAL_GOLDENS) == 60


@pytest.mark.parametrize('path', REAL_GOLDENS, ids=lambda p: p.stem)
def test_real_parity(path):
    patches = Path.home() / 'pluck-ds/patches'
    if not patches.is_dir():
        pytest.skip(f'gen1 real patches unavailable: {patches}')
    golden = json.loads(path.read_text())
    image_path = patches / golden['filename']
    assert image_path.is_file(), f'Missing sampled patch: {image_path}'
    frame = cv2.imread(str(image_path))
    assert frame is not None
    assert_parity(frame, golden)


def test_gate_thresholds_and_unmeasurable_paleness():
    from dataclasses import replace
    from nest_clear.vision.hair import HairCandidate

    cfg = HairDetectConfig(um_per_px=6.959)
    c = HairCandidate((0., 0.), 0., (), (0., 0., 1., 1.), (0.5, 0.5),
                      300., 25000, None, None, 'no background', 'cv-dark')
    assert gen1_pixel_gates(c, cfg) == (True, None)
    assert gen1_pixel_gates(replace(c, length_px=100), cfg) == (False, 'short')
    assert gen1_pixel_gates(replace(c, length_px=600), cfg) == (False, 'thin')
    assert gen1_pixel_gates(replace(c, gray_ratio=0.71), cfg) == (False, 'pale')
    assert gen1_pixel_gates(replace(c, dark_frac=0.14), cfg) == (False, 'pale')
