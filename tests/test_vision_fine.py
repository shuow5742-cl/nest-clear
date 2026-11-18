"""合成候选验证精定位裁剪、状态和面积排序。"""
from unittest.mock import patch

import numpy as np
import pytest

from nest_clear.vision.fine import fine_camera_xy, select_fine_target
from nest_clear.vision.geometry import Calibration, pixel_to_module
from nest_clear.vision.hair import HairCandidate, HairDetectConfig


@pytest.fixture
def setup():
    k = np.array([[100., 0, 35], [0, 100., 45], [0, 0, 1]])
    calibration = Calibration(k, np.zeros(5), np.diag([1., -1., -1.]),
                              np.array([2., 3., 100.]), 0.)
    return np.full((100, 100, 3), 200, np.uint8), (10., 20., 0.), calibration


def candidate(x0, y0, x1, y1, pick=None):
    return HairCandidate(pick or ((x0 + x1) / 2, (y0 + y1) / 2), None,
                         ((x0, y0), (x1, y0), (x1, y1), (x0, y1)),
                         (x0, y0, x1, y1), ((x0 + x1) / 2, (y0 + y1) / 2),
                         x1 - x0, (x1 - x0) * (y1 - y0), None, None, None, "test")


def select(setup, candidates, **kwargs):
    image, pose, calibration = setup
    options = dict(box_half_mm=10., is_attempted=lambda c: False,
                   passes_gate=lambda c: True, cfg=HairDetectConfig())
    options.update(kwargs)
    return select_fine_target(image, candidates, pose, calibration, **options)


def test_camera_uses_calibration_principal_point(setup):
    _, pose, calibration = setup
    target = pixel_to_module(40, 50, pose, calibration)
    assert fine_camera_xy(target, pose, calibration) == pytest.approx((15, 15, 0))


def test_center_and_clipped_long_hair(setup):
    _, pose, calibration = setup
    for c in (candidate(32, 42, 38, 48), candidate(10, 42, 38, 48, (10, 45))):
        result = select(setup, [c])
        assert result.status == "ok"
        center = pixel_to_module(35, 45, pose, calibration)
        assert abs(result.target_xyz[0] - center[0]) <= 10
        assert abs(result.target_xyz[1] - center[1]) <= 10
        if c.pick_uv[0] == 10:
            assert result.pick_uv[0] >= 25
            assert result.offset_mm > 0


def test_empty_and_rejected(setup):
    c = candidate(32, 42, 38, 48)
    assert select(setup, [candidate(70, 70, 80, 80)]).status == "fine_empty"
    assert select(setup, [c], passes_gate=lambda c: False).status == "fine_rejected"
    assert select(setup, [c], is_attempted=lambda c: True).status == "fine_rejected"


def test_point_outside(setup):
    c = candidate(32, 42, 38, 48)
    with patch("nest_clear.vision.fine._hair_pick_point", return_value=(50, 50, 1, 0)):
        assert select(setup, [c]).status == "fine_point_outside"


def test_largest_physical_area_wins(setup):
    small = candidate(30, 42, 34, 46)
    large = candidate(35, 40, 43, 49)
    result = select(setup, [small, large])
    assert result.status == "ok"
    assert result.candidate is large


def test_fine_identity_is_projected_candidate_identity(setup):
    _, pose, calibration = setup
    c = candidate(10, 42, 38, 48, (10, 45))
    result = select(setup, [c])
    assert result.status == "ok"
    assert result.identity == pytest.approx(pixel_to_module(*c.identity_uv, pose, calibration))
    assert result.identity != result.target_xyz
