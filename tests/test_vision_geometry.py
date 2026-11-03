"""用合成投影和目标集合检验视觉基础能力。"""
import cv2
import numpy as np
import pytest

from nest_clear.guard import GuardError
from nest_clear.vision.geometry import Calibration, pixel_to_module
from nest_clear.vision.detector import ModuleTarget, PixelTarget, postprocess


def test_projection_round_trip() -> None:
    """带畸变的投影通过去畸变及射线求交回到原坐标。"""
    k = np.array([[900., 0, 320], [0, 910, 240], [0, 0, 1]])
    d = np.array([0.01, -0.002, 0.001, -0.001, 0.])
    rotation = np.diag([1., -1., -1.])
    pose = np.array([12., 20., 5.])
    translation = np.array([2., 3., 100.])
    point = np.array([23., 28., 0.])
    camera_point = rotation.T @ (point - pose - translation)
    pixel, _ = cv2.projectPoints(camera_point.reshape(1, 3), np.zeros(3), np.zeros(3), k, d)
    calibration = Calibration(k, d, rotation, translation, 0.)
    assert pixel_to_module(*pixel[0, 0], tuple(pose), calibration) == pytest.approx(point, abs=1e-5)


def test_postprocess() -> None:
    """高分优先，重复目标剔除并限制数量。"""
    targets = [ModuleTarget(PixelTarget(0, 0, score), (x, 0, 0)) for x, score in [(0, .4), (.2, .9), (3, .8), (5, .7)]]
    assert [t.pixel.score for t in postprocess(targets, 1., 2)] == [.9, .8]
    with pytest.raises(GuardError):
        postprocess([ModuleTarget(PixelTarget(0, 0, float('nan')), (0, 0, 0))], 1, 2)
