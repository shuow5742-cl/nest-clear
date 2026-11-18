import cv2
import numpy as np
import pytest

from sim.charuco_scene import (CharucoScene, K, DISTORTION, ROTATION,
                               BOARD_ORIGIN, BOARD_ROTATION)
from nest_clear.camcal.session import _board


@pytest.mark.parametrize('pose', [(0, 0, 110), (15, -15, 110), (-10, 10, 105)])
def test_rendered_corners_match_projection(pose):
    scene = CharucoScene()
    image = scene.render(pose)
    assert image.shape == (3036, 4024)
    corners, ids, _, _ = cv2.aruco.CharucoDetector(_board(scene.config)).detectBoard(image)
    assert ids is not None and len(ids) > 50
    points = _board(scene.config).getChessboardCorners()[ids.ravel()]
    camera_points = (points @ BOARD_ROTATION.T + BOARD_ORIGIN - pose) @ ROTATION
    expected = cv2.projectPoints(camera_points, np.zeros(3), np.zeros(3), K, DISTORTION)[0].reshape(-1, 2)
    assert np.max(np.linalg.norm(corners.reshape(-1, 2) - expected, axis=1)) < .1
