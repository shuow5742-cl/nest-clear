"""真实渲染与检测的分层误差、解算精度回归。"""
import cv2
import numpy as np

from nest_clear.camcal import CameraCalibConfig, CameraCalibrationSession, solve_offline
from nest_clear.camcal.session import _board
from nest_clear.vision.geometry import Calibration, load_calibration, pixel_to_module
from sim.charuco_scene import (CharucoScene, K, DISTORTION, ROTATION,
                               BOARD_ORIGIN, BOARD_ROTATION)


def test_rendered_session_accuracy_by_depth():
    cfg = CameraCalibConfig()
    nominal = load_calibration('config/calibration/examples/camera.yaml')
    scene = CharucoScene()
    session = CameraCalibrationSession(cfg, nominal, scene.size, (0, 0, 110))
    board_points = _board(cfg).getChessboardCorners()
    pose = np.array([0., 0., 110.])
    groups = {}
    for _ in range(53):
        step = session.add(scene.render(pose), pose)
        row = session.session_record()['observations'][-1]
        ids = np.asarray(row['ids'])
        camera_points = (board_points[ids] @ BOARD_ROTATION.T + BOARD_ORIGIN - pose) @ ROTATION
        projected = cv2.projectPoints(camera_points, np.zeros(3), np.zeros(3), K, DISTORTION)[0].reshape(-1, 2)
        groups.setdefault(round(pose[2] - 110, 1), []).append(np.asarray(row['pixels']) - projected)
        if step.kind != 'move':
            break
        pose = np.asarray(step.target_xyz)
    assert step.kind == 'done'
    for z, parts in sorted(groups.items()):
        delta = np.concatenate(parts)
        rms = np.sqrt(np.mean(np.sum(delta**2, axis=1)))
        print(f'Z={z:+.1f} n={len(delta)} mean_uv={delta.mean(axis=0)} '
              f'mean_length={np.linalg.norm(delta, axis=1).mean():.6f}px rms={rms:.6f}px')
        assert np.max(np.abs(delta.mean(axis=0))) < .01
        assert rms < .04
    assert set(groups) == {-5., -2.5, 0., 2.5, 5.}
    truth = Calibration(K, DISTORTION, ROTATION, np.zeros(3), 0., False, BOARD_ORIGIN, BOARD_ROTATION[:, 2])
    principal = np.asarray(pixel_to_module(K[0, 2], K[1, 2], (0, 0, 110), truth))
    rows = session.session_record()['observations']
    for label, observations in [('all', rows), ('inner', [r for r in rows if abs(r['actual_xyz'][2]-110) < 3])]:
        result = solve_offline({'observations': observations}, cfg, nominal, scene.size)
        regions = [[], [], []]
        for u in np.linspace(0, 4023, 41):
            for v in np.linspace(0, 3035, 31):
                actual = np.asarray(pixel_to_module(u, v, (0, 0, 110), result.calibration))
                reference = np.asarray(pixel_to_module(u, v, (0, 0, 110), truth))
                error = np.linalg.norm((actual - reference)[:2])
                radius = np.linalg.norm((reference - principal)[:2])
                regions[2].append(error)
                if radius <= 10:
                    regions[1].append(error)
                if radius <= 5:
                    regions[0].append(error)
        maxima = [max(region) for region in regions]
        print(f'{label}: max error 5mm/10mm/full = {maxima} mm')
        assert maxima[0] < .005 and maxima[2] < .01
