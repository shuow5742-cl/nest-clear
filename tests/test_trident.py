import cv2
import numpy as np
import pytest
from nest_clear.tooltip.trident import TridentTipDetector


def scene():
    y, x = np.indices((600, 600))
    image = np.clip(225 + 30*np.sin(x/18)*np.sin(y/17), 0, 255).astype(np.uint8)
    image[:, :70] = 0
    image[:, 530:] = 0
    cv2.rectangle(image, (250, 0), (350, 100), 20, -1)
    cv2.rectangle(image, (285, 100), (315, 400), 20, -1)
    for u, end in ((290, 440), (300, 450), (310, 445)):
        cv2.line(image, (u, 400), (u, end), 30, 3)
    return image


def test_geometry_and_definition():
    result = TridentTipDetector().detect(scene())
    assert result.point_uv is not None
    assert result.status in ('精定位', '粗定位')
    assert abs(result.point_uv[0] - 300) < 1
    assert abs(result.point_uv[1] - 452) < 2
    axis = result.axis
    d = np.array(axis['direction']); p = np.array(result.point_uv); o = np.array(axis['point'])
    far = np.array(result.farthest_prong_uv)
    assert abs(d[0]*(p-o)[1] - d[1]*(p-o)[0]) < 1e-8
    assert abs(np.dot(d, p-far)) < 1e-8


def test_gradient_falls_with_blur():
    image = scene(); detector = TridentTipDetector(); scores = []; esf_widths = []
    for sigma in (0, 1, 2, 3):
        blurred = cv2.GaussianBlur(image, (0, 0), sigma) if sigma else image
        r = detector.detect(blurred)
        assert r.point_uv is not None
        scores.append(r.sharpness['normalized_gradient'])
        esf_widths.append(r.sharpness['esf_width_px'])
    assert all(a > b for a, b in zip(scores, scores[1:]))
    assert all(a < b for a, b in zip(esf_widths, esf_widths[1:]))


def test_missing_and_invalid():
    result = TridentTipDetector().detect(np.full((600, 600), 255, np.uint8))
    assert result.status == '未识别' and result.point_uv is None
    with pytest.raises(ValueError):
        TridentTipDetector().detect(np.zeros((20, 20, 3), np.uint8))


def test_rolling_shutter_distortion_is_reported():
    image = scene()
    distorted = np.stack([np.roll(row, 8 if (y // 9) % 2 else -8)
                          for y, row in enumerate(image)])
    result = TridentTipDetector().detect(distorted)
    assert result.axis_rms_px > 2.0
    assert result.status != '精定位'
    assert '工具在动或画面畸变' in result.diagnostic


def test_sequence_quality_gates_motion_without_hiding_positions():
    detector = TridentTipDetector()
    image = scene()
    stable = detector.detect_sequence([image] * 5)
    assert all(r.status == '精定位' and '五帧一致性检查通过' in r.diagnostic for r in stable)
    moved = detector.detect_sequence([np.roll(image, i, axis=1) for i in range(5)])
    assert all(r.status == '粗定位' and '标准差超限' in r.diagnostic for r in moved)
    assert all(r.point_uv is not None for r in moved)
    with pytest.raises(ValueError):
        detector.detect_sequence([image])


def test_axis_rms_ignores_isolated_edge_artifacts():
    image = scene()
    detector = TridentTipDetector()
    reference = detector.detect(image)
    for y in (240, 270, 300):
        noisy = image.copy()
        cv2.rectangle(noisy, (315, y), (322, y + 2), 20, -1)
        result = detector.detect(noisy)
        assert abs(result.axis_rms_px - reference.axis_rms_px) < .02
        assert np.linalg.norm(np.asarray(result.point_uv) - reference.point_uv) < .1


def test_axis_uses_outer_edge_of_reflective_facet():
    frame = scene()
    frame[100:400, :][frame[100:400, :] > 100] = 250
    frame[100:400, 301:316] = 195
    result = TridentTipDetector().detect(frame)
    assert result.focused
    assert abs(result.point_uv[0] - 300) < .3
    assert result.edge_fit['right']['background_gray'] == pytest.approx(250)
    assert result.edge_fit['right']['tool_gray'] == pytest.approx(195)
    assert abs(result.edge_fit['right']['slope_du_dv']) < .001


@pytest.mark.parametrize('mirror', (False, True))
def test_narrow_bright_halo_does_not_pull_the_axis(mirror):
    frame = scene()
    shaft = frame[100:400]
    shaft[shaft > 100] = 250
    frame[170:400, 283:285] = 225
    if mirror:
        frame = frame[:, ::-1].copy()
    result = TridentTipDetector().detect(frame)
    assert result.focused
    expected = frame.shape[1] - 1 - 300 if mirror else 300
    assert abs(result.point_uv[0] - expected) < .25
