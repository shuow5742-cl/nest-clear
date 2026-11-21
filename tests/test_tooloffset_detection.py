import cv2
import numpy as np

from nest_clear.tooloffset.detection import detect


def test_detect_synthetic_ring_and_cross():
    image = np.full((600, 800), 255, np.uint8)
    x, y, scale = 300, 300, 20
    cv2.circle(image, (x, y), round(1.85 * scale), 0, 6, cv2.LINE_AA)
    cv2.line(image, (x-100, y), (x+100, y), 0, 2, cv2.LINE_AA)
    cv2.line(image, (x, y-100), (x, y+100), 0, 2, cv2.LINE_AA)
    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 12, 80)
    image[y-180:y-100, x+100:x+180] = marker
    targets = detect(image)
    assert len(targets) == 1
    target = targets[0]
    assert target.label == 'T12'
    assert np.linalg.norm(np.array(target.ring_uv) - [x,y]) < .5
    assert target.cross_error_mm < .08


def test_crop_contains_marker_and_is_rendered_from_source():
    from nest_clear.workers.tooloffset import render_crop, render_annotations
    image = np.full((600,800),255,np.uint8); x=y=300
    cv2.circle(image, (x,y), 37, 0, 6, cv2.LINE_AA)
    cv2.line(image, (x-100,y), (x+100,y), 0, 2, cv2.LINE_AA)
    cv2.line(image, (x,y-100), (x,y+100), 0, 2, cv2.LINE_AA)
    image[120:200,400:480] = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 12, 80)
    target = detect(image)[0]
    before = image.copy()
    crop = render_crop(image, target, (300,300))
    overlay = render_annotations(image, [target], 12, (300,300))
    assert crop.shape[1] == 480 and crop.ndim == 3
    assert (crop < 50).all(axis=2).sum() > 100
    assert np.array_equal(image,before) and overlay.shape == (600,800,3)


def test_selected_target_has_yellow_box():
    from nest_clear.workers.tooloffset import render_annotations
    image = np.full((600,800), 255, np.uint8)
    cv2.circle(image, (300,300), 37, 0, 6, cv2.LINE_AA)
    cv2.line(image, (200,300), (400,300), 0, 2, cv2.LINE_AA)
    cv2.line(image, (300,200), (300,400), 0, 2, cv2.LINE_AA)
    image[120:200,400:480] = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 12, 80)
    target = detect(image)[0]
    selected = render_annotations(image, [target], 12, (500,500))
    ordinary = render_annotations(image, [target], None, (500,500))
    assert ((selected[:,:,0] < 20) & (selected[:,:,1] > 240) & (selected[:,:,2] > 240)).sum() > 1000
    assert not ((ordinary[:,:,0] < 20) & (ordinary[:,:,1] > 240) & (ordinary[:,:,2] > 240)).any()
