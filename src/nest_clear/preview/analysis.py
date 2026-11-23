"""预览图分析；不访问设备或归档。"""
import cv2
import numpy as np

from nest_clear.camcal.config import CameraCalibConfig
from nest_clear.camcal.session import _board


def detect_charuco(gray, config):
    """使用标定板配置检测原图角点，兼容新旧 OpenCV。"""
    cfg = CameraCalibConfig(**config)
    if hasattr(cv2.aruco, 'CharucoDetector'):
        board = _board(cfg)
    else:
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.dictionary))
        board = cv2.aruco.CharucoBoard_create(cfg.cols, cfg.rows, cfg.square_mm, cfg.marker_mm, dictionary)
    try:
        if hasattr(cv2.aruco, 'CharucoDetector'):
            corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(gray)
        else:
            dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.dictionary))
            markers, marker_ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
            if marker_ids is None:
                return board, None, None
            _, corners, ids = cv2.aruco.interpolateCornersCharuco(markers, marker_ids, gray, board)
    except cv2.error:
        corners, ids = None, None
    return board, corners, ids


def analyze(image, config, actual_xyz=None):
    cfg = CameraCalibConfig(**config)
    h, w = image.shape[:2]
    scale = min(1., 1600. / max(h, w))
    view = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image.copy()
    if view.ndim == 2:
        view = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
    height, width = view.shape[:2]
    gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
    center_crop = gray[height // 3:2 * height // 3, width // 3:2 * width // 3]
    sharpness = float(cv2.Laplacian(center_crop, cv2.CV_64F).var())
    board, corners, ids = detect_charuco(gray, config)
    count = 0 if ids is None else len(ids)
    offset_mm = offset_px = None
    if count:
        cv2.aruco.drawDetectedCornersCharuco(view, corners, ids)
    if count >= 8:
        points = np.asarray(board.getChessboardCorners())[ids.ravel(), :2]
        pixels = np.asarray(corners).reshape(-1, 2)
        matrix, _ = cv2.findHomography(points, pixels, cv2.RANSAC, 3.)
        if matrix is not None:
            try:
                bc = np.array([[[cfg.cols * cfg.square_mm / 2, cfg.rows * cfg.square_mm / 2]]], np.float32)
                ic = np.array([[[width / 2, height / 2]]], np.float32)
                projected = cv2.perspectiveTransform(bc, matrix)[0, 0]
                inverse = cv2.perspectiveTransform(ic, np.linalg.inv(matrix))[0, 0]
                offset_mm = (bc[0, 0] - inverse).tolist()
                offset_px = ((projected - ic[0, 0]) / scale).tolist()
                px, py = map(int, projected)
                cv2.drawMarker(view, (px, py), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
            except (np.linalg.LinAlgError, cv2.error):
                pass
    cv2.line(view, (width // 2, 0), (width // 2, height - 1), (255, 255, 0), 1)
    cv2.line(view, (0, height // 2), (width - 1, height // 2), (255, 255, 0), 1)
    cv2.circle(view, (width // 2, height // 2), 8, (255, 255, 0), 1)
    lines = [f'XYZ: {actual_xyz if actual_xyz is not None else "--"}',
             f'Sharpness: {sharpness:.1f}  Corners: {count}',
             f'Board offset mm: {offset_mm}' if offset_mm is not None else 'Board: not detected']
    for index, line in enumerate(lines):
        cv2.putText(view, line, (12, 28 + 28 * index), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 0), 4)
        cv2.putText(view, line, (12, 28 + 28 * index), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1)
    return view, {'actual_xyz': actual_xyz, 'sharpness': sharpness, 'corners': count,
                  'board_offset_mm': offset_mm, 'offset_px': offset_px}
