"""对焦预览内的曝光扫描与人工确认后的持久化。"""
from datetime import datetime
import math

import cv2
import numpy as np

from nest_clear.preview.analysis import detect_charuco
from nest_clear.preview.focus import analyze_focus


def measure_exposure(image, config):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, corners, ids = detect_charuco(gray, config)
    count = 0 if ids is None else len(ids)
    h, w = gray.shape
    mask = np.zeros((h, w), np.uint8)
    if count >= 3:
        hull = cv2.convexHull(np.rint(np.asarray(corners).reshape(-1, 2)).astype(np.int32))
        if cv2.contourArea(hull) > 0:
            cv2.fillConvexPoly(mask, hull, 255)
    if not cv2.countNonZero(mask):
        mask[h//10:h-h//10, w//10:w-w//10] = 255
    total = cv2.countNonZero(mask)
    hist = cv2.calcHist([gray], [0], mask, [256], [0, 256]).ravel()
    cumulative = np.cumsum(hist)
    p99 = int(np.searchsorted(cumulative, total * .99))
    maximum = np.max(image, axis=2) if image.ndim == 3 else gray
    saturated = cv2.countNonZero(cv2.bitwise_and(cv2.inRange(maximum, int(config.get('saturation_level', 250)), 255), mask)) / total
    return {'saturated_fraction': float(saturated), 'p99': p99,
            'dark_fraction': float(cumulative[5] / total), 'corners': count,
            'sharpness': analyze_focus(image)['scores'][4],
            'region': 'board' if count >= 3 and cv2.contourArea(hull) > 0 else 'center'}


def select_exposure(rows, mode, max_saturated_fraction):
    if not rows:
        return None, '没有有效扫描数据'
    threshold = max_saturated_fraction if mode == 'calibration' else .005
    eligible = [r for r in rows if r['saturated_fraction'] <= threshold]
    if not eligible:
        return None, '所有档都过曝：收小光圈'
    if max(r['p99'] for r in rows) < 120:
        return None, '最长曝光仍太暗：开大光圈或加光'
    if mode == 'calibration':
        best = max(r['corners'] for r in eligible)
        if not best:
            return None, '未检出标定板'
        tied = sorted((r for r in eligible if r['corners'] >= .98 * best), key=lambda r: r['exposure_us'])
        return tied[(len(tied)-1)//2], ''
    if mode == 'brightness':
        return min(eligible, key=lambda r: (abs(r['p99']-220), r['exposure_us'])), ''
    raise ValueError('未知曝光自测模式')


class ExposureRestoreError(RuntimeError):
    pass


def run_sweep(worker, settings, config, mode, *, progress=lambda _: None, cancelled=lambda: False, measure=measure_exposure):
    if mode not in ('calibration', 'brightness'):
        raise ValueError('未知曝光自测模式')
    original = worker.readback.get('exposure_us')
    if original is None:
        raise ValueError('无法回读当前曝光，不能安全恢复')
    auto_mode = worker.exposure_auto_mode() if hasattr(worker, 'exposure_auto_mode') else None
    total = settings['steps']
    rows = []
    result = {'state': 'running', 'mode': mode, 'total': total, 'index': 0, 'rows': rows,
              'original_us': original, 'recommendation': None, 'reason': ''}
    try:
        for index in range(total):
            if cancelled():
                result.update(state='cancelled', reason='已取消'); break
            exposure = settings['min_us'] * (settings['max_us']/settings['min_us']) ** (index/(total-1))
            result.update(index=index+1, phase='capture')
            progress(result)
            actual = worker.set_exposure(exposure)
            for _ in range(settings['settle_frames']):
                if cancelled(): break
                worker.execute({'stage': 'preview'})
            if cancelled():
                result.update(state='cancelled', reason='已取消'); break
            image, _ = worker.execute({'stage': 'preview'})
            row = measure(image, config)
            row['exposure_us'] = actual
            rows.append(row)
            progress(result)
        else:
            chosen, reason = select_exposure(rows, mode, config.get('max_saturated_fraction', .01))
            result.update(state='done', recommendation=chosen, reason=reason)
    finally:
        try:
            worker.set_exposure(original)
            if hasattr(worker, 'restore_exposure_auto_mode'):
                worker.restore_exposure_auto_mode(auto_mode)
        except Exception as exc:
            raise ExposureRestoreError(f'恢复原曝光失败：{exc}') from exc
    progress(result)
    return result


def save_exposure(path, result):
    from nest_clear.tooloffset.store import save_mapping
    if result.get('state') != 'done' or not result.get('recommendation'):
        raise ValueError('没有可应用的曝光推荐')
    exposure = result['recommendation']['exposure_us']
    if not math.isfinite(exposure) or exposure <= 0:
        raise ValueError('曝光推荐无效')
    data = {'exposure_us': float(exposure), 'mode': result['mode'],
            'time': datetime.now().astimezone().isoformat(),
            'scan_summary': {'original_us': result['original_us'], 'rows': result['rows']}}
    return save_mapping(path, data, 'camera_exposure')


def demo_sweep():
    """合成标定板和线性曝光相机，使用实际扫描及角点检测。"""
    cfg = {'cols': 8, 'rows': 6, 'square_mm': 5.0, 'marker_mm': 3.75,
           'dictionary': 'DICT_5X5_250', 'saturation_level': 250, 'max_saturated_fraction': .01}
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    if hasattr(cv2.aruco, 'CharucoDetector'):
        board = cv2.aruco.CharucoBoard((8, 6), 5.0, 3.75, dictionary)
        target = board.generateImage((800, 600), marginSize=12)
    else:
        board = cv2.aruco.CharucoBoard_create(8, 6, 5.0, 3.75, dictionary)
        target = board.draw((800, 600), marginSize=12)
    source = np.full((720, 960), 170, np.uint8)
    source[60:660, 80:880] = (target.astype(np.float32) * .72 + 12).astype(np.uint8)
    class MockExposureCamera:
        def __init__(self): self.readback = {'exposure_us': 6000.0}
        def set_exposure(self, value): self.readback['exposure_us'] = float(value); return float(value)
        def execute(self, payload):
            image = np.clip(source.astype(np.float32) * (self.readback['exposure_us']/10000), 0, 255).astype(np.uint8)
            return image, {}
    return run_sweep(MockExposureCamera(), {'min_us': 500, 'max_us': 30000, 'steps': 10, 'settle_frames': 1}, cfg, 'calibration')
