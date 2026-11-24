"""Manual exposure measurement for bright-field silhouettes."""
import math
import numpy as np
import cv2

DEFAULTS = dict(target_low=200, target_high=235, max_saturated_fraction=0.001, roi_fraction=0.8, settle_frames=3, max_iterations=12, min_exposure=1, max_exposure=1600)

class ExposureError(ValueError):
    def __init__(self, message, iterations):
        super().__init__(message)
        self.iterations = list(iterations)

def measure(image, roi_fraction=0.8):
    h, w = image.shape; dy = int(h * (1-roi_fraction)/2); dx = int(w * (1-roi_fraction)/2)
    roi = image[dy:h-dy or h, dx:w-dx or w]
    # Otsu segmentation is only trusted when a small, genuinely dark component exists.
    threshold, mask = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    dark = mask != 0
    contrast = None
    if 0.0001 < dark.mean() < 0.35 and threshold < np.percentile(roi, 90) * .8:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(dark.astype('uint8'))
        if n > 1:
            component = labels == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            if component.sum() >= 5:
                excluded = cv2.dilate(component.astype('uint8'), np.ones((5,5), np.uint8)) != 0
                background = roi[~excluded]
                if background.size:
                    contrast = float(np.median(background) - np.median(roi[component]))
                    roi = background
    return dict(p99=float(np.percentile(roi, 99)), saturated_fraction=float(np.mean(roi >= 250)), contrast=contrast)

def auto_exposure(camera, **options):
    """Set exposure and return measurement/iterations; raises ValueError on failure."""
    initial = options.pop('initial_exposure', None)
    cfg = {**DEFAULTS, **options}
    minimum, maximum = cfg['min_exposure'], cfg['max_exposure']
    if initial is None:
        if hasattr(camera, 'get_exposure'):
            initial = camera.get_exposure()
        else:
            initial = getattr(camera, 'config', {}).get('camera_controls', {}).get('exposure', 100)
    value = max(minimum, min(maximum, int(initial)))
    low = high = None  # Last exposures known to be too dark / too bright.
    history = []
    while len(history) < cfg['max_iterations']:
        camera.set_exposure(value)
        for _ in range(cfg['settle_frames']): camera.capture()
        if hasattr(camera, 'get_exposure'):
            actual = camera.get_exposure()
            if actual != value:
                raise ExposureError(f'曝光回读不一致：期望 {value}，实际 {actual}', history)
        row = {'exposure': value, **measure(camera.capture(), cfg['roi_fraction'])}; history.append(row)
        bright = row['p99'] > cfg['target_high'] or row['saturated_fraction'] >= cfg['max_saturated_fraction']
        dark = row['p99'] < cfg['target_low']
        if not bright and not dark: return {**row, 'iterations': history}
        if bright:
            if value == minimum: raise ExposureError(f'曝光达到 {minimum} 仍过曝', history)
            high = value
        else:
            if value == maximum: raise ExposureError(f'曝光达到 {maximum} 仍偏暗', history)
            low = value
        if low is None:
            value = max(minimum, value // 2)
        elif high is None:
            value = min(maximum, value * 2)
        else:
            if high - low <= 1: break
            value = max(low + 1, min(high - 1, round(math.sqrt(low * high))))
    raise ExposureError('自动测曝光未收敛：目标亮度与饱和比例无法同时满足或迭代次数不足', history)
