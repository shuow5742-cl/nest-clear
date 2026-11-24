"""原分辨率对焦和曝光分析。"""
from collections import deque
from .focus_record import write_record

import cv2
import numpy as np


def analyze_focus(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape
    region = gray[h//10:h-h//10, w//10:w-w//10]
    rh, rw = region.shape
    scores, laps, means = [], [], []
    for y in range(3):
        for x in range(3):
            cell = region[y*rh//3:(y+1)*rh//3, x*rw//3:(x+1)*rw//3]
            f = cell.astype(np.float32)
            gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
            mean = float(cv2.mean(cell)[0])
            scores.append(float((cv2.mean(gx*gx)[0] + cv2.mean(gy*gy)[0]) / max(mean, 1)**2))
            laps.append(float(cv2.Laplacian(f, cv2.CV_32F).var()))
            means.append(mean)
    hist = cv2.calcHist([region], [0], None, [256], [0, 256]).ravel()
    cumulative = np.cumsum(hist)
    n = region.size
    p1, p99 = (int(np.searchsorted(cumulative, n*p)) for p in (.01, .99))
    color_region = image[h//10:h-h//10, w//10:w-w//10]
    maximum = cv2.max(cv2.max(color_region[:,:,0], color_region[:,:,1]), color_region[:,:,2]) if image.ndim == 3 else region
    saturated = float(cv2.countNonZero(cv2.inRange(maximum, 250, 255)) / n * 100)
    dark = float(cumulative[5] / n * 100)
    return {'scores': scores, 'laplacian': laps, 'cell_means': means,
            'mean': float(cv2.mean(region)[0]), 'p1': p1, 'p99': p99,
            'saturated_pct': saturated, 'dark_pct': dark}


def exposure_hint(data):
    return '过曝：收小光圈或减曝光' if data['saturated_pct'] > .5 else '偏暗' if data['p99'] < 120 else ''


class FocusSession:
    def __init__(self):
        self.history = deque(maxlen=3)
        self.peaks = [0.] * 9

    def reset(self):
        self.peaks = [0.] * 9

    def update(self, image):
        data = analyze_focus(image)
        self.history.append(data.copy())
        for key in ('scores', 'laplacian', 'cell_means'):
            data[key] = np.median([d[key] for d in self.history], axis=0).tolist()
        for key in ('mean', 'p1', 'p99', 'saturated_pct', 'dark_pct'):
            data[key] = float(np.median([d[key] for d in self.history]))
        self.peaks = [max(a, b) for a, b in zip(self.peaks, data['scores'])]
        data['peaks'] = self.peaks.copy()
        data['percent'] = [100*s/p if p else 0 for s, p in zip(data['scores'], self.peaks)]
        data['center'] = data['scores'][4]
        data['median'] = float(np.median(data['scores']))
        corners = [(0, '左上'), (2, '右上'), (6, '左下'), (8, '右下')]
        different = [name for i, name in corners if abs(data['percent'][i] - data['percent'][4]) > 10]
        data['tilt'] = '四角清晰度不一致，可能镜头/标定纸不平或场曲：' + '、'.join(different) if different else ''
        data['exposure_hint'] = exposure_hint(data)
        return data


def draw_focus(image, data):
    view = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    h, w = view.shape[:2]
    x0, x1, y0, y1 = w//10, w-w//10, h//10, h-h//10
    for i in range(9):
        x, y = i % 3, i // 3
        a, b = x0+x*(x1-x0)//3, y0+y*(y1-y0)//3
        c, d = x0+(x+1)*(x1-x0)//3, y0+(y+1)*(y1-y0)//3
        pct = data.get('percent', [0]*9)[i]
        color = (40, 190, 40) if pct >= 98 else (0, 200, 255) if pct >= 90 else (40, 40, 240)
        cv2.rectangle(view, (a,b), (c,d), color, 2)
        cv2.putText(view, f'{pct:.1f}%', (a+6,b+25), cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2)
    return view
