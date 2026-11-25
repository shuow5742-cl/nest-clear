"""Exposure-normalized focus and silhouette edge measurements."""
import cv2
import numpy as np
from collections import deque

def focus_metrics(image, tip=None, roi_size=160, um_per_px=15.2, silhouette_mask=None):
    image = np.asarray(image)
    if image.ndim != 2 or not image.size or image.dtype != np.uint8: raise ValueError('须为非空 uint8 灰度图')
    h, w = image.shape; x, y = tip if tip is not None else (w/2, h/2)
    x, y = int(round(x)), int(round(y)); half = roi_size // 2
    roi = image[max(0,y-half):min(h,y+half), max(0,x-half):min(w,x+half)]
    if roi.size == 0: raise ValueError('ROI 在图像之外')
    threshold, mask = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    dark = roi[mask == 0]; bright = roi[mask != 0]
    contrast = float(np.median(bright)-np.median(dark)) if dark.size and bright.size else 0.
    f = roi.astype(np.float32); gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    ten = float(np.mean(gx*gx+gy*gy)/contrast**2) if contrast > 0 else 0.
    # Sample along the local Sobel normal, rejecting profiles without stable plateaus.
    magnitude = cv2.magnitude(gx, gy); candidates = (magnitude > max(1., contrast * 0.5))
    if silhouette_mask is not None:
        mask_roi = silhouette_mask[max(0,y-half):min(h,y+half), max(0,x-half):min(w,x+half)].astype(np.uint8)
        boundary = cv2.morphologyEx(mask_roi, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        candidates &= cv2.dilate(boundary, np.ones((7, 7), np.uint8)) != 0
    coords = np.argwhere(candidates)[::max(1, int(candidates.sum()/300))]; widths = []
    t = np.arange(-20, 20.01, .1, dtype=np.float32)
    for yy, xx in coords:
        mag = magnitude[yy,xx]; nx, ny = gx[yy,xx]/mag, gy[yy,xx]/mag
        xs = xx+t*nx; ys = yy+t*ny
        if xs.min()<0 or ys.min()<0 or xs.max()>roi.shape[1]-1 or ys.max()>roi.shape[0]-1: continue
        vals = cv2.remap(f, xs.astype(np.float32).reshape(1,-1), ys.astype(np.float32).reshape(1,-1), cv2.INTER_LINEAR).ravel()
        low, high = np.median(vals[:30]), np.median(vals[-30:])
        if high-low < contrast*.65: continue
        def crossing(level):
            inds = np.flatnonzero(vals >= level)
            if not len(inds) or inds[0] == 0: return None
            i = inds[0]; a, b = vals[i-1:i+1]
            return float(t[i-1] + (level-a)/(b-a)*.1) if b>a else float(t[i])
        a, b = crossing(low+.1*(high-low)), crossing(low+.9*(high-low))
        if a is not None and b is not None and b>a: widths.append(b-a)
    width = float(np.median(widths)) if widths else None
    return {'source': 'tip' if tip is not None else 'center', 'tenengrad': ten, 'edge_width_px': width,
            'edge_width_um': width*um_per_px if width is not None else None, 'valid_profiles': len(widths), 'contrast': contrast}

class PeakTracker:
    def __init__(self, reset_distance=None):
        self.reset_distance = reset_distance; self.reset()
    def reset(self):
        self.current = None; self.peak = None; self.position = None
    def update(self, value, position=None):
        if position is not None and self.position is not None and self.reset_distance is not None and np.linalg.norm(np.asarray(position)-self.position) > self.reset_distance: self.reset()
        self.current = float(value); self.peak = max(self.current, self.peak if self.peak is not None else self.current)
        self.position = np.asarray(position, dtype=float) if position is not None else None
        return {'current': self.current, 'peak': self.peak, 'percent': 100*self.current/self.peak if self.peak else 0.}


class EdgeWidthTracker:
    def __init__(self, reset_distance=50):
        self.reset_distance = reset_distance
        self.reset()

    def reset(self):
        self.best = None
        self.position = None
        self.widths = deque(maxlen=3)

    def update(self, width, position=None):
        if width is None or not np.isfinite(width) or width <= 0:
            return {'best': self.best, 'percent': None}
        if position is not None and self.position is not None and np.linalg.norm(np.asarray(position) - self.position) > self.reset_distance:
            self.reset()
        if position is not None:
            self.position = np.asarray(position, dtype=float)
        self.widths.append(float(width))
        if len(self.widths) < 3:
            return {'best': self.best, 'percent': None}
        median = float(np.median(self.widths))
        self.best = min(median, self.best if self.best is not None else median)
        return {'best': self.best, 'percent': min(100., 100. * self.best / width)}


class BandGradientTracker:
    """Five-frame display median and three-frame median peak reference."""
    def __init__(self): self.reset()

    def reset(self):
        self.values = deque(maxlen=5)
        self.best = None

    def update(self, value, position=None):
        if value is None or not np.isfinite(value) or value <= 0:
            return {'current': None, 'best': self.best, 'percent': None}
        self.values.append(float(value))
        if len(self.values) >= 3:
            candidate = float(np.median(list(self.values)[-3:]))
            self.best = max(candidate, self.best if self.best is not None else candidate)
        current = float(np.median(self.values)) if len(self.values) == 5 else None
        return {'current': current, 'best': self.best,
                'percent': 100 * current / self.best if current is not None and self.best else None}
