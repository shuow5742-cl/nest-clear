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
    def __init__(self, reset_distance=50, smooth_frames=3, pass_percent=92.0):
        self.reset_distance = reset_distance
        self.smooth_frames = max(1, int(smooth_frames))
        self.pass_percent = float(pass_percent)
        self.reset()

    def reset(self):
        self.best = None
        self.position = None
        self.widths = deque(maxlen=self.smooth_frames)
        self.sample_count = 0
        self.peak_confirmed = False

    def update(self, width, position=None):
        if width is None or not np.isfinite(width) or width <= 0:
            return {'raw': None, 'current': None, 'best': self.best, 'percent': None,
                    'sample_count': self.sample_count, 'peak_confirmed': self.peak_confirmed}
        if position is not None and self.position is not None and np.linalg.norm(np.asarray(position) - self.position) > self.reset_distance:
            self.reset()
        if position is not None:
            self.position = np.asarray(position, dtype=float)
        raw = float(width)
        self.sample_count += 1
        self.widths.append(raw)
        if len(self.widths) < self.smooth_frames:
            return {'raw': raw, 'current': None, 'best': self.best, 'percent': None,
                    'sample_count': self.sample_count, 'peak_confirmed': self.peak_confirmed}
        median = float(np.median(self.widths))
        if self.best is None or median < self.best:
            self.best = median
            self.peak_confirmed = False
        percent = min(100., 100. * self.best / raw) if self.best else None
        if percent is not None and percent <= self.pass_percent:
            self.peak_confirmed = True
        return {'raw': raw, 'current': median, 'best': self.best, 'percent': percent,
                'sample_count': self.sample_count, 'peak_confirmed': self.peak_confirmed}


class BandGradientTracker:
    """N-frame smoothed median, smoothed peak reference, and overshoot confirmation."""
    def __init__(self, smooth_frames=5, pass_percent=92.0, peak_reset_margin_percent=3.0):
        self.smooth_frames = max(1, int(smooth_frames))
        self.pass_percent = float(pass_percent)
        self.peak_reset_margin_percent = float(peak_reset_margin_percent)
        if not (0.0 <= self.peak_reset_margin_percent < 100.0 - self.pass_percent):
            raise ValueError(
                f"peak_reset_margin_percent must satisfy 0 <= x < 100 - pass_percent ({100.0 - self.pass_percent:g})"
            )
        self.reset()

    def reset(self):
        self.values = deque(maxlen=self.smooth_frames)
        self.best = None
        self._confirmed_peak = None
        self.sample_count = 0
        self.peak_confirmed = False
        self.last_peak_event = ""

    def update(self, value, position=None):
        if value is None or not np.isfinite(value) or value <= 0:
            return {
                'raw': None, 'current': None, 'best': self.best, 'percent': None,
                'sample_count': self.sample_count, 'peak_confirmed': self.peak_confirmed,
                'best_updated': False, 'peak_confirm_kept': False,
                'peak_event': '', 'last_peak_event': self.last_peak_event,
            }
        raw = float(value)
        self.sample_count += 1
        self.values.append(raw)
        if len(self.values) < self.smooth_frames:
            return {
                'raw': raw, 'current': None, 'best': self.best, 'percent': None,
                'sample_count': self.sample_count, 'peak_confirmed': self.peak_confirmed,
                'best_updated': False, 'peak_confirm_kept': False,
                'peak_event': '', 'last_peak_event': self.last_peak_event,
            }
        current = float(np.median(self.values))
        best_updated = False
        peak_confirm_kept = False
        peak_event = ''
        if self.best is None:
            self.best = current
            self._confirmed_peak = None
            self.peak_confirmed = False
            best_updated = True
            peak_event = 'init'
            self.last_peak_event = peak_event
        elif current > self.best:
            best_updated = True
            was_confirmed = self.peak_confirmed
            # Anchor reset check to the confirmed peak so slow ramp steps cannot bypass the margin.
            ref_peak = self._confirmed_peak if (was_confirmed and self._confirmed_peak is not None) else self.best
            exceeds_margin = current >= ref_peak * (1.0 + self.peak_reset_margin_percent / 100.0)
            self.best = current
            if not was_confirmed:
                peak_event = 'new_peak'
            elif exceeds_margin:
                self.peak_confirmed = False
                self._confirmed_peak = None
                peak_event = 'reset'
            else:
                peak_confirm_kept = True
                peak_event = 'kept'
            self.last_peak_event = peak_event
        percent = 100 * current / self.best if self.best else None
        if percent is not None and percent <= self.pass_percent:
            if not self.peak_confirmed:
                peak_event = 'confirmed'
                self.last_peak_event = peak_event
            self.peak_confirmed = True
            self._confirmed_peak = self.best
        return {
            'raw': raw, 'current': current, 'best': self.best, 'percent': percent,
            'sample_count': self.sample_count, 'peak_confirmed': self.peak_confirmed,
            'best_updated': best_updated, 'peak_confirm_kept': peak_confirm_kept,
            'peak_event': peak_event, 'last_peak_event': self.last_peak_event,
        }
