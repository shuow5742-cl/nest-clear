"""Tool-aware exposure sweep, executed in the camera owner thread."""
import math
import cv2
import numpy as np
from scipy.signal import find_peaks


class ExposureScanError(ValueError):
    def __init__(self, message, rows):
        super().__init__(message)
        self.rows = rows


def _prongs(mask):
    """Approximate teeth by separated distal silhouette segments.

    TridentResult.prong_tips are sampled profiles, not a tooth count. Count
    contiguous dark segments on distal rows instead; the maximum over several
    rows allows teeth of unequal lengths. This is an image-space approximation.
    """
    if mask is None or not np.any(mask):
        return 0
    ys = np.flatnonzero(np.any(mask, axis=1))
    end = int(ys[-1]); start = max(int(ys[0]), end - 100)
    counts = []
    for row in mask[start:end + 1]:
        indices = np.flatnonzero(row)
        if not len(indices):
            counts.append(0)
            continue
        groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
        counts.append(sum(len(group) >= 2 for group in groups))
    # Require ten contiguous rows to reduce transient splits from texture.
    return max((min(counts[i:i + 10]) for i in range(len(counts) - 9)), default=0)


def _profile_prongs(gray, mask):
    """Count persistent dark prong profiles near the distal silhouette end.

    At the accepted 0.2/0.3 ms, partial-volume pixels can merge the three
    prongs into two binary components even while their grey-level valleys are
    distinct. Count the dark valleys with 15-grey-level local prominence on
    each of ten consecutive rows in the last 35 pixels and take the minimum.
    This rejects isolated texture; it does not track individual valleys across rows.
    """
    if mask is None or not np.any(mask):
        return 0
    ys, xs = np.nonzero(mask)
    end = int(ys.max())
    # Only the terminal cross-section, with background on both sides.
    xlo = max(0, int(xs[ys >= end - 35].min()) - 12)
    xhi = min(gray.shape[1], int(xs[ys >= end - 35].max()) + 13)
    counts = []
    for y in range(max(0, end - 35), end + 1):
        row = gray[y, xlo:xhi].astype(np.float32)
        if len(row) < 5:
            counts.append(0); continue
        row = cv2.GaussianBlur(row.reshape(1, -1), (3, 1), 0).ravel()
        background = float(np.percentile(row, 90))
        peaks, _ = find_peaks(-row, prominence=15, distance=3)
        counts.append(int(np.count_nonzero(row[peaks] <= background - 40)))
    return max((min(counts[i:i + 10]) for i in range(len(counts) - 9)), default=0)


def _frame_metrics(frame, detector):
    gray = np.asarray(frame)
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError('曝光扫描需要 uint8 灰度图')
    det = detector.detect(gray)
    raw_mask = getattr(det, 'tool_mask', None)
    mask = np.asarray(raw_mask, dtype=bool) if raw_mask is not None else np.zeros(gray.shape, dtype=bool)
    if mask.shape != gray.shape:
        raise ValueError('检测器工具区域尺寸不匹配')
    h, w = gray.shape
    # Exclude the outer fifth on each side (frequent black support bars), and
    # a margin around the detected tool; retain the brighter half of the ROI.
    roi = np.zeros(gray.shape, dtype=bool)
    roi[:, w // 5:w - w // 5] = True
    excluded = cv2.dilate(mask.astype('uint8'), np.ones((11, 11), np.uint8)).astype(bool)
    background = gray[roi & ~excluded]
    if background.size:
        background = background[background >= np.percentile(background, 50)]
    # Fill holes inside the detected silhouette so washed-out pixels within
    # the tool are measured instead of being discarded by segmentation.
    contours, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros(gray.shape, dtype='uint8')
    if contours:
        cv2.drawContours(filled, contours, -1, 1, cv2.FILLED)
    tool = gray[filled.astype(bool)]
    sharp = (getattr(det, 'sharpness', None) or {}).get('band_gradient_p995')
    contrast = getattr(det, 'tip_contrast', None)
    return {'recognized': getattr(det, 'status', '') == '精定位' and '畸变' not in (getattr(det, 'diagnostic', '') or ''),
            'diagnostic': getattr(det, 'diagnostic', '') or '', 'prongs': max(_prongs(mask), _profile_prongs(gray, mask)),
            'contrast': float(contrast) if contrast is not None and math.isfinite(float(contrast)) else 0.,
            'sharpness': float(sharp) if sharp is not None and math.isfinite(float(sharp)) else 0.,
            'background': float(np.median(background)) if background.size else 0.,
            'p99': float(np.percentile(background, 99)) if background.size else 0.,
            'background_saturation': float(np.mean(background >= 250)) if background.size else 0.,
            'tool_whiteness': float(np.mean(tool >= 230)) if tool.size else 1.}


def absolute_criteria(metrics, expected_prongs, contrast_min, saturation_max):
    """Frame gates; expected_prongs is retained for API compatibility only.

    A three-pronged tool may project as two at some angles, so neither camera
    has a fixed expected count. The scan also applies a relative count gate.
    """
    return {'recognized': metrics['recognized'],
            'prongs': metrics['prongs'] >= 2,
            'contrast': metrics['contrast'] >= contrast_min,
            'whiteness': metrics['tool_whiteness'] <= saturation_max}


def scan_exposure(camera, detector, expected_prongs=None, *, levels, frames_per_level=3,
                  discard_per_level=1, contrast_min, sharpness_ratio=0.9,
                  saturation_max, progress=None):
    initial = camera.get_exposure()
    limits = camera.exposure_limits()
    values = list(range(1, levels + 1)) if isinstance(levels, int) else list(levels)
    if limits:
        values = [v for v in values if limits['min'] <= v <= limits['max'] and (v-limits['min']) % limits['step'] == 0]
    if not values or frames_per_level < 1 or discard_per_level < 0:
        raise ValueError('曝光扫描范围或帧数无效')
    rows = []
    try:
        for k, exposure in enumerate(values, 1):
            camera.set_exposure(exposure)
            for _ in range(discard_per_level):
                camera.capture()
            if camera.get_exposure() != exposure:
                raise ValueError(f'曝光回读不一致：期望 {exposure}')
            frames = [_frame_metrics(camera.capture(), detector) for _ in range(frames_per_level)]
            row = {'exposure': exposure, 'frames': frames,
                   'recognized': all(f['recognized'] for f in frames),
                   'prongs': int(np.median([f['prongs'] for f in frames])),
                   'contrast': min(f['contrast'] for f in frames),
                   'sharpness': float(np.median([f['sharpness'] for f in frames])),
                   'background': float(np.median([f['background'] for f in frames])),
                   'p99': float(np.median([f['p99'] for f in frames])),
                   'saturated_fraction': float(np.median([f['background_saturation'] for f in frames])),
                   'tool_whiteness': max(f['tool_whiteness'] for f in frames)}
            rows.append(row)
            if progress: progress(k, len(values), row)
        peak = max((r['sharpness'] for r in rows if r['recognized']), default=0.)
        reference = max((r['prongs'] for r in rows if r['recognized']), default=0)
        for row in rows:
            gates = absolute_criteria(row, expected_prongs, contrast_min, saturation_max)
            reasons = []
            if not gates['recognized']: reasons.append('未全部精定位或畸变')
            if reference < 2: reasons.append('未识别到足够的齿，检查工具或视角')
            elif row['prongs'] < reference: reasons.append(f'细齿被吃掉/未分辨（{row["prongs"]}/{reference}）')
            elif row['prongs'] > reference: reasons.append(f'齿数与参考值不一致（{row["prongs"]}/{reference}）')
            if not gates['contrast']: reasons.append('对比度不足')
            if not peak or row['sharpness'] < peak * sharpness_ratio: reasons.append('清晰度不足')
            if not gates['whiteness']: reasons.append('工具发白')
            row.update(prong_ref=reference, sharpness_percent=100 * row['sharpness'] / peak if peak else 0., reasons=reasons, qualified=not reasons, selected=False)
        eligible = [r for r in rows if r['qualified']]
        if not eligible:
            prefix = '未识别到足够的齿，检查工具或视角；' if reference < 2 else ''
            raise ExposureScanError(prefix + '没有合格曝光档：' + '；'.join(f'{r["exposure"]/10:g} ms：' + '、'.join(r['reasons']) for r in rows), rows)
        best = max(eligible, key=lambda r: (r['background'], -r['exposure']))
        camera.set_exposure(best['exposure'])
        if camera.get_exposure() != best['exposure']:
            raise ValueError('选中曝光回读不一致')
        best['selected'] = True
        return {'exposure': best['exposure'], 'rows': rows, 'prong_ref': reference, 'reason': f'参考齿数 {reference}；合格档中背景最亮，同亮度取较短曝光',
                'p99': best['p99'], 'saturated_fraction': best['saturated_fraction'], 'contrast': best['contrast']}
    except Exception:
        camera.set_exposure(initial)
        raise
