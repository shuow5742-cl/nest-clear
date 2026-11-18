"""The twelve gen1 detection stages and final pick gates."""
from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np

from . import core
from .config import HairDetectConfig


@dataclass(frozen=True)
class HairCandidate:
    pick_uv: tuple[float, float]
    angle_deg: float | None
    polygon: tuple[tuple[int, int], ...]
    bbox: tuple[float, float, float, float]
    identity_uv: tuple[float, float]
    length_px: float
    area_px: int
    gray_ratio: float | None
    dark_frac: float | None
    paleness_unmeasurable_reason: str | None
    source: str


def gen1_pixel_gates(c: HairCandidate, cfg: HairDetectConfig) -> tuple[bool, str | None]:
    ppm = 1000.0 / cfg.um_per_px
    length = round(cfg.min_length_mm * ppm)
    area = round(cfg.min_area_mm2 * ppm * ppm)
    width = round(cfg.min_width_mm * ppm)
    by_len = c.length_px >= length
    by_area = c.area_px >= area
    by_width = width <= 0 or c.length_px <= 0 or c.area_px / c.length_px >= width
    if not by_len or not by_area:
        return False, 'short'
    if not by_width:
        return False, 'thin'
    if ((cfg.max_gray_ratio > 0 and c.gray_ratio is not None and c.gray_ratio > cfg.max_gray_ratio)
            or (cfg.min_dark_frac > 0 and c.dark_frac is not None and c.dark_frac < cfg.min_dark_frac)):
        return False, 'pale'
    return True, None


def detect_candidates(image_bgr: np.ndarray, cfg: HairDetectConfig, *,
                      timings_ms: dict[str, float] | None = None,
                      stage_callback=None) -> list[HairCandidate]:
    """Detect with an already scaled config; optionally collect stage timings."""
    started = perf_counter() if timings_ms is not None else 0.0
    def mark(name):
        nonlocal started
        if stage_callback is not None:
            stage_callback(name, tuple(tuple(map(tuple, c.polygon_xy)) for c in clumps if c.polygon_xy))
        if timings_ms is not None:
            now = perf_counter()
            timings_ms[name] = (now - started) * 1000.0
            started = now
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    bg60 = float(np.percentile(gray[::4, ::4], 60))
    dark_max = int(max(35, min(cfg.hair_dark_max, bg60 - cfg.core_bg_margin)))
    mask = (gray < dark_max).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((cfg.dark_open_k, cfg.dark_open_k), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((cfg.dark_close_k, cfg.dark_close_k), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clumps = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue
        sub = np.zeros((h, w), np.uint8)
        cv2.drawContours(sub, [contour - [x, y]], -1, 255, -1)
        if not np.any(sub):
            continue
        dist = cv2.distanceTransform(sub, cv2.DIST_L2, 5)
        # Gen1 also overflows the float32 median on solid contours that fill
        # their bounding box: OpenCV returns float32 max for every distance.
        radius = float(np.median(dist[sub > 0]))
        px, py, _, angle = core._hair_pick_point(sub, dist, cfg=cfg)
        clumps.append(core.HairClump(pick_point_xy=(float(px + x), float(py + y)),
            bbox=core._BBox(float(x), float(y), float(x + w), float(y + h)),
            polygon_xy=[(int(pt[0][0]), int(pt[0][1])) for pt in contour],
            confidence=float(min(1.0, area / cfg.hair_area_norm)), mask_area=int(area),
            pick_angle_deg=angle, object_type=core.HAIR_TYPE, thick_radius=radius))
    clipped = np.minimum(gray, np.uint8(min(255, round(bg60))))
    line_dark = cv2.subtract(core._blur_background(clipped, cfg=cfg), gray)
    mark("dark_core")
    covered = np.zeros(gray.shape, np.uint8)
    for det in clumps:
        if det.polygon_xy:
            cv2.fillPoly(covered, [np.asarray(det.polygon_xy, np.int32)], 255)
    covered_bool = cv2.dilate(covered, np.ones((cfg.covered_dilate_k, cfg.covered_dilate_k), np.uint8)) > 0
    clumps.extend(core._line_hairs(gray, covered_bool, line_dark, bg_ceiling=int(bg60), cfg=cfg))
    mark("line")
    for det in clumps:
        if det.pick_method == 'cv-line' and det.polygon_xy:
            cv2.fillPoly(covered, [np.asarray(det.polygon_xy, np.int32)], 255)
    covered_bool = cv2.dilate(covered, np.ones((cfg.covered_dilate_k, cfg.covered_dilate_k), np.uint8)) > 0
    clumps.extend(core._fiber_density_clumps(gray, line_dark, covered_bool, bg_ceiling=int(bg60), cfg=cfg))
    mark("fiber")
    shape = image_bgr.shape[:2]
    clumps = core._bridge_link_clumps(clumps, gray, shape, cfg=cfg)
    mark("bridge")
    clumps = core._linefit_link_clumps(clumps, gray, shape, dark_max=dark_max, cfg=cfg)
    mark("linefit")
    clumps = core._merge_nearby_clumps(clumps, shape, cfg=cfg)
    mark("merge")
    clumps = core._suppress_contained(clumps, cfg=cfg)
    mark("suppress_before")
    clumps = core._seed_grow_clumps(clumps, gray, shape, dark_max=dark_max, cfg=cfg)
    mark("seed_grow")
    clumps = core._axis_trace_extend(clumps, gray, shape, cfg=cfg)
    mark("axis_trace")
    clumps = core._suppress_contained(clumps, cfg=cfg)
    mark("suppress_after")
    clumps = core._refine_pick_points(clumps, gray, line_dark, cfg=cfg)
    mark("refine")
    clumps.sort(key=lambda c: -c.mask_area)
    result = []
    for c in clumps:
        ratio, fraction, reason = core.clump_paleness(c, gray, cfg=cfg)
        bbox = (c.bbox.x1, c.bbox.y1, c.bbox.x2, c.bbox.y2)
        result.append(HairCandidate(c.pick_point_xy, c.pick_angle_deg, tuple(map(tuple, c.polygon_xy)),
            bbox, ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
            core._clump_length_px(c, cfg=cfg), c.mask_area, ratio, fraction,
            reason,
            c.pick_method))
    mark("relabel_output")
    return result
