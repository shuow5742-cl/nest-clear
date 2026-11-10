"""Gen1 hair detection primitives, ported from pluck-hair detect.py."""
from __future__ import annotations
import heapq
from dataclasses import dataclass, replace
import cv2
import numpy as np
from skimage.morphology import skeletonize
HAIR_TYPE = "hair"
CRUMB_TYPE = "crumb"

def clump_paleness(det, gray_full, *, cfg) -> tuple:
    """回 ``(中位灰/局部背景, 外接框内偏暗占比)``; 算不出来回 ``(None, None)``。

    **逐字复刻** ``src/workers/vision.py`` 的 ``_target_darkness``: 局部背景取轮廓**外**、
    padded ROI **内**那一圈的中位灰; 偏暗占比量的是**外接框内**(不是轮廓内 —— 轮廓本来
    就是沿着暗区画的, 算在里面会饱和)。两处定义必须一致, 见上方 ⚠️。

    算不出来一律回 ``(None, None)`` 而不是抛 —— 判据那边把 ``None`` 当"这道门不表态",
    也就是**放行**。失效方向选放行: 一个算不出暗度的目标去戳一下, 代价是一针; 而拒绝
    启动或静默丢弃, 代价是整轮。
    """
    try:
        poly = getattr(det, 'polygon_xy', None) or []
        if len(poly) < 3:
            return (None, None, 'polygon has fewer than three points')
        arr = np.asarray(poly, np.int32)
        h, w = gray_full.shape[:2]
        x, y, bw, bh = cv2.boundingRect(arr)
        if bw <= 0 or bh <= 0:
            return (None, None, 'empty bounding box')
        pad = max(12, int(0.25 * max(bw, bh)))
        x1, y1 = (max(0, x - pad), max(0, y - pad))
        x2, y2 = (min(w, x + bw + pad), min(h, y + bh + pad))
        if x2 - x1 < 3 or y2 - y1 < 3:
            return (None, None, 'padded region too small')
        gray = gray_full[y1:y2, x1:x2]
        mask = np.zeros(gray.shape, np.uint8)
        cv2.fillPoly(mask, [arr - [x1, y1]], 255)
        inside = gray[mask > 0]
        outside = gray[mask == 0]
        if inside.size < 8 or outside.size < 8:
            return (None, None, 'insufficient inside or background pixels')
        if inside.size > cfg.pale_sample_max:
            inside = inside[::max(1, inside.size // cfg.pale_sample_max)]
        if outside.size > cfg.pale_sample_max:
            outside = outside[::max(1, outside.size // cfg.pale_sample_max)]
        bg = float(np.median(outside))
        if bg <= 1e-06:
            return (None, None, 'background is too dark')
        ratio = float(np.median(inside.astype(np.float32))) / bg
        bx1, by1 = (max(0, x - x1), max(0, y - y1))
        bx2, by2 = (min(gray.shape[1], bx1 + bw), min(gray.shape[0], by1 + bh))
        dfrac = None
        if bx2 > bx1 + 2 and by2 > by1 + 2:
            sub = gray[by1:by2, bx1:bx2].astype(np.float32)
            dfrac = float((sub < bg - cfg.pale_dark_margin).mean())
        return (ratio, dfrac, None if dfrac is not None else 'bounding box too small for dark fraction')
    except (ValueError, TypeError, OverflowError, FloatingPointError, ArithmeticError, cv2.error):
        return (None, None, 'numeric measurement error')

def _blur_background(gray: np.ndarray, *, cfg) -> np.ndarray:
    """The σ15 local-background estimate for the line / fiber-density passes.

    A σ15 Gaussian keeps only the low-frequency background, so we can estimate it at
    ``_LINE_BLUR_SCALE`` resolution (INTER_AREA downscale -> σ15*scale blur -> INTER_LINEAR
    upscale) instead of full res, cutting the ~12ms full-res Gaussian roughly in half with a
    near-identical result. ``_LINE_BLUR_SCALE >= 1.0`` runs the exact full-res blur (baseline)."""
    s = cfg.line_blur_scale
    if s >= 1.0:
        return cv2.GaussianBlur(gray, (0, 0), cfg.line_blur_sigma)
    small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    bg = cv2.GaussianBlur(small, (0, 0), cfg.line_blur_sigma * s)
    return cv2.resize(bg, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)

@dataclass(frozen=True)
class _BBox:
    x1: float
    y1: float
    x2: float
    y2: float

@dataclass(frozen=True)
class HairClump:
    """A detected hair blob. Duck-typed to the SegDetection surface so
    ``render_pipeline`` draws it (polygon fill + bbox + pick cross) unchanged.

    ``eligible_for_pick`` controls robot action only: all clumps are rendered, but only
    eligible clumps are returned as Hair targets to the pluck loop.
    """
    pick_point_xy: tuple[float, float]
    bbox: _BBox
    polygon_xy: list
    confidence: float
    mask_area: int
    object_type: str = 'hair'
    pick_angle_deg: float | None = None
    pick_method: str = 'cv-dark'
    eligible_for_pick: bool = True
    reject_reason: str | None = None
    thick_radius: float = 0.0
    length_px: float = 0.0

def _hair_pick_point(filled: np.ndarray, dist: np.ndarray | None=None, *, cfg) -> tuple[int, int, float, float]:
    """A grasp point ON the blob: the THICKEST pixel of the MID-LENGTH slice.

    The contour centroid lands in the concave bay of a C-curved hair (off the hair),
    which is the bug this fixes. Instead: PCA major axis -> take the middle slice along
    its length -> pick the distance-transform-thickest pixel there (mid-section, on the
    hair, with the most material). If that mid point is too thin, fall back to the
    thickest pixel overall (most material / usually the anchored root).
    Returns (px, py, thickness_px, angle_deg) in the ``filled`` image's local coords.

    ``dist`` is the L2 distance transform of ``filled``; pass the one the caller already
    computed for its median-thickness gate so it isn't recomputed here (it was, on every blob).
    """
    if dist is None:
        dist = cv2.distanceTransform(filled, cv2.DIST_L2, 5)
    ys, xs = np.where(filled > 0)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    mean = pts.mean(axis=0)
    cen = pts - mean
    svd_in = cen[::-(-len(cen) // cfg.geom_subsample_target)] if len(cen) > cfg.geom_subsample_over else cen
    _, _, vt = np.linalg.svd(svd_in, full_matrices=False)
    axis = vt[0]
    proj = cen @ axis
    span = float(proj.max() - proj.min()) or 1.0
    band = np.abs(proj - np.median(proj)) <= cfg.hair_mid_band * span
    if not band.any():
        band = np.ones(len(proj), dtype=bool)
    cand = pts[band].astype(int)
    d_cand = dist[cand[:, 1], cand[:, 0]]
    j = int(np.argmax(d_cand))
    px, py, thick = (int(cand[j, 0]), int(cand[j, 1]), float(d_cand[j]))
    if thick < cfg.hair_min_thick:
        py, px = (int(v) for v in np.unravel_index(int(np.argmax(dist)), dist.shape))
        thick = float(dist[py, px])
    return (px, py, thick, float(np.degrees(np.arctan2(axis[1], axis[0]))))

def _line_hairs(gray: np.ndarray, covered: np.ndarray, dark: np.ndarray | None=None, *, bg_ceiling: int | None=None, cfg) -> list[HairClump]:
    """Local-contrast LINE pathway -> a list of HairClump to OR-union with the dark-core
    clumps. Catches thin/faint hairs that are darker than their LOCAL neighbourhood but
    not below the absolute dark-core band (see the ``_LINE_*`` note). ``covered`` is a
    bool mask of the dark-core clumps (filled + dilated); a candidate whose body mostly
    (>=50%) falls inside it is a fragment of an already-found dark blob and is skipped —
    so this path only ADDS what dark-core missed (a true union, not a double count).
    Each candidate is emitted with default eligibility/object_type; the FINAL length/area
    relabel in ``detect_hairs_cv`` decides pluckability for every clump (a long thin wisp now
    qualifies by length rather than being pre-rejected as "thin").

    ``dark`` is the local-darkness map ``subtract(σ15 blur, gray)``; pass the one the caller
    already computed (the fiber-density pass shares the same σ15 blur) so it isn't recomputed.
    """
    if dark is None:
        dark = cv2.subtract(cv2.GaussianBlur(gray, (0, 0), cfg.line_blur_sigma), gray)
    bg_cap = bg_ceiling if bg_ceiling is not None else cfg.line_dark_max
    mask = ((dark >= cfg.line_contrast) & (gray < bg_cap)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, lab, stats, _c = cv2.connectedComponentsWithStatsWithAlgorithm(mask, 8, cv2.CV_32S, cv2.CCL_SPAGHETTI)
    out: list[HairClump] = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < cfg.line_min_area:
            continue
        x, y, w, h = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]), int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
        comp = lab[y:y + h, x:x + w] == i
        if float(covered[y:y + h, x:x + w][comp].mean()) >= 0.5:
            continue
        if float(np.percentile(dark[y:y + h, x:x + w][comp], 90)) < cfg.comp_p90_min:
            continue
        skel_len = int(skeletonize(comp).sum())
        if skel_len < cfg.line_skel_min:
            continue
        if area / skel_len > cfg.line_width_max:
            continue
        sub = comp.astype(np.uint8) * 255
        dist = cv2.distanceTransform(sub, cv2.DIST_L2, 5)
        thick_radius = float(np.median(dist[sub > 0]))
        lpx, lpy, _thick, angle = _hair_pick_point(sub, dist, cfg=cfg)
        px, py = (float(lpx + x), float(lpy + y))
        cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        poly = max(cnts, key=cv2.contourArea) if cnts else None
        polygon_xy = [(int(p[0][0]) + x, int(p[0][1]) + y) for p in poly] if poly is not None else []
        out.append(HairClump(pick_point_xy=(px, py), bbox=_BBox(float(x), float(y), float(x + w), float(y + h)), polygon_xy=polygon_xy, confidence=float(min(1.0, area / cfg.hair_area_norm)), mask_area=area, pick_angle_deg=angle, pick_method='cv-line', object_type=HAIR_TYPE, eligible_for_pick=True, reject_reason=None, thick_radius=thick_radius))
    return out

def _fiber_density_clumps(gray: np.ndarray, dark: np.ndarray, covered: np.ndarray, bg_ceiling: int | None=None, *, cfg) -> list[HairClump]:
    """Fiber-density pathway -> HairClump list for the radiating FEATHER SPRAY the dark-core and
    line/bridge passes miss (sparse, non-collinear barbs). Density = a 31px box-count of the
    fine-dark mask ``(dark>=10)&(gray<120)``; a region thresholded at ``dens>_FIBER_DENS_TH`` and
    morph-cleaned is accepted only if it clears the area floor, has a feather's DARK FRACTION (>=
    ``_FIBER_DARK_FRAC_MIN`` of it is gray<``_FIBER_DARK_GRAY_MAX`` — genuinely-dark barbs, NOT bright
    gel-fold ridges) AND its mean density clears the gel ceiling (``_FIBER_MEAN_DENS_MIN``). ``dark``
    is REUSED from the line path (same σ15 blur). ``covered`` is the filled+dilated dark-core/line
    coverage mask; a region mostly (>=50%) inside it is a fragment of an already-found clump and is
    skipped, so this only ADDS the feather (no double count). Accepted regions are pluckable by the
    same area gate the dark-core path uses (method ``cv-fiber``).

    COST: the density box-mean + OPEN/CLOSE morphology + connected-components scale with pixel count
    and dominate the pass (~36ms full-frame). A feather is a LARGE-SCALE texture, so all of that runs
    at HALF resolution (``_FIBER_DS``, 4x fewer pixels); ``INTER_AREA`` makes each half-pixel the
    fine-fraction of its ds×ds block, so a half-res box-mean is the SAME "fraction of fine pixels in
    the window" the full-res boxFilter computed (window/CLOSE kernels scaled by ds). The area + mean-
    density guards run on the half-res cues; each surviving component is then upscaled (NEAREST) ON
    ITS OWN BBOX for the BRIGHTNESS guards (covered overlap + dark-barb fraction) and the outline /
    pick — so a no-candidate frame never touches a full-res array and a rejected candidate costs only
    its own bbox, yet a real candidate is still judged on its REAL full-res pixels and its clump is
    extracted exactly as before — same detection, a few px of bbox jitter from the ds grid."""
    h, w = gray.shape
    ds = cfg.fiber_ds
    hs, ws = (h // ds, w // ds)
    fine_cap = bg_ceiling if bg_ceiling is not None else cfg.fiber_fine_gray_max
    fine = ((dark >= cfg.fiber_fine_dark) & (gray < fine_cap)).astype(np.float32)
    fine_h = cv2.resize(fine, (ws, hs), interpolation=cv2.INTER_AREA)
    dens = cv2.boxFilter(fine_h, -1, (cfg.fiber_dens_win_ds, cfg.fiber_dens_win_ds))
    mask = (dens > cfg.fiber_dens_th).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.fiber_open_k, cfg.fiber_open_k)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.fiber_close_k_ds, cfg.fiber_close_k_ds)))
    n, lab, stats, _c = cv2.connectedComponentsWithStatsWithAlgorithm(mask, 8, cv2.CV_32S, cv2.CCL_SPAGHETTI)
    out: list[HairClump] = []
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) * ds * ds < cfg.fiber_min_area:
            continue
        x, y, ww, hh = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]), int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
        comp = lab[y:y + hh, x:x + ww] == i
        if float(dens[y:y + hh, x:x + ww][comp].mean()) < cfg.fiber_mean_dens_min:
            continue
        fx, fy, fww, fhh = (x * ds, y * ds, ww * ds, hh * ds)
        sub = cv2.resize(comp.astype(np.uint8) * 255, (fww, fhh), interpolation=cv2.INTER_NEAREST)
        m = sub > 0
        if not m.any():
            continue
        if float(covered[fy:fy + fhh, fx:fx + fww][m].mean()) >= 0.5:
            continue
        fsub = (fine[fy:fy + fhh, fx:fx + fww] > 0) & m
        if not fsub.any():
            continue
        if float(np.percentile(dark[fy:fy + fhh, fx:fx + fww][fsub], 90)) < cfg.comp_p90_min:
            continue
        area = int(m.sum())
        dist = cv2.distanceTransform(sub, cv2.DIST_L2, 5)
        thick_radius = float(np.median(dist[sub > 0]))
        lpx, lpy, _thick, angle = _hair_pick_point(sub, dist, cfg=cfg)
        px, py = (float(lpx + fx), float(lpy + fy))
        cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        poly = max(cnts, key=cv2.contourArea) if cnts else None
        polygon_xy = [(int(p[0][0]) + fx, int(p[0][1]) + fy) for p in poly] if poly is not None else []
        out.append(HairClump(pick_point_xy=(px, py), bbox=_BBox(float(fx), float(fy), float(fx + fww), float(fy + fhh)), polygon_xy=polygon_xy, confidence=float(min(1.0, area / cfg.hair_area_norm)), mask_area=area, pick_angle_deg=angle, pick_method='cv-fiber', eligible_for_pick=True, reject_reason=None, thick_radius=thick_radius))
    return out

def _rebuild_merged_clumps(local: np.ndarray, members: list[HairClump], ox: int, oy: int, *, cfg) -> list[HairClump]:
    """Rebuild clumps from a fused LOCAL-crop mask of >=2 members (``ox,oy`` = crop origin in
    full-frame px). Emits ONE clump PER connected contour in the union — NOT just the largest:
    grouping (cheap, half-res) is deliberately a touch more eager than the local CLOSE bridge, so a
    bucket can contain members that don't actually fuse into one shape; keeping only the largest
    contour would silently DROP the others (a real bug — it lost a whole detected feather). Each
    contour recomputes its own outline/area; pick/angle/thickness are NOT recomputed here — the
    fused mask is filled-polygon ∪ CLOSE fill ∪ bridge corridors (GEOMETRY, not hair), so a
    thickest-pixel pick on it can land on morphological fill / mid-corridor. Picks are decided
    ONCE, at the end of the pipeline, by ``_refine_pick_points``; until then each contour just
    inherits its largest inside-member's seed pick/angle/thickness (a real-material point that
    keeps the downstream location probes and the low-evidence fallback sane). pick_method carries
    over from whether any member falling inside it was a dark-core (has_dark). Eligibility/
    object_type are left as defaults here — the FINAL length/area relabel sets them."""
    cnts, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[HairClump] = []
    for c in cnts:
        area = int(cv2.contourArea(c))
        if area < cfg.crumb_min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        mem = [m for m in members if cv2.pointPolygonTest(c, (float(m.pick_point_xy[0] - ox), float(m.pick_point_xy[1] - oy)), False) >= 0]
        if not mem:
            mem = members
        if mem:
            lead = max(mem, key=lambda m: m.mask_area)
            pick_xy, pick_angle, thick = (lead.pick_point_xy, lead.pick_angle_deg, lead.thick_radius)
            has_dark = any((m.pick_method == 'cv-dark' for m in mem))
        else:
            sub = np.zeros((h, w), np.uint8)
            cv2.drawContours(sub, [c - [x, y]], -1, 255, -1)
            dist = cv2.distanceTransform(sub, cv2.DIST_L2, 5)
            thick = float(np.median(dist[sub > 0])) if np.any(sub) else 0.0
            lpx, lpy, _t, pick_angle = _hair_pick_point(sub, dist, cfg=cfg)
            pick_xy = (float(lpx + x + ox), float(lpy + y + oy))
            has_dark = False
        out.append(HairClump(pick_point_xy=pick_xy, bbox=_BBox(float(x + ox), float(y + oy), float(x + w + ox), float(y + h + oy)), polygon_xy=[(int(p[0][0]) + ox, int(p[0][1]) + oy) for p in c], confidence=float(min(1.0, area / cfg.hair_area_norm)), mask_area=area, pick_angle_deg=pick_angle, pick_method='cv-dark' if has_dark else 'cv-line', object_type=HAIR_TYPE, eligible_for_pick=True, reject_reason=None, thick_radius=thick))
    return out

def _close_local_bounded(local: np.ndarray, gap: int, *, cfg) -> np.ndarray:
    """BIT-IDENTICAL replacement for ``cv2.morphologyEx(local, MORPH_CLOSE, ELLIPSE(gap|1))``.
    When a bucket's members are far apart the crop is huge but mostly EMPTY, so closing the whole
    sparse array wastes ~28ms. A closing = dilate(r) then erode(r) with r = (gap|1)//2, so its output
    is contained in the r-dilation of the foreground: any output pixel lies within ``r`` of some
    foreground pixel, and two foreground regions can only interact when they are within ~3r of each
    other. We therefore split the foreground into clusters (connected components grouped when their
    bboxes, padded by the SE size, overlap — a conservative >3r separation guarantee), run the SAME
    closing on each cluster's own tight crop (padded by 3r so morphology sees the identical
    neighbourhood, matching cv2's default border), and OR the per-cluster results back. Clusters that
    are actually within ``gap`` collapse into one group whose crop equals the old combined bbox, so
    the dense case is unchanged; only the sparse case is spared the empty area. Result is provably the
    same array as the full-crop closing."""
    se = gap | 1
    r = se // 2
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se, se))
    hc, wc = local.shape
    if local.size <= cfg.merge_close_bound_min_px:
        return cv2.morphologyEx(local, cv2.MORPH_CLOSE, ker)
    ncomp, _lbl, cstats, _cent = cv2.connectedComponentsWithStatsWithAlgorithm(local, 8, cv2.CV_32S, cv2.CCL_SPAGHETTI)
    if ncomp <= 1:
        return local
    tight = []
    pad = se
    padded = []
    for i in range(1, ncomp):
        x, y, bw, bh, _ar = cstats[i]
        tight.append((x, y, x + bw, y + bh))
        padded.append((x - pad, y - pad, x + bw + pad, y + bh + pad))
    parent = list(range(len(tight)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i in range(len(padded)):
        ax1, ay1, ax2, ay2 = padded[i]
        for j in range(i + 1, len(padded)):
            bx1, by1, bx2, by2 = padded[j]
            if ax1 < bx2 and bx1 < ax2 and (ay1 < by2) and (by1 < ay2):
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(tight)):
        groups.setdefault(find(i), []).append(i)
    out = np.zeros_like(local)
    for members in groups.values():
        gx1 = min((tight[i][0] for i in members))
        gy1 = min((tight[i][1] for i in members))
        gx2 = max((tight[i][2] for i in members))
        gy2 = max((tight[i][3] for i in members))
        ix1, iy1 = (max(0, gx1 - r), max(0, gy1 - r))
        ix2, iy2 = (min(wc, gx2 + r), min(hc, gy2 + r))
        cx1, cy1 = (max(0, ix1 - 2 * r), max(0, iy1 - 2 * r))
        cx2, cy2 = (min(wc, ix2 + 2 * r), min(hc, iy2 + 2 * r))
        sub_closed = cv2.morphologyEx(local[cy1:cy2, cx1:cx2], cv2.MORPH_CLOSE, ker)
        np.maximum(out[iy1:iy2, ix1:ix2], sub_closed[iy1 - cy1:iy2 - cy1, ix1 - cx1:ix2 - cx1], out=out[iy1:iy2, ix1:ix2])
    return out

def _merge_nearby_clumps(clumps: list[HairClump], shape: tuple[int, int], gap: int=None, *, cfg) -> list[HairClump]:
    if gap is None:
        gap = cfg.merge_gap_px
    'Fuse clumps whose filled masks lie within ``gap`` px into one. GROUPING is cheap: fill every\n    polygon full-res (so thin wisps survive), downscale (INTER_AREA keeps any nonzero), DILATE by\n    gap/2 so within-gap masks touch, connected-components, and bucket each clump by which component\n    its body falls in. Single-member buckets pass through UNCHANGED; each multi-member bucket is\n    rebuilt on a LOCAL bbox crop (not full-frame — that was the cost) by _rebuild_merged_clump.'
    if gap <= 0:
        return clumps
    polyd = [(c, np.asarray(c.polygon_xy, np.int32)) for c in clumps if c.polygon_xy]
    if len(polyd) < 2:
        return clumps
    h, w = shape
    s = cfg.merge_group_scale
    fill = np.zeros((h, w), np.uint8)
    for _c, poly in polyd:
        cv2.fillPoly(fill, [poly], 255)
    small = (cv2.resize(fill, (w // s, h // s), interpolation=cv2.INTER_AREA) > 0).astype(np.uint8)
    hs, ws = small.shape
    r = max(1, gap // (2 * s))
    grown = cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))
    _n, comp = cv2.connectedComponentsWithAlgorithm(grown, 8, cv2.CV_32S, cv2.CCL_SPAGHETTI)
    buckets: dict[int, list[HairClump]] = {}
    for i, (c, poly) in enumerate(polyd):
        cid = 0
        sx, sy = (int(c.pick_point_xy[0]) // s, int(c.pick_point_xy[1]) // s)
        for px, py in [(sx, sy), *((int(p[0]) // s, int(p[1]) // s) for p in poly)]:
            if 0 <= py < hs and 0 <= px < ws and comp[py, px]:
                cid = int(comp[py, px])
                break
        buckets.setdefault(cid if cid else -(i + 1), []).append(c)
    out: list[HairClump] = []
    for members in buckets.values():
        if len(members) == 1:
            out.append(members[0])
            continue
        pad = gap + 4
        x1 = max(0, min((int(m.bbox.x1) for m in members)) - pad)
        y1 = max(0, min((int(m.bbox.y1) for m in members)) - pad)
        x2 = min(w, max((int(m.bbox.x2) for m in members)) + pad)
        y2 = min(h, max((int(m.bbox.y2) for m in members)) + pad)
        local = np.zeros((y2 - y1, x2 - x1), np.uint8)
        for m in members:
            cv2.fillPoly(local, [np.asarray(m.polygon_xy, np.int32) - [x1, y1]], 255)
        local = _close_local_bounded(local, gap, cfg=cfg)
        rebuilt = _rebuild_merged_clumps(local, members, x1, y1, cfg=cfg)
        out.extend(rebuilt if rebuilt else members)
    return out

def _densify_poly(poly: np.ndarray, step: float=5.0, max_pts: int=None, *, cfg) -> np.ndarray:
    if max_pts is None:
        max_pts = cfg.bridge_peri_max_pts
    "Resample a polygon's PERIMETER to ~``step``-px spacing (float Nx2), capped at ``max_pts``\n    evenly-spaced points. The stored polygons are CHAIN_APPROX_SIMPLE (sparse vertices), so a raw\n    nearest-vertex search between two clumps can miss the true closest boundary points by tens of px;\n    densifying makes the nearest-point hop accurate, and the cap keeps the pairwise search cheap."
    pts = np.asarray(poly, np.float32)
    if len(pts) < 2:
        return pts.reshape(-1, 2)
    closed = np.vstack([pts, pts[:1]])
    seg = closed[1:] - closed[:-1]
    out = [pts]
    for i in range(len(seg)):
        n = int(np.hypot(seg[i, 0], seg[i, 1]) // step)
        if n > 1:
            ts = np.linspace(0.0, 1.0, n, endpoint=False)[1:, None]
            out.append(closed[i] + ts * seg[i])
    dense = np.vstack(out)
    if len(dense) > max_pts:
        dense = dense[np.linspace(0, len(dense) - 1, max_pts).astype(int)]
    return dense

def _bbox_gap(a: _BBox, b: _BBox, *, cfg) -> float:
    """L-inf-ish rectangle gap between two bboxes (0 if they overlap). Cheap pair prefilter."""
    dx = max(0.0, a.x1 - b.x2, b.x1 - a.x2)
    dy = max(0.0, a.y1 - b.y2, b.y1 - a.y2)
    return float(max(dx, dy))

def _clump_axis_deg(c: HairClump, *, cfg) -> float | None:
    """The clump's trusted long-axis angle (deg) for the collinearity test, or None if the blob is
    too round for its axis to be meaningful (a stepping-stone stone — exempt from the test)."""
    if c.pick_angle_deg is None:
        return None
    w = max(1.0, c.bbox.x2 - c.bbox.x1)
    h = max(1.0, c.bbox.y2 - c.bbox.y1)
    if max(w, h) / min(w, h) < cfg.bridge_elongation_min:
        return None
    return float(c.pick_angle_deg)

def _angle_gap_deg(a: float, b: float, *, cfg) -> float:
    """Smallest unsigned difference between two undirected line angles (deg), in [0, 90]."""
    d = abs(a - b) % 180.0
    return d if d <= 90.0 else 180.0 - d

def _bridge_eval(p: np.ndarray, q: np.ndarray, g: np.ndarray, bg_cap: float, *, cfg) -> tuple[float, bool]:
    """Measure the straight segment p->q as a candidate hair trail. Returns ``(margin, over_cap)``:
    ``margin`` = how much DARKER the centre line is than its two flank lines (flank median - line
    median, gray; larger = more hair-like, brightness-invariant — an embedded segment at gray~85 still
    scores high as long as its flanks are brighter); ``over_cap`` = the line's max climbed above
    ``bg_cap`` (it rode across a bright gel highlight — disqualifying regardless of margin).

    The caller decides: a normal hop needs ``margin >= _BRIDGE_MARGIN`` and not ``over_cap``; a
    strongly-dark hop (``margin >= _BRIDGE_STRONG_MARGIN``) waives the collinearity guard; the FIX-B
    core-chain accepts a weak margin on a short tightly-collinear hop between two genuine dark cores —
    but ALL paths still reject ``over_cap`` (never run a corridor across a bright highlight)."""
    h, w = g.shape
    L = int(max(abs(p[0] - q[0]), abs(p[1] - q[1])))
    if L < 2:
        return (float(cfg.bridge_strong_margin), False)
    t = np.linspace(0.0, 1.0, L)
    xs = p[0] + t * (q[0] - p[0])
    ys = p[1] + t * (q[1] - p[1])
    d = np.array([q[1] - p[1], -(q[0] - p[0])], float)
    d /= np.linalg.norm(d) + 1e-06

    def samp(off: float) -> np.ndarray:
        X = np.clip(xs + off * d[0], 0, w - 1).astype(int)
        Y = np.clip(ys + off * d[1], 0, h - 1).astype(int)
        return g[Y, X]
    line = samp(0.0)
    over_cap = float(line.max()) > bg_cap
    flank = np.concatenate([samp(o) for W in cfg.bridge_flank_w for o in (W, -W)])
    margin = float(np.median(flank)) - float(np.median(line))
    return (margin, over_cap)

class _UnionFind:

    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, i: int) -> int:
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, i: int, j: int) -> None:
        self.p[self.find(i)] = self.find(j)

def _bridge_link_clumps(clumps: list[HairClump], gray: np.ndarray, shape: tuple[int, int], *, cfg) -> list[HairClump]:
    """Link dark cores that lie along one darker-than-flanking hair trail into single clumps.

    Nodes = clumps with a polygon and area >= ``_BRIDGE_NODE_MIN_AREA`` (small crumb cores INCLUDED
    as stepping stones). For every near pair, find the nearest boundary points and measure the hop with
    ``_bridge_eval`` (its darker-than-flanks margin + whether it crosses a bright highlight). A hop is
    accepted when it never crosses a highlight AND either: (FIX A) the centre line is DARKER than its
    flanks by ``_BRIDGE_MARGIN`` and agrees with any elongated endpoint's long axis within
    ``_BRIDGE_COLLINEAR_DEG`` — the collinearity guard WAIVED when the margin is strong
    (``_BRIDGE_STRONG_MARGIN``), since real hairs bend; or (FIX B core-chain) the contrast cue is blind
    (weak margin in a deeply-embedded gap) but BOTH endpoints are genuine dark cores (``cv-dark``), the
    hop is short (``_BRIDGE_CHAIN_HOP``) and tightly collinear (``_BRIDGE_CHAIN_COLLINEAR_DEG``).
    Union-find the accepted hops; each multi-node group is fused (member masks + the accepted corridor
    segments) and rebuilt into one clump. Non-grouped clumps pass through UNCHANGED, so with no accepted
    hop the output equals the input."""
    h, w = shape
    nodes = [c for c in clumps if c.polygon_xy and c.mask_area >= cfg.bridge_node_min_area]
    passthrough = [c for c in clumps if not (c.polygon_xy and c.mask_area >= cfg.bridge_node_min_area)]
    if len(nodes) < 2:
        return clumps
    bg_cap = float(np.percentile(gray[::4, ::4], 60)) + cfg.bridge_line_max_over_bg
    peri = [_densify_poly(np.asarray(c.polygon_xy, np.float32), cfg=cfg) for c in nodes]
    axes = [_clump_axis_deg(c, cfg=cfg) for c in nodes]
    uf = _UnionFind(len(nodes))
    edges: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if _bbox_gap(nodes[i].bbox, nodes[j].bbox, cfg=cfg) > cfg.bridge_max_hop:
                continue
            ai, aj = (peri[i], peri[j])
            d2 = ((ai[:, None, :] - aj[None, :, :]) ** 2).sum(-1)
            fi, fj = np.unravel_index(int(np.argmin(d2)), d2.shape)
            p, q = (ai[fi], aj[fj])
            if float(np.hypot(*p - q)) > cfg.bridge_max_hop:
                continue
            hop_deg = float(np.degrees(np.arctan2(q[1] - p[1], q[0] - p[0])))
            ends = [a for a in (axes[i], axes[j]) if a is not None]
            collinear_gap = min((_angle_gap_deg(hop_deg, a, cfg=cfg) for a in ends)) if ends else None
            margin, over_cap = _bridge_eval(p, q, gray, bg_cap, cfg=cfg)
            if over_cap:
                continue
            strong = margin >= cfg.bridge_strong_margin
            if collinear_gap is not None and collinear_gap > cfg.bridge_collinear_deg and (not strong):
                continue
            link = margin >= cfg.bridge_margin
            if not link:
                both_dark = nodes[i].pick_method == 'cv-dark' and nodes[j].pick_method == 'cv-dark'
                tight = collinear_gap is not None and collinear_gap <= cfg.bridge_chain_collinear_deg
                short = float(np.hypot(*p - q)) <= cfg.bridge_chain_hop
                link = both_dark and tight and short
            if not link:
                continue
            uf.union(i, j)
            edges.append((i, j, p, q))
    if not edges:
        return clumps
    groups: dict[int, list[int]] = {}
    for idx in range(len(nodes)):
        groups.setdefault(uf.find(idx), []).append(idx)
    out: list[HairClump] = list(passthrough)
    for members_idx in groups.values():
        if len(members_idx) == 1:
            out.append(nodes[members_idx[0]])
            continue
        members = [nodes[k] for k in members_idx]
        pad = cfg.bridge_corridor_th + 4
        x1 = max(0, min((int(m.bbox.x1) for m in members)) - pad)
        y1 = max(0, min((int(m.bbox.y1) for m in members)) - pad)
        x2 = min(w, max((int(m.bbox.x2) for m in members)) + pad)
        y2 = min(h, max((int(m.bbox.y2) for m in members)) + pad)
        local = np.zeros((y2 - y1, x2 - x1), np.uint8)
        for m in members:
            cv2.fillPoly(local, [np.asarray(m.polygon_xy, np.int32) - [x1, y1]], 255)
        mset = set(members_idx)
        for i, j, p, q in edges:
            if i in mset and j in mset:
                cv2.line(local, (int(p[0] - x1), int(p[1] - y1)), (int(q[0] - x1), int(q[1] - y1)), 255, cfg.bridge_corridor_th)
        local = cv2.morphologyEx(local, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        rebuilt = _rebuild_merged_clumps(local, members, x1, y1, cfg=cfg)
        out.extend(rebuilt if rebuilt else members)
    return out

def _linefit_link_clumps(clumps: list[HairClump], gray: np.ndarray, shape: tuple[int, int], dark_max: int=None, *, cfg) -> list[HairClump]:
    if dark_max is None:
        dark_max = cfg.hair_dark_max
    "Fuse a dense collinear STRING of tiny dark cores (each below the bridge node floor) into one\n    hair clump. Candidate points = centroids of every ``gray < _HAIR_DARK_MAX`` connected component\n    with area >= ``_LINE_CC_MIN_AREA``. RANSAC-fit lines; accept one only with >= ``_LINE_N_MIN``\n    inliers that SPREAD over >= ``_LINE_MIN_SPAN`` with no gap > ``_LINE_MAX_GAP`` and no bright-\n    highlight crossing between consecutive cores. Each accepted line's core pixels + a thin corridor\n    (extended ``_LINE_EXT_PX`` past the ends, clipped at highlights) + any existing clump the corridor\n    overlaps are unioned and rebuilt into one clump. No accepted line -> input returned UNCHANGED."
    h, w = shape
    dark = (gray < dark_max).astype(np.uint8)
    ncc, labels, stats, cents = cv2.connectedComponentsWithStatsWithAlgorithm(dark, 8, cv2.CV_32S, cv2.CCL_SPAGHETTI)
    cand = [(float(cents[l, 0]), float(cents[l, 1]), l) for l in range(1, ncc) if int(stats[l, cv2.CC_STAT_AREA]) >= cfg.line_cc_min_area]
    if len(cand) < cfg.line_n_min:
        return clumps
    P = np.array([[c[0], c[1]] for c in cand], float)
    labs = np.array([c[2] for c in cand])
    bg_cap = float(np.percentile(gray[::4, ::4], 60)) + cfg.bridge_line_max_over_bg
    used = np.zeros(len(cand), bool)
    accepted: list[tuple[np.ndarray, np.ndarray, np.ndarray, float, float]] = []
    for _line in range(cfg.line_max_lines):
        avail = np.where(~used)[0]
        if len(avail) < cfg.line_n_min:
            break
        Pa = P[avail]
        best_cnt, best_mask = (0, None)
        for ii in range(len(avail)):
            for jj in range(ii + 1, len(avail)):
                seg = Pa[jj] - Pa[ii]
                L = float(np.hypot(seg[0], seg[1]))
                if L < 1.0 or L > cfg.line_seed_max:
                    continue
                d = seg / L
                rel = Pa - Pa[ii]
                proj = rel @ d
                perp = np.abs(rel[:, 0] * -d[1] + rel[:, 1] * d[0])
                m = (perp <= cfg.line_band) & (proj >= -1.0) & (proj <= L + 1.0)
                cnt = int(m.sum())
                if cnt > best_cnt:
                    best_cnt, best_mask = (cnt, m)
        if best_cnt < cfg.line_n_min:
            break
        inl = avail[best_mask]
        cen = P[inl].mean(0)
        _u, _s, vt = np.linalg.svd(P[inl] - cen, full_matrices=False)
        d = vt[0]
        rel = P[avail] - cen
        proj_in = (P[inl] - cen) @ d
        lo0, hi0 = (float(proj_in.min()), float(proj_in.max()))
        m2 = (np.abs(rel[:, 0] * -d[1] + rel[:, 1] * d[0]) <= cfg.line_band) & (rel @ d >= lo0 - 1.0) & (rel @ d <= hi0 + 1.0)
        inl = avail[m2]
        used[inl] = True
        if len(inl) < cfg.line_n_min:
            continue
        proj = (P[inl] - cen) @ d
        order = np.argsort(proj)
        inl, proj = (inl[order], proj[order])
        if float(proj[-1] - proj[0]) < cfg.line_min_span:
            continue
        if float(np.diff(proj).max()) > cfg.line_max_gap:
            continue
        if any((_bridge_eval(P[inl[k]], P[inl[k + 1]], gray, bg_cap, cfg=cfg)[1] for k in range(len(inl) - 1))):
            continue
        accepted.append((inl, cen, d, float(proj[0]), float(proj[-1])))
    if not accepted:
        return clumps
    out = list(clumps)
    for inl, cen, d, lo, hi in accepted:
        e_lo, e_hi = (cen + lo * d, cen + hi * d)
        x_lo = e_lo if _bridge_eval(e_lo, cen + (lo - cfg.line_ext_px) * d, gray, bg_cap, cfg=cfg)[1] else cen + (lo - cfg.line_ext_px) * d
        x_hi = e_hi if _bridge_eval(e_hi, cen + (hi + cfg.line_ext_px) * d, gray, bg_cap, cfg=cfg)[1] else cen + (hi + cfg.line_ext_px) * d
        path = np.vstack([x_lo, P[inl], x_hi])
        pad = cfg.bridge_corridor_th + 6
        lx1 = max(0, int(path[:, 0].min()) - pad)
        ly1 = max(0, int(path[:, 1].min()) - pad)
        lx2 = min(w, int(path[:, 0].max()) + pad)
        ly2 = min(h, int(path[:, 1].max()) + pad)
        if lx2 <= lx1 or ly2 <= ly1:
            continue
        linemask = np.isin(labels[ly1:ly2, lx1:lx2], labs[inl]).astype(np.uint8) * 255
        for k in range(len(path) - 1):
            cv2.line(linemask, (int(path[k, 0] - lx1), int(path[k, 1] - ly1)), (int(path[k + 1, 0] - lx1), int(path[k + 1, 1] - ly1)), 255, cfg.bridge_corridor_th)
        members, rest = ([], [])
        for c in out:
            b = c.bbox
            if not c.polygon_xy or b.x2 < lx1 or b.x1 > lx2 or (b.y2 < ly1) or (b.y1 > ly2):
                rest.append(c)
                continue
            cm = np.zeros((ly2 - ly1, lx2 - lx1), np.uint8)
            cv2.fillPoly(cm, [np.asarray(c.polygon_xy, np.int32) - [lx1, ly1]], 255)
            (members if np.any(cv2.bitwise_and(cm, linemask)) else rest).append(c)
        fx1 = max(0, min([lx1] + [int(m.bbox.x1) for m in members]) - pad)
        fy1 = max(0, min([ly1] + [int(m.bbox.y1) for m in members]) - pad)
        fx2 = min(w, max([lx2] + [int(m.bbox.x2) for m in members]) + pad)
        fy2 = min(h, max([ly2] + [int(m.bbox.y2) for m in members]) + pad)
        local = np.zeros((fy2 - fy1, fx2 - fx1), np.uint8)
        local[ly1 - fy1:ly2 - fy1, lx1 - fx1:lx2 - fx1] = linemask
        for m in members:
            cv2.fillPoly(local, [np.asarray(m.polygon_xy, np.int32) - [fx1, fy1]], 255)
        local = cv2.morphologyEx(local, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        rebuilt = _rebuild_merged_clumps(local, members, fx1, fy1, cfg=cfg)
        rebuilt = [replace(rc, pick_method='cv-dark') for rc in rebuilt]
        out = rest + (rebuilt if rebuilt else members)
    return out

def _suppress_contained(clumps: list[HairClump], overlap: float=None, *, cfg) -> list[HairClump]:
    if overlap is None:
        overlap = cfg.contain_overlap
    "Drop any clump with >= ``overlap`` of its bbox AREA inside a strictly-LARGER clump's bbox\n    (containment-NMS). Such a box is redundant: picking goes biggest-first, so a box (mostly) wrapped\n    inside a bigger one is never this round's target; if it's a genuinely separate object it reappears\n    once the big one is picked and the next recognize frame is taken — so this DEFERS, never\n    permanently loses. Strict area inequality means equal-area clumps don't mutually suppress."
    keep: list[HairClump] = []
    for b in clumps:
        barea = max(1.0, (b.bbox.x2 - b.bbox.x1) * (b.bbox.y2 - b.bbox.y1))
        swallowed = False
        for a in clumps:
            if a is b or a.mask_area <= b.mask_area:
                continue
            ix = max(0.0, min(b.bbox.x2, a.bbox.x2) - max(b.bbox.x1, a.bbox.x1))
            iy = max(0.0, min(b.bbox.y2, a.bbox.y2) - max(b.bbox.y1, a.bbox.y1))
            if ix * iy >= overlap * barea:
                swallowed = True
                break
        if not swallowed:
            keep.append(b)
    return keep

def _skeleton_geodesic_px(poly_xy: list, *, cfg) -> float:
    """CURVE (arc) length px of a clump: the longest end-to-end (geodesic) path along the
    skeleton of the clump's FILLED polygon, computed on its bbox-crop (like ``_line_hairs``,
    so cost scales with the clump, not the frame). Diagonal skeleton steps are weighted sqrt(2)
    so the path length matches true pixel distance.

    Why the LONGEST-GEODESIC and not the total-skeleton-pixel-SUM: a curved hair's skeleton is a
    single strand whose sum ≈ its arc, but a BRANCHY / bushy blob's skeleton sprouts side twigs, and
    summing every twig OVER-counts — a compact tuft could sum past 200px and false-pick. The
    endpoint-to-endpoint path ignores side branches (it only walks the two farthest tips), so a real
    curved hair reads its arc while a bushy blob reads only its longest span (short) -> no false pick.
    """
    arr = np.asarray(poly_xy, np.int32)
    x, y, w, h = cv2.boundingRect(arr)
    if w <= 0 or h <= 0:
        return 0.0
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [arr - [x, y]], 255)
    skel = skeletonize(m > 0)
    ys, xs = np.nonzero(skel)
    n = int(len(xs))
    if n <= 1:
        return float(n)
    idx = {(int(ys[k]), int(xs[k])): k for k in range(n)}
    r2 = 2.0 ** 0.5
    steps = ((-1, -1, r2), (-1, 0, 1.0), (-1, 1, r2), (0, -1, 1.0), (0, 1, 1.0), (1, -1, r2), (1, 0, 1.0), (1, 1, r2))
    adj: list[list] = [[] for _ in range(n)]
    for k in range(n):
        yy, xx = (int(ys[k]), int(xs[k]))
        for dy, dx, wt in steps:
            j = idx.get((yy + dy, xx + dx))
            if j is not None:
                adj[k].append((j, wt))

    def _farthest(src: int) -> tuple[int, float]:
        dist = [float('inf')] * n
        dist[src] = 0.0
        pq = [(0.0, src)]
        best, bestd = (src, 0.0)
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            if d > bestd:
                bestd, best = (d, u)
            for j, wt in adj[u]:
                nd = d + wt
                if nd < dist[j]:
                    dist[j] = nd
                    heapq.heappush(pq, (nd, j))
        return (best, bestd)
    a, _ = _farthest(0)
    _, span = _farthest(a)
    return float(span)

def _clump_length_px(clump: HairClump, *, cfg) -> float:
    """The clump's 'length' px for the >=2mm pick standard = its true CURVE (arc) length, taken as
    ``max(straight chord, skeleton geodesic)``. The minAreaRect chord (old metric) is the straight
    end-to-end EXTENT: for a CURVED hair it under-reads the real length, so a hair whose arc is >=2mm
    but bends gets a sub-_PICK_MIN_LEN_PX chord and is wrongly skipped. The skeleton geodesic recovers the arc; the
    ``max`` keeps a STRAIGHT hair at its chord unchanged (its medial axis retracts a bit at the ends,
    so arc<=chord there) and guarantees length only ever GROWS (arc>=chord) — so no hair that picked on
    chord can drop. Falls back to the bbox diagonal for a degenerate (<3 pt) polygon.

    COST GATE: final eligibility is ``length >= _PICK_MIN_LEN_PX AND mask_area >= _PICK_MIN_AREA_PX``
    with ``length = max(chord, geodesic)``. Since arc>=chord only ever GROWS length, the geodesic can
    flip a clump skip->pick ONLY when it ALREADY passes area (``mask_area >= _PICK_MIN_AREA_PX``) but
    is still short on chord (``chord < _PICK_MIN_LEN_PX``) — only then does recovering the arc matter.
    A clump that already meets the chord length, or that fails area (never pluckable however long), gets
    its chord returned directly (skipping skeletonize — the costly part on big/long/bulky clumps).
    Eligibility is bit-identical; only the L=..mm label on such a box may read chord."""
    poly = clump.polygon_xy
    if poly and len(poly) >= 3:
        _c, (w, h), _a = cv2.minAreaRect(np.asarray(poly, np.float32))
        chord = float(max(w, h))
        if chord >= round(cfg.min_length_mm * 1000.0 / cfg.um_per_px) or clump.mask_area < round(cfg.min_area_mm2 * (1000.0 / cfg.um_per_px) ** 2):
            return chord
        return max(chord, _skeleton_geodesic_px(poly, cfg=cfg))
    b = clump.bbox
    return float(((b.x2 - b.x1) ** 2 + (b.y2 - b.y1) ** 2) ** 0.5)

def _trace_ridge(gray: np.ndarray, start: np.ndarray, direction: np.ndarray, bg_cap: float, *, cfg) -> list[tuple[float, float]]:
    """Walk from ``start`` along ``direction``, following a darker-than-both-flanks ridge that sits
    between two bright gel flanks. Returns the accepted ridge points (full-frame float x,y). Stops when
    the ridge fades (``_TRACE_MAX_MISS`` weak steps), bends past ``_TRACE_BEND_MAX``, or hits the step
    cap. Selective by construction: gel texture with no continuing bright-flanked ridge is not followed."""
    h, w = gray.shape
    pos = np.asarray(start, float)
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d) + 1e-09
    pts: list[tuple[float, float]] = []
    misses = 0
    ts = np.arange(-cfg.trace_half_win, cfg.trace_half_win + 1, dtype=float)
    fs = np.array([s for s in cfg.trace_flank] + [-s for s in cfg.trace_flank], dtype=float)

    def sample(cx: np.ndarray, cy: np.ndarray) -> np.ndarray:
        xs = np.clip(np.rint(cx), 0, w - 1).astype(np.intp)
        ys = np.clip(np.rint(cy), 0, h - 1).astype(np.intp)
        return gray[ys, xs]
    for _ in range(cfg.trace_max_steps):
        center = pos + cfg.trace_step * d
        perp = np.array([-d[1], d[0]])
        vals = sample(center[0] + ts * perp[0], center[1] + ts * perp[1])
        bi = int(np.argmin(vals))
        best_v = int(vals[bi])
        ridge = center + ts[bi] * perp
        fv = sample(ridge[0] + fs * perp[0], ridge[1] + fs * perp[1])
        fl = float(np.median(fv[:len(cfg.trace_flank)]))
        fr = float(np.median(fv[len(cfg.trace_flank):]))
        margin = float(min(fl, fr)) - best_v
        step_vec = ridge - pos
        nn = float(np.linalg.norm(step_vec))
        bend = 999.0 if nn <= 1e-06 else float(np.degrees(np.arccos(np.clip(np.dot(step_vec / nn, d), -1.0, 1.0))))
        ok = margin >= cfg.trace_margin and min(fl, fr) >= cfg.trace_flank_min and (best_v <= cfg.trace_ridge_max) and (best_v < bg_cap) and (bend <= cfg.trace_bend_max)
        if ok:
            pts.append((float(ridge[0]), float(ridge[1])))
            d = (1.0 - cfg.trace_adapt) * d + cfg.trace_adapt * (step_vec / nn)
            d /= np.linalg.norm(d) + 1e-09
            pos = ridge
            misses = 0
        else:
            misses += 1
            pos = center
            if misses > cfg.trace_max_miss:
                break
    return pts

def _seed_grow_clumps(clumps: list[HairClump], gray_full: np.ndarray, shape: tuple[int, int], dark_max: int=None, *, cfg) -> list[HairClump]:
    if dark_max is None:
        dark_max = cfg.hair_dark_max
    'Extend confident dark-core clumps to their seed-connected loose-dark reconstruction (feather\n    barbs attached to the core), without flooding gel. See the block comment above for the method.\n\n    marker = gray_full < _HAIR_DARK_MAX ; mask = (gray_full < _SEED_GROW_LOOSE) | (local-dark >= _SEED_\n    GROW_LOCAL). reconstructed = the loose-mask components that contain >=1 marker (seed) pixel. Each\n    cv-dark/cv-line clump is mapped to the reconstructed component covering it; same-component seeds fuse\n    into ONE rebuilt clump (grown polygon/bbox/area/length; pick kept from the largest seed member). Other\n    clumps and any component that fails the growth floor pass through unchanged. Nothing is dropped.'
    if not cfg.seed_grow_enable:
        return clumps
    h, w = shape

    def _seedable(c: HairClump) -> bool:
        return c.pick_method in ('cv-dark', 'cv-line') and bool(c.polygon_xy)
    seeds = [c for c in clumps if _seedable(c)]
    if not seeds:
        return clumps
    s = cfg.seed_grow_scale
    hs, ws = (max(1, int(round(h * s))), max(1, int(round(w * s))))
    gray_s = cv2.resize(gray_full, (ws, hs), interpolation=cv2.INTER_AREA)
    bg60 = float(np.percentile(gray_s[::2, ::2], 60))
    clipped_s = np.minimum(gray_s, np.uint8(min(255, round(bg60))))
    local_dark_s = cv2.subtract(cv2.GaussianBlur(clipped_s, (0, 0), cfg.line_blur_sigma * s), gray_s)
    loose_max = bg60 - cfg.seed_grow_bg_margin
    loose = ((gray_s < loose_max) | (local_dark_s >= cfg.seed_grow_local)).astype(np.uint8)
    if cfg.seed_grow_close > 0:
        k = max(1, int(round(cfg.seed_grow_close * s)))
        loose = cv2.morphologyEx(loose, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1,) * 2))
    marker = gray_s < dark_max
    n, lab = cv2.connectedComponentsWithAlgorithm(loose, 8, cv2.CV_32S, cv2.CCL_SPAGHETTI)
    if n <= 1:
        return clumps
    seed_labels = np.unique(lab[marker & (lab > 0)])
    keep = np.zeros(n, np.uint8)
    keep[seed_labels] = 1
    sx, sy = (ws / w, hs / h)
    buckets: dict[int, list[HairClump]] = {}
    passthrough: list[HairClump] = [c for c in clumps if not _seedable(c)]
    for c in seeds:
        lbl = 0
        px, py = (int(round(c.pick_point_xy[0])), int(round(c.pick_point_xy[1])))
        probes = [(px, py), *((int(vx), int(vy)) for vx, vy in c.polygon_xy)]
        for qx, qy in probes:
            if 0 <= qy < h and 0 <= qx < w:
                dqx = min(ws - 1, int(qx * sx))
                dqy = min(hs - 1, int(qy * sy))
                L = int(lab[dqy, dqx])
                if L > 0 and keep[L]:
                    lbl = L
                    break
        if lbl == 0:
            passthrough.append(c)
        else:
            buckets.setdefault(lbl, []).append(c)
    out: list[HairClump] = list(passthrough)
    for lbl, members in buckets.items():
        comp_s = (lab == lbl).astype(np.uint8)
        dx, dy, dcw, dch = cv2.boundingRect(comp_s)
        pad = 4
        fx0 = max(0, int(dx / sx) - pad)
        fy0 = max(0, int(dy / sy) - pad)
        fx1 = min(w, int(np.ceil((dx + dcw) / sx)) + pad)
        fy1 = min(h, int(np.ceil((dy + dch) / sy)) + pad)
        for m in members:
            mp = np.asarray(m.polygon_xy, np.int32)
            fx0 = min(fx0, max(0, int(mp[:, 0].min()) - pad))
            fy0 = min(fy0, max(0, int(mp[:, 1].min()) - pad))
            fx1 = max(fx1, min(w, int(mp[:, 0].max()) + pad))
            fy1 = max(fy1, min(h, int(mp[:, 1].max()) + pad))
        if fx1 <= fx0 or fy1 <= fy0:
            out.extend(members)
            continue
        src_xs = np.minimum((np.arange(fx0, fx1) * sx).astype(np.intp), ws - 1)
        src_ys = np.minimum((np.arange(fy0, fy1) * sy).astype(np.intp), hs - 1)
        comp = np.ascontiguousarray(comp_s[np.ix_(src_ys, src_xs)] * 255)
        for m in members:
            cv2.fillPoly(comp, [np.asarray(m.polygon_xy, np.int32) - [fx0, fy0]], 255)
        if cfg.seed_grow_close > 0:
            k = max(1, cfg.seed_grow_close)
            comp = cv2.morphologyEx(comp, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1,) * 2))
        own_area = max((int(m.mask_area) for m in members))
        x, y, cw, ch = cv2.boundingRect(comp)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        big = max(cnts, key=cv2.contourArea) if cnts else None
        new_area = int(cv2.contourArea(big)) if big is not None else 0
        if big is None or new_area < own_area * cfg.seed_grow_min_growth:
            out.extend(members)
            continue
        if new_area > cfg.seed_grow_max_area:
            out.extend(members)
            continue
        lead = max(members, key=lambda m: m.mask_area)
        sub = np.zeros((ch, cw), np.uint8)
        cv2.drawContours(sub, [big - [x, y]], -1, 255, -1)
        dist = cv2.distanceTransform(sub, cv2.DIST_L2, 5)
        thick = float(np.median(dist[sub > 0])) if np.any(sub) else lead.thick_radius
        new_poly = [(int(p[0][0]) + fx0, int(p[0][1]) + fy0) for p in big]
        out.append(replace(lead, polygon_xy=new_poly, bbox=_BBox(float(x + fx0), float(y + fy0), float(x + fx0 + cw), float(y + fy0 + ch)), confidence=float(min(1.0, new_area / cfg.hair_area_norm)), mask_area=new_area, thick_radius=thick, pick_method='cv-dark'))
    return out

def _axis_trace_extend(clumps: list[HairClump], gray: np.ndarray, shape: tuple[int, int], *, cfg) -> list[HairClump]:
    """Grow each would-be-SKIPPED fragment along its own axis by tracing the faint continuation
    (see the block comment). Already-pluckable clumps (>=2mm AND bulky) and non-CV / tiny clumps are
    returned UNCHANGED, so no existing target is touched. A traced fragment's polygon/bbox/area/length
    are rebuilt from its mask + the traced corridor; its pick point/angle are kept (still on the real
    core). Nothing is dropped."""
    h, w = shape
    bg_cap = float(np.percentile(gray[::4, ::4], 60)) + cfg.bridge_line_max_over_bg
    out: list[HairClump] = []
    for c in clumps:
        would_pick = _clump_length_px(c, cfg=cfg) >= round(cfg.min_length_mm * 1000.0 / cfg.um_per_px) and c.mask_area >= round(cfg.min_area_mm2 * (1000.0 / cfg.um_per_px) ** 2)
        seedable = c.pick_method in ('cv-dark', 'cv-line') and c.pick_angle_deg is not None and (c.mask_area >= cfg.trace_seed_min_area) and (len(c.polygon_xy) >= 3)
        if would_pick or not seedable:
            out.append(c)
            continue
        poly = np.asarray(c.polygon_xy, float)
        cen = poly - poly.mean(axis=0)
        _u, _s, vt = np.linalg.svd(cen, full_matrices=False)
        axis = vt[0]
        proj = cen @ axis
        lo = poly[int(proj.argmin())]
        hi = poly[int(proj.argmax())]
        pts_hi = _trace_ridge(gray, hi, axis, bg_cap, cfg=cfg)
        pts_lo = _trace_ridge(gray, lo, -axis, bg_cap, cfg=cfg)
        if not pts_hi and (not pts_lo):
            out.append(c)
            continue
        allpts = np.vstack([poly, np.asarray(pts_hi + pts_lo, float)])
        x1 = max(0, int(allpts[:, 0].min()) - 3)
        y1 = max(0, int(allpts[:, 1].min()) - 3)
        x2 = min(w, int(allpts[:, 0].max()) + 3)
        y2 = min(h, int(allpts[:, 1].max()) + 3)
        local = np.zeros((y2 - y1, x2 - x1), np.uint8)
        cv2.fillPoly(local, [np.asarray(c.polygon_xy, np.int32) - [x1, y1]], 255)
        for seed, chain in ((hi, pts_hi), (lo, pts_lo)):
            path = [seed] + chain
            for a, b in zip(path[:-1], path[1:]):
                cv2.line(local, (int(round(a[0])) - x1, int(round(a[1])) - y1), (int(round(b[0])) - x1, int(round(b[1])) - y1), 255, cfg.trace_corridor_th)
        cnts, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            out.append(c)
            continue
        big = max(cnts, key=cv2.contourArea)
        new_poly = [(int(p[0][0]) + x1, int(p[0][1]) + y1) for p in big]
        _c, (rw, rh), _a = cv2.minAreaRect(np.asarray(new_poly, np.float32))
        new_len = float(max(rw, rh))
        if new_len <= _clump_length_px(c, cfg=cfg):
            out.append(c)
            continue
        out.append(replace(c, polygon_xy=new_poly, bbox=_BBox(float(x1), float(y1), float(x2), float(y2)), mask_area=int(cv2.contourArea(big)), length_px=new_len))
    return out

def _refine_pick_points(clumps: list[HairClump], gray_full: np.ndarray, line_dark_full: np.ndarray, *, cfg) -> list[HairClump]:
    """Move each final clump's pick point to the DENSEST real-hair-evidence pixel inside its
    final polygon (see the block comment above). Evidence = detection's own cues: absolute dark
    (``gray < _HAIR_DARK_MAX``) OR locally-dark fine fiber (``line_dark >= _FIBER_FINE_DARK`` and
    ``gray < _FIBER_FINE_GRAY_MAX``). Score = σ10 Gaussian density of the DARKNESS-GRADED
    evidence (each pixel weighted by how dark it is, 1.0 at gray<=55 → 0 at gray=120) × √(depth
    inside the clump polygon) — densest darkest material wins, but dense-hugging-the-outline
    loses to nearly-as-dense-in-the-body (a boundary pick is fragile: half the tool lands on
    gel). The
    argmax runs over evidence pixels sitting an ADAPTIVE MARGIN inside the evidence boundary;
    thin hairs shrink the margin (~1.5px = centerline) so they never empty out, and if the inner
    band still starves, the full evidence set is used. The pick angle is refit by PCA over the
    evidence pixels (same subsample rule as
    ``_hair_pick_point``). A clump with too little evidence keeps its existing pick unchanged;
    geometry/eligibility are untouched (the final length/area relabel keys on those alone).

    ⚠️ 上面那段描述的是 ``_PICK_SCORE_MODE == "legacy"`` 那条路。2026-08-19 起默认走
    ``"thick_central"``: **粗^1 x 中心^2**, 完整依据(49 个人工取点真值)见 该常量上方的 ★★。
    两条路都只动 pick_point_xy / pick_angle_deg, 不碰几何量与可挑性。"""
    if not cfg.pick_refine_enable:
        return clumps
    h, w = gray_full.shape[:2]
    pad = int(3 * cfg.pick_dens_sigma)
    out: list[HairClump] = []
    for c in clumps:
        if not c.polygon_xy or len(c.polygon_xy) < 3:
            out.append(c)
            continue
        x1 = max(0, int(c.bbox.x1) - pad)
        y1 = max(0, int(c.bbox.y1) - pad)
        x2 = min(w, int(c.bbox.x2) + pad)
        y2 = min(h, int(c.bbox.y2) + pad)
        if x2 <= x1 or y2 <= y1:
            out.append(c)
            continue
        poly = np.zeros((y2 - y1, x2 - x1), np.uint8)
        cv2.fillPoly(poly, [np.asarray(c.polygon_xy, np.int32) - [x1, y1]], 255)
        g = gray_full[y1:y2, x1:x2]
        ld = line_dark_full[y1:y2, x1:x2]
        ev = ((g < cfg.hair_dark_max) | (ld >= cfg.fiber_fine_dark) & (g < cfg.fiber_fine_gray_max)) & (poly > 0)
        if int(np.count_nonzero(ev)) < cfg.pick_evidence_min:
            out.append(c)
            continue
        wdark = np.where(ev, np.clip((cfg.fiber_fine_gray_max - g.astype(np.float32)) / float(cfg.fiber_fine_gray_max - cfg.hair_dark_max), 0.0, 1.0), 0.0).astype(np.float32)
        dens = (cv2.GaussianBlur(wdark, (0, 0), cfg.pick_dens_sigma)
                if cfg.dens_pow > 0 else None)
        thick = cv2.distanceTransform(ev.astype(np.uint8), cv2.DIST_L2, 5)
        dmax = float(thick.max())
        ys_e, xs_e = np.nonzero(ev)
        cxe, cye = (float(xs_e.mean()), float(ys_e.mean()))
        yy, xx = np.mgrid[0:ev.shape[0], 0:ev.shape[1]]
        rad = np.sqrt((xx - cxe) ** 2 + (yy - cye) ** 2).astype(np.float32)
        rmax = max(1.0, float(rad[ev].max()))
        central = np.clip(1.0 - rad / rmax, 0.0, 1.0)
        tn = thick / max(1e-06, dmax)
        score = (np.power(tn, cfg.thick_pow) * np.power(central, cfg.central_pow)).astype(np.float32)
        if cfg.dens_pow > 0:
            dn_ = dens / max(1e-06, float(dens.max()))
            score = (score * np.power(dn_, cfg.dens_pow)).astype(np.float32)
        cand = ev
        score = np.asarray(score, np.float32)
        score[~cand] = -1.0
        py, px = np.unravel_index(int(np.argmax(score)), score.shape)
        ys, xs = np.nonzero(ev)
        pts = np.column_stack([xs, ys]).astype(np.float32)
        angle = c.pick_angle_deg
        if len(pts) >= 3:
            cen = pts - pts.mean(axis=0)
            svd_in = cen[::-(-len(cen) // cfg.geom_subsample_target)] if len(cen) > cfg.geom_subsample_over else cen
            _u, _s, vt = np.linalg.svd(svd_in, full_matrices=False)
            angle = float(np.degrees(np.arctan2(vt[0][1], vt[0][0])))
        out.append(replace(c, pick_point_xy=(float(px + x1), float(py + y1)), pick_angle_deg=angle))
    return out
