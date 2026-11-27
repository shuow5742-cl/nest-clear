"""菲林与 ChArUco 标定板测点布局选择及模组坐标中的平面拟合。"""

import math
import statistics

import numpy as np

from nest_clear.guard import GuardError
from nest_clear.tooloffset.session import nominal_targets


def charuco_label(corner_id):
    """将 ChArUco 内角点编号格式化为测点标签（如 C045）。"""
    return f'C{int(corner_id):03d}'


def charuco_corner_id(label):
    """从测点标签（如 C045）解析 ChArUco 内角点编号；不匹配时返回 None。"""
    if isinstance(label, str) and len(label) >= 2 and label[0] == 'C' and label[1:].isdigit():
        return int(label[1:])
    return None


def fit_board_to_module(board_xy, module_xy, *, allow_scale=True):
    """由检测到的 ChArUco 板坐标与模组 XY 拟合二维刚体/相似变换（允许正反面投影翻转）。"""
    src_all = np.asarray(board_xy, dtype=float)
    dst_all = np.asarray(module_xy, dtype=float)
    if src_all.ndim != 2 or dst_all.ndim != 2 or src_all.shape != dst_all.shape or src_all.shape[1] != 2:
        raise ValueError('ChArUco 板与模组坐标形状无效')
    if len(src_all) < 6 or not (np.isfinite(src_all).all() and np.isfinite(dst_all).all()):
        raise ValueError('至少需要 6 个有效 ChArUco 角点才能拟合标定板位姿')
    src_mean = src_all.mean(axis=0)
    dst_mean = dst_all.mean(axis=0)
    src = src_all - src_mean
    dst = dst_all - dst_mean
    _, s_src, _ = np.linalg.svd(src, full_matrices=False)
    if s_src[1] < 1.0:
        raise ValueError('检出 ChArUco 角点近似共线，无法稳定确定标定板二维位姿')
    den = float(np.sum(src * src))
    if den < 1e-12:
        raise ValueError('检出 ChArUco 角点重合')
    u, singular_vals, vt = np.linalg.svd(src.T @ dst)
    rot = vt.T @ u.T
    scale = float(np.sum(singular_vals) / den) if allow_scale else 1.0
    if not (0.8 <= scale <= 1.25):
        raise ValueError(f'ChArUco 二维拟合尺度 {scale:.4f} 超出合理范围 [0.8, 1.25]')
    pred = scale * (src @ rot.T) + dst_mean
    errs = np.linalg.norm(dst_all - pred, axis=1)
    ex = rot @ np.array([1.0, 0.0])
    angle_deg = float(math.degrees(math.atan2(ex[1], ex[0])))

    def transform(points):
        arr = np.asarray(points, dtype=float)
        single = arr.ndim == 1
        pts = arr.reshape(1, 2) if single else arr
        out = scale * ((pts - src_mean) @ rot.T) + dst_mean
        return out[0] if single else out

    fit_info = {
        'corner_count': int(len(src_all)),
        'scale': float(scale),
        'angle_deg': angle_deg,
        'rms_residual_mm': float(np.sqrt(np.mean(errs * errs))),
        'max_residual_mm': float(np.max(errs)),
        'det': float(np.linalg.det(rot)),
    }
    return transform, fit_info


def charuco_board_outer_quad(camcal_cfg, transform):
    """用拟合位姿变换求 ChArUco 板四个外角在模组坐标中的四边形顶点。"""
    width = float(camcal_cfg['cols']) * float(camcal_cfg['square_mm'])
    height = float(camcal_cfg['rows']) * float(camcal_cfg['square_mm'])
    outer_board = np.array([
        [0.0, 0.0],
        [width, 0.0],
        [width, height],
        [0.0, height],
    ], dtype=float)
    return np.asarray(transform(outer_board), dtype=float)


def _inward_edge_normals(quad):
    """返回凸四边形各条边的单位内法向量（与顶点顺序无关）。"""
    q = np.asarray(quad, dtype=float)
    if q.shape != (4, 2) or not np.isfinite(q).all():
        return None
    centroid = q.mean(axis=0)
    normals = []
    for i in range(4):
        a = q[i]
        b = q[(i + 1) % 4]
        ab = b - a
        length = float(math.hypot(ab[0], ab[1]))
        if length < 1e-9:
            return None
        n = np.array([-ab[1] / length, ab[0] / length], dtype=float)
        if float(np.dot(centroid - a, n)) < 0.0:
            n = -n
        normals.append(n)
    return normals


def charuco_safe_quad(outer_quad, keepout_mm):
    """将板外轮廓四边形各边向内缩 keepout_mm，返回允许区顶点（退化为空时返回 None）。"""
    q = np.asarray(outer_quad, dtype=float)
    normals = _inward_edge_normals(q)
    if normals is None:
        return None
    d = float(keepout_mm)
    offsets = [float(np.dot(normals[i], q[i])) + d for i in range(4)]
    verts = []
    for i in range(4):
        j = (i + 1) % 4
        mat = np.vstack((normals[i], normals[j]))
        if abs(float(np.linalg.det(mat))) < 1e-9:
            return None
        verts.append(np.linalg.solve(mat, np.array([offsets[i], offsets[j]], dtype=float)))
    verts_arr = np.array(verts, dtype=float)
    for i in range(4):
        if np.any(verts_arr @ normals[i] < offsets[i] - 1e-6):
            return None
    return verts_arr


def point_in_board_safe_zone(point_xy, outer_quad, keepout_mm, *, tol_mm=0.05):
    """判断模组 XY 点是否落在板外轮廓向内缩 keepout_mm 的允许区内。"""
    p = np.asarray(point_xy, dtype=float)
    if p.shape != (2,) or not np.isfinite(p).all():
        return False
    q = np.asarray(outer_quad, dtype=float)
    normals = _inward_edge_normals(q)
    if normals is None:
        return False
    d = float(keepout_mm) - float(tol_mm)
    return all(float(np.dot(p - q[i], normals[i])) >= d for i in range(4))


def charuco_layout(camcal_cfg, board_points, transform):
    """生成 ChArUco 板内角点模组坐标布局、5 主测点、4 加测点与模组方位名称。"""
    cols = int(camcal_cfg['cols'])
    rows = int(camcal_cfg['rows'])
    square_mm = float(camcal_cfg['square_mm'])
    nc, nr = cols - 1, rows - 1
    if nc <= 0 or nr <= 0:
        raise ValueError('ChArUco 标定板行列数无效')
    c_min, c_max = (1, nc - 2) if cols >= 5 else (0, nc - 1)
    r_min, r_max = (1, nr - 2) if rows >= 5 else (0, nr - 1)
    board_2d = np.asarray(board_points, dtype=float)[:, :2]
    layout = {}
    for r in range(r_min, r_max + 1):
        for c in range(c_min, c_max + 1):
            cid = r * nc + c
            layout[charuco_label(cid)] = np.asarray(transform(board_2d[cid]), dtype=float)

    if cols >= 5 and rows >= 5:
        x_lo, x_hi = 2.0 * square_mm, (cols - 2.0) * square_mm
        y_lo, y_hi = 2.0 * square_mm, (rows - 2.0) * square_mm
        x_mid, y_mid = 0.5 * cols * square_mm, 0.5 * rows * square_mm
    else:
        xs = [board_2d[r * nc + c, 0] for r in range(r_min, r_max + 1) for c in range(c_min, c_max + 1)]
        ys = [board_2d[r * nc + c, 1] for r in range(r_min, r_max + 1) for c in range(c_min, c_max + 1)]
        x_lo, x_hi, x_mid = min(xs), max(xs), 0.5 * (min(xs) + max(xs))
        y_lo, y_hi, y_mid = min(ys), max(ys), 0.5 * (min(ys) + max(ys))

    used = set()

    def nearest_unused(target_mod_xy):
        pool = [lbl for lbl in layout if lbl not in used] or list(layout)
        chosen = min(pool, key=lambda lbl: (round(float(np.linalg.norm(layout[lbl] - target_mod_xy)), 6), lbl))
        used.add(chosen)
        return chosen

    corner_nom = [(x_lo, y_lo), (x_hi, y_lo), (x_hi, y_hi), (x_lo, y_hi)]
    raw_corners = [nearest_unused(transform(xy)) for xy in corner_nom]
    center_lbl = nearest_unused(transform((x_mid, y_mid)))
    edge_nom = [(x_mid, y_lo), (x_hi, y_mid), (x_mid, y_hi), (x_lo, y_mid)]
    raw_edges = [nearest_unused(transform(xy)) for xy in edge_nom]

    # 按模组 XY 相对板中心分配人眼方位名：上=+Y, 下=-Y, 左=-X, 右=+X
    by_y_desc = sorted(raw_corners, key=lambda lbl: (-float(layout[lbl][1]), float(layout[lbl][0]), lbl))
    top_pair = sorted(by_y_desc[:2], key=lambda lbl: (float(layout[lbl][0]), lbl))
    bot_pair = sorted(by_y_desc[2:], key=lambda lbl: (float(layout[lbl][0]), lbl))
    tl, tr = top_pair[0], top_pair[1]
    bl, br = bot_pair[0], bot_pair[1]

    rem_edges = list(raw_edges)
    tm = max(rem_edges, key=lambda lbl: (float(layout[lbl][1]), -float(layout[lbl][0])))
    rem_edges.remove(tm)
    bm = min(rem_edges, key=lambda lbl: (float(layout[lbl][1]), float(layout[lbl][0])))
    rem_edges.remove(bm)
    lm, rm = sorted(rem_edges, key=lambda lbl: (float(layout[lbl][0]), lbl))

    primary = [tl, tr, br, bl, center_lbl]
    extra = [tm, rm, bm, lm]
    positions = {
        tl: '左上', tr: '右上', br: '右下', bl: '左下', center_lbl: '中心',
        tm: '上中', rm: '右中', bm: '下中', lm: '左中',
    }
    return layout, primary, extra, positions


def default_points(layout):
    """从布局边界选四角、中心和四条边的中点，距离相同按编号。"""
    low_x, low_y = np.min(list(layout.values()), axis=0)
    high_x, high_y = np.max(list(layout.values()), axis=0)
    mid_x, mid_y = (low_x + high_x) / 2, (low_y + high_y) / 2

    def nearest(x, y):
        return min(layout, key=lambda label: (np.linalg.norm(layout[label] - (x, y)), label))

    return (
        [nearest(*xy) for xy in ((low_x, high_y), (high_x, high_y), (high_x, low_y),
                                  (low_x, low_y), (mid_x, mid_y))],
        [nearest(*xy) for xy in ((mid_x, high_y), (high_x, mid_y), (mid_x, low_y), (low_x, mid_y))],
    )


def film_layout():
    """V2 现场摆放：卡片 +X 对设备 +Y，卡片 +Y 对设备 −X。"""
    card = {f'T{i:02d}': xy for i, xy in nominal_targets().items()}
    primary, extra = default_points(card)
    return {label: np.array([-y, x]) for label, (x, y) in card.items()}, primary, extra


def corner_labels_for_plan(layout, plan):
    """从初始计划点中识别四个角点（若不足 5 点则全部视为角点，5 点时排除最靠近中心者）。"""
    valid = [lbl for lbl in (plan or ()) if lbl in layout]
    if len(valid) < 5:
        return set(valid)
    pts = np.array([layout[lbl] for lbl in valid], dtype=float)
    center = pts.mean(axis=0)
    center_lbl = min(valid, key=lambda lbl: (float(np.linalg.norm(layout[lbl] - center)), lbl))
    return {lbl for lbl in valid if lbl != center_lbl}


def fit_plane(results, *, surface_type='film', board_thickness_mm=None):
    xy = np.array([result['xy'] for result in results])
    z = np.array([result['z_peak'] for result in results])
    center = xy.mean(axis=0)
    design = np.column_stack((xy - center, np.ones(len(xy))))
    (a, b, offset), _, rank, _ = np.linalg.lstsq(design, z, rcond=None)
    if rank != 3:
        raise GuardError('DEPTH-PLANE', '测点共线，无法拟合料盘平面', '平面判定',
                         '选择在 X 和 Y 方向均有足够分布的五个靶点。')
    residuals = z - design @ [a, b, offset]
    plane = {
        'a': float(a), 'b': float(b), 'c': float(offset - center @ [a, b]),
        'residuals': {result['label']: float(residual) for result, residual in zip(results, residuals)},
        'max_residual': float(np.max(np.abs(residuals))),
        'tilt_mm_per_100mm': float(100 * np.hypot(a, b)),
        'surface_type': surface_type,
    }
    if surface_type == 'charuco' and board_thickness_mm is not None:
        thickness = float(board_thickness_mm)
        plane.update({
            'plane_kind': 'board_top_surface',
            'board_thickness_mm': thickness,
            'plane_note': f'板面平面（板厚 {thickness:.2f}mm，盘底 = 板面 − 板厚方向换算，暂不用于生产）',
        })
    return plane


def repeatability(results, threshold, *, plane=None, target_type='film'):
    max_range = max(result['range'] for result in results)
    std = [result['std'] for result in results if result['std'] is not None]
    max_std = max(std) if std else None
    insufficient_items = [
        f"{result['label']}#{fit['repeat']}"
        for result in results
        for fit in (result.get('fits') or [])
        if not fit.get('failed') and not fit.get('quality_ok', True)
    ]
    total_fits = sum(
        1 for result in results for fit in (result.get('fits') or []) if not fit.get('failed')
    )
    out = {
        'max_range': max_range,
        'max_std': max_std,
        'threshold': threshold,
        'insufficient_fits': len(insufficient_items),
        'insufficient_items': insufficient_items,
        'total_fits': total_fits,
        'met': len(insufficient_items) == 0 and max_std is not None and max(max_range, max_std) <= threshold,
    }
    if target_type == 'charuco':
        out['target_type'] = 'charuco'
        if isinstance(plane, dict) and isinstance(plane.get('max_residual'), (int, float)):
            res = float(plane['max_residual'])
            out['plane_max_residual'] = res
            out['plane_verdict'] = '可行' if res <= 0.02 else ('勉强' if res <= 0.05 else '不可行')
    return out


def dof_summary(results, dof_ratio):
    """汇总所有测点 DOF 均值/最小/最大（未覆盖的细扫及全未覆盖测点不参与统计）。"""
    point_dofs = [
        float(r['dof_mm']) for r in results
        if isinstance(r.get('dof_mm'), (int, float)) and math.isfinite(r['dof_mm'])
    ]
    uncovered_items = [
        f"{r['label']}#{fit['repeat']}"
        for r in results
        for fit in (r.get('fits') or [])
        if not fit.get('failed') and fit.get('dof_covered') is False
    ]
    total_fits = sum(
        1 for r in results for fit in (r.get('fits') or []) if not fit.get('failed')
    )
    covered_fits = sum(
        1 for r in results for fit in (r.get('fits') or [])
        if not fit.get('failed') and fit.get('dof_covered') is True
    )
    return {
        'dof_ratio': float(dof_ratio),
        'mean_mm': float(statistics.mean(point_dofs)) if point_dofs else None,
        'min_mm': float(min(point_dofs)) if point_dofs else None,
        'max_mm': float(max(point_dofs)) if point_dofs else None,
        'covered_points': len(point_dofs),
        'total_points': len(results),
        'covered_fits': covered_fits,
        'total_fits': total_fits,
        'uncovered_fits': len(uncovered_items),
        'uncovered_items': uncovered_items,
    }
