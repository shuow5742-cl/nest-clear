"""菲林布局测点选择与模组坐标中的平面拟合。"""

import math
import statistics

import numpy as np

from nest_clear.guard import GuardError
from nest_clear.tooloffset.session import nominal_targets


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


def fit_plane(results):
    xy = np.array([result['xy'] for result in results])
    z = np.array([result['z_peak'] for result in results])
    center = xy.mean(axis=0)
    design = np.column_stack((xy - center, np.ones(len(xy))))
    (a, b, offset), _, rank, _ = np.linalg.lstsq(design, z, rcond=None)
    if rank != 3:
        raise GuardError('DEPTH-PLANE', '测点共线，无法拟合料盘平面', '平面判定',
                         '选择在 X 和 Y 方向均有足够分布的五个靶点。')
    residuals = z - design @ [a, b, offset]
    return {'a': float(a), 'b': float(b), 'c': float(offset - center @ [a, b]),
            'residuals': {result['label']: float(residual) for result, residual in zip(results, residuals)},
            'max_residual': float(np.max(np.abs(residuals))),
            'tilt_mm_per_100mm': float(100 * np.hypot(a, b))}


def repeatability(results, threshold):
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
    return {
        'max_range': max_range,
        'max_std': max_std,
        'threshold': threshold,
        'insufficient_fits': len(insufficient_items),
        'insufficient_items': insufficient_items,
        'total_fits': total_fits,
        'met': len(insufficient_items) == 0 and max_std is not None and max(max_range, max_std) <= threshold,
    }


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
