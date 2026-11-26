"""挑取补偿计算及确认后的原子持久化。"""
import math
from pathlib import Path

import numpy as np
import yaml
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import QhullError


def evaluate(result, xy):
    """凸包内分段线性插值，凸包外使用最近节点。"""
    if not result:
        return (0., 0.)
    if result['type'] == 'constant':
        value = np.asarray(result['constant_xy'], dtype=float)
    elif result['type'] == 'table':
        nodes = result['nodes']
        points = np.asarray([[n['x'], n['y']] for n in nodes], dtype=float)
        values = np.asarray([[n['dx'], n['dy']] for n in nodes], dtype=float)
        point = np.asarray(xy, dtype=float)
        value = np.asarray(LinearNDInterpolator(points, values)(point)).reshape(2)
        if not np.isfinite(value).all():
            value = values[np.argmin(np.linalg.norm(points - point, axis=1))]
    else:
        raise ValueError('未知挑取补偿类型')
    if value.shape != (2,) or not np.isfinite(value).all():
        raise ValueError('挑取补偿值无效')
    return tuple(float(v) for v in value)


def has_calibrated_z(result):
    """检查该工具是否已完成挑取下针深度 Z 标定。"""
    return isinstance(result, dict) and result.get('z_type') in ('constant', 'table')


def evaluate_z(result, xy):
    """计算标定状态下的下针深度 Z（未含换针后的 offset Z 变化）。

    未标定（缺失或无 z_type）返回 None；
    constant 返回全盘最大值；
    table 在凸包内分段线性插值，凸包外或降维点集取最近节点（并列取最大即最高最安全）。
    """
    if not has_calibrated_z(result):
        return None
    z_type = result['z_type']
    if z_type == 'constant':
        value = result.get('z_mm')
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('挑取下针 Z 值无效')
        return float(value)
    if z_type == 'table':
        nodes = result.get('z_nodes')
        if not isinstance(nodes, list) or not nodes:
            raise ValueError('挑取下针 Z 表节点为空')
        try:
            points = np.asarray([[float(n['x']), float(n['y'])] for n in nodes], dtype=float)
            values = np.asarray([float(n['z']) for n in nodes], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('挑取下针 Z 表节点格式无效') from exc
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all() or not np.isfinite(values).all():
            raise ValueError('挑取下针 Z 表节点数值无效')
        point = np.asarray(xy, dtype=float)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError('查询 XY 坐标无效')
        if len(points) == 1:
            return float(values[0])
        try:
            interp = float(np.asarray(LinearNDInterpolator(points, values)(point)).reshape(-1)[0])
        except QhullError:
            interp = float('nan')
        if not math.isfinite(interp):
            # 凸包外或共线点集使用最近节点；若距离并列，取 Z 最大值（最高=最安全）。
            dists = np.linalg.norm(points - point, axis=1)
            dmin = float(dists.min())
            interp = float(np.max(values[dists - dmin <= 1e-9]))
        return interp
    raise ValueError('未知挑取下针 Z 类型')


def resolve_pick_z(result, xy, default_pick_depth_z, current_offset_z):
    """解析生产/菲林模式下发给 PLC 的模组挑取 Z，返回 (pick_z, is_calibrated)。

    符号推导（同 calculate_pick_coords 与 tooltip/core.py）：
    - calculate_pick_coords 约定 pick_z = pick_depth_z + offset[2]；
    - 换针后若新针更长，对准基准需要模组上移（+Z），tools.yaml 的 offset_xyz[2] 增大 (+Δz)，
      生产挑取的模组 Z 也相应增大 (+Δz) 才不会戳底；
    - 挑取精度标定记录的 z_calib 是在当时工具（tools.yaml 中记录为 z_ref）下实测的模组 Z，
      已包含 z_ref；换针后 offset_xyz[2] 变为 current_offset_z，变化量为 (current_offset_z - z_ref)，
      因此下针模组 Z = z_calib(xy) + (current_offset_z - z_ref)；
    - 若未标定下针 Z，回退到 default_pick_depth_z + current_offset_z，并返回 False。
    """
    z_calib = evaluate_z(result, xy)
    if z_calib is None:
        return float(default_pick_depth_z) + float(current_offset_z), False
    z_ref = result.get('z_ref', 0.0) if 'z_ref' in result else 0.0
    if isinstance(z_ref, bool) or not isinstance(z_ref, (int, float)) or not math.isfinite(z_ref):
        raise ValueError('挑取下针 Z 结果缺少有效的 z_ref')
    return float(z_calib) + (float(current_offset_z) - float(z_ref)), True


def read(path):
    path = Path(path)
    if not path.exists(): return {}
    data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    if not isinstance(data, dict): raise ValueError('挑取补偿文件格式无效')
    return data


def save(path, tool, result):
    if tool not in (1, 2): raise ValueError('工具号无效')
    path = Path(path)
    # 先校验，避免不完整结果覆盖已有标定。
    evaluate(result, (0., 0.))
    if has_calibrated_z(result):
        evaluate_z(result, (0., 0.))
        if 'z_ref' in result:
            z_ref = result['z_ref']
            if isinstance(z_ref, bool) or not isinstance(z_ref, (int, float)) or not math.isfinite(z_ref):
                raise ValueError('挑取下针 Z 参考值 z_ref 无效')
    data = read(path); data[f'tool{tool}'] = result
    from nest_clear.tooloffset.store import save_mapping
    return save_mapping(path, data, 'pick_compensation')
