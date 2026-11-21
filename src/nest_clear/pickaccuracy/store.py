"""挑取补偿计算及确认后的原子持久化。"""
from pathlib import Path

import numpy as np
import yaml
from scipy.interpolate import LinearNDInterpolator


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
    data = read(path); data[f'tool{tool}'] = result
    from nest_clear.tooloffset.store import save_mapping
    return save_mapping(path, data, 'pick_compensation')
