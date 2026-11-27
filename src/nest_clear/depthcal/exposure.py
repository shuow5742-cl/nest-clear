"""下针深度标定曝光模式持久化与参数校验（无 cv2/numpy/yaml 依赖）。"""

from datetime import datetime
import json
import math
import os
from pathlib import Path
import tempfile

from nest_clear.depth_config import DEFAULTS as DEPTH_DEFAULTS
from nest_clear.paths import depthcal_exposure_file


def load_depthcal_exposure(root, cfg=None):
    """读取 runs/state/depthcal_exposure.json；缺失或损坏时回退配置默认值。"""
    base = cfg or DEPTH_DEFAULTS
    default_mode = base.get('exposure_mode', 'auto')
    default_manual = base.get('manual_exposure_us')
    exp_min = float(base.get('exposure_min_us', DEPTH_DEFAULTS['exposure_min_us']))
    exp_max = float(base.get('exposure_max_us', DEPTH_DEFAULTS['exposure_max_us']))
    path = depthcal_exposure_file(root)
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ValueError('not a dict')
        mode = data.get('mode', default_mode)
        if mode not in ('auto', 'manual'):
            mode = default_mode
        manual = data.get('manual_exposure_us', default_manual)
        if manual is not None:
            manual = float(manual)
            if not math.isfinite(manual) or manual <= 0:
                manual = default_manual
            else:
                manual = min(max(manual, exp_min), exp_max)
        return {'mode': mode, 'manual_exposure_us': manual}
    except (OSError, ValueError, TypeError):
        return {'mode': default_mode, 'manual_exposure_us': default_manual}


def validate_depthcal_exposure(mode, manual_exposure_us, cfg=None):
    """校验并规范化下针深度标定曝光设置；无效时抛 ValueError。"""
    base = cfg or DEPTH_DEFAULTS
    exp_min = float(base.get('exposure_min_us', DEPTH_DEFAULTS['exposure_min_us']))
    exp_max = float(base.get('exposure_max_us', DEPTH_DEFAULTS['exposure_max_us']))
    if mode not in ('auto', 'manual'):
        raise ValueError('曝光模式必须为自动或手动')
    manual_val = None
    if manual_exposure_us is not None and str(manual_exposure_us).strip() != '':
        try:
            manual_val = float(manual_exposure_us)
        except (TypeError, ValueError):
            raise ValueError('请输入有效的手动曝光数值（µs）') from None
        if not math.isfinite(manual_val) or manual_val <= 0:
            raise ValueError('手动曝光须为正的有限数（µs）')
        if not exp_min <= manual_val <= exp_max:
            raise ValueError(f'手动曝光须在 {exp_min:g}–{exp_max:g} µs 范围内')
    elif mode == 'manual':
        raise ValueError('手动模式下请输入曝光数值（µs）')
    return mode, manual_val


def save_depthcal_exposure(root, mode, manual_exposure_us, *, seq=None, clock=None):
    """原子持久化下针深度标定曝光模式与手动值到 runs/state/depthcal_exposure.json。"""
    path = depthcal_exposure_file(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = (clock() if clock else datetime.now().astimezone()).isoformat()
    payload = {
        'mode': mode,
        'manual_exposure_us': manual_exposure_us,
        'updated_at': now,
    }
    if seq is not None:
        payload['seq'] = int(seq)
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', dir=path.parent, prefix='.depthcal_exp_', delete=False
        ) as stream:
            tmp_name = stream.name
            json.dump(payload, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
    finally:
        if tmp_name:
            Path(tmp_name).unlink(missing_ok=True)
    return payload
