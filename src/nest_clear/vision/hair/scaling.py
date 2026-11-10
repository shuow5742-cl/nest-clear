"""Single inventory of reference tuning and conversion to the current lens."""
from dataclasses import fields, replace
import math

from .config import HairDetectConfig

# Integer lengths have a one-pixel floor; odd kernels have a three-pixel floor:
# 1x1 morphology removes/joins nothing and is not a useful neighbourhood.
# Optional seed_grow_close is a radius with 0 meaning disabled, not an odd kernel.
_LENGTH = {
    'bridge_chain_hop', 'bridge_flank_w', 'bridge_max_hop', 'hair_min_thick',
    'line_band', 'line_blur_sigma', 'line_ext_px', 'line_max_gap', 'line_min_span',
    'line_seed_max', 'line_skel_min', 'line_width_max', 'merge_gap_px',
    'pick_dens_sigma', 'trace_flank', 'trace_half_win', 'trace_step',
    'pale_min_pad', 'densify_step', 'bridge_corridor_th', 'trace_corridor_th',
    'merge_pad_extra', 'bridge_pad_extra', 'linefit_pad_extra', 'trace_crop_pad', 'seed_grow_pad',
}
_AREA = {'bridge_node_min_area', 'crumb_min_area', 'fiber_min_area', 'hair_area_norm',
         'line_cc_min_area', 'line_min_area', 'pick_evidence_min', 'seed_grow_max_area',
         'trace_seed_min_area', 'merge_close_bound_min_px'}
_KERNEL = {'dark_open_k', 'dark_close_k', 'covered_dilate_k', 'line_close_k', 'line_open_k',
           'link_close_k'}
_FIBER_KERNEL = {'fiber_close_k_ds', 'fiber_dens_win_ds', 'fiber_open_k'}
_DOWNSAMPLE = {'fiber_ds', 'merge_group_scale', 'seed_grow_scale', 'line_blur_scale'}
_OPTIONAL_RADIUS = {'seed_grow_close'}
_REASONS = {
    'length': '像素距离、宽度、半径或空间标准差，保持对应的物理长度',
    'area': '像素面积或像素数量门槛，保持对应的物理面积',
    'kernel': '形态学核或窗口，缩放后保持奇数及有效邻域',
    'radius_optional': '生长闭运算全分辨率半径；处理图上乘生长比例，零表示关闭',
    'downsample': '整数倍数 max(1, round(一代倍数×k))；浮点比例 min(1, 一代比例÷k)',
    'fiber_kernel': '纤维处理图核：参考处理图核×参考 fiber_ds×k÷实际 fiber_ds，取奇数且至少 3',
    'fixed': '灰度、比例、角度、性能预算、开关或毫米制参数，不按像素缩放',
}
_SPECIFIC_REASONS = {
    'geom_subsample_over': '大团块 PCA 点数阈值，性能预算；不是物理面积门',
    'geom_subsample_target': 'PCA 采样点数目标，性能预算；不是物理尺寸',
    'bridge_peri_max_pts': '桥接轮廓采样点数上限，控制两两搜索成本',
    'line_max_lines': '线拟合每帧的最大轮数，性能预算',
    'pale_sample_max': '灰度统计采样量上限，性能预算',
    'line_n_min': '拟合所需独立连通域数量，不是面积或长度',
    'trace_max_steps': '追踪迭代次数上限，性能预算；步长单独缩放',
    'trace_max_miss': '追踪容许连续失败次数，算法计数；步长单独缩放',
    'fiber_ds': '纤维路径整数降采样倍数=max(1,round(参考倍数×k))；处理图核乘参考倍数×k÷实际倍数',
    'merge_group_scale': '合并分组整数降采样倍数=max(1,round(参考倍数×k))；分组半径由实际像素间距÷倍数求出',
    'seed_grow_scale': '生长图比例=min(1,参考比例÷k)；处理图半径和高斯σ由全分辨率值乘此比例',
    'line_blur_scale': '背景图比例=min(1,参考比例÷k)；处理图高斯σ由全分辨率σ乘此比例',
    'merge_close_bound_min_px': 'local.size 性能分支门槛，按原图像素面积×k²；两分支设计为相同闭运算',
}
_DEFAULT = HairDetectConfig()
# name -> (gen1 default, category, reason). Scale calibration is metadata.
INVENTORY = {}
for f in fields(HairDetectConfig):
    category = ('length' if f.name in _LENGTH else 'area' if f.name in _AREA else
                'kernel' if f.name in _KERNEL else 'fiber_kernel' if f.name in _FIBER_KERNEL else
                'downsample' if f.name in _DOWNSAMPLE else
                'radius_optional' if f.name in _OPTIONAL_RADIUS else 'fixed')
    default = 6.959 if f.name == 'um_per_px' else getattr(_DEFAULT, f.name)
    INVENTORY[f.name] = (default, category, _SPECIFIC_REASONS.get(f.name, _REASONS[category]))


def _odd(value: float) -> int:
    # Nearest odd; ties choose the larger neighbourhood. Minimum 3.
    return max(3, 2 * math.floor(value / 2) + 1)


def scaled(cfg: HairDetectConfig) -> HairDetectConfig:
    """Convert reference-pixel settings once; pass the result to the detector.

    This is deliberately not idempotent: apply to reference settings only.
    """
    if not (math.isfinite(cfg.um_per_px) and cfg.um_per_px > 0 and
            math.isfinite(cfg.reference_um_per_px) and cfg.reference_um_per_px > 0):
        raise ValueError('像素尺度必须为正的有限数')
    k = cfg.reference_um_per_px / cfg.um_per_px
    if not math.isfinite(k) or k == 0 or not math.isfinite(k * k):
        raise ValueError("像素尺度比例超出数值范围")
    if cfg.pick_point != "thick_central":
        raise ValueError("仅支持 thick_central 取点模式")
    if k == 1:
        return cfg
    # Fibers specify kernels in the reference processing grid, not the
    # reference full-resolution grid. Convert through physical full pixels.
    fiber_ds_eff = max(1, round(cfg.fiber_ds * k))
    values = {}
    for name, (_, category, _) in INVENTORY.items():
        value = getattr(cfg, name)
        if category == 'length':
            def length(x, integer):
                return max(1, round(x * k)) if integer else max(1e-6, x * k)
            ref_type = getattr(_DEFAULT, name)
            values[name] = (tuple(length(x, isinstance(ref, int)) for x, ref in zip(value, ref_type))
                            if isinstance(value, tuple) else length(value, isinstance(ref_type, int)))
        elif category == 'area':
            v = value * k * k
            values[name] = max(1, round(v)) if isinstance(getattr(_DEFAULT, name), int) else max(1.0, v)
        elif category == 'kernel':
            values[name] = _odd(value * k)
        elif category == 'fiber_kernel':
            values[name] = _odd(value * cfg.fiber_ds * k / fiber_ds_eff)
        elif category == 'downsample':
            values[name] = (max(1, round(value * k)) if isinstance(getattr(_DEFAULT, name), int)
                            else min(1.0, value / k))
        elif category == 'radius_optional':
            values[name] = 0 if value == 0 else max(1, round(value * k))
    return replace(cfg, **values)


def effective_pixel_values(cfg: HairDetectConfig, *, already_scaled: bool = False) -> dict:
    """Return actual parameters for startup logging (call once on reference config)."""
    result = cfg if already_scaled else scaled(cfg)
    values = {name: getattr(result, name) for name, (_, category, _) in INVENTORY.items()
              if category != 'fixed'}
    ppm = 1000.0 / result.um_per_px
    values.update(pick_min_length_px=round(result.min_length_mm * ppm),
                  pick_min_area_px=round(result.min_area_mm2 * ppm * ppm),
                  pick_min_width_px=round(result.min_width_mm * ppm))
    return values
