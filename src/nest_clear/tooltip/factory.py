"""Configuration-based detector selection for camera-owning processes."""
from .detector import SilhouetteTipDetector
from .trident import TridentConfig, TridentTipDetector


def make_tip_detector(config, silhouette_class=SilhouetteTipDetector):
    config = dict(config or {})
    method = config.pop('method', 'trident')
    if method == 'silhouette':
        return silhouette_class(**{k: v for k, v in config.items() if k in (
            'direction', 'threshold', 'min_area', 'probe_offset_px', 'max_edge_width_px')})
    if method == 'trident':
        return TridentTipDetector({k: v for k, v in config.items() if k in TridentConfig.__dataclass_fields__})
    raise ValueError(f'未知针尖检测方法：{method}')
