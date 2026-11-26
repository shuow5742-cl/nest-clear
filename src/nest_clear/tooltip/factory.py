"""Configuration-based detector selection for camera-owning processes."""
from .detector import SILHOUETTE_VERSION, SilhouetteTipDetector
from .trident import DETECTOR_VERSION, TridentConfig, TridentTipDetector


def make_tip_detector(config, silhouette_class=SilhouetteTipDetector):
    config = dict(config or {})
    method = config.pop('method', 'trident')
    if method == 'silhouette':
        return silhouette_class(**{k: v for k, v in config.items() if k in (
            'direction', 'threshold', 'min_area', 'probe_offset_px', 'max_edge_width_px')})
    if method == 'trident':
        return TridentTipDetector({k: v for k, v in config.items() if k in TridentConfig.__dataclass_fields__})
    raise ValueError(f'未知针尖检测方法：{method}')


BASELINE_DETECTOR_WARNING = '基准由旧版识别算法建立，建议重建基准'


def detector_identity(config):
    method = config.get('method', 'trident')
    version = DETECTOR_VERSION if method == 'trident' else SILHOUETTE_VERSION
    return {'method': method, 'version': version}


def baseline_detector_warning(detector, configs):
    current = {name: detector_identity(configs.get(name, {})) for name in ('front', 'side')}
    return BASELINE_DETECTOR_WARNING if detector != current else ''
