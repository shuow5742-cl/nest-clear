"""Gen1-compatible hair detection."""
from .config import HairDetectConfig
from .steps import HairCandidate, detect_candidates, gen1_pixel_gates

__all__ = ['HairDetectConfig', 'HairCandidate', 'detect_candidates', 'gen1_pixel_gates']
