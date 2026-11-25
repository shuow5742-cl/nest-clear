"""Synthetic trident integration, without cameras or production calibration."""
from types import SimpleNamespace
import numpy as np
import pytest

from test_trident import scene
from nest_clear.baseline.preparation import analyze_tip, draw_tip_preview
from nest_clear.baseline.session import BaselineSession
from nest_clear.tooltip.factory import make_tip_detector
from nest_clear.tooltip.focus_metrics import BandGradientTracker
from nest_clear.tooltip.detector import SilhouetteTipDetector


def test_trident_preview_and_baseline_detection():
    image = scene()
    config = SimpleNamespace(tool_calibration=SimpleNamespace(detection={'front': {}, 'side': {}}))
    session = object.__new__(BaselineSession)
    session.config = config
    detections = session._detect({'front': image, 'side': image})
    assert all(d.status == '精定位' for d in detections.values())
    tracker = BandGradientTracker()
    for _ in range(5):
        status, reason, metrics, tracked = analyze_tip(image, detections['front'], tracker, 13.5, 40)
    assert status == '精定位' and not reason
    assert metrics['band_gradient'] > 0 and tracked['percent'] == pytest.approx(100)
    assert draw_tip_preview(image, 'front', detections['front'], {}, 13.5).shape == (*image.shape, 3)


def test_distorted_baseline_rejected_even_when_manually_clicked():
    from nest_clear.tooltip.click import ClickResult
    image = scene()
    distorted = np.stack([np.roll(row, 8 if (y // 9) % 2 else -8) for y, row in enumerate(image)])
    config = SimpleNamespace(tool_calibration=SimpleNamespace(detection={'front': {}, 'side': {}}))
    session = object.__new__(BaselineSession)
    session.config = config
    ds = session._detect({'front': distorted, 'side': image})
    assert '工具在动或画面畸变' in ds['front'].diagnostic
    with pytest.raises(ValueError, match='工具在动或画面畸变'):
        session._evaluate({'front': distorted, 'side': image}, ds, {'front': ClickResult((300, 450), 'manual', 1, 0)})


def test_select_old_silhouette_and_smooth_gradient():
    assert isinstance(make_tip_detector({'method': 'silhouette'}), SilhouetteTipDetector)
    tracker = BandGradientTracker()
    for x in (10, 10, 100, 10, 10): result = tracker.update(x)
    assert result['current'] == 10 and result['best'] == 10 and result['percent'] == 100
