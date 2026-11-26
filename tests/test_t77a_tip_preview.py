import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from nest_clear.baseline.preparation import analyze_tip, draw_tip_preview
from nest_clear.tooltip.detector import SilhouetteTipDetector, TipDetection
from nest_clear.tooltip.focus_metrics import EdgeWidthTracker


def frame(sigma=0):
    image = np.full((480, 640), 220, np.uint8)
    cv2.line(image, (320, 50), (320, 240), 25, 12)
    return cv2.GaussianBlur(image, (0, 0), sigma) if sigma else image


def test_validity_and_best_with_synthetic_frames():
    tracker = EdgeWidthTracker()
    detector = SilhouetteTipDetector()
    sharp = frame(.5)
    blur = frame(3)
    status, reason, metrics, result = analyze_tip(sharp, detector.detect(sharp), tracker, 15.2, 40)
    assert result['percent'] is None
    for _ in range(2):
        status, reason, metrics, result = analyze_tip(sharp, detector.detect(sharp), tracker, 15.2, 40)
    assert status == '精定位' and not reason and result['percent'] == 100
    best = result['best']
    for image in (np.full_like(sharp, 255), np.random.default_rng(13).integers(99, 102, sharp.shape, dtype=np.uint8)):
        _, reason, metrics, result = analyze_tip(image, detector.detect(image), tracker, 15.2, 40)
        assert reason and metrics['edge_width_px'] is None and result['best'] == best and result['percent'] is None
    noise = np.random.default_rng(14).integers(99, 102, sharp.shape, dtype=np.uint8)
    _, reason, _, result = analyze_tip(noise, TipDetection((20, 20), 1), tracker, 15.2, 40)
    assert reason and result['best'] == best
    _, reason, metrics, result = analyze_tip(sharp, TipDetection((320, 240), 1, '退化', coarse=True), tracker, 15.2, 40)
    assert reason and metrics['edge_width_px'] is None and result['best'] == best
    _, _, _, result = analyze_tip(blur, detector.detect(blur), tracker, 15.2, 40)
    assert result['percent'] < 100
    _, _, _, result = analyze_tip(sharp, detector.detect(sharp), tracker, 15.2, 40)
    assert result['percent'] == pytest.approx(100)


def test_tracker_consecutive_movement_and_reset():
    tracker = EdgeWidthTracker()
    tracker.update(1, (10, 10))
    tracker.update(1, (10, 10)); tracker.update(1, (10, 10))
    assert tracker.update(3, (70, 10))['percent'] is None
    tracker.update(3, (70, 10)); tracker.update(3, (70, 10))
    tracker.update(None, None)
    assert tracker.update(4, (200, 10))['best'] is None
    tracker.reset()
    assert tracker.update(4, (200, 10))['percent'] is None


def test_preview_overlay_uses_tip_roi_only():
    image = frame()
    marked = draw_tip_preview(image, 'side', TipDetection((320, 240), 1), {}, 15.2)
    assert tuple(marked[160, 240]) == (0, 220, 0)
    missed = draw_tip_preview(image, 'side', TipDetection(None, 0, 'no dark component'), {}, 15.2)
    assert tuple(missed[160, 240]) == (220, 220, 220)


def test_ui_import_and_model_with_optional_modules_blocked(tmp_path):
    code = '''
import sys
sys.modules.update({name: None for name in ("cv2", "numpy", "yaml")})
from nest_clear.ui.app import App
from nest_clear.ui.model import baseline_preview_display
data = baseline_preview_display({"on": True}, {"percent": None, "detection_status": "未识别"})
assert not data["no_live_data"]
'''
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'src'))
    subprocess.run(['/usr/bin/python3', '-c', code], cwd=tmp_path, env=env, check=True)


def test_tooltip_host_receives_main_baseline_settings(tmp_path, monkeypatch):
    from nest_clear.baseline.session import settings
    from nest_clear.main import ROOT
    from nest_clear.supervisor.tooltip_host import TooltipHost
    monkeypatch.setattr('nest_clear.baseline.preparation.enumerate_capture_devices', lambda: [])
    monkeypatch.setattr('nest_clear.supervisor.tooltip_host.CalibCameraWorker', lambda *args, **kwargs: SimpleNamespace(cameras={}))
    config = SimpleNamespace(record_directory=str(tmp_path), capture_timeout_s=1,
                                 vision=SimpleNamespace(calibration=tmp_path / "camera.yaml"),
                                 tool_calibration=SimpleNamespace(directory=tmp_path, cameras={}))
    events = SimpleNamespace(clock=None, monotonic=None, emit=lambda *args, **kwargs: None)
    host = TooltipHost(config, events)
    assert host.tooltip_preparation.ui_config == settings(ROOT / 'config/tooltip_baseline.yaml')
    host.tooltip_preparation.close()


def test_exposure_change_and_retest_reset_best(tmp_path, monkeypatch):
    from nest_clear.baseline.preparation import Preparation
    from nest_clear.devices.tooltip_camera_store import save_measurement
    path = tmp_path / 'settings.yaml'
    save_measurement(path, 'front', {'device': 'front'}, {'exposure': 100, 'p99': 210, 'saturated_fraction': 0, 'contrast': 100})
    class Camera:
        config = {'device': 'front', 'settings_path': str(path)}
        exposure = 100
        sigma = .5
        def open(self): pass
        def close(self): pass
        def get_exposure(self): return self.exposure
        def set_exposure(self, value): self.exposure = value
        def settings_snapshot(self): return {'device': 'front'}
        def capture(self): return frame(self.sigma)
    camera = Camera()
    events = SimpleNamespace(directory=tmp_path, emit=lambda *args, **kwargs: None)
    monkeypatch.setattr('nest_clear.baseline.preparation.enumerate_capture_devices', lambda: [])
    monkeypatch.setattr('nest_clear.baseline.preparation.auto_exposure', lambda *args, **kwargs: {'exposure': camera.exposure, 'p99': 210, 'saturated_fraction': 0, 'contrast': 100})
    config = SimpleNamespace(tool_calibration=SimpleNamespace(detection={n: {'method': 'silhouette'} for n in ('front', 'side')}))
    prep = Preparation(events, SimpleNamespace(cameras={'front': camera, 'side': camera}), config)
    def wait_for(predicate):
        end = time.monotonic() + 5
        while time.monotonic() < end:
            if predicate(): return
            time.sleep(.02)
        pytest.fail(str(prep.state['front']))
    try:
        prep.command('tooltip_preview', {'camera': 'front', 'on': True})
        wait_for(lambda: prep.state['front'].get('percent') == 100)
        camera.sigma = 3
        wait_for(lambda: prep.state['front'].get('percent') is not None and prep.state['front']['percent'] < 90)
        camera.exposure = 200
        wait_for(lambda: prep.state['front'].get('exposure_ms') == 20 and prep.state['front'].get('percent') == 100)
        camera.sigma = .5
        wait_for(lambda: prep.state['front'].get('peak') is not None and prep.state['front']['peak'] < 2)
        camera.sigma = 3
        wait_for(lambda: prep.state['front'].get('percent') is not None and prep.state['front']['percent'] < 90)
        prep.command('tooltip_exposure_retest', {'camera': 'front'})
        wait_for(lambda: prep.state['front'].get('percent') == 100)
    finally:
        prep.close()
        prep.threads['front'].join(5)
