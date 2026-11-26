"""Synthetic exposure response; no hardware, production runs, or calibration."""
import cv2
import numpy as np
import pytest
from test_trident import scene
from nest_clear.tooltip.exposure_scan import scan_exposure
from nest_clear.tooltip.trident import TridentTipDetector


class Camera:
    def __init__(self, fail=False):
        self.exposure = 8
        self.fail = fail
        self.counts = {}
    def get_exposure(self): return self.exposure
    def exposure_limits(self): return {'min': 1, 'max': 4, 'step': 1}
    def set_exposure(self, value): self.exposure = value
    def capture(self):
        self.counts[self.exposure] = self.counts.get(self.exposure, 0) + 1
        im = scene().astype(float)
        if self.exposure == 1 or self.fail:
            # Dim backlight and visible LED texture; insufficient contrast.
            im = im * .4
        elif self.exposure == 2:
            im = im * .7
        elif self.exposure == 3:
            im = im * 1.0
        else:
            im = np.clip(im * 1.2, 0, 255)
            # Backlight consumes two fine teeth while leaving the shaft intact.
            im[402:460, 285:295] = 255
            im[402:460, 305:315] = 255
        return np.clip(im, 0, 255).astype(np.uint8)


def test_scans_three_fresh_frames_and_selects_brightest_quality_level():
    cam = Camera(); updates = []
    result = scan_exposure(cam, TridentTipDetector(), 3, levels=4, contrast_min=40,
                           saturation_max=.05, progress=lambda k, n, row: updates.append((k,n)))
    assert result['exposure'] == cam.exposure == 3
    assert cam.counts == {1:4, 2:4, 3:4, 4:4}
    assert updates == [(i,4) for i in range(1,5)]
    assert result['rows'][2]['selected'] and not result['rows'][3]['qualified']
    assert result['rows'][3]['prongs'] < 3
    assert result['rows'][0]['background'] < result['rows'][2]['background']


def test_no_qualifying_level_restores_original_and_reports_every_level():
    cam = Camera(fail=True)
    with pytest.raises(ValueError, match='没有合格曝光档') as error:
        scan_exposure(cam, TridentTipDetector(), 3, levels=4, contrast_min=40, saturation_max=.05)
    assert cam.exposure == 8
    assert len(error.value.rows) == 4
    assert all(row['reasons'] and not row['qualified'] for row in error.value.rows)


@pytest.mark.parametrize('host_type', ['main', 'supervisor'])
@pytest.mark.parametrize('has_saved', [True, False])
def test_owner_hosts_execute_scan_and_reply(tmp_path, monkeypatch, host_type, has_saved):
    import json
    import time
    from datetime import datetime
    from types import SimpleNamespace
    from nest_clear.baseline.preparation import Preparation
    from nest_clear.devices.tooltip_camera_store import save_measurement, load_settings
    from test_t77b_capture import wait_for

    class OwnedCamera(Camera):
        def __init__(self):
            super().__init__()
            self.exposure = 3
            self.device = None
            self.config = {'device': 'front', 'settings_path': str(tmp_path / 'cameras.yaml')}
        def open(self): self.device = object()
        def close(self): self.device = None
        def settings_snapshot(self): return {'device': 'front'}
        def capture(self):
            time.sleep(.005)
            return super().capture()
    camera = OwnedCamera()
    if has_saved:
        save_measurement(tmp_path / 'cameras.yaml', 'front', {'device': 'front'},
                         {'exposure': 3, 'p99': 230, 'saturated_fraction': 0, 'contrast': 100})
    worker = SimpleNamespace(cameras={'front': camera, 'side': camera})
    monkeypatch.setattr('nest_clear.baseline.preparation.enumerate_capture_devices', lambda: [])
    events_seen = []
    events = SimpleNamespace(directory=tmp_path, clock=lambda: datetime.now().astimezone(), monotonic=time.monotonic,
                             emit=lambda event, **data: events_seen.append((event, data)))
    config = SimpleNamespace(record_directory=str(tmp_path), capture_timeout_s=1,
        tool_calibration=SimpleNamespace(cameras={}, detection={'front': {}, 'side': {}},
                                         auto_exposure={}, exposure_scan={'max_exposure':4}))
    host = None
    if host_type == 'supervisor':
        from nest_clear.supervisor.tooltip_host import TooltipHost
        monkeypatch.setattr('nest_clear.supervisor.tooltip_host.CalibCameraWorker', lambda *a, **k: worker)
        monkeypatch.setattr('nest_clear.baseline.session.settings', lambda path: {})
        host = TooltipHost(config, events)
        prep = host.tooltip_preparation
    else:
        prep = Preparation(events, worker, config)
    try:
        prep.command('tooltip_preview', {'camera': 'front', 'on': True})
        if not has_saved:
            wait_for(lambda: prep.state['front'].get('exposure_saved'), timeout=15)
            assert any(event == 'tooltip_exposure_scan' and data['success'] for event, data in events_seen)
        wait_for(lambda: prep.state['front'].get('detection_status') == '精定位')
        prep.command('tooltip_exposure_retest', {'camera': 'front', '_seq': 51})
        reply = wait_for(lambda: (x if (x := prep.state['front'].get('exposure_result', {})).get('seq') == 51 else None), timeout=15)
        assert reply['state'] == 'done' and camera.exposure == 3
        assert load_settings(tmp_path / 'cameras.yaml')['measurements']['front']['exposure'] == 3
        assert len(prep.state['front']['exposure_scan_rows']) == 4
        paths = list(tmp_path.glob('*/tooltip_exposure/*_front.json'))
        # Reply is sent at completion; wait for record publication too.
        if not paths:
            paths = wait_for(lambda: list(tmp_path.glob('*/tooltip_exposure/*_front.json')))
        assert json.loads(paths[0].read_text())['success']
        assert any(event == 'tooltip_exposure_scan' and data['success'] for event, data in events_seen)
        assert any('exposure_progress' in data.get('preview', {}).get('front', {}) for event, data in events_seen if event == 'baseline_state')
        wait_for(lambda: prep.state['front'].get('detection_status') == '精定位')
        camera.fail = True
        prep.command('tooltip_exposure_retest', {'camera': 'front', '_seq': 52})
        failed = wait_for(lambda: (x if (x := prep.state['front'].get('exposure_result', {})).get('seq') == 52 else None), timeout=15)
        assert failed['state'] == 'error' and '没有合格曝光档' in failed['message']
        assert camera.exposure == 3
        assert any(event == 'tooltip_exposure_scan' and not data['success'] for event, data in events_seen)
        assert len(prep.state['front']['exposure_scan_rows']) == 4
    finally:
        if host: host.close()
        else: prep.stop_previews(('front',)); prep.close()


def test_equal_background_chooses_shorter_exposure():
    class TiedCamera(Camera):
        def capture(self): return scene()
    cam = TiedCamera()
    result = scan_exposure(cam, TridentTipDetector(), 3, levels=4, contrast_min=40, saturation_max=.05)
    assert result['exposure'] == 1


def test_whiteness_in_segmentation_hole_is_measured():
    from types import SimpleNamespace
    from nest_clear.tooltip.exposure_scan import _frame_metrics
    image = scene()
    image[200:260, 294:306] = 250
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[100:401, 285:316] = 1
    mask[200:260, 294:306] = 0
    class Detector:
        def detect(self, frame):
            return SimpleNamespace(tool_mask=mask, status='精定位', diagnostic='', tip_contrast=100,
                                   sharpness={'band_gradient_p995': 100})
    assert _frame_metrics(image, Detector())['tool_whiteness'] > .05
