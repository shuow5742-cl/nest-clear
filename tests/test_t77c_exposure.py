import threading
import time
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from nest_clear.devices.tooltip_exposure import auto_exposure, measure
from test_t77b_capture import make_preparation, wait_for


@pytest.mark.parametrize('command', ['tooltip_exposure_set', 'tooltip_exposure_save'])
def test_command_reader_preserves_execution_sequence(tmp_path, command):
    from nest_clear.tooloffset.commands import CommandReader, send
    from nest_clear.flow.leaves import PREVIEW_COMMANDS
    events = SimpleNamespace(emit=lambda *args, **kwargs: None)
    seq = send(tmp_path, command, {'camera': 'front', 'exposure_ms': '.3'})
    name, params = CommandReader(tmp_path, events).poll(PREVIEW_COMMANDS)
    assert name == command and params['_seq'] == seq


def test_white_with_needle_is_never_classified_dark():
    image = np.full((1080, 1920), 255, np.uint8)
    image[250:850, 955:965] = 0
    result = measure(image)
    assert result['p99'] == 255 and result['saturated_fraction'] > .99


def test_stale_pipeline_and_discrete_closest():
    class DelayedCamera:
        def __init__(self): self.exposure = 100; self.pipeline = [100] * 4
        def get_exposure(self): return self.exposure
        def set_exposure(self, value): self.exposure = value
        def capture(self):
            self.pipeline.append(self.exposure)
            level = min(255, self.pipeline.pop(0) * 80)
            image = np.full((160, 200), level, np.uint8)
            image[20:140, 98:102] = 0
            return image
    camera = DelayedCamera()
    result = auto_exposure(camera, initial_exposure=1)
    assert result['approximate'] and result['exposure'] == 3 and result['p99'] == 240
    assert result['saturated_fraction'] < .001 and camera.get_exposure() == 3
    assert [row['exposure'] for row in result['iterations']][:2] == [1, 2]


def test_stale_dark_samples_can_disagree_with_white_preview():
    class Camera:
        def __init__(self): self.exposure = 100; self.stale = 0
        def get_exposure(self): return self.exposure
        def set_exposure(self, value): self.exposure = value; self.stale = 4
        def capture(self):
            if self.stale:
                self.stale -= 1
                level = 0
            else:
                level = min(255, self.exposure * 80)
            image = np.full((100, 100), level, np.uint8)
            image[20:80, 49:51] = 0
            return image
    camera = Camera()
    assert measure(np.full((100, 100), 255, np.uint8))['p99'] == 255
    for value in (100, 200, 400, 800, 1600):
        camera.set_exposure(value)
        for _ in range(3): camera.capture()
        assert measure(camera.capture())['p99'] < 200
    result = auto_exposure(camera, initial_exposure=1)
    assert result['approximate'] and result['exposure'] == 3


def test_search_obeys_device_limits_and_step():
    class Camera:
        def __init__(self): self.value = 4
        def exposure_limits(self): return {'min': 2, 'max': 20, 'step': 2}
        def get_exposure(self): return self.value
        def set_exposure(self, value):
            assert 2 <= value <= 20 and value % 2 == 0
            self.value = value
        def capture(self): return np.full((50, 50), min(255, self.value * 28), np.uint8)
    camera = Camera()
    result = auto_exposure(camera, initial_exposure=2)
    assert result['exposure'] == 8 and result['p99'] == 224


@pytest.mark.parametrize('host_type', ['main', 'supervisor'])
def test_set_restore_validation_and_busy(tmp_path, monkeypatch, host_type):
    prep, camera = make_preparation(tmp_path, monkeypatch, .01)
    camera.exposure_limits = lambda: {'min': 2, 'max': 100, 'step': 2}
    host = None
    if host_type == 'supervisor':
        from nest_clear.supervisor.tooltip_host import TooltipHost
        worker = prep.worker
        prep.close()
        monkeypatch.setattr('nest_clear.supervisor.tooltip_host.CalibCameraWorker', lambda *args, **kwargs: worker)
        monkeypatch.setattr('nest_clear.baseline.session.settings', lambda path: {})
        config = SimpleNamespace(record_directory=str(tmp_path), capture_timeout_s=1, tool_calibration=SimpleNamespace(cameras={}, detection={'front': {}, 'side': {}}))
        events = SimpleNamespace(clock=lambda: datetime.now().astimezone(), monotonic=time.monotonic, emit=lambda *args, **kwargs: None)
        host = TooltipHost(config, events)
        prep = host.tooltip_preparation
    saved = (tmp_path / 'cameras.yaml').read_bytes()
    def issue(seq, action, value=None):
        prep.command(action, {'camera': 'front', '_seq': seq, 'exposure_ms': value})
        return wait_for(lambda: (result if (result := prep.state['front'].get('exposure_result', {})).get('seq') == seq else None))
    try:
        prep.command('tooltip_preview', {'camera': 'front', 'on': True})
        wait_for(lambda: prep.state['front'].get('exposure_limits'))
        assert issue(1, 'tooltip_exposure_set', '.4')['state'] == 'done'
        assert camera.exposure == 4 and prep.state['front']['exposure_source'] == '临时'
        assert issue(2, 'tooltip_exposure_set', '.3')['state'] == 'error'
        assert issue(3, 'tooltip_exposure_set', '100')['state'] == 'error'
        assert issue(4, 'tooltip_exposure_restore')['state'] == 'done'
        assert camera.exposure == 100 and prep.state['front']['exposure_source'] == '已存（自动测）'
        with prep.lock: prep.state['front']['exposure_busy'] = True
        assert issue(5, 'tooltip_exposure_set', '.4')['state'] == 'error'
        assert (tmp_path / 'cameras.yaml').read_bytes() == saved
        with prep.lock: prep.state['front']['exposure_busy'] = False
        (tmp_path / 'cameras.yaml').unlink()
        assert issue(6, 'tooltip_exposure_restore')['state'] == 'done'
        assert camera.exposure == 100 and prep.state['front']['exposure_source'] == '默认'
    finally:
        if host: host.close()
        else: prep.stop_previews(('front',)); prep.close()
    assert len({owner for _, owner, _ in camera.operations}) == 1


def test_close_during_exposure_reports_error(tmp_path, monkeypatch):
    prep, camera = make_preparation(tmp_path, monkeypatch, .01)
    camera.exposure_limits = lambda: {'min': 1, 'max': 100, 'step': 1}
    entered, release = threading.Event(), threading.Event()
    original = camera.set_exposure
    def slow_set(value):
        entered.set()
        assert release.wait(3)
        original(value)
    camera.set_exposure = slow_set
    try:
        prep.command('tooltip_preview', {'camera': 'front', 'on': True})
        wait_for(lambda: prep.state['front'].get('exposure_limits'))
        prep.command('tooltip_exposure_set', {'camera': 'front', '_seq': 31, 'exposure_ms': '.3'})
        assert entered.wait(3)
        prep.command('tooltip_preview', {'camera': 'front', 'on': False})
        release.set()
        prep.threads['front'].join(5)
        assert prep.state['front']['exposure_result']['seq'] == 31
        assert prep.state['front']['exposure_result']['state'] == 'error'
    finally:
        release.set()
        prep.stop_previews(('front',))
        prep.close()


def test_set_is_rejected_during_retest(tmp_path, monkeypatch):
    prep, camera = make_preparation(tmp_path, monkeypatch, .01)
    camera.exposure_limits = lambda: {'min': 1, 'max': 100, 'step': 1}
    entered, release = threading.Event(), threading.Event()
    def retest(camera, **options):
        entered.set()
        assert release.wait(3)
        return {'exposure': 100, 'p99': 220, 'saturated_fraction': 0, 'contrast': 100}
    monkeypatch.setattr('nest_clear.baseline.preparation.auto_exposure', retest)
    try:
        prep.command('tooltip_preview', {'camera': 'front', 'on': True})
        wait_for(lambda: prep.state['front'].get('exposure_limits'))
        prep.command('tooltip_exposure_retest', {'camera': 'front'})
        assert entered.wait(3)
        prep.command('tooltip_exposure_set', {'camera': 'front', '_seq': 99, 'exposure_ms': '.3'})
        assert prep.state['front']['exposure_result']['state'] == 'error'
        assert '进行中' in prep.state['front']['exposure_result']['message']
        prep.command('tooltip_exposure_save', {'camera': 'front', '_seq': 100})
        assert prep.state['front']['exposure_result']['state'] == 'error'
        assert '进行中' in prep.state['front']['exposure_result']['message']
    finally:
        release.set()
        prep.stop_previews(('front',))
        prep.close()


@pytest.mark.parametrize('host_type', ['main', 'supervisor'])
def test_manual_save_restore_reopen_and_rejections(tmp_path, monkeypatch, host_type):
    from nest_clear.devices.tooltip_camera_store import load_settings, measurement_for_device
    prep, camera = make_preparation(tmp_path, monkeypatch, .01)
    camera.exposure_limits = lambda: {'min': 1, 'max': 100, 'step': 1}
    events_seen = []
    host = None
    if host_type == 'supervisor':
        from nest_clear.supervisor.tooltip_host import TooltipHost
        worker = prep.worker
        prep.close()
        monkeypatch.setattr('nest_clear.supervisor.tooltip_host.CalibCameraWorker', lambda *args, **kwargs: worker)
        monkeypatch.setattr('nest_clear.baseline.session.settings', lambda path: {})
        config = SimpleNamespace(record_directory=str(tmp_path), capture_timeout_s=1, tool_calibration=SimpleNamespace(cameras={}, detection={'front': {}, 'side': {}}))
        events = SimpleNamespace(clock=lambda: datetime.now().astimezone(), monotonic=time.monotonic, emit=lambda name, **kw: events_seen.append((name, kw)))
        host = TooltipHost(config, events)
        prep = host.tooltip_preparation
    else:
        prep.events.emit = lambda name, **kw: events_seen.append((name, kw))
    def issue(seq, action, **params):
        prep.command(action, {'camera': 'front', '_seq': seq, **params})
        return wait_for(lambda: (result if (result := prep.state['front'].get('exposure_result', {})).get('seq') == seq else None))
    path = tmp_path / 'cameras.yaml'
    try:
        assert issue(1, 'tooltip_exposure_save')['state'] == 'error'
        prep.command('tooltip_preview', {'camera': 'front', 'on': True})
        wait_for(lambda: prep.state['front'].get('exposure_limits'))
        assert issue(2, 'tooltip_exposure_set', exposure_ms='.4')['state'] == 'done'
        assert issue(3, 'tooltip_exposure_save', expected_exposure=5)['state'] == 'error'
        assert issue(4, 'tooltip_exposure_save', expected_exposure=4)['state'] == 'done'
        saved = measurement_for_device(load_settings(path), 'front')
        assert saved['source'] == 'manual' and saved['exposure'] == 4 and saved['time'] and saved['p99'] is not None
        event = [fields for event, fields in events_seen if event == 'tooltip_exposure_saved'][-1]
        assert event == dict(camera='front', device='front', exposure=4, source='manual', old_value=100)
        assert (tmp_path / 'history').is_dir()
        assert issue(5, 'tooltip_exposure_set', exposure_ms='.6')['state'] == 'done'
        assert issue(6, 'tooltip_exposure_restore')['state'] == 'done'
        assert camera.exposure == 4 and prep.state['front']['exposure_source'] == '已存（手动）'
        with prep.lock: prep.state['front']['exposure_busy'] = True
        assert issue(7, 'tooltip_exposure_save')['state'] == 'error'
        with prep.lock: prep.state['front']['exposure_busy'] = False
        prep.command('tooltip_preview', {'camera': 'front', 'on': False})
        prep.threads['front'].join(5)
        prep.command('tooltip_preview', {'camera': 'front', 'on': True})
        wait_for(lambda: prep.state['front'].get('exposure_limits'))
        assert prep.state['front']['exposure_source'] == '已存（手动）'
        assert not prep.retest['front'].is_set()
        assert camera.exposure == 4
        monkeypatch.setattr('nest_clear.baseline.preparation.measure', lambda frame: (_ for _ in ()).throw(ValueError('bad frame')))
        assert issue(8, 'tooltip_exposure_save')['state'] == 'done'
        assert measurement_for_device(load_settings(path), 'front')['p99'] is None
    finally:
        if host: host.close()
        else: prep.stop_previews(('front',)); prep.close()


def test_legacy_measurement_has_auto_source(tmp_path):
    from nest_clear.devices.tooltip_camera_store import load_settings, measurement_for_device
    from nest_clear.baseline.preparation import exposure_source
    path = tmp_path / 'legacy.yaml'
    path.write_text('cameras:\n  front:\n    device: front\n    measured: true\n    exposure: 42\n', encoding='utf-8')
    measured = measurement_for_device(load_settings(path), 'front')
    assert measured['exposure'] == 42 and exposure_source(measured) == '已存（自动测）'
