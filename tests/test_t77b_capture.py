import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from nest_clear.baseline.preparation import Preparation
from nest_clear.baseline.latest_capture import FrameBroker
from nest_clear.devices.tooltip_camera_store import save_measurement
from nest_clear.tooltip.focus_metrics import EdgeWidthTracker


def test_best_median_rejects_outlier_and_jump_across_invalid():
    tracker = EdgeWidthTracker()
    assert tracker.update(2, (0, 0))['percent'] is None
    assert tracker.update(2, (0, 0))['percent'] is None
    assert tracker.update(2, (0, 0))['best'] == 2
    assert tracker.update(.2, (0, 0))['best'] == 2
    assert tracker.update(None, None)['best'] == 2
    assert tracker.update(4, (60, 0))['best'] is None
    assert tracker.update(4, (60, 0))['percent'] is None
    assert tracker.update(4, (60, 0))['best'] == 4


def test_broker_copies_reused_camera_buffers():
    broker = FrameBroker()
    frame = np.zeros((2, 2), np.uint8)
    first = broker.publish(frame, 1., 'first', 10)
    frame.fill(5)
    broker.publish(frame, 2., 'second', 10)
    assert int(first.frame[0, 0]) == 0


def make_preparation(tmp_path, monkeypatch, delay):
    settings = tmp_path / 'cameras.yaml'
    save_measurement(settings, 'front', {'device': 'front'}, {'exposure': 100, 'p99': 210, 'saturated_fraction': 0, 'contrast': 100})
    class Camera:
        config = {'device': 'front', 'settings_path': str(settings)}
        def __init__(self):
            self.exposure = 100
            self.count = 0
            self.operations = []
            self.closed = False
            self.device = None
        def record(self, action): self.operations.append((action, threading.get_ident(), time.monotonic()))
        def open(self): self.record('open'); self.device = object()
        def close(self): self.record('close'); self.closed = True; self.device = None
        def get_exposure(self): self.record('get'); return self.exposure
        def set_exposure(self, value): self.record('set'); self.exposure = value
        def settings_snapshot(self): return {'device': 'front'}
        def capture(self):
            self.record('read')
            time.sleep(delay)
            self.count += 1
            return np.full((80, 100), self.count % 255, np.uint8)
    camera = Camera()
    events = SimpleNamespace(directory=tmp_path, emit=lambda *args, **kwargs: None)
    monkeypatch.setattr('nest_clear.baseline.preparation.enumerate_capture_devices', lambda: [])
    prep = Preparation(events, SimpleNamespace(cameras={'front': camera, 'side': camera}))
    return prep, camera


def wait_for(predicate, timeout=6):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(.01)
    pytest.fail('等待预览状态超时')


@pytest.mark.parametrize('read_delay', [.005, .12])
def test_latest_frame_slow_processing_and_shutdown(tmp_path, monkeypatch, read_delay):
    prep, camera = make_preparation(tmp_path, monkeypatch, read_delay)
    from nest_clear.baseline import preparation
    draw = preparation.draw_tip_preview
    def slow_draw(*args):
        time.sleep(.25)
        return draw(*args)
    monkeypatch.setattr(preparation, 'draw_tip_preview', slow_draw)
    try:
        prep.command('tooltip_preview', {'camera': 'front', 'on': True})
        first = wait_for(lambda: dict(prep._preview_cameras.get('front') or {}))
        time.sleep(1.1)
        latest = dict(prep._preview_cameras['front'])
        assert latest['frame_seq'] > first['frame_seq']
        assert latest['frame_age_ms'] < 900
        assert latest['processing_ms'] >= 200
        assert latest['fps'] > 0
        if read_delay < .01:
            assert latest['frame_seq'] - first['frame_seq'] > 10
    finally:
        prep.stop_previews(('front',))
        prep.close()
    assert camera.closed and not prep.capture_threads['front'].is_alive()
    assert len({owner for _, owner, _ in camera.operations}) == 1


def test_capture_after_returns_k_new_frames_and_timeout(tmp_path, monkeypatch):
    prep, camera = make_preparation(tmp_path, monkeypatch, .02)
    try:
        prep.command('tooltip_preview', {'camera': 'front', 'on': True})
        wait_for(lambda: camera.count > 0)
        pressed = time.monotonic()
        frames = prep.capture_after('front', pressed, 5, 3)
        assert len(frames) == 5
        assert all(item.captured_mono > pressed and item.captured_at for item in frames)
        assert [item.seq for item in frames] == sorted({item.seq for item in frames})
        with pytest.raises(TimeoutError, match='等待 2 帧超时'):
            prep.capture_after('front', time.monotonic() + 10, 2, .05)
    finally:
        prep.stop_previews(('front',))
        prep.close()
    with pytest.raises(RuntimeError, match='未采集'):
        prep.capture_after('front', time.monotonic(), 1, .1)


def test_retest_is_serial_and_reads_after_set(tmp_path, monkeypatch):
    prep, camera = make_preparation(tmp_path, monkeypatch, .01)
    def retest(proxy, **kwargs):
        proxy.set_exposure(30)
        for _ in range(4):
            previous = proxy.after_seq
            proxy.capture()
            assert proxy.after_seq > previous
        return {'exposure': 30, 'p99': 210, 'saturated_fraction': 0, 'contrast': 100}
    monkeypatch.setattr('nest_clear.baseline.preparation.auto_exposure', retest)
    try:
        prep.command('tooltip_preview', {'camera': 'front', 'on': True})
        wait_for(lambda: camera.count > 0)
        prep.command('tooltip_exposure_retest', {'camera': 'front'})
        wait_for(lambda: camera.exposure == 30 and len([op for op in camera.operations if op[0] == 'read']) >= 5)
        wait_for(lambda: prep.state['front'].get('exposure_source', '').startswith('已存'))
    finally:
        prep.stop_previews(('front',))
        prep.close()
    operations = camera.operations
    index = next(index for index, item in enumerate(operations) if item[0] == 'set')
    reads = [item for item in operations[index+1:] if item[0] == 'read']
    assert len(reads) >= 4 and all(item[2] > operations[index][2] for item in reads[:4])
    assert len({owner for _, owner, _ in operations}) == 1


@pytest.mark.parametrize('host_type', ['main', 'supervisor'])
def test_capture_interface_matches_both_hosts(tmp_path, monkeypatch, host_type):
    from datetime import datetime
    prep, camera = make_preparation(tmp_path, monkeypatch, .02)
    host = None
    if host_type == 'supervisor':
        from nest_clear.supervisor.tooltip_host import TooltipHost
        worker = prep.worker
        prep.close()
        monkeypatch.setattr('nest_clear.supervisor.tooltip_host.CalibCameraWorker', lambda *args, **kwargs: worker)
        monkeypatch.setattr('nest_clear.baseline.session.settings', lambda path: {'main_tol_mm': .15})
        config = SimpleNamespace(record_directory=str(tmp_path), capture_timeout_s=1, tool_calibration=SimpleNamespace(cameras={}, detection={'front': {}, 'side': {}}))
        events = SimpleNamespace(clock=lambda: datetime.now().astimezone(), monotonic=time.monotonic, emit=lambda *args, **kwargs: None)
        host = TooltipHost(config, events)
        prep = host.tooltip_preparation
    try:
        prep.command('tooltip_preview', {'camera': 'front', 'on': True})
        wait_for(lambda: camera.count > 0)
        pressed = time.monotonic()
        frames = prep.capture_after('front', pressed, 2, 2)
        assert all(item.captured_mono > pressed for item in frames)
        assert frames[0].seq < frames[1].seq
    finally:
        if host:
            host.close()
        else:
            prep.stop_previews(('front',))
            prep.close()
    assert camera.closed and not prep.capture_threads['front'].is_alive()
