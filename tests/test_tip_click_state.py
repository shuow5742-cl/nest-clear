import pytest
from nest_clear.tooltip.click import ClickState


def test_click_pan_zoom_and_keys():
    state = ClickState((100, 80), (40., 30.), (200, 100))
    state.click(104, 54)
    assert state.point == pytest.approx((40.5, 30.5))
    assert state.source == 'manual'
    state.pan(8, -16)
    state.key(ord('+'))
    state.click(100, 50)
    assert state.point == pytest.approx((39., 32.))
    state.click(-10000, 0)
    assert state.point == pytest.approx((39., 32.))
    state.key(ord('r'))
    assert state.point == (40., 30.) and state.source == 'auto'
    state.key(13)
    assert state.done
    state = ClickState((10, 10))
    state.key(13)
    assert not state.done
    state.key(27)
    assert state.done and state.source == 'cancel'


def test_capture_reference_click_and_no_confirm(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    from nest_clear.tools import tooltip_baseline as baseline
    from nest_clear.tooltip.click import ClickResult
    class Camera:
        def capture(self): return np.zeros((20, 20), dtype=np.uint8)
    class Client:
        def read_real32_block(self, *args): return [0., 0., 0.]
    class Detector:
        def __init__(self, **kwargs): pass
        def detect(self, image): return SimpleNamespace(point_uv=(5., 6.), width_px=10., focused=True)
    monkeypatch.setattr(baseline, 'SilhouetteTipDetector', Detector)
    monkeypatch.setattr(baseline, 'confirm_tip', lambda *args: ClickResult((6., 8.), 'manual', 8., 1.))
    config = {'detection': {'front': {}, 'side': {}}, 'center_warning_px': 100}
    cameras = {'front': Camera(), 'side': Camera()}
    frame = baseline._capture(cameras, Client(), 0, tmp_path, 0, config, tool=1)
    assert frame['points']['front'] == [6., 8.]
    assert frame['clicks']['front']['correction_px'] == pytest.approx(5 ** .5)
    assert frame['widths']['front'] == 10
    frame = baseline._capture(cameras, Client(), 0, tmp_path, 0, config, no_confirm=True)
    assert frame['points']['front'] == [5., 6.]
    assert frame['clicks']['front']['source'] == 'auto'


def test_compensation_file_and_conflict(tmp_path):
    from nest_clear.tools.tooltip_baseline import _read_compensation, main
    path = tmp_path / 'compensation.yaml'
    path.write_text('tool: 1\noffset_xyz: [1, 2, 3]\ndepth_z: 4\nsource: test\ntimestamp: now\n')
    assert _read_compensation(str(path), 1)[0] == [1, 2, 4]
    with pytest.raises(ValueError, match='工具号'):
        _read_compensation(str(path), 2)
    with pytest.raises(SystemExit):
        main(['--tool', '1', '--offset', '1', '2', '3', '--compensation-from', str(path)])


def test_display_failure_is_chinese_guard_error(monkeypatch):
    import cv2
    import numpy as np
    from nest_clear.guard import GuardError
    from nest_clear.tooltip import click
    monkeypatch.delenv('DISPLAY', raising=False)
    with pytest.raises(GuardError, match='无法打开针尖确认窗口（无显示环境）'):
        click.confirm_tip(np.zeros((2, 2), dtype=np.uint8))
    monkeypatch.setattr(click, '_check_display', lambda: None)
    def broken(*args):
        raise cv2.error('NULL window handler')
    monkeypatch.setattr(click, '_confirm_tip', broken)
    with pytest.raises(GuardError, match='工控机本地桌面运行'):
        click._confirm_local(np.zeros((2, 2), dtype=np.uint8), None, '测试')


def _crash_window(connection, image, detection, title):
    import os
    os._exit(1)


def test_window_process_crash_returns_display_error(monkeypatch):
    import numpy as np
    from nest_clear.guard import GuardError
    from nest_clear.tooltip import click
    monkeypatch.setattr(click, '_check_display', lambda: None)
    monkeypatch.setattr(click, '_window_process', _crash_window)
    with pytest.raises(GuardError, match='无法打开针尖确认窗口（无显示环境）'):
        click.confirm_tip(np.zeros((2, 2), dtype=np.uint8))
