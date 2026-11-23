import json
import cv2
import numpy as np
import pytest

from nest_clear.preview.focus import FocusSession, analyze_focus, exposure_hint, write_record
from nest_clear.ui.model import preview_controls


def pattern():
    rng = np.random.default_rng(123)
    return rng.integers(55, 165, (600, 800), dtype=np.uint8)


def test_blur_and_peak():
    image = pattern()
    scores = [analyze_focus(cv2.GaussianBlur(image, (0, 0), sigma))['scores'][4] for sigma in (.5, 1., 1.5)]
    assert scores[0] > scores[1] > scores[2] and scores[0]/scores[2] > 1.3
    session = FocusSession()
    first = session.update(image)
    for _ in range(4): result = session.update(cv2.GaussianBlur(image, (0, 0), 2))
    assert result['peaks'][4] == first['center'] and result['center'] < result['peaks'][4]
    session.reset()
    assert session.update(cv2.GaussianBlur(image, (0, 0), 2))['percent'][4] == pytest.approx(100)


def test_brightness_normalization():
    image = pattern()
    scores = [analyze_focus((image.astype(float)*factor).astype('uint8'))['scores'][4] for factor in (.5, 1., 1.5)]
    assert max(scores)/min(scores) < 1.1


def test_tilt_and_exposure():
    image = pattern(); session = FocusSession()
    session.update(image)
    altered = image.copy(); altered[:260, :340] = cv2.GaussianBlur(altered[:260, :340], (0, 0), 2)
    for _ in range(3): data = session.update(altered)
    assert '左上' in data['tilt'] and '右下' not in data['tilt']
    assert exposure_hint(analyze_focus(np.full((80,80), 255, np.uint8))).startswith('过曝')
    assert exposure_hint(analyze_focus(np.full((80,80), 80, np.uint8))) == '偏暗'


def test_record_and_controls(tmp_path):
    data = FocusSession().update(pattern()); data.update(exposure_us=100, gain_db=1)
    row = write_record(tmp_path, data, b'jpeg', 'F4')
    path = next(tmp_path.glob('focus_records/*/records.jsonl'))
    assert json.loads(path.read_text()) == row
    assert (path.parent / row['thumbnail']).read_bytes() == b'jpeg'
    assert all(k in row for k in ('time','note','scores','peaks','mean','p1','p99','saturated_pct','dark_pct','exposure_us','gain_db'))
    assert not preview_controls({'pid':1,'child_pid':2}, 1)['focus_enabled']
    controls = preview_controls({'pid':1,'preview':{'on':True,'mode':'focus'}}, 1)
    assert controls['focus_enabled'] and not controls['preview_enabled']


def test_mode_commands_are_exclusive(tmp_path, monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace
    from nest_clear.config import load_config
    from nest_clear.supervisor.runner import Supervisor
    from nest_clear.paths import ui_command_file
    class PLC:
        def read(self, name): return 0
    class Process:
        pid = 42
        exitcode = 0
        def __init__(self, target, args): self.args = args; self.alive = False
        def start(self): self.alive = True
        def is_alive(self): return self.alive
        def terminate(self): self.alive = False
        def join(self, timeout=None): pass
    monkeypatch.setattr('nest_clear.supervisor.runner.multiprocessing.get_context', lambda _: SimpleNamespace(Process=Process))
    sup = Supervisor(replace(load_config('config/nest_clear.yaml'), record_directory=str(tmp_path)), PLC())
    path = ui_command_file(tmp_path); path.parent.mkdir(parents=True, exist_ok=True)
    def command(seq, mode, on=True):
        path.write_text(json.dumps({'seq':seq, 'mode':mode, 'preview':on}))
        sup._preview_tick()
    command(1, 'focus'); first = sup.preview_process
    assert first.args[1] == 'focus'
    command(2, 'preview'); assert sup.preview_process is first
    command(3, 'focus', False); assert sup.preview_process is None and not first.alive
    command(4, 'preview'); assert sup.preview_process.args[1] == 'preview'


def test_demo_live_runs_analysis(tmp_path):
    from nest_clear.ui.demo import write_demo
    from nest_clear.paths import supervisor_state_file, main_status_file
    write_demo(tmp_path, initial=True, focus_state='live')
    data = json.loads(supervisor_state_file(tmp_path).read_text())['preview']
    assert data['mode'] == 'focus' and data['center'] > 0 and len(data['peaks']) == 9
    main = json.loads(main_status_file(tmp_path).read_text())
    assert main['mode'] == '空闲' and not main['camcal']
    assert len(data['demo_history']) >= 25
    values = [point[1] for point in data['demo_history']]
    assert max(values) > 1.3 * min(values)
    assert data['cell_means'][0] > 2 * data['cell_means'][2]
    for step in range(1, 6):
        write_demo(tmp_path, step, focus_state='live')
    data = json.loads(supervisor_state_file(tmp_path).read_text())['preview']
    assert data['percent'][4] < 90 and '左上' in data['tilt']


def test_exposure_selection_and_failure_reasons():
    from nest_clear.preview.exposure import select_exposure
    def row(exposure, corners, p99=200, sat=0):
        return {'exposure_us': exposure, 'corners': corners, 'p99': p99, 'saturated_fraction': sat}
    rows = [row(500, 90, 60), row(1000, 100, 120), row(2000, 99, 205),
            row(4000, 100, 220), row(8000, 110, 255, .02)]
    assert select_exposure(rows, 'calibration', .01)[0]['exposure_us'] == 2000
    assert select_exposure(rows, 'brightness', .01)[0]['exposure_us'] == 4000
    assert '过曝' in select_exposure([row(1, 0, 255, .1)], 'calibration', .01)[1]
    assert '太暗' in select_exposure([row(1, 0, 50), row(2, 0, 90)], 'calibration', .01)[1]
    assert '未检出' in select_exposure([row(1, 0, 150)], 'calibration', .01)[1]


def test_exposure_scan_restore_and_settle():
    from nest_clear.preview.exposure import run_sweep
    class Camera:
        def __init__(self): self.readback = {'exposure_us': 5000.}; self.captures = 0
        def set_exposure(self, x): self.readback['exposure_us'] = x; return x
        def execute(self, _): self.captures += 1; return self.readback['exposure_us'], {}
    camera = Camera(); seen = []
    settings = {'min_us': 500, 'max_us': 8000, 'steps': 5, 'settle_frames': 1}
    def measure(value, _):
        return {'p99': min(255, int(value/20)), 'corners': 50 if value < 8000 else 60,
                'saturated_fraction': .1 if value >= 8000 else 0, 'sharpness': 1}
    result = run_sweep(camera, settings, {}, 'calibration', measure=measure, progress=lambda v: seen.append(v['index']))
    assert result['state'] == 'done' and result['recommendation']['exposure_us'] == pytest.approx(1000)
    assert camera.readback['exposure_us'] == 5000 and camera.captures == 10 and max(seen) == 5
    camera.captures = 0
    result = run_sweep(camera, settings, {}, 'brightness', measure=measure, cancelled=lambda: camera.captures >= 2)
    assert result['state'] == 'cancelled' and camera.readback['exposure_us'] == 5000
    def fail(*_): raise RuntimeError('失败')
    with pytest.raises(RuntimeError): run_sweep(camera, settings, {}, 'brightness', measure=fail)
    assert camera.readback['exposure_us'] == 5000


def test_mock_exposure_and_real_demo():
    from nest_clear.workers.camera import CameraWorker
    from nest_clear.preview.exposure import demo_sweep
    camera = CameraWorker('mock', 'mock', exposure_us=10000, image_size=(320, 240))
    camera.open()
    try:
        camera.set_exposure(5000); a = camera.execute({'stage':'preview'})[0]
        camera.set_exposure(20000); b = camera.execute({'stage':'preview'})[0]
        assert a.mean() < b.mean() and np.max(a) <= 127 and np.count_nonzero(b == 255) > 100
    finally:
        camera.camera.close()
    result = demo_sweep()
    assert result['state'] == 'done' and len(result['rows']) == 10
    assert result['recommendation'] and result['recommendation']['corners'] > 0
    assert any(r['saturated_fraction'] > .01 for r in result['rows'])
    assert any(r['region'] == 'board' for r in result['rows'])


def test_exposure_override_backup_and_config(tmp_path):
    import yaml
    from nest_clear.config import load_config
    from nest_clear.preview.exposure import save_exposure
    config_path = tmp_path / 'settings.yaml'
    config_path.write_text(open('config/nest_clear.yaml', encoding='utf-8').read(), encoding='utf-8')
    path = tmp_path / 'calibration' / 'camera_exposure.yaml'
    assert load_config(config_path, require_tools=False).exposure_source == '配置'
    row = {'exposure_us': 5000., 'p99': 200, 'corners': 40, 'saturated_fraction': 0., 'dark_fraction': 0., 'sharpness': 1.}
    result = {'state': 'done', 'recommendation': row, 'mode': 'calibration', 'original_us': 10000., 'rows': [row]}
    save_exposure(path, result)
    loaded = load_config(config_path, require_tools=False)
    assert loaded.exposure_us == 5000 and loaded.exposure_source.startswith('自测 ')
    result['recommendation'] = {**row, 'exposure_us': 6000.}
    backup = save_exposure(path, result)
    assert yaml.safe_load(backup.read_text())['exposure_us'] == 5000
    assert yaml.safe_load(path.read_text())['exposure_us'] == 6000
    assert load_config(config_path, require_tools=False).exposure_us == 6000
    assert yaml.safe_load(config_path.read_text())['camera'].get('exposure_us') is None


def test_preview_process_sweep_cancel(tmp_path):
    import time
    from dataclasses import replace
    from nest_clear.config import load_config
    from nest_clear.supervisor.runner import Supervisor
    from nest_clear.paths import ui_command_file
    class PLC:
        def read(self, name): return 0
    cfg = load_config('config/nest_clear.yaml')
    settings = dict(cfg.ui_preview, exposure_sweep={'min_us': 500., 'max_us': 30000., 'steps': 10, 'settle_frames': 8})
    cfg = replace(cfg, record_directory=str(tmp_path), camera_sn='mock', camera_image_size=(320, 240),
                  exposure_us=6000., ui_preview=settings, vision=replace(cfg.vision, camera_backend='mock'))
    sup = Supervisor(cfg, PLC())
    command = ui_command_file(tmp_path); command.parent.mkdir(parents=True, exist_ok=True)
    command.write_text(json.dumps({'seq': 1, 'preview': True, 'mode': 'focus'}))
    sup._preview_tick()
    status = tmp_path / 'state' / 'preview_status.json'
    action = tmp_path / 'state' / 'focus_action.json'
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if status.exists() and json.loads(status.read_text()).get('scores'): break
            time.sleep(.03)
        else: pytest.fail('预览未启动')
        action.write_text(json.dumps({'seq': 1, 'action': 'exposure_sweep', 'mode': 'calibration'}))
        while time.monotonic() < deadline:
            data = json.loads(status.read_text())
            if data.get('exposure_sweep', {}).get('state') == 'running': break
            time.sleep(.01)
        else: pytest.fail('扫描未启动')
        action.write_text(json.dumps({'seq': 2, 'action': 'cancel_sweep'}))
        while time.monotonic() < deadline:
            data = json.loads(status.read_text())
            if data.get('exposure_sweep', {}).get('state') == 'cancelled' and data.get('exposure_us') == 6000: break
            time.sleep(.02)
        else: pytest.fail(f'扫描未取消或曝光未恢复: {data}')
        assert data['exposure_sweep']['original_us'] == 6000
    finally:
        sup._stop_preview('test')


def test_brightness_target_scans_actual_mock_frames():
    from nest_clear.preview.exposure import run_sweep
    class Camera:
        def __init__(self): self.readback = {'exposure_us': 1234.}
        def set_exposure(self, x): self.readback['exposure_us'] = x; return x
        def execute(self, _):
            level = min(255, round(self.readback['exposure_us'] * .055))
            return np.full((180, 240), level, np.uint8), {}
    camera = Camera()
    result = run_sweep(camera, {'min_us': 500., 'max_us': 8000., 'steps': 5, 'settle_frames': 1}, {}, 'brightness')
    assert result['recommendation']['exposure_us'] == pytest.approx(4000)
    assert result['recommendation']['p99'] == 220
    assert camera.readback['exposure_us'] == 1234
