"""通过注入会话验证真实行为树，不依赖标定求解器。"""
from dataclasses import replace
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest
from autoweaver.motion_policy.world_board import WorldBoard
from autoweaver.worker.clock import BTClock
from autoweaver.worker import AsyncRequestWorker

from nest_clear.workers.camcal import CamcalWorker
from nest_clear.config import load_config
from nest_clear.flow.batch import build_batch
from nest_clear.flow.context import RequestTracker
from nest_clear.flow.modes import Mode
from nest_clear.record.events import EventRecorder
from nest_clear.record.report import report


class PLC(AsyncRequestWorker):
    name = 'plc'
    def __init__(self):
        super().__init__()
        self.request = (0, 0, None)
        self.actual = (0., 0., 0.)
        self.replies = {}
    def execute(self, payload):
        op = payload['operation']
        if op == 'request': return self.request
        if op == 'actual': return self.actual
        if op == 'actual_probe': return [{'xyz': list(self.actual), 'words': [0]*6} for _ in range(payload.get('count', 1))]
        if op == 'reply': self.replies[payload['task_id']] = (payload['func'], payload['coordinates'])


class Camera(AsyncRequestWorker):
    name = 'camera'
    fail = False
    def execute(self, payload):
        if self.fail: raise RuntimeError('相机采图失败')
        return np.zeros((12, 16, 3), dtype=np.uint8), {}


class FakeSession:
    outcome = 'done'
    target = (1., 0., 0.)
    def __init__(self, cfg, nominal, image_size, start_xyz):
        self.samples = []
        self.start = start_xyz
    def add(self, image, actual_xyz):
        self.samples.append(actual_xyz)
        kw = dict(target_xyz=None, result=None, code=None, reason=None, detail={'count': len(self.samples)}, overlay=None)
        if self.outcome == 'fail':
            return SimpleNamespace(kind='fail', **{**kw, 'code': 'CAMCAL-GLARE', 'reason': '亮斑，请调整光源'})
        if len(self.samples) == 1:
            return SimpleNamespace(kind='move', **{**kw, 'target_xyz': self.target})
        result = SimpleNamespace(calibration='new-calibration', report={'rms_px': 0.1, 'axis_scale': {'x': 1}, 'camera_tilt_deg': 2, 'um_per_px_at_principal': 10})
        return SimpleNamespace(kind='done', **{**kw, 'result': result})
    def session_record(self): return {'samples': self.samples}


@pytest.fixture
def rig(tmp_path, request):
    cfg = load_config('config/nest_clear.yaml')
    cfg = replace(cfg, travel=((-100., 100.),)*3, camera_calibration={**cfg.camera_calibration, 'start_xyz': [0., 0., 0.]})
    events = EventRecorder(tmp_path, run=True)
    board, plc = WorldBoard(), PLC()
    clock = BTClock(board)
    camera = Camera()
    for w in (plc, camera, CamcalWorker()): clock.attach_worker(w)
    updated = []
    path = tmp_path / 'calibration' / 'camera.yaml'
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {'save_fail': False}
    def save(result, path, history):
        if state['save_fail']: raise OSError('保存失败')
        path.write_text('placeholder: false\n')
    batch = build_batch(board, cfg, [(0.,0.,0.)], RequestTracker(0), events, calibration='old',
                        camcal_factory=FakeSession, camcal_save=save, calibration_path=path,
                        calibration_updated=updated.append)
    handle = clock.submit(batch)
    def request(task, func):
        plc.request = (task, func, 2 if func in (100, 110) else None)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            clock.tick_once()
            if task in plc.replies: return plc.replies[task]
            assert handle.result is None
            time.sleep(.001)
        raise AssertionError('未应答')
    yield SimpleNamespace(plc=plc, request=request, batch=batch, events=events, updated=updated, path=path, camera=camera, state=state, clock=clock, cfg=cfg)
    clock.shutdown()
    FakeSession.outcome = 'done'
    FakeSession.target = (1., 0., 0.)


def rows(rig):
    return [json.loads(x) for x in (rig.events.run_directory / 'events.jsonl').read_text().splitlines()]


def test_complete(rig):
    assert rig.request(1, 80) == (80, {'scan_xyz': (1., 0., 0.)})
    rig.plc.actual = (1., 0., 0.)
    assert rig.request(2, 80)[0] == 81
    assert rig.path.exists() and rig.updated == ['new-calibration']
    assert rig.batch.blackboard.read('mode') == Mode.IDLE
    assert list(rig.events.run_directory.glob('camcal/*/session_record.json'))
    raw_files = list(rig.events.run_directory.rglob('*camcal*raw*'))
    assert len(raw_files) == 2
    captures = [r for r in rows(rig) if r['event'] == 'capture' and r.get('kind') == 'camcal']
    assert {rig.events.directory / r['raw_path'] for r in captures} == set(raw_files)
    text = report(rig.events.directory, rig.events.clock().date())
    assert '相机标定：' in text and 'RMS 0.1' in text


def test_glare_failure(rig):
    FakeSession.outcome = 'fail'
    assert rig.request(1, 80)[0] == 82
    assert rig.batch.blackboard.read('mode') == Mode.IDLE
    assert any(r.get('code') == 'CAMCAL-GLARE' and '亮斑' in r.get('reason', '') for r in rows(rig))
    assert '相机标定 82' in report(rig.events.directory, rig.events.clock().date())


def test_wrong_position_and_abort(rig):
    assert rig.request(1, 80)[0] == 80
    assert rig.request(2, 80)[0] == 999
    assert rig.batch.blackboard.read('mode') == Mode.IDLE
    assert rig.request(3, 80)[0] == 80
    assert rig.request(4, 1)[0] == 999
    assert rig.batch.blackboard.read('mode') == Mode.IDLE
    assert any(r['event'] == 'camcal_aborted' for r in rows(rig))


def test_limit_and_picking(rig):
    FakeSession.target = (26., 0., 0.)
    assert rig.request(1, 80)[0] == 82
    assert rig.request(2, 1)[0] == 1
    assert rig.request(3, 80)[0] == 999
    assert rig.batch.blackboard.read('mode') == Mode.PICKING


def test_abort_on_shutdown(rig):
    assert rig.request(1, 80)[0] == 80
    rig.events.abort_camcal('PLC 初始化')
    assert rig.batch.blackboard.read('mode') == Mode.IDLE
    assert list(rig.events.run_directory.glob('camcal/*/session_record.json'))
    assert any(r['event'] == 'camcal_aborted' and '初始化' in r['reason'] for r in rows(rig))


def test_invalid_config(tmp_path):
    import yaml
    from nest_clear.guard import GuardError
    source = yaml.safe_load(open('config/nest_clear.yaml', encoding='utf-8'))
    source['camera_calibration']['min_corner_ratio'] = 1.2
    path = tmp_path / 'bad.yaml'
    path.write_text(yaml.safe_dump(source), encoding='utf-8')
    with pytest.raises(GuardError, match='camera_calibration.min_corner_ratio'):
        load_config(path)


@pytest.mark.parametrize('value', [0, -1, 1.5, True])
def test_invalid_glare_min_lost_corners(tmp_path, value):
    import yaml
    from nest_clear.guard import GuardError
    source = yaml.safe_load(open('config/nest_clear.yaml', encoding='utf-8'))
    source['camera_calibration']['glare_min_lost_corners'] = value
    path = tmp_path / 'bad.yaml'
    path.write_text(yaml.safe_dump(source), encoding='utf-8')
    with pytest.raises(GuardError, match='camera_calibration.glare_min_lost_corners'):
        load_config(path)


def test_capture_failure(rig):
    rig.camera.fail = True
    assert rig.request(1, 80)[0] == 82
    assert rig.batch.blackboard.read('mode') == Mode.IDLE
    assert any(r['event'] == 'camcal_failed' and '采图失败' in r['reason'] for r in rows(rig))
    record = next(rig.events.run_directory.glob('camcal/*/session_record.json'))
    assert json.loads(record.read_text())['observations'] == []
    summary = json.loads((record.parent / 'plc_actual_summary.json').read_text())
    assert len(summary['shots']) == 1
    assert len(json.loads((record.parent / 's01_plc_actual.json').read_text())['before']) == 5


def test_save_failure(rig):
    rig.state['save_fail'] = True
    assert rig.request(1, 80)[0] == 80
    rig.plc.actual = (1., 0., 0.)
    assert rig.request(2, 80)[0] == 82
    assert rig.updated == []
    assert any(r['event'] == 'camcal_failed' and '保存失败' in r['reason'] for r in rows(rig))


def test_vision_replaces_calibration_and_clears_scale_cache():
    from nest_clear.workers.vision import HairDetector, VisionWorker
    from nest_clear.vision.hair import HairDetectConfig
    detector = HairDetector(HairDetectConfig())
    detector.cache[1] = 'old scale'
    detector._logged_keys.add(1)
    worker = VisionWorker(detector, 'old', 1, 1)
    worker.update_calibration('new')
    assert worker.calibration == detector.calibration == 'new'
    assert detector.cache == {} and detector._logged_keys == set()


def test_slow_calibration_runs_outside_tick(rig, monkeypatch, caplog):
    original = FakeSession.add
    def slow(self, image, xyz):
        time.sleep(.12)
        return original(self, image, xyz)
    monkeypatch.setattr(FakeSession, 'add', slow)
    caplog.set_level('WARNING')
    assert rig.request(1, 80)[0] == 80
    rig.plc.actual = (1., 0., 0.)
    assert rig.request(2, 80)[0] == 81
    assert not [r for r in caplog.records if 'slow tick' in r.getMessage().lower()]


def reply_row(rig, task):
    deadline=time.monotonic()+2
    while time.monotonic()<deadline:
        rig.clock.tick_once()
        found=[r for r in rows(rig) if r['event']=='reply' and r['task_id']==task]
        if found: return found[-1]
        time.sleep(.001)
    raise AssertionError('缺少应答事件')

def assert_diagnostic(row, code):
    assert row['func']==code
    assert all(row.get(k) for k in ('reason_code','what','where','hint'))

def test_reply_diagnostics_protocol_arrival_camera_and_82(rig):
    assert rig.request(1, 80)[0]==80
    assert rig.request(2, 80)[0]==999
    arrival=reply_row(rig,2); assert_diagnostic(arrival,999); assert '超差' in arrival['what']
    FakeSession.outcome='fail'
    assert rig.request(3,80)[0]==82
    glare=reply_row(rig,3); assert_diagnostic(glare,82); assert glare['reason_code']=='CAMCAL-GLARE' and '亮斑' in glare['what']
    FakeSession.outcome='done'; rig.camera.fail=True
    assert rig.request(4,80)[0]==82
    capture=reply_row(rig,4); assert_diagnostic(capture,82); assert '采图失败' in capture['what']
    assert rig.request(5,655)[0]==999
    protocol=reply_row(rig,5); assert_diagnostic(protocol,999); assert protocol['reason_code']=='WORK-PROTOCOL'

def test_reply_diagnostic_internal_and_tooltip(rig,monkeypatch):
    def broken(*args): raise RuntimeError('模拟内部异常')
    monkeypatch.setattr(FakeSession,'add',broken)
    assert rig.request(1,80)[0]==82
    row=reply_row(rig,1); assert_diagnostic(row,82); assert '模拟内部异常' in row['what']
    assert rig.request(2,100)[0]==131
    row=reply_row(rig,2); assert_diagnostic(row,131)

def test_internal_999_reply_diagnostic(rig, monkeypatch):
    from nest_clear.flow.camera_calibration import Prepare
    def broken(self): raise RuntimeError('模拟位置检查内部异常')
    monkeypatch.setattr(Prepare, 'perform', broken)
    assert rig.request(1,80)[0] == 999
    row=reply_row(rig,1); assert_diagnostic(row,999)
    assert row['reason_code']=='INTERNAL-ERROR' and '模拟位置检查内部异常' in row['what']


def test_calibration_complete_and_raw(rig):
    import hashlib
    production = __import__('pathlib').Path('config/calibration/examples/camera.yaml')
    before = hashlib.sha256(production.read_bytes()).digest()
    assert rig.request(1, 80)[0] == 80
    rig.plc.actual = (1., 0., 0.)
    assert rig.request(2, 80)[0] == 81
    assert rig.path.exists() and rig.path.parent.name == 'calibration'
    assert hashlib.sha256(production.read_bytes()).digest() == before
    assert len(list(rig.events.run_directory.glob('camcal/*/*.png'))) == 2
    assert list(rig.events.run_directory.glob('camcal/*/session_record.json'))


def test_calibration_raw_and_overlay_are_discoverable(rig, monkeypatch):
    from nest_clear.record.images import ImageArchive
    original = FakeSession.add
    def add_with_overlay(self, image, actual_xyz):
        step = original(self, image, actual_xyz)
        step.overlay = image.copy()
        return step
    monkeypatch.setattr(FakeSession, 'add', add_with_overlay)
    archive = ImageArchive(rig.events, True, 'png', 3, 30)
    rig.events.archive = archive
    try:
        assert rig.request(1, 80)[0] == 80
        archive.executor.submit(lambda: None).result(timeout=5)
        raw = list(rig.events.run_directory.rglob('*camcal*raw*'))
        overlay = list(rig.events.run_directory.rglob('*camcal*ovl*'))
        assert len(raw) == len(overlay) == 1
        assert overlay[0].name == raw[0].name.replace('_raw.png', '_ovl.jpg')
        assert raw[0].is_file() and overlay[0].is_file()
    finally:
        archive.close()
        del rig.events.archive


@pytest.mark.parametrize('start,target', [((0.,0.,0.), (11.,0.,0.))])
def test_camcal_travel(rig, start, target):
    object.__setattr__(rig.cfg, 'travel', ((-10., 10.),)*3)
    rig.plc.actual = start
    FakeSession.target = target
    assert rig.request(1, 80)[0] == 999
    row = reply_row(rig, 1)
    assert row['reason_code']


def test_rendered_real_session(tmp_path):
    from hashlib import sha256
    from nest_clear.vision.geometry import load_calibration
    from sim.charuco_scene import CharucoScene

    cfg = load_config('config/nest_clear.yaml')
    cfg = replace(cfg, camera_calibration={**cfg.camera_calibration, 'start_xyz': [130., 70., 110.]}, travel=((80., 180.), (-10., 150.), (100., 120.)))
    production = __import__('pathlib').Path('config/calibration/examples/camera.yaml')
    before = sha256(production.read_bytes()).digest()
    path = tmp_path / 'calibration' / 'camera.yaml'
    path.parent.mkdir()
    path.write_bytes(production.read_bytes())
    events = EventRecorder(tmp_path / 'runs', run=True)
    board, plc = WorldBoard(), PLC()
    plc.actual = (130., 70., 110.)
    scene = CharucoScene()
    class RenderCamera(AsyncRequestWorker):
        name = 'camera'
        def execute(self, payload):
            x, y, z = plc.actual
            return scene.render((x - 130., y - 70., z)), {}
    clock = BTClock(board)
    for worker in (plc, RenderCamera(), CamcalWorker()): clock.attach_worker(worker)
    calibration = load_calibration(path)
    handle = clock.submit(build_batch(board, cfg, [(130., 70., 0.)], RequestTracker(0), events,
                                     calibration=calibration, calibration_path=path))
    try:
        for task in range(1, 55):
            plc.request = (task, 80, None)
            deadline = time.monotonic() + 120
            while task not in plc.replies and time.monotonic() < deadline:
                clock.tick_once()
                assert handle.result is None
                time.sleep(.001)
            assert task in plc.replies
            code, coords = plc.replies[task]
            if code == 81: break
            assert code == 80, (code, rows(SimpleNamespace(events=events))[-10:])
            plc.actual = coords['scan_xyz']
        assert code == 81 and task == 53
        assert len(list(events.run_directory.glob('camcal/*/*_raw.png'))) == 53
        assert list(events.run_directory.glob('camcal/*/session_record.json'))
        assert list((path.parent / 'history').glob('*.yaml'))
        assert sha256(production.read_bytes()).digest() == before
        done = [r for r in rows(SimpleNamespace(events=events)) if r['event'] == 'camcal_done'][-1]['report']
        assert all(k in done for k in ('K', 'distortion', 'rms_px', 'repeat_px', 'axis_scale', 'um_per_px_at_principal'))
    finally:
        clock.shutdown()


def test_start_unset(rig):
    rig.cfg.camera_calibration['start_xyz'] = None
    assert rig.request(1, 80)[0] == 999
    row = reply_row(rig, 1)
    assert row['reason_code'] == 'CAMCAL-START-UNSET'
    assert 'camera_calibration.start_xyz' in row['hint']
    assert not [r for r in rows(rig) if r['event'] == 'capture']


def test_goto_start_then_capture(rig):
    rig.cfg.camera_calibration['start_xyz'] = [92.651, 88.445, -11.904]
    rig.plc.actual = (-19.59, 88.445, -11.904)
    assert rig.request(1, 80) == (80, {'scan_xyz': (92.651, 88.445, -11.904)})
    assert not [r for r in rows(rig) if r['event'] == 'capture']
    event = next(r for r in rows(rig) if r['event'] == 'camcal_goto_start')
    assert event['actual'][0] == -19.59 and event['target'] == [92.651, 88.445, -11.904]
    rig.plc.actual = (92.651, 88.445, -11.904)
    FakeSession.target = rig.plc.actual
    assert rig.request(2, 80)[0] == 80
    assert len([r for r in rows(rig) if r['event'] == 'capture']) == 1


def test_start_outside_config(tmp_path):
    import yaml
    from nest_clear.guard import GuardError
    source = yaml.safe_load(open('config/nest_clear.yaml', encoding='utf-8'))
    source['camera_calibration']['start_xyz'] = [79, 88, -12]
    path = tmp_path / 'bad.yaml'
    path.write_text(yaml.safe_dump(source), encoding='utf-8')
    with pytest.raises(GuardError, match='camera_calibration.start_xyz'):
        load_config(path)


def test_goto_start_outside_actual_travel(rig):
    object.__setattr__(rig.cfg, 'travel', ((80., 180.), (-10., 150.), (-15., 0.)))
    rig.cfg.camera_calibration['start_xyz'] = [92.651, 88.445, -11.904]
    rig.plc.actual = (-19.59, 88.445, -11.904)
    target = (92.651, 88.445, -11.904)
    assert rig.request(1, 80) == (80, {'scan_xyz': target})
    assert rig.request(2, 80)[0] == 999
    assert rig.batch.blackboard.read('mode') == Mode.IDLE
    assert rig.request(3, 80) == (80, {'scan_xyz': target})
    rig.plc.actual = target
    FakeSession.target = target
    assert rig.request(4, 80)[0] == 80
    assert len([r for r in rows(rig) if r['event'] == 'capture']) == 1


@pytest.mark.parametrize('start', [[0, 0], [0, 0, float('nan')], [0, True, 0], '1,2,3'])
def test_start_invalid_config(tmp_path, start):
    import yaml
    from nest_clear.guard import GuardError
    source = yaml.safe_load(open('config/nest_clear.yaml', encoding='utf-8'))
    source['camera_calibration']['start_xyz'] = start
    path = tmp_path / 'bad.yaml'
    path.write_text(yaml.safe_dump(source), encoding='utf-8')
    with pytest.raises(GuardError, match='camera_calibration.start_xyz'):
        load_config(path)
