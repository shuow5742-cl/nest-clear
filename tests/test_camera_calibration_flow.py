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
        self.request = (0, 0)
        self.actual = (0., 0., 0.)
        self.replies = {}
    def execute(self, payload):
        op = payload['operation']
        if op == 'request': return self.request
        if op == 'actual': return self.actual
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
def rig(tmp_path):
    cfg = load_config('config/nest_clear.yaml')
    cfg = replace(cfg, travel=((-100., 100.),)*3)
    events = EventRecorder(tmp_path, run=True)
    board, plc = WorldBoard(), PLC()
    clock = BTClock(board)
    camera = Camera()
    for w in (plc, camera): clock.attach_worker(w)
    updated = []
    path = tmp_path / 'camera.yaml'
    state = {'save_fail': False}
    def save(result, path, history):
        if state['save_fail']: raise OSError('保存失败')
        path.write_text('placeholder: false\n')
    batch = build_batch(board, cfg, [(0.,0.,0.)], RequestTracker(0), events, calibration='old',
                        camcal_factory=FakeSession, camcal_save=save, calibration_path=path,
                        calibration_updated=updated.append)
    handle = clock.submit(batch)
    def request(task, func):
        plc.request = (task, func)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            clock.tick_once()
            if task in plc.replies: return plc.replies[task]
            assert handle.result is None
            time.sleep(.001)
        raise AssertionError('未应答')
    yield SimpleNamespace(plc=plc, request=request, batch=batch, events=events, updated=updated, path=path, camera=camera, state=state)
    clock.shutdown()
    FakeSession.outcome = 'done'
    FakeSession.target = (1., 0., 0.)


def rows(rig):
    return [json.loads(x) for x in (rig.events.run_directory / 'events.jsonl').read_text().splitlines()]


def test_complete(rig):
    assert rig.request(1, 80) == (80, {'camcal_xyz': (1., 0., 0.)})
    rig.plc.actual = (1., 0., 0.)
    assert rig.request(2, 80)[0] == 81
    assert rig.path.exists() and rig.updated == ['new-calibration']
    assert rig.batch.blackboard.read('mode') == Mode.IDLE
    assert list(rig.events.run_directory.glob('camcal/*/session_record.json'))
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


def test_capture_failure(rig):
    rig.camera.fail = True
    assert rig.request(1, 80)[0] == 82
    assert rig.batch.blackboard.read('mode') == Mode.IDLE
    assert any(r['event'] == 'camcal_failed' and '采图失败' in r['reason'] for r in rows(rig))


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
