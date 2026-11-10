from dataclasses import replace
import numpy as np
import pytest
from nest_clear.vision.geometry import Calibration
from nest_clear.vision.hair import HairDetectConfig, HairCandidate
from nest_clear.vision.hair_integration import measure, render
from nest_clear.workers.vision import VisionWorker
from nest_clear.config import load_config

@pytest.fixture
def calibration():
    return Calibration(np.eye(3), np.zeros(5), np.eye(3), np.zeros(3), 1)

def candidate(w=3, h=1, length=3, ratio=.5, dark=.5, x=0):
    return HairCandidate((x+1,.5), None, ((x,0),(x+w,0),(x+w,h),(x,h)), (x,0,x+w,h), (x+w/2,h/2), length, w*h, ratio, dark, None, 'test')

def test_physical_gates(calibration):
    cfg = HairDetectConfig()
    o = measure(candidate(), (0,0,0), calibration, cfg)
    assert o.length == 3 and o.area == 3 and o.reason is None
    assert measure(candidate(length=1), (0,0,0), calibration, cfg).reason == 'short'
    assert measure(candidate(w=1,h=1), (0,0,0), calibration, cfg).reason == 'small'
    assert measure(candidate(w=3,h=.4,length=4), (0,0,0), calibration, cfg).reason == 'thin'
    assert measure(candidate(ratio=.71), (0,0,0), calibration, cfg).reason == 'pale'
    assert measure(candidate(w=2,h=.6,length=2,ratio=.7,dark=.15), (0,0,0), calibration, replace(cfg,min_width_mm=.6)).reason is None

def test_worker_review_and_skip(calibration):
    cfg = load_config('config/nest_clear.yaml').vision
    class Detector:
        def detect(self, image): return [candidate(x=0), candidate(w=4,x=10)]
    worker = VisionWorker(Detector(), calibration, 1, 10, cfg)
    payload = {'image': np.zeros((20,20,3), np.uint8), 'pose': (0,0,0), 'attempted': [], 'pending': None}
    d = worker.execute(payload)
    assert d['eligible'][0].area == 4
    identity = d['eligible'][0].identity
    payload['pending'] = {'identity': identity, 'scan_index': 1, 'target_index': 1, 'tool': 1}
    d = worker.execute(payload)
    assert not d['result']['success'] and d['failed_count'] == 1 and len(d['eligible']) == 1
    payload['pending']['identity'] = (100,100,1)
    assert worker.execute(payload)['result']['success']

def test_render_color(calibration):
    o = measure(candidate(w=10,h=5), (0,0,0), calibration, HairDetectConfig())
    img = render(np.zeros((30,30,3), np.uint8), [o])
    assert tuple(img[0,0]) == (0,255,0)
    o.status = 'failed'
    assert tuple(render(np.zeros((30,30,3), np.uint8), [o])[0,0]) == (0,0,255)

class Board:
    def __init__(self, **data): self.data = data
    def read(self, key, *args): return self.data.get(key, args[0] if args else None)
    def write(self, key, value, *args): self.data[key] = value

class Events:
    def __init__(self): self.rows = []
    def emit(self, event, **kw): self.rows.append((event, kw))

def test_return_to_scan_and_placeholder(calibration):
    from nest_clear.flow.leaves import ReturnToScan, PreparePick
    from nest_clear.guard import GuardError
    cfg = load_config('config/nest_clear.yaml')
    bb = Board(scan=(1,2,3), targets=[measure(candidate(), (0,0,0), calibration, cfg.vision.hair_detector)], tool_index=0, task_id=2, next_index=1, view_picks=0)
    leaf = ReturnToScan(); leaf._blackboard = bb; leaf.perform()
    assert bb.read('reply_func') == 1 and bb.read('reply_coordinates') == {'scan_xyz': (1,2,3)}
    leaf = PreparePick(cfg, True); leaf._blackboard = bb
    with pytest.raises(GuardError, match='占位'):
        leaf.perform()
    assert bb.read('tool_index') == 0 and bb.read('pending') is None

def test_view_limit(calibration):
    from nest_clear.flow.leaves import RecordDetection
    cfg = load_config('config/nest_clear.yaml')
    class Detector:
        def detect(self, image): return [candidate()]
    worker = VisionWorker(Detector(), calibration, 1, 10, cfg.vision)
    d = worker.execute({'image': np.zeros((20,20,3),np.uint8), 'pose': (0,0,0)})
    bb = Board(detection=d, view_picks=cfg.vision.max_picks_per_view, next_index=1, task_id=1)
    events = Events(); leaf = RecordDetection(events, cfg); leaf._blackboard = bb; leaf.perform()
    assert bb.read('targets') == []
    assert any(name == 'warning' for name, _ in events.rows)

def test_report_recognition(tmp_path):
    import json
    from datetime import date
    from nest_clear.record.report import report
    day = date(2026,1,2); directory = tmp_path / day.isoformat(); directory.mkdir()
    (directory/'summary.jsonl').write_text(json.dumps({'tray_id':'t', 'targets':5, 'eligible':3, 'scans':2, 'picks':{'1':2}, 'success':1, 'failed':1, 'rejections':{'short':2}, 'detection_ms':20})+'\n')
    text = report(tmp_path, day)
    assert '识别与挑取' in text and '成功率 50.0%' in text and "'short': 2" in text and '10.0 ms' in text

def test_port_free_tray_sequence(calibration):
    from nest_clear.flow.leaves import PrepareScan, PreparePick, ReturnToScan, RecordDetection
    cfg = load_config('config/nest_clear.yaml')
    cfg = replace(cfg, tools=((0,0,0),(0,0,0)), travel=((-100,100),(-100,100),(-100,100)))
    class Detector:
        calls = 0
        def detect(self, image):
            self.calls += 1
            return [candidate(x=0), candidate(x=10)] if self.calls == 1 else ([candidate(x=10)] if self.calls == 2 else [])
    worker = VisionWorker(Detector(), calibration, 1, 10, cfg.vision)
    events = Events()
    bb = Board(next_index=0, scan=None, view_picks=0, tool_index=0, attempted=[], pending=None, task_id=1, image=np.zeros((30,30,3),np.uint8), actual=(0,0,0))
    def run(leaf):
        leaf._blackboard = bb
        leaf.perform()
    run(PrepareScan([(0,0,0)]))
    assert bb.read('reply_func') == 1
    for number in range(3):
        bb.write('detection', worker.execute({'image': bb.read('image'), 'pose': bb.read('actual'), 'attempted': bb.read('attempted'), 'pending': bb.read('pending')}))
        run(RecordDetection(events, cfg))
        if number < 2:
            run(PreparePick(cfg))
            assert bb.read('reply_func') == 3 and bb.read('reply_tool') == number + 1
            run(ReturnToScan())
            assert bb.read('reply_coordinates') == {'scan_xyz': (0,0,0)}
        else:
            run(PrepareScan([(0,0,0)]))
            assert bb.read('reply_func') == 9
    assert len([e for e, row in events.rows if e == 'pick_result' and row['success']]) == 2


def test_placeholder_guard_returns_999_with_chinese_diagnostic(calibration):
    from autoweaver.motion_policy.nodes.node import Status
    from nest_clear.flow.leaves import PrepareError, PreparePick

    cfg = load_config('config/nest_clear.yaml')
    assert not cfg.vision.allow_placeholder_calibration
    observation = measure(candidate(), (0, 0, 0), calibration, cfg.vision.hair_detector)
    bb = Board(targets=[observation], tool_index=0, task_id=4, next_index=1, request_func=1, raw_path=None)
    pick = PreparePick(cfg, placeholder=True)
    pick._blackboard = bb
    assert pick.on_start() == Status.FAILURE
    events = Events()
    error = PrepareError(events)
    error._blackboard = bb
    error.perform()
    assert bb.read('reply_func') == 999 and bb.read('reply_coordinates') == {}
    reason = bb.read('error')
    assert all(part in reason for part in ('CALIBRATION-PLACEHOLDER', '发生了什么：', '位置：', '排查建议：', '占位'))
    assert events.rows[-1][1]['reason'] == reason


def test_render_scales_labels_and_distinguishes_reappeared(calibration, monkeypatch):
    import cv2
    from nest_clear.vision.hair_integration import COLORS

    calls = []
    real_put_text = cv2.putText
    def capture(image, text, origin, font, scale, color, thickness, line_type):
        calls.append((text, scale, color, thickness))
        return real_put_text(image, text, origin, font, scale, color, thickness, line_type)
    monkeypatch.setattr(cv2, 'putText', capture)
    observation = measure(candidate(w=10, h=5), (0, 0, 0), calibration, HairDetectConfig())
    render(np.zeros((300, 400, 3), np.uint8), [observation])
    small_scale = calls[0][1]
    calls.clear()
    observation.status = 'attempted'
    render(np.zeros((3036, 4024, 3), np.uint8), [observation])
    assert 'REAPPEARED' in calls[0][0]
    assert calls[0][1] > small_scale and calls[1][1] > calls[0][1]
    assert calls[0][2] == COLORS['attempted'] and COLORS['attempted'] != COLORS['rejected']
    assert calls[0][3] > 1


def test_preview_does_not_block_archive(tmp_path, monkeypatch, caplog):
    import logging
    import threading
    from nest_clear.record.events import EventRecorder
    from nest_clear.record.images import ImageArchive

    entered = threading.Event()
    release = threading.Event()
    def stuck_imshow(name, image):
        entered.set()
        assert release.wait(3)
        raise RuntimeError('模拟窗口失败')
    monkeypatch.setenv('DISPLAY', ':test')
    monkeypatch.setattr('nest_clear.record.images.cv2.imshow', stuck_imshow)
    monkeypatch.setattr('nest_clear.record.images.cv2.destroyWindow', lambda name: None)
    events = EventRecorder(tmp_path)
    archive = ImageArchive(events, True, 'png', 3, 30)
    image = np.zeros((20, 20, 3), np.uint8)
    try:
        archive.rendered(image, 1, 1, None, preview=True)
        assert entered.wait(2)
        # 窗口还在等待时，归档队列已经可以独立排空。
        archive.executor.submit(lambda: None).result(timeout=2)
        assert list(tmp_path.glob('*/images/*overlay.png'))
        assert list(tmp_path.glob('*/live/latest.jpg'))
    finally:
        release.set()
        archive.close()
    assert any(record.levelno == logging.WARNING and '模拟窗口失败' in record.message for record in caplog.records)


def test_responsibility_and_ignore(calibration):
    """按扎点分配三视野，边界相等归本位，屏蔽不进入观察结果。"""
    from nest_clear.workers.vision import owns_target
    centers = [(0, 0), (10, 0), (20, 0)]
    assert owns_target((5, 0), (0, 0), centers)
    assert owns_target((5, 0), (10, 0), centers)
    assert not owns_target((11, 0), (0, 0), centers)
    cfg = replace(load_config('config/nest_clear.yaml').vision, ignore_regions_px=((9, 0, 15, 3),))
    class Detector:
        def detect(self, image):
            return [candidate(x=0), candidate(x=10), candidate(x=20)]
    worker = VisionWorker(Detector(), calibration, 1, 10, cfg)
    result = worker.execute({'image': np.zeros((30, 30, 3), np.uint8), 'pose': (0, 0, 0), 'scan': (0, 0, 0), 'scan_path': [(0, 0, 0), (10, 0, 0), (20, 0, 0)], 'attempted': []})
    assert result['ignored_count'] == 1 and result['other_view_count'] == 1
    assert len(result['eligible']) == 1 and result['count'] == 2
    bb = Board(detection=result, attempted=[], view_picks=0, next_index=1, task_id=1)
    leaf = __import__('nest_clear.flow.leaves', fromlist=['RecordDetection']).RecordDetection(Events(), load_config('config/nest_clear.yaml'))
    leaf._blackboard = bb
    leaf.perform()
    assert bb.read('attempted') == []
