from dataclasses import replace
from pathlib import Path
import json

import cv2
import numpy as np
import pytest
import yaml

from nest_clear.camcal import CameraCalibConfig, CameraCalibrationSession
from nest_clear.vision.geometry import load_calibration
from test_camcal_core import CFG, rendered_board

FIELD = Path('/home/shuow/nest-clear-field/20261122/runs_commtest/2026-11-22/190532_4708')
SESSION = FIELD / 'camcal/20261122_190601_235331'


def test_field_tenth_frame():
    if not (SESSION / 's10_raw.png').exists():
        pytest.skip('现场图像不可用')
    record = json.loads((SESSION / 'session_record.json').read_text())
    config = yaml.safe_load(Path('/home/shuow/nest-clear-field/20261122/config/commtest.yaml').read_text())
    cfg = CameraCalibConfig(**config['camera_calibration'])
    nominal = load_calibration('config/commtest_calibration/camera.yaml')
    session = CameraCalibrationSession(cfg, nominal, record['image_size'], record['start_xyz'])
    session._obs = record['observations'][:9]
    session._plan = record['plan']
    events = [json.loads(line) for line in (FIELD / 'events.jsonl').read_text().splitlines()]
    event = next(e for e in events if e.get('event') == 'camcal_step' and e.get('sequence') == 10)
    step = session.add(cv2.imread(str(SESSION / 's10_raw.png')), event['actual_xyz'])
    assert step.kind == 'move', (step.code, step.reason)
    assert step.detail['detected_corners'] == 202
    assert step.detail['predicted_corners'] == 250
    assert step.detail['glare_missing_corners'] <= 2
    assert step.detail['usable_corners'] >= .6 * step.detail['predicted_corners']


def test_glare_removal_and_primary_cause():
    image, nominal = rendered_board()
    detector = cv2.aruco.CharucoDetector(CameraCalibrationSession(CFG, nominal, (640,480), (0,0,110))._board)
    corners, ids, _, _ = detector.detectBoard(image)
    pts = corners.reshape(-1, 2)
    class Detector:
        def __init__(self, keep): self.keep = keep
        def detectBoard(self, gray): return corners[self.keep], ids[self.keep], None, None
    def run(radius):
        frame = image.copy()
        cv2.circle(frame, (320, 240), radius, 255, -1)
        # Keep some points inside the saturated region to exercise exclusion.
        distances = np.linalg.norm(pts - [320, 240], axis=1)
        keep = (distances > radius) | (np.arange(len(ids)) % 5 == 0)
        cfg = replace(CFG, max_saturated_fraction=1, glare_blob_min_px=100)
        session = CameraCalibrationSession(cfg, nominal, (640,480), (0,0,110))
        session._detector = Detector(keep)
        return session.add(frame, (0,0,110))
    small = run(40)
    assert small.kind == 'move'
    assert small.detail['glare_missing_ids'] and small.detail['glare_excluded_ids']
    assert not set(small.detail['glare_excluded_ids']) & set(small.detail.get('missing_ids'))
    large = run(220)
    assert large.code == 'CAMCAL-GLARE' and '亮斑' in large.reason


def test_worker_emits_warning_for_accepted_glare(tmp_path):
    from types import SimpleNamespace
    from nest_clear.workers.camcal import CamcalWorker
    from nest_clear.camcal.session import Step
    class Events:
        archive = None
        directory = tmp_path
        run_directory = None
        def __init__(self): self.rows = []
        def next_capture(self): return 1, 'id'
        def emit(self, event, **kwargs): self.rows.append((event, kwargs))
    class Engine:
        plan = []
        def add(self, image, actual):
            return Step('move', detail={'glare_blobs': [{'bbox': [1, 2, 3, 4]}], 'glare_excluded_corners': 2})
    events = Events()
    ctx = SimpleNamespace(events=events, config=SimpleNamespace(run_mode='test'))
    session = {'shots': 0, 'target': (0, 0, 0), 'engine': Engine()}
    CamcalWorker().execute({'context': ctx, 'session': session, 'captured': np.zeros((4, 4), np.uint8),
                            'actual': (0, 0, 0), 'task_id': 1})
    warning = next(row for event, row in events.rows if event == 'camcal_glare_warning')
    assert warning['image_index'] == 1 and warning['bboxes'] == [[1, 2, 3, 4]]
    assert warning['excluded_corners'] == 2
