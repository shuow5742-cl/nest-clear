import json
import queue
import threading
from types import SimpleNamespace

import cv2
import numpy as np

from nest_clear.paths import live_image_file, main_status_file
from nest_clear.record.live import write_live
from nest_clear.ui.app import App, Picture
from nest_clear.ui.model import camcal_progress
from nest_clear.ui.status import MainStatus


def test_session_progress_survives_requests_and_read_failures(tmp_path):
    status = MainStatus(tmp_path)
    app = SimpleNamespace(files={})
    path = main_status_file(tmp_path)
    status.observe({'event': 'request', 'func': 80})
    status.observe({'event': 'camcal_start'})
    seen = []
    for sequence in range(1, 5):
        status.observe({'event': 'camcal_step', 'sequence': sequence, 'plan': [[1, 2, 3]]})
        status.observe({'event': 'camcal_glare_warning', 'sequence': sequence})
        status.flush()
        seen.append(camcal_progress(App.file_data(app, path))['sequence'])
        status.observe({'event': 'request', 'func': 80})
        status.flush()
        data = App.file_data(app, path)
        seen.append(camcal_progress(data)['sequence'])
        assert data['camcal']['plan'] == [[1, 2, 3]]
        path.write_text('{"camcal":', encoding='utf-8')
        seen.append(camcal_progress(App.file_data(app, path))['sequence'])
        path.unlink()
        seen.append(camcal_progress(App.file_data(app, path))['sequence'])
    assert seen == sorted(seen) and min(seen) == 1


def test_glare_warning_preserves_step_data(tmp_path):
    status = MainStatus(tmp_path)
    status.observe({'event': 'camcal_start'})
    status.observe({'event': 'camcal_step', 'sequence': 3, 'plan': [[1, 2, 3]], 'detail': {'detected_corners': 35}})
    status.observe({'event': 'camcal_glare_warning', 'sequence': 99, 'plan': [], 'detail': {}, 'bboxes': [[1, 2, 3, 4]]})
    status.observe({'event': 'request', 'func': 80})
    camcal = status.data['camcal']
    assert camcal['event'] == 'camcal_step' and camcal['sequence'] == 3
    assert camcal['plan'] == [[1, 2, 3]] and camcal['detail'] == {'detected_corners': 35}
    assert camcal['glare_warning']['bboxes'] == [[1, 2, 3, 4]]


def test_format_pose_numpy():
    from nest_clear.camcal.session import format_pose
    assert format_pose(np.array([105.18527221679688, 79.96, -14.904])) == '(105.185, 79.960, -14.904)'


def test_camcal_picture_keeps_image_on_missing_and_bad_data(tmp_path, monkeypatch):
    path = live_image_file(tmp_path, 'camcal_latest')
    original = object()
    pic = SimpleNamespace(path=path, min_mtime_ns=0, mtime=1, size=(100, 100),
                          image=original, photo=original, busy=False, results=queue.Queue(),
                          keep_on_missing=True, aspect_changed=None,
                          winfo_width=lambda: 100, winfo_height=lambda: 100,
                          configure=lambda **kwargs: (_ for _ in ()).throw(AssertionError(kwargs)))
    Picture.poll(pic)
    assert pic.image is original and pic.photo is original
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'invalid jpeg')
    class ImmediateThread:
        def __init__(self, target, daemon): self.target = target
        def start(self): self.target()
    monkeypatch.setattr(threading, 'Thread', ImmediateThread)
    Picture.poll(pic)
    assert pic.image is original and pic.photo is original


def test_atomic_status_and_live_image_under_concurrent_reads(tmp_path):
    status = MainStatus(tmp_path)
    path = main_status_file(tmp_path)
    image_path = live_image_file(tmp_path, 'camcal_latest')
    write_live(tmp_path, 'camcal_latest', np.zeros((30, 30, 3), dtype=np.uint8))
    errors = []
    def writer():
        for n in range(1, 50):
            status.observe({'event': 'camcal_step', 'sequence': n, 'plan': [[0, 0, 0]]})
            status.flush()
            write_live(tmp_path, 'camcal_latest', np.full((30, 30, 3), n, dtype=np.uint8))
    thread = threading.Thread(target=writer)
    thread.start()
    while thread.is_alive():
        try:
            json.loads(path.read_text(encoding='utf-8'))
            if cv2.imdecode(np.frombuffer(image_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR) is None:
                errors.append('invalid image')
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
    thread.join()
    assert not errors
