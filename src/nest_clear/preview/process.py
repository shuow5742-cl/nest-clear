"""独立预览进程：相机采集与低频分析不阻塞父程序。"""
import json
import os
import signal
import threading
import time
from pathlib import Path

import cv2

from nest_clear.paths import live_image_file, supervisor_state_file
from nest_clear.preview.analysis import analyze
from nest_clear.workers.camera import CameraWorker


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_bytes(data)
    os.replace(tmp, path)


def run_preview(config):
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    root = Path(config.record_directory)
    status_path = root / 'state' / 'preview_status.json'
    settings = config.ui_preview
    sn = config.camera_sn
    if sn == 'auto':
        from nest_clear.commtest import resolve_camera
        sn = resolve_camera()
    worker = CameraWorker(config.vision.camera_backend, sn, timeout_s=config.capture_timeout_s,
                          exposure_us=config.exposure_us, gain_db=config.gain_db,
                          balance_ratio=config.balance_ratio, image_size=config.camera_image_size)
    values = {'sharpness': None, 'corners': 0, 'board_offset_mm': None}
    analysis_thread = None
    started = time.monotonic()
    frames = 0
    try:
        worker.open()
        while not stop.is_set():
            begin = time.monotonic()
            image, _ = worker._execute({'stage': 'preview'})
            captured = time.monotonic()
            width = int(settings['width'])
            if image.shape[1] > width:
                image = cv2.resize(image, (width, round(image.shape[0] * width / image.shape[1])), interpolation=cv2.INTER_AREA)
            try:
                parent = json.loads(supervisor_state_file(root).read_text(encoding='utf-8'))
                xyz = parent.get('preview_xyz') if parent.get('plc_connected') else None
            except (OSError, ValueError):
                xyz = None
            if analysis_thread is None or not analysis_thread.is_alive():
                def calculate(frame):
                    try:
                        _, result = analyze(frame, config.camera_calibration)
                        values.update(result)
                    except Exception:
                        pass
                if begin - getattr(run_preview, '_last_analysis', 0) >= settings['analysis_interval_s']:
                    run_preview._last_analysis = begin
                    analysis_thread = threading.Thread(target=calculate, args=(image.copy(),), daemon=True)
                    analysis_thread.start()
            view = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
            h, w = view.shape[:2]
            cv2.line(view, (w//2, 0), (w//2, h-1), (255,255,0), 1)
            cv2.line(view, (0, h//2), (w-1, h//2), (255,255,0), 1)
            snapshot = dict(values, actual_xyz=xyz)
            lines = [f'XYZ: {xyz if xyz is not None else "--"}',
                     f'Sharpness: {snapshot.get("sharpness")}  Corners: {snapshot.get("corners")}',
                     f'Board offset mm: {snapshot.get("board_offset_mm")}']
            for i, line in enumerate(lines):
                cv2.putText(view, line, (12, 28 + 28*i), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,0,0), 4)
                cv2.putText(view, line, (12, 28 + 28*i), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 1)
            ok, data = cv2.imencode('.jpg', view, [cv2.IMWRITE_JPEG_QUALITY, int(settings['jpeg_quality'])])
            if not ok:
                raise ValueError('JPEG 编码失败')
            atomic_write(live_image_file(root, 'preview_latest'), data.tobytes())
            frames += 1
            snapshot.update(frames=frames, fps=round(frames / max(time.monotonic()-started, .001), 2), last_frame_time=time.time(), capture_to_publish_s=round(time.monotonic() - captured, 4), error=None)
            atomic_write(status_path, json.dumps(snapshot, ensure_ascii=False).encode())
            stop.wait(max(0, settings['interval_s'] - (time.monotonic()-begin)))
    except Exception as exc:
        atomic_write(status_path, json.dumps({'error': str(exc)}, ensure_ascii=False).encode())
    finally:
        worker.camera.close()
