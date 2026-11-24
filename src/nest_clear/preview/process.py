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


def run_preview(config, mode="preview"):
    """Standalone host: open and close a camera owned by this process."""
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    sn = config.camera_sn
    if sn == 'auto' and config.vision.camera_backend == 'daheng':
        from nest_clear.devices.camera_select import resolve_camera
        sn = resolve_camera()
    worker = CameraWorker(config.vision.camera_backend, sn, timeout_s=config.capture_timeout_s,
                          exposure_us=config.exposure_us, gain_db=config.gain_db,
                          balance_ratio=config.balance_ratio, image_size=config.camera_image_size)
    try:
        try:
            worker.open()
        except Exception as exc:
            status_path = Path(config.record_directory) / 'state' / 'preview_status.json'
            atomic_write(status_path, json.dumps({'error': str(exc)}, ensure_ascii=False).encode())
            raise
        return run_preview_session(config, mode, worker, stop)
    finally:
        worker.camera.close()


def run_preview_session(config, mode, worker, stop, *, shared=False):
    """Run the same preview renderer with a caller-owned, already open worker."""
    root = Path(config.record_directory)
    status_path = root / 'state' / 'preview_status.json'
    settings = config.ui_preview
    values = {'sharpness': None, 'corners': 0, 'board_offset_mm': None}
    analysis_thread = None
    if mode == 'focus':
        from nest_clear.preview.focus import FocusSession, draw_focus
        focus_session = FocusSession()
    try:
        last_action = json.loads((root / 'state' / 'focus_action.json').read_text(encoding='utf-8')).get('seq')
    except (OSError, ValueError):
        last_action = None
    reset_pending = False
    sweep_result = None
    exposure_source = config.exposure_source
    started = time.monotonic()
    frames = 0
    try:
        while not stop.is_set():
            begin = time.monotonic()
            image, _ = (worker.execute if shared else worker._execute)({'stage': 'preview'})
            captured = time.monotonic()
            if mode == 'focus':
                action_path = root / 'state' / 'focus_action.json'
                try:
                    action = json.loads(action_path.read_text(encoding='utf-8'))
                    if action.get('seq') != last_action:
                        last_action = action.get('seq')
                        if action.get('action') == 'reset':
                            reset_pending = True
                        elif action.get('action') == 'exposure_sweep':
                            from nest_clear.preview.exposure import run_sweep, ExposureRestoreError
                            sweep_result = None
                            values.pop('exposure_apply', None)
                            request_seq = last_action
                            def publish_sweep(result):
                                values['exposure_sweep'] = dict(result, configured_us=config.exposure_us, exposure_source=config.exposure_source)
                                snapshot = dict(values, mode=mode, frames=frames, fps=round(frames / max(time.monotonic()-started, .001), 2), error=None)
                                atomic_write(status_path, json.dumps(snapshot, ensure_ascii=False).encode())
                            def is_cancelled():
                                if stop.is_set(): return True
                                try:
                                    pending = json.loads(action_path.read_text(encoding='utf-8'))
                                    return pending.get('seq') != request_seq and pending.get('action') == 'cancel_sweep'
                                except (OSError, ValueError):
                                    return False
                            try:
                                sweep_result = run_sweep(worker, settings['exposure_sweep'], config.camera_calibration,
                                                         action.get('mode'), progress=publish_sweep, cancelled=is_cancelled)
                            except ExposureRestoreError:
                                raise
                            except Exception as exc:
                                values['exposure_sweep'] = {'state': 'error', 'reason': str(exc), 'rows': []}
                            try:
                                pending = json.loads(action_path.read_text(encoding='utf-8'))
                                if pending.get('seq') != request_seq and pending.get('action') in ('set_exposure', 'restore_exposure'):
                                    from nest_clear.preview.temporary_exposure import change_exposure
                                    values['temporary_exposure'] = dict(change_exposure(worker, pending, config.exposure_us, settings['exposure_sweep'], root, busy=True), request_seq=pending['seq'])
                                last_action = pending.get('seq')
                            except (OSError, ValueError):
                                pass
                        elif action.get('action') in ('set_exposure', 'restore_exposure'):
                            from nest_clear.preview.temporary_exposure import change_exposure
                            result = change_exposure(worker, action, config.exposure_us, settings['exposure_sweep'], root)
                            values['temporary_exposure'] = dict(result, request_seq=last_action)
                            if result['state'] == 'done':
                                exposure_source = '临时' if action['action'] == 'set_exposure' else config.exposure_source
                            values.update(exposure_us=worker.readback.get('exposure_us'), exposure_source=exposure_source)
                        elif action.get('action') == 'white_balance':
                            from nest_clear.preview.white_balance import run_white_balance
                            def publish_wb(result):
                                values['white_balance'] = dict(result, request_seq=last_action)
                                snapshot = dict(values, mode=mode, frames=frames, error=None)
                                atomic_write(status_path, json.dumps(snapshot, ensure_ascii=False).encode())
                            run_white_balance(worker, settings['white_balance'], config.camera_calibration,
                                              config.white_balance_override_path, root, publish_wb)
                            try:
                                pending = json.loads(action_path.read_text(encoding='utf-8'))
                                if pending.get('seq') != last_action and pending.get('action') in ('set_exposure', 'restore_exposure'):
                                    from nest_clear.preview.temporary_exposure import change_exposure
                                    values['temporary_exposure'] = dict(change_exposure(worker, pending, config.exposure_us, settings['exposure_sweep'], root, busy=True), request_seq=pending['seq'])
                                    last_action = pending['seq']
                            except (OSError, ValueError):
                                pass
                        elif action.get('action') == 'apply_exposure':
                            try:
                                from nest_clear.preview.exposure import save_exposure
                                if not config.exposure_override_path:
                                    raise ValueError('未配置曝光覆盖文件路径')
                                backup = save_exposure(config.exposure_override_path, sweep_result or {})
                                values['exposure_apply'] = {'state': 'done', 'message': '已保存，主程序下次启动生效', 'backup': str(backup) if backup else None}
                            except Exception as exc:
                                values['exposure_apply'] = {'state': 'error', 'message': str(exc)}
                except (OSError, ValueError):
                    pass
                if analysis_thread is None or not analysis_thread.is_alive():
                    do_reset = reset_pending
                    reset_pending = False
                    frame_exposure = worker.readback.get('exposure_us')
                    def calculate_focus(frame, frame_exposure):
                        try:
                            if do_reset:
                                focus_session.reset()
                            result = focus_session.update(frame)
                            result.update(white_balance_source=(('自测 ' + values['white_balance']['time']) if values.get('white_balance', {}).get('state') == 'success' else config.white_balance_source), gain_db=config.gain_db, analysis_time=time.time())
                            values.update(result)
                        except Exception:
                            import logging
                            logging.exception('对焦分析失败')
                    analysis_thread = threading.Thread(target=calculate_focus, args=(image.copy(), frame_exposure), daemon=True)
                    analysis_thread.start()
            width = int(settings['width'])
            if image.shape[1] > width:
                image = cv2.resize(image, (width, round(image.shape[0] * width / image.shape[1])), interpolation=cv2.INTER_AREA)
            try:
                parent = json.loads(supervisor_state_file(root).read_text(encoding='utf-8'))
                xyz = parent.get('preview_xyz') if parent.get('plc_connected') else None
            except (OSError, ValueError):
                xyz = None
            if mode != 'focus' and (analysis_thread is None or not analysis_thread.is_alive()):
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
            view = draw_focus(image, dict(values)) if mode == 'focus' else (cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy())
            h, w = view.shape[:2]
            cv2.line(view, (w//2, 0), (w//2, h-1), (255,255,0), 1)
            cv2.line(view, (0, h//2), (w-1, h//2), (255,255,0), 1)
            if mode == 'focus':
                values.update(exposure_us=worker.readback.get('exposure_us'), exposure_source=exposure_source)
            snapshot = dict(values, actual_xyz=xyz)
            lines = [] if mode == 'focus' else [f'XYZ: {xyz if xyz is not None else "--"}',
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
            snapshot.update(mode=mode, frames=frames, fps=round(frames / max(time.monotonic()-started, .001), 2), last_frame_time=time.time(), capture_to_publish_s=round(time.monotonic() - captured, 4), error=None)
            atomic_write(status_path, json.dumps(snapshot, ensure_ascii=False).encode())
            stop.wait(max(0, settings['interval_s'] - (time.monotonic()-begin)))
    except Exception as exc:
        atomic_write(status_path, json.dumps({'error': str(exc)}, ensure_ascii=False).encode())
        if shared:
            raise
