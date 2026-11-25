"""Independent background acquisition for baseline preparation."""
from datetime import datetime
import json
import logging
import math
from pathlib import Path
import os
import tempfile
import threading
import time
import cv2
import numpy as np

from nest_clear.devices.calib_cameras import enumerate_capture_devices
from nest_clear.devices.tooltip_camera_store import load_settings, save_measurement, measurement_for_device
from nest_clear.devices.tooltip_exposure import auto_exposure
from nest_clear.devices.v4l2_controls import DEFAULTS as CAMERA_DEFAULTS
from nest_clear.paths import tooltip_preview_file
from nest_clear.tooltip.detector import SilhouetteTipDetector
from nest_clear.tooltip.focus_metrics import focus_metrics, EdgeWidthTracker
from nest_clear.baseline.latest_capture import FrameBroker

logger = logging.getLogger(__name__)
LOW_FREQUENCY_PREVIEW_KEYS = ('on', 'device', 'error', 'exposure_source', 'exposure_error', 'exposure_limits', 'exposure_result', 'exposure_busy')


def analyze_tip(frame, detection, tracker, scale, min_contrast, threshold=127):
    tip = detection.point_uv
    status = '未识别' if tip is None else '粗定位' if detection.coarse else '精定位'
    reason = detection.diagnostic or ('未识别到针尖' if tip is None else '针尖为粗定位' if detection.coarse else '')
    metrics = {'tenengrad': None, 'edge_width_px': None, 'edge_width_um': None, 'contrast': None,
               'source': 'none' if tip is None else 'coarse' if detection.coarse else 'tip'}
    if status == '精定位':
        count, labels, stats, _ = cv2.connectedComponentsWithStats((frame <= threshold).astype(np.uint8), connectivity=8)
        silhouette = labels == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])) if count > 1 else np.zeros(frame.shape, dtype=bool)
        metrics = focus_metrics(frame, tip=tip, um_per_px=scale, silhouette_mask=silhouette)
        if metrics['contrast'] < min_contrast:
            reason = f'对比度不足（{metrics["contrast"]:.1f} < {min_contrast:g}）'
        elif metrics['edge_width_px'] is None or metrics['edge_width_px'] <= 0:
            reason = '针尖附近无有效边缘'
        else:
            reason = ''
    valid = status == '精定位' and not reason
    if not valid:
        metrics['edge_width_px'] = metrics['edge_width_um'] = None
    tracked = tracker.update(metrics['edge_width_px'], tip if valid else None)
    return status, reason, metrics, tracked


def draw_tip_preview(frame, name, detection, ui_config, scale):
    from nest_clear.baseline.session import _draw_labels
    image = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    height, width = frame.shape[:2]
    tip = detection.point_uv
    center = ui_config.get('main_target_uv') or ((width - 1) / 2, (height - 1) / 2)
    if name == 'front':
        x, y = map(round, center)
        cv2.drawMarker(image, (x, y), (255, 0, 255), cv2.MARKER_CROSS, 40, 2)
        radius = round(ui_config.get('main_tol_mm', .1) * 1000 / scale)
        cv2.circle(image, (x, y), max(1, radius), (255, 0, 255), 2)
    labels = []
    if tip is None:
        labels.append(((20, 20), '未识别到针尖：' + (detection.diagnostic or '无结果'), 'Tip not detected: ' + detection.diagnostic, (0, 0, 255), 36))
    else:
        x, y = map(round, tip)
        color = (0, 220, 255) if detection.coarse else (0, 220, 0)
        cv2.drawMarker(image, (x, y), color, cv2.MARKER_CROSS, 48, 4)
        cv2.rectangle(image, (max(0, x-80), max(0, y-80)), (min(width-1, x+80), min(height-1, y+80)), color, 3)
        status = '粗定位' if detection.coarse else '精定位'
        position = f'{status} ({tip[0]:.1f}, {tip[1]:.1f})'
        if name == 'front':
            position += f'  Δu={tip[0]-center[0]:+.1f}（右+） Δv={tip[1]-center[1]:+.1f}（下+）'
        labels.append(((20, 20), position, f'{status} ({tip[0]:.1f}, {tip[1]:.1f})', color, 36))
    return _draw_labels(image, labels)


class Preparation:
    def __init__(self, events, worker=None, config=None, ui_config=None, *,
                 clock=None, monotonic=None, preview_interval=0.2, summary_interval=10.0):
        self.events, self.worker, self.config = events, worker, config
        self.ui_config = ui_config or {}
        self._clock = clock or getattr(events, 'clock', None) or (lambda: datetime.now().astimezone())
        self._monotonic = monotonic or getattr(events, 'monotonic', time.monotonic)
        self.preview_interval = preview_interval
        self.summary_interval = summary_interval
        self.enabled = {'front': False, 'side': False}
        self.state = {name: {'on': False} for name in self.enabled}
        self.stops = {}; self.threads = {}; self.retained = set(); self.retest = {n: threading.Event() for n in self.enabled}
        self.lock = threading.RLock()
        self.busy = False
        self.assignment_thread = None
        self.devices = []
        self.trackers = {n: EdgeWidthTracker(reset_distance=50) for n in self.enabled}
        self.reset_peak = {n: threading.Event() for n in self.enabled}
        self.history = {n: [] for n in self.enabled}
        self._preview_cameras = {}
        self.streams = {}
        self.capture_threads = {}
        self.exposure_requests = {name: None for name in self.enabled}
        self._preview_last_write = None
        self._preview_timer = None
        self._last_summary = {n: None for n in self.enabled}
        self._closed = False
        if worker:
            threading.Thread(target=self._discover, daemon=True, name='tooltip-discovery').start()

    @property
    def active(self):
        return self.busy or any(self.enabled.values()) or any(t.is_alive() for n, t in self.threads.items() if not self.enabled[n])

    def _discover(self):
        try:
            if hasattr(self.worker, 'resolve_roles') and any(
                    camera.config.get('backend') == 'uvc' for camera in self.worker.cameras.values()):
                roles = self.worker.resolve_roles()
                devices = list(roles.values())
            else:
                devices = enumerate_capture_devices()
                roles = {n: self.worker.cameras[n].config.get('device') for n in self.enabled}
            with self.lock:
                self.devices = devices
                for name in self.enabled:
                    self.state[name]['device'] = roles.get(name)
                    if not self.enabled[name]: self.state[name].pop('error', None)
        except Exception as exc:
            if not isinstance(exc, ValueError): logger.exception('识别对刀相机失败')
            with self.lock:
                self.devices = []
                for name in self.enabled:
                    self.state[name]['device'] = None
                    self.state[name]['error'] = str(exc)
        self.publish()

    def publish(self):
        """发布低频对刀相机准备状态（不含每帧高频指标）。"""
        with self.lock:
            state = {
                n: {k: v[k] for k in LOW_FREQUENCY_PREVIEW_KEYS if k in v}
                for n, v in self.state.items()
            }
            devices = list(self.devices)
        self.events.emit('baseline_state', preview=state, detected_cameras=devices)

    def _flush_preview_file(self):
        try:
            with self.lock:
                self._preview_timer = None
                if self._closed:
                    return
                self._write_preview_file(force=True)
        except Exception:
            logger.exception('刷新对刀预览状态文件失败')

    def _write_preview_file(self, *, force=False, now=None):
        """以不超过 5Hz 的频率原子写入高频预览状态文件。"""
        directory = getattr(self.events, 'directory', None)
        if directory is None:
            return
        path = tooltip_preview_file(directory)
        with self.lock:
            if now is None:
                now = self._monotonic()
            elapsed = (now - self._preview_last_write) if self._preview_last_write is not None else math.inf
            if not force and elapsed < self.preview_interval:
                if self._preview_timer is None and not self._closed:
                    delay = max(0.01, self.preview_interval - elapsed)
                    timer = threading.Timer(delay, self._flush_preview_file)
                    timer.daemon = True
                    self._preview_timer = timer
                    timer.start()
                return
            if self._preview_timer is not None:
                self._preview_timer.cancel()
                self._preview_timer = None
            payload = {
                'updated_at': self._clock().isoformat(),
                'cameras': {k: dict(v) for k, v in self._preview_cameras.items()},
            }
            self._preview_last_write = now
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, temp = tempfile.mkstemp(prefix='.tooltip_preview_', suffix='.tmp', dir=path.parent)
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                    json.dump(payload, stream, ensure_ascii=False, allow_nan=False)
                os.replace(temp, path)
            except Exception:
                try:
                    os.unlink(temp)
                except OSError:
                    pass
                raise

    def refresh_existing(self):
        if self.config:
            try:
                from nest_clear.baseline.session import existing_baselines
                self.events.emit('baseline_state', existing=existing_baselines(self.config, self.config.vision.calibration))
            except Exception as exc:
                self.events.emit('warning', reason=f'刷新对刀基准状态失败：{exc}')

    def capture_after(self, name, t_mono, count, timeout):
        """Return count new frames; timestamps mark the start of each read.

        Frames are collected from the latest slot and subsequent reads, not a
        historical queue. Raises TimeoutError unless the full batch arrives.
        """
        with self.lock:
            stream = self.streams.get(name)
        if stream is None:
            raise RuntimeError(f'{name} 对刀相机预览未采集')
        return stream.capture_after(t_mono, count, timeout)

    def exposure_reply(self, name, seq, state, message):
        if name not in self.state:
            return
        with self.lock:
            self.state[name]['exposure_result'] = {'seq': seq, 'state': state, 'message': message}
        self.publish()

    def _change_exposure(self, name, camera, stream, request, path, device, stop):
        seq = request.get('_seq')
        try:
            limits = camera.exposure_limits()
            if request['action'] == 'tooltip_exposure_restore':
                measured = measurement_for_device(load_settings(path), device)
                value = measured.get('exposure') if measured else camera.config.get('camera_controls', {}).get('exposure', CAMERA_DEFAULTS['exposure'])
                source = ('已测（取最接近值）' if measured.get('approximate') else '已测') if measured else '默认'
            else:
                from decimal import Decimal, InvalidOperation
                try:
                    units = Decimal(str(request.get('exposure_ms'))) * 10
                    if not units.is_finite() or units != units.to_integral_value():
                        raise ValueError('曝光须按 0.1 ms 单位输入')
                    value = int(units)
                except (InvalidOperation, ValueError, TypeError):
                    raise ValueError('请输入有效曝光，单位 0.1 ms') from None
                source = '临时'
            if type(value) is not int or not limits['min'] <= value <= limits['max'] or (value - limits['min']) % limits['step']:
                raise ValueError(f"曝光范围 {limits['min']/10:g}–{limits['max']/10:g} ms，步进 {limits['step']/10:g} ms")
            stream.invalidate()
            self.reset_peak[name].set()
            camera.set_exposure(value)
            actual = camera.get_exposure()
            if stop.is_set():
                raise RuntimeError('预览已关闭，曝光操作未确认')
            if actual != value:
                raise ValueError(f'曝光回读不一致：期望 {value}，实际 {actual}')
            with self.lock:
                self.state[name]['exposure_source'] = source
                self.state[name].pop('exposure_error', None)
            self.exposure_reply(name, seq, 'done', f'曝光回读 {actual/10:.1f} ms（{source}）')
        except Exception as exc:
            self.exposure_reply(name, seq, 'error', str(exc))

    def _capture_loop(self, name, stop, stream, camera):
        class ExposureCamera:
            def __init__(self, owner):
                self.owner = owner
                self.config = camera.config
                self.after_seq = 0
                self.after_time = 0.
            def get_exposure(self):
                return camera.get_exposure()
            def exposure_limits(self):
                return limits
            def set_exposure(self, value):
                camera.set_exposure(value)
                self.after_time = time.monotonic()
                with stream.condition:
                    self.after_seq = stream.seq
            def capture(self):
                while True:
                    item = self.owner._capture_frame(camera, stream, stop, preview=False)
                    if item.seq > self.after_seq and item.captured_mono >= self.after_time:
                        self.after_seq = item.seq
                        return item.frame
        try:
            camera.open()
            self.events.emit('tooltip_preview_open', camera=name)
            path = camera.config.get('settings_path', 'config/calibration/tooltip_cameras.yaml')
            device = camera.settings_snapshot().get('device', camera.config.get('device'))
            measured = measurement_for_device(load_settings(path), device)
            try:
                limits = camera.exposure_limits()
            except (AttributeError, RuntimeError, OSError, ValueError):
                limits = None
            with self.lock:
                source = ('已测（取最接近值）' if measured.get('approximate') else '已测') if measured else '默认'
                self.state[name].update(device=device, exposure_limits=limits, exposure_source=source, exposure_busy=not bool(measured))
            self.publish()
            if not measured:
                self.retest[name].set()
            while not stop.is_set():
                if self.retest[name].is_set():
                    with self.lock:
                        self.state[name]['exposure_busy'] = True
                        self.retest[name].clear()
                    stream.invalidate()
                    self.reset_peak[name].set()
                    with self.lock:
                        previous_source = self.state[name].get('exposure_source', '默认')
                        self.state[name]['exposure_source'] = '正在测曝光…'; self.state[name].pop('exposure_error', None)
                    self.publish()
                    try:
                        initial = camera.get_exposure()
                        try:
                            options = dict(getattr(self.config.tool_calibration, 'auto_exposure', {}) if self.config else {})
                            if not measured:
                                options['initial_exposure'] = max(1, limits['min'] if limits else 1)
                            result = auto_exposure(ExposureCamera(self), **options)
                        except Exception:
                            camera.set_exposure(initial); raise
                        camera.exposure_status = '已测曝光'
                        if hasattr(camera, 'actual_settings'): camera.actual_settings['exposure_status'] = '已测曝光'
                        save_measurement(path, name, camera.settings_snapshot(), result)
                        with self.lock:
                            self.state[name]['exposure_source'] = '已测（取最接近值）' if result.get('approximate') else '已测'
                            measured = measurement_for_device(load_settings(path), device)
                        self.events.emit('tooltip_exposure_measured', camera=name, device=device, exposure=result['exposure'])
                        self.publish()
                        self.refresh_existing()
                    except Exception as exc:
                        if stop.is_set():
                            break
                        with self.lock:
                            self.state[name]['exposure_source'] = previous_source; self.state[name]['exposure_error'] = str(exc)
                        self.events.emit('tooltip_exposure_failed', camera=name, reason=str(exc))
                        self.publish()
                    finally:
                        self.reset_peak[name].set()
                        with self.lock:
                            self.state[name]['exposure_busy'] = False
                        self.publish()
                with self.lock:
                    request = self.exposure_requests[name]
                    self.exposure_requests[name] = None
                if request:
                    if stop.is_set():
                        self.exposure_reply(name, request.get('_seq'), 'error', '预览已关闭')
                    else:
                        self._change_exposure(name, camera, stream, request, path, device, stop)
                if not stop.is_set():
                    self._capture_frame(camera, stream, stop)
        except Exception as exc:
            if not stop.is_set():
                stream.finish(exc)
        finally:
            with self.lock:
                request = self.exposure_requests[name]
                self.exposure_requests[name] = None
                self.state[name]['exposure_busy'] = False
            if request:
                self.exposure_reply(name, request.get('_seq'), 'error', '预览已关闭')
            stream.finish()
            if stop not in self.retained:
                camera.close()

    def _capture_frame(self, camera, stream, stop, preview=True):
        if stop.is_set():
            raise RuntimeError('预览采集已停止')
        captured_mono = time.monotonic()
        captured_at = self._clock().isoformat()
        frame = camera.capture()
        if stop.is_set():
            raise RuntimeError('预览采集已停止')
        try:
            exposure = camera.get_exposure() / 10
        except Exception:
            exposure = None
        return stream.publish(frame, captured_mono, captured_at, exposure, preview=preview)

    def command(self, name, params):
        if name in ('tooltip_assign', 'tooltip_swap'):
            raise ValueError('对刀相机已按插口固定，不可在界面修改')
        camera = params.get('camera')
        if camera not in self.enabled:
            raise ValueError('请选择主或辅助对刀相机')
        if name == 'tooltip_preview':
            on = params.get('on')
            if type(on) is not bool: raise ValueError('预览开关无效')
            if on and not self.worker: raise ValueError('对刀相机未启用')
            if self.busy and not params.get('_internal'): raise ValueError('相机分配正在更新')
            retry = on and self.enabled[camera] and bool(self.state[camera].get('error'))
            if self.enabled[camera] == on and not retry: return
            if retry and camera in self.stops: self.stops[camera].set()
            with self.lock:
                self.enabled[camera] = on
                self.state[camera].update(on=on, error=None)
                self.state[camera]['exposure_busy'] = on
                self.state[camera].pop('exposure_limits', None)
                for key in ('tenengrad','peak','percent','edge_width_um','edge_width_px','contrast','source','tip_uv','history','exposure_error','exposure_ms','fps'):
                    self.state[camera].pop(key, None)
                self._preview_cameras.pop(camera, None)
            if not on:
                self._write_preview_file(force=True)
            if on and self.worker:
                threading.Thread(target=self._discover, daemon=True, name='tooltip-discovery').start()
                stop = threading.Event(); previous = self.threads.get(camera)
                self.stops[camera] = stop
                thread = threading.Thread(target=self._run, args=(camera, stop, previous), daemon=True, name='tooltip-'+camera)
                self.threads[camera] = thread; thread.start()
            elif not on and camera in self.stops:
                self.stops[camera].set()
                if not any(self.enabled.values()) and not params.get('_internal'): self._record_finished()
            self.events.emit('baseline_state', phase='prepare' if self.active else 'idle',
                             message='请调焦后开始对刀基准建立' if self.active else '请开始对刀基准建立')
            if not on: self.events.emit('tooltip_preview_close', camera=camera)
        elif name == 'tooltip_exposure_retest':
            self.retest[camera].set()
        elif name in ('tooltip_exposure_set', 'tooltip_exposure_restore'):
            with self.lock:
                if not self.enabled[camera] or camera not in self.streams:
                    error = '预览未打开'
                elif self.state[camera].get('exposure_busy') or self.retest[camera].is_set() or self.exposure_requests[camera] is not None:
                    error = '曝光操作进行中，请稍后重试'
                else:
                    error = None
                    self.exposure_requests[camera] = {**params, 'action': name}
            if error:
                self.exposure_reply(camera, params.get('_seq'), 'error', error)
        elif name == 'tooltip_peak_reset':
            self.reset_peak[camera].set()
        else:
            raise ValueError('未知对刀相机命令')
        self.publish()

    def _run(self, name, stop, previous):
        if previous: previous.join()
        if stop.is_set(): return
        camera = self.worker.cameras[name]
        self.trackers[name].reset(); self.history[name] = []
        self._last_summary[name] = None
        target = Path(self.events.directory) / 'live' / f'baseline_{name}.jpg'
        target.unlink(missing_ok=True)
        stream = FrameBroker()
        with self.lock:
            self.streams[name] = stream
        capture_thread = threading.Thread(target=self._capture_loop, args=(name, stop, stream, camera), daemon=True, name='tooltip-capture-'+name)
        self.capture_threads[name] = capture_thread
        capture_thread.start()
        try:
            last = None
            last_exposure = None
            last_seq = 0
            while not stop.is_set():
                if last is not None:
                    delay = .2 - (self._monotonic() - last)
                    if delay > 0 and stop.wait(delay):
                        break
                item = stream.next_latest(last_seq, stop)
                if item is None:
                    break
                frame, exposure, last_seq = item.frame, item.exposure_ms, item.seq
                processing_start = time.monotonic()
                if exposure is not None and last_exposure is not None and exposure != last_exposure:
                    self.trackers[name].reset()
                    self.history[name] = []
                if exposure is not None:
                    last_exposure = exposure
                now = self._monotonic()
                fps = round(1 / (now - last), 1) if last is not None and now > last else None
                last = now
                detector_cfg = self.config.tool_calibration.detection[name] if self.config else {}
                detection = SilhouetteTipDetector(**detector_cfg).detect(frame)
                tip = detection.point_uv
                target_uv = self.ui_config.get('main_target_uv') or ((frame.shape[1]-1)/2, (frame.shape[0]-1)/2)
                delta_uv = [tip[0] - target_uv[0], tip[1] - target_uv[1]] if name == 'front' and tip is not None else None
                scale = self.ui_config.get('main_um_per_px' if name == 'front' else 'aux_um_per_px', 15.2)
                if self.reset_peak[name].is_set():
                    self.trackers[name].reset(); self.reset_peak[name].clear()
                    self.history[name] = []
                status, reason, metrics, tracked = analyze_tip(frame, detection, self.trackers[name], scale, self.ui_config.get('min_focus_contrast', 40), detector_cfg.get('threshold', 127))
                self.history[name] = [(t,v) for t,v in self.history[name] if now-t <= 30]
                if tracked['percent'] is not None:
                    self.history[name].append((now, tracked['percent']))
                with self.lock:
                    self.state[name]['peak_edge_width_um'] = tracked['best'] * scale if tracked['best'] is not None else None
                    self.state[name].update(tenengrad=metrics['tenengrad'], peak=tracked['best'], percent=tracked['percent'],
                        edge_width_um=metrics['edge_width_um'], edge_width_px=metrics['edge_width_px'], contrast=metrics['contrast'],
                        source=metrics['source'], tip_uv=list(tip) if tip is not None else None,
                        detection_status=status, detection_reason=detection.diagnostic, focus_reason=reason,
                        history=[v for _,v in self.history[name]], fps=fps)
                image = draw_tip_preview(frame, name, detection, self.ui_config, scale)
                image = cv2.resize(image, (960, max(1, round(frame.shape[0]*960/frame.shape[1]))), interpolation=cv2.INTER_AREA)
                ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
                with stream.condition:
                    if item.generation != stream.generation:
                        continue
                if ok:
                    target = Path(self.events.directory) / 'live' / f'baseline_{name}.jpg'
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_suffix('.preview.tmp'); temporary.write_bytes(encoded.tobytes()); os.replace(temporary, target)
                with self.lock:
                    self.state[name].update(exposure_ms=exposure)
                    published = time.monotonic()
                    self._preview_cameras[name] = {
                        'tenengrad': metrics['tenengrad'],
                        'peak': tracked['best'],
                        'percent': tracked['percent'],
                        'edge_width_um': metrics['edge_width_um'],
                        'edge_width_px': metrics['edge_width_px'],
                        'peak_edge_width_um': self.state[name].get('peak_edge_width_um'),
                        'contrast': metrics['contrast'],
                        'source': metrics['source'],
                        'tip_uv': list(tip) if tip is not None else None,
                        'detection_status': status,
                        'detection_reason': detection.diagnostic,
                        'focus_reason': reason,
                        'delta_uv': delta_uv,
                        'history': list(self.state[name]['history']),
                        'fps': fps,
                        'frame_seq': item.seq,
                        'captured_mono': item.captured_mono,
                        'captured_at': item.captured_at,
                        'frame_age_ms': max(0., (published - item.captured_mono) * 1000),
                        'processing_ms': max(0., (published - processing_start) * 1000),
                        'exposure_ms': exposure,
                        'updated_at': self._clock().isoformat(),
                    }
                self._write_preview_file(now=now)
                last_sum = self._last_summary[name]
                if last_sum is None or now - last_sum >= self.summary_interval:
                    self._last_summary[name] = now
                    self.events.emit(
                        'tooltip_focus_summary',
                        camera=name,
                        current=metrics['edge_width_px'],
                        peak=tracked['best'],
                        percent=tracked['percent'],
                        edge_width_um=metrics['edge_width_um'],
                        source=metrics['source'],
                    )
        except Exception as exc:
            detail = str(exc)
            display = next((line.removeprefix('发生了什么：') for line in detail.splitlines() if line.startswith('发生了什么：')), detail)
            with self.lock:
                self.state[name]['error'] = display
                self._preview_cameras.pop(name, None)
            self._write_preview_file(force=True)
            self.events.emit('tooltip_preview_error', camera=name, reason=detail); self.publish()
        finally:
            stop.set()
            capture_thread.join()
            with self.lock:
                if self.streams.get(name) is stream:
                    self.streams.pop(name, None)
            if stop not in self.retained:
                target.unlink(missing_ok=True)

    def _record_finished(self):
        self.events.emit('tooltip_preparation_finished', cameras={name: {
            key: self.state[name].get(key) for key in ('tenengrad', 'peak', 'percent', 'edge_width_um', 'edge_width_px', 'peak_edge_width_um')
        } for name in self.enabled})

    def stop_previews(self, cameras):
        """Synchronously stop selected acquisitions and release their devices."""
        assignment = self.assignment_thread
        if assignment and assignment is not threading.current_thread():
            assignment.join()
        stopped = []
        for name in cameras:
            if name not in self.enabled:
                continue
            if self.enabled[name]:
                self.command('tooltip_preview', {'camera': name, 'on': False, '_internal': True})
                stopped.append(name)
            thread = self.threads.get(name)
            if thread and thread is not threading.current_thread() and thread.ident is not None:
                thread.join()
        return stopped

    def handoff(self):
        """Stop acquisition without closing devices; joining occurs on session thread."""
        if any(self.enabled.values()): self._record_finished()
        with self.lock:
            for name in self.enabled:
                if self.enabled[name]:
                    self.retained.add(self.stops[name])
                    self.enabled[name] = False
                    self.state[name]['on'] = False
                    self._preview_cameras.pop(name, None)
                    self.stops[name].set()
        self._write_preview_file(force=True)
        self.publish()

    def await_handoff(self):
        for thread in list(self.threads.values()):
            thread.join()
        self.retained.clear()

    def close(self):
        with self.lock:
            if self._preview_timer is not None:
                self._preview_timer.cancel()
                self._preview_timer = None
                try:
                    self._write_preview_file(force=True)
                except Exception:
                    logger.exception('关闭时刷新对刀预览状态文件失败')
            self._closed = True
            stops = list(self.stops.values())
        for stop in stops: stop.set()
