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

from nest_clear.devices.calib_cameras import enumerate_capture_devices
from nest_clear.devices.tooltip_camera_store import load_settings, save_measurement, measurement_for_device
from nest_clear.devices.tooltip_exposure import auto_exposure
from nest_clear.paths import tooltip_preview_file
from nest_clear.tooltip.detector import SilhouetteTipDetector
from nest_clear.tooltip.focus_metrics import focus_metrics, PeakTracker

logger = logging.getLogger(__name__)
LOW_FREQUENCY_PREVIEW_KEYS = ('on', 'device', 'error', 'exposure_source', 'exposure_error')


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
        self.trackers = {n: PeakTracker(reset_distance=50) for n in self.enabled}
        self.reset_peak = {n: threading.Event() for n in self.enabled}
        self.history = {n: [] for n in self.enabled}
        self._preview_cameras = {}
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
        try:
            camera.open()
            self.events.emit('tooltip_preview_open', camera=name)
            path = camera.config.get('settings_path', 'config/calibration/tooltip_cameras.yaml')
            device = camera.settings_snapshot().get('device', camera.config.get('device'))
            measured = measurement_for_device(load_settings(path), device)
            with self.lock:
                self.state[name].update(device=device, exposure_source=('已测 '+str(measured.get('time', '')) if measured else '未测'))
            self.publish()
            if not measured: self.retest[name].set()
            last = None
            while not stop.is_set():
                if self.retest[name].is_set():
                    self.retest[name].clear()
                    with self.lock:
                        self.state[name]['exposure_source'] = '正在测曝光…'; self.state[name].pop('exposure_error', None)
                    self.publish()
                    try:
                        initial = camera.get_exposure()
                        try:
                            result = auto_exposure(camera, **(getattr(self.config.tool_calibration, 'auto_exposure', {}) if self.config else {}))
                        except Exception:
                            camera.set_exposure(initial); raise
                        camera.exposure_status = '已测曝光'
                        if hasattr(camera, 'actual_settings'): camera.actual_settings['exposure_status'] = '已测曝光'
                        save_measurement(path, name, camera.settings_snapshot(), result)
                        with self.lock:
                            self.state[name]['exposure_source'] = '已测 ' + str(load_settings(path)['cameras'][name]['time'])
                        self.events.emit('tooltip_exposure_measured', camera=name, device=device, exposure=result['exposure'])
                        self.publish()
                        self.refresh_existing()
                    except Exception as exc:
                        with self.lock:
                            self.state[name]['exposure_source'] = '未测'; self.state[name]['exposure_error'] = str(exc)
                        self.events.emit('tooltip_exposure_failed', camera=name, reason=str(exc))
                        self.publish()
                frame = camera.capture()
                now = self._monotonic()
                if last is not None and now - last < .2:
                    stop.wait(.2 - (now - last))
                    continue
                fps = round(1 / (now - last), 1) if last is not None and now > last else None
                last = now
                detector_cfg = self.config.tool_calibration.detection[name] if self.config else {}
                tip = SilhouetteTipDetector(**detector_cfg).detect(frame).point_uv
                scale = self.ui_config.get('main_um_per_px' if name == 'front' else 'aux_um_per_px', 15.2)
                metrics = focus_metrics(frame, tip=tip, um_per_px=scale)
                if self.reset_peak[name].is_set():
                    self.trackers[name].reset(); self.reset_peak[name].clear()
                tracked = self.trackers[name].update(metrics['tenengrad'], tip)
                self.history[name] = [(t,v) for t,v in self.history[name] if now-t <= 30]
                self.history[name].append((now, tracked['percent']))
                with self.lock:
                    if tracked['current'] == tracked['peak']:
                        self.state[name]['peak_edge_width_um'] = metrics['edge_width_um']
                    self.state[name].update(tenengrad=metrics['tenengrad'], peak=tracked['peak'], percent=tracked['percent'],
                        edge_width_um=metrics['edge_width_um'], edge_width_px=metrics['edge_width_px'], contrast=metrics['contrast'],
                        source=metrics['source'], tip_uv=list(tip) if tip is not None else None,
                        history=[v for _,v in self.history[name]], fps=fps)
                image = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                h, w = frame.shape[:2]
                if name == 'front':
                    center = self.ui_config.get('main_target_uv') or ((w-1)/2, (h-1)/2)
                    x,y = map(round, center)
                    cv2.drawMarker(image, (x,y), (255,0,255), cv2.MARKER_CROSS, 40, 1)
                    radius = round(self.ui_config.get('main_tol_mm', .1)*1000/scale)
                    cv2.circle(image, (x,y), max(1,radius), (255,0,255), 1)
                if tip is not None: cv2.circle(image, tuple(map(round, tip)), 5, (0,200,255), 2)
                image = cv2.resize(image, (960, max(1, round(frame.shape[0]*960/frame.shape[1]))), interpolation=cv2.INTER_AREA)
                ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    target = Path(self.events.directory) / 'live' / f'baseline_{name}.jpg'
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_suffix('.preview.tmp'); temporary.write_bytes(encoded.tobytes()); os.replace(temporary, target)
                try: exposure = camera.get_exposure() / 10
                except Exception: exposure = None
                with self.lock:
                    self.state[name].update(exposure_ms=exposure)
                    self._preview_cameras[name] = {
                        'tenengrad': metrics['tenengrad'],
                        'peak': tracked['peak'],
                        'percent': tracked['percent'],
                        'edge_width_um': metrics['edge_width_um'],
                        'edge_width_px': metrics['edge_width_px'],
                        'peak_edge_width_um': self.state[name].get('peak_edge_width_um'),
                        'contrast': metrics['contrast'],
                        'source': metrics['source'],
                        'tip_uv': list(tip) if tip is not None else None,
                        'history': list(self.state[name]['history']),
                        'fps': fps,
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
                        current=metrics['tenengrad'],
                        peak=tracked['peak'],
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
            if stop not in self.retained:
                camera.close()
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
