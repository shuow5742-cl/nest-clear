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
from nest_clear.devices.tooltip_exposure import auto_exposure, measure
from nest_clear.devices.v4l2_controls import DEFAULTS as CAMERA_DEFAULTS
from nest_clear.paths import tooltip_preview_file
from nest_clear.tooltip.detector import SilhouetteTipDetector
from nest_clear.tooltip.factory import make_tip_detector
from nest_clear.tooltip.focus_metrics import focus_metrics, EdgeWidthTracker, BandGradientTracker
from nest_clear.tooltip.trident import TridentResult
from nest_clear.tooltip.exposure_scan import scan_exposure
from nest_clear.baseline.latest_capture import FrameBroker
from nest_clear.baseline.focus_rules import (
    CAMERA_AXES,
    EDGE_MARGIN_MESSAGE,
    STAGE_ASCENDING_TEXT,
    evaluate_camera_readiness,
    resolve_focus_config,
)

logger = logging.getLogger(__name__)
LOW_FREQUENCY_PREVIEW_KEYS = ('on', 'device', 'error', 'exposure_source', 'exposure_error', 'exposure_limits', 'exposure_result', 'exposure_busy', 'exposure_saved', 'exposure_progress', 'exposure_scan_rows')


def exposure_source(measured):
    return ('已存（手动）' if measured.get('source') == 'manual' else '已存（自动测）') if measured else '默认'


def make_tip_tracker(detector_cfg, ui_config=None):
    """按检测方法构造对应的清晰度峰值与越峰跟踪器。"""
    method = (detector_cfg or {}).get('method', 'trident')
    focus_cfg = resolve_focus_config(ui_config, ui_config)
    if method == 'trident':
        return BandGradientTracker(
            smooth_frames=int(focus_cfg['focus_smooth_frames']),
            pass_percent=float(focus_cfg['focus_pass_percent']),
        )
    return EdgeWidthTracker(reset_distance=50, pass_percent=float(focus_cfg['focus_pass_percent']))


class ReadinessTracker:
    """按帧跟踪单路对刀相机的连续达标帧数与就绪结论。"""

    def __init__(self) -> None:
        self.consecutive = 0

    def reset(self) -> None:
        self.consecutive = 0

    def update(self, name, payload_fields, ui_config=None, *, manual_confirmed=False):
        probe = evaluate_camera_readiness(
            name,
            {**payload_fields, 'consecutive_ready': None},
            config=ui_config,
            manual_confirmed=manual_confirmed,
        )
        self.consecutive = (self.consecutive + 1) if probe['frame_ok'] else 0
        return evaluate_camera_readiness(
            name,
            {**payload_fields, 'consecutive_ready': self.consecutive},
            config=ui_config,
            manual_confirmed=manual_confirmed,
        )


def write_tooltip_preview_file(directory, cameras, updated_at):
    """原子写入高频对刀实时清晰度状态文件 runs/state/tooltip_preview.json。"""
    path = tooltip_preview_file(directory)
    payload = {
        'updated_at': updated_at,
        'cameras': {k: dict(v) for k, v in cameras.items()},
    }
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


def build_preview_camera_payload(
    *,
    metrics,
    tracked,
    status,
    reason,
    detection,
    tip,
    delta_uv,
    history,
    fps,
    scale,
    detector_cfg,
    ui_config,
    updated_at,
    exposure_ms=None,
    frame_seq=None,
    captured_mono=None,
    captured_at=None,
    frame_age_ms=None,
    processing_ms=None,
    image_size=None,
    readiness=None,
    manual_confirmed=False,
    consecutive_ready=None,
):
    """构造单路写入 tooltip_preview.json 的标准化实时清晰度字典。"""
    method = (detector_cfg or {}).get('method', 'trident')
    axis_rms_limit = detector_cfg.get('max_axis_rms_px', 0.7) if method == 'trident' else None
    focus_cfg = resolve_focus_config(ui_config, ui_config)
    peak_edge_um = (
        tracked['best'] * scale
        if metrics.get('source') != 'band_gradient_p995' and tracked.get('best') is not None
        else None
    )
    sample_count = int(tracked.get('sample_count', 0) or 0)
    peak_valid = (
        bool(readiness['peak_valid'])
        if readiness is not None
        else (tracked.get('best') is not None and sample_count >= int(focus_cfg['peak_min_frames']))
    )
    consec = (
        int(readiness['consecutive_ready'])
        if readiness is not None and readiness.get('consecutive_ready') is not None
        else int(consecutive_ready or 0)
    )
    return {
        'tenengrad': metrics['tenengrad'],
        'raw_band_gradient': metrics.get('raw_band_gradient'),
        'band_gradient': metrics.get('band_gradient'),
        'axis_rms_px': metrics.get('axis_rms_px'),
        'axis_rms_limit': axis_rms_limit,
        'focus_smooth_frames': focus_cfg['focus_smooth_frames'],
        'focus_pass_percent': focus_cfg['focus_pass_percent'],
        'focus_good_px': focus_cfg['focus_good_px'],
        'focus_warn_px': focus_cfg['focus_warn_px'],
        'focus_max_px': focus_cfg['focus_max_px'],
        'focus_ready_percent': focus_cfg['focus_ready_percent'],
        'focus_ready_frames': focus_cfg['focus_ready_frames'],
        'peak_min_frames': focus_cfg['peak_min_frames'],
        'edge_margin_px': focus_cfg['edge_margin_px'],
        'peak': tracked['best'],
        'percent': tracked['percent'],
        'sample_count': sample_count,
        'peak_valid': peak_valid,
        'peak_confirmed': bool(readiness['peak_confirmed']) if readiness is not None else bool(tracked.get('peak_confirmed')),
        'stage': readiness['stage'] if readiness is not None else 'ascending',
        'stage_text': readiness['stage_text'] if readiness is not None else STAGE_ASCENDING_TEXT,
        'wide_edge_note': readiness['wide_edge_note'] if readiness is not None else '',
        'consecutive_ready': consec,
        'edge_near': bool(readiness['edge_near']) if readiness is not None else False,
        'manual_confirmed': bool(manual_confirmed),
        'ready': bool(readiness['ready']) if readiness is not None else False,
        'ready_reason': readiness['reason'] if readiness is not None else '',
        'edge_width_um': metrics['edge_width_um'],
        'edge_width_px': metrics['edge_width_px'],
        'peak_edge_width_um': peak_edge_um,
        'contrast': metrics['contrast'],
        'source': metrics['source'],
        'tip_uv': list(tip) if tip is not None else None,
        'image_size': list(image_size) if image_size is not None else None,
        'detection_status': status,
        'detection_reason': getattr(detection, 'diagnostic', ''),
        'focus_reason': reason,
        'delta_uv': delta_uv,
        'history': list(history),
        'fps': fps,
        'frame_seq': frame_seq,
        'captured_mono': captured_mono,
        'captured_at': captured_at,
        'frame_age_ms': frame_age_ms,
        'processing_ms': processing_ms,
        'exposure_ms': exposure_ms,
        'updated_at': updated_at,
    }


def analyze_tip(frame, detection, tracker, scale, min_contrast, threshold=127, *, manual_confirmed=False, manual_tip=None):
    tip = manual_tip if (manual_confirmed and manual_tip is not None) else detection.point_uv
    if isinstance(detection, TridentResult):
        status = detection.status
        diag = detection.diagnostic or ''
        has_distortion = '工具在动或画面畸变' in diag
        reason = diag if status != '精定位' else ''
        can_track = (status == '精定位') or (manual_confirmed and not has_distortion)
        value = (detection.sharpness or {}).get('band_gradient_p995') if can_track else None
        tracked = tracker.update(value, tip if value is not None else None)
        edge_px = detection.edge_width_px
        edge_um = edge_px * scale if edge_px is not None else None
        metrics = {'tenengrad': None, 'edge_width_px': edge_px, 'edge_width_um': edge_um,
                   'contrast': detection.tip_contrast, 'source': 'band_gradient_p995',
                   'raw_band_gradient': tracked.get('raw'),
                   'band_gradient': tracked['current'], 'axis_rms_px': detection.axis_rms_px}
        return status, reason, metrics, tracked
    status = '未识别' if detection.point_uv is None else '粗定位' if detection.coarse else '精定位'
    reason = detection.diagnostic or ('未识别到针尖' if detection.point_uv is None else '针尖为粗定位' if detection.coarse else '')
    metrics = {'tenengrad': None, 'edge_width_px': None, 'edge_width_um': None, 'contrast': None,
               'source': 'none' if tip is None else 'coarse' if detection.coarse else 'tip'}
    can_measure = (status == '精定位') or (manual_confirmed and tip is not None)
    if can_measure:
        count, labels, stats, _ = cv2.connectedComponentsWithStats((frame <= threshold).astype(np.uint8), connectivity=8)
        silhouette = labels == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])) if count > 1 else np.zeros(frame.shape, dtype=bool)
        metrics = focus_metrics(frame, tip=tip, um_per_px=scale, silhouette_mask=silhouette)
        if metrics['contrast'] < min_contrast:
            reason = f'对比度不足（{metrics["contrast"]:.1f} < {min_contrast:g}）'
        elif metrics['edge_width_px'] is None or metrics['edge_width_px'] <= 0:
            reason = '针尖附近无有效边缘'
        else:
            reason = '' if status == '精定位' or manual_confirmed else reason
    valid = can_measure and not reason
    if not valid:
        metrics['edge_width_px'] = metrics['edge_width_um'] = None
    tracked = tracker.update(metrics['edge_width_px'], tip if valid else None)
    return status, reason, metrics, tracked


def draw_tip_preview(frame, name, detection, ui_config, scale):
    from nest_clear.baseline.session import _draw_labels
    readiness = ui_config.get('_live_readiness') if isinstance(ui_config, dict) else None
    tracked = ui_config.get('_live_tracked') if isinstance(ui_config, dict) else None
    image = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    height, width = frame.shape[:2]
    tip = detection.point_uv
    center = ui_config.get('main_target_uv') or ((width - 1) / 2, (height - 1) / 2)
    if name == 'front':
        cx, cy = map(round, center)
        cv2.drawMarker(image, (cx, cy), (160, 165, 175), cv2.MARKER_CROSS, 24, 1)
    labels = []
    if isinstance(detection, TridentResult):
        if detection.axis and detection.sampling_rows:
            p, d = detection.axis['point'], detection.axis['direction']
            a, b = min(detection.sampling_rows), max(detection.sampling_rows)
            if abs(d[1]) > 1e-9:
                cv2.line(image, (round(p[0]+(a-p[1])*d[0]/d[1]), a),
                         (round(p[0]+(b-p[1])*d[0]/d[1]), b), (255, 180, 0), 1)
            mask = detection.tool_mask
            if mask is not None:
                yy, xx = np.nonzero(mask[a:b+1])
                if len(xx): cv2.rectangle(image, (int(xx.min()), a), (int(xx.max()), b), (255, 180, 0), 1)
        if detection.farthest_prong_uv:
            cv2.circle(image, tuple(map(round, detection.farthest_prong_uv)), 5, (255, 180, 0), 2)
        if detection.axis_rms_px is not None and '工具在动或画面畸变' in detection.diagnostic:
            labels.append(((20, 95), detection.diagnostic, 'Motion or distortion', (0, 0, 255), 30))
    if tip is None:
        labels.append(((20, 20), '未识别到针尖：' + (detection.diagnostic or '无结果'), 'Tip not detected: ' + detection.diagnostic, (0, 0, 255), 36))
    else:
        x, y = map(round, tip)
        color = (0, 220, 255) if detection.coarse else (0, 220, 0)
        cv2.drawMarker(image, (x, y), color, cv2.MARKER_CROSS, 48, 4)
        if not isinstance(detection, TridentResult):
            cv2.rectangle(image, (max(0, x-80), max(0, y-80)), (min(width-1, x+80), min(height-1, y+80)), color, 3)
        status = '粗定位' if detection.coarse else '精定位'
        position = f'{status} ({tip[0]:.1f}, {tip[1]:.1f})'
        if name == 'front':
            position += f'  中心参考偏 Δu={tip[0]-center[0]:+.1f} Δv={tip[1]-center[1]:+.1f}'
        labels.append(((20, 20), position, f'{status} ({tip[0]:.1f}, {tip[1]:.1f})', color, 32))
        axis_hint = CAMERA_AXES.get(name, {}).get('jog_hint', '')
        pct = (tracked or {}).get('percent')
        edge_px = getattr(detection, 'edge_width_px', None)
        pct_str = f'{pct:.0f}%' if isinstance(pct, (int, float)) else '—%'
        edge_str = f'{edge_px:.2f}px' if isinstance(edge_px, (int, float)) else '—px'
        if readiness and readiness.get('edge_near'):
            focus_line = EDGE_MARGIN_MESSAGE
            focus_en = 'Near edge'
            focus_color = (0, 0, 220)
        elif readiness and readiness.get('ready'):
            stage_txt = readiness.get('stage_text') or '✔ 最清晰'
            focus_line = f'清晰度 {pct_str}  边缘 {edge_str}  {stage_txt}'
            focus_en = f'Focus {pct_str} edge {edge_str} READY'
            focus_color = (0, 165, 0)
        else:
            stage_or_reason = (readiness or {}).get('stage_text') or (readiness or {}).get('short_reason') or axis_hint
            focus_line = f'清晰度 {pct_str}  边缘 {edge_str}  {stage_or_reason}'
            focus_en = f'Focus {pct_str} edge {edge_str}'
            focus_color = (0, 145, 220) if (readiness or {}).get('color_key') == 'yellow' else (0, 0, 220)
        labels.append(((20, 58), focus_line, focus_en, focus_color, 26))
        if readiness and readiness.get('wide_edge_note'):
            labels.append(((20, 90), readiness['wide_edge_note'], 'Wide edge accepted', (0, 145, 220), 22))
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
        detector_settings = getattr(getattr(config, 'tool_calibration', None), 'detection', {})
        self.trackers = {n: make_tip_tracker(detector_settings.get(n, {}), self.ui_config) for n in self.enabled}
        self.readiness_trackers = {n: ReadinessTracker() for n in self.enabled}
        self.reset_peak = {n: threading.Event() for n in self.enabled}
        self.history = {n: [] for n in self.enabled}
        self._preview_cameras = {}
        self.streams = {}
        self.capture_threads = {}
        self.exposure_requests = {name: None for name in self.enabled}
        self.retest_seq = {name: None for name in self.enabled}
        self._preview_last_write = None
        self._preview_timer = None
        self._last_summary = {n: None for n in self.enabled}
        self._closed = False
        from nest_clear.baseline.runout import RunoutController
        self.runout = RunoutController(self)
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
            cameras = {k: dict(v) for k, v in self._preview_cameras.items()}
            self._preview_last_write = now
            write_tooltip_preview_file(directory, cameras, self._clock().isoformat())

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
            if request['action'] == 'tooltip_exposure_save':
                if not device:
                    raise ValueError('无法确定相机设备路径')
                actual = camera.get_exposure()
                if type(actual) is not int or actual <= 0:
                    raise ValueError('无法读取有效硬件曝光')
                if request.get('expected_exposure') is not None and request['expected_exposure'] != actual:
                    raise ValueError('曝光已变化，请重新确认')
                old = measurement_for_device(load_settings(path), device)
                metrics = {'p99': None, 'saturated_fraction': None, 'contrast': None}
                try:
                    metrics.update(measure(camera.capture()))
                except Exception:
                    pass
                if stop.is_set():
                    raise RuntimeError('预览已关闭，曝光未保存')
                if camera.get_exposure() != actual:
                    raise ValueError('曝光已变化，请重新确认')
                snapshot = camera.settings_snapshot()
                if snapshot.get('device') != device:
                    raise ValueError('相机设备路径已变化，请重新打开预览')
                saved = save_measurement(path, name, snapshot, {'exposure': actual, **metrics}, source='manual')
                with self.lock:
                    self.state[name].update(exposure_source=exposure_source(saved), exposure_saved=saved)
                    self.state[name].pop('exposure_error', None)
                self.events.emit('tooltip_exposure_saved', camera=name, device=device, exposure=actual, source='manual', old_value=old.get('exposure'))
                self.exposure_reply(name, seq, 'done', f'已存曝光 {actual/10:.1f} ms（手动）')
                self.refresh_existing()
                return
            limits = camera.exposure_limits()
            if request['action'] == 'tooltip_exposure_restore':
                measured = measurement_for_device(load_settings(path), device)
                value = measured.get('exposure') if measured else camera.config.get('camera_controls', {}).get('exposure', CAMERA_DEFAULTS['exposure'])
                source = exposure_source(measured)
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
                source = exposure_source(measured)
                self.state[name].update(device=device, exposure_limits=limits, exposure_source=source, exposure_saved=measured, exposure_busy=not bool(measured))
            self.publish()
            if not measured:
                self.retest[name].set()
            while not stop.is_set():
                if self.retest[name].is_set():
                    with self.lock:
                        self.state[name]['exposure_busy'] = True
                        self.retest[name].clear()
                        request_seq = self.retest_seq[name]
                        self.retest_seq[name] = None
                    stream.invalidate()
                    self.reset_peak[name].set()
                    with self.lock:
                        previous_source = self.state[name].get('exposure_source', '默认')
                        self.state[name]['exposure_source'] = '正在测曝光…'; self.state[name].pop('exposure_error', None)
                        self.state[name]['exposure_progress'] = '正在测曝光，准备采集'
                        self.state[name]['exposure_scan_rows'] = []
                    self.publish()
                    rows = []
                    scan_options = getattr(getattr(self.config, 'tool_calibration', None), 'exposure_scan', {}) or {}
                    method = scan_options.get('method', 'quality_scan')
                    scan_error = None
                    result = None
                    initial = None
                    saved_ok = False
                    try:
                        initial = camera.get_exposure()
                        try:
                            options = dict(getattr(self.config.tool_calibration, 'auto_exposure', {}) if self.config else {})
                            if method == 'brightness':
                                if not measured:
                                    options['initial_exposure'] = max(1, limits['min'] if limits else 1)
                                result = auto_exposure(ExposureCamera(self), **options)
                            else:
                                detector_cfg = self.config.tool_calibration.detection[name] if self.config else {}
                                detector = make_tip_detector(detector_cfg, SilhouetteTipDetector)
                                def progress(k, total, row):
                                    rows.append(row)
                                    with self.lock:
                                        self.state[name]['exposure_progress'] = f'正在测曝光 第{k}/{total}档'
                                    self.publish()
                                result = scan_exposure(ExposureCamera(self), detector,
                                    levels=scan_options.get('max_exposure', 10), contrast_min=scan_options.get('contrast_min', 40.),
                                    sharpness_ratio=scan_options.get('sharpness_ratio', .9), saturation_max=scan_options.get('saturation_max', .05), progress=progress)
                                rows = result['rows']
                        except Exception:
                            camera.set_exposure(initial); raise
                        snapshot = camera.settings_snapshot()
                        snapshot['exposure_status'] = '已测曝光'
                        save_measurement(path, name, snapshot, result)
                        saved_ok = True
                        camera.exposure_status = '已测曝光'
                        if hasattr(camera, 'actual_settings'): camera.actual_settings['exposure_status'] = '已测曝光'
                        with self.lock:
                            measured = measurement_for_device(load_settings(path), device)
                            self.state[name].update(exposure_source=exposure_source(measured), exposure_saved=measured)
                        self.events.emit('tooltip_exposure_measured', camera=name, device=device, exposure=result['exposure'])
                        self.publish()
                        self.refresh_existing()
                    except Exception as exc:
                        scan_error = str(exc)
                        rows = getattr(exc, 'rows', rows)
                        if result is not None and initial is not None and not saved_ok:
                            try: camera.set_exposure(initial)
                            except Exception: logger.exception('保存曝光失败后恢复原曝光失败')
                        if stop.is_set():
                            break
                        with self.lock:
                            self.state[name]['exposure_source'] = previous_source; self.state[name]['exposure_error'] = str(exc)
                        self.events.emit('tooltip_exposure_failed', camera=name, reason=str(exc))
                        self.publish()
                    finally:
                        if method == 'quality_scan':
                            record = {'camera': name, 'device': device, 'time': self._clock().isoformat(), 'rows': rows,
                                      'exposure': result.get('exposure') if result else None,
                                      'prong_ref': rows[0].get('prong_ref') if rows else None,
                                      'reason': scan_error or (result.get('reason') if result else None), 'success': result is not None and scan_error is None}
                            try:
                                now = self._clock()
                                directory = Path(self.events.directory) / now.date().isoformat() / 'tooltip_exposure'
                                directory.mkdir(parents=True, exist_ok=True)
                                path_record = directory / (now.strftime('%H%M%S_%f') + '_' + name + '.json')
                                path_record.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
                                self.events.emit('tooltip_exposure_scan', camera=name, success=record['success'], path=str(path_record), reason=record['reason'])
                            except Exception as exc:
                                self.events.emit('warning', reason=f'曝光扫描记录失败：{exc}')
                                self.events.emit('tooltip_exposure_scan', camera=name, success=record['success'], path=None, reason=record['reason'])
                            with self.lock:
                                self.state[name]['exposure_scan_rows'] = rows
                        with self.lock:
                            self.state[name].pop('exposure_progress', None)
                        self.reset_peak[name].set()
                        with self.lock:
                            self.state[name]['exposure_busy'] = False
                        self.publish()
                        if request_seq is not None:
                            if scan_error:
                                self.exposure_reply(name, request_seq, 'error', '预览已关闭' if stop.is_set() else scan_error)
                            elif result:
                                self.exposure_reply(name, request_seq, 'done', f'曝光已设为 {result["exposure"]/10:g} ms')
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
        if name.startswith('tooltip_runout_'):
            self.runout.command(name, params.get('_seq'))
            return
        if name == 'tooltip_peak_reset':
            camera = params.get('camera')
            if camera in (None, 'all'):
                for cam_name in self.enabled:
                    self.reset_peak[cam_name].set()
                self.publish()
                return
            if camera not in self.enabled:
                raise ValueError('请选择主或辅助对刀相机')
            self.reset_peak[camera].set()
            self.publish()
            return
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
                for key in ('tenengrad','peak','percent','edge_width_um','edge_width_px','contrast','source','tip_uv','history','exposure_error','exposure_ms','fps','detection_status','detection_reason'):
                    self.state[camera].pop(key, None)
                self._preview_cameras.pop(camera, None)
            if not on:
                self.runout.interrupt('预览已关闭，离心测试结束')
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
            with self.lock:
                if not self.enabled[camera] or camera not in self.streams:
                    error = '预览未打开'
                elif self.state[camera].get('exposure_busy') or self.retest[camera].is_set() or self.exposure_requests[camera] is not None:
                    error = '曝光操作进行中，请稍后重试'
                elif self.state[camera].get('detection_status') != '精定位':
                    error = '当前未精定位工具'
                else:
                    error = None
                    self.retest_seq[camera] = params.get('_seq')
                    self.state[camera]['exposure_busy'] = True
                    self.retest[camera].set()
            if error: self.exposure_reply(camera, params.get('_seq'), 'error', error)
        elif name in ('tooltip_exposure_set', 'tooltip_exposure_restore', 'tooltip_exposure_save'):
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
        else:
            raise ValueError('未知对刀相机命令')
        self.publish()

    def _run(self, name, stop, previous):
        if previous: previous.join()
        if stop.is_set(): return
        camera = self.worker.cameras[name]
        self.trackers[name].reset(); self.readiness_trackers[name].reset(); self.history[name] = []
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
                    self.readiness_trackers[name].reset()
                    self.history[name] = []
                if exposure is not None:
                    last_exposure = exposure
                now = self._monotonic()
                fps = round(1 / (now - last), 1) if last is not None and now > last else None
                last = now
                detector_cfg = self.config.tool_calibration.detection[name] if self.config else {}
                detection = make_tip_detector(detector_cfg, SilhouetteTipDetector).detect(frame)
                tip = detection.point_uv
                target_uv = self.ui_config.get('main_target_uv') or ((frame.shape[1]-1)/2, (frame.shape[0]-1)/2)
                delta_uv = [tip[0] - target_uv[0], tip[1] - target_uv[1]] if name == 'front' and tip is not None else None
                scale = self.ui_config.get('main_um_per_px' if name == 'front' else 'aux_um_per_px', 13.5)
                if self.reset_peak[name].is_set():
                    self.trackers[name].reset(); self.readiness_trackers[name].reset(); self.reset_peak[name].clear()
                    self.history[name] = []
                status, reason, metrics, tracked = analyze_tip(frame, detection, self.trackers[name], scale, self.ui_config.get('min_focus_contrast', 40), detector_cfg.get('threshold', 127))
                image_size = (frame.shape[1], frame.shape[0])
                readiness = self.readiness_trackers[name].update(
                    name,
                    {
                        'detection_status': status,
                        'detection_reason': detection.diagnostic,
                        'tip_uv': list(tip) if tip is not None else None,
                        'image_size': list(image_size),
                        'percent': tracked['percent'],
                        'peak': tracked['best'],
                        'sample_count': tracked.get('sample_count', 0),
                        'peak_confirmed': tracked.get('peak_confirmed', False),
                        'edge_width_px': metrics['edge_width_px'],
                    },
                    self.ui_config,
                )
                self.history[name] = [(t,v) for t,v in self.history[name] if now-t <= 30]
                if tracked['percent'] is not None:
                    self.history[name].append((now, tracked['percent']))
                with self.lock:
                    peak_edge_um = tracked['best'] * scale if metrics['source'] != 'band_gradient_p995' and tracked['best'] is not None else None
                    self.state[name]['peak_edge_width_um'] = peak_edge_um
                    self.state[name].update(tenengrad=metrics['tenengrad'], peak=tracked['best'], percent=tracked['percent'],
                        band_gradient=metrics.get('band_gradient'), axis_rms_px=metrics.get('axis_rms_px'),
                        edge_width_um=metrics['edge_width_um'], edge_width_px=metrics['edge_width_px'], contrast=metrics['contrast'],
                        source=metrics['source'], tip_uv=list(tip) if tip is not None else None,
                        detection_status=status, detection_reason=detection.diagnostic, focus_reason=reason,
                        ready=readiness['ready'], ready_reason=readiness['reason'],
                        peak_confirmed=readiness['peak_confirmed'], stage=readiness['stage'],
                        stage_text=readiness['stage_text'], wide_edge_note=readiness['wide_edge_note'],
                        sample_count=tracked.get('sample_count', 0), consecutive_ready=readiness['consecutive_ready'],
                        history=[v for _,v in self.history[name]], fps=fps)
                preview_cfg = {**self.ui_config, '_live_readiness': readiness, '_live_tracked': tracked}
                image = draw_tip_preview(frame, name, detection, preview_cfg, scale)
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
                    self._preview_cameras[name] = build_preview_camera_payload(
                        metrics=metrics,
                        tracked=tracked,
                        status=status,
                        reason=reason,
                        detection=detection,
                        tip=tip,
                        delta_uv=delta_uv,
                        history=self.state[name]['history'],
                        fps=fps,
                        scale=scale,
                        detector_cfg=detector_cfg,
                        ui_config=self.ui_config,
                        updated_at=self._clock().isoformat(),
                        exposure_ms=exposure,
                        frame_seq=item.seq,
                        captured_mono=item.captured_mono,
                        captured_at=item.captured_at,
                        frame_age_ms=max(0., (published - item.captured_mono) * 1000),
                        processing_ms=max(0., (published - processing_start) * 1000),
                        image_size=image_size,
                        readiness=readiness,
                    )
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
        self.runout.interrupt('相机已交接，离心测试结束')
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
        self.runout.interrupt('相机关闭，离心测试结束')
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
