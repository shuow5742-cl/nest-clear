"""离心测试的采集、逐帧质量检查和记录，运行于相机拥有者进程。"""
from concurrent.futures import ThreadPoolExecutor
import copy
from datetime import datetime
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from nest_clear.tooltip.factory import detector_identity, make_tip_detector
from nest_clear.tooltip.runout import RunoutSession


FRAME_COUNT = 5
FRAME_QUALITY_PASSED = '五帧质量及一致性检查通过'


def measure_frames(frames, configs, std_max=0.3, *, rms_max=None, capture_times=None):
    """Measure each view, with the same population std gate used for runout."""
    representatives, views, reasons = {}, {}, []
    for camera, label in (('front', '正面'), ('side', '侧面')):
        config = configs.get(camera, {})
        detector = make_tip_detector(config)
        limit = rms_max if rms_max is not None else config.get('max_distortion_axis_rms_px', 2.0)
        records, view_reasons = [], []
        starts = (capture_times or {}).get(camera)
        for i, frame in enumerate(frames.get(camera, ()), 1):
            result = detector.detect(frame)
            if i == 1:
                representatives[camera] = result
            point = result.point_uv
            rms = getattr(result, 'axis_rms_px', None)
            detail = result.diagnostic
            reason = ''
            if point is None or not np.all(np.isfinite(point)):
                detail = {'no dark component': '图像中未找到暗色针尖区域',
                          'largest component is too small': '图像中的暗色针尖区域过小'}.get(detail, detail)
                reason = f'无法识别针尖：{detail or "未识别"}'
                point = None
            elif rms is not None and rms > limit:
                reason = f'工具在动或画面畸变（axis_rms {rms:.3f} > {limit:g} px）'
            elif result.coarse:
                reason = f'未通过单帧精定位检查：{detail}'
            elif not result.focused:
                reason = f'针尖模糊：{detail}'
            elif rms is None or not np.isfinite(rms):
                reason = '缺少有效 axis_rms，无法检查畸变'
            if reason:
                view_reasons.append(f'{label}第 {i} 帧{reason}')
            records.append({
                'index': i, 'uv': list(map(float, point)) if point is not None else None,
                'status': getattr(result, 'status', None),
                'focused': result.focused, 'coarse': result.coarse, 'diagnostic': detail,
                'axis_rms_px': float(rms) if rms is not None and np.isfinite(rms) else None,
                'tip_contrast': getattr(result, 'tip_contrast', None),
                'edge_fit': getattr(result, 'edge_fit', None),
                'edge_width_px': result.edge_width_px, 'width_px': result.width_px,
                'reason': reason, 'capture_mono': starts[i-1] if starts is not None else None,
            })
        std = median = None
        if len(records) != FRAME_COUNT:
            view_reasons.append(f'{label}应采集 {FRAME_COUNT} 帧，实际 {len(records)} 帧')
        elif all(row['uv'] is not None for row in records):
            points = [row['uv'] for row in records]
            std = np.std(points, axis=0).tolist()
            median = np.median(points, axis=0).tolist()
            for axis, spread in zip(('u', 'v'), std):
                if spread > std_max:
                    view_reasons.append(f'{label}五帧 {axis} 标准差 {spread:.3f} > {std_max:g} px')
        reasons.extend(view_reasons)
        views[camera] = {
            'detector': detector_identity(config), 'frames': records,
            'std_uv': std, 'median_uv': median, 'accepted': not view_reasons,
            'reasons': view_reasons, 'axis_rms_max_px': limit,
            'max_edge_width_px': detector.max_edge_width_px,
        }
    return representatives, {'views': views, 'std_max_px': std_max,
                             'rejected': bool(reasons), 'reason': '；'.join(reasons)}


def detect_frames(frames, configs, std_max=0.3):
    _, measurement = measure_frames(frames, configs, std_max)
    coords = {name: [row['uv'] for row in view['frames']]
              for name, view in measurement['views'].items()}
    if measurement['rejected']:
        return coords, {'rejected': True, 'reason': measurement['reason']}
    return coords, None


def add_coordinates(coords, session, scales):
    maps = [np.eye(2) * scale / 1000 for scale in scales]
    return session.add_press(coords['front'], coords['side'], *maps,
                             meta={'scale_um_per_px': list(scales), 'scale_source': '11-25 实测：Y 每 0.04mm 侧面平移 2.95px'})


def evaluate_frames(frames, configs, session, std_max=0.3, scales=(13.5, 13.5)):
    coords, result = detect_frames(frames, configs, std_max)
    return coords, result if result is not None else add_coordinates(coords, session, scales)


class RunoutController:
    def __init__(self, preparation):
        self.prep = preparation
        self.events = preparation.events
        self.lock = threading.RLock()
        self.session = None
        self.thread = None
        self.state = {'active': False, 'busy': False}
        self.attempt = 0

    def publish(self, **updates):
        self.state.update(updates)
        self.events.emit('baseline_state', runout=dict(self.state))

    def ready_reason(self):
        with self.prep.lock:
            for name, label in (('front', '正面'), ('side', '侧面')):
                if not self.prep.enabled[name] or name not in self.prep.streams:
                    return f'请打开{label}相机预览'
                if self.prep.state[name].get('detection_status') != '精定位':
                    return f'{label}针尖未精定位：{self.prep.state[name].get("detection_reason") or "等待识别"}'
        return ''

    def command(self, name, seq):
        with self.lock:
            if name == 'tooltip_runout_end':
                self.end('用户结束', seq)
                return
            reason = self.ready_reason()
            if reason:
                self.publish(reply_seq=seq, message=reason)
                return
            if name == 'tooltip_runout_start':
                if self.session:
                    self.publish(reply_seq=seq, message='离心测试已开始')
                    return
                now = datetime.now().astimezone()
                root = Path(self.events.directory) / now.date().isoformat() / 'tooltip_runout'
                root.mkdir(parents=True, exist_ok=True)
                self.directory = root / now.strftime('%H%M%S_%f')
                self.directory.mkdir()
                self.session = RunoutSession(max_mm=self.prep.ui_config.get('runout_max_mm', .05))
                self.attempt = 0
                self.events.emit('tooltip_runout_start', directory=str(self.directory))
                self.publish(active=True, busy=False, result=None, rejected_reason='', reply_seq=seq, message='请转动工具并停稳后记录')
            elif name == 'tooltip_runout_press':
                if not self.session or self.state.get('busy'):
                    self.publish(reply_seq=seq, message='请先开始离心测试或等待计算完成')
                    return
                self.attempt += 1
                pressed = time.monotonic()
                self.publish(busy=True, message='正在采集并计算')
                self.pending_seq = seq
                snapshot = copy.deepcopy(self.session)
                self.thread = threading.Thread(target=self._press, args=(pressed, seq, self.attempt, self.session, snapshot, self.directory), daemon=True, name='tooltip-runout')
                self.thread.start()

    def _press(self, pressed, seq, attempt, session, snapshot, directory):
        frames, coords = {}, {}
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = {n: pool.submit(self.prep.capture_after, n, pressed, FRAME_COUNT, 5) for n in ('front', 'side')}
                for n, f in futures.items():
                    try:
                        frames[n] = [item.frame for item in f.result()]
                    except Exception as exc:
                        raise RuntimeError(f'{"正面" if n == "front" else "侧面"}采集失败：{exc}') from exc
            for n in ('front', 'side'):
                if not cv2.imwrite(str(directory / f'{attempt:03d}_{n}.png'), frames[n][0]):
                    raise OSError(f'{n} 原图保存失败')
            configs = self.prep.config.tool_calibration.detection if self.prep.config else {}
            scales = (self.prep.ui_config.get('main_um_per_px', 13.5), self.prep.ui_config.get('aux_um_per_px', 13.5))
            coords, result = detect_frames(frames, configs, self.prep.ui_config.get('runout_frame_std_max_px', .3))
            if result is None:
                with self.lock:
                    active = self.session is session
                if active:
                    result = add_coordinates(coords, snapshot, scales)
        except Exception as exc:
            result = {'rejected': True, 'reason': str(exc)}
        with self.lock:
            interrupted = self.session is not session
            if interrupted:
                result = {'rejected': True, 'reason': '离心测试已结束，本次记录被中断'}
            elif not result['rejected']:
                self.session = snapshot
            record = {'attempt': attempt, 'time': datetime.now().astimezone().isoformat(), 'coordinates': coords, 'result': result, 'interrupted': interrupted}
            with (directory / 'session.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            self.events.emit('tooltip_runout_press', **record)
            if result['rejected']:
                self.events.emit('tooltip_runout_rejected', **record)
            if not interrupted:
                self.publish(busy=False, reply_seq=seq, result=self.state.get('result') if result['rejected'] else result,
                             rejected_reason=result.get('reason') or self.state.get('rejected_reason', ''), message=result.get('reason') or result['message'])

    def end(self, reason, seq=None):
        if self.session:
            (self.directory / 'summary.json').write_text(json.dumps(self.session.to_record(), ensure_ascii=False, indent=2), encoding='utf-8')
            self.events.emit('tooltip_runout_end', reason=reason, directory=str(self.directory))
            self.session = None
        reply_seq = seq if seq is not None else (getattr(self, 'pending_seq', None) if self.state.get('busy') else self.state.get('reply_seq'))
        self.publish(active=False, busy=False, reply_seq=reply_seq, message=reason)

    def interrupt(self, reason):
        with self.lock:
            if self.session:
                self.end(reason)
