"""离心测试的采集、逐帧质量检查和记录，运行于相机拥有者进程。"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from nest_clear.tooltip.factory import make_tip_detector
from nest_clear.tooltip.runout import RunoutSession


def evaluate_frames(frames, configs, session, std_max=0.3, scales=(13.5, 13.5)):
    coords, reasons = {}, []
    for camera, label in (('front', '正面'), ('side', '侧面')):
        detector = make_tip_detector(configs.get(camera, {}))
        points = []
        for i, frame in enumerate(frames[camera], 1):
            result = detector.detect(frame)
            point = result.point_uv
            points.append(list(map(float, point)) if point is not None else None)
            rms = getattr(result, 'axis_rms_px', None)
            limit = configs.get(camera, {}).get('max_distortion_axis_rms_px', 2.0)
            if rms is None:
                reasons.append(f'{label}第 {i} 帧缺少 axis_rms，无法检查畸变')
            elif rms > limit:
                reasons.append(f'{label}第 {i} 帧工具在动或画面畸变（axis_rms {rms:.3f} > {limit:g} px）')
            elif point is None or result.coarse:
                reasons.append(f'{label}第 {i} 帧未通过单帧质量检查：{result.diagnostic or "未识别"}')
        coords[camera] = points
        if len(points) != 5:
            reasons.append(f'{label}帧数不足 5 帧')
        elif all(p is not None for p in points):
            std = np.std(points, axis=0)
            for j, axis in enumerate(('u', 'v')):
                if std[j] > std_max:
                    reasons.append(f'{label}五帧 {axis} 标准差 {std[j]:.3f} > {std_max:g} px')
    if reasons:
        return coords, {'rejected': True, 'reason': '；'.join(reasons)}
    maps = [np.eye(2) * scale / 1000 for scale in scales]
    return coords, session.add_press(coords['front'], coords['side'], *maps,
                                     meta={'scale_um_per_px': list(scales), 'scale_source': '11-25 实测：Y 每 0.04mm 侧面平移 2.95px'})


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
                self.thread = threading.Thread(target=self._press, args=(pressed, seq, self.attempt), daemon=True, name='tooltip-runout')
                self.thread.start()

    def _press(self, pressed, seq, attempt):
        frames, coords = {}, {}
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = {n: pool.submit(self.prep.capture_after, n, pressed, 5, 5) for n in ('front', 'side')}
                for n, f in futures.items():
                    try:
                        frames[n] = [item.frame for item in f.result()]
                    except Exception as exc:
                        raise RuntimeError(f'{"正面" if n == "front" else "侧面"}采集失败：{exc}') from exc
            for n in ('front', 'side'):
                if not cv2.imwrite(str(self.directory / f'{attempt:03d}_{n}.png'), frames[n][0]):
                    raise OSError(f'{n} 原图保存失败')
            configs = self.prep.config.tool_calibration.detection if self.prep.config else {}
            scales = (self.prep.ui_config.get('main_um_per_px', 13.5), self.prep.ui_config.get('aux_um_per_px', 13.5))
            coords, result = evaluate_frames(frames, configs, self.session, self.prep.ui_config.get('runout_frame_std_max_px', .3), scales)
        except Exception as exc:
            result = {'rejected': True, 'reason': str(exc)}
        with self.lock:
            record = {'attempt': attempt, 'time': datetime.now().astimezone().isoformat(), 'coordinates': coords, 'result': result}
            with (self.directory / 'session.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            self.events.emit('tooltip_runout_press', **record)
            if result['rejected']:
                self.events.emit('tooltip_runout_rejected', **record)
            self.publish(busy=False, reply_seq=seq, result=self.state.get('result') if result['rejected'] else result,
                         rejected_reason=result.get('reason') or self.state.get('rejected_reason', ''), message=result.get('reason') or result['message'])

    def end(self, reason, seq=None):
        # The caller may be the PLC handoff; allow the worker to finish before summary.
        thread = self.thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            self.lock.release()
            try: thread.join()
            finally: self.lock.acquire()
        if self.session:
            (self.directory / 'summary.json').write_text(json.dumps(self.session.to_record(), ensure_ascii=False, indent=2), encoding='utf-8')
            self.events.emit('tooltip_runout_end', reason=reason, directory=str(self.directory))
            self.session = None
        self.publish(active=False, busy=False, reply_seq=seq if seq is not None else self.state.get('reply_seq'), message=reason)

    def interrupt(self, reason):
        with self.lock:
            if self.session:
                self.end(reason)
