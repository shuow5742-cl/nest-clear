"""协调低优先级预览；设备请求仍通过正常 worker 通道。"""
import json
import os
import threading
import time
from datetime import datetime

from autoweaver.motion_policy.nodes.leaf.call_worker import next_request_id
from nest_clear.paths import live_image_file, ui_command_file


class PreviewController:
    def __init__(self, root, board, events, config, interval=.5, timeout=3., camera=None):
        self.root, self.board, self.events, self.config = root, board, events, config
        self.camera = camera
        self.interval, self.timeout = max(.05, float(interval)), timeout
        self.state = dict(on=False, seq=0, blocked_reason=None, fps=0., frames=0,
                          last_frame_time=None, actual_xyz=None, sharpness=None, corners=0,
                          board_offset_mm=None, offset_px=None, error=None)
        self.command_stamp = None
        self.request_active = False
        self.camcal_starting = False
        self.capture_id = self.xyz_id = None
        self.capture_started = self.last_submit = self.last_status = 0.
        self.capture_generation = None
        self.started = None
        self.generation = 0
        self.analysis_thread = None
        self.analysis_deadline = 0.
        self.result = None
        self.lock = threading.Lock()
        self.xyz = None
        self.xyz_frame = self.current_frame = None
        self.pending_image = None
        self.image_deadline = 0.
        self.events.preview_controller = self

    def observe(self, event, fields):
        if event == 'request':
            self.request_active = True
            if fields.get('func') == 80:
                self.camcal_starting = True
                self.off('camcal_start')
        elif event in ('camcal_done', 'camcal_failed', 'camcal_aborted'):
            self.state['blocked_reason'] = None
        elif event in ('reply', 'request_abandoned'):
            self.request_active = False
            self.camcal_starting = False

    def off(self, source):
        if self.state['on']:
            duration = time.monotonic() - self.started
            self.state['on'] = False
            self.generation += 1
            self.events.emit('preview_off', source=source, reason=source, duration_s=duration, frames=self.state['frames'])

    def poll_command(self):
        path = ui_command_file(self.root)
        try:
            stamp = path.stat().st_mtime_ns
            if stamp == self.command_stamp:
                return
            command = json.loads(path.read_text(encoding='utf-8'))
            seq = command['seq']
            if not isinstance(seq, int) or isinstance(seq, bool) or not isinstance(command['preview'], bool):
                return
            self.command_stamp = stamp
            if seq <= self.state['seq']:
                return
            self.state['seq'] = seq
            if command['preview'] and (self.camcal_starting or getattr(self.events, 'active_camcal', None)):
                self.state['blocked_reason'] = '相机标定进行中'
            elif command['preview'] and self.analysis_thread is not None and self.analysis_thread.is_alive() and time.monotonic() > self.analysis_deadline:
                self.state['blocked_reason'] = '预览处理尚未结束'
            elif command['preview']:
                self.state['blocked_reason'] = None
                if not self.state['on']:
                    self.state.update(on=True, frames=0, fps=0., error=None)
                    self.started = time.monotonic()
                    self.generation += 1
                    self.events.emit('preview_on', source='ui', duration_s=0., frames=0)
            else:
                self.state['blocked_reason'] = None
                self.off('ui')
        except (OSError, ValueError, KeyError, TypeError):
            return

    def submit(self, worker, payload):
        ident = next_request_id()
        self.board.pass_note(worker, 'execute', {**payload, '__request_id__': ident}, sender='preview')
        return ident

    def idle(self, name):
        return not self.board.read_state(name + '.pending_requests', frozenset()) and not any(
            n.namespace == name for n in self.board._pending_notes)

    def analyze(self, image, generation, deadline, xyz):
        try:
            from nest_clear.preview.analysis import analyze
            import cv2
            view, values = analyze(image, self.config, xyz)
            ok, data = cv2.imencode('.jpg', view, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                raise ValueError('JPEG 编码失败')
            path = live_image_file(self.root, 'preview_latest')
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix('.tmp')
            tmp.write_bytes(data.tobytes())
            with self.lock:
                if generation != self.generation or time.monotonic() > deadline:
                    tmp.unlink(missing_ok=True)
                    return
                os.replace(tmp, path)
                self.result = (generation, values, None)
        except Exception as exc:
            with self.lock:
                if generation == self.generation and time.monotonic() <= deadline:
                    self.result = (generation, None, str(exc))

    def tick(self):
        now = time.monotonic()
        self.poll_command()
        if self.xyz_id is not None:
            outcome = self.board.read_state('plc.results', {}).get(self.xyz_id)
            if outcome is not None:
                if self.xyz_frame == self.current_frame:
                    self.xyz = outcome.get('value')
                self.xyz_id = None
        if self.capture_id is not None:
            outcome = self.board.read_state('camera.results', {}).get(self.capture_id)
            if outcome is not None:
                ident, self.capture_id = self.capture_id, None
                results = dict(self.board.read_state('camera.results', {}))
                results.pop(ident, None)
                self.camera.write_state('camera.results', results)
                image, self.camera.preview_image = self.camera.preview_image, None
                if 'error' in outcome:
                    self.state['error'] = outcome['error']
                    self.off('error')
                elif self.capture_generation == self.generation and self.state['on'] and not self.request_active and not getattr(self.events, 'active_camcal', None):
                    if image is not None:
                        self.pending_image = image
                        self.image_deadline = now + min(self.timeout, .5)
        if self.pending_image is not None:
            if not self.state['on'] or self.request_active or self.capture_generation != self.generation:
                self.pending_image = None
            elif self.xyz_id is None or now >= self.image_deadline:
                image, self.pending_image = self.pending_image, None
                self.analysis_deadline = now + self.timeout
                self.analysis_thread = threading.Thread(target=self.analyze, args=(image, self.generation, self.analysis_deadline, self.xyz), daemon=True)
                self.analysis_thread.start()
        if self.state['on'] and self.analysis_thread is not None and self.analysis_thread.is_alive() and now > self.analysis_deadline:
            self.state['error'] = '预览图处理超时'
            self.off('error')
        with self.lock:
            result, self.result = self.result, None
        if result and result[0] == self.generation and self.state['on']:
            if result[2]:
                self.state['error'] = result[2]
            else:
                self.state.update(result[1]); self.state['error'] = None
                self.state['frames'] += 1
                self.state['fps'] = round(self.state['frames'] / max(now - self.started, .001), 2)
                self.state['last_frame_time'] = datetime.now().astimezone().isoformat()
        if (self.state['on'] and not self.request_active and not getattr(self.events, 'active_camcal', None)
                and self.capture_id is None and self.pending_image is None and (self.analysis_thread is None or not self.analysis_thread.is_alive())
                and now - self.last_submit >= self.interval and self.idle('camera')):
            self.capture_generation = self.generation
            self.capture_id = self.submit('camera', {'stage': 'preview'})
            self.current_frame = self.capture_id
            self.capture_started = self.last_submit = now
            self.xyz = None
            if self.xyz_id is None:
                self.xyz_id = self.submit('plc', {'operation': 'actual'})
                self.xyz_frame = self.current_frame
        if now - self.last_status >= .5:
            if self.events.status:
                self.events.status.update_preview(self.state)
            self.last_status = now

    def close(self):
        self.off('ui')
        if self.events.status:
            self.events.status.update_preview(self.state)
