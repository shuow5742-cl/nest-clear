"""Main-process preview host, borrowing the application's open camera worker."""
import json
import os
import threading
import time
from pathlib import Path

from nest_clear.paths import supervisor_state_file, ui_command_file
from nest_clear.preview.eligibility import preview_eligibility
from nest_clear.preview.process import atomic_write, run_preview_session
from nest_clear.preview.control import REQUEST_CAMERAS
from nest_clear.preview.white_balance import read_ratios, write_ratios
from nest_clear.devices.gx_enum import enum_value


class MainPreview:
    def __init__(self, config, worker, events):
        self.config, self.worker, self.events = config, worker, events
        self.root = Path(config.record_directory)
        self.command_path = ui_command_file(self.root)
        self.state_path = self.root / 'state' / 'main_preview_state.json'
        self.status_path = self.root / 'state' / 'preview_status.json'
        self.seq = self._existing_seq()
        self.thread = None
        self.stop_event = None
        self.mode = None
        self.restore_error = None
        self._state_lock = threading.RLock()
        self._finished = False
        self._publish(False)

    def _existing_seq(self):
        try:
            handoff = json.loads((self.root / 'state' / 'main_preview_handoff.json').read_text(encoding='utf-8'))
            if handoff.get('pid') == os.getppid() and type(handoff.get('seq')) is int:
                return handoff['seq']
        except (OSError, ValueError, AttributeError):
            pass
        try:
            value = json.loads(self.command_path.read_text(encoding='utf-8')).get('seq')
            return value if type(value) is int else 0
        except (OSError, ValueError, AttributeError):
            return 0

    def _publish(self, on, error=None):
        data = {'on': on, 'pid': os.getpid(), 'seq': self.seq, 'mode': self.mode}
        if error:
            data['error'] = str(error)
        with self._state_lock:
            atomic_write(self.state_path, json.dumps(data, ensure_ascii=False).encode())

    def _eligible(self, mode):
        try:
            path = supervisor_state_file(self.root)
            age = max(0, time.time() - path.stat().st_mtime)
            supervisor = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(supervisor, dict): supervisor = {}
        except (OSError, ValueError, TypeError):
            supervisor, age = {}, None
        return preview_eligibility(supervisor, age, mode.value, 'main')

    def poll(self, mode):
        try:
            cmd = json.loads(self.command_path.read_text(encoding='utf-8'))
            seq, wanted = cmd['seq'], cmd['preview']
            wanted_mode = cmd.get('mode', 'preview')
            if type(seq) is not int or type(wanted) is not bool or wanted_mode not in ('preview', 'focus') or seq <= self.seq:
                return
        except (OSError, ValueError, KeyError, TypeError):
            return
        self.seq = seq
        if not wanted:
            self.stop()
            self._publish(False)
            return
        allowed, reason = self._eligible(mode)
        if not allowed:
            self.events.emit('preview_rejected', camera='main', reason=reason)
            self._publish(bool(self.thread and self.thread.is_alive()), reason)
            return
        if self.thread:
            with self._state_lock:
                active = self.thread.is_alive() and not self._finished
                if active:
                    self._publish(True)
            if active:
                return
            self.stop()
        if getattr(self.events, 'request_pending', False) and 'main' in REQUEST_CAMERAS.get(getattr(self.events, 'request_func', None), ()):
            reason = '主相机请求执行中，请在应答后重试'
            self.events.emit('preview_rejected', camera='main', reason=reason)
            self._publish(False, reason)
            return
        if self.restore_error:
            self._publish(False, self.restore_error)
            return
        self.mode = wanted_mode
        self.status_path.unlink(missing_ok=True)
        self.stop_event = threading.Event()
        self._finished = False
        self.thread = threading.Thread(target=self._run, name='main-camera-preview', daemon=True)
        self._publish(True)
        self.thread.start()
        self.events.emit('preview_start', camera='main', mode=self.mode, seq=self.seq)

    def _run(self):
        worker = self.worker
        original_exposure = original_auto = original_ratios = original_wb_auto = None
        wb_auto = None
        error = None
        try:
            original_exposure = worker.readback.get('exposure_us')
            original_auto = worker.exposure_auto_mode()
            try:
                original_ratios = read_ratios(worker)
            except (ValueError, AttributeError):
                original_ratios = None
            device = getattr(worker.camera, '_cam', None)
            wb_auto = getattr(device, 'BalanceWhiteAuto', None) if device is not None else None
            original_wb_auto = enum_value(wb_auto.get()) if wb_auto is not None else None
            if self.mode == 'focus' and (original_exposure is None or original_ratios is None):
                raise ValueError('无法回读流程曝光或白平衡，不能安全启动对焦辅助')
            run_preview_session(self.config, self.mode, worker, self.stop_event, shared=True)
        except Exception as exc:
            error = str(exc)
        finally:
            try:
                if original_exposure is not None:
                    worker.set_exposure(original_exposure)
                if original_auto is not None:
                    worker.restore_exposure_auto_mode(original_auto)
                if original_ratios is not None:
                    write_ratios(worker, original_ratios)
                if original_wb_auto is not None:
                    wb_auto.set(original_wb_auto)
                if worker.needs_reopen:
                    raise RuntimeError('预览采图超时，相机需要重开')
            except Exception as exc:
                self.restore_error = f'恢复流程相机参数失败：{exc}'
                error = self.restore_error
            with self._state_lock:
                self._finished = True
                self._publish(False, error)
            if error:
                self.events.emit('preview_error', camera='main', reason=error)

    def stop(self):
        thread = self.thread
        if thread is None:
            if self.restore_error:
                raise RuntimeError(self.restore_error)
            return False
        was_on = thread.is_alive()
        self.stop_event.set()
        thread.join()
        self.thread = None
        if self.restore_error:
            raise RuntimeError(self.restore_error)
        return was_on
