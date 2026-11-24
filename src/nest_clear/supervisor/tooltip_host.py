"""Supervisor-owned tooltip preparation while the application is stopped."""
import json
import os
import threading
from pathlib import Path

from nest_clear.baseline.preparation import Preparation
from nest_clear.flow.leaves import PREVIEW_COMMANDS, handle_preview_command
from nest_clear.flow.modes import Mode
from nest_clear.preview.process import atomic_write
from nest_clear.tooloffset.commands import CommandReader
from nest_clear.workers.calib_camera import CalibCameraWorker


class TooltipHost:
    def __init__(self, config, events, last_seq=0):
        self.config, self.events = config, events
        self.directory = Path(config.record_directory)
        self.clock, self.monotonic = events.clock, events.monotonic
        self.lock = threading.RLock()
        self.status = {'pid': os.getpid(), 'command_seq': last_seq,
                       'baseline': {'phase': 'idle', 'message': '请开始对刀基准建立'}}
        self.status_path = self.directory / 'state' / 'supervisor_tooltip_status.json'
        self.request_pending = False
        self.closed = False
        self.baseline_session = None
        self.reader = CommandReader(self.directory, self, last_seq=last_seq)
        self.worker = CalibCameraWorker(config.tool_calibration.cameras,
                                        timeout_s=config.capture_timeout_s, events=self)
        self.tooltip_preparation = Preparation(self, self.worker, config)
        self._write()

    def _write(self):
        with self.lock:
            atomic_write(self.status_path, json.dumps(self.status, ensure_ascii=False).encode())

    def emit(self, event, **fields):
        self.events.emit(event, **fields)
        with self.lock:
            if event == 'baseline_state':
                self.status['baseline'].update(fields)
                self._write()
            elif event == 'tooloffset_command':
                self.status['command_seq'] = fields['seq']
                self._write()

    def poll(self):
        if self.closed:
            return
        command = self.reader.poll(PREVIEW_COMMANDS, defer_unallowed=True)
        if command:
            handle_preview_command(self, Mode.IDLE, *command)

    def close(self):
        # This joins assignment and acquisition threads before ownership changes.
        self.tooltip_preparation.stop_previews(('front', 'side'))
        self.tooltip_preparation.close()
        self.closed = True
        for camera in self.worker.cameras.values():
            if camera.device is not None or getattr(camera, '_ownership_fd', None) is not None:
                camera.close()
        still_open = [name for name, camera in self.worker.cameras.items()
                      if camera.device is not None or getattr(camera, '_ownership_fd', None) is not None]
        if still_open:
            raise RuntimeError('对刀相机未释放：' + ', '.join(still_open))
