"""跨进程单槽命令通道；发送端等待状态中的确认序号后再发送。"""
from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
import time


def command_path(root):
    return Path(root) / 'live' / 'command.json'


def send(root, name, params=None, *, acknowledged=0):
    path = command_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        old = json.loads(path.read_text(encoding='utf-8'))
        previous = old.get('seq', 0)
        previous = previous if type(previous) is int else 0
    except (OSError, ValueError, AttributeError):
        previous = 0
    seq = max(time.time_ns(), previous + 1, acknowledged + 1)
    item = {'seq': seq, 'command': name, 'params': params or {},
            'time': datetime.now().astimezone().isoformat()}
    name_tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, prefix='.command_', delete=False) as f:
            name_tmp = f.name
            json.dump(item, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name_tmp, path)
    finally:
        if name_tmp:
            Path(name_tmp).unlink(missing_ok=True)
    return seq


class CommandReader:
    def __init__(self, root, events, *, last_seq=0):
        self.path, self.events, self.last_seq = command_path(root), events, last_seq

    def pending_command(self):
        """Return unread command name without advancing last_seq."""
        try:
            cmd = json.loads(self.path.read_text(encoding='utf-8'))
            seq, name, params = cmd['seq'], cmd['command'], cmd['params']
            if type(seq) is int and seq > self.last_seq and isinstance(name, str) and isinstance(params, dict):
                return name
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return None

    def poll(self, allowed, *, defer_unallowed=False, rejection_message=None):
        try:
            cmd = json.loads(self.path.read_text(encoding='utf-8'))
            seq, name, params = cmd['seq'], cmd['command'], cmd['params']
            if type(seq) is not int or seq <= self.last_seq or not isinstance(name, str) or not isinstance(params, dict):
                return None
        except (OSError, ValueError, KeyError, TypeError):
            return None
        if name in ('tooltip_assign', 'tooltip_swap'):
            self.last_seq = seq
            self.events.emit('tooloffset_command', seq=seq, command=name, params=params, accepted=False)
            self.events.emit('baseline_state', message='对刀相机已按插口固定，不可在界面修改')
            return None
        if defer_unallowed and name not in allowed:
            return None
        self.last_seq = seq
        accepted = name in allowed
        self.events.emit('tooloffset_command', seq=seq, command=name, params=params, accepted=accepted)
        if name == 'baseline_start' and not accepted:
            self.events.emit('baseline_state', message=rejection_message or '当前正在挑毛或标定，忽略开始请求')
        if accepted and (name.startswith('tooltip_runout_') or name in (
                'tooltip_exposure_retest', 'tooltip_exposure_set', 'tooltip_exposure_restore',
                'tooltip_exposure_save', 'depthcal_exposure')):
            params = {**params, '_seq': seq}
        return (name, params) if accepted else None


def startup_reader(root, events):
    """Respect a supervisor handoff, or discard stale preview/session starts."""
    try:
        handoff = json.loads((Path(root) / 'state' / 'tooltip_handoff.json').read_text(encoding='utf-8'))
        if handoff.get('pid') == os.getppid() and type(handoff.get('seq')) is int:
            return CommandReader(root, events, last_seq=handoff['seq']), True
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    try:
        prior = json.loads(command_path(root).read_text(encoding='utf-8'))
        if prior.get('command') in {'baseline_start', 'tooltip_preview', 'tooltip_exposure_retest', 'tooltip_exposure_set', 'tooltip_exposure_restore', 'tooltip_exposure_save', 'tooltip_peak_reset', 'tooltip_assign', 'tooltip_swap', 'tooltip_runout_start', 'tooltip_runout_press', 'tooltip_runout_end'} and type(prior.get('seq')) is int:
            return CommandReader(root, events, last_seq=prior['seq']), False
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return CommandReader(root, events), False
