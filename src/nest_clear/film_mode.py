"""主流程菲林测试模式的持久状态。"""
import json
import os
from pathlib import Path
import tempfile


def read_mode(root):
    path = Path(root) / 'state' / 'film_mode.json'
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return False
    if not isinstance(value, dict) or type(value.get('enabled')) is not bool:
        raise ValueError('菲林测试模式状态文件无效')
    return value['enabled']


def write_mode(root, enabled):
    if type(enabled) is not bool:
        raise ValueError('模式须为布尔值')
    path = Path(root) / 'state' / 'film_mode.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, prefix='.film_mode_', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump({'enabled': enabled}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def may_change_mode(supervisor, main, active_tray=True):
    """主程序未运行、空闲或挑毛盘已结束时允许切换。"""
    if not isinstance(supervisor, dict) or not supervisor.get('pid'):
        return False
    if supervisor.get('child_pid'):
        return not active_tray and main.get('mode') in ('空闲', '挑毛中')
    return supervisor.get('child_state') not in ('启动中', '运行', '就绪')


from contextlib import contextmanager
import fcntl

@contextmanager
def mode_lock(root):
    directory = Path(root) / 'state'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'film_mode.lock').open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def start_tray(root):
    """与界面切换互斥，先登记活动盘再读取模式。"""
    with mode_lock(root):
        enabled = read_mode(root)
        (Path(root) / 'state' / 'film_tray.pid').write_text(str(os.getpid()), encoding='ascii')
        return enabled


def tray_active(root, child_pid):
    try:
        return int((Path(root) / 'state' / 'film_tray.pid').read_text()) == int(child_pid)
    except (OSError, ValueError, TypeError):
        return False


def end_tray(root):
    with mode_lock(root):
        (Path(root) / 'state' / 'film_tray.pid').unlink(missing_ok=True)


def clear_stale_tray(root):
    with mode_lock(root):
        path = Path(root) / 'state' / 'film_tray.pid'
        try:
            owner = int(path.read_text(encoding='ascii'))
        except (OSError, ValueError):
            owner = None
        if owner != os.getpid():
            path.unlink(missing_ok=True)
