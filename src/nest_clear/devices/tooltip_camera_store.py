"""Persist measured camera settings and compare snapshots."""
from datetime import datetime, timezone
from pathlib import Path
import os
import shutil
import tempfile
import yaml

def load_settings(path):
    path = Path(path)
    if not path.exists(): return {'cameras': {}}
    with path.open(encoding='utf-8') as f: data = yaml.safe_load(f)
    if not isinstance(data, dict) or not isinstance(data.get('cameras'), dict): raise ValueError('相机标定文件格式无效')
    return data

def save_measurement(path, name, snapshot, result):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    data = load_settings(path)
    data['cameras'][name] = {**snapshot, 'measured': True, 'time': datetime.now(timezone.utc).isoformat(),
                             'exposure': result['exposure'], 'p99': result['p99'],
                             'saturated_fraction': result['saturated_fraction'], 'contrast': result['contrast']}
    if path.exists():
        history = path.parent / 'history'; history.mkdir(exist_ok=True)
        fd, backup = tempfile.mkstemp(prefix='tooltip_cameras_', suffix='.yaml', dir=history); os.close(fd)
        shutil.copy2(path, backup)
    fd, temp = tempfile.mkstemp(prefix='.tooltip_cameras_', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f: yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp): os.unlink(temp)

def settings_snapshot(camera):
    return camera.settings_snapshot()

def compare_settings(a, b):
    result = []
    def walk(x, y, prefix=''):
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(x.keys() | y.keys()): walk(x.get(k, '<缺失>'), y.get(k, '<缺失>'), f'{prefix}.{k}' if prefix else k)
        elif x != y: result.append({'field': prefix, 'a': x, 'b': y})
    walk(a, b); return result
