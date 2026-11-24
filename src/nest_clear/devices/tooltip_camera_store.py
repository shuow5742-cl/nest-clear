"""Persist measured camera settings and compare snapshots."""
from datetime import datetime, timezone
from pathlib import Path
import os
import shutil
import tempfile
import threading

_lock = threading.RLock()
import yaml

def load_settings(path):
    path = Path(path)
    if not path.exists(): return {'cameras': {}}
    with path.open(encoding='utf-8') as f: data = yaml.safe_load(f)
    if not isinstance(data, dict) or not isinstance(data.get('cameras', {}), dict) or not isinstance(data.get('devices', {}), dict): raise ValueError('相机标定文件格式无效')
    data.setdefault('cameras', {})
    return data

def _write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        history = path.parent / 'history'; history.mkdir(exist_ok=True)
        fd, backup = tempfile.mkstemp(prefix='tooltip_cameras_', suffix='.yaml', dir=history); os.close(fd)
        shutil.copy2(path, backup)
    fd, temp = tempfile.mkstemp(prefix='.tooltip_cameras_', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp): os.unlink(temp)

def save_measurement(path, name, snapshot, result):
    with _lock:
        data = load_settings(path)
        measurements = data.setdefault('measurements', {})
        for old in data['cameras'].values():
            if isinstance(old, dict) and old.get('device') and old.get('measured'):
                measurements.setdefault(old['device'], old)
        data['cameras'][name] = {**snapshot, 'measured': True, 'time': datetime.now(timezone.utc).isoformat(),
                                 'exposure': result['exposure'], 'p99': result['p99'],
                                 'saturated_fraction': result['saturated_fraction'], 'contrast': result['contrast']}
        if snapshot.get('device'): measurements[snapshot['device']] = data['cameras'][name]
        _write(path, data)

def save_assignments(path, devices):
    """Atomically persist the complete front/side assignment with a backup."""
    if set(devices) != {'front', 'side'}:
        raise ValueError('须提供主、辅相机路径')
    for device in devices.values():
        if device and (not isinstance(device, str) or not device.startswith('/dev/v4l/by-path/') or not device.endswith('-video-index0')):
            raise ValueError('相机路径须为 by-path video-index0')
    if devices['front'] and devices['front'] == devices['side']:
        raise ValueError('主、辅相机不能使用同一设备')
    with _lock:
        data = load_settings(path)
        data['devices'] = dict(devices)
        _write(path, data)

def measurement_for_device(data, device):
    """Exposure belongs to a physical path, even after swapping views."""
    measured = data.get('measurements', {}).get(device, {})
    if isinstance(measured, dict) and measured.get('measured') and type(measured.get('exposure')) is int:
        return measured
    return next((item for item in data.get('cameras', {}).values()
                 if isinstance(item, dict) and item.get('device') == device
                 and item.get('measured') and type(item.get('exposure')) is int), {})

def settings_snapshot(camera):
    return camera.settings_snapshot()

def compare_settings(a, b):
    result = []
    def walk(x, y, prefix=''):
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(x.keys() | y.keys()): walk(x.get(k, '<缺失>'), y.get(k, '<缺失>'), f'{prefix}.{k}' if prefix else k)
        elif x != y: result.append({'field': prefix, 'a': x, 'b': y})
    walk(a, b); return result

def baseline_differences(config, recorded):
    """Compare recorded device settings against current assignments and measured controls."""
    if not isinstance(recorded, dict) or not recorded:
        return []
    path = Path(config.tool_calibration.directory) / 'tooltip_cameras.yaml'
    data = load_settings(path)
    differences = []
    for name, snapshot in recorded.items():
        if not isinstance(snapshot, dict) or not snapshot: continue
        cfg = config.tool_calibration.cameras.get(name, {})
        device = data.get('devices', {}).get(name) or cfg.get('device')
        if snapshot.get('device') != device:
            differences.append(f'{name}.device')
        measured = measurement_for_device(data, device)
        if measured:
            for key in ('format', 'width', 'height'):
                if key in measured and snapshot.get(key) != measured[key]: differences.append(f'{name}.{key}')
            controls = dict(measured.get('controls') or {})
            if 'exposure' in measured: controls['exposure'] = measured['exposure']
            for key, value in controls.items():
                if (snapshot.get('controls') or {}).get(key) != value: differences.append(f'{name}.controls.{key}')
        for key, value in cfg.get('camera_controls', {}).items():
            if value is None: continue  # Device-reported defaults are not explicit configuration.
            if key in ('width', 'height'):
                actual = snapshot.get(key)
            elif key == 'exposure' and measured:
                continue
            else: actual = (snapshot.get('controls') or {}).get(key)
            if actual != value and f'{name}.controls.{key}' not in differences: differences.append(f'{name}.{key}')
    return differences
