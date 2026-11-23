"""预览相机的临时曝光控制，不修改配置。"""
import json
import math
from datetime import datetime
from pathlib import Path


def change_exposure(worker, action, configured_us, limits, root, busy=False):
    old = worker.readback.get('exposure_us')
    try:
        if busy:
            raise ValueError('白平衡或曝光自测执行中，不能设置曝光')
        try:
            value = float(configured_us if action['action'] == 'restore_exposure' else action.get('exposure_us'))
        except (TypeError, ValueError):
            raise ValueError('请输入有效的曝光数值') from None
        if not math.isfinite(value):
            raise ValueError('请输入有效的曝光数值')
        if action['action'] != 'restore_exposure' and not limits['min_us'] <= value <= limits['max_us']:
            raise ValueError(f"曝光须在 {limits['min_us']:g}–{limits['max_us']:g} us 范围内")
        actual = worker.set_exposure(value)
        now = datetime.now().astimezone()
        directory = Path(root) / 'focus_records' / now.date().isoformat()
        directory.mkdir(parents=True, exist_ok=True)
        event = {'time': now.isoformat(), 'event': action['action'], 'old_us': old, 'new_us': value, 'readback_us': actual}
        with (directory / 'exposure.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(event, ensure_ascii=False) + '\n')
        return {'state': 'done', 'message': f'曝光回读 {actual:g} us', 'readback_us': actual}
    except Exception as exc:
        return {'state': 'error', 'message': str(exc) if str(exc) else '请输入有效的曝光数值'}
