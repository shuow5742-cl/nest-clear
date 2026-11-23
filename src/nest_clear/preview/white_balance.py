"""灰卡白平衡、验证及可追溯记录。图像通道顺序为 BGR。"""
from datetime import datetime
import json
import time
from pathlib import Path

import numpy as np

DEFAULTS = {'roi_fraction': .6, 'max_saturated_fraction': .001, 'min_green': 40., 'max_green': 190., 'max_p99': 235., 'max_color_error': .03, 'max_block_color_error': .05, 'max_brightness_cv': .15, 'timeout_s': 5.}


def roi(image, fraction):
    h, w = image.shape[:2]
    rh, rw = max(1, int(h*fraction)), max(1, int(w*fraction))
    return image[(h-rh)//2:(h-rh)//2+rh, (w-rw)//2:(w-rw)//2+rw]


def evaluate(images, settings=None, saturation_level=250):
    s = {**DEFAULTS, **(settings or {})}
    regions = [roi(im, s['roi_fraction']) for im in images]
    if not regions or any(im.ndim != 3 or im.shape[2] != 3 for im in regions):
        raise ValueError('白平衡需要彩色图像')
    pixels = np.concatenate([im.reshape(-1, 3) for im in regions]).astype(np.float32)
    b, g, r = pixels.mean(axis=0)
    rg, bg = (r/g, b/g) if g else (0., 0.)
    sat = float(np.mean(np.max(pixels, axis=1) >= saturation_level))
    p99 = float(np.percentile(pixels[:, 1], 99))
    blocks = []
    for im in regions:
        for row in np.array_split(im, 6, axis=0):
            for block in np.array_split(row, 8, axis=1):
                if block.size:
                    bb, gg, rr = block.mean(axis=(0, 1))
                    if gg > 0: blocks.append((rr/gg, bb/gg, gg))
    a = np.array(blocks)
    deviation = float(max(np.max(np.abs(a[:, 0]-rg)), np.max(np.abs(a[:, 1]-bg)))) if len(a) else 0.
    cv = float(np.std(a[:, 2])/np.mean(a[:, 2])) if len(a) and np.mean(a[:, 2]) else 0.
    warnings = []
    if deviation > s['max_block_color_error']: warnings.append('画面各处色偏不一致（光源不均或目标不平整）')
    if cv > s['max_brightness_cv']: warnings.append('亮度不均')
    reason = ''
    if sat > s['max_saturated_fraction'] or g > s['max_green'] or p99 > s['max_p99']:
        reason = '画面过亮或饱和（疑似白纸），请用 18% 灰卡做白平衡'
    elif g < s['min_green']:
        reason = '画面过暗，请检查光源/曝光后用 18% 灰卡做白平衡'
    elif abs(rg-1) > s['max_color_error'] or abs(bg-1) > s['max_color_error']:
        reason = f'校正后仍偏色 R/G={rg:.3f}, B/G={bg:.3f}，请用 18% 灰卡做白平衡'
    return {'green_mean': float(g), 'p99': p99, 'saturated_fraction': sat, 'r_g': float(rg), 'b_g': float(bg), 'block_color_deviation': deviation, 'brightness_cv': cv, 'warnings': warnings, 'reason': reason}


def read_ratios(worker):
    camera = worker.camera
    device = getattr(camera, '_cam', None)
    if device is None:
        value = getattr(camera, 'balance_ratio', None) or worker.readback.get('balance_ratio')
        if value is None: raise ValueError('无法回读当前白平衡参数')
        return list(map(float, value))
    selector = device.BalanceRatioSelector
    previous = selector.get()
    channels = camera._gx.GxBalanceRatioSelectorEntry
    try:
        values = []
        for channel in (channels.RED, channels.GREEN, channels.BLUE):
            selector.set(channel); values.append(float(device.BalanceRatio.get()))
        return values
    finally: selector.set(previous)


def write_ratios(worker, values):
    camera = worker.camera
    device = getattr(camera, '_cam', None)
    if device is None:
        camera.balance_ratio = list(values)
    else:
        auto = getattr(device, 'BalanceWhiteAuto', None)
        if auto is not None and (not hasattr(auto, 'is_writable') or auto.is_writable()): auto.set(camera._gx.GxAutoEntry.OFF)
        selector = device.BalanceRatioSelector
        previous = selector.get(); channels = camera._gx.GxBalanceRatioSelectorEntry
        try:
            for channel, value in zip((channels.RED, channels.GREEN, channels.BLUE), values):
                selector.set(channel); device.BalanceRatio.set(float(value))
        finally: selector.set(previous)
    actual = read_ratios(worker)
    worker.readback['balance_ratio'] = actual
    return actual


def run_white_balance(worker, settings, calibration, path, root, progress=lambda _: None):
    from nest_clear.tooloffset.store import save_mapping
    s = {**DEFAULTS, **settings}
    events = []
    def event(name, **data):
        events.append({'event': name, 'time': datetime.now().astimezone().isoformat(), **data})
    result = {'state': 'running'}
    event('start'); progress(dict(result))
    original = None
    success = False
    try:
        original = read_ratios(worker)
        device = getattr(worker.camera, '_cam', None)
        node = getattr(device, 'BalanceWhiteAuto', None) if device is not None else None
        gx = getattr(worker.camera, '_gx', None)
        supported = node is not None and hasattr(gx.GxAutoEntry, 'ONCE') and (not hasattr(node, 'is_writable') or node.is_writable())
        method = 'once' if supported else 'software'
        if supported:
            # Some cameras expose no AWB ROI. Leave unsupported ROI nodes untouched.
            try:
                width, height = worker.size
                for name, value in [('AWBROIWidth', int(width*.6)), ('AWBROIHeight', int(height*.6)), ('AWBROIOffsetX', int(width*.2)), ('AWBROIOffsetY', int(height*.2))]:
                    feature = getattr(device, name, None)
                    if feature is not None and (not hasattr(feature, 'is_writable') or feature.is_writable()): feature.set(value)
            except Exception:
                pass
            try:
                node.set(gx.GxAutoEntry.ONCE)
            except Exception:
                supported = False
                method = 'software'
        if supported:
            deadline = time.monotonic() + s['timeout_s']
            while True:
                worker.execute({'stage': 'preview'})
                if node.get() == gx.GxAutoEntry.OFF: break
                if time.monotonic() >= deadline: raise TimeoutError('自动白平衡超时，请用 18% 灰卡做白平衡')
            ratios = read_ratios(worker)
        else:
            image, _ = worker.execute({'stage': 'preview'})
            b, g, r = roi(image, s['roi_fraction']).mean(axis=(0, 1))
            if min(r, g, b) <= 0: raise ValueError('画面过暗，请检查光源/曝光后用 18% 灰卡做白平衡')
            ratios = write_ratios(worker, [original[0]*g/r, original[1], original[2]*g/b])
        event('readback', method=method, balance_ratio=ratios, original_ratio=original)
        worker.execute({'stage': 'preview'})
        images = [worker.execute({'stage': 'preview'})[0] for _ in range(3)]
        metrics = evaluate(images, s, calibration.get('saturation_level', 250))
        event('judgement', metrics=metrics)
        result = {'state': 'fail' if metrics['reason'] else 'success', 'method': method, 'balance_ratio': ratios, 'metrics': metrics, 'reason': metrics['reason']}
        if not metrics['reason']:
            stamp = datetime.now().astimezone().isoformat()
            backup = save_mapping(path, {'balance_ratio': ratios, 'time': stamp, 'method': method, 'metrics': metrics}, 'camera_white_balance')
            result.update(time=stamp, backup=str(backup) if backup else None)
            worker.readback['balance_ratio'] = ratios
            success = True
    except Exception as exc:
        result = {'state': 'fail', 'reason': f'{exc}，请用 18% 灰卡做白平衡'}
    finally:
        if not success and original is not None:
            try: write_ratios(worker, original)
            except Exception as exc: result['reason'] += f'；恢复原白平衡失败：{exc}'
        event('result', **result)
        directory = Path(root) / 'focus_records' / datetime.now().astimezone().date().isoformat()
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / 'white_balance.jsonl').open('a', encoding='utf-8') as f:
            for row in events: f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    progress(result)
    return result
