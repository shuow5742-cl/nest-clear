#!/usr/bin/env python3
"""回放五帧转角组，使用与现场相同的逐帧质量及稳定性规则。"""
import argparse
from pathlib import Path
import cv2
from nest_clear.baseline.runout import evaluate_frames
from nest_clear.tooltip.runout import RunoutSession


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path('/home/shuow/nest-clear-field/20261124_tipsweep/device'))
    parser.add_argument('--output', type=Path, default=Path('/home/shuow/nest-clear-field/20261125_t79eval/runout_replay.md'))
    args = parser.parse_args()
    session = RunoutSession()
    lines = ['# 离心测试回放', '', '五帧逐帧质量检查；各坐标总体标准差 ≤ 0.3 px，坐标取中位数；比例尺 13.5 µm/px。', '', '| 组 | 采用/作废及原因 | 累计最大离心量 µm | 变化 µm |', '|---|---|---:|---:|']
    maximum = 0.0
    last = None
    for i in range(1, 13):
        folder = args.input / f'set_{i:03d}'
        frames = {n: [cv2.imread(str(folder / f'{n}_{j}.png'), cv2.IMREAD_GRAYSCALE) for j in range(1, 6)] for n in ('front', 'side')}
        if any(x is None for images in frames.values() for x in images):
            raise FileNotFoundError(f'{folder} 缺少图像')
        _, result = evaluate_frames(frames, {}, session)
        previous = maximum
        if result['rejected']:
            description = '作废：' + result['reason']
        else:
            maximum = result['max_deviation_um']; last = result
            description = '采用'
        lines.append(f'| set_{i:03d} | {description} | {maximum:.1f} | {maximum-previous:+.1f} |')
    if last:
        lines += ['', f'最终最大离心量 **{maximum:.1f} µm**，与人工标注 521 µm 相差 **{abs(maximum-521):.1f} µm**；{"合格" if last["pass"] else "不合格"}。']
    else:
        lines += ['', '没有可采用的组。']
    lines += ['', '## 已知静止帧异常补充检查', '', '以下组独立检查，不计入上面的 12 次转角回放。', '', '| 组 | 采用/作废及原因 |', '|---|---|']
    for device, index in (('device2', 6), ('device2', 8), ('device2', 10), ('device', 14)):
        folder = args.input.parent / device / f'set_{index:03d}'
        frames = {n: [cv2.imread(str(folder / f'{n}_{j}.png'), cv2.IMREAD_GRAYSCALE) for j in range(1, 6)] for n in ('front', 'side')}
        if any(x is None for images in frames.values() for x in images):
            raise FileNotFoundError(f'{folder} 缺少图像')
        _, check = evaluate_frames(frames, {}, RunoutSession())
        description = '作废：' + check['reason'] if check['rejected'] else '采用：单帧质量和五帧标准差均在门限内'
        lines.append(f'| {device}/set_{index:03d} | {description} |')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines))

if __name__ == '__main__':
    main()
