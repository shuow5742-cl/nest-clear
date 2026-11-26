#!/usr/bin/env python3
"""Evaluate the production T81 absolute criteria on stored tip-sweep frames.

PYTHONPATH=src .venv/bin/python tools/tip_eval/exposure_criteria_check.py
Inputs are read-only; the CSV, failure overlays, and report go under --out.
"""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2
import numpy as np
import yaml
from nest_clear.tooltip.exposure_scan import _frame_metrics, absolute_criteria
from nest_clear.tooltip.factory import make_tip_detector

DATA = Path('/home/shuow/nest-clear-field/20261124_tipsweep')
OUT = Path('/home/shuow/nest-clear-field/20261125_t79eval/exposure_check')
REPORT = OUT.parent / 'exposure_replay.md'
NAMES = ('recognized', 'prongs', 'contrast', 'whiteness')
LABELS = {'recognized': '精定位且非畸变', 'prongs': '齿数完整', 'contrast': '对比度', 'whiteness': '工具不发白'}


def evaluate(job):
    path, root, config = job
    cv2.setNumThreads(1)
    path, root = Path(path), Path(root)
    camera = path.name.split('_')[0]
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f'图像无法读取：{path}')
    detector = make_tip_detector(config['detection'][camera])
    class RecordingDetector:
        def detect(self, image):
            self.result = detector.detect(image)
            return self.result
    recording = RecordingDetector()
    metrics = _frame_metrics(image, recording)
    result = recording.result
    rms = getattr(result, 'axis_rms_px', None)
    expected = 3 if camera == 'front' else 2
    gates = absolute_criteria(metrics, expected, config['contrast_min'], config['saturation_max'])
    row = {'frame': str(path.relative_to(root)), 'camera': camera,
           'axis_rms_px': rms, 'excluded': rms is not None and rms > 2,
           'diagnostic': metrics['diagnostic'], 'expected_prongs': expected,
           'prongs_value': metrics['prongs'], 'contrast_value': metrics['contrast'],
           'whiteness_value': metrics['tool_whiteness'], **gates}
    return row


def write_overlay(path, row, config, destination):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    det = make_tip_detector(config['detection'][row['camera']]).detect(image)
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    mask = getattr(det, 'tool_mask', None)
    if mask is not None:
        tint = canvas.copy(); tint[mask != 0] = (0, 0, 255)
        canvas = cv2.addWeighted(canvas, .75, tint, .25, 0)
    if det.point_uv is not None:
        cv2.drawMarker(canvas, tuple(map(round, det.point_uv)), (0, 255, 255), cv2.MARKER_CROSS, 25, 2)
    if getattr(det, 'farthest_prong_uv', None) is not None:
        cv2.circle(canvas, tuple(map(round, det.farthest_prong_uv)), 8, (0, 255, 0), 2)
    if getattr(det, 'tool_mask', None) is not None and np.any(det.tool_mask):
        ys, xs = np.nonzero(det.tool_mask)
        end = int(ys.max())
        distal_x = xs[ys >= end - 100]
        x0, x1 = max(0, int(distal_x.min())-25), min(image.shape[1], int(distal_x.max())+26)
        y1 = min(image.shape[0], end+25); y0 = max(0, end-100)
        canvas = canvas[y0:y1, x0:x1]
        canvas = cv2.resize(canvas, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    canvas = cv2.copyMakeBorder(canvas, 54, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    cv2.putText(canvas, row['frame'], (5, 19), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"teeth={row['prongs_value']}/{row['expected_prongs']} contrast={row['contrast_value']:.1f} white={row['whiteness_value']:.3f}", (5, 43), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(destination), canvas)


def summarize(rows, out, report, config):
    eligible = [r for r in rows if not r['excluded']]
    lines = ['## T81 绝对判定实拍校核', '',
             f"范围：device 和 device2 全部 {len(rows)} 帧；按 axis_rms > 2 px 排除 {len(rows)-len(eligible)} 帧，剩余 {len(eligible)} 帧。axis_rms 缺失帧保留并按实际检测结果计入。", '',
             f"对比度门限 {config['contrast_min']:g}；工具灰度 ≥230 的比例上限 {config['saturation_max']:.1%}。各项单独计数，同一帧可出现在多项失败清单。", '',
             '修正后同时检查末端 35 像素内的灰度暗谷：局部突显度至少 15 灰度、暗于背景至少 40 灰度，连续 5 行的暗谷数均达到该数；算法不追踪跨行的单个暗谷。取灰度剖面与二值连通段两者较大值。该补充针对实拍中三齿被二值阈值粘连的情况，仍要求正面 3、侧面 2 齿；合成缺一齿和加噪回归测试用于验证不会只因一条轮廓就通过。', '',
             '| 判定 | 通过/总数 | 通过率 |', '|---|---:|---:|']
    for name in NAMES:
        passed = sum(bool(r[name]) for r in eligible)
        pct = 100 * passed / len(eligible) if eligible else 0
        lines.append(f'| {LABELS[name]} | {passed}/{len(eligible)} | {pct:.2f}% |')
    if len(rows) == 420 and any(r['frame'] == 'device/set_005/front_1.png' for r in rows):
        lines += ['', '本批修正前二值连通段齿数通过 315/412（76.46%）。正面 device/set_005 至 set_008 共 20 帧只有两个可分离的齿投影，可能存在遮挡，也不能排除缺齿；正面单独的齿数通过率低于 95%。不能仅凭该图像把两个投影判为三个，现场仍可能出现没有合格曝光档，需重新摆放或增加独立视角确认。']
    lines += ['', '### 按相机分开统计', '', '| 相机 | 判定 | 通过/总数 | 通过率 |', '|---|---|---:|---:|']
    for camera in ('front', 'side'):
        subset = [r for r in eligible if r['camera'] == camera]
        for name in NAMES:
            passed = sum(bool(r[name]) for r in subset)
            percent = 100 * passed / len(subset) if subset else 0
            lines.append(f'| {camera} | {LABELS[name]} | {passed}/{len(subset)} | {percent:.2f}% |')
    lines += ['', '### 各项失败帧', '']
    for name in NAMES:
        lines += [f'**{LABELS[name]}**', '']
        failed = [r for r in eligible if not r[name]]
        if not failed:
            lines += ['无。', '']; continue
        lines += ['| 组/相机/帧 | 数值或原因 |', '|---|---|']
        for r in failed:
            value = {'recognized': f"{r['diagnostic'] or '未精定位'}；axis_rms={r['axis_rms_px']}；对比度={r['contrast_value']:.1f}", 'prongs': f"{r['prongs_value']}/{r['expected_prongs']}",
                     'contrast': f"{r['contrast_value']:.1f}", 'whiteness': f"{r['whiteness_value']:.2%}"}[name]
            lines.append(f"| {r['frame']} | {value.replace('|', '/')} |")
        lines.append('')
    low = [r for r in eligible if '末端对比度不足' in r['diagnostic']]
    lines += ['### 检测器报告末端对比度不足', '',
              ('、'.join(f"{r['frame']}（{r['contrast_value']:.1f}）" for r in low) if low else '无。'), '',
              f'逐帧全部指标和排除标记见 [{out.name}/frames.csv]({out.name}/frames.csv)；失败样例叠加图见 `{out.name}/overlays/`。',
              '齿数来自末端轮廓近似，不能替代显微镜或逐齿标注；以上为固定合适曝光下的绝对门限校核，不是分档选档验证。', '']
    previous = report.read_text(encoding='utf-8') if report.exists() else '# 曝光回放\n'
    marker = '## T81 绝对判定实拍校核'
    previous = previous.split(marker)[0].rstrip()
    report.write_text(previous + '\n\n' + '\n'.join(lines), encoding='utf-8')
    return eligible


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--report', type=Path, default=REPORT)
    parser.add_argument('--config', type=Path, default=Path('config/nest_clear.yaml'))
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    tc = yaml.safe_load(args.config.read_text(encoding='utf-8'))['tool_calibration']
    scan = tc.get('exposure_scan') or {}
    config = {'detection': tc['detection'], 'contrast_min': scan.get('contrast_min', 40.), 'saturation_max': scan.get('saturation_max', .05)}
    paths = sorted(p for device in ('device', 'device2') for p in (args.data / device).glob('set_*/[fs]*_*.png') if p.name.startswith(('front_', 'side_')))
    if not paths: raise SystemExit('没有找到实拍帧')
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [(str(p), str(args.data), config) for p in paths]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(evaluate, jobs, chunksize=2))
    with (args.out / 'frames.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    eligible = summarize(rows, args.out, args.report, config)
    overlay_dir = args.out / 'overlays'; overlay_dir.mkdir(exist_ok=True)
    failures = [r for r in eligible if not all(r[name] for name in NAMES)]
    # Sample every failure category and camera, not just the first group.
    selected = []
    for name in NAMES:
        for camera in ('front', 'side'):
            selected += [r for r in failures if r['camera'] == camera and not r[name]][:2]
    seen = set()
    for row in selected:
        if row['frame'] in seen: continue
        seen.add(row['frame'])
        write_overlay(args.data / row['frame'], row, config, overlay_dir / (row['frame'].replace('/', '_')))
    print(f'frames={len(rows)} excluded={len(rows)-len(eligible)} included={len(eligible)}')
    for name in NAMES:
        passed = sum(bool(r[name]) for r in eligible)
        percent = 100 * passed / len(eligible) if eligible else 0.
        print(f'{name}: {passed}/{len(eligible)} = {percent:.2f}%')
    for camera in ('front', 'side'):
        subset = [r for r in eligible if r['camera'] == camera]
        print(camera, ', '.join(f'{name}={sum(bool(r[name]) for r in subset)}/{len(subset)}' for name in NAMES))
    print('report:', args.report)


if __name__ == '__main__': main()
