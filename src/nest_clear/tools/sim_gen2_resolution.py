"""在固定随机图块上比较参考尺度与二代尺度，输出可复查数据。"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import math
from pathlib import Path
import random

import cv2
import numpy as np
from nest_clear.vision.geometry import Calibration
from nest_clear.vision.hair import HairDetectConfig, detect_candidates, gen1_pixel_gates, scaled
from nest_clear.vision.hair_integration import measure

REF = 6.959
GEN2 = 23.585
RATIO = REF / GEN2
THRESHOLD_PX = 0.3 * 1000 / REF


def _calibration(scale):
    return Calibration(np.array([[1000., 0, 0], [0, 1000., 0], [0, 0, 1.]]), np.zeros(5),
                       np.diag([1., -1., -1.]), np.array([0., 0., scale]), 0.)


def _touches(poly, width, height):
    return any(x <= 0 or y <= 0 or x >= width - 1 or y >= height - 1 for x, y in poly)


def _serialize(c):
    return {'polygon': c.polygon, 'pick': c.pick_uv, 'source': c.source}


def _run(path):
    cv2.setNumThreads(1)
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f'无法读取 {path}')
    h, w = image.shape[:2]
    small = cv2.resize(image, None, fx=RATIO, fy=RATIO, interpolation=cv2.INTER_AREA)
    ref_cfg = scaled(HairDetectConfig(um_per_px=REF))
    gen_cfg = scaled(HairDetectConfig(um_per_px=GEN2))
    refs = [c for c in detect_candidates(image, ref_cfg) if gen1_pixel_gates(c, ref_cfg)[0]]
    found = [c for c in detect_candidates(small, gen_cfg) if measure(c, (0., 0., 0.), _calibration(GEN2), gen_cfg).reason is None]
    # 用实际缩放后的像素中心坐标映回原图；原图块须为同一分辨率。
    sx, sy = w / small.shape[1], h / small.shape[0]
    edges = []
    for i, ref in enumerate(refs):
        ref_poly = np.asarray(ref.polygon, np.int32)
        if len(ref_poly) < 3:
            continue
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [ref_poly], 255)
        dist = cv2.distanceTransform(255-mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        for j, other in enumerate(found):
            poly = np.rint(np.array(other.polygon) * [sx, sy]).astype(np.int32)
            if len(poly) < 3:
                continue
            mask2 = np.zeros((h, w), np.uint8)
            cv2.fillPoly(mask2, [poly], 255)
            d = float(dist[mask2 > 0].min()) if np.any(mask2) else float('inf')
            pick = (other.pick_uv[0] * sx, other.pick_uv[1] * sy)
            pick_dist = max(0., -cv2.pointPolygonTest(ref_poly, pick, True))
            if min(d, pick_dist) <= THRESHOLD_PX:
                edges.append((min(d, pick_dist), i, j, pick_dist * REF / 1000,
                              math.dist(pick, ref.pick_uv) * REF / 1000))
    pairs, used_ref, used_gen = [], set(), set()
    for _, i, j, on_hair, along in sorted(edges):
        if i not in used_ref and j not in used_gen:
            used_ref.add(i); used_gen.add(j)
            pairs.append({'ref': i, 'gen': j, 'on_hair_mm': on_hair, 'along_mm': along})
    return {'path': path, 'size': [w, h], 'small_size': [small.shape[1], small.shape[0]],
            'ref': [{**_serialize(c), 'truncated': _touches(c.polygon, w, h)} for c in refs],
            'gen': [{**_serialize(c), 'truncated': _touches(c.polygon, small.shape[1], small.shape[0])} for c in found],
            'pairs': pairs}


def _stats(rows, truncated):
    n = matched = extra = 0
    errors, along = [], []
    for row in rows:
        refs = [i for i, r in enumerate(row['ref']) if r['truncated'] == truncated]
        pairs = [p for p in row['pairs'] if p['ref'] in refs]
        n += len(refs); matched += len(pairs)
        used = {p['gen'] for p in row['pairs']}
        extra += sum(i not in used and g['truncated'] == truncated for i, g in enumerate(row['gen']))
        errors.extend(p['on_hair_mm'] for p in pairs)
        along.extend(p['along_mm'] for p in pairs)
    def stats(values):
        return {'mean': float(np.mean(values)), 'p95': float(np.percentile(values, 95)), 'max': float(max(values))} if values else None
    return {'reference': n, 'matched': matched, 'recall': matched / n if n else None, 'extra': extra,
            'on_hair_mm': stats(errors), 'along_hair_mm': stats(along)}


def _example(row, out, index):
    image = cv2.imread(row['path']); h, w = image.shape[:2]
    small = cv2.resize(image, None, fx=RATIO, fy=RATIO, interpolation=cv2.INTER_AREA)
    right = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    for c in row['ref']:
        cv2.polylines(image, [np.array(c['polygon'], np.int32)], True, (0, 255, 0), 2)
        cv2.drawMarker(image, tuple(map(round, c['pick'])), (0, 0, 255))
    sx, sy = w / small.shape[1], h / small.shape[0]
    for c in row['gen']:
        poly = np.rint(np.array(c['polygon']) * [sx, sy]).astype(np.int32)
        cv2.polylines(right, [poly], True, (0, 255, 0), 2)
        cv2.drawMarker(right, (round(c['pick'][0]*sx), round(c['pick'][1]*sy)), (0, 0, 255))
    path = out / f'example_{index}.png'
    cv2.imwrite(str(path), np.hstack([image, right]))
    return str(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patches', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261117)
    parser.add_argument('--count', type=int, default=3000)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.patches.glob('*.png'))
    if len(paths) < args.count:
        raise ValueError(f'图块不足：期望 {args.count}，实际 {len(paths)}')
    selected = random.Random(args.seed).sample(paths, args.count)
    (args.out / 'sample_manifest.txt').write_text('\n'.join(str(p) for p in selected) + '\n', encoding='utf-8')
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, row in enumerate(executor.map(_run, map(str, selected), chunksize=4), 1):
            rows.append(row)
            if index % 100 == 0:
                print(f'已完成 {index}/{args.count}', flush=True)
    (args.out / 'results.json').write_text(json.dumps(rows, ensure_ascii=False), encoding='utf-8')
    stats = {'untruncated': _stats(rows, False), 'truncated': _stats(rows, True)}
    ranked = [r for r in rows if any(not c['truncated'] and not any(p['ref'] == i for p in r['pairs']) for i, c in enumerate(r['ref']))]
    if len(ranked) < 5:
        ranked += [r for r in rows if r['ref'] and r not in ranked]
    examples = [_example(row, args.out, i+1) for i, row in enumerate(ranked[:5])]
    report = {'seed': args.seed, 'images': len(rows), 'reference_um_per_px': REF, 'gen2_um_per_px': GEN2,
              'match_threshold_mm': .3, 'stats': stats, 'examples': examples,
              'note': '定位误差为二代挑点到一代候选轮廓距离；沿毛挑点差异不等于定位误差。参考端使用一代像素挑取门，二代使用工作面物理门；无真图及光学退化模拟。'}
    (args.out / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)

if __name__ == '__main__':
    main()
