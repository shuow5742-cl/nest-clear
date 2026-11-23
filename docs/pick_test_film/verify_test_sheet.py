#!/usr/bin/env python3
"""Render at camera scale and validate with the production detector."""
import csv
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'src'))
from nest_clear.tooloffset.detection import detect

PDF = HERE / '燕窝二代_主流程测试纸_A4打印版_20261123.pdf'
SCALE = 968 / 25.4
rows = list(csv.DictReader((HERE / 'test_sheet_targets.csv').open(encoding='utf-8')))
design = np.array([[float(r['x_mm']), float(r['y_mm'])] for r in rows])
assert len(rows) == 105
with tempfile.TemporaryDirectory(prefix='nest_test_sheet_', dir='/tmp') as tmp:
    prefix = str(Path(tmp) / 'sheet')
    subprocess.run(['pdftoppm', '-r', '968', '-gray', '-png', '-singlefile', str(PDF), prefix], check=True)
    page = cv2.imread(prefix + '.png', cv2.IMREAD_GRAYSCALE)
    if page is None:
        raise RuntimeError('Failed to load rendered page')
    preview = cv2.resize(page, (2000, round(page.shape[0] * 2000 / page.shape[1])), interpolation=cv2.INTER_AREA)
    assert cv2.imwrite(str(HERE / 'test_sheet_preview.png'), preview)
    x0, x1 = round(26 * SCALE), round(271 * SCALE)
    y0, y1 = round(45 * SCALE), round(165 * SCALE)
    pattern = page[y0:y1, x0:x1]
    rng = np.random.default_rng(20261123)
    blurred = cv2.GaussianBlur(pattern, (0, 0), 1)
    noisy = np.clip(blurred.astype(np.float32) + rng.normal(0, 4, blurred.shape), 0, 255).astype(np.uint8)
    passed = True
    for name, image in [('原图', pattern), ('模糊 sigma=1px + 噪声 sigma=4', noisy)]:
        targets = detect(image)
        positions = np.array([[(t.ring_uv[0] + x0 + .5) / SCALE - 148.5,
                               105 - (t.ring_uv[1] + y0 + .5) / SCALE] for t in targets]).reshape(-1, 2)
        distances = np.linalg.norm(positions[:, None, :] - design[None, :, :], axis=2)
        nearest = np.argmin(distances, axis=1)
        match_dist = distances[np.arange(len(targets)), nearest]
        ids_ok = all(t.index == int(rows[j]['aruco_id']) for t, j in zip(targets, nearest))
        unique = len(set(nearest.tolist())) == 105
        if len(targets):
            residuals = positions - design[nearest]
            residuals -= residuals.mean(axis=0)
            max_error = float(np.linalg.norm(residuals, axis=1).max())
            max_cross = max(t.cross_error_mm for t in targets)
            max_match = float(match_dist.max())
        else:
            max_error = max_cross = max_match = math.inf
        ok = len(targets) == 105 and unique and ids_ok and max_match < 1 and max_error <= .05 and max_cross <= .05
        passed &= ok
        print(f'{name}: 检出 {len(targets)}/105，位置唯一={unique}，ID 一致={ids_ok}，最大匹配距离={max_match:.5f} mm，平移拟合后最大圆环偏差={max_error:.5f} mm，最大 cross_error={max_cross:.5f} mm；{"通过" if ok else "失败"}')
    for x, y, vertical, name in [(48, 23, False, 'X'), (12, 55, True, 'Y')]:
        centers = []
        for i in (0, 100):
            px = (x + (.5 if vertical else i)) * SCALE - .5
            py = (210 - y - (i if vertical else .5)) * SCALE - .5
            if vertical:
                k = round(py); coords = np.arange(k - 12, k + 13)
                profile = 255 - page[k-12:k+13, round(px)-3:round(px)+4].mean(axis=1)
            else:
                k = round(px); coords = np.arange(k - 12, k + 13)
                profile = 255 - page[round(py)-3:round(py)+4, k-12:k+13].mean(axis=0)
            if profile.sum() <= 100:
                raise RuntimeError(f'{name} ruler tick {i} missing')
            centers.append(float(coords @ profile / profile.sum()))
        length = abs(centers[1] - centers[0]) / SCALE
        ok = abs(length - 100) <= .1
        passed &= ok
        print(f'{name} 100mm 校验尺: {length:.5f} mm；{"通过" if ok else "失败"}')
    print('PASS' if passed else 'FAIL')
    if not passed:
        sys.exit(1)
