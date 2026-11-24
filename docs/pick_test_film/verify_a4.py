#!/usr/bin/env python3
"""Raster checks for the A4 print sheet; run from this directory."""
import csv, math, re, subprocess, tempfile
from pathlib import Path
import cv2
import numpy as np

pdf = '燕窝二代_挑取测试菲林片_A4打印版_20261123.pdf'
original = '燕窝二代_挑取精度测试菲林片_V2_245x120.pdf'
scale = 20
with tempfile.TemporaryDirectory() as tmp:
    images = []
    for name, label in [(pdf, 'a4'), (original, 'original')]:
        prefix = str(Path(tmp) / label)
        subprocess.run(['pdftocairo', '-png', '-r', '508', '-singlefile', name, prefix], check=True)
        images.append(cv2.imread(prefix + '.png', cv2.IMREAD_GRAYSCALE))
a4, orig = images
# V1/V2 的每个目标局部（含编号和 ArUco）在同一渲染网格上逐像素比对。
with tempfile.TemporaryDirectory() as tmp:
    prefix = str(Path(tmp) / 'v1')
    subprocess.run(['pdftocairo', '-png', '-r', '508', '-singlefile',
                    '燕窝二代_挑取精度测试菲林片_V1_245x120.pdf', prefix], check=True)
    v1 = cv2.imread(prefix + '.png', cv2.IMREAD_GRAYSCALE)
v1_equal = True
for y in (45, 15, -15, -45):
    for x in (-105, -75, -45, -15, 15, 45, 75, 105):
        # 覆盖十字、圆环、编号、坐标文字和码，避开箭头及页眉。
        x0, x1 = int((x+132.5-8)*scale), int((x+132.5+10)*scale)
        y0, y1 = int((70-y-10)*scale), int((70-y+10)*scale)
        v1_equal &= np.array_equal(v1[y0:y1, x0:x1], orig[y0:y1, x0:x1])
rows = list(csv.DictReader(open('targets.csv', encoding='utf-8')))
corners, ids, _ = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)).detectMarkers(a4)
ids = [] if ids is None else ids.flatten().tolist()
errors = []
for corner, i in zip(corners, ids):
    x, y = corner[0].mean(axis=0)
    errors.append(math.hypot((x+.5)/scale-148.5-float(rows[i]['aruco_center_x_mm']), 105-(y+.5)/scale-float(rows[i]['aruco_center_y_mm'])))
# Fit the two ring boundaries from radial intensity crossings, avoiding cross arms.
ring_errors = []
for row in rows:
    x, y = float(row['x_mm']), float(row['y_mm'])
    cx, cy = (x+148.5)*scale-.5, (105-y)*scale-.5
    boundaries = [[], []]
    for angle in np.linspace(0, 2*np.pi, 720, endpoint=False):
        if min(abs(np.sin(angle)), abs(np.cos(angle))) < .13: continue
        rr = np.arange(1.45, 2.25, 1/(scale*4))
        mx = (cx+rr*scale*np.cos(angle)).astype(np.float32)
        my = (cy+rr*scale*np.sin(angle)).astype(np.float32)
        vals = cv2.remap(a4, mx.reshape(1,-1), my.reshape(1,-1), cv2.INTER_LINEAR)[0]
        hits = np.flatnonzero(vals < 128)
        if len(hits) < 2: continue
        for points, k in zip(boundaries, (hits[0], hits[-1])):
            points.append([cx+rr[k]*scale*np.cos(angle), cy+rr[k]*scale*np.sin(angle)])
    centers = []
    for points in boundaries:
        p = np.array(points); A = np.column_stack((2*p[:,0], 2*p[:,1], np.ones(len(p))))
        centers.append(np.linalg.lstsq(A, (p*p).sum(axis=1), rcond=None)[0][:2])
    ring_errors.append(np.linalg.norm(np.mean(centers, axis=0)-[cx,cy])/scale)
# Read every tick from a strip above/right of its baseline; fit intensity centroids.
tick_errors = []
for x,y,length,vertical in [(48,23,100,False),(183,23,50,False),(12,55,100,True),(278,80,50,True)]:
    centers=[]
    for i in range(length+1):
        px=(x+(0.5 if vertical else i))*scale-.5
        py=(210-y-(i if vertical else 0.5))*scale-.5
        if vertical:
            k=round(py); profile=255-a4[k-6:k+7,round(px)-2:round(px)+3].mean(axis=1); coords=np.arange(k-6,k+7)
        else:
            k=round(px); profile=255-a4[round(py)-2:round(py)+3,k-6:k+7].mean(axis=0); coords=np.arange(k-6,k+7)
        assert profile.sum()>100, (x,y,i)
        centers.append(float(coords@profile/profile.sum()))
    steps=np.abs(np.diff(centers))/scale
    tick_errors.extend(abs(steps-1))
# Compare the common pattern area at identical scale, excluding the boundary stroke.
pattern_a4=a4[46*scale:164*scale,27*scale:270*scale]
pattern_orig=orig[11*scale:129*scale,11*scale:254*scale]
diff=np.abs(pattern_a4.astype(np.int16)-pattern_orig.astype(np.int16))
info=subprocess.check_output(['pdfinfo',pdf],text=True)
m=re.search(r'Page size:\s+([\d.]+) x ([\d.]+) pts',info)
dims=[float(m.group(i))*25.4/72 for i in (1,2)]
# Font rasterization can vary slightly with translation; require changed pixels
# to remain within one pixel of existing ink in the other image.
kernel = np.ones((3,3), np.uint8)
a_ink = (pattern_a4 < 200).astype(np.uint8)
b_ink = (pattern_orig < 200).astype(np.uint8)
unmatched = int(np.sum(a_ink * (1-cv2.dilate(b_ink,kernel))) + np.sum(b_ink * (1-cv2.dilate(a_ink,kernel))))
passed=(v1_equal and sorted(ids)==list(range(32)) and max(errors)<.05 and max(ring_errors)<.05 and max(tick_errors)<.05 and all(abs(a-b)<.01 for a,b in zip(dims,(297,210))) and unmatched==0)
report=f'''\nV1 与 V2：32 个目标中心、编号及 ArUco 局部逐像素一致：{v1_equal}。\nA4 打印版 20261123 自检（pdftocairo 508 dpi，20 px/mm）：
PDF 页面：{dims[0]:.5f} × {dims[1]:.5f} mm（要求 297 × 210 mm）。
ArUco：检出 {len(ids)}/32，唯一 ID {len(set(ids))} 个，与 0–31 一致：{sorted(ids)==list(range(32))}；码中心最大偏差 {max(errors):.5f} mm。
32 个目标圆环中心相对 targets.csv 名义坐标最大偏差：{max(ring_errors):.5f} mm（要求 < 0.05 mm）。
X、Y 的 100 mm 和 50 mm 校验尺：逐一检测全部 304 个毫米刻度，相邻间距最大偏差 {max(tick_errors):.5f} mm。
原菲林图案内部区域同分辨率像素比较：最大灰度差 {np.max(diff)}；偏离另一图墨迹超过 1 像素的像素数 {unmatched}（文字边缘存在抗锯齿差异）。
方法：ArUco 角点均值；避开十字线的圆环内外边缘拟合圆；刻度线灰度重心。\n最终结论：{'全部通过' if passed else '未通过'}。\n'''
path=Path('自检结果.txt'); old=path.read_text(encoding='utf-8'); old=old.split('\nV1 与 V2：')[0].split('\nA4 打印版 20261123 自检')[0]; path.write_text(old.rstrip()+'\n'+report,encoding='utf-8')
print(report)
if not passed: raise SystemExit(1)
