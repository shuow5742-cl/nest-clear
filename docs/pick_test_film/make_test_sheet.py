#!/usr/bin/env python3
"""Generate the 1:1 A4 main-flow test sheet and its design coordinates."""
import csv
import itertools
import math
from pathlib import Path

import cv2
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

HERE = Path(__file__).resolve().parent
PDF = HERE / '燕窝二代_主流程测试纸_A4打印版_20261123.pdf'
pdfmetrics.registerFont(TTFont('CJK', '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'))
c = canvas.Canvas(str(PDF), pagesize=(297 * mm, 210 * mm), pageCompression=1)

def pt(x, y):
    return (x + 148.5) * mm, (y + 105) * mm

def line(x1, y1, x2, y2, width):
    c.setLineWidth(width * mm)
    c.line(*pt(x1, y1), *pt(x2, y2))

def runs(s, special='·'):
    return [('Helvetica' if latin else 'CJK', ''.join(group)) for latin, group in
            itertools.groupby(s, key=lambda ch: ord(ch) < 128 or ch in special)]

def centered_text(parts, x, y, size):
    width = sum(pdfmetrics.stringWidth(s, font, size) for font, s in parts)
    t = c.beginText(x - width / 2, y)
    for font, s in parts:
        t.setFont(font, size)
        t.textOut(s)
    c.drawText(t)
    return width

c.setStrokeGray(.5)
c.setLineWidth(.1 * mm)
c.rect(26 * mm, 45 * mm, 245 * mm, 120 * mm, stroke=1, fill=0)
c.setStrokeGray(0)
c.setFillGray(0)
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
markers = [(cv2.aruco.generateImageMarker(dictionary, i, 6) == 0) for i in range(32)]
rows = [(i, f'T{i % 32:02d}', -105 + 15 * col, 42 - 15 * row, i % 32)
        for row in range(7) for col in range(15) for i in [row * 15 + col]]
for i, label, x, y, marker_id in rows:
    # Conservative font metric rectangle, in pattern coordinates.
    width = pdfmetrics.stringWidth(label, 'Helvetica', 1.6 * mm) / mm
    ascent, descent = pdfmetrics.getAscentDescent('Helvetica', 1.6 * mm)
    left, right = x - 4.8, x - 4.8 + width
    bottom, top = y - 4.6 + descent / mm, y - 4.6 + ascent / mm
    dx = max(left - x, 0, x - right)
    dy = max(bottom - y, 0, y - top)
    assert math.hypot(dx, dy) >= 2.6
    assert dx >= 1.0 and dy >= 1.0
    for _, _, ox, oy, _ in rows:
        if (ox, oy) == (x, y):
            continue
        mx1, mx2, my1, my2 = ox + 5, ox + 9, oy + 5, oy + 9
        gap_x = max(mx1 - right, left - mx2, 0)
        gap_y = max(my1 - top, bottom - my2, 0)
        assert math.hypot(gap_x, gap_y) >= .8
    c.setLineWidth(.3 * mm)
    c.circle(*pt(x, y), 1.85 * mm, stroke=1, fill=0)
    line(x - 5, y, x + 5, y, .1)
    line(x, y - 5, x, y + 5, .1)
    for r in range(6):
        for q in range(6):
            if markers[marker_id][r][q]:
                a, b = pt(x + 5 + q * 4 / 6, y + 9 - (r + 1) * 4 / 6)
                c.rect(a, b, 4 / 6 * mm, 4 / 6 * mm, stroke=0, fill=1)
    c.setFont('Helvetica', 1.6 * mm)
    c.drawString(*pt(x - 4.8, y - 4.6), label)

p = c.beginPath()
p.moveTo(*pt(-120, 55)); p.lineTo(*pt(-115, 55)); p.lineTo(*pt(-120, 50)); p.close()
c.drawPath(p, stroke=0, fill=1)
header = '燕窝二代 主流程测试纸 · 105 目标 @15mm · 编号 T00–T31 循环 · 放置方向不限 · 仅用于主流程测试模式，不可用于工具偏移/挑取精度标定 · 2026-11-23'
parts = runs(header, '·–')
size = 2.5 * mm
width = sum(pdfmetrics.stringWidth(s, font, size) for font, s in parts)
size *= min(1, 240 * mm / width)
assert sum(pdfmetrics.stringWidth(s, font, size) for font, s in parts) <= 240 * mm + 1e-8
centered_text(parts, 148.5 * mm, (105 + 55.5) * mm, size)

def ruler(x, y, length, vertical=False):
    c.setStrokeGray(0); c.setLineWidth(.12 * mm)
    if vertical: c.line(x * mm, y * mm, x * mm, (y + length) * mm)
    else: c.line(x * mm, y * mm, (x + length) * mm, y * mm)
    for i in range(length + 1):
        tick = 2.5 if i % 10 == 0 else 1.6 if i % 5 == 0 else .8
        if vertical: c.line(x * mm, (y + i) * mm, (x + tick) * mm, (y + i) * mm)
        else: c.line((x + i) * mm, y * mm, (x + i) * mm, (y + tick) * mm)
        if i % 10 == 0:
            c.setFont('Helvetica', 1.5 * mm)
            if vertical: c.drawString((x + 3) * mm, (y + i - .5) * mm, str(i))
            else: c.drawCentredString((x + i) * mm, (y - 2.5) * mm, str(i))
    c.setFont('Helvetica', 2 * mm)
    if vertical: c.drawString((x - 1) * mm, (y + length + 3) * mm, f'Y {length} mm')
    else: c.drawCentredString((x + length / 2) * mm, (y + 5) * mm, f'X {length} mm')

ruler(48, 23, 100); ruler(183, 23, 50)
ruler(12, 55, 100, True); ruler(278, 80, 50, True)
instruction = "打印时选择'实际大小/100%'，不要'适合页面'；打印后用钢尺量校验尺 100mm 应为 100.0±0.3mm，不符请调打印设置。贴纸只贴图案外四边，保持平整，不得被针尖压动。"
parts = runs(instruction, '±')
assert sum(pdfmetrics.stringWidth(s, font, 1.75 * mm) for font, s in parts) < 287 * mm
centered_text(parts, 148.5 * mm, 190 * mm, 1.75 * mm)
c.setFont('Helvetica', 2 * mm)
c.drawCentredString(148.5 * mm, 180 * mm, 'A4 print version 20261123')
c.showPage(); c.save()
with (HERE / 'test_sheet_targets.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['index', 'label', 'x_mm', 'y_mm', 'aruco_id'])
    writer.writerows(rows)
