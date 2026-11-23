"""换针对刀图像标注：基准、针尖与画面方向偏差。"""
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from nest_clear.display import format_number


def render_tooltip(image, target, point, *, um_per_px=15.2, radius_mm=.05, manual=False, threshold=127):
    """绘制两相机共用的标注，偏差采用未取整的原始像素。"""
    image = np.asarray(image)
    rendered = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours((gray <= threshold).astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(rendered, [max(contours, key=cv2.contourArea)], -1, (0, 165, 255), 1)
    t = tuple(round(float(v)) for v in target)
    radius = max(1, round(radius_mm * 1000 / um_per_px))
    cv2.circle(rendered, t, radius, (0, 160, 0), 1)
    cv2.drawMarker(rendered, t, (180, 0, 180), cv2.MARKER_CROSS, 12, 1)
    if point is not None:
        p = tuple(round(float(v)) for v in point)
        cv2.circle(rendered, p, 3, (210, 0, 180) if manual else (0, 140, 255), -1)
    # 放大基准局部，确保小于数像素的范围圈仍可检查。
    height, width = rendered.shape[:2]
    inset = min(150, max(80, width // 4), height // 2)
    span = 40
    x0, y0 = t[0] - span // 2, t[1] - span // 2
    padded = cv2.copyMakeBorder(rendered, span, span, span, span, cv2.BORDER_CONSTANT, value=(245,245,245))
    crop = padded[y0+span:y0+span+span, x0+span:x0+span+span]
    if crop.size and width >= inset + 8:
        crop = cv2.resize(crop, (inset, inset), interpolation=cv2.INTER_NEAREST)
        rendered[8:8+inset, width-inset-8:width-8] = crop
        cv2.rectangle(rendered, (width-inset-8, 8), (width-8, 8+inset), (160,160,160), 1)
    rgb = Image.fromarray(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(rgb)
    size = max(14, min(24, width // 42))
    font = next((ImageFont.truetype(str(p), size) for p in (Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'), Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')) if p.is_file()), ImageFont.load_default())
    if point is None:
        lines = [('对刀基准点 · 基准范围圈', (95,0,110)), ('状态：未识别针尖', (180,35,24))]
    else:
        du, dv = ((float(point[i])-float(target[i])) * um_per_px / 1000 for i in (0, 1))
        inside = math.hypot(du, dv) <= radius_mm
        color = (20,130,65) if inside else (180,35,24)
        lines = [('对刀基准点 · 基准范围圈', (95,0,110)),
                 (f'横向 {format_number(du, signed=True)} mm  竖向 {format_number(dv, signed=True)} mm', color),
                 ('状态：在圈内' if inside else '状态：超出范围圈', color),
                 ('紫色：人工点选' if manual else '橙色：自动识别', (130,0,140) if manual else (155,85,0))]
    for i, (text, color) in enumerate(lines):
        draw.text((8, 8 + i*(size+4)), text, font=font, fill=color, stroke_width=1, stroke_fill=(255,255,255))
    return cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
