"""Pure Pillow annotation operations and persistence; no display required."""
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image, ImageDraw, ImageChops


class Document:
    def __init__(self, image, labels=None):
        self.image = image.convert('L')
        self.labels = labels.copy().convert('L') if labels is not None else Image.new('L', image.size)
        if self.labels.size != self.image.size:
            raise ValueError('标注尺寸与图片不一致')
        self.undo_stack, self.redo_stack = [], []
        self.manual = None
        self.threshold_enabled, self.threshold = False, 127

    def apply(self, shape, value, threshold=False):
        if threshold and value:
            allowed = self.image.point(lambda p: 255 if p <= self.threshold else 0)
            shape = ImageChops.multiply(shape, allowed)
        before = self.labels.copy()
        self.labels.paste(value, mask=shape)
        if ImageChops.difference(before, self.labels).getbbox():
            self.undo_stack.append(before)
            self.undo_stack = self.undo_stack[-30:]
            self.redo_stack.clear()

    def brush(self, points, size, value, threshold=False):
        if not points: return
        mask = Image.new('L', self.image.size)
        draw = ImageDraw.Draw(mask)
        # Round ends and joins, with the requested diameter in image pixels.
        r = (size - 1) / 2
        if len(points) > 1: draw.line(points, fill=255, width=size, joint='curve')
        for x, y in points:
            draw.rectangle((round(x), round(y), round(x), round(y)), fill=255) if size == 1 else draw.ellipse((round(x-r), round(y-r), round(x-r)+size-1, round(y-r)+size-1), fill=255)
        self.apply(mask, value, threshold)

    def polygon(self, points, value, threshold=False):
        if len(points) < 3: return
        mask = Image.new('L', self.image.size)
        ImageDraw.Draw(mask).polygon(points, fill=255)
        self.apply(mask, value, threshold)

    def undo(self):
        if self.undo_stack:
            self.redo_stack.append(self.labels.copy()); self.labels = self.undo_stack.pop()

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append(self.labels.copy()); self.labels = self.redo_stack.pop()

    def derive(self):
        # Principal axis of the body, oriented toward the tip-mask centroid.
        # PCA uses all body pixels and supports needles at any rotation.
        w, h = self.labels.size
        counts = [0, 0]; sums = [[0., 0.], [0., 0.]]
        xx = xy = yy = 0.
        data = self.labels.load()
        for y in range(h):
            for x in range(w):
                v = data[x, y]
                if v == 2:
                    counts[0] += 1; sums[0][0] += x; sums[0][1] += y
                    xx += x*x; xy += x*y; yy += y*y
                elif v == 3:
                    counts[1] += 1; sums[1][0] += x; sums[1][1] += y
        if not counts[0] or not counts[1]:
            missing = '针身' if not counts[0] else '针尖'
            return {'status': '缺少标注：' + missing}
        n = counts[0]; cx, cy = sums[0][0]/n, sums[0][1]/n
        a, b, c = xx/n-cx*cx, xy/n-cx*cy, yy/n-cy*cy
        if n < 2 or (abs(a-c) < 1e-12 and abs(b) < 1e-12):
            return {'status': '针身标注不足，无法确定中轴线'}
        angle = .5 * math.atan2(2*b, a-c); dx, dy = math.cos(angle), math.sin(angle)
        tx, ty = sums[1][0]/counts[1]-cx, sums[1][1]/counts[1]-cy
        if dx*tx + dy*ty < 0: dx, dy = -dx, -dy
        best = float('-inf'); far = None
        for y in range(h):
            for x in range(w):
                if data[x, y] == 3:
                    projection = (x-cx)*dx + (y-cy)*dy
                    if projection > best: best, far = projection, [x, y]
        return {'status': 'ok', 'axis': {'point': [cx, cy], 'direction': [dx, dy]},
                'farthest_tip': far, 'tool_tip': [cx+best*dx, cy+best*dy]}


def paths(path):
    path = Path(path); base = path.parent / 'annotations' / path.stem
    return Path(str(base) + '.labels.png'), Path(str(base) + '.json')


def save(path, doc):
    labels, metadata = paths(path); labels.parent.mkdir(parents=True, exist_ok=True)
    result = doc.derive(); hist = doc.labels.histogram()
    record = {'pixel_counts': {'背景': hist[0], '黑胶': hist[1], '针身': hist[2], '针尖': hist[3]},
              'derived': result, 'manual_tip': doc.manual,
              'threshold_enabled': doc.threshold_enabled, 'threshold': doc.threshold,
              'saved_at': datetime.now(timezone.utc).isoformat()}
    doc.labels.save(labels); metadata.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')


def load(path):
    image = Image.open(path); labels, metadata = paths(path)
    doc = Document(image, Image.open(labels) if labels.exists() else None)
    if metadata.exists():
        record = json.loads(metadata.read_text(encoding='utf-8'))
        doc.manual = record.get('manual_tip'); doc.threshold = record.get('threshold', 127)
        doc.threshold_enabled = record.get('threshold_enabled', False)
    return doc
