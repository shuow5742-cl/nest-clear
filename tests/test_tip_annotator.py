import sys
from pathlib import Path
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools' / 'tip_annotator'))
from core import Document, load, save, paths


def test_brush_polygon_threshold_and_history():
    image = Image.new('L', (40, 40), 200)
    ImageDraw.Draw(image).rectangle((0, 0, 19, 39), fill=60)
    doc = Document(image)
    doc.brush([(5, 5), (30, 5)], 3, 1, True)
    assert doc.labels.getpixel((5, 5)) == 1
    assert doc.labels.getpixel((30, 5)) == 0
    doc.polygon([(10, 10), (30, 10), (30, 30), (10, 30)], 2, True)
    assert doc.labels.getpixel((15, 15)) == 2
    assert doc.labels.getpixel((25, 15)) == 0
    doc.undo(); assert doc.labels.getpixel((15, 15)) == 0
    doc.redo(); assert doc.labels.getpixel((15, 15)) == 2
    doc.brush([(15, 15)], 1, 0); assert doc.labels.getpixel((15, 15)) == 0
    doc.undo(); assert doc.labels.getpixel((15, 15)) == 2
    for i in range(35): doc.brush([(i, 38)], 1, 3)
    assert len(doc.undo_stack) >= 30


def test_roundtrip(tmp_path):
    p = tmp_path / 'frame.png'; Image.new('L', (30, 40), 80).save(p)
    doc = load(p); doc.brush([(5, 5), (5, 20)], 3, 2); doc.brush([(5, 25)], 1, 3)
    doc.manual = [5.25, 27.5]; doc.threshold = 88; doc.threshold_enabled = True
    save(p, doc); other = load(p)
    assert other.labels.tobytes() == doc.labels.tobytes()
    assert other.manual == doc.manual and other.threshold == 88 and other.threshold_enabled
    assert all(x.exists() for x in paths(p))


def test_three_teeth_and_missing():
    doc = Document(Image.new('L', (100, 120)))
    assert '缺少标注' in doc.derive()['status']
    draw = ImageDraw.Draw(doc.labels); draw.rectangle((45, 10, 55, 80), fill=2)
    assert '缺少标注' in doc.derive()['status']
    for x, end in [(46, 100), (50, 106), (54, 102)]: draw.line((x, 81, x, end), fill=3)
    result = doc.derive(); x, y = result['tool_tip']
    assert abs(x-50) < .5 and abs(y-106) < .5
    assert result['farthest_tip'] == [50, 106]
