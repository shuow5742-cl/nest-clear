import json
import math
import pytest
from datetime import date
from nest_clear.ui.model import decode_status, status_indicators, classify_event, EventTail, read_json, camcal_progress, tooltip_progress

@pytest.mark.parametrize('mode,page', [
    ('相机标定中', '相机标定'), ('工具偏移标定中', '工具偏移标定'),
    ('挑取精度标定中', '挑取精度标定'), ('对刀基准建立中', '对刀基准建立'),
    ('对刀中(工具1)', '换针对刀'), ('对刀中(工具2)', '换针对刀'),
    ('挑毛中', '主页'), ('空闲', None), ('未知', None), (None, None),
])
def test_page_for_mode(mode, page):
    from nest_clear.ui.model import page_for_mode
    assert page_for_mode({'mode': mode}) == page
    assert page_for_mode({}) is None

def test_recent_replies_limit_and_order(tmp_path):
    from nest_clear.ui.status import MainStatus
    status = MainStatus(tmp_path)
    assert status.data['recent_replies'] == []
    status.observe({'event': 'request', 'func': 60})
    for i in range(7):
        status.observe({'event': 'reply', 'wall': str(i), 'task_id': i, 'reply_func': i})
    status.flush()
    assert status.data['recent_replies'] == [
        {'time': str(i), 'task_id': i, 'request_func': 60, 'reply_func': i} for i in range(6, 1, -1)]
    assert status.data['last_reply_func'] == 6

def test_decode_and_status():
    bits=decode_status((1<<0)|(1<<6)|(1<<10)|(1<<11))
    assert [bits[i]['color'] for i in (0,1,6,10,11)] == ['green','gray','red','yellow','red']
    assert status_indicators({}, {})['PLC 连接'][0]=='gray'
    assert status_indicators({'plc_connected':False}, {})['PLC 连接'][0]=='red'

def test_categories():
    for row, category in [({'event':'camcal_failed'},'标定'),({'event':'plc_disconnected'},'通信'),({'event':'error','code':'CONFIG-X'},'配置'),({'event':'error','code':'CAMERA-X'},'相机'),({'event':'supervisor_reply_999','process':'supervisor'},'父程序'),({'event':'error'},'算法')]:
        assert classify_event(row)['category']==category
    assert classify_event({'event':'capture'}) is None
    assert classify_event({'event':'plc_restored'})['color']=='green'

def test_tail_partial_cross_runs_and_days(tmp_path):
    day=date(2026,1,1); d=tmp_path/day.isoformat(); a=d/'run1'/'events.jsonl'; b=d/'run2'/'events.jsonl'; a.parent.mkdir(parents=True); b.parent.mkdir(parents=True)
    a.write_text('{"event":"error","wall":"1"}\n{"event":',encoding='utf8'); b.write_text('{"event":"camcal_failed","wall":"2"}\n',encoding='utf8')
    tail=EventTail(tmp_path); assert [x['time'] for x in tail.poll(day)]==['2','1']
    with a.open('a') as f: f.write('"error","wall":"3"}\n')
    assert [x['time'] for x in tail.poll(day)]==['3','2','1']; assert len(tail.poll(day))==3
    assert [x['time'] for x in tail.poll(date(2026,1,2))]==['3','2','1']

def test_bad_data(tmp_path):
    p=tmp_path/'a'; assert read_json(p)=={}; p.write_text('{'); assert read_json(p)=={}
    assert camcal_progress({})=={} and tooltip_progress({'tooltip':None})=={}


def test_display_formatting():
    from datetime import datetime, timezone, timedelta
    from nest_clear.ui.model import display_time, relative_time, picking_text
    now=datetime(2026,1,2,12,0,40,tzinfo=timezone.utc)
    assert display_time('2026-01-02T11:00:00+00:00',now.date()) == '11:00:00'
    assert display_time('2026-01-01T11:00:00+00:00',now.date()) == '01-01 11:00:00'
    assert relative_time((now-timedelta(seconds=11)).isoformat(),now)==('11 秒前','yellow')
    assert relative_time((now-timedelta(seconds=31)).isoformat(),now)==('31 秒前','red')
    assert picking_text({'tray':1,'view_index':10,'total_views':24,'coarse_shots':3,'removable_count':5,'red_count':1,'sent_count':7}) == '第 1 盘 · 视野 11/24 · 粗拍 3 次 · 可去 5 · 红框 1 · 已下发 7'
    row=classify_event({'event':'reply','func':999,'reason_code':'TEST','what':'现象','where':'位置','hint':'建议'})
    assert (row['func'],row['phenomenon'],row['location'],row['suggestion'])==(999,'现象','位置','建议')

def test_font_selection_does_not_synthesize_bold(monkeypatch):
    from types import SimpleNamespace
    from nest_clear.ui.app import select_font
    monkeypatch.setattr('nest_clear.ui.app.subprocess.run',lambda *a,**k: SimpleNamespace(stdout='WenQuanYi Micro Hei\tRegular\n'))
    assert select_font()==('WenQuanYi Micro Hei',False)
    monkeypatch.setattr('nest_clear.ui.app.subprocess.run',lambda *a,**k: SimpleNamespace(stdout='Noto Sans CJK SC\tRegular\nNoto Sans CJK SC\tBold\n'))
    assert select_font()==('Noto Sans CJK SC',True)


def test_edge_corner_labels_match_camera_order():
    from nest_clear.ui.model import edge_corner_label
    assert [edge_corner_label(i) for i in range(4)] == ['左上', '右上', '左下', '右下']


def test_pickacc_geometry_is_equal_scale_and_arrows_are_readable():
    from nest_clear.ui.model import pickacc_tray_geometry
    for width,height in ((600,350),(500,220),(300,500)):
        left,bottom,scale,multiplier=pickacc_tray_geometry(width,height)
        assert 100*scale <= width-90 and 160*scale <= height-45
        assert 0 <= left and bottom <= height
        assert 20 <= .05*scale*multiplier <= 40
        assert multiplier % 100 == 0


def test_pickacc_text_and_notes():
    from nest_clear.ui.model import pickacc_infield, pickacc_note, pickacc_stage, pickacc_type
    assert pickacc_type('constant')=='固定补偿' and pickacc_type('table')=='补偿表'
    assert [pickacc_stage(x) for x in ('A','B','repeat','verify','infield','simple')]==['第一轮','补点','重复性','验证','视野内','工具2简化']
    assert pickacc_note({'stage':'infield','corner':[-1,1]})=='视野内-左上'
    assert pickacc_note({'outlier':True,'retest':True})=='离群、重测'
    assert pickacc_note({'verification':True})=='验证'
    assert pickacc_infield([])=='视野内：未开始'
    assert '0.070 mm ⚠' in pickacc_infield([{'distance_mm':.07,'warning':True}])
    assert pickacc_infield([{'distance_mm':.02}]*4).endswith('正常')


def test_pickacc_demo_uses_real_plans_and_geometry():
    from nest_clear.ui.demo import pickacc_demo
    first=['T02','T03','T05','T10','T12','T21','T26','T28','T29']
    sixteen={'T02','T03','T04','T05','T10','T11','T12','T13','T18','T19','T20','T21','T26','T27','T28','T29'}
    select=pickacc_demo('select','2026-01-01')
    assert [q['target'] for q in select['queue']]==first
    table=pickacc_demo('table','2026-01-01')
    assert [s['target'] for s in table['samples'] if s['stage']=='A']==first
    assert {s['target'] for s in table['samples'] if s['stage'] in ('A','B')}==sixteen
    simple=pickacc_demo('tool_in_place','2026-01-01')
    assert [s['target'] for s in simple['samples']]+[q['target'] for q in simple['queue']]==['T02','T05','T26','T29','T12']
    p=table['predicted']
    assert abs(math.dist(p['T02'],p['T03'])-30.15)<1e-6
    assert abs(math.dist(p['T02'],p['T10'])-30.15)<1e-6


def test_tooltip_display_formatting():
    from nest_clear.ui.model import tooltip_vector, tooltip_deviation, tooltip_pixels, tooltip_quality, tooltip_time
    assert tooltip_vector([.12,-.08,.04], signed=True, labels='XYZ') == 'X +0.120 Y -0.080 Z +0.040'
    assert tooltip_vector(None) == '—'
    assert tooltip_vector([125.12, 73., 110.04], compact=True) == '125.120 73.000 110.040'
    assert tooltip_vector([float('nan'),0,0]) == '—'
    text = tooltip_deviation({'front':[.061,.03], 'side':[-.03,.03]})
    assert text == '主相机 横向 +0.061 竖向 +0.030 mm；辅相机 横向 -0.030 竖向 +0.030 mm'
    assert tooltip_deviation(None) == '—'
    assert tooltip_pixels({'front':[320,241], 'side':[321,242]}) == '320,241 / 321,242'
    assert tooltip_deviation({'front':[-.00001,.007], 'side':[-.0,.007]}, compact=True) == '主 +0.000/+0.007 辅 +0.000/+0.007'
    assert tooltip_vector([.2,-.00001,-0.0], signed=True, compact=True) == '+0.200 +0.000 +0.000'
    assert tooltip_vector([-.00001,-0.0,1], signed=False) == '0.000 / 0.000 / 1.000'
    assert tooltip_quality('未测', {'rms_px':{'front':.2,'side':.3},'condition_number':1.4}) == '未测'
    assert tooltip_quality('已存', {'rms_px':{'front':.2,'side':.3},'condition_number':1.4}) == '已存（RMS 主 0.20px 辅 0.30px，条件数 1.4）'
    assert tooltip_time('2026-11-22T12:34:56.123456-08:00') == '12:34:56'
    assert tooltip_time(None) == '—'


def test_tooltip_demo_states_use_detection_and_consistent_measurements(tmp_path):
    import json
    import math
    from nest_clear.tools.tooltip_demo_image import main
    from nest_clear.paths import live_image_file
    states = {}
    for state in ('idle','probe','confirm_move','in_range','done','fail'):
        root = tmp_path/state
        main(root, state)
        states[state] = json.loads((root/'tooltip_demo.json').read_text())
        assert all(live_image_file(root, f'calib_{view}_latest').exists() for view in ('front','side'))
        if state != 'idle':
            for view, point in states[state]['pixels'].items():
                assert states[state]['deviation_mm'][view] == pytest.approx([(point[0]-320)*.0152,(point[1]-242)*.0152])
    assert states['idle']['rounds'] == [] and states['idle']['tool'] is None
    assert states['probe']['rounds'][0]['stage'] == '探测' and states['probe']['jacobian_quality'] is None
    assert math.hypot(*states['confirm_move']['deviation_mm']['front']) > .05
    assert all(math.hypot(*p) < .05 for p in states['in_range']['deviation_mm'].values())
    assert states['done']['new_compensation'] and states['done']['compensation_difference']
    assert '质量不合格' in states['fail']['message']


def test_tooltip_overlay_marks_baseline_even_when_detection_fails_and_manual_is_purple():
    import numpy as np
    from nest_clear.tooltip.overlay import render_tooltip
    raw = np.full((480,640), 245, np.uint8)
    missing = render_tooltip(raw, (320,240), None)
    assert not np.array_equal(missing[240,320], [245,245,245])
    manual = render_tooltip(raw, (320,240), (400,300), manual=True)
    auto = render_tooltip(raw, (320,240), (400,300), manual=False)
    # OpenCV BGR：紫色有高蓝和红，橙色以红绿为主。
    assert manual[300,400,0] > 150 and auto[300,400,0] < 30


def test_tooltip_overlay_displays_no_negative_zero(monkeypatch):
    import numpy as np
    from nest_clear.tooltip import overlay
    draw_original = overlay.ImageDraw.Draw
    texts = []
    class CapturingDraw:
        def __init__(self, image): self.draw = draw_original(image)
        def text(self, xy, text, **kwargs):
            texts.append(text)
            self.draw.text(xy, text, **kwargs)
    monkeypatch.setattr(overlay.ImageDraw, 'Draw', CapturingDraw)
    overlay.render_tooltip(np.full((480,640),245,np.uint8), (320,240), (319.99999,239.99999))
    assert any('横向 +0.000 mm  竖向 +0.000 mm' in text for text in texts)
    assert all('-0.000' not in text for text in texts)
