import json
import math
from datetime import date
from nest_clear.ui.model import decode_status, status_indicators, classify_event, EventTail, read_json, camcal_progress, tooltip_progress

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
