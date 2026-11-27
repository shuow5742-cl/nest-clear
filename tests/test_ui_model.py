import json
import math
import pytest
from datetime import date
from nest_clear.ui.model import decode_status, status_indicators, classify_event, EventTail, read_json, camcal_progress, tooltip_progress

@pytest.mark.parametrize('mode,page', [
    ('相机标定中', '相机标定'), ('工具偏移标定中', '工具偏移标定'),
    ('下针深度标定中', '下针深度标定'),
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
    assert status_indicators({'pid': 1, 'plc_connected':False}, {}, supervisor_age=0)['PLC 连接'][0]=='red'

def test_preview_unavailable_when_main_running():
    from nest_clear.ui.model import preview_controls
    assert preview_controls({'pid': 1, 'child_pid': 2}, 0)['reason'] == '主程序运行中，预览不可用'

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
    from nest_clear.ui.model import pickacc_z_missing_tools, pickacc_z_summary_text, pickacc_z_type
    first=['T02','T03','T05','T10','T12','T21','T26','T28','T29']
    sixteen={'T02','T03','T04','T05','T10','T11','T12','T13','T18','T19','T20','T21','T26','T27','T28','T29'}
    select=pickacc_demo('select','2026-01-01')
    assert [q['target'] for q in select['queue']]==first
    table=pickacc_demo('table','2026-01-01')
    assert [s['target'] for s in table['samples'] if s['stage']=='A']==first
    assert {s['target'] for s in table['samples'] if s['stage'] in ('A','B')}==sixteen
    assert table['candidate']['z_type'] == 'table'
    constant=pickacc_demo('constant','2026-01-01')
    assert constant['candidate']['z_type'] == 'constant'
    z_step=pickacc_demo('z_step','2026-01-01')
    assert z_step['phase'] == 'z_step' and z_step['pending_sample']['target'] == 'T12'
    assert pickacc_z_missing_tools(z_step['existing']) == [2]
    assert pickacc_z_missing_tools(constant['existing']) == []
    assert '全盘最大值' in pickacc_z_summary_text(constant)
    assert 'Z 表插值' in pickacc_z_summary_text(table)
    assert pickacc_z_type('constant') == '全盘最大值' and pickacc_z_type('table') == 'Z 表插值'
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


def test_main_status_running_stopped_and_recovery():
    main = {'mode': '挑毛中'}
    running = {'pid': 1, 'child_pid': 2, 'child_state': '就绪'}
    assert status_indicators(running, main, supervisor_age=0)['主程序'] == ('green', '挑毛中')
    assert status_indicators({**running, 'child_pid': None, 'child_state': '退出'}, main, supervisor_age=0)['主程序'] == ('gray', '主程序未运行')
    assert status_indicators(running, main, supervisor_age=31)['主程序'] == ('gray', '主程序未运行')
    assert status_indicators(running, main, supervisor_age=None)['主程序'] == ('gray', '主程序未运行')
    assert status_indicators(running, main, supervisor_age=0)['主程序'] == ('green', '挑毛中')


def test_command_timeout_message_distinguishes_main_running_and_stopped():
    from nest_clear.ui.model import command_timeout_message, main_running
    running = {'pid': 1, 'child_pid': 2, 'child_state': '就绪'}
    stopped = {'pid': 1, 'child_pid': None, 'child_state': '未启动'}
    assert main_running(running, supervisor_age=0) is True
    assert main_running(stopped, supervisor_age=0) is False
    assert main_running(running, supervisor_age=31) is False
    assert main_running({}, {'pid': 2}, main_alive_age=0) is True
    assert main_running({}, {'pid': 2}, main_alive_age=31) is False
    assert command_timeout_message('baseline_start', stopped, supervisor_age=0) == '主程序未运行，请先让 PLC 初始化'
    assert command_timeout_message('baseline_start', running, supervisor_age=0) == '主程序未响应命令【baseline_start】（已放弃，可重试），请查看日志'


def test_reply_time_crosses_day_boundary():
    from nest_clear.ui.model import display_time
    assert display_time('2026-11-22T07:00:15-08:00', date(2026, 11, 23)) == '11-22 07:00:15'
    assert display_time('2026-11-23T07:00:15-08:00', date(2026, 11, 23)) == '07:00:15'


def test_supervisor_status_and_process_recover():
    from nest_clear.ui.model import parent_process_row, child_process_row, SUPERVISOR_STALE_SECONDS
    from datetime import datetime, timezone
    now = datetime(2026, 11, 23, 12, tzinfo=timezone.utc)
    state = {'pid': 123, 'child_pid': 456, 'child_state': '就绪',
             'start_time': '2026-11-23T11:00:00+00:00',
             'plc_last_read': now.isoformat(), 'heartbeat_last_write': now.isoformat(),
             'plc_connected': True, 'algorithm_ready': 1}
    main = {'mode': '挑毛中'}
    for age in (SUPERVISOR_STALE_SECONDS + 1, None):
        indicators = status_indicators(state, main, now, age)
        assert indicators['父程序'][0] == 'gray'
        assert indicators['主程序'] == ('gray', '主程序未运行')
        assert all(indicators[name][0] == 'gray' for name in ('PLC 连接', '心跳', '40932 就绪'))
        assert all('过期' in indicators[name][1] for name in ('PLC 连接', '心跳', '40932 就绪'))
        row = parent_process_row(state, age, now)
        assert row[1] == '—' and row[4] == '未运行'
        assert child_process_row(state, age, now)[1:5:3] == ('—', '未运行')
    indicators = status_indicators(state, main, now, 0)
    assert indicators['父程序'] == ('green', '123')
    assert indicators['主程序'] == ('green', '挑毛中')
    assert all(indicators[name][0] == 'green' for name in ('PLC 连接', '心跳', '40932 就绪'))
    assert child_process_row(state, 0, now)[1] == 456
    assert parent_process_row(state, 0, now)[1:5] == (123, '11:00:00', '01:00:00', '运行')
    assert status_indicators({}, {}, now, None)['父程序'] == ('gray', '无数据')
    assert parent_process_row({}, None, now)[4] == '未运行'


def test_baseline_preview_error_clears_stale_metrics():
    from nest_clear.ui.model import baseline_preview_display
    data = baseline_preview_display({'error':'USB 带宽不足','on':True,'percent':98,'tenengrad':1.2,
        'peak':1.3,'edge_width_um':20,'edge_width_px':2,'contrast':220,'source':'tip','exposure_ms':10,'history':[98]})
    assert data['button'] == '预览失败，点击重试'
    assert set(data['metrics'].values()) == {'—'}
    assert data['history'] == [] and '集线器' in data['hint']


def test_tooltip_focus_summary_not_anomaly_and_preserves_status(tmp_path):
    from nest_clear.ui.status import MainStatus
    row = {'event': 'tooltip_focus_summary', 'camera': 'front', 'current': 1.2, 'peak': 1.3,
           'percent': 92.3, 'edge_width_um': 20.0, 'source': 'tip'}
    assert classify_event(row) is None
    status = MainStatus(tmp_path)
    status.observe({'event': 'tooltip_start', 'tool': 1})
    status.observe(row)
    assert status.data['tooltip']['event'] == 'tooltip_start'
    assert 'current' not in status.data['tooltip']
    if status.timer:
        status.timer.cancel()


def test_baseline_preview_reads_state_file_and_handles_expiry(tmp_path):
    import os
    from datetime import datetime, timedelta, timezone
    from nest_clear.paths import tooltip_preview_file
    from nest_clear.ui.model import baseline_preview_display, read_tooltip_preview

    low = {'on': True, 'device': 'usb-0:4', 'exposure_source': '已测 2026-11-23'}
    # Missing file -> no live data
    assert read_tooltip_preview(tmp_path) == {}
    missing_disp = baseline_preview_display(low, live=None)
    assert missing_disp['no_live_data'] is True
    assert missing_disp['message'] == '无实时数据'
    assert set(missing_disp['metrics'].values()) == {'—'}
    assert missing_disp['history'] == []

    now = datetime.now().astimezone()
    path = tooltip_preview_file(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        'updated_at': now.isoformat(),
        'cameras': {
            'front': {'tenengrad': 1.28, 'peak': 1.30, 'percent': 98.5, 'edge_width_um': 27.2,
                      'edge_width_px': 1.79, 'contrast': 210, 'source': 'tip', 'exposure_ms': 10.0,
                      'history': [80, 90, 98.5], 'updated_at': now.isoformat()},
            'side': {'tenengrad': 1.10, 'peak': 1.30, 'percent': 84.6, 'edge_width_um': 30.0,
                     'edge_width_px': 1.97, 'contrast': 195, 'source': 'tip', 'exposure_ms': 10.0,
                     'history': [70, 80, 84.6], 'updated_at': (now - timedelta(seconds=4)).isoformat()},
        }
    }), encoding='utf-8')
    fresh = read_tooltip_preview(tmp_path, now=now)
    assert 'front' in fresh and 'side' not in fresh
    front_disp = baseline_preview_display(low, live=fresh['front'])
    assert front_disp['no_live_data'] is False
    assert front_disp['live']['percent'] == 98.5
    assert front_disp['history'] == [80, 90, 98.5]
    side_disp = baseline_preview_display(low, live=fresh.get('side'))
    assert side_disp['no_live_data'] is True
    assert side_disp['message'] == '无实时数据'
    assert set(side_disp['metrics'].values()) == {'—'}

    # File mtime expired (> 3s) -> all cameras report no live data
    stale_epoch = now.timestamp() - 5.0
    os.utime(path, (stale_epoch, stale_epoch))
    assert read_tooltip_preview(tmp_path, now=now) == {}


def test_main_preview_controls_use_mode_and_hosted_focus_state():
    from nest_clear.ui.model import preview_controls
    status = {'pid': 1, 'child_pid': 2, 'word': 0,
              'preview': {'on': True, 'mode': 'focus'}}
    focus = preview_controls(status, 0, '空闲')
    assert focus['focus_enabled'] and focus['white_balance_enabled']
    blocked = preview_controls(status, 0, '挑毛中')
    assert not blocked['preview_enabled'] and not blocked['focus_enabled']
    assert '挑毛中' in blocked['reason']
    status['word'] = 1  # 手动模式
    assert preview_controls(status, 0, '挑毛中')['focus_enabled']
    status['preview'] = {'on': False}
    assert preview_controls(status, 0, '对刀基准建立中')['preview_enabled']
    assert not preview_controls(status, 0, '空闲')['white_balance_enabled']


def test_depthcal_model_helpers_and_status(tmp_path):
    from nest_clear.ui.demo import depthcal_demo
    from nest_clear.ui.model import (
        classify_event, depthcal_arrival_summary_text, depthcal_curve_dof,
        depthcal_dof_summary_text, depthcal_exposure_display, depthcal_feasibility_text,
        depthcal_fine_plan_text, depthcal_latest_dof_text, depthcal_point_arrival_text,
        depthcal_stage_info, depthcal_subplot_layout,
    )
    from nest_clear.ui.status import MainStatus
    assert depthcal_stage_info({'phase': 'idle'})[0] == '阶段：空闲（等待 PLC 160）'
    assert depthcal_stage_info({'phase': 'done'})[0] == '阶段：完成（已回 161）'
    assert depthcal_stage_info({'phase': 'failed'})[0] == '阶段：失败（已回 999）'
    assert depthcal_stage_info({'phase': 'fine', 'message': '细扫 第 2/3 次'})[0] == '阶段：细扫 第 2/3 次'
    assert '到位异常 3 次' in depthcal_stage_info({
        'phase': 'fine', 'message': '细扫 第 2/3 次',
        'arrival_summary': {'total_arrivals': 100, 'total_anomalies': 3},
    })[0]
    assert '到位异常 2/50 次' in depthcal_stage_info({
        'phase': 'failed',
        'arrival_summary': {'total_arrivals': 50, 'total_anomalies': 2},
    })[0]

    # 到位列文字、汇总行文字及 depthcal_arrival_warning 事件分类
    assert depthcal_point_arrival_text({'arrival': {'total': 126, 'anomalies': 0}}) == ('—', False)
    assert depthcal_point_arrival_text({
        'arrival': {'total': 126, 'anomalies': 3, 'max_delta_axis': 'Z', 'max_delta_mm': 0.023}
    }) == ('3/126 次 · 最大 Z +0.023', True)
    warn_sum, warn_bg, _ = depthcal_arrival_summary_text({
        'phase': 'done',
        'arrival_summary': {
            'total_arrivals': 775, 'total_anomalies': 4, 'anomaly_points': ['T02', 'T26'],
            'max_delta_axis': 'Z', 'max_delta_mm': 0.023, 'position_tolerance_mm': 0.01, 'avg_elapsed_s': 0.52,
        },
    })
    assert warn_sum == '到位超差点：T02、T26（共 4 次，最大 Z +0.023 mm，详见 arrivals.csv）'
    assert warn_bg == '#fef3c7'
    ok_sum, ok_bg, _ = depthcal_arrival_summary_text({
        'phase': 'done',
        'arrival_summary': {
            'total_arrivals': 630, 'total_anomalies': 0, 'anomaly_points': [],
            'position_tolerance_mm': 0.01, 'avg_elapsed_s': 0.51,
        },
    })
    assert ok_sum == '到位：全部在 ±0.01 mm 内（平均等待 0.51 s）'
    assert ok_bg == '#ecfdf3'
    ev = classify_event({'event': 'depthcal_arrival_warning', 'label': 'T02', 'phase': 'fine1', 'what': 'T02 fine1 到位超差'})
    assert ev['color'] == 'yellow' and ev['category'] == '标定'

    ok_text, ok_bg, _ = depthcal_feasibility_text({'feasibility': {'max_range': 0.004, 'max_std': 0.002, 'threshold': 0.02, 'met': True}})
    assert '满足下针深度标定误差预算' in ok_text and ok_bg == '#ecfdf3'
    bad_text, bad_bg, _ = depthcal_feasibility_text({'feasibility': {'max_range': 0.035, 'max_std': 0.018, 'threshold': 0.02, 'met': False}})
    assert '不满足下针深度标定误差预算' in bad_text and bad_bg == '#fff1f0'
    insuf_text, insuf_bg, _ = depthcal_feasibility_text({
        'feasibility': {'max_range': 0.003, 'max_std': 0.001, 'threshold': 0.02,
                        'insufficient_fits': 2, 'insufficient_items': ['T02#1', 'T02#2'], 'total_fits': 15, 'met': False}
    })
    assert '拟合点不足 2/15 次' in insuf_text and '不计入满足' in insuf_text and insuf_bg == '#fff1f0'

    summary_auto, notice_auto = depthcal_fine_plan_text({
        'label': 'T02',
        'fine_plan': {'mode': 'auto', 'half_width_mm': 1.5, 'fine_step_mm': 0.075, 'coarse_fwhm_mm': 2.0,
                      'fallback': False, 'truncated': False},
    })
    assert '±1.500 mm' in summary_auto and '0.075 mm' in summary_auto and notice_auto == ''
    summary_fb, notice_fb = depthcal_fine_plan_text({
        'label': 'T05',
        'fine_plan': {'mode': 'fixed_fallback', 'half_width_mm': 0.6, 'fine_step_mm': 0.02,
                      'fallback': True, 'fallback_reason': '高斯拟合失败',
                      'truncated': True, 'truncation_note': '下限受 camera_z_min_mm 截断'},
    })
    assert '回退 fixed' in summary_fb and '回退 fixed' in notice_fb and 'Z 约束截断' in notice_fb

    demo_done, _, _ = depthcal_demo('done')
    assert 1.8 <= demo_done['results'][0]['fits'][0]['fwhm'] <= 2.2
    assert demo_done['dof_summary']['dof_ratio'] == pytest.approx(0.8)
    assert demo_done['results'][0]['dof_mm'] > 0

    # DOF 辅助函数：纯 Python 插值、粗扫估计、细扫、未覆盖、全盘汇总
    pts = [[-6.0, 0.5], [-5.5, 0.8], [-5.0, 1.0], [-4.5, 0.8], [-4.0, 0.5]]
    cdof = depthcal_curve_dof(pts, 0.8)
    assert cdof['dof_covered'] and cdof['dof_mm'] == pytest.approx(1.0)
    assert cdof['dof_z_lower'] == pytest.approx(-5.5) and cdof['dof_z_upper'] == pytest.approx(-4.5)
    assert depthcal_curve_dof([[-5.5, 0.9], [-5.0, 1.0], [-4.5, 0.7]], 0.8)['dof_covered'] is False

    assert depthcal_latest_dof_text({'label': 'T02', 'coarse_fit': {'dof_mm': 1.25, 'fwhm': 2.1, 'dof_covered': True}}) == '景深：1.250 mm（粗扫估计，FWHM 2.100 mm）'
    assert depthcal_latest_dof_text({'label': 'T02', 'current_fits': [{'repeat': 2, 'dof_mm': 1.14, 'fwhm': 2.0, 'dof_covered': True}]}) == '景深：1.140 mm（细扫2，FWHM 2.000 mm）'
    assert depthcal_latest_dof_text({'label': 'T02', 'current_fits': [{'repeat': 1, 'dof_mm': None, 'fwhm': 2.0, 'dof_covered': False}]}) == '景深：未覆盖（细扫1，FWHM 2.000 mm）'

    sum_ok, sum_bg, _ = depthcal_dof_summary_text(demo_done)
    assert 'dof_ratio=0.80' in sum_ok and '均值' in sum_ok and '最小' in sum_ok and '最大' in sum_ok and sum_bg == '#ecfdf3'
    sum_uncov, sum_uncov_bg, _ = depthcal_dof_summary_text({
        'dof_ratio': 0.8,
        'dof_summary': {'mean_mm': 1.15, 'min_mm': 1.12, 'max_mm': 1.18, 'covered_points': 4, 'total_points': 5,
                        'uncovered_fits': 3, 'total_fits': 15, 'uncovered_items': ['T05#1', 'T05#2', 'T05#3']},
    })
    assert '未覆盖 3/15 次' in sum_uncov and '不计入统计' in sum_uncov and sum_uncov_bg == '#fff7ed'

    boxes = depthcal_subplot_layout(['T02', 'T05', 'T29', 'T26', 'T12'], 900, 400)
    assert len(boxes) == 5
    for _, (x0, y0, x1, y1) in boxes:
        assert 0 <= x0 < x1 <= 900 and 24 <= y0 < y1 <= 400

    st = MainStatus(tmp_path)
    st.observe({'event': 'depthcal_exposure_state', 'exposure_mode': 'manual', 'manual_exposure_us': 12000.0,
                'exposure_command_message': '已保存曝光设置'})
    assert st.data['depthcal']['exposure_mode'] == 'manual'
    assert st.data['depthcal']['manual_exposure_us'] == 12000.0
    assert '12000 µs' in depthcal_exposure_display({
        'current_exposure_us': 12000.0, 'point_exposure_mode': 'manual', 'roi_mean': 102.5, 'roi_p99': 188.0,
    })
    st.observe({'event': 'depthcal_state', 'session': 's1', 'phase': 'coarse', 'metric': 'tenengrad',
                'sample': {'label': 'T02', 'pass': 'coarse', 'z_actual': -5.0, 'tenengrad': 0.5}})
    st.observe({'event': 'depthcal_state', 'session': 's1', 'phase': 'fine', 'metric': 'tenengrad',
                'sample': {'label': 'T02', 'pass': 1, 'z_actual': -5.02, 'tenengrad': 0.8}})
    st.observe({'event': 'depthcal_abort_requested'})
    assert st.data['depthcal']['curves']['T02']['coarse'] == [[-5.0, 0.5]]
    assert st.data['depthcal']['curves']['T02']['1'] == [[-5.02, 0.8]]
    assert st.data['depthcal']['abort_requested'] is True
    if st.timer:
        st.timer.cancel()


def test_depthcal_auto_page_switch_model():
    from nest_clear.ui.model import depthcal_session_key, next_auto_page

    # 空闲且无新会话时不切页
    page, last_mode, last_session = next_auto_page({'mode': '空闲', 'depthcal': {}}, None, None)
    assert page is None and last_mode == '空闲' and last_session is None

    # 两次轮询之间发生“空闲→下针深度标定中→空闲（立即失败）”：mode 仍是空闲，仍须切到下针深度标定页
    failed_status = {
        'mode': '空闲',
        'depthcal': {'session': 's1', 'phase': 'failed', 'reason_code': 'POSITION-TRAVEL', 'what': 'X 超限'},
    }
    page, last_mode, last_session = next_auto_page(failed_status, last_mode, last_session)
    assert page == '下针深度标定' and last_session == ('session', 's1')

    # 同一会话内再次轮询（用户已切走）不再抢页
    page, last_mode, last_session = next_auto_page(failed_status, last_mode, last_session)
    assert page is None and last_session == ('session', 's1')

    # 新会话 s2 到来时再次切页
    page, last_mode, last_session = next_auto_page(
        {'mode': '下针深度标定中', 'depthcal': {'session': 's2', 'phase': 'center'}},
        last_mode, last_session,
    )
    assert page == '下针深度标定' and last_session == ('session', 's2')
    # 同会话推进阶段不再重复切页
    page, last_mode, last_session = next_auto_page(
        {'mode': '下针深度标定中', 'depthcal': {'session': 's2', 'phase': 'coarse'}},
        last_mode, last_session,
    )
    assert page is None and last_session == ('session', 's2')

    # 无 Session 即失败（如缺少 camera_z_min_mm）也能凭事件标识切页一次
    pre_fail = {'mode': '空闲', 'depthcal': {'phase': 'failed', 'seq_event': 42, 'reason_code': 'DEPTH-Z-MIN'}}
    assert depthcal_session_key(pre_fail) == ('event', '42')
    page, last_mode, last_session = next_auto_page(pre_fail, last_mode, last_session)
    assert page == '下针深度标定'
    page, _, _ = next_auto_page(pre_fail, last_mode, last_session)
    assert page is None

    # 其他工作模式进行中（如挑毛中）收到异常 depthcal 状态不抢页
    page, _, _ = next_auto_page(
        {'mode': '挑毛中', 'depthcal': {'phase': 'failed', 'seq_event': 43, 'reason_code': 'DEPTH-MODE'}},
        '空闲', last_session,
    )
    assert page == '主页'
