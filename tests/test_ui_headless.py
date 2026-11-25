"""无显示服务时也遍历窗口构造、刷新、页面名和截图控制路径。"""
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from nest_clear.ui.demo import write_demo


class Widget:
    def __init__(self, parent=None, **kwargs):
        self.parent = parent; self.master = parent; self.options = dict(kwargs); self.children = []; self.rows = {}; self.sel = ()
        self.tk = self; self.destroyed = False; self.text_calls = []; self.row_options = {}; self.bindings = {}
        if isinstance(parent, Widget): parent.children.append(self)
    def call(self, *args): pass
    def winfo_fpixels(self, arg): return 96.0
    def winfo_screenwidth(self): return 1366
    def winfo_screenheight(self): return 768
    def winfo_width(self): return 600
    def winfo_height(self): return 300
    def winfo_reqheight(self): return 22
    def winfo_reqwidth(self): return 100
    def transient(self, *args): pass
    def grab_set(self): pass
    def grab_release(self): pass
    def lift(self): pass
    def focus_force(self): pass
    def protocol(self, *args): pass
    def pack(self, *args, **kwargs): pass
    def winfo_rootx(self): return 5
    def winfo_rooty(self): return 7
    def rowconfigure(self, index, **kwargs): self.row_options.setdefault(index, {}).update(kwargs)
    def columnconfigure(self, *args, **kwargs): pass
    def grid(self, *args, **kwargs): self.visible = True
    def grid_remove(self): self.visible = False
    def tkraise(self): pass
    def grid_configure(self, **kwargs): self.options.update(kwargs)
    def grid_propagate(self, *args): pass
    def bind(self, event, callback, *args, **kwargs): self.bindings[event] = callback
    def option_add(self, *args): pass
    def title(self, *args): pass
    def minsize(self, *args): pass
    def geometry(self, *args): pass
    def state(self, *args): pass
    def attributes(self, *args): pass
    def after(self, *args): pass
    def configure(self, **kwargs): self.options.update(kwargs)
    config = configure
    def update_idletasks(self): pass
    def destroy(self): self.destroyed = True
    def add(self, item, **kwargs): self.children.append(item)
    def select(self, index): self.selected = index
    def heading(self, *args, **kwargs): pass
    def column(self, *args, **kwargs): pass
    def tag_configure(self, *args, **kwargs): pass
    def yview(self, *args): pass
    def yview_moveto(self, *args): pass
    def set(self, *args): pass
    def insert(self, parent, where, iid=None, **kwargs):
        iid = str(iid) if iid is not None else str(len(self.rows))
        self.rows[iid] = kwargs; return iid
    def item(self, iid, option=None, **kwargs):
        self.rows[iid].update(kwargs)
        return self.rows[iid].get(option) if option else self.rows[iid]
    def get_children(self): return tuple(self.rows)
    def delete(self, *args):
        for key in args: self.rows.pop(key, None)
    def selection(self): return self.sel
    def create_text(self, *args, **kwargs): self.text_calls.append(kwargs.get('text'))
    def create_oval(self, *args, **kwargs): pass
    def create_rectangle(self, *args, **kwargs): pass
    def create_line(self, *args, **kwargs): pass
    def create_polygon(self, *args, **kwargs): pass
    def create_image(self, *args, **kwargs): pass
    def create_window(self, *args, **kwargs): return 1
    def itemconfigure(self, *args, **kwargs): pass
    def scan_mark(self, *args): pass
    def scan_dragto(self, *args, **kwargs): pass


class Style:
    def __init__(self, root): pass
    def theme_use(self, theme): assert theme == 'clam'
    def configure(self, *args, **kwargs): pass
    def map(self, *args, **kwargs): pass


def test_home_picture_width_tracks_height_and_image_ratio():
    from nest_clear.ui.app import resize_home_picture
    container = Widget(); body = Widget(); picture = SimpleNamespace(image=None)
    resize_home_picture(container, body, picture)
    assert container.options['width'] == round(300 * 4024 / 3036) + 24
    picture.image = SimpleNamespace(size=(960, 600))
    resize_home_picture(container, body, picture)
    assert container.options['width'] == 504
    container.winfo_width = lambda: 503
    container.options.clear()
    resize_home_picture(container, body, picture)
    assert not container.options


def test_demo_images_use_camera_aspect_ratio(tmp_path):
    from PIL import Image
    from nest_clear.paths import live_image_file
    write_demo(tmp_path)
    for name in ('latest', 'camcal_latest', 'preview_latest', 'calib_front_latest', 'calib_side_latest'):
        with Image.open(live_image_file(tmp_path, name)) as image:
            assert image.size == (1006, 759)


def test_all_page_construction_refresh_and_screenshot(tmp_path, monkeypatch):
    from nest_clear.ui import app
    write_demo(tmp_path, initial=True)
    import sys
    for module in ('cv2', 'numpy', 'yaml'):
        monkeypatch.setitem(sys.modules, module, None)
    for name in ('Tk', 'Frame', 'Label', 'Canvas', 'Toplevel', 'Text', 'Button'):
        monkeypatch.setattr(app.tk, name, Widget)
    for name in ('Notebook', 'Treeview', 'Progressbar', 'Scrollbar', 'Button', 'Entry', 'Combobox', 'Spinbox'):
        monkeypatch.setattr(app.ttk, name, Widget)
    monkeypatch.setattr(app.ttk, 'Style', Style)
    monkeypatch.setattr(app.tkfont, 'families', lambda root: ['Noto Sans Mono CJK SC'])
    class FakeCard(Widget):
        def __init__(self, parent, title, theme):
            super().__init__(parent); self.body = Widget(self); self.title_label = Widget(self)
    class FakePicture(Widget):
        def __init__(self, parent, path, theme): super().__init__(parent)
        def poll(self): pass
    class FakeKeyValueGrid(Widget):
        def __init__(self, parent, keys, theme): super().__init__(parent); self.values_data = {}
        def set(self, values): self.values_data = values
    monkeypatch.setattr(app, 'Card', FakeCard)
    monkeypatch.setattr(app, 'Picture', FakePicture)
    monkeypatch.setattr(app, 'KeyValueGrid', FakeKeyValueGrid)
    for page in app.App.PAGE_NAMES:
        ui = app.App(tmp_path, page=page, ignore_before_ns=0)
        if page == '对刀基准建立':
            assert not hasattr(ui, 'baseline_choose') and not hasattr(ui, 'baseline_swap')
            assert all('choose' not in view for view in ui.baseline_views.values())
            from nest_clear.ui.model import supervisor_tooltip_main
            ui.baseline_pending = 8
            supervisor = {'pid': 123, 'child_pid': None, 'child_state': '停止'}
            main = {'command_seq': 99}
            host = {'pid': 123, 'command_seq': 7, 'baseline': {'phase': 'prepare'}}
            ui.update_baseline(supervisor_tooltip_main(main, supervisor, host), supervisor)
            assert ui.baseline_ack == 7 and ui.baseline_pending == 8
            host['command_seq'] = 8
            ui.update_baseline(supervisor_tooltip_main(main, supervisor, host), supervisor)
            assert ui.baseline_ack == 8 and ui.baseline_pending == 0
            host['baseline'] = {'phase': 'live', 'candidate': {'p0': [1, 2, 3]}}
            ui.update_baseline(supervisor_tooltip_main(main, supervisor, host), supervisor)
            for button in [ui.baseline_start, *ui.baseline_tools, ui.baseline_measure,
                           ui.baseline_manual, ui.baseline_aux, ui.baseline_save, ui.baseline_end]:
                assert button.options['state'] == 'disabled'
            assert '主程序未运行（等待 PLC 初始化）' in ui.baseline_message.options['text']
            assert ui.baseline_views['front']['toggle'].options['state'] == 'normal'
            ui.baseline_pending = 9; ui.baseline_pending_since = 0; ui.baseline_pending_name = 'tooltip_assign'
            ui.update_baseline({'command_seq': 9}, {'child_pid': 42})
            assert ui.baseline_pending == 0
        left = ui.home.children[0]
        device_content = ui.bits[0].parent
        pick_content = ui.pick_values['盘号'].parent
        log_card = ui.table.parent.master
        left.winfo_reqheight = lambda: 280 + 4 * device_content.row_options[0]['minsize']
        log_card.winfo_reqheight = lambda: 160
        ui.home.winfo_height = lambda: 700
        ui.home.bindings['<Configure>'](None)
        assert device_content.master.options['pady'] == (0, 0)
        assert device_content.master.master.options['pady'] == 1
        ui.root.winfo_height = lambda: 1080
        ui.home.bindings['<Configure>'](None)
        assert device_content.master.options['pady'] == (2, 6)
        assert device_content.master.master.options['pady'] == 3
        assert ui.home.row_options[0]['weight'] == 0
        assert ui.home.row_options[1] == {'weight': 1, 'minsize': 166}
        ui.home.winfo_height = lambda: 500
        ui.home.bindings['<Configure>'](None)
        assert ui.home.row_options[0]['weight'] == 1
        assert left.row_options[0]['weight'] == 1
        assert device_content.row_options[0]['minsize'] == 22
        assert pick_content.row_options[0]['minsize'] == 22
        ui.home.bindings['<Configure>'](None)
        assert device_content.row_options[0]['minsize'] == 22
        ui.home.winfo_height = lambda: 700
        ui.home.bindings['<Configure>'](None)
        assert device_content.row_options[0]['minsize'] == 30
        ui.root.winfo_height = lambda: 300
        assert any(child.options.get('text') == app.FONT_WARNING for child in ui.root.children)
        for index, name in enumerate(app.App.PAGE_NAMES):
            ui.select_page(name + '页' if name != '主页' else name)
            assert ui.tabs.selected == index
        ui.select_page(page)
        ui.refresh()
        assert ui.tabs.selected == app.App.PAGE_NAMES.index(page)
        ui.update_preview({'pid': 1, 'child_pid': 2})
        assert ui.focus_button.options['state'] == 'disabled'
        assert ui.preview_button.options['state'] == 'disabled'
        focus = {'on': True, 'mode': 'focus', 'scores': [1.0]*9, 'peaks': [2.0]*9,
                 'percent': [50.0]*9, 'center': 1.0, 'median': 1.0, 'laplacian': [10.0]*9,
                 'cell_means': [100.0]*9, 'mean': 100.0, 'p1': 20, 'p99': 180,
                 'saturated_pct': 0, 'dark_pct': 0, 'tilt': '左上', 'exposure_hint': ''}
        ui.update_preview({'pid': 1, 'preview': focus})
        assert ui.focus_button.options['state'] == 'normal'
        assert ui.preview_button.options['state'] == 'disabled'
        assert ui.focus_percent.options['text'] == '50%'
        assert ui.focus_alert.options['text'] == '左上'
        assert ui.focus_curve.options['height'] >= 120
        assert len(ui.focus_score_cells) == len(ui.focus_mean_cells) == 9
        row = {'exposure_us': 3000, 'saturated_fraction': 0, 'p99': 200, 'corners': 30, 'sharpness': 1.2}
        focus['exposure_sweep'] = {'state': 'done', 'mode': 'calibration', 'index': 10, 'total': 10,
                                   'original_us': 6000, 'rows': [row], 'recommendation': row}
        ui.update_preview({'pid': 1, 'preview': focus})
        assert ui.exposure_apply.options['state'] == 'normal'
        assert any('推荐 3000.0' in str(value) for value in ui.exposure_message.options.values())
        assert next(iter(ui.exposure_table.rows.values()))['tags'] == ('recommended',)
        ui.update_preview({'pid': 1, 'preview': {'on': False}})
        assert all(b.options['state'] == 'disabled' for b in ui.offset_tool_buttons + ui.offset_mode_buttons)
        assert ui.offset_start.options['state'] == 'disabled'
        assert ui.offset_tool_buttons[0].options['style'] == 'OffsetSelected.TButton'
        ui.update_offset({'tooloffset': {'phase':'select'}, 'command_seq':0})
        assert all(b.options['state'] == 'normal' for b in ui.offset_tool_buttons + ui.offset_mode_buttons)
        ui.update_offset({'tooloffset': {'phase':'edge', 'tool':2, 'mode':'center_edge', 'edges':[{'target':'T12','corner':2,'delta_xy_mm':[0,0], 'distance_mm':0}]}, 'command_seq':0})
        assert all(b.options['state'] == 'disabled' for b in ui.offset_tool_buttons + ui.offset_mode_buttons)
        assert ui.offset_edges.rows['0']['values'][1] == '左下'
        ui.update_pickacc({'pickacc':{'phase':'select'},'command_seq':0})
        ui.choose_pickacc_tool(2)
        ui.update_pickacc({'pickacc':{'phase':'select'},'command_seq':0})
        assert ui.pickacc_start.options['state'] == 'normal'
        ui.update_pickacc({'pickacc':{'phase':'tool_in_place','tool':2,'target':'T12','predicted':{'T12':[100,50]},'samples':[{'target':'T12','target_xy_mm':[100,50],'residual_xy_mm':[.02,.01]}]},'command_seq':0})
        assert ui.pickacc_calculate.options['state'] == 'normal'
        assert len(ui.pickacc_table.rows) == 1
        ui.update_pickacc({'pickacc':{'phase':'wait_tool','tool':2},'command_seq':0})
        assert ui.pickacc_calculate.options['state'] == 'disabled'
        ui.update_baseline({'baseline':{'phase':'select'},'command_seq':0})
        assert all(b.options['state'] == 'normal' for b in ui.baseline_tools)
        ui.update_baseline({'baseline':{'phase':'live','tool':1,'live_delta':[0.01,-0.02],'live_inside':True,'existing':{'tool1':{'tool_offset_missing':True}}},'command_seq':0})
        assert ui.baseline_measure.options['state'] == 'normal'
        assert ui.baseline_save.options['state'] == 'disabled'
        assert '尚无工具偏移标定结果' in ui.baseline_notice.options['text']
        ui.update_baseline({'baseline':{'phase':'fail','tool':1,'message':'基准标定失败'},'command_seq':0})
        assert ui.baseline_message.options['fg'] == app.COLORS['red']
        ui.update_baseline({'baseline':{'phase':'success','tool':1,'candidate':{'p0':[1,2,3], 'pixels':{'front':[1,2], 'side':[3,4]}}},'command_seq':0})
        assert ui.baseline_save.options['state'] == 'normal'
        assert ui.baseline_tools[0].options['state'] == 'disabled'
        assert ui.baseline_tools[0].options['style'] == 'OffsetSelected.TButton'
        ui.update_baseline({'mode':'对刀基准建立中', 'command_seq':0, 'baseline':{
            'phase':'prepare', 'preview':{'front':{'on':True,'device':'/dev/v4l/by-path/pci-usb-0:4:1.0-video-index0','percent':98,'tenengrad':1.0,'peak':1.1,'source':'tip'},
                                           'side':{'on':True,'error':'USB 带宽不足'}},
            'existing':{'tool1':{'camera_differences':['front.device']}}}})
        assert ui.baseline_views['front']['info'].options['fg'] == app.COLORS['green']
        assert 'usb-0:4' in ui.baseline_views['front']['path'].options['text']
        assert 'USB 带宽不足' in ui.baseline_views['side']['info'].options['text']
        assert ui.baseline_views['side']['toggle'].options['text'] == '辅助对刀相机（侧面）预览：开'
        assert '集线器' in ui.baseline_views['side']['hint'].options['text']
        assert '对比度 —' in ui.baseline_views['side']['detail'].options['text']
        assert ui.baseline_views['side']['path'].parent.row_options[0]['weight'] == 0
        assert ui.baseline_views['side']['path'].parent.row_options[2]['weight'] == 1
        # Eligibility controls every action independently of whether preview is on.
        from nest_clear.paths import supervisor_state_file
        import json
        supervisor = {'pid': 1, 'child_pid': 2, 'child_state': '就绪', 'word': 0}
        supervisor_state_file(tmp_path).write_text(json.dumps(supervisor))
        active = {'mode': '对刀基准建立中', 'command_seq': 0, 'baseline': {
            'phase': 'live', 'preview': {'front': {'on': True}, 'side': {'on': True}}}}
        ui.update_baseline(active, supervisor)
        for view in ui.baseline_views.values():
            assert all(view[key].options['state'] == 'disabled' for key in ('toggle', 'exposure', 'reset'))
            assert '不可用：对刀基准建立中' in view['reason'].options['text']
            assert '不可用' not in view['toggle'].options['text']
        active['mode'] = '挑毛中'
        ui.update_baseline(active, supervisor)
        for view in ui.baseline_views.values():
            assert all(view[key].options['state'] == 'normal' for key in ('toggle', 'exposure', 'reset'))
            assert not view['reason'].visible
        active['baseline']['preview']['front']['on'] = False
        ui.update_baseline(active, supervisor)
        assert all(ui.baseline_views['front'][key].options['state'] == 'disabled' for key in ('exposure', 'reset'))
        ui.update_baseline({'mode':'对刀基准建立中', 'command_seq':0, 'baseline':{
            'phase':'prepare', 'preview':{'front':{'on':True},'side':{'on':True,'error':'USB 带宽不足'}},
            'existing':{'tool1':{'camera_differences':['front.device']}}}})
        assert '基准需重建' in ui.baseline_notice.options['text']
        assert ui.baseline_results_card.visible is False
        assert ui.baseline_existing_card.visible is False
        assert ui.baseline_live.visible is False

        # Off camera shows '识别来源 —', and active session shows bottom cards again
        ui.update_baseline({'mode': '空闲', 'command_seq': 0, 'baseline': {
            'phase': 'idle', 'preview': {'front': {'on': False}, 'side': {'on': False}}}})
        assert '识别来源 —' in ui.baseline_views['front']['detail'].options['text']
        assert ui.baseline_results_card.visible is True
        assert ui.baseline_existing_card.visible is True
        ui.update_baseline({'mode': '对刀基准建立中', 'command_seq': 0, 'baseline': {
            'phase': 'live', 'tool': 1, 'preview': {'front': {'on': False}, 'side': {'on': False}}}})
        assert ui.baseline_results_card.visible is True
        assert ui.baseline_existing_card.visible is True
        assert ui.baseline_live.visible is True

        # Read live metrics from tooltip_preview.json and clear when expired (>3s)
        import json, os, time
        from datetime import datetime
        from nest_clear.paths import tooltip_preview_file
        preview_file = tooltip_preview_file(tmp_path)
        preview_file.parent.mkdir(parents=True, exist_ok=True)
        now_iso = datetime.now().astimezone().isoformat()
        preview_file.write_text(json.dumps({
            'updated_at': now_iso,
            'cameras': {'front': {'tenengrad': 1.28, 'peak': 1.30, 'percent': 98.0, 'edge_width_um': 27.2,
                                  'edge_width_px': 1.79, 'contrast': 210, 'source': 'tip', 'exposure_ms': 10.0,
                                  'history': [80, 98], 'updated_at': now_iso}}
        }), encoding='utf-8')
        low_main = {'mode': '对刀基准建立中', 'command_seq': 0, 'baseline': {
            'phase': 'prepare', 'preview': {'front': {'on': True, 'device': 'usb-0:4', 'exposure_source': '已测 2026-11-23'},
                                            'side': {'on': True, 'device': 'usb-0:8', 'exposure_source': '已测 2026-11-23'}}}}
        ui.update_baseline(low_main)
        assert '98%' in ui.baseline_views['front']['info'].options['text']
        low_main['baseline']['preview']['front']['exposure_limits'] = {'min': 1, 'max': 50, 'step': 1}
        ui.baseline_exposure_pending['front'] = (123, time.monotonic())
        low_main['command_seq'] = 123
        ui.update_baseline(low_main)
        assert 'front' in ui.baseline_exposure_pending
        low_main['baseline']['preview']['front']['exposure_result'] = {'seq': 123, 'state': 'done', 'message': '曝光回读 0.3 ms'}
        ui.update_baseline(low_main)
        assert 'front' not in ui.baseline_exposure_pending
        assert '0.3 ms' in ui.baseline_views['front']['exposure_result'].options['text']
        ui.baseline_exposure_pending['front'] = (124, time.monotonic() - 6)
        ui.update_baseline(low_main)
        assert '未完成' in ui.baseline_views['front']['exposure_result'].options['text']
        assert ui.baseline_views['side']['info'].options['text'] == '无实时数据'
        assert '对比度 —' in ui.baseline_views['side']['detail'].options['text']
        preview_file.write_text(json.dumps({'updated_at': now_iso, 'cameras': {'front': {
            'percent': None, 'detection_status': '未识别', 'detection_reason': 'no dark component',
            'focus_reason': '未识别到针尖', 'frame_age_ms': 1100, 'updated_at': now_iso}}}), encoding='utf-8')
        ui.update_baseline(low_main)
        assert '—%' in ui.baseline_views['front']['info'].options['text']
        assert '未识别到针尖' in ui.baseline_views['front']['detail'].options['text']
        assert '画面延迟' in ui.baseline_views['front']['detail'].options['text']
        assert ui.baseline_views['front']['detail'].options['fg'] == app.COLORS['red']
        old_t = time.time() - 5.0
        os.utime(preview_file, (old_t, old_t))
        ui.update_baseline(low_main)
        assert ui.baseline_views['front']['info'].options['text'] == '无实时数据'
        assert '对比度 —' in ui.baseline_views['front']['detail'].options['text']
        preview_file.unlink(missing_ok=True)

        assert len(ui.table.rows) == 3
        ui.update_tip({'phase':'confirm_move', 'tool':1, 'rounds':[{'iteration':1, 'stage':'对准', 'pixels':{}, 'delta':[.1,0,0], 'target':[1,2,3]}]})
        assert len(ui.tiptable.rows) == 1
        assert ui.tip_buttons['tooltip_confirm_move'].options['state'] == 'normal'
        assert ui.tip_buttons['tooltip_confirm_result'].options['state'] == 'disabled'
        ui.update_tip({'phase':'limit', 'tool':1, 'message':'修正量过大'})
        assert ui.tip_buttons['tooltip_confirm_move'].options['state'] == 'disabled'
        assert ui.tip_buttons['tooltip_click'].options['state'] == 'normal'
        ui.update_tip({'phase':'done', 'tool':1, 'rounds':[{'iteration':2,'stage':'对准', 'pixels':{'front':[320,241],'side':[321,242]}, 'deviation_mm':{'front':[.01,.02],'side':[-.03,.04]}, 'delta':[.01,.02,.03], 'target':None, 'confirmed_at':'2026-11-22T12:34:56.123456-08:00'}], 'jacobian_status':'已存', 'old_compensation':[1,2,3], 'new_compensation':[1.1,2,3], 'compensation_difference':[.1,0,0]})
        values = ui.tiptable.rows['0']['values']
        assert values[3] == '主 +0.010/+0.020 辅 -0.030/+0.040'
        assert values[5] == '—' and values[7] == '12:34:56'
        assert ui.tip_message.options['fg'] == app.COLORS['green']
        assert ui.tip_message.options['font'] == ui.tip_heading.options['font']
        assert 'None' not in ui.tip_details.options['text'] and '[' not in ui.tip_details.options['text'] and '{' not in ui.tip_details.options['text']
        ui.update_tip({'phase':'fail', 'tool':1, 'message':'检测失败', 'rounds':[]})
        assert ui.tip_details.options['text'].startswith('第 1 轮（已失败）')
        ui.update_tip({'phase':'limit', 'tool':1, 'message':'修正量过大'})
        assert ui.process_table.options['height'] == 2
        assert ui.pick_values['视野'].options['text'] == '11/24'
        assert ui.cam_step.values_data['检出角点'] == '35/40'
        assert ui.cam_step.values_data['步骤'] == '移动拍照'
        assert ui.cam_step.values_data['亮斑'] == '(510,120)–(560,170)'
        plan_titles = list(ui.plan.text_calls)
        for kind, expected in [('done','完成'), ('fail','失败'), ('other','other')]:
            ui.update_cam({'kind':kind, 'detail':{'bbox':[1,2,3,4],'area_px':42}})
            assert ui.cam_step.values_data['步骤'] == expected
            assert ui.cam_step.values_data['亮斑'] == '(1,2)–(3,4)，面积 42 px'
        ui.update_cam({'kind':'move', 'detail':{}})
        assert ui.cam_step.values_data['亮斑'] == '无'
        ui.update_cam({'kind': 'fail', 'code': 'CAMCAL-FLOW', 'reason': '到位超差', 'hint': '检查位置'})
        assert ui.cam_result_card.options['row'] == 1
        assert ui.cam_plan_card.options['row'] == 4
        assert ui.cam_result.options['height'] >= 5
        assert ui._cam_result_cache[0] == '失败：CAMCAL-FLOW\n到位超差\n建议：检查位置'
        ui.update_cam({'kind': 'move'})
        assert ui.cam_result_card.options['row'] == 4
        assert ui.cam_plan_card.options['row'] == 1
        assert len([t for t in plan_titles if t and t.startswith('Z')]) == 5
        assert {'Z0', 'Z-5', 'Z-2.5', 'Z+2.5', 'Z+5'} == {t.split(' · ')[0] for t in plan_titles if t and t.startswith('Z')}
        assert ui.reply_table.options['height'] == 5
        assert len(ui.reply_table.rows) == 5
        assert ui.tip_message.options['text'] == '修正量过大'
        if page == '主页':
            automatic = app.App(tmp_path, ignore_before_ns=0)
            automatic.refresh()
            assert automatic.tabs.selected == 1
            main = {'mode': '空闲'}
            original_file_data = ui.file_data
            ui.file_data = lambda path: main if path.name == 'main_status.json' else {}
            ui.refresh()
            ui.select_page('主页')
            for mode, expected in [('相机标定中', 1), ('挑毛中', 0),
                                   ('对刀中(工具1)', 5), ('对刀基准建立中', 4)]:
                main['mode'] = mode
                ui.refresh()
                assert ui.tabs.selected == expected
                if mode == '相机标定中':
                    ui.select_page('主页')
                    ui.refresh()
                    assert ui.tabs.selected == 0
                    ui.select_page('相机标定')
            main['mode'] = '空闲'
            ui.refresh()
            assert ui.tabs.selected == 4
            ui.last_mode = None  # 界面重启后首次读到正在运行的模式
            main['mode'] = '相机标定中'
            ui.refresh()
            assert ui.tabs.selected == 1
            ui.file_data = original_file_data
            ui.refresh()
        with patch('nest_clear.film_mode.read_mode', return_value=True):
            ui.refresh()
            assert ui.film_banner.visible
            assert ui.film_button.options['fg'] == '#b00020'
        original_file_data = ui.file_data
        ui.file_data = lambda path: ({'mode': '挑毛中', 'film_mode': True} if path.name == 'main_status.json' else original_file_data(path))
        with patch('nest_clear.film_mode.read_mode', return_value=False):
            ui.refresh()
            assert not ui.film_banner.visible
        import os
        from nest_clear.film_mode import start_tray, end_tray, read_mode
        supervisor = {'pid': 1, 'child_pid': os.getpid()}
        main = {'mode': '挑毛中', 'film_mode': True}
        ui.file_data = lambda path: main if path.name == 'main_status.json' else supervisor
        end_tray(tmp_path)
        ui.refresh()
        assert ui.film_button.options['state'] == 'normal'
        with patch.object(app, 'read_json', side_effect=lambda path: main if path.name == 'main_status.json' else supervisor):
            previous = read_mode(tmp_path)
            ui.toggle_film_mode()
            assert read_mode(tmp_path) is not previous
            start_tray(tmp_path)
            with patch.object(app.messagebox, 'showwarning') as warning:
                ui.toggle_film_mode()
                warning.assert_called_once()
            assert read_mode(tmp_path) is not previous
        ui.refresh()
        assert ui.film_button.options['state'] == 'disabled'
        assert ui.film_reason.options['text'] == '挑毛进行中，本盘结束后可切换'
        end_tray(tmp_path)
        main['mode'] = '相机标定中'
        ui.refresh()
        assert ui.film_button.options['state'] == 'disabled'
        assert ui.film_reason.options['text'] == '主程序正在执行其他任务，空闲后可切换'
        ui.file_data = original_file_data
        target = tmp_path / 'screen' / (page + '.png')
        class Capture:
            def save(self, path): path.write_bytes(b'screenshot')
        with patch.object(app.ImageGrab, 'grab', return_value=Capture()) as grab:
            ui.capture_and_exit(target)
            grab.assert_called_once_with(bbox=(5, 7, 605, 307))
        assert ui.root.destroyed and target.is_file()


def test_invalid_page_rejected(tmp_path, monkeypatch):
    from nest_clear.ui import app
    ui = object.__new__(app.App); ui.tabs = Widget()
    with pytest.raises(ValueError, match='字符串'): ui.select_page(Widget())
    with pytest.raises(ValueError, match='未知页面'): ui.select_page('无此页')


def test_reply_legacy_fallback_and_cache():
    from nest_clear.ui.app import App
    ui = object.__new__(App); ui.reply_table = Widget(); ui.cache = {}
    main = {'last_reply_time': '2026-01-02T11:00:00', 'last_reply_task_id': 42,
            'last_reply_request_func': 20, 'last_reply_func': 9}
    ui.update_reply(main, [])
    assert list(ui.reply_table.rows.values())[0]['values'][1:] == (42, '20 → 9', '正常应答')
    ui.update_reply(main, [{'task_id': 42, 'func': 9, 'phenomenon': '故障'}])
    assert list(ui.reply_table.rows.values())[0]['values'][3] == '故障'
    ui.update_reply({'recent_replies': []}, [])
    assert not ui.reply_table.rows


@pytest.mark.parametrize('alias,index', [('main',0),('home',0),('camcal',1),('camera',1),('offset',2),('tool-offset',2),('pickacc',3),('pick-accuracy',3),('baseline',4),('toolcal',5),('tooltip',5)])
def test_page_aliases(alias,index):
    from nest_clear.ui.app import App
    ui = object.__new__(App); ui.tabs = Widget(); ui.select_page(alias)
    assert ui.tabs.selected == index


def test_font_backend_check():
    from nest_clear.ui.app import supports_xft
    class TK:
        def __init__(self, result): self.result = result
        def call(self, *args):
            assert args == ('::tk::pkgconfig','get','fontsystem')
            if isinstance(self.result, Exception): raise self.result
            return self.result
    assert supports_xft(SimpleNamespace(tk=TK('xft')))
    assert not supports_xft(SimpleNamespace(tk=TK('x11')))
    assert not supports_xft(SimpleNamespace(tk=TK(RuntimeError('not supported'))))

def test_plan_layers_are_separate_and_show_relative_progress():
    from nest_clear.ui.app import App
    ui = object.__new__(App); ui.plan = Widget(); ui.cache = {}; ui.theme = SimpleNamespace(table=('x',11),body=('x',12))
    points = [[0,0,100], [1,0,100], [0,1,95], [0,0,97.5], [0,0,102.5], [0,0,105]]
    ui.draw_plan(points, 3, origin_z=100)
    labels = ui.plan.text_calls
    assert labels == ['Z0 · 2/2','Z-5 · 0/1','Z-2.5 · 0/1','Z+2.5 · 0/1','Z+5 · 0/1']

def test_key_value_wrap_tracks_card_width():
    from nest_clear.ui.app import KeyValueGrid
    class Key:
        def winfo_reqwidth(self): return 80
    class Value:
        def configure(self, **kwargs): self.options=kwargs
    value=Value()
    fake=SimpleNamespace(grid_slaves=lambda column: [Key()], values={'原因':value})
    KeyValueGrid._rewrap(fake, SimpleNamespace(width=320))
    assert value.options['wraplength']==228


@pytest.mark.parametrize('tool,color', [(1, '#2563eb'), (2, '#d56713')])
def test_offset_confirm_identifies_tool(tmp_path, monkeypatch, tool, color):
    from nest_clear.ui import app
    class Dialog(Widget):
        def title(self, value): self.window_title = value
    monkeypatch.setattr(app.tk, 'Toplevel', Dialog)
    monkeypatch.setattr(app.tk, 'Label', Widget)
    monkeypatch.setattr(app.ttk, 'Button', Widget)
    ui = object.__new__(app.App)
    ui.root = Widget(); ui.theme = SimpleNamespace(title=('font',13), body=('font',12))
    ui.offset_dialog = None
    ui.offset_confirm({'tool':tool, 'candidate':{}}, {})
    expected = f'确认覆盖【工具{tool}】的偏移标定结果'
    assert ui.offset_dialog.window_title == expected
    assert any(child.options.get('text') == expected and child.options.get('bg') == color for child in ui.offset_dialog.children)


@pytest.mark.parametrize('tool,color', [(1, '#2563eb'), (2, '#d56713')])
def test_baseline_confirm_identifies_tool(monkeypatch, tool, color):
    from nest_clear.ui import app
    class Dialog(Widget):
        def title(self, value): self.window_title = value
        def pack(self, *args, **kwargs): pass
    class PackWidget(Widget):
        def pack(self, *args, **kwargs): pass
    monkeypatch.setattr(app.tk, 'Toplevel', Dialog)
    monkeypatch.setattr(app.tk, 'Label', PackWidget)
    monkeypatch.setattr(app.tk, 'Frame', PackWidget)
    monkeypatch.setattr(app.ttk, 'Button', PackWidget)
    ui = object.__new__(app.App)
    ui.root = Widget(); ui.theme = SimpleNamespace(title=('font',13), body=('font',12))
    ui.baseline_dialog = None
    ui.baseline_confirm({'tool':tool, 'candidate':{'p0':[1,2,3], 'pixels':{'front':[320,240], 'side':[321,240]}},
                         'existing':{f'tool{tool}':{'time':'2026-11-21T01:00:00', 'p0':[1,2,2.9], 'pixels':{'front':[319,240], 'side':[320,240]}}}})
    expected = f'确认覆盖【工具{tool}】的对刀基准'
    assert ui.baseline_dialog.window_title == expected
    assert any(child.options.get('text') == expected and child.options.get('bg') == color for child in ui.baseline_dialog.children)


def test_restart_clears_old_images_keeps_status(tmp_path, monkeypatch):
    import json
    import os
    import time
    from datetime import datetime, timedelta
    from PIL import Image
    from nest_clear.paths import live_image_file, main_status_file, supervisor_state_file
    from nest_clear.ui import app

    names = (
        'latest', 'camcal_latest', 'preview_latest', 'tooloffset_latest', 'tooloffset_crop',
        'pickacc_latest', 'pickacc_crop', 'baseline_front', 'baseline_side',
        'calib_front_latest', 'calib_side_latest',
    )
    old_ns = time.time_ns() - 60_000_000_000
    for name in names:
        path = live_image_file(tmp_path, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (40, 30), 'blue').save(path, format='JPEG')
        os.utime(path, ns=(old_ns, old_ns))

    old_iso = (datetime.now().astimezone() - timedelta(minutes=5)).isoformat()
    now_iso = datetime.now().astimezone().isoformat()
    main_file = main_status_file(tmp_path)
    main_file.parent.mkdir(parents=True, exist_ok=True)
    main_file.write_text(json.dumps({
        'mode': '挑毛中',
        'picking': {'tray': 9, 'view_index': 2, 'total_views': 10, 'wall': old_iso},
        'camcal': {'event': 'camcal_done', 'sequence': 5, 'plan': [[0, 0, 0]], 'detail': {'detected_corners': 35, 'predicted_corners': 40}, 'wall': old_iso},
        'tooltip': {'tool': 1, 'iterations': 2, 'event': 'tooltip_aligned', 'wall': old_iso, 'history': [{'event': 'tooltip_detect', 'wall': old_iso}]},
        'last_reply_time': old_iso,
        'last_reply_task_id': 99,
        'last_reply_func': 1,
        'recent_replies': [{'time': old_iso, 'task_id': 99, 'request_func': 1, 'reply_func': 1}],
    }), encoding='utf-8')
    os.utime(main_file, ns=(old_ns, old_ns))
    sup_file = supervisor_state_file(tmp_path)
    sup_file.write_text(json.dumps({
        'pid': 123,
        'child_state': '运行',
        'plc_connected': True,
        'plc_last_read': now_iso,
        'heartbeat_last_write': now_iso,
        'algorithm_ready': 1,
    }), encoding='utf-8')
    os.utime(sup_file, ns=(old_ns, old_ns))

    for name in ('Tk', 'Frame', 'Label', 'Canvas', 'Toplevel', 'Text', 'Button'):
        monkeypatch.setattr(app.tk, name, Widget)
    for name in ('Notebook', 'Treeview', 'Progressbar', 'Scrollbar', 'Button', 'Entry', 'Combobox', 'Spinbox'):
        monkeypatch.setattr(app.ttk, name, Widget)
    monkeypatch.setattr(app.ttk, 'Style', Style)
    monkeypatch.setattr(app.tkfont, 'families', lambda root: ['Noto Sans Mono CJK SC'])

    class FakeCard(Widget):
        def __init__(self, parent, title, theme):
            super().__init__(parent); self.body = Widget(self); self.title_label = Widget(self)

    class FakeKeyValueGrid(Widget):
        def __init__(self, parent, keys, theme):
            super().__init__(parent); self.values_data = {}
        def set(self, values):
            self.values_data = values

    class FakePhoto:
        def __init__(self, img):
            self.img = img

    class FakePic(Widget):
        def __init__(self, parent, path, theme):
            super().__init__(parent, text='暂无图像')
            self.path = path
            self.min_mtime_ns = 0
            self.mtime = None
            self.image = None
            self.photo = None
            self.results = app.queue.Queue()
            self.busy = False
            self.size = None
            self.aspect_changed = None
        poll = app.Picture.poll

    monkeypatch.setattr(app, 'Card', FakeCard)
    monkeypatch.setattr(app, 'KeyValueGrid', FakeKeyValueGrid)
    monkeypatch.setattr(app, 'Picture', FakePic)
    monkeypatch.setattr(app.ImageTk, 'PhotoImage', FakePhoto)

    ui = app.App(tmp_path)
    ui.refresh()
    # 业务历史仍显示，过期的父程序状态须标记过期。
    assert ui.lights['父程序'][1].options['text'].startswith('未运行(')
    assert ui.lights['父程序'][0].options['fg'] == app.COLORS['gray']
    assert ui.lights['40932 就绪'][1].options['text'] == '1(过期)'
    os.utime(sup_file, None)
    ui.refresh()
    assert ui.lights['父程序'][1].options['text'] == '123'
    assert ui.lights['40932 就绪'][1].options['text'] == '1'
    assert ui.pick_values['盘号'].options['text'] == '9'
    assert len(ui.reply_table.rows) == 1
    assert ui.cam_step.values_data['检出角点'] == '35/40'
    for pic in ui.pictures:
        pic.poll()
        assert pic.image is None
        assert pic.options.get('text') == '暂无图像'

    # 写入启动后新图片，窗口应正常加载显示。
    for name in names:
        Image.new('RGB', (40, 30), 'green').save(live_image_file(tmp_path, name), format='JPEG')
    for pic in ui.pictures:
        pic.poll()
        for _ in range(40):
            pic.poll()
            if pic.image is not None:
                break
            time.sleep(0.01)
        assert pic.image is not None
        assert pic.options.get('text') == ''


def test_reply_detail_shows_full_reason(monkeypatch):
    from nest_clear.ui import app
    created = []
    class Text(Widget):
        def insert(self, index, value):
            self.content = value
            created.append(self)
    monkeypatch.setattr(app.tk, 'Toplevel', Widget)
    monkeypatch.setattr(app.tk, 'Text', Text)
    monkeypatch.setattr(app.ttk, 'Scrollbar', Widget)
    ui = object.__new__(app.App)
    ui.root = Widget(); ui.theme = SimpleNamespace(mono=('monospace', 10))
    ui.reply_table = Widget()
    reason = '相机标定失败：' + '检查位姿与光照。' * 30
    row = ui.reply_table.insert('', 'end', values=('12:00', 42, '80 → 82', reason))
    ui.reply_table.sel = (row,)
    ui.reply_detail()
    assert reason in created[0].content
    assert created[0].options['state'] == 'disabled'


def test_cam_scroll_geometry_coalesces_feedback_and_ignores_content_width():
    from nest_clear.ui.app import ScrollGeometry

    class Canvas:
        def __init__(self):
            self.queue = []; self.width_updates = []; self.regions = []; self.content_width = 343
        def after_idle(self, callback): self.queue.append(callback)
        def bbox(self, _tag): return (0, 0, self.content_width, 1030)
        def itemconfigure(self, _window, **kwargs):
            self.width_updates.append(kwargs['width'])
            geometry.content_changed(None)
        def configure(self, **kwargs):
            self.regions.append(kwargs['scrollregion'])
            self.content_width = 346 if self.content_width == 343 else 343
            geometry.content_changed(None)

    canvas = Canvas(); geometry = ScrollGeometry(canvas, 1)
    for _ in range(10): geometry.content_changed(None)
    assert not canvas.queue and not canvas.regions
    for _ in range(10): geometry.viewport_changed(SimpleNamespace(width=450))
    assert len(canvas.queue) == 1
    steps = 0
    while canvas.queue and steps < 10:
        canvas.queue.pop(0)(); steps += 1
    assert not canvas.queue and steps == 2
    assert canvas.width_updates == [450]
    assert canvas.regions == [(0, 0, 450, 1030)]
    canvas.content_width = 343
    geometry.content_changed(None)
    canvas.queue.pop(0)()
    assert not canvas.queue and len(canvas.regions) == 1


def test_page_commands_timeout_and_recover(tmp_path, monkeypatch, caplog):
    from nest_clear.ui import app
    now = [100.0]
    monkeypatch.setattr(app.time, 'monotonic', lambda: now[0])
    ui = object.__new__(app.App)
    ui.directory = tmp_path
    for page in ('baseline', 'offset', 'pickacc', 'tip'):
        setattr(ui, page + '_message', Widget())
        setattr(ui, page + '_pending', 0)
        ui._send_page_command(page, page + '_test')
        seq = getattr(ui, page + '_pending')
        ui._send_page_command(page, 'second')
        assert getattr(ui, page + '_pending') == seq
        assert getattr(ui, page + '_message').options['text'] == '上一条命令处理中'
        now[0] += 5.1
        ui._check_page_command(page, 0)
        assert getattr(ui, page + '_pending') == 0
        assert page + '_test' in getattr(ui, page + '_command_message')
        ui._send_page_command(page, 'retry')
        ui._check_page_command(page, getattr(ui, page + '_pending'))
        assert not getattr(ui, page + '_pending')
    assert '未被确认' in caplog.text


def test_system_python_ui_imports_without_vision_dependencies(tmp_path):
    import os
    import subprocess
    code = '''
import sys
for name in ('cv2', 'numpy', 'scipy', 'yaml'):
    sys.modules[name] = None
from nest_clear.ui import app
from nest_clear.ui.__main__ import add_view_arguments
from nest_clear.devices.v4l2_enum import enumerate_capture_devices
from nest_clear.tooloffset.commands import send
from nest_clear.preview.eligibility import preview_eligibility
from nest_clear.film_mode import may_change_mode, read_mode, write_mode, mode_lock, tray_active
from nest_clear.record.events import EventRecorder
from nest_clear.ui.model import supervisor_tooltip_main
from nest_clear.preview.focus_record import write_record
assert enumerate_capture_devices(sys.argv[1]) == []
'''
    env = dict(os.environ, PYTHONPATH=str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
    result = subprocess.run(['/usr/bin/python3', '-c', code, str(tmp_path)], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_focus_requests_timeout(tmp_path, monkeypatch, caplog):
    from nest_clear.ui import app
    monkeypatch.setattr(app.time, 'monotonic', lambda: 10)
    ui = object.__new__(app.App)
    for name in ('temporary', 'wb', 'sweep'):
        label = Widget()
        setattr(ui, name + '_pending', True)
        setattr(ui, name + '_pending_since', 4)
        ui._expire_focus_request(name, label)
        assert not getattr(ui, name + '_pending')
        assert '未被确认' in label.options['text']
    assert '未被确认' in caplog.text


def test_exposure_sweep_ack_matches_request(monkeypatch):
    import json
    from nest_clear.ui import app
    monkeypatch.setattr(app.time, 'monotonic', lambda: 10)
    ui = object.__new__(app.App)
    ui.sweep_pending = True; ui.sweep_pending_since = 9; ui.sweep_request = 42
    ui.wb_pending = False; ui.exposure_button = Widget(); ui.exposure_mode = Widget()
    ui.exposure_message = Widget(); ui.exposure_visible = True
    stale = {'state': 'done', 'request_seq': 41}
    ui.exposure_cache = json.dumps([stale, None], ensure_ascii=False, sort_keys=True)
    ui.update_exposure({'exposure_sweep': stale})
    assert ui.sweep_pending
    completed = {'state': 'done', 'request_seq': 42}
    ui.exposure_cache = json.dumps([completed, None], ensure_ascii=False, sort_keys=True)
    ui.update_exposure({'exposure_sweep': completed})
    assert not ui.sweep_pending
