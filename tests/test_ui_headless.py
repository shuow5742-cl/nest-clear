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
    def winfo_rootx(self): return 5
    def winfo_rooty(self): return 7
    def rowconfigure(self, index, **kwargs): self.row_options.setdefault(index, {}).update(kwargs)
    def columnconfigure(self, *args, **kwargs): pass
    def grid(self, *args, **kwargs): pass
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
    def create_polygon(self, *args, **kwargs): pass
    def create_image(self, *args, **kwargs): pass
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
    for name in ('Tk', 'Frame', 'Label', 'Canvas', 'Toplevel', 'Text'):
        monkeypatch.setattr(app.tk, name, Widget)
    for name in ('Notebook', 'Treeview', 'Progressbar', 'Scrollbar', 'Button'):
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
        ui = app.App(tmp_path, page=page)
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
        ui.refresh()
        assert all(b.options['state'] == 'disabled' for b in ui.offset_tool_buttons + ui.offset_mode_buttons)
        assert ui.offset_start.options['state'] == 'disabled'
        assert ui.offset_tool_buttons[0].options['style'] == 'OffsetSelected.TButton'
        ui.update_offset({'tooloffset': {'phase':'select'}, 'command_seq':0})
        assert all(b.options['state'] == 'normal' for b in ui.offset_tool_buttons + ui.offset_mode_buttons)
        ui.update_offset({'tooloffset': {'phase':'edge', 'tool':2, 'mode':'center_edge', 'edges':[{'target':'T12','corner':2,'delta_xy_mm':[0,0], 'distance_mm':0}]}, 'command_seq':0})
        assert all(b.options['state'] == 'disabled' for b in ui.offset_tool_buttons + ui.offset_mode_buttons)
        assert ui.offset_edges.rows['0']['values'][1] == '左下'
        assert len(ui.table.rows) == 3
        assert len(ui.tiptable.rows) == 2
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
        assert len([t for t in plan_titles if t and t.startswith('Z')]) == 5
        assert {'Z0', 'Z-5', 'Z-2.5', 'Z+2.5', 'Z+5'} == {t.split(' · ')[0] for t in plan_titles if t and t.startswith('Z')}
        assert ui.reply_table.options['height'] == 5
        assert len(ui.reply_table.rows) == 5
        assert ui.tip_result.values_data['迭代次数'] == 2
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
