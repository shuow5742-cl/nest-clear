"""基于共享文件的 Tk 界面；预览命令写入文件，图片解码与缩放在后台线程完成。"""
import json
import logging
import os
import time
from datetime import datetime
import queue
import subprocess
import threading
from datetime import date
from pathlib import Path
import tkinter as tk
from tkinter import font as tkfont, ttk

from PIL import Image, ImageGrab, ImageTk

from nest_clear.paths import live_image_file, main_status_file, supervisor_state_file, ui_command_file
from .model import (EventTail, camcal_progress, decode_status, display_time, duration,
                    read_json, status_indicators, tooltip_progress, preview_controls)

COLORS = {'green': '#168448', 'yellow': '#916300', 'red': '#b42318',
          'gray': '#85909f', 'blue': '#2563eb'}
BG = '#F3F4F6'
BORDER = '#D0D5DD'
INK = '#1d2939'
MUTED = '#667085'
FONT_WARNING = '界面字体系统不支持抗锯齿，请用系统 Python 运行'
logger = logging.getLogger(__name__)
WHITE = '#ffffff'


def select_font():
    """从 fontconfig 的实际字体族和字重中选择，避免合成粗体。"""
    try:
        output = subprocess.run(['fc-list', '--format=%{family}\t%{style}\n'], capture_output=True,
                                text=True, timeout=3, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return 'TkDefaultFont', False
    rows = [line.split('\t', 1) for line in output.splitlines() if '\t' in line]
    for family in ('Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'AR PL UMing'):
        matches = [(names, style) for names, style in rows if any(
            n.strip() == family or (family == 'AR PL UMing' and n.strip().startswith(family))
            for n in names.split(','))]
        if matches:
            chosen = next((n.strip() for n in matches[0][0].split(',')
                           if n.strip() == family or n.strip().startswith(family)), family)
            return chosen, any('bold' in style.lower() for _, style in matches)
    return 'TkDefaultFont', False


def supports_xft(root):
    try:
        return str(root.tk.call('::tk::pkgconfig', 'get', 'fontsystem')).lower() == 'xft'
    except Exception:
        return False


class Theme:
    """所有显示字体、DPI 和 ttk 外观集中在这里。"""
    def __init__(self, root):
        root.tk.call('tk', 'scaling', root.winfo_fpixels('1i') / 72.0)
        family, has_bold = select_font()
        available = set(tkfont.families(root))
        mono = 'Noto Sans Mono CJK SC' if 'Noto Sans Mono CJK SC' in available else family
        self.body = (family, 12)
        self.title = (family, 13, 'bold' if has_bold else 'normal')
        self.status = (family, 13)
        self.tab = (family, 13)
        self.table = (family, 11)
        self.mono = (mono, 12)
        self.mono_status = (mono, 13)
        self.mono_table = (mono, 11)
        self.large = (mono, 18, 'bold' if has_bold and mono == family else 'normal')
        root.option_add('*Font', self.body)
        root.configure(bg=BG)
        style = ttk.Style(root)
        style.theme_use('clam')
        style.configure('.', font=self.body, background=BG, foreground=INK)
        style.configure('TFrame', background=BG)
        style.configure('TNotebook', background=BG, borderwidth=0)
        style.configure('TNotebook.Tab', font=self.tab, padding=(14, 6), background='#e4e7ec', foreground=INK)
        style.map('TNotebook.Tab', background=[('selected', WHITE)], foreground=[('selected', '#175cd3')])
        style.configure('Treeview', font=self.mono_table, rowheight=26, background=WHITE,
                        fieldbackground=WHITE, foreground=INK, bordercolor=BORDER, borderwidth=1)
        style.configure('Treeview.Heading', font=self.table, background='#eaecf0', foreground=INK, relief='flat')
        style.map('Treeview', background=[('selected', '#dbeafe')], foreground=[('selected', INK)])
        style.configure('Horizontal.TProgressbar', background=COLORS['blue'], troughcolor='#e4e7ec',
                        bordercolor='#e4e7ec', lightcolor=COLORS['blue'], darkcolor=COLORS['blue'])


class Card(tk.Frame):
    def __init__(self, parent, title, theme):
        super().__init__(parent, bg=WHITE, highlightbackground=BORDER, highlightcolor=BORDER, highlightthickness=1)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        tk.Label(self, text=title, font=theme.title, bg=WHITE, fg=INK, anchor='w').grid(
            row=0, column=0, sticky='ew', padx=8, pady=(5, 0))
        self.body = tk.Frame(self, bg=WHITE)
        self.body.grid(row=1, column=0, sticky='nsew', padx=8, pady=(2, 6))
        self.body.rowconfigure(0, weight=1)
        self.body.columnconfigure(0, weight=1)


class KeyValueGrid(tk.Frame):
    """按字体自然高度排布多行文本；宽度变化时更新值列换行。"""
    def __init__(self, parent, keys, theme):
        super().__init__(parent, bg=WHITE)
        self.columnconfigure(1, weight=1)
        self.values = {}
        for index, key in enumerate(keys):
            label = tk.Label(self, text=key, font=theme.body, bg=WHITE, fg=MUTED, anchor='ne', justify='right')
            label.grid(row=index, column=0, sticky='ne', padx=(0, 8), pady=2)
            value = tk.Label(self, text='—', font=theme.mono, bg=WHITE, fg=INK, anchor='nw', justify='left', wraplength=200)
            value.grid(row=index, column=1, sticky='new', pady=2)
            self.values[key] = value
        self.bind('<Configure>', self._rewrap)

    def _rewrap(self, event):
        key_width = max((child.winfo_reqwidth() for child in self.grid_slaves(column=0)), default=0)
        width = max(40, event.width - key_width - 12)
        for value in self.values.values():
            value.configure(wraplength=width)

    def set(self, values):
        for key, widget in self.values.items():
            value = values.get(key)
            widget.configure(text=str(value) if value is not None else '—')


class Picture(tk.Label):
    def __init__(self, parent, path, theme):
        super().__init__(parent, text='暂无图片', anchor='center', bg='#eef1f5', fg=MUTED, font=theme.body)
        self.path = path
        self.mtime = None
        self.image = None
        self.photo = None
        self.results = queue.Queue()
        self.busy = False
        self.size = None
        self.bind('<Button-1>', self.enlarge)

    def poll(self):
        try:
            stamp = self.path.stat().st_mtime_ns
        except OSError:
            stamp = None
        size = (max(1, self.winfo_width()), max(1, self.winfo_height()))
        if stamp is not None and (stamp != self.mtime or size != self.size) and not self.busy and min(size) > 1:
            self.busy = True
            path = self.path
            def load():
                try:
                    with Image.open(path) as source:
                        original = source.copy()
                    preview = original.copy()
                    preview.thumbnail(size, Image.Resampling.LANCZOS)
                    self.results.put((path, stamp, size, original, preview))
                except Exception:
                    self.results.put((path, None, size, None, None))
            threading.Thread(target=load, daemon=True).start()
        try:
            path, stamp, size, original, preview = self.results.get_nowait()
        except queue.Empty:
            return
        self.busy = False
        if original is not None and path == self.path:
            self.image, self.mtime, self.size = original, stamp, size
            self.photo = ImageTk.PhotoImage(preview)
            self.configure(image=self.photo, text='')

    def enlarge(self, _event=None):
        if self.image is None:
            return
        top = tk.Toplevel(self)
        top.title(str(self.path))
        top.rowconfigure(0, weight=1); top.columnconfigure(0, weight=1)
        canvas = tk.Canvas(top, width=1000, height=700, scrollregion=(0, 0, *self.image.size))
        canvas.grid(row=0, column=0, sticky='nsew')
        photo = ImageTk.PhotoImage(self.image)
        canvas.create_image(0, 0, image=photo, anchor='nw'); canvas.image = photo
        canvas.bind('<ButtonPress-1>', lambda event: canvas.scan_mark(event.x, event.y))
        canvas.bind('<B1-Motion>', lambda event: canvas.scan_dragto(event.x, event.y, gain=1))


class App:
    PAGE_NAMES = ('主页', '相机标定', '工具偏移标定', '挑取精度标定', '对刀基准建立', '换针对刀')
    PAGE_ALIASES = {'home': '主页', 'main': '主页', 'camera': '相机标定', 'camcal': '相机标定',
                    'offset': '工具偏移标定', 'tool-offset': '工具偏移标定',
                    'pickacc': '挑取精度标定', 'pick-accuracy': '挑取精度标定',
                    'baseline': '对刀基准建立', 'toolcal': '换针对刀', 'tooltip': '换针对刀'}

    def __init__(self, root, refresh_ms=300, fullscreen=False, page='主页', screenshot=None, screenshot_after_ms=None):
        self.directory = Path(root); self.interval = refresh_ms
        self.root = tk.Tk(); self.root.title('Nest Clear 算法状态')
        self.theme = Theme(self.root)
        self.root.minsize(1366, 768)
        self.root.geometry(f'{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}+0+0')
        try:
            self.root.state('zoomed')
        except tk.TclError:
            try:
                self.root.attributes('-zoomed', True)
            except tk.TclError:
                pass
        if fullscreen:
            self.root.attributes('-fullscreen', True)
        self.root.bind('<Escape>', lambda event: self.root.attributes('-fullscreen', False))
        self.root.columnconfigure(0, weight=1); self.root.rowconfigure(2, weight=1)
        if not supports_xft(self.root):
            logger.warning(FONT_WARNING)
            banner = tk.Label(self.root, text=FONT_WARNING, bg='#fef0c7', fg='#7a4b00', font=self.theme.status, anchor='w', padx=8, pady=4)
            banner.grid(row=0, column=0, sticky='ew', padx=6, pady=(4, 0))
        self.commtest_banner = tk.Label(self.root, text='通信联调模式：下发坐标为假点', bg='#b00020', fg='white', font=self.theme.title, padx=8, pady=4)
        self.tail = EventTail(root); self.pictures = []; self.cache = {}; self.events = []; self.files = {}
        self.build_status()
        self.tabs = ttk.Notebook(self.root)
        self.tabs.grid(row=2, column=0, sticky='nsew', padx=6, pady=(0, 6))
        self.home = self.new_page('主页'); self.build_home()
        self.cam = self.new_page('相机标定'); self.build_cam()
        for title in ('工具偏移标定', '挑取精度标定', '对刀基准建立'):
            placeholder = self.new_page(title)
            self.build_placeholder(placeholder, title)
        self.tip = self.new_page('换针对刀'); self.build_tip()
        self.select_page(page)
        self.root.after(50, self.refresh)
        if screenshot is not None and screenshot_after_ms is not None:
            self.root.after(screenshot_after_ms, lambda: self.capture_and_exit(screenshot))

    def new_page(self, title):
        frame = tk.Frame(self.tabs, bg=BG)
        self.tabs.add(frame, text=title)
        return frame

    def select_page(self, page_name):
        if not isinstance(page_name, str):
            raise ValueError('页面名称必须为字符串')
        page_name = self.PAGE_ALIASES.get(page_name, page_name)
        if page_name.endswith('页') and page_name not in self.PAGE_NAMES:
            page_name = page_name[:-1]
        if page_name not in self.PAGE_NAMES:
            raise ValueError(f'未知页面：{page_name}')
        self.tabs.select(self.PAGE_NAMES.index(page_name))

    def capture_and_exit(self, path):
        try:
            self.root.update_idletasks()
            x, y = self.root.winfo_rootx(), self.root.winfo_rooty()
            width, height = self.root.winfo_width(), self.root.winfo_height()
            target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
            ImageGrab.grab(bbox=(x, y, x + width, y + height)).save(target)
        finally:
            self.root.destroy()

    def card(self, parent, title, row, column, **options):
        card = Card(parent, title, self.theme)
        card.grid(row=row, column=column, sticky='nsew', padx=3, pady=3, **options)
        return card.body

    def label(self, parent, text='', **kwargs):
        kwargs.setdefault('bg', WHITE); kwargs.setdefault('fg', INK)
        kwargs.setdefault('font', self.theme.body); kwargs.setdefault('anchor', 'w')
        return tk.Label(parent, text=text, **kwargs)

    def picture(self, parent, name):
        widget = Picture(parent, live_image_file(self.directory, name), self.theme)
        widget.grid(row=0, column=0, sticky='nsew')
        self.pictures.append(widget)
        return widget

    def build_status(self):
        bar = tk.Frame(self.root, bg=BG)
        bar.grid(row=1, column=0, sticky='ew', padx=3, pady=3)
        self.lights = {}
        for i, name in enumerate(('父程序', '主程序', 'PLC 连接', '心跳', '40932 就绪')):
            bar.columnconfigure(i, weight=1, uniform='pills')
            pill = tk.Frame(bar, bg=WHITE, highlightbackground=BORDER, highlightthickness=1)
            pill.grid(row=0, column=i, sticky='ew', padx=3)
            pill.columnconfigure(2, weight=1)
            dot = tk.Label(pill, text='●', bg=WHITE, fg=COLORS['gray'], font=self.theme.status)
            dot.grid(row=0, column=0, padx=(8, 3), pady=4)
            label = tk.Label(pill, text=name, bg=WHITE, fg=INK, font=self.theme.status, anchor='w')
            label.grid(row=0, column=1, sticky='w', pady=4)
            value = tk.Label(pill, text='无数据', bg=WHITE, fg=INK, font=self.theme.mono_status, anchor='w')
            value.grid(row=0, column=2, sticky='ew', padx=(6, 0), pady=4)
            for widget in (pill, dot, label, value):
                widget.bind('<Button-1>', lambda event, item=name: self.show_status(item))
            self.lights[name] = (dot, value)

    def table_widget(self, parent, columns, widths, stretch=(), height=1):
        tree = ttk.Treeview(parent, columns=columns, show='headings', height=height)
        tree.grid(row=0, column=0, sticky='nsew')
        for column, width in zip(columns, widths):
            tree.heading(column, text=column)
            tree.column(column, width=width, minwidth=35, stretch=column in stretch, anchor='w')
        return tree

    def build_home(self):
        self.home.columnconfigure(0, weight=38, uniform='homecols')
        self.home.columnconfigure(1, weight=62, uniform='homecols')
        self.home.rowconfigure(0, weight=65, uniform='homerows')
        self.home.rowconfigure(1, weight=35, uniform='homerows')
        left = tk.Frame(self.home, bg=BG)
        left.grid(row=0, column=0, sticky='nsew')
        left.columnconfigure(0, weight=1)
        for i, weight in enumerate((4, 3, 3, 2)):
            left.rowconfigure(i, weight=weight)
        left.rowconfigure(1, minsize=120)
        device = self.card(left, '设备状态 40901', 0, 0)
        for i in range(4): device.rowconfigure(i, weight=1)
        for i in range(3): device.columnconfigure(i, weight=1, uniform='bits')
        self.bits = []
        for i in range(12):
            label = self.label(device)
            label.grid(row=i // 3, column=i % 3, sticky='w')
            self.bits.append(label)
        process = self.card(left, '进程', 1, 0)
        cols = ('程序', 'PID', '启动时间', '运行时长', '状态', '最近退出原因')
        self.process_table = self.table_widget(process, cols, (55, 55, 75, 75, 60, 100), stretch=('最近退出原因',), height=2)
        self.process_table.insert('', 'end', iid='parent'); self.process_table.insert('', 'end', iid='child')
        pick = self.card(left, '挑毛进度', 2, 0)
        self.pick_values = {}
        for i, title in enumerate(('盘号', '视野', '粗拍次数', '可去数量', '红框数量', '已下发')):
            r, c = divmod(i, 2)
            pick.rowconfigure(r, weight=1); pick.columnconfigure(c * 2 + 1, weight=1)
            self.label(pick, title + '  ', fg=MUTED).grid(row=r, column=c * 2, sticky='w')
            value = self.label(pick, '—', font=self.theme.mono)
            value.grid(row=r, column=c * 2 + 1, sticky='w'); self.pick_values[title] = value
        reply = self.card(left, '最近应答', 3, 0)
        self.reply_info = KeyValueGrid(reply, ('时间', 'Task_ID', '功能码', '原因'), self.theme)
        self.reply_info.grid(row=0, column=0, sticky='new')
        image = self.card(self.home, '最新标注图 · 点击放大', 0, 1)
        self.picture(image, 'latest')
        logs = self.card(self.home, '异常日志 · 双击查看详情', 1, 0, columnspan=2)
        cols = ('时间', '类别', 'Task_ID', '功能码', '现象', '位置', '排查建议')
        self.table = self.table_widget(logs, cols, (90, 70, 70, 70, 250, 200, 250), stretch=('现象', '排查建议'))
        for color in ('red', 'yellow', 'green'):
            for stripe, bg in (('even', WHITE), ('odd', '#f8fafc')):
                self.table.tag_configure(f'{color}_{stripe}', foreground=COLORS[color], background=bg)
        self.table.bind('<Double-1>', self.detail)
        scrollbar = ttk.Scrollbar(logs, orient='vertical', command=self.table.yview)
        scrollbar.grid(row=0, column=1, sticky='ns'); self.table.configure(yscrollcommand=scrollbar.set)

    def build_cam(self):
        self.cam.columnconfigure(0, weight=65, uniform='camcols'); self.cam.columnconfigure(1, weight=35, uniform='camcols')
        self.cam.rowconfigure(0, weight=1)
        self.cam_picture = self.picture(self.card(self.cam, '主相机实时标注 · 点击放大', 0, 0), 'camcal_latest')
        right = tk.Frame(self.cam, bg=BG); right.grid(row=0, column=1, sticky='nsew')
        right.columnconfigure(0, weight=1)
        for row, weight in enumerate((3, 3, 3, 5, 2)): right.rowconfigure(row, weight=weight)
        progress = self.card(right, '标定进度', 0, 0)
        progress.rowconfigure(0, weight=1); progress.rowconfigure(1, weight=1)
        self.progress_text = self.label(progress, '无数据', font=self.theme.large)
        self.progress_text.grid(row=0, column=0, sticky='w')
        self.progress = ttk.Progressbar(progress, maximum=1)
        self.progress.grid(row=1, column=0, sticky='ew', pady=4)
        self.preview_button = ttk.Button(progress, text='打开相机预览', command=self.toggle_preview)
        self.preview_button.grid(row=2, column=0, sticky='w')
        self.preview_status = self.label(progress, '已关闭')
        self.preview_status.grid(row=3, column=0, sticky='w')
        plan = self.card(right, '计划点 · XY 平面', 1, 0)
        self.plan = tk.Canvas(plan, bg=WHITE, highlightthickness=0)
        self.plan.grid(row=0, column=0, sticky='nsew')
        step = self.card(right, '当前步', 2, 0)
        self.cam_step = KeyValueGrid(step, ('步骤', '检出角点', '饱和比例', '亮斑'), self.theme)
        self.cam_step.grid(row=0, column=0, sticky='new')
        preview = self.card(right, '预览读数', 3, 0)
        self.preview_values = KeyValueGrid(preview, ('模组 XYZ', '清晰度', '检出角点', '板中心偏移 mm', '提示'), self.theme)
        self.preview_values.grid(row=0, column=0, sticky='new')
        result = self.card(right, '结果', 4, 0)
        self.cam_result = tk.Label(result, text='标定进行中', bg='#ecfdf3', fg=COLORS['green'],
                                   font=self.theme.body, justify='left', anchor='nw', wraplength=460, padx=6, pady=4)
        self.cam_result.grid(row=0, column=0, sticky='nsew')

    def build_tip(self):
        self.tip.columnconfigure(0, weight=1, uniform='tipcols'); self.tip.columnconfigure(1, weight=1, uniform='tipcols')
        self.tip.rowconfigure(0, weight=55, uniform='tiprows'); self.tip.rowconfigure(1, weight=45, uniform='tiprows')
        self.picture(self.card(self.tip, '正面相机 · 点击放大', 0, 0), 'calib_front_latest')
        self.picture(self.card(self.tip, '侧面相机 · 点击放大', 0, 1), 'calib_side_latest')
        iterations = self.card(self.tip, '对刀迭代', 1, 0)
        cols = ('轮次', 'Δx', 'Δy', 'Δz', '正面宽度/清晰', '侧面宽度/清晰', '可靠性')
        self.tiptable = self.table_widget(iterations, cols, (50, 65, 65, 65, 135, 135, 95),
                                          stretch=('正面宽度/清晰', '侧面宽度/清晰', '可靠性'))
        result = self.card(self.tip, '对刀结果', 1, 1)
        self.tip_result = KeyValueGrid(result, ('工具号', '迭代次数', '状态', '原因', '补偿'), self.theme)
        self.tip_result.grid(row=0, column=0, sticky='new')

    def build_placeholder(self, parent, title):
        parent.columnconfigure(0, weight=1); parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)
        for col, label in enumerate(('图像窗', '步骤列表')):
            body = self.card(parent, title + ' · ' + label, 0, col)
            tk.Label(body, text='流程待定', bg=WHITE, fg=MUTED, font=self.theme.title).grid(row=0, column=0)

    def show_status(self, name):
        top = tk.Toplevel(self.root); top.title(name + ' 状态详情')
        top.rowconfigure(0, weight=1); top.columnconfigure(0, weight=1)
        text = tk.Text(top, width=90, height=20, font=self.theme.mono)
        text.grid(row=0, column=0, sticky='nsew')
        path = main_status_file(self.directory) if name == '主程序' else supervisor_state_file(self.directory)
        text.insert('1.0', json.dumps(self.file_data(path), ensure_ascii=False, indent=2))
        text.config(state='disabled')

    def detail(self, _event):
        selected = self.table.selection()
        if not selected: return
        row = self.events[int(selected[0])]['raw']
        top = tk.Toplevel(self.root); top.title('异常详情')
        top.rowconfigure(0, weight=1); top.columnconfigure(0, weight=1)
        text = tk.Text(top, width=100, height=24, font=self.theme.mono)
        text.grid(row=0, column=0, sticky='nsew')
        text.insert('1.0', json.dumps(row, ensure_ascii=False, indent=2)); text.config(state='disabled')
        index = 1
        for key, value in row.items():
            if isinstance(value, str) and 'path' in key:
                path = Path(value) if Path(value).is_absolute() else self.directory / value
                if path.is_file():
                    ttk.Button(top, text='查看 ' + key, command=lambda image_path=path: self.open_image(image_path)).grid(row=index, column=0, sticky='w')
                    index += 1

    def open_image(self, path):
        try:
            widget = Picture(self.root, Path(path), self.theme)
            with Image.open(path) as source: widget.image = source.copy()
            widget.enlarge(); widget.destroy()
        except OSError:
            pass

    def file_data(self, path):
        try: stamp = path.stat().st_mtime_ns
        except OSError: stamp = None
        if path not in self.files or self.files[path][0] != stamp:
            self.files[path] = (stamp, read_json(path))
        return self.files[path][1]

    @staticmethod
    def val(data, *keys, default='—'):
        return next((data[key] for key in keys if data.get(key) is not None), default)

    def refresh(self):
        supervisor = self.file_data(supervisor_state_file(self.directory))
        main = self.file_data(main_status_file(self.directory))
        if main.get('run_mode') == 'commtest': self.commtest_banner.grid(row=0, column=0, sticky='ew')
        elif hasattr(self.commtest_banner, 'grid_remove'): self.commtest_banner.grid_remove()
        for name, (color, value) in status_indicators(supervisor, main).items():
            dot, label = self.lights[name]; dot.config(fg=COLORS[color]); label.config(text=str(value))
        for label, item in zip(self.bits, decode_status(supervisor.get('word'))):
            label.config(text=f"● {item['name']}", fg=COLORS[item['color']])
        self.process_table.item('parent', values=('父程序', supervisor.get('pid') or '—', display_time(supervisor.get('start_time')),
            duration(supervisor.get('start_time')), '运行' if supervisor.get('pid') else '无数据', '—'))
        self.process_table.item('child', values=('主程序', supervisor.get('child_pid') or '—', display_time(supervisor.get('child_start_time')),
            duration(supervisor.get('child_start_time')) if supervisor.get('child_pid') else '—', supervisor.get('child_state') or '无数据',
            supervisor.get('child_last_exit_reason') or '无'))
        picking = main.get('picking') if isinstance(main.get('picking'), dict) else {}
        view = self.val(picking, 'view_index', 'view')
        if isinstance(view, int) and 'view_index' in picking: view += 1
        pick_fields = {'盘号': self.val(picking, 'tray_id', 'tray'), '视野': f"{view}/{self.val(picking, 'total_views', 'total')}",
                       '粗拍次数': self.val(picking, 'coarse_shots'), '可去数量': self.val(picking, 'removable_count', 'n0'),
                       '红框数量': self.val(picking, 'red_count'), '已下发': self.val(picking, 'sent_count', 'attempts')}
        for name, value in pick_fields.items(): self.pick_values[name].config(text=str(value))
        rows = self.tail.poll()
        self.update_reply(main, rows)
        self.update_cam(camcal_progress(main))
        self.update_preview(main)
        self.update_tip(tooltip_progress(main))
        key = (date.today(), json.dumps(rows, ensure_ascii=False, default=str))
        if key != self.cache.get('events'):
            self.events = list(rows); self.table.delete(*self.table.get_children())
            for i, row in enumerate(rows):
                values = [display_time(row.get('time'))] + [row.get(k) for k in ('category', 'task_id', 'func', 'phenomenon', 'location', 'suggestion')]
                self.table.insert('', 'end', iid=str(i), values=values, tags=(f"{row['color']}_{'odd' if i % 2 else 'even'}",))
            self.cache['events'] = key
        for picture in self.pictures: picture.poll()
        self.root.after(self.interval, self.refresh)

    def update_reply(self, main, rows):
        code = main.get('last_reply_func')
        if code is None:
            self.reply_info.set({}); return
        task = main.get('last_reply_task_id')
        matching = next((r for r in rows if r.get('task_id') == task and r.get('func') == code), None)
        reason = matching['phenomenon'] if matching else ('正常应答' if code not in (999, 131, 82) else '原因无数据')
        self.reply_info.set({'时间': display_time(main.get('last_reply_time')), 'Task_ID': task, '功能码': f"{main.get('last_reply_request_func', '—')} → {code}", '原因': reason})

    def update_cam(self, data):
        points = data.get('plan') or []; count = data.get('sequence') or 0
        total = len(points) + 1 if points else 0
        self.progress.config(maximum=max(1, total), value=count)
        self.progress_text.config(text=f'第 {count}/{total or "—"} 张')
        detail = data.get('detail') if isinstance(data.get('detail'), dict) else {}
        fraction = detail.get('saturated_fraction')
        saturation = f'{fraction:.2%}' if isinstance(fraction, (float, int)) else '—'
        bbox = detail.get('bbox')
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            glare = f'({x1},{y1})–({x2},{y2})'
            if detail.get('area_px') is not None:
                glare += f"，面积 {detail['area_px']} px"
        else:
            glare = '无'
        kind = data.get('kind') or '—'
        step = {'move': '移动拍照', 'done': '完成', 'fail': '失败'}.get(kind, kind)
        code = data.get('code')
        self.cam_step.set({'步骤': f'{step}  {code}' if code else step, '检出角点': f"{detail.get('detected_corners', '—')}/{detail.get('predicted_corners', '—')}", '饱和比例': saturation, '亮斑': glare})

        failed = data.get('event') in ('camcal_failed', 'camcal_aborted') or data.get('kind') == 'fail'
        done = data.get('event') == 'camcal_done' or data.get('kind') == 'done'
        if failed:
            reason = data.get('reason') or '标定失败'
            hint = data.get('hint') or ('请' + reason.rsplit('请', 1)[1] if '请' in reason else '检查标定板、光照和相机连接。')
            text = f"{data.get('code') or 'CAMCAL-FAILED'}\n{reason}\n建议：{hint}"
            self.cam_result.config(text=text, bg='#b42318', fg=WHITE)
        elif done:
            report = data.get('report') if isinstance(data.get('report'), dict) else detail
            text = (f"标定成功  RMS：{report.get('rms_px', '—')} px\n"
                    f"主点尺度：{report.get('um_per_px_at_principal', '—')} µm/px  "
                    f"轴比例：{report.get('axis_scale', '—')}\n倾角：{report.get('camera_tilt_deg', '—')}°\n"
                    f"文件：{data.get('calibration_path') or '—'}  备份：{data.get('backup_path') or '—'}")
            self.cam_result.config(text=text, bg='#ecfdf3', fg=COLORS['green'])
        else:
            self.cam_result.config(text='标定进行中' if data else '无数据', bg='#ecfdf3', fg=COLORS['green'])
        start = data.get('start_xyz')
        origin_z = start[2] if isinstance(start, (list, tuple)) and len(start) == 3 else None
        self.draw_plan(points, count, origin_z)

    def toggle_preview(self):
        path = ui_command_file(self.directory)
        try:
            old = read_json(path)
            seq = max(time.time_ns(), int(old.get('seq', 0)) + 1, int(self.preview_data.get('seq', 0)) + 1)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix('.tmp')
            tmp.write_text(json.dumps({'preview': not self.preview_data.get('on', False), 'seq': seq,
                                       'time': datetime.now().astimezone().isoformat()}), encoding='utf-8')
            os.replace(tmp, path)
        except (OSError, ValueError, TypeError):
            logger.exception('写预览命令失败')

    def update_preview(self, main):
        try:
            age = max(0, time.time() - main_status_file(self.directory).stat().st_mtime)
        except OSError:
            age = None
        controls = preview_controls(main, age)
        self.preview_data = main.get('preview') if isinstance(main.get('preview'), dict) else {}
        self.preview_button.config(text='关闭相机预览' if controls['on'] else '打开相机预览',
                                   state='normal' if controls['enabled'] else 'disabled')
        text = ('不可用：' + controls['reason'] if controls['reason'] else
                f"预览中 {self.preview_data.get('fps', 0)} fps" if controls['on'] else
                '不可用：' + str(self.preview_data['error']) if self.preview_data.get('error') else '已关闭')
        self.preview_status.config(text=text)
        path = live_image_file(self.directory, 'preview_latest' if controls['on'] else 'camcal_latest')
        if getattr(self.cam_picture, 'path', None) != path:
            self.cam_picture.path = path
            self.cam_picture.mtime = None
            self.cam_picture.image = self.cam_picture.photo = None
            self.cam_picture.config(image='', text='暂无图片')
        p = self.preview_data
        self.preview_values.set({'模组 XYZ': p.get('actual_xyz'), '清晰度': p.get('sharpness'),
                           '检出角点': p.get('corners'), '板中心偏移 mm': p.get('board_offset_mm'),
                           '提示': '偏移接近 0 且清晰度最大处即为推荐起始位'})

    def update_tip(self, data):
        key = json.dumps(data, ensure_ascii=False, default=str)
        if key == self.cache.get('tip'): return
        self.cache['tip'] = key
        self.tiptable.delete(*self.tiptable.get_children())
        current = None
        for row in data.get('history', []):
            if row.get('event') == 'tooltip_detect':
                focus = row.get('focus') or {}; delta = row.get('delta') or [None, None, None]
                delta = list(delta)[:3] + [None] * max(0, 3-len(delta))
                def view(name):
                    item = focus.get(name) or {}
                    return f"{item.get('edge_width_px', '—')} px / {'清晰' if item.get('focused') else '不清晰'}"
                current = self.tiptable.insert('', 'end', values=(len(self.tiptable.get_children()) + 1,
                    *[str(v) if v is not None else '—' for v in delta], view('front'), view('side'), '—'))
            elif current and row.get('event') == 'tooltip_reliability':
                values = list(self.tiptable.item(current, 'values'))
                values[-1] = ('可靠' if row.get('reliable') else '不可靠') + ' ' + '；'.join(row.get('reasons') or [])
                self.tiptable.item(current, values=values)
        states = {'tooltip_aligned': '对刀完成', 'tooltip_failed': '对刀失败', 'tooltip_detect': '检测中', 'tooltip_offset': '调整中', 'tooltip_reliability': '可靠性检查'}
        state = states.get(data.get('event'), '进行中' if data else '无数据')
        self.tip_result.set({'工具号': data.get('tool'), '迭代次数': data.get('iteration', data.get('iterations')), '状态': state, '原因': data.get('reason') or '—', '补偿': data.get('new_compensation', data.get('new'))})

    def draw_plan(self, points, count, origin_z=None):
        key = (str(points), count, origin_z, self.plan.winfo_width(), self.plan.winfo_height())
        if self.cache.get('plan') == key: return
        self.cache['plan'] = key
        self.plan.delete('all')
        if not points:
            self.plan.create_text(12, 12, text='暂无计划点', fill=MUTED, anchor='nw', font=self.theme.body)
            return
        try:
            layers = {}
            for index, point in enumerate(points):
                z = round(float(point[2]), 6)
                layers.setdefault(z, []).append((index, float(point[0]), float(point[1])))
            levels = sorted(layers)
            base = float(origin_z) if origin_z is not None else levels[len(levels) // 2]
            center = min(levels, key=lambda value: abs(value - base))
            outer = [z for z in levels if z != center]
            width = max(120, self.plan.winfo_width()); height = max(100, self.plan.winfo_height())
            top_height = height * .48
            slots = [(center, 0, 0, width, top_height)]
            slot_width = width / 4
            for slot, z in enumerate(outer[:4]):
                slots.append((z, slot * slot_width, top_height, (slot + 1) * slot_width, height))
            for z, left, top, right, bottom in slots:
                group = layers[z]
                done = sum(index < count - 1 for index, _, _ in group)
                offset = z - base
                ztext = '0' if abs(offset) < 1e-6 else f'{offset:+g}'
                self.plan.create_rectangle(left + 2, top + 2, right - 2, bottom - 2, outline=BORDER)
                self.plan.create_text(left + 8, top + 5, text=f'Z{ztext} · {done}/{len(group)}',
                                      fill=INK, anchor='nw', font=self.theme.table)
                xs = [p[1] for p in group]; ys = [p[2] for p in group]
                # 点中心始终留出边距，外圈也不会被小图裁剪。
                x0, x1 = left + 12, max(left + 12, right - 12)
                y0, y1 = top + 29, max(top + 29, bottom - 12)
                seen = {}
                for index, x, y in group:
                    px = (x0 + x1) / 2 if max(xs) == min(xs) else x0 + (x1-x0) * (x-min(xs))/(max(xs)-min(xs))
                    py = (y0 + y1) / 2 if max(ys) == min(ys) else y0 + (y1-y0) * (y-min(ys))/(max(ys)-min(ys))
                    occurrence = seen.get((x, y), 0); seen[(x, y)] = occurrence + 1
                    px = max(x0, min(x1, px + (occurrence % 3 - 1) * 4 if occurrence else px))
                    py = max(y0, min(y1, py + (occurrence // 3) * 4))
                    current = index == count - 1
                    color = COLORS['green' if index < count - 1 else 'blue' if current else 'gray']
                    self.plan.create_oval(px-4, py-4, px+4, py+4,
                                          fill=color if index < count - 1 or current else WHITE,
                                          outline=color, width=2)
                    if current:
                        self.plan.create_oval(px-7, py-7, px+7, py+7, outline=color, width=1)
        except (TypeError, ValueError, IndexError, OverflowError):
            logger.warning('相机标定计划点格式无效', exc_info=True)

    def run(self):
        self.root.mainloop()
