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
from tkinter import font as tkfont, ttk, messagebox

from PIL import Image, ImageGrab, ImageTk

from nest_clear.paths import live_image_file, main_status_file, supervisor_state_file, ui_command_file, tooltip_preview_file
from .model import (EventTail, camcal_progress, decode_status, display_time, duration,
                    read_json, status_indicators, parent_process_row, child_process_row, tooltip_progress, preview_controls, edge_corner_label,
                    pickacc_tray_geometry, pickacc_type, pickacc_stage, pickacc_note, pickacc_infield,
                    tooltip_vector, tooltip_deviation, tooltip_pixels, tooltip_quality, tooltip_time, page_for_mode,
                    baseline_preview_display, extract_tooltip_preview)

COLORS = {'green': '#168448', 'yellow': '#916300', 'red': '#b42318',
          'gray': '#85909f', 'blue': '#2563eb'}
BG = '#F3F4F6'
BORDER = '#D0D5DD'
INK = '#1d2939'
MUTED = '#667085'
FONT_WARNING = '界面字体系统不支持抗锯齿，请用系统 Python 运行'
logger = logging.getLogger(__name__)
WHITE = '#ffffff'


def modal_window(top, parent):
    """Keep a dialog above its parent, including when the parent is fullscreen."""
    top.transient(parent)
    top.grab_set()
    top.lift()
    top.focus_force()


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
        self.title_label = tk.Label(self, text=title, font=theme.title, bg=WHITE, fg=INK, anchor='w')
        self.title_label.grid(row=0, column=0, sticky='ew', padx=8, pady=(5, 0))
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
        self._wrap_width = None
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
        if width == getattr(self, '_wrap_width', None):
            return
        self._wrap_width = width
        for value in self.values.values():
            value.configure(wraplength=width)

    def set(self, values):
        for key, widget in self.values.items():
            value = values.get(key)
            widget.configure(text=str(value) if value is not None else '—')


class ScrollGeometry:
    """Coalesce geometry events and update the canvas only when values change."""
    def __init__(self, canvas, window):
        self.canvas, self.window = canvas, window
        self.width = self.applied_width = self.region = None
        self.pending = False

    def viewport_changed(self, event):
        if event.width > 1:
            self.width = event.width
            self.schedule()

    def content_changed(self, _event):
        if self.width is not None:
            self.schedule()

    def schedule(self):
        if not self.pending:
            self.pending = True
            self.canvas.after_idle(self.apply)

    def apply(self):
        self.pending = False
        if self.width is None:
            return
        if self.width != self.applied_width:
            self.applied_width = self.width
            self.canvas.itemconfigure(self.window, width=self.width)
        bounds = self.canvas.bbox('all')
        region = (0, 0, self.width, max(0, bounds[3]) if bounds else 0)
        if region != self.region:
            self.region = region
            self.canvas.configure(scrollregion=region)


class Picture(tk.Label):
    def __init__(self, parent, path, theme, *, min_mtime_ns=0):
        super().__init__(parent, text='暂无图像', anchor='center', bg='#eef1f5', fg=MUTED, font=theme.body)
        self.path = path
        self.min_mtime_ns = min_mtime_ns
        self.mtime = None
        self.image = None
        self.photo = None
        self.results = queue.Queue()
        self.busy = False
        self.size = None
        self.aspect_changed = None
        self.keep_on_missing = False
        self.bind('<Button-1>', self.enlarge)

    def poll(self):
        try:
            stamp = self.path.stat().st_mtime_ns
        except OSError:
            stamp = None
        if stamp is not None and self.min_mtime_ns and stamp < self.min_mtime_ns:
            stamp = None
        size = (max(1, self.winfo_width()), max(1, self.winfo_height()))
        if stamp is None:
            if not getattr(self, 'keep_on_missing', False) and (self.image is not None or self.photo is not None or self.mtime is not None):
                self.image = self.photo = self.mtime = self.size = None
                self.configure(image='', text='暂无图像')
        elif (stamp != self.mtime or size != self.size) and not self.busy and min(size) > 1:
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
        if original is not None and path == self.path and (not self.min_mtime_ns or (stamp is not None and stamp >= self.min_mtime_ns)):
            old_size = self.image.size if self.image is not None else None
            self.image, self.mtime, self.size = original, stamp, size
            self.photo = ImageTk.PhotoImage(preview)
            self.configure(image=self.photo, text='')
            if self.aspect_changed and (old_size is None or old_size[0] * original.height != original.width * old_size[1]):
                self.aspect_changed()

    def enlarge(self, _event=None):
        if self.image is None:
            return
        parent = self.winfo_toplevel()
        top = tk.Toplevel(parent)
        modal_window(top, parent)
        top.title(str(self.path))
        top.rowconfigure(0, weight=1); top.columnconfigure(0, weight=1)
        canvas = tk.Canvas(top, width=1000, height=700, scrollregion=(0, 0, *self.image.size))
        canvas.grid(row=0, column=0, sticky='nsew')
        photo = ImageTk.PhotoImage(self.image)
        canvas.create_image(0, 0, image=photo, anchor='nw'); canvas.image = photo
        canvas.bind('<ButtonPress-1>', lambda event: canvas.scan_mark(event.x, event.y))
        canvas.bind('<B1-Motion>', lambda event: canvas.scan_dragto(event.x, event.y, gain=1))


def resize_home_picture(container, body, picture):
    height = body.winfo_height()
    if height <= 1:
        return
    width, original_height = picture.image.size if picture.image is not None else (4024, 3036)
    target = round(height * width / original_height) + 24
    if abs(container.winfo_width() - target) >= 2:
        container.configure(width=target)


class App:
    PAGE_NAMES = ('主页', '相机标定', '工具偏移标定', '挑取精度标定', '对刀基准建立', '换针对刀')
    PAGE_ALIASES = {'home': '主页', 'main': '主页', 'camera': '相机标定', 'camcal': '相机标定',
                    'offset': '工具偏移标定', 'tool-offset': '工具偏移标定',
                    'pickacc': '挑取精度标定', 'pick-accuracy': '挑取精度标定',
                    'baseline': '对刀基准建立', 'toolcal': '换针对刀', 'tooltip': '换针对刀'}

    def __init__(self, root, refresh_ms=300, fullscreen=False, page=None, screenshot=None, screenshot_after_ms=None, ignore_before_ns=None):
        self.directory = Path(root); self.interval = refresh_ms
        self.started_at_ns = time.time_ns() if ignore_before_ns is None else ignore_before_ns
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
        self.film_banner = tk.Label(self.root, text='主流程测试模式（菲林片）已开启：识别菲林片目标，非燕窝挑毛',
                                    bg='#b00020', fg='white', font=self.theme.status, anchor='center', padx=8, pady=5)
        self.tail = EventTail(root); self.pictures = []; self.cache = {}; self.events = []; self.files = {}
        self.last_mode = read_json(main_status_file(self.directory)).get('mode') if page is not None else None
        self.build_status()
        self.tabs = ttk.Notebook(self.root)
        self.tabs.grid(row=2, column=0, sticky='nsew', padx=6, pady=(0, 6))
        self.home = self.new_page('主页'); self.build_home()
        self.cam = self.new_page('相机标定'); self.build_cam()
        self.offset = self.new_page('工具偏移标定'); self.build_offset()
        self.pickacc = self.new_page('挑取精度标定'); self.build_pickacc()
        self.baseline = self.new_page('对刀基准建立'); self.build_baseline()
        self.tip = self.new_page('换针对刀'); self.build_tip()
        self.select_page(page if page is not None else '主页')
        self.root.after(50, self.refresh)
        self.root.after(50, self.poll_pictures)
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
        widget.min_mtime_ns = self.started_at_ns
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
        self.home.columnconfigure(0, weight=1)
        self.home.rowconfigure(1, weight=1)
        left = tk.Frame(self.home, bg=BG)
        left.grid(row=0, column=0, sticky='nsew')
        left.columnconfigure(0, weight=3, uniform='homeleft')
        left.columnconfigure(1, weight=2, uniform='homeleft')
        device = self.card(left, '设备状态 40901', 0, 0)
        device_content = tk.Frame(device, bg=WHITE)
        device_content.grid(row=0, column=0, sticky='new')
        for i in range(4): device_content.rowconfigure(i, minsize=30)
        for i in range(3): device_content.columnconfigure(i, weight=1, uniform='bits')
        self.bits = []
        for i in range(12):
            label = self.label(device_content)
            label.grid(row=i // 3, column=i % 3, sticky='w')
            self.bits.append(label)
        process = self.card(left, '进程', 1, 0, columnspan=2)
        cols = ('程序', 'PID', '启动时间', '运行时长', '状态', '最近退出原因')
        self.process_table = self.table_widget(process, cols, (55, 55, 75, 75, 60, 100), stretch=('最近退出原因',), height=2)
        self.process_table.insert('', 'end', iid='parent'); self.process_table.insert('', 'end', iid='child')
        pick = self.card(left, '挑毛进度', 0, 1)
        pick_content = tk.Frame(pick, bg=WHITE)
        pick_content.grid(row=0, column=0, sticky='new')
        for i in range(4): pick_content.rowconfigure(i, minsize=30)
        self.pick_values = {}
        for i, title in enumerate(('盘号', '视野', '粗拍次数', '可去数量', '红框数量', '已下发', '不可去', '失败')):
            r, c = divmod(i, 2)
            pick_content.columnconfigure(c * 2 + 1, weight=1)
            label = self.label(pick_content, title + '  ', fg=MUTED)
            label.grid(row=r, column=c * 2, sticky='w')
            value = self.label(pick_content, '—', font=self.theme.mono)
            value.grid(row=r, column=c * 2 + 1, sticky='w'); self.pick_values[title] = value
        pick.grid_configure(columnspan=2)
        self.film_button = tk.Button(pick.master, text='主流程测试模式（菲林片）', command=self.toggle_film_mode)
        self.film_button.grid(row=0, column=1, sticky='e', padx=(0, 8))
        self.film_reason = self.label(pick.master, '', fg=MUTED)
        self.film_reason.grid(row=1, column=1, sticky='e', padx=(0, 8))
        reply = self.card(left, '最近应答', 2, 0, columnspan=2)
        self.reply_table = self.table_widget(reply, ('时间', 'Task_ID', '功能码', '原因'),
                                             (150, 65, 85, 120), stretch=('原因',), height=5)
        self.reply_table.bind('<Double-1>', self.reply_detail)
        self.reply_table.bind('<Motion>', self.reply_hover)
        self.reply_table.bind('<Leave>', self.reply_hover_close)
        image_container = tk.Frame(self.home, bg=BG, width=400)
        image_container.grid(row=0, column=1, sticky='nsew')
        image_container.grid_propagate(False)
        image_container.rowconfigure(0, weight=1); image_container.columnconfigure(0, weight=1)
        image = self.card(image_container, '最新标注图 · 点击放大', 0, 0)
        picture = self.picture(image, 'latest')
        def resize_image(_event=None):
            resize_home_picture(image_container, image, picture)
        image.bind('<Configure>', resize_image)
        picture.aspect_changed = resize_image
        logs = self.card(self.home, '异常日志 · 双击查看详情', 1, 0, columnspan=2)
        cols = ('时间', '类别', 'Task_ID', '功能码', '现象', '位置', '排查建议')
        self.table = self.table_widget(logs, cols, (90, 70, 70, 70, 250, 200, 250), stretch=('现象', '排查建议'), height=4)
        layout_state = {}
        def resize_home(_event):
            if self.home.winfo_height() <= 1 or logs.master.winfo_reqheight() <= 1:
                return
            small = self.root.winfo_height() <= 800
            if layout_state.get('small') != small:
                for body in (device, pick, process, reply, logs):
                    body.master.grid_configure(pady=1 if small else 3)
                    body.master.title_label.grid_configure(pady=(0, 0) if small else (5, 0))
                    body.grid_configure(pady=(0, 0) if small else (2, 6))
                layout_state['small'] = small
            log_height = logs.master.winfo_reqheight() + 6
            current_row = layout_state.get('row_height', 30)
            natural_row = max(widget.winfo_reqheight() for widget in self.bits)
            full_height = left.winfo_reqheight() + 4 * (
                max(30, natural_row) - max(current_row, natural_row))
            shortage = max(0, full_height + log_height - self.home.winfo_height())
            row_height = max(22, 30 - (shortage + 3) // 4)
            if current_row != row_height:
                for index in range(4): device_content.rowconfigure(index, minsize=row_height)
                for index in range(4): pick_content.rowconfigure(index, minsize=row_height)
                layout_state['row_height'] = row_height
            compact = self.home.winfo_height() < left.winfo_reqheight() + log_height
            if layout_state.get('log_height') != log_height:
                self.home.rowconfigure(1, minsize=log_height)
                layout_state['log_height'] = log_height
            if layout_state.get('compact') != compact:
                self.home.rowconfigure(0, weight=int(compact))
                left.rowconfigure(0, weight=int(compact))
                layout_state['compact'] = compact
        self.home.bind('<Configure>', resize_home)
        left.bind('<Configure>', resize_home)
        logs.master.bind('<Configure>', resize_home)
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
        self.cam_picture.keep_on_missing = True
        container = tk.Frame(self.cam, bg=BG); container.grid(row=0, column=1, sticky='nsew')
        container.rowconfigure(0, weight=1); container.columnconfigure(0, weight=1)
        viewport = self.cam_viewport = tk.Canvas(container, bg=BG, highlightthickness=0)
        viewport.grid(row=0, column=0, sticky='nsew')
        scroll = ttk.Scrollbar(container, orient='vertical', command=viewport.yview)
        scroll.grid(row=0, column=1, sticky='ns'); viewport.configure(yscrollcommand=scroll.set)
        right = self.cam_right = tk.Frame(viewport, bg=BG)
        window = viewport.create_window((0, 0), window=right, anchor='nw', width=400)
        geometry = self.cam_scroll_geometry = ScrollGeometry(viewport, window)
        right.bind('<Configure>', geometry.content_changed)
        viewport.bind('<Configure>', geometry.viewport_changed)
        viewport.bind('<MouseWheel>', lambda event: viewport.yview_scroll(-int(event.delta / 120), 'units'))
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
        self.focus_button = ttk.Button(progress, text='对焦/光圈辅助', command=lambda: self.toggle_preview('focus'))
        self.focus_button.grid(row=2, column=0, sticky='e')
        self.focus_panel = tk.Frame(right, bg=WHITE)
        self.focus_panel.grid(row=1, column=0, sticky='nsew', padx=3)
        self.focus_panel.columnconfigure(0, weight=1)
        self.focus_panel.rowconfigure(7, weight=1)
        headline = tk.Frame(self.focus_panel, bg=WHITE); headline.grid(row=0, column=0, sticky='ew')
        headline.columnconfigure(1, weight=1)
        self.focus_percent = self.label(headline, '—%', font=(self.theme.mono[0], -32))
        self.focus_percent.grid(row=0, column=0, sticky='w', padx=(3, 12))
        self.focus_summary = self.label(headline, '等待对焦数据', font=self.theme.table)
        self.focus_summary.config(justify='left')
        self.focus_summary.grid(row=0, column=1, sticky='ew')
        self.focus_bar = tk.Canvas(self.focus_panel, height=16, bg='#e4e7ec', highlightthickness=0)
        self.focus_bar.grid(row=1, column=0, sticky='ew')
        details_frame = tk.Frame(self.focus_panel, bg=WHITE); details_frame.grid(row=2, column=0, sticky='ew')
        self.focus_details = self.label(details_frame, '', font=self.theme.table)
        self.focus_details.config(justify='left', anchor='nw')
        self.focus_details.grid(row=0, column=0, sticky='ew')
        manual = tk.Frame(details_frame, bg=WHITE); manual.grid(row=1, column=0, sticky='ew')
        self.label(manual, '曝光 us').grid(row=0, column=0)
        self.temporary_entry = ttk.Entry(manual, width=8); self.temporary_entry.grid(row=0, column=1)
        self.temporary_set = ttk.Button(manual, text='设置曝光', command=lambda: self.change_temporary_exposure(False)); self.temporary_set.grid(row=0, column=2)
        self.temporary_restore = ttk.Button(manual, text='恢复配置曝光', command=lambda: self.change_temporary_exposure(True)); self.temporary_restore.grid(row=0, column=3)
        self.temporary_message = self.label(manual, '', font=self.theme.table); self.temporary_message.grid(row=1, column=0, columnspan=4, sticky='w')
        self.temporary_pending = False; self.sweep_pending = False
        self.focus_alert = self.label(self.focus_panel, '', font=self.theme.table)
        self.focus_alert.config(justify='left', anchor='nw', fg='#b54708', wraplength=430)
        self.focus_alert.grid(row=3, column=0, sticky='ew')
        tables = tk.Frame(self.focus_panel, bg=WHITE); tables.grid(row=4, column=0, sticky='ew')
        tables.columnconfigure(0, weight=3, uniform='focus_tables'); tables.columnconfigure(1, weight=1, uniform='focus_tables')
        score_frame = tk.Frame(tables, bg=WHITE); score_frame.grid(row=0, column=0, sticky='ew')
        mean_frame = tk.Frame(tables, bg=WHITE); mean_frame.grid(row=0, column=1, sticky='ew')
        self.label(score_frame, '各格 当前/峰值  %', font=self.theme.table).grid(row=0, column=0, columnspan=3, sticky='w')
        self.label(mean_frame, '灰度均值', font=self.theme.table).grid(row=0, column=0, columnspan=3, sticky='w')
        self.focus_score_cells, self.focus_mean_cells = [], []
        for col in range(3):
            score_frame.columnconfigure(col, weight=1, uniform='scores')
            mean_frame.columnconfigure(col, weight=1, uniform='means')
        for i in range(9):
            score = self.label(score_frame, '—', font=(self.theme.mono[0], -12), anchor='center', bg='#f3f5f8')
            mean = self.label(mean_frame, '—', font=(self.theme.mono[0], -12), anchor='center', bg='#f3f5f8')
            score.grid(row=1+i//3, column=i%3, sticky='ew', padx=1, pady=1)
            mean.grid(row=1+i//3, column=i%3, sticky='ew', padx=1, pady=1)
            self.focus_score_cells.append(score); self.focus_mean_cells.append(mean)
        self.focus_curve = tk.Canvas(self.focus_panel, height=120, bg='#f1f5f9', highlightthickness=0)
        self.focus_curve.grid(row=5, column=0, sticky='ew', pady=(3, 0))
        self.focus_record_title = self.label(self.focus_panel, '本次记录', font=self.theme.table)
        self.focus_record_title.grid(row=6, column=0, sticky='w')
        self.focus_table = ttk.Treeview(self.focus_panel, columns=('time','note','center','median','mean','sat'), show='headings', height=1)
        for col, title, width in [('time','时间',65),('note','备注',90),('center','中心',65),('median','中位数',65),('mean','均值',50),('sat','饱和%',55)]:
            self.focus_table.heading(col, text=title); self.focus_table.column(col, width=width, minwidth=30, stretch=True)
        self.focus_table.grid(row=7, column=0, sticky='nsew')
        self.exposure_panel = tk.Frame(self.focus_panel, bg=WHITE)
        self.exposure_panel.columnconfigure(0, weight=1); self.exposure_panel.rowconfigure(1, weight=1)
        self.exposure_message = self.label(self.exposure_panel, '曝光自测', font=self.theme.table)
        self.exposure_message.grid(row=0, column=0, sticky='ew')
        self.exposure_table = ttk.Treeview(self.exposure_panel, columns=('us','sat','p99','corners','sharp'), show='headings', height=4)
        for col, title, width in [('us','曝光 us',80),('sat','饱和%',65),('p99','P99',50),('corners','角点',50),('sharp','清晰度',65)]:
            self.exposure_table.heading(col, text=title); self.exposure_table.column(col, width=width, minwidth=30, stretch=True)
        self.exposure_table.tag_configure('recommended', background='#d1fadf', foreground='#067647')
        self.exposure_table.grid(row=1, column=0, sticky='nsew')
        exposure_actions = tk.Frame(self.exposure_panel, bg=WHITE); exposure_actions.grid(row=2, column=0, sticky='ew')
        self.exposure_apply = ttk.Button(exposure_actions, text='应用推荐曝光', command=self.apply_exposure)
        self.exposure_apply.grid(row=0, column=0, sticky='w')
        ttk.Button(exposure_actions, text='返回曲线', command=self.close_exposure_results).grid(row=0, column=1, sticky='e')
        exposure_actions.columnconfigure(1, weight=1)
        self.exposure_panel.grid(row=5, column=0, rowspan=3, sticky='nsew'); self.exposure_panel.grid_remove()
        self.exposure_visible = False; self.exposure_cache = None
        exposure_controls = tk.Frame(self.focus_panel, bg=WHITE); exposure_controls.grid(row=8, column=0, sticky='ew')
        self.exposure_mode = ttk.Combobox(exposure_controls, values=('标定用（角点最多）', '亮度目标'), state='readonly', width=18)
        self.exposure_mode.set('标定用（角点最多）'); self.exposure_mode.grid(row=0, column=0, sticky='ew')
        exposure_controls.columnconfigure(0, weight=1)
        self.exposure_button = ttk.Button(exposure_controls, text='曝光自测', command=self.toggle_exposure_sweep)
        self.exposure_button.grid(row=0, column=1, sticky='e')
        self.wb_button = ttk.Button(exposure_controls, text='白平衡', command=self.start_white_balance)
        self.wb_button.grid(row=0, column=2, sticky='e')
        self.wb_message = self.label(headline, '', font=self.theme.table)
        self.wb_message.config(justify='left', anchor='w', wraplength=350)
        self.wb_message.grid(row=1, column=0, columnspan=2, sticky='ew')
        self.wb_pending = False
        actions = tk.Frame(self.focus_panel, bg=WHITE); actions.grid(row=9, column=0, sticky='ew')
        ttk.Button(actions, text='重置峰值', command=self.reset_focus).grid(row=0, column=0)
        self.label(actions, '备注').grid(row=0, column=1, padx=(5, 2))
        self.focus_note = ttk.Entry(actions, width=8); self.focus_note.grid(row=0, column=2, sticky='ew'); actions.columnconfigure(2, weight=1)
        ttk.Button(actions, text='记录当前', command=self.record_focus).grid(row=0, column=3)
        self.focus_panel.grid_remove()
        self.focus_history = []
        self.focus_last_sample = None
        self.focus_was_on = False
        self.preview_status = self.label(progress, '已关闭')
        self.preview_status.grid(row=3, column=0, sticky='w')
        plan = self.card(right, '计划点 · XY 平面', 1, 0)
        self.plan = tk.Canvas(plan, bg=WHITE, highlightthickness=0, height=150)
        self.plan.grid(row=0, column=0, sticky='nsew')
        step = self.card(right, '当前步', 2, 0)
        self.cam_step = KeyValueGrid(step, ('步骤', '检出角点', '饱和比例', '亮斑'), self.theme)
        self.cam_step.grid(row=0, column=0, sticky='new')
        preview = self.card(right, '预览读数', 3, 0)
        self.preview_values = KeyValueGrid(preview, ('模组 XYZ', '清晰度', '检出角点', '板中心偏移 mm', '提示'), self.theme)
        self.preview_values.grid(row=0, column=0, sticky='new')
        result = self.card(right, '结果', 4, 0)
        self.cam_plan_card, self.cam_result_card = plan.master, result.master
        self.cam_other_cards = [plan.master, step.master, preview.master, result.master]
        self.cam_result = tk.Text(result, height=5, width=20, wrap='word', state='disabled', bg='#ecfdf3', fg=COLORS['green'],
                                  font=self.theme.body, padx=6, pady=4)
        self.cam_result.grid(row=0, column=0, sticky='nsew')
        result_scroll = ttk.Scrollbar(result, orient='vertical', command=self.cam_result.yview)
        result_scroll.grid(row=0, column=1, sticky='ns'); self.cam_result.configure(yscrollcommand=result_scroll.set)

    def build_offset(self):
        self.offset_tool = None; self.offset_mode = 'center'; self.offset_pending = 0; self.offset_dialog = None; self.offset_last_phase = None
        self.offset.columnconfigure(0, weight=60); self.offset.columnconfigure(1, weight=40)
        self.offset.rowconfigure(1, weight=1)
        header = self.card(self.offset, '工具偏移标定', 0, 0, columnspan=2)
        header.columnconfigure(0, weight=1)
        heading = tk.Frame(header, bg=WHITE); heading.grid(row=0, column=0, sticky='ew')
        self.offset_heading = tk.Label(heading, text='当前：未选择', bg='#667085', fg=WHITE, font=(self.theme.title[0], 28, 'bold'), anchor='center', padx=18)
        self.offset_heading.grid(row=0, column=0, sticky='w', padx=(0,15))
        self.offset_mode_text = tk.Label(heading, text='模式：仅中心', bg=WHITE, fg=INK, font=self.theme.title)
        self.offset_mode_text.grid(row=0, column=1, sticky='w')
        self.offset_message = tk.Label(header, text='等待 PLC 请求', bg=WHITE, fg=INK, font=(self.theme.title[0], 20), anchor='w')
        self.offset_message.grid(row=1, column=0, sticky='ew')
        controls = tk.Frame(header, bg=WHITE); controls.grid(row=2, column=0, sticky='ew', pady=3)
        controls.columnconfigure(1, weight=1)
        left = tk.Frame(controls, bg=WHITE); left.grid(row=0, column=0, sticky='w')
        style = ttk.Style(self.root)
        style.configure('OffsetSelected.TButton', foreground='white', background='#2563eb')
        style.map('OffsetSelected.TButton', background=[('disabled', '#a9c9f8')], foreground=[('disabled', '#405575')])
        style.configure('OffsetPlace.TButton', foreground='white', background='#168448', font=(self.theme.title[0], 17, 'bold'))
        self.offset_tool_buttons = []
        for col, tool in enumerate((1,2)):
            button = ttk.Button(left, text=f'工具{tool}', command=lambda n=tool: self.choose_offset_tool(n))
            button.grid(row=0, column=col, padx=2); self.offset_tool_buttons.append(button)
        tk.Label(left, text='｜', bg=WHITE, fg=MUTED).grid(row=0, column=2)
        self.offset_mode_buttons = []
        for col, (label, mode) in enumerate((('仅中心','center'), ('中心+边缘验证','center_edge')), 3):
            button = ttk.Button(left, text=label, command=lambda m=mode: self.choose_offset_mode(m))
            button.grid(row=0, column=col, padx=2); self.offset_mode_buttons.append(button)
        tk.Label(left, text='｜', bg=WHITE, fg=MUTED).grid(row=0, column=5)
        self.offset_start = ttk.Button(left, text='开始', command=self.start_offset); self.offset_start.grid(row=0, column=6, padx=3)
        right_controls = tk.Frame(controls, bg=WHITE); right_controls.grid(row=0, column=2, sticky='e')
        self.offset_place = ttk.Button(right_controls, text='工具到位', style='OffsetPlace.TButton', command=lambda: self.offset_command('tool_in_place')); self.offset_place.grid(row=0, column=0, padx=5)
        self.offset_abort = ttk.Button(right_controls, text='放弃本次标定', command=lambda: self.offset_command('abort')); self.offset_abort.grid(row=0, column=1, padx=5)
        self.offset_warning = tk.Label(header, text='', bg=WHITE, fg=COLORS['red'], font=self.theme.body, anchor='w')
        self.offset_warning.grid(row=3, column=0, sticky='ew')
        self.offset_caution = tk.Label(header, text='', bg=WHITE, fg=COLORS['yellow'], font=self.theme.body, anchor='w')
        self.offset_caution.grid(row=4, column=0, sticky='ew')
        self.picture(self.card(self.offset, '标定图像 · 点击放大', 1, 0), 'tooloffset_latest')
        right = tk.Frame(self.offset, bg=BG); right.grid(row=1, column=1, sticky='nsew')
        right.columnconfigure(0, weight=1); right.rowconfigure(0, weight=4, minsize=135); right.rowconfigure(1, weight=0); right.rowconfigure(2, weight=2, minsize=145)
        self.offset.bind('<Configure>', lambda event: right.rowconfigure(2, minsize=175 if self.offset.winfo_height() > 850 else 145))
        self.picture(self.card(right, '当前目标放大图 · 绿：圆环中心/拟合线  紫：十字中心  蓝：主点', 0, 0), 'tooloffset_crop')
        existing = self.card(right, '已有结果与当前补偿', 1, 0)
        self.offset_existing = tk.Label(existing, text='无数据', bg=WHITE, fg=INK, font=self.theme.body, justify='left', anchor='w', wraplength=680)
        self.offset_existing.grid(row=0, column=0, sticky='ew')
        tables = tk.Frame(right, bg=BG); tables.grid(row=2, column=0, sticky='nsew')
        tables.columnconfigure(0, weight=1); tables.columnconfigure(1, weight=1); tables.rowconfigure(0, weight=1)
        samples = self.card(tables, '样本与统计', 0, 0)
        self.offset_samples = self.table_widget(samples, ('序号','目标','dx','dy','均值差','离群'), (38,43,55,55,55,42))
        scrollbar = ttk.Scrollbar(samples, orient='vertical', command=self.offset_samples.yview)
        scrollbar.grid(row=0, column=1, sticky='ns'); self.offset_samples.configure(yscrollcommand=scrollbar.set)
        style.configure('Offset.Treeview', rowheight=19, font=(self.theme.body[0], 9))
        self.offset_samples.configure(style='Offset.Treeview')
        self.offset_stats = tk.Label(samples, text='均值 / std / SE：—', bg=WHITE, fg=INK, font=(self.theme.body[0], 9), anchor='w')
        self.offset_stats.grid(row=1, column=0, sticky='ew')
        edge = self.card(tables, '边缘验证', 0, 1)
        self.offset_edges = self.table_widget(edge, ('目标','角','Δx','Δy','差值'), (48,35,58,58,60))
        self.offset_edges.configure(style='Offset.Treeview')
        self.offset_edges.tag_configure('warning', foreground=COLORS['yellow'])
        scrollbar = ttk.Scrollbar(edge, orient='vertical', command=self.offset_edges.yview)
        scrollbar.grid(row=0, column=1, sticky='ns'); self.offset_edges.configure(yscrollcommand=scrollbar.set)

    def build_baseline(self):
        self.baseline_pending = 0; self.baseline_dialog = None
        self.baseline.columnconfigure(0, weight=1, uniform='baseline'); self.baseline.columnconfigure(1, weight=1, uniform='baseline')
        self.baseline.rowconfigure(1, weight=1)
        head = self.card(self.baseline, '对刀基准建立', 0, 0, columnspan=2)
        bar = tk.Frame(head, bg=WHITE); bar.grid(row=0, column=0, sticky='ew'); bar.columnconfigure(1, weight=1)
        self.baseline_heading = tk.Label(bar, text='当前：未选择', bg='#667085', fg=WHITE, font=(self.theme.title[0], 19, 'bold'), padx=10)
        self.baseline_heading.grid(row=0, column=0, padx=3)
        self.baseline_message = tk.Label(bar, text='请开始对刀基准建立', bg=WHITE, fg=INK, font=(self.theme.title[0], 16), anchor='w', justify='left', wraplength=900)
        self.baseline_message.grid(row=0, column=1, sticky='ew', padx=8)
        bar.bind('<Configure>', lambda event: self.baseline_message.config(wraplength=max(240, event.width - self.baseline_heading.winfo_reqwidth() - 32)))
        controls = tk.Frame(head, bg=WHITE); controls.grid(row=1, column=0, sticky='w', pady=4)
        self.baseline_start = ttk.Button(controls, text='开始对刀基准建立', command=lambda:self.baseline_command('baseline_start'))
        self.baseline_start.grid(row=0, column=0, padx=2)
        self.baseline_tools = []
        for i in (1, 2):
            b = ttk.Button(controls, text=f'工具{i}', command=lambda i=i:self.baseline_command('baseline_select', {'tool':i}))
            b.grid(row=0, column=i, padx=2); self.baseline_tools.append(b)
        tk.Label(controls, text='｜', bg=WHITE, fg=MUTED).grid(row=0, column=3, padx=2)
        self.baseline_measure = ttk.Button(controls, text='对刀基准', style='OffsetPlace.TButton', command=lambda:self.baseline_command('baseline_measure'))
        self.baseline_measure.grid(row=0, column=4, padx=5)
        self.baseline_manual = ttk.Button(controls, text='识别不准，手动确认', command=lambda:self.baseline_command('baseline_manual')); self.baseline_manual.grid(row=0, column=5, padx=2)
        self.baseline_aux = ttk.Button(controls, text='辅助相机手动重设', command=lambda:self.baseline_command('baseline_aux')); self.baseline_aux.grid(row=0, column=6, padx=2)
        self.baseline_save = ttk.Button(controls, text='保存基准', command=lambda:self.baseline_command('baseline_save')); self.baseline_save.grid(row=0, column=7, padx=2)
        self.baseline_end = ttk.Button(controls, text='结束/放弃', command=lambda:self.baseline_command('baseline_end')); self.baseline_end.grid(row=0, column=8, padx=2)
        runout = tk.Frame(head, bg=WHITE, padx=6, pady=3)
        runout.grid(row=4, column=0, sticky='ew')
        tk.Label(runout, text='离心测试', bg=WHITE, fg=INK).grid(row=0, column=0, sticky='w', padx=4)
        runout_controls = tk.Frame(runout, bg=WHITE)
        runout_controls.grid(row=1, column=0, sticky='w')
        self.runout_start = ttk.Button(runout_controls, text='开始离心测试', command=lambda:self.baseline_command('tooltip_runout_start'))
        self.runout_start.grid(row=0, column=0, padx=3)
        self.runout_press = ttk.Button(runout_controls, text='记录并计算', command=lambda:self.baseline_command('tooltip_runout_press'))
        self.runout_press.grid(row=0, column=1, padx=3)
        self.runout_end = ttk.Button(runout_controls, text='结束', command=lambda:self.baseline_command('tooltip_runout_end'))
        self.runout_end.grid(row=0, column=2, padx=3)
        self.runout_max = tk.Label(runout_controls, text='最大离心量 — µm', bg=WHITE, fg=INK, font=(self.theme.title[0], 20, 'bold'))
        self.runout_max.grid(row=0, column=3, padx=12)
        self.runout_detail = tk.Label(runout, text='请打开两路预览并精定位', bg=WHITE, fg=MUTED, anchor='w', justify='left')
        self.runout_detail.grid(row=2, column=0, sticky='ew')
        self.baseline_live = tk.Label(head, text='实时辅助：等待针尖', bg=WHITE, fg=MUTED, anchor='w', font=self.theme.body)
        self.baseline_live.grid(row=2, column=0, sticky='ew')
        self.baseline_notice = tk.Label(head, text='', bg=WHITE, fg=COLORS['yellow'], anchor='w', font=self.theme.body)
        self.baseline_notice.grid(row=3, column=0, sticky='ew')
        self.baseline_views = {}
        self.baseline_exposure_pending = {}
        style = ttk.Style(self.root)
        style.map('TooltipAction.TButton', foreground=[('disabled', '#9ca3af'), ('!disabled', INK)],
                  background=[('disabled', '#e5e7eb')])
        for column, (name, title) in enumerate((('front','主对刀相机（正面）'), ('side','辅助对刀相机（侧面）'))):
            body = self.card(self.baseline, title + ' · 点击放大', 1, column)
            body.columnconfigure(0, weight=1); body.rowconfigure(0, weight=0); body.rowconfigure(1, weight=0); body.rowconfigure(2, weight=1)
            path = self.label(body, '未分配', fg=MUTED); path.grid(row=0, column=0, sticky='w')
            actions = tk.Frame(body, bg=WHITE); actions.grid(row=1, column=0, sticky='w')
            toggle = tk.Button(actions, text=title+'预览：关', command=lambda n=name:self.baseline_toggle(n))
            toggle.grid(row=0, column=0)
            exposure = ttk.Button(actions, style='TooltipAction.TButton', text='重新测曝光', command=lambda n=name:self.baseline_command('tooltip_exposure_retest', {'camera':n})); exposure.grid(row=0, column=2)
            reset = ttk.Button(actions, style='TooltipAction.TButton', text='清晰度峰值清零', command=lambda n=name:self.baseline_command('tooltip_peak_reset', {'camera':n})); reset.grid(row=0, column=3)
            reason = self.label(actions, '', fg=MUTED); reason.grid(row=1, column=0, columnspan=4, sticky='w')
            manual = tk.Frame(actions, bg=WHITE); manual.grid(row=2, column=0, columnspan=4, sticky='w')
            self.label(manual, '曝光 ms').grid(row=0, column=0)
            entry = ttk.Spinbox(manual, from_=0.1, to=500, increment=0.1, width=7); entry.grid(row=0, column=1)
            set_exposure = ttk.Button(manual, text='设置曝光', command=lambda n=name: self.baseline_exposure(n, False)); set_exposure.grid(row=0, column=2)
            restore = ttk.Button(manual, text='恢复曝光', command=lambda n=name: self.baseline_exposure(n, True)); restore.grid(row=0, column=3)
            save = ttk.Button(manual, text='设为实际曝光', command=lambda n=name: self.baseline_save_exposure(n)); save.grid(row=0, column=4)
            limits = self.label(manual, '范围 —'); limits.grid(row=0, column=5)
            result = self.label(manual, ''); result.grid(row=1, column=0, columnspan=6, sticky='w')
            saved = self.label(manual, '已存曝光：无'); saved.grid(row=2, column=0, columnspan=6, sticky='w')
            save_reason = self.label(manual, ''); save_reason.grid(row=3, column=0, columnspan=6, sticky='w')
            pic = self.picture(body, 'baseline_'+name); pic.grid_configure(row=2)
            info = tk.Label(body, text='预览关闭', bg=WHITE, fg=MUTED, font=self.theme.large, anchor='w', justify='left'); info.grid(row=3, column=0, sticky='ew')
            hint = self.label(body, '', fg=COLORS['red']); hint.grid(row=4, column=0, sticky='ew')
            detail = self.label(body, '', justify='left'); detail.grid(row=5, column=0, sticky='ew')
            graph = tk.Canvas(body, height=24, bg=WHITE, highlightthickness=0); graph.grid(row=6, column=0, sticky='ew')
            body.bind('<Configure>', lambda event, info=info, hint=hint, detail=detail: (info.config(wraplength=max(100, event.width-12)), hint.config(wraplength=max(100, event.width-12)), detail.config(wraplength=max(100, event.width-12))))
            self.baseline_views[name] = dict(path=path, toggle=toggle, exposure=exposure, reset=reset, reason=reason, info=info, hint=hint, detail=detail, graph=graph, picture=pic, exposure_entry=entry, exposure_set=set_exposure, exposure_restore=restore, exposure_save=save, exposure_saved=saved, exposure_save_reason=save_reason, exposure_limits=limits, exposure_result=result)
        results = self.card(self.baseline, '本次结果', 2, 0)
        self.baseline_results_card = results.master
        self.baseline_result = tk.Label(results, text='等待测量', bg=WHITE, fg=INK, anchor='nw', justify='left', font=self.theme.table, width=1, wraplength=600)
        self.baseline_result.grid(row=0, column=0, sticky='nsew')
        results.bind('<Configure>', lambda event: self.baseline_result.config(wraplength=max(200, event.width - 12)))
        old = self.card(self.baseline, '现有基准与工具补偿', 2, 1)
        self.baseline_existing_card = old.master
        self.baseline_existing = tk.Label(old, text='—', bg=WHITE, fg=INK, anchor='nw', justify='left', font=self.theme.table, width=1, wraplength=600)
        self.baseline_existing.grid(row=0, column=0, sticky='nsew')
        old.bind('<Configure>', lambda event: self.baseline_existing.config(wraplength=max(200, event.width - 12)))

    def baseline_toggle(self, name):
        item = getattr(self, 'baseline_preview', {}).get(name) or {}
        self.baseline_command('tooltip_preview', {'camera':name, 'on':True if item.get('error') else not item.get('on', False)})

    def _send_page_command(self, page, name, params=None):
        from nest_clear.tooloffset.commands import send
        if getattr(self, page + '_pending', 0):
            setattr(self, page + '_command_message', '上一条命令处理中')
            getattr(self, page + '_message').config(text='上一条命令处理中', fg=COLORS['yellow'])
            return
        try:
            seq = send(self.directory, name, params, acknowledged=getattr(self, page + '_ack', 0))
        except OSError:
            logger.exception('写命令失败：%s', name)
            setattr(self, page + '_command_message', '命令写入失败，请重试')
            return
        setattr(self, page + '_pending', seq)
        setattr(self, page + '_pending_name', name)
        setattr(self, page + '_pending_since', time.monotonic())
        setattr(self, page + '_command_message', '')
        getattr(self, page + '_message').config(text='上一条命令处理中', fg=COLORS['yellow'])
        return seq

    def _check_page_command(self, page, ack):
        pending = getattr(self, page + '_pending', 0)
        if not pending:
            return
        if ack >= pending:
            setattr(self, page + '_pending', 0)
            setattr(self, page + '_command_message', '')
        elif time.monotonic() - getattr(self, page + '_pending_since', time.monotonic()) >= 5:
            name = getattr(self, page + '_pending_name', '未知')
            message = f'上一条命令【{name}】未被确认（主程序/父程序可能未运行），已放弃，可重新操作'
            logger.warning(message)
            setattr(self, page + '_pending', 0)
            setattr(self, page + '_command_message', message)

    def baseline_command(self, name, params=None):
        return self._send_page_command('baseline', name, params)

    def baseline_exposure(self, name, restore):
        params = {'camera': name}
        if not restore:
            params['exposure_ms'] = self.baseline_views[name]['exposure_entry'].get()
        seq = self.baseline_command('tooltip_exposure_restore' if restore else 'tooltip_exposure_set', params)
        if seq:
            self.baseline_exposure_pending[name] = (seq, time.monotonic())
            self.baseline_views[name]['exposure_result'].config(text='设置中…', fg=INK)

    def baseline_save_exposure(self, name):
        view = self.baseline_views[name]
        current = view.get('current_exposure')
        if not isinstance(current, (int, float)):
            return
        saved = view.get('saved_exposure') or {}
        old = saved.get('exposure')
        old_text = f'{old/10:.1f} ms' if type(old) is int else '无'
        side = '正面' if name == 'front' else '侧面'
        if not messagebox.askyesno('设为实际曝光', f'将{side}相机曝光 {current:.1f} ms 设为实际参数（覆盖已存的 {old_text}）？', parent=self.root):
            return
        seq = self.baseline_command('tooltip_exposure_save', {'camera': name, 'expected_exposure': round(current * 10)})
        if seq:
            self.baseline_exposure_pending[name] = (seq, time.monotonic())
            view['exposure_result'].config(text='保存中…', fg=INK)

    def update_baseline(self, main, supervisor=None):
        data = main.get('baseline') or {}; phase = data.get('phase') or 'idle'
        self.baseline_ack = main.get('command_seq', 0)
        runout = data.get('runout') or {}
        pending_name = getattr(self, 'baseline_pending_name', '')
        ack = (runout.get('reply_seq') or 0) if pending_name.startswith('tooltip_runout_') else self.baseline_ack
        self._check_page_command('baseline', ack)
        tool = data.get('tool'); candidate = data.get('candidate') or {}; existing = data.get('existing') or {}
        self.baseline_heading.config(text=f'当前：工具{tool}' if tool else '当前：未选择', bg='#2563eb' if tool == 1 else '#d56713' if tool == 2 else '#667085')
        no_main = supervisor is not None and not supervisor.get('child_pid')
        message = getattr(self, 'baseline_command_message', '') or ('上一条命令处理中' if self.baseline_pending else data.get('message') or '请开始对刀基准建立')
        if phase == 'unknown': message = '对刀状态未知或已过期'
        if no_main: message = '主程序未运行（等待 PLC 初始化）　' + message
        msg_color = COLORS['red'] if phase == 'fail' or (phase == 'done' and any(word in message for word in ('失败', '异常'))) else COLORS['green'] if phase in ('success', 'saved') else INK
        self.baseline_message.config(text=message, fg=COLORS['red'] if getattr(self, 'baseline_command_message', '') else COLORS['yellow'] if self.baseline_pending else msg_color)
        live = data.get('live_delta')
        self.baseline_live.config(text=f'实时辅助：离中心 横向 {live[0]:+.3f} mm / 竖向 {live[1]:+.3f} mm' if isinstance(live, list) and len(live) == 2 else '实时辅助：等待针尖', fg=COLORS['green'] if data.get('live_inside') else COLORS['red'] if live else MUTED)
        notice = data.get('notice') or ''
        if not notice and tool in (1, 2):
            selected_old = existing.get(f'tool{tool}') or {}
            if selected_old.get('tool_offset_missing'): notice = f'提示：工具{tool} 尚无工具偏移标定结果（不阻止建基准）'
            elif selected_old.get('offset_newer'): notice = f'提示：工具{tool} 的工具偏移标定结果比现有对刀基准新（不阻止建基准）'
        camera_diffs = [f'工具{i}: ' + '、'.join((existing.get(f'tool{i}') or {}).get('camera_differences') or []) for i in (1,2) if (existing.get(f'tool{i}') or {}).get('camera_differences')]
        if camera_diffs: notice = '对刀相机设置已变更（' + '；'.join(camera_diffs) + '），基准需重建'
        self.baseline_notice.config(text=notice)
        def enable(widget, condition): widget.config(state='normal' if condition and not self.baseline_pending else 'disabled')
        enable(self.baseline_start, phase in ('idle', 'done', 'prepare') and (supervisor is None or bool(supervisor.get('child_pid'))))
        self.baseline_preview = data.get('preview') or {}; self.baseline_devices = data.get('detected_cameras') or []
        preparing = phase in ('idle', 'done', 'prepare') and any(
            (self.baseline_preview.get(n) or {}).get('on') or (self.baseline_preview.get(n) or {}).get('error')
            for n in ('front', 'side')
        )
        if preparing:
            self.baseline_results_card.grid_remove()
            self.baseline_existing_card.grid_remove()
            self.baseline_live.grid_remove()
            self.baseline_notice.grid_remove()
        else:
            self.baseline_results_card.grid()
            self.baseline_existing_card.grid()
            self.baseline_live.grid()
            if notice: self.baseline_notice.grid()
            else: self.baseline_notice.grid_remove()
        if supervisor is None:  # Legacy callers without a supervisor snapshot.
            permissions = {n: (phase in ('idle', 'done', 'prepare') and main.get('mode', '空闲') in ('空闲', '对刀基准建立中'), None) for n in ('front', 'side')}
        else:
            from nest_clear.preview.eligibility import preview_eligibility
            try:
                age = max(0, time.time() - supervisor_state_file(self.directory).stat().st_mtime)
            except OSError:
                age = None
            permissions = {n: preview_eligibility(supervisor, age, main.get('mode'), n) for n in ('front', 'side')}
        preview_path = tooltip_preview_file(self.directory)
        preview_doc = self.file_data(preview_path)
        try:
            preview_age = max(0.0, time.time() - preview_path.stat().st_mtime)
        except OSError:
            preview_age = None
        ready_reason = ''
        for name, label in (('front', '正面'), ('side', '侧面')):
            camera = (data.get('preview') or {}).get(name) or {}
            live_cam = extract_tooltip_preview(preview_doc, name, file_age=preview_age)
            if not camera.get('on'):
                ready_reason = f'请打开{label}相机预览'; break
            if not live_cam or live_cam.get('detection_status') != '精定位':
                ready_reason = f'{label}针尖未精定位'; break
        active = bool(runout.get('active')); busy = bool(runout.get('busy'))
        enable(self.runout_start, not active and not ready_reason)
        enable(self.runout_press, active and not busy and not ready_reason)
        enable(self.runout_end, active and not busy)
        result = runout.get('result') or {}
        maximum = result.get('max_deviation_um')
        self.runout_max.config(text=f'最大离心量 {maximum:.1f} µm　{"合格" if result.get("pass") else "不合格"}' if maximum is not None else '最大离心量 — µm',
                               fg=(COLORS['green'] if result.get('pass') else COLORS['red']) if maximum is not None else INK)
        if result:
            diameter = result.get('circle_fit_diameter_um')
            noise = result.get('noise_um') or {}
            detail = (f"次数 {result['count']}　ΔX {result['delta_x_um']:+.1f} / ΔY {result['delta_y_um']:+.1f} µm　"
                      f"{'合格' if result['pass'] else '不合格'}　{'已稳定' if result['stable'] else '未稳定'}　"
                      f"拟合圆直径 {f'{diameter:.1f}' if diameter is not None else '—'} µm（参考）　"
                      f"噪声 正面 {noise.get('front', 0):.1f} / 侧面 {noise.get('side', 0):.1f} µm")
        else:
            detail = '次数 0　等待记录'
        message = runout.get('message') or ''
        if message and not (ready_reason and message == '请打开两路预览并精定位'):
            detail += '　' + message
        if ready_reason: detail += '　' + ready_reason
        if runout.get('rejected_reason'): detail += '\n最近一次作废：' + runout['rejected_reason']
        self.runout_detail.config(text=detail)
        for name, widgets in self.baseline_views.items():
            may_prepare, blocked_reason = permissions[name]
            item = self.baseline_preview.get(name) or {}; on = item.get('on', False)
            device = item.get('device') or '未分配'
            import re
            short = re.search(r'usb-[0-9]+:[0-9.]+', str(device))
            rule = '口号小=正面' if name == 'front' else '口号大=侧面'
            widgets['path'].config(text='设备：' + (short.group(0) if short else str(device)) + f'（自动：{rule}）')
            title = '主对刀相机（正面）' if name == 'front' else '辅助对刀相机（侧面）'
            live = extract_tooltip_preview(preview_doc, name, file_age=preview_age)
            display = baseline_preview_display(item, live=live, allow_inline=(preview_age is None and not preview_doc))
            button_text = title+'预览：'+('未知' if phase == 'unknown' else '开' if on else '关')
            widgets['reason'].config(text='' if may_prepare else '不可用：' + (blocked_reason or '会话进行中'))
            if phase == 'unknown': may_prepare = False
            if may_prepare: widgets['reason'].grid_remove()
            else: widgets['reason'].grid()
            widgets['toggle'].config(text=button_text, fg=COLORS['red'] if display['failed'] else INK, bg='#fee4e2' if display['failed'] else '#b7ebc6' if on else '#e5e7eb', state='normal' if may_prepare and not self.baseline_pending else 'disabled')
            enable(widgets['exposure'], on and may_prepare and not item.get('exposure_busy')); enable(widgets['reset'], on and may_prepare)
            limits = item.get('exposure_limits') or {}
            valid_limits = all(type(limits.get(key)) is int for key in ('min', 'max', 'step')) and limits.get('step', 0) > 0
            if valid_limits:
                widgets['exposure_entry'].config(from_=limits['min']/10, to=limits['max']/10, increment=limits['step']/10)
                widgets['exposure_limits'].config(text=f"范围 {limits['min']/10:g}–{limits['max']/10:g} ms，步进 {limits['step']/10:g}")
            else:
                widgets['exposure_limits'].config(text='范围 —')
            pending = self.baseline_exposure_pending.get(name)
            result = item.get('exposure_result') or {}
            if pending and result.get('seq') == pending[0] and result.get('state') in ('done', 'error'):
                self.baseline_exposure_pending.pop(name, None)
                widgets['exposure_result'].config(text=result.get('message', ''), fg=COLORS['red'] if result['state'] == 'error' else INK)
            elif pending and time.monotonic() - pending[1] >= 5:
                self.baseline_exposure_pending.pop(name, None)
                widgets['exposure_result'].config(text='曝光操作未完成或未确认，请检查预览', fg=COLORS['red'])
            pending = name in self.baseline_exposure_pending
            for key in ('exposure_entry', 'exposure_set', 'exposure_restore'):
                enable(widgets[key], on and may_prepare and valid_limits and not pending and not item.get('exposure_busy'))
            saved = item.get('exposure_saved') or {}
            widgets['saved_exposure'] = saved
            saved_value = saved.get('exposure')
            saved_text = f'{saved_value/10:.1f} ms　{saved.get("time") or "时间未知"}' if type(saved_value) is int else '无'
            widgets['exposure_saved'].config(text='已存曝光：' + saved_text)
            current = (live or {}).get('exposure_ms') if isinstance(live, dict) else None
            if current is None:
                current = item.get('exposure_ms')
            widgets['current_exposure'] = current
            save_reason = ('预览未打开' if not on else '曝光操作进行中' if item.get('exposure_busy') or pending else '当前曝光未读回' if not isinstance(current, (int, float)) else '当前不可操作' if not may_prepare else '')
            enable(widgets['exposure_save'], not save_reason)
            widgets['exposure_save_reason'].config(text='不可保存：' + save_reason if save_reason else '')
            def number(value): return f'{value:.2f}' if isinstance(value, (float,int)) else '—'
            delay_warning = False
            if display['failed']:
                text = display['message']; color = COLORS['red']
                detail = '清晰度 —　边缘 — µm / — px　对比度 —　识别来源 —　曝光 —'
            elif not display['on']:
                text = '状态未知或已过期' if phase == 'unknown' else '预览关闭'; color = MUTED
                detail = f'边缘 — µm / — px　对比度 —　识别来源 —　曝光 — ms {item.get("exposure_source", "未测")}'
            elif display['no_live_data']:
                text = display['message']
                color = COLORS['red'] if item.get('exposure_error') else MUTED
                detail = (str(item.get('exposure_error')) + '　' if item.get('exposure_error') else '') + f'边缘 — µm / — px　对比度 —　识别来源 —　曝光 — ms {item.get("exposure_source", "未测")}'
                cameras = preview_doc.get('cameras') if isinstance(preview_doc, dict) else None
                stale = cameras.get(name, {}) if isinstance(cameras, dict) else {}
                try:
                    age = max(0, (datetime.now().astimezone() - datetime.fromisoformat(stale['captured_at'])).total_seconds() * 1000)
                    if age > 1000:
                        detail += f'　帧龄 {age:.0f} ms　画面延迟'
                        delay_warning = True
                except (KeyError, ValueError, TypeError):
                    pass
            else:
                src = display['live']
                pct = src.get('percent')
                color = COLORS['green'] if isinstance(pct, (int,float)) and pct >= 95 else COLORS['yellow'] if isinstance(pct, (int,float)) and pct >= 80 else COLORS['red']
                percent_text = f'{pct:.0f}' if isinstance(pct, (int, float)) else '—'
                if src.get('source') == 'band_gradient_p995':
                    text = f'清晰度参考值 {percent_text}%　当前 {number(src.get("band_gradient"))} / 本次最佳 {number(src.get("peak"))}'
                else:
                    text = f'相对清晰度 {percent_text}%　当前 {number(src.get("edge_width_px"))} px / 本次最佳 {number(src.get("peak"))} px'
                tip = src.get('tip_uv')
                delta = src.get('delta_uv')
                source = f'{src.get("detection_status", "—")} {str(src.get("detection_reason") or "")}　针尖 ' + (f'({number(tip[0])}, {number(tip[1])})' if tip else '—')
                if name == 'front' and delta:
                    source += f'　Δu {number(delta[0])}（右+） Δv {number(delta[1])}（下+）px'
                if src.get('axis_rms_px') is not None:
                    source += f'　axis_rms {number(src.get("axis_rms_px"))} px'
                if src.get('focus_reason'):
                    source += '　清晰度无效：' + src['focus_reason']
                exposure_source = item.get('exposure_source') or src.get('exposure_source') or '未测'
                exposure_value = src.get('exposure_ms')
                exposure_text = f'{exposure_value:.1f}' if isinstance(exposure_value, (int, float)) else '—'
                detail = (str(item.get('exposure_error')) + '\n' if item.get('exposure_error') else '') + f'{source}\n边缘 {number(src.get("edge_width_um"))} µm / {number(src.get("edge_width_px"))} px　最佳 {number(src.get("peak_edge_width_um"))} µm　对比度 {number(src.get("contrast"))}　曝光 {exposure_text} ms（{exposure_source}）'
                age = src.get('frame_age_ms')
                if isinstance(age, (int, float)):
                    try:
                        age += max(0, (datetime.now().astimezone() - datetime.fromisoformat(src['updated_at'])).total_seconds() * 1000)
                    except (KeyError, ValueError, TypeError):
                        pass
                    detail += f'　帧龄 {age:.0f} ms' + ('　画面延迟' if age > 1000 else '')
                    delay_warning = age > 1000
            widgets['info'].config(text=text, fg=color)
            hint_text = display.get('hint', '')
            widgets['hint'].config(text=hint_text)
            if hint_text: widgets['hint'].grid()
            else: widgets['hint'].grid_remove()
            widgets['detail'].config(text=detail, fg=COLORS['red'] if delay_warning else INK)
            graph=widgets['graph']; graph.delete('all'); values=display.get('history') or []
            if len(values)>1:
                width=max(2, graph.winfo_width()); height=max(2, graph.winfo_height())
                points=[coord for i,v in enumerate(values) for coord in (i*width/(len(values)-1), height-max(0,min(100,v))*height/100)]
                graph.create_line(*points, fill=COLORS['blue'], width=2)

        def enable_main(widget, condition): enable(widget, condition and not no_main)
        for i, button in enumerate(self.baseline_tools, 1):
            enable_main(button, phase == 'select'); button.config(style='OffsetSelected.TButton' if i == tool else 'TButton')
        for b in (self.baseline_measure, self.baseline_manual): enable_main(b, phase in ('live','fail','success','saved'))
        enable_main(self.baseline_aux, phase in ('success','saved') and bool(candidate.get('p0')))
        enable_main(self.baseline_save, phase == 'success')
        enable_main(self.baseline_end, phase not in ('idle','done'))
        pixels = candidate.get('pixels') or {}; focus = candidate.get('focus') or {}; click = candidate.get('click') or {}
        def vector(value, digits=2):
            return '(' + ', '.join(f'{x:.{digits}f}' for x in value) + ')' if isinstance(value, (list,tuple)) else '—'
        lines = [f"主相机偏差 mm：{vector(candidate.get('delta_mm'), 3)}　{'在圈内' if candidate.get('inside') else '不在圈内' if candidate else '—'}",
                 f"主/辅针尖：{vector(pixels.get('front'))} / {vector(pixels.get('side'))}　P0：{vector(candidate.get('p0'), 3)}"]
        for key, title in (('front','主'), ('side','辅助')):
            item = focus.get(key) or {}
            conf = item.get('confidence'); conf = f'{conf:.2f}' if isinstance(conf, (int, float)) else '—'
            focused = item.get('focused'); focused_text = ('清晰' if focused else '模糊') if isinstance(focused, bool) else '—'
            edge = item.get('edge_width_px'); edge = f'{edge:.2f}' if isinstance(edge, (int,float)) else '—'
            coarse = '（粗定位）' if item.get('coarse') else ''
            lines.append(f"{title}相机：可靠度 {conf}　{focused_text}{coarse}　边缘 {edge}px　人工 {'是' if key in click else '否'}")
        self.baseline_result.config(text='\n'.join(lines))
        lines = []
        for i in (1, 2):
            old = existing.get(f'tool{i}') or {}
            flags = ' '.join(x for x in ('雅可比待定' if old.get('jacobian_pending') else '', '基准过时' if old.get('stale') else '') if x)
            offset_note = '工具偏移不存在' if old.get('tool_offset_missing') else '工具偏移比基准新' if old.get('offset_newer') else ''
            comp = vector(old.get('offset_xyz'), 3)
            lines.append(f"工具{i}：{display_time(old.get('time'))}　P0 {vector(old.get('p0'), 3)}　补偿 {comp}　{flags}".rstrip())
            lines.append(f"　主/辅 {vector(old.get('pixels',{}).get('front'))} / {vector(old.get('pixels',{}).get('side'))}　{offset_note}".rstrip())
        cfg = data.get('config') or {}; selected_comp = data.get('offset_xyz')
        lines.append(f"当前所选补偿：{vector(selected_comp, 3)}　主/辅比例：{cfg.get('main_um_per_px',13.5)} / {cfg.get('aux_um_per_px',13.5)} μm/px（11-25 实测）")
        self.baseline_existing.config(text='\n'.join(lines))
        if phase == 'confirm' and self.baseline_dialog is None: self.baseline_confirm(data)
        elif phase != 'confirm' and self.baseline_dialog is not None: self.baseline_dialog.destroy(); self.baseline_dialog = None

    def baseline_confirm(self, data):
        tool = data.get('tool'); candidate = data.get('candidate') or {}; old = (data.get('existing') or {}).get(f'tool{tool}') or {}
        top = tk.Toplevel(self.root); self.baseline_dialog = top; top.title(f'确认覆盖【工具{tool}】的对刀基准')
        modal_window(top, self.root)
        tk.Label(top, text=f'确认覆盖【工具{tool}】的对刀基准', bg='#2563eb' if tool == 1 else '#d56713', fg=WHITE, font=(self.theme.title[0], 20, 'bold'), padx=20, pady=8).pack(fill='x')
        def difference(a, b):
            return [round(x-y, 4) for x,y in zip(a,b)] if isinstance(a, (tuple,list)) and isinstance(b, (tuple,list)) and len(a)==len(b) else '—'
        def vec(a):
            return '(' + ', '.join(f'{v:.3f}' for v in a) + ')' if isinstance(a, (tuple,list)) else '—'
        op = old.get('pixels') or {}; npix = candidate.get('pixels') or {}
        text = f"旧：{old.get('time','无')}  P0 {vec(old.get('p0'))}  主 {vec(op.get('front'))}  辅 {vec(op.get('side'))}\n新：待保存  P0 {vec(candidate.get('p0'))}  主 {vec(npix.get('front'))}  辅 {vec(npix.get('side'))}\n差值：P0 {vec(difference(candidate.get('p0'),old.get('p0')))}  主 {vec(difference(npix.get('front'),op.get('front')))}  辅 {vec(difference(npix.get('side'),op.get('side')))}"
        tk.Label(top, text=text, justify='left', font=self.theme.body, padx=20, pady=16).pack()
        row=tk.Frame(top); row.pack()
        ttk.Button(row, text='覆盖保存', command=lambda:self.baseline_command('baseline_confirm', {'accept':True})).pack(side='left', padx=15, pady=10)
        ttk.Button(row, text='不保存', command=lambda:self.baseline_command('baseline_confirm', {'accept':False})).pack(side='left', padx=15, pady=10)
        if hasattr(top, 'protocol'): top.protocol('WM_DELETE_WINDOW', lambda:self.baseline_command('baseline_confirm', {'accept':False}))
        if hasattr(top, 'winfo_reqwidth'):
            top.update_idletasks()
            x = self.root.winfo_rootx() + max(0, (self.root.winfo_width() - top.winfo_reqwidth()) // 2)
            y = self.root.winfo_rooty() + max(0, (self.root.winfo_height() - top.winfo_reqheight()) // 2)
            top.geometry(f'+{x}+{y}')

    def build_pickacc(self):
        self.pickacc_tool = None; self.pickacc_pending = 0; self.pickacc_dialog = None; self.pickacc_last_phase = None
        self.pickacc.columnconfigure(0, weight=1); self.pickacc.columnconfigure(1, weight=1)
        self.pickacc.rowconfigure(1, weight=3, uniform='pickacc'); self.pickacc.rowconfigure(2, weight=2, uniform='pickacc')
        head = self.card(self.pickacc, '挑取精度标定', 0, 0, columnspan=2)
        bar = tk.Frame(head, bg=WHITE); bar.grid(row=0, column=0, sticky='ew'); bar.columnconfigure(2, weight=1)
        self.pickacc_heading = tk.Label(bar, text='当前：未选择', bg='#667085', fg=WHITE, font=(self.theme.title[0],20,'bold'), padx=12)
        self.pickacc_heading.grid(row=0,column=0,padx=4)
        self.pickacc_message = tk.Label(bar, text='等待 PLC 请求', bg=WHITE, fg=INK, font=(self.theme.title[0],16), anchor='w')
        self.pickacc_message.grid(row=0,column=1,columnspan=2,sticky='ew',padx=8)
        controls=tk.Frame(head,bg=WHITE); controls.grid(row=1,column=0,sticky='ew'); controls.columnconfigure(4,weight=1)
        self.pickacc_buttons=[]
        for n in (1,2):
            b=ttk.Button(controls,text=f'工具{n}',command=lambda n=n:self.choose_pickacc_tool(n)); b.grid(row=0,column=n-1,padx=3); self.pickacc_buttons.append(b)
        self.pickacc_start=ttk.Button(controls,text='开始',command=lambda:self.pickacc_command('pickacc_start',{'tool':self.pickacc_tool})); self.pickacc_start.grid(row=0,column=2,padx=4)
        self.pickacc_calculate=ttk.Button(controls,text='计算补偿',style='OffsetPlace.TButton',command=lambda:self.pickacc_command('pickacc_calculate')); self.pickacc_calculate.grid(row=0,column=5,padx=4)
        self.pickacc_discard=ttk.Button(controls,text='本点作废重测',command=lambda:self.pickacc_command('pickacc_discard')); self.pickacc_discard.grid(row=0,column=6,padx=4)
        self.pickacc_abort=ttk.Button(controls,text='放弃本次标定',command=lambda:self.pickacc_command('abort')); self.pickacc_abort.grid(row=0,column=7,padx=4)
        self.pickacc_existing=tk.Label(head,text='',bg=WHITE,fg=COLORS['yellow'],font=self.theme.body,anchor='w',justify='left'); self.pickacc_existing.grid(row=2,column=0,sticky='ew')
        images=tk.Frame(self.pickacc,bg=BG); images.grid(row=1,column=0,sticky='nsew'); images.columnconfigure(0,weight=3); images.columnconfigure(1,weight=2); images.rowconfigure(0,weight=1)
        self.picture(self.card(images,'相机画面 · 点击放大',0,0),'pickacc_latest')
        self.picture(self.card(images,'当前目标放大图',0,1),'pickacc_crop')
        tray=self.card(self.pickacc,'料盘示意',1,1)
        self.pickacc_tray_title=tray.master.title_label
        self.pickacc_tray=tk.Canvas(tray,bg=WHITE,highlightthickness=0); self.pickacc_tray.grid(row=0,column=0,sticky='nsew')
        table=self.card(self.pickacc,'测点记录',2,0,columnspan=2)
        columns=('序号','目标','阶段','下发 XY','实到 XY','dx','dy','|r|','备注')
        self.pickacc_table=self.table_widget(table,columns,(55,60,95,165,165,75,75,75,190))
        scroll=ttk.Scrollbar(table,orient='vertical',command=self.pickacc_table.yview); scroll.grid(row=0,column=1,sticky='ns'); self.pickacc_table.configure(yscrollcommand=scroll.set,style='Offset.Treeview')
        self.pickacc_stats=tk.Label(table,text='',bg=WHITE,fg=INK,font=self.theme.body,anchor='w'); self.pickacc_stats.grid(row=1,column=0,sticky='ew')

    def choose_pickacc_tool(self, tool):
        self.pickacc_tool=tool

    def pickacc_command(self, name, params=None):
        self._send_page_command('pickacc', name, params)

    def update_pickacc(self,main):
        data=main.get('pickacc') or {}; phase=data.get('phase'); self.pickacc_ack=main.get('command_seq',0)
        if phase=='select' and self.pickacc_last_phase!='select': self.pickacc_tool=None
        self.pickacc_last_phase=phase
        self._check_page_command('pickacc', self.pickacc_ack)
        tool=data.get('tool') or self.pickacc_tool
        self.pickacc_heading.config(text=f'当前：工具{tool}' if tool else '当前：未选择',bg='#2563eb' if tool==1 else '#d56713' if tool==2 else '#667085')
        self.pickacc_message.config(text=getattr(self, 'pickacc_command_message', '') or ('上一条命令处理中' if self.pickacc_pending else data.get('message')) or '等待 PLC 请求', fg=COLORS['red'] if getattr(self, 'pickacc_command_message', '') else INK)
        for i,b in enumerate(self.pickacc_buttons,1): b.config(state='normal' if phase=='select' and not self.pickacc_pending else 'disabled',style='OffsetSelected.TButton' if i==tool else 'TButton')
        self.pickacc_start.config(state='normal' if phase=='select' and self.pickacc_tool and not self.pickacc_pending else 'disabled')
        for b in (self.pickacc_calculate,self.pickacc_discard): b.config(state='normal' if phase=='tool_in_place' and not self.pickacc_pending else 'disabled')
        self.pickacc_abort.config(state='normal' if phase not in (None,'idle','done','failed') and not self.pickacc_pending else 'disabled')
        summaries=[]
        for i in (1,2):
            old=(data.get('existing') or {}).get(f'tool{i}') or {}
            summaries.append(f"工具{i}：{pickacc_type(old.get('type'))} {display_time(old.get('time'))} 验证 {len(old.get('verification',[]))}点" + (f'　工具{i} 偏移/相机标定已变更，挑取补偿需重新验证' if old.get('stale') else ''))
        self.pickacc_existing.config(text='　｜　'.join(summaries))
        self.pickacc_table.delete(*self.pickacc_table.get_children())
        import math
        for i,s in enumerate(data.get('samples') or [],1):
            r=s.get('residual_xy_mm'); sent=s.get('sent_xyz_mm'); actual=s.get('actual_xyz_mm')
            xy=lambda v: f'{v[0]:.3f}, {v[1]:.3f}' if v and len(v)>=2 else '—'
            self.pickacc_table.insert('','end',values=(i,s.get('target'),pickacc_stage(s.get('stage')),xy(sent),xy(actual),f'{r[0]:.3f}' if r else '—',f'{r[1]:.3f}' if r else '—',f'{math.hypot(*r):.3f}' if r else '—',pickacc_note(s)))
        decision=data.get('decision') or {}; candidate=data.get('candidate') or {}
        mode=pickacc_type(candidate['type']) if candidate.get('type') else '待判定'
        validation=[math.hypot(*s['residual_xy_mm']) for s in data.get('samples',[]) if s.get('verification') and s.get('residual_xy_mm') and not s.get('superseded')]
        validation_text=f'验证：{len(validation)} 点，最大 {max(validation):.3f} mm' if validation else '验证：未开始'
        self.pickacc_stats.config(text=f"判定：{mode}  最大偏差 {decision.get('max_dev_mm','—')} / 标准差 {decision.get('std_mm','—')}　{pickacc_infield(data.get('infield') or [])}　{validation_text}")
        self.draw_pickacc_tray(data)
        if phase=='confirm' and self.pickacc_dialog is None: self.pickacc_confirm(data)
        elif phase!='confirm' and self.pickacc_dialog is not None: self.pickacc_dialog.destroy(); self.pickacc_dialog=None

    def draw_pickacc_tray(self,data):
        import math
        c=self.pickacc_tray; c.delete('all'); w=max(c.winfo_width(),200); h=max(c.winfo_height(),120)
        left,bottom,scale,multiplier=pickacc_tray_geometry(w,h)
        self.pickacc_tray_title.config(text=f'料盘示意 · 残差箭头 ×{multiplier}  绿≤0.03 黄≤0.05 红>0.05 mm')
        def pos(x,y): return (left+(x-80)*scale, bottom-(y+10)*scale)
        a=pos(80,-10); b=pos(180,150); c.create_rectangle(a[0],b[1],b[0],a[1],outline='#667085')
        samples=[s for s in data.get('samples') or [] if s.get('target_xy_mm') and s.get('residual_xy_mm') and not s.get('verification')]
        candidate=data.get('candidate') or {}; nodes=candidate.get('nodes') or []
        if candidate.get('type')=='table' and len(nodes)>=3:
            cells=[(80+(gx+.5)*5,-10+(gy+.5)*8) for gx in range(20) for gy in range(20)]
            magnitudes=[]
            for px,py in cells:
                nearest=min(nodes,key=lambda n:(n['x']-px)**2+(n['y']-py)**2)
                magnitudes.append(math.hypot(nearest['dx'],nearest['dy']))
            for gx in range(20):
                for gy in range(20):
                    value=magnitudes[gx*20+gy]
                    color='#f1f9f3' if value<=.03 else '#fffae9' if value<=.05 else '#fdf1ef'
                    x0,y0=pos(80+gx*5,-10+gy*8); x1,y1=pos(85+gx*5,-2+gy*8)
                    c.create_rectangle(x0,y1,x1,y0,fill=color,outline=color)
            c.create_rectangle(a[0],b[1],b[0],a[1],outline='#667085')
        planned=set()
        planned.update(x.get('target') for x in data.get('queue',[]))
        planned.update(x.get('target') for x in data.get('samples',[]))
        planned.add(data.get('target'))
        verification={x.get('target') for x in data.get('queue',[]) if x.get('stage')=='verify'}
        skipped={x.get('target') for x in data.get('skipped',[])}
        for label,xy in (data.get('predicted') or {}).items():
            if label not in planned or not (80<=xy[0]<=180 and -10<=xy[1]<=150): continue
            x,y=pos(*xy)
            if label in skipped:
                c.create_line(x-4,y-4,x+4,y+4,fill='#b42318',width=2); c.create_line(x-4,y+4,x+4,y-4,fill='#b42318',width=2)
            elif label in verification: c.create_polygon(x,y-5,x+5,y,x,y+5,x-5,y,fill=WHITE,outline='#667085')
            else: c.create_oval(x-3,y-3,x+3,y+3,outline='#98a2b3')
            c.create_text(x+5,y-5,text=label,anchor='sw',fill='#667085',font=(self.theme.body[0],8))
        for s in data.get('samples') or []:
            xy=s.get('target_xy_mm'); r=s.get('residual_xy_mm')
            if not xy: continue
            if s.get('discarded') and not r:
                x,y=pos(*xy); c.create_line(x-5,y-5,x+5,y+5,fill='#b42318',width=2); c.create_line(x-5,y+5,x+5,y-5,fill='#b42318',width=2); continue
            if not r: continue
            x,y=pos(*xy); d=math.hypot(*r); color='#168448' if d<=.03 else '#d69e00' if d<=.05 else '#b42318'
            if s.get('verification'): c.create_polygon(x,y-5,x+5,y,x,y+5,x-5,y,fill=color)
            else: c.create_oval(x-4,y-4,x+4,y+4,fill=color,outline=color)
            c.create_line(x,y,x+r[0]*multiplier*scale,y-r[1]*multiplier*scale,fill=color,width=2,arrow='last',arrowshape=(8,10,4))
        for xval in range(80,181,20):
            x,y=pos(xval,-10); c.create_line(x,y,x,y+4,fill=INK); c.create_text(x,y+6,text=str(xval),anchor='n',fill=INK,font=(self.theme.body[0],8))
        for yval in range(-10,151,20):
            x,y=pos(80,yval); c.create_line(x-4,y,x,y,fill=INK); c.create_text(x-7,y,text=str(yval),anchor='e',fill=INK,font=(self.theme.body[0],8))
        c.create_text(b[0]+20,a[1]+6,text='X mm',anchor='nw',fill=INK,font=(self.theme.body[0],8))
        c.create_text(a[0]-8,b[1]-2,text='Y mm',anchor='se',fill=INK,font=(self.theme.body[0],8))
        target=(data.get('predicted') or {}).get(data.get('target'))
        if target:
            x,y=pos(*target); c.create_oval(x-8,y-8,x+8,y+8,outline='#2563eb',width=2)

    def pickacc_confirm(self,data):
        tool=data.get('tool'); candidate=data.get('candidate') or {}; old=(data.get('existing') or {}).get(f'tool{tool}') or {}
        top=tk.Toplevel(self.root); self.pickacc_dialog=top; top.title(f'确认覆盖【工具{tool}】挑取补偿'); modal_window(top, self.root)
        tk.Label(top,text=f'确认覆盖【工具{tool}】挑取补偿',bg='#2563eb' if tool==1 else '#d56713',fg=WHITE,font=self.theme.large,padx=16).pack(fill='x')
        prior=old.get('constant_xy'); current=candidate.get('constant_xy')
        def mean_value(value,nodes):
            if isinstance(value,list) and len(value)==2: return value
            if nodes: return [sum(n[k] for n in nodes)/len(nodes) for k in ('dx','dy')]
            return None
        old_mean=mean_value(prior,old.get('nodes',[])); new_mean=mean_value(current,candidate.get('nodes',[]))
        delta=[round(new_mean[i]-old_mean[i],4) for i in range(2)] if old_mean is not None and new_mean is not None else '无旧结果'
        import math
        valid=[math.hypot(*s['residual_xy_mm']) for s in candidate.get('verification',[]) if s.get('residual_xy_mm')]
        tk.Label(top,text=f"旧：{pickacc_type(old.get('type'))} {prior or str(len(old.get('nodes',[])))+' 节点'}\n新：{pickacc_type(candidate.get('type'))} {current or str(len(candidate.get('nodes',[])))+' 节点'}\n均值差：{delta}\n验证：{len(candidate.get('verification',[]))} 点通过，最大 {max(valid,default=0):.3f} mm",justify='left',font=self.theme.body,padx=25,pady=15).pack()
        row=tk.Frame(top); row.pack(); ttk.Button(row,text='覆盖保存',command=lambda:self.pickacc_command('pickacc_confirm',{'accept':True})).pack(side='left',padx=15,pady=10); ttk.Button(row,text='不保存',command=lambda:self.pickacc_command('pickacc_confirm',{'accept':False})).pack(side='left',padx=15,pady=10)
        top.protocol('WM_DELETE_WINDOW',lambda:self.pickacc_command('pickacc_confirm',{'accept':False}))
        top.update_idletasks()
        x=self.root.winfo_rootx()+max(0,(self.root.winfo_width()-top.winfo_reqwidth())//2)
        y=self.root.winfo_rooty()+max(0,(self.root.winfo_height()-top.winfo_reqheight())//2)
        top.geometry(f'+{x}+{y}')

    def choose_offset_tool(self, tool):
        self.offset_tool = tool
        self.offset_heading.config(text=f"当前：工具{tool}", bg='#2563eb' if tool == 1 else '#d56713')
        for i, button in enumerate(self.offset_tool_buttons, 1):
            button.config(style='OffsetSelected.TButton' if i == tool else 'TButton')

    def choose_offset_mode(self, mode):
        self.offset_mode = mode
        for i, button in enumerate(self.offset_mode_buttons):
            button.config(style='OffsetSelected.TButton' if mode == ('center','center_edge')[i] else 'TButton')
        self.offset_mode_text.config(text='模式：' + ('仅中心' if mode == 'center' else '中心+边缘验证'))
        if self.offset_tool: self.choose_offset_tool(self.offset_tool)

    def offset_command(self, name, params=None):
        self._send_page_command('offset', name, params)

    def start_offset(self):
        if self.offset_tool:
            self.offset_command('select_tool_and_start', {'tool':self.offset_tool, 'mode':self.offset_mode})

    def update_offset(self, main):
        data = main.get('tooloffset') or {}; phase = data.get('phase')
        if phase == 'select' and self.offset_last_phase != 'select':
            self.offset_tool, self.offset_mode = None, 'center'
        self.offset_last_phase = phase
        self.offset_ack = main.get('command_seq', 0)
        self._check_page_command('offset', self.offset_ack)
        tool = data.get('tool') or self.offset_tool
        mode = data.get('mode') or self.offset_mode
        self.offset_heading.config(text=f"当前：{'工具' + str(tool) if tool else '未选择'}", bg='#2563eb' if tool == 1 else '#d56713' if tool == 2 else '#667085')
        self.offset_mode_text.config(text='模式：' + ('仅中心' if mode == 'center' else '中心+边缘验证'))
        for i, button in enumerate(self.offset_tool_buttons, 1): button.config(style='OffsetSelected.TButton' if i == tool else 'TButton')
        for i, button in enumerate(self.offset_mode_buttons): button.config(style='OffsetSelected.TButton' if mode == ('center','center_edge')[i] else 'TButton')
        skipped = data.get('skipped') or []
        skip_text = '；跳过：' + '、'.join(f"{x.get('target')} {x.get('reason')}" for x in skipped) if skipped else ''
        self.offset_message.config(text=(getattr(self, 'offset_command_message', '') or ('上一条命令处理中' if self.offset_pending else data.get('message')) or '等待 PLC 请求') + skip_text, fg=COLORS['red'] if getattr(self, 'offset_command_message', '') else INK)
        selection_state = 'normal' if phase == 'select' and not self.offset_pending else 'disabled'
        for button in self.offset_tool_buttons + self.offset_mode_buttons:
            button.config(state=selection_state)
        self.offset_start.config(state=selection_state)
        self.offset_place.config(state='normal' if phase == 'tool' and not self.offset_pending else 'disabled')
        self.offset_abort.config(state='normal' if phase in ('select','tool','confirm','moving','wait90','arrival') and not self.offset_pending else 'disabled')
        existing = data.get('existing') or {}
        text = []; warnings = []
        for i in (1,2):
            old = existing.get(f'tool{i}') or {}
            text.append(f"工具{i} {display_time(old.get('time'))}  XY={old.get('offset_xy_mm','—')}  n={old.get('n','—')}  SE={old.get('se','—')}  当前={old.get('current_tools_xy','—')}")
            if old.get('calibration_changed'): warnings.append(f'工具{i}：相机标定已变更，原工具偏移可能失效')
            if old.get('baseline_stale'): warnings.append(f'工具{i} 偏移已重新标定，需重建对刀基准后才能换针对刀')
        self.offset_existing.config(text='\n'.join(text))
        self.offset_warning.config(text='；'.join(w for w in warnings if '对刀基准' in w))
        self.offset_caution.config(text='；'.join([*(w for w in warnings if '相机标定已变更' in w), *([data['recognition_warning']] if data.get('recognition_warning') else [])]))
        if hasattr(self, 'tip_stale'): self.tip_stale.config(text='；'.join(w for w in warnings if '对刀基准' in w))
        self.offset_samples.delete(*self.offset_samples.get_children())
        for i, sample in enumerate(data.get('samples') or [], 1):
            xy = sample.get('offset_xy_mm', [0,0])
            self.offset_samples.insert('', 'end', values=(i, sample.get('target'), f'{xy[0]:.4f}', f'{xy[1]:.4f}', f"{sample.get('distance_from_mean_mm',0):.4f}", '离群' if sample.get('outlier') else ''))
        se = data.get('se'); good = isinstance(se, list) and len(se) == 2 and sum(not x.get('outlier') for x in data.get('samples', [])) >= data.get('min_points',5) and max(se) <= data.get('se_tol_mm', .02)
        self.offset_stats.config(text=f"均值{data.get('mean') or '—'} std{data.get('std') or '—'} SE{se or '—'}", fg=COLORS['green'] if good else INK)
        self.offset_edges.delete(*self.offset_edges.get_children())
        for edge in data.get('edges') or []:
            xy = edge.get('delta_xy_mm', [0,0]); distance = edge.get('distance_mm',0)
            self.offset_edges.insert('', 'end', values=(edge.get('target'), edge_corner_label(edge.get('corner')), f'{xy[0]:.4f}', f'{xy[1]:.4f}', f'{distance:.4f}' + (' ⚠' if distance > data.get('edge_warn_mm',.05) else '')), tags=('warning',) if distance > data.get('edge_warn_mm',.05) else ())
        if phase == 'confirm' and self.offset_dialog is None:
            self.offset_confirm(data, existing.get(f'tool{tool}', {}))
        elif phase != 'confirm' and self.offset_dialog is not None:
            self.offset_dialog.destroy(); self.offset_dialog = None

    def offset_confirm(self, data, old):
        tool = data.get('tool')
        heading = f'确认覆盖【工具{tool}】的偏移标定结果'
        top = tk.Toplevel(self.root); self.offset_dialog = top; top.title(heading)
        modal_window(top, self.root)
        candidate = data.get('candidate') or {}
        prior, new = old.get('offset_xy_mm'), candidate.get('offset_xy_mm')
        delta = [round(new[i]-prior[i], 5) for i in range(2)] if isinstance(prior, list) and isinstance(new, list) and len(prior)==len(new)==2 else '—'
        tk.Label(top, text=heading, bg='#2563eb' if tool == 1 else '#d56713', fg=WHITE, font=(self.theme.title[0], 20, 'bold'), padx=12, pady=6).grid(row=0, column=0, columnspan=2, sticky='ew')
        columns = [('旧结果', old), ('新结果', candidate)]
        for col, (title, result) in enumerate(columns):
            tk.Label(top, text=title, font=self.theme.title).grid(row=1, column=col, padx=24, pady=5)
            details = (f"时间：{result.get('time','本次' if col else '无')}\n"
                       f"偏移 XY：{result.get('offset_xy_mm','—')}\n"
                       f"有效点数：{result.get('n','—')}\nSE：{result.get('se','—')}")
            tk.Label(top, text=details, justify='left', font=self.theme.body).grid(row=2, column=col, padx=24, sticky='nw')
        tk.Label(top, text=f"偏移差值：{delta}　　tools.yaml 当前 XY：{old.get('current_tools_xy','—')}", font=self.theme.body).grid(row=3, column=0, columnspan=2, pady=8)
        ttk.Button(top, text='覆盖保存', command=lambda: self.offset_command('confirm_overwrite', {'accept':True})).grid(row=4, column=0, pady=8)
        ttk.Button(top, text='不保存', command=lambda: self.offset_command('confirm_overwrite', {'accept':False})).grid(row=4, column=1, pady=8)
        if hasattr(top, 'protocol'): top.protocol('WM_DELETE_WINDOW', lambda: self.offset_command('confirm_overwrite', {'accept':False}))
        if hasattr(top, 'winfo_reqwidth'):
            top.update_idletasks()
            x = self.root.winfo_rootx() + max(0, (self.root.winfo_width() - top.winfo_reqwidth()) // 2)
            y = self.root.winfo_rooty() + max(0, (self.root.winfo_height() - top.winfo_reqheight()) // 2)
            top.geometry(f'+{x}+{y}')

    def build_tip(self):
        self.tip.columnconfigure(0, weight=1, uniform='tipcols'); self.tip.columnconfigure(1, weight=1, uniform='tipcols')
        self.tip.rowconfigure(1, weight=6); self.tip.rowconfigure(2, weight=3)
        head = tk.Frame(self.tip, bg=WHITE); head.grid(row=0, column=0, columnspan=2, sticky='ew')
        self.tip_heading = tk.Label(head, text='等待 PLC 100/101', bg=COLORS['gray'], fg=WHITE, font=self.theme.large, padx=12)
        self.tip_heading.grid(row=0, column=0, padx=4, pady=4)
        self.tip_message = tk.Label(head, text='等待 PLC 100/101', bg=WHITE, fg=INK, font=self.theme.large, wraplength=850, justify='left')
        self.tip_message.grid(row=0, column=1, columnspan=4, padx=8, sticky='w')
        self.tip_buttons = {}
        for column, (text, name) in enumerate((('确认移动','tooltip_confirm_move'), ('确认对刀结果','tooltip_confirm_result'), ('识别不准，手动点击','tooltip_click'), ('放弃对刀','tooltip_abort'))):
            button = ttk.Button(head, text=text, style='OffsetPlace.TButton' if name in ('tooltip_confirm_move','tooltip_confirm_result') else 'TButton', command=lambda n=name:self.tip_command(n))
            button.grid(row=1, column=column, padx=3, pady=2, sticky='w'); self.tip_buttons[name] = button
        self.tip_pending = 0; self.tip_ack = 0
        self.picture(self.card(self.tip, '主相机 · 点击放大', 1, 0), 'calib_front_latest')
        self.picture(self.card(self.tip, '辅助相机 · 点击放大', 1, 1), 'calib_side_latest')
        iterations = self.card(self.tip, '本次会话迭代', 2, 0)
        cols = ('轮次', '阶段', '主/辅针尖像素', '偏差 mm', 'Δxyz', '下发坐标', '点击', '确认时间')
        self.tiptable = self.table_widget(iterations, cols, (55, 55, 180, 295, 220, 225, 55, 105), stretch=('主/辅针尖像素', '偏差 mm', 'Δxyz', '下发坐标'))
        ttk.Style(self.root).configure('Tip.Treeview', font=self.theme.body)
        ttk.Style(self.root).configure('Tip.Treeview.Heading', font=self.theme.body)
        self.tiptable.configure(style='Tip.Treeview')
        tip_scroll = ttk.Scrollbar(iterations, orient='horizontal', command=lambda *args: self.tiptable.xview(*args))
        tip_scroll.grid(row=1, column=0, sticky='ew'); self.tiptable.configure(xscrollcommand=tip_scroll.set)
        result = self.card(self.tip, '本次计算与补偿', 2, 1)
        self.tip_details = tk.Label(result, text='等待 PLC 请求', bg=WHITE, fg=INK, anchor='nw', justify='left', font=self.theme.body, wraplength=620)
        self.tip_details.grid(row=0, column=0, sticky='nsew')

    def tip_command(self, name):
        self._send_page_command('tip', name)

    def build_placeholder(self, parent, title):
        parent.columnconfigure(0, weight=1); parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)
        for col, label in enumerate(('图像窗', '步骤列表')):
            body = self.card(parent, title + ' · ' + label, 0, col)
            tk.Label(body, text='流程待定', bg=WHITE, fg=MUTED, font=self.theme.title).grid(row=0, column=0)

    def show_status(self, name):
        top = tk.Toplevel(self.root); top.title(name + ' 状态详情'); modal_window(top, self.root)
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
        top = tk.Toplevel(self.root); top.title('异常详情'); modal_window(top, self.root)
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

    def reply_detail(self, _event=None):
        selected = self.reply_table.selection()
        if not selected: return
        values = self.reply_table.item(selected[0], 'values')
        if not values: return
        top = tk.Toplevel(self.root); top.title('应答详情'); modal_window(top, self.root)
        top.rowconfigure(0, weight=1); top.columnconfigure(0, weight=1)
        text = tk.Text(top, width=80, height=16, wrap='word', font=self.theme.mono)
        text.grid(row=0, column=0, sticky='nsew')
        scroll = ttk.Scrollbar(top, orient='vertical', command=text.yview)
        scroll.grid(row=0, column=1, sticky='ns'); text.configure(yscrollcommand=scroll.set)
        text.insert('1.0', '\n'.join(f'{name}：{value}' for name, value in zip(('时间', 'Task_ID', '功能码', '原因'), values)))
        text.config(state='disabled')

    def reply_hover_close(self, _event=None):
        tip = getattr(self, '_reply_tip', None)
        if tip is not None: tip.destroy()
        self._reply_tip = None; self._reply_tip_row = None

    def reply_hover(self, event):
        row = self.reply_table.identify_row(event.y)
        if not row or self.reply_table.identify_column(event.x) != '#4':
            self.reply_hover_close(); return
        if row == getattr(self, '_reply_tip_row', None): return
        self.reply_hover_close()
        values = self.reply_table.item(row, 'values')
        if len(values) < 4: return
        tip = self._reply_tip = tk.Label(self.root, text=str(values[3]), justify='left',
                                         wraplength=600, bg='#fffbe6', relief='solid', borderwidth=1)
        self._reply_tip_row = row
        tip.place(x=event.x_root - self.root.winfo_rootx() + 12,
                  y=event.y_root - self.root.winfo_rooty() + 16)
        tip.lift()

    def open_image(self, path):
        try:
            widget = Picture(self.root, Path(path), self.theme)
            with Image.open(path) as source: widget.image = source.copy()
            widget.enlarge(); widget.destroy()
        except OSError:
            pass

    def file_data(self, path):
        try:
            stamp = path.stat().st_mtime_ns
            if path not in self.files or self.files[path][0] != stamp:
                value = json.loads(path.read_text(encoding='utf-8'))
                if not isinstance(value, dict):
                    raise ValueError('状态必须是字典')
                self.files[path] = (stamp, value)
        except (OSError, ValueError, UnicodeError):
            pass
        return self.files.get(path, (None, {}))[1]

    @staticmethod
    def val(data, *keys, default='—'):
        return next((data[key] for key in keys if data.get(key) is not None), default)

    def toggle_film_mode(self):
        from nest_clear.film_mode import may_change_mode, read_mode, write_mode, mode_lock, tray_active
        from nest_clear.record.events import EventRecorder
        supervisor = read_json(supervisor_state_file(self.directory))
        main = read_json(main_status_file(self.directory))
        try:
            with mode_lock(self.directory):
                supervisor = read_json(supervisor_state_file(self.directory))
                main = read_json(main_status_file(self.directory))
                busy = tray_active(self.directory, supervisor.get('child_pid'))
                if not may_change_mode(supervisor, main, busy):
                    messagebox.showwarning('无法切换', '请等待本盘或当前任务结束后再切换。', parent=self.root)
                    return
                previous = read_mode(self.directory)
                write_mode(self.directory, not previous)
            EventRecorder(self.directory, process='ui').emit('film_mode_change', previous_enabled=previous, enabled=not previous)
        except (OSError, ValueError) as exc:
            messagebox.showerror('切换失败', str(exc), parent=self.root)
        self.refresh()

    def refresh(self):
        supervisor = self.file_data(supervisor_state_file(self.directory))
        main = self.file_data(main_status_file(self.directory))
        from nest_clear.film_mode import may_change_mode, read_mode, tray_active
        try:
            enabled = read_mode(self.directory)
        except (OSError, ValueError):
            enabled = False
        if enabled:
            self.film_banner.grid(row=0, column=0, sticky='ew', padx=6, pady=(4, 0))
        else:
            self.film_banner.grid_remove()
        busy = tray_active(self.directory, supervisor.get('child_pid'))
        allowed = may_change_mode(supervisor, main, busy) and not busy
        self.film_button.config(text='主流程测试模式（菲林片）：' + ('开' if enabled else '关'),
                                fg='#b00020' if enabled else '#222222',
                                disabledforeground='#b00020' if enabled else '#777777',
                                state='normal' if allowed else 'disabled')
        if allowed:
            reason = ''
        elif busy:
            reason = '挑毛进行中，本盘结束后可切换'
        elif supervisor.get('child_pid') and main.get('mode'):
            reason = '主程序正在执行其他任务，空闲后可切换'
        else:
            reason = '主程序状态未就绪或正在运行，空闲后可切换'
        self.film_reason.config(text=reason)
        mode = main.get('mode')
        if mode != self.last_mode:
            self.last_mode = mode
            page = page_for_mode(main)
            if page is not None: self.select_page(page)
        try:
            supervisor_age = max(0, time.time() - supervisor_state_file(self.directory).stat().st_mtime)
        except OSError:
            supervisor_age = None
        for name, (color, value) in status_indicators(supervisor, main, supervisor_age=supervisor_age).items():
            dot, label = self.lights[name]; dot.config(fg=COLORS[color]); label.config(text=str(value))
        for label, item in zip(self.bits, decode_status(supervisor.get('word'))):
            label.config(text=f"● {item['name']}", fg=COLORS[item['color']])
        self.process_table.item('parent', values=parent_process_row(supervisor, supervisor_age))
        self.process_table.item('child', values=child_process_row(supervisor, supervisor_age))
        picking = main.get('picking') if isinstance(main.get('picking'), dict) else {}
        view = self.val(picking, 'view_index', 'view')
        if isinstance(view, int) and 'view_index' in picking: view += 1
        pick_fields = {'盘号': self.val(picking, 'tray_id', 'tray'), '视野': f"{view}/{self.val(picking, 'total_views', 'total')}",
                       '粗拍次数': self.val(picking, 'coarse_shots'), '可去数量': self.val(picking, 'removable_count', 'n0'),
                       '红框数量': self.val(picking, 'red_count'), '已下发': self.val(picking, 'sent_count', 'attempts'),
                       '不可去': self.val(picking, 'non_pickable_count'), '失败': self.val(picking, 'failed_count')}
        for name, value in pick_fields.items(): self.pick_values[name].config(text=str(value))
        rows = self.tail.poll()
        self.update_reply(main, rows)
        self.update_cam(camcal_progress(main))
        self.update_preview(supervisor, main)
        self.tip_ack = main.get('command_seq', 0)
        self._check_page_command('tip', self.tip_ack)
        tip_data = tooltip_progress(main)
        self.update_tip(tip_data)
        self.update_offset(main)
        self.update_pickacc(main)
        from nest_clear.ui.model import supervisor_tooltip_main
        tooltip = self.file_data(self.directory / 'state' / 'supervisor_tooltip_status.json')
        self.update_baseline(supervisor_tooltip_main(main, supervisor, tooltip, supervisor_age=supervisor_age), supervisor)
        key = (date.today(), json.dumps(rows, ensure_ascii=False, default=str))
        if key != self.cache.get('events'):
            self.events = list(rows); self.table.delete(*self.table.get_children())
            for i, row in enumerate(rows):
                values = [display_time(row.get('time'))] + [row.get(k) for k in ('category', 'task_id', 'func', 'phenomenon', 'location', 'suggestion')]
                self.table.insert('', 'end', iid=str(i), values=values, tags=(f"{row['color']}_{'odd' if i % 2 else 'even'}",))
            self.cache['events'] = key
        self.root.after(self.interval, self.refresh)

    def poll_pictures(self):
        for picture in self.pictures:
            picture.poll()
        self.root.after(50, self.poll_pictures)

    def update_reply(self, main, rows):
        replies = main.get('recent_replies')
        if replies is None:
            replies = [{'time': main.get('last_reply_time'), 'task_id': main.get('last_reply_task_id'),
                        'request_func': main.get('last_reply_request_func', '—'), 'reply_func': main.get('last_reply_func')}]
        values = []
        for reply in replies[:5]:
            code = reply.get('reply_func')
            if code is None: continue
            task = reply.get('task_id')
            matching = next((r for r in rows if r.get('task_id') == task and r.get('func') == code), None)
            reason = matching['phenomenon'] if matching else ('正常应答' if code not in (999, 131, 82) else '原因无数据')
            values.append((display_time(reply.get('time')), task, f"{reply.get('request_func', '—')} → {code}", reason))
        key = tuple(values)
        if key == self.cache.get('replies'): return
        self.reply_table.delete(*self.reply_table.get_children())
        for value in values: self.reply_table.insert('', 'end', values=value)
        self.cache['replies'] = key

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
        step = '前往起始位' if data.get('event') == 'camcal_goto_start' else {'move': '移动拍照', 'done': '完成', 'fail': '失败'}.get(kind, kind)
        code = data.get('code')
        self.cam_step.set({'步骤': f'{step}  {code}' if code else step, '检出角点': f"{detail.get('detected_corners', '—')}/{detail.get('predicted_corners', '—')}", '饱和比例': saturation, '亮斑': glare})

        failed = data.get('event') in ('camcal_failed', 'camcal_aborted') or data.get('kind') == 'fail'
        done = data.get('event') == 'camcal_done' or data.get('kind') == 'done'
        if failed != getattr(self, '_cam_failed', False):
            self.cam_result_card.grid_configure(row=1 if failed else 4)
            self.cam_plan_card.grid_configure(row=4 if failed else 1)
            if failed: self.cam_viewport.yview_moveto(0)
            self._cam_failed = failed
        if failed:
            reason = data.get('reason') or '标定失败'
            hint = data.get('hint') or ('请' + reason.rsplit('请', 1)[1] if '请' in reason else '检查标定板、光照和相机连接。')
            text = f"失败：{data.get('code') or 'CAMCAL-FAILED'}\n{reason}\n建议：{hint}"
            color, background = COLORS['red'], '#fff1f0'
        elif done:
            report = data.get('report') if isinstance(data.get('report'), dict) else detail
            text = (f"标定成功  RMS：{report.get('rms_px', '—')} px\n"
                    f"中心重拍：{report.get('repeat_shift_mm', '—')} mm / 上限 {report.get('repeat_limit_mm', '—')} mm\n"
                    f"主点尺度：{report.get('um_per_px_at_principal', '—')} µm/px  "
                    f"轴比例：{report.get('axis_scale', '—')}\n倾角：{report.get('camera_tilt_deg', '—')}°\n"
                    f"文件：{data.get('calibration_path') or '—'}  备份：{data.get('backup_path') or '—'}")
            color, background = COLORS['green'], '#ecfdf3'
        else:
            text = '标定进行中' if data else '无数据'
            color, background = COLORS['green'], '#ecfdf3'
        if not failed:
            text = f'当前步骤：{step}' + (f'  {code}' if code else '') + '\n' + text
        if (text, color, background) != getattr(self, '_cam_result_cache', None):
            self.cam_result.config(state='normal', bg=background, fg=color)
            self.cam_result.delete('1.0', 'end'); self.cam_result.insert('1.0', text)
            self.cam_result.config(state='disabled')
            self._cam_result_cache = (text, color, background)
        start = data.get('start_xyz')
        origin_z = start[2] if isinstance(start, (list, tuple)) and len(start) == 3 else None
        self.draw_plan(points, count, origin_z)

    def toggle_preview(self, mode="preview"):
        path = ui_command_file(self.directory)
        try:
            old = read_json(path)
            seq = max(time.time_ns(), int(old.get('seq', 0)) + 1, int(self.preview_data.get('seq', 0)) + 1)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix('.tmp')
            tmp.write_text(json.dumps({'preview': not self.preview_data.get('on', False), 'mode': mode, 'seq': seq,
                                       'time': datetime.now().astimezone().isoformat()}), encoding='utf-8')
            os.replace(tmp, path)
        except (OSError, ValueError, TypeError):
            logger.exception('写预览命令失败')

    def update_preview(self, supervisor, main=None):
        try:
            age = max(0, time.time() - supervisor_state_file(self.directory).stat().st_mtime)
        except OSError:
            age = None
        controls = preview_controls(supervisor, age, (main or {}).get('mode'))
        self.preview_data = supervisor.get('preview') if isinstance(supervisor.get('preview'), dict) else {}
        self.preview_button.config(text='关闭相机预览' if controls['on'] and controls['mode'] == 'preview' else '打开相机预览',
                                   state='normal' if controls['preview_enabled'] else 'disabled')
        focus_on = controls['on'] and controls['mode'] == 'focus'
        self.focus_button.config(text='关闭对焦/光圈辅助' if focus_on else '对焦/光圈辅助', state='normal' if controls['focus_enabled'] else 'disabled')
        if focus_on != self.focus_was_on:
            if focus_on:
                self.focus_history.clear(); self.focus_last_sample = None
                self.exposure_cache = None
                self.close_exposure_results()
                self.focus_table.delete(*self.focus_table.get_children())
                for card in self.cam_other_cards: card.grid_remove()
                self.progress_text.grid_remove(); self.progress.grid_remove()
                for row in range(5): self.cam_right.rowconfigure(row, weight=1 if row == 1 else 0)
                self.focus_panel.grid(); self.focus_panel.tkraise()
            else:
                self.focus_panel.grid_remove()
                for card in self.cam_other_cards: card.grid()
                self.progress_text.grid(); self.progress.grid()
                for row, weight in enumerate((3, 3, 3, 5, 2)): self.cam_right.rowconfigure(row, weight=weight)
        if focus_on:
            self.update_focus(self.preview_data)
            if not controls['white_balance_enabled']:
                self.wb_button.config(state='disabled')
        self.focus_was_on = focus_on
        text = ('不可用：' + controls['reason'] if controls['reason'] else
                f"预览中 {self.preview_data.get('fps', 0)} fps" if controls['on'] else
                '不可用：' + str(self.preview_data['error']) if self.preview_data.get('error') else '已关闭')
        self.preview_status.config(text=text)
        path = live_image_file(self.directory, 'preview_latest' if controls['on'] else 'camcal_latest')
        if getattr(self.cam_picture, 'path', None) != path:
            self.cam_picture.path = path
            self.cam_picture.mtime = None
            self.cam_picture.image = self.cam_picture.photo = None
            self.cam_picture.config(image='', text='暂无图像')
        p = self.preview_data
        def vector(value):
            if not isinstance(value, (list, tuple)): return '—'
            try: return '[' + ', '.join(f'{float(v):.3f}' for v in value) + ']'
            except (TypeError, ValueError): return str(value)
        self.preview_values.set({'模组 XYZ': vector(p.get('actual_xyz')), '清晰度': p.get('sharpness'),
                           '检出角点': p.get('corners'), '板中心偏移 mm': vector(p.get('board_offset_mm')),
                           '提示': '偏移接近 0 且清晰度最大处即为推荐起始位'})

    def focus_action(self, action, **fields):
        path = self.directory / 'state' / 'focus_action.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps({'seq': time.time_ns(), 'action': action, **fields}), encoding='utf-8')
        os.replace(tmp, path)
        return json.loads(path.read_text(encoding='utf-8'))['seq']

    def change_temporary_exposure(self, restore):
        self.temporary_pending = True
        self.temporary_timeout_message = ''
        self.temporary_pending_since = time.monotonic()
        self.temporary_set.config(state='disabled'); self.temporary_restore.config(state='disabled')
        self.temporary_message.config(text='设置中…')
        self.temporary_request = self.focus_action('restore_exposure' if restore else 'set_exposure', exposure_us=self.temporary_entry.get())

    def _expire_focus_request(self, name, label):
        if getattr(self, name + '_pending', False) and time.monotonic() - getattr(self, name + '_pending_since', time.monotonic()) >= 5:
            setattr(self, name + '_pending', False)
            message = '上一条命令未被确认（主程序/父程序可能未运行），已放弃，可重新操作'
            logger.warning('%s：%s', name, message)
            setattr(self, name + '_timeout_message', message)
            label.config(text=message, fg=COLORS['red'])
            return True
        return False

    def update_temporary_exposure(self, p):
        result = p.get('temporary_exposure') or {}
        if result.get('request_seq') == getattr(self, 'temporary_request', None) and result.get('state'):
            self.temporary_pending = False
            self.temporary_timeout_message = ''
        expired = self._expire_focus_request('temporary', self.temporary_message)
        busy = (p.get('white_balance') or {}).get('state') == 'running' or (p.get('exposure_sweep') or {}).get('state') == 'running' or self.wb_pending or self.sweep_pending or self.temporary_pending
        self.temporary_set.config(state='disabled' if busy else 'normal')
        self.temporary_restore.config(state='disabled' if busy else 'normal')
        if result.get('state') and not self.temporary_pending and not expired and not getattr(self, 'temporary_timeout_message', ''):
            self.temporary_message.config(text=result.get('message', ''), fg=COLORS['red'] if result['state'] == 'error' else INK)

    def start_white_balance(self):
        from tkinter import messagebox
        if messagebox.askokcancel('白平衡', '请在视野中央放置 18% 灰卡并保持光源与标定/挑毛时一致', parent=self.root):
            self.wb_pending = True
            self.wb_timeout_message = ''
            self.wb_pending_since = time.monotonic()
            self.wb_button.config(state='disabled', text='白平衡中…')
            self.temporary_set.config(state='disabled'); self.temporary_restore.config(state='disabled')
            self.wb_message.config(text='白平衡中…', fg=INK)
            self.wb_request = self.focus_action('white_balance')

    def update_white_balance(self, p):
        wb = p.get('white_balance') or {}
        state = wb.get('state')
        if wb.get('request_seq') == getattr(self, 'wb_request', None) and state:
            self.wb_pending = False
            self.wb_timeout_message = ''
        expired = self._expire_focus_request('wb', self.wb_message)
        running = state == 'running' or self.wb_pending
        sweep_running = (p.get('exposure_sweep') or {}).get('state') == 'running'
        self.wb_button.config(state='disabled' if running or sweep_running else 'normal', text='白平衡中…' if running else '白平衡')
        if expired or getattr(self, 'wb_timeout_message', ''):
            return
        if running:
            self.wb_message.config(text='白平衡中…', fg=INK)
        elif state == 'success':
            r, g, b = wb['balance_ratio']; m = wb['metrics']
            message = f"白平衡成功 R={r:.3f} G={g:.3f} B={b:.3f}，偏色 R/G {m['r_g']:.3f} B/G {m['b_g']:.3f}，已保存，主程序下次启动生效"
            warnings = m.get('warnings') or []
            self.wb_message.config(text=message + ('\n警告：' + '；'.join(warnings) if warnings else ''), fg='#b54708' if warnings else '#168448')
        elif state == 'fail':
            self.wb_message.config(text='白平衡失败：' + wb.get('reason', ''), fg=COLORS['red'])

    def toggle_exposure_sweep(self):
        sweep = self.preview_data.get('exposure_sweep') or {}
        if sweep.get('state') == 'running':
            self.focus_action('cancel_sweep')
        else:
            mode = 'brightness' if self.exposure_mode.get() == '亮度目标' else 'calibration'
            self.exposure_cache = None
            self.sweep_pending = True
            self.sweep_timeout_message = ''
            self.sweep_pending_since = time.monotonic()
            self.temporary_set.config(state='disabled'); self.temporary_restore.config(state='disabled')
            self.sweep_request = self.focus_action('exposure_sweep', mode=mode)
            self.exposure_message.config(text='正在启动曝光自测…')
            self.show_exposure_results()

    def show_exposure_results(self):
        self.exposure_visible = True
        self.focus_curve.grid_remove(); self.focus_record_title.grid_remove(); self.focus_table.grid_remove()
        self.exposure_panel.grid(); self.exposure_panel.tkraise()

    def close_exposure_results(self):
        self.exposure_visible = False
        self.exposure_panel.grid_remove()
        self.focus_curve.grid(); self.focus_record_title.grid(); self.focus_table.grid()

    def apply_exposure(self):
        sweep = self.preview_data.get('exposure_sweep') or {}
        recommendation = sweep.get('recommendation')
        if sweep.get('state') != 'done' or not recommendation:
            return
        old, new = sweep.get('original_us'), recommendation['exposure_us']
        if messagebox.askyesno('应用推荐曝光', f'当前曝光 {old} us → 推荐曝光 {new:.1f} us\n写入覆盖文件，主程序下次启动生效。确定应用？', parent=self.root):
            self.focus_action('apply_exposure')
            self.exposure_message.config(text='正在保存曝光覆盖文件…')

    def update_exposure(self, p):
        sweep = p.get('exposure_sweep') or {}
        running = sweep.get('state') == 'running'
        if sweep.get('request_seq') == getattr(self, 'sweep_request', None) and sweep.get('state'):
            self.sweep_pending = False
            self.sweep_timeout_message = ''
        expired = self._expire_focus_request('sweep', self.exposure_message)
        self.exposure_button.config(text='取消' if running else '曝光自测', state='disabled' if (p.get('white_balance') or {}).get('state') == 'running' or self.wb_pending else 'normal')
        self.exposure_mode.config(state='disabled' if running else 'readonly')
        if (running or (p.get('demo_exposure') and self.exposure_cache is None)) and not self.exposure_visible: self.show_exposure_results()
        if expired or getattr(self, 'sweep_timeout_message', '') or not sweep: return
        key = json.dumps([sweep, p.get('exposure_apply')], ensure_ascii=False, sort_keys=True)
        if key == self.exposure_cache: return
        self.exposure_cache = key
        self.exposure_table.delete(*self.exposure_table.get_children())
        recommendation = sweep.get('recommendation') or {}
        for row in sweep.get('rows', []):
            selected = row['exposure_us'] == recommendation.get('exposure_us')
            self.exposure_table.insert('', 'end', values=(f"{row['exposure_us']:.1f}", f"{row['saturated_fraction']*100:.2f}", row['p99'], row['corners'], f"{row['sharpness']:.3f}"), tags=('recommended',) if selected else ())
        self.exposure_apply.config(state='normal' if sweep.get('state') == 'done' and recommendation and not p.get('demo_exposure') else 'disabled')
        reason = sweep.get('reason') or ''
        status = f"第 {sweep.get('index', 0)}/{sweep.get('total', 0)} 档" if running else '扫描完成' if sweep.get('state') == 'done' else '已取消' if sweep.get('state') == 'cancelled' else '自测失败'
        old = sweep.get('original_us')
        suggested = recommendation.get('exposure_us')
        configured = sweep.get('configured_us', old)
        text = f"{status}  当前配置曝光 {configured if configured is not None else '未指定'} us"
        if configured is None and old is not None: text += f"（相机回读 {old:.1f} us）"
        if suggested is not None: text += f"  推荐 {suggested:.1f} us"
        if p.get('demo_exposure'): text += '\n演示结果（不可应用）'
        if reason: text += '\n' + reason
        applied = p.get('exposure_apply') or {}
        if applied.get('message'): text += '\n' + applied['message']
        self.exposure_message.config(text=text, fg=COLORS['red'] if reason or applied.get('state') == 'error' else INK)

    def reset_focus(self):
        path = self.directory / 'state' / 'focus_action.json'
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix('.tmp')
            tmp.write_text(json.dumps({'seq': time.time_ns(), 'action': 'reset'}), encoding='utf-8')
            os.replace(tmp, path)
        except OSError:
            logger.exception('重置峰值失败')

    def record_focus(self):
        p = self.preview_data
        if not p.get('scores') or p.get('mode') != 'focus' or not p.get('on'):
            return
        try:
            from nest_clear.preview.focus_record import write_record
            record = write_record(self.directory, p, live_image_file(self.directory, 'preview_latest').read_bytes(), self.focus_note.get())
            self.focus_table.insert('', 'end', values=(record['time'][11:19], record['note'], f"{p['center']:.4f}", f"{p['median']:.4f}", f"{p['mean']:.1f}", f"{p['saturated_pct']:.2f}"))
        except OSError:
            logger.exception('记录对焦数据失败')

    def update_focus(self, p):
        if not p.get('scores'):
            return
        pct = p['percent'][4]
        color = '#168448' if pct >= 98 else '#916300' if pct >= 90 else '#b42318'
        self.focus_percent.config(text=f'{pct:.0f}%', fg=color)
        self.focus_summary.config(text=f"中心 {p['center']:.4f} / 峰值 {p['peaks'][4]:.4f}\n九格中位 {p['median']:.4f}   Laplacian {p['laplacian'][4]:.1f}")
        self.focus_bar.delete('all'); w = max(self.focus_bar.winfo_width(), 100)
        self.focus_bar.create_rectangle(0, 0, w * min(pct, 100) / 100, 16, fill=color, outline='')
        self.update_exposure(p)
        self.update_white_balance(p)
        self.update_temporary_exposure(p)
        self.focus_details.config(text=(f"曝光 {p.get('exposure_us')} us（{p.get('exposure_source', '配置')}）  增益 {p.get('gain_db')} dB  白平衡 {p.get('white_balance_source', '配置')}\n"
            f"均值 {p['mean']:.1f}  P1/P99 {p['p1']:.0f}/{p['p99']:.0f}  饱和 {p['saturated_pct']:.2f}%  过暗 {p['dark_pct']:.2f}%"))
        self.focus_alert.config(text='  '.join(x for x in (p.get('exposure_hint'), p.get('tilt')) if x),
                                fg=COLORS['red'] if p.get('saturated_pct', 0) > .5 else '#b54708')
        for i, (score, peak, percent, mean) in enumerate(zip(p['scores'], p['peaks'], p['percent'], p['cell_means'])):
            fg = '#168448' if percent >= 98 else '#916300' if percent >= 90 else '#b42318'
            self.focus_score_cells[i].config(text=f'{score:.2f}/{peak:.2f} {percent:.0f}%', fg=fg)
            self.focus_mean_cells[i].config(text=f'{mean:.0f}')
        stamp = p.get('analysis_time')
        # 演示数据带有实际分析生成的历史；真实相机会逐帧累积。
        for t, value in p.get('demo_history', []):
            if not self.focus_history or t > self.focus_history[-1][0]:
                self.focus_history.append((t, value))
        if stamp and stamp != self.focus_last_sample and (not self.focus_history or stamp > self.focus_history[-1][0]):
            self.focus_history.append((stamp, p['center']))
        self.focus_last_sample = stamp
        now = time.time()
        self.focus_history = [(t,v) for t,v in self.focus_history if 0 <= now-t <= 30]
        c = self.focus_curve; c.delete('all'); w = max(c.winfo_width(), 100); h = max(c.winfo_height(), 120)
        left, right, top, bottom = 52, w-8, 20, h-20
        peak = max(p['peaks'][4], max((v for _, v in self.focus_history), default=0), 1e-9)
        for fraction in (0, .5, 1):
            y = bottom - (bottom-top)*fraction
            c.create_line(left, y, right, y, fill='#cbd5e1', dash=(3, 3))
            c.create_text(left-4, y, text=f'{peak*fraction:.2f}', anchor='e', fill='#475569', font=self.theme.table)
        c.create_line(left, top, right, top, fill='#b54708', dash=(6, 3))
        c.create_text(right, top, text='峰值', anchor='ne', fill='#b54708', font=self.theme.table)
        if len(self.focus_history) > 1:
            coords = []
            for t, value in self.focus_history:
                coords.extend((left + (right-left) * (1-(now-t)/30), bottom-(bottom-top)*value/peak))
            c.create_line(*coords, fill='#2563eb', width=2)
        c.create_text(left, 2, text='中心清晰度 · 最近 30 秒', anchor='nw', fill='#334155', font=self.theme.table)
        c.create_text(left, h-2, text='−30 秒', anchor='sw', fill='#475569', font=self.theme.table)
        c.create_text(right, h-2, text='现在', anchor='se', fill='#475569', font=self.theme.table)

    def update_tip(self, data):
        key = json.dumps([data, self.tip_pending, getattr(self, 'tip_command_message', '')], ensure_ascii=False, default=str)
        if key == self.cache.get('tip'): return
        self.cache['tip'] = key
        self.tiptable.delete(*self.tiptable.get_children())
        for row in data.get('rounds', []):
            self.tiptable.insert('', 'end', values=(row.get('iteration') or '—', row.get('stage') or '—', tooltip_pixels(row.get('pixels')), tooltip_deviation(row.get('deviation_mm'), compact=True), tooltip_vector(row.get('delta'), signed=True, compact=True), tooltip_vector(row.get('target'), compact=True), '是' if row.get('click') else '否', tooltip_time(row.get('confirmed_at'))))
        phase = data.get('phase', 'idle'); tool = data.get('tool')
        self.tip_heading.config(text=f'当前：工具{tool}' if tool else '当前：未选择', bg='#2563eb' if tool == 1 else '#d56713' if tool == 2 else COLORS['gray'])
        warning = data.get('camera_settings_warning')
        self.tip_message.config(text=getattr(self, 'tip_command_message', '') or ('上一条命令处理中' if self.tip_pending else warning or data.get('message')) or '等待 PLC 100/101', fg=COLORS['red'] if getattr(self, 'tip_command_message', '') else COLORS['yellow'] if warning else COLORS['red'] if phase in ('limit','fail') else COLORS['green'] if phase == 'done' else INK)
        for name, button in self.tip_buttons.items():
            enabled = (name == 'tooltip_confirm_move' and phase in ('probe','confirm_move') or name == 'tooltip_confirm_result' and phase == 'in_range' or name in ('tooltip_click','tooltip_abort') and phase in ('probe','confirm_move','in_range','limit','click_failed'))
            button.config(state='normal' if enabled and not self.tip_pending else 'disabled')
        rounds = data.get('rounds') or []
        waiting = phase in ('starting', 'probe', 'confirm_move', 'in_range', 'limit', 'clicking', 'click_failed')
        number = len(rounds) + int(waiting) if rounds or waiting else None
        xyz = lambda value, signed=False: tooltip_vector(value, signed=signed, labels='XYZ')
        heading = (f'第 {number or len(rounds) + 1} 轮（已失败）' if phase == 'fail'
                   else f'第 {number} 轮' if number else '等待会话')
        self.tip_details.config(text=(heading + '\n') +
            f"{tooltip_deviation(data.get('deviation_mm'))}\n"
            f"Δ {xyz(data.get('delta'), True)} mm　目标 {xyz(data.get('target'))}\n"
            f"累计相对 P0：{xyz(data.get('cumulative'), True)} mm　手动点击：{'是' if data.get('click') is True else '否' if data.get('click') is False else '—'}\n"
            f"雅可比：{tooltip_quality(data.get('jacobian_status'), data.get('jacobian_quality'))}\n"
            f"旧补偿：{xyz(data.get('old_compensation'), True)} mm\n"
            f"新补偿：{xyz(data.get('new_compensation'), True)} mm\n"
            f"补偿差值：{xyz(data.get('compensation_difference'), True)} mm")


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
