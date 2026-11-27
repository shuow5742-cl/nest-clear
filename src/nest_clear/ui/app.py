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

from nest_clear.depthcal.exposure import validate_depthcal_exposure
from nest_clear.paths import live_image_file, main_status_file, supervisor_state_file, ui_command_file, tooltip_preview_file
from .model import (EventTail, camcal_progress, decode_status, display_time, duration,
                    read_json, status_indicators, parent_process_row, child_process_row, tooltip_progress, preview_controls, edge_corner_label,
                    pickacc_tray_geometry, pickacc_type, pickacc_z_type, pickacc_z_summary_text, pickacc_z_missing_tools,
                    pickacc_stage, pickacc_note, pickacc_infield,
                    tooltip_vector, tooltip_deviation, tooltip_pixels, tooltip_quality, tooltip_time,
                    page_for_mode, depthcal_session_key, next_auto_page,
                    baseline_preview_display, extract_tooltip_preview,
                    depthcal_stage_info, depthcal_feasibility_text, depthcal_fine_plan_text, depthcal_subplot_layout,
                    depthcal_curve_dof, depthcal_latest_dof_text, depthcal_dof_summary_text,
                    depthcal_point_arrival_text, depthcal_arrival_summary_text, depthcal_exposure_display)

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


def bind_text_tooltip(widget, source):
    """Expose the full status text when its row has limited space."""
    timer = tip = None

    def show():
        nonlocal timer, tip
        timer = None
        text = source.cget('text')
        if not text:
            return
        tip = tk.Toplevel(widget)
        tip.overrideredirect(True)
        tk.Label(tip, text=text, bg=WHITE, fg=INK, font=source.cget('font'),
                 relief='solid', borderwidth=1, padx=6, pady=4, wraplength=480,
                 justify='left').pack()
        tip.update_idletasks()
        x = min(widget.winfo_rootx(), widget.winfo_screenwidth() - tip.winfo_reqwidth())
        y = widget.winfo_rooty() + widget.winfo_height() + 3
        if y + tip.winfo_reqheight() > widget.winfo_screenheight():
            y = widget.winfo_rooty() - tip.winfo_reqheight() - 3
        tip.geometry(f'+{max(0, x)}+{max(0, y)}')

    def enter(_event):
        nonlocal timer
        timer = widget.after(450, show)

    def leave(_event):
        nonlocal timer, tip
        if timer is not None:
            widget.after_cancel(timer)
            timer = None
        if tip is not None:
            tip.destroy()
            tip = None

    widget.bind('<Enter>', enter)
    widget.bind('<Leave>', leave)
    widget.bind('<Destroy>', leave)


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
    PAGE_NAMES = ('主页', '相机标定', '工具偏移标定', '挑取精度标定', '对刀基准建立', '换针对刀', '下针深度标定')
    PAGE_ALIASES = {'home': '主页', 'main': '主页', 'camera': '相机标定', 'camcal': '相机标定',
                    'offset': '工具偏移标定', 'tool-offset': '工具偏移标定',
                    'pickacc': '挑取精度标定', 'pick-accuracy': '挑取精度标定',
                    'baseline': '对刀基准建立', 'toolcal': '换针对刀', 'tooltip': '换针对刀',
                    'depth': '下针深度标定', 'depthcal': '下针深度标定', 'depth-calib': '下针深度标定'}

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
        self.pick_z_banner = tk.Label(self.root, text='', bg='#fef0c7', fg='#b42318', font=self.theme.title, anchor='center', padx=8, pady=4)
        self.tail = EventTail(root); self.pictures = []; self.cache = {}; self.events = []; self.files = {}
        initial_status = read_json(main_status_file(self.directory)) if page is not None else {}
        self.last_mode = initial_status.get('mode') if page is not None else None
        self.last_depth_session = depthcal_session_key(initial_status) if page is not None else None
        self.build_status()
        self.tabs = ttk.Notebook(self.root)
        self.tabs.grid(row=2, column=0, sticky='nsew', padx=6, pady=(0, 6))
        self.home = self.new_page('主页'); self.build_home()
        self.cam = self.new_page('相机标定'); self.build_cam()
        self.offset = self.new_page('工具偏移标定'); self.build_offset()
        self.pickacc = self.new_page('挑取精度标定'); self.build_pickacc()
        self.baseline = self.new_page('对刀基准建立'); self.build_baseline()
        self.tip = self.new_page('换针对刀'); self.build_tip()
        self.depth = self.new_page('下针深度标定'); self.build_depth()
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
            shot = ImageGrab.grab(bbox=(x, y, x + width, y + height))
            if hasattr(shot, 'convert'):
                shot = shot.convert('RGB')
            shot.save(target)
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

    def table_widget(self, parent, columns, widths, stretch=(), height=1, anchor='w', heading_anchor=None):
        tree = ttk.Treeview(parent, columns=columns, show='headings', height=height)
        tree.grid(row=0, column=0, sticky='nsew')
        for column, width in zip(columns, widths):
            if heading_anchor is not None:
                tree.heading(column, text=column, anchor=heading_anchor)
            else:
                tree.heading(column, text=column)
            tree.column(column, width=width, minwidth=35, stretch=column in stretch, anchor=anchor)
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
        self.offset_dialog = None
        self.offset_pending = 0
        self.offset.columnconfigure(0, weight=58, uniform='offset_columns')
        self.offset.columnconfigure(1, weight=42, uniform='offset_columns')
        self.offset.rowconfigure(1, weight=1)

        header = self.card(self.offset, '工具偏移标定', 0, 0, columnspan=2)
        header.columnconfigure(0, weight=1)
        heading = tk.Frame(header, bg=WHITE)
        heading.grid(row=0, column=0, sticky='ew')
        heading.columnconfigure(3, weight=1)
        self.offset_heading = tk.Label(heading, text='当前：等待 PLC 指定', bg='#667085', fg=WHITE, font=(self.theme.title[0], 24, 'bold'), anchor='center', padx=16, pady=2)
        self.offset_heading.grid(row=0, column=0, sticky='w', padx=(0, 12))
        self.offset_plc_id = tk.Label(heading, text='40905=—', bg='#eef2f6', fg=INK, font=self.theme.body, padx=10, pady=4)
        self.offset_plc_id.grid(row=0, column=1, sticky='w', padx=(0, 12))
        self.offset_stage = tk.Label(heading, text='阶段：空闲', bg='#eef2f6', fg=INK, font=self.theme.title, padx=12, pady=3)
        self.offset_stage.grid(row=0, column=2, sticky='w')
        self.offset_abort = ttk.Button(heading, text='放弃本次标定', command=lambda: self.offset_command('tooloffset_abort'))
        self.offset_abort.grid(row=0, column=4, sticky='e', padx=4)

        self.offset_message = tk.Label(header, text='等待 PLC 90 (40905=1)', bg=WHITE, fg=INK, font=(self.theme.title[0], 17), anchor='w', justify='left', wraplength=1260)
        self.offset_message.grid(row=1, column=0, sticky='ew', pady=(4, 2))
        self.offset_failure_detail = tk.Label(header, text='', bg='#fff1f0', fg=COLORS['red'], font=self.theme.body, anchor='w', justify='left', wraplength=1260)
        self.offset_failure_detail.grid(row=2, column=0, sticky='ew')
        self.offset_failure_detail.grid_remove()
        self.offset_warning = tk.Label(header, text='', bg=WHITE, fg=COLORS['red'], font=self.theme.body, anchor='w', wraplength=1260)
        self.offset_warning.grid(row=3, column=0, sticky='ew')
        self.offset_warning.grid_remove()
        self.offset_caution = tk.Label(header, text='', bg=WHITE, fg=COLORS['yellow'], font=self.theme.body, anchor='w', wraplength=1260)
        self.offset_caution.grid(row=4, column=0, sticky='ew')
        self.offset_caution.grid_remove()

        left = tk.Frame(self.offset, bg=BG)
        left.grid(row=1, column=0, sticky='nsew')
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1, minsize=140)
        left.rowconfigure(1, weight=2, minsize=250)

        tri_card = self.card(left, '主相机三角定位视角叠加图 · 点击放大', 0, 0)
        for col in range(5):
            tri_card.columnconfigure(col, weight=1, uniform='tri_thumbs')
        tri_card.rowconfigure(0, weight=1)
        tri_card.grid_propagate(False)
        self.offset_tri_pics = []
        self.offset_tri_labels = []
        for idx in range(5):
            slot = tk.Frame(tri_card, bg=WHITE)
            slot.grid(row=0, column=idx, sticky='nsew', padx=3)
            slot.columnconfigure(0, weight=1)
            slot.rowconfigure(0, weight=1)
            pic = self.picture(slot, f'tooloffset_tri_{idx + 1}')
            caption = tk.Label(slot, text=f'视角 {idx + 1}：—', bg=WHITE, fg=MUTED, font=(self.theme.body[0], 9), anchor='center')
            caption.grid(row=1, column=0, sticky='ew')
            self.offset_tri_pics.append(pic)
            self.offset_tri_labels.append(caption)

        side_card = self.card(left, '对刀相机实时叠加（工具尖 + 标定尖 + 过工具尖 Z 方向线）', 1, 0)
        side_card.columnconfigure(0, weight=1, uniform='offset_side')
        side_card.columnconfigure(1, weight=1, uniform='offset_side')
        side_card.rowconfigure(0, weight=1)
        side_card.grid_propagate(False)
        front_box = tk.Frame(side_card, bg=WHITE)
        front_box.grid(row=0, column=0, sticky='nsew', padx=(0, 4))
        front_box.columnconfigure(0, weight=1)
        front_box.rowconfigure(0, weight=1)
        side_box = tk.Frame(side_card, bg=WHITE)
        side_box.grid(row=0, column=1, sticky='nsew', padx=(4, 0))
        side_box.columnconfigure(0, weight=1)
        side_box.rowconfigure(0, weight=1)
        self.offset_front_pic = self.picture(front_box, 'tooloffset_front_latest')
        self.offset_side_pic = self.picture(side_box, 'tooloffset_side_latest')
        tk.Label(front_box, text='正面相机 (X / Z)', bg=WHITE, fg=MUTED, font=(self.theme.body[0], 10)).grid(row=1, column=0, sticky='ew')
        tk.Label(side_box, text='侧面相机 (Y / Z)', bg=WHITE, fg=MUTED, font=(self.theme.body[0], 10)).grid(row=1, column=0, sticky='ew')

        right = tk.Frame(self.offset, bg=BG)
        right.grid(row=1, column=1, sticky='nsew')
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=4)
        right.rowconfigure(1, weight=5)
        right.rowconfigure(2, weight=5)

        tri_info = self.card(right, '三角定位三维结果', 0, 0)
        self.offset_tri_grid = KeyValueGrid(
            tri_info,
            ('标定尖 XYZ', '目标机台 XY', '视角 / 夹角', '射线最大残差'),
            self.theme,
        )
        self.offset_tri_grid.grid(row=0, column=0, sticky='nsew')

        align_info = self.card(right, '对刀姿态、对准与防撞', 1, 0)
        self.offset_align_grid = KeyValueGrid(
            align_info,
            ('姿态测量', '对准 ΔX / ΔY', '安全间隙', 'P0 / P_final', '130 核对'),
            self.theme,
        )
        self.offset_align_grid.grid(row=0, column=0, sticky='nsew')

        cmp_info = self.card(right, '新旧偏移对比与档案', 2, 0)
        self.offset_compare_grid = KeyValueGrid(
            cmp_info,
            ('修正后 / 旧偏移', '差值 (新 - 旧)', '工具1 档案', '工具2 档案', '基准状态'),
            self.theme,
        )
        self.offset_compare_grid.grid(row=0, column=0, sticky='nsew')
        for grid in (self.offset_tri_grid, self.offset_align_grid, self.offset_compare_grid):
            for widget in getattr(grid, 'winfo_children', lambda: ())():
                widget.configure(font=(self.theme.body[0], 10), pady=1)
                widget.grid_configure(pady=0)

    def build_baseline(self):
        self.baseline_pending = 0
        self.baseline_dialog = None
        self.baseline_scan_dialogs = {}
        small = (self.theme.body[0], 10)
        style = ttk.Style(self.root)
        style.configure('Baseline.TButton', font=small, padding=(5, 2))
        style.configure('Selected.Baseline.TButton', foreground=WHITE, background='#2563eb')
        style.map('Selected.Baseline.TButton', background=[('disabled', '#a9c9f8')], foreground=[('disabled', '#405575')])
        style.configure('Measure.Baseline.TButton', font=(small[0], 11, 'bold'), foreground=WHITE, background='#168448')
        style.map('Action.Baseline.TButton', foreground=[('disabled', '#9ca3af'), ('!disabled', INK)],
                  background=[('disabled', '#e5e7eb')])

        def label(parent, text='', **options):
            options.setdefault('font', small)
            options.setdefault('height', 1)
            return self.label(parent, text, **options)

        def button(parent, text, command, style='Baseline.TButton'):
            return ttk.Button(parent, text=text, command=command, style=style, width=0)

        for column in (0, 1):
            self.baseline.columnconfigure(column, weight=1, uniform='baseline')
        self.baseline.rowconfigure(1, weight=1)
        head = tk.Frame(self.baseline, bg=WHITE, highlightbackground=BORDER, highlightthickness=1, padx=5, pady=2)
        head.grid(row=0, column=0, columnspan=2, sticky='ew', padx=3, pady=3)
        head.columnconfigure(0, weight=1)
        bar = tk.Frame(head, bg=WHITE)
        bar.grid(row=0, column=0, sticky='ew')
        bar.columnconfigure(1, weight=2)
        bar.columnconfigure(3, weight=1)
        self.baseline_heading = label(bar, '当前：未选择', bg=MUTED, fg=WHITE, font=(small[0], 12, 'bold'), padx=7)
        self.baseline_heading.grid(row=0, column=0, sticky='w')
        self.baseline_message = label(bar, '请开始对刀基准建立', width=1)
        self.baseline_message.grid(row=0, column=1, sticky='ew', padx=6)
        self.baseline_live = label(bar, '实时辅助：等待针尖', fg=MUTED)
        self.baseline_live.grid(row=0, column=2, sticky='w')
        self.baseline_notice = label(bar, fg=COLORS['yellow'], width=1)
        self.baseline_notice.grid(row=0, column=3, sticky='ew', padx=6)
        for status in (self.baseline_message, self.baseline_notice):
            bind_text_tooltip(status, status)

        controls = tk.Frame(head, bg=WHITE)
        controls.grid(row=1, column=0, sticky='ew', pady=(2, 0))

        def action(column, text, command, *, params=None, style='Baseline.TButton'):
            widget = button(controls, text, lambda: self.baseline_command(command, params), style)
            widget.grid(row=0, column=column, padx=(0, 3))
            return widget

        self.baseline_start = action(0, '开始对刀基准建立', 'baseline_start')
        self.baseline_tools = [action(i, f'工具{i}', 'baseline_select', params={'tool': i}) for i in (1, 2)]
        self.baseline_measure = action(3, '对刀基准', 'baseline_measure', style='Measure.Baseline.TButton')
        self.baseline_manual = action(4, '识别不准，手动确认', 'baseline_manual')
        self.baseline_aux = action(5, '辅助相机手动重设', 'baseline_aux')
        self.baseline_save = action(6, '保存基准', 'baseline_save')
        self.baseline_end = action(7, '结束/放弃', 'baseline_end')
        label(controls, '│', fg=MUTED).grid(row=0, column=8, padx=(0, 3))
        self.runout_start = action(9, '开始离心测试', 'tooltip_runout_start')
        self.runout_press = action(10, '记录并计算', 'tooltip_runout_press')
        self.runout_end = action(11, '结束', 'tooltip_runout_end')
        self.runout_detail = label(controls, '请打开两路预览并精定位', fg=MUTED, width=1, height=1, anchor='nw', justify='left')
        self.runout_detail.grid(row=0, column=13, sticky='ew')
        controls.columnconfigure(13, weight=1)
        self.runout_max = label(controls, '最大离心量 — µm')
        self.runout_max.grid(row=0, column=12, padx=(4, 7))
        bind_text_tooltip(self.runout_detail, self.runout_detail)

        self.baseline_views = {}
        self.baseline_exposure_pending = {}
        camera_cards = []
        for column, (name, title) in enumerate((('front', '主对刀相机（正面）'), ('side', '辅助对刀相机（侧面）'))):
            body = tk.Frame(self.baseline, bg=WHITE, highlightbackground=BORDER, highlightthickness=1, padx=4, pady=2)
            camera_cards.append(body)
            body.grid(row=1, column=column, sticky='nsew', padx=3, pady=(0, 3))
            body.columnconfigure(0, weight=1)
            body.rowconfigure(3, weight=1)
            actions = tk.Frame(body, bg=WHITE)
            actions.grid(row=0, column=0, sticky='ew')
            actions.columnconfigure(3, weight=2)
            toggle = tk.Button(actions, text=title+'预览：关', font=small, padx=4, pady=0,
                               command=lambda n=name: self.baseline_toggle(n))
            toggle.grid(row=0, column=0, padx=(0, 3))
            exposure = button(actions, '重新测曝光', lambda n=name: self.baseline_retest_exposure(n), 'Action.Baseline.TButton')
            exposure.grid(row=0, column=1, padx=(0, 3))
            reset = button(actions, '清晰度峰值清零', lambda n=name: self.baseline_command('tooltip_peak_reset', {'camera': n}), 'Action.Baseline.TButton')
            reset.grid(row=0, column=2, padx=(0, 3))
            path = label(actions, '未分配', fg=MUTED, width=1)
            path.grid(row=0, column=3, sticky='ew')
            reason = label(actions, fg=MUTED, width=1)
            reason.grid(row=0, column=4, sticky='ew')
            actions.columnconfigure(4, weight=1)
            bind_text_tooltip(path, path)
            bind_text_tooltip(toggle, reason)

            manual = tk.Frame(body, bg=WHITE)
            manual.grid(row=1, column=0, sticky='ew', pady=(2, 0))
            manual.columnconfigure(6, weight=1)
            label(manual, '曝光 ms').grid(row=0, column=0)
            entry = ttk.Spinbox(manual, from_=0.1, to=500, increment=0.1, width=6, font=small)
            entry.grid(row=0, column=1, padx=(2, 3))
            set_exposure = button(manual, '设置曝光', lambda n=name: self.baseline_exposure(n, False))
            set_exposure.grid(row=0, column=2, padx=(0, 3))
            restore = button(manual, '恢复曝光', lambda n=name: self.baseline_exposure(n, True))
            restore.grid(row=0, column=3, padx=(0, 3))
            save = button(manual, '设为实际曝光', lambda n=name: self.baseline_save_exposure(n))
            save.grid(row=0, column=4, padx=(0, 3))
            limits = label(manual, '范围 —')
            limits.grid(row=0, column=5, padx=(0, 3))
            saved = label(manual, '已存曝光：无', width=1)
            saved.grid(row=0, column=6, sticky='ew')
            save_reason = label(manual)
            bind_text_tooltip(save, save_reason)
            bind_text_tooltip(saved, saved)

            scan = tk.Frame(body, bg=WHITE)
            scan.grid(row=2, column=0, sticky='ew', pady=(2, 0))
            scan.columnconfigure(1, weight=2)
            scan.columnconfigure(2, weight=1)
            scan_button = button(scan, '扫描详情', lambda n=name: self.show_baseline_scan(n))
            scan_button.grid(row=0, column=0, padx=(0, 3))
            scan_status = label(scan, '扫描：—', fg=MUTED, width=1)
            scan_status.grid(row=0, column=1, sticky='ew')
            result = label(scan, width=1)
            result.grid(row=0, column=2, sticky='ew')
            label(scan, '点击画面放大', fg=MUTED).grid(row=0, column=3, padx=(3, 0))
            bind_text_tooltip(scan_status, scan_status)
            bind_text_tooltip(result, result)

            viewport = tk.Frame(body, bg='#eef1f5')
            viewport.grid(row=3, column=0, sticky='nsew', pady=(2, 0))
            # Place ignores the photo's requested size when sizing the viewport.
            pic = Picture(viewport, live_image_file(self.directory, 'baseline_'+name), self.theme)
            pic.min_mtime_ns = self.started_at_ns
            pic.config(borderwidth=0, highlightthickness=0, padx=0, pady=0)
            self.pictures.append(pic)

            def fit_picture(_event=None, viewport=viewport, pic=pic):
                source_width, source_height = pic.image.size if pic.image is not None else (16, 9)
                width, height = viewport.winfo_width(), viewport.winfo_height()
                fitted_width = min(width, round(height * source_width / source_height))
                fitted_height = min(height, round(width * source_height / source_width))
                pic.place(relx=.5, rely=.5, anchor='center', width=max(1, fitted_width), height=max(1, fitted_height))

            viewport.bind('<Configure>', fit_picture)
            pic.aspect_changed = fit_picture

            readings = tk.Frame(body, bg=WHITE)
            readings.grid(row=4, column=0, sticky='ew')
            readings.columnconfigure(1, weight=1)
            info = label(readings, '预览关闭', fg=MUTED, font=(self.theme.large[0], 13))
            info.grid(row=0, column=0, sticky='w')
            hint = label(readings, fg=COLORS['red'], width=1)
            hint.grid(row=0, column=1, sticky='ew')
            graph = tk.Canvas(readings, width=100, height=16, bg=WHITE, highlightthickness=0)
            graph.grid(row=0, column=2, sticky='e', padx=(3, 0))
            detail = label(readings, width=1, height=2, anchor='nw', justify='left')
            detail.grid(row=1, column=0, columnspan=3, sticky='ew')
            bind_text_tooltip(info, info)
            bind_text_tooltip(hint, hint)
            bind_text_tooltip(detail, detail)
            self.baseline_views[name] = dict(
                title=title, path=path, toggle=toggle, exposure=exposure, reset=reset, reason=reason,
                info=info, hint=hint, detail=detail, graph=graph, picture=pic, exposure_entry=entry,
                exposure_set=set_exposure, exposure_restore=restore, exposure_save=save, exposure_saved=saved,
                exposure_save_reason=save_reason, exposure_limits=limits, exposure_result=result,
                scan_button=scan_button, scan_status=scan_status, scan_table_text='')

        footer = tk.Frame(self.baseline, bg=BG)
        footer.grid(row=2, column=0, columnspan=2, sticky='ew')
        footer.columnconfigure(0, weight=1)
        canvas = tk.Canvas(footer, height=113, highlightthickness=0, bg=BG)
        canvas.grid(row=0, column=0, sticky='ew')
        scroll = ttk.Scrollbar(footer, orient='vertical', command=canvas.yview)
        scroll.grid(row=0, column=1, sticky='ns')
        canvas.configure(yscrollcommand=scroll.set)
        cards = tk.Frame(canvas, bg=BG)
        window = canvas.create_window(0, 0, window=cards, anchor='nw')
        geometry = ScrollGeometry(canvas, window)

        def fit_results(_event):
            # Camera cards request only their controls and readings, since the pictures use place.
            page_width = self.baseline.winfo_width()
            if page_width <= 1:
                return
            camera_height = max(card.winfo_reqheight() for card in camera_cards)
            picture_height = round((page_width // 2 - 16) * 9 / 16)
            spare = self.baseline.winfo_height() - head.winfo_reqheight() - camera_height - picture_height - 9
            height = min(cards.winfo_reqheight(), max(64, spare))
            if height != canvas.winfo_reqheight():
                canvas.configure(height=height)

        self.baseline.bind('<Configure>', fit_results)
        canvas.bind('<Configure>', geometry.viewport_changed)
        cards.bind('<Configure>', geometry.content_changed)
        cards.bind('<Configure>', fit_results, add='+')
        for column in (0, 1):
            cards.columnconfigure(column, weight=1, uniform='baseline_results')
        for column, (title, key, initial) in enumerate((
            ('本次结果', 'result', '等待测量'), ('现有基准与工具补偿', 'existing', '—')
        )):
            body = self.card(cards, title, 0, column)
            card = body.master
            card.title_label.config(font=(small[0], 11, 'bold'))
            value = label(body, initial, anchor='nw', justify='left', width=1, height=0, wraplength=600)
            value.grid(row=0, column=0, sticky='nsew')
            body.bind('<Configure>', lambda event, value=value: value.config(wraplength=max(100, event.width - 8)))
            for widget in (card, card.title_label, body, value):
                widget.bind('<Button-4>', lambda _: canvas.yview_scroll(-1, 'units'))
                widget.bind('<Button-5>', lambda _: canvas.yview_scroll(1, 'units'))
            setattr(self, 'baseline_' + key, value)
            setattr(self, 'baseline_' + ('results' if key == 'result' else key) + '_card', card)
        self.baseline_footer = footer

    def show_baseline_scan(self, name):
        if name in self.baseline_scan_dialogs:
            self.baseline_scan_dialogs[name]['window'].lift()
            return
        view = self.baseline_views[name]
        top = tk.Toplevel(self.root)
        top.title(view['title'] + ' · 扫描详情')
        top.transient(self.root)
        font = (self.theme.body[0], 11)
        status = self.label(top, view['scan_status'].cget('text'), font=font, fg=MUTED)
        status.pack(fill='x', padx=12, pady=(8, 0))
        table = self.label(top, view['scan_table_text'], font=font, justify='left')
        table.pack(fill='both', expand=True, padx=12, pady=(0, 8))
        self.baseline_scan_dialogs[name] = dict(window=top, status=status, table=table)
        top.bind('<Destroy>', lambda event: self.baseline_scan_dialogs.pop(name, None) if event.widget is top else None)

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

    def baseline_retest_exposure(self, name):
        seq = self.baseline_command('tooltip_exposure_retest', {'camera': name})
        if seq:
            self.baseline_exposure_pending[name] = (seq, time.monotonic())
            self.baseline_views[name]['exposure_result'].config(text='等待扫描…', fg=INK)

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
            self.baseline_footer.grid_remove()
            self.baseline_results_card.grid_remove()
            self.baseline_existing_card.grid_remove()
            self.baseline_live.grid_remove()
            self.baseline_notice.grid_remove()
        else:
            self.baseline_footer.grid()
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
            scan_reason = ('请打开相机预览' if not on else '正在测曝光' if item.get('exposure_busy') else '请先精定位工具' if not live or live.get('detection_status') != '精定位' else '当前不可操作' if not may_prepare else '')
            enable(widgets['exposure'], not scan_reason and name not in self.baseline_exposure_pending); enable(widgets['reset'], on and may_prepare)
            scan_status = item.get('exposure_progress') or scan_reason
            scan_rows = item.get('exposure_scan_rows') or []
            lines = []
            if scan_rows:
                ref = scan_rows[0].get('prong_ref', '—')
                lines = [f'曝光 ms　识别　齿数（参考 {ref}）　对比度　清晰度%　背景　背景饱和/工具发白　合格']
                for row in scan_rows:
                    lines.append(f"{row.get('exposure', 0)/10:g}　{'是' if row.get('recognized') else '否'}　{row.get('prongs', 0)}　{row.get('contrast', 0):.1f}　{row.get('sharpness_percent', 0):.0f}%　{row.get('background', 0):.0f}　{row.get('saturated_fraction', 0):.1%}/{row.get('tool_whiteness', 0):.1%}　{'是' if row.get('qualified') else '否'}{' ★' if row.get('selected') else ''}　{'、'.join(row.get('reasons') or [])}")
            selected = next((row for row in scan_rows if row.get('selected')), None)
            summary = f"选 {selected.get('exposure', 0)/10:g} ms ★" if selected is not None else ''
            if scan_rows and selected is None and not item.get('exposure_busy') and not any(row.get('qualified') for row in scan_rows):
                reasons = dict.fromkeys(reason for row in scan_rows for reason in (row.get('reasons') or []))
                summary = '无合格档'
                if reasons:
                    summary += '：' + '、'.join(reasons)
            parts = (scan_status, summary) if item.get('exposure_progress') else (summary, scan_status)
            status_text = '扫描：' + ('　'.join(text for text in parts if text) or '—')
            table_text = '\n'.join(lines)
            widgets['scan_status'].config(text=status_text)
            widgets['scan_table_text'] = table_text
            scan_dialog = self.baseline_scan_dialogs.get(name)
            if scan_dialog is not None:
                scan_dialog['status'].config(text=status_text)
                scan_dialog['table'].config(text=table_text)
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
                widgets['last_exposure_reply'] = result.get('seq')
            elif pending and time.monotonic() - pending[1] >= 5:
                self.baseline_exposure_pending.pop(name, None)
                widgets['exposure_result'].config(text='曝光操作未完成或未确认，请检查预览', fg=COLORS['red'])
            elif not pending and result.get('seq') is not None and result.get('seq') != widgets.get('last_exposure_reply') and result.get('state') in ('done', 'error'):
                widgets['last_exposure_reply'] = result['seq']
                widgets['exposure_result'].config(text=result.get('message', ''), fg=COLORS['red'] if result['state'] == 'error' else INK)
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
            enable_main(button, phase == 'select'); button.config(style='Selected.Baseline.TButton' if i == tool else 'Baseline.TButton')
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
        self.pickacc_pending = 0; self.pickacc_dialog = None
        self.pickacc.columnconfigure(0, weight=1); self.pickacc.columnconfigure(1, weight=1)
        self.pickacc.rowconfigure(1, weight=3, uniform='pickacc'); self.pickacc.rowconfigure(2, weight=2, uniform='pickacc')
        head = self.card(self.pickacc, '挑取精度标定', 0, 0, columnspan=2)
        bar = tk.Frame(head, bg=WHITE); bar.grid(row=0, column=0, sticky='ew'); bar.columnconfigure(2, weight=1)
        self.pickacc_heading = tk.Label(bar, text='当前：等待 PLC 指定', bg='#667085', fg=WHITE, font=(self.theme.title[0],20,'bold'), padx=12)
        self.pickacc_heading.grid(row=0,column=0,padx=4)
        self.pickacc_message = tk.Label(bar, text='等待 PLC 请求', bg=WHITE, fg=INK, font=(self.theme.title[0],16), anchor='w')
        self.pickacc_plc_id = tk.Label(bar, text='40905=—', bg=WHITE, fg=INK, font=self.theme.body)
        self.pickacc_plc_id.grid(row=0, column=1, padx=8)
        self.pickacc_message.grid(row=0,column=2,sticky='ew',padx=8)
        controls=tk.Frame(head,bg=WHITE); controls.grid(row=1,column=0,sticky='ew'); controls.columnconfigure(4,weight=1)
        self.pickacc_start=ttk.Button(controls,text='开始',command=lambda:self.pickacc_command('pickacc_start')); self.pickacc_start.grid(row=0,column=0,padx=4)
        self.pickacc_calculate=ttk.Button(controls,text='计算补偿',style='OffsetPlace.TButton',command=lambda:self.pickacc_command('pickacc_calculate')); self.pickacc_calculate.grid(row=0,column=5,padx=4)
        self.pickacc_record_z=ttk.Button(controls,text='记录下针 Z',style='OffsetPlace.TButton',command=lambda:self.pickacc_command('pickacc_record_z')); self.pickacc_record_z.grid(row=0,column=6,padx=4)
        self.pickacc_discard=ttk.Button(controls,text='本点作废重测',command=lambda:self.pickacc_command('pickacc_discard')); self.pickacc_discard.grid(row=0,column=7,padx=4)
        self.pickacc_abort=ttk.Button(controls,text='放弃本次标定',command=lambda:self.pickacc_command('abort')); self.pickacc_abort.grid(row=0,column=8,padx=4)
        self.pickacc_existing=tk.Label(head,text='',bg=WHITE,fg=COLORS['yellow'],font=self.theme.body,anchor='w',justify='left'); self.pickacc_existing.grid(row=2,column=0,sticky='ew')
        self.pickacc_z_info=tk.Label(head,text='',bg=WHITE,fg=INK,font=self.theme.body,anchor='w',justify='left'); self.pickacc_z_info.grid(row=3,column=0,sticky='ew')
        images=tk.Frame(self.pickacc,bg=BG); images.grid(row=1,column=0,sticky='nsew'); images.columnconfigure(0,weight=3); images.columnconfigure(1,weight=2); images.rowconfigure(0,weight=1)
        self.picture(self.card(images,'相机画面 · 点击放大',0,0),'pickacc_latest')
        self.picture(self.card(images,'当前目标放大图',0,1),'pickacc_crop')
        tray=self.card(self.pickacc,'料盘示意',1,1)
        self.pickacc_tray_title=tray.master.title_label
        self.pickacc_tray=tk.Canvas(tray,bg=WHITE,highlightthickness=0); self.pickacc_tray.grid(row=0,column=0,sticky='nsew')
        table=self.card(self.pickacc,'测点记录',2,0,columnspan=2)
        columns=('序号','目标','阶段','下发 XY','实到 XY','dx','dy','|r|','下针 Z','备注')
        self.pickacc_table=self.table_widget(table,columns,(50,55,90,155,155,70,70,70,80,175))
        scroll=ttk.Scrollbar(table,orient='vertical',command=self.pickacc_table.yview); scroll.grid(row=0,column=1,sticky='ns'); self.pickacc_table.configure(yscrollcommand=scroll.set,style='Offset.Treeview')
        self.pickacc_stats=tk.Label(table,text='',bg=WHITE,fg=INK,font=self.theme.body,anchor='w'); self.pickacc_stats.grid(row=1,column=0,sticky='ew')

    def pickacc_command(self, name, params=None):
        self._send_page_command('pickacc', name, params)

    def update_pickacc(self,main):
        data=main.get('pickacc') or {}; phase=data.get('phase'); self.pickacc_ack=main.get('command_seq',0)
        self._check_page_command('pickacc', self.pickacc_ack)
        tool=data.get('tool')
        self.pickacc_plc_id.config(text=f"40905={data.get('plc_tool_id', '—')}")
        self.pickacc_heading.config(text=f'当前：工具{tool}（PLC 指定）' if tool in (1, 2) else '当前：等待 PLC 指定',bg='#2563eb' if tool==1 else '#d56713' if tool==2 else '#667085')
        self.pickacc_message.config(text=getattr(self, 'pickacc_command_message', '') or ('上一条命令处理中' if self.pickacc_pending else data.get('message')) or '等待 PLC 请求', fg=COLORS['red'] if getattr(self, 'pickacc_command_message', '') else COLORS['blue'] if phase=='z_step' else INK)
        self.pickacc_start.config(state='normal' if phase=='select' and tool in (1, 2) and not self.pickacc_pending else 'disabled')
        self.pickacc_calculate.config(state='normal' if phase=='tool_in_place' and not self.pickacc_pending else 'disabled')
        self.pickacc_record_z.config(state='normal' if phase=='z_step' and not self.pickacc_pending else 'disabled')
        self.pickacc_discard.config(state='normal' if phase in ('tool_in_place','z_step') and not self.pickacc_pending else 'disabled')
        self.pickacc_abort.config(state='normal' if phase not in (None,'idle','done','failed') and not self.pickacc_pending else 'disabled')
        summaries=[]
        for i in (1,2):
            old=(data.get('existing') or {}).get(f'tool{i}') or {}
            z_type=old.get('z_type')
            if z_type=='constant' and isinstance(old.get('z_mm'), (int, float)):
                z_label=f"Z=全盘 {old['z_mm']:.3f}mm"
            elif z_type=='table':
                z_label=f"Z=Z表({len(old.get('z_nodes') or [])}点)"
            else:
                z_label=f"⚠ 工具{i} 未做下针深度标定"
            summaries.append(f"工具{i}：{pickacc_type(old.get('type'))} {z_label} {display_time(old.get('time'))} 验证 {len(old.get('verification',[]))}点" + (f'　工具{i} 偏移/相机标定已变更，挑取补偿需重新验证' if old.get('stale') else ''))
        self.pickacc_existing.config(text='　｜　'.join(summaries))
        jitter=data.get('z_jitter_mm')
        self.pickacc_z_info.config(text=pickacc_z_summary_text(data), fg=COLORS['red'] if isinstance(jitter, (int, float)) and jitter > 0.02 else COLORS['blue'] if phase=='z_step' else INK)
        self.pickacc_table.delete(*self.pickacc_table.get_children())
        import math
        rows_to_show=list(data.get('samples') or [])
        if phase=='z_step' and isinstance(data.get('pending_sample'), dict):
            rows_to_show.append(data['pending_sample'])
        for i,s in enumerate(rows_to_show,1):
            r=s.get('residual_xy_mm'); sent=s.get('sent_xyz_mm'); actual=s.get('actual_xyz_mm')
            xy=lambda v: f'{v[0]:.3f}, {v[1]:.3f}' if v and len(v)>=2 else '—'
            pz=s.get('pick_z_mm')
            z_text=f'{pz:.3f}' if isinstance(pz,(int,float)) else ('待测(压塞尺)' if s is data.get('pending_sample') else '—')
            self.pickacc_table.insert('','end',values=(i,s.get('target'),pickacc_stage(s.get('stage')),xy(sent),xy(actual),f'{r[0]:.3f}' if r else '—',f'{r[1]:.3f}' if r else '—',f'{math.hypot(*r):.3f}' if r else '—',z_text,pickacc_note(s)))
        decision=data.get('decision') or {}; candidate=data.get('candidate') or {}; z_src=candidate if candidate.get('z_type') else (data.get('z_summary') or {})
        mode=pickacc_type(candidate['type']) if candidate.get('type') else '待判定'
        z_mode=pickacc_z_type(z_src.get('z_type'))
        if z_src.get('z_type')=='constant' and isinstance(z_src.get('z_mm'),(int,float)):
            z_mode+=f" {z_src['z_mm']:.3f}mm"
        elif z_src.get('z_type')=='table':
            z_dec=z_src.get('z_decision') or {}
            z_mode+=f" ({len(z_src.get('z_nodes') or [])}点 {z_dec.get('min_z_mm',0):.3f}~{z_dec.get('max_z_mm',0):.3f}mm)"
        validation=[math.hypot(*s['residual_xy_mm']) for s in data.get('samples',[]) if s.get('verification') and s.get('residual_xy_mm') and not s.get('superseded')]
        validation_text=f'验证：{len(validation)} 点，最大 {max(validation):.3f} mm' if validation else '验证：未开始'
        self.pickacc_stats.config(text=f"XY 判定：{mode}  最大偏差 {decision.get('max_dev_mm','—')} / 标准差 {decision.get('std_mm','—')}　|　下针 Z：{z_mode}　|　{pickacc_infield(data.get('infield') or [])}　{validation_text}")
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
        def z_desc(entry):
            if not isinstance(entry, dict) or not entry.get('z_type'): return '未标定'
            if entry['z_type']=='constant' and isinstance(entry.get('z_mm'),(int,float)):
                return f"全盘最大值 {entry['z_mm']:.3f} mm (Z_ref={entry.get('z_ref',0.0):.3f})"
            nodes=entry.get('z_nodes') or []; zs=[n['z'] for n in nodes if isinstance(n.get('z'),(int,float))]
            rng=f"{min(zs):.3f}~{max(zs):.3f} mm" if zs else '—'
            return f"Z 表 {len(nodes)} 节点 ({rng}, Z_ref={entry.get('z_ref',0.0):.3f})"
        tk.Label(top,text=f"旧 XY：{pickacc_type(old.get('type'))} {prior or str(len(old.get('nodes',[])))+' 节点'}\n新 XY：{pickacc_type(candidate.get('type'))} {current or str(len(candidate.get('nodes',[])))+' 节点'}\n均值差：{delta}\n旧 Z：{z_desc(old)}\n新 Z：{z_desc(candidate)}\n验证：{len(candidate.get('verification',[]))} 点通过，最大 {max(valid,default=0):.3f} mm",justify='left',font=self.theme.body,padx=25,pady=15).pack()
        row=tk.Frame(top); row.pack(); ttk.Button(row,text='覆盖保存',command=lambda:self.pickacc_command('pickacc_confirm',{'accept':True})).pack(side='left',padx=15,pady=10); ttk.Button(row,text='不保存',command=lambda:self.pickacc_command('pickacc_confirm',{'accept':False})).pack(side='left',padx=15,pady=10)
        top.protocol('WM_DELETE_WINDOW',lambda:self.pickacc_command('pickacc_confirm',{'accept':False}))
        top.update_idletasks()
        x=self.root.winfo_rootx()+max(0,(self.root.winfo_width()-top.winfo_reqwidth())//2)
        y=self.root.winfo_rooty()+max(0,(self.root.winfo_height()-top.winfo_reqheight())//2)
        top.geometry(f'+{x}+{y}')

    def offset_command(self, name, params=None):
        self._send_page_command('offset', name, params)

    def update_offset(self, main):
        data = main.get('tooloffset') or {}
        phase = data.get('phase') or 'idle'
        self.offset_ack = main.get('command_seq', 0)
        self._check_page_command('offset', self.offset_ack)

        tool = data.get('tool')
        plc_tool_id = data.get('plc_tool_id', main.get('plc_tool_id'))
        self.offset_heading.config(
            text=f"当前：{'工具' + str(tool) + '（PLC 100/110 指定）' if tool in (1, 2) else '等待 PLC 100/110 指定'}",
            bg='#2563eb' if tool == 1 else '#d56713' if tool == 2 else '#667085',
        )
        self.offset_plc_id.config(text=f"40905={plc_tool_id if plc_tool_id is not None else '—'}")

        tri = data.get('triangulation') or {}
        tri_views = tri.get('views') or []
        align = data.get('alignment') or {}
        probe_step = align.get('probe_step') or 0
        align_iter = align.get('align_iteration') or 0

        stage_map = {
            'idle': ('空闲（等待 PLC 90）', '#eef2f6', INK),
            'triangulate': (f"三角定位 第 {len(tri_views)}/≥3 张", '#dbeafe', '#1d4ed8'),
            'wait_tool_start': ('三角定位完成 · 等待 100/110', '#e0f2fe', '#0369a1'),
            'probing': (f"姿态测量 第 {max(1, probe_step)}/4 步", '#fef3c7', '#92400e'),
            'aligning': (f"对准 第 {max(1, align_iter)} 轮", '#fef3c7', '#92400e'),
            'wait_130': ('已对准 · 等待 PLC 130 核对', '#dcfce7', '#166534'),
            'done': ('完成并已保存', '#dcfce7', '#166534'),
            'failed': ('失败（已回 99/999）', '#fee2e2', '#b91c1c'),
        }
        stage_text, stage_bg, stage_fg = stage_map.get(phase, (f'阶段：{phase}', '#eef2f6', INK))
        self.offset_stage.config(text=f"阶段：{stage_text}", bg=stage_bg, fg=stage_fg)

        msg = getattr(self, 'offset_command_message', '') or ('上一条命令处理中' if self.offset_pending else data.get('message')) or '等待 PLC 请求 90 (40905=1)'
        msg_fg = COLORS['red'] if getattr(self, 'offset_command_message', '') or phase == 'failed' else '#168448' if phase == 'done' else INK
        self.offset_message.config(text=msg, fg=msg_fg)

        if phase == 'failed' and any(data.get(k) for k in ('failure_code', 'failure_what', 'failure_where', 'failure_hint')):
            detail_lines = []
            if data.get('failure_code'):
                detail_lines.append(f"错误码：{data['failure_code']}    位置：{data.get('failure_where') or '工具偏移标定'}")
            detail_lines.append(f"现象：{data.get('failure_what') or data.get('message') or '—'}")
            detail_lines.append(f"建议：{data.get('failure_hint') or '检查标定尖、相机视野与 PLC 坐标。'}")
            self.offset_failure_detail.config(text='\n'.join(detail_lines))
            self.offset_failure_detail.grid()
            if not data.get('failure_what') or data['failure_what'] == msg:
                self.offset_message.grid_remove()
            else:
                self.offset_message.grid()
        else:
            self.offset_failure_detail.config(text='')
            self.offset_failure_detail.grid_remove()
            self.offset_message.grid()

        active = phase in ('triangulate', 'wait_tool_start', 'probing', 'aligning', 'wait_130') and not self.offset_pending and not data.get('abort_requested')
        self.offset_abort.config(state='normal' if active else 'disabled')

        for idx, lbl in enumerate(self.offset_tri_labels, 1):
            view = tri_views[idx - 1] if idx - 1 < len(tri_views) else None
            if view and view.get('uv'):
                uv = view['uv']
                lbl.config(text=f"视角{idx} ({uv[0]:.1f},{uv[1]:.1f}) 质={view.get('quality', 0):.2f}", fg=INK)
            elif view and not view.get('ok', True):
                lbl.config(text=f"视角{idx}：失败", fg=COLORS['red'])
            else:
                lbl.config(text=f"视角{idx}：—", fg=MUTED)

        xyz = tri.get('xyz')
        target_xy = tri.get('target_xy')
        max_res = tri.get('max_residual_mm')
        ratio = tri.get('max_baseline_ratio')
        angle_deg = tri.get('max_angle_deg')
        views_geom = f"{len(tri_views)} 张"
        if ratio is not None and angle_deg is not None:
            views_geom += f"  (基线比 {ratio:.3f} / {angle_deg:.1f}°)"
        self.offset_tri_grid.set({
            '标定尖 XYZ': f"({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}) mm" if xyz else '—',
            '目标机台 XY': f"({target_xy[0]:.4f}, {target_xy[1]:.4f}) mm" if target_xy else '—',
            '视角 / 夹角': views_geom if tri_views else '—',
            '射线最大残差': f"{max_res:.4f} mm" if max_res is not None else '—',
        })

        dxy = align.get('delta_xy_mm')
        gap_mm = align.get('gap_mm')
        pred_gap = align.get('predicted_gap_mm')
        p0 = data.get('p0')
        p_final = data.get('p_final')
        plc_final = data.get('plc_final')
        jacs = align.get('jacobians') or {}
        if jacs:
            f_rms = (jacs.get('front') or {}).get('rms_px')
            s_rms = (jacs.get('side') or {}).get('rms_px')
            pose_text = f"4/4 完成  RMS {f_rms:.2f} / {s_rms:.2f} px" if f_rms is not None and s_rms is not None else '4/4 完成'
        elif probe_step:
            pose_text = f"第 {probe_step}/4 步进行中"
        else:
            pose_text = '未开始'
        gap_str = f"当前 {gap_mm:.2f} mm" if gap_mm is not None else '—'
        if pred_gap is not None:
            gap_str += f"  (预测 {pred_gap:.2f})"
        verify_str = '—'
        if plc_final and p_final:
            diff_max = max(abs(a - b) for a, b in zip(plc_final, p_final))
            verify_str = f"已核对，最大轴差 {diff_max:.4f} mm"
        elif p_final:
            verify_str = '等待 PLC 130'
        p0_s = f"({p0[0]:.2f},{p0[1]:.2f},{p0[2]:.2f})" if p0 else '—'
        pf_s = f"({p_final[0]:.2f},{p_final[1]:.2f},{p_final[2]:.2f})" if p_final else '—'
        self.offset_align_grid.set({
            '姿态测量': pose_text,
            '对准 ΔX / ΔY': (f"第{align_iter}轮  {dxy[0]:+.4f}, {dxy[1]:+.4f} mm" if dxy else f"第{align_iter}轮" if align_iter else '—'),
            '安全间隙': gap_str,
            'P0 / P_final': f"{p0_s}  /  {pf_s}" if (p0 or p_final) else '—',
            '130 核对': verify_str,
        })

        existing = data.get('existing') or {}
        new_off = data.get('offset_xy')
        old_off = data.get('old_offset_xy')
        if new_off and old_off:
            cmp_text = f"({new_off[0]:+.3f},{new_off[1]:+.3f}) / ({old_off[0]:+.3f},{old_off[1]:+.3f})"
            diff_text = f"({new_off[0] - old_off[0]:+.4f}, {new_off[1] - old_off[1]:+.4f}) mm"
        elif new_off:
            cmp_text = f"({new_off[0]:+.4f}, {new_off[1]:+.4f}) mm"
            diff_text = '—'
        else:
            cmp_text = '—'
            diff_text = '—'
        warnings = []
        stale_flags = []
        tool_summaries = {}
        for i in (1, 2):
            rec = existing.get(f'tool{i}') or {}
            xy = rec.get('offset_xy_mm')
            cxy = rec.get('current_tools_xy')
            xy_s = f"{xy[0]:+.3f},{xy[1]:+.3f}" if xy else '未标'
            cxy_s = f"{cxy[0]:+.3f},{cxy[1]:+.3f}" if cxy else '—'
            tool_summaries[i] = f"off=({xy_s})  tools=({cxy_s})"
            if rec.get('calibration_changed'):
                warnings.append(f'工具{i}：相机标定已变更')
            if rec.get('baseline_stale'):
                stale_flags.append(f'工具{i} 需重建基准')
        self.offset_compare_grid.set({
            '修正后 / 旧偏移': cmp_text,
            '差值 (新 - 旧)': diff_text,
            '工具1 档案': tool_summaries[1],
            '工具2 档案': tool_summaries[2],
            '基准状态': '；'.join(stale_flags) if stale_flags else '均有效',
        })
        if stale_flags:
            self.offset_warning.config(text='；'.join(stale_flags))
            self.offset_warning.grid()
        else:
            self.offset_warning.config(text='')
            self.offset_warning.grid_remove()
        if warnings:
            self.offset_caution.config(text='；'.join(warnings))
            self.offset_caution.grid()
        else:
            self.offset_caution.config(text='')
            self.offset_caution.grid_remove()
        if hasattr(self, 'tip_stale'):
            self.tip_stale.config(text='；'.join(w + '，需重建对刀基准后才能换针对刀' for w in stale_flags))
        if self.offset_dialog is not None:
            self.offset_dialog.destroy()
            self.offset_dialog = None

    def build_tip(self):
        self.tip.columnconfigure(0, weight=1, uniform='tipcols'); self.tip.columnconfigure(1, weight=1, uniform='tipcols')
        self.tip.rowconfigure(1, weight=6); self.tip.rowconfigure(2, weight=3)
        head = tk.Frame(self.tip, bg=WHITE); head.grid(row=0, column=0, columnspan=2, sticky='ew')
        self.tip_heading = tk.Label(head, text='等待 PLC 100/110', bg=COLORS['gray'], fg=WHITE, font=self.theme.large, padx=12)
        self.tip_heading.grid(row=0, column=0, padx=4, pady=4)
        self.tip_message = tk.Label(head, text='等待 PLC 100/110', bg=WHITE, fg=INK, font=self.theme.large, wraplength=850, justify='left')
        self.tip_plc_id = tk.Label(head, text='40905=—', bg=WHITE, fg=INK, font=self.theme.body)
        self.tip_plc_id.grid(row=0, column=1, padx=8)
        self.tip_message.grid(row=0, column=2, columnspan=3, padx=8, sticky='w')
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

    def build_depth(self):
        self.depth_pending = 0
        self.depth_ack = 0
        self.depth.columnconfigure(0, weight=64, uniform='depthcols')
        self.depth.columnconfigure(1, weight=36, uniform='depthcols')
        self.depth.rowconfigure(1, weight=3, uniform='depthrows')
        self.depth.rowconfigure(2, weight=2, uniform='depthrows')

        head = self.card(self.depth, '下针深度标定 · 料盘多点对焦测试', 0, 0, columnspan=2)
        head.columnconfigure(0, weight=1)
        bar = tk.Frame(head, bg=WHITE)
        bar.grid(row=0, column=0, sticky='ew')
        bar.columnconfigure(4, weight=1)
        self.depth_stage = tk.Label(bar, text='阶段：空闲（等待 PLC 160）', bg='#eef2f6', fg=INK,
                                    font=(self.theme.title[0], 13, 'bold'), padx=12, pady=3)
        self.depth_stage.grid(row=0, column=0, sticky='w', padx=(0, 10))
        self.depth_target = tk.Label(bar, text='靶点：—', bg='#eef2f6', fg=INK, font=self.theme.mono, padx=10, pady=3)
        self.depth_target.grid(row=0, column=1, sticky='w', padx=(0, 10))
        self.depth_z = tk.Label(bar, text='PLC Z：— mm', bg='#eef2f6', fg=INK, font=self.theme.mono, padx=10, pady=3)
        self.depth_z.grid(row=0, column=2, sticky='w', padx=(0, 10))
        self.depth_dof = tk.Label(bar, text='景深：—（FWHM —）', bg='#eef2f6', fg=INK, font=self.theme.mono, padx=10, pady=3)
        self.depth_dof.grid(row=0, column=3, sticky='w', padx=(0, 10))
        self.depth_abort = ttk.Button(bar, text='放弃本次标定', command=lambda: self.depth_command('depthcal_abort'))
        self.depth_abort.grid(row=0, column=5, sticky='e', padx=4)

        subbar = tk.Frame(head, bg=WHITE)
        subbar.grid(row=1, column=0, sticky='ew', pady=(4, 0))
        self.depth_metric = tk.Label(subbar, text='指标：tenengrad · gauss', bg=WHITE, fg=MUTED, font=self.theme.body)
        self.depth_metric.grid(row=0, column=0, sticky='w', padx=(0, 12))
        self.depth_fine_info = tk.Label(subbar, text='细扫：等待粗扫完成后确定', bg='#eef2f6', fg=INK,
                                        font=self.theme.mono, padx=8, pady=2)
        self.depth_fine_info.grid(row=0, column=1, sticky='w')

        expbar = tk.Frame(head, bg=WHITE)
        expbar.grid(row=2, column=0, sticky='ew', pady=(4, 0))
        tk.Label(expbar, text='曝光：', bg=WHITE, fg=MUTED, font=self.theme.body).grid(row=0, column=0, sticky='w')
        self.depth_exp_mode = 'auto'
        self.depth_exp_user_dirty = False
        self._depth_exp_synced = None
        self.depth_exp_auto_btn = tk.Button(
            expbar, text='◉ 自动', command=lambda: self.select_depth_exposure_mode('auto'),
            bg='#dbeafe', fg='#1d4ed8', font=self.theme.body, bd=1, relief='sunken', padx=8, pady=1,
        )
        self.depth_exp_auto_btn.grid(row=0, column=1, sticky='w', padx=(2, 4))
        self.depth_exp_manual_btn = tk.Button(
            expbar, text='○ 手动', command=lambda: self.select_depth_exposure_mode('manual'),
            bg='#f8fafc', fg=INK, font=self.theme.body, bd=1, relief='raised', padx=8, pady=1,
        )
        self.depth_exp_manual_btn.grid(row=0, column=2, sticky='w', padx=(0, 8))
        tk.Label(expbar, text='手动(µs):', bg=WHITE, fg=MUTED, font=self.theme.body).grid(row=0, column=3, sticky='w')
        self.depth_exp_entry = ttk.Entry(expbar, width=9, font=self.theme.mono)
        self.depth_exp_entry.grid(row=0, column=4, sticky='w', padx=(4, 4))
        self.depth_exp_apply = ttk.Button(expbar, text='应用', command=self.apply_depth_exposure)
        self.depth_exp_apply.grid(row=0, column=5, sticky='w', padx=(0, 10))
        self.depth_exp_live = tk.Label(
            expbar, text='当前曝光：— µs  |  ROI 均值 — / p99 —',
            bg='#eef2f6', fg=INK, font=self.theme.mono, padx=8, pady=2,
        )
        self.depth_exp_live.grid(row=0, column=6, sticky='w', padx=(0, 10))
        self.depth_exp_status = tk.Label(expbar, text='', bg=WHITE, fg=MUTED, font=self.theme.body)
        self.depth_exp_status.grid(row=0, column=7, sticky='w')

        self.depth_message = tk.Label(head, text='等待 PLC 160 发起料盘多点对焦测试', bg=WHITE, fg=INK,
                                      font=(self.theme.title[0], 13), anchor='w', justify='left', wraplength=1360)
        self.depth_message.grid(row=3, column=0, sticky='ew', pady=(4, 2))
        self.depth_failure_detail = tk.Label(head, text='', bg='#fff1f0', fg=COLORS['red'],
                                             font=self.theme.body, anchor='w', justify='left', padx=8, pady=4, wraplength=1360)
        self.depth_failure_detail.grid(row=4, column=0, sticky='ew', pady=(0, 2))
        self.depth_failure_detail.grid_remove()
        self.depth_notice = tk.Label(head, text='', bg='#fff7ed', fg=COLORS['yellow'],
                                     font=self.theme.body, anchor='w', justify='left', padx=8, pady=3, wraplength=1360)
        self.depth_notice.grid(row=5, column=0, sticky='ew', pady=(0, 2))
        self.depth_notice.grid_remove()
        self.depth_feasibility = tk.Label(head, text='可行性结论：等待测点完成后评估（总误差预算 0.05mm，对焦环节约 0.02mm）',
                                          bg='#f8fafc', fg=MUTED, font=(self.theme.title[0], 11, 'bold'),
                                          anchor='w', justify='left', padx=8, pady=4, wraplength=1360)
        self.depth_feasibility.grid(row=6, column=0, sticky='ew')
        self.depth_dof_summary = tk.Label(head, text='全盘景深（dof_ratio=0.80）：等待测点完成后汇总',
                                          bg='#f8fafc', fg=MUTED, font=(self.theme.title[0], 10, 'bold'),
                                          anchor='w', justify='left', padx=8, pady=3, wraplength=1360)
        self.depth_dof_summary.grid(row=7, column=0, sticky='ew', pady=(2, 0))

        curves_body = self.card(self.depth, '清晰度曲线（粗扫灰线 · 细扫彩线 · 竖线=峰位 · 浅蓝=DOF 区间）', 1, 0)
        curves_body.grid_propagate(False)
        self.depth_curves = tk.Canvas(curves_body, bg=WHITE, highlightthickness=0)
        self.depth_curves.grid(row=0, column=0, sticky='nsew')

        right = tk.Frame(self.depth, bg=BG)
        right.grid(row=1, column=1, sticky='nsew')
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        right.rowconfigure(1, weight=0)

        roi_body = self.card(right, 'ROI 实时缩略图 · 点击放大', 0, 0)
        roi_body.grid_propagate(False)
        self.depth_roi_pic = self.picture(roi_body, 'depthcal_roi')

        plane_body = self.card(right, '平面拟合结果', 1, 0)
        self.depth_plane_grid = KeyValueGrid(
            plane_body,
            ('平面方程', '最大 |残差| / 倾斜', '加测判定', '各点残差 (mm)'),
            self.theme,
        )
        self.depth_plane_grid.grid(row=0, column=0, sticky='nsew')
        for widget in getattr(self.depth_plane_grid, 'winfo_children', lambda: ())():
            widget.configure(font=(self.theme.body[0], 10), pady=1)
            widget.grid_configure(pady=0)

        table_body = self.card(self.depth, '各测点对焦重复性结果', 2, 0, columnspan=2)
        table_body.rowconfigure(0, weight=0)
        table_body.rowconfigure(1, weight=1)
        self.depth_arrival_summary = tk.Label(
            table_body, text='到位：等待测点到位数据（详见 arrivals.csv）',
            bg='#f8fafc', fg=MUTED, font=(self.theme.title[0], 10, 'bold'),
            anchor='w', justify='left', padx=8, pady=3, wraplength=1360,
        )
        self.depth_arrival_summary.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 4))
        cols = ('靶点', '模组 XY (mm)', '细扫1 Z', '细扫2 Z', '细扫3 Z', '均值 Z_peak', '极差', '标准差', 'FWHM', 'DOF 均值', 'DOF 极差', 'R²', '到位超差')
        widths = (50, 128, 92, 92, 92, 96, 68, 68, 66, 88, 76, 64, 215)
        self.depth_table = self.table_widget(
            table_body, cols, widths, stretch=cols, height=5,
            anchor='center', heading_anchor='center',
        )
        self.depth_table.grid(row=1, column=0, sticky='nsew')
        self.depth_table.tag_configure('arrival_warn', background='#fef3c7', foreground='#916300')
        depth_scroll = ttk.Scrollbar(table_body, orient='vertical', command=self.depth_table.yview)
        depth_scroll.grid(row=1, column=1, sticky='ns')
        self.depth_table.configure(yscrollcommand=depth_scroll.set)

    def depth_command(self, name, params=None):
        return self._send_page_command('depth', name, params)

    def select_depth_exposure_mode(self, mode, *, user=True):
        if mode not in ('auto', 'manual'):
            return
        self.depth_exp_mode = mode
        if user:
            self.depth_exp_user_dirty = True
        auto_on = mode == 'auto'
        self.depth_exp_auto_btn.config(
            text='◉ 自动' if auto_on else '○ 自动',
            bg='#dbeafe' if auto_on else '#f8fafc',
            fg='#1d4ed8' if auto_on else INK,
            relief='sunken' if auto_on else 'raised',
        )
        self.depth_exp_manual_btn.config(
            text='○ 手动' if auto_on else '◉ 手动',
            bg='#f8fafc' if auto_on else '#dbeafe',
            fg=INK if auto_on else '#1d4ed8',
            relief='raised' if auto_on else 'sunken',
        )

    def apply_depth_exposure(self):
        raw = self.depth_exp_entry.get().strip() if hasattr(self.depth_exp_entry, 'get') else ''
        try:
            mode, manual_val = validate_depthcal_exposure(self.depth_exp_mode, raw or None)
        except ValueError as exc:
            self.depth_exp_status.config(text=str(exc), fg=COLORS['red'])
            return None
        self.depth_exp_user_dirty = False
        self.depth_exp_status.config(text='正在应用曝光设置…', fg=INK)
        return self.depth_command('depthcal_exposure', {'mode': mode, 'manual_exposure_us': manual_val})

    def update_depthcal(self, main):
        data = main.get('depthcal') or {}
        phase = data.get('phase') or 'idle'
        self.depth_ack = main.get('command_seq', 0)
        self._check_page_command('depth', self.depth_ack)

        stage_text, stage_bg, stage_fg = depthcal_stage_info(data)
        self.depth_stage.config(text=stage_text, bg=stage_bg, fg=stage_fg)

        label = data.get('label')
        point = data.get('point')
        total = data.get('total')
        target_text = f"靶点：{label or '—'}" + (f" ({point}/{total})" if point and total else '')
        self.depth_target.config(text=target_text)

        z_val = data.get('z')
        self.depth_z.config(text=f"PLC Z：{z_val:.4f} mm" if isinstance(z_val, (int, float)) else 'PLC Z：— mm')
        self.depth_dof.config(text=depthcal_latest_dof_text(data))

        metric = data.get('metric') or 'tenengrad'
        method = data.get('method') or 'gauss'
        ratio = float(data.get('dof_ratio') or 0.8)
        self.depth_metric.config(text=f'指标：{metric} · {method} · dof_ratio={ratio:.2f}')
        fine_text, fine_notice = depthcal_fine_plan_text(data)
        self.depth_fine_info.config(text=fine_text)

        active = phase in ('center', 'exposure', 'coarse_up', 'coarse_overshoot', 'coarse', 'overshoot', 'fine', 'peak', 'plane') and not self.depth_pending and not data.get('abort_requested')
        self.depth_abort.config(state='normal' if active else 'disabled')

        exp_mode = data.get('exposure_mode') or 'auto'
        manual_us = data.get('manual_exposure_us')
        if not self.depth_exp_user_dirty and (exp_mode, manual_us) != self._depth_exp_synced:
            self._depth_exp_synced = (exp_mode, manual_us)
            self.select_depth_exposure_mode(exp_mode, user=False)
            if isinstance(manual_us, (int, float)):
                self.depth_exp_entry.delete(0, 'end')
                self.depth_exp_entry.insert(0, f'{float(manual_us):g}')
        self.depth_exp_live.config(text=depthcal_exposure_display(data))
        exp_err = data.get('exposure_error') or ''
        exp_warn = data.get('exposure_warning') or ''
        exp_msg = data.get('exposure_command_message') or ''
        if exp_err:
            self.depth_exp_status.config(text=exp_err, fg=COLORS['red'])
        elif exp_warn:
            self.depth_exp_status.config(text=exp_warn, fg=COLORS['yellow'])
        elif exp_msg:
            self.depth_exp_status.config(text=exp_msg, fg=COLORS['green'])
        elif not self.depth_pending:
            self.depth_exp_status.config(text='', fg=MUTED)

        default_msg = {
            'idle': '等待 PLC 160 发起料盘多点对焦测试',
            'done': '料盘多点对焦测试完成（已回 161）',
        }.get(phase, stage_text.replace('阶段：', ''))
        msg = getattr(self, 'depth_command_message', '') or ('上一条命令处理中' if self.depth_pending else data.get('message')) or default_msg
        if phase == 'done' and data.get('feasibility'):
            msg = '料盘多点对焦测试完成（已回 161）'
        msg_fg = COLORS['red'] if getattr(self, 'depth_command_message', '') or phase == 'failed' else COLORS['green'] if phase == 'done' else INK
        self.depth_message.config(text=msg, fg=msg_fg)

        if phase == 'failed' and any(data.get(k) for k in ('reason_code', 'what', 'where', 'hint')):
            lines = [f"错误码：{data.get('reason_code') or 'DEPTH-FAILED'}    位置：{data.get('where') or '下针深度标定'}"]
            lines.append(f"现象：{data.get('what') or data.get('message') or '—'}")
            lines.append(f"建议：{data.get('hint') or '检查菲林、相机和 PLC 实时坐标。'}")
            self.depth_failure_detail.config(text='\n'.join(lines))
            self.depth_failure_detail.grid()
        else:
            self.depth_failure_detail.config(text='')
            self.depth_failure_detail.grid_remove()

        notices = [n for n in (fine_notice, exp_warn) if n]
        if notices:
            self.depth_notice.config(text='提示：' + '；'.join(notices))
            self.depth_notice.grid()
        else:
            self.depth_notice.config(text='')
            self.depth_notice.grid_remove()

        feas_text, feas_bg, feas_fg = depthcal_feasibility_text(data)
        self.depth_feasibility.config(text=feas_text, bg=feas_bg, fg=feas_fg)
        dof_sum_text, dof_sum_bg, dof_sum_fg = depthcal_dof_summary_text(data)
        self.depth_dof_summary.config(text=dof_sum_text, bg=dof_sum_bg, fg=dof_sum_fg)
        arr_text, arr_bg, arr_fg = depthcal_arrival_summary_text(data)
        self.depth_arrival_summary.config(text=arr_text, bg=arr_bg, fg=arr_fg)

        plane = data.get('plane') or {}
        initial_plane = data.get('initial_plane') or {}
        if plane and 'a' in plane:
            eq_text = f"z = {plane['a']:+.5f}x {plane['b']:+.5f}y {plane['c']:+.4f}"
            res_tilt = f"{plane.get('max_residual', 0):.4f} mm  /  倾斜 {plane.get('tilt_mm_per_100mm', 0):.4f} mm/100mm"
            if plane.get('after_extra') and initial_plane:
                extra_text = f"{plane['after_extra']}（{initial_plane.get('max_residual', 0):.4f} → {plane.get('max_residual', 0):.4f} mm）"
            else:
                extra_text = '未触发加测（≤ 0.02 mm）'
            residuals = plane.get('residuals') or {}
            res_text = '  '.join(f'{k}:{v:+.4f}' for k, v in residuals.items()) or '—'
        else:
            eq_text = res_tilt = extra_text = res_text = '—'
        self.depth_plane_grid.set({
            '平面方程': eq_text,
            '最大 |残差| / 倾斜': res_tilt,
            '加测判定': extra_text,
            '各点残差 (mm)': res_text,
        })

        results = data.get('results') or []
        table_key = json.dumps(results, ensure_ascii=False, default=str)
        if table_key != self.cache.get('depth_table'):
            self.cache['depth_table'] = table_key
            self.depth_table.delete(*self.depth_table.get_children())
            for item in results:
                xy = item.get('xy') or []
                xy_text = f'({xy[0]:.2f}, {xy[1]:.2f})' if len(xy) >= 2 else '—'
                fits = item.get('fits') or []
                p_vals = []
                for i in range(3):
                    if i < len(fits) and isinstance(fits[i].get('z_peak'), (int, float)):
                        suffix = '(点不足)' if fits[i].get('quality_ok') is False else ''
                        p_vals.append(f"{fits[i]['z_peak']:.4f}{suffix}")
                    else:
                        p_vals.append('—')
                if len(fits) > 3:
                    p_vals[2] += f' (+{len(fits) - 3})'
                fwhms = [f['fwhm'] for f in fits if isinstance(f.get('fwhm'), (int, float))]
                r2s = [f['r2'] for f in fits if isinstance(f.get('r2'), (int, float))]
                z_mean = item.get('z_peak')
                z_rng = item.get('range')
                z_std = item.get('std')
                dof_mean = item.get('dof_mm')
                dof_rng = item.get('dof_range_mm')
                dof_uncov = int(item.get('dof_uncovered') or 0)
                if isinstance(dof_mean, (int, float)):
                    dof_mean_txt = f'{dof_mean:.3f}' + (f' ({dof_uncov}未覆盖)' if dof_uncov else '')
                elif dof_uncov > 0 or fits:
                    dof_mean_txt = '未覆盖'
                else:
                    dof_mean_txt = '—'
                dof_rng_txt = f'{dof_rng:.4f}' if isinstance(dof_rng, (int, float)) else '—'
                arr_cell, arr_warn = depthcal_point_arrival_text(item)
                self.depth_table.insert('', 'end', values=(
                    item.get('label') or '—',
                    xy_text,
                    *p_vals,
                    f'{z_mean:.4f}' if isinstance(z_mean, (int, float)) else '—',
                    f'{z_rng:.4f}' if isinstance(z_rng, (int, float)) else '—',
                    f'{z_std:.4f}' if isinstance(z_std, (int, float)) else '—',
                    f'{sum(fwhms)/len(fwhms):.3f}' if fwhms else '—',
                    dof_mean_txt,
                    dof_rng_txt,
                    f'{sum(r2s)/len(r2s):.4f}' if r2s else '—',
                    arr_cell,
                ), tags=('arrival_warn',) if arr_warn else ())

        self.draw_depthcal_curves(data)

    def draw_depthcal_curves(self, data):
        c = self.depth_curves
        w = max(240, c.winfo_width())
        h = max(160, c.winfo_height())
        curves = data.get('curves') or {}
        plan = list(data.get('plan') or [])
        for lbl in curves:
            if lbl not in plan:
                plan.append(lbl)
        results_by_label = {r.get('label'): r for r in (data.get('results') or []) if r.get('label')}
        current_label = data.get('label')
        dof_ratio = float(data.get('dof_ratio') or 0.8)
        current_fits = data.get('current_fits') or []
        key = (json.dumps([curves, plan, current_label, data.get('results'), current_fits, dof_ratio],
                          ensure_ascii=False, default=str), w, h)
        if self.cache.get('depth_curves') == key:
            return
        self.cache['depth_curves'] = key
        c.delete('all')

        # 顶部图例
        legend_items = [('粗扫(灰)', '#98a2b3', ()), ('细扫1', '#2563eb', ()), ('细扫2', '#168448', ()),
                        ('细扫3', '#d97706', ()), ('明显模糊', '#9333ea', ()),
                        ('峰位(竖线)', '#b42318', (4, 2)), (f'DOF({dof_ratio:.0%})', '#0284c7', (3, 2))]
        lx = 12
        for text, color, dash in legend_items:
            c.create_line(lx, 12, lx + 14, 12, fill=color, width=2, dash=dash)
            c.create_text(lx + 18, 12, text=text, anchor='w', fill=INK, font=(self.theme.body[0], 9))
            lx += 78 if '粗扫' in text else 84 if '模糊' in text else 88 if '竖线' in text else 86 if 'DOF' in text else 60

        if not plan:
            c.create_text(w / 2, h / 2, text='暂无清晰度扫描曲线，等待 PLC 160 发起测试',
                          fill=MUTED, font=self.theme.body)
            return

        pass_colors = {'1': '#2563eb', '2': '#168448', '3': '#d97706', '4': '#7c3aed', '5': '#0891b2'}
        for label, (x0, y0, x1, y1) in depthcal_subplot_layout(plan, w, h):
            is_current = (label == current_label) and data.get('phase') not in ('idle', 'done', 'failed')
            border = '#2563eb' if is_current else BORDER
            c.create_rectangle(x0, y0, x1, y1, outline=border, width=2 if is_current else 1, fill='#fcfcfd')
            res = results_by_label.get(label)
            title = f'{label}'
            if res and isinstance(res.get('z_peak'), (int, float)):
                dof_part = (f" DOF={res['dof_mm']:.2f}" if isinstance(res.get('dof_mm'), (int, float))
                            else ' DOF=未覆盖')
                title += f" 峰={res['z_peak']:.4f} Δ={res.get('range', 0):.4f}{dof_part}"
            elif is_current:
                title += '  (扫描中)'
            c.create_text(x0 + 8, y0 + 5, text=title, anchor='nw',
                          fill='#1d4ed8' if is_current else INK, font=(self.theme.body[0], 9, 'bold'))

            series = curves.get(label) or {}
            all_pts = [pt for pts in series.values() for pt in pts if len(pt) >= 2]
            if not all_pts:
                c.create_text((x0 + x1) / 2, (y0 + y1) / 2 + 6, text='等待扫描', fill=MUTED, font=(self.theme.body[0], 9))
                continue

            px0, py0, px1, py1 = x0 + 36, y0 + 24, x1 - 10, y1 - 20
            if px1 <= px0 + 10 or py1 <= py0 + 10:
                continue
            zs = [pt[0] for pt in all_pts]
            ys = [pt[1] for pt in all_pts]
            z_lo, z_hi = min(zs), max(zs)
            if z_hi - z_lo < 1e-6:
                z_lo -= 0.1
                z_hi += 0.1
            y_lo, y_hi = 0.0, max(max(ys) * 1.08, 1e-9)

            def to_px(zv, yv):
                sx = px0 + (px1 - px0) * (zv - z_lo) / (z_hi - z_lo)
                sy = py1 - (py1 - py0) * (yv - y_lo) / (y_hi - y_lo)
                return sx, sy

            fine_keys = sorted([k for k in series if k != 'coarse' and len(series.get(k) or []) >= 3],
                               key=lambda k: int(k) if k.isdigit() else 999)
            ref_pts = series.get(fine_keys[-1]) if fine_keys else series.get('coarse')
            curve_dof = depthcal_curve_dof(ref_pts, dof_ratio) if ref_pts else None
            y_thresh = curve_dof['threshold'] if curve_dof else (max(ys) * dof_ratio)
            z_dof_lo = curve_dof['dof_z_lower'] if curve_dof else None
            z_dof_hi = curve_dof['dof_z_upper'] if curve_dof else None
            fit_pool = (res.get('fits') if res else None) or (current_fits if label == current_label else None) or []
            covered_fits = [f for f in fit_pool if f.get('dof_covered') and isinstance(f.get('dof_z_lower'), (int, float))
                            and isinstance(f.get('dof_z_upper'), (int, float))]
            if covered_fits:
                latest_fit = covered_fits[-1]
                z_dof_lo = latest_fit['dof_z_lower']
                z_dof_hi = latest_fit['dof_z_upper']
                fit_pts = series.get(str(latest_fit.get('repeat', '')), ref_pts)
                if fit_pts:
                    y_thresh = max(pt[1] for pt in fit_pts if len(pt) >= 2) * dof_ratio
            elif fit_pool:
                latest_fit = fit_pool[-1]
                if latest_fit.get('dof_covered') is False:
                    z_dof_lo = latest_fit.get('dof_z_lower')
                    z_dof_hi = latest_fit.get('dof_z_upper')

            sy_thr = None
            if y_lo <= y_thresh <= y_hi:
                _, sy_thr = to_px(z_lo, y_thresh)
                if (z_dof_lo is not None and z_dof_hi is not None
                        and z_lo <= z_dof_lo < z_dof_hi <= z_hi):
                    sx_lo, _ = to_px(z_dof_lo, y_lo)
                    sx_hi, _ = to_px(z_dof_hi, y_lo)
                    c.create_rectangle(sx_lo, sy_thr, sx_hi, py1, fill='#e0f2fe', outline='')

            c.create_line(px0, py0, px0, py1, fill='#cbd5e1')
            c.create_line(px0, py1, px1, py1, fill='#cbd5e1')
            c.create_text(px0 - 3, py0, text=f'{y_hi:.2f}', anchor='ne', fill=MUTED, font=(self.theme.body[0], 8))
            c.create_text(px0 - 3, py1, text='0', anchor='se', fill=MUTED, font=(self.theme.body[0], 8))
            c.create_text(px0, py1 + 3, text=f'{z_lo:.2f}', anchor='nw', fill=MUTED, font=(self.theme.body[0], 8))
            c.create_text(px1, py1 + 3, text=f'{z_hi:.2f}', anchor='ne', fill=MUTED, font=(self.theme.body[0], 8))

            ordered_passes = ['coarse'] + sorted([k for k in series if k != 'coarse'])
            for p_name in ordered_passes:
                pts = sorted((pt for pt in series.get(p_name, []) if len(pt) >= 2), key=lambda item: item[0])
                if not pts:
                    continue
                color = '#98a2b3' if p_name == 'coarse' else pass_colors.get(p_name, '#2563eb')
                width_px = 1.5 if p_name == 'coarse' else 2.0
                coords = []
                for zv, yv in pts:
                    sx, sy = to_px(zv, yv)
                    coords.extend((sx, sy))
                    if p_name == 'coarse':
                        c.create_oval(sx - 2, sy - 2, sx + 2, sy + 2, fill=color, outline=color)
                if len(coords) >= 4:
                    c.create_line(*coords, fill=color, width=width_px)

            if sy_thr is not None:
                c.create_line(px0, sy_thr, px1, sy_thr, fill='#38bdf8', dash=(4, 2), width=1.5)
                for z_cross in (z_dof_lo, z_dof_hi):
                    if isinstance(z_cross, (int, float)) and z_lo <= z_cross <= z_hi:
                        sx_c, _ = to_px(z_cross, y_lo)
                        c.create_line(sx_c, max(py0, sy_thr - 8), sx_c, py1, fill='#0284c7', width=1.5, dash=(2, 2))
                        c.create_oval(sx_c - 2.5, sy_thr - 2.5, sx_c + 2.5, sy_thr + 2.5,
                                      fill='#0284c7', outline='#0284c7')

            bounds = (res.get('coarse_bounds') if res else None) or (
                data.get('coarse_bounds') if label == current_label else None
            ) or (res.get('fine_plan') if res else None) or (
                data.get('fine_plan') if label == current_label else None
            )
            if isinstance(bounds, dict):
                coarse_pts = series.get('coarse') or []
                for z_key, flag_key in (('blur_low_z', 'blur_low_reached'), ('blur_high_z', 'blur_high_reached')):
                    bz = bounds.get(z_key)
                    if bounds.get(flag_key) and isinstance(bz, (int, float)) and z_lo <= bz <= z_hi:
                        if coarse_pts:
                            nearest = min(coarse_pts, key=lambda pt: abs(pt[0] - bz))
                            sx_b, sy_b = to_px(nearest[0], nearest[1])
                            c.create_polygon(sx_b, sy_b - 4.5, sx_b + 4.5, sy_b, sx_b, sy_b + 4.5, sx_b - 4.5, sy_b,
                                             fill='#9333ea', outline='#ffffff', width=1)
                        else:
                            sx_b, _ = to_px(bz, y_lo)
                            c.create_line(sx_b, py0, sx_b, py1, fill='#9333ea', dash=(2, 2), width=1)

            if res:
                for idx, fit in enumerate(res.get('fits') or [], 1):
                    pk = fit.get('z_peak')
                    if isinstance(pk, (int, float)) and z_lo <= pk <= z_hi:
                        sx, _ = to_px(pk, y_lo)
                        c.create_line(sx, py0, sx, py1, fill=pass_colors.get(str(idx), '#b42318'), dash=(3, 2), width=1.5)
                mean_pk = res.get('z_peak')
                if isinstance(mean_pk, (int, float)) and z_lo <= mean_pk <= z_hi:
                    sx, _ = to_px(mean_pk, y_lo)
                    c.create_line(sx, py0, sx, py1, fill='#b42318', width=2)

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
        missing_z = pickacc_z_missing_tools((main.get('pickacc') or {}).get('existing'))
        if missing_z:
            self.pick_z_banner.config(text='⚠ ' + '；'.join(f'工具{t} 未做下针深度标定' for t in missing_z) + '（生产与菲林模式将回退使用配置 pick_depth_z）')
            self.pick_z_banner.grid(row=3, column=0, sticky='ew', padx=6, pady=(0, 4))
        else:
            self.pick_z_banner.grid_remove()
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
        page, self.last_mode, self.last_depth_session = next_auto_page(
            main, self.last_mode, getattr(self, 'last_depth_session', None),
        )
        if page is not None:
            self.select_page(page)
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
        self.update_depthcal(main)
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
        self.tip_plc_id.config(text=f"40905={data.get('plc_tool_id', '—')}")
        self.tip_heading.config(text=f'当前：工具{tool}（PLC 功能码指定）' if tool else '当前：等待 PLC 指定', bg='#2563eb' if tool == 1 else '#d56713' if tool == 2 else COLORS['gray'])
        warning = data.get('camera_settings_warning')
        self.tip_message.config(text=getattr(self, 'tip_command_message', '') or ('上一条命令处理中' if self.tip_pending else warning or data.get('message')) or '等待 PLC 100/110', fg=COLORS['red'] if getattr(self, 'tip_command_message', '') else COLORS['yellow'] if warning else COLORS['red'] if phase in ('limit','fail') else COLORS['green'] if phase == 'done' else INK)
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
