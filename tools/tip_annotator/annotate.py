#!/usr/bin/python3
"""本地针尖标注工具。"""
import argparse
import sys
import tkinter as tk
from tkinter import ttk, messagebox, font
from pathlib import Path
from PIL import Image, ImageTk
from core import load, save, paths

COLORS = {1: '#ef4444', 2: '#22c55e', 3: '#3b82f6'}

class App:
    def __init__(self, root, directory):
        self.root = root
        self.files = sorted(p for p in Path(directory).rglob('*') if p.is_file() and p.suffix.lower() == '.png' and 'annotations' not in p.relative_to(directory).parts)
        if not self.files: raise ValueError('目录中没有 PNG 图片')
        self.directory = Path(directory); self.index = 0; self.doc = None; self.dirty = False
        self.zoom = 1.; self.ox = self.oy = 0; self.pan = None; self.stroke = []; self.poly = []; self.result = {}
        self.cls = tk.IntVar(value=1); self.tool = tk.StringVar(value='brush'); self.size = tk.IntVar(value=8)
        self.alpha = tk.IntVar(value=45); self.visible = tk.BooleanVar(value=True); self.threshold = tk.IntVar(value=127); self.enabled = tk.BooleanVar(value=False)
        root.title('针尖标注工具'); root.geometry('1400x850')
        families = set(font.families(root))
        for name in ('Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'Microsoft YaHei', 'SimHei'):
            if name in families:
                font.nametofont('TkDefaultFont').configure(family=name, size=10); font.nametofont('TkTextFont').configure(family=name, size=10); break
        main = ttk.Frame(root); main.pack(fill='both', expand=True)
        side = ttk.Frame(main, width=255); side.pack(side='left', fill='y'); side.pack_propagate(False)
        ttk.Label(side, text='图片列表（✓ 已标注）').pack(anchor='w', padx=6, pady=4)
        self.listbox = tk.Listbox(side, exportselection=False); self.listbox.pack(fill='both', expand=True, padx=5)
        self.listbox.bind('<<ListboxSelect>>', self.select)
        ttk.Button(side, text='复制上一张标注到本张', command=self.copy_previous).pack(fill='x', padx=5, pady=5)
        right = ttk.Frame(main); right.pack(side='left', fill='both', expand=True)
        bar = ttk.Frame(right); bar.pack(fill='x')
        for text, value in [('黑胶 (1)', 1), ('针身 (2)', 2), ('针尖 (3)', 3)]: ttk.Radiobutton(bar, text=text, variable=self.cls, value=value).pack(side='left')
        for text, value in [('画笔 B', 'brush'), ('橡皮 E', 'erase'), ('多边形 P', 'polygon'), ('人工点 T', 'manual')]: ttk.Radiobutton(bar, text=text, variable=self.tool, value=value, command=self.cancel_polygon).pack(side='left')
        ttk.Button(bar, text='清除人工点', command=self.clear_manual).pack(side='left')
        controls = ttk.Frame(right); controls.pack(fill='x')
        ttk.Button(controls, text='适应窗口', command=self.fit).pack(side='left')
        ttk.Label(controls, text='笔尺寸').pack(side='left'); ttk.Spinbox(controls, from_=1, to=60, width=4, textvariable=self.size).pack(side='left')
        ttk.Checkbutton(controls, text='阈值辅助', variable=self.enabled, command=self.settings).pack(side='left')
        ttk.Label(controls, text='阈值').pack(side='left'); ttk.Scale(controls, from_=0, to=255, variable=self.threshold, command=lambda _: self.settings()).pack(side='left', fill='x', expand=True)
        ttk.Label(controls, textvariable=self.threshold, width=4).pack(side='left')
        ttk.Checkbutton(controls, text='显示叠加 H', variable=self.visible, command=self.render).pack(side='left')
        ttk.Label(controls, text='透明度').pack(side='left'); ttk.Scale(controls, from_=0, to=100, variable=self.alpha, command=lambda _: self.render()).pack(side='left', fill='x', expand=True)
        self.canvas = tk.Canvas(right, bg='#242424', highlightthickness=0); self.canvas.pack(fill='both', expand=True)
        self.status = tk.StringVar(); ttk.Label(right, textvariable=self.status, anchor='w').pack(fill='x')
        self.canvas.bind('<Configure>', lambda e: self.render()); self.canvas.bind('<MouseWheel>', self.wheel)
        self.canvas.bind('<Button-4>', lambda e: self.wheel(e, 1)); self.canvas.bind('<Button-5>', lambda e: self.wheel(e, -1))
        self.canvas.bind('<ButtonPress-1>', self.down); self.canvas.bind('<B1-Motion>', self.move); self.canvas.bind('<ButtonRelease-1>', self.up)
        self.canvas.bind('<Double-Button-1>', self.double)
        for b in (2, 3):
            self.canvas.bind(f'<ButtonPress-{b}>', self.pan_start); self.canvas.bind(f'<B{b}-Motion>', self.pan_move)
        self.canvas.bind('<Motion>', self.motion)
        for key, val in [('b','brush'), ('e','erase'), ('p','polygon'), ('t','manual')]: root.bind(key, lambda e, v=val: self.set_tool(v))
        for i in (1, 2, 3): root.bind(str(i), lambda e, n=i: self.cls.set(n))
        root.bind('h', lambda e: (self.visible.set(not self.visible.get()), self.render()))
        root.bind('[', lambda e: self.size.set(max(1, self.size.get()-1))); root.bind(']', lambda e: self.size.set(min(60, self.size.get()+1)))
        root.bind('<Control-z>', lambda e: self.history(False)); root.bind('<Control-y>', lambda e: self.history(True)); root.bind('<Control-s>', lambda e: self.save(force=True))
        root.bind('<Left>', lambda e: self.change(self.index-1)); root.bind('<Right>', lambda e: self.change(self.index+1))
        root.bind('<Return>', lambda e: self.finish_poly()); root.bind('<Escape>', lambda e: self.cancel_polygon())
        root.protocol('WM_DELETE_WINDOW', self.close); self.refresh_list(); self.open_image(0)
        root.after(150, self.fit)

    def refresh_list(self):
        self.listbox.delete(0, 'end')
        for p in self.files: self.listbox.insert('end', ('✓ ' if paths(p)[0].exists() else '○ ') + str(p.relative_to(self.directory)))
        self.listbox.selection_set(self.index); self.listbox.see(self.index)
    def select(self, event):
        selected = self.listbox.curselection()
        if selected and selected[0] != self.index: self.change(selected[0])
    def open_image(self, i):
        self.index = i; self.doc = load(self.files[i]); self.threshold.set(self.doc.threshold); self.enabled.set(self.doc.threshold_enabled)
        self.poly = []; self.dirty = False; self.derive(); self.refresh_list(); self.fit()
    def change(self, i):
        if 0 <= i < len(self.files) and i != self.index:
            self.save(); self.open_image(i)
    def save(self, force=False):
        if self.doc and (self.dirty or force):
            self.settings(); save(self.files[self.index], self.doc); self.dirty = False; self.refresh_list()
    def close(self):
        try: self.save(); self.root.destroy()
        except Exception as e: messagebox.showerror('保存失败', str(e))
    def settings(self):
        if self.doc:
            v = (self.enabled.get(), self.threshold.get())
            if v != (self.doc.threshold_enabled, self.doc.threshold): self.dirty = True
            self.doc.threshold_enabled, self.doc.threshold = v
    def copy_previous(self):
        if self.index == 0: messagebox.showinfo('提示', '没有上一张图片'); return
        previous = load(self.files[self.index-1])
        if previous.labels.size != self.doc.labels.size: messagebox.showerror('无法复制', '图片尺寸不同'); return
        self.doc.undo_stack.append(self.doc.labels.copy()); self.doc.undo_stack = self.doc.undo_stack[-30:]; self.doc.redo_stack.clear()
        self.doc.labels = previous.labels.copy(); self.doc.manual = previous.manual; self.dirty = True; self.derive(); self.render()
    def derive(self): self.result = self.doc.derive()
    def fit(self):
        if not self.doc: return
        w, h = self.doc.image.size; cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 2 or ch < 2: return
        self.zoom = min(cw/w, ch/h, 16); self.ox = (cw-w*self.zoom)/2; self.oy = (ch-h*self.zoom)/2; self.render()
    def render(self):
        if not self.doc or self.canvas.winfo_width() < 2: return
        self.canvas.delete('all'); w, h = self.doc.image.size; z = self.zoom
        # Render only the visible crop, keeping 16x zoom memory bounded.
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        x0 = max(0, int(-self.ox/z)); y0 = max(0, int(-self.oy/z)); x1 = min(w, int((cw-self.ox)/z)+2); y1 = min(h, int((ch-self.oy)/z)+2)
        if x1 > x0 and y1 > y0:
            box = (x0, y0, x1, y1); img = self.doc.image.crop(box).convert('RGB')
            if self.visible.get():
                labels = self.doc.labels.crop(box)
                for v, color in COLORS.items():
                    mask = labels.point(lambda p, v=v: int(self.alpha.get()*2.55) if p == v else 0)
                    img.paste(color, mask=mask)
            img = img.resize((max(1, round((x1-x0)*z)), max(1, round((y1-y0)*z))), Image.Resampling.NEAREST)
            self.photo = ImageTk.PhotoImage(img); self.canvas.create_image(self.ox+x0*z, self.oy+y0*z, image=self.photo, anchor='nw')
        if self.result.get('status') == 'ok':
            axis = self.result['axis']; x,y = axis['point']; dx,dy = axis['direction']; length = max(w,h)*2
            self.canvas.create_line(self.ox+(x-dx*length)*z, self.oy+(y-dy*length)*z, self.ox+(x+dx*length)*z, self.oy+(y+dy*length)*z, fill='yellow', dash=(5,4))
            self.marker(self.result['farthest_tip'], 'cyan', False); self.marker(self.result['tool_tip'], 'yellow', True)
        if self.doc.manual: self.marker(self.doc.manual, '#ff80ff', True)
        if self.poly:
            coords = [v for x,y in self.poly for v in (self.ox+x*z, self.oy+y*z)]
            if len(coords) >= 4: self.canvas.create_line(*coords, fill='white', width=2)
        self.motion_status(None)
    def marker(self, p, color, cross):
        x,y = self.ox+p[0]*self.zoom, self.oy+p[1]*self.zoom
        if cross:
            self.canvas.create_line(x-8,y,x+8,y,fill=color,width=2); self.canvas.create_line(x,y-8,x,y+8,fill=color,width=2)
        else: self.canvas.create_oval(x-4,y-4,x+4,y+4,outline=color,width=2)
    def motion_status(self, e):
        pointer = ''
        if e:
            x,y = self.point(e); ix,iy = round(x),round(y)
            if 0 <= ix < self.doc.image.width and 0 <= iy < self.doc.image.height: pointer = f'坐标 ({x:.2f}, {y:.2f}) 灰度 {self.doc.image.getpixel((ix,iy))}  |  '
        r = self.result; tip = r.get('tool_tip')
        self.status.set(pointer + (f'推导工具尖 ({tip[0]:.2f}, {tip[1]:.2f})' if tip else r.get('status','')) + f'  |  缩放 {self.zoom:.2f}×')
    def motion(self, e): self.motion_status(e)
    def point(self, e): return ((e.x-self.ox)/self.zoom, (e.y-self.oy)/self.zoom)
    def wheel(self, e, direction=None):
        factor = 1.25 if (direction or e.delta) > 0 else 1/1.25
        x,y = self.point(e); self.zoom = max(.02, min(16, self.zoom*factor)); self.ox = e.x-x*self.zoom; self.oy = e.y-y*self.zoom; self.render()
    def pan_start(self,e): self.pan = (e.x,e.y,self.ox,self.oy)
    def pan_move(self,e): self.ox = self.pan[2]+e.x-self.pan[0]; self.oy = self.pan[3]+e.y-self.pan[1]; self.render()
    def down(self,e):
        p = self.point(e); tool = self.tool.get()
        if tool == 'manual': self.doc.manual = list(p); self.dirty = True; self.render()
        elif tool == 'polygon': self.poly.append(p); self.render()
        else: self.stroke = [p]
    def move(self,e):
        if self.stroke: self.stroke.append(self.point(e))
    def up(self,e):
        if self.stroke:
            self.stroke.append(self.point(e)); self.doc.brush(self.stroke, max(1,min(60,self.size.get())), 0 if self.tool.get() == 'erase' else self.cls.get(), self.enabled.get())
            self.stroke = []; self.dirty = True; self.derive(); self.render()
    def double(self,e):
        if self.tool.get() == 'polygon': self.finish_poly(); return 'break'
    def finish_poly(self):
        if len(self.poly) >= 3:
            self.doc.polygon(self.poly, self.cls.get(), self.enabled.get()); self.dirty = True; self.derive()
        self.poly = []; self.render()
    def cancel_polygon(self): self.poly = []; self.render()
    def set_tool(self, tool): self.tool.set(tool); self.cancel_polygon()
    def clear_manual(self): self.doc.manual = None; self.dirty = True; self.render()
    def history(self, redo):
        (self.doc.redo if redo else self.doc.undo)(); self.dirty = True; self.derive(); self.render()


def main():
    parser = argparse.ArgumentParser(description='本地针尖标注工具'); parser.add_argument('directory', type=Path); args = parser.parse_args()
    if not args.directory.is_dir(): parser.error('图片目录不存在')
    root = tk.Tk()
    try: App(root, args.directory)
    except Exception as e: messagebox.showerror('启动失败', str(e)); root.destroy(); return 1
    root.mainloop(); return 0

if __name__ == '__main__': sys.exit(main())
