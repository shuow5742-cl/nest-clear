"""针尖点击确认及可无窗口驱动的坐标状态机。"""
from dataclasses import dataclass
import time
import os
import multiprocessing as mp
import socket
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from nest_clear.guard import GuardError

_DISPLAY_ERROR = "无法打开针尖确认窗口（无显示环境），请在工控机本地桌面运行，或加 --no-confirm（仅测试用）"


def _check_display():
    """在调用 Qt 前检查 X11 连接，避免无显示环境触发 Qt 进程中止。"""
    if not sys.platform.startswith("linux"):
        return
    display = os.environ.get("DISPLAY", "")
    try:
        host, number = display.rsplit(":", 1)
        number = int(number.split(".", 1)[0])
        if not host or host == "unix":
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            address = f"/tmp/.X11-unix/X{number}"
        else:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            address = (host, 6000 + number)
        with sock:
            sock.settimeout(.5)
            sock.connect(address)
    except (ValueError, OSError):
        raise GuardError("TIP-CLICK-DISPLAY", _DISPLAY_ERROR, "针尖确认", "检查本地桌面和显示连接。") from None


@dataclass(frozen=True)
class ClickResult:
    point_uv: tuple[float, float] | None
    source: str
    zoom: float
    elapsed_s: float


class ClickState:
    def __init__(self, image_size, auto_point=None, viewport=(1000, 700), zoom=8.0):
        self.width, self.height = image_size
        self.viewport = viewport
        self.auto_point = tuple(auto_point) if auto_point is not None else None
        self.point = self.auto_point
        self.center = list(self.point or ((self.width - 1) / 2, (self.height - 1) / 2))
        self.zoom = max(2., min(32., float(zoom)))
        self.full = False
        self.source = 'auto'
        self.done = False

    def transform(self):
        w, h = self.viewport
        scale = min(w / self.width, h / self.height) if self.full else self.zoom
        return scale, self.center[0] - w / (2 * scale), self.center[1] - h / (2 * scale)

    def click(self, x, y):
        scale, left, top = self.transform()
        u, v = left + x / scale, top + y / scale
        if 0 <= u <= self.width - 1 and 0 <= v <= self.height - 1:
            self.point = (float(u), float(v))
            self.source = 'manual'

    def pan(self, dx, dy):
        scale, _, _ = self.transform()
        self.center[0] = max(0., min(self.width - 1., self.center[0] - dx / scale))
        self.center[1] = max(0., min(self.height - 1., self.center[1] - dy / scale))

    def key(self, key):
        if key in (10, 13) and self.point is not None:
            self.done = True
        elif key == 27:
            self.point, self.source, self.done = None, 'cancel', True
        elif key in (ord('r'), ord('R')):
            self.point, self.source = self.auto_point, 'auto'
            if self.point is not None:
                self.center = list(self.point)
        elif key == 32:
            self.full = not self.full
            if self.full:
                self.center = [(self.width - 1) / 2, (self.height - 1) / 2]
            elif self.point is not None:
                self.center = list(self.point)
        elif key in (ord('+'), ord('=')):
            self.zoom = min(32., self.zoom * 2)
        elif key in (ord('-'), ord('_')):
            self.zoom = max(2., self.zoom / 2)


def confirm_tip(image, detection=None, title='针尖确认'):
    """在隔离进程运行 highgui，Qt 初始化异常退出不会终止建基准工具。"""
    _check_display()
    if mp.current_process().name != "MainProcess":
        return _confirm_local(image, detection, title)
    parent, child = mp.get_context("spawn").Pipe(False)
    process = mp.get_context("spawn").Process(target=_window_process, args=(child, image, detection, title))
    try:
        process.start()
        child.close()
        while not parent.poll(.1):
            if not process.is_alive():
                raise GuardError("TIP-CLICK-DISPLAY", _DISPLAY_ERROR, "针尖确认", "检查本地桌面和显示连接。")
        try:
            result = parent.recv()
        except EOFError:
            raise GuardError("TIP-CLICK-DISPLAY", _DISPLAY_ERROR, "针尖确认", "检查本地桌面和显示连接。") from None
        if isinstance(result, str):
            raise GuardError("TIP-CLICK-DISPLAY", _DISPLAY_ERROR, "针尖确认", result)
        return result
    finally:
        parent.close()
        child.close()
        if process.is_alive():
            process.terminate()
        if process.pid is not None:
            process.join()


def _window_process(connection, image, detection, title):
    try:
        connection.send(_confirm_local(image, detection, title))
    except Exception as exc:
        connection.send(str(exc))
    finally:
        connection.close()


def _confirm_local(image, detection, title):
    """将 highgui 错误转换为现场可读的错误。"""
    try:
        return _confirm_tip(image, detection, title)
    except cv2.error as exc:
        raise GuardError("TIP-CLICK-DISPLAY", _DISPLAY_ERROR, "针尖确认", "检查本地桌面和显示连接。") from exc


def _confirm_tip(image, detection=None, title='针尖确认'):
    """在调用进程的主线程运行 highgui；返回用户选择。"""
    start = time.monotonic()
    point = getattr(detection, 'point_uv', detection)
    state = ClickState((image.shape[1], image.shape[0]), point)
    window = str(title)
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    dragging = [None, False]
    def mouse(event, x, y, flags, param):
        y -= 45
        if event == cv2.EVENT_LBUTTONDOWN:
            dragging[:] = [(x, y), False]
        elif event == cv2.EVENT_MOUSEMOVE and dragging[0] is not None:
            old = dragging[0]
            if abs(x-old[0]) + abs(y-old[1]) > 0:
                state.pan(x-old[0], y-old[1]); dragging[:] = [(x,y), True]
        elif event == cv2.EVENT_LBUTTONUP:
            if not dragging[1] and y >= 0: state.click(x, y)
            dragging[:] = [None, False]
        elif event == cv2.EVENT_MOUSEWHEEL:
            state.key(ord('+') if flags > 0 else ord('-'))
    cv2.setMouseCallback(window, mouse)
    try:
        while not state.done:
            w, h = state.viewport
            scale, left, top = state.transform()
            matrix = np.array([[scale, 0, -left*scale], [0, scale, -top*scale]], dtype=np.float32)
            canvas = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_NEAREST)
            if canvas.ndim == 2: canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
            if not state.full and scale >= 8:
                for u in range(max(0, int(left)), min(state.width, int(left+w/scale)+2)):
                    x = round((u-left)*scale); cv2.line(canvas, (x,0), (x,h-1), (80,80,80), 1)
                for v in range(max(0, int(top)), min(state.height, int(top+h/scale)+2)):
                    y = round((v-top)*scale); cv2.line(canvas, (0,y), (w-1,y), (80,80,80), 1)
            if state.point:
                x, y = (round((state.point[0]-left)*scale), round((state.point[1]-top)*scale))
                cv2.drawMarker(canvas, (x,y), (0,0,255), cv2.MARKER_CROSS, 20, 2)
            header = Image.new('RGB', (w,45), (30,30,30))
            font = None
            for path in ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'):
                try: font = ImageFont.truetype(path, 18); break
                except OSError: pass
            if font:
                ImageDraw.Draw(header).text((8,10), '点击选点  拖动平移  滚轮/±缩放  空格全图  R恢复  Enter确认  Esc取消', font=font, fill='white')
            cv2.imshow(window, np.vstack((cv2.cvtColor(np.asarray(header), cv2.COLOR_RGB2BGR), canvas)))
            state.key(cv2.waitKey(30) & 0xff)
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1: state.key(27)
    finally:
        cv2.destroyWindow(window)
    return ClickResult(state.point, state.source, state.zoom, time.monotonic()-start)
