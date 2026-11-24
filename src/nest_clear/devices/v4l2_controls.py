"""Small Linux V4L2 control interface; ioctl can be injected for tests."""
import ctypes
import fcntl
import os

class Control(ctypes.Structure):
    _fields_ = [('id', ctypes.c_uint32), ('value', ctypes.c_int32)]

class QueryControl(ctypes.Structure):
    _fields_ = [('id', ctypes.c_uint32), ('type', ctypes.c_uint32), ('name', ctypes.c_char * 32),
                ('minimum', ctypes.c_int32), ('maximum', ctypes.c_int32), ('step', ctypes.c_int32),
                ('default_value', ctypes.c_int32), ('flags', ctypes.c_uint32), ('reserved', ctypes.c_uint32 * 2)]

VIDIOC_G_CTRL = 0xC008561B
VIDIOC_S_CTRL = 0xC008561C
VIDIOC_QUERYCTRL = 0xC0445624
IDS = {'auto_exposure': 0x009a0901, 'exposure': 0x009a0902, 'dynamic_framerate': 0x009a0903,
       'white_balance_auto': 0x0098090c, 'white_balance_temperature': 0x0098091a,
       'gain': 0x00980913, 'sharpness': 0x0098091b, 'backlight_compensation': 0x0098091c,
       'gamma': 0x00980910, 'brightness': 0x00980900, 'contrast': 0x00980901,
       'saturation': 0x00980902, 'hue': 0x00980903, 'power_line_frequency': 0x00980918}
DEFAULTS = {'width': 1920, 'height': 1080, 'auto_exposure': 1, 'exposure': 100,
            'dynamic_framerate': 0, 'white_balance_auto': 0, 'white_balance_temperature': 4600,
            'sharpness': 0, 'backlight_compensation': 0, 'gain': 0, 'gamma': 100,
            'brightness': None, 'contrast': None, 'saturation': None, 'hue': None,
            'power_line_frequency': 1}

class V4L2Controls:
    def __init__(self, path, ioctl=fcntl.ioctl):
        self.fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
        self.ioctl = ioctl
    def close(self):
        os.close(self.fd)
    def query(self, name):
        q = QueryControl(id=IDS[name]); self.ioctl(self.fd, VIDIOC_QUERYCTRL, q)
        if q.flags & 1: raise ValueError(f'{name} 控制项已禁用')
        return q
    def get(self, name):
        c = Control(id=IDS[name]); self.ioctl(self.fd, VIDIOC_G_CTRL, c); return c.value
    def set(self, name, value):
        q = self.query(name)
        if type(value) is not int or not q.minimum <= value <= q.maximum or (q.step > 0 and (value-q.minimum) % q.step):
            raise ValueError(f'{name}={value} 超出设备范围 {q.minimum}–{q.maximum}，步长 {q.step}')
        c = Control(id=IDS[name], value=value); self.ioctl(self.fd, VIDIOC_S_CTRL, c)
    def apply(self, settings, settle=None):
        expected = {}
        # Disable automatic modes before their dependent manual controls, regardless of mapping order.
        names = [name for name in ("auto_exposure", "white_balance_auto") if name in settings]
        names += [name for name in settings if name not in names]
        for name in names:
            value = settings[name]
            if name not in IDS: continue
            if value is None: value = self.query(name).default_value
            self.set(name, value); expected[name] = value
        if settle: settle()
        actual = {name: self.get(name) for name in expected}
        differences = [f'{name} 期望 {value}，实际 {actual[name]}' for name, value in expected.items() if actual[name] != value]
        if differences: raise ValueError('控制项回读不一致：' + '；'.join(differences))
        return actual
