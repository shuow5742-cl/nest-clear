import ctypes
import os
import cv2
import numpy as np
import pytest
from nest_clear.devices.v4l2_controls import V4L2Controls, VIDIOC_QUERYCTRL, VIDIOC_G_CTRL, VIDIOC_S_CTRL
from nest_clear.devices.tooltip_exposure import auto_exposure
from nest_clear.devices.tooltip_camera_store import save_measurement, load_settings, compare_settings
from nest_clear.devices.calib_cameras import CalibCamera
from nest_clear.tooltip.focus_metrics import focus_metrics, PeakTracker


def test_controls(tmp_path):
    path = tmp_path/'device'; path.touch(); values = {}
    def ioctl(fd, op, c):
        if op == VIDIOC_QUERYCTRL: c.minimum=0; c.maximum=100; c.step=1; c.default_value=2
        elif op == VIDIOC_S_CTRL: values[c.id]=c.value
        elif op == VIDIOC_G_CTRL: c.value=values[c.id]
    controls = V4L2Controls(str(path), ioctl)
    assert controls.apply({'gain': 4}) == {'gain': 4}
    with pytest.raises(ValueError, match='超出设备范围'): controls.set('gain', 101)
    def mismatch(fd, op, c):
        ioctl(fd, op, c)
        if op == VIDIOC_G_CTRL: c.value += 1
    controls.ioctl = mismatch
    with pytest.raises(ValueError, match='期望 4，实际 5'): controls.apply({'gain': 4})
    controls.close()

class Fake:
    def __init__(self, scale): self.scale = scale; self.exposure = 1
    def set_exposure(self, value): self.exposure = value
    def capture(self):
        b = min(255, self.exposure*self.scale)
        a = np.full((200,200), b, np.uint8); a[20:160,90:110] = b*.2
        return a

def test_exposure():
    result = auto_exposure(Fake(.5))
    assert 200 <= result['p99'] <= 235 and len(result['iterations']) <= 12
    with pytest.raises(ValueError, match='仍偏暗'): auto_exposure(Fake(.001))
    with pytest.raises(ValueError, match='仍过曝'): auto_exposure(Fake(255))

def test_focus():
    widths=[]; tens=[]
    for sigma in (.5, 1, 2, 4):
        im=np.full((240,240), 220, np.uint8); im[:, :120]=20
        im=cv2.GaussianBlur(im, (0,0), sigma)
        m=focus_metrics(im, (120,120)); widths.append(m['edge_width_px']); tens.append(m['tenengrad'])
        assert abs(m['edge_width_px']-2.563*sigma)/(2.563*sigma)<.15
    assert widths == sorted(widths) and tens == sorted(tens, reverse=True)
    tracker=PeakTracker(10); tracker.update(10,(0,0)); assert tracker.update(5,(1,0))['percent']==50
    assert tracker.update(3,(20,0))['percent']==100
    tracker.reset(); assert tracker.peak is None

def test_store_and_placeholder(tmp_path):
    path=tmp_path/'tooltip_cameras.yaml'; result=dict(exposure=42,p99=210,saturated_fraction=0.,contrast=100.)
    save_measurement(path,'front',{'device':'/dev/test','controls':{}},result)
    save_measurement(path,'front',{'device':'/dev/test','controls':{}},result)
    assert load_settings(path)['cameras']['front']['exposure']==42
    assert len(list((tmp_path/'history').iterdir()))==1
    assert compare_settings({'x':1},{'x':2})[0]['field']=='x'
    with pytest.raises(Exception, match='辅助对刀相机未配置'):
        CalibCamera('side', {'backend':'uvc','device':'待接入：插好后填路径'}).open()

def test_focus_exposure_invariance():
    original=np.full((240,240), 120, np.uint8); original[:, :120]=20
    original=cv2.GaussianBlur(original,(0,0),2)
    brighter=(original.astype(np.uint16)*2).astype(np.uint8)
    a=focus_metrics(original)['tenengrad']; b=focus_metrics(brighter)['tenengrad']
    assert abs(a-b)/a < .1

def test_config_devices_and_controls(tmp_path):
    import yaml
    from pathlib import Path
    from nest_clear.config import load_config
    data=yaml.safe_load(Path('config/nest_clear.yaml').read_text())
    data['tool_calibration']['directory']=str(tmp_path/'calibration')
    data['vision']['calibration']=str(tmp_path/'calibration'/'camera.yaml')
    data['tool_calibration']['cameras']['front']['device']=0
    data['tool_calibration']['cameras']['side']['device']='/dev/v4l/by-path/example-video-index0'
    path=tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(data))
    cfg=load_config(path)
    assert cfg.tool_calibration.cameras['front']['device']==0
    assert cfg.tool_calibration.cameras['side']['device'].startswith('/dev/v4l/by-path/')
    data['tool_calibration']['camera_controls']={'white_balance_temperature': 6501}
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(Exception, match='white_balance_temperature'): load_config(path)

def test_worker_keeps_other_camera_open(monkeypatch):
    from nest_clear.workers.calib_camera import CalibCameraWorker
    class Device:
        def close(self): self.closed = True
    front = Device(); front.closed = False
    def fake_open(camera):
        if camera.name == 'side': raise ValueError('side failure')
        camera.device = front
    monkeypatch.setattr(CalibCamera, 'open', fake_open)
    worker = CalibCameraWorker({'front': {'backend': 'mock'}, 'side': {'backend': 'mock'}})
    with pytest.raises(Exception, match='side failure'): worker.open()
    assert worker.cameras['front'].device is front and not front.closed
    worker.on_stop(); assert front.closed

class SyntheticUVC:
    """Backlit silhouette with optical blur, clipping and repeatable sensor noise."""
    def __init__(self, exposure=100, scale=.48, sigma=1.5, noise=0.8):
        self.exposure = exposure
        self.scale = scale
        self.sigma = sigma
        self.noise = noise
        self.rng = np.random.default_rng(29)

    def get_exposure(self):
        return self.exposure

    def set_exposure(self, value):
        self.exposure = value

    def capture(self):
        background = self.exposure * self.scale
        image = np.full((200, 240), background, dtype=np.float32)
        cv2.fillPoly(image, [np.array([[104, 0], [136, 0], [120, 155]])], background * .18)
        image = cv2.GaussianBlur(image, (0, 0), self.sigma)
        image += self.rng.normal(0, self.noise, image.shape).astype(np.float32)
        return np.clip(image, 0, 255).astype(np.uint8)


def test_exposure_expands_from_current_and_brackets():
    camera = SyntheticUVC()
    result = auto_exposure(camera)
    values = [item['exposure'] for item in result['iterations']]
    assert values[0] == 100
    assert 200 <= result['p99'] <= 235 and result['saturated_fraction'] < .001
    assert result['contrast'] is not None
    assert len(values) <= 12 and max(values) <= result['exposure'] * 2
    assert camera.get_exposure() == result['exposure']
    saved = SyntheticUVC(exposure=420)
    assert auto_exposure(saved)['iterations'][0]['exposure'] == 420
    class ConfiguredFake(Fake):
        config = {'camera_controls': {'exposure': 150}}
    assert auto_exposure(ConfiguredFake(.5))['iterations'][0]['exposure'] == 150


def test_exposure_expands_down_and_endpoint_failures():
    from nest_clear.devices.tooltip_exposure import ExposureError
    result = auto_exposure(SyntheticUVC(exposure=800, scale=1))
    assert [x['exposure'] for x in result['iterations']][:3] == [800, 400, 200]
    with pytest.raises(ExposureError, match='1600 仍偏暗') as error:
        auto_exposure(SyntheticUVC(scale=.001, noise=0))
    assert [x['exposure'] for x in error.value.iterations] == [100, 200, 400, 800, 1600]
    with pytest.raises(ExposureError, match='1 仍过曝') as error:
        auto_exposure(SyntheticUVC(scale=300, noise=0))
    assert [x['exposure'] for x in error.value.iterations] == [100, 50, 25, 12, 6, 3, 1]


def test_control_order_independent_of_override_order(tmp_path):
    from nest_clear.devices.v4l2_controls import IDS
    path = tmp_path / 'device'; path.touch()
    writes, values = [], {}
    def ioctl(fd, op, c):
        if op == VIDIOC_QUERYCTRL:
            c.minimum = 0; c.maximum = 10000; c.step = 1
        elif op == VIDIOC_S_CTRL:
            writes.append(c.id); values[c.id] = c.value
        elif op == VIDIOC_G_CTRL:
            c.value = values[c.id]
    controls = V4L2Controls(str(path), ioctl)
    try:
        # Mirrors a per-camera override with dependent values inserted first.
        controls.apply({'exposure': 100, 'white_balance_temperature': 4600,
                        'white_balance_auto': 0, 'auto_exposure': 1})
    finally:
        controls.close()
    assert writes.index(IDS['auto_exposure']) < writes.index(IDS['exposure'])
    assert writes.index(IDS['white_balance_auto']) < writes.index(IDS['white_balance_temperature'])


def test_open_error_messages(tmp_path, monkeypatch):
    import errno
    import struct
    path = tmp_path / 'by-path-camera'
    with pytest.raises(Exception, match='未找到主对刀相机：' + str(path) + '，检查线缆与插口'):
        CalibCamera('front', {'backend': 'uvc', 'device': str(path)}).open()
    with pytest.raises(Exception, match='辅助对刀相机未配置'):
        CalibCamera('side', {'backend': 'uvc', 'device': '待接入：待确认'}).open()
    path.touch()
    def querycap_non_capture(fd, request, buffer, mutate):
        struct.pack_into('II', buffer, 84, 0, 0)
    monkeypatch.setattr('nest_clear.devices.calib_cameras.fcntl.ioctl', querycap_non_capture)
    with pytest.raises(Exception, match='不是视频 capture 节点'):
        CalibCamera('front', {'backend': 'uvc', 'device': str(path)}).open()
    def querycap_capture(fd, request, buffer, mutate):
        struct.pack_into('II', buffer, 84, 1, 0)
    monkeypatch.setattr('nest_clear.devices.calib_cameras.fcntl.ioctl', querycap_capture)
    class NoBandwidth:
        def isOpened(self): return True
        def set(self, *args): return True
        def read(self): raise OSError(errno.ENOSPC, 'No space left on device')
        def release(self): pass
    monkeypatch.setattr('cv2.VideoCapture', lambda *args: NoBandwidth())
    with pytest.raises(Exception, match='USB 带宽不足：两台对刀相机须插在不同 USB 控制器上'):
        CalibCamera('front', {'backend': 'uvc', 'device': str(path)}).open()
