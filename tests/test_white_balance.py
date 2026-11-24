import json
from types import SimpleNamespace

import numpy as np
import yaml

from nest_clear.preview.white_balance import evaluate, run_white_balance


def frame(b=110, g=110, r=110):
    return np.full((120, 160, 3), (b, g, r), dtype=np.uint8)


def test_judgement():
    assert not evaluate([frame()])['reason']
    assert '灰卡' in evaluate([frame(252, 252, 252)])['reason']
    assert '偏色' in evaluate([frame(r=119)])['reason']
    image = frame()
    image[:, :80, 2] = 100
    image[:, 80:, 2] = 120
    result = evaluate([image])
    assert not result['reason'] and result['warnings']


class EnumNode:
    def __init__(self, value=0, writable=True):
        self.value = value
        self.writable = writable
    def get(self):
        return (self.value, {0: 'Off', 1: 'Once', 2: 'Blue'}.get(self.value, 'Unknown'))
    def set(self, value):
        if type(value) is not int:
            raise TypeError(f"EnumFeature.set: Expected enum_value type is int, not {type(value)}")
        self.value = value
    def is_writable(self): return self.writable


class Ratio:
    def __init__(self, device): self.device = device
    def get(self): return self.device.values[self.device.BalanceRatioSelector.value]
    def set(self, value): self.device.values[self.device.BalanceRatioSelector.value] = value


class Worker:
    def __init__(self, once=True, image=None, timeout=False):
        self.image = image if image is not None else frame()
        self.readback = {'balance_ratio': [1., 1., 1.]}
        self.size = (160, 120)
        self.count = 0; self.timeout = timeout
        device = SimpleNamespace(values=[1., 1., 1.], BalanceRatioSelector=EnumNode(2), BalanceWhiteAuto=EnumNode(writable=once))
        device.BalanceRatio = Ratio(device)
        gx = SimpleNamespace(GxAutoEntry=SimpleNamespace(OFF=0, ONCE=1), GxBalanceRatioSelectorEntry=SimpleNamespace(RED=0, GREEN=1, BLUE=2))
        self.camera = SimpleNamespace(_cam=device, _gx=gx)
    def execute(self, payload):
        self.count += 1
        device = self.camera._cam
        if device.BalanceWhiteAuto.value == 1 and self.count >= 2 and not self.timeout:
            device.values = [1.2, 1., 1.3]; device.BalanceWhiteAuto.set(0)
        return self.image, {}


def test_once_save_backup(tmp_path):
    path = tmp_path / 'camera_white_balance.yaml'
    path.write_text('old: true\n')
    worker = Worker()
    result = run_white_balance(worker, {}, {}, path, tmp_path)
    assert result['state'] == 'success' and result['method'] == 'once'
    assert worker.count >= 6
    assert worker.camera._cam.BalanceRatioSelector.get() == (2, 'Blue')
    assert yaml.safe_load(path.read_text())['balance_ratio'] == [1.2, 1., 1.3]
    assert list((tmp_path / 'history').iterdir())
    assert len((next((tmp_path / 'focus_records').iterdir()) / 'white_balance.jsonl').read_text().splitlines()) == 4


def test_software_and_failure_restore(tmp_path):
    path = tmp_path / 'camera_white_balance.yaml'
    worker = Worker(once=False)
    result = run_white_balance(worker, {}, {}, path, tmp_path)
    assert result['state'] == 'success' and result['method'] == 'software'
    assert worker.camera._cam.BalanceRatioSelector.value == 2
    before = path.read_text()
    worker = Worker(image=frame(252, 252, 252))
    result = run_white_balance(worker, {}, {}, path, tmp_path)
    assert result['state'] == 'fail' and '灰卡' in result['reason']
    assert worker.camera._cam.values == [1., 1., 1.]
    assert worker.camera._cam.BalanceRatioSelector.value == 2
    assert path.read_text() == before


def test_timeout_restores(tmp_path):
    worker = Worker(timeout=True)
    result = run_white_balance(worker, {'timeout_s': .001}, {}, tmp_path / 'camera_white_balance.yaml', tmp_path)
    assert result['state'] == 'fail' and '超时' in result['reason'] and '灰卡' not in result['reason']
    assert worker.camera._cam.values == [1., 1., 1.]
    assert not (tmp_path / 'camera_white_balance.yaml').exists()


def test_camera_exception_is_not_target_failure(tmp_path):
    worker = Worker()
    def broken(): raise RuntimeError('相机断开')
    worker.camera._cam.BalanceRatio.get = broken
    result = run_white_balance(worker, {}, {}, tmp_path / 'camera_white_balance.yaml', tmp_path)
    assert '白平衡执行异常：相机断开' in result['reason']
    assert '请联系维护' in result['reason'] and '灰卡' not in result['reason']
    assert not (tmp_path / 'camera_white_balance.yaml').exists()


def test_enum_normalization_and_camera_worker():
    from nest_clear.devices.gx_enum import enum_value
    from nest_clear.workers.camera import CameraWorker
    assert enum_value(2) == enum_value((2, 'Blue')) == enum_value([2, 'Blue']) == 2
    fake = Worker()
    fake.camera.open = lambda: True
    fake.camera._cam.ExposureAuto = EnumNode(1)
    worker = CameraWorker('mock', 'fake', camera=fake.camera)
    worker.open()
    assert worker.readback['balance_ratio'] == [1., 1., 1.]
    assert fake.camera._cam.BalanceRatioSelector.value == 2
    mode = worker.exposure_auto_mode()
    assert mode == 1
    fake.camera._cam.ExposureAuto.set(0)
    worker.restore_exposure_auto_mode(mode)
    assert fake.camera._cam.ExposureAuto.get() == (1, 'Once')
    worker.restore_exposure_auto_mode((1, 'Once'))



def test_exposure_sweep_restores_tuple_enum(tmp_path):
    from nest_clear.preview.exposure import run_sweep
    from nest_clear.workers.camera import CameraWorker
    fake = Worker()
    device = fake.camera._cam
    device.ExposureAuto = EnumNode(1)
    class FloatNode:
        def __init__(self): self.value = 5000.
        def set(self, value): self.value = value
        def get(self): return self.value
    device.ExposureTime = FloatNode()
    worker = CameraWorker('mock', 'fake', camera=fake.camera)
    worker.readback['exposure_us'] = 5000.
    worker.execute = lambda _: (frame(), {})
    result = run_sweep(worker, {'min_us': 1000., 'max_us': 2000., 'steps': 2, 'settle_frames': 0}, {}, 'brightness',
                       measure=lambda *_: {'p99': 200, 'saturated_fraction': 0., 'corners': 1})
    assert result['state'] == 'done'
    assert device.ExposureTime.get() == 5000.
    assert device.ExposureAuto.get() == (1, 'Once')


def test_config_override_and_corruption(tmp_path):
    import pytest
    from nest_clear.config import load_config
    from nest_clear.guard import GuardError
    config = tmp_path / 'settings.yaml'
    config.write_text(open('config/nest_clear.yaml', encoding='utf-8').read(), encoding='utf-8')
    path = tmp_path / 'calibration' / 'camera_white_balance.yaml'
    path.parent.mkdir()
    path.write_text('balance_ratio: [1.2, 1, 1.4]\ntime: today\n')
    loaded = load_config(config, require_tools=False)
    assert loaded.balance_ratio == (1.2, 1., 1.4)
    assert loaded.white_balance_source == '自测 today'
    path.write_text('balance_ratio: [broken')
    with pytest.raises(GuardError, match='camera_white_balance'):
        load_config(config, require_tools=False)


def test_ui_controls_and_result():
    from nest_clear.ui.model import preview_controls
    from nest_clear.ui.app import App
    assert not preview_controls({'pid': 1, 'child_pid': 2, 'preview': {'on': True, 'mode': 'focus'}}, 1)['white_balance_enabled']
    assert preview_controls({'pid': 1, 'preview': {'on': True, 'mode': 'focus'}}, 1)['white_balance_enabled']
    class Widget:
        def config(self, **kwargs): self.values = kwargs
    ui = SimpleNamespace(wb_button=Widget(), wb_message=Widget(), wb_pending=False)
    ui._expire_focus_request = lambda name, label: App._expire_focus_request(ui, name, label)
    App.update_white_balance(ui, {'white_balance': {'state': 'success', 'balance_ratio': [1., 1., 1.], 'metrics': {'r_g': 1., 'b_g': 1., 'warnings': []}}})
    assert '白平衡成功' in ui.wb_message.values['text']
    App.update_white_balance(ui, {'white_balance': {'state': 'fail', 'reason': '请用 18% 灰卡做白平衡'}})
    assert '白平衡失败' in ui.wb_message.values['text']
