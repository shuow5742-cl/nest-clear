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


class Node:
    def __init__(self, value=0): self.value = value
    def get(self): return self.value
    def set(self, value): self.value = value


class Ratio(Node):
    def __init__(self, device): self.device = device
    def get(self): return self.device.values[self.device.BalanceRatioSelector.get()]
    def set(self, value): self.device.values[self.device.BalanceRatioSelector.get()] = value


class Worker:
    def __init__(self, once=False, image=None, timeout=False):
        self.image = image if image is not None else frame()
        self.readback = {'balance_ratio': [1., 1., 1.]}
        self.size = (160, 120)
        self.count = 0; self.timeout = timeout
        if once:
            device = SimpleNamespace(values=[1., 1., 1.], BalanceRatioSelector=Node(), BalanceWhiteAuto=Node())
            device.BalanceRatio = Ratio(device)
            gx = SimpleNamespace(GxAutoEntry=SimpleNamespace(OFF=0, ONCE=1), GxBalanceRatioSelectorEntry=SimpleNamespace(RED=0, GREEN=1, BLUE=2))
            self.camera = SimpleNamespace(_cam=device, _gx=gx)
        else:
            self.camera = SimpleNamespace(balance_ratio=[1., 1., 1.])
    def execute(self, payload):
        self.count += 1
        device = getattr(self.camera, '_cam', None)
        if device and device.BalanceWhiteAuto.get() == 1 and self.count >= 2 and not self.timeout:
            device.values = [1.2, 1., 1.3]; device.BalanceWhiteAuto.set(0)
        return self.image, {}


def test_once_save_backup(tmp_path):
    path = tmp_path / 'camera_white_balance.yaml'
    path.write_text('old: true\n')
    result = run_white_balance(Worker(True), {}, {}, path, tmp_path)
    assert result['state'] == 'success' and result['method'] == 'once'
    assert yaml.safe_load(path.read_text())['balance_ratio'] == [1.2, 1., 1.3]
    assert list((tmp_path / 'history').iterdir())
    assert len((next((tmp_path / 'focus_records').iterdir()) / 'white_balance.jsonl').read_text().splitlines()) == 4


def test_software_and_failure_restore(tmp_path):
    path = tmp_path / 'camera_white_balance.yaml'
    worker = Worker()
    result = run_white_balance(worker, {}, {}, path, tmp_path)
    assert result['state'] == 'success' and result['method'] == 'software'
    before = path.read_text()
    worker = Worker(image=frame(252, 252, 252))
    result = run_white_balance(worker, {}, {}, path, tmp_path)
    assert result['state'] == 'fail' and worker.camera.balance_ratio == [1., 1., 1.]
    assert path.read_text() == before


def test_timeout_restores(tmp_path):
    worker = Worker(True, timeout=True)
    result = run_white_balance(worker, {'timeout_s': .001}, {}, tmp_path / 'camera_white_balance.yaml', tmp_path)
    assert result['state'] == 'fail' and '超时' in result['reason']
    assert worker.camera._cam.values == [1., 1., 1.]


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
    App.update_white_balance(ui, {'white_balance': {'state': 'success', 'balance_ratio': [1., 1., 1.], 'metrics': {'r_g': 1., 'b_g': 1., 'warnings': []}}})
    assert '白平衡成功' in ui.wb_message.values['text']
    App.update_white_balance(ui, {'white_balance': {'state': 'fail', 'reason': '请用 18% 灰卡做白平衡'}})
    assert '白平衡失败' in ui.wb_message.values['text']
