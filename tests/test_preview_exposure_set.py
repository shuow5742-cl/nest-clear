import json
import threading
from types import SimpleNamespace

from nest_clear.preview.temporary_exposure import change_exposure
from nest_clear.workers.camera import CameraWorker
from nest_clear.ui.app import App


class Node:
    def __init__(self, value): self.value = value
    def set(self, value): self.value = value
    def get(self): return self.value


class Enum(Node):
    def get(self): return (self.value, 'Off')


def test_temporary_exposure_and_restore(tmp_path):
    worker = CameraWorker.__new__(CameraWorker)
    worker.capture_lock = threading.Lock()
    device = SimpleNamespace(ExposureAuto=Enum(0), ExposureTime=Node(10000.0))
    worker.camera = SimpleNamespace(_cam=device, _gx=SimpleNamespace(GxAutoEntry=SimpleNamespace(OFF=0)))
    worker.readback = {'exposure_us': 10000.0}
    limits = {'min_us': 500, 'max_us': 30000}
    def change(action, busy=False):
        return change_exposure(worker, action, 10000, limits, tmp_path, busy)
    assert change({'action': 'set_exposure', 'exposure_us': '20000'})['readback_us'] == 20000
    assert change({'action': 'set_exposure', 'exposure_us': '30001'})['state'] == 'error'
    assert change({'action': 'set_exposure', 'exposure_us': 'abc'})['state'] == 'error'
    assert change({'action': 'set_exposure', 'exposure_us': '25000'}, True)['state'] == 'error'
    assert device.ExposureTime.get() == 20000
    assert change({'action': 'restore_exposure'})['readback_us'] == 10000
    assert not list(tmp_path.rglob('*.yaml'))
    events = [json.loads(x) for x in next(tmp_path.rglob('exposure.jsonl')).read_text().splitlines()]
    assert [(e['old_us'], e['new_us'], e['readback_us']) for e in events] == [(10000, 20000, 20000), (20000, 10000, 10000)]


def test_controls_and_message():
    class Widget:
        def config(self, **kwargs): self.__dict__.update(kwargs)
    ui = SimpleNamespace(temporary_set=Widget(), temporary_restore=Widget(), temporary_message=Widget(), wb_pending=False, sweep_pending=False, temporary_pending=False)
    ui._expire_focus_request = lambda name, label: App._expire_focus_request(ui, name, label)
    App.update_temporary_exposure(ui, {'white_balance': {'state': 'running'}})
    assert ui.temporary_set.state == ui.temporary_restore.state == 'disabled'
    App.update_temporary_exposure(ui, {'temporary_exposure': {'state': 'done', 'message': '曝光回读 20000 us'}})
    assert ui.temporary_set.state == 'normal'
    assert ui.temporary_message.text == '曝光回读 20000 us'
