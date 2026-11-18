"""沙箱外运行：真实 Modbus TCP 驱动相机标定闭环。"""
from dataclasses import replace
import threading
import time
from types import SimpleNamespace

import numpy as np
from autoweaver.motion_policy.world_board import WorldBoard
from autoweaver.worker import AsyncRequestWorker
from autoweaver.worker.clock import BTClock

from nest_clear.comm import Handshake, PLCClient
from nest_clear.config import load_config
from nest_clear.flow.batch import build_batch
from nest_clear.flow.context import RequestTracker
from nest_clear.record.events import EventRecorder
from nest_clear.workers.plc import PlcWorker
from sim.fake_plc import FakePLC


class Camera(AsyncRequestWorker):
    name = 'camera'
    def execute(self, payload): return np.zeros((8, 8, 3), np.uint8), {}


class FakeSession:
    def __init__(self, cfg, nominal, image_size, start_xyz): self.count = 0
    def add(self, image, actual_xyz):
        self.count += 1
        fields = dict(target_xyz=None, result=None, code=None, reason=None, detail={}, overlay=None)
        if self.count == 1:
            fields['target_xyz'] = (1., 0., 0.)
            return SimpleNamespace(kind='move', **fields)
        fields['result'] = SimpleNamespace(calibration='new', report={'rms_px': 0.1})
        return SimpleNamespace(kind='done', **fields)
    def session_record(self): return {'count': self.count}


def test_camcal_tcp(tmp_path):
    with FakePLC() as plc:
        config = replace(load_config('config/nest_clear.yaml'), travel=((-100., 100.),)*3)
        io = PLCClient(plc.host, plc.port, 1, plc.contract)
        board = WorldBoard()
        clock = BTClock(board)
        for worker in (PlcWorker(Handshake(io, plc.contract)), Camera()): clock.attach_worker(worker)
        events = EventRecorder(tmp_path, run=True)
        path = tmp_path / 'camera.yaml'
        updated = []
        def save(result, path, history): path.write_text('placeholder: false\n')
        clock.submit(build_batch(board, config, [(0., 0., 0.)], RequestTracker(0), events,
                                 calibration='old', camcal_factory=FakeSession, camcal_save=save,
                                 calibration_updated=updated.append, calibration_path=path))
        stop = threading.Event()
        def run():
            while not stop.is_set():
                clock.tick_once()
                time.sleep(.002)
        thread = threading.Thread(target=run)
        thread.start()
        try:
            replies = plc.start_camera_calibration((0., 0., 0.), timeout_s=5)
            assert [r['func_code'] for r in replies] == [80, 81]
            assert replies[0]['camcal_xyz'] == [1., 0., 0.]
            assert path.exists() and updated == ['new']
            assert 40972 in [address for address, _ in plc.writes]
        finally:
            stop.set()
            thread.join(3)
            clock.shutdown()
            io.close()
