"""真实行为树、内存 PLC 与监督器心跳共同验证点击等待不阻塞。"""
from dataclasses import replace
import time
from types import SimpleNamespace

import numpy as np
from autoweaver.motion_policy.world_board import WorldBoard
from autoweaver.worker.clock import BTClock
from autoweaver.worker import AsyncRequestWorker

from nest_clear.config import load_config
from nest_clear.flow.batch import build_batch
from nest_clear.flow.context import RequestTracker
from nest_clear.record.events import EventRecorder
from nest_clear.supervisor.runner import Supervisor
from nest_clear.tooltip import ToolTipBaseline, ViewCalib
from nest_clear.tooltip.click import ClickResult
from nest_clear.tooltip.store import save_baseline
from nest_clear.vision.geometry import Calibration


def delayed_click(connection, images, points, tool):
    """子进程模拟打开窗口后操作员两秒才点击。"""
    time.sleep(2)
    connection.send({name: ClickResult((0., 0.), 'manual', 8., 2.) for name in ('front', 'side')})
    connection.close()


class MemoryPLC(AsyncRequestWorker):
    name = 'plc'
    def __init__(self):
        super().__init__()
        self.request = (1, 100)
        self.replies = {}
        self.heartbeats = []
    def execute(self, payload):
        op = payload['operation']
        if op == 'request': return self.request
        if op == 'actual': return (0., 0., 0.)
        if op == 'reply':
            self.replies[payload['task_id']] = payload['func']
            return None
        raise AssertionError(op)
    def read(self, key):
        return 0
    def write(self, key, value):
        if key == 'heartbeat': self.heartbeats.append((time.monotonic(), value))


class Camera(AsyncRequestWorker):
    name = 'calib_camera'
    def execute(self, payload):
        return {name: np.zeros((20, 20), np.uint8) for name in ('front', 'side')}


class CoarseDetector:
    max_edge_width_px = 6
    def __init__(self, **kwargs): pass
    def detect(self, image):
        return SimpleNamespace(point_uv=(0., 0.), focused=True, edge_width_px=1., width_px=10., coarse=True, diagnostic='coarse')


def test_delayed_click_does_not_block_ticks_or_heartbeat(tmp_path, monkeypatch, caplog):
    import nest_clear.flow.calibration as flow
    from nest_clear.tooloffset.commands import send
    monkeypatch.setattr(flow, '_click_process', delayed_click)
    monkeypatch.setattr(flow, 'SilhouetteTipDetector', CoarseDetector)
    publish = flow.ConfirmCalibration.publish
    def auto_confirm(node, phase, message, **fields):
        publish(node, phase, message, **fields)
        if phase == 'limit': send(node.ctx.events.directory, 'tooltip_click')
        elif phase == 'in_range': send(node.ctx.events.directory, 'tooltip_confirm_result')
    monkeypatch.setattr(flow.ConfirmCalibration, 'publish', auto_confirm)
    views = [ViewCalib('front', (0, 0), ((10, 0, 0), (0, 0, 10))),
             ViewCalib('side', (0, 0), ((0, 10, 0), (0, 0, 10)))]
    save_baseline(ToolTipBaseline('tool1', (0, 0, 0), views), tmp_path / 'tool1_baseline.yaml', tool=1, source='test')
    cfg = load_config('config/nest_clear.yaml')
    cfg = replace(cfg, record_directory=str(tmp_path / 'runs'),
                  tool_calibration=replace(cfg.tool_calibration, directory=str(tmp_path), unreliable_action='click', click_timeout_s=10))
    events = EventRecorder(tmp_path / 'runs', 'main')
    board, plc = WorldBoard(), MemoryPLC()
    clock = BTClock(board)
    for worker in (plc, Camera()): clock.attach_worker(worker)
    batch = build_batch(board, cfg, [(0., 0., 0.)], RequestTracker(0), events,
                        calibration=Calibration(np.eye(3), np.zeros(5), np.eye(3), np.zeros(3), 1))
    handle = clock.submit(batch)
    supervisor = Supervisor(cfg, plc)
    durations = []
    started = time.monotonic()
    try:
        while 1 not in plc.replies and time.monotonic() - started < 8:
            begin = time.monotonic()
            clock.tick_once()
            durations.append(time.monotonic() - begin)
            supervisor.tick()
            time.sleep(.005)
        assert plc.replies.get(1) == 111
        assert time.monotonic() - started >= 2
        assert len(plc.heartbeats) >= 2
        assert len(durations) > 100
        assert max(durations) < .3, f'最长 tick {max(durations):.3f}s'
        assert not [record for record in caplog.records if 'slow tick' in record.getMessage().lower() and '300' in record.getMessage()]
        assert handle.result is None
    finally:
        clock.shutdown()
