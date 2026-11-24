from dataclasses import replace
from datetime import date
import json
import time

import numpy as np
import pytest
from autoweaver.motion_policy.world_board import WorldBoard
from autoweaver.worker.clock import BTClock
from nest_clear.config import load_config
from nest_clear.flow.batch import build_batch
from nest_clear.flow.context import RequestTracker
from nest_clear.record.events import EventRecorder
from nest_clear.record.report import report
from nest_clear.ui.model import classify_event
from nest_clear.vision.geometry import Calibration
from test_flow_recovery_no_tcp import MemoryPLC, MemoryCamera, EmptyVision


def run_arrival_flow(tmp_path, values, timeout, warning, stable, film_mode):
    class PLC(MemoryPLC):
        def read(self, key): return 0
        def write(self, key, value):
            if key == 'heartbeat': self.heartbeats.append(value)
        def execute(self, payload):
            if payload['operation'] == 'actual':
                self.reads += 1
                return (values[min(self.reads - 1, len(values)-1)], 0., 0.)
            return super().execute(payload)
    class Vision(EmptyVision):
        def execute(self, payload):
            self.pose = payload['pose']
            return super().execute(payload)
    from nest_clear.film_mode import write_mode
    write_mode(tmp_path, film_mode)
    config = load_config('config/nest_clear.yaml')
    config = replace(config, record_directory=str(tmp_path), arrival_settle=dict(sample_interval_s=.01, settle_samples=4, jitter_tol_mm=.02, settle_timeout_s=timeout), vision=replace(config.vision, detector='stub', pixels=()))
    board = WorldBoard(); clock = BTClock(board); plc = PLC(); plc.reads = 0; plc.heartbeats = []; vision = Vision()
    for worker in (plc, MemoryCamera(), vision): clock.attach_worker(worker)
    events = EventRecorder(tmp_path, 'main')
    batch = build_batch(board, config, [(0., 0., 0.)], RequestTracker(0), events, calibration=Calibration(np.eye(3), np.zeros(5), np.eye(3), np.zeros(3), 1))
    clock.submit(batch)
    from nest_clear.supervisor.runner import Supervisor
    supervisor = Supervisor(config, plc)
    def request(task, func):
        plc.request = (task, func); deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            start = time.monotonic(); clock.tick_once(); assert time.monotonic() - start < .3
            supervisor._next_heartbeat = 0; supervisor.tick()
            if task in plc.replies: return plc.replies[task][0]
            time.sleep(.001)
        pytest.fail('request timeout')
    try:
        assert request(1, 1) == 1
        assert batch.blackboard.read('work_cycle').film_mode == film_mode
        assert request(2, 20) == 9
        rows = [json.loads(line) for path in tmp_path.rglob('events.jsonl') for line in path.read_text().splitlines()]
        row = next(r for r in rows if r['event'] in ('arrival', 'arrival_warning'))
        assert (row['event'] == 'arrival_warning') == warning
        assert row['stable'] == stable
        assert row['stage'] == 'coarse'
        assert row['tolerance_mm'] == .1
        assert {'task_id', 'stage', 'expected', 'actual', 'delta', 'tolerance_mm', 'stable', 'settle_elapsed_s', 'samples', 'over_tolerance', 'ok'} <= row.keys()
        assert row['settle_elapsed_s'] >= .03
        assert row['settle_elapsed_s'] * 1000 >= row['samples'][-1][0] - .001
        assert row['settle_elapsed_s'] * 1000 - row['samples'][-1][0] < 50
        assert len(row['samples']) >= 4
        assert len(plc.heartbeats) >= 2
        assert vision.pose == pytest.approx(row['actual'])
        assert row['actual'][0] == pytest.approx(sum(s[1] for s in row['samples'][-4:]) / 4)
        if warning:
            assert classify_event(row)['color'] == 'yellow'
        if values[0] == .2:
            assert '超差次数 1' in report(tmp_path, date.today())
    finally:
        clock.shutdown()
    return row


@pytest.mark.parametrize('values,timeout,warning,stable', [
    ([0, .2, -.04, .075, .08, .085, .08], .3, False, True),
    ([.08] * 8, .3, False, True),
    ([.14] * 8, .3, True, True),
    ([.2] * 8, .3, True, True),
    ([0, .2] * 100, .06, True, False),
])
def test_arrival_sampling_and_flow(tmp_path, values, timeout, warning, stable):
    production = tmp_path / 'production'
    film = tmp_path / 'film'
    production.mkdir()
    film.mkdir()
    normal_row = run_arrival_flow(production, values, timeout, warning, stable, False)
    film_row = run_arrival_flow(film, values, timeout, warning, stable, True)
    assert set(normal_row) == set(film_row)
    for key in ('event', 'stage', 'expected', 'stable', 'over_tolerance', 'tolerance_mm', 'ok'):
        assert normal_row[key] == film_row[key]


def test_actual_pose_drives_pick_coordinates():
    from nest_clear.flow.context import FlowContext
    from nest_clear.flow.picking import WorkRequest, FinishCapture
    from nest_clear.vision.geometry import pixel_to_module
    from nest_clear.vision.fine import FineResult
    from nest_clear.vision.hair_integration import measure
    from test_hair_integration import Board, Events, candidate
    config = replace(load_config('config/nest_clear.yaml'), travel=((-100., 100.),) * 3)
    cal = Calibration(np.eye(3), np.zeros(5), np.eye(3), np.zeros(3), 1)
    def pick(x):
        ctx = FlowContext(None, config, [(0, 0, 0)], RequestTracker(0), Events(), cal)
        bb = Board(work_cycle=None, request_func=1, task_id=1)
        req = WorkRequest(ctx); req._blackboard = bb; req.perform()
        bb.write('request_func', 20); req.perform()
        bb.write('actual', (x, 0, 0))
        bb.write('detection', {'observations': [measure(candidate(), (x, 0, 0), cal, config.vision.hair_detector)]})
        finish = FinishCapture(ctx); finish._blackboard = bb; finish.perform()
        req.perform()
        bb.write('detection', {'fine_result': FineResult('ok', target_xyz=pixel_to_module(2, 3, bb.read('actual'), cal))})
        finish.perform()
        return bb.read('reply_coordinates')['pick1_xyz']
    assert pick(.14)[0] - pick(0)[0] == pytest.approx(.14)


def test_arrival_settle_validation(tmp_path):
    import yaml
    from nest_clear.guard import GuardError
    original = yaml.safe_load(open('config/nest_clear.yaml'))
    for key, value in [('settle_samples', 1), ('settle_samples', 41), ('sample_interval_s', 0), ('jitter_tol_mm', float('nan')), ('settle_timeout_s', -1)]:
        data = dict(original); data['arrival_settle'] = dict(original['arrival_settle'], **{key: value})
        path = tmp_path / 'config.yaml'; path.write_text(yaml.safe_dump(data))
        with pytest.raises(GuardError): load_config(path)


def test_warning_does_not_overwrite_status(tmp_path):
    from nest_clear.ui.status import MainStatus
    status = MainStatus(tmp_path)
    status.observe({'event': 'work_cycle', 'state': 'WAIT_FINE'})
    before = dict(status.data['picking'])
    status.observe({'event': 'arrival_warning', 'task_id': 2, 'stage': 'fine'})
    assert status.data['picking'] == before
    item = classify_event({'event': 'arrival_warning', 'task_id': 2})
    assert item['category'] == '算法' and item['color'] == 'yellow'


def test_invalid_actual_still_fails(tmp_path):
    class PLC(MemoryPLC):
        def execute(self, payload):
            if payload['operation'] == 'actual': return (float('nan'), 0., 0.)
            return super().execute(payload)
    config = load_config('config/nest_clear.yaml')
    board = WorldBoard(); clock = BTClock(board); plc = PLC()
    for worker in (plc, MemoryCamera(), EmptyVision()): clock.attach_worker(worker)
    events = EventRecorder(tmp_path, 'main')
    clock.submit(build_batch(board, config, [(0., 0., 0.)], RequestTracker(0), events, calibration=Calibration(np.eye(3), np.zeros(5), np.eye(3), np.zeros(3), 1)))
    try:
        for task, func in [(1, 1), (2, 20)]:
            plc.request = task, func
            deadline = time.monotonic() + 3
            while task not in plc.replies and time.monotonic() < deadline:
                clock.tick_once(); time.sleep(.001)
            assert plc.replies[task][0] == (1 if task == 1 else 999)
        rows = [json.loads(line) for path in tmp_path.rglob('events.jsonl') for line in path.read_text().splitlines()]
        assert not any(r['event'] == 'arrival_warning' for r in rows)
    finally:
        clock.shutdown()
