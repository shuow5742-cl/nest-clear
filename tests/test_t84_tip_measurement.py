"""T84 five-frame quality, baseline provenance and PLC handshakes."""
from dataclasses import replace
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from autoweaver.motion_policy.world_board import WorldBoard
from autoweaver.worker.clock import BTClock

from nest_clear.baseline import runout
from nest_clear.baseline.session import publish_existing_baselines, summary
from nest_clear.comm import Handshake, load_contract
from nest_clear.config import load_config
from nest_clear.flow.batch import build_batch
from nest_clear.flow.context import RequestTracker
from nest_clear.record.events import EventRecorder
from nest_clear.supervisor.tooltip_host import TooltipHost
from nest_clear.tooltip import ToolTipBaseline, ViewCalib
from nest_clear.tooltip.factory import BASELINE_DETECTOR_WARNING
from nest_clear.tooltip.store import save_baseline
from nest_clear.tooltip.trident import DETECTOR_VERSION, TridentResult
from nest_clear.ui.model import supervisor_tooltip_main
from nest_clear.workers.calib_camera import CalibCameraWorker
from nest_clear.workers.plc import PlcWorker
from test_flow_recovery_no_tcp import MemoryPLC
from test_trident import scene


@pytest.mark.parametrize('failure', [None, 'motion', 'coarse', 'std'])
def test_five_frame_quality_and_median(failure, monkeypatch):
    points = [(10.1, 19.9), (10.2, 20.1), (10, 20), (9.9, 20.2), (9.8, 19.8)]
    front = [TridentResult(point_uv=p, status='精定位', axis_rms_px=.2) for p in points]
    if failure == 'motion':
        front[2] = replace(front[2], axis_rms_px=2.01)
    elif failure == 'coarse':
        front[2] = replace(front[2], status='粗定位', diagnostic='末端对比度不足')
    elif failure == 'std':
        front[2] = replace(front[2], point_uv=(10, 21))
    side = [TridentResult(point_uv=(25, 26), status='精定位', axis_rms_px=.1)] * 5

    class Detector:
        max_edge_width_px = None

        def detect(self, frame):
            return frame

    monkeypatch.setattr(runout, 'make_tip_detector', lambda cfg: Detector())
    _, measurement = runout.measure_frames({'front': front, 'side': side}, {}, .3, rms_max=2)
    report = measurement['views']['front']
    assert report['detector'] == {'method': 'trident', 'version': DETECTOR_VERSION}
    assert [row['index'] for row in report['frames']] == [1, 2, 3, 4, 5]
    assert report['std_uv'] == pytest.approx(np.std([r.point_uv for r in front], axis=0))
    assert measurement['rejected'] is (failure is not None)
    if failure is None:
        assert report['median_uv'] == [10, 20]
        assert measurement['views']['side']['median_uv'] == [25, 26]
    else:
        expected = {'motion': '正面第 3 帧工具在动或画面畸变',
                    'coarse': '正面第 3 帧未通过单帧精定位检查',
                    'std': '正面五帧 v 标准差'}
        assert expected[failure] in measurement['reason']


def app_config(tmp_path):
    config = load_config('config/nest_clear.yaml')
    cfg = replace(config.tool_calibration, enabled=True, directory=str(tmp_path),
                  detection={'front': {'threshold': 127}, 'side': {'threshold': 127}},
                  cameras={name: {'backend': 'mock'} for name in ('front', 'side')})
    return replace(config, tool_calibration=cfg, record_directory=str(tmp_path),
                   travel=((-100., 100.),) * 3)


def seed_baseline(config, detector=None):
    path = Path(config.tool_calibration.directory) / 'tool1_baseline.yaml'
    views = [
        ViewCalib('front', (300, 452), ((20, 0, 0), (0, 0, 20))),
        ViewCalib('side', (300, 452), ((0, 20, 0), (0, 0, 20))),
    ]
    save_baseline(ToolTipBaseline('tool1', (0, 0, 0), views, detector=detector),
                  path, tool=1, source='test')
    return path


@pytest.mark.parametrize('version', [None, 'trident-2', DETECTOR_VERSION])
def test_baseline_version_warning_agrees_in_both_hosts(tmp_path, version):
    config = app_config(tmp_path)
    detector = {name: {'method': 'trident', 'version': version} for name in ('front', 'side')} if version else None
    path = seed_baseline(config, detector)
    main_events = EventRecorder(tmp_path / 'main', 'main', run=True)
    sup_events = EventRecorder(tmp_path / 'supervisor', 'supervisor')
    stamp = main_events.clock().isoformat()
    host = TooltipHost(replace(config, record_directory=str(sup_events.directory)), sup_events, start_time=stamp)
    try:
        publish_existing_baselines(main_events, config, config.vision.calibration, phase='idle')
        supervisor = {'pid': os.getpid(), 'start_time': stamp, 'child_pid': None}
        from_supervisor = supervisor_tooltip_main({}, supervisor, host.status, supervisor_age=0)
        from_main = supervisor_tooltip_main(main_events.status.data, {**supervisor, 'child_pid': 42}, {}, supervisor_age=0)
        assert from_supervisor['baseline']['existing'] == from_main['baseline']['existing']
        warning = '' if version == DETECTOR_VERSION else BASELINE_DETECTOR_WARNING
        assert summary(path)['detector_warning'] == warning
        assert (BASELINE_DETECTOR_WARNING in from_main['baseline'].get('notice', '')) == bool(warning)
        assert from_main['baseline'].get('notice') == from_supervisor['baseline'].get('notice')
        for events in (main_events, sup_events):
            rows = [json.loads(line) for file in events.directory.rglob('events.jsonl')
                    for line in file.read_text().splitlines()]
            assert any(row.get('reason') == BASELINE_DETECTOR_WARNING for row in rows) == bool(warning)
    finally:
        host.close()


@pytest.mark.parametrize(('reply', 'groups'), [(131, 3), (111, 3), (999, 1)])
def test_five_frame_replies_keep_task_id(tmp_path, monkeypatch, reply, groups):
    from nest_clear.tooloffset.commands import send
    config = app_config(tmp_path)
    seed_baseline(config)  # A missing detector version must not block acquisition.
    board = WorldBoard()
    clock = BTClock(board)
    contract = load_contract('config/plc_contract.yaml')

    class Registers:
        def __init__(self):
            self.words = {}
            self.writes = []

        def read_u16(self, address):
            return self.words.get(address, 0)

        def write_u16(self, address, value):
            self.words[address] = value
            self.writes.append((address, value))

        def read_real32_block(self, address, count):
            assert address == contract['actual_x'].address and count == 3
            return (0, 0, 0)

    registers = Registers()
    events = EventRecorder(tmp_path, 'main', run=True)
    plc = PlcWorker(Handshake(registers, contract, events=events))
    camera = CalibCameraWorker(config.tool_calibration.cameras)
    frame = scene()
    cv2.fillPoly(frame, [np.array([[300, 540], [290, 599], [310, 599]], dtype=np.int32)], 25)
    blank = np.full_like(frame, 255)
    reads = {'front': [], 'side': []}

    def front():
        reads['front'].append(time.monotonic())
        reject = reply == 131 or (reply == 111 and len(reads['front']) <= 10)
        return blank if reject and len(reads['front']) % 5 == 3 else frame

    def side():
        reads['side'].append(time.monotonic())
        return frame

    monkeypatch.setattr(camera.cameras['front'], 'capture', front)
    monkeypatch.setattr(camera.cameras['side'], 'capture', side)
    if reply == 999:
        def broken(*args, **kwargs):
            raise RuntimeError('测试检测器异常')
        monkeypatch.setattr('nest_clear.flow.calibration.measure_frames', broken)
    for worker in (plc, camera):
        clock.attach_worker(worker)
    batch = build_batch(board, config, [(0, 0, 0)], RequestTracker(0), events)
    clock.submit(batch)
    task = 73
    try:
        requested = time.monotonic()
        registers.words[contract['request_func'].address] = 100
        registers.words[contract['plc_tool_id'].address] = 2
        registers.words[contract['task_id'].address] = task
        deadline = requested + 8
        sent = False
        while registers.read_u16(40933) != task and time.monotonic() < deadline:
            clock.tick_once()
            if reply == 111 and not sent and events.status.data['tooltip'].get('phase') == 'in_range':
                send(tmp_path, 'tooltip_confirm_result')
                sent = True
            time.sleep(.002)
        assert registers.writes == [(40935, 2), (40934, reply), (40933, task)]
        assert all(len(starts) == 5 * groups and starts[0] > requested for starts in reads.values())
        record = json.loads((events.run_directory / 'tooltip' / str(task) / 'session_record.json').read_text())
        attempts = record['measurements']
        assert [a['attempt'] for a in attempts] == list(range(1, groups + 1))
        assert all(a['task_id'] == task for a in attempts)
        if reply == 999:
            assert attempts[0]['reason'] == '测试检测器异常'
        else:
            rejected = attempts if reply == 131 else attempts[:-1]
            assert all('正面第 3 帧无法识别针尖' in a['reason'] for a in rejected)
            assert all(len(a['measurement']['views']['side']['frames']) == 5 for a in attempts)
        assert BASELINE_DETECTOR_WARNING in record['baseline_warning']
    finally:
        clock.shutdown()


def test_baseline_session_persists_trident_median_and_version(tmp_path, monkeypatch):
    from nest_clear.baseline.session import BaselineSession
    from nest_clear.tooltip.store import initialize_tools, load_baseline

    config = app_config(tmp_path)
    initialize_tools(tmp_path / 'tools.yaml')
    events = EventRecorder(tmp_path, 'main', run=True)
    worker = CalibCameraWorker(config.tool_calibration.cameras)
    def capture(position):
        return np.roll(scene(), position, axis=1)

    shifts = [iter((2, 0, 1, 0, 0)) for _ in range(2)]
    execute = worker.execute
    batching = False

    def execute_with_batch(payload):
        nonlocal batching
        batching = payload.get('frames_per_view') == 5
        return execute(payload)

    monkeypatch.setattr(worker, 'execute', execute_with_batch)
    for camera, shift in zip(worker.cameras.values(), shifts):
        monkeypatch.setattr(camera, 'capture',
                            lambda shift=shift: capture(next(shift) if batching else 0))
    config = replace(config, tool_calibration=replace(config.tool_calibration, frame_std_max_px=1))
    session = BaselineSession(config, worker, MemoryPLC(), events, config.vision.calibration,
                              {'main_target_uv': [300, 452]})

    def command(name, phase, **params):
        session.submit(name, params, len(session.commands) + 1)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and session.phase != phase:
            time.sleep(.02)
        assert session.phase == phase, session.message

    try:
        monkeypatch.setattr(session, '_stable_position', lambda: [0, 0, 0])
        command('baseline_select', 'live', tool=1)
        command('baseline_measure', 'success')
        assert session.phase == 'success'
        for name, view in session.measurement['views'].items():
            assert session.candidate['pixels'][name] == view['median_uv']
            assert abs(view['frames'][0]['uv'][0] - view['median_uv'][0]) > 1
        command('baseline_save', 'confirm')
        command('baseline_confirm', 'saved', accept=True)
        saved = load_baseline(tmp_path / 'tool1_baseline.yaml')
        assert saved.detector == {name: {'method': 'trident', 'version': DETECTOR_VERSION}
                                  for name in ('front', 'side')}
        assert summary(tmp_path / 'tool1_baseline.yaml')['detector_warning'] == ''
    finally:
        session.end()
        session.thread.join(5)
