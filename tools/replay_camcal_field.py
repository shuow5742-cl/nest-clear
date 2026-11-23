#!/usr/bin/env python3
"""回放现场图像及 PLC 实测位姿，不修改现场数据。

PYTHONPATH=src .venv/bin/python tools/replay_camcal_field.py SESSION_DIR RUN_META_JSON
输出写入临时目录；需使用现场 run 的 meta.json 以恢复当时配置和名义标定。
"""
import argparse
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace

import cv2
import yaml
from autoweaver.motion_policy.world_board import WorldBoard
from autoweaver.worker.clock import BTClock
from nest_clear.camcal.config import CameraCalibConfig
from nest_clear.record.events import EventRecorder
from nest_clear.vision.geometry import load_calibration
from nest_clear.workers.camcal import CamcalWorker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session', type=Path)
    parser.add_argument('meta', type=Path)
    parser.add_argument('--reproduce-pause', action='store_true', help='末帧模拟旧主循环暂停 tick，再恢复以展示结果发布')
    args = parser.parse_args()
    meta = json.loads(args.meta.read_text())
    cfg = yaml.safe_load(meta['config']['content'])['camera_calibration']
    cfg.pop('solve_timeout_s', None)
    with tempfile.TemporaryDirectory(prefix='camcal-replay-') as tmp:
        nominal = Path(tmp) / 'nominal.yaml'
        nominal.write_text(meta['calibration']['content'])
        ctx = SimpleNamespace(events=EventRecorder(Path(tmp)), config=SimpleNamespace(camera_calibration=cfg),
                              calibration=load_calibration(nominal), camcal_factory=None)
        step_written = threading.Event()
        emit = ctx.events.emit
        def observe(event, **fields):
            emit(event, **fields)
            if event == 'camcal_step' and fields.get('sequence') == 53:
                step_written.set()
        ctx.events.emit = observe
        record = json.loads((args.session / 'session_record.json').read_text())
        session = dict(shots=0, engine=None, directory='replay', start=record['start_xyz'], target=None)
        board = WorldBoard()
        clock = BTClock(board)
        worker = CamcalWorker()
        clock.attach_worker(worker)
        try:
            for i in range(1, 54):
                image = cv2.imread(str(args.session / f'camcal_s{i:02d}_raw.png'))
                if image is None:
                    raise FileNotFoundError(f'frame {i}')
                probe = json.loads((args.session / f's{i:02d}_plc_actual.json').read_text())
                actual = probe['algorithm_actual_xyz']
                board.pass_note('camcal', 'execute', {'__request_id__': i, 'context': ctx, 'session': session,
                                'captured': image, 'actual': actual, 'task_id': i}, sender='replay')
                if i == 53 and args.reproduce_pause:
                    clock.tick_once()  # 投递请求后模拟旧版 gate 暂停，不再 tick
                    if not step_written.wait(120):
                        raise TimeoutError('末帧 worker 未产生 camcal_step')
                    if board.read_state('camcal.results', {}).get(i) is not None:
                        raise AssertionError('暂停 tick 期间结果意外发布')
                    print('复现：camcal_step 已写入，但暂停 tick 导致结果未发布；现恢复 tick', flush=True)
                deadline = time.monotonic() + 120
                while time.monotonic() < deadline:
                    clock.tick_once()
                    result = board.read_state('camcal.results', {}).get(i)
                    if result is not None:
                        break
                    time.sleep(.002)
                else:
                    raise TimeoutError(f'frame {i}: worker result not delivered')
                if 'error' in result:
                    raise RuntimeError(result['error'])
                step = result['value']['step']
                print(f'{i}: {step.kind} {step.code or ""} {step.reason or ""}', flush=True)
                if step.kind != 'move':
                    break
                session['target'] = step.target_xyz
        finally:
            clock.shutdown()


if __name__ == '__main__':
    main()
