"""真实 TCP、主程序和 ChArUco 图像；需要在允许本地监听的环境运行。"""
from dataclasses import replace
from datetime import date
import json
import re
import shutil
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from nest_clear.config import load_config
from nest_clear.main import Application
from nest_clear.vision.geometry import Calibration, load_calibration, pixel_to_module
from sim.charuco_scene import (CharucoScene, SimulatedCharucoCamera, K, DISTORTION, ROTATION,
                               BOARD_ORIGIN, BOARD_ROTATION)
from sim.fake_plc import FakePLC

ROOT = Path(__file__).resolve().parents[1]

# 第七帧约预计 176 个角点，60% 覆盖门槛约 106 个；此亮斑遮挡后剩约 78 个。
LARGE_GLARE_AXES = (1200, 1000)


@pytest.fixture
def running(tmp_path, monkeypatch, request):
    # 用户要求本次工作不执行 git 命令。
    monkeypatch.setattr('nest_clear.main.repository_commit', lambda *_: 'unknown')
    monkeypatch.setattr('nest_clear.main.repository_dirty', lambda *_: None)
    monkeypatch.setattr('nest_clear.main.framework_commit', lambda: 'unknown')
    with FakePLC() as plc:
        calibration_dir = tmp_path / 'calibration'
        shutil.copytree(ROOT / 'config/calibration/examples', calibration_dir)
        path = calibration_dir / 'camera.yaml'
        cfg = load_config(ROOT / 'config/nest_clear.yaml')
        camcal_cfg = {**cfg.camera_calibration, 'start_xyz': [0., 0., 110.]}
        if getattr(request, 'param', None) == 'large_glare':
            # 让大亮斑进入角点覆盖检查，而非先触发整体过曝。
            camcal_cfg['max_saturated_fraction'] = .5
            camcal_cfg['min_corner_ratio'] = .6
        cfg = replace(cfg, travel=((-500., 500.),)*3, camera_calibration=camcal_cfg, plc=replace(cfg.plc, ip=plc.host, port=plc.port),
                      record_directory=str(tmp_path / 'record'),
                      vision=replace(cfg.vision, calibration=str(path), camera_backend='mock',
                                     detector='stub', allow_placeholder_calibration=True),
                      tool_calibration=replace(cfg.tool_calibration, enabled=False, directory=str(calibration_dir)))
        camera = SimulatedCharucoCamera(plc)
        app = Application(cfg, ROOT / 'config/nest_clear.yaml', camera=camera)
        stop = threading.Event()
        errors = []
        def loop():
            try:
                while not stop.is_set():
                    app.step()
                    time.sleep(.002)
            except BaseException as exc:
                errors.append(exc)
        thread = threading.Thread(target=loop)
        thread.start()
        try:
            yield plc, app, camera, path
        finally:
            stop.set()
            thread.join(10)
            app.close()
            assert not errors


def events(app):
    return [json.loads(line) for line in (app.events.run_directory / 'events.jsonl').read_text().splitlines()]


def test_normal_and_offline(running, tmp_path, caplog):
    plc, app, camera, path = running
    caplog.set_level('WARNING')
    import nest_clear.camcal.session as camcal
    original_solve = camcal.solve_offline
    durations = []
    def timed_solve(*args, **kwargs):
        start_solve = time.monotonic()
        try:
            return original_solve(*args, **kwargs)
        finally:
            durations.append(time.monotonic() - start_solve)
    from unittest.mock import patch
    start = time.monotonic()
    with patch.object(camcal, 'solve_offline', timed_solve):
        replies = plc.start_camera_calibration((0., 0., 110.), timeout_s=120)
    elapsed = time.monotonic() - start
    assert len(replies) == 53 and replies[-1]['func_code'] == 81
    # result_id 已由 PLC worker 写入，但行为树可能尚未记录 reply。
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if any(r['event'] == 'reply' and r.get('task_id') == replies[-1]['task_id'] for r in events(app)):
            break
        time.sleep(.01)
    else:
        pytest.fail('最终 PLC 应答未记录')
    cal = load_calibration(path)
    assert not cal.placeholder
    assert np.max(np.abs(cal.K.diagonal()[:2] / K.diagonal()[:2] - 1)) < .001
    assert np.max(np.abs(cal.K[:2, 2] - K[:2, 2])) < 2
    angle = lambda r: np.degrees(np.arccos(np.clip(abs(r[2, 2]), 0, 1)))
    assert abs(angle(cal.rotation) - angle(ROTATION)) < .05
    truth = Calibration(K, DISTORTION, ROTATION, np.zeros(3), 0., False, BOARD_ORIGIN, BOARD_ROTATION[:, 2])
    principal = np.asarray(pixel_to_module(K[0, 2], K[1, 2], (0, 0, 110), truth))
    for u in np.linspace(0, 4023, 33):
        for v in np.linspace(0, 3035, 25):
            a = np.asarray(pixel_to_module(u, v, (0, 0, 110), cal))
            b = np.asarray(pixel_to_module(u, v, (0, 0, 110), truth))
            error = np.linalg.norm((a - b)[:2])
            assert error < .01
            if np.linalg.norm((b - principal)[:2]) <= 5:
                assert error < .005
    assert list((path.parent / 'history').glob('*.yaml'))
    assert app.vision.calibration is app.calibration
    app.archive.executor.submit(lambda: None).result(timeout=60)
    rows = events(app)
    assert len([r for r in rows if r['event'] == 'camcal_step']) == 53
    assert list(app.events.run_directory.rglob('*camcal*raw*'))
    assert list(app.events.run_directory.rglob('*camcal*ovl*'))
    record = next(app.events.run_directory.rglob('session_record.json'))
    offline = tmp_path / 'offline.yaml'
    subprocess.run([sys.executable, '-m', 'nest_clear.tools.camera_calib_solve', str(record),
                    '--nominal', str(next((path.parent / 'history').glob('*.yaml'))), '--out', str(offline)],
                   check=True, timeout=120)
    other = load_calibration(offline)
    np.testing.assert_allclose(other.K, cal.K, rtol=1e-6, atol=1e-6)
    for name in ('rotation', 'distortion', 'plane_point', 'plane_normal'):
        np.testing.assert_allclose(getattr(other, name), getattr(cal, name), rtol=1e-6, atol=1e-8)
    from nest_clear.record.report import report
    daily = report(app.events.directory, date.today())
    assert '相机标定：' in daily and '53 张，成功' in daily
    assert len(durations) == 1
    # 同进程的全分辨率仿真渲染会争用 GIL/CPU，偶发超过 40ms；生产环境没有该渲染器。
    # 仍检查 300ms 以上的阻塞，以捕获检测或求解误落在 tick 中。
    for record in caplog.records:
        message = record.getMessage()
        if 'slow tick' in message.lower():
            match = re.search(r'slow tick.*?:\s*([0-9]+(?:\.[0-9]+)?)\s*ms', message, re.IGNORECASE)
            assert match is not None, f'无法解析 slow tick 耗时：{message}'
            assert float(match.group(1)) < 300, message
    print(f'完整 TCP 墙钟耗时: {elapsed:.3f}s，在线求解: {durations[0]:.3f}s')



def test_glare_geometry_without_tcp():
    """以真实渲染器和生产参数覆盖第七帧亮斑及完整小亮斑会话。"""
    import copy
    from nest_clear.camcal import CameraCalibConfig, CameraCalibrationSession

    settings = dict(load_config(ROOT / 'config/nest_clear.yaml').camera_calibration)
    settings['start_xyz'] = [0., 0., 110.]
    settings['min_corner_ratio'] = 0.6
    cfg = CameraCalibConfig(**settings)
    scene = CharucoScene()
    nominal = load_calibration(ROOT / 'config/calibration/examples/camera.yaml')
    session = CameraCalibrationSession(cfg, nominal, scene.size, settings['start_xyz'])
    pose = tuple(settings['start_xyz'])
    for _ in range(6):
        step = session.add(scene.render(pose), pose)
        assert step.kind == 'move', (step.code, step.reason)
        pose = step.target_xyz

    # 与 TCP 大亮斑夹具一致：仅为测试覆盖率分支提高饱和比例门槛。
    large = copy.copy(session)
    large._obs = list(session._obs)
    large.cfg = replace(cfg, max_saturated_fraction=.5)
    step = large.add(scene.render(pose, glare=((2012, 1518), LARGE_GLARE_AXES)), pose)
    assert step.kind == 'fail' and step.code == 'CAMCAL-GLARE'
    assert step.detail['glare_blobs']
    assert step.detail['usable_corners'] < cfg.min_corner_ratio * step.detail['predicted_corners']

    # 小亮斑按生产饱和阈值继续运行，整个真实求解会话仍须成功。
    step = session.add(scene.render(pose, glare=((2012, 1518), (100, 80))), pose)
    assert step.kind == 'move' and step.detail['glare_blobs']
    assert step.detail['usable_corners'] >= cfg.min_corner_ratio * step.detail['predicted_corners']
    for _ in range(52):
        if step.kind != 'move':
            break
        pose = step.target_xyz
        step = session.add(scene.render(pose), pose)
    assert step.kind == 'done', (step.code, step.reason)


def test_small_glare_is_accepted(running):
    plc, app, camera, _ = running
    camera.glare_frames.add(7)
    replies = plc.start_camera_calibration((0, 0, 110), timeout_s=120)
    assert replies[-1]['func_code'] == 81
    warnings = [r for r in events(app) if r['event'] == 'camcal_glare_warning' and r['image_index'] == 7]
    assert len(warnings) == 1
    assert warnings[0]['sequence'] == 7
    assert warnings[0]['bboxes'] and len(warnings[0]['bboxes'][0]) == 4
    assert isinstance(warnings[0]['excluded_corners'], int) and warnings[0]['excluded_corners'] >= 0


@pytest.mark.parametrize('running', ['large_glare'], indirect=True)
def test_large_glare_and_restart(running):
    plc, app, camera, _ = running
    camera.glare_frames.add(7)
    camera.glare_axes = LARGE_GLARE_AXES
    replies = plc.start_camera_calibration((0, 0, 110), timeout_s=120)
    assert replies[-1]['func_code'] == 82
    failed = [r for r in events(app) if r['event'] == 'camcal_failed'][-1]
    assert failed['code'] == 'CAMCAL-GLARE' and failed['detail']['bbox']
    detail = failed['detail']
    assert detail['image_index'] == 7
    assert detail['usable_corners'] < .6 * detail['predicted_corners']
    # 回到起始位再重新启动，否则会触发板面偏心检查。
    plc.set_actual_xyz((0, 0, 110))
    plc.physical_xyz = [0, 0, 110]
    assert plc.wait_reply(plc.request(80), 120)['func_code'] == 80


def test_axis_scale(running):
    plc, app, _, _ = running
    app.config.camera_calibration["max_axis_scale_error"] = .002
    replies = plc.start_camera_calibration((0, 0, 110), timeout_s=120, x_scale_error=.003)
    assert replies[-1]['func_code'] == 82
    assert [r for r in events(app) if r['event'] == 'camcal_failed'][-1]['code'] == 'CAMCAL-AXIS-SCALE'


def test_arrival_error(running):
    plc, _, _, _ = running
    replies = plc.start_camera_calibration((0, 0, 110), timeout_s=120,
                                           position_error=(.2, 0, 0), error_step=3)
    assert replies[-1]['func_code'] == 999
