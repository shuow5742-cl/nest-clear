import cv2
import numpy as np
import pytest
import yaml

from nest_clear.config import load_config
from nest_clear.guard import GuardError
from nest_clear.tooltip import SilhouetteTipDetector
from nest_clear.tools.tip_jitter import main


def needle():
    image = np.full((800, 800), 220, np.uint8)
    cv2.rectangle(image, (336, 0), (464, 330), 30, -1)
    cv2.ellipse(image, (400, 330), (64, 70), 0, 0, 180, 30, -1)
    return cv2.resize(image, (100, 100), interpolation=cv2.INTER_AREA)


def test_focus_blur_noise_and_direction():
    image = needle()
    detector = SilhouetteTipDetector()
    result = detector.detect(image)
    assert np.all(np.abs(np.array(result.point_uv) - [49.5, 49.5]) < .3)
    assert result.focused and result.edge_width_px <= 3
    blurred = detector.detect(cv2.GaussianBlur(image, (0, 0), 6))
    assert not blurred.focused
    assert np.linalg.norm(np.array(blurred.point_uv) - [49.5, 49.5]) < (8 if blurred.coarse else 3)
    rng = np.random.default_rng(7)
    samples = [detector.detect(np.clip(image.astype(float) + rng.normal(0, 3, image.shape), 0, 255).astype(np.uint8)).point_uv for _ in range(50)]
    assert np.all(np.std(samples, axis=0) < .15)
    horizontal = SilhouetteTipDetector(direction=(1, 0)).detect(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))
    assert np.all(np.abs(np.array(horizontal.point_uv) - [49.5, 49.5]) < .3)


@pytest.mark.parametrize('key,value', [('probe_offset_px', 0), ('probe_offset_px', 1.5), ('max_edge_width_px', -1), ('max_edge_width_px', float('nan'))])
def test_config_invalid(tmp_path, key, value):
    config = yaml.safe_load(open('config/nest_clear.yaml'))
    config['tool_calibration']['detection']['front'][key] = value
    path = tmp_path / 'bad.yaml'
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(GuardError, match=key):
        load_config(path, require_tools=False)


def test_jitter(tmp_path, capsys):
    for i in range(3):
        cv2.imwrite(str(tmp_path / f'{i}.png'), needle())
    assert main([str(tmp_path)]) == 0
    text = capsys.readouterr().out
    assert 'std=' in text and '0.03mm 阈值稳' in text


def test_defocus_hold_and_clear_alignment(monkeypatch):
    from types import SimpleNamespace
    from nest_clear.flow.calibration import PrepareCalibrationReply
    from test_hair_integration import Board, Events
    import nest_clear.flow.calibration as calibration

    monkeypatch.setattr(calibration, 'is_aligned', lambda result, tolerance: True)
    monkeypatch.setattr(calibration, 'guard_position', lambda *args: None)
    config = load_config('config/nest_clear.yaml', require_tools=False)
    events = Events()
    node = PrepareCalibrationReply(SimpleNamespace(config=config, events=events))
    session = {'tool': 1, 'iterations': 0, 'offset': False, 'ended': False}
    focus = {'front': {'focused': False, 'edge_width_px': 12, 'width_px': 16, 'coarse': False, 'max_edge_width_px': 6.0},
             'side': {'focused': True, 'edge_width_px': 2, 'width_px': 16, 'coarse': False, 'max_edge_width_px': 6.0}}
    board = Board(tooltip_session=session, tooltip_result=SimpleNamespace(delta_xyz=np.zeros(3)),
                  tooltip_focus=focus, actual=(0, 0, 0), task_id=1)
    node._blackboard = board
    node.perform()
    assert board.read('reply_func') == 100
    node.perform()
    assert board.read('reply_func') == 110
    with pytest.raises(GuardError, match='镜头焦点与基准位置不一致'):
        node.perform()
    focus['front']['focused'] = True
    node.perform()
    assert board.read('reply_func') == 111


def test_old_detection_config(tmp_path):
    config = yaml.safe_load(open('config/nest_clear.yaml'))
    for view in config['tool_calibration']['detection'].values():
        view.pop('probe_offset_px')
        view.pop('max_edge_width_px')
    path = tmp_path / 'old.yaml'
    path.write_text(yaml.safe_dump(config))
    assert load_config(path, require_tools=False).tool_calibration.detection['front']['min_area'] == 5


def test_tapered_tip_focus():
    scale = 8
    image = np.full((140 * scale, 100 * scale), 220, np.uint8)
    cv2.rectangle(image, (42 * scale, 0), (58 * scale, 60 * scale), 30, -1)
    cone = np.array([[42 * scale, 60 * scale], [58 * scale, 60 * scale],
                     [50 * scale, 90 * scale]], np.int32)
    cv2.fillPoly(image, [cone], 30)
    image = cv2.resize(image, (100, 140), interpolation=cv2.INTER_AREA)
    detector = SilhouetteTipDetector()
    sharp = detector.detect(cv2.GaussianBlur(image, (0, 0), 1.0))
    blurry = detector.detect(cv2.GaussianBlur(image, (0, 0), 3.0))
    assert sharp.point_uv is not None and sharp.focused, sharp
    assert blurry.point_uv is not None and not blurry.focused, blurry
    if blurry.edge_width_px is None:
        assert '灰度不稳定' in blurry.diagnostic


def test_report_focus_width_format(tmp_path):
    import json
    from datetime import date
    from nest_clear.record.report import report

    day = date(2026, 1, 2)
    directory = tmp_path / day.isoformat()
    directory.mkdir()
    rows = [
        {'tool': 1, 'defocus_iterations': 2, 'final_edge_width_px': {'front': .786666, 'side': None}},
        {'tool': 2},
    ]
    (directory / 'tooltip_summary.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    result = report(tmp_path, day)
    assert '离焦轮数 2，最终边缘宽度 正面 0.8 px / 侧面 未测' in result
    assert '最终边缘宽度 未记录' in result
    assert 'front' not in result and 'side' not in result


@pytest.mark.parametrize('configured', [None, 4.25])
def test_detect_records_actual_focus_threshold(configured):
    from types import SimpleNamespace
    from nest_clear.flow.calibration import DetectTips
    from test_hair_integration import Board, Events

    config = load_config('config/nest_clear.yaml', require_tools=False)
    for view in ('front', 'side'):
        settings = config.tool_calibration.detection[view]
        if configured is None:
            settings.pop('max_edge_width_px')
        else:
            settings['max_edge_width_px'] = configured
    node = DetectTips(SimpleNamespace(config=config, events=Events()))
    views = [SimpleNamespace(name=name, target_uv=(0, 0)) for name in ('front', 'side')]
    board = Board(tooltip_images={'front': needle(), 'side': needle()},
                  tooltip_session={'tool': 1, 'images': [{'overlay': {}, 'raw': {}, 'iteration': 0}]},
                  tooltip_baseline=SimpleNamespace(views=views), task_id=1)
    node._blackboard = board
    node.perform()
    for view in ('front', 'side'):
        actual = SilhouetteTipDetector(**config.tool_calibration.detection[view]).max_edge_width_px
        assert board.read('tooltip_focus')[view]['max_edge_width_px'] == actual
