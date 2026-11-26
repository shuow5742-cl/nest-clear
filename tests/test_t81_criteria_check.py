"""Exercise offline reporting using only temporary synthetic images."""
import cv2
from test_trident import scene
from tools.tip_eval.exposure_criteria_check import evaluate, summarize
from nest_clear.tooltip.exposure_scan import _frame_metrics, absolute_criteria
from nest_clear.tooltip.trident import TridentTipDetector


def test_evaluation_uses_production_frame_metrics_and_gates(tmp_path):
    path = tmp_path / 'device' / 'set_001' / 'front_1.png'
    path.parent.mkdir(parents=True)
    image = scene()
    assert cv2.imwrite(str(path), image)
    config = {'detection': {'front': {}}, 'contrast_min': 40, 'saturation_max': .05}
    result = evaluate((str(path), str(tmp_path), config))
    metrics = _frame_metrics(image, TridentTipDetector())
    expected = absolute_criteria(metrics, 3, 40, .05)
    assert all(result[key] == value for key, value in expected.items())
    assert result['frame'] == 'device/set_001/front_1.png'


def test_report_excludes_only_rms_over_two_and_lists_failures(tmp_path):
    base = dict(frame='device/set_001/front_1.png', camera='front', axis_rms_px=None,
                diagnostic='末端对比度不足', expected_prongs=2, prongs_value=1,
                contrast_value=20., whiteness_value=.1, recognized=False, prongs=False,
                contrast=False, whiteness=False, excluded=False)
    rows = [base, {**base, 'frame': 'excluded.png', 'axis_rms_px': 2.1, 'excluded': True}]
    report = tmp_path / 'report.md'
    eligible = summarize(rows, tmp_path / 'exposure_check', report, {'contrast_min':40, 'saturation_max':.05})
    text = report.read_text(encoding='utf-8')
    assert len(eligible) == 1 and '0/1' in text
    assert 'device/set_001/front_1.png' in text and 'excluded.png' not in text
    assert '末端对比度不足' in text


def test_group_consistency_requires_five_non_distorted_frames():
    from tools.tip_eval.exposure_criteria_check import group_consistency
    def frame(group, number, count, excluded=False):
        return {'frame': f'device/{group}/front_{number}.png', 'camera': 'front',
                'prongs_value': count, 'excluded': excluded}
    rows = [frame('set_001', i, 2) for i in range(1, 6)]
    rows += [frame('set_002', i, 3 if i < 5 else 2) for i in range(1, 6)]
    rows += [frame('set_003', i, 2, excluded=i == 5) for i in range(1, 6)]
    groups = group_consistency(rows)
    assert groups[0]['consistent'] is True and groups[0]['counts'] == [2] * 5
    assert groups[1]['consistent'] is False
    assert groups[2]['consistent'] is None and groups[2]['eligible_frames'] == 4
