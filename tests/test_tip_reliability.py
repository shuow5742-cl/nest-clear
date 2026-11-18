from types import SimpleNamespace
import time
import numpy as np
import pytest
from autoweaver.motion_policy.nodes.node import Status
from nest_clear.flow.calibration import SolveTips, ReliableTips
from nest_clear.tooltip import ViewCalib, ToolTipBaseline
from nest_clear.guard import GuardError


class Board:
    def __init__(self, **values): self.values = values
    def read(self, key): return self.values.get(key)
    def write(self, key, value, name): self.values[key] = value


class Events:
    def emit(self, *args, **kwargs): pass


def context():
    cfg = SimpleNamespace(max_residual_mm=.1, max_condition=10., z_disagreement_mm=.1,
                          tip_tolerance_mm=.03, click_timeout_s=.01, quality_hard_fail=True)
    return SimpleNamespace(config=SimpleNamespace(tool_calibration=cfg), events=Events())


def board():
    baseline = ToolTipBaseline('tool1', (0,0,0), [
        ViewCalib('front', (0,0), ((10,0,0),(0,0,10)), 10),
        ViewCalib('side', (0,0), ((0,10,0),(0,0,10)), 10)])
    focus = {v: dict(focused=True, coarse=False, width_px=10) for v in ('front','side')}
    return Board(tooltip_baseline=baseline, tooltip_points={'front': (0.,0.), 'side': (0.,0.)},
                 tooltip_focus=focus, tooltip_session={'tool': 1, 'images': [{}]}, task_id=1)


@pytest.mark.parametrize('kind,reason', [('missing','检测失败'), ('coarse','coarse'), ('blur','不清晰'),
                                         ('width','宽度'), ('z','Z 分歧'), ('condition','条件数')])
def test_reliability_reasons(kind, reason):
    ctx, bb = context(), board()
    if kind == 'missing': bb.values['tooltip_points']['front'] = None
    if kind == 'coarse': bb.values['tooltip_focus']['front']['coarse'] = True
    if kind == 'blur': bb.values['tooltip_focus']['front']['focused'] = False
    if kind == 'width': bb.values['tooltip_focus']['front']['width_px'] = 14
    if kind == 'z': bb.values['tooltip_points']['side'] = (0, 10)
    if kind == 'condition': ctx.config.tool_calibration.max_condition = 1.01
    node = SolveTips(ctx); node._blackboard = bb
    with pytest.raises(GuardError, match=reason): node.perform()


def test_click_timeout_is_nonblocking():
    ctx = context(); node = ReliableTips(ctx); node._blackboard = board()
    class Process:
        def is_alive(self): return True
        def terminate(self): pass
        def join(self, timeout): pass
    class Pipe:
        def poll(self): return False
        def close(self): pass
    node.process, node.connection, node.started = Process(), Pipe(), time.monotonic()
    before = time.monotonic()
    assert node.on_running() == Status.RUNNING
    assert time.monotonic() - before < .3
    node.started -= 1
    assert node.on_running() == Status.FAILURE
    assert '超时' in node._blackboard.read('error')


def test_click_result_and_cancel():
    from nest_clear.tooltip.click import ClickResult
    class Process:
        def join(self, timeout): pass
    class Pipe:
        def __init__(self, result): self.result = result
        def poll(self): return True
        def recv(self): return self.result
        def close(self): pass
    for cancelled in (False, True):
        ctx = context(); bb = board(); node = ReliableTips(ctx); node._blackboard = bb
        point = None if cancelled else (0., 0.)
        result = ClickResult(point, 'cancel' if cancelled else 'manual', 8., 2.)
        node.process, node.connection = Process(), Pipe({'front': result, 'side': result})
        assert node.on_running() == (Status.FAILURE if cancelled else Status.SUCCESS)
        if not cancelled:
            assert len(bb.read('tooltip_session')['tip_clicks']) == 2


@pytest.mark.parametrize('kind', ['residual', 'condition', 'z'])
@pytest.mark.parametrize('hard', [False, True])
def test_quality_hard_fail_controls_quality_only(monkeypatch, kind, hard):
    from dataclasses import replace
    import nest_clear.flow.calibration as flow
    ctx, bb = context(), board()
    ctx.config.tool_calibration.quality_hard_fail = hard
    solve = flow.solve_correction
    def altered(*args, **kwargs):
        result = solve(*args, **kwargs)
        return replace(result, condition_number=20. if kind == 'condition' else 1.,
                       axis_disagreement={'z': .2 if kind == 'z' else 0.})
    monkeypatch.setattr(flow, 'solve_correction', altered)
    monkeypatch.setattr(flow, 'check_quality', lambda *a: SimpleNamespace(
        ok=False, residuals_mm={'front': .2 if kind == 'residual' else 0.}, condition_number=20.))
    node = SolveTips(ctx); node._blackboard = bb
    if hard:
        with pytest.raises(GuardError) as exc:
            node.perform()
        text = str(exc.value)
        assert {'residual': ('正面相机投影残差', '0.200', '0.100'),
                'condition': ('条件数', '20.000', '10.000'),
                'z': ('Z 分歧', '0.200', '0.100')}[kind][0] in text
        assert all(v in text for v in {'residual': ('0.200', '0.100'), 'condition': ('20.000', '10.000'), 'z': ('0.200', '0.100')}[kind])
    else:
        node.perform()
        bb.read('tooltip_focus')['front']['coarse'] = True
        with pytest.raises(GuardError, match='coarse'):
            node.perform()


def test_missing_detection_keeps_original_diagnostic_and_advice():
    ctx, bb = context(), board()
    bb.read('tooltip_points')['front'] = None
    bb.read('tooltip_focus')['front']['diagnostic'] = 'no dark component'
    node = SolveTips(ctx); node._blackboard = bb
    with pytest.raises(GuardError) as exc:
        node.perform()
    assert '正面相机无法识别针尖：图像中未找到暗色针尖区域' in str(exc.value)
    assert '检查背光与曝光，以及针尖是否在画面内。' in str(exc.value)
