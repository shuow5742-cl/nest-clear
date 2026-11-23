import json
from types import SimpleNamespace

import numpy as np
import pytest

from nest_clear.camcal.config import CameraCalibConfig
from nest_clear.camcal.probe import make_probe, summarize
from nest_clear.camcal.session import CameraCalibrationSession
from nest_clear.config import _camera_calibration
from nest_clear.ui.status import MainStatus


@pytest.mark.parametrize('px,kind', [(1.70, 'done'), (3., 'fail')])
def test_repeat_mm(monkeypatch, px, kind):
    cfg = CameraCalibConfig()
    session = CameraCalibrationSession.__new__(CameraCalibrationSession)
    session.cfg, session._obs, session._plan = cfg, [{}] * 52, [None] * 52
    session.nominal, session.image_size = None, (1, 1)
    report = {'repeat_px': px, 'repeat_shift_mm': px * 24.84 / 1000,
              'repeat_limit_mm': .05, 'repeat_plc_delta_mm': .03,
              'coverage': 1., 'axis_scale': {'x': 1., 'y': 1.}, 'rms_px': 0., 'per_image_rms': [0.]}
    monkeypatch.setattr('nest_clear.camcal.session.solve_offline', lambda *args: SimpleNamespace(report=report))
    monkeypatch.setattr(session, 'session_record', lambda: {})
    result = session._accept_observation(np.zeros(3), np.array([1]), np.zeros((1, 2)), {}, None)
    assert result.kind == kind
    if kind == 'fail':
        assert result.code == 'CAMCAL-REPEAT' and 'mm' in result.reason and '上限' in result.reason


def test_legacy_config():
    cfg = CameraCalibConfig(**_camera_calibration({'max_repeat_px': .5}))
    assert cfg.module_repeatability_mm == .02


def test_probe_summary_and_status(tmp_path):
    def samples(values): return [{'xyz': [x, 2., 3.], 'words': [0] * 6} for x in values]
    a = make_probe(2, [1., 2., 3.], samples([1., 1.01, 1., 1., 1.]), samples([1.02]), [1., 2., 3.])
    b = make_probe(28, [1., 2., 3.], samples([1.03] * 5), samples([1.03]), [1.03, 2., 3.])
    summary = summarize([a, b])
    assert summary['shots'][0]['before_jitter_mm'][0] == pytest.approx(.01)
    assert summary['shots'][0]['after_minus_before_mm'][0] == pytest.approx(.02)
    assert summary['repeated_targets'][0]['pairs'][0]['delta_xy_mm'] == pytest.approx(.03)
    assert summary['shots'][1]['before_jitter_mm'] == [0., 0., 0.]
    status = MainStatus(tmp_path)
    status.observe({'event': 'camcal_step', 'sequence': 2, 'plan': [[0, 0, 0]] * 3, 'plc_actual_probe': a})
    status.observe({'event': 'request', 'func': 80})
    assert status.data['camcal']['sequence'] == 2
