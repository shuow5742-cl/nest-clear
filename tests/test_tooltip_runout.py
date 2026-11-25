import json

import numpy as np
import pytest

from nest_clear.tooltip.runout import RunoutSession


def samples(diameter, rotated=False):
    rng = np.random.default_rng(780)
    angles = rng.uniform(0, 2 * np.pi, 10)
    theta = np.deg2rad(2 if rotated else 0)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    mapping = rotation * 0.004
    for angle in angles:
        x, y = diameter / 2000 * np.array([np.cos(angle), np.sin(angle)])
        front = np.linalg.solve(mapping, [x, 0]) + rng.normal(0, 0.15, (5, 2))
        side = np.linalg.solve(mapping, [y, 0]) + rng.normal(0, 0.15, (5, 2))
        yield front.tolist(), side.tolist(), mapping, mapping


def run(diameter, rotated=False):
    session = RunoutSession()
    for args in samples(diameter, rotated):
        result = session.add_press(*args, meta={"source": "标定"})
        assert not result["rejected"]
    return session, result


@pytest.mark.parametrize("diameter,passes", [(40, True), (70, False)])
def test_circle_and_pass(diameter, passes):
    session, result = run(diameter)
    assert diameter - 4 <= result["max_deviation_um"] <= diameter + 1
    assert result["pass"] is passes
    assert result["circle_fit_diameter_um"] == pytest.approx(diameter, abs=3)
    assert len(session.to_record()["presses"]) == 10
    json.dumps(session.to_record(), ensure_ascii=False)


def test_rejection_preserves_session():
    session, result = run(40)
    before = session.to_record()
    identity = np.eye(2)
    rejected = session.add_press([(0, 0), None], [(0, 0), (0, 0)], identity, identity)
    assert rejected["rejected"] and "第 2 帧未识别到针尖" in rejected["reason"]
    rejected = session.add_press([(0, 0), (2, 0)], [(0, 0), (0, 0)], identity, identity)
    assert rejected["rejected"] and "工具未停稳（正面 极差 2.00 px）" in rejected["reason"]
    assert session.to_record() == before
    assert before["presses"][-1]["result"] == result


def test_stability_and_insufficient_circle():
    session = RunoutSession(stable_presses=3, stable_delta_um=2)
    mapping = np.eye(2) * 0.001
    for i, x in enumerate([0, 10, 10, 10, 10]):
        result = session.add_press([(x, 0)] * 5, [(0, 0)] * 5, mapping, mapping)
        assert result["stable"] == (i == 4)
        assert result["circle_fit_diameter_um"] is None
        assert result["circle_fit_reason"] == "点数或角度覆盖不足"


def test_rotated_mapping():
    _, normal = run(40)
    _, rotated = run(40, rotated=True)
    assert rotated["max_deviation_um"] == pytest.approx(normal["max_deviation_um"], abs=1)
    assert rotated["circle_fit_diameter_um"] == pytest.approx(40, abs=3)
