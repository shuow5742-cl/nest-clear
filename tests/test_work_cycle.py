from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from nest_clear.flow.work_cycle import ProtocolViolation, State, WorkCycle
from nest_clear.vision.fine import FineResult


@dataclass
class Observation:
    target: object
    area: float = 1
    reason: str | None = None
    status: str = "eligible"


def obs(x, area=1, status="eligible", reason=None):
    return Observation(SimpleNamespace(xyz=(float(x), 0., 0.)), area, reason, status)


def cycle(views=((0., 0., 5.),), **kwargs):
    params = dict(skip_radius_mm=0.1, max_fine_targets=10, fine_box_half_mm=4., cap_ratio=1.2,
                  tool_check_min_attempts=3, tool_check_ratio=0.5, max_coarse_shots_per_view=30)
    params.update(kwargs)
    return WorkCycle(list(views), lambda xyz, pose: (xyz[0] + pose[0], pose[1] + 7, pose[2]),
                     lambda xyz: ((xyz[0] + 1, 0, 1), (xyz[0] + 2, 0, 2)), SimpleNamespace(**params))


def fine(c, x):
    return c.on_fine((0., 0., 5.), FineResult("ok", target_xyz=(float(x), 0., 0.)))


def test_multiview_and_failures():
    c = cycle(views=((0., 0., 5.), (100., 0., 5.)), tool_check_min_attempts=99, cap_ratio=3)
    assert c.expect(1) == "start"
    assert c.start().scan_xyz == (0., 0., 5.)
    r = c.on_coarse((3., 2., 5.), [obs(1), obs(2, 2)])
    assert (r.removable_count, r.scan_xyz) == (2, (5., 9., 5.))
    r = fine(c, 2.05)
    assert (r.func_code, r.tool1_xyz, r.tool2_xyz) == (3, (3.05, 0, 1), (4.05, 0, 2))
    assert c.on_picked().scan_xyz == (4., 9., 5.)
    for i, failure in enumerate(("fine_empty", "fine_point_outside", "fine_rejected")):
        r = c.on_fine((0, 0, 5), FineResult(failure))
        assert r.scan_xyz == (0., 0., 5.)
        assert c.is_attempted((1. if i == 0 else 10. * i, 0, 0))
        previous = obs(1)
        r = c.on_coarse((0, 0, 5), [previous, obs(10 * (i + 1))] if i < 2 else [previous])
        assert previous.status == "failed" and r.removable_count == (1 if i < 2 else 0)
    assert r.scan_xyz == (100., 0., 5.) and c.view_index == 1
    assert c.on_coarse((100, 0, 5), []).func_code == 9
    assert [a.reason for a in c.attempted] == ["ok", "fine_empty", "fine_point_outside", "fine_rejected"]
    assert c.is_attempted((2.05, 0, 0))


def test_priority_limit_and_filtering():
    c = cycle()
    c.start()
    entries = [obs(i, i) for i in range(12)] + [obs(50, 100, "other_view"), obs(51, 100, "rejected", "small")]
    r = c.on_coarse((0, 0, 5), entries)
    assert r.removable_count == 12
    assert [o.area for o in c.targets] == list(range(11, 1, -1))
    assert c.current_target().area == 11


@pytest.mark.parametrize("initial, allowed", [(1, 2), (10, 12)])
def test_cap_final_check(initial, allowed):
    c = cycle(tool_check_min_attempts=99)
    c.start()
    r = c.on_coarse((0, 0, 5), [obs(i * 10) for i in range(initial)])
    assert c.n0 == initial and c.allowed_attempts == allowed
    for i in range(allowed):
        if c.state == State.WAIT_COARSE:
            c.on_coarse((0, 0, 5), [obs(1000 + i * 10)])
        fine(c, c.current_target().target.xyz[0])
        c.on_picked()
    assert c.final_check and c.state == State.WAIT_COARSE
    r = c.on_coarse((0, 0, 5), [obs(9999)])
    assert r.func_code == 9 and r.removable_count == 1 and r.event["reason"] == "cap"


@pytest.mark.parametrize("min_attempts, red, expected", [(1, True, 10), (1, False, 9), (2, True, 9)])
def test_tool_check(min_attempts, red, expected):
    c = cycle(tool_check_min_attempts=min_attempts, cap_ratio=1)
    c.start()
    c.on_coarse((0, 0, 5), [obs(1)])
    fine(c, 1)
    c.on_picked()
    r = c.on_coarse((0, 0, 5), [obs(1)] if red else [])
    assert r.func_code == expected
    assert r.event["tool_check_total"] == int(red)


def test_coarse_limit_and_failure_cap():
    c = cycle(cap_ratio=100, max_coarse_shots_per_view=1, tool_check_min_attempts=99)
    c.start()
    c.on_coarse((0, 0, 5), [obs(1)])
    c.on_fine((0, 0, 5), FineResult("fine_empty"))
    r = c.on_coarse((0, 0, 5), [obs(2)])
    assert r.func_code == 9 and r.event["warning"] and r.event["reason"] == "coarse_limit"
    c = cycle(cap_ratio=1, tool_check_min_attempts=99)
    c.start()
    c.on_coarse((0, 0, 5), [obs(1)])
    c.on_fine((0, 0, 5), FineResult("fine_empty"))
    assert c.final_check
    assert c.on_coarse((0, 0, 5), [obs(2)]).event["reason"] == "cap"


def test_protocol_all_states():
    c = cycle(cap_ratio=1, tool_check_min_attempts=1)
    def check(valid):
        for code in (1, 20, 60, 100, 999):
            if code != valid:
                with pytest.raises(ProtocolViolation, match=f"实际功能码 {code}"):
                    c.expect(code)
    check(1)
    c.start(); check(20)
    c.on_coarse((0, 0, 5), [obs(1)]); check(20)
    fine(c, 1); check(60)
    c.on_picked()
    c.on_coarse((0, 0, 5), [obs(1)]); assert c.state == State.TOOL_CHANGE
    check(None)
    d = cycle(); d.start(); d.on_coarse((0, 0, 5), [])
    assert d.state == State.DONE
    for code in (1, 20, 60, 100, 999):
        with pytest.raises(ProtocolViolation):
            d.expect(code)
