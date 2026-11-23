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
    assert (c.on_picked(), c.on_next())[1].scan_xyz == (4., 9., 5.)
    for i, failure in enumerate(("fine_empty", "fine_point_outside", "fine_rejected")):
        r = c.on_fine((0, 0, 5), FineResult(failure))
        assert r.scan_xyz == (0., 0., 5.)
        assert c.is_attempted((1. if i == 0 else 10. * i, 0, 0))
        previous = obs(1)
        r = c.on_coarse((0, 0, 5), [previous, obs(10 * (i + 1))] if i < 2 else [previous])
        assert previous.status == "fine_failed" and r.removable_count == (1 if i < 2 else 0)
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
        c.on_picked(); c.on_next()
    assert c.final_check and c.state == State.WAIT_COARSE
    r = c.on_coarse((0, 0, 5), [obs(9999)])
    assert r.func_code == 9 and r.removable_count == 1 and r.event["reason"] == "cap"


@pytest.mark.parametrize("min_attempts, red, expected", [(1, True, 10), (1, False, 9), (2, True, 9)])
def test_tool_check(min_attempts, red, expected):
    c = cycle(tool_check_min_attempts=min_attempts, cap_ratio=1)
    c.start()
    c.on_coarse((0, 0, 5), [obs(1)])
    fine(c, 1)
    c.on_picked(); c.on_next()
    r = c.on_coarse((0, 0, 5), [obs(1)] if red else [])
    assert r.func_code == expected
    assert r.event["tool_check_total"] == int(red)


@pytest.mark.parametrize("status, reason, expected_status, red, code", [
    ("other_view", None, "other_view", 0, 9),
    ("rejected", "small", "attempted", 1, 10),
])
def test_attempted_observation_respects_responsibility(status, reason, expected_status, red, code):
    c = cycle(tool_check_min_attempts=1, cap_ratio=1)
    c.start()
    c.on_coarse((0, 0, 5), [obs(1)])
    fine(c, 1)
    c.on_picked(); c.on_next()
    observation = obs(1, status=status, reason=reason)
    reply = c.on_coarse((0, 0, 5), [observation])
    assert observation.status == expected_status
    assert c.red_count == reply.event["red_count"] == red
    assert reply.removable_count == 0
    assert reply.event["tool_check_total"] == red
    assert reply.func_code == code


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
    assert c.on_picked().func_code == 60
    assert c.state == State.WAIT_NEXT
    check(1)
    c.on_next()
    c.on_coarse((0, 0, 5), [obs(1)]); assert c.state == State.TOOL_CHANGE
    check(None)
    d = cycle(); d.start(); d.on_coarse((0, 0, 5), [])
    assert d.state == State.DONE
    for code in (1, 20, 60, 100, 999):
        with pytest.raises(ProtocolViolation):
            d.expect(code)


def test_attempt_matching_uses_identity_not_pick():
    c = cycle(tool_check_min_attempts=99)
    c.start()
    original = obs(10)
    original.identity = (1., 0., 0.)
    c.on_coarse((0, 0, 5), [original])
    c.on_fine((0, 0, 5), FineResult('fine_empty'))
    repeat = obs(20)
    repeat.identity = (1.02, 0., 0.)
    c.on_coarse((0, 0, 5), [repeat])
    assert repeat.status == 'fine_failed'
    assert not c.is_attempted((10., 0., 0.))


def test_attempt_matches_either_coarse_or_fine_identity():
    c = cycle(skip_radius_mm=0.1)
    c.start()
    c.on_coarse((0, 0, 5), [obs(1)])
    c.on_fine((0, 0, 5), FineResult("ok", target_xyz=(5., 0., 0.), identity=(9., 0., 0.)))
    assert c.is_attempted((1.05, 0, 0))
    assert c.is_attempted((9.05, 0, 0))
    assert not c.is_attempted((5., 0, 0))


def test_pick_region_excludes_counts_and_preserves_other_priority():
    c = cycle(pick_region_mm=((0, 2), (-1, 1)), tool_check_min_attempts=1, cap_ratio=1)
    c.start()
    inside, outside, other = obs(2), obs(3, 100), obs(4, 100, 'other_view')
    reply = c.on_coarse((0, 0, 5), [inside, outside, other])
    assert (reply.removable_count, c.n0, c.targets) == (1, 1, (inside,))
    assert (outside.status, other.status) == ('out_of_range', 'other_view')
    reply = fine(c, 3)
    assert reply.func_code == 1 and reply.tool1_xyz is None
    assert reply.event['reason'] == 'fine_out_of_range'
    reply = c.on_coarse((0, 0, 5), [obs(3), obs(4)])
    assert reply.func_code == 9 and reply.event['tool_check_total'] == 0


def test_picked_ack_then_next_target_and_exhausted():
    c = cycle(cap_ratio=3, tool_check_min_attempts=99)
    c.start()
    c.on_coarse((0, 0, 5), [obs(1), obs(2)])
    fine(c, 1)
    ack = c.on_picked()
    assert (ack.func_code, ack.scan_xyz, ack.tool1_xyz, ack.tool2_xyz, ack.removable_count) == (60, None, None, None, None)
    assert c.expect(1) == "next"
    for code in (20, 60):
        with pytest.raises(ProtocolViolation) as error:
            c.expect(code)
        assert (error.value.state, error.value.expected, error.value.actual) == (State.WAIT_NEXT, 1, code)
    assert c.on_next().scan_xyz == (2., 7., 5.)
    fine(c, 2)
    assert c.on_picked().func_code == 60
    reply = c.on_next()
    assert reply.scan_xyz == (0., 0., 5.) and reply.event["reason"] == "list_exhausted"
    assert c.state == State.WAIT_COARSE


def test_daily_report_counts_pick_ack():
    from nest_clear.record.report import _work_stats
    assert _work_stats([{"event": "reply", "func": 60}])["replies"][60] == 1


def test_reachability_filters_coarse_and_fine():
    from nest_clear.vision.fine import FineResult
    c = cycle()
    c._reachable = lambda xyz: xyz[0] < 5
    c.start()
    bad = obs(10)
    good = obs(1)
    c.on_coarse((0, 0, 0), [bad, good])
    assert bad.status == 'out_of_range'
    reply = c.on_fine((0, 0, 0), FineResult('ok', target_xyz=(10, 0, 0)))
    assert reply.event['reason'] == 'fine_out_of_range' and reply.tool1_xyz is None
    review = obs(1)
    c.on_coarse((0, 0, 0), [review])
    assert review.status == 'out_of_range'
