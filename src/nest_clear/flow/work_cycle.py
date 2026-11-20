"""一盘工作模式的纯逻辑状态机。"""
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Callable

from nest_clear.guard import GuardError

XYZ = tuple[float, float, float]


class State(str, Enum):
    IDLE = "IDLE"
    WAIT_COARSE = "WAIT_COARSE"
    WAIT_FINE = "WAIT_FINE"
    WAIT_PICK = "WAIT_PICK"
    WAIT_NEXT = "WAIT_NEXT"
    DONE = "DONE"
    TOOL_CHANGE = "TOOL_CHANGE"


class ProtocolViolation(GuardError):
    """请求功能码与当前状态不符。"""

    def __init__(self, state: State, expected: int | None, actual: Any, view_index: int) -> None:
        self.state, self.expected, self.actual, self.view_index = state, expected, actual, view_index
        super().__init__("WORK-PROTOCOL", f"当前状态 {state.value}，期望功能码 {expected if expected is not None else '无'}，实际功能码 {actual}",
                         f"视野序号 {view_index}", "请检查 PLC 工作流程。")


@dataclass(frozen=True)
class Reply:
    func_code: int
    scan_xyz: XYZ | None = None
    tool1_xyz: XYZ | None = None
    tool2_xyz: XYZ | None = None
    removable_count: int | None = None
    event: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Attempt:
    coarse_xyz: XYZ
    fine_xyz: XYZ | None
    reason: str
    target_id: str | None = None


class WorkCycle:
    """保存一盘的状态；视野序号从零开始，未启动时为 -1。

    observations 使用 Observation：status 为 eligible/rejected/other_view，
    target.xyz 为挑取点，area 为物理面积。调用方负责事先计算负责区。
    params 使用 VisionConfig 或具有同名属性的对象，匹配半径为 skip_radius_mm。
    """

    def __init__(self, views: list[XYZ], fine_coord: Callable[[XYZ, XYZ], XYZ],
                 pick_coords: Callable[[XYZ], tuple[XYZ, XYZ]], params: Any,
                 record: Callable[..., None] | None = None, tray: str = "0000") -> None:
        if not views:
            raise ValueError("视野列表不能为空")
        self._views = tuple(tuple(v) for v in views)
        self._fine_coord, self._pick_coords, self._params = fine_coord, pick_coords, params
        self._state = State.IDLE
        self._view_index = -1
        self._attempted: list[Attempt] = []
        self.record = record or (lambda *args, **kwargs: None)
        self.tray = tray
        self.identities: dict[int, dict[str, XYZ]] = {}
        self.attempts_ledger: list[dict[str, Any]] = []
        self.active_attempt: dict[str, Any] | None = None
        self._reset_view()

    def _reset_view(self) -> None:
        self._n0 = None
        self._allowed = 0
        self._attempts = 0
        self._coarse_shots = 0
        self._red_count = 0
        self._final_check = False
        self._targets: list[Any] = []
        self._target_index = 0
        self._coarse_pose: XYZ | None = None

    @property
    def state(self) -> State:
        return self._state

    @property
    def view_index(self) -> int:
        return self._view_index

    @property
    def n0(self) -> int | None:
        return self._n0

    @property
    def allowed_attempts(self) -> int:
        return self._allowed

    @property
    def attempt_count(self) -> int:
        return self._attempts

    @property
    def coarse_shots(self) -> int:
        return self._coarse_shots

    @property
    def red_count(self) -> int:
        return self._red_count

    @property
    def final_check(self) -> bool:
        return self._final_check

    @property
    def attempted(self) -> tuple[Attempt, ...]:
        return tuple(self._attempted)

    @property
    def targets(self) -> tuple[Any, ...]:
        return tuple(self._targets)

    def expect(self, func_code: int) -> str:
        expected, action = {
            State.IDLE: (1, "start"), State.WAIT_COARSE: (20, "coarse"),
            State.WAIT_FINE: (20, "fine"), State.WAIT_PICK: (60, "picked"),
            State.WAIT_NEXT: (1, "next"),
        }.get(self._state, (None, ""))
        if expected is None or func_code != expected:
            raise ProtocolViolation(self._state, expected, func_code, self._view_index)
        return action

    def _require(self, state: State, code: int) -> None:
        if self._state != state:
            expected = {State.IDLE: 1, State.WAIT_COARSE: 20, State.WAIT_FINE: 20, State.WAIT_PICK: 60, State.WAIT_NEXT: 1}.get(self._state)
            raise ProtocolViolation(self._state, expected, code, self._view_index)

    def start(self) -> Reply:
        self._require(State.IDLE, 1)
        self._view_index = 0
        self._state = State.WAIT_COARSE
        return Reply(1, scan_xyz=self._views[0], event={"reason": "start", "view_index": 0})

    def is_attempted(self, xyz: XYZ) -> bool:
        return any(math.dist(xyz[:2], point[:2]) <= self._params.skip_radius_mm
                   for attempt in self._attempted
                   for point in (attempt.coarse_xyz, attempt.fine_xyz) if point is not None)

    def current_target(self) -> Any:
        if self._state not in (State.WAIT_FINE, State.WAIT_PICK):
            return None
        return self._targets[self._target_index]

    def _fine_reply(self, event: dict) -> Reply:
        xyz = self.current_target().target.xyz
        return Reply(1, scan_xyz=self._fine_coord(xyz, self._coarse_pose), event=event)

    def match_observations(self, observations: list[Any], capture_id: str | None = None, *, create: bool = True) -> None:
        """同视野按身份点最近距离一对一匹配。"""
        known = self.identities.setdefault(self._view_index, {})
        available = set(known)
        for obs in observations:
            if obs.status == "other_view":
                continue
            identity = tuple(getattr(obs, "identity", obs.target.xyz))
            matches = sorted((math.dist(identity[:2], known[key][:2]), key) for key in available
                             if math.dist(identity[:2], known[key][:2]) <= self._params.skip_radius_mm)
            if matches:
                distance, target_id = matches[0]
                obs.match_distance_mm = distance
                if len(matches) > 1:
                    self.record("match_ambiguous", capture_id=capture_id, target_id=target_id,
                                matches=[key for _, key in matches], threshold_mm=self._params.skip_radius_mm)
                self.record("target_match", capture_id=capture_id, target_id=target_id,
                            distance_mm=distance, threshold_mm=self._params.skip_radius_mm)
                available.remove(target_id)
            else:
                if not create:
                    continue
                target_id = f"{self.tray}-{self._view_index + 1}-{len(known) + 1:03d}"
            known[target_id] = identity
            obs.target_id = target_id

    def begin_fine(self, capture_id: str | None, task_id: int | None = None,
                   request_func: int | None = None, xid: int | None = None) -> dict[str, Any]:
        target = self.current_target()
        target_id = getattr(target, "target_id", None)
        if target_id is None:
            self.match_observations([target])
            target_id = target.target_id
        count = sum(a["target_id"] == target_id for a in self.attempts_ledger) + 1
        attempt = {"attempt_id": f"{target_id}#a{count}", "target_id": target_id,
                   "view_index": self._view_index, "capture_id": capture_id,
                   "coarse_capture_id": getattr(target, "capture_id", None),
                   "identity": tuple(getattr(target, "identity", target.target.xyz)),
                   "task_id": task_id, "request_func": request_func, "xid": xid,
                   "sent": False, "done": False, "outcome": None}
        self.attempts_ledger.append(attempt)
        self.active_attempt = attempt
        self.record("fine_capture", **attempt)
        return attempt

    def mark_reply(self, task_id: int, xid: int, coordinates: dict[str, Any]) -> None:
        attempt = self.active_attempt
        if attempt and attempt["outcome"] is None:
            attempt.update(sent=True, reply_task_id=task_id, reply_xid=xid, coordinates=coordinates)
            self.record("reply_sent", attempt_id=attempt["attempt_id"], target_id=attempt["target_id"],
                        capture_id=attempt["capture_id"], task_id=task_id, xid=xid, coordinates=coordinates)
            # 先记可替换的待判状态；异常断电时仍可从日志知道这次下发尚无复拍结论。
            self.record("pick_result", target_id=attempt["target_id"], attempt_id=attempt["attempt_id"],
                        capture_id=attempt["capture_id"], coarse_capture_id=attempt["coarse_capture_id"],
                        task_id=task_id, request_func=attempt["request_func"], xid=xid,
                        outcome="pending", reason="等待复拍")
            attempt["pending_recorded"] = True

    def mark_done(self, task_id: int, xid: int) -> None:
        attempt = self.active_attempt
        if attempt and attempt["sent"] and attempt["outcome"] is None:
            attempt["done"] = True
            self.record("plc_done", attempt_id=attempt["attempt_id"], target_id=attempt["target_id"],
                        capture_id=attempt["capture_id"], task_id=task_id, request_func=60, xid=xid)

    def _result(self, attempt: dict[str, Any], outcome: str, **fields: Any) -> None:
        if attempt["outcome"] is not None:
            return
        attempt["outcome"] = outcome
        self.record("pick_result", target_id=attempt["target_id"], attempt_id=attempt["attempt_id"],
                    capture_id=attempt["capture_id"], coarse_capture_id=attempt["coarse_capture_id"],
                    task_id=attempt["task_id"], request_func=attempt["request_func"], xid=attempt["xid"], outcome=outcome, **fields)

    def pending(self) -> None:
        for attempt in self.attempts_ledger:
            if attempt["sent"] and attempt["outcome"] is None:
                if attempt.get("pending_recorded"):
                    attempt["outcome"] = "pending"
                else:
                    self._result(attempt, "pending", reason="离开视野或进程结束前未复拍")

    def fine_not_sent(self, reason: str) -> None:
        attempt = self.active_attempt
        if attempt and not attempt["sent"] and attempt["outcome"] is None:
            self._result(attempt, "not_sent", reason=reason)

    def in_pick_region(self, xyz: XYZ) -> bool:
        region = getattr(self._params, "pick_region_mm", None)
        return region is None or all(low <= value <= high for value, (low, high) in zip(xyz[:2], region))

    def on_coarse(self, pose: XYZ, observations: list[Any], capture_id: str | None = None) -> Reply:
        self._require(State.WAIT_COARSE, 20)
        self._coarse_shots += 1
        self._coarse_pose = pose
        for obs in observations:
            if obs.status != "other_view" and not self.in_pick_region(getattr(obs, "identity", obs.target.xyz)):
                obs.status = "out_of_range"
        self.match_observations(observations, capture_id)
        matched = {getattr(o, "target_id", None): o for o in observations if o.status != "other_view"}
        for attempt in self.attempts_ledger:
            if attempt["view_index"] == self._view_index and attempt["done"] and attempt["outcome"] is None:
                found = matched.get(attempt["target_id"])
                distance = getattr(found, "match_distance_mm", None) if found else None
                self._result(attempt, "failed" if found else "success", review_capture_id=capture_id,
                             match_distance_mm=distance)
        guard_red_count = 0
        for obs in observations:
            obs.capture_id = capture_id
            if obs.status in ("eligible", "rejected", "failed", "attempted") and self.is_attempted(getattr(obs, "identity", obs.target.xyz)):
                guard_red_count += 1
                attempts = [a for a in self.attempts_ledger if a["target_id"] == getattr(obs, "target_id", None)]
                outcome = attempts[-1]["outcome"] if attempts else None
                obs.status = "failed" if outcome == "failed" else "fine_failed" if outcome == "not_sent" else "attempted"
        self._red_count = guard_red_count + sum(o.status in ("failed", "attempted") for o in observations if not self.is_attempted(getattr(o, "identity", o.target.xyz)))
        eligible = sorted((o for o in observations if o.status == "eligible" and o.reason is None),
                          key=lambda o: o.area, reverse=True)
        count = len(eligible)
        if self._n0 is None:
            self._n0 = count
            self._allowed = math.ceil(self._params.cap_ratio * count)
        exceeded = self._coarse_shots > self._params.max_coarse_shots_per_view
        reason = "coarse_limit" if exceeded else "cap" if self._final_check else "empty" if count == 0 else None
        self._targets = eligible[:self._params.max_fine_targets] if reason is None else []
        self._target_index = 0
        event = {"type": "coarse", "view_index": self._view_index, "coarse_shots": self._coarse_shots,
                 "n0": self._n0, "allowed_attempts": self._allowed, "attempts": self._attempts,
                 "removable_count": count, "red_count": self._red_count,
                 "targets": [o.target.xyz for o in self._targets],
                 "red_xyz": [o.target.xyz for o in observations if o.status in ("failed", "attempted")],
                 "match_radius_mm": self._params.skip_radius_mm}
        if reason is not None:
            change = (self._attempts >= self._params.tool_check_min_attempts and
                      self._red_count + count > self._params.tool_check_ratio * self._n0)
            event.update(reason=reason, tool_change=change, tool_check_total=self._red_count + count,
                         tool_check_threshold=self._params.tool_check_ratio * self._n0,
                         tool_check_min_attempts=self._params.tool_check_min_attempts,
                         warning=exceeded)
            if change:
                self._state = State.TOOL_CHANGE
                return Reply(10, removable_count=count, event=event)
            if self._view_index + 1 == len(self._views):
                self._state = State.DONE
                return Reply(9, removable_count=count, event=event)
            self._view_index += 1
            self._reset_view()
            return Reply(1, scan_xyz=self._views[self._view_index], removable_count=count, event=event)
        self._state = State.WAIT_FINE
        reply = self._fine_reply(event)
        return Reply(reply.func_code, scan_xyz=reply.scan_xyz, removable_count=count, event=event)

    def on_fine(self, pose: XYZ, result: Any) -> Reply:
        self._require(State.WAIT_FINE, 20)
        if result.status not in ("ok", "fine_empty", "fine_point_outside", "fine_rejected", "fine_out_of_range"):
            raise ValueError(f"未知精定位结果：{result.status}")
        coarse = tuple(self.current_target().target.xyz)
        fine = tuple(result.target_xyz) if result.status == "ok" and result.target_xyz is not None else None
        if result.status == "ok" and fine is None:
            raise ValueError("精定位成功时缺少 target_xyz")
        if fine is not None and not self.in_pick_region(fine):
            from dataclasses import replace
            result = replace(result, status="fine_out_of_range")
            fine = None
        # 注入的计算可能抛异常；成功后才提交状态。
        tools = self._pick_coords(fine) if fine is not None else None
        attempt = self.active_attempt
        if attempt is None or attempt["outcome"] is not None or attempt.get("fine_result") is not None:
            attempt = self.begin_fine(None)
        attempt["fine_result"] = result.status
        self.record("fine_result", attempt_id=attempt["attempt_id"], target_id=attempt["target_id"],
                    capture_id=attempt["capture_id"], task_id=attempt["task_id"], request_func=attempt["request_func"],
                    xid=attempt["xid"], result=result.status)
        if fine is None:
            self._result(attempt, "not_sent", reason=result.status)
        identity = tuple(getattr(self.current_target(), "identity", coarse))
        fine_identity = tuple(result.identity) if result.identity is not None else None
        self._attempted.append(Attempt(identity, fine_identity, result.status, attempt["target_id"]))
        self._attempts += 1
        event = {"type": "fine", "view_index": self._view_index, "reason": result.status,
                 "coarse_xyz": coarse, "fine_xyz": fine, "pose": pose,
                 "offset_mm": math.dist(coarse[:2], fine[:2]) if fine is not None else None, "attempts": self._attempts}
        if tools is not None:
            self._state = State.WAIT_PICK
            event.update(tool1_xyz=tools[0], tool2_xyz=tools[1])
            return Reply(3, tool1_xyz=tools[0], tool2_xyz=tools[1], event=event)
        self._final_check = self._attempts >= self._allowed
        self._targets = []
        self._state = State.WAIT_COARSE
        return Reply(1, scan_xyz=self._views[self._view_index], event=event)

    def on_picked(self) -> Reply:
        self._require(State.WAIT_PICK, 60)
        self._state = State.WAIT_NEXT
        return Reply(60, event={"type": "picked", "reason": "acknowledged", "view_index": self._view_index})

    def on_next(self) -> Reply:
        self._require(State.WAIT_NEXT, 1)
        if self._attempts >= self._allowed:
            self._final_check = True
            reason = "cap"
        elif self._target_index + 1 < len(self._targets):
            self._target_index += 1
            self._state = State.WAIT_FINE
            return self._fine_reply({"type": "picked", "reason": "next_target", "view_index": self._view_index})
        else:
            reason = "list_exhausted"
        self._state = State.WAIT_COARSE
        return Reply(1, scan_xyz=self._views[self._view_index],
                     event={"type": "picked", "reason": reason, "view_index": self._view_index})
