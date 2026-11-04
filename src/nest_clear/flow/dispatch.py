"""功能码分发表、统一模式检查和业务失败兜底。"""

from collections.abc import Callable, Mapping
import logging

from autoweaver.motion_policy.nodes.control.fallback import Fallback
from autoweaver.motion_policy.nodes.control.sequence import Sequence
from autoweaver.motion_policy.nodes.node import TreeNode

from .calibration import calibration_placeholder, start_branch, continue_branch
from .context import FlowContext
from .leaves import Instant, Predicate, PrepareError
from .modes import Mode
from .picking import next_branch, scan_branch

logger = logging.getLogger("nest_clear.flow.dispatch")
BranchFactory = Callable[[FlowContext], TreeNode]
Branch = tuple[frozenset[Mode], BranchFactory]
BRANCHES: dict[int, Branch] = {
    1: (frozenset((Mode.IDLE, Mode.PICKING)), scan_branch),
    60: (frozenset((Mode.PICKING,)), next_branch),
    100: (frozenset((Mode.IDLE,)), start_branch),
    101: (frozenset((Mode.IDLE,)), start_branch),
    110: (frozenset((Mode.CALIBRATING_1,)), continue_branch),
    120: (frozenset((Mode.CALIBRATING_2,)), continue_branch),
    130: (frozenset((Mode.CALIBRATING_1, Mode.CALIBRATING_2)), calibration_placeholder),
}


class RejectMode(Instant):
    """记录模式拒绝原因，由统一错误节点准备 999。"""

    def __init__(self, ctx: FlowContext) -> None:
        """绑定事件记录器。"""
        super().__init__(name="记录模式拒绝")
        self.events = ctx.events

    def perform(self) -> None:
        """记录模式、盘号、扫描位和请求，保留挑毛状态。"""
        bb = self._blackboard
        mode = bb.read("mode")
        func = bb.read("request_func")
        if mode == Mode.PICKING and func in (100, 101):
            reason = "挑毛进行中收到对刀指令，须先停止挑毛并由 PLC 重新初始化"
        else:
            reason = f"当前模式 {mode.value} 不允许功能码 {func}"
        bb.write("error", reason, self.name)
        logger.warning("模式拒绝：当前模式=%s，盘号=%s，扫描位=%s，功能码=%s，原因=%s", mode.value, self.events.tray_id, bb.read("next_index"), func, reason)
        self.events.emit("mode_rejected", mode=mode.value, scan_index=bb.read("next_index"), request_func=func, task_id=bb.read("task_id"), reason=reason)


def build_dispatch(ctx: FlowContext, branches: Mapping[int, Branch] | None = None) -> TreeNode:
    """每项先匹配功能码，再统一检查模式并执行对应工厂。"""
    entries = BRANCHES if branches is None else branches
    nodes = []
    for code, (allowed, factory) in entries.items():
        match = Predicate(f"功能码 == {code}", lambda bb, code=code: bb.read("request_func") == code)
        business = Fallback([factory(ctx), PrepareError(ctx.events)], name=f"功能码 {code} 业务或 999")
        checked = Fallback(
            [
                Sequence([Predicate("模式允许", lambda bb, allowed=allowed: bb.read("mode") in allowed), business], name="模式允许后执行"),
                Sequence([RejectMode(ctx), PrepareError(ctx.events)], name="模式拒绝并应答 999"),
            ],
            name="模式检查",
        )
        nodes.append(Sequence([match, checked], name=f"功能码 {code}"))
    nodes.append(PrepareError(ctx.events, unknown=True))
    return Fallback(nodes, name="模式检查+功能码分发")
