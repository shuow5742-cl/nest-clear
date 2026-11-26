"""功能码分发表、统一模式检查和业务失败兜底。"""

from collections.abc import Callable, Mapping
import logging

from autoweaver.motion_policy.nodes.control.fallback import Fallback
from autoweaver.motion_policy.nodes.control.sequence import Sequence
from autoweaver.motion_policy.nodes.node import TreeNode

from .camera_calibration import camera_branch, finish
from .tool_offset import ToolOffset, finish as finish_tooloffset
from .pick_accuracy import PickAccuracy, finish as finish_pickaccuracy
from .calibration import finish_branch, change_tool_branch
from .context import FlowContext
from .leaves import Instant, Predicate, PrepareError
from .modes import Mode
from .picking import next_branch, scan_branch
from .work_cycle import ProtocolViolation
from nest_clear.guard import GuardError, diagnostic
from nest_clear.preview.control import REQUEST_CAMERAS, close_previews_for

logger = logging.getLogger("nest_clear.flow.dispatch")
BranchFactory = Callable[[FlowContext], TreeNode]
Branch = tuple[frozenset[Mode], BranchFactory]
BRANCHES: dict[int, Branch] = {
    1: (frozenset((Mode.IDLE, Mode.PICKING)), scan_branch),
    20: (frozenset((Mode.IDLE, Mode.PICKING)), scan_branch),
    60: (frozenset((Mode.IDLE, Mode.PICKING)), next_branch),
    80: (frozenset((Mode.IDLE, Mode.CAMERA_CALIBRATING)), camera_branch),
    90: (frozenset((Mode.IDLE, Mode.TOOL_OFFSET)), ToolOffset),
    100: (frozenset((Mode.IDLE, Mode.CALIBRATING_1, Mode.TOOL_OFFSET)), change_tool_branch),
    110: (frozenset((Mode.IDLE, Mode.CALIBRATING_2, Mode.TOOL_OFFSET)), change_tool_branch),
    130: (frozenset((Mode.CALIBRATING_1, Mode.CALIBRATING_2, Mode.TOOL_OFFSET)), finish_branch),
    150: (frozenset((Mode.IDLE, Mode.PICK_ACCURACY)), PickAccuracy),
    155: (frozenset((Mode.IDLE, Mode.PICK_ACCURACY)), PickAccuracy),
    156: (frozenset((Mode.IDLE, Mode.PICK_ACCURACY)), PickAccuracy),
}


class CloseRequestPreviews(Instant):
    def __init__(self, ctx):
        super().__init__(name='释放请求使用的预览相机')
        self.events = ctx.events

    def perform(self):
        bb = self._blackboard
        mode = bb.read('mode')
        func = bb.read('request_func')
        close_previews_for(self.events, REQUEST_CAMERAS.get(func, ()), 'plc_request',
                           bb.read('task_id'), func, mode=mode.value)


class RejectMode(Instant):
    """记录模式拒绝原因，由统一错误节点准备 999。"""

    def __init__(self, ctx: FlowContext) -> None:
        """绑定事件记录器。"""
        super().__init__(name="记录模式拒绝")
        self.events = ctx.events
        self.ctx = ctx

    def perform(self) -> None:
        """记录模式、盘号、扫描位和请求，保留挑毛状态。"""
        bb = self._blackboard
        mode = bb.read("mode")
        func = bb.read("request_func")
        cycle = bb.read("work_cycle", None)
        if mode == Mode.TOOL_OFFSET:
            finish_tooloffset(self.ctx, bb, "aborted", reason=f"标定中收到非工具偏移功能码 {func}")
        if mode == Mode.PICK_ACCURACY:
            finish_pickaccuracy(self.ctx, bb, "aborted", reason=f"标定中收到非 150/155/156 功能码 {func}")
        if mode == Mode.CAMERA_CALIBRATING:
            finish(self.ctx, bb, "aborted", reason=f"标定中收到非 80 功能码 {func}")
        if mode == Mode.BASELINE:
            reason = "对刀基准建立中，请先在界面结束"
        elif mode == Mode.PICKING and cycle is not None:
            try:
                cycle.expect(func)
            except ProtocolViolation as exc:
                reason = str(exc)
                bb.write("error_type", type(exc).__name__, self.name)
                bb.write("error_diagnostic", diagnostic(exc), self.name)
        elif mode == Mode.PICKING and func in (100, 110):
            reason = "挑毛进行中收到对刀指令，须先停止挑毛并由 PLC 重新初始化"
        else:
            reason = f"当前模式 {mode.value} 不允许功能码 {func}"
        bb.write("error", reason, self.name)
        if not bb.read("error_diagnostic", None):
            bb.write("error_diagnostic", diagnostic(reason, code="WORK-PROTOCOL", where="请求分发", hint="检查 PLC 功能码和当前运行模式。"), self.name)
        logger.warning("模式拒绝：当前模式=%s，盘号=%s，扫描位=%s，功能码=%s，原因=%s", mode.value, self.events.tray_id, bb.read("next_index"), func, reason)
        self.events.emit("mode_rejected", mode=mode.value, scan_index=bb.read("next_index"), request_func=func, task_id=bb.read("task_id"), reason=reason)



class CheckToolID(Instant):
    def __init__(self, ctx: FlowContext) -> None:
        super().__init__(name="校验 PLC 工具 ID")
        self.events = ctx.events

    def perform(self) -> None:
        bb = self._blackboard
        func, tool_id, mode = (bb.read(key) for key in ("request_func", "plc_tool_id", "mode"))
        if func in (150, 155, 156):
            session = bb.read("pickaccuracy_session", None)
            expected = session["tool"] if session else None
            page = "pickacc_state"
            tool = expected if expected is not None else tool_id
        else:
            session = bb.read("tooltip_session", None)
            expected = {Mode.TOOL_OFFSET: 1, Mode.CALIBRATING_1: 2, Mode.CALIBRATING_2: 2}.get(mode)
            page = "tooloffset_state" if func == 90 or tool_id == 1 or mode == Mode.TOOL_OFFSET else "tooltip_state"
            tool = session["tool"] if session else {100: 1, 110: 2}.get(func)
        if tool_id not in (1, 2):
            reason = f"40905={tool_id} 非法，必须为 1 或 2"
        elif expected is not None and tool_id != expected:
            reason = f"流程中 40905 从 {expected} 变为 {tool_id}"
        elif func == 90 and tool_id != 1:
            reason = f"工具偏移标定要求 40905=1，实际为 {tool_id}"
        elif func in (100, 110) and tool_id == 1 and not bb.read("tooloffset_session", None):
            reason = "未收到 90 却收到工具偏移标定请求"
        else:
            return
        fields = {"phase": "protocol_error" if page == "pickacc_state" and session else
                           "fail" if page == "tooltip_state" else "failed",
                  "message": reason, "plc_tool_id": tool_id}
        if page != "tooloffset_state":
            fields["tool"] = tool if tool in (1, 2) else None
        self.events.emit(page, **fields)
        raise GuardError("PLC-TOOL-ID", reason, "PLC 地址 40905", "检查工具 ID 及当前标定会话。")


def build_dispatch(ctx: FlowContext, branches: Mapping[int, Branch] | None = None) -> TreeNode:
    """每项先匹配功能码，再统一检查模式并执行对应工厂。"""
    entries = BRANCHES if branches is None else branches
    nodes = []
    for code, (allowed, factory) in entries.items():
        match = Predicate(f"功能码 == {code}", lambda bb, code=code: bb.read("request_func") == code)
        steps = [CloseRequestPreviews(ctx), factory(ctx)]
        if code in (100, 110, 130) and factory in (change_tool_branch, finish_branch):
            steps = [Fallback([
                Sequence([Predicate("工具偏移标定", lambda bb: bb.read("plc_tool_id") == 1),
                          CloseRequestPreviews(ctx), ToolOffset(ctx)], name="执行工具偏移标定"),
                Sequence(steps, name="执行换针对刀"),
            ], name="按工具 ID 分流")]
        business = Fallback([Sequence(steps, name="执行请求"), PrepareError(ctx.events)], name=f"功能码 {code} 业务或 999")
        checked = Fallback(
            [
                Sequence([Predicate("模式允许", lambda bb, allowed=allowed: bb.read("mode") != Mode.BASELINE and bb.read("mode") in allowed), business], name="模式允许后执行"),
                Sequence([RejectMode(ctx), PrepareError(ctx.events)], name="模式拒绝并应答 999"),
            ],
            name="模式检查",
        )
        if code in (90, 100, 110, 130, 150, 155, 156):
            checked = Fallback([
                Sequence([CheckToolID(ctx), checked], name="工具 ID 有效后检查模式"),
                PrepareError(ctx.events),
            ], name="工具 ID 检查")
        nodes.append(Sequence([match, checked], name=f"功能码 {code}"))
    nodes.append(Sequence([RejectMode(ctx), PrepareError(ctx.events, unknown=True)], name="拒绝未知功能码"))
    return Fallback(nodes, name="模式检查+功能码分发")
