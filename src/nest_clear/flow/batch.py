"""用 AutoWeaver 组合节点构造每盘行为树和功能码分发表。"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from autoweaver.motion_policy.batch import Batch
from autoweaver.motion_policy.nodes.control.fallback import Fallback
from autoweaver.motion_policy.nodes.control.sequence import Sequence
from autoweaver.motion_policy.nodes.decorator.repeat_until import RepeatUntil
from autoweaver.motion_policy.nodes.decorator.retry import Retry
from autoweaver.motion_policy.nodes.node import TreeNode

from nest_clear.config import Config
from nest_clear.record.events import EventRecorder
from .leaves import (
    CallWorker,
    CheckArrival,
    CommitReply,
    Predicate,
    PrepareError,
    PreparePick,
    PrepareScan,
    RecordDetection,
    RecordCapture,
    WaitRequest,
)


class RequestTracker:
    """仅跨盘保留 PLC 观察基线，防止重复接收上一盘请求。"""

    def __init__(self, baseline: int) -> None:
        """启动时忽略当前编号；盘内业务状态放在 Blackboard。"""
        self.observed = baseline
        self.processed = None


@dataclass(frozen=True)
class FlowContext:
    """构树依赖；不保存当前盘的可变业务状态。"""

    board: Any
    config: Config
    path: list[tuple[float, float, float]]
    tracker: RequestTracker
    events: EventRecorder


def choose_next(ctx: FlowContext) -> TreeNode:
    """目标存在时挑取，否则下发下一扫描位或结束。"""
    return Fallback(
        [
            Sequence(
                [
                    Predicate("有待挑目标", lambda bb: bool(bb.read("targets"))),
                    PreparePick(ctx.config),
                ],
                name="挑取目标",
            ),
            Sequence(
                [
                    Predicate("没有待挑目标", lambda bb: not bb.read("targets")),
                    PrepareScan(ctx.path),
                ],
                name="继续扫描",
            ),
        ],
        name="选择挑取或扫描",
    )


def scan_branch(ctx: FlowContext) -> TreeNode:
    """功能码 1：首次直接扫描，后续到位后拍照和检测。"""
    first = Sequence(
        [
            Predicate("尚未下发扫描位", lambda bb: bb.read("scan") is None),
            PrepareScan(ctx.path),
        ],
        name="首次扫描",
    )
    arrived = Sequence(
        [
            Predicate("等待扫描到位", lambda bb: bb.read("last_action") == 1),
            CallWorker(ctx.board, "plc", lambda bb: {"operation": "actual"}, "actual", "读取实时坐标"),
            CheckArrival(ctx.config, ctx.events),
            Predicate("到位合格", lambda bb: bb.read("arrival_ok")),
            CallWorker(ctx.board, "camera", lambda bb: {}, "image", "拍照"),
            RecordCapture(ctx.events),
            CallWorker(
                ctx.board,
                "vision",
                lambda bb: {"image": bb.read("image"), "pose": bb.read("actual")},
                "targets",
                "检测并换算目标",
            ),
            RecordDetection(ctx.events),
            choose_next(ctx),
        ],
        name="到位拍照检测",
    )
    return Fallback([first, arrived], name="首次或到位")


def next_branch(ctx: FlowContext) -> TreeNode:
    """功能码 60：挑取结束后使用剩余目标，不重新拍照。"""
    return Sequence(
        [
            Predicate("等待挑取完成", lambda bb: bb.read("last_action") == 3),
            choose_next(ctx),
        ],
        name="挑取完成后继续",
    )


BranchFactory = Callable[[FlowContext], TreeNode]
BRANCHES: dict[int, BranchFactory] = {1: scan_branch, 60: next_branch}


def build_tree(ctx: FlowContext, branches: Mapping[int, BranchFactory] | None = None) -> TreeNode:
    """由分发表构造功能码分支；公共应答不属于任何业务分支。"""
    factories = BRANCHES if branches is None else branches
    dispatch = []
    for code, factory in factories.items():
        match = Predicate(f"功能码 == {code}", lambda bb, code=code: bb.read("request_func") == code)
        business = Fallback(
            [
                factory(ctx),
                PrepareError(ctx.events),
            ],
            name=f"功能码 {code} 业务或 999",
        )
        dispatch.append(Sequence([match, business], name=f"功能码 {code}"))
    dispatch.append(PrepareError(ctx.events, unknown=True))
    dispatch_tree = Fallback(dispatch, name="功能码分发")
    reply = CallWorker(
        ctx.board,
        "plc",
        lambda bb: {
            "operation": "reply",
            "task_id": bb.read("task_id"),
            "func": bb.read("reply_func"),
            "coordinates": bb.read("reply_coordinates"),
        },
        name="写入应答",
    )
    iteration = Sequence(
        [
            WaitRequest(ctx.board, ctx.tracker, ctx.config.plc.comm_fail_timeout_s, ctx.events),
            dispatch_tree,
            Retry(max_attempts=3, child=reply, name="应答最多三次"),
            CommitReply(ctx.events),
        ],
        name="处理一次请求",
    )
    return RepeatUntil(
        cond=lambda snapshot, bb: bb.read("done", False),
        child=iteration,
        name="一盘直到结束",
    )


def format_tree(tree: TreeNode) -> str:
    """递归输出组合和装饰节点，供启动日志与现场检查。"""
    lines: list[str] = []

    def visit(node: TreeNode, depth: int) -> None:
        """按树深度添加名称并遍历实际子节点。"""
        lines.append(f"{'  ' * depth}{type(node).__name__}: {node.name}")
        for child in getattr(node, "children", []):
            visit(child, depth + 1)
        child = getattr(node, "child", None)
        if child is not None:
            visit(child, depth + 1)

    visit(tree, 0)
    return "\n".join(lines)


def build_batch(
    board: Any,
    config: Config,
    path: list[tuple[float, float, float]],
    tracker: RequestTracker,
    events: EventRecorder,
    branches: Mapping[int, BranchFactory] | None = None,
) -> Batch:
    """创建全新树及私有 Blackboard；业务状态均在参数中初始化。"""
    ctx = FlowContext(board, config, path, tracker, events)
    params = {
        "next_index": 0,
        "scan": None,
        "targets": [],
        "tool_index": 0,
        "last_action": 0,
        "task_id": 0,
        "request_func": 0,
        "actual": None,
        "done": False,
        "error": "",
        "raw_path": None,
    }
    return Batch(lambda: build_tree(ctx, branches), params=params, name="一盘")
