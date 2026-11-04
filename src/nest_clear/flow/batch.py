"""组装等待请求、模式检查与分发、统一应答出口。"""

from collections.abc import Mapping
from typing import Any

from autoweaver.motion_policy import format_tree
from autoweaver.motion_policy.batch import Batch
from autoweaver.motion_policy.nodes.control.sequence import Sequence
from autoweaver.motion_policy.nodes.decorator.repeat_until import RepeatUntil
from autoweaver.motion_policy.nodes.decorator.retry import Retry
from autoweaver.motion_policy.nodes.node import TreeNode

from nest_clear.config import Config
from nest_clear.record.events import EventRecorder
from .context import FlowContext, RequestTracker
from .dispatch import Branch, build_dispatch
from .leaves import CallWorker, CommitReply, WaitRequest
from .modes import Mode


def build_tree(ctx: FlowContext, branches: Mapping[int, Branch] | None = None) -> TreeNode:
    """装配每次请求的三个阶段，所有分支共用应答出口。"""
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
            build_dispatch(ctx, branches),
            Sequence([Retry(max_attempts=3, child=reply, name="应答最多三次"), CommitReply(ctx.events)], name="统一应答出口"),
        ],
        name="处理一次请求",
    )
    return RepeatUntil(cond=lambda snapshot, bb: bb.read("done", False), child=iteration, name="一盘直到结束")


def build_batch(
    board: Any,
    config: Config,
    path: list[tuple[float, float, float]],
    tracker: RequestTracker,
    events: EventRecorder,
    branches: Mapping[int, Branch] | None = None,
) -> Batch:
    """创建全新树及私有 Blackboard，初始模式为空闲。"""
    ctx = FlowContext(board, config, path, tracker, events)
    params = {
        "mode": Mode.IDLE,
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
