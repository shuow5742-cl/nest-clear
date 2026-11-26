"""组装等待请求、模式检查与分发、统一应答出口。"""

from collections.abc import Mapping
from typing import Any

from autoweaver.motion_policy import format_tree
from autoweaver.motion_policy.batch import Batch
from autoweaver.motion_policy.nodes.control.sequence import Sequence
from autoweaver.motion_policy.nodes.decorator.repeat_until import RepeatUntil
from autoweaver.motion_policy.nodes.node import TreeNode

from nest_clear.config import Config
from nest_clear.record.events import EventRecorder
from .context import FlowContext, RequestTracker
from .dispatch import Branch, build_dispatch
from .leaves import CallWorker, CommitReply, WaitRequest
from .modes import Mode


def build_tree(ctx: FlowContext, branches: Mapping[int, Branch] | None = None) -> TreeNode:
    """装配每次请求的三个阶段，所有分支共用应答出口。"""
    class ReplyWorker(CallWorker):
        def on_running(self):
            status = super().on_running()
            from autoweaver.motion_policy.nodes.node import Status
            if status == Status.FAILURE and self._blackboard.read("reply_func") == 131:
                ctx.events.emit("error", task_id=self._blackboard.read("task_id"), operation="write_131", reason=f"对刀失败应答 131 写入失败：{self._blackboard.read('error')}")
            if status == Status.FAILURE:
                ctx.events.emit("error", task_id=self._blackboard.read("task_id"), operation="reply", reason=f"应答写入失败，将持续重试：{self._blackboard.read('error')}")
                return self.on_start()
            return status
    def reply_payload(bb):
        coordinates = dict(bb.read("reply_coordinates"))
        tool_id = bb.read("plc_tool_id")
        if tool_id is not None:
            coordinates["pc_tool_id"] = tool_id
        return {
            "operation": "reply",
            "task_id": bb.read("task_id"),
            "func": bb.read("reply_func"),
            "coordinates": coordinates,
            "diagnostic": bb.read("reply_diagnostic", None),
        }

    reply = ReplyWorker(
        ctx.board,
        "plc",
        reply_payload,
        result_key="committed_func",
        name="写入应答",
    )
    iteration = Sequence(
        [
            WaitRequest(ctx.board, ctx.tracker, ctx.events),
            build_dispatch(ctx, branches),
            Sequence([reply, CommitReply(ctx.events, ctx)], name="统一应答出口"),
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
    calibration: Any = None,
    camcal_factory: Any = None,
    camcal_save: Any = None,
    calibration_updated: Any = None,
    calibration_path: Any = None,
) -> Batch:
    """创建全新树及私有 Blackboard，初始模式为空闲。"""
    ctx = FlowContext(board, config, path, tracker, events, calibration, camcal_factory, camcal_save, calibration_updated, calibration_path)
    params = {
        "work_cycle": None,
        "tooloffset_session": None,
        "pickaccuracy_session": None,
        "mode": Mode.IDLE,
        "next_index": 0,
        "scan": None,
        "targets": [],
        "attempted": [],
        "pending": None,
        "view_picks": 0,
        "tool_index": 0,
        "last_action": 0,
        "task_id": 0,
        "request_func": 0,
        "actual": None,
        "done": False,
        "error": "",
        "error_type": None,
        "raw_path": None,
    }
    return Batch(lambda: build_tree(ctx, branches), params=params, name="一盘")
