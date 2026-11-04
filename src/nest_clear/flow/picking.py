"""挑毛请求的行为子树。"""

from autoweaver.motion_policy.nodes.control.fallback import Fallback
from autoweaver.motion_policy.nodes.control.sequence import Sequence
from autoweaver.motion_policy.nodes.node import TreeNode

from .context import FlowContext
from .leaves import CallWorker, CheckArrival, Predicate, PreparePick, PrepareScan, RecordCapture, RecordDetection


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
            CallWorker(ctx.board, "plc", lambda bb: {"operation": "actual"}, "actual", name="读取实时坐标"),
            CheckArrival(ctx.config, ctx.events),
            Predicate("到位合格", lambda bb: bb.read("arrival_ok")),
            CallWorker(ctx.board, "camera", lambda bb: {}, "image", name="拍照"),
            RecordCapture(ctx.events),
            CallWorker(
                ctx.board,
                "vision",
                lambda bb: {"image": bb.read("image"), "pose": bb.read("actual")},
                "targets",
                name="检测并换算目标",
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

