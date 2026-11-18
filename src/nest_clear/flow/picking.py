"""将工作模式请求接入纯状态机及异步视觉处理。"""
import logging
from typing import Any

from autoweaver.motion_policy.nodes.control.sequence import Sequence
from autoweaver.motion_policy.nodes.node import TreeNode

from nest_clear.guard import GuardError
from nest_clear.plan.scan_path import guard_position
from nest_clear.record.images import record_image_error
from nest_clear.vision.fine import fine_camera_xy
from nest_clear.vision.hair_integration import render, render_fine
from .context import FlowContext
from .leaves import CallWorker, CheckArrival, Instant, Predicate, RecordCapture
from .work_cycle import WorkCycle

logger = logging.getLogger("nest_clear.flow.picking")


class WorkRequest(Instant):
    """检查请求与状态，并在无需拍照时准备状态机应答。"""

    def __init__(self, ctx: FlowContext) -> None:
        super().__init__(name="检查工作状态")
        self.ctx = ctx

    def perform(self) -> None:
        bb = self._blackboard
        cycle = bb.read("work_cycle")
        if cycle is None:
            calibration = self.ctx.calibration
            def fine_coord(target: tuple[float, float, float], pose: tuple[float, float, float]) -> tuple[float, float, float]:
                """按标定主点计算精定位拍照坐标。"""
                xyz = fine_camera_xy(target, pose, calibration)
                guard_position(xyz, self.ctx.config.travel, "精定位拍照坐标")
                return xyz
            def pick_coords(target: tuple[float, float, float]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
                """分别补偿两把工具并守卫行程。"""
                if getattr(self.ctx.events, "calibration_placeholder", False) and not self.ctx.config.vision.allow_placeholder_calibration:
                    raise GuardError("CALIBRATION-PLACEHOLDER", "相机标定为占位值，拒绝下发挑取坐标", "挑取坐标守卫", "完成相机标定。")
                result = []
                for tool, offset in enumerate(self.ctx.config.tools, 1):
                    xyz = (target[0] + offset[0], target[1] + offset[1], self.ctx.config.pick_depth_z + offset[2])
                    for axis, value, bounds in zip("XYZ", xyz, self.ctx.config.travel):
                        if not __import__("math").isfinite(value) or not bounds[0] <= value <= bounds[1]:
                            raise GuardError("POSITION-TRAVEL", f"工具 {tool} 的 {axis} 轴目标值 {value} 超出行程范围 {bounds}", f"挑取坐标 Task_ID={bb.read('task_id')}", "检查工具补偿、目标和设备行程。")
                    result.append(xyz)
                return result[0], result[1]
            cycle = WorkCycle(self.ctx.path, fine_coord, pick_coords, self.ctx.config.vision)
            bb.write("work_cycle", cycle, self.name)
        action = cycle.expect(bb.read("request_func"))
        bb.write("work_action", action, self.name)
        if action == "start":
            apply_reply(bb, cycle.start(), self.ctx)
        elif action == "picked":
            apply_reply(bb, cycle.on_picked(), self.ctx)


def apply_reply(bb: Any, reply: Any, ctx: FlowContext) -> None:
    """将状态机应答映射到统一握手出口。"""
    payload = {}
    if reply.scan_xyz is not None:
        payload["scan_xyz"] = reply.scan_xyz
        bb.write("scan", reply.scan_xyz, "工作状态机")
    if reply.tool1_xyz is not None:
        payload["pick1_xyz"] = reply.tool1_xyz
        payload["pick2_xyz"] = reply.tool2_xyz
    if reply.removable_count is not None:
        payload["removable_count"] = reply.removable_count
    bb.write("reply_func", reply.func_code, "工作状态机")
    bb.write("reply_coordinates", payload, "工作状态机")
    bb.write("reply_tool", None, "工作状态机")
    bb.write("next_index", bb.read("work_cycle").view_index + 1, "工作状态机")
    if reply.event.get("warning"):
        logger.warning("视野 %s 粗拍次数超限，强制离开", reply.event.get("view_index"))
        ctx.events.emit("warning", task_id=bb.read("task_id"), reason="粗拍次数超限", scan_index=reply.event.get("view_index"))
    ctx.events.emit("work_cycle", task_id=bb.read("task_id"), reply_func=reply.func_code, **reply.event)


class UnpackCapture(Instant):
    def __init__(self, ctx: FlowContext) -> None:
        super().__init__(name="记录采图元数据")
        self.ctx = ctx

    def perform(self) -> None:
        bb = self._blackboard
        value = bb.read("image")
        if not isinstance(value, tuple):
            return
        image, meta = value
        bb.write("image", image, self.name)
        bb.write("capture_meta", meta, self.name)
        if self.ctx.events.run_directory is None:
            self.ctx.events.emit("capture", task_id=bb.read("task_id"), stage=bb.read("work_action"), **meta)


class PlaceholderWarning(Instant):
    def __init__(self, ctx: FlowContext) -> None:
        super().__init__(name="检查占位标定放行")
        self.ctx = ctx

    def perform(self) -> None:
        bb = self._blackboard
        if bb.read("work_action") == "coarse" and self.ctx.config.vision.allow_placeholder_calibration and getattr(self.ctx.events, "calibration_placeholder", False):
            logger.warning("占位标定放行中，Task_ID=%s", bb.read("task_id"))
            self.ctx.events.emit("placeholder_calibration", task_id=bb.read("task_id"), reason="占位标定放行中，请完成相机标定。")


class FinishCapture(Instant):
    """把视觉结果送入相应状态转移。"""
    def __init__(self, ctx: FlowContext) -> None:
        super().__init__(name="处理工作拍照")
        self.ctx = ctx

    def perform(self) -> None:
        bb = self._blackboard
        cycle = bb.read("work_cycle")
        detection = bb.read("detection")
        stage = bb.read("work_action")
        if stage == "coarse":
            reply = cycle.on_coarse(bb.read("actual"), detection["observations"])
        else:
            reply = cycle.on_fine(bb.read("actual"), detection["fine_result"])
        # 状态转移和应答不受可选画面处理的异常影响。
        archive = getattr(self.ctx.events, "archive", None)
        if archive:
            try:
                if stage == "coarse":
                    overlay = render(bb.read("image"), detection["observations"], None,
                                     self.ctx.config.vision.ignore_regions_px, cycle.targets)
                else:
                    overlay = render_fine(bb.read("image"), detection.get("observations", []), detection["fine_result"],
                                          bb.read("actual"), self.ctx.calibration, self.ctx.config.vision.fine_box_half_mm)
                archive.rendered(overlay, bb.read("next_index"), bb.read("task_id"), bb.read("raw_path"),
                                 self.ctx.config.vision.preview_window,
                                 **({"capture_id": bb.read("capture_id")} if getattr(self.ctx.events, "run_directory", None) is not None else {}))
            except Exception as exc:
                record_image_error(self.ctx.events, f"{stage}叠加图 {bb.read('raw_path')}", exc, bb.read("task_id"))
        apply_reply(bb, reply, self.ctx)


def work_branch(ctx: FlowContext) -> TreeNode:
    """状态校验后按需校验到位、拍照并处理视觉结果。"""
    from autoweaver.motion_policy.nodes.control.fallback import Fallback
    capture = Sequence([
        Predicate("需要拍照", lambda bb: bb.read("work_action") in ("coarse", "fine")),
        CallWorker(ctx.board, "plc", lambda bb: {"operation": "actual"}, "actual", name="读取实时坐标"),
        CheckArrival(ctx.config, ctx.events),
        Predicate("到位合格", lambda bb: bb.read("arrival_ok")),
        PlaceholderWarning(ctx),
        CallWorker(ctx.board, "camera", lambda bb: {"stage": bb.read("work_action"), "task_id": bb.read("task_id")}, "image", name="拍照"),
        UnpackCapture(ctx),
        RecordCapture(ctx.events),
        CallWorker(ctx.board, "vision", lambda bb: {"image": bb.read("image"), "pose": bb.read("actual"),
                   "work_action": bb.read("work_action"), "cycle": bb.read("work_cycle"),
                   "scan": ctx.path[bb.read("work_cycle").view_index], "scan_path": ctx.path},
                   "detection", name="检测并换算目标"),
        FinishCapture(ctx),
    ], name="到位拍照")
    return Sequence([WorkRequest(ctx), Fallback([capture, Predicate("无需拍照", lambda bb: bb.read("work_action") in ("start", "picked"))])], name="工作模式")


scan_branch = work_branch
next_branch = work_branch
