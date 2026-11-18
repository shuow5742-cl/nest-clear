"""将工作模式请求接入纯状态机及异步视觉处理。"""
import logging
from collections import Counter
import math
from datetime import datetime
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

REASONS = {"short": "过短", "small": "面积过小", "thin": "过细", "pale": "颜色过浅"}


def record_detection(ctx: FlowContext, bb: Any, cycle: WorkCycle, detection: dict,
                     stage: str, view: int, coarse_target: Any = None) -> None:
    """记录可复核的候选信息；日志失败不会改变状态转移。"""
    try:
        observations = detection.get("observations", [])
        result = detection.get("fine_result")
        if stage == "fine" and result is not None and coarse_target is not None:
            cycle.match_observations(observations, bb.read("capture_id"), create=False)
        ranked = {id(o): i for i, o in enumerate(cycle.targets, 1)}
        candidates = []
        for obs in observations:
            candidate = obs.candidate
            candidates.append({"target_id": getattr(obs, "target_id", None), "bbox_px": candidate.bbox,
                "identity_px": candidate.identity_uv, "identity_xyz": obs.identity,
                "pick_px": candidate.pick_uv, "pick_xyz": obs.target.xyz,
                "length_mm": obs.length, "area_mm2": obs.area,
                "width_mm": obs.area / obs.length if obs.length else 0,
                "gray_ratio": candidate.gray_ratio, "dark_frac": candidate.dark_frac,
                "gate_passed": obs.reason is None, "reject_reason": REASONS.get(obs.reason, obs.reason),
                "responsibility": "OTHER" if obs.status == "other_view" else "本视野",
                "status": obs.status, "rank": ranked.get(id(obs))})
        counts = Counter(c["status"] for c in candidates)
        rejections = Counter(c["reject_reason"] for c in candidates if c["reject_reason"])
        fields = {"task_id": bb.read("task_id"), "request_func": bb.read("request_func"),
                  "capture_id": bb.read("capture_id"), "kind": stage, "view": view + 1,
                  "elapsed_ms": detection.get("elapsed_ms"), "count": len(candidates),
                  "status_counts": dict(counts), "eligible_count": counts["eligible"],
                  "other_view_count": counts["other_view"], "rejections": dict(rejections),
                  "um_per_px": detection.get("um_per_px"), "z_mm": bb.read("actual")[2],
                  "candidates": candidates}
        if stage == "fine" and result is not None:
            selected = next((obs for obs in observations if obs.candidate is result.candidate), None)
            if selected is not None and getattr(selected, "target_id", None) != getattr(coarse_target, "target_id", None):
                ctx.events.emit("match_ambiguous", task_id=bb.read("task_id"), capture_id=bb.read("capture_id"),
                                target_id=getattr(coarse_target, "target_id", None),
                                selected_target_id=getattr(selected, "target_id", None), reason="精定位选中目标与粗拍身份不一致")
            fields.update(fine_box_half_mm=ctx.config.vision.fine_box_half_mm,
                          selected_target_id=getattr(selected, "target_id", None),
                          requested_target_id=getattr(coarse_target, "target_id", None),
                          box_candidate_count=result.box_candidate_count,
                          coarse_xyz=coarse_target.target.xyz if coarse_target else None,
                          fine_xyz=result.target_xyz,
                          coarse_fine_delta_mm=[b - a for a, b in zip(coarse_target.target.xyz, result.target_xyz)] if coarse_target and result.target_xyz is not None else None,
                          offset_mm=math.dist(coarse_target.target.xyz[:2], result.target_xyz[:2]) if coarse_target and result.target_xyz is not None else None,
                          failure_reason=None if result.status == "ok" else result.status)
        ctx.events.emit("detection", **fields)
    except Exception:
        logger.exception("检测事件记录失败")


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
            def record(event: str, **fields: Any) -> None:
                fields.setdefault("task_id", bb.read("task_id"))
                fields.setdefault("request_func", bb.read("request_func"))
                fields.setdefault("xid", getattr(self.ctx.events, "xid", 0))
                self.ctx.events.emit(event, **fields)
            cycle = WorkCycle(self.ctx.path, fine_coord, pick_coords, self.ctx.config.vision,
                              record=record, tray=getattr(self.ctx.events, "tray_id", None) or "0000")
            self.ctx.events.active_cycle = cycle
            bb.write("work_cycle", cycle, self.name)
        action = cycle.expect(bb.read("request_func"))
        bb.write("work_action", action, self.name)
        if action == "start":
            apply_reply(bb, cycle.start(), self.ctx)
        elif action == "picked":
            cycle.mark_done(bb.read("task_id"), getattr(self.ctx.events, "xid", 0))
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


class BeginCapture(Instant):
    """在采图前建立身份，采图失败时仍能结束精定位尝试。"""
    def __init__(self, ctx: FlowContext) -> None:
        super().__init__(name="分配拍照身份")
        self.ctx = ctx

    def perform(self) -> None:
        if getattr(self.ctx.events, "run_directory", None) is None:
            return
        bb = self._blackboard
        sequence, capture_id = self.ctx.events.next_capture()
        bb.write("capture_sequence", sequence, self.name)
        bb.write("capture_id", capture_id, self.name)
        if bb.read("work_action") == "fine":
            bb.read("work_cycle").begin_fine(capture_id, bb.read("task_id"),
                                             bb.read("request_func"), self.ctx.events.xid)


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
        view = cycle.view_index
        coarse_target = cycle.current_target() if stage == "fine" else None
        if stage == "coarse":
            reply = cycle.on_coarse(bb.read("actual"), detection["observations"], bb.read("capture_id"))
        else:
            reply = cycle.on_fine(bb.read("actual"), detection["fine_result"])
        record_detection(self.ctx, bb, cycle, detection, stage, view, coarse_target)
        # 状态转移和应答不受可选画面处理的异常影响。
        archive = getattr(self.ctx.events, "archive", None)
        if archive:
            info = {"capture_id": bb.read("capture_id"), "wall": "",
                    "tray": getattr(self.ctx.events, "tray_id", None), "view": view + 1,
                    "shot": bb.read("capture_shot"), "task_id": bb.read("task_id"),
                    "eligible": reply.removable_count if stage == "coarse" else None,
                    "failed": None, "um_per_px": None}
            try:
                capture_meta = bb.read("capture_meta") or {}
                timestamp = capture_meta.get("pc_time_after")
                wall = datetime.fromtimestamp(timestamp).astimezone().isoformat() if timestamp is not None else ""
                scale = detection.get("um_per_px")
                if scale is None:
                    try:
                        from nest_clear.vision.geometry import scale_um_per_px
                        scale = scale_um_per_px(self.ctx.calibration, self.ctx.calibration.plane_z, bb.read("actual")[2])
                    except Exception:
                        logger.warning("无法计算画面尺度", exc_info=True)
                info.update(wall=wall,
                            eligible=reply.removable_count if stage == "coarse" else sum(o.status == "eligible" for o in detection.get("observations", [])),
                            failed=sum(o.status == "failed" for o in detection.get("observations", [])),
                            um_per_px=round(scale, 3) if scale is not None else None)
                if stage == "coarse":
                    overlay = render(bb.read("image"), detection["observations"], None,
                                     self.ctx.config.vision.ignore_regions_px, cycle.targets, info=info)
                else:
                    import cv2
                    import numpy as np
                    from dataclasses import replace
                    coarse_uv = None
                    if coarse_target is not None:
                        camera_xyz = (np.asarray(coarse_target.target.xyz) - (np.asarray(bb.read("actual")) + self.ctx.calibration.translation)) @ self.ctx.calibration.rotation
                        uv, _ = cv2.projectPoints(camera_xyz.reshape(1, 3), np.zeros(3), np.zeros(3), self.ctx.calibration.K, self.ctx.calibration.distortion)
                        coarse_uv = tuple(uv.reshape(2))
                    info.update(attempt_id=cycle.active_attempt["attempt_id"] if cycle.active_attempt else None,
                                tool1=None, tool2=None, not_sent_reason=detection["fine_result"].status)
                    result = detection["fine_result"]
                    if getattr(self.ctx.events, "run_directory", None) is not None:
                        bb.write("fine_render_data", {"image": bb.read("image"), "observations": detection.get("observations", []),
                                 "result": result, "pose": bb.read("actual"), "calibration": self.ctx.calibration,
                                 "half_mm": self.ctx.config.vision.fine_box_half_mm, "info": info,
                                 "coarse_uv": coarse_uv, "capture_id": bb.read("capture_id"),
                                 "raw_path": bb.read("raw_path"), "scan": bb.read("next_index"),
                                 "preview": self.ctx.config.vision.preview_window}, self.name)
                        result = replace(result, pick_uv=None, target_xyz=None)
                    overlay = render_fine(bb.read("image"), detection.get("observations", []), result,
                                          bb.read("actual"), self.ctx.calibration, self.ctx.config.vision.fine_box_half_mm,
                                          info=info, coarse_uv=coarse_uv)
                if stage != "fine" or getattr(self.ctx.events, "run_directory", None) is None:
                    archive.rendered(overlay, bb.read("next_index"), bb.read("task_id"), bb.read("raw_path"),
                                     self.ctx.config.vision.preview_window,
                                     **({"capture_id": bb.read("capture_id")} if getattr(self.ctx.events, "run_directory", None) is not None else {}))
            except Exception as exc:
                record_image_error(self.ctx.events, f"{stage}叠加图 {bb.read('raw_path')}", exc, bb.read("task_id"))
                if stage == "coarse":
                    try:
                        fallback = render(bb.read("image"), [], info=info)
                        archive.rendered(fallback, bb.read("next_index"), bb.read("task_id"), bb.read("raw_path"),
                                         self.ctx.config.vision.preview_window,
                                         **({"capture_id": bb.read("capture_id")} if getattr(self.ctx.events, "run_directory", None) is not None else {}))
                    except Exception as fallback_exc:
                        record_image_error(self.ctx.events, "粗拍信息栏", fallback_exc, bb.read("task_id"))
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
        BeginCapture(ctx),
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
