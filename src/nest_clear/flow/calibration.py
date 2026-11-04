"""双相机对刀迭代子树及单一职责业务叶。"""

from dataclasses import replace
import logging
import math
from pathlib import Path

import numpy as np
import yaml
from autoweaver.motion_policy.nodes.control.fallback import Fallback
from autoweaver.motion_policy.nodes.control.sequence import Sequence
from autoweaver.motion_policy.nodes.node import TreeNode

from nest_clear.guard import GuardError
from nest_clear.plan.scan_path import guard_position
from nest_clear.record.summary import write_tooltip_summary
from nest_clear.tooltip import SilhouetteTipDetector, check_quality, compute_compensation, is_aligned, solve_correction
from nest_clear.tooltip.store import load_baseline, save_tools
from .context import FlowContext
from .leaves import CallWorker, Instant
from .modes import Mode, transition

logger = logging.getLogger("nest_clear.flow.calibration")


def fail(reason: str, hint: str = "检查对刀基准、相机和设备位置。") -> None:
    """将对刀业务错误交给子树的 131 出口。"""
    raise GuardError("TOOLTIP-FAILED", reason, "对刀", hint)


def change_mode(bb: object, ctx: FlowContext, target: Mode, name: str) -> None:
    """执行合法模式切换并记录请求上下文。"""
    current = bb.read("mode")
    if current != target:
        bb.write("mode", transition(current, target), name)
        ctx.events.emit("mode_change", previous_mode=current.value, mode=target.value,
                        task_id=bb.read("task_id"), request_func=bb.read("request_func"),
                        scan_index=bb.read("next_index"))


class CheckSession(Instant):
    """130 仅接受已经返回完成码的会话。"""
    def __init__(self) -> None:
        super().__init__(name="检查对刀会话已结束")

    def perform(self) -> None:
        session = self._blackboard.read("tooltip_session")
        if not session or not session["ended"]:
            fail("收到 130 时对刀会话尚未结束（未回 111/121）")


class SaveCompensation(Instant):
    """核对最终位置，持久化后更新运行中的补偿。"""
    def __init__(self, ctx: FlowContext) -> None:
        super().__init__(name="核对最终坐标并保存补偿")
        self.ctx = ctx

    def perform(self) -> None:
        bb = self._blackboard
        session = bb.read("tooltip_session")
        baseline = bb.read("tooltip_baseline")
        final = bb.read("tooltip_final")
        actual = bb.read("actual")
        expected = session["last_position"] if session["offset"] else baseline.p0
        cfg = self.ctx.config.tool_calibration
        tol = cfg.consistency_tolerance_mm if session["offset"] else cfg.position_tolerance_mm

        def matches(value: object) -> bool:
            return (isinstance(value, (tuple, list)) and len(value) == 3
                    and all(math.isfinite(a) and abs(a - b) <= tol for a, b in zip(value, expected)))

        ok = matches(final) and (session["offset"] or matches(actual))
        self.ctx.events.emit("tooltip_consistency", task_id=bb.read("task_id"), tool=session["tool"],
                             expected=expected, actual=final, realtime=actual if not session["offset"] else None,
                             tolerance_mm=tol, ok=ok)
        if not ok:
            fail(f"最终坐标不一致：期望 {expected}，实际 {final}，实时 {actual}，公差 {tol} mm")
        result = compute_compensation(baseline, final)
        new = tuple(float(x) for x in result.offset_xyz)
        delta = [float(x) for x in result.delta_xyz]
        tool = session["tool"]
        old = self.ctx.config.tools[tool - 1]
        try:
            save_tools(Path(cfg.directory) / "tools.yaml", tool, new, source="calibration",
                       task_id=bb.read("task_id"), details={"P0": list(baseline.p0), "P_final": list(final),
                       "delta": delta, "old_compensation": list(old), "new_compensation": list(new)})
        except Exception:
            logger.exception("对刀补偿保存失败，工具 %s", tool)
            raise
        offsets = list(self.ctx.config.tools)
        offsets[tool - 1] = new
        object.__setattr__(self.ctx.config, "tools", tuple(offsets))
        self.ctx.events.emit("tooltip_compensation", task_id=bb.read("task_id"), tool=tool,
                             old=old, new=new, delta=delta)
        session.update(delta=delta, old_compensation=list(old), new_compensation=list(new))
        write_tooltip_summary(self.ctx.events, session, "成功")
        change_mode(bb, self.ctx, Mode.IDLE, self.name)
        bb.write("tooltip_session", None, self.name)
        bb.write("tooltip_baseline", None, self.name)
        bb.write("reply_func", 130, self.name)
        bb.write("reply_coordinates", {}, self.name)
        bb.write("reply_tool", None, self.name)


def finish_branch(ctx: FlowContext) -> TreeNode:
    """130 完成校验及保存；失败统一回 131。"""
    steps = [
        RecordTask(ctx),
        CheckSession(),
        CallWorker(ctx.board, "plc", lambda bb: {"operation": "calibration_final",
                   "tool": bb.read("tooltip_session")["tool"]}, "tooltip_final", name="读取最终标定坐标"),
        CallWorker(ctx.board, "plc", lambda bb: {"operation": "actual"}, "actual", name="读取实时坐标"),
        SaveCompensation(ctx),
    ]
    return Fallback([Sequence(steps, name="完成对刀并保存"), CalibrationFailed(ctx)], name="完成对刀或 131")


class StartCalibration(Instant):
    """检查开关与基准并准备新会话。"""
    def __init__(self, ctx: FlowContext) -> None:
        """绑定对刀配置和事件记录器。"""
        super().__init__(name="检查对刀基准")
        self.ctx = ctx

    def perform(self) -> None:
        bb = self._blackboard
        tool = 1 if bb.read("request_func") == 100 else 2
        self.ctx.events.emit("tooltip_start", task_id=bb.read("task_id"), tool=tool)
        session = {"tool": tool, "iterations": 0, "offset": False, "last_position": None,
                   "ended": False, "start_time": self.ctx.events.clock().isoformat(),
                   "task_ids": [bb.read("task_id")], "images": [], "delta": None,
                   "old_compensation": list(self.ctx.config.tools[tool - 1]), "new_compensation": None}
        bb.write("tooltip_session", session, self.name)
        if not self.ctx.config.tool_calibration.enabled:
            fail("对刀功能未启用（tool_calibration.enabled=false）")
        path = Path(self.ctx.config.tool_calibration.directory) / f"tool{tool}_baseline.yaml"
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("status") == "占位，未标定":
            fail(f"工具 {tool} 基准为占位，未标定：{path}")
        baseline = load_baseline(path)
        if baseline.tool_name != f"tool{tool}" or {v.name for v in baseline.views} != {"front", "side"}:
            fail(f"工具 {tool} 基准工具号或视图不匹配")
        bb.write("tooltip_baseline", baseline, self.name)
        session["last_position"] = baseline.p0
        bb.write("tooltip_session", session, self.name)


class RecordTask(Instant):
    """将继续和结束请求加入当前对刀会话。"""
    def __init__(self, ctx: FlowContext) -> None:
        """绑定记录上下文。"""
        super().__init__(name="记录对刀请求")
        self.ctx = ctx

    def perform(self) -> None:
        session = self._blackboard.read("tooltip_session")
        if session:
            session["task_ids"].append(self._blackboard.read("task_id"))


class ArchiveTips(Instant):
    """采图后先提交两个原图，保证检测失败仍可追查。"""
    def __init__(self, ctx: FlowContext) -> None:
        """绑定图片归档器。"""
        super().__init__(name="归档对刀原图")
        self.ctx = ctx

    def perform(self) -> None:
        bb = self._blackboard
        session = bb.read("tooltip_session")
        task = bb.read("task_id")
        iteration = len(session["images"]) + 1
        paths = {"task_id": task, "iteration": iteration, "raw": {}, "overlay": {}}
        archive = getattr(self.ctx.events, "archive", None)
        for view, image in bb.read("tooltip_images").items():
            paths["raw"][view] = archive.calibration(image, session["tool"], task, iteration, view) if archive else None
        session["images"].append(paths)


class CheckPosition(Instant):
    """逐轴核对初始位置或上次偏移位置。"""
    def __init__(self, ctx: FlowContext, start: bool) -> None:
        """绑定对刀配置和事件记录器。"""
        super().__init__(name="检查对刀实时位置")
        self.ctx = ctx
        self.start = start

    def perform(self) -> None:
        bb = self._blackboard
        session = bb.read("tooltip_session")
        if not session or (not self.start and session["ended"]):
            fail("对刀会话不存在或已经结束")
        actual = bb.read("actual")
        expected = session["last_position"]
        tol = self.ctx.config.tool_calibration.position_tolerance_mm
        ok = len(actual) == 3 and all(math.isfinite(a) and abs(a - b) <= tol for a, b in zip(actual, expected))
        self.ctx.events.emit("tooltip_position_check", task_id=bb.read("task_id"), tool=session["tool"],
                             actual=actual, expected=expected, tolerance_mm=tol, ok=ok)
        if not ok:
            fail(f"对刀位置超差：实际 {actual}，期望 {expected}，公差 {tol} mm")
        if self.start:
            target = Mode.CALIBRATING_1 if session["tool"] == 1 else Mode.CALIBRATING_2
            change_mode(bb, self.ctx, target, self.name)


class DetectTips(Instant):
    """用配置的剪影检测器识别两台相机的针尖。"""
    def __init__(self, ctx: FlowContext) -> None:
        """绑定对刀配置和事件记录器。"""
        super().__init__(name="检测两视图针尖")
        self.ctx = ctx

    def perform(self) -> None:
        bb = self._blackboard
        images = bb.read("tooltip_images")
        points = {}
        for view in ("front", "side"):
            detection = SilhouetteTipDetector(**self.ctx.config.tool_calibration.detection[view]).detect(images[view])
            if detection.point_uv is None:
                diagnostic = {
                    "no dark component": "图像中未找到暗色针尖区域",
                    "largest component is too small": "图像中的暗色针尖区域过小",
                }.get(detection.diagnostic, "图像中未能识别有效针尖区域")
                camera = "正面" if view == "front" else "侧面"
                fail(f"{camera}相机无法识别针尖：{diagnostic}", "检查背光与曝光，以及针尖是否在画面内。")
            points[view] = tuple(float(x) for x in detection.point_uv)
        session = bb.read("tooltip_session")
        paths = session["images"][-1]
        archive = getattr(self.ctx.events, "archive", None)
        targets = {view.name: view.target_uv for view in bb.read("tooltip_baseline").views}
        for view in ("front", "side"):
            paths["overlay"][view] = archive.calibration(
                images[view], session["tool"], bb.read("task_id"), paths["iteration"], view,
                points[view], targets[view], paths["raw"].get(view)) if archive else None
        bb.write("tooltip_points", points, self.name)


class SolveTips(Instant):
    """求解修正量、以正面 Z 为准并记录质量诊断。"""
    def __init__(self, ctx: FlowContext) -> None:
        """绑定对刀配置和事件记录器。"""
        super().__init__(name="求解针尖修正并检查质量")
        self.ctx = ctx

    def perform(self) -> None:
        bb = self._blackboard
        baseline = bb.read("tooltip_baseline")
        points = bb.read("tooltip_points")
        result = solve_correction(points, baseline, primary_view="front")
        estimates = result.axis_estimates["z"]
        if "front" not in estimates:
            fail("正面相机基准无法观测 Z 轴")
        delta = result.delta_xyz.copy()
        delta[2] = estimates["front"]
        residuals = {}
        mm = {}
        for view in baseline.views:
            j = np.asarray(view.jacobian)
            residuals[view.name] = j @ delta - (np.asarray(view.target_uv) - points[view.name])
            mm[view.name] = np.linalg.pinv(j) @ residuals[view.name]
        result = replace(result, delta_xyz=delta, residuals_uv=residuals, residuals_mm=mm)
        cfg = self.ctx.config.tool_calibration
        quality = check_quality(result, cfg.max_residual_mm, cfg.max_condition)
        disagreement = result.axis_disagreement.get("z", 0.0)
        self.ctx.events.emit("tooltip_detect", task_id=bb.read("task_id"), tool=bb.read("tooltip_session")["tool"],
                             images=bb.read("tooltip_session")["images"][-1],
                             pixels=points, delta=delta.tolist(),
                             residuals_uv={k: v.tolist() for k, v in residuals.items()},
                             residuals_mm=quality.residuals_mm, condition_number=result.condition_number,
                             z_disagreement_mm=disagreement)
        if not quality.ok or disagreement > cfg.z_disagreement_mm:
            logger.warning("对刀质量告警：残差=%s，条件数=%s，Z 分歧=%s",
                           quality.residuals_mm, quality.condition_number, disagreement)
            self.ctx.events.emit("tooltip_quality_warning", task_id=bb.read("task_id"),
                                 residuals_mm=quality.residuals_mm, condition_number=quality.condition_number,
                                 z_disagreement_mm=disagreement, quality_ok=quality.ok)
        bb.write("tooltip_result", result, self.name)


class PrepareCalibrationReply(Instant):
    """判断收敛或守卫并准备下一次偏移。"""
    def __init__(self, ctx: FlowContext) -> None:
        """绑定对刀配置和事件记录器。"""
        super().__init__(name="准备对刀偏移或完成")
        self.ctx = ctx

    def perform(self) -> None:
        bb = self._blackboard
        session = dict(bb.read("tooltip_session"))
        tool = session["tool"]
        result = bb.read("tooltip_result")
        if is_aligned(result, self.ctx.config.tool_calibration.tip_tolerance_mm):
            session["ended"] = True
            func = 111 if tool == 1 else 121
            coordinates = {}
            self.ctx.events.emit("tooltip_aligned", task_id=bb.read("task_id"), tool=tool,
                                 iterations=session["iterations"])
        else:
            if session["iterations"] >= self.ctx.config.tool_calibration.max_iterations:
                fail(f"对刀偏移超过最大次数 {session['iterations']}")
            xyz = tuple(float(a + d) for a, d in zip(bb.read("actual"), result.delta_xyz))
            guard_position(xyz, self.ctx.config.travel, f"对刀 Task_ID={bb.read('task_id')} 工具 {tool}")
            func = (110 if tool == 1 else 120) if session["offset"] else (100 if tool == 1 else 101)
            coordinates = {f"calibration{tool}_xyz": xyz}
            session.update(iterations=session["iterations"] + 1, offset=True, last_position=xyz)
            self.ctx.events.emit("tooltip_offset", task_id=bb.read("task_id"), tool=tool,
                                 coordinates=xyz, iteration=session["iterations"])
        bb.write("tooltip_session", session, self.name)
        bb.write("reply_func", func, self.name)
        bb.write("reply_coordinates", coordinates, self.name)
        bb.write("reply_tool", tool, self.name)


class CalibrationFailed(Instant):
    """业务失败清除会话、恢复空闲并准备 131。"""
    def __init__(self, ctx: FlowContext) -> None:
        """绑定对刀配置和事件记录器。"""
        super().__init__(name="对刀失败并准备 131")
        self.ctx = ctx

    def perform(self) -> None:
        bb = self._blackboard
        reason = bb.read("error")
        logger.warning("对刀失败：%s", reason)
        session = bb.read("tooltip_session")
        self.ctx.events.emit("tooltip_failed", task_id=bb.read("task_id"),
                             tool=session["tool"] if session else None,
                             images=session["images"] if session else [],
                             request_func=bb.read("request_func"), reason=reason)
        if session:
            write_tooltip_summary(self.ctx.events, session, "失败", str(reason))
        change_mode(bb, self.ctx, Mode.IDLE, self.name)
        bb.write("tooltip_session", None, self.name)
        bb.write("tooltip_baseline", None, self.name)
        bb.write("reply_func", 131, self.name)
        bb.write("reply_coordinates", {}, self.name)
        bb.write("reply_tool", None, self.name)


def calibration_branch(ctx: FlowContext, start: bool) -> TreeNode:
    """组合启动或继续的对刀步骤，失败统一走 131。"""
    steps = []
    if start:
        steps.append(StartCalibration(ctx))
    else:
        steps.append(RecordTask(ctx))
    steps.extend([
        CallWorker(ctx.board, "plc", lambda bb: {"operation": "actual"}, "actual", name="读取对刀实时坐标"),
        CheckPosition(ctx, start),
        CallWorker(ctx.board, "calib_camera", lambda bb: {}, "tooltip_images", name="采集对刀双视图"),
        ArchiveTips(ctx), DetectTips(ctx), SolveTips(ctx), PrepareCalibrationReply(ctx),
    ])
    return Fallback([Sequence(steps, name="对刀迭代"), CalibrationFailed(ctx)], name="对刀业务或 131")


def start_branch(ctx: FlowContext) -> TreeNode:
    """构造 100/101 开始分支。"""
    return calibration_branch(ctx, True)


def continue_branch(ctx: FlowContext) -> TreeNode:
    """构造 110/120 继续分支。"""
    return calibration_branch(ctx, False)
