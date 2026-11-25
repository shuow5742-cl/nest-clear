"""双相机对刀迭代子树及单一职责业务叶。"""

from dataclasses import replace
import logging
import json
import math
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import yaml
from autoweaver.motion_policy.nodes.control.fallback import Fallback
from autoweaver.motion_policy.nodes.control.sequence import Sequence
from autoweaver.motion_policy.nodes.node import TreeNode, Status

from nest_clear.guard import GuardError, diagnostic, format_pose
from nest_clear.plan.scan_path import guard_position
from nest_clear.record.summary import write_tooltip_summary
from nest_clear.tooltip import SilhouetteTipDetector, check_quality, compute_compensation, is_aligned, solve_correction, fit_view_jacobian
from nest_clear.tooltip.factory import make_tip_detector
from nest_clear.tooltip.store import load_baseline, save_tools
from nest_clear.tooltip.click import confirm_tip, _DISPLAY_ERROR
from .context import FlowContext
from .leaves import CallWorker, Instant
from .modes import Mode, transition
from nest_clear.preview.control import close_for_mode

logger = logging.getLogger("nest_clear.flow.calibration")


def save_session_record(ctx, session, result, reason=None):
    directory = getattr(ctx.events, 'run_directory', None)
    if directory is None:
        return
    path = Path(directory) / 'tooltip' / str(session['task_ids'][0]) / 'session_record.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**session, 'result': result, 'reason': reason}, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def view_settings():
    from nest_clear.baseline.session import settings
    return settings(Path(__file__).resolve().parents[3] / 'config' / 'tooltip_baseline.yaml')


def fail(reason: str, hint: str = "检查对刀基准、相机和设备位置。") -> None:
    """将对刀业务错误交给子树的 131 出口。"""
    raise GuardError("TOOLTIP-FAILED", reason, "对刀", hint)


def change_mode(bb: object, ctx: FlowContext, target: Mode, name: str) -> None:
    """执行合法模式切换并记录请求上下文。"""
    current = bb.read("mode")
    if current != target:
        if current == Mode.IDLE:
            close_for_mode(ctx.events, target.value, bb.read('task_id'), bb.read('request_func'))
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
            fail(f"最终坐标不一致：期望 {format_pose(expected)}，实际 {format_pose(final)}，实时 {format_pose(actual)}，公差 {tol} mm")
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
        self.ctx.events.meta_update(task_id=bb.read("task_id"), request_func=bb.read("request_func"),
                                    tool=tool, old=old, new=new)
        self.ctx.events.emit("tooltip_compensation", task_id=bb.read("task_id"), tool=tool,
                             old=old, new=new, delta=delta)
        session.update(delta=delta, old_compensation=list(old), new_compensation=list(new))
        self.ctx.events.emit('tooltip_state', phase='done', message='完成', tool=tool,
                             old_compensation=list(old), new_compensation=list(new), compensation_difference=[n-o for n, o in zip(new, old)], rounds=session.get('rounds', []))
        save_session_record(self.ctx, session, '成功')
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
        self.ctx.events.emit("tooltip_start", task_id=bb.read("task_id"), start_task_id=bb.read("task_id"),
                             start_token=self.ctx.events.clock().isoformat(), tool=tool, phase='starting', message='检查基准和到位')
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
        from nest_clear.devices.tooltip_camera_store import baseline_differences
        differences = baseline_differences(self.ctx.config, data.get('camera_settings')) if isinstance(data, dict) else []
        if differences:
            message = '对刀相机设置已变更（' + '、'.join(differences) + '），基准需重建'
            self.ctx.events.emit('tooltip_settings_warning', camera_settings_warning=message, reason=message)
        baseline = load_baseline(path)
        if baseline.tool_name != f"tool{tool}" or {v.name for v in baseline.views} != {"front", "side"}:
            fail(f"工具 {tool} 基准工具号或视图不匹配")
        session['probing'] = any(v.jacobian is None for v in baseline.views)
        session['probe_samples'] = []
        session['rounds'] = []
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
        images = bb.read("tooltip_images")
        for view, image in images.items():
            if self.ctx.events.run_directory is not None:
                paths["raw"][view] = getattr(images, "raw_paths", {}).get(view)
            else:
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
            fail(f"对刀位置超差：实际 {format_pose(actual)}，期望 {format_pose(expected)}，公差 {tol} mm")
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
        focus = {}
        for view in ("front", "side"):
            detector = make_tip_detector(self.ctx.config.tool_calibration.detection[view], SilhouetteTipDetector)
            detection = detector.detect(images[view])
            points[view] = tuple(float(x) for x in detection.point_uv) if detection.point_uv is not None else None
            focus[view] = {"focused": detection.focused, "edge_width_px": detection.edge_width_px, "width_px": detection.width_px, "coarse": detection.coarse, "max_edge_width_px": detector.max_edge_width_px, "diagnostic": detection.diagnostic, 'method': self.ctx.config.tool_calibration.detection[view].get('method', 'trident')}
        bb.write("tooltip_focus", focus, self.name)
        session = bb.read("tooltip_session")
        paths = session["images"][-1]
        archive = getattr(self.ctx.events, "archive", None)
        targets = {view.name: view.target_uv for view in bb.read("tooltip_baseline").views}
        for view in ("front", "side"):
            visual = view_settings()
            paths["overlay"][view] = archive.calibration(
                images[view], session["tool"], bb.read("task_id"), paths["iteration"], view,
                points[view], targets[view], paths["raw"].get(view), focus=focus,
                um_per_px=visual['main_um_per_px' if view == 'front' else 'aux_um_per_px'],
                radius_mm=visual['main_tol_mm'], threshold=self.ctx.config.tool_calibration.detection[view]['threshold']) if archive else None
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
        cfg = self.ctx.config.tool_calibration
        focus = bb.read("tooltip_focus")
        reasons = []
        for view in baseline.views:
            f = focus[view.name]
            camera = {"front": "正面", "side": "侧面"}.get(view.name, view.name)
            if points[view.name] is None:
                diagnostic = {"no dark component": "图像中未找到暗色针尖区域",
                              "largest component is too small": "图像中的暗色针尖区域过小"}.get(f.get("diagnostic"), f.get("diagnostic") or "检测失败")
                reasons.append(f"{camera}相机无法识别针尖：{diagnostic}")
            if f["coarse"]: reasons.append(f"{camera}相机 coarse 粗定位：{f.get('diagnostic') or '识别不准'}")
            if view.width_px is not None and f["width_px"] is not None and abs(f["width_px"] / view.width_px - 1) > .3:
                reasons.append(f"{camera}相机针尖宽度超差")
        if any(points[v] is None for v in points):
            self.ctx.events.emit("tooltip_reliability", task_id=bb.read("task_id"), reliable=False, reasons=reasons)
            fail("；".join(reasons), "检查背光与曝光，以及针尖是否在画面内。")
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
                             pixels=points, focus=bb.read("tooltip_focus"), delta=delta.tolist(),
                             residuals_uv={k: v.tolist() for k, v in residuals.items()},
                             residuals_mm=quality.residuals_mm, condition_number=result.condition_number,
                             z_disagreement_mm=disagreement)
        if not quality.ok or disagreement > cfg.z_disagreement_mm:
            logger.warning("对刀质量告警：残差=%s，条件数=%s，Z 分歧=%s",
                           quality.residuals_mm, quality.condition_number, disagreement)
            self.ctx.events.emit("tooltip_quality_warning", task_id=bb.read("task_id"),
                                 residuals_mm=quality.residuals_mm, condition_number=quality.condition_number,
                                 z_disagreement_mm=disagreement, quality_ok=quality.ok)
        if cfg.quality_hard_fail:
            for name, value in quality.residuals_mm.items():
                if value > cfg.max_residual_mm:
                    camera = {"front": "正面", "side": "侧面"}.get(name, name)
                    reasons.append(f"{camera}相机投影残差 {value:.3f} mm 超过阈值 {cfg.max_residual_mm:.3f} mm")
            if result.condition_number > cfg.max_condition:
                reasons.append(f"条件数 {result.condition_number:.3f} 超过阈值 {cfg.max_condition:.3f}")
            if disagreement > cfg.z_disagreement_mm:
                reasons.append(f"两视图 Z 分歧 {disagreement:.3f} mm 超过阈值 {cfg.z_disagreement_mm:.3f} mm")
        if is_aligned(result, cfg.tip_tolerance_mm):
            reasons.extend(f"{'正面' if v == 'front' else '侧面'}相机不清晰" for v in focus if not focus[v]["focused"])
        self.ctx.events.emit("tooltip_reliability", task_id=bb.read("task_id"), reliable=not reasons, reasons=reasons)
        if reasons: fail("对刀不可靠：" + "；".join(reasons))
        bb.write("tooltip_result", result, self.name)


def _click_process(connection, images, points, tool):
    try:
        results = {}
        for view in ("front", "side"):
            results[view] = confirm_tip(images[view], points[view], f"{view} 工具{tool} 对刀")
            if results[view].point_uv is None:
                break
        connection.send(results)
    except BaseException as exc:
        connection.send(str(exc))
    finally:
        connection.close()


class ReliableTips(TreeNode):
    """不可靠时在独立进程显示窗口，tick 只轮询管道。"""
    def __init__(self, ctx):
        super().__init__(name="针尖可靠性和人工确认")
        self.ctx = ctx
        self.process = None

    def on_start(self):
        self.process = None
        bb = self._blackboard
        try:
            detect = DetectTips(self.ctx); detect._blackboard = bb; detect.perform()
            solve = SolveTips(self.ctx); solve._blackboard = bb; solve.perform()
            return Status.SUCCESS
        except Exception as exc:
            bb.write("error", str(exc), self.name)
            self.reason = str(exc)
            focus = bb.read('tooltip_focus', {})
            if any(item.get('method') == 'trident' and (item.get('coarse') or not item.get('focused')) for item in focus.values()):
                return Status.FAILURE
            if not isinstance(exc, GuardError) or self.ctx.config.tool_calibration.unreliable_action != "click":
                return Status.FAILURE
            parent, child = mp.get_context("spawn").Pipe(False)
            self.connection = parent
            self.process = mp.get_context("spawn").Process(target=_click_process, args=(child, bb.read("tooltip_images"), bb.read("tooltip_points"), bb.read("tooltip_session")["tool"]))
            self.process.start(); child.close()
            self.started = time.monotonic()
            return Status.RUNNING

    def on_halted(self):
        if self.process is not None and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=0.1)
        if self.process is not None:
            self.connection.close()

    def on_running(self):
        bb = self._blackboard
        if self.connection.poll():
            try: result = self.connection.recv()
            except EOFError: result = _DISPLAY_ERROR
            self.process.join(timeout=0)
            self.connection.close()
            if isinstance(result, dict) and (len(result) != 2 or any(r.point_uv is None for r in result.values())):
                for view, r in result.items():
                    auto = bb.read("tooltip_points")[view]
                    correction = float(np.linalg.norm(np.asarray(r.point_uv) - auto)) if r.point_uv is not None and auto is not None else None
                    info = dict(camera=view, tool=bb.read("tooltip_session")["tool"], auto_point=auto, manual_point=r.point_uv,
                                correction_px=correction, correction_mm=correction * view_settings()['main_um_per_px' if view == 'front' else 'aux_um_per_px'] / 1000 if correction is not None else None,
                                elapsed_s=r.elapsed_s, source=r.source)
                    self.ctx.events.emit("tip_click", task_id=bb.read("task_id"), **info)
                    bb.read("tooltip_session").setdefault("tip_clicks", []).append(info)
            if not isinstance(result, dict) or len(result) != 2 or any(r.point_uv is None for r in result.values()):
                bb.write("error", getattr(self, "reason", "") + "；人工点击取消或窗口失败：" + str(result), self.name)
                return Status.FAILURE
            points = bb.read("tooltip_points")
            focus = bb.read("tooltip_focus")
            session = bb.read("tooltip_session")
            for view, r in result.items():
                auto = points[view]
                correction = float(np.linalg.norm(np.asarray(r.point_uv) - auto)) if auto is not None else None
                info = dict(camera=view, tool=session["tool"], auto_point=auto, manual_point=r.point_uv,
                            correction_px=correction, correction_mm=correction * view_settings()['main_um_per_px' if view == 'front' else 'aux_um_per_px'] / 1000 if correction is not None else None,
                            elapsed_s=r.elapsed_s, source=r.source)
                self.ctx.events.emit("tip_click", task_id=bb.read("task_id"), **info)
                session.setdefault("tip_clicks", []).append(info)
                points[view] = r.point_uv
                focus[view].update(coarse=False, focused=True, width_px=None)
            bb.write("tooltip_points", points, self.name)
            bb.write("tooltip_focus", focus, self.name)
            try:
                solve = SolveTips(self.ctx); solve._blackboard = bb; solve.perform()
                return Status.SUCCESS
            except Exception as exc:
                bb.write("error", str(exc), self.name); return Status.FAILURE
        if time.monotonic() - self.started >= self.ctx.config.tool_calibration.click_timeout_s or not self.process.is_alive():
            reason = "人工点击超时" if self.process.is_alive() else _DISPLAY_ERROR
            if self.process.is_alive(): self.process.terminate()
            self.process.join(timeout=0.1); self.connection.close()
            bb.write("error", getattr(self, "reason", "") + "；" + reason, self.name)
            return Status.FAILURE
        return Status.RUNNING


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
        focus = bb.read("tooltip_focus")
        aligned = is_aligned(result, self.ctx.config.tool_calibration.tip_tolerance_mm)
        focused = all(focus[v]["focused"] for v in ("front", "side"))
        session["final_edge_width_px"] = {v: focus[v]["edge_width_px"] for v in focus}
        if not focused:
            session["defocus_iterations"] = session.get("defocus_iterations", 0) + 1
        session["defocus_hold_count"] = session.get("defocus_hold_count", 0) + 1 if aligned and not focused else 0
        if aligned and not focused:
            details = "；".join(f"{'正面' if v == 'front' else '侧面'}相机边缘宽度 {focus[v]['edge_width_px']} px，阈值 {focus[v]['max_edge_width_px']} px" for v in focus if not focus[v]["focused"])
            logger.warning("对刀位置仍离焦：%s；检查对焦/针尖是否在景深内", details)
            self.ctx.events.emit("tooltip_defocus_hold", task_id=bb.read("task_id"), tool=tool, focus=focus, count=session["defocus_hold_count"])
            bb.write("tooltip_session", session, self.name)
            if session["defocus_hold_count"] >= 3:
                fail("针尖在对准位置仍不清晰，镜头焦点与基准位置不一致", "检查对焦/针尖是否在景深内。")
        if aligned and focused:
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
            self.ctx.events.emit('tooltip_state', phase='fail', message=str(reason), tool=session['tool'], rounds=session.get('rounds', []))
            save_session_record(self.ctx, session, '失败', str(reason))
            write_tooltip_summary(self.ctx.events, session, "失败", str(reason))
        change_mode(bb, self.ctx, Mode.IDLE, self.name)
        bb.write("tooltip_session", None, self.name)
        bb.write("tooltip_baseline", None, self.name)
        bb.write("reply_diagnostic", bb.read("error_diagnostic", None) or diagnostic(reason, code="TOOLTIP-FAILED", where="换针对刀", hint="检查对刀基准、相机和设备位置。"), self.name)
        bb.write("reply_func", 131, self.name)
        bb.write("reply_coordinates", {}, self.name)
        bb.write("reply_tool", None, self.name)


class ConfirmCalibration(TreeNode):
    """每个 PLC 请求持续运行到操作员确认；不会在等待期间写坐标。"""
    def __init__(self, ctx):
        super().__init__('等待对刀人工确认')
        self.ctx = ctx
        self.process = None

    def on_start(self):
        self.process = None
        self.manual = False
        try:
            node = DetectTips(self.ctx); node._blackboard = self._blackboard; node.perform()
            bb = self._blackboard
            self.ctx.events.emit('tooltip_observation', task_id=bb.read('task_id'), tool=bb.read('tooltip_session')['tool'],
                                 stage='探测' if bb.read('tooltip_session')['probing'] else '对准',
                                 pixels=bb.read('tooltip_points'), focus=bb.read('tooltip_focus'),
                                 images=bb.read('tooltip_session')['images'][-1], click=False)
            return self.prepare()
        except Exception as exc:
            self._blackboard.write('error', str(exc), self.name)
            return Status.FAILURE

    def publish(self, phase, message, **fields):
        from .tool_offset import reader
        bb = self._blackboard
        session = bb.read('tooltip_session')
        self.ctx.events.emit('tooltip_state', phase=phase, message=message, tool=session['tool'],
                             command_seq=reader(self.ctx).last_seq, rounds=session['rounds'],
                             jacobian_status=session.get('jacobian_status', '未测' if session['probing'] else '已存'),
                             jacobian_quality=session.get('jacobian_quality'), **fields)

    def fit_probe(self):
        bb = self._blackboard; session = bb.read('tooltip_session'); baseline = bb.read('tooltip_baseline')
        samples = session['probe_samples'] + [{'actual': list(bb.read('actual')), 'points': dict(bb.read('tooltip_points'))}]
        try:
            quality_path = Path(__file__).resolve().parents[3] / 'config' / 'tooltip_baseline.yaml'
            quality_config = (yaml.safe_load(quality_path.read_text(encoding='utf-8')) or {}).get('quality', {})
            max_fit_rms = float(quality_config.get('max_fit_rms_px', 2.0))
            if not math.isfinite(max_fit_rms) or max_fit_rms <= 0:
                raise ValueError('拟合残差阈值无效')
            x = np.asarray([s['actual'] for s in samples], dtype=float)
            x -= x.mean(axis=0)
            views = []; rms = {}
            for view in baseline.views:
                y = np.asarray([s['points'][view.name] for s in samples], dtype=float)
                y -= y.mean(axis=0)
                fit = fit_view_jacobian(list(zip(x, y)))
                if fit.rms_px > max_fit_rms or fit.condition_number > self.ctx.config.tool_calibration.max_condition:
                    raise ValueError('拟合残差或位移矩阵条件数超限')
                rms[view.name] = float(fit.rms_px)
                views.append(replace(view, jacobian=fit.jacobian.tolist()))
            matrix = np.vstack([v.jacobian for v in views])
            condition = float(np.linalg.cond(matrix))
            if np.linalg.matrix_rank(matrix) < 3 or not math.isfinite(condition) or condition > self.ctx.config.tool_calibration.max_condition or np.linalg.norm(np.asarray(views[0 if views[0].name == 'front' else 1].jacobian)[:, 2]) < 1e-6:
                raise ValueError('联合雅可比降秩或条件数超限')
            measured = replace(baseline, views=tuple(views), jacobian_pending=False,
                               jacobian_measured_at=self.ctx.events.clock().isoformat(),
                               jacobian_session=str(session['task_ids'][0]))
            from nest_clear.baseline.session import baseline_mapping
            from nest_clear.tooloffset.store import save_mapping
            mapping = baseline_mapping(measured)
            mapping['jacobian_pending'] = False
            for item, view in zip(mapping['views'], measured.views):
                item['jacobian'] = [list(row) for row in view.jacobian]
            mapping['jacobian_measured_at'] = measured.jacobian_measured_at
            mapping['jacobian_session'] = measured.jacobian_session
            path = Path(self.ctx.config.tool_calibration.directory) / f"tool{session['tool']}_baseline.yaml"
            save_mapping(path, mapping, f"tool{session['tool']}_baseline")
            bb.write('tooltip_baseline', measured, self.name)
            session.update(probing=False, probe_samples=samples, jacobian_status='本次测得', jacobian_quality={'rms_px': rms, 'condition_number': condition})
            self.ctx.events.emit('tooltip_jacobian', tool=session['tool'], quality=session['jacobian_quality'], samples=samples)
        except Exception as exc:
            fail('雅可比测量质量不合格：检查针尖识别/对焦', str(exc))

    def prepare(self):
        bb = self._blackboard; session = bb.read('tooltip_session'); cfg = self.ctx.config.tool_calibration
        points = bb.read('tooltip_points')
        if any(points[v] is None for v in ('front', 'side')):
            if not session['probing']:
                solve = SolveTips(self.ctx); solve._blackboard = bb; solve.perform()
            fail('相机无法识别针尖：检查针尖识别/对焦')
        if session['probing']:
            focus = bb.read('tooltip_focus')
            if any(focus[v]['coarse'] or not focus[v]['focused'] for v in ('front', 'side')):
                self.aligned = False; self.too_large = True
                self.publish('limit', '探测识别不可靠，请手动点击或放弃')
                return Status.RUNNING
        if session['probing'] and len(session['probe_samples']) == 4:
            self.fit_probe()
        baseline = bb.read('tooltip_baseline'); actual = np.asarray(bb.read('actual'), dtype=float)
        if session['probing']:
            index = len(session['probe_samples']); p0 = np.asarray(baseline.p0)
            target = p0.copy()
            if index < 3: target[index] += cfg.probe_step_mm
            delta = target - actual; aligned = False
            message = f'探测中 第{index + 1}/4步：' + (f'探测移动 +{cfg.probe_step_mm:g}mm {"XYZ"[index]}，用于测量相机像素与模组移动关系' if index < 3 else '返回 P0')
            stage = '探测'
        else:
            solve = SolveTips(self.ctx); solve._blackboard = bb
            try:
                solve.perform()
            except GuardError as exc:
                self.aligned = False; self.too_large = True
                self.publish('limit', str(exc) + '，请手动点击或放弃')
                return Status.RUNNING
            result = bb.read('tooltip_result'); delta = np.asarray(result.delta_xyz)
            aligned = is_aligned(result, cfg.tip_tolerance_mm)
            target = actual + delta
            message = '已在范围内，请确认' if aligned else '等待确认移动'
            stage = '对准'
            if not aligned and session['iterations'] >= cfg.max_iterations:
                fail(f"对刀偏移超过最大次数 {session['iterations']}")
        too_large = bool(np.any(np.abs(delta) > cfg.max_step_mm + 1e-9) or np.any(np.abs(target - baseline.p0) > cfg.max_total_mm + 1e-9))
        if not aligned and not too_large:
            guard_position(tuple(target), self.ctx.config.travel, f"对刀工具 {session['tool']}")
        if too_large:
            message = '修正量过大，疑似识别错误，请手动点击或放弃'
        self.aligned, self.too_large, self.target, self.delta, self.stage = aligned, too_large, target, delta, stage
        self.sample = {'actual': actual.tolist(), 'points': {k: list(v) for k, v in points.items()}}
        visual = view_settings()
        deviations = {v.name: ((np.asarray(points[v.name]) - v.target_uv) * visual['main_um_per_px' if v.name == 'front' else 'aux_um_per_px'] / 1000).tolist() for v in baseline.views}
        session['pending'] = {'stage': stage, 'pixels': self.sample['points'], 'deviation_mm': deviations,
                              'delta': delta.tolist(), 'target': target.tolist(), 'cumulative': (target-baseline.p0).tolist(),
                              'click': self.manual, 'images': session['images'][-1]}
        save_session_record(self.ctx, session, '进行中')
        phase = 'limit' if too_large else 'in_range' if aligned else 'probe' if session['probing'] else 'confirm_move'
        self.publish(phase, message, pixels=self.sample['points'], deviation_mm=deviations,
                     delta=delta.tolist(), target=target.tolist(), cumulative=(target - baseline.p0).tolist(), click=self.manual)
        return Status.RUNNING

    def on_running(self):
        bb = self._blackboard; session = bb.read('tooltip_session')
        try:
            if self.process is not None:
                if not self.connection.poll():
                    if self.process.is_alive(): return Status.RUNNING
                    fail('人工点击窗口失败')
                result = self.connection.recv(); self.process.join(timeout=.1); self.connection.close(); self.process = None
                if not isinstance(result, dict) or len(result) != 2 or any(r.point_uv is None for r in result.values()):
                    self.publish('click_failed', '人工点击取消，请重试或放弃')
                    return Status.RUNNING
                points = bb.read('tooltip_points'); focus = bb.read('tooltip_focus')
                for view, r in result.items():
                    correction = float(np.linalg.norm(np.asarray(r.point_uv) - points[view])) if points[view] is not None else None
                    scale = view_settings()['main_um_per_px' if view == 'front' else 'aux_um_per_px'] / 1000
                    info = dict(camera=view, tool=session['tool'], auto_point=points[view], manual_point=r.point_uv,
                                correction_px=correction, correction_mm=correction * scale if correction is not None else None,
                                source=r.source, elapsed_s=r.elapsed_s)
                    self.ctx.events.emit('tip_click', task_id=bb.read('task_id'), **info)
                    session.setdefault('tip_clicks', []).append(info)
                    points[view] = r.point_uv; focus[view].update(coarse=False, focused=True, width_px=None)
                bb.write('tooltip_points', points, self.name); bb.write('tooltip_focus', focus, self.name)
                archive = getattr(self.ctx.events, 'archive', None)
                if archive:
                    visual = view_settings(); paths = session['images'][-1]
                    for view in bb.read('tooltip_baseline').views:
                        paths['overlay'][view.name] = archive.calibration(bb.read('tooltip_images')[view.name], session['tool'], bb.read('task_id'), paths['iteration'], view.name,
                                            points[view.name], view.target_uv, paths['raw'].get(view.name), focus=focus,
                                            um_per_px=visual['main_um_per_px' if view.name == 'front' else 'aux_um_per_px'], radius_mm=visual['main_tol_mm'], manual=True,
                                            threshold=self.ctx.config.tool_calibration.detection[view.name]['threshold'])
                self.manual = True
                return self.prepare()
            from .tool_offset import reader
            allowed = {'tooltip_abort', 'tooltip_click'}
            if not self.too_large: allowed.add('tooltip_confirm_result' if self.aligned else 'tooltip_confirm_move')
            command = reader(self.ctx).poll(allowed)
            if not command: return Status.RUNNING
            name, _ = command
            if name == 'tooltip_abort': fail('用户放弃')
            if name == 'tooltip_click':
                parent, child = mp.get_context('spawn').Pipe(False); self.connection = parent
                self.process = mp.get_context('spawn').Process(target=_click_process, args=(child, bb.read('tooltip_images'), bb.read('tooltip_points'), session['tool']))
                self.process.start(); child.close(); self.publish('clicking', '请依次点选两台相机的针尖'); return Status.RUNNING
            now = self.ctx.events.clock().isoformat()
            row = dict(session.pop('pending'), iteration=len(session['rounds']) + 1, confirmed_at=now)
            if self.aligned:
                row['target'] = None
            session['rounds'].append(row)
            save_session_record(self.ctx, session, '进行中')
            self.ctx.events.emit('tooltip_confirmed', task_id=bb.read('task_id'), tool=session['tool'], **row)
            if self.aligned:
                session['ended'] = True; func = 111 if session['tool'] == 1 else 121; coords = {}
                self.ctx.events.emit('tooltip_aligned', task_id=bb.read('task_id'), tool=session['tool'], iterations=session['iterations'])
                self.publish('wait130', '等待 PLC 130')
            else:
                if session['probing']: session['probe_samples'].append(self.sample)
                else: session['iterations'] += 1
                func = (110 if session['tool'] == 1 else 120) if session['offset'] else (100 if session['tool'] == 1 else 101)
                target = tuple(float(v) for v in self.target)
                session.update(offset=True, last_position=target)
                coords = {f"calibration{session['tool']}_xyz": target}
                self.ctx.events.emit('tooltip_offset', task_id=bb.read('task_id'), tool=session['tool'], coordinates=target, iteration=session['iterations'])
                self.publish('wait_plc', '等待 PLC 到位')
            bb.write('reply_func', func, self.name); bb.write('reply_coordinates', coords, self.name); bb.write('reply_tool', session['tool'], self.name)
            return Status.SUCCESS
        except Exception as exc:
            bb.write('error', str(exc), self.name); return Status.FAILURE

    def on_halted(self):
        if self.process is not None:
            if self.process.is_alive(): self.process.terminate()
            self.process.join(timeout=.1); self.connection.close(); self.process = None


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
        CallWorker(ctx.board, "calib_camera", lambda bb: {"task_id": bb.read("task_id"),
                   "request_func": bb.read("request_func"), "actual": bb.read("actual"),
                   "target": bb.read("tooltip_session")["last_position"],
                   "iteration": len(bb.read("tooltip_session")["images"]) + 1}, "tooltip_images", name="采集对刀双视图"),
        ArchiveTips(ctx), ConfirmCalibration(ctx),
    ])
    return Fallback([Sequence(steps, name="对刀迭代"), CalibrationFailed(ctx)], name="对刀业务或 131")


def start_branch(ctx: FlowContext) -> TreeNode:
    """构造 100/101 开始分支。"""
    return calibration_branch(ctx, True)


def continue_branch(ctx: FlowContext) -> TreeNode:
    """构造 110/120 继续分支。"""
    return calibration_branch(ctx, False)
