"""主相机标定的单次请求行为树。"""
import json
import math
from pathlib import Path

from autoweaver.motion_policy.nodes.control.fallback import Fallback
from autoweaver.motion_policy.nodes.control.sequence import Sequence

from nest_clear.plan.scan_path import guard_position
from nest_clear.guard import GuardError, diagnostic
from .calibration import change_mode
from .leaves import CallWorker, Instant, Predicate
from .modes import Mode


def finish(ctx, bb, outcome, **fields):
    session = bb.read("camcal_session", None)
    if session:
        engine = session.get("engine")
        if engine is not None and ctx.events.run_directory is not None:
            try:
                path = ctx.events.run_directory / "camcal" / session["directory"] / "session_record.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(engine.session_record(), ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
            except Exception as exc:
                ctx.events.emit("error", operation="camcal_record", reason=str(exc))
        ctx.events.emit("camcal_" + outcome, task_id=bb.read("task_id"), start_time=session["start_time"],
                        end_time=ctx.events.clock().isoformat(), n_images=session["shots"], **fields)
    if bb.read("mode") == Mode.CAMERA_CALIBRATING:
        change_mode(bb, ctx, Mode.IDLE, "结束相机标定")
    bb.write("camcal_session", None, "结束相机标定")
    ctx.events.active_camcal = None
    ctx.events.abort_camcal = None


class Prepare(Instant):
    def __init__(self, ctx):
        super().__init__(name="检查相机标定位置")
        self.ctx = ctx

    def perform(self):
        bb = self._blackboard
        session = bb.read("camcal_session", None)
        actual = bb.read("actual")
        if not isinstance(actual, (tuple, list)) or len(actual) != 3 or not all(math.isfinite(x) for x in actual):
            raise ValueError("相机标定实时坐标无效")
        bb.write("camcal_goto_start", False, self.name)
        tolerance = self.ctx.config.camera_calibration.get("position_tolerance_mm", 0.05)
        if session is None:
            start = self.ctx.config.camera_calibration.get("start_xyz")
            if start is None:
                raise GuardError("CAMCAL-START-UNSET", "未配置相机标定起始坐标", "相机标定", "在配置 camera_calibration.start_xyz 填写起始坐标。")
            target = tuple(start)
            move = any(abs(a-b) > tolerance for a, b in zip(actual, target))
            if move:
                guard_position(target, self.ctx.config.travel, "相机标定")
            now = self.ctx.events.clock()
            session = {"start": None if move else tuple(actual), "target": target if move else None, "shots": 0, "engine": None,
                       "start_time": now.isoformat(), "directory": now.strftime("%Y%m%d_%H%M%S_%f")}
            bb.write("camcal_session", session, self.name)
            self.ctx.events.active_camcal = session
            self.ctx.events.abort_camcal = lambda reason: finish(self.ctx, bb, "aborted", reason=reason)
            change_mode(bb, self.ctx, Mode.CAMERA_CALIBRATING, self.name)
            self.ctx.events.emit("camcal_start", task_id=bb.read("task_id"), start_xyz=target)
            if move:
                bb.write("camcal_goto_start", True, self.name)
                bb.write("reply_coordinates", {"camcal_xyz": target}, self.name)
                bb.write("reply_func", 80, self.name)
                bb.write("reply_tool", None, self.name)
                self.ctx.events.emit("camcal_goto_start", task_id=bb.read("task_id"), actual=actual, target=target)
        elif session["target"] is not None:
            if any(abs(a-b) > tolerance for a,b in zip(actual, session["target"])):
                reason = f"相机标定到位超差：期望 {session['target']}，实际 {actual}，公差 {tolerance} mm"
                finish(self.ctx, bb, "aborted", reason=reason)
                raise ValueError(reason)
            if session["start"] is None:
                session["start"] = tuple(actual)


class Process(Instant):
    def __init__(self, ctx):
        super().__init__(name="相机标定采样及求解")
        self.ctx = ctx

    def perform(self):
        bb, ctx = self._blackboard, self.ctx
        session = bb.read("camcal_session")
        result = bb.read("camcal_result")
        step, raw = result['step'], result['raw_path']
        bb.write("raw_path", raw, self.name)
        if step.kind == "fail":
            hint = "请" + step.reason.rsplit("请", 1)[1] if step.reason and "请" in step.reason else "检查标定板、相机光照和运动位置。"
            bb.write("reply_diagnostic", diagnostic(step.reason, code=step.code or "CAMCAL-FAILED", where="相机标定", hint=hint), self.name)
            finish(ctx, bb, "failed", code=step.code, reason=step.reason, detail=step.detail, raw_path=raw)
            reply = 82
        elif step.kind == "done":
            if ctx.camcal_save is None:
                from nest_clear.camcal.session import save_calibration
                save = save_calibration
            else:
                save = ctx.camcal_save
            path = Path(ctx.calibration_path or ctx.config.vision.calibration)
            backup = save(step.result, path, path.parent / "history")
            if ctx.calibration_updated:
                ctx.calibration_updated(step.result.calibration)
            ctx.calibration = step.result.calibration
            ctx.events.calibration_placeholder = False
            from .pick_accuracy import refresh_existing
            refresh_existing(ctx)
            ctx.events.meta_update(task_id=bb.read("task_id"), request_func=80, calibration=str(path), backup=str(backup) if backup else None, report=step.result.report)
            report = dict(step.result.report)
            calibration = step.result.calibration
            if hasattr(calibration, "K"):
                report.update(K=calibration.K.tolist(), distortion=calibration.distortion.tolist())
            finish(ctx, bb, "done", report=report, calibration_path=str(path), backup_path=str(backup) if backup else None)
            reply = 81
        elif step.kind == "move":
            target = tuple(step.target_xyz)
            cfg = session["cfg"]
            if len(target) != 3 or any(not math.isfinite(v) for v in target) or any(abs(target[i]-session["start"][i]) > (cfg.xy_limit_mm if i < 2 else cfg.z_limit_mm) for i in range(3)):
                raise ValueError(f"相机标定目标超出起始位置相对行程：{target}")
            session["target"] = target
            bb.write("reply_coordinates", {"camcal_xyz": target}, self.name)
            reply = 80
        else:
            raise ValueError(f"未知相机标定步骤 {step.kind!r}")
        if reply != 80:
            bb.write("reply_coordinates", {}, self.name)
        bb.write("reply_func", reply, self.name)
        bb.write("reply_tool", None, self.name)


class CheckTarget(Instant):
    def __init__(self, ctx):
        super().__init__(name="检查相机标定终点")
        self.ctx = ctx

    def perform(self):
        bb = self._blackboard
        if bb.read("reply_func") == 80:
            try:
                guard_position(bb.read("reply_coordinates")["camcal_xyz"], self.ctx.config.travel, "相机标定")
            except GuardError as exc:
                finish(self.ctx, bb, "failed", code=exc.code, reason=exc.what)
                raise


class Failed(Instant):
    def __init__(self, ctx):
        super().__init__(name="相机标定失败并准备 82")
        self.ctx = ctx

    def perform(self):
        bb = self._blackboard
        reason = str(bb.read("error"))
        if bb.read("camcal_session", None):
            finish(self.ctx, bb, "failed", code="CAMCAL-FLOW", reason=reason, detail={}, raw_path=bb.read("raw_path"))
        bb.write("reply_diagnostic", bb.read("error_diagnostic", None) or diagnostic(reason, code="CAMCAL-FLOW", where="相机标定", hint="检查相机连接、标定板和光照。"), self.name)
        bb.write("reply_func", 82, self.name)
        bb.write("reply_coordinates", {}, self.name)
        bb.write("reply_tool", None, self.name)


def camera_branch(ctx):
    # 位置检查失败由统一 999 出口处理；采图和业务失败返回 82。
    sample = Fallback([Sequence([CallWorker(ctx.board, "camera", lambda bb: {"stage":"camcal", "task_id":bb.read("task_id")}, "image", name="采集相机标定图"), CallWorker(ctx.board, "camcal", lambda bb: {
        "context": ctx, "session": bb.read("camcal_session"), "captured": bb.read("image"),
        "actual": bb.read("actual"), "task_id": bb.read("task_id")}, "camcal_result", name="异步相机标定"),
        Process(ctx)], name="采样及求解"), Failed(ctx)], name="相机标定或 82")
    return Sequence([CallWorker(ctx.board, "plc", lambda bb: {"operation":"actual"}, "actual", name="读取相机标定坐标"), Prepare(ctx), Fallback([Predicate("前往起始位", lambda bb: bb.read("camcal_goto_start")), sample]), CheckTarget(ctx)], name="相机标定")
