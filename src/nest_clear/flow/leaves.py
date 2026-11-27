"""一盘行为树的单一职责叶节点及 Worker 调用协议。"""

from collections.abc import Callable
import logging
import math
import time
import json
from typing import Any

from autoweaver.motion_policy import CallWorker as FrameworkCallWorker
from autoweaver.motion_policy.nodes.leaf.condition import Condition
from autoweaver.motion_policy.nodes.node import Status, TreeNode

from nest_clear.config import Config
from nest_clear.guard import GuardError, diagnostic
from nest_clear.vision.hair_integration import render
from nest_clear.plan.scan_path import guard_position
from nest_clear.record.events import EventRecorder
from nest_clear.record.images import record_image_error, tray_number
from .work_cycle import ProtocolViolation
from .modes import Mode, transition
from nest_clear.paths import supervisor_state_file
from nest_clear.preview.eligibility import preview_eligibility
from nest_clear.preview.control import close_for_mode


def preview_command_allowed(events, mode, name, params):
    """Apply the same fresh-supervisor and per-camera rules as the UI."""
    try:
        path = supervisor_state_file(events.directory)
        age = max(0, time.time() - path.stat().st_mtime)
        supervisor = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(supervisor, dict): supervisor = {}
    except (OSError, ValueError, TypeError, AttributeError):
        supervisor, age = {}, None
    cameras = ('front', 'side') if name == 'tooltip_swap' or name.startswith('tooltip_runout_') else (params.get('camera'),)
    for camera in cameras:
        if camera not in ('front', 'side'):
            return False, '请选择主或辅助对刀相机'
        allowed, reason = preview_eligibility(supervisor, age, mode.value, camera)
        if not allowed: return False, reason
    return True, None

logger = logging.getLogger("nest_clear.flow")
REQUEST_NAMES = {1: "请求扫描位或下一任务", 60: "挑取完成"}
PREVIEW_COMMANDS = frozenset({'tooltip_preview', 'tooltip_exposure_retest', 'tooltip_exposure_set', 'tooltip_exposure_restore', 'tooltip_exposure_save', 'tooltip_peak_reset', 'tooltip_assign', 'tooltip_swap', 'tooltip_runout_start', 'tooltip_runout_press', 'tooltip_runout_end'})


def handle_preview_command(events, mode, name, params):
    if name in ('tooltip_assign', 'tooltip_swap'):
        events.emit('baseline_state', message='对刀相机已按插口固定，不可在界面修改')
        return
    preparation = getattr(events, 'tooltip_preparation', None)
    if not preparation:
        return
    baseline = getattr(events, 'baseline_session', None)
    allowed, reason = preview_command_allowed(events, mode, name, params)
    if baseline and not baseline.closed:
        allowed, reason = False, '对刀基准建立中使用对刀相机，预览不可用'
    if not allowed:
        if name.startswith('tooltip_runout_'):
            preparation.runout.publish(reply_seq=params.get('_seq'), message=reason)
        if name in ('tooltip_exposure_retest', 'tooltip_exposure_set', 'tooltip_exposure_restore', 'tooltip_exposure_save'):
            preparation.exposure_reply(params.get('camera'), params.get('_seq'), 'error', reason)
        events.emit('baseline_state', message=reason)
        return
    try:
        worker = getattr(preparation, 'worker', None)
        if name == 'tooltip_preview' and params.get('on') and getattr(worker, 'opened', False):
            # The calibration worker keeps its devices open between requests.
            # Release that ownership before preparation opens either camera.
            if getattr(events, 'request_pending', False):
                events.emit('baseline_state', message='对刀请求正在使用相机，请在应答后重试')
                return
            worker.on_stop()
        preparation.command(name, params)
    except Exception as exc:
        if name in ('tooltip_exposure_retest', 'tooltip_exposure_set', 'tooltip_exposure_restore', 'tooltip_exposure_save'):
            preparation.exposure_reply(params.get('camera'), params.get('_seq'), 'error', str(exc))
        events.emit('baseline_state', message=str(exc))


class CallWorker(FrameworkCallWorker):
    """保留设备调用失败的任务日志，结果关联交由框架处理。"""

    def on_running(self) -> Status:
        """记录框架报告的失败，供现场定位请求。"""
        status = super().on_running()
        if status == Status.FAILURE:
            logger.warning("%s失败，Task_ID=%s：%s", self.name, self._blackboard.read("task_id"), self._blackboard.read("error"))
            self._blackboard.write("error_diagnostic", diagnostic(self._blackboard.read("error"), code="CAMERA-CAPTURE" if "采集" in self.name else "WORKER-ERROR", where=self.name, hint="检查相机连接、供电与采图参数。" if "采集" in self.name else "检查对应设备连接和程序日志。"), self.name)
        return status


class WaitRequest(CallWorker):
    """持续读取请求，断线时记录并重试。"""

    def __init__(self, board: Any, tracker: Any, events: EventRecorder, clock: Callable[[], float] = time.monotonic) -> None:
        super().__init__(board, "plc", lambda bb: {"operation": "request"}, result_key="poll_result", name="等待新请求")
        self.tracker = tracker
        self.events = events
        self.clock = clock
        self.failure_since = None
        self.failures = 0

    def on_running(self) -> Status:
        from .tool_offset import reader
        # 没有挂起的标定请求时，也消费并记录不合时宜的界面命令。
        reader_ctx = type("CommandContext", (), {"events": self.events})()
        accuracy_session = self._blackboard.read("pickaccuracy_session", None)
        offset_session = self._blackboard.read("tooloffset_session", None)
        depth_session = self._blackboard.read("depth_session", None)
        baseline_commands = {'baseline_start', 'baseline_select', 'baseline_measure', 'baseline_manual',
                             'baseline_aux', 'baseline_save', 'baseline_confirm', 'baseline_end'}
        preview_commands = PREVIEW_COMMANDS
        abort_commands = set()
        if accuracy_session:
            abort_commands.add("abort")
        if offset_session:
            abort_commands.update(("tooloffset_abort", "abort"))
        if depth_session:
            abort_commands.add("depthcal_abort")
        command = reader(reader_ctx).poll(baseline_commands | preview_commands | abort_commands)
        preparation = getattr(self.events, 'tooltip_preparation', None)
        baseline = getattr(self.events, 'baseline_session', None)
        if baseline and baseline.closed:
            if self._blackboard.read('mode') == Mode.BASELINE:
                self._blackboard.write('mode', transition(Mode.BASELINE, Mode.IDLE), self.name)
                self.events.emit('mode_change', previous_mode=Mode.BASELINE.value, mode=Mode.IDLE.value)
            self.events.baseline_session = baseline = None
        if command and command[0] in preview_commands:
            name, params = command
            handle_preview_command(self.events, self._blackboard.read('mode'), name, params)
        if command and command[0].startswith('baseline_'):
            name, params = command
            if name == 'baseline_start':
                if self._blackboard.read('mode') == Mode.IDLE and baseline is None and not (preparation and getattr(preparation, 'busy', False)):
                    try:
                        close_for_mode(self.events, Mode.BASELINE.value)
                        baseline = self.events.start_baseline()
                        self.events.baseline_session = baseline
                        self._blackboard.write('mode', transition(Mode.IDLE, Mode.BASELINE), self.name)
                        self.events.emit('mode_change', previous_mode=Mode.IDLE.value, mode=Mode.BASELINE.value)
                    except Exception as exc:
                        self.events.emit('baseline_state', phase='idle', message=f'无法开始对刀基准建立：{exc}')
                else:
                    self.events.emit('baseline_state', message='当前正在挑毛或标定，忽略开始请求')
            elif baseline:
                baseline.submit(name, params, reader(reader_ctx).last_seq)
            elif name == 'baseline_end' and preparation and preparation.active:
                for camera in ('front', 'side'):
                    if preparation.enabled[camera]: preparation.command('tooltip_preview', {'camera':camera, 'on':False})
        if command and command[0] == "depthcal_abort":
            depth_session.abort_requested = True
            self.events.emit("depthcal_abort_requested")
        if command and command[0] in ("abort", "tooloffset_abort"):
            if accuracy_session and command[0] == "abort":
                accuracy_session["commands"].append({"seq": reader(reader_ctx).last_seq, "command": "abort", "params": command[1]})
                callback = getattr(self.events, "abort_pickaccuracy", None)
                if callback:
                    callback("用户放弃")
                self.events.emit("pickacc_state", phase="failed", message="用户放弃")
            elif offset_session:
                offset_session["commands"].append({"seq": reader(reader_ctx).last_seq, "command": command[0], "params": command[1]})
                offset_session["abort_requested"] = True
                self.events.emit("tooloffset_state", abort_requested=True, message="用户已点击放弃，下一个 PLC 请求将回 99 结束")
        status = super().on_running()
        if status == Status.RUNNING:
            return status
        if status == Status.FAILURE:
            if self.failure_since is None:
                self.failure_since = self.clock()
                self.events.emit("plc_disconnected", reason=self._blackboard.read("error"), operation="poll")
                self.events.plc("plc_disconnected", reason=self._blackboard.read("error"), failures=1)
            self.failures += 1
            return CallWorker.on_start(self)
        if self.failure_since is not None:
            self.events.emit("plc_restored", operation="poll", failures=self.failures, duration_s=self.clock() - self.failure_since)
            self.events.plc("plc_restored", failures=self.failures, duration_s=self.clock() - self.failure_since)
            self.failure_since = None
            self.failures = 0
        task, func, tool_id = self._blackboard.read("poll_result")
        previous = self.tracker.observed
        self.tracker.observed = task
        if task == 0 or task == previous or task == self.tracker.processed:
            return CallWorker.on_start(self)
        last = self.tracker.processed
        if last is not None and task != last + 1 and task != 1:
            logger.warning("Task_ID 跳变：%s → %s", last, task)
        self.tracker.processed = task
        self._blackboard.write("task_id", task, self.name)
        self._blackboard.write("request_func", func, self.name)
        self._blackboard.write("depth_request", False, self.name)
        self._blackboard.write("plc_tool_id", tool_id, self.name)
        self._blackboard.write("raw_path", None, self.name)
        self._blackboard.write("fine_render_data", None, self.name)
        self._blackboard.write("capture_id", None, self.name)
        self._blackboard.write("capture_sequence", None, self.name)
        self._blackboard.write("started", self.clock(), self.name)
        self._blackboard.write("error", f"功能码 {func} 在当前状态不允许", self.name)
        self._blackboard.write("error_type", None, self.name)
        self._blackboard.write("error_diagnostic", None, self.name)
        self._blackboard.write("reply_diagnostic", None, self.name)
        self.events.begin_request(task, func)
        self.events.plc("request", task_id=task, request_func=func, plc_tool_id=tool_id)
        self.events.emit("request", task_id=task, func=func, plc_tool_id=tool_id, name=REQUEST_NAMES.get(func, "未知功能码"))
        return Status.SUCCESS


class Predicate(Condition):
    """树中的无副作用条件叶，以黑板内容选择分支。"""

    def __init__(self, name: str, predicate: Callable[[Any], bool]) -> None:
        """保存谓词和可读节点名。"""
        super().__init__(name=name)
        self.predicate = predicate

    def check(self) -> bool:
        """读取本盘黑板并返回分支是否可执行。"""
        return self.predicate(self._blackboard)


class Instant(TreeNode):
    """同步业务叶基类：将可预期异常写入黑板供 999 兜底。"""

    def on_start(self) -> Status:
        """执行单次业务计算，异常转为可定位的分支失败。"""
        try:
            self.perform()
            return Status.SUCCESS
        except Exception as exc:
            self._blackboard.write("error", str(exc), self.name)
            self._blackboard.write("error_type", type(exc).__name__, self.name)
            self._blackboard.write("error_diagnostic", diagnostic(exc, where=self.name), self.name)
            return Status.FAILURE

    def on_running(self) -> Status:
        """单拍叶不会持续运行，满足框架叶接口。"""
        return Status.SUCCESS

    def perform(self) -> None:
        """由具体叶实现单项业务操作。"""
        raise NotImplementedError


class PrepareScan(Instant):
    """选择下一扫描位或本盘结束，并准备对应应答。"""

    def __init__(self, path: list[tuple[float, float, float]]) -> None:
        """保存预先守卫的扫描路径。"""
        super().__init__(name="准备扫描位或结束")
        self.path = path

    def perform(self) -> None:
        """把扫描位置和待应答数据写入本盘黑板。"""
        bb = self._blackboard
        index = bb.read("next_index")
        if index >= len(self.path):
            bb.write("reply_func", 9, self.name)
            bb.write("reply_coordinates", {}, self.name)
        else:
            scan = self.path[index]
            bb.write("scan", scan, self.name)
            bb.write("view_picks", 0, self.name)
            bb.write("next_index", index + 1, self.name)
            bb.write("reply_func", 1, self.name)
            bb.write("reply_coordinates", {"scan_xyz": scan}, self.name)
        bb.write("reply_tool", None, self.name)


class CheckArrival(CallWorker):
    """异步重复读取扫描位，等待稳定或超时，不阻塞行为树 tick。"""

    def __init__(self, board: Any, config: Config, events: EventRecorder, clock=time.monotonic) -> None:
        super().__init__(board, "plc", {"operation": "actual"}, "arrival_sample", name="读取实时坐标")
        self.config, self.events, self.clock = config, events, clock

    def on_start(self) -> Status:
        self.started = self.clock()
        self.samples = []
        self.pending = True
        return super().on_start()

    def on_running(self) -> Status:
        now = self.clock()
        if not self.pending:
            if now - self.last_read < self.config.arrival_settle["sample_interval_s"] and now - self.started < self.config.arrival_settle["settle_timeout_s"]:
                return Status.RUNNING
            if now - self.started >= self.config.arrival_settle["settle_timeout_s"]:
                return self.finish(False, now)
            self.pending = True
            return super().on_start()
        status = super().on_running()
        if status != Status.SUCCESS:
            return status
        now = self.clock()
        actual = self._blackboard.read("arrival_sample")
        try:
            valid = len(actual) == 3 and all(math.isfinite(v) for v in actual)
        except (TypeError, ValueError):
            valid = False
        if not valid:
            reason = "读取实时坐标无效"
            self._blackboard.write("error", reason, self.name)
            self._blackboard.write("error_diagnostic", diagnostic(reason, code="MOTION-ARRIVAL", where="扫描位到位校验", hint="检查实时坐标和通信。"), self.name)
            return Status.FAILURE
        self.samples.append((now - self.started, tuple(actual)))
        self.samples = self.samples[-40:]
        self.last_read = now
        self.pending = False
        n = self.config.arrival_settle["settle_samples"]
        window = self.samples[-n:]
        stable = len(window) == n and all(max(x[1][axis] for x in window) - min(x[1][axis] for x in window) <= self.config.arrival_settle["jitter_tol_mm"] for axis in range(3))
        if stable or now - self.started >= self.config.arrival_settle["settle_timeout_s"]:
            return self.finish(stable, now)
        return Status.RUNNING

    def finish(self, stable: bool, now: float) -> Status:
        bb = self._blackboard
        window = self.samples[-self.config.arrival_settle["settle_samples"]:]
        actual = tuple(sum(sample[1][i] for sample in window) / len(window) for i in range(3))
        expected = bb.read("scan")
        delta = tuple(a - e for a, e in zip(actual, expected))
        over = any(abs(v) > self.config.arrival_tolerance_mm for v in delta)
        bb.write("actual", actual, self.name)
        bb.write("arrival_ok", not over and stable, self.name)
        self.events.emit("arrival_warning" if over or not stable else "arrival", task_id=bb.read("task_id"),
                         stage=bb.read("work_action"), expected=expected, actual=actual, delta=delta,
                         reason=(f"扫描位{'未稳定、' if not stable else ''}{'超差' if over else '稳定等待超时'}；偏差 {delta} mm" if over or not stable else "到位稳定"),
                         where="扫描位到位监督",
                         tolerance_mm=self.config.arrival_tolerance_mm, stable=stable, over_tolerance=over,
                         settle_elapsed_s=now - self.started, samples=[[round(t * 1000, 3), *xyz] for t, xyz in self.samples],
                         ok=not over and stable)
        return Status.SUCCESS


class RecordCapture(Instant):
    """拍照后立即排队原图，检测失败仍可追溯。"""

    def __init__(self, events: EventRecorder) -> None:
        """保存记录器，供拍照与检测之间的叶节点使用。"""
        super().__init__(name="存档原图")
        self.events = events

    def perform(self) -> None:
        """记录原图相对路径并放入黑板，供错误应答引用。"""
        bb = self._blackboard
        archive = getattr(self.events, "archive", None)
        path = None
        if archive is not None:
            try:
                if self.events.run_directory is not None:
                    sequence, capture_id = bb.read("capture_sequence"), bb.read("capture_id")
                    if sequence is None or capture_id is None:
                        sequence, capture_id = self.events.next_capture()
                        if bb.read("work_action") == "fine":
                            bb.read("work_cycle").begin_fine(capture_id, bb.read("task_id"), bb.read("request_func"), self.events.xid)
                    cycle = bb.read("work_cycle")
                    stage = bb.read("work_action")
                    view = cycle.view_index + 1
                    shot = f"c{cycle.coarse_shots + 1:02d}" if stage == "coarse" else f"f{cycle.attempt_count + 1:02d}"
                    bb.write("capture_shot", shot, self.name)
                    tray = self.events.tray_id or "unknown"
                    stem = f"{sequence:06d}_{stage}_t{tray_number(tray)}_v{view:02d}_{shot}"
                    base = self.events.run_directory.relative_to(self.events.directory) / "images"
                    raw = str(base / f"{stem}_raw.png") if archive.enabled else None
                    overlay = str(base / f"{stem}_ovl.jpg") if archive.enabled else None
                    bb.write("capture_id", capture_id, self.name)
                    meta = dict(bb.read("capture_meta"))
                    meta["capture_seq"] = sequence
                    self.events.emit("capture", capture_id=capture_id, xid=self.events.xid, kind=stage, tray=tray,
                                     view=view, shot=shot, task_id=bb.read("task_id"),
                                     request_func=bb.read("request_func"), actual_xyz=bb.read("actual"),
                                     target_xyz=bb.read("scan"), raw_path=raw, ovl_path=overlay,
                                     queued_wall=self.events.clock().isoformat(), **meta)
                    path = archive.capture_raw(bb.read("image"), sequence=sequence, capture_id=capture_id,
                                               kind=stage, tray=tray, view=view, shot=shot,
                                               task_id=bb.read("task_id"))
                else:
                    path = archive.raw(bb.read("image"), bb.read("next_index"), bb.read("task_id"), bb.read("work_action"))
            except Exception as exc:
                record_image_error(self.events, f"{bb.read('work_action')}原图", exc, bb.read("task_id"))
        bb.write("raw_path", path, self.name)


class PrepareError(Instant):
    """将业务分支的失败原因转换为 999 应答并记录。"""

    def __init__(self, events: EventRecorder, unknown: bool = False) -> None:
        """区分未知功能码兜底与已知分支失败。"""
        super().__init__(name="准备未知功能码 999" if unknown else "准备业务错误 999")
        self.events = events
        self.unknown = unknown

    def perform(self) -> None:
        """保留失败上下文并准备空坐标的 999 应答。"""
        bb = self._blackboard
        func = bb.read("request_func")
        reason = f"未知功能码 {func}" if self.unknown else bb.read("error")
        cycle = bb.read("work_cycle")
        error_type = bb.read("error_type")
        if cycle is not None:
            cycle.fine_not_sent(str(reason))
        if not self.unknown and error_type != ProtocolViolation.__name__:
            state = cycle.state.value if cycle is not None else "IDLE"
            reason = f"当前状态 {state}，收到功能码 {func}，处理时出错：{error_type or '业务错误'}: {reason}"
        logger.error("Task_ID=%s 处理失败：%s", bb.read("task_id"), reason)
        self.events.emit(
            "error",
            task_id=bb.read("task_id"),
            request_func=func,
            reason=reason,
            scan_index=bb.read("next_index"),
            raw_path=bb.read("raw_path"),
        )
        details = (None if self.unknown else bb.read("error_diagnostic", None)) or diagnostic(reason, code="WORK-PROTOCOL" if self.unknown or error_type == ProtocolViolation.__name__ else "FLOW-ERROR", where="请求处理", hint="检查 PLC 功能码、当前状态和设备运行情况。")
        bb.write("reply_diagnostic", details, self.name)
        bb.write("reply_func", 999, self.name)
        bb.write("reply_coordinates", {}, self.name)
        bb.write("reply_tool", None, self.name)


class CommitReply(Instant):
    """应答提交成功后记录结果，并更新本盘上次应答状态。"""

    def __init__(self, events: EventRecorder, ctx: Any = None) -> None:
        """保存事件记录器。"""
        super().__init__(name="记录已提交应答")
        self.events = events
        self.ctx = ctx

    def perform(self) -> None:
        """只有 PLC 确认写入后才记录应答和盘结束。"""
        bb = self._blackboard
        func = bb.read("committed_func")
        if func is None:
            func = bb.read("reply_func")
        if func != bb.read("reply_func"):
            if bb.read("mode") == Mode.PICK_ACCURACY:
                from .pick_accuracy import finish as finish_pickaccuracy
                finish_pickaccuracy(self.ctx, bb, "aborted", reason="标定应答未提交")
            if bb.read("mode") == Mode.TOOL_OFFSET:
                from .tool_offset import finish as finish_tooloffset
                finish_tooloffset(self.ctx, bb, "aborted", reason="标定应答未提交")
            if bb.read("mode") == Mode.CAMERA_CALIBRATING:
                from .camera_calibration import finish
                finish(self.ctx, bb, "aborted", reason="标定应答未提交")
            bb.write("reply_coordinates", {}, self.name)
        current = bb.read("mode")
        target = None
        if func == 1 and current == Mode.IDLE:
            target = Mode.PICKING
        if target is not None:
            close_for_mode(self.events, target.value, bb.read('task_id'), bb.read('request_func'))
            bb.write("mode", transition(current, target), self.name)
            self.events.emit(
                "mode_change", previous_mode=current.value, mode=target.value,
                task_id=bb.read("task_id"), request_func=bb.read("request_func"),
                scan_index=bb.read("next_index"),
            )
        if func in (1, 3):
            bb.write("last_action", func, self.name)
        cycle = bb.read("work_cycle")
        if func == 3 and cycle is not None:
            cycle.mark_reply(bb.read("task_id"), getattr(self.events, "xid", 0), bb.read("reply_coordinates"))
            if getattr(cycle, "film_mode", False):
                self.events.emit("work_cycle", film_mode=True, tray_id=cycle.tray, view_index=cycle.view_index,
                                 sent_count=sum(bool(a["sent"]) for a in cycle.attempts_ledger),
                                 failed_count=sum(a["outcome"] == "failed" for a in cycle.attempts_ledger))
        elif cycle is not None and bb.read("reply_func") == 3:
            cycle.fine_not_sent("挑取应答未提交")
        details = {}
        if func in (999, 131, 99, 82):
            actual = getattr(self.events, "reply_diagnostics", {}).pop(bb.read("task_id"), None)
            details = actual or bb.read("reply_diagnostic", None) or diagnostic(bb.read("error"), code="REPLY-ERROR", where="应答提交")
        self.events.emit(
            "reply",
            task_id=bb.read("task_id"),
            request_func=bb.read("request_func"),
            func=func,
            coordinates=bb.read("reply_coordinates"),
            tool=bb.read("reply_tool"),
            scan_index=bb.read("next_index"),
            raw_path=bb.read("raw_path"),
            elapsed_ms=round((time.monotonic() - bb.read("started")) * 1000, 3),
            **details,
        )
        committed_depth = getattr(self.events, "depth_reply_committed", None)
        if bb.read("depth_request", False) and committed_depth:
            committed_depth(func, details)
        self.events.request_pending = False
        render_data = bb.read("fine_render_data")
        archive = getattr(self.events, "archive", None)
        if render_data and archive:
            try:
                from dataclasses import replace
                from nest_clear.vision.hair_integration import render_fine
                info = dict(render_data["info"])
                coordinates = bb.read("reply_coordinates") or {}
                info.update(tool1=coordinates.get("pick1_xyz"), tool2=coordinates.get("pick2_xyz"))
                if func != 3 and render_data["result"].status == "ok":
                    info["not_sent_reason"] = "reply_failed"
                result = render_data["result"] if func == 3 else replace(render_data["result"], pick_uv=None, target_xyz=None)
                overlay = render_fine(render_data["image"], render_data["observations"], result,
                                      render_data["pose"], render_data["calibration"], render_data["half_mm"],
                                      info=info, coarse_uv=render_data["coarse_uv"])
                archive.rendered(overlay, render_data["scan"], bb.read("task_id"), render_data["raw_path"],
                                 render_data["preview"], capture_id=render_data["capture_id"])
            except Exception as exc:
                record_image_error(self.events, "精定位已提交应答叠加图", exc, bb.read("task_id"))
            bb.write("fine_render_data", None, self.name)
        if func in (9, 10) or (func == 999 and cycle is None and not bb.read("depth_request", False)):
            from nest_clear.film_mode import end_tray
            try:
                end_tray(self.events.directory)
            except (OSError, AttributeError):
                logger.exception("清理菲林盘状态失败")
        if func in (9, 10):
            self.events.emit("tray_end", task_id=bb.read("task_id"), scans=bb.read("next_index"))
