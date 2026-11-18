"""一盘行为树的单一职责叶节点及 Worker 调用协议。"""

from collections.abc import Callable
import logging
import math
import time
from typing import Any

from autoweaver.motion_policy import CallWorker as FrameworkCallWorker
from autoweaver.motion_policy.nodes.leaf.condition import Condition
from autoweaver.motion_policy.nodes.node import Status, TreeNode

from nest_clear.config import Config
from nest_clear.guard import GuardError
from nest_clear.vision.hair_integration import render
from nest_clear.plan.scan_path import guard_position
from nest_clear.record.events import EventRecorder
from nest_clear.record.images import record_image_error, tray_number
from .work_cycle import ProtocolViolation
from .modes import Mode, transition

logger = logging.getLogger("nest_clear.flow")
REQUEST_NAMES = {1: "扫描到位或请求首个扫描位", 60: "挑取完成下一步"}


class CallWorker(FrameworkCallWorker):
    """保留设备调用失败的任务日志，结果关联交由框架处理。"""

    def on_running(self) -> Status:
        """记录框架报告的失败，供现场定位请求。"""
        status = super().on_running()
        if status == Status.FAILURE:
            logger.warning("%s失败，Task_ID=%s：%s", self.name, self._blackboard.read("task_id"), self._blackboard.read("error"))
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
        task, func = self._blackboard.read("poll_result")
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
        self._blackboard.write("raw_path", None, self.name)
        self._blackboard.write("fine_render_data", None, self.name)
        self._blackboard.write("capture_id", None, self.name)
        self._blackboard.write("capture_sequence", None, self.name)
        self._blackboard.write("started", self.clock(), self.name)
        self._blackboard.write("error", f"功能码 {func} 在当前状态不允许", self.name)
        self._blackboard.write("error_type", None, self.name)
        self.events.begin_request(task, func)
        self.events.plc("request", task_id=task, request_func=func)
        self.events.emit("request", task_id=task, func=func, name=REQUEST_NAMES.get(func, "未知功能码"))
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


class CheckArrival(Instant):
    """计算并记录扫描位到位结果，供紧随其后的 Condition 判断。"""

    def __init__(self, config: Config, events: EventRecorder) -> None:
        """绑定到位公差和事件记录器。"""
        super().__init__(name="记录到位校验")
        self.config = config
        self.events = events

    def perform(self) -> None:
        """按轴比较实测与目标；无效坐标也作为到位失败记录。"""
        bb = self._blackboard
        actual = bb.read("actual")
        expected = bb.read("scan")
        ok = isinstance(actual, (list, tuple)) and len(actual) == 3
        if ok:
            ok = all(
                math.isfinite(value) and abs(value - target) <= self.config.arrival_tolerance_mm
                for value, target in zip(actual, expected)
            )
        bb.write("arrival_ok", ok, self.name)
        self.events.emit("arrival", task_id=bb.read("task_id"), expected=expected, actual=actual, ok=ok)
        if not ok:
            reason = f"到位超差：期望 {expected}，实际 {actual}，公差 {self.config.arrival_tolerance_mm} mm"
            bb.write("error", reason, self.name)


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
        if error_type != ProtocolViolation.__name__:
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
        bb.write("reply_func", 999, self.name)
        bb.write("reply_coordinates", {}, self.name)
        bb.write("reply_tool", None, self.name)


class CommitReply(Instant):
    """应答提交成功后记录结果，并更新本盘上次应答状态。"""

    def __init__(self, events: EventRecorder) -> None:
        """保存事件记录器。"""
        super().__init__(name="记录已提交应答")
        self.events = events

    def perform(self) -> None:
        """只有 PLC 确认写入后才记录应答和盘结束。"""
        bb = self._blackboard
        func = bb.read("committed_func")
        if func is None:
            func = bb.read("reply_func")
        if func != bb.read("reply_func"):
            bb.write("reply_coordinates", {}, self.name)
        current = bb.read("mode")
        target = None
        if func == 1 and current == Mode.IDLE:
            target = Mode.PICKING
        if target is not None:
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
        elif cycle is not None and bb.read("reply_func") == 3:
            cycle.fine_not_sent("挑取应答未提交")
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
        )
        render_data = bb.read("fine_render_data")
        archive = getattr(self.events, "archive", None)
        if render_data and archive:
            try:
                from dataclasses import replace
                from nest_clear.vision.hair_integration import render_fine
                info = dict(render_data["info"])
                coordinates = bb.read("reply_coordinates") or {}
                info.update(tool1=coordinates.get("pick1_xyz"), tool2=coordinates.get("pick2_xyz"))
                result = render_data["result"] if func == 3 else replace(render_data["result"], pick_uv=None, target_xyz=None)
                overlay = render_fine(render_data["image"], render_data["observations"], result,
                                      render_data["pose"], render_data["calibration"], render_data["half_mm"],
                                      info=info, coarse_uv=render_data["coarse_uv"])
                archive.rendered(overlay, render_data["scan"], bb.read("task_id"), render_data["raw_path"],
                                 render_data["preview"], capture_id=render_data["capture_id"])
            except Exception as exc:
                record_image_error(self.events, "精定位已提交应答叠加图", exc, bb.read("task_id"))
            bb.write("fine_render_data", None, self.name)
        if func in (9, 10):
            self.events.emit("tray_end", task_id=bb.read("task_id"), scans=bb.read("next_index"))
