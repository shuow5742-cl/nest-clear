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
from nest_clear.plan.scan_path import guard_position
from nest_clear.record.events import EventRecorder

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
    """循环异步读取请求；短时断线重试，持续超时才让整盘失败。"""

    def __init__(self, board: Any, tracker: Any, timeout: float, events: EventRecorder) -> None:
        """保留跨盘基线和通信失败预算。"""
        super().__init__(
            board,
            "plc",
            lambda bb: {"operation": "request"},
            result_key="poll_result",
            name="等待新请求",
        )
        self.tracker = tracker
        self.timeout = timeout
        self.events = events

    def on_start(self) -> Status:
        """开始本次等待；失败起点保存在本盘黑板。"""
        self._blackboard.write("poll_failure_since", None, self.name)
        return super().on_start()

    def on_running(self) -> Status:
        """处理轮询结果、编号变化及持续失败超时。"""
        status = super().on_running()
        if status == Status.RUNNING:
            since = self._blackboard.read("poll_failure_since")
            if since is not None and time.monotonic() - since >= self.timeout:
                reason = self._blackboard.read("error")
                logger.error("PLC 连续轮询失败超过 %.3f 秒：%s", self.timeout, reason)
                self.events.emit("error", reason=reason, operation="poll")
                return Status.FAILURE
            return status
        if status == Status.FAILURE:
            now = time.monotonic()
            since = self._blackboard.read("poll_failure_since")
            if since is None:
                since = now
                self._blackboard.write("poll_failure_since", since, self.name)
            if now - since >= self.timeout:
                reason = self._blackboard.read("error")
                logger.error("PLC 连续轮询失败超过 %.3f 秒：%s", self.timeout, reason)
                self.events.emit("error", reason=reason, operation="poll")
                return Status.FAILURE
            return CallWorker.on_start(self)
        self._blackboard.write("poll_failure_since", None, self.name)
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
        self._blackboard.write("started", time.monotonic(), self.name)
        self._blackboard.write("error", f"功能码 {func} 在当前状态不允许", self.name)
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
            path = archive.raw(bb.read("image"), bb.read("next_index"), bb.read("task_id"))
        bb.write("raw_path", path, self.name)


class RecordDetection(Instant):
    """检测完成后记录目标数，供后续分支选择挑取或继续扫描。"""

    def __init__(self, events: EventRecorder) -> None:
        """保存事件记录器。"""
        super().__init__(name="记录检测目标数")
        self.events = events

    def perform(self) -> None:
        """记录本次结果数量及扫描序号。"""
        bb = self._blackboard
        archive = getattr(self.events, "archive", None)
        if archive is not None:
            archive.overlay(
                bb.read("image"),
                bb.read("targets"),
                bb.read("next_index"),
                bb.read("task_id"),
                bb.read("raw_path"),
            )
        self.events.emit(
            "detection",
            task_id=bb.read("task_id"),
            count=len(bb.read("targets")),
            scan_index=bb.read("next_index"),
        )


class PreparePick(Instant):
    """按工具补偿计算挑取坐标，守卫通过后准备 3 应答。"""

    def __init__(self, config: Config) -> None:
        """保存工具、深度和行程配置。"""
        super().__init__(name="计算并守卫挑取坐标")
        self.config = config

    def perform(self) -> None:
        """只在坐标合法后消费目标并切换工具。"""
        bb = self._blackboard
        targets = bb.read("targets")
        target = targets[0]
        tool_index = bb.read("tool_index")
        offset = self.config.tools[tool_index]
        xyz = (
            target.xyz[0] + offset[0],
            target.xyz[1] + offset[1],
            self.config.pick_depth_z + offset[2],
        )
        where = f"Task_ID={bb.read('task_id')}，扫描位 {bb.read('next_index')}"
        guard_position(xyz, self.config.travel, where)
        tool = tool_index + 1
        bb.write("targets", targets[1:], self.name)
        bb.write("tool_index", 1 - tool_index, self.name)
        bb.write("reply_func", 3, self.name)
        bb.write("reply_coordinates", {f"pick{tool}_xyz": xyz}, self.name)
        bb.write("reply_tool", tool, self.name)


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
        func = bb.read("reply_func")
        if func in (1, 3):
            bb.write("last_action", func, self.name)
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
        if func == 9:
            bb.write("done", True, self.name)
            self.events.emit("tray_end", task_id=bb.read("task_id"), scans=bb.read("next_index"))
