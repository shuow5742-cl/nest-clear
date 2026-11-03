"""用行为树节点编排一盘 PLC 请求和 Worker 操作。"""
import logging
import math
import time
from typing import Any

from autoweaver.motion_policy.batch import Batch
from autoweaver.motion_policy.nodes.node import Status, TreeNode
from autoweaver.worker.base import next_request_id

from nest_clear.config import Config
from nest_clear.guard import GuardError
from nest_clear.plan.scan_path import guard_position
from nest_clear.record.events import EventRecorder

logger = logging.getLogger("nest_clear.flow")
REQUEST_NAMES = {1: "扫描到位或请求首个扫描位", 60: "挑取完成下一步"}


class RequestTracker:
    """跨盘保存已观察及已处理编号，避免批次切换重复接收。"""
    def __init__(self, baseline: int) -> None:
        """以启动时的 PLC 编号作为忽略基线。"""
        self.observed = baseline
        self.processed = None


class TrayTree(TreeNode):
    """按请求驱动有限状态行为树；所有设备操作通过 note/state 完成。"""
    def __init__(self, board: Any, config: Config, path: list[tuple[float, float, float]], tracker: RequestTracker, events: EventRecorder) -> None:
        """创建一盘私有流程状态和可扩展的功能码分发表。"""
        super().__init__(name="一盘流程")
        self.board = board
        self.config = config
        self.path = path
        self.tracker = tracker
        self.events = events
        self.dispatch = {1: self._request_scan, 60: self._request_next}
        self.phase = "poll"
        self.rid = 0
        self.next_index = 0
        self.scan = None
        self.targets = []
        self.tool = 0
        self.task_id = 0
        self.request_func = 0
        self.started = 0.0
        self.reply_func = 0
        self.last_action = 0
        self.reply_coordinates = {}
        self.reply_tool = None
        self.pose = None

    def _send(self, worker: str, payload: dict[str, Any], phase: str) -> None:
        """发送编号 note；每次只允许一个操作在途，保证状态与请求配对。"""
        self.rid = next_request_id()
        self.worker = worker
        self.phase = phase
        self.board.pass_note(namespace=worker, name="execute", payload={**payload, "__request_id__": self.rid}, sender=self.name)

    def _result(self) -> tuple[bool, Any, str]:
        """检查本次完成编号和结果错误，不能仅凭完成编号判断成功。"""
        result = self.snapshot.get(f"{self.worker}.result", {})
        if result.get("id") != self.rid or self.snapshot.get(f"{self.worker}.last_completed_id", 0) < self.rid:
            return False, None, ""
        return True, result.get("value"), result.get("error", "")

    def _reply(self, func: int, coordinates: dict[str, Any] | None = None, tool: int | None = None) -> None:
        """通过 PLC Worker 按握手提交顺序异步应答。"""
        self.reply_func = func
        self.reply_coordinates = coordinates or {}
        self.reply_tool = tool
        self._send("plc", {"operation": "reply", "task_id": self.task_id, "func": func, "coordinates": self.reply_coordinates}, "reply")

    def _error(self, reason: str) -> None:
        """记录请求上下文并发送 999，让 PLC 得知可恢复的业务错误。"""
        logger.error("Task_ID=%s 处理失败：%s", self.task_id, reason)
        self.events.emit("error", task_id=self.task_id, request_func=self.request_func, reason=reason)
        self._reply(999)

    def _advance(self) -> None:
        """目标耗尽后下发下一扫描位，路径耗尽则结束本盘。"""
        if self.next_index >= len(self.path):
            self._reply(9)
            return
        self.scan = self.path[self.next_index]
        self.next_index += 1
        self._reply(1, {"scan_xyz": self.scan})

    def _pick(self) -> None:
        """根据第八节工具偏移计算挑取坐标，行程合格才下发。"""
        target = self.targets[0]
        offset = self.config.tools[self.tool]
        xyz = (target.xyz[0] + offset[0], target.xyz[1] + offset[1], self.config.pick_depth_z + offset[2])
        guard_position(xyz, self.config.travel, f"Task_ID={self.task_id}，扫描位 {self.next_index}")
        self.targets.pop(0)
        tool = self.tool + 1
        self.tool = 1 - self.tool
        self._reply(3, {f"pick{tool}_xyz": xyz}, tool)

    def _request_scan(self) -> None:
        """首次请求只下发扫描位；后续请求先校验到位再拍照。"""
        if self.scan is None:
            self._advance()
        elif self.last_action != 1:
            self._error("当前未等待扫描到位，不能处理功能码 1")
        else:
            self._send("plc", {"operation": "actual"}, "actual")

    def _request_next(self) -> None:
        """挑取完成后消费已检测目标，不重新拍照。"""
        if self.last_action != 3:
            self._error("当前未等待挑取完成，不能处理功能码 60")
        elif self.targets:
            self._pick()
        else:
            self._advance()

    def on_start(self) -> Status:
        """首次 tick 开始轮询请求。"""
        self._send("plc", {"operation": "request"}, "poll")
        return Status.RUNNING

    def on_running(self) -> Status:
        """推进一个非阻塞状态转换，失败时发送带原因的 999。"""
        ready, value, error = self._result()
        if not ready:
            return Status.RUNNING
        if error:
            if self.phase in ("poll", "reply"):
                logger.error("PLC 通信失败：%s", error)
                self.events.emit("error", task_id=self.task_id, reason=error)
                return Status.FAILURE
            self._error(error)
            return Status.RUNNING
        try:
            if self.phase == "poll":
                task, func = value
                previous = self.tracker.observed
                self.tracker.observed = task
                if task == 0 or task == previous or task == self.tracker.processed:
                    self._send("plc", {"operation": "request"}, "poll")
                    return Status.RUNNING
                if self.tracker.processed is not None and task != self.tracker.processed + 1 and task != 1:
                    logger.warning("Task_ID 跳变：%s → %s", self.tracker.processed, task)
                self.tracker.processed = task
                self.task_id = task
                self.request_func = func
                self.started = time.monotonic()
                self.events.emit("request", task_id=task, func=func, name=REQUEST_NAMES.get(func, "未知功能码"))
                handler = self.dispatch.get(func)
                if handler is None:
                    self._error(f"未知功能码 {func}")
                else:
                    handler()
            elif self.phase == "actual":
                self.pose = tuple(value)
                ok = len(self.pose) == 3 and all(math.isfinite(v) and abs(v - expected) <= self.config.arrival_tolerance_mm for v, expected in zip(self.pose, self.scan))
                self.events.emit("arrival", task_id=self.task_id, expected=self.scan, actual=self.pose, ok=ok)
                if not ok:
                    self._error(f"到位超差：期望 {self.scan}，实际 {self.pose}，公差 {self.config.arrival_tolerance_mm} mm")
                else:
                    self._send("camera", {}, "camera")
            elif self.phase == "camera":
                self._send("vision", {"image": value, "pose": self.pose}, "vision")
            elif self.phase == "vision":
                self.targets = list(value)
                self.events.emit("detection", task_id=self.task_id, count=len(self.targets), scan_index=self.next_index)
                if self.targets:
                    self._pick()
                else:
                    self._advance()
            elif self.phase == "reply":
                if self.reply_func in (1, 3):
                    self.last_action = self.reply_func
                self.events.emit("reply", task_id=self.task_id, request_func=self.request_func, func=self.reply_func, coordinates=self.reply_coordinates, tool=self.reply_tool, elapsed_ms=round((time.monotonic() - self.started) * 1000, 3))
                if self.reply_func == 9:
                    self.events.emit("tray_end", task_id=self.task_id, scans=self.next_index)
                    return Status.SUCCESS
                self._send("plc", {"operation": "request"}, "poll")
        except Exception as exc:
            self._error(str(exc))
        return Status.RUNNING


def build_batch(board: Any, config: Config, path: list[tuple[float, float, float]], tracker: RequestTracker, events: EventRecorder) -> Batch:
    """为新盘构造全新行为树和独立 Batch 状态。"""
    return Batch(lambda: TrayTree(board, config, path, tracker, events), name="一盘")
