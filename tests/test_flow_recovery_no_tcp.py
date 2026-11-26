"""不打开 TCP，回归真实行为树在拒绝请求后的恢复。"""
from dataclasses import replace
import time
from typing import Any

import numpy as np
from autoweaver.motion_policy.world_board import WorldBoard
from autoweaver.worker.clock import BTClock
from autoweaver.worker import AsyncRequestWorker

from nest_clear.config import load_config
from nest_clear.flow.batch import build_batch
from nest_clear.flow.context import RequestTracker
from nest_clear.flow.work_cycle import State
from nest_clear.record.events import EventRecorder
from nest_clear.vision.geometry import Calibration


class MemoryPLC(AsyncRequestWorker):
    """以内存保存请求、坐标和应答，使用与真实 PLC 相同的异步 Worker 接口。"""
    name = "plc"

    def __init__(self) -> None:
        """创建空请求及应答表。"""
        super().__init__()
        self.request = (0, 0, None)
        self.actual = (0.0, 0.0, 0.0)
        self.replies: dict[int, tuple[int, dict[str, Any]]] = {}

    def execute(self, payload: dict[str, Any]) -> Any:
        """返回当前请求或到位坐标，记录已提交应答。"""
        operation = payload["operation"]
        if operation == "request":
            return self.request
        if operation == "actual":
            return self.actual
        if operation == "reply":
            self.replies[payload["task_id"]] = (payload["func"], payload["coordinates"])
            return None
        raise AssertionError(operation)


class MemoryCamera(AsyncRequestWorker):
    """提供固定图像，不依赖设备和随机数据。"""
    name = "camera"

    def execute(self, payload: dict[str, Any]) -> np.ndarray:
        """返回确定性的空图。"""
        return np.zeros((8, 8, 3), dtype=np.uint8)


class EmptyVision(AsyncRequestWorker):
    """提供空粗拍结果，单独验证分发与工作状态。"""
    name = "vision"

    def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        """确认粗拍请求已经到达视觉 Worker。"""
        assert payload["work_action"] == "coarse"
        return {"observations": []}


def test_rejected_request_then_valid_arrival(tmp_path: Any) -> None:
    """100 被拒绝后，新编号的 20 必须经过到位、拍照并得到应答。"""
    config = load_config("config/nest_clear.yaml")
    config = replace(config, vision=replace(config.vision, detector="stub", pixels=()))
    calibration = Calibration(np.eye(3), np.zeros(5), np.eye(3), np.zeros(3), 1)
    board = WorldBoard()
    clock = BTClock(board)
    plc = MemoryPLC()
    for worker in (plc, MemoryCamera(), EmptyVision()):
        clock.attach_worker(worker)
    tracker = RequestTracker(0)
    events = EventRecorder(tmp_path, "main")
    batch = build_batch(board, config, [(0.0, 0.0, 0.0)], tracker, events, calibration=calibration)
    handle = clock.submit(batch)

    def request(task: int, func: int) -> tuple[int, dict[str, Any]]:
        """持续推进真实时钟，直到对应编号得到应答。"""
        plc.request = (task, func, 2 if func in (100, 110) else None)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            clock.tick_once()
            if task in plc.replies:
                return plc.replies[task]
            assert handle.result is None
            time.sleep(0.001)
        raise AssertionError(f"任务 {task} 未得到应答")

    try:
        assert request(1, 1)[0] == 1
        cycle = batch.blackboard.read("work_cycle")
        assert cycle.state == State.WAIT_COARSE
        assert request(2, 100)[0] == 999
        assert cycle.state == State.WAIT_COARSE
        func, data = request(3, 20)
        assert func == 9
        assert data["removable_count"] == 0
        assert cycle.coarse_shots == 1
        assert cycle.state == State.DONE
        assert tracker.processed == 3
    finally:
        clock.shutdown()
