"""行为树装配所需的共享依赖和请求基线。"""

from dataclasses import dataclass
from typing import Any

from nest_clear.config import Config
from nest_clear.record.events import EventRecorder


class RequestTracker:
    """仅跨盘保留 PLC 观察基线，防止重复接收上一盘请求。"""

    def __init__(self, baseline: int) -> None:
        """启动时忽略当前编号；盘内业务状态放在 Blackboard。"""
        self.observed = baseline
        self.processed = None


@dataclass(frozen=True)
class FlowContext:
    """构树依赖；不保存当前盘的可变业务状态。"""

    board: Any
    config: Config
    path: list[tuple[float, float, float]]
    tracker: RequestTracker
    events: EventRecorder


