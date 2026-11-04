"""对刀功能码的占位子树；实际对刀流程留待后续实现。"""

from autoweaver.motion_policy.nodes.node import TreeNode

from .context import FlowContext
from .leaves import Instant


class CalibrationUnavailable(Instant):
    """说明尚未实现，并交给公共错误出口准备应答。"""

    def __init__(self) -> None:
        """为占位节点设置可读名称。"""
        super().__init__(name="对刀功能尚未实现")

    def perform(self) -> None:
        """保存明确原因，使公共错误出口返回 999。"""
        self._blackboard.write("error", "对刀功能尚未实现", self.name)
        from nest_clear.guard import GuardError

        raise GuardError("CALIBRATION-UNAVAILABLE", "对刀功能尚未实现", "对刀", "等待对刀功能实现。")


def calibration_placeholder(ctx: FlowContext) -> TreeNode:
    """构造对刀占位节点，接口与其他分支工厂一致。"""
    return CalibrationUnavailable()
