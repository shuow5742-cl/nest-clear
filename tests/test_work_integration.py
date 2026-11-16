"""不依赖 TCP，验证状态机接线和握手数据提交顺序。"""
from dataclasses import replace

import numpy as np
import pytest

from nest_clear.comm import Handshake, load_contract
from nest_clear.config import load_config
from nest_clear.flow.context import FlowContext, RequestTracker
from nest_clear.flow.picking import FinishCapture, WorkRequest
from nest_clear.vision.fine import FineResult
from nest_clear.vision.geometry import Calibration
from nest_clear.vision.hair_integration import measure
from test_hair_integration import Board, Events, candidate


def test_work_adapter_both_tools_and_terminal():
    """粗拍、精拍与挑完通过同一状态机传递，终态继续拒绝请求。"""
    config = load_config("config/nest_clear.yaml")
    config = replace(config, tools=((1, 2, 3), (4, 5, 6)), travel=((-100, 100),) * 3)
    calibration = Calibration(np.eye(3), np.zeros(5), np.eye(3), np.zeros(3), 1)
    ctx = FlowContext(None, config, [(0, 0, 0)], RequestTracker(0), Events(), calibration)
    bb = Board(work_cycle=None, request_func=1, task_id=1)
    request = WorkRequest(ctx)
    request._blackboard = bb
    request.perform()
    assert bb.read("reply_coordinates")["scan_xyz"] == (0, 0, 0)
    bb.write("request_func", 20)
    request.perform()
    obs = measure(candidate(), (0, 0, 0), calibration, config.vision.hair_detector)
    bb.write("actual", (0, 0, 0))
    bb.write("detection", {"observations": [obs]})
    finish = FinishCapture(ctx)
    finish._blackboard = bb
    finish.perform()
    assert bb.read("reply_coordinates")["removable_count"] == 1
    request.perform()
    bb.write("detection", {"fine_result": FineResult("ok", target_xyz=(2, 3, 1))})
    finish.perform()
    assert bb.read("reply_func") == 3
    assert bb.read("reply_coordinates") == {"pick1_xyz": (3, 5, config.pick_depth_z + 3),
                                             "pick2_xyz": (6, 8, config.pick_depth_z + 6)}
    bb.write("request_func", 60)
    request.perform()
    bb.write("request_func", 20)
    request.perform()
    bb.write("detection", {"observations": []})
    finish.perform()
    assert bb.read("reply_func") == 9
    with pytest.raises(Exception, match="DONE"):
        request.perform()


def test_data_precedes_commit():
    """数量及两工具坐标均在功能码和提交编号之前写入。"""
    class IO:
        def __init__(self):
            self.writes = []
        def write_u16(self, address, value):
            self.writes.append(address)
        def write_real32_block(self, address, values):
            self.writes.append(address)
    contract = load_contract("config/plc_contract.yaml")
    io = IO()
    handshake = Handshake(io, contract)
    handshake.write_reply(1, 3, {"pick1_xyz": (1, 2, 3), "pick2_xyz": (4, 5, 6)})
    assert io.writes == [contract[name].address for name in ("pick1_x", "pick2_x", "reply_func", "result_id")]
    io.writes.clear()
    handshake.write_reply(2, 9, {"removable_count": 0})
    assert io.writes == [contract[name].address for name in ("removable_count", "reply_func", "result_id")]


def test_placeholder_rejects_pick_without_consuming_attempt():
    """占位标定时拒绝挑取坐标，状态与尝试次数保持不变。"""
    from nest_clear.guard import GuardError
    config = load_config("config/nest_clear.yaml")
    calibration = Calibration(np.eye(3), np.zeros(5), np.eye(3), np.zeros(3), 1)
    events = Events()
    events.calibration_placeholder = True
    ctx = FlowContext(None, config, [(0, 0, 0)], RequestTracker(0), events, calibration)
    bb = Board(work_cycle=None, request_func=1, task_id=1)
    request = WorkRequest(ctx)
    request._blackboard = bb
    request.perform()
    cycle = bb.read("work_cycle")
    observation = measure(candidate(), (0, 0, 0), calibration, config.vision.hair_detector)
    cycle.on_coarse((0, 0, 0), [observation])
    with pytest.raises(GuardError, match="占位"):
        cycle.on_fine((0, 0, 0), FineResult("ok", target_xyz=(1, 1, 1)))
    assert cycle.attempt_count == 0
    assert cycle.state.value == "WAIT_FINE"
