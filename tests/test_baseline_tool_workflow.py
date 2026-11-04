"""以真实 TCP 仿真 PLC 验证建基准工具的各条操作路径。"""

import ast
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml
from pymodbus.client import ModbusTcpClient

from nest_clear.tools.tooltip_baseline import main
from nest_clear.tooltip.store import load_baseline, load_tools
from sim.fake_plc import FakePLC


MATRICES = {
    "front": [[20, 0, 0], [0, 0, 20]],
    "side": [[0, 20, 0], [0, 0, 20]],
}
POSITIONS = [
    [1.2, 2, 3],
    [0.8, 2, 3],
    [1, 2.2, 3],
    [1, 1.8, 3],
    [1, 2, 3.2],
    [1, 2, 2.8],
]


@pytest.fixture
def setup_tool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """隔离配置与数据，模拟 HMI 点动并在每条路径后断言零客户端写入。"""
    with FakePLC() as plc:
        app = yaml.safe_load(Path("config/nest_clear.yaml").read_text(encoding="utf-8"))
        app["plc"].update(ip=plc.host, port=plc.port)
        app["record"]["directory"] = str(tmp_path / "runs")
        app_path = tmp_path / "app.yaml"
        app_path.write_text(yaml.safe_dump(app), encoding="utf-8")
        config = yaml.safe_load(Path("config/tooltip_baseline.yaml").read_text(encoding="utf-8"))
        for name, matrix in MATRICES.items():
            config["cameras"][name] = {
                "backend": "mock",
                "size": [480, 640],
                "target_uv": [320, 240],
                "p0": [1, 2, 3],
                "jacobian": matrix,
            }
        config_path = tmp_path / "baseline.yaml"
        calibration = tmp_path / "calibration"
        shutil.copytree("config/calibration", calibration)
        plc.set_actual_xyz([1, 2, 3])
        args = [
            "--tool",
            "1",
            "--app-config",
            str(app_path),
            "--config",
            str(config_path),
            "--calibration-dir",
            str(calibration),
            "--offset",
            "1",
            "2",
            "3",
            "--note",
            "test",
        ]

        def run(positions: list = POSITIONS, extra: list | None = None, reset_answer: str = "") -> int:
            """同步注入每步位姿及确认答案，让被测工具执行完整交互。"""
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            steps = iter(positions)

            def answer(prompt: str) -> str:
                """通过 PLC 内部数据区模拟 HMI，而非发送 Modbus 写请求。"""
                if "HMI" in prompt:
                    plc.set_actual_xyz(next(steps))
                    return ""
                return reset_answer

            monkeypatch.setattr("builtins.input", answer)
            return main(args + (extra or []))

        yield plc, config, calibration, tmp_path / "runs", run
        assert plc.writes == [], f"工具发送了客户端写入：{plc.writes}"


def test_full_workflow_jacobians_and_p0(setup_tool) -> None:
    """完整采样拟合两个真值矩阵，验证 P0 和原图位姿归档。"""
    plc, config, calibration, runs, run = setup_tool
    assert run(extra=["--no-reset"]) == 0
    baseline = load_baseline(calibration / "tool1_baseline.yaml")
    assert baseline.p0 == pytest.approx([1, 2, 3])
    for view in baseline.views:
        assert np.allclose(view.jacobian, MATRICES[view.name], atol=1e-4)
    directory = next(runs.glob("*/tooltip_sessions/*"))
    session = json.loads((directory / "session.json").read_text(encoding="utf-8"))
    assert len(session["frames"]) == 7
    assert len(list(directory.glob("*.png"))) == 14


def test_front_center_warning_does_not_block(setup_tool, capsys) -> None:
    """正面偏离中心应提示但不影响建立基准。"""
    plc, config, calibration, runs, run = setup_tool
    config["cameras"]["front"]["target_uv"] = [180, 240]
    assert run(extra=["--no-reset"]) == 0
    assert "警告：正面针尖离中心" in capsys.readouterr().out
    assert load_baseline(calibration / "tool1_baseline.yaml").p0 == pytest.approx([1, 2, 3])


def test_insufficient_samples_fail_with_advice(setup_tool, capsys) -> None:
    """采样步数不足时拒绝保存并给出独立方向建议。"""
    plc, config, calibration, runs, run = setup_tool
    assert run(extra=["--steps", "2", "--no-reset"]) == 1
    output = capsys.readouterr().out
    assert "采样步数不足" in output and "三个线性独立" in output
    assert "占位" in (calibration / "tool1_baseline.yaml").read_text(encoding="utf-8")


def test_collinear_samples_fail_with_advice(setup_tool, capsys) -> None:
    """共线点动不能拟合三轴，失败时应提示补采独立方向。"""
    plc, config, calibration, runs, run = setup_tool
    positions = [[1.2, 2, 3], [0.8, 2, 3], [1.4, 2, 3]]
    assert run(positions, ["--steps", "3", "--no-reset"]) == 1
    assert "至少采集三个独立方向" in capsys.readouterr().out
    assert "占位" in (calibration / "tool1_baseline.yaml").read_text(encoding="utf-8")


def test_verify_platform_displacement_delta(setup_tool, capsys) -> None:
    """平台偏移后验证三个修正轴的数值和符号。"""
    plc, config, calibration, runs, run = setup_tool
    assert run(extra=["--no-reset"]) == 0
    capsys.readouterr()
    plc.set_actual_xyz([1.2, 1.8, 3.2])
    assert run(extra=["--verify"]) == 0
    output = capsys.readouterr().out
    delta_text = output.split("修正 Δ=", 1)[1].split(" mm", 1)[0]
    assert ast.literal_eval(delta_text) == pytest.approx([-0.2, 0.2, -0.2], abs=1e-4)
    assert "质量=" in output


def test_replay_read_only_and_save(setup_tool, capsys) -> None:
    """回放复现基准且默认不写配置；显式保存才产生备份和历史。"""
    plc, config, calibration, runs, run = setup_tool
    assert run(extra=["--no-reset"]) == 0
    original = load_baseline(calibration / "tool1_baseline.yaml")
    directory = next(runs.glob("*/tooltip_sessions/*"))
    before = {path.name: path.read_bytes() for path in calibration.iterdir()}
    assert run(extra=["--replay", str(directory)]) == 0
    assert "重算 P0=" in capsys.readouterr().out
    assert {path.name: path.read_bytes() for path in calibration.iterdir()} == before
    assert run(extra=["--replay", str(directory), "--save"]) == 0
    assert load_baseline(calibration / "tool1_baseline.yaml") == original
    assert len(list(calibration.glob("tool1_baseline.yaml.bak.*"))) == 2
    history = (calibration / "tools_history.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(history[-1])["source"] == "replay"


def test_no_reset_preserves_current_offsets(setup_tool) -> None:
    """禁止重置时，当前工具补偿文件及其历史保持不变。"""
    plc, config, calibration, runs, run = setup_tool
    before = (calibration / "tools.yaml").read_bytes()
    assert run(extra=["--no-reset"]) == 0
    assert (calibration / "tools.yaml").read_bytes() == before
    assert not list(calibration.glob("tools.yaml.bak.*"))


def test_reset_current_offsets(setup_tool) -> None:
    """默认确认重置时，只更新目标工具的补偿。"""
    plc, config, calibration, runs, run = setup_tool
    before = load_tools(calibration / "tools.yaml")["tools"]["tool2"]
    assert run() == 0
    tools = load_tools(calibration / "tools.yaml")["tools"]
    assert tools["tool1"]["offset_xyz"] == [1, 2, 3]
    assert tools["tool2"] == before
    assert list(calibration.glob("tools.yaml.bak.*"))


def test_fake_plc_distinguishes_client_writes() -> None:
    """正向验证记录器能发现外部写入，内部状态变化则不计入。"""
    with FakePLC() as plc:
        plc.set_status(3)
        plc.set_actual_xyz([1, 2, 3])
        plc.request(1)
        assert plc.writes == []
        plc.io.write("reply_func", 7)
        assert plc.writes == [(plc.contract["reply_func"].address, [7])]


def test_fake_plc_records_other_client_write_functions() -> None:
    """真实客户端的掩码、读写组合和线圈写请求也不能漏记。"""
    with FakePLC() as plc:
        client = ModbusTcpClient(plc.host, port=plc.port)
        try:
            assert client.connect()
            client.mask_write_register(address=10, and_mask=65535, or_mask=1)
            client.readwrite_registers(read_address=10, read_count=1, write_address=11, values=[42])
            client.write_coil(0, True)
            assert plc.writes == [(40011, [65535, 1]), (40012, [42]), (1, [True])]
        finally:
            client.close()
