"""验证对刀配置及补偿文件的启动预检。"""

from pathlib import Path
import yaml
import pytest

from nest_clear.config import load_config
from nest_clear.guard import GuardError

ROOT = Path(__file__).resolve().parents[1]


def test_missing_tools_fails_with_baseline_hint(tmp_path: Path) -> None:
    """缺少补偿文件时明确提示建立基准。"""
    data = yaml.safe_load((ROOT / "config/nest_clear.yaml").read_text())
    data["tool_calibration"]["directory"] = str(tmp_path)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(GuardError, match="tooltip_baseline"):
        load_config(path)


@pytest.mark.parametrize("field,value", [("max_iterations", 0), ("tip_tolerance_mm", -1), ("max_condition", 0.5)])
def test_bad_calibration_config(tmp_path: Path, field: str, value: object) -> None:
    """非法迭代次数、容差及质量阈值均在启动前拒绝。"""
    data = yaml.safe_load((ROOT / "config/nest_clear.yaml").read_text())
    data["tool_calibration"][field] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(GuardError, match="配置项"):
        load_config(path)
