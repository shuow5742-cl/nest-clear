"""Keep default configuration calibration paths private to each test."""
from pathlib import Path
import shutil

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / 'config/calibration/examples'


@pytest.fixture
def calibration_dir(tmp_path_factory):
    """A writable copy of the repository's placeholder calibration templates."""
    directory = tmp_path_factory.mktemp('test_calibration')
    shutil.copytree(EXAMPLES, directory, dirs_exist_ok=True)
    camera = directory / 'camera.yaml'
    data = yaml.safe_load(camera.read_text())
    # At scan Z=-3 this synthetic geometry gives 26.2 µm/px.
    data['translation'][2] = 125.1968
    camera.write_text(yaml.safe_dump(data))
    return directory


@pytest.fixture(autouse=True)
def isolate_default_calibration(calibration_dir, monkeypatch):
    """Redirect default paths while leaving explicitly supplied test paths intact."""
    original = yaml.safe_load

    def load(stream):
        data = original(stream)
        if isinstance(data, dict) and isinstance(data.get('vision'), dict) and isinstance(data.get('tool_calibration'), dict):
            vision = data['vision']
            tool = data['tool_calibration']
            source = getattr(stream, 'name', None)
            if vision.get('calibration') == 'config/calibration/camera.yaml' or (
                vision.get('calibration') == 'calibration/camera.yaml'
                and source is not None
                and Path(source).resolve() == ROOT / 'config/nest_clear.yaml'
            ):
                vision['calibration'] = str(calibration_dir / 'camera.yaml')
            if tool.get('directory') == 'config/calibration':
                tool['directory'] = str(calibration_dir)
        return data

    monkeypatch.setattr(yaml, 'safe_load', load)
