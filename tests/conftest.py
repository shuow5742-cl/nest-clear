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
def isolate_default_calibration(calibration_dir, tmp_path_factory, monkeypatch):
    """Redirect default paths while leaving explicitly supplied test paths intact."""
    import nest_clear.baseline.preparation as preparation
    import nest_clear.devices.calib_cameras as calib_cameras
    import nest_clear.devices.tooltip_camera_store as tooltip_camera_store

    runs_dir = tmp_path_factory.mktemp('test_runs')
    original = yaml.safe_load

    def load(stream):
        data = original(stream)
        if isinstance(data, dict) and isinstance(data.get('vision'), dict) and isinstance(data.get('tool_calibration'), dict):
            vision = data['vision']
            tool = data['tool_calibration']
            record = data.get('record')
            source = getattr(stream, 'name', None)
            is_default_config = source is not None and Path(source).resolve() == ROOT / 'config/nest_clear.yaml'
            if vision.get('calibration') == 'config/calibration/camera.yaml' or (
                vision.get('calibration') == 'calibration/camera.yaml' and is_default_config
            ):
                vision['calibration'] = str(calibration_dir / 'camera.yaml')
            if tool.get('directory') == 'config/calibration':
                tool['directory'] = str(calibration_dir)
            if isinstance(record, dict) and record.get('directory') == 'runs' and is_default_config:
                record['directory'] = str(runs_dir)
        return data

    monkeypatch.setattr(yaml, 'safe_load', load)

    original_load_settings = tooltip_camera_store.load_settings
    default_tooltip_settings = ROOT / 'config/calibration/tooltip_cameras.yaml'

    def isolated_load_settings(path):
        if Path(path).resolve() == default_tooltip_settings:
            path = calibration_dir / 'tooltip_cameras.yaml'
        return original_load_settings(path)

    for module in (tooltip_camera_store, calib_cameras, preparation):
        monkeypatch.setattr(module, 'load_settings', isolated_load_settings)
