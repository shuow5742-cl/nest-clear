import pytest
import yaml

from nest_clear.tooloffset import load_offset, save_result


def test_save_preserves_other_tool_and_backs_up(tmp_path):
    path = tmp_path / 'tool_offset.yaml'
    old = {'tool1': {'offset_xy_mm': [1, 2]}, 'tool2': {'offset_xy_mm': [3, 4], 'n': 8}}
    path.write_text(yaml.safe_dump(old), encoding='utf-8')
    backup = save_result(path, 1, {'offset_xy_mm': [1.5, -2], 'n': 5})
    assert yaml.safe_load(backup.read_text()) == old
    assert yaml.safe_load(path.read_text())['tool2'] == old['tool2']
    assert load_offset(path, 1) == (1.5, -2.0)
    assert load_offset(path, 2) == (3.0, 4.0)


def test_missing_and_invalid_results(tmp_path):
    path = tmp_path / 'tool_offset.yaml'
    assert load_offset(path, 1) is None
    with pytest.raises(ValueError):
        save_result(path, 1, {'offset_xy_mm': [float('nan'), 0]})
    assert not path.exists()
    with pytest.raises(ValueError):
        load_offset(path, True)


def test_confirmed_updates_tools_and_preserves_z(tmp_path):
    from nest_clear.tooloffset.store import save_confirmed
    from nest_clear.tooltip.store import initialize_tools, load_tools, save_tools
    tools = tmp_path / 'tools.yaml'
    initialize_tools(tools)
    save_tools(tools, 1, [1, 2, 3], source='test')
    old_tools = tools.read_bytes()
    path = tmp_path / 'tool_offset.yaml'
    _, backup, xyz = save_confirmed(path, tools, 1, {'offset_xy_mm': [4, 5]})
    assert backup.read_bytes() == old_tools
    assert xyz == (4, 5, 3)
    assert load_tools(tools)['tools']['tool1']['offset_xyz'] == [4, 5, 3]
    assert yaml.safe_load(path.read_text())['tool1']['baseline_stale'] is True


def test_sync_failure_restores_original(tmp_path, monkeypatch):
    from nest_clear.tooloffset.store import save_confirmed
    from nest_clear.tooltip.store import initialize_tools
    tools = tmp_path / 'tools.yaml'; initialize_tools(tools)
    path = tmp_path / 'tool_offset.yaml'
    save_result(path, 1, {'offset_xy_mm': [1, 2]})
    before, before_tools = path.read_bytes(), tools.read_bytes()
    def fail(*args, **kwargs): raise OSError('disk error')
    monkeypatch.setattr('nest_clear.tooltip.store.save_tools', fail)
    with pytest.raises(OSError): save_confirmed(path, tools, 1, {'offset_xy_mm': [4, 5]})
    assert path.read_bytes() == before
    assert tools.read_bytes() == before_tools
