from dataclasses import replace
import logging

from nest_clear.config import load_config
from nest_clear.paths import ui_command_file
from nest_clear.supervisor.runner import Supervisor


def test_missing_preview_command_is_silent(tmp_path, caplog):
    config = replace(load_config('config/nest_clear.yaml'), record_directory=str(tmp_path))
    supervisor = Supervisor(config, None)
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            supervisor._preview_tick()
    assert not caplog.records


def test_preview_command_errors_are_deduplicated_and_reset(tmp_path, caplog):
    config = replace(load_config('config/nest_clear.yaml'), record_directory=str(tmp_path))
    supervisor = Supervisor(config, None)
    path = ui_command_file(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text('{', encoding='utf-8')
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            supervisor._preview_tick()
        assert len(caplog.records) == 1
        path.write_text('{}', encoding='utf-8')
        supervisor._preview_tick()
        assert len(caplog.records) == 2
        path.unlink()
        supervisor._preview_tick()
        path.write_text('{}', encoding='utf-8')
        supervisor._preview_tick()
        assert len(caplog.records) == 3
        path.write_text('{"seq": 1, "preview": false}', encoding='utf-8')
        supervisor._preview_tick()
        path.write_text('{}', encoding='utf-8')
        supervisor._preview_tick()
        assert len(caplog.records) == 4
