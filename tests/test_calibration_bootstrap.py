from concurrent.futures import ThreadPoolExecutor
import hashlib

from nest_clear.calibration_bootstrap import CALIBRATION_FILES, TEMPLATE_DIRECTORY, ensure_calibration_files


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_missing_directory_and_files_are_created_from_templates(tmp_path, caplog):
    directory = tmp_path / "nested" / "calibration"
    ensure_calibration_files(directory)
    for name in CALIBRATION_FILES:
        assert (directory / name).read_bytes() == (TEMPLATE_DIRECTORY / name).read_bytes()
        assert f"使用占位模板：{name}" in caplog.text


def test_existing_files_never_change(tmp_path):
    for index, name in enumerate(CALIBRATION_FILES):
        (tmp_path / name).write_bytes(b"real calibration\x00" + bytes([index]))
    before = {name: digest(tmp_path / name) for name in CALIBRATION_FILES}
    ensure_calibration_files(tmp_path)
    ensure_calibration_files(tmp_path)
    assert {name: digest(tmp_path / name) for name in CALIBRATION_FILES} == before


def test_concurrent_calls_are_idempotent(tmp_path):
    directory = tmp_path / "calibration"
    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(ensure_calibration_files, [directory] * 32))
    assert sorted(p.name for p in directory.iterdir()) == sorted(CALIBRATION_FILES)
    assert all(digest(directory / name) == digest(TEMPLATE_DIRECTORY / name) for name in CALIBRATION_FILES)
