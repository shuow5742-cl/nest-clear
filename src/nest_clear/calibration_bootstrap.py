"""Create missing calibration files from the shipped placeholder templates."""
import logging
import os
from pathlib import Path
import tempfile

TEMPLATE_DIRECTORY = Path(__file__).resolve().parents[2] / "config/calibration/examples"
CALIBRATION_FILES = ("camera.yaml", "tools.yaml", "tool1_baseline.yaml", "tool2_baseline.yaml", "tooltip_cameras.yaml")
logger = logging.getLogger(__name__)


def ensure_calibration_files(directory: str | Path) -> None:
    """Atomically publish templates only where no directory entry exists.

    A hard link provides atomic create-if-absent semantics; os.replace would
    overwrite a calibration created by another process after an existence check.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in CALIBRATION_FILES:
        target = directory / name
        if os.path.lexists(target):
            continue
        fd, temporary = tempfile.mkstemp(prefix=f".{name}.", dir=directory)
        try:
            with os.fdopen(fd, "wb") as output, (TEMPLATE_DIRECTORY / name).open("rb") as source:
                output.write(source.read())
            try:
                os.link(temporary, target)
            except FileExistsError:
                continue
            logger.warning("使用占位模板：%s", name)
        finally:
            os.unlink(temporary)
