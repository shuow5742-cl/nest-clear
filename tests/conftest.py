"""Initialize repository placeholder files needed by tests that read them directly."""
from pathlib import Path

from nest_clear.calibration_bootstrap import ensure_calibration_files

ensure_calibration_files(Path(__file__).resolve().parents[1] / "config/calibration")
