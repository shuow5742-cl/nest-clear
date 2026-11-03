"""主程序装配与每盘 Batch 的可暂停 tick 循环。"""
import argparse
import logging
from pathlib import Path
import subprocess
import time
from typing import Any, Protocol

from autoweaver.motion_policy.world_board import WorldBoard
from autoweaver.worker.clock import BTClock

from nest_clear.comm import Handshake, PLCClient, load_contract
from nest_clear.config import Config, load_config
from nest_clear.flow.batch import RequestTracker, build_batch
from nest_clear.guard import GuardError
from nest_clear.plan.scan_path import build_scan_path
from nest_clear.record.events import EventRecorder
from nest_clear.vision.detector import PixelTarget, StubDetector
from nest_clear.vision.geometry import load_calibration
from nest_clear.workers.camera import CameraWorker
from nest_clear.workers.plc import PlcWorker
from nest_clear.workers.vision import VisionWorker

logger = logging.getLogger("nest_clear.main")
ROOT = Path(__file__).resolve().parents[2]


class Gate(Protocol):
    """父程序可注入的暂停和活性接口。"""
    def beat(self) -> None:
        """上报主进程活性。"""
        ...

    def paused(self) -> bool:
        """返回是否暂停业务 tick。"""
        ...


def framework_commit() -> str:
    """读取框架提交号用于现场追溯，非 Git 安装时返回未知。"""
    try:
        import autoweaver
        directory = Path(autoweaver.__file__).resolve().parent
        return subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL, timeout=2).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


class Application:
    """持有一次进程的设备、时钟和跨盘握手基线。"""
    def __init__(self, config: Config, config_path: str | Path, *, detector: Any = None, camera: Any = None, events: EventRecorder | None = None) -> None:
        """完成配置、路径、标定、PLC 和相机预检后装配 Workers。"""
        self.config = config
        self.events = events or EventRecorder(config.record_directory)
        self.events.emit("process_start", framework_commit=framework_commit(), plc=f"{config.plc.ip}:{config.plc.port}", camera_sn=config.camera_sn)
        self.path = build_scan_path(config.scan, config.travel)
        calibration_path = Path(config.vision.calibration)
        if not calibration_path.is_absolute():
            calibration_path = Path(config_path).resolve().parent / calibration_path
        calibration = load_calibration(calibration_path)
        contract = load_contract(ROOT / "config/plc_contract.yaml")
        self.io = PLCClient(config.plc.ip, config.plc.port, config.plc.timeout_s, contract)
        self.plc = PlcWorker(Handshake(self.io, contract))
        self.camera = None
        self.clock = None
        try:
            baseline = self.plc.initialize()
            self.events.emit("baseline", task_id=baseline, ignored=bool(baseline))
            self.tracker = RequestTracker(baseline)
            self.camera = CameraWorker(config.vision.camera_backend, config.camera_sn, camera)
            self.camera.open()
            detector = detector or StubDetector([PixelTarget(*pixel) for pixel in config.vision.pixels])
            vision = VisionWorker(detector, calibration, config.vision.dedup_radius_mm, config.vision.max_targets)
            self.board = WorldBoard()
            self.clock = BTClock(self.board)
            for worker in (self.plc, self.camera, vision):
                self.clock.attach_worker(worker)
            self.handle = None
        except BaseException:
            if self.clock is not None:
                self.clock.shutdown()
            elif self.camera is not None:
                self.camera.camera.close()
            self.io.close()
            raise

    def step(self, gate: Gate | None = None) -> None:
        """每拍维持活性；暂停时不提交或推进 Batch。"""
        if gate is not None:
            gate.beat()
            if gate.paused():
                return
        if self.handle is None or self.handle.result is not None:
            if self.handle is not None:
                result = self.handle.result
                if result.reason.value != "completed":
                    raise GuardError("BATCH-FAILED", f"一盘执行失败：{result}", "主程序", "查看前面的 PLC 或行为树错误。")
            self.handle = self.clock.submit(build_batch(self.board, self.config, self.path, self.tracker, self.events))
        self.clock.tick_once()

    def close(self) -> None:
        """释放时钟、相机和 PLC；当前流程没有设备动作收尾树。"""
        if self.clock is not None:
            self.clock.shutdown()
        self.io.close()
        self.events.emit("process_stop")


def run(config_path: str | Path = ROOT / "config/nest_clear.yaml", gate: Gate | None = None) -> None:
    """持续处理各盘；暂停时仍按节拍向父程序报告活性。"""
    config = load_config(config_path)
    app = Application(config, config_path)
    try:
        while True:
            app.step(gate)
            time.sleep(0.02)
    finally:
        app.close()


def main() -> None:
    """解析入口配置并运行主程序。"""
    parser = argparse.ArgumentParser(description="燕窝挑毛算法主程序")
    parser.add_argument("--config", default=str(ROOT / "config/nest_clear.yaml"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        run(args.config)
    except GuardError as exc:
        logger.error("启动或运行失败：%s", exc)
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        logger.info("主程序退出")


if __name__ == "__main__":
    main()
