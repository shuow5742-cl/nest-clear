"""主程序装配与每盘 Batch 的可暂停 tick 循环。"""

import argparse
import logging
import hashlib
import signal
import yaml
from pathlib import Path
import subprocess
import time
from typing import Any, Protocol

from autoweaver.motion_policy import format_tree
from autoweaver.motion_policy.world_board import WorldBoard
from autoweaver.worker.clock import BTClock

from nest_clear.comm import Handshake, PLCClient, load_contract
from nest_clear.config import Config, load_config
from nest_clear.flow.batch import FlowContext, RequestTracker, build_batch, build_tree
from nest_clear.guard import GuardError
from nest_clear.plan.scan_path import build_scan_path, scan_details
from nest_clear.record.events import EventRecorder
from nest_clear.record.summary import TraySummary
from nest_clear.record.images import ImageArchive
from nest_clear.record.log_setup import configure
from nest_clear.paths import supervisor_state_file
from nest_clear.supervisor.gate import SupervisorGate
from nest_clear.vision.detector import PixelTarget, StubDetector
from nest_clear.vision.geometry import load_calibration
from nest_clear.workers.camera import CameraWorker
from nest_clear.workers.calib_camera import CalibCameraWorker
from nest_clear.workers.plc import PlcWorker
from nest_clear.workers.vision import VisionWorker, HairDetector
from nest_clear.tooltip.store import load_baseline

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
        return subprocess.check_output(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def repository_commit(path: Path) -> str:
    """查询仓库提交，非 Git 环境返回 unknown。"""
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL, timeout=2
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


class Application:
    """持有一次进程的设备、时钟和跨盘握手基线。"""

    def __init__(
        self,
        config: Config,
        config_path: str | Path,
        *,
        detector: Any = None,
        camera: Any = None,
        events: EventRecorder | None = None,
    ) -> None:
        """完成配置、路径、标定、PLC 和相机预检后装配 Workers。"""
        self.config = config
        for tool in (1, 2):
            baseline_path = Path(config.tool_calibration.directory) / f"tool{tool}_baseline.yaml"
            try:
                baseline_data = yaml.safe_load(baseline_path.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError) as exc:
                raise GuardError(
                    "TOOLTIP-BASELINE", f"读取工具 {tool} 基准失败：{exc}", str(baseline_path),
                    "请用 python -m nest_clear.tools.tooltip_baseline 建立基准。",
                ) from exc
            if isinstance(baseline_data, dict) and baseline_data.get("status") == "占位，未标定":
                logger.warning("工具 %s 基准为占位，未建立基准：%s", tool, baseline_path)
            else:
                baseline = load_baseline(baseline_path)
                if baseline.tool_name != f"tool{tool}":
                    raise GuardError(
                        "TOOLTIP-BASELINE", f"基准工具号不匹配：{baseline.tool_name}", str(baseline_path),
                        "请重新建立对应工具的基准。",
                    )
        self.events = events or EventRecorder(config.record_directory)
        config_bytes = Path(config_path).read_bytes()
        calibration_path = Path(config.vision.calibration)
        if not calibration_path.is_absolute():
            calibration_path = Path(config_path).resolve().parent / calibration_path
        calibration_info = yaml.safe_load(calibration_path.read_text(encoding="utf-8"))
        commit = repository_commit(ROOT)
        metadata = {
            "nest_commit": commit,
            "framework_commit": framework_commit(),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "plc": f"{config.plc.ip}:{config.plc.port}",
            "camera_sn": config.camera_sn,
            "camera_backend": config.vision.camera_backend,
            "calibration_placeholder": calibration_info.get("placeholder", False),
            "calibration_sha256": hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
            "calibration_mtime": calibration_path.stat().st_mtime,
        }
        logger.info("主程序启动信息：%s", metadata)
        self.events.emit("process_start", **metadata)
        self.summary = TraySummary(self.events)
        self.archive = ImageArchive(
            self.events,
            config.image_enabled,
            config.image_format,
            config.image_quality,
            config.image_retention_days,
            config.image_queue_size,
        )
        self.events.archive = self.archive
        self.path = build_scan_path(config.scan, config.travel)
        details = scan_details(config.scan, self.path)
        logger.info("扫描规划：%s", details)
        self.events.emit("scan_plan", **details)
        calibration_path = Path(config.vision.calibration)
        if not calibration_path.is_absolute():
            calibration_path = Path(config_path).resolve().parent / calibration_path
        calibration = load_calibration(calibration_path)
        self.calibration = calibration
        if detector is None and config.vision.detector == "hair":
            detector = HairDetector(config.vision.hair_detector, calibration, self.path[0][2], config.vision.scale_z_bin_mm)
            for z in {point[2] for point in self.path}:
                detector.for_z(z)
        self.events.calibration_placeholder = calibration.placeholder
        contract = load_contract(ROOT / "config/plc_contract.yaml")
        self.io = PLCClient(config.plc.ip, config.plc.port, config.plc.timeout_s, contract)
        self.plc = PlcWorker(Handshake(self.io, contract, config.plc.write_readback, self.events))
        self.camera = None
        self.calib_camera = None
        self.clock = None
        try:
            self.camera = CameraWorker(config.vision.camera_backend, config.camera_sn, camera, timeout_s=config.capture_timeout_s, exposure_us=config.exposure_us, gain_db=config.gain_db, balance_ratio=config.balance_ratio, principal=(float(calibration.K[0][2]), float(calibration.K[1][2])), image_size=config.camera_image_size)
            self.camera.open()
            if config.tool_calibration.enabled:
                self.calib_camera = CalibCameraWorker(config.tool_calibration.cameras, timeout_s=config.capture_timeout_s, events=self.events)
                self.calib_camera.open()
            detector = detector or (HairDetector(config.vision.hair_detector, calibration, config.scan.z, config.vision.scale_z_bin_mm) if config.vision.detector == "hair" else StubDetector([PixelTarget(*pixel) for pixel in config.vision.pixels]))
            vision = VisionWorker(
                detector, calibration, config.vision.dedup_radius_mm, config.vision.max_targets, config.vision
            )
            self.board = WorldBoard()
            self.clock = BTClock(self.board)
            for worker in (self.plc, self.camera, vision):
                self.clock.attach_worker(worker)
            if self.calib_camera is not None:
                self.clock.attach_worker(self.calib_camera)
            self.handle = None
            baseline = self.plc.initialize()
            self.events.emit("baseline", task_id=baseline, ignored=bool(baseline))
            self.tracker = RequestTracker(baseline)
            context = FlowContext(self.board, config, self.path, self.tracker, self.events, calibration)
            logger.info("主程序行为树结构：\n%s", format_tree(build_tree(context)))
            self.io.write("algorithm_ready", 1)
        except BaseException:
            if self.clock is not None:
                self.clock.shutdown()
            if self.calib_camera is not None:
                self.calib_camera.on_stop()
            if self.camera is not None:
                self.camera.camera.close()
            self.io.close()
            self.archive.close()
            raise

    def mark(self, event: str, **fields: Any) -> None:
        """将父程序暂停变化接入事件与盘计时。"""
        self.events.emit(event, **fields)

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
                    raise GuardError(
                        "BATCH-FAILED", f"一盘执行失败：{result}", "主程序", "查看前面的 PLC 或行为树错误。"
                    )
            self.handle = self.clock.submit(
                build_batch(self.board, self.config, self.path, self.tracker, self.events, calibration=self.calibration)
            )
        self.clock.tick_once()

    def close(self) -> None:
        """释放时钟、相机和 PLC；当前流程没有设备动作收尾树。"""
        if self.clock is not None:
            self.clock.shutdown()
        try:
            self.io.write("algorithm_ready", 0)
        except Exception as exc:
            logger.error("就绪位 40932 清零失败：%s", exc)
            self.events.emit("error", operation="clear_algorithm_ready", reason=str(exc))
        self.io.close()
        self.summary.finish("interrupted", self.events.clock().isoformat(), self.events.monotonic())
        self.archive.close()
        self.events.emit("process_stop")


def run(
    config_path: str | Path = ROOT / "config/nest_clear.yaml",
    gate: Gate | None = None,
    supervised: bool = False,
) -> None:
    """持续处理各盘；暂停时仍按节拍向父程序报告活性。"""
    config = load_config(config_path)
    app = Application(config, config_path)
    if gate is None and (supervised or supervisor_state_file(config.record_directory).exists()):
        gate = SupervisorGate(config.record_directory, mark=app.mark)
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
    parser.add_argument("--supervised", action="store_true")
    args = parser.parse_args()
    fallback = ROOT / "runs/startup_error.log"
    fallback.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(fallback, encoding="utf-8")
    logging.getLogger().addHandler(handler)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.getLogger().setLevel(logging.INFO)
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        config = load_config(args.config)
        root_logger = logging.getLogger()
        for existing in list(root_logger.handlers):
            root_logger.removeHandler(existing)
        configure(config.record_directory, "main")
        root_logger.addHandler(handler)
        run(args.config, supervised=args.supervised)
    except GuardError as exc:
        logger.error("启动或运行失败：%s", exc)
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        logger.info("主程序退出")
    except Exception as exc:
        logger.exception("启动或运行失败：%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
