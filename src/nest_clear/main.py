"""主程序装配与每盘 Batch 的可暂停 tick 循环。"""

import argparse
import logging
import hashlib
import json
from dataclasses import asdict, replace
from importlib.metadata import version, PackageNotFoundError
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
from nest_clear.workers.camcal import CamcalWorker
from nest_clear.workers.tooloffset import ToolOffsetWorker
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


def repository_dirty(path: Path) -> bool | None:
    """记录工作树状态；无法查询时明确标记未知。"""
    try:
        result = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"],
                                         text=True, stderr=subprocess.DEVNULL, timeout=2)
        return bool(result.strip())
    except (OSError, subprocess.SubprocessError):
        return None


def file_snapshot(path: Path) -> dict[str, Any]:
    content = path.read_bytes()
    return {"path": str(path), "content": content.decode("utf-8"),
            "sha256": hashlib.sha256(content).hexdigest(), "mtime": path.stat().st_mtime}


class ProductionAssembly:
    """默认装配策略，直接连接生产 worker 与坐标逻辑。"""
    def prepare(self, config, path):
        pass

    def detector(self, config, calibration, path):
        detector = HairDetector(config.vision.hair_detector, calibration, path[0][2], config.vision.scale_z_bin_mm)
        for z in {point[2] for point in path}:
            detector.for_z(z)
        return detector

    def camera_sn(self, config):
        if config.camera_sn == "auto" and config.vision.camera_backend == "daheng":
            from nest_clear.devices.camera_select import resolve_camera
            return resolve_camera()
        return config.camera_sn

    def vision(self, real, config, path, events):
        return real

    def branches(self):
        return None

    def scale_error(self, exc, z):
        raise exc


def write_meta(events: EventRecorder, config_path: Path, calibration_path: Path,
               config: Config, camera: CameraWorker, detector: Any, path: list,
               calib_camera: CalibCameraWorker | None = None, assembly: Any = None) -> None:
    """记录启动快照；记录故障不改变设备握手。"""
    if events.run_directory is None:
        return
    try:
        try:
            framework_version = version("autoweaver")
        except PackageNotFoundError:
            framework_version = "unknown"
        from nest_clear.vision.geometry import scale_um_per_px
        from nest_clear.vision.hair import scaled
        calibration = load_calibration(calibration_path)
        scales = []
        for z in sorted({point[2] for point in path}):
            try:
                if isinstance(detector, HairDetector) and detector.calibration is not None:
                    parameters = detector.for_z(z)
                else:
                    scale = scale_um_per_px(calibration, calibration.plane_z, z)
                    parameters = scaled(replace(config.vision.hair_detector, um_per_px=scale))
                scales.append({"z_mm": z, "um_per_px": parameters.um_per_px,
                               "parameters": asdict(parameters)})
            except Exception as exc:
                scales.append((assembly or ProductionAssembly()).scale_error(exc, z))
        calibration_cameras = {}
        if calib_camera is not None:
            for name, device in calib_camera.cameras.items():
                info = {"sn": device.config.get("sn"), "model": None,
                        "width": None, "height": None, "settings": device.config,
                        "readback": {"exposure_us": None, "gain_db": None, "balance_ratio": None}}
                try:
                    if device.backend == "mock":
                        height, width = device.config.get("size", [480, 640])
                        info.update(model="mock", width=width, height=height)
                    elif device.backend == "uvc":
                        import cv2
                        info.update(model="uvc", width=int(device.device.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                    height=int(device.device.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                    elif device.device:
                        width, height = device.device.get_frame_size()
                        info.update(model=type(device.device).__name__, width=width, height=height)
                except Exception:
                    logger.warning("对刀相机 %s 参数回读不可用", name, exc_info=True)
                calibration_cameras[name] = info
        metadata = {
            "run_id": events.run_id, "machine_id": config.machine_id,
            "code": {"sha": repository_commit(ROOT), "dirty": repository_dirty(ROOT)},
            "auto_weaver": {"version": framework_version, "sha": framework_commit()},
            "config": file_snapshot(config_path),
            "contract": file_snapshot(ROOT / "config/plc_contract.yaml"),
            "calibration": file_snapshot(calibration_path),
            "tools": file_snapshot(Path(config.tool_calibration.directory) / "tools.yaml"),
            "camera": {"sn": camera.sn, "model": camera.model or type(camera.camera).__name__,
                       "width": camera.size[0] if camera.size else None,
                       "height": camera.size[1] if camera.size else None,
                       "settings": camera.settings, "readback": camera.readback},
            "calibration_cameras": calibration_cameras,
            "detection_scales": scales,
        }
        events.run_directory.mkdir(parents=True, exist_ok=True)
        (events.run_directory / "meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    except Exception:
        logger.exception("启动快照 meta.json 写入失败")


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
        self.events = events or EventRecorder(config.record_directory, machine_id=config.machine_id, run=True)
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
        assembly = ProductionAssembly()
        assembly.prepare(config, self.path)
        details = scan_details(config.scan, self.path)
        logger.info("扫描规划：%s", details)
        self.events.emit("scan_plan", **details)
        calibration_path = Path(config.vision.calibration)
        if not calibration_path.is_absolute():
            calibration_path = Path(config_path).resolve().parent / calibration_path
        calibration = load_calibration(calibration_path)
        self.calibration = calibration
        if detector is None and config.vision.detector == "hair":
            detector = assembly.detector(config, calibration, self.path)
        self.events.calibration_placeholder = calibration.placeholder
        contract = load_contract(ROOT / "config/plc_contract.yaml")
        self.io = PLCClient(config.plc.ip, config.plc.port, config.plc.timeout_s, contract)
        self.plc = PlcWorker(Handshake(self.io, contract, config.plc.write_readback, self.events))
        self.camera = None
        self.calib_camera = None
        self.clock = None
        try:
            camera_sn = assembly.camera_sn(config)
            self.camera = CameraWorker(config.vision.camera_backend, camera_sn, camera, timeout_s=config.capture_timeout_s, exposure_us=config.exposure_us, gain_db=config.gain_db, balance_ratio=config.balance_ratio, principal=(float(calibration.K[0][2]), float(calibration.K[1][2])), image_size=config.camera_image_size)
            self.camera.open()
            if config.tool_calibration.enabled:
                self.calib_camera = CalibCameraWorker(config.tool_calibration.cameras, timeout_s=config.capture_timeout_s, events=self.events)
                self.calib_camera.open()
            detector = detector or (HairDetector(config.vision.hair_detector, calibration, config.scan.z, config.vision.scale_z_bin_mm) if config.vision.detector == "hair" else StubDetector([PixelTarget(*pixel) for pixel in config.vision.pixels]))
            write_meta(self.events, Path(config_path), calibration_path, config, self.camera, detector, self.path, self.calib_camera, assembly)
            vision = VisionWorker(
                detector, calibration, config.vision.dedup_radius_mm, config.vision.max_targets, config.vision
            )
            vision = assembly.vision(vision, config, self.path, self.events)
            self.vision = vision
            self.calibration_path = calibration_path
            from nest_clear.flow.tool_offset import _existing
            from types import SimpleNamespace
            self.events.emit('tooloffset_state', phase='idle', message='等待 PLC 请求',
                             existing=_existing(SimpleNamespace(config=config), calibration_path.parent / 'tool_offset.yaml', metadata['calibration_sha256']))
            from nest_clear.flow.pick_accuracy import existing as accuracy_existing
            accuracy = accuracy_existing(SimpleNamespace(config=config), Path(config.tool_calibration.directory) / 'pick_compensation.yaml', metadata['calibration_sha256'])
            self.events.emit('pickacc_state', phase='idle', message='等待 PLC 请求', existing=accuracy)
            from nest_clear.baseline.session import BaselineSession, existing_baselines, settings as baseline_settings
            baseline_config = baseline_settings(ROOT / 'config/tooltip_baseline.yaml')
            def start_baseline():
                if self.calib_camera is None:
                    self.calib_camera = CalibCameraWorker(config.tool_calibration.cameras, timeout_s=config.capture_timeout_s, events=self.events)
                    if self.clock is not None:
                        self.clock.attach_worker(self.calib_camera)
                return BaselineSession(config, self.calib_camera, self.plc, self.events, calibration_path, baseline_config)
            self.events.start_baseline = start_baseline
            self.events.baseline_session = None
            self.events.emit(
                'baseline_state',
                phase='idle',
                message='请开始对刀基准建立',
                existing=existing_baselines(config, calibration_path),
                config=baseline_config,
            )
            # 主程序未运行期间遗留的开始命令不能在 PLC 初始化后自动启动会话。
            from nest_clear.tooloffset.commands import CommandReader, command_path
            try:
                prior_command = json.loads(command_path(self.events.directory).read_text(encoding='utf-8'))
                if prior_command.get('command') == 'baseline_start' and type(prior_command.get('seq')) is int:
                    seq = prior_command['seq']
                    self.events.command_reader = CommandReader(self.events.directory, self.events, last_seq=seq)
                    self.events.emit('tooloffset_command', seq=seq, command='baseline_start', params={}, accepted=False)
                    self.events.emit('baseline_state', phase='idle', message='主程序未运行时的开始请求已忽略，请重新开始')
            except (OSError, ValueError, TypeError, AttributeError):
                pass
            for tool in (1, 2):
                if not accuracy[f'tool{tool}'].get('type'):
                    self.events.emit('warning', reason=f'工具{tool}挑取补偿未标定', tool=tool)
            self.board = WorldBoard()
            self.clock = BTClock(self.board)
            for worker in (self.plc, self.camera, vision, CamcalWorker(), ToolOffsetWorker()):
                self.clock.attach_worker(worker)
            if self.calib_camera is not None:
                self.clock.attach_worker(self.calib_camera)
            self.handle = None
            baseline = self.plc.initialize()
            self.events.emit("baseline", task_id=baseline, ignored=bool(baseline))
            self.tracker = RequestTracker(baseline)
            context = FlowContext(self.board, config, self.path, self.tracker, self.events, calibration)
            self.branches = assembly.branches()
            logger.info("主程序行为树结构：\n%s", format_tree(build_tree(context, self.branches)))
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

    def update_calibration(self, calibration):
        self.vision.update_calibration(calibration)
        self.calibration = calibration
        self.camera.principal = (float(calibration.K[0][2]), float(calibration.K[1][2]))

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
                build_batch(self.board, self.config, self.path, self.tracker, self.events, branches=self.branches, calibration=self.calibration, calibration_updated=self.update_calibration, calibration_path=self.calibration_path)
            )
        self.clock.tick_once()

    def close(self) -> None:
        """释放时钟、相机和 PLC；当前流程没有设备动作收尾树。"""
        baseline = getattr(self.events, 'baseline_session', None)
        if baseline is not None:
            baseline.end('主程序停止')
            baseline.thread.join(timeout=self.config.capture_timeout_s + 2)
        if self.clock is not None:
            self.clock.shutdown()
        calib_camera = getattr(self, "calib_camera", None)
        if calib_camera is not None and calib_camera.opened:
            calib_camera.on_stop()
        abort_camcal = getattr(self.events, "abort_camcal", None)
        if abort_camcal is not None:
            abort_camcal("主程序停止，PLC 初始化或急停时会话中断")
        abort_offset = getattr(self.events, "abort_tooloffset", None)
        if abort_offset is not None:
            abort_offset("主程序停止，PLC 初始化或急停时会话中断")
        abort_accuracy = getattr(self.events, "abort_pickaccuracy", None)
        if abort_accuracy is not None:
            abort_accuracy("主程序停止，PLC 初始化或急停时会话中断")
        cycle = getattr(self.events, "active_cycle", None)
        if cycle is not None:
            cycle.pending()
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
    events: EventRecorder | None = None,
) -> None:
    """持续处理各盘；暂停时仍按节拍向父程序报告活性。"""
    config = load_config(config_path)
    app = Application(config, config_path, events=events)
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
        events = EventRecorder(config.record_directory, machine_id=config.machine_id, run=True)
        configure(config.record_directory, "main", run_directory=events.run_directory)
        root_logger.addHandler(handler)
        run(args.config, supervised=args.supervised, events=events)
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
