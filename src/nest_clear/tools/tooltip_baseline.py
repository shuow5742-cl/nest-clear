"""人工点动建立对刀基准；本程序绝不写 PLC。"""

import argparse
from datetime import datetime
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from nest_clear.comm.client import PLCClient
from nest_clear.comm.contract import load_contract
from nest_clear.config import load_config
from nest_clear.devices.calib_cameras import CalibCamera
from nest_clear.guard import GuardError
from nest_clear.paths import daily_file
from nest_clear.record.events import EventRecorder
from nest_clear.record.log_setup import configure
from nest_clear.tooltip import (
    SilhouetteTipDetector,
    ToolTipBaseline,
    ViewCalib,
    check_quality,
    fit_view_jacobian,
    solve_correction,
)
from nest_clear.tooltip import store
from nest_clear.tooltip.click import confirm_tip


def _fail(message: str, hint: str) -> GuardError:
    """生成供现场操作人员阅读的诊断。"""
    return GuardError("BASELINE-TOOL", message, "tools.tooltip_baseline", hint)


def _fit(session: dict, config: dict) -> ToolTipBaseline:
    """从原始观测拟合两个视图并检查残差与可观测性。"""
    reference = session["frames"][0]
    views = []
    for name in ("front", "side"):
        samples = [
            (
                np.asarray(frame["xyz"]) - reference["xyz"],
                np.asarray(frame["points"][name]) - reference["points"][name],
            )
            for frame in session["frames"][1:]
        ]
        try:
            fit = fit_view_jacobian(samples)
        except GuardError as exc:
            raise _fail(str(exc), "至少采集三个独立方向；多采几步，检查 PLC 坐标变化。") from exc
        print(
            f"{name}: RMS={fit.rms_px:.4f}px，位移条件数={fit.condition_number:.3f}，J={fit.jacobian.tolist()}"
        )
        if (
            fit.rms_px > config["quality"]["max_fit_rms_px"]
            or fit.condition_number > config["quality"]["max_condition"]
        ):
            raise _fail(f"{name} 拟合质量不达标", "多采几步、增大步距并确认针尖无遮挡。")
        views.append(ViewCalib(name, reference["points"][name], fit.jacobian,
                               reference.get("widths", {}).get(name), reference.get("clicks", {}).get(name)))
    stacked = np.vstack([view.jacobian for view in views])
    condition = float(np.linalg.cond(stacked))
    print(f"联合矩阵条件数={condition:.3f}")
    if not np.isfinite(condition) or condition > config["quality"]["max_condition"]:
        raise _fail(
            "相机联合矩阵不可观测或条件数过大",
            "检查两相机能否看到对应轴，调整角度或增加步距后重新采集。",
        )
    return ToolTipBaseline(
        f"tool{session['tool']}",
        reference["xyz"],
        views,
        session["offset_xyz"],
        session["timestamp"],
        session["note"],
    )


def _detect(image: np.ndarray, name: str, config: dict, reference: bool = False) -> list[float]:
    """检测单视图针尖，失败时中止而不是保存坏基准。"""
    result = SilhouetteTipDetector(**config["detection"][name]).detect(image)
    if result.point_uv is None:
        raise _fail(
            f"{name} 相机无法识别针尖：{result.diagnostic}",
            "检查背光、阈值、方向、遮挡和相机视野。",
        )
    if not result.focused:
        message = f"{name} 相机针尖不清晰，边缘宽度 {result.edge_width_px} px"
        if reference:
            raise _fail(message, "请调焦或调整 Z/深度，使针尖位于景深内后重新采集基准。")
        print(f"警告：{message}；步进样本仍将使用。")
    return list(result.point_uv)


def _capture(
    cameras: dict, client: PLCClient, address: int, session_dir: Path, index: int, config: dict,
    no_confirm: bool = False, events=None, tool=None
) -> dict:
    """读取实时坐标及两张原图，并保存可重算的观测。"""
    xyz = client.read_real32_block(address, 3)
    points = {}
    images = {}
    clicks, widths = {}, {}
    for name, camera in cameras.items():
        image = camera.capture()
        filename = f"{index:03d}_{name}.png"
        if not cv2.imwrite(str(session_dir / filename), image):
            raise _fail(f"无法保存 {filename}", "检查磁盘空间与目录权限。")
        images[name] = filename
        detection = SilhouetteTipDetector(**config["detection"][name]).detect(image)
        auto = detection.point_uv
        widths[name] = float(detection.width_px) if detection.width_px is not None and detection.width_px > 0 else None
        needs_click = index == 0 or auto is None or not detection.focused
        if needs_click and not no_confirm:
            selected = confirm_tip(image, detection, f"{name} 工具{tool} 步骤{index}")
            if selected.point_uv is None:
                if events is not None:
                    events.emit("tip_click", camera=name, tool=tool, auto_point=auto, manual_point=None,
                                correction_px=None, correction_mm=None, elapsed_s=selected.elapsed_s, source="cancel")
                raise _fail(f"{name} 点击已取消", "重新采集。")
            point = selected.point_uv
            correction = float(np.linalg.norm(np.asarray(point) - auto)) if auto is not None else None
            info = {"camera": name, "tool": tool, "auto_point": list(auto) if auto is not None else None,
                    "manual_point": list(point), "correction_px": correction,
                    "correction_mm": correction * 0.0152 if correction is not None else None,
                    "elapsed_s": selected.elapsed_s, "source": selected.source}
            clicks[name] = info
            if events is not None: events.emit("tip_click", **info)
        else:
            if auto is None: raise _fail(f"{name} 相机无法识别针尖：{detection.diagnostic}", "检查相机。")
            point = auto
            clicks[name] = {"source": "auto", "correction_px": 0.0, "correction_mm": 0.0}
        points[name] = list(point)
        if index == 0 and name == "front":
            distance = float(
                np.linalg.norm(np.asarray(points[name]) - np.array([image.shape[1] / 2, image.shape[0] / 2]))
            )
            if distance > config["center_warning_px"]:
                print(f"警告：正面针尖离中心 {distance:.1f}px，超过阈值；仍可继续。")
    print(f"采集 {index}：坐标 {xyz}，针尖 {points}")
    return {"xyz": xyz, "points": points, "images": images, "clicks": clicks, "widths": widths}


def _read_compensation(path: str, tool: int) -> tuple[list[float], dict]:
    with open(path, encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict) or not {"tool", "offset_xyz", "depth_z", "source", "timestamp"} <= data.keys():
        raise ValueError("补偿文件需要 tool、offset_xyz、depth_z、source 和 timestamp")
    if data["tool"] not in (tool, f"tool{tool}"):
        raise ValueError("补偿文件工具号不匹配")
    values = np.asarray(data["offset_xyz"], dtype=float)
    depth = float(data["depth_z"])
    if values.shape != (3,) or not np.isfinite(values).all() or not np.isfinite(depth):
        raise ValueError("补偿和下针深度必须为有限数值")
    if not isinstance(data["source"], str) or not data["source"] or not data["timestamp"]:
        raise ValueError("补偿来源和时间不能为空")
    return [float(values[0]), float(values[1]), depth], data


def main(argv: list[str] | None = None) -> int:
    """解析命令并执行采集、验证或离线重算。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", type=int, choices=(1, 2), required=True)
    parser.add_argument("--config", default="config/tooltip_baseline.yaml")
    parser.add_argument("--app-config", default="config/nest_clear.yaml")
    parser.add_argument("--contract", default="config/plc_contract.yaml")
    parser.add_argument("--calibration-dir", default="config/calibration")
    compensation = parser.add_mutually_exclusive_group()
    compensation.add_argument("--offset", nargs=3, type=float)
    compensation.add_argument("--compensation-from")
    parser.add_argument("--no-confirm", action="store_true")
    parser.add_argument("--note")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--yes", action="store_true", help="无需回车，自动采集并重置补偿")
    parser.add_argument("--no-reset", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--replay")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args(argv)
    client = None
    cameras = {}
    events = None
    try:
        with open(args.config, encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        app = load_config(args.app_config, require_tools=False)
        configure(app.record_directory, "tool")
        events = EventRecorder(app.record_directory, process="tool")
        events.emit("tooltip_start", tool=args.tool, verify=args.verify, replay=args.replay)
        calibration = Path(args.calibration_dir)
        from nest_clear.calibration_bootstrap import ensure_calibration_files
        ensure_calibration_files(calibration)
        baseline_path = calibration / f"tool{args.tool}_baseline.yaml"
        if args.replay:
            directory = Path(args.replay)
            session = json.loads((directory / "session.json").read_text(encoding="utf-8"))
            if session["tool"] != args.tool:
                raise _fail("会话工具号不匹配", "确认 --tool 和会话目录。")
            for index, frame in enumerate(session["frames"]):
                if frame.get("clicks"):
                    continue  # 保留人工确认的原始观测
                frame["points"] = {
                    name: _detect(cv2.imread(str(directory / filename), cv2.IMREAD_GRAYSCALE), name, config, reference=index == 0)
                    for name, filename in frame["images"].items()
                }
            baseline = _fit(session, config)
            if args.save:
                store.save_baseline(baseline, baseline_path, tool=args.tool, source="replay")
            print(f"重算 P0={baseline.p0}，补偿={baseline.offset_xyz}")
        else:
            contract = load_contract(args.contract)
            client = PLCClient(app.plc.ip, app.plc.port, app.plc.timeout_s, contract)
            address = contract["actual_x"].address
            def position() -> list[float]:
                """只读实时坐标，让模拟相机跟随平台而不控制 PLC。"""
                return client.read_real32_block(address, 3)

            from nest_clear.workers.calib_camera import CalibCameraWorker
            role_worker = CalibCameraWorker(app.tool_calibration.cameras, events=events)
            for name in ("front", "side"):
                camera_config = dict(config["cameras"][name])
                if camera_config.get('backend') == 'uvc':
                    camera_config['_role_resolver'] = role_worker.resolve_roles
                camera = CalibCamera(name, camera_config, position)
                cameras[name] = camera
                camera.open()
            now = datetime.now().astimezone()
            directory = daily_file(app.record_directory, "tooltip_sessions", now.date()) / now.strftime(
                "%H%M%S_%f"
            )
            directory.mkdir(parents=True)
            frames = [_capture(cameras, client, address, directory, 0, config, args.no_confirm, events, args.tool)]
            session = {"tool": args.tool, "timestamp": now.isoformat(), "frames": frames}
            (directory / "session.json").write_text(
                json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if args.verify:
                baseline = store.load_baseline(baseline_path)
                result = solve_correction(frames[0]["points"], baseline)
                quality = check_quality(
                    result, config["quality"]["max_residual_mm"], config["quality"]["max_condition"]
                )
                (directory / "session.json").write_text(
                    json.dumps(
                        {"tool": args.tool, "timestamp": now.isoformat(), "mode": "verify", "frames": frames},
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                print(f"修正 Δ={result.delta_xyz.tolist()} mm，质量={quality}")
            else:
                if args.steps < 3:
                    raise _fail("采样步数不足", "至少采集三个线性独立位移。")
                for index in range(1, args.steps + 1):
                    if not args.yes:
                        input(
                            f"请用 HMI 点动第 {index} 步（建议 ±X、±Y、±Z，"
                            f"{config['jog_step_mm']} mm），稳定后回车："
                        )
                    frames.append(_capture(cameras, client, address, directory, index, config, args.no_confirm, events, args.tool))
                    (directory / "session.json").write_text(
                        json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                print(f"请将针尖点动回 P0={frames[0]['xyz']}；本工具不会控制设备。")
                if args.compensation_from:
                    offset, compensation_data = _read_compensation(args.compensation_from, args.tool)
                    session["compensation"] = compensation_data
                else:
                    offset = (
                    args.offset
                    if args.offset is not None
                    else [
                        float(value)
                        for value in input("输入人工确定的 XYZ 基准补偿（空格分隔，mm）：").split()
                    ]
                )
                note = args.note if args.note is not None else input("输入备注：")
                session.update(offset_xyz=offset, note=note)
                (directory / "session.json").write_text(
                    json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                baseline = _fit(session, config)
                store.save_baseline(baseline, baseline_path, tool=args.tool, source="baseline_tool")
                reset = not args.no_reset and (
                    args.yes
                    or input("重置当前补偿为基准补偿？[Y/n]：").strip().lower() in ("", "y", "yes")
                )
                if reset:
                    tools_path = calibration / "tools.yaml"
                    if not tools_path.exists():
                        store.initialize_tools(tools_path)
                    store.save_tools(
                        calibration / "tools.yaml", args.tool, baseline.offset_xyz, source="baseline_tool"
                    )
            print(f"会话目录：{directory}")
        events.emit("tooltip_complete", tool=args.tool)
        return 0
    except (GuardError, OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        print(f"操作失败：{exc}")
        if events is not None:
            events.emit("tooltip_error", tool=args.tool, error=str(exc))
        return 1
    finally:
        for camera in cameras.values():
            camera.close()
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
