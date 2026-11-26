"""对刀相机标定尖检测、±3°倾斜相机雅可比对准、防撞保护与完整 Modbus 寄存器协议端到端测试（T86-3）。"""
from dataclasses import replace
import json
import math
import threading
import time

import cv2
import numpy as np
import pytest
import yaml

from nest_clear.camcal import CalibResult, save_calibration
from nest_clear.config import Scan, load_config
from nest_clear.main import Application
from nest_clear.tooloffset.alignment import (
    AlignConfig,
    ToolAlignmentSession,
    predict_collision_gap_mm,
    solve_view_delta,
)
from nest_clear.tooloffset.ref_tip_side import (
    RefTipSideConfig,
    RefTipSideDetector,
    measure_dual_tips,
    render_tooloffset_side_overlay,
)
from nest_clear.tooloffset.store import load_offset
from nest_clear.tooltip import ToolTipBaseline, ViewCalib
from nest_clear.tooltip.store import initialize_tools, load_tools, save_baseline, save_tools
from nest_clear.tooltip.trident import TridentTipDetector
from nest_clear.vision.geometry import Calibration
from sim.fake_plc import FakePLC


def _render_backlit_scene(
    tool_tip_uv: tuple[float, float],
    ref_tip_uv: tuple[float, float] | None,
    *,
    size: tuple[int, int] = (640, 480),
    tilt_deg: float = 0.0,
    noise_sigma: float = 0.0,
    seed: int = 0,
    extra_cone: tuple[float, float] | None = None,
) -> np.ndarray:
    """绘制与真机三棱针背光一致的合成图（顶部入镜三棱针 + 底部向上标定锥）。"""
    w, h = size
    ss = 4
    shift = 0.5 * (ss - 1)
    hi = np.full((h * ss, w * ss), 235.0, dtype=np.float32)

    # 顶部三棱针：轴向略带 tilt_deg 倾斜，末端含三齿，主针长覆盖 y=0..tip_y-12
    rad = math.radians(tilt_deg)
    d = np.array([math.sin(rad), math.cos(rad)], dtype=float)
    n = np.array([d[1], -d[0]], dtype=float)
    tx, ty = tool_tip_uv
    tip = np.array([tx, ty], dtype=float)
    prong_len = 36.0
    fork = tip - prong_len * d
    shaft_len = max(140.0, fork[1] + 20.0)
    top_center = fork - shaft_len * d

    # 画主针杆（半宽 11 px，保证直线段长度与边缘清晰度满足 Trident 精定位）
    shaft_poly = np.array([
        top_center - 11.0 * n,
        top_center + 11.0 * n,
        fork + 11.0 * n,
        fork - 11.0 * n,
    ])
    cv2.fillPoly(hi, [np.round(shaft_poly * ss + shift).astype(np.int32)], 28.0, lineType=cv2.LINE_AA)

    # 画三根针齿，中齿最长到 tip
    for lateral, length in ((-7.0, 26.0), (0.0, prong_len), (7.0, 28.0)):
        p0 = fork + lateral * n - 4.0 * d
        p1 = fork + lateral * n + length * d
        pts_p = np.array([p0 - 1.6 * n, p0 + 1.6 * n, p1 + 1.6 * n, p1 - 1.6 * n])
        cv2.fillPoly(hi, [np.round(pts_p * ss + shift).astype(np.int32)], 30.0, lineType=cv2.LINE_AA)

    # 画底部向上伸入的标定锥
    for cone_uv in [ref_tip_uv, extra_cone]:
        if cone_uv is None:
            continue
        rx, ry = cone_uv
        base_half_w = max(8.0, (h - ry) * math.tan(math.radians(16.0)))
        cone_poly = np.array([
            [rx, ry],
            [rx - base_half_w, float(h + 2)],
            [rx + base_half_w, float(h + 2)],
        ])
        cv2.fillPoly(hi, [np.round(cone_poly * ss + shift).astype(np.int32)], 25.0, lineType=cv2.LINE_AA)

    img = cv2.resize(hi, (w, h), interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (3, 3), 0.7)
    if noise_sigma > 0:
        rng = np.random.default_rng(seed)
        img += rng.normal(0.0, noise_sigma, img.shape).astype(np.float32)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def _tilted_jacobians(tilt_deg: float = 3.0, scale_px_per_mm: float = 65.0) -> tuple[np.ndarray, np.ndarray]:
    """构造非横平竖直、各绕不同轴倾斜 ±3° 的 front/side 相机雅可比矩阵（px/mm）。"""
    rf = cv2.Rodrigues(np.radians([2.1, -1.8, tilt_deg]))[0]
    rs = cv2.Rodrigues(np.radians([-2.0, 1.5, -tilt_deg]))[0]
    # front 名义映射 (X -> u, Z -> -v)，side 名义映射 (Y -> u, Z -> -v)
    proj_front = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]) * scale_px_per_mm
    proj_side = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]) * scale_px_per_mm
    j_front = proj_front @ rf
    j_side = proj_side @ rs
    return j_front, j_side


def test_ref_tip_side_detector_cone_pin_and_errors():
    det = RefTipSideDetector()
    t_det = TridentTipDetector()
    ref_true = (318.4, 385.6)
    img = _render_backlit_scene((325.0, 210.0), ref_true, tilt_deg=2.5, noise_sigma=1.0, seed=11)

    res_t = t_det.detect(img)
    res_r = det.detect(img)
    assert res_t.point_uv is not None and res_t.focused
    assert res_r.point_uv is not None and res_r.focused
    assert res_r.quality > 0.4
    assert math.hypot(res_r.point_uv[0] - ref_true[0], res_r.point_uv[1] - ref_true[1]) < 0.7

    ovl = render_tooloffset_side_overlay(
        img,
        "front",
        tool_uv=res_t.point_uv,
        ref_uv=res_r.point_uv,
        ref_result=res_r,
        z_dir_uv=(2.0, -65.0),
        delta_mm=-0.1,
        gap_mm=2.69,
    )
    assert ovl.shape == (480, 640, 3)

    # 仅工具无标定锥 -> 识别失败
    img_no_ref = _render_backlit_scene((325.0, 210.0), None)
    assert det.detect(img_no_ref).point_uv is None
    assert "未识别到对刀标定尖" in det.detect(img_no_ref).diagnostic

    # 双标定锥 -> 不唯一报错
    img_two = _render_backlit_scene((325.0, 210.0), (220.0, 390.0), extra_cone=(420.0, 395.0))
    r_dup = det.detect(img_two)
    assert r_dup.point_uv is None
    assert "不唯一" in r_dup.diagnostic

    # 平行直柱销（无锥角）回退中心线穿越检测
    pin = np.full((480, 640), 235, dtype=np.uint8)
    cv2.rectangle(pin, (315, 420), (325, 479), 20, -1)
    r_pin = det.detect(pin)
    assert r_pin.point_uv is not None
    assert abs(r_pin.point_uv[0] - 320.0) < 0.6
    assert abs(r_pin.point_uv[1] - 420.0) < 1.2

    # measure_dual_tips 只要任意帧缺标定尖均拒收
    frames_bad = {
        "front": [img, img, img_no_ref, img, img],
        "side": [img] * 5,
    }
    dual = measure_dual_tips(frames_bad, {"front": {"method": "trident"}, "side": {"method": "trident"}}, None)
    assert dual["ok"] is False
    assert "正面标定尖" in dual["reason"]


def test_alignment_converges_under_3deg_camera_tilt_and_collision_guard():
    j_f, j_s = _tilted_jacobians(tilt_deg=3.0, scale_px_per_mm=65.0)
    target_xy = (112.40, 58.60)
    true_offset_xy = (12.35, -8.42)
    # 标定尖三维虚拟位置：当工具走到 true_tool_at_tip 时，两视图中工具尖与标定尖重合
    true_aligned_xyz = (target_xy[0] + true_offset_xy[0], target_xy[1] + true_offset_xy[1], -6.5)
    r_f = np.array([320.0, 395.0])
    r_s = np.array([320.0, 395.0])
    p0 = (true_aligned_xyz[0] - 0.85, true_aligned_xyz[1] + 0.72, -2.5)  # 竖直间隙 4.0mm（Z-2 时仍留 2.0mm > 1mm）

    def obs(pose):
        dxyz = np.asarray(pose, dtype=float) - np.asarray(true_aligned_xyz, dtype=float)
        return (
            {"front": tuple((r_f + j_f @ dxyz).tolist()), "side": tuple((r_s + j_s @ dxyz).tolist())},
            {"front": tuple(r_f.tolist()), "side": tuple(r_s.tolist())},
        )

    travel = ((80.0, 200.0), (-10.0, 150.0), (-15.0, 5.0))
    cfg = AlignConfig(align_tol_mm=0.05, max_align_iterations=10, max_step_mm=0.5, max_total_mm=3.0, min_gap_mm=1.0)
    sess = ToolAlignmentSession(1, target_xy, travel, cfg)

    pose = p0
    probes_seen = []
    align_iters = 0
    while True:
        tp, rp = obs(pose)
        dec = sess.step(pose, tp, rp)
        if dec["status"] == "converged":
            break
        assert dec["status"] == "move", dec
        nxt = dec["target"]
        if dec["phase"] == "probing":
            probes_seen.append(tuple(round(nxt[i] - p0[i], 4) for i in range(3)))
        else:
            align_iters += 1
            assert abs(nxt[0] - pose[0]) <= 0.5 + 1e-9
            assert abs(nxt[1] - pose[1]) <= 0.5 + 1e-9
            assert abs(nxt[2] - pose[2]) < 1e-9  # 对准步保持 Z 不变
            assert max(abs(nxt[i] - p0[i]) for i in range(3)) <= 3.0 + 1e-9
        pose = nxt

    assert probes_seen == [(0.2, 0.0, 0.0), (0.0, 0.2, 0.0), (0.0, 0.0, 2.0), (0.0, 0.0, -2.0)]
    assert 1 <= align_iters <= 5
    assert np.allclose(sess.jacobians["front"], j_f, atol=1e-6)
    assert np.allclose(sess.jacobians["side"], j_s, atol=1e-6)
    assert abs(dec["offset_xy"][0] - true_offset_xy[0]) <= 0.05
    assert abs(dec["offset_xy"][1] - true_offset_xy[1]) <= 0.05
    dx, _, _ = solve_view_delta(obs(pose)[0]["front"], r_f, j_f[:, 0], j_f[:, 2])
    dy, _, _ = solve_view_delta(obs(pose)[0]["side"], r_s, j_s[:, 1], j_s[:, 2])
    assert abs(dx) <= 0.05 and abs(dy) <= 0.05

    # 验证防撞：P0 距离标定尖仅 2.2mm，做 Z+2 安全（4.2mm），但下一轮 Z-2（相对 P0 将变成 0.2mm < 1mm）必须被拦截！
    p0_close = (true_aligned_xyz[0] - 0.3, true_aligned_xyz[1] + 0.2, -4.3)
    sess_col = ToolAlignmentSession(1, target_xy, travel, cfg)
    pose = p0_close
    for expected_step in range(3):
        tp, rp = obs(pose)
        d = sess_col.step(pose, tp, rp)
        assert d["status"] == "move"
        pose = d["target"]
    # 此时刚测完 X+0.2, Y+0.2, Z+2，准备下发 Z-2（预计间隙 0.2mm < 1.0mm）-> 必须防撞拦下！
    tp, rp = obs(pose)
    d_col = sess_col.step(pose, tp, rp)
    assert d_col["status"] == "collision"
    assert "会碰到标定尖" in d_col["reason"]
    assert d_col["predicted_gap_mm"] < 1.0


def _oblique_main_cal() -> tuple[Calibration, tuple[int, int]]:
    K = np.array([[1162.5, 0.0, 505.0], [0.0, 1163.75, 377.5], [0.0, 0.0, 1.0]])
    D = np.array([-0.06, 0.03, 0.0002, -0.0001, 0.0])
    R = cv2.Rodrigues(np.array([0.006, 0.0086, 0.02]))[0] @ np.diag([1.0, -1.0, -1.0])
    T = np.array([-52.5, -40.0, 0.0])
    return Calibration(K, D, R, T, 0.0), (1006, 759)


def _render_cone_for_pose(pose, tip_xyz, cal, size=(1006, 759)):
    from nest_clear.tooloffset.triangulation import project_point

    w, h = size
    tx, ty, tz = tip_xyz
    apex = np.array([tx, ty, tz], dtype=float)
    base_c = np.array([tx, ty, tz - 20.0], dtype=float)
    angles = np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False)
    base_pts = [base_c + np.array([1.2 * math.cos(a), 1.2 * math.sin(a), 0.0]) for a in angles]
    uv_a = project_point(apex, pose, cal)
    uv_b = [project_point(p, pose, cal) for p in base_pts]
    pts_2d = np.array([uv_a, *[p for p in uv_b if p is not None]], dtype=np.float32)
    hull = pts_2d[cv2.convexHull(pts_2d, returnPoints=False).ravel()]
    ss = 4
    hi = np.full((h * ss, w * ss), 225.0, dtype=np.float32)
    cv2.fillPoly(hi, [np.round(hull * ss + 0.5 * (ss - 1)).astype(np.int32)], 40.0, lineType=cv2.LINE_AA)
    return cv2.GaussianBlur(cv2.resize(hi, (w, h), interpolation=cv2.INTER_AREA), (3, 3), 0.6).astype(np.uint8)


def test_fake_plc_full_register_sequence_tool1_tool2_and_guards(tmp_path, monkeypatch):
    """完整真实 Modbus TCP 寄存器交互验证：90->91->100(4步+对准)->111->130，工具2，130不一致，999与40905=2防撞。"""
    main_cal, main_size = _oblique_main_cal()
    tip_true = (92.0, 30.0, 12.0)
    tool_offsets = {1: (10.5, -5.2), 2: (-8.3, 6.4)}
    ref_uv = {"front": (320.0, 430.0), "side": (320.0, 430.0)}
    j_front, j_side = _tilted_jacobians(tilt_deg=3.0, scale_px_per_mm=50.0)
    # 标定尖高度对应的工具尖理论重合 Z=-10.0；工具停在 Z=-6.8 时间隙 3.2mm（Z+2 到 5.2mm 仍够针身，Z-2 到 1.2mm 安全）
    tip_z_contact = -10.0

    # 保存真实 config + baseline + tools + camera
    save_calibration(CalibResult(main_cal, {}), tmp_path / "camera.yaml", tmp_path / "history")
    initialize_tools(tmp_path / "tools.yaml")
    save_tools(tmp_path / "tools.yaml", 1, (0.0, 0.0, -9.2), source="init")
    save_tools(tmp_path / "tools.yaml", 2, (0.0, 0.0, -9.8), source="init")
    # 也准备一套 40905=2 用的基准（target 在 ref 上方 3mm，与 40905=2 兼容）
    for tool in (1, 2):
        save_baseline(
            ToolTipBaseline(
                f"tool{tool}",
                (100.0, 30.0, -6.8),
                (
                    ViewCalib("front", (320.0, 280.0), j_front),
                    ViewCalib("side", (320.0, 280.0), j_side),
                ),
                (0.0, 0.0, -9.2 if tool == 1 else -9.8),
            ),
            tmp_path / f"tool{tool}_baseline.yaml",
            tool=tool,
            source="test",
        )

    cfg = load_config("config/nest_clear.yaml")
    tc = replace(
        cfg.tool_calibration,
        enabled=True,
        directory=str(tmp_path),
        cameras={"front": {"backend": "mock"}, "side": {"backend": "mock"}},
    )
    cfg = replace(
        cfg,
        tools=((0.0, 0.0, -9.2), (0.0, 0.0, -9.8)),
        travel=((50.0, 260.0), (-20.0, 160.0), (-15.0, 150.0)),
        scan=Scan(((160.0, 0.0), (100.0, 0.0), (160.0, 140.0)), 60.0, 70.0, 110.0),
        vision=replace(
            cfg.vision,
            calibration=str(tmp_path / "camera.yaml"),
            camera_backend="mock",
            responsibility_zone=False,
            hair_detector=replace(cfg.vision.hair_detector, expected_um_per_px=94.6),
        ),
        tool_calibration=tc,
        record_directory=str(tmp_path / "runs"),
        tool_offset={
            **cfg.tool_offset,
            "settle_window_s": 0.005,
            "triangulation": {
                **cfg.tool_offset["triangulation"],
                "approx_tip_z_mm": 12.0,
                "max_ray_residual_mm": 0.15,
            },
            "align": {
                **cfg.tool_offset["align"],
                "max_align_iterations": 10,
                "align_tol_mm": 0.05,
                "consistency_tolerance_mm": 0.05,
                "min_gap_mm": 1.0,
            },
        },
    )

    state = {"active_tool": 1, "tooltip_tip_offset": [0.0, 0.0, 0.0], "hide_cone": False}

    with FakePLC() as plc:
        cfg = replace(cfg, plc=replace(cfg.plc, ip=plc.host, port=plc.port))
        app = Application(cfg, "config/nest_clear.yaml")

        # 重写主相机工作线程采集：返回真实 3D 斜视圆锥渲染图
        def mock_main_execute(payload):
            pose = app.io.read_real32_block(plc.contract["actual_x"].address, 3)
            img = _render_cone_for_pose(pose, tip_true, main_cal, main_size)
            return img, {}

        app.camera.execute = mock_main_execute

        # 重写两台对刀相机模拟帧：同时渲染顶部三棱针 + 底部标定锥
        def make_calib_capture(view: str):
            J = j_front if view == "front" else j_side

            def cap():
                pose = np.asarray(app.io.read_real32_block(plc.contract["actual_x"].address, 3), dtype=float)
                mode = app.events.status.data.get("mode", "空闲")
                if mode.startswith("对刀中"):
                    # 40905=2：相对 baseline.p0 (100, 30, -6.8) 偏移
                    base_target = np.array([320.0, 280.0], dtype=float)
                    tool_uv = base_target + J @ (pose - np.array([100.0, 30.0, -6.8]) + np.asarray(state["tooltip_tip_offset"]))
                else:
                    # 40905=1：工具贴合点为 tip_true[:2] + tool_offsets[tool], z=tip_z_contact
                    off = tool_offsets[state["active_tool"]]
                    aligned = np.array([tip_true[0] + off[0], tip_true[1] + off[1], tip_z_contact], dtype=float)
                    tool_uv = np.asarray(ref_uv[view], dtype=float) + J @ (pose - aligned)
                cone = None if state["hide_cone"] else ref_uv[view]
                return _render_backlit_scene((float(tool_uv[0]), float(tool_uv[1])), cone, tilt_deg=2.0)

            return cap

        for view_name in ("front", "side"):
            cam_obj = app.calib_camera.cameras[view_name]
            cam_obj.capture = make_calib_capture(view_name)

        stop = threading.Event()
        errors = []

        def loop():
            try:
                while not stop.is_set():
                    app.step()
                    time.sleep(0.002)
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=loop)
        thread.start()
        try:
            # 1) 会话外收到 100 且 40905=1 -> 回 999 并写 40935=1
            r_oos = plc.wait_reply(plc.request(100, tool_id=1), 5)
            assert (r_oos["func_code"], r_oos["pc_tool_id"]) == (999, 1)

            def run_full_tool_offset(tool: int, p0_xy: tuple[float, float], p0_z: float = -6.8):
                state["active_tool"] = tool
                plc.set_actual_xyz((162.0, 70.0, 122.0))
                tri_steps = 0
                while True:
                    rep = plc.wait_reply(plc.request(90, tool_id=1), 8)
                    assert rep["pc_tool_id"] == 1
                    tri_steps += 1
                    if rep["func_code"] == 91:
                        break
                    assert rep["func_code"] == 90, app.events.status.data.get("tooloffset")
                    plc.set_actual_xyz(rep["scan_xyz"])
                assert tri_steps >= 3

                plc.set_actual_xyz((p0_xy[0], p0_xy[1], p0_z))
                req_code = 100 if tool == 1 else 110
                done_code = 111 if tool == 1 else 121
                probes = 0
                align_cmds = []
                while True:
                    rep = plc.wait_reply(plc.request(req_code, tool_id=1), 8)
                    assert rep["pc_tool_id"] == 1
                    if rep["func_code"] == done_code:
                        break
                    assert rep["func_code"] == req_code, app.events.status.data.get("tooloffset", {}).get("message")
                    nxt = rep[f"calibration{tool}_xyz"]
                    if probes < 4:
                        probes += 1
                    else:
                        align_cmds.append(nxt)
                    plc.set_actual_xyz(nxt)
                assert probes == 4 and len(align_cmds) >= 1

                p_final = plc.io.read_real32_block(plc.contract["actual_x"].address, 3)
                plc.set_calibration_final(tool, p_final)
                rep130 = plc.wait_reply(plc.request(130, tool_id=1), 8)
                assert (rep130["func_code"], rep130["pc_tool_id"]) == (130, 1)
                return p_final

            # 2) 完成工具 1
            p0_1 = (tip_true[0] + tool_offsets[1][0] - 0.65, tip_true[1] + tool_offsets[1][1] + 0.55)
            run_full_tool_offset(1, p0_1)
            off1 = load_offset(tmp_path / "tool_offset.yaml", 1)
            assert off1 == pytest.approx(tool_offsets[1], abs=0.06)
            t1_doc = yaml.safe_load((tmp_path / "tool_offset.yaml").read_text(encoding="utf-8"))["tool1"]
            assert t1_doc["baseline_stale"] is True
            tools_disk = load_tools(tmp_path / "tools.yaml")["tools"]
            assert tools_disk["tool1"]["offset_xyz"] == pytest.approx([off1[0], off1[1], -9.2], abs=1e-6)
            assert app.config.tools[0] == pytest.approx((off1[0], off1[1], -9.2), abs=1e-6)

            # 3) 紧接着从 90 重启标工具 2
            p0_2 = (tip_true[0] + tool_offsets[2][0] + 0.52, tip_true[1] + tool_offsets[2][1] - 0.48)
            run_full_tool_offset(2, p0_2)
            off2 = load_offset(tmp_path / "tool_offset.yaml", 2)
            assert off2 == pytest.approx(tool_offsets[2], abs=0.06)
            tools_disk2 = load_tools(tmp_path / "tools.yaml")["tools"]
            assert tools_disk2["tool2"]["offset_xyz"] == pytest.approx([off2[0], off2[1], -9.8], abs=1e-6)
            # 确认 tool1 未被破坏
            assert tools_disk2["tool1"]["offset_xyz"] == pytest.approx([off1[0], off1[1], -9.2], abs=1e-6)

            # 4) 验证保存路径：session_record.json 与 原图/叠加图 均落盘
            sessions_dir = list((app.events.run_directory / "tooloffset").iterdir())
            assert len(sessions_dir) == 2
            for sd in sessions_dir:
                assert (sd / "session_record.json").is_file()
                rec = json.loads((sd / "session_record.json").read_text(encoding="utf-8"))
                assert rec["outcome"] == "done"
                assert any(sd.glob("tri_s01_ovl.png"))
                assert any(sd.glob("align_s04_front_ovl.png"))

            # 5) 130 坐标不一致 -> 回 99，且不覆写已保存结果
            state["active_tool"] = 1
            plc.set_actual_xyz((162.0, 70.0, 122.0))
            while True:
                rep = plc.wait_reply(plc.request(90, tool_id=1), 8)
                if rep["func_code"] == 91:
                    break
                plc.set_actual_xyz(rep["scan_xyz"])
            plc.set_actual_xyz((p0_1[0] + 0.2, p0_1[1] - 0.2, -6.8))
            while True:
                rep = plc.wait_reply(plc.request(100, tool_id=1), 8)
                if rep["func_code"] == 111:
                    break
                plc.set_actual_xyz(rep["calibration1_xyz"])
            p_f = plc.io.read_real32_block(plc.contract["actual_x"].address, 3)
            plc.set_calibration_final(1, (p_f[0] + 0.5, p_f[1], p_f[2]))
            rep_bad130 = plc.wait_reply(plc.request(130, tool_id=1), 8)
            assert (rep_bad130["func_code"], rep_bad130["pc_tool_id"]) == (99, 1)

            # 6) 40905=1 防撞测试：工具离标定尖太近 (z=-7.8 -> 间隙 2.2mm，Z-2 后仅 0.2mm) -> 回 99
            plc.set_actual_xyz((162.0, 70.0, 122.0))
            while True:
                rep = plc.wait_reply(plc.request(90, tool_id=1), 8)
                if rep["func_code"] == 91:
                    break
                plc.set_actual_xyz(rep["scan_xyz"])
            plc.set_actual_xyz((p0_1[0], p0_1[1], -7.8))
            got_99 = False
            for _ in range(6):
                rep = plc.wait_reply(plc.request(100, tool_id=1), 8)
                assert rep["pc_tool_id"] == 1
                if rep["func_code"] == 99:
                    got_99 = True
                    break
                plc.set_actual_xyz(rep["calibration1_xyz"])
            assert got_99
            assert "会碰到标定尖" in app.events.status.data["tooloffset"]["message"]

            # 7) 40905=2（换针对刀）防撞 -> 回 999 且界面显示“会碰到标定尖”；缺标定尖 -> 回 131
            plc.set_actual_xyz((100.0, 30.0, -6.8))
            # baseline target_uv=(320, 280), ref_uv=(320, 430), 间隙 = (430-280)/50 = 3.0mm;
            # 让针尖向下错位 tip_offset_z = -2.5mm -> 间隙 0.5mm < 1mm，应直接回 999！
            state["tooltip_tip_offset"] = [0.0, 0.0, -2.5]
            rep_col2 = plc.wait_reply(plc.request(100, tool_id=2), 8)
            assert (rep_col2["func_code"], rep_col2["pc_tool_id"]) == (999, 2)
            assert "会碰到标定尖" in app.events.status.data["tooltip"]["message"]

            state["tooltip_tip_offset"] = [0.0, 0.0, 0.0]
            state["hide_cone"] = True
            rep_miss2 = plc.wait_reply(plc.request(100, tool_id=2), 8)
            assert (rep_miss2["func_code"], rep_miss2["pc_tool_id"]) == (131, 2)
        finally:
            stop.set()
            thread.join(3)
            app.close()
            assert not errors
