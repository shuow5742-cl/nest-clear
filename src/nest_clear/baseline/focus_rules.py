"""对刀基准建立的纯 Python 清晰度就绪判定与提示格式化（不依赖 cv2/numpy/yaml）。"""

import math
from typing import Any

BASELINE_RULE = "focus_both_v1"

FOCUS_DEFAULTS: dict[str, Any] = {
    "focus_smooth_frames": 5,
    "focus_pass_percent": 92.0,
    "focus_ready_percent": 95.0,
    "focus_good_px": 2.0,
    "focus_warn_px": 3.0,
    "focus_max_px": 3.0,
    "focus_ready_frames": 5,
    "peak_min_frames": 10,
    "edge_margin_px": 40.0,
}

CAMERA_AXES: dict[str, dict[str, str]] = {
    "front": {
        "label": "正面",
        "axis": "Y",
        "jog_hint": "点动 Y",
        "geometry_hint": "正面相机沿 Y 看 → 点动 Y 找正面最清晰",
    },
    "side": {
        "label": "侧面",
        "axis": "X",
        "jog_hint": "点动 X",
        "geometry_hint": "侧面相机沿 X 看 → 点动 X 找侧面最清晰",
    },
}

GEOMETRY_SUMMARY = (
    "正面沿 Y 看 → 点动 Y 找正面最清晰；侧面沿 X 看 → 点动 X 找侧面最清晰；"
    "Z 两路均在竖直方向，只需使针尖在画面内且不靠边"
)
STAGE_ASCENDING_TEXT = "继续同方向点动（尚未越过最清晰点）"
STAGE_OVERSHOT_TEXT = "已越峰，往回点动到峰值附近"
STAGE_READY_TEXT = "✔ 最清晰"
PEAK_OVERSHOOT_HINT = STAGE_ASCENDING_TEXT
EDGE_MARGIN_MESSAGE = "针尖太靠近画面边缘，点动 Z/X/Y 移回画面中部"


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    num = float(value)
    return num if math.isfinite(num) else None


def resolve_focus_config(live: dict[str, Any] | None = None, config: dict[str, Any] | None = None) -> dict[str, float | int]:
    """合并预览帧与配置中的清晰度判定参数，缺失时回退默认值。"""
    src_live = live if isinstance(live, dict) else {}
    src_cfg = config if isinstance(config, dict) else {}

    def pick_float(key: str, default: float, *, minimum: float = 0.0, allow_zero: bool = False) -> float:
        for source in (src_live, src_cfg):
            val = _finite_number(source.get(key))
            if val is not None and (val >= minimum if allow_zero else val > minimum):
                return val
        return default

    def pick_int(key: str, default: int) -> int:
        for source in (src_live, src_cfg):
            raw = source.get(key)
            if not isinstance(raw, bool) and isinstance(raw, (int, float)) and math.isfinite(raw) and int(raw) >= 1:
                return int(raw)
        return default

    good_px = pick_float("focus_good_px", float(FOCUS_DEFAULTS["focus_good_px"]))
    warn_px = pick_float("focus_warn_px", max(good_px, float(FOCUS_DEFAULTS["focus_warn_px"])))
    if warn_px < good_px:
        warn_px = good_px
    max_px = pick_float("focus_max_px", max(good_px, float(FOCUS_DEFAULTS["focus_max_px"])))
    if max_px < good_px:
        max_px = good_px
    ready_pct = min(100.0, pick_float("focus_ready_percent", float(FOCUS_DEFAULTS["focus_ready_percent"])))
    pass_pct = pick_float("focus_pass_percent", float(FOCUS_DEFAULTS["focus_pass_percent"]))
    if pass_pct >= ready_pct:
        pass_pct = max(1.0, ready_pct - 3.0)
    return {
        "focus_smooth_frames": pick_int("focus_smooth_frames", int(FOCUS_DEFAULTS["focus_smooth_frames"])),
        "focus_pass_percent": pass_pct,
        "focus_ready_percent": ready_pct,
        "focus_good_px": good_px,
        "focus_warn_px": warn_px,
        "focus_max_px": max_px,
        "focus_ready_frames": pick_int("focus_ready_frames", int(FOCUS_DEFAULTS["focus_ready_frames"])),
        "peak_min_frames": pick_int("peak_min_frames", int(FOCUS_DEFAULTS["peak_min_frames"])),
        "edge_margin_px": pick_float("edge_margin_px", float(FOCUS_DEFAULTS["edge_margin_px"]), minimum=0.0, allow_zero=True),
    }


def is_near_edge(
    tip_uv: list[float] | tuple[float, float] | None,
    image_size: list[int] | tuple[int, int] | None,
    margin_px: float,
) -> bool:
    """判断针尖是否距离画面四边小于 margin_px。"""
    if tip_uv is None or image_size is None or len(tip_uv) < 2 or len(image_size) < 2:
        return False
    u = _finite_number(tip_uv[0])
    v = _finite_number(tip_uv[1])
    w = _finite_number(image_size[0])
    h = _finite_number(image_size[1])
    if u is None or v is None or w is None or h is None or w <= 0 or h <= 0:
        return False
    return (u < margin_px) or ((w - 1.0 - u) < margin_px) or (v < margin_px) or ((h - 1.0 - v) < margin_px)


def evaluate_camera_readiness(
    name: str,
    live: dict[str, Any] | None,
    *,
    config: dict[str, Any] | None = None,
    manual_confirmed: bool = False,
) -> dict[str, Any]:
    """判定单路对刀相机是否已越峰确认并达到“最清晰可建基准”条件。"""
    axis_meta = CAMERA_AXES.get(name, {"label": name, "axis": "X/Y", "jog_hint": "点动调焦", "geometry_hint": ""})
    label = axis_meta["label"]
    jog_hint = axis_meta["jog_hint"]
    cfg = resolve_focus_config(live, config)
    src = live if isinstance(live, dict) else {}

    if not src:
        return {
            "camera": name,
            "label": label,
            "jog_hint": jog_hint,
            "ready": False,
            "frame_ok": False,
            "edge_near": False,
            "peak_valid": False,
            "peak_confirmed": False,
            "stage": "ascending",
            "stage_text": STAGE_ASCENDING_TEXT,
            "wide_edge_note": "",
            "manual_confirmed": bool(manual_confirmed),
            "color_key": "gray",
            "reason": f"{label}等待实时检测数据（{jog_hint}）",
            "short_reason": f"{label}无实时数据",
            "percent": None,
            "edge_width_px": None,
            "sample_count": 0,
            "consecutive_ready": 0,
            "config": cfg,
        }

    is_manual = bool(manual_confirmed or src.get("manual_confirmed"))
    det_raw = src.get("detection_status")
    det_str = "未识别" if det_raw in (None, "", "未检出") else str(det_raw)
    det_reason = str(src.get("detection_reason") or "")
    tip_uv = src.get("tip_uv")
    image_size = src.get("image_size")
    pct = _finite_number(src.get("percent"))
    edge_px = _finite_number(src.get("edge_width_px"))
    peak = _finite_number(src.get("peak"))
    sample_count_raw = src.get("sample_count")
    sample_count = int(sample_count_raw) if isinstance(sample_count_raw, (int, float)) and not isinstance(sample_count_raw, bool) else None
    consec_raw = src.get("consecutive_ready")
    consecutive = int(consec_raw) if isinstance(consec_raw, (int, float)) and not isinstance(consec_raw, bool) else None

    if "edge_near" in src:
        edge_near = bool(src["edge_near"])
    else:
        edge_near = is_near_edge(tip_uv, image_size, float(cfg["edge_margin_px"]))

    has_distortion = "工具在动或画面畸变" in det_reason
    det_ok = (det_str == "精定位") or (is_manual and not has_distortion)

    if sample_count is not None:
        peak_valid = peak is not None and sample_count >= int(cfg["peak_min_frames"])
    elif "peak_valid" in src:
        peak_valid = bool(src["peak_valid"])
    else:
        peak_valid = peak is not None

    if "peak_confirmed" in src:
        peak_confirmed = bool(src["peak_confirmed"])
    elif "ready" in src:
        peak_confirmed = bool(src["ready"])
    elif sample_count is not None:
        peak_confirmed = False
    else:
        # 兼容仅提供静态清晰度字段、无会话状态字段的单元测试输入
        peak_confirmed = True

    pct_ok = pct is not None and pct >= float(cfg["focus_ready_percent"])
    edge_ok = edge_px is not None and 0.0 < edge_px <= float(cfg["focus_max_px"])

    frame_ok = (
        (tip_uv is not None or is_manual)
        and not edge_near
        and det_ok
        and peak_valid
        and peak_confirmed
        and pct_ok
        and edge_ok
    )

    if consecutive is not None:
        ready = frame_ok and consecutive >= int(cfg["focus_ready_frames"])
    elif "ready" in src and not (is_manual and not src.get("manual_confirmed")):
        ready = bool(src["ready"]) and peak_confirmed and edge_ok
    else:
        ready = frame_ok

    pct_txt = f"{pct:.0f}%" if pct is not None else "—%"
    edge_txt = f"{edge_px:.2f}px" if edge_px is not None else "—px"
    wide_edge_note = (
        f"该路最佳边缘 {edge_px:.1f}px 偏宽，可接受"
        if (ready and edge_px is not None and edge_px > float(cfg["focus_good_px"]))
        else ""
    )

    if ready:
        stage = "ready"
        stage_text = STAGE_READY_TEXT
    elif peak_confirmed:
        stage = "overshot"
        stage_text = STAGE_OVERSHOT_TEXT
    else:
        stage = "ascending"
        stage_text = STAGE_ASCENDING_TEXT

    if edge_near:
        short_reason = f"{label}靠边（{EDGE_MARGIN_MESSAGE}）"
        reason = EDGE_MARGIN_MESSAGE
        color_key = "red"
    elif tip_uv is None and not is_manual:
        short_reason = f"{label}未识别（{jog_hint}）"
        reason = f"{label}未识别到针尖（{jog_hint}）"
        color_key = "red"
    elif has_distortion:
        short_reason = f"{label}晃动/畸变"
        reason = f"{label}工具在动或画面畸变"
        color_key = "red"
    elif not det_ok:
        detail_bits = [jog_hint]
        if pct is not None:
            detail_bits.append(f"当前 {pct_txt}")
        elif edge_px is not None:
            detail_bits.append(f"边缘 {edge_txt}")
        short_reason = f"{label}未精定位（{'，'.join(detail_bits)}）"
        reason = f"{label}未精定位（可{jog_hint}或点【人工点选】）"
        color_key = "red" if (edge_px is None or edge_px > float(cfg["focus_warn_px"])) else "yellow"
    elif not peak_valid:
        cnt = sample_count if sample_count is not None else 0
        need = int(cfg["peak_min_frames"])
        short_reason = f"{label}峰值样本不足（{cnt}/{need}，{jog_hint}）"
        reason = f"{label}峰值样本不足（{cnt}/{need} 帧），请{jog_hint}寻找最清晰点"
        color_key = "yellow"
    elif not peak_confirmed:
        short_reason = f"{label}尚未越峰（{jog_hint}，当前 {pct_txt}）"
        reason = f"{label}{STAGE_ASCENDING_TEXT}（{jog_hint}，需先降到 ≤{cfg['focus_pass_percent']:g}% 峰值确认）"
        color_key = "yellow"
    elif not pct_ok:
        short_reason = f"{label}模糊（{jog_hint}，当前 {pct_txt}）"
        reason = f"{label}{STAGE_OVERSHOT_TEXT}（当前 {pct_txt} < {cfg['focus_ready_percent']:g}%，{jog_hint}）"
        color_key = "yellow" if (pct is not None and pct >= 80.0) else "red"
    elif not edge_ok:
        short_reason = f"{label}边缘过宽 {edge_txt}（上限 {cfg['focus_max_px']:g}px，{jog_hint}）"
        reason = f"{label}边缘宽 {edge_txt} > 兜底上限 {cfg['focus_max_px']:g}px（{jog_hint}）"
        color_key = "red"
    elif not ready:
        cur_c = consecutive if consecutive is not None else 0
        need_c = int(cfg["focus_ready_frames"])
        short_reason = f"{label}稳定中（{cur_c}/{need_c} 帧）"
        reason = f"{label}已达清晰门限，保持稳定（{cur_c}/{need_c} 帧）"
        color_key = "yellow"
    else:
        short_reason = f"{label} ✔ 最清晰"
        reason = wide_edge_note
        color_key = "green" if not wide_edge_note else "yellow"

    return {
        "camera": name,
        "label": label,
        "jog_hint": jog_hint,
        "ready": ready,
        "frame_ok": frame_ok,
        "edge_near": edge_near,
        "peak_valid": peak_valid,
        "peak_confirmed": peak_confirmed,
        "stage": stage,
        "stage_text": stage_text,
        "wide_edge_note": wide_edge_note,
        "manual_confirmed": is_manual,
        "color_key": color_key,
        "reason": reason,
        "short_reason": short_reason,
        "percent": pct,
        "edge_width_px": edge_px,
        "sample_count": sample_count,
        "consecutive_ready": consecutive,
        "config": cfg,
    }


def evaluate_baseline_readiness(
    front_live: dict[str, Any] | None,
    side_live: dict[str, Any] | None,
    *,
    config: dict[str, Any] | None = None,
    manual_confirmed: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """综合两路相机状态生成顶部总状态与【对刀基准】可用性原因。"""
    manual_map = manual_confirmed or {}
    cams = {
        "front": evaluate_camera_readiness(
            "front", front_live, config=config, manual_confirmed=bool(manual_map.get("front"))
        ),
        "side": evaluate_camera_readiness(
            "side", side_live, config=config, manual_confirmed=bool(manual_map.get("side"))
        ),
    }
    unready = [cams[n] for n in ("front", "side") if not cams[n]["ready"]]
    if not unready:
        wide_notes = [f"{cams[n]['label']}{cams[n]['wide_edge_note']}" for n in ("front", "side") if cams[n]["wide_edge_note"]]
        suffix = f"（{'；'.join(wide_notes)}）" if wide_notes else ""
        return {
            "ready": True,
            "banner_text": f"✔ 两路针尖都最清晰，可作为对刀基准 → 点【对刀基准】{suffix}",
            "banner_color": "green",
            "disabled_reason": "",
            "cameras": cams,
        }
    parts = [item["short_reason"] for item in unready]
    joined = " / ".join(parts)
    if all(item["color_key"] == "gray" for item in unready):
        banner_color = "gray"
    elif any(item["color_key"] == "red" for item in unready):
        banner_color = "red"
    else:
        banner_color = "yellow"
    return {
        "ready": False,
        "banner_text": f"✖ 还需对准：{joined}",
        "banner_color": banner_color,
        "disabled_reason": joined,
        "cameras": cams,
    }
