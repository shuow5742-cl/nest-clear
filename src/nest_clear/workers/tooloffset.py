"""工具偏移图像处理放在 worker 线程，不阻塞行为树时钟。"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from autoweaver.worker import AsyncRequestWorker

from nest_clear.tooloffset.detection import detect
from nest_clear.record.live import write_live

RING_COLOR = (60, 220, 60)       # BGR: 绿色圆环中心与拟合线
CROSS_COLOR = (220, 30, 220)     # 紫色十字中心
PRINCIPAL_COLOR = (255, 150, 0)  # 蓝色主点


def ellipse_points(target):
    c = np.asarray(target.marker, dtype=float)
    right = ((c[1]-c[0]) + (c[2]-c[3])) / 8
    down = ((c[3]-c[0]) + (c[2]-c[1])) / 8
    (_, _), (a, b), angle = target.ellipse
    radians = np.deg2rad(angle)
    rotation = np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]])
    result = []
    for theta in np.linspace(0, 2*np.pi, 150):
        local = rotation @ np.array([a/2*np.cos(theta), b/2*np.sin(theta)]) / 20
        result.append(np.asarray(target.ring_uv) + right*local[0] + down*local[1])
    return np.asarray(result)


def render_annotations(image, targets, current, principal):
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    for target in targets:
        selected = target.index == current
        color, thickness = ((0, 220, 255), 4) if selected else ((150, 150, 150), 1)
        p = tuple(np.rint(target.ring_uv).astype(int))
        cv2.polylines(overlay, [np.rint(ellipse_points(target)).astype(np.int32)], True, color, thickness, cv2.LINE_AA)
        cv2.drawMarker(overlay, tuple(np.rint(target.cross_uv).astype(int)), color, cv2.MARKER_CROSS, 35 if selected else 15, thickness)
        if selected:
            marker = np.asarray(target.marker, dtype=float)
            scale = (np.linalg.norm(marker[1]-marker[0]) + np.linalg.norm(marker[3]-marker[0])) / 8
            center = np.asarray(target.ring_uv)
            low = np.minimum(center - 5.5*scale, marker.min(axis=0)) - 8
            high = np.maximum(center + 5.5*scale, marker.max(axis=0)) + 8
            x0, y0 = np.rint(low).astype(int); x1, y1 = np.rint(high).astype(int)
            cv2.rectangle(overlay, (x0,y0), (x1,y1), (0,255,255), 12, cv2.LINE_AA)
            cv2.putText(overlay, target.label, (max(0,x0), max(70,y0-18)), cv2.FONT_HERSHEY_SIMPLEX, 6.0, (0,255,255), 9, cv2.LINE_AA)
        else:
            cv2.putText(overlay, target.label, (p[0]+12,p[1]-12), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 1, cv2.LINE_AA)
    cv2.drawMarker(overlay, tuple(np.rint(principal).astype(int)), PRINCIPAL_COLOR, cv2.MARKER_CROSS, 70, 4)
    return overlay


def render_crop(image, target, principal):
    source = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image
    center = np.asarray(target.ring_uv)
    marker = np.asarray(target.marker)
    scale = np.mean([np.linalg.norm(marker[1]-marker[0]), np.linalg.norm(marker[3]-marker[0])]) / 4
    # 至少覆盖十字臂周围 6.5 mm，且完整包含相邻 ArUco。
    low = np.minimum(center - 6.5*scale, marker.min(axis=0) - scale)
    high = np.maximum(center + 6.5*scale, marker.max(axis=0) + scale)
    margin = 2
    x0, y0 = np.maximum(np.floor(low).astype(int)-margin, 0)
    x1, y1 = np.minimum(np.ceil(high).astype(int)+margin, [source.shape[1],source.shape[0]])
    if x1 <= x0 or y1 <= y0: raise ValueError('目标放大图范围无效')
    factor = 480. / (x1-x0)
    crop = cv2.resize(source[y0:y1,x0:x1], (480, max(1,round((y1-y0)*factor))), interpolation=cv2.INTER_CUBIC)
    def pixel(p): return tuple(np.rint((np.asarray(p)-[x0,y0])*factor).astype(int))
    points = np.rint((ellipse_points(target)-[x0,y0])*factor).astype(np.int32)
    cv2.polylines(crop, [points], True, RING_COLOR, 1, cv2.LINE_AA)
    cv2.circle(crop, pixel(target.ring_uv), 3, RING_COLOR, -1, cv2.LINE_AA)
    cv2.drawMarker(crop, pixel(target.cross_uv), CROSS_COLOR, cv2.MARKER_CROSS, 13, 1, cv2.LINE_AA)
    if x0 <= principal[0] < x1 and y0 <= principal[1] < y1:
        cv2.drawMarker(crop, pixel(principal), PRINCIPAL_COLOR, cv2.MARKER_CROSS, 20, 1, cv2.LINE_AA)
    # OpenCV 自带字体不支持中文。
    rgb = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(rgb)
    for path in ('/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc', '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'):
        try:
            font = ImageFont.truetype(path, 16); break
        except OSError: continue
    else: font = ImageFont.load_default()
    label = f'{target.label} 圆环-十字差 {target.cross_error_mm:.3f}mm'
    draw.rectangle((0,0,480,23), fill='white')
    draw.text((4,2), label, font=font, fill=(0,0,0))
    return cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)


class ToolOffsetWorker(AsyncRequestWorker):
    name = 'tooloffset'

    def _execute_triangulate_shot(self, payload):
        """在后台线程检测 3 帧主相机标定尖并保存原图与诊断叠加图。"""
        from nest_clear.tooloffset.ref_tip_top import RefTipTopDetector, render_ref_tip_top_overlay

        ctx, session = payload['context'], payload['session']
        frames = payload['frames']
        shot_idx = int(payload['shot_index'])
        cfg_tri = ctx.config.tool_offset['triangulation']
        detector = RefTipTopDetector(ctx.config.tool_offset['ref_tip_top'])

        raw_paths = []
        results = []
        rep_image = None
        for f_idx, item in enumerate(frames, 1):
            image, _ = item if isinstance(item, tuple) else (item, {})
            if rep_image is None:
                rep_image = image
            raw_rel = None
            if ctx.events.run_directory is not None:
                raw_file = ctx.events.run_directory / 'tooloffset' / session['directory'] / f'tri_s{shot_idx:02d}_f{f_idx}_raw.png'
                raw_file.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(raw_file), image):
                    raise OSError(f'保存三角定位原图失败：{raw_file}')
                raw_rel = str(raw_file.relative_to(ctx.events.directory))
            raw_paths.append(raw_rel)
            results.append(detector.detect(image))

        principal = (float(ctx.calibration.K[0, 2]), float(ctx.calibration.K[1, 2])) if ctx.calibration is not None else None
        rep_res = results[0]
        overlay = render_ref_tip_top_overlay(
            rep_image,
            rep_res,
            shot_index=shot_idx,
            principal=principal,
            margin_frac=cfg_tri['center_margin_frac'],
        )
        ovl_rel = None
        if ctx.events.run_directory is not None:
            ovl_file = ctx.events.run_directory / 'tooloffset' / session['directory'] / f'tri_s{shot_idx:02d}_ovl.png'
            ovl_file.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(ovl_file), overlay):
                raise OSError(f'保存三角定位叠加图失败：{ovl_file}')
            ovl_rel = str(ovl_file.relative_to(ctx.events.directory))
        write_live(ctx.events.directory, 'tooloffset_latest', overlay)
        if 1 <= shot_idx <= 5:
            write_live(ctx.events.directory, f'tooloffset_tri_{shot_idx}', overlay)

        frame_records = []
        reasons = []
        pts = []
        for f_idx, res in enumerate(results, 1):
            frame_records.append({
                'frame': f_idx,
                'uv': list(res.point_uv) if res.point_uv is not None else None,
                'quality': res.quality,
                'focused': res.focused,
                'contrast': res.contrast,
                'edge_rms_px': res.edge_rms_px,
                'apex_angle_deg': res.apex_angle_deg,
                'diagnostic': res.diagnostic,
            })
            if res.point_uv is None or not res.focused:
                reasons.append(f'第 {f_idx} 帧{res.diagnostic}')
            else:
                pts.append(res.point_uv)

        std_uv = None
        median_uv = None
        if not reasons:
            arr = np.asarray(pts, dtype=float)
            std = np.std(arr, axis=0)
            std_uv = (float(std[0]), float(std[1]))
            median_uv = tuple(float(v) for v in np.median(arr, axis=0))
            max_std = float(cfg_tri['frame_std_max_px'])
            for axis_name, val in zip(('u', 'v'), std_uv):
                if val > max_std:
                    reasons.append(f'三帧 {axis_name} 标准差 {val:.3f} > {max_std:g} px（疑似未停稳）')

        h, w = rep_image.shape[:2]
        return {
            'ok': not reasons,
            'reason': '；'.join(reasons),
            'uv': median_uv,
            'std_uv': std_uv,
            'quality': float(np.median([r.quality for r in results])) if results else 0.0,
            'frames': frame_records,
            'raw_paths': raw_paths,
            'raw_path': raw_paths[0] if raw_paths else None,
            'overlay_path': ovl_rel,
            'size': (w, h),
        }

    def _execute_align_step(self, payload):
        """在后台线程保存对刀原始图像、同时检测五帧工具尖与标定尖，并计算本轮对准/防撞结果。"""
        from nest_clear.tooloffset.alignment import predict_collision_gap_mm, solve_view_delta
        from nest_clear.tooloffset.ref_tip_side import measure_dual_tips, render_tooloffset_side_overlay

        ctx, session = payload['context'], payload['session']
        images = payload['images']
        step_idx = int(payload['step_index'])
        align_session = payload['align_session']
        actual = tuple(float(v) for v in payload['actual'])
        align_cfg = ctx.config.tool_offset['align']
        ref_cfg = ctx.config.tool_offset['ref_tip_side']

        raw_paths = {}
        rep_images = {}
        try:
            for view in ('front', 'side'):
                view_frames = list(images.frames.get(view, ()))
                if not view_frames:
                    view_frames = [images[view]]
                rep_images[view] = view_frames[0]
                raw_paths[view] = []
                for f_idx, frame in enumerate(view_frames, 1):
                    raw_rel = None
                    if ctx.events.run_directory is not None:
                        raw_file = (
                            ctx.events.run_directory
                            / 'tooloffset'
                            / session['directory']
                            / f'align_s{step_idx:02d}_{view}_f{f_idx}_raw.png'
                        )
                        raw_file.parent.mkdir(parents=True, exist_ok=True)
                        if not cv2.imwrite(str(raw_file), frame):
                            raise OSError(f'保存对刀原图失败：{raw_file}')
                        raw_rel = str(raw_file.relative_to(ctx.events.directory))
                    raw_paths[view].append(raw_rel)

            dual = measure_dual_tips(
                images.frames,
                ctx.config.tool_calibration.detection,
                ref_cfg,
                std_max=float(align_cfg.get('frame_std_max_px', 0.3)),
                capture_times=images.capture_times,
            )
        finally:
            images.frames.clear()

        decision = None
        if dual['ok']:
            decision = align_session.step(actual, dual['tool_points'], dual['ref_points'])

        ovl_paths = {}
        per_view_delta = {}
        per_view_gap = {}
        jacs = align_session.jacobians
        for view in ('front', 'side'):
            t_uv = dual['tool_points'].get(view)
            r_uv = dual['ref_points'].get(view)
            z_dir = None
            delta_v = None
            gap_v = None
            if view in jacs and t_uv is not None and r_uv is not None:
                J = jacs[view]
                z_dir = (float(J[0, 2]), float(J[1, 2]))
                xy_col = 0 if view == 'front' else 1
                try:
                    dv, _, gv = solve_view_delta(t_uv, r_uv, J[:, xy_col], J[:, 2])
                    delta_v, gap_v = dv, gv
                except ValueError:
                    pass
            elif t_uv is not None and r_uv is not None:
                _, per_g = predict_collision_gap_mm(
                    dual['tool_points'],
                    dual['ref_points'],
                    jacs,
                    fallback_um_per_px=float(align_cfg['fallback_um_per_px']),
                )
                gap_v = per_g.get(view)
            per_view_delta[view] = delta_v
            per_view_gap[view] = gap_v
            status_text = '对准'
            if not dual['ok']:
                status_text = '识别失败'
            elif decision and decision['status'] == 'collision':
                status_text = '会碰到标定尖'
            elif decision and decision['status'] == 'converged':
                status_text = '已对准'
            elif not jacs:
                status_text = f'测姿 {min(4, max(1, align_session.probe_index))}/4'
            ovl = render_tooloffset_side_overlay(
                rep_images[view],
                view,
                tool_uv=t_uv,
                ref_uv=r_uv,
                ref_result=dual['ref_reps'].get(view),
                z_dir_uv=z_dir,
                delta_mm=delta_v,
                gap_mm=gap_v,
                status_text=status_text,
            )
            ovl_rel = None
            if ctx.events.run_directory is not None:
                ovl_file = (
                    ctx.events.run_directory
                    / 'tooloffset'
                    / session['directory']
                    / f'align_s{step_idx:02d}_{view}_ovl.png'
                )
                ovl_file.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(ovl_file), ovl):
                    raise OSError(f'保存对刀叠加图失败：{ovl_file}')
                ovl_rel = str(ovl_file.relative_to(ctx.events.directory))
            ovl_paths[view] = ovl_rel
            write_live(ctx.events.directory, f'tooloffset_{view}_latest', ovl)

        return {
            'ok': bool(dual['ok']),
            'reason': dual['reason'],
            'tool_points': {k: list(v) for k, v in dual['tool_points'].items()},
            'ref_points': {k: list(v) for k, v in dual['ref_points'].items()},
            'raw_paths': raw_paths,
            'overlay_paths': ovl_paths,
            'decision': decision,
            'per_view_delta': per_view_delta,
            'per_view_gap': per_view_gap,
            'tool_measurement': dual['tool_measurement'],
            'ref_measurement': dual['ref_measurement'],
        }

    def execute(self, payload):
        if payload.get('operation') == 'triangulate_shot':
            return self._execute_triangulate_shot(payload)
        if payload.get('operation') == 'align_step':
            return self._execute_align_step(payload)
        captured, ctx, session = payload['captured'], payload['context'], payload['session']
        image, meta = captured if isinstance(captured, tuple) else (captured, {})
        session['shots'] += 1
        raw = None
        if ctx.events.run_directory is not None:
            path = ctx.events.run_directory / session.get('kind', 'tooloffset') / session['directory'] / f"s{session['shots']:03d}_raw.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(path), image): raise OSError(f'保存工具偏移原图失败：{path}')
            raw = str(path.relative_to(ctx.events.directory))
        targets = detect(image)
        principal = (float(ctx.calibration.K[0,2]), float(ctx.calibration.K[1,2]))
        prefix = session.get('kind', 'tooloffset')
        annotated = render_annotations(image, targets, session['engine'].current, principal)
        if prefix == 'pickacc' and payload.get('pose') is not None:
            from nest_clear.vision.geometry import pixel_to_module
            cal = session['engine'].cal
            pose = np.asarray(payload['pose'])
            center = np.asarray(pixel_to_module(*principal, pose, cal))
            half = session['engine'].fine_half
            corners = np.asarray([center + [x, y, 0] for x, y in ((-half,-half),(half,-half),(half,half),(-half,half))])
            camera = (corners - (pose + cal.translation)) @ cal.rotation
            uv, _ = cv2.projectPoints(camera, np.zeros(3), np.zeros(3), cal.K, cal.distortion)
            cv2.polylines(annotated, [np.rint(uv.reshape(-1,2)).astype(np.int32)], True, (255,150,0), 4, cv2.LINE_AA)
        write_live(ctx.events.directory, prefix + '_latest', annotated)
        current = next((t for t in targets if t.index == session['engine'].current), None)
        if current is not None:
            write_live(ctx.events.directory, prefix + '_crop', render_crop(image, current, principal))
        details = [{'target': t.label, 'ring_uv': t.ring_uv, 'cross_uv': t.cross_uv,
                    'cross_error_mm': t.cross_error_mm, 'marker': t.marker} for t in targets]
        ctx.events.emit(prefix + '_detection', task_id=payload['task_id'], raw_path=raw, targets=details)
        return {'targets': targets, 'size': (image.shape[1], image.shape[0]), 'raw_path': raw}
