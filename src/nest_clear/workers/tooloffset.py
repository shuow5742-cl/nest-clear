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

    def execute(self, payload):
        if payload.get('operation') == 'triangulate_shot':
            return self._execute_triangulate_shot(payload)
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
