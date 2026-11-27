"""生成有代表性的演示数据并启动界面。"""
import argparse
import json
import math
import os
import subprocess
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from nest_clear.paths import supervisor_state_file, main_status_file, live_image_file, tooltip_preview_file


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
    os.replace(tmp, path)


def pickacc_demo(state, stamp):
    """演示数据按菲林目标、实际点序和各测量阶段生成。"""
    import csv
    path = Path(__file__).resolve().parents[3] / 'docs/pick_test_film/targets.csv'
    with path.open(encoding='utf-8', newline='') as stream:
        nominal = {int(row['index']): (float(row['x_mm']), float(row['y_mm'])) for row in csv.DictReader(stream)}
    angle = .02
    def position(index):
        x,y = nominal[index]
        return [130 + 1.005*(math.cos(angle)*x-math.sin(angle)*y),
                70 + 1.005*(math.sin(angle)*x+math.cos(angle)*y)]
    first = (2,3,5,10,12,21,26,28,29)
    table = (2,3,4,5,10,11,12,13,18,19,20,21,26,27,28,29)
    simple = (2,5,26,29,12)
    tool = 2 if state == 'tool_in_place' else 1
    samples=[]
    def z_for(index, xy):
        if state == 'table':
            return round(-10.02 - 0.0025 * (xy[0] - 85.0) - 0.0012 * (xy[1] - 10.0), 3)
        return round(-10.02 - 0.006 * (index % 7), 3)
    def add(index, stage, *, corner=None, retest=False, verification=False):
        xy=position(index); number=len(samples)
        residual=[.012,.008] if verification else [.02+(number % 3)*.02, -.02]
        sent=[xy[0]+1,xy[1]-1,-10]
        actual=[sent[0]+residual[0],sent[1]+residual[1],-10]
        samples.append({'target':f'T{index:02d}', 'target_xy_mm':xy, 'sent_xyz_mm':sent,
                        'actual_xyz_mm':actual, 'residual_xy_mm':residual, 'stage':stage,
                        'corner':corner, 'retest':retest, 'verification':verification,
                        'pick_z_mm': z_for(index, xy)})
    pending_sample = None
    if state == 'wait_tool':
        for index in first[:4]: add(index,'A')
        pending=first[4:]; current=12
    elif state == 'tool_in_place':
        for index in simple[:4]: add(index,'simple')
        pending=simple[4:]; current=12
    elif state == 'z_step':
        for index in first[:4]: add(index,'A')
        pending=first[5:]; current=12
        xy = position(12)
        pending_sample = {'target':'T12', 'target_xy_mm':xy, 'sent_xyz_mm':[xy[0]+1,xy[1]-1,-10],
                          'actual_xyz_mm':[xy[0]+1.03,xy[1]-1.02,-10], 'residual_xy_mm':[.03,-.02],
                          'stage':'A', 'corner':None, 'retest':False, 'verification':False}
    elif state in ('table','confirm','constant'):
        for index in first: add(index,'A')
        for corner in ((-1,-1),(-1,1),(1,-1),(1,1)): add(12,'infield',corner=corner)
        if state == 'table':
            for index in table:
                if index not in first: add(index,'B')
            for index in (2,29,12): add(index,'repeat',retest=True)
        for index in (19,11): add(index,'verify',verification=True)
        pending=(); current=12
    else:
        pending=first; current=None
    phases={'select':('select','请按【开始】'), 'wait_tool':('wait_tool','请等待工具前往'),
            'tool_in_place':('tool_in_place','工具已到位，调整好后按【计算补偿】'),
            'z_step':('z_step','XY 已记录；请放 0.05mm 塞尺并点动工具尖刚好压住塞尺，按【记录下针 Z】'),
            'constant':('done','保存成功（全盘下针 Z 最大值）'),
            'table':('done','保存成功（挑取 Z 表）'), 'confirm':('confirm','等待确认覆盖')}
    phase,message=phases[state]
    infield=[{'corner':corner,'distance_mm':distance,'warning':distance>.05} for corner,distance in zip(((-1,-1),(-1,1),(1,-1),(1,1)),(.021,.028,.070,.032))] if state in ('table','confirm','constant') else []
    candidate={}
    z_rows=[s for s in samples if not s.get('discarded') and s.get('pick_z_mm') is not None]
    z_summary={}
    if z_rows:
        grouped={}
        for s in z_rows: grouped.setdefault(s['target'],[]).append(s)
        z_nodes=[]
        for tgt in sorted(grouped):
            rows=grouped[tgt]
            z_nodes.append({'target':tgt,'x':sum(r['target_xy_mm'][0] for r in rows)/len(rows),'y':sum(r['target_xy_mm'][1] for r in rows)/len(rows),'z':sum(r['pick_z_mm'] for r in rows)/len(rows),'n':len(rows)})
        zs=[n['z'] for n in z_nodes]; z_min,z_max=min(zs),max(zs); z_rng=z_max-z_min
        z_dec={'z_type':'constant' if z_rng<=.1 else 'table','min_z_mm':z_min,'max_z_mm':z_max,'range_mm':z_rng,'threshold_mm':.1,'point_count':len(z_nodes)}
        z_summary={'z_type':'constant','z_mm':z_max,'z_decision':z_dec} if z_rng<=.1 else {'z_type':'table','z_nodes':z_nodes,'z_decision':z_dec}
    if state=='table':
        nodes=[]
        for index in table:
            sample=next(s for s in samples if s['target']==f'T{index:02d}' and s['stage'] in ('A','B'))
            xy=sample['target_xy_mm']; r=sample['residual_xy_mm']
            nodes.append({'x':xy[0],'y':xy[1],'dx':r[0],'dy':r[1]})
        candidate={'type':'table','nodes':nodes,**z_summary,'z_ref':0.0}
    elif state in ('confirm','constant'):
        candidate={'type':'constant','constant_xy':[.081,-.048],**z_summary,'z_ref':0.0,'verification':[s for s in samples if s['verification']]}
    existing_tool2 = {'type':'table','time':stamp,'verification':[{}],'stale':True}
    if state != 'z_step':
        existing_tool2.update(z_type='constant', z_mm=-10.050, z_calibrated=True, z_ref=0.0)
    return {'phase':phase, 'tool':tool, 'plc_tool_id':tool, 'message':message,
            'target':f'T{current:02d}' if current is not None else None,
            'predicted':{f'T{i:02d}':position(i) for i in set(table) | set(simple)},
            'queue':[{'target':f'T{i:02d}','stage':'simple' if tool==2 else 'A'} for i in pending],
            'samples':samples, 'pending_sample':pending_sample,
            'z_jitter_mm': 0.009 if state in ('z_step','constant','table','confirm') else None,
            'z_summary': z_summary,
            'infield':infield, 'decision':({'max_dev_mm':.071,'std_mm':.027} if state=='table' else {'max_dev_mm':.022,'std_mm':.010}) if candidate else {},
            'existing':{'tool1':{'type':'constant','constant_xy':[.08,-.05],'z_type':'constant','z_mm':-10.020,'z_calibrated':True,'z_ref':0.0,'time':stamp,'verification':[{},{}]},
                        'tool2':existing_tool2}, 'candidate':candidate}


def _draw_offset_demo_images(root: Path, offset_state: str, step: int = 0) -> None:
    """用纯 PIL 绘制新协议工具偏移标定的主相机三角定位缩略图与对刀相机叠加图（无 cv2/numpy 依赖）。"""
    views_count = 2 if offset_state == 'triangulate' else 3
    for idx in range(1, 6):
        path = live_image_file(root, f'tooloffset_tri_{idx}')
        path.parent.mkdir(parents=True, exist_ok=True)
        if idx > views_count:
            path.unlink(missing_ok=True)
            continue
        img = Image.new('RGB', (1006, 759), (232, 236, 240))
        draw = ImageDraw.Draw(img)
        # 中央 80% 安全区域框
        draw.rectangle((100, 76, 906, 683), outline=(40, 160, 200), width=3)
        cx, cy = 460 + idx * 55, 360 + (idx - 2) * 35
        # 斜视锥体深色剪影及两侧拟合线
        draw.polygon([(cx, cy), (cx - 125, cy - 195), (cx - 45, cy - 230)], fill=(38, 42, 48))
        draw.line((cx - 120, cy - 187, cx, cy), fill=(0, 220, 255), width=3)
        draw.line((cx - 43, cy - 220, cx, cy), fill=(0, 220, 255), width=3)
        draw.ellipse((cx - 10, cy - 10, cx + 10, cy + 10), outline=(255, 60, 0), width=3)
        draw.line((cx - 18, cy, cx + 18, cy), fill=(255, 60, 0), width=2)
        draw.line((cx, cy - 18, cx, cy + 18), fill=(255, 60, 0), width=2)
        draw.rectangle((16, 16, 440, 78), fill=(255, 255, 255), outline=(200, 210, 220))
        draw.text((28, 26), f'Shot #{idx}  RefTip=({cx:.1f}, {cy:.1f}) px  Q=0.92', fill=(20, 120, 60))
        tmp = path.with_suffix('.tmp')
        img.save(tmp, format='JPEG')
        os.replace(tmp, path)

    banner_font = ImageFont.truetype('DejaVuSans.ttf', 28)
    for view in ('front', 'side'):
        path = live_image_file(root, f'tooloffset_{view}_latest')
        path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.new('RGB', (640, 480), (242, 244, 247))
        draw = ImageDraw.Draw(img)
        view_name = 'Front' if view == 'front' else 'Side'
        banner_color = (90, 100, 115)
        if offset_state == 'triangulate':
            info = f'{view_name} | Awaiting PLC 100/110'
        else:
            tx = 335 if (offset_state == 'align' and view == 'front') else 312 if offset_state == 'align' else 320
            ty = 265 if offset_state == 'align' else 245 if offset_state == 'done' else 362
            rx, ry = 320, 395
            # 顶部三棱针剪影
            draw.polygon([(tx - 16, 0), (tx + 16, 0), (tx + 12, ty - 48), (tx - 12, ty - 48)], fill=(30, 34, 40))
            for off, ln in ((-8, 32), (0, 48), (8, 34)):
                draw.rectangle((tx + off - 2, ty - 48, tx + off + 2, ty - 48 + ln), fill=(30, 34, 40))
            # 底部朝上标定锥
            draw.polygon([(rx, ry), (rx - 28, 480), (rx + 28, 480)], fill=(25, 28, 34))
            # 过工具尖沿 Z 方向直线（略倾斜 3°）与交点标记
            slope = 0.052 if view == 'front' else -0.052
            draw.line((tx - 240 * slope, 0, tx + 240 * slope, 480), fill=(0, 165, 255), width=2)
            draw.line((tx - 12, ty, tx + 12, ty), fill=(0, 185, 65), width=3)
            draw.line((tx, ty - 12, tx, ty + 12), fill=(0, 185, 65), width=3)
            draw.ellipse((rx - 8, ry - 8, rx + 8, ry + 8), outline=(240, 50, 20), width=3)
            draw.line((tx, ty, rx, ry), fill=(190, 70, 210), width=1)
            banner_color = (185, 28, 28) if offset_state == 'failed' else (22, 130, 65) if offset_state == 'done' else (25, 40, 60)
            info = (
                f"{'Front dX=+0.082mm' if view == 'front' else 'Side dY=-0.058mm'} | Gap=2.15mm"
                if offset_state == 'align'
                else f"{'Front dX=+0.012mm' if view == 'front' else 'Side dY=-0.009mm'} | Gap=3.20mm\nAligned"
                if offset_state == 'done'
                else f"{view_name} | Gap=0.72mm < 1.00mm\nCollision"
            )
        _, _, _, bottom = draw.multiline_textbbox((10, 5), info, font=banner_font)
        draw.rectangle((0, 0, 640, bottom + 5), fill=(255, 255, 255))
        draw.multiline_text((10, 5), info, font=banner_font, fill=banner_color)
        tmp = path.with_suffix('.tmp')
        img.save(tmp, format='JPEG')
        os.replace(tmp, path)


def _draw_depthcal_demo_roi(root: Path, depthcal_state: str, z_val: float, label: str) -> None:
    """用纯 PIL 绘制下针深度标定 ROI 实时缩略图（无 cv2/numpy 依赖）。"""
    path = live_image_file(root, 'depthcal_roi')
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new('RGB', (280, 280), (226, 230, 236))
    draw = ImageDraw.Draw(img)
    ring_color = (28, 32, 38) if depthcal_state in ('fine', 'done') else (85, 92, 102)
    draw.ellipse((96, 96, 184, 184), outline=ring_color, width=6)
    draw.line((55, 140, 225, 140), fill=ring_color, width=2)
    draw.line((140, 55, 140, 225), fill=ring_color, width=2)
    draw.rectangle((4, 4, 276, 30), fill=(255, 255, 255), outline=(200, 210, 220))
    draw.text((10, 10), f'{label}  Z={z_val:.3f}mm  {depthcal_state}', fill=(20, 40, 80))
    tmp = path.with_suffix('.tmp')
    img.save(tmp, format='JPEG')
    os.replace(tmp, path)


def depthcal_demo(state: str, step: int = 0) -> tuple[dict, float, str]:
    """生成下针深度标定页演示状态（粗扫中/细扫中/完成/失败）。"""
    plan = ['T02', 'T05', 'T29', 'T26', 'T12']
    coords = {
        'T02': [96.0, 25.0],
        'T05': [96.0, 115.0],
        'T29': [164.0, 115.0],
        'T26': [164.0, 25.0],
        'T12': [130.0, 70.0],
    }
    true_peaks = {
        'T02': -5.032,
        'T05': -5.028,
        'T29': -4.994,
        'T26': -4.998,
        'T12': -5.013,
    }

    def curve_val(z: float, z0: float, amp: float = 0.85, sigma: float = 0.85) -> float:
        return round(0.04 + amp * math.exp(-0.5 * ((z - z0) / sigma) ** 2), 4)

    def make_coarse(lbl: str, limit: int | None = None):
        z0 = true_peaks[lbl]
        zs = [-2.0 - 0.2 * i for i in range(26)]
        if limit is not None:
            zs = zs[:limit]
        return [[round(z, 3), curve_val(z, z0, amp=0.84, sigma=0.85)] for z in zs]

    def make_fine(lbl: str, offset: float, limit: int | None = None):
        z0 = true_peaks[lbl] + offset
        zs = [round(true_peaks[lbl] + 1.50 - 0.075 * i, 4) for i in range(41)]
        if limit is not None:
            zs = zs[:limit]
        return [[z, curve_val(z, z0, amp=0.92, sigma=0.85)] for z in zs]

    def make_fine_plan():
        return {
            'mode': 'auto', 'configured_mode': 'auto', 'fallback': False, 'fallback_reason': None,
            'coarse_fwhm_mm': 2.002, 'half_width_mm': 1.500, 'effective_half_width_mm': 1.500,
            'lower_half_mm': 1.500, 'upper_half_mm': 1.500, 'fine_step_mm': 0.075,
            'truncated': False, 'truncation_note': None,
        }

    def make_result(lbl: str):
        offsets = (-0.0018, 0.0012, 0.0005)
        peaks = [round(true_peaks[lbl] + off, 5) for off in offsets]
        mean_pk = sum(peaks) / len(peaks)
        rng = max(peaks) - min(peaks)
        var = sum((p - mean_pk) ** 2 for p in peaks) / (len(peaks) - 1)
        fits = [{'repeat': i + 1, 'z_peak': p, 'fwhm': 1.985 + i * 0.008, 'r2': 0.9974 - i * 0.0003,
                 'quality_ok': True, 'insufficient_points': False}
                for i, p in enumerate(peaks)]
        return {'label': lbl, 'xy': coords[lbl], 'fits': fits, 'fine_plan': make_fine_plan(),
                'insufficient_repeats': [], 'z_peak': mean_pk, 'range': rng, 'std': math.sqrt(var)}

    curves = {}
    results = []
    plane = None
    initial_plane = None
    feasibility = None

    if state == 'coarse':
        lbl = 'T02'
        curves[lbl] = {'coarse': make_coarse(lbl, limit=18)}
        z_cur = curves[lbl]['coarse'][-1][0]
        return ({
            'phase': 'coarse', 'message': 'T02 粗扫中（Z 向下步进 0.20 mm）',
            'label': lbl, 'z': z_cur, 'point': 1, 'total': 5, 'plan': plan,
            'plane': None, 'initial_plane': None, 'results': [], 'feasibility': None,
            'metric': 'tenengrad', 'method': 'gauss', 'fine_plan': None, 'session': 'demo_depthcal',
            'curves': curves, 'abort_requested': False,
        }, z_cur, lbl)

    if state == 'fine':
        for done_lbl in ('T02', 'T05'):
            curves[done_lbl] = {
                'coarse': make_coarse(done_lbl),
                '1': make_fine(done_lbl, -0.0018),
                '2': make_fine(done_lbl, 0.0012),
                '3': make_fine(done_lbl, 0.0005),
            }
            results.append(make_result(done_lbl))
        lbl = 'T29'
        curves[lbl] = {
            'coarse': make_coarse(lbl),
            '1': make_fine(lbl, -0.0014),
            '2': make_fine(lbl, 0.0011, limit=25),
        }
        z_cur = curves[lbl]['2'][-1][0]
        return ({
            'phase': 'fine', 'message': 'T29 细扫 第 2/3 次（单向向下 0.075 mm）',
            'label': lbl, 'z': z_cur, 'point': 3, 'total': 5, 'plan': plan,
            'plane': None, 'initial_plane': None, 'results': results, 'feasibility': None,
            'metric': 'tenengrad', 'method': 'gauss', 'fine_plan': make_fine_plan(), 'session': 'demo_depthcal',
            'curves': curves, 'abort_requested': False,
        }, z_cur, lbl)

    if state == 'done':
        for lbl in plan:
            curves[lbl] = {
                'coarse': make_coarse(lbl),
                '1': make_fine(lbl, -0.0018),
                '2': make_fine(lbl, 0.0012),
                '3': make_fine(lbl, 0.0005),
            }
            results.append(make_result(lbl))
        residuals = {'T02': -0.0012, 'T05': 0.0015, 'T29': -0.0009, 'T26': 0.0011, 'T12': -0.0005}
        plane = {'a': 0.00050, 'b': 0.00004, 'c': -5.0808, 'residuals': residuals,
                 'max_residual': 0.0015, 'tilt_mm_per_100mm': 0.0502}
        initial_plane = plane
        feasibility = {'max_range': 0.0030, 'max_std': 0.0016, 'threshold': 0.02,
                       'insufficient_fits': 0, 'insufficient_items': [], 'total_fits': 15, 'met': True}
        z_cur = results[-1]['z_peak']
        return ({
            'phase': 'done', 'message': '料盘多点对焦测试完成（已回 161）',
            'label': 'T12', 'z': z_cur, 'point': 5, 'total': 5, 'plan': plan,
            'plane': plane, 'initial_plane': initial_plane, 'results': results, 'feasibility': feasibility,
            'metric': 'tenengrad', 'method': 'gauss', 'fine_plan': make_fine_plan(), 'session': 'demo_depthcal',
            'curves': curves, 'abort_requested': False,
        }, z_cur, 'T12')

    # failed
    curves['T02'] = {
        'coarse': make_coarse('T02'),
        '1': make_fine('T02', -0.0018),
        '2': make_fine('T02', 0.0012),
        '3': make_fine('T02', 0.0005),
    }
    results.append(make_result('T02'))
    curves['T05'] = {'coarse': [[round(-8.0 - 0.2 * i, 3), round(0.12 + 0.03 * i, 4)] for i in range(21)]}
    z_cur = -12.0
    return ({
        'phase': 'failed',
        'message': '对焦峰不在扫描范围内，最大清晰度在 Z−（向下）边界 Z=-12.0000 mm',
        'reason_code': 'DEPTH-PEAK-BOUNDARY',
        'what': '对焦峰不在扫描范围内，最大清晰度在 Z−（向下）边界 Z=-12.0000 mm',
        'where': 'T05 粗扫',
        'hint': '核查工具间隙及安全 Z 下限，再评估是否扩大向下的可用扫描范围。',
        'label': 'T05', 'z': z_cur, 'point': 2, 'total': 5, 'plan': plan,
        'plane': None, 'initial_plane': None, 'results': results, 'feasibility': None,
        'metric': 'tenengrad', 'method': 'gauss', 'fine_plan': None, 'session': 'demo_depthcal',
        'curves': curves, 'abort_requested': False,
    }, z_cur, 'T05')


def write_demo(root, step=0, *, initial=False, preview_state="on", offset_state=None, pickacc_state=None, baseline_state=None, tooltip_state=None, focus_state=None, camcal_state=None, depthcal_state=None, wb_state=None, temporary_exposure=False, film_mode=False):
    root = Path(root)
    from nest_clear.film_mode import write_mode
    write_mode(root, film_mode)
    now = datetime.now().astimezone(); stamp = now.isoformat()
    start = (now - timedelta(hours=2, minutes=18)).isoformat()
    points = [[x, y, z] for z in (-5, -2.5, 0, 2.5, 5) for x in (-20, -10, 0, 10, 20) for y in (-20, -10, 0, 10, 20)]
    sequence = 63 + step % 5
    atomic(supervisor_state_file(root), {'pid': os.getpid(), 'start_time': start, 'word': 0x126,
        'plc_connected': True, 'plc_last_read': stamp, 'heartbeat_last_write': stamp,
        'algorithm_ready': 1, 'child_pid': os.getpid() if preview_state == 'blocked' else None, 'child_state': '运行' if preview_state == 'blocked' else '未启动', 'child_start_time': start,
        'preview': {'on': preview_state == 'on', 'seq': 1, 'fps': 6.0, 'frames': 36, 'actual_xyz': [125.2, 73.4, 110.0], 'sharpness': 568.3, 'corners': 124, 'board_offset_mm': [0.3, -0.2]},
        'child_last_exit_reason': None})
    history = [
        {'event':'tooltip_detect', 'delta':[0.12, -0.08, 0.04], 'focus':{
            'front':{'edge_width_px':2.4,'focused':True},'side':{'edge_width_px':2.8,'focused':True}}},
        {'event':'tooltip_reliability','reliable':True,'reasons':[]},
        {'event':'tooltip_detect', 'delta':[0.02, -0.01, 0.01], 'focus':{
            'front':{'edge_width_px':2.2,'focused':True},'side':{'edge_width_px':2.5,'focused':True}}},
        {'event':'tooltip_reliability','reliable':True,'reasons':[]},
    ]
    atomic(main_status_file(root), {'mode': '相机标定中', 'picking': {
        'tray':1,'view_index':10,'total_views':24,'coarse_shots':3,'removable_count':5,'red_count':1,'sent_count':7},
        'last_reply_time':stamp,'last_reply_task_id':107,'last_reply_request_func':1,'last_reply_func':1,
        'recent_replies': [{'time': (now - timedelta(seconds=i)).isoformat(), 'task_id': task,
                            'request_func': request, 'reply_func': reply}
                           for i, (task, request, reply) in enumerate(((107, 1, 1), (106, 60, 60), (105, 3, 3), (104, 20, 9), (103, 80, 82)))],
        'camcal':{'event': 'camcal_step' if preview_state == 'blocked' else 'camcal_done', 'sequence':sequence,'plan':points,'start_xyz':[0,0,0],'kind':'move','detail':{
            'detected_corners':35,'predicted_corners':40,'saturated_fraction':0.01,'bbox':[510,120,560,170]}},
        'tooltip':{'tool':1,'plc_tool_id':2,'iterations':2,'event':'tooltip_aligned','history':history,'new':[0.02,-0.01,0.01]},
        'command_seq':0, 'tooloffset':{'phase':'idle', 'tool':None, 'plc_tool_id':None,
            'message':'等待 PLC 请求 90 (40905=1)',
            'existing':{'tool1':{'saved_at':stamp,'offset_xy_mm':[10.480,-5.240],'current_tools_xy':[10.480,-5.240],'baseline_stale':True}, 'tool2':{}}}})
    if camcal_state:
        path = main_status_file(root); data = json.loads(path.read_text(encoding='utf-8'))
        data['camcal'].update(event='camcal_failed' if camcal_state == 'failed' else 'camcal_step',
                              kind='fail' if camcal_state == 'failed' else 'move')
        if camcal_state == 'failed':
            data['camcal'].update(code='CAMCAL-FLOW', reason='相机标定到位超差：期望 (105.185, 79.960, -14.904)，实际 (105.480, 80.120, -14.800)，公差 0.05 mm。请检查模组位置、运动状态及标定板，确认设备已到达目标位置后重新标定。', hint='检查模组运动、位置反馈和标定板，然后重新启动标定。')
        atomic(path, data)
    if offset_state:
        path = main_status_file(root)
        data = json.loads(path.read_text(encoding='utf-8'))
        data['mode'] = '空闲' if offset_state in ('done', 'failed') else '工具偏移标定中'
        data['plc_tool_id'] = 1
        d = data['tooloffset']
        views_3 = [
            {'index': 1, 'pose': [162.0, 70.0, 122.0], 'uv': [515.2, 360.4], 'quality': 0.91, 'ok': True},
            {'index': 2, 'pose': [174.5, 91.6, 122.0], 'uv': [570.4, 395.1], 'quality': 0.93, 'ok': True},
            {'index': 3, 'pose': [174.5, 48.4, 122.0], 'uv': [625.1, 325.8], 'quality': 0.90, 'ok': True},
        ]
        tri_done = {
            'views': views_3,
            'xyz': [92.008, 30.012, 11.994],
            'target_xy': [92.008, 30.012],
            'max_residual_mm': 0.0082,
            'rms_residual_mm': 0.0049,
            'max_baseline_ratio': 0.224,
            'max_angle_deg': 12.8,
        }
        jac_info = {
            'front': {'rms_px': 0.12, 'condition_number': 18.4},
            'side': {'rms_px': 0.14, 'condition_number': 19.1},
        }
        d.update(
            plc_tool_id=1,
            existing={
                'tool1': {'saved_at': stamp, 'offset_xy_mm': [10.480, -5.240], 'current_tools_xy': [10.480, -5.240], 'baseline_stale': True},
                'tool2': {'saved_at': stamp, 'offset_xy_mm': [-8.310, 6.420], 'current_tools_xy': [-8.310, 6.420], 'baseline_stale': False},
            },
            old_offset_xy=[10.480, -5.240],
        )
        if offset_state == 'triangulate':
            d.update(
                phase='triangulate',
                tool=None,
                message='三角定位 第 2/≥3 张完成，下发下一相机坐标 (174.50, 48.40, 122.00) 并回 90',
                triangulation={'views': views_3[:2], 'xyz': None, 'target_xy': None, 'max_residual_mm': None},
                alignment=None,
            )
        elif offset_state == 'align':
            d.update(
                phase='aligning',
                tool=1,
                message='工具 1 对准 第 2 轮（ΔX=+0.082, ΔY=-0.058, 间隙 2.15 mm）',
                triangulation=tri_done,
                p0=[101.850, 25.350, -6.800],
                alignment={
                    'probe_step': 4, 'align_iteration': 2, 'delta_xy_mm': [0.082, -0.058],
                    'gap_mm': 2.15, 'predicted_gap_mm': 2.14, 'jacobians': jac_info,
                },
            )
        elif offset_state == 'done':
            d.update(
                phase='done',
                tool=1,
                message='工具 1 偏移标定完成并已保存：offset=(+10.5020, -5.2080) mm',
                triangulation=tri_done,
                p0=[101.850, 25.350, -6.800],
                p_final=[102.498, 24.813, -6.800],
                plc_final=[102.498, 24.814, -6.800],
                offset_xy=[10.502, -5.208],
                alignment={
                    'probe_step': 4, 'align_iteration': 3, 'delta_xy_mm': [0.012, -0.009],
                    'gap_mm': 3.20, 'predicted_gap_mm': 3.20, 'jacobians': jac_info,
                },
                existing={
                    'tool1': {'saved_at': stamp, 'offset_xy_mm': [10.502, -5.208], 'current_tools_xy': [10.502, -5.208], 'baseline_stale': True},
                    'tool2': {'saved_at': stamp, 'offset_xy_mm': [-8.310, 6.420], 'current_tools_xy': [-8.310, 6.420], 'baseline_stale': False},
                },
            )
        elif offset_state == 'failed':
            d.update(
                phase='failed',
                tool=1,
                message='会碰到标定尖：移动后预测竖直间隙 0.720 mm 小于安全下限 1 mm',
                failure_code='TOOLOFFSET-COLLISION',
                failure_what='会碰到标定尖：移动后预测竖直间隙 0.720 mm 小于安全下限 1 mm',
                failure_where='工具 1 防撞检查（间隙 0.720 mm）',
                failure_hint='抬高工具 Z 坐标或调整工具装夹高度后再重新标定。',
                triangulation=tri_done,
                p0=[101.850, 25.350, -7.800],
                alignment={
                    'probe_step': 3, 'align_iteration': 0, 'delta_xy_mm': [0.180, -0.110],
                    'gap_mm': 2.18, 'predicted_gap_mm': 0.72, 'jacobians': None,
                },
            )
        atomic(path, data)
        if initial:
            _draw_offset_demo_images(root, offset_state, step)
    if pickacc_state:
        path = main_status_file(root); data = json.loads(path.read_text(encoding='utf-8'))
        data['mode']='挑取精度标定中'; data['pickacc']=pickacc_demo(pickacc_state,stamp)
        atomic(path,data)
        if initial:
            project=Path(__file__).resolve().parents[3]; env=dict(os.environ,PYTHONPATH=str(project/'src'))
            subprocess.run([str(project/'.venv/bin/python'),'-m','nest_clear.tools.offset_demo_image',str(root),'pickacc'],cwd=project,env=env,check=True,timeout=30)
    if baseline_state:
        if initial:
            project = Path(__file__).resolve().parents[3]; env = dict(os.environ, PYTHONPATH=str(project/'src'))
            subprocess.run([str(project/'.venv/bin/python'), '-m', 'nest_clear.tools.baseline_demo_image', str(root), 'live' if baseline_state.startswith('prepare') else baseline_state], cwd=project, env=env, check=True, timeout=30)
        path = main_status_file(root); data = json.loads(path.read_text(encoding='utf-8'))
        candidate = json.loads((root/'baseline_demo.json').read_text(encoding='utf-8'))
        from nest_clear.baseline.session import format_outside_message
        delta = candidate.get('delta_mm') or [0.0, 0.0]
        fail_msg = format_outside_message(delta)
        phases = {'select':('select','请选择工具'), 'live':('live','请点动针尖到主相机中心后按【对刀基准】'),
                  'success':('success','基准建立成功（待保存）'), 'fail':('fail', fail_msg), 'confirm':('confirm','请确认覆盖保存'),
                  'prepare':('prepare','请调焦后开始对刀基准建立'), 'prepare_error':('prepare','请检查相机连接'), 'prepare_unassigned':('idle','请分配相机')}
        phase,message = phases[baseline_state]
        data['mode']='对刀基准建立中'; data['baseline']={'phase':phase, 'message':message, 'tool':None if phase=='select' else 1,
            'candidate':candidate if phase in ('success','fail','confirm') else None, 'offset_xyz':None if phase=='select' else [1.2,-.3,.05],
            'notice':'提示：工具1 的工具偏移标定结果比现有对刀基准新（不阻止建基准）' if phase != 'select' else '',
            'live_delta':candidate['delta_mm'] if phase != 'select' else None, 'live_inside':candidate['inside'],
            'existing':{'tool1':{'time':stamp,'p0':[125,73,110],'offset_xyz':[1.2,-.3,.05],'pixels':{'front':[319,240],'side':[322,240]},'jacobian_pending':True,'stale':True,'offset_newer':True},
                        'tool2':{'time':stamp,'p0':[126,73,110],'offset_xyz':[2.1,.4,.02],'pixels':{'front':[320,240],'side':[321,240]},'jacobian_pending':True}}}
        if baseline_state.startswith('prepare'):
            def low_view(port):
                return {'on': True, 'device': f'/dev/v4l/by-path/pci-0000:00:14.0-usb-0:{port}:1.0-video-index0', 'exposure_source': '已测', 'exposure_limits': {'min': 1, 'max': 5000, 'step': 1}}
            def live_view(current, peak):
                percent = 100 * current / peak
                return {'percent': percent, 'tenengrad': current, 'peak': 1.60, 'peak_edge_width_um': 24.32, 'edge_width_um': 27.2, 'edge_width_px': 1.79, 'contrast': 210, 'source': 'tip', 'detection_status': '精定位', 'detection_reason': '', 'focus_reason': '', 'delta_uv': [12.0, -8.0], 'exposure_ms': 10.0, 'fps': 5.0, 'frame_seq': 20, 'frame_age_ms': 80, 'processing_ms': 45, 'tip_uv': [320, 240], 'history': [60, 65, 73, 82, 90, percent], 'updated_at': stamp}
            data['baseline']['tool'] = None
            data['baseline']['preview'] = {'front': low_view(4), 'side': low_view(8)}
            data['baseline']['detected_cameras'] = [data['baseline']['preview'][n]['device'] for n in ('front', 'side')]
            preview_cameras = {'front': live_view(1.28, 1.30), 'side': live_view(1.13, 1.30)}
            if baseline_state == 'prepare_error':
                data['baseline']['preview']['side']['error'] = 'USB 带宽不足'
                preview_cameras.pop('side', None)
                live_image_file(root, 'baseline_side').unlink(missing_ok=True)
            if baseline_state == 'prepare_unassigned':
                data['mode'] = '空闲'
                data['baseline']['preview'] = {'front': {'on': False}, 'side': {'on': False}}
                preview_cameras = {}
                for name in ('front', 'side'):
                    live_image_file(root, 'baseline_' + name).unlink(missing_ok=True)
            if preview_cameras:
                atomic(tooltip_preview_file(root), {'updated_at': stamp, 'cameras': preview_cameras})
            else:
                tooltip_preview_file(root).unlink(missing_ok=True)
        atomic(path,data)
    if tooltip_state:
        if initial:
            project = Path(__file__).resolve().parents[3]; env = dict(os.environ, PYTHONPATH=str(project/'src'))
            subprocess.run([str(project/'.venv/bin/python'), '-m', 'nest_clear.tools.tooltip_demo_image', str(root), tooltip_state], cwd=project, env=env, check=True, timeout=30)
        path = main_status_file(root); data = json.loads(path.read_text(encoding='utf-8'))
        data['tooltip'] = json.loads((root/'tooltip_demo.json').read_text(encoding='utf-8'))
        if data['tooltip'].get('tool'):
            data['tooltip']['plc_tool_id'] = 2
        data['mode'] = '空闲' if tooltip_state in ('idle', 'done', 'fail') else '对刀中(工具1)'
        atomic(path, data)
    if depthcal_state:
        path = main_status_file(root); data = json.loads(path.read_text(encoding='utf-8'))
        depth_data, z_val, lbl = depthcal_demo(depthcal_state, step)
        data['depthcal'] = depth_data
        data['mode'] = '空闲' if depthcal_state in ('done', 'failed') else '下针深度标定中'
        atomic(path, data)
        if initial:
            _draw_depthcal_demo_roi(root, depthcal_state, z_val, lbl)
    names = ('latest', 'camcal_latest', 'preview_latest', 'calib_front_latest', 'calib_side_latest')
    if tooltip_state: names = tuple(n for n in names if not n.startswith('calib_'))
    if not offset_state: names += ('tooloffset_latest', 'tooloffset_crop')
    for name in names:
        path = live_image_file(root, name); path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new('RGB', (1006, 759), '#253444'); draw = ImageDraw.Draw(image)
        draw.rectangle((100 + step % 20 * 3, 100, 600, 450), outline='#00ff88', width=4)
        for x in range(170, 590, 65):
            for y in range(150, 440, 60): draw.ellipse((x-4,y-4,x+4,y+4), fill='#46a7ff')
        draw.rectangle((510,120,560,170),outline='#ffdd55',width=3)
        if name == 'preview_latest':
            draw.line((503, 0, 503, 759), fill='#00ffff', width=2); draw.line((0, 379, 1006, 379), fill='#00ffff', width=2)
            draw.line((488, 379, 518, 379), fill='red', width=3); draw.line((503, 364, 503, 394), fill='red', width=3)
        draw.text((30, 30), f'{name}  frame {step}', fill='white')
        temp = path.with_suffix('.tmp'); image.save(temp, format='JPEG'); os.replace(temp, path)
    if focus_state in ('live', 'exposure'):
        import math
        import cv2
        import numpy as np
        from nest_clear.preview.focus import FocusSession, draw_focus
        if initial or not hasattr(write_demo, '_focus_session'):
            write_demo._focus_session = FocusSession()
            write_demo._focus_history = []
        main_path = main_status_file(root)
        main = json.loads(main_path.read_text(encoding='utf-8'))
        main['mode'] = '空闲'
        main['camcal'] = {}
        atomic(main_path, main)
        yy, xx = np.indices((759, 1006))
        light = (1.0 - 0.96 * xx / 1005)
        source = ((((xx // 32 + yy // 32) % 2) * 150 + 70) * light).astype(np.uint8)
        now_s = time.time()
        for tick in (range(-25, 1) if initial else (step,)):
            sigma = 1.75 + 1.25 * math.cos(2 * math.pi * tick / 20)
            board = cv2.GaussianBlur(source, (0, 0), sigma)
            if tick >= -3:
                patch = board[:330, :410]
                board[:330, :410] = cv2.GaussianBlur(patch, (0, 0), 3.0)
            data = write_demo._focus_session.update(board)
            timestamp = now_s + tick if initial else now_s
            write_demo._focus_history.append((timestamp, data['center']))
        write_demo._focus_history = [(t, v) for t, v in write_demo._focus_history if now_s-t <= 30]
        data.update(mode='focus', on=True, seq=1, fps=6, exposure_us=10000, gain_db=0,
                    analysis_time=now_s, demo_history=write_demo._focus_history)
        if temporary_exposure:
            data.update(exposure_us=20000, exposure_source='临时', temporary_exposure={'state': 'done', 'message': '曝光回读 20000 us'})
        if wb_state:
            metrics = {'r_g': 1.002, 'b_g': .998, 'warnings': []}
            data['white_balance'] = {'state': wb_state, 'balance_ratio': [1.2, 1.0, 1.4], 'metrics': metrics, 'reason': '画面过亮或饱和（疑似白纸），请用 18% 灰卡做白平衡'}
        if focus_state == 'exposure':
            if initial or not hasattr(write_demo, '_exposure_result'):
                from nest_clear.preview.exposure import demo_sweep
                write_demo._exposure_result = demo_sweep()
            data['exposure_sweep'] = write_demo._exposure_result
            data['demo_exposure'] = True
        supervisor_path = supervisor_state_file(root)
        supervisor = json.loads(supervisor_path.read_text(encoding='utf-8'))
        supervisor['preview'] = data
        atomic(supervisor_path, supervisor)
        ok, encoded = cv2.imencode('.jpg', draw_focus(board, data))
        if ok: live_image_file(root, 'preview_latest').write_bytes(encoded.tobytes())
    if initial:
        path = root / now.date().isoformat() / 'demo' / 'events.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        examples = [
            (999, 101, 'WORK-PROTOCOL', '请求功能码与当前状态不符', '请求分发', '检查 PLC 功能码和运行模式。'),
            (131, 102, 'TOOLTIP-FAILED', '正面相机边缘不清晰', '换针对刀', '检查正面相机焦距及针尖。'),
            (82, 103, 'CAMCAL-GLARE', '标定板角点附近检测到亮斑', '主相机标定', '调整照明角度后重新标定。'),
        ]
        with path.open('a', encoding='utf-8') as file:
            for code, task, reason_code, what, where, hint in examples:
                row = {'wall':stamp, 'event':'reply', 'func':code, 'reply_func':code, 'task_id':task,
                       'reason_code':reason_code, 'what':what, 'where':where, 'hint':hint}
                file.write(json.dumps(row, ensure_ascii=False) + '\n')


def generate(root):
    write_demo(root, initial=True)
    step = 1
    while True:
        time.sleep(1)
        write_demo(root, step)
        step += 1


def main():
    from .__main__ import add_view_arguments, validate_view_arguments
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True); parser.add_argument('--preview-state', choices=('on', 'off', 'blocked'), default='on'); parser.add_argument('--offset-state', choices=('triangulate','align','done','failed')); parser.add_argument('--pickacc-state', choices=('select','wait_tool','tool_in_place','z_step','constant','table','confirm')); parser.add_argument('--baseline-state', choices=('select','live','success','fail','confirm','prepare','prepare_error','prepare_unassigned')); parser.add_argument('--tooltip-state', choices=('idle','probe','confirm_move','in_range','done','fail')); add_view_arguments(parser)
    parser.add_argument('--film-mode', action='store_true')
    parser.add_argument('--temporary-exposure', action='store_true')
    parser.add_argument('--wb-state', choices=('running','success','fail'), default=None)
    parser.add_argument('--focus-state', choices=('off','live','exposure'), default=None)
    parser.add_argument('--camcal-state', choices=('running', 'failed', 'preview'), default=None)
    parser.add_argument('--depthcal-state', choices=('coarse', 'fine', 'done', 'failed'), default=None)
    args = parser.parse_args(); validate_view_arguments(parser, args)
    if args.camcal_state:
        args.preview_state = 'on' if args.camcal_state == 'preview' else 'blocked'
    if args.offset_state is None and args.page in ('offset', 'tool-offset', '工具偏移标定', '工具偏移标定页'):
        args.offset_state = 'align'
    if args.depthcal_state is None and args.page in ('depth', 'depthcal', 'depth-calib', '下针深度标定', '下针深度标定页'):
        args.depthcal_state = 'done'
    if args.wb_state: args.focus_state = 'live'
    if args.focus_state == 'off':
        args.preview_state = 'off'
    elif args.focus_state in ('live', 'exposure'):
        args.preview_state = 'on'
    # 先同步生成第一帧，让短延迟截图也有完整内容。
    write_demo(args.root, initial=True, preview_state=args.preview_state, offset_state=args.offset_state, pickacc_state=args.pickacc_state, baseline_state=args.baseline_state, tooltip_state=args.tooltip_state, focus_state=args.focus_state, camcal_state=args.camcal_state, depthcal_state=args.depthcal_state, wb_state=args.wb_state, temporary_exposure=args.temporary_exposure, film_mode=args.film_mode)
    def update():
        step = 1
        while True:
            time.sleep(1); write_demo(args.root, step, preview_state=args.preview_state, offset_state=args.offset_state, pickacc_state=args.pickacc_state, baseline_state=args.baseline_state, tooltip_state=args.tooltip_state, focus_state=args.focus_state, camcal_state=args.camcal_state, depthcal_state=args.depthcal_state, wb_state=args.wb_state, temporary_exposure=args.temporary_exposure, film_mode=args.film_mode); step += 1
    threading.Thread(target=update, daemon=True).start()
    from .app import App
    app = App(args.root, page=args.page or '主页', screenshot=args.screenshot, screenshot_after_ms=args.screenshot_after_ms, ignore_before_ns=0)
    if args.screenshot:
        app.root.overrideredirect(True)
        app.root.attributes('-topmost', True)
    if args.geometry:
        try:
            app.root.state('normal')
            app.root.attributes('-zoomed', False)
        except Exception:
            pass
        app.root.geometry(args.geometry)
    app.run()

if __name__ == '__main__': main()
