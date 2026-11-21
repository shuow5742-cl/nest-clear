"""生成有代表性的演示数据并启动界面。"""
import argparse
import json
import os
import subprocess
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from PIL import Image, ImageDraw
from nest_clear.paths import supervisor_state_file, main_status_file, live_image_file


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
    os.replace(tmp, path)


def write_demo(root, step=0, *, initial=False, preview_state="on", offset_state=None):
    root = Path(root)
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
        'tooltip':{'tool':1,'iterations':2,'event':'tooltip_aligned','history':history,'new':[0.02,-0.01,0.01]},
        'command_seq':0, 'tooloffset':{'phase':'tool', 'tool':1, 'mode':'center', 'target':'T12',
            'message':'请手动移动工具1到 T12 中心，到位后按【工具到位】',
            'existing':{'tool1':{'time':stamp,'offset_xy_mm':[1.2,-0.3],'n':5,'se':[.01,.01], 'current_tools_xy':[1.2,-0.3], 'baseline_stale':True}, 'tool2':{}},
            'samples':[{'target':'T02','offset_xy_mm':[1.2,-.3], 'distance_from_mean_mm':.01, 'outlier':False}],
            'mean':[1.2,-.3], 'std':[.01,.01], 'se':[.005,.005], 'edges':[]}})
    if offset_state:
        path = main_status_file(root)
        data = json.loads(path.read_text(encoding='utf-8'))
        data['mode'] = '工具偏移标定中'
        d = data['tooloffset']
        d.update(existing={
            'tool1':{'time':stamp,'offset_xy_mm':[1.2,-.3],'n':8,'se':[.012,.011], 'current_tools_xy':[1.21,-.31], 'baseline_stale':True, 'calibration_changed':True},
            'tool2':{'time':stamp,'offset_xy_mm':[2.1,.4],'n':7,'se':[.013,.014], 'current_tools_xy':[2.1,.4]}},
            samples=[{'target':f'T{i:02d}','offset_xy_mm':[1.201+i*.001,-.301+i*.001], 'distance_from_mean_mm':.008, 'outlier':False} for i in range(5)],
            mean=[1.203,-.299], std=[.012,.011], se=[.005,.005])
        if offset_state == 'select': d.update(phase='select',tool=None,message='请选择工具和模式并开始')
        elif offset_state == 'wait_tool': d.update(phase='tool',tool=1,message='请手动移动工具1到 T12 中心，到位后按【工具到位】')
        elif offset_state == 'edge':
            d.update(phase='wait90',tool=1,mode='center_edge',message='边缘验证中，不需要移动工具', edges=[{'target':'T12','corner':i,'delta_xy_mm':[.012,.008] if i != 2 else [.06,.02], 'distance_mm':.014 if i != 2 else .063} for i in range(4)])
        elif offset_state == 'confirm':
            d.update(phase='confirm',tool=1,message='等待确认覆盖',candidate={'time':stamp,'offset_xy_mm':[1.203,-.299],'n':8,'se':[.005,.005]})
        atomic(path, data)
        if initial:
            project = Path(__file__).resolve().parents[3]
            env = dict(os.environ, PYTHONPATH=str(project / 'src'))
            subprocess.run([str(project / '.venv/bin/python'), '-m', 'nest_clear.tools.offset_demo_image', str(root)], cwd=project, env=env, check=True, timeout=30)
    names = ('latest', 'camcal_latest', 'preview_latest', 'calib_front_latest', 'calib_side_latest')
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
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True); parser.add_argument('--preview-state', choices=('on', 'off', 'blocked'), default='on'); parser.add_argument('--geometry', help='窗口尺寸，例如 1366x768'); parser.add_argument('--offset-state', choices=('select','wait_tool','edge','confirm'));  add_view_arguments(parser)
    args = parser.parse_args(); validate_view_arguments(parser, args)
    if args.offset_state is None and args.page in ('offset', 'tool-offset', '工具偏移标定', '工具偏移标定页'):
        args.offset_state = 'wait_tool'
    # 先同步生成第一帧，让短延迟截图也有完整内容。
    write_demo(args.root, initial=True, preview_state=args.preview_state, offset_state=args.offset_state)
    def update():
        step = 1
        while True:
            time.sleep(1); write_demo(args.root, step, preview_state=args.preview_state, offset_state=args.offset_state); step += 1
    threading.Thread(target=update, daemon=True).start()
    from .app import App
    app = App(args.root, page=args.page, screenshot=args.screenshot, screenshot_after_ms=args.screenshot_after_ms)
    if args.geometry:
        try:
            app.root.state('normal')
            app.root.attributes('-zoomed', False)
        except Exception:
            pass
        app.root.geometry(args.geometry)
    app.run()

if __name__ == '__main__': main()
