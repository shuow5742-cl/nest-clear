"""生成有代表性的演示数据并启动只读界面。"""
import argparse
import json
import os
import subprocess
import sys
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


def write_demo(root, step=0, *, initial=False):
    root = Path(root)
    now = datetime.now().astimezone(); stamp = now.isoformat()
    start = (now - timedelta(hours=2, minutes=18)).isoformat()
    points = [[x, y, z] for z in (-5, -2.5, 0, 2.5, 5) for x in (-20, -10, 0, 10, 20) for y in (-20, -10, 0, 10, 20)]
    sequence = 63 + step % 5
    atomic(supervisor_state_file(root), {'pid': os.getpid(), 'start_time': start, 'word': 0x126,
        'plc_connected': True, 'plc_last_read': stamp, 'heartbeat_last_write': stamp,
        'algorithm_ready': 1, 'child_pid': os.getpid(), 'child_state': '运行', 'child_start_time': start,
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
        'last_reply_time':stamp,'last_reply_task_id':103,'last_reply_request_func':80,'last_reply_func':82,
        'camcal':{'sequence':sequence,'plan':points,'start_xyz':[0,0,0],'kind':'move','detail':{
            'detected_corners':35,'predicted_corners':40,'saturated_fraction':0.01,'bbox':[510,120,560,170]}},
        'tooltip':{'tool':1,'iterations':2,'event':'tooltip_aligned','history':history,'new':[0.02,-0.01,0.01]}})
    for name in ('latest', 'camcal_latest', 'calib_front_latest', 'calib_side_latest'):
        path = live_image_file(root, name); path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new('RGB', (960, 600), '#253444'); draw = ImageDraw.Draw(image)
        draw.rectangle((100 + step % 20 * 3, 100, 600, 450), outline='#00ff88', width=4)
        for x in range(170, 590, 65):
            for y in range(150, 440, 60): draw.ellipse((x-4,y-4,x+4,y+4), fill='#46a7ff')
        draw.rectangle((510,120,560,170),outline='#ffdd55',width=3)
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
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True); add_view_arguments(parser)
    args = parser.parse_args(); validate_view_arguments(parser, args)
    # 先同步生成第一帧，让短延迟截图也有完整内容。
    write_demo(args.root, initial=True)
    def update():
        step = 1
        while True:
            time.sleep(1); write_demo(args.root, step); step += 1
    threading.Thread(target=update, daemon=True).start()
    command = [sys.executable, '-m', 'nest_clear.ui', '--root', args.root, '--page', args.page]
    if args.screenshot is not None:
        command += ['--screenshot', args.screenshot, '--screenshot-after-ms', str(args.screenshot_after_ms)]
    subprocess.run(command, check=False)

if __name__ == '__main__': main()
