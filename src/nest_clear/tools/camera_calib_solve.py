"""从采集记录离线重解相机标定。"""
import argparse
import json
from dataclasses import fields
from pathlib import Path
import yaml
from nest_clear.camcal import CameraCalibConfig, solve_offline, save_calibration
from nest_clear.vision.geometry import load_calibration

def main():
    parser=argparse.ArgumentParser(description='离线重解主相机标定')
    parser.add_argument('session_record'); parser.add_argument('--config',help='标定参数 YAML'); parser.add_argument('--out',help='输出标定 YAML'); parser.add_argument('--nominal',default='config/calibration/camera.yaml',help='名义相机标定 YAML')
    args=parser.parse_args()
    with open(args.session_record,encoding='utf-8') as f: record=json.load(f)
    data={}
    if args.config:
        with open(args.config,encoding='utf-8') as f: data=yaml.safe_load(f) or {}
        data=data.get('camera_calibration',data)
    cfg=CameraCalibConfig(**{k:v for k,v in data.items() if k in {x.name for x in fields(CameraCalibConfig)}})
    from nest_clear.calibration_bootstrap import ensure_calibration_files
    ensure_calibration_files(Path(args.nominal).parent)
    result=solve_offline(record,cfg,load_calibration(args.nominal),tuple(record['image_size']))
    print('相机标定报告：'); print(json.dumps(result.report,ensure_ascii=False,indent=2))
    if args.out:
        path=Path(args.out); save_calibration(result,path,path.parent/'history'); print(f'已保存：{path}')
if __name__=='__main__': main()
