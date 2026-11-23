"""合成换针画面，使用生产检测、求解和标注生成演示。"""
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from nest_clear.paths import live_image_file
from nest_clear.record.live import write_live
from nest_clear.tooltip import SilhouetteTipDetector, ToolTipBaseline, ViewCalib, solve_correction
from nest_clear.tooltip.overlay import render_tooltip


def main(root, state):
    root = Path(root)
    if state == 'idle':
        for name in ('front', 'side'):
            write_live(root, f'calib_{name}_latest', np.full((480,640), 235, np.uint8))
        data = {'phase':'idle', 'message':'等待 PLC 100/101', 'tool':None, 'rounds':[]}
    else:
        # 两个画面的竖直偏移一致，与相机雅可比的公共 Z 方向相符。
        positions = {'probe':(324,320,242), 'confirm_move':(329,314,245),
                     'in_range':(320,320,241), 'done':(320,320,241), 'fail':(331,314,245)}
        xf, xs, y = positions[state]
        matrices = {'front':((20,0,0),(0,0,20)), 'side':((0,20,0),(0,0,20))}
        target = (320., 242.)
        detector = SilhouetteTipDetector(direction=(0,1), threshold=127, min_area=5)
        pixels = {}
        for name, x in (('front',xf), ('side',xs)):
            frame = np.full((480,640), 245, np.uint8)
            cv2.line(frame, (x,120), (x,y-2), 15, 3)
            detection = detector.detect(frame)
            if detection.point_uv is None: raise RuntimeError('演示针尖检测失败')
            pixels[name] = list(detection.point_uv)
            write_live(root, f'calib_{name}_latest', render_tooltip(frame, target, detection.point_uv))
        deviation = {n:[(pixels[n][i]-target[i])*.0152 for i in range(2)] for n in pixels}
        baseline = ToolTipBaseline('tool1', (125.,73.,110.), [ViewCalib(n,target,j) for n,j in matrices.items()])
        delta = solve_correction(pixels, baseline, primary_view='front').delta_xyz.tolist()
        phase = state
        message = {'probe':'探测中 第 2/4 步：探测移动 +0.2mm Y', 'confirm_move':'等待确认移动',
                   'in_range':'已在范围内，请确认', 'done':'完成', 'fail':'雅可比测量质量不合格：检查针尖识别/对焦'}[state]
        if state == 'probe': delta = [-.2,.2,0.]
        actual = ([125.2,73.,110.] if state == 'probe' else
                  [125.12,72.92,110.04] if state in ('in_range', 'done') else [125.,73.,110.])
        target_xyz = [a+d for a,d in zip(actual,delta)]
        cumulative = [x-p for x,p in zip(target_xyz if state in ('probe','confirm_move') else actual,baseline.p0)]
        row = {'iteration':1, 'stage':'探测' if state == 'probe' else '对准', 'pixels':pixels,
               'deviation_mm':deviation, 'delta':delta, 'target':target_xyz if state in ('probe','confirm_move') else None,
               'click':False, 'confirmed_at':'2026-11-22T12:34:56-08:00'}
        if state == 'probe':
            reference = np.full((480,640), 245, np.uint8)
            cv2.line(reference, (320,120), (320,240), 15, 3)
            reference_point = list(detector.detect(reference).point_uv)
            row.update(pixels={'front':reference_point, 'side':reference_point},
                       deviation_mm={n:[(reference_point[i]-target[i])*.0152 for i in range(2)] for n in ('front','side')},
                       delta=[.2,0.,0.], target=[125.2,73.,110.])
        data = {'phase':phase, 'message':message, 'tool':1, 'rounds':[row] if state in ('probe', 'done') else [], 'pixels':pixels,
                'deviation_mm':deviation, 'jacobian_status':'未测' if state == 'probe' else '已存',
                'jacobian_quality':None if state in ('probe','fail') else {'condition_number':1.4,'rms_px':{'front':.2,'side':.3}},
                'delta':delta, 'target':target_xyz if state in ('probe','confirm_move') else None,
                'cumulative':cumulative, 'old_compensation':[1.2,-.3,.05]}
        if state == 'done':
            data.update(new_compensation=[1.32,-.38,.09], compensation_difference=[.12,-.08,.04])
        if state == 'fail': data['jacobian_status'] = '未测'
    (root/'tooltip_demo.json').write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__': main(*sys.argv[1:])
