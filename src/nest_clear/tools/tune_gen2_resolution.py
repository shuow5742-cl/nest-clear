"""对已固定抽样及参考结果作受限网格试调，不改物理挑取门。"""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import partial
import json
from pathlib import Path
import cv2
import numpy as np
from nest_clear.vision.hair import HairDetectConfig, detect_candidates, scaled
from nest_clear.vision.hair_integration import measure
from nest_clear.tools.sim_gen2_resolution import GEN2, RATIO, THRESHOLD_PX, REF, _calibration, _touches, _stats


def evaluate(row, changes):
    cv2.setNumThreads(1)
    image = cv2.imread(row['path']); h, w = image.shape[:2]
    small = cv2.resize(image, None, fx=RATIO, fy=RATIO, interpolation=cv2.INTER_AREA)
    sx, sy = w / small.shape[1], h / small.shape[0]
    cfg = replace(scaled(HairDetectConfig(um_per_px=GEN2)), **changes)
    found = [c for c in detect_candidates(small, cfg) if measure(c, (0,0,0), _calibration(GEN2), cfg).reason is None]
    edges=[]
    for i, ref in enumerate(row['ref']):
        contour = np.array(ref['polygon'], np.int32)
        mask = np.zeros((h,w), np.uint8); cv2.fillPoly(mask, [contour], 255)
        dist = cv2.distanceTransform(255-mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        for j, c in enumerate(found):
            poly=np.rint(np.array(c.polygon)*[sx,sy]).astype(np.int32)
            if len(poly)<3: continue
            target=np.zeros((h,w),np.uint8);cv2.fillPoly(target,[poly],255)
            if not np.any(target): continue
            d=float(dist[target>0].min()); pick=(c.pick_uv[0]*sx,c.pick_uv[1]*sy)
            pd=max(0.,-cv2.pointPolygonTest(contour,pick,True))
            if min(d,pd) <= THRESHOLD_PX:
                edges.append((min(d,pd),i,j,pd*REF/1000,float(np.linalg.norm(np.array(pick)-ref['pick']))*REF/1000))
    used_r=set();used_g=set();pairs=[]
    for _,i,j,on,along in sorted(edges):
        if i not in used_r and j not in used_g:
            used_r.add(i);used_g.add(j);pairs.append({'ref':i,'gen':j,'on_hair_mm':on,'along_mm':along})
    return {'ref':row['ref'],'gen':[{'truncated':_touches(c.polygon,small.shape[1],small.shape[0])} for c in found],'pairs':pairs}


def main():
    out=Path('runs/sim_gen2_3000');rows=json.loads((out/'results.json').read_text())
    # 低分辨率目标面积缩小，尝试闭运算及种子区域扩张阈值；挑取门保持原样。
    trials=[{'dark_close_k': v} for v in (3,5,9)] + [{'seed_grow_bg_margin': v} for v in (24,36,60)]
    result=[]
    for changes in trials:
        with ProcessPoolExecutor(max_workers=8) as pool:
            measured=list(pool.map(partial(evaluate,changes=changes),rows,chunksize=8))
        entry={'changes':changes,'untruncated':_stats(measured,False),'truncated':_stats(measured,True)}
        result.append(entry); print(json.dumps(entry,ensure_ascii=False),flush=True)
        (out/'tuning.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
