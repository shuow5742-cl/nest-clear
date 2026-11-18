"""分析固定样本的尺度测量偏差；所有阈值只用于此离线脚本。"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import partial
import json
import math
from pathlib import Path

import cv2
import numpy as np
from nest_clear.vision.hair import HairDetectConfig, detect_candidates, gen1_pixel_gates, scaled
from nest_clear.vision.hair_integration import measure
from nest_clear.tools.sim_gen2_resolution import REF, GEN2, RATIO, THRESHOLD_PX, _calibration, _touches


def values(c, cfg, scale):
    obs = measure(c, (0., 0., 0.), _calibration(scale), cfg)
    return {'area': obs.area, 'length': obs.length,
            'width': obs.area / obs.length if obs.length > 0 else None,
            'gray_ratio': c.gray_ratio, 'dark_frac': c.dark_frac, 'reason': obs.reason}


def evaluate(row, threshold):
    cv2.setNumThreads(1)
    image = cv2.imread(row['path'])
    if image is None:
        raise FileNotFoundError(row['path'])
    h, w = image.shape[:2]
    small = cv2.resize(image, None, fx=RATIO, fy=RATIO, interpolation=cv2.INTER_AREA)
    sx, sy = w / small.shape[1], h / small.shape[0]
    cfg_ref = scaled(HairDetectConfig(um_per_px=REF))
    cfg_gen = replace(scaled(HairDetectConfig(um_per_px=GEN2)), hair_dark_max=threshold)
    refs = detect_candidates(image, cfg_ref)
    gens = detect_candidates(small, cfg_gen)
    ref_data = [{**values(c, cfg_ref, REF), 'eligible': gen1_pixel_gates(c, cfg_ref)[0],
                 'truncated': _touches(c.polygon, w, h), 'polygon': c.polygon} for c in refs]
    gen_data = [{**values(c, cfg_gen, GEN2),
                 'truncated': _touches(c.polygon, small.shape[1], small.shape[0]),
                 'polygon': c.polygon} for c in gens]
    # 预先生成二代轮廓掩码局部区域，按上一版仿真的距离和贪心规则配对。
    gen_shapes = []
    for c in gens:
        poly = np.rint(np.asarray(c.polygon) * [sx, sy]).astype(np.int32)
        if len(poly) < 3:
            gen_shapes.append(None); continue
        x, y, bw, bh = cv2.boundingRect(poly)
        x0, y0, x1, y1 = max(0,x), max(0,y), min(w,x+bw), min(h,y+bh)
        if x0 >= x1 or y0 >= y1:
            gen_shapes.append(None); continue
        mask = np.zeros((y1-y0,x1-x0), np.uint8)
        cv2.fillPoly(mask, [poly - [x0,y0]], 255)
        gen_shapes.append((x0,y0,x1,y1,mask>0, (c.pick_uv[0]*sx,c.pick_uv[1]*sy)))
    edges = []
    for i, c in enumerate(refs):
        poly = np.asarray(c.polygon, np.int32)
        if len(poly)<3: continue
        rx, ry, rw, rh = cv2.boundingRect(poly)
        dist = None
        for j, shape in enumerate(gen_shapes):
            if shape is None: continue
            x0,y0,x1,y1,sel,pick=shape
            if not sel.any(): continue
            gap_x=max(rx-x1, x0-(rx+rw), 0)
            gap_y=max(ry-y1, y0-(ry+rh), 0)
            point_x=max(rx-pick[0], pick[0]-(rx+rw), 0)
            point_y=max(ry-pick[1], pick[1]-(ry+rh), 0)
            if math.hypot(gap_x,gap_y) > THRESHOLD_PX and math.hypot(point_x,point_y) > THRESHOLD_PX:
                continue
            if dist is None:
                mask=np.zeros((h,w),np.uint8); cv2.fillPoly(mask,[poly],255)
                dist=cv2.distanceTransform(255-mask,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
            d=float(dist[y0:y1,x0:x1][sel].min())
            point_distance=max(0.,-cv2.pointPolygonTest(poly,pick,True))
            if min(d,point_distance)<=THRESHOLD_PX:
                edges.append((min(d,point_distance),i,j))
    edges.sort()
    def match(eligible):
        used_r=set(); used_g=set(); out=[]
        for distance,i,j in edges:
            if eligible and (not ref_data[i]['eligible'] or gen_data[j]['reason'] is not None):
                continue
            if i not in used_r and j not in used_g:
                used_r.add(i);used_g.add(j);out.append({'ref':i,'gen':j,'distance_px':distance})
        return out
    return {'path':row['path'], 'ref':ref_data, 'gen':gen_data, 'pairs_all':match(False), 'pairs_eligible':match(True)}


def summary_stats(nums):
    a=np.asarray([x for x in nums if x is not None and np.isfinite(x)],dtype=float)
    if not len(a): return {'n':0}
    return {'n':len(a),'mean':float(np.mean(a)),'median':float(np.median(a)),
            'p5':float(np.percentile(a,5)),'p95':float(np.percentile(a,95))}


def summarize(rows, mode, truncated):
    samples={k:[] for k in ('area_ratio','length_ratio','width_ratio','gray_ratio_diff','dark_frac_diff')}
    n=0
    for row in rows:
        for p in row[mode]:
            r,g=row['ref'][p['ref']],row['gen'][p['gen']]
            if r['truncated'] != truncated: continue
            n+=1
            for key in ('area','length','width'):
                a,b=r[key],g[key]
                samples[key+'_ratio'].append(b/a if a is not None and b is not None and a>0 else None)
            for key in ('gray_ratio','dark_frac'):
                a,b=r[key],g[key]
                samples[key+'_diff'].append(b-a if a is not None and b is not None else None)
    return {'pairs':n, **{k:summary_stats(v) for k,v in samples.items()}}


def recall(rows, truncated):
    n=matched=extra=0
    for row in rows:
        refs={i for i,c in enumerate(row['ref']) if c['eligible'] and c['truncated']==truncated}
        n+=len(refs)
        matched+=sum(p['ref'] in refs for p in row['pairs_eligible'])
        used={p['gen'] for p in row['pairs_eligible']}
        extra+=sum(i not in used and c['reason'] is None and c['truncated']==truncated for i,c in enumerate(row['gen']))
    return {'reference':n,'matched':matched,'recall':matched/n if n else None,'extra':extra}


def misses(rows, old):
    bypath={row['path']:row for row in rows}; found=[]
    for orig in old:
        for i, ref in enumerate(orig['ref']):
            if ref['truncated'] or any(p['ref']==i for p in orig['pairs']): continue
            row=bypath[orig['path']]
            # 用原参考轮廓精确关联，避免同图其他目标被误认。
            ri=next((j for j,c in enumerate(row['ref']) if np.array_equal(c['polygon'], ref['polygon'])),None)
            if ri is None:
                found.append({'path':orig['path'],'error':'参考轮廓无法复现'});continue
            ref_poly=np.asarray(ref['polygon'],np.int32)
            image=cv2.imread(orig['path']);h,w=image.shape[:2]; sw,sh=orig['small_size']
            mask=np.zeros((h,w),np.uint8);cv2.fillPoly(mask,[ref_poly],255)
            dist=cv2.distanceTransform(255-mask,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
            options=[]
            for j,gen in enumerate(row['gen']):
                poly=np.rint(np.asarray(gen['polygon'])*[w/sw,h/sh]).astype(np.int32)
                if len(poly)<3: continue
                gm=np.zeros((h,w),np.uint8);cv2.fillPoly(gm,[poly],255)
                if not gm.any():continue
                d=float(dist[gm>0].min())
                if d<=THRESHOLD_PX:
                    options.append({'candidate':j,'distance_px':d,'reason':gen['reason'],
                                    'area_over_gate':gen['area']/1.2,'length_over_gate':gen['length']/2,
                                    'width_over_gate':gen['width']/.35 if gen['width'] is not None else None,
                                    'assigned_to_other_reference':any(p['gen']==j and p['ref']!=ri for p in row['pairs_eligible'])})
            found.append({'path':orig['path'],'reference_index':ri,'neighbors':sorted(options,key=lambda x:x['distance_px'])})
    return found


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=Path('runs/sim_gen2_3000'))
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--force',action='store_true',help='重新计算所有阈值，忽略已有汇总')
    args=parser.parse_args()
    old=json.loads((args.out/'results.json').read_text())
    output = args.out/'measurement_bias.json'
    result=json.loads(output.read_text()) if output.exists() and not args.force else {}
    for threshold in (55,60,65,70):
        if str(threshold) in result:
            continue
        if threshold != 55:
            m=result['55']['all_untruncated']['area_ratio'].get('median')
            if m is None or m >= .92:
                break
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows=[]
            for index, row in enumerate(pool.map(partial(evaluate,threshold=threshold),old,chunksize=4), 1):
                rows.append(row)
                if index % 500 == 0:
                    print(f"阈值 {threshold}：{index}/{len(old)}", flush=True)
        entry={f'{label}_{group}': summarize(rows,mode,trunc)
               for label,mode in [('all','pairs_all'),('eligible','pairs_eligible')]
               for group,trunc in [('untruncated',False),('truncated',True)]}
        entry['recall_untruncated']=recall(rows,False);entry['recall_truncated']=recall(rows,True)
        if threshold==55:
            entry['misses']=misses(rows,old)
        result[str(threshold)]=entry
        (args.out/'measurement_bias.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        print(f'阈值 {threshold} 完成；不触边面积比中位数 {entry["all_untruncated"]["area_ratio"].get("median")}',flush=True)
if __name__=='__main__':main()
