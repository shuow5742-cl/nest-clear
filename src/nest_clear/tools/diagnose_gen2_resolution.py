import json, collections, cv2, numpy as np
from nest_clear.vision.hair import HairDetectConfig, scaled, detect_candidates
from nest_clear.tools.sim_gen2_resolution import _calibration, GEN2, RATIO
from nest_clear.vision.hair_integration import measure
cv2.setNumThreads(1)
rows=json.load(open('runs/sim_gen2_3000/results.json')); cfg=scaled(HairDetectConfig(um_per_px=GEN2)); output=[]
for row in rows:
    missing=[(i,c) for i,c in enumerate(row['ref']) if not any(p['ref']==i for p in row['pairs'])]
    if not missing: continue
    img=cv2.imread(row['path']); small=cv2.resize(img,None,fx=RATIO,fy=RATIO,interpolation=cv2.INTER_AREA); stages={}
    candidates=detect_candidates(small,cfg,stage_callback=lambda name,polys: stages.update({name:polys}))
    sx=small.shape[1]/img.shape[1]; sy=small.shape[0]/img.shape[0]
    for i,ref in missing:
        poly=np.rint(np.array(ref['polygon'])*[sx,sy]).astype(np.int32)
        refmask=np.zeros(small.shape[:2],np.uint8); cv2.fillPoly(refmask,[poly],255)
        dist=cv2.distanceTransform(255-refmask,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        def near(p):
            if len(p)<3:return False
            mask=np.zeros(small.shape[:2],np.uint8);cv2.fillPoly(mask,[np.array(p,np.int32)],255)
            return bool(np.any(mask) and dist[mask>0].min() <= .3*1000/GEN2)
        present={name:any(near(p) for p in polys) for name,polys in stages.items()}
        if present['relabel_output']:
            matches=[measure(c,(0,0,0),_calibration(GEN2),cfg).reason for c in candidates if near(c.polygon)]
            loss='物理门/一对一匹配: '+str(matches)
        else:
            last=next((name for name in reversed(list(present)) if present[name]),None)
            names=list(present)
            loss=names[names.index(last)+1] if last else 'dark_core/line/fiber 均未形成候选'
        output.append({'path':row['path'],'ref_index':i,'truncated':ref['truncated'],'stages':present,'loss':loss})
print('未触边',collections.Counter(x['loss'] for x in output if not x['truncated']))
print('触边',collections.Counter(x['loss'] for x in output if x['truncated']))
open('runs/sim_gen2_3000/diagnosis.json','w').write(json.dumps(output,ensure_ascii=False,indent=2))
