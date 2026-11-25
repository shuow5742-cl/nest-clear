#!/usr/bin/env python3
"""Evaluate detectors against the read-only T79 annotations and frame sets."""
import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from nest_clear.tooltip.detector import SilhouetteTipDetector

DATA = Path('/home/shuow/nest-clear-field/20261124_annot')
OUT = Path('/home/shuow/nest-clear-field/20261125_t79eval')

def baseline():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted(DATA.glob('*.png')):
        ref = json.loads((DATA / 'annotations' / (p.stem + '.json')).read_text())['derived']['tool_tip']
        result = SilhouetteTipDetector().detect(cv2.imread(str(p), 0))
        err = np.linalg.norm(np.asarray(result.point_uv) - ref) if result.point_uv else float('nan')
        rows.append([p.stem, result.point_uv, err, result.diagnostic])
    with (OUT / 'baseline.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['image', 'point_uv', 'error_px', 'diagnostic']); w.writerows(rows)
    print('baseline', len(rows), 'median error', np.nanmedian([r[2] for r in rows]), 'px')


def evaluate():
    import math
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from nest_clear.tooltip.trident import TridentTipDetector
    from nest_clear.tooltip.runout import RunoutSession
    cv2.setNumThreads(1); OUT.mkdir(parents=True,exist_ok=True); (OUT/'overlays').mkdir(exist_ok=True)
    detector=TridentTipDetector(); rows=[]; found={}; refs={}
    for p in sorted(DATA.glob('*.png')):
        img=cv2.imread(str(p),0); labels=cv2.imread(str(DATA/'annotations'/(p.stem+'.labels.png')),0)
        ref=json.loads((DATA/'annotations'/(p.stem+'.json')).read_text())['derived']; r=detector.detect(img); found[p.stem]=r; refs[p.stem]=ref
        straight_axis=None; supplemental=None
        if r.sampling_rows:
            sy,sx=np.where((labels==2)&(np.indices(labels.shape)[0]>=min(r.sampling_rows))&(np.indices(labels.shape)[0]<=max(r.sampling_rows)))
            if len(sx)>=10:
                pts=np.column_stack((sx,sy)); mean=pts.mean(axis=0); _,vectors=np.linalg.eigh(np.cov(pts,rowvar=False)); sd=vectors[:,-1]
                if sd[1]<0: sd=-sd
                straight_axis=(mean,sd)
                mask=cv2.dilate((labels==3).astype(np.uint8),np.ones((5,5),np.uint8))
                admissible=[np.array(t) for t in r.prong_tips if 0<=round(t[1])<labels.shape[0] and 0<=round(t[0])<labels.shape[1] and mask[round(t[1]),round(t[0])]]
                if admissible:
                    endpt=max(admissible,key=lambda t:np.dot(t-mean,sd)); supplemental=mean+np.dot(endpt-mean,sd)*sd

        row={'image':p.stem,'status':r.status,'diagnostic':r.diagnostic,'ms':r.elapsed_ms}
        canvas=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR); target=np.array(ref['tool_tip']); cv2.drawMarker(canvas,tuple(np.round(target).astype(int)),(255,0,0),cv2.MARKER_CROSS,15,2)
        if r.point_uv is not None:
            point=np.array(r.point_uv); direction=np.array(ref['axis']['direction']); normal=np.array([-direction[1],direction[0]]); delta=point-target
            angle=math.degrees(math.acos(np.clip(np.dot(direction,r.axis['direction']),-1,1)))
            farerr=float(np.linalg.norm(np.array(r.farthest_prong_uv)-ref['farthest_tip']))
            # Approximate body and tip partitions: fixed terminal 75 px, not trained on labels.
            end=int(np.max(np.nonzero(r.tool_mask)[0])); yy=np.indices(labels.shape)[0]
            for name,truth,pred in [('body',labels==2,(r.tool_mask>0)&(yy<end-75)),('tip',labels==3,(r.tool_mask>0)&(yy>=end-75))]:
                row[name+'_iou']=float(np.sum(truth&pred)/max(1,np.sum(truth|pred)))
            row.update(u=point[0],v=point[1],error=float(np.linalg.norm(delta)),along=float(delta@direction),cross=float(delta@normal),far_error=farerr,angle_deg=angle,gradient=r.sharpness['normalized_gradient'],width=r.sharpness['edge_width_px'])
            o=np.array(r.axis['point']); d=np.array(r.axis['direction'])
            ro=np.array(ref['axis']['point']); rd=np.array(ref['axis']['direction']); cv2.line(canvas,tuple(np.round(ro-200*rd).astype(int)),tuple(np.round(ro+300*rd).astype(int)),(255,0,0),1)
            if straight_axis is not None:
                so,sd=straight_axis; cv2.line(canvas,tuple(np.round(so-200*sd).astype(int)),tuple(np.round(so+300*sd).astype(int)),(255,255,0),1)
                refv=target[1]; refx=ro[0]+(refv-ro[1])*rd[0]/rd[1]; sx=so[0]+(refv-so[1])*sd[0]/sd[1]
                row['straight_axis_offset']=sx-refx
            if supplemental is not None:
                row['subref_u'],row['subref_v']=map(float,supplemental); row['subref_error']=float(np.linalg.norm(point-supplemental)); row['subref_shift']=float(np.linalg.norm(supplemental-target)); cv2.drawMarker(canvas,tuple(np.round(supplemental).astype(int)),(0,165,255),cv2.MARKER_TILTED_CROSS,13,1)
            cv2.line(canvas,tuple(np.round(o-200*d).astype(int)),tuple(np.round(o+300*d).astype(int)),(0,255,0),1)
            for tip in r.prong_tips: cv2.circle(canvas,tuple(np.round(tip).astype(int)),4,(0,255,255),1)
            cv2.drawMarker(canvas,tuple(np.round(point).astype(int)),(0,0,255),cv2.MARKER_CROSS,15,2)
            for y in r.sampling_rows[::10]:
                x=int(o[0]+(y-o[1])*d[0]/d[1]); cv2.line(canvas,(x-25,y),(x+25,y),(255,0,255),1)
        cv2.imwrite(str(OUT/'overlays'/(p.stem+'.png')),canvas); rows.append(row)
    fields=['image','status','diagnostic','u','v','error','along','cross','far_error','angle_deg','body_iou','tip_iou','gradient','width','ms','straight_axis_offset','subref_u','subref_v','subref_error','subref_shift']
    with (OUT/'annotations.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fields); wr.writeheader(); wr.writerows(rows)
    def runout(use_ref):
        session=RunoutSession(stable_px=1000); result=None
        for i in range(1,13):
            a=f'rot{i:02d}_front'; b=f'rot{i:02d}_side'
            f=refs[a]['tool_tip'] if use_ref else found[a].point_uv; s=refs[b]['tool_tip'] if use_ref else found[b].point_uv
            if f is None or s is None: return None
            result=session.add_press([f],[s],np.eye(2)*.0135,np.eye(2)*.0135)
        return result['max_deviation_um']
    scan=[]; repeats=[]; frame_rows=[]; rotation_frames={}
    groups=[('device',i,float('nan')) for i in range(1,13)]+[('device',i,-.503+(i-13)*.04) for i in range(13,29)]+[('device2',i, .062 if i==1 else .060+(i-2)*(.479/12)) for i in range(1,15)]
    root=Path('/home/shuow/nest-clear-field/20261124_tipsweep')
    for device,i,y in groups:
        for camera in ('front','side'):
            results=[detector.detect(cv2.imread(str(root/device/f'set_{i:03d}'/f'{camera}_{j}.png'),0)) for j in range(1,6)]
            valid=[r for r in results if r.point_uv is not None]
            for j,r in enumerate(results,1):
                frame_rows.append([device,i,camera,j,r.status,*(r.point_uv or (None,None)),*(r.farthest_prong_uv or (None,None))])
                if device=='device' and i in (3,10) and camera=='side' and r.point_uv is not None:
                    image=cv2.imread(str(root/device/f'set_{i:03d}'/f'{camera}_{j}.png'))
                    cv2.drawMarker(image,tuple(np.round(r.point_uv).astype(int)),(0,0,255),cv2.MARKER_CROSS,15,2)
                    cv2.circle(image,tuple(np.round(r.farthest_prong_uv).astype(int)),4,(0,255,255),1)
                    cv2.imwrite(str(OUT/f'round2_{i}_{camera}_{j}.png'),image)

            if device=='device' and i<=12: rotation_frames[(i,camera)]=[r.point_uv for r in results]
            if valid:
                pts=np.array([r.point_uv for r in valid]); std=np.std(pts,axis=0); repeats.append((device,i,camera,len(valid),*std))
                for metric in ('normalized_gradient','edge_width_px'):
                    if np.isnan(y): continue
                    vals=[r.sharpness[metric] for r in valid if r.sharpness[metric] is not None]
                    scan.append({'device':device,'set':i,'y_mm':y,'camera':camera,'metric':metric,'mean':np.mean(vals) if vals else np.nan,'std':np.std(vals) if vals else np.nan,'n':len(vals)})
    with (OUT/'scan.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,['device','set','y_mm','camera','metric','mean','std','n']); wr.writeheader(); wr.writerows(scan)
    with (OUT/'frames.csv').open('w',newline='') as f:
        wr=csv.writer(f); wr.writerow(['device','set','camera','frame','status','u','v','far_u','far_v']); wr.writerows(frame_rows)
    with (OUT/'repeatability.csv').open('w',newline='') as f:
        wr=csv.writer(f); wr.writerow(['device','set','camera','n','std_u_px','std_v_px']); wr.writerows(repeats)
    fig,axs=plt.subplots(2,1,figsize=(9,8))
    for ax,metric in zip(axs,('normalized_gradient','edge_width_px')):
        for camera in ('front','side'):
            ss=[x for x in scan if x['camera']==camera and x['metric']==metric]; ax.errorbar([x['y_mm'] for x in ss],[x['mean'] for x in ss],yerr=[x['std'] for x in ss],label=camera,marker='.',capsize=2)
        ax.set(xlabel='Y (mm; approximate)',ylabel=metric); ax.grid(); ax.legend()
    fig.tight_layout(); fig.savefig(OUT/'sharpness.png',dpi=150); plt.close(fig)
    session=RunoutSession(); raw_result=None
    for i in range(1,13):
        raw_result=session.add_press(rotation_frames[(i,'front')], rotation_frames[(i,'side')], np.eye(2)*.0135,np.eye(2)*.0135)
        if raw_result.get('rejected'): break
    raw_runout=raw_result.get('max_deviation_um'); raw_summary=(f'{raw_runout:.1f} µm，差 {abs(runout(True)-raw_runout):.1f} µm' if raw_runout is not None and not raw_result.get('rejected') else f'不可计算（{raw_result.get("reason", "输入不足")}）')
    errs=[r['error'] for r in rows if 'error' in r]; times=[r['ms'] for r in rows]; rr=runout(True); pr=runout(False)
    lines=['# 三棱针评测','', '基线 SilhouetteTipDetector 在 38 张标注图均未返回坐标，详见 baseline.csv。检测方法和参数见工作区 docs/trident_detector.md；结果仅为实验性评估。', '',f'标注图 {len(rows)} 张，精定位 {sum(r["status"]=="精定位" for r in rows)} 张。',f'工具尖误差中位 {np.median(errs):.3f} px，最大 {max(errs):.3f} px；耗时中位 {np.median(times):.1f} ms，最大 {max(times):.1f} ms。',f'所有原始组坐标标准差最大 {max(max(x[4:]) for x in repeats):.3f} px（分别按 u/v，样本总体标准差）。',f'转角组 RunoutSession 最大偏差：参考 {rr:.1f} µm，标注帧检测 {pr:.1f} µm，差 {abs(rr-pr):.1f} µm；原始五帧检测 {raw_summary}；使用 RunoutSession 默认稳定性门限。','', '![清晰度曲线](sharpness.png)','', '扫描 Y 坐标按题给步距近似，非逐帧实测；误差棒为五帧总体标准差。清晰度指标取针身边缘归一化最大梯度和 10–90% 宽度中位数。', '', '| 图像 | 状态 | 误差 px | 沿轴 px | 横轴 px | 齿尖误差 px | 轴角差 ° | 针身 IoU | 针尖 IoU | ms |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        def fmt(k): return f'{r[k]:.3f}' if k in r else '—'
        lines.append('| '+' | '.join([r['image'],r['status']]+[fmt(k) for k in ('error','along','cross','far_error','angle_deg','body_iou','tip_iou','ms')])+' |')
    front=[x for x in scan if x['camera']=='front' and x['metric']=='normalized_gradient']
    side=[x for x in scan if x['camera']=='side' and x['metric']=='normalized_gradient']
    peak=max(front,key=lambda x:x['mean']); lines += ['', f'归一化梯度正面峰值位于 Y={peak["y_mm"]:.3f} mm；侧面全范围相对均值波动 {(max(x["mean"] for x in side)-min(x["mean"] for x in side))/np.mean([x["mean"] for x in side])*100:.1f}%。']
    for pct in (1,5,10):
        selected=[x['y_mm'] for x in front if x['mean']>=peak['mean']*(1-pct/100)]
        lines.append(f'峰值下降 {pct}% 以内的采样点范围：{min(selected):.3f} 至 {max(selected):.3f} mm；可能不连续。')
    ordered=sorted(front,key=lambda x:x['y_mm']); k=ordered.index(peak)
    violations=sum(ordered[i+1]['mean']<ordered[i]['mean'] for i in range(k))+sum(ordered[i+1]['mean']>ordered[i]['mean'] for i in range(k,len(ordered)-1))
    lines += [f'正面梯度曲线相对峰值两侧共有 {violations} 处违反单调趋势；侧面波动亦较大，因此两个指标均未证明满足清晰度验收要求，暂不推荐用于对刀。', '', '失败样例：' + '、'.join(f'[{r["image"]}](overlays/{r["image"]}.png)' for r in sorted(rows,key=lambda x:x.get('error',-1),reverse=True)[:5]), '', '验收：精定位数量、误差、五帧重复性未达标；耗时为单线程 OpenCV 下的标注帧测量；离心量差见上。当前粗定位结果不可当作精定位使用。', '', '针身和针尖 IoU 使用检测末端前 75 px 的固定分界，仅用于暴露分割不足；算法没有独立的语义分割。人工参考是像素掩膜 PCA 和最远整数像素，存在标注边界与亚像素定义差异。叠加图蓝十字为参考，红十字为检测，绿线为轴，黄圈为连通末端，紫线为清晰度采样位置。', '', '局限：局部阈值、断齿和背景暗物可能造成漏检或误检；指标未经过独立数据验证，不宜作为实际对刀判据。']
    lines += ['', '## 第一项：五帧异常', '', '旧版最大波动为 device/set_010/side，u 标准差 7.628 px；次大为 set_003/side，7.279 px。前十组各帧轴线、末端与工具尖坐标见 round1_worst_frames.csv；本版逐帧坐标见 frames.csv。原始版本和本版的异常组叠加图分别以 old_ 和 round2_ 为前缀。', 'set_010/side 第 1 帧针身较直，第 2–5 帧呈周期性锯齿（见 [五帧针身拼图](round1_set010_side_montage.png)）；旧版轴线外推位置约 775 至 796 px，变化并非仅仅选中了不同齿。不能仅凭图像确定是逐行成像、振动还是其他原因，也不能通过强行跨帧对齐声称原始坐标重复性达标。', '本版沿轴平行扫描局部中间灰度穿越，要求五个连续采样点，并限制在原连通域末端之后 5 px 内以减少背景暗物误报；该限制也可能漏掉完全断开的更长灰齿。', '', '## 第二项：参考轴和补充亚像素参考', '']
    subrows=[r for r in rows if 'subref_shift' in r]
    lines.append(f'成功构造 {len(subrows)}/{len(rows)} 张补充参考。先使用检测轴的同一批轴向剖面候选，再保留位于人工针尖掩膜膨胀 2 px 内的候选，将最远者投影到人工笔直段 PCA 轴。它受检测器采样范围与人工掩膜约束，不是独立真值。原始参考及误差保持不变。')
    lines.append(f'补充参考相对整数像素参考位移中位 {np.median([r["subref_shift"] for r in subrows]):.3f} px，最大 {max(r["subref_shift"] for r in subrows):.3f} px；差值包含轴与边缘定义差异，不能作为统计置信区间。')
    for camera in ('front','side'):
        group=[r for r in rows if r['image'].endswith(camera) and 'straight_axis_offset' in r]
        lines.append(f'{camera}：检测相对原参考横轴误差中位 {np.median([r["cross"] for r in group]):.3f} px，沿轴误差中位 {np.median([r["along"] for r in group]):.3f} px；人工笔直段轴相对原人工轴的横向差中位 {np.median([r["straight_axis_offset"] for r in group]):.3f} px。')
    lines += ['', '人工笔直段与全针身 PCA 轴的横向中位差远小于检测轴偏差，提示仅改变人工掩膜纵向区间不足以解释系统偏差；还需检查标注左右轮廓与灰度边缘位置。沿轴正偏与半灰度边缘相对最远暗像素中心的定义差一致，但单凭本评测不能证明其为唯一原因。', '', '| 图像 | 人工笔直段轴相对原轴横向差 px | 补充参考 u | 补充参考 v | 检测至补充参考 px |', '|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append('| '+' | '.join([r['image']]+[f'{r[k]:.3f}' if k in r else '—' for k in ('straight_axis_offset','subref_u','subref_v','subref_error')])+' |')
    lines += ['', '蓝线为原人工 PCA 轴，青线为人工笔直段 PCA 轴，绿线为检测轴，橙色斜十字为补充参考。整数掩膜像素中心有约半像素量化尺度；实际标注、灰度阈值与轴向选择引入额外且未标定的不确定度。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n'); print('\n'.join(lines[:7]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--baseline', action='store_true')
    args = parser.parse_args()
    baseline() if args.baseline else evaluate()
