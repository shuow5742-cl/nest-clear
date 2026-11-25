"""三棱针背光图像检测。坐标为 (u, v)，方向指向针尖。"""
from dataclasses import dataclass
from time import perf_counter
import cv2
import numpy as np

@dataclass(frozen=True)
class TridentConfig:
    x_range: tuple[int, int] | None = None
    min_height: int = 100

@dataclass
class TridentResult:
    point_uv: tuple | None = None
    axis: dict | None = None
    farthest_prong_uv: tuple | None = None
    prong_tips: tuple = ()
    status: str = '未识别'
    diagnostic: str = ''
    tool_mask: np.ndarray | None = None
    sharpness: dict | None = None
    elapsed_ms: float = 0.0
    sampling_rows: tuple = ()

class TridentTipDetector:
    def __init__(self, config=None):
        self.config = config or TridentConfig()
        if isinstance(self.config, dict): self.config = TridentConfig(**self.config)

    def detect(self, gray):
        start = perf_counter()
        def done(**kw):
            return TridentResult(elapsed_ms=(perf_counter()-start)*1000, **kw)
        if not isinstance(gray, np.ndarray) or gray.ndim != 2 or gray.dtype != np.uint8 or not gray.size:
            raise ValueError('需要非空 uint8 灰度图')
        h,w = gray.shape
        # Downsampling makes selection inexpensive; touching bottom excludes the support bars.
        small = cv2.resize(gray, (w//4,h//4), interpolation=cv2.INTER_AREA)
        n, lab, stats, _ = cv2.connectedComponentsWithStats((small < 110).astype(np.uint8), 8)
        candidates=[]
        for x,y,ww,hh,area in stats[1:]:
            cx=(x+ww/2)*4
            if y <= 1 and y+hh < small.shape[0]-1 and x>0 and x+ww<small.shape[1]-1 and hh*4>=self.config.min_height and hh>ww*1.5 and (self.config.x_range is None or self.config.x_range[0]<=cx<self.config.x_range[1]):
                candidates.append((area,x,ww))
        if len(candidates)!=1:
            return done(diagnostic='顶部细长候选数量不唯一' if candidates else '未找到顶部细长候选')
        _,x,ww=candidates[0]; lo=max(0,x*4-35); hi=min(w,(x+ww)*4+35)
        crop=gray[:,lo:hi]
        # Local bright envelope, estimated over a wide neighborhood, accommodates LED shading.
        bg=cv2.dilate(crop, cv2.getStructuringElement(cv2.MORPH_RECT,(61,31)))
        binary=(crop.astype(np.float32) < bg.astype(np.float32)*0.78).astype(np.uint8)
        n, labels, stats, _=cv2.connectedComponentsWithStats(binary,8)
        choices=[i for i in range(1,n) if stats[i,1]<=2 and stats[i,1]+stats[i,3]<h-1 and stats[i,4]>100]
        if len(choices)!=1: return done(diagnostic='局部阈值下工具连通域缺失或歧义')
        idx=choices[0]; comp=(labels==idx); yy,xx=np.nonzero(comp)
        end=int(yy.max()); width=hi-lo
        # Robust straight section: exclude glue above and the final 80 pixels of prongs.
        start_y=max(int(end*.40),end-300); stop_y=end-80
        if stop_y-start_y<30: return done(diagnostic='针身直线段不足')
        centers=[]; rows=[]; gradients=[]; widths=[]
        for y in range(start_y,stop_y,3):
            xs=np.flatnonzero(comp[y]);
            if len(xs)<5: continue
            l,r=int(xs[0]),int(xs[-1]); row=crop[y].astype(float)
            if l<8 or r+8>=width or r-l<8: continue
            def edge(k, rising):
                body=np.median(row[l+4:r-3]); back=np.median(row[k+3:k+9] if rising else row[k-8:k-2]); contrast=back-body
                if contrast<20: return None
                a=max(0,k-5); b=min(width,k+6); vals=row[a:b]; inds=np.arange(a,b)
                def cross(level):
                    v=vals-level; ix=np.flatnonzero((v[:-1]<=0)&(v[1:]>0) if rising else (v[:-1]>0)&(v[1:]<=0))
                    if not len(ix): return None
                    j=ix[np.argmin(abs(inds[ix]-k))]; return float(inds[j]+(level-vals[j])/(vals[j+1]-vals[j]))
                c=cross(body+.5*contrast); c1=cross(body+.1*contrast); c9=cross(body+.9*contrast)
                grad=np.max(np.abs(np.diff(vals)))/contrast
                return c, abs(c9-c1) if c1 is not None and c9 is not None else None,grad
            left=edge(l,False); right=edge(r,True)
            if left is None or right is None or left[0] is None or right[0] is None: continue
            centers.append((left[0]+right[0])/2+lo); rows.append(y)
            gradients.extend([left[2],right[2]])
            widths.extend(z[1] for z in (left,right) if z[1] is not None)
        if len(rows)<10: return done(diagnostic='有效针身边缘不足')
        slope, intercept=np.polyfit(rows,centers,1); residual=np.asarray(centers)-(slope*np.asarray(rows)+intercept)
        good=np.abs(residual-np.median(residual))<max(1.,3*np.median(np.abs(residual-np.median(residual))))
        slope,intercept=np.polyfit(np.asarray(rows)[good],np.asarray(centers)[good],1)
        d=np.array([slope,1.]); d/=np.linalg.norm(d); origin=np.array([intercept+slope*np.mean(rows),np.mean(rows)])
        # Components below the straight body: retain only connected tool pixels.
        tipmask=comp.copy(); tipmask[:max(0,end-75)]=0
        k,tl,ts,_=cv2.connectedComponentsWithStats(tipmask.astype(np.uint8),8)
        tips=[]
        for i in range(1,k):
            if ts[i,4]<3: continue
            ys,xs=np.nonzero(tl==i); proj=(xs+lo-origin[0])*d[0]+(ys-origin[1])*d[1]; j=np.argmax(proj)
            u=int(xs[j]); v=int(ys[j]);
            # interpolate the end profile to local half contrast
            bgval=float(bg[v,u]); body=float(np.percentile(crop[max(0,v-8):v+1,max(0,u-1):u+2],10)); level=(bgval+body)/2
            sub=float(v)
            if v+1<h and crop[v+1,u]>crop[v,u] and crop[v,u]<=level<=crop[v+1,u]: sub=v+(level-float(crop[v,u]))/(float(crop[v+1,u])-float(crop[v,u]))
            tips.append((float(u+lo),sub))
        if not tips: return done(diagnostic='未找到连接针身的齿尖')
        far=max(tips,key=lambda p: np.dot(np.asarray(p)-origin,d)); point=origin+np.dot(np.asarray(far)-origin,d)*d
        mask=np.zeros_like(gray); mask[:,lo:hi]=comp.astype(np.uint8)
        return done(point_uv=tuple(point),axis={'point':tuple(origin),'direction':tuple(d)},farthest_prong_uv=far,prong_tips=tuple(tips),status='精定位' if len(tips)>=2 else '粗定位',diagnostic='针身边缘拟合与连接齿尖' if len(tips)>=2 else '未能区分多个齿尖，末端位置可能有歧义',tool_mask=mask,sharpness={'edge_width_px':float(np.median(widths)) if widths else None,'normalized_gradient':float(np.median(gradients))},sampling_rows=tuple(rows))
