"""三棱针背光图像检测。坐标为 (u, v)，方向指向针尖。"""
from dataclasses import dataclass, replace
from time import perf_counter
import cv2
import numpy as np

DETECTOR_VERSION = "trident-3"


@dataclass(frozen=True)
class TridentConfig:
    x_range: tuple[int, int] | None = None
    min_height: int = 100
    max_axis_rms_px: float = 0.7
    # 逐行曝光时运动会造成明显畸变；与精定位门限分别配置。
    max_distortion_axis_rms_px: float = 2.0
    min_tip_contrast: float = 40.0
    max_sequence_std_px: float = 0.1

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
    axis_rms_px: float | None = None
    tip_contrast: float | None = None
    edge_fit: dict | None = None

    @property
    def coarse(self): return self.status != '精定位'

    @property
    def focused(self): return self.status == '精定位'

    @property
    def edge_width_px(self): return (self.sharpness or {}).get('edge_width_px')

    @property
    def width_px(self): return None

    @property
    def confidence(self): return 1.0 if self.status == '精定位' else 0.0

def _outer_edge(profile):
    """Find the white-to-tool transition in a profile ordered outside-in."""
    # Samples 13–20 px outside the connected-component edge clear both the
    # blurred transition and a reflective facet omitted by the component.
    back = float(np.median(profile[:8]))
    # Require a sustained departure from the local white level. Three-pixel
    # support suppresses single-pixel noise while retaining a bright facet.
    deficit = back - profile
    support = (deficit[:-2] + deficit[1:-1] + deficit[2:]) / 3
    candidates = np.flatnonzero(support[8:] >= max(10, .05 * back))
    if not len(candidates):
        return None
    seed = int(candidates[0]) + 8
    # Estimate the tool platform close to the transition so brightness on the
    # opposite facet cannot pull the half-height crossing into the shaft.
    inner = profile[seed+3:seed+6]
    if len(inner) < 3:
        return None
    body = float(np.median(inner))
    contrast = back - body
    if contrast < 20:
        return None
    start, stop = max(0, seed-5), min(len(profile)-1, seed+5)
    vals = profile[start:stop+1]

    def cross(fraction):
        level = body + fraction * contrast
        indices = np.flatnonzero((vals[:-1] > level) & (vals[1:] <= level))
        if not len(indices):
            return None
        j = indices[np.argmin(abs(indices+start-seed))]
        return float(start+j+(level-vals[j])/(vals[j+1]-vals[j]))

    # A fractional drop from local white is less sensitive to the changing
    # intensity of a reflective facet. Keep the 10–90% width measured against
    # the local tool and background levels for the sharpness diagnostic.
    center = cross(1 - .06*back/contrast)
    dark, bright = cross(.1), cross(.9)
    if center is None:
        return None
    # A narrow halo beside a high-contrast silhouette can cross the white
    # threshold before the optical edge. Use the deeper transition only if
    # the core is close to opaque and the two crossings are within the
    # same few-pixel transition. A resolved bright facet keeps its own edge.
    # The component seed is at column 20 in this outside-in profile.
    core = float(np.median(profile[23:28]))
    core_fraction = core / back
    opacity_weight = min(1., max(0., (.25 - core_fraction) / .05))
    if opacity_weight > 0:
        level = back + .4 * (core - back)
        crossings = np.flatnonzero((profile[10:25] > level) & (profile[11:26] <= level))
        if len(crossings):
            j = int(crossings[0]) + 10
            core_edge = j + (level-profile[j])/(profile[j+1]-profile[j])
            gap = core_edge - center
            if gap > 0:
                center += opacity_weight * gap * np.exp(-(gap/4)**2)

    width = abs(dark-bright) if dark is not None and bright is not None else None
    gradient = float(np.max(-np.diff(vals))/contrast)
    return center, width, gradient, body, back


class TridentTipDetector:
    def __init__(self, config=None):
        self.config = config or TridentConfig()
        if isinstance(self.config, dict): self.config = TridentConfig(**self.config)
        self.max_edge_width_px = None

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
        # Robust straight section: exclude glue above and the final 45 pixels near the prongs. The
        # lower shaft increases leverage when extrapolating the axis to the tip.
        start_y=max(int(end*.40),end-300); stop_y=end-45
        if stop_y-start_y<30: return done(diagnostic='针身直线段不足')
        rows, edges, platforms, gradients, widths, esf_profiles, band_gradients = [], [], [], [], [], [], []
        offsets_esf = np.arange(-6, 6.01, .25)
        pixel_columns = np.arange(width)
        previous_span = None
        for y in range(start_y, stop_y, 3):
            xs = np.flatnonzero(comp[y])
            if len(xs) < 9:
                continue
            l, r = int(xs[0]), int(xs[-1])
            if y >= end-80 and previous_span is not None:
                old_l, old_r = previous_span
                # An abrupt loss of one side near the fork marks the end of
                # the straight shaft. Translation moves both sides together.
                cut = max(6, .2*(old_r-old_l))
                if ((l-old_l >= cut and abs(r-old_r) <= 1)
                        or (old_r-r >= cut and abs(l-old_l) <= 1)):
                    break
            previous_span = l, r
            if l < 20 or r+20 >= width or r-l < 8:
                continue
            row = crop[y].astype(float)
            left = _outer_edge(row[l-20:l+9])
            right = _outer_edge(row[r-8:r+21][::-1])
            if left is None or right is None:
                continue
            left_u, right_u = l-20+left[0], r+20-right[0]
            rows.append(y)
            edges.append((left_u + lo, right_u + lo))
            platforms.append((left[3:], right[3:]))
            for u, sign, info in ((left_u, -1, left), (right_u, 1, right)):
                _, edge_width, gradient, body, background = info
                gradients.append(gradient)
                if edge_width is not None:
                    widths.append(edge_width)
                coordinates = u+sign*offsets_esf
                if coordinates.min() >= 0 and coordinates.max() < width:
                    esf_profiles.append((np.interp(coordinates, pixel_columns, row)-body)/(background-body))
            band_gradients.extend(np.abs(np.diff(row[int(left_u)-5:int(right_u)+7])))
        if len(rows) < 10:
            return done(diagnostic='有效针身边缘不足')
        edges = np.asarray(edges)
        centers = edges.mean(axis=1)
        # Keep only continuous pairs of edges within the straight shaft band.
        # If many rows disagree, include them in RMS below to expose motion.
        slope, intercept = np.polyfit(rows, edges, 1)
        residual = edges - (np.asarray(rows)[:, None]*slope + intercept)
        deviation = np.abs(residual - np.median(residual, axis=0))
        # Four MADs are about 2.7 standard deviations for Gaussian edge
        # noise. Test both edges, since a reflection can move just one edge.
        good = np.all(deviation < np.maximum(1., 4*np.median(deviation, axis=0)), axis=1)
        if np.count_nonzero(good) < 10:
            return done(diagnostic='连续针身边缘不足')
        edge_slopes, edge_intercepts = np.polyfit(np.asarray(rows)[good], edges[good], 1)
        slope, intercept = float(edge_slopes.mean()), float(edge_intercepts.mean())
        levels = np.median(np.asarray(platforms)[good], axis=0)
        edge_fit = {side: {'tool_gray': float(body), 'background_gray': float(background),
                           'slope_du_dv': float(edge_slope)}
                    for side, (body, background), edge_slope
                    in zip(('left', 'right'), levels, edge_slopes)}
        d=np.array([slope,1.]); d/=np.linalg.norm(d); origin=np.array([intercept+slope*np.mean(rows),np.mean(rows)])
        esf_width=None
        if len(esf_profiles)>=20:
            esf=np.mean(esf_profiles,axis=0)
            def esf_cross(level):
                indices=np.flatnonzero((esf[:-1]<=level)&(esf[1:]>level))
                if not len(indices): return None
                i=indices[np.argmin(np.abs(offsets_esf[indices]))]
                return float(offsets_esf[i]+(level-esf[i])*(offsets_esf[i+1]-offsets_esf[i])/(esf[i+1]-esf[i]))
            x10,x90=esf_cross(.1),esf_cross(.9)
            if x10 is not None and x90 is not None and x90>x10: esf_width=x90-x10
        # Measure the same inliers used by the robust fit; a few spurious edge
        # crossings must not dominate RMS. If many rows were rejected, retain
        # them in the quality metric: a moving tool can produce two modes.
        quality_rows = good if np.mean(good) >= .8 else np.ones(len(good), dtype=bool)
        axis_rms=float(np.sqrt(np.mean((np.asarray(centers)[quality_rows]-(slope*np.asarray(rows)[quality_rows]+intercept))**2)))
        # Sample axis-parallel profiles inside the body envelope. This is independent
        # of whether thin prongs remain attached in the thresholded component.
        normal=np.array([d[1],-d[0]])
        sample_y=np.arange(max(0,end-90), min(h-2,end+6), dtype=np.float32)
        mid_y=end-90
        row_x=np.flatnonzero(comp[max(0,mid_y)])
        half=max(8., min(35., (float(row_x[-1]-row_x[0])/2+8) if len(row_x) else 20.))
        offsets=np.arange(-half,half+0.1,.5,dtype=np.float32)
        t=(sample_y-origin[1])/d[1]
        xmap=(origin[0]+d[0]*t[:,None]+normal[0]*offsets[None,:]-lo).astype(np.float32)
        ymap=np.broadcast_to((origin[1]+d[1]*t[:,None]+normal[1]*offsets[None,:]),xmap.shape).astype(np.float32)
        values=cv2.remap(crop,xmap,ymap,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        backgrounds=cv2.remap(bg,xmap,ymap,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        # Local dark platform per profile; require measurable darkness over several
        # adjacent samples so isolated dust or LED texture is not a valid endpoint.
        body=np.percentile(values[:min(45,len(values))],10,axis=0)
        contrast=backgrounds-body[None,:]
        darkness=(backgrounds-values)/np.maximum(contrast,1)
        dark=darkness>=.5
        supported=dark.copy()
        supported[:4] = False
        for k in range(1,5): supported[4:] &= dark[4-k:len(dark)-k]
        supported &= contrast>=30
        tips=[]; tip_contrasts=[]
        for col in range(len(offsets)):
            ys=np.flatnonzero(supported[:,col])
            if not len(ys): continue
            j=int(ys[-1])
            if j+1>=len(sample_y) or darkness[j+1,col]>=.5: continue
            den=darkness[j,col]-darkness[j+1,col]
            frac=float(np.clip((darkness[j,col]-.5)/den,0,1)) if den>0 else 0.
            yy=float(sample_y[j]+frac)
            tt=(yy-origin[1])/d[1]
            pt=origin+d*tt+normal*float(offsets[col])
            tips.append(tuple(pt)); tip_contrasts.append(float(contrast[j,col]))
        if not tips: return done(diagnostic='针尖区域缺少可靠的轴向灰度穿越', edge_fit=edge_fit)
        far_index=max(range(len(tips)),key=lambda i:np.dot(np.asarray(tips[i])-origin,d)); far=tips[far_index]; tip_contrast=tip_contrasts[far_index]; point=origin+np.dot(np.asarray(far)-origin,d)*d
        reasons=[]
        if axis_rms>self.config.max_distortion_axis_rms_px: reasons.append('工具在动或画面畸变')
        elif axis_rms>self.config.max_axis_rms_px: reasons.append('针身轴拟合残差过大')
        if tip_contrast<self.config.min_tip_contrast: reasons.append('末端对比度不足')
        if len(tips)<3: reasons.append('末端有效剖面不足')
        if far[1]>=end+4: reasons.append('末端靠近搜索边界，可能存在断齿或背景干扰')
        status='粗定位' if reasons else '精定位'
        diagnostic='；'.join(reasons) if reasons else '单帧质量检查通过；尚未验证五帧一致性'
        mask=np.zeros_like(gray); mask[:,lo:hi]=comp.astype(np.uint8)
        return done(point_uv=tuple(point),axis={'point':tuple(origin),'direction':tuple(d)},farthest_prong_uv=far,prong_tips=tuple(tips),status=status,diagnostic=diagnostic,tool_mask=mask,sharpness={'edge_width_px':float(np.median(widths)) if widths else None,'normalized_gradient':float(np.median(gradients)), 'esf_width_px':esf_width, 'band_gradient_p995':float(np.percentile(band_gradients,99.5)) if band_gradients else None},sampling_rows=tuple(rows),axis_rms_px=axis_rms,tip_contrast=tip_contrast,edge_fit=edge_fit)

    def detect_sequence(self, images):
        """Evaluate five independent frames, retaining positions even on rejection."""
        images=list(images)
        if len(images)!=5:
            raise ValueError('五帧一致性检查需要恰好五帧')
        results=[self.detect(image) for image in images]
        if any(r.point_uv is None for r in results):
            reason='五帧中存在未识别帧'
        else:
            spread=np.std(np.asarray([r.point_uv for r in results]),axis=0)
            reason=(f'五帧坐标标准差超限（u={spread[0]:.3f}，v={spread[1]:.3f} px）'
                    if np.any(spread>self.config.max_sequence_std_px) else '')
        if reason:
            return [replace(r,status='粗定位' if r.point_uv is not None else r.status,
                            diagnostic=r.diagnostic+'；'+reason) for r in results]
        return [replace(r,diagnostic=r.diagnostic.replace('尚未验证五帧一致性','五帧一致性检查通过')) for r in results]
