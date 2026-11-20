"""ChArUco 平移标定、质量检查与离线重解。"""
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import math
import os
import shutil
import tempfile
import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares
from .config import CameraCalibConfig
from nest_clear.vision.geometry import Calibration, pixel_to_module

@dataclass
class Step:
    kind: str
    target_xyz: tuple[float, float, float] | None = None
    result: 'CalibResult | None' = None
    code: str | None = None
    reason: str | None = None
    detail: dict = field(default_factory=dict)
    overlay: 'np.ndarray | None' = None

@dataclass
class CalibResult:
    calibration: Calibration
    report: dict

def _board(cfg):
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.dictionary))
    return cv2.aruco.CharucoBoard((cfg.cols, cfg.rows), cfg.square_mm, cfg.marker_mm, dictionary)

def _rot(v):
    return cv2.Rodrigues(np.asarray(v, dtype=float))[0]

def _visible_mask(board_points, square_mm, rvec, tvec, nominal, image_size):
    offsets = np.array([[-1, -1, 0], [-1, 1, 0], [1, -1, 0], [1, 1, 0]], float) * square_mm
    points = (board_points[:, None, :] + offsets).reshape(-1, 3)
    uv = cv2.projectPoints(points, rvec, tvec, nominal.K, nominal.distortion)[0].reshape(-1, 4, 2)
    w, h = image_size
    return np.all((uv[:, :, 0] >= 8) & (uv[:, :, 0] < w-8) &
                  (uv[:, :, 1] >= 8) & (uv[:, :, 1] < h-8), axis=1)

def _glare_margin(predicted, cfg):
    """Median projected spacing of horizontally and vertically adjacent corners."""
    if predicted is not None and np.shape(predicted) == ((cfg.cols-1)*(cfg.rows-1), 2):
        grid = np.asarray(predicted).reshape(cfg.rows-1, cfg.cols-1, 2)
        distances = np.concatenate((np.linalg.norm(np.diff(grid, axis=0), axis=2).ravel(),
                                    np.linalg.norm(np.diff(grid, axis=1), axis=2).ravel()))
        distances = distances[np.isfinite(distances) & (distances > 0)]
        if len(distances): return float(np.median(distances))
    return 10.0

def _plan(cfg, start, center):
    s = np.asarray(start, float); c = np.asarray(center, float)
    if np.any(np.abs(c[:2]-s[:2]) > cfg.xy_limit_mm-5):
        raise ValueError('板中心偏离过大，请调整板或相机位置')
    h = min(cfg.grid_half_mm, *(cfg.xy_limit_mm - np.abs(c[:2]-s[:2])))
    points = [tuple(c)]
    def grid(z, offsets):
        for j, y in enumerate(offsets):
            for x in offsets[::1 if j % 2 == 0 else -1]:
                points.append((c[0]+x, c[1]+y, s[2]+z))
    grid(0, [-h, -h/2, 0, h/2, h])
    points.append(tuple(c))
    grid(cfg.z_outer_mm, [-h, 0, h]); grid(-cfg.z_outer_mm, [-h, 0, h])
    for z in (cfg.z_inner_mm, -cfg.z_inner_mm):
        for x, y in ((0, 0), (h, h), (-h, -h)):
            points.append((c[0]+x, c[1]+y, s[2]+z))
    points.append(tuple(c))
    arr = np.array(points); arr[:, :2] = np.clip(arr[:, :2], s[:2]-cfg.xy_limit_mm, s[:2]+cfg.xy_limit_mm); arr[:, 2] = np.clip(arr[:, 2], s[2]-cfg.z_limit_mm, s[2]+cfg.z_limit_mm)
    assert np.all(np.abs(arr[:,:2]-s[:2]) <= cfg.xy_limit_mm+1e-9) and np.all(np.abs(arr[:,2]-s[2]) <= cfg.z_limit_mm+1e-9)
    return [tuple(map(float, p)) for p in arr]

def _initial(obs, board_points, nominal):
    first = obs[0]; ids = np.asarray(first['ids'], int); pts = np.asarray(first['pixels'], float)
    ok, rv, tv = cv2.solvePnP(board_points[ids], pts, nominal.K, nominal.distortion, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: raise ValueError('第一张图的板位姿求解失败')
    rb = nominal.rotation @ _rot(rv.ravel()); tb = np.asarray(first['actual_xyz']) + nominal.rotation @ tv.ravel()
    return np.r_[nominal.K[0,0], nominal.K[1,1], nominal.K[0,2], nominal.K[1,2], np.pad(nominal.distortion[:5], (0, max(0,5-len(nominal.distortion))))[:5], cv2.Rodrigues(nominal.rotation)[0].ravel(), cv2.Rodrigues(rb)[0].ravel(), tb]

def solve_offline(session_record, cfg, nominal, image_size):
    obs = session_record['observations']; board_points = np.asarray(_board(cfg).getChessboardCorners(), float)
    if len(obs) < 3: raise ValueError('至少需要三张图才能求解')
    indices = np.concatenate([np.asarray(o['ids'], int) for o in obs]); pixels = np.concatenate([np.asarray(o['pixels'], float).reshape(-1,2) for o in obs]); poses = np.concatenate([np.tile(o['actual_xyz'], (len(o['ids']),1)) for o in obs]); xyz = board_points[indices]
    if len(pixels) < 30 or not np.isfinite(pixels).all() or not np.isfinite(poses).all(): raise ValueError('观测不足或数据无效')
    p0 = _initial(obs, board_points, nominal)
    def residual(p, scaled=False):
        r = _rot(p[9:12]); rb = _rot(p[12:15]); t = p[15:18]
        position = poses * (np.r_[p[18:20], 1.] if scaled else 1.)
        pc = (xyz @ rb.T + t - position) @ r
        if np.any(pc[:,2] <= 0): return np.full(pixels.size, 1e6)
        uv = cv2.projectPoints(pc, np.zeros(3), np.zeros(3), np.array([[p[0],0,p[2]],[0,p[1],p[3]],[0,0,1.]]), p[4:9])[0].reshape(-1,2)
        return (uv-pixels).ravel()
    fit = least_squares(residual, p0, method='trf', loss='huber', f_scale=.5, x_scale='jac', max_nfev=1000)
    if not fit.success or not np.isfinite(fit.x).all() or np.any(fit.x[:2] <= 0): raise ValueError('联合优化未收敛：'+fit.message)
    lower = np.r_[np.full(18, -np.inf), [.9, .9]]
    upper = np.r_[np.full(18, np.inf), [1.1, 1.1]]
    diag = least_squares(lambda p: residual(p, True), np.r_[fit.x, [1.,1.]], bounds=(lower, upper), method='trf', loss='huber', f_scale=.5, x_scale='jac', max_nfev=1000)
    if not diag.success: raise ValueError('轴比例诊断未收敛：'+diag.message)
    p = fit.x; r = _rot(p[9:12]); rb = _rot(p[12:15]); t = p[15:18]; normal = rb[:,2]
    k = np.array([[p[0],0,p[2]],[0,p[1],p[3]],[0,0,1.]])
    cal = Calibration(k, p[4:9], r, np.zeros(3), float(t[2]), False, t, normal)
    err = residual(p).reshape(-1,2); per = []; offset = 0
    for o in obs:
        e = err[offset:offset+len(o['ids'])]; offset += len(e); per.append(float(np.sqrt(np.mean(np.sum(e*e, axis=1)))))
    rms = float(np.sqrt(np.mean(np.sum(err*err,axis=1))))
    cells = set()
    for u,v in pixels:
        if 0 <= u < image_size[0] and 0 <= v < image_size[1]: cells.add((min(7,int(u*8/image_size[0])),min(5,int(v*6/image_size[1]))))
    repeats = [o for o in obs if o.get('repeat', False)]
    shifts = []
    for a,b in zip(repeats,repeats[1:]):
        aa = dict(zip(a['ids'],a['pixels'])); bb = dict(zip(b['ids'],b['pixels']))
        distances = [np.linalg.norm(np.asarray(aa[i])-bb[i]) for i in aa.keys() & bb.keys()]
        if distances: shifts.append(float(np.mean(distances)))
    repeat = max(shifts, default=0.)
    try:
        cov = np.linalg.pinv(fit.jac.T @ fit.jac, rcond=1e-15) * (np.dot(fit.fun, fit.fun) / max(1, len(fit.fun)-len(p)))
        std = np.sqrt(np.maximum(0,np.diag(cov)))
        origin0 = np.asarray(obs[0]['actual_xyz'], float)
        def derived(q):
            direction = _rot(q[9:12])[:,2]
            n = _rot(q[12:15])[:,2]
            height = origin0[2] + direction[2] * (n @ (q[15:18] - origin0)) / (n @ direction)
            angle = np.degrees(np.arccos(np.clip(abs(direction[2]), 0., 1.)))
            return np.array([angle, height])
        grad = np.empty((2, len(p)))
        for j in range(len(p)):
            step = 1e-5 * max(1., abs(p[j])); q1=p.copy(); q2=p.copy(); q1[j]+=step; q2[j]-=step
            grad[:,j] = (derived(q1)-derived(q2))/(2*step)
        derived_std = np.sqrt(np.maximum(0, np.diag(grad @ cov @ grad.T)))
        param_std = {'fx':float(std[0]), 'fy':float(std[1]), 'cx':float(std[2]), 'cy':float(std[3]), 'camera_tilt_deg':float(derived_std[0]), 'plane_height_mm':float(derived_std[1])}
    except np.linalg.LinAlgError: param_std = {}
    origin = np.asarray(obs[0]['actual_xyz'],float)
    center = np.asarray(pixel_to_module(p[2], p[3], tuple(origin), cal)); dx = np.asarray(pixel_to_module(p[2]+1,p[3],tuple(origin),cal)); dy = np.asarray(pixel_to_module(p[2],p[3]+1,tuple(origin),cal))
    tilt = lambda n: float(np.degrees(np.arccos(np.clip(abs(n[2])/np.linalg.norm(n),0,1))))
    report = {'rms_px':rms, 'per_image_rms':per, 'axis_scale':{'x':float(diag.x[18]), 'y':float(diag.x[19]), 'z':'不可观测（与焦距/距离耦合），未检查'}, 'axis_scale_note':'Z 比例误差影响相机高度和工作平面绝对 Z 的估计；在标定工作平面上不影响像素到毫米的 XY 换算。', 'param_std':param_std, 'coverage':len(cells)/48, 'repeat_px':repeat, 'um_per_px_at_principal':float(500*(np.linalg.norm(dx-center)+np.linalg.norm(dy-center))), 'focus_z':float(origin[2]), 'n_images':len(obs), 'board_tilt_deg':tilt(normal), 'camera_tilt_deg':tilt(r[:,2])}
    return CalibResult(cal, report)

def save_calibration(result, path, history_dir):
    path = Path(path); history_dir = Path(history_dir); path.parent.mkdir(parents=True,exist_ok=True)
    backup = None
    if path.exists():
        history_dir.mkdir(parents=True,exist_ok=True)
        backup = history_dir / ('camera_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.yaml')
        shutil.copy2(path, backup)
    c = result.calibration
    data = {'K':c.K.tolist(),'distortion':c.distortion.tolist(),'rotation':c.rotation.tolist(),'translation':c.translation.tolist(),'plane_z':float(c.plane_z),'plane_point':c.plane_point.tolist(),'plane_normal':c.plane_normal.tolist(),'placeholder':False}
    name = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False) as f:
            name = f.name; yaml.safe_dump(data,f,allow_unicode=True,sort_keys=False); f.flush(); os.fsync(f.fileno())
        os.replace(name,path)
    finally:
        if name and os.path.exists(name): os.unlink(name)
    return backup

class CameraCalibrationSession:
    def __init__(self, cfg, nominal, image_size, start_xyz):
        self.cfg, self.nominal, self.image_size = cfg, nominal, tuple(image_size)
        self.start_xyz = tuple(map(float,start_xyz)); self._plan = []; self._obs = []; self._terminal = None
        self._board = _board(cfg); self._detector = cv2.aruco.CharucoDetector(self._board)
        self._board_points = np.asarray(self._board.getChessboardCorners(),float); self._rb = self._tb = None
    @property
    def plan(self): return list(self._plan)
    def session_record(self):
        return {'version':1, 'image_size':list(self.image_size), 'start_xyz':list(self.start_xyz), 'plan':[list(p) for p in self._plan], 'observations':self._obs}
    def _fail(self, code, reason, detail, overlay):
        self._terminal = Step('fail',code=code,reason=reason,detail=detail,overlay=overlay); return self._terminal
    def add(self, image, actual_xyz):
        if self._terminal is not None: return self._terminal
        cfg=self.cfg; pose=np.asarray(actual_xyz,float); n=len(self._obs)+1
        if pose.shape != (3,) or not np.isfinite(pose).all() or image is None or image.shape[:2] != (self.image_size[1],self.image_size[0]):
            return self._fail('CAMCAL-SOLVE','图像尺寸或实时坐标无效，请检查采集。',{'image_index':n},None)
        if n == 1 and np.linalg.norm(pose-np.asarray(self.start_xyz)) > cfg.position_tolerance_mm:
            return self._fail('CAMCAL-SOLVE','起始实际位置与起始位偏差超限，请检查模组。',{'image_index':n,'actual_xyz':pose.tolist(),'target_xyz':list(self.start_xyz)},None)
        if self._plan and np.linalg.norm(pose-np.asarray(self._plan[n-2])) > cfg.position_tolerance_mm:
            return self._fail('CAMCAL-SOLVE','实际位置与目标位置偏差超限，请检查模组。',{'image_index':n,'actual_xyz':pose.tolist(),'target_xyz':list(self._plan[n-2])},None)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        overlay = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image[:,:,:3].copy()
        saturated = np.any(image >= cfg.saturation_level,axis=2) if image.ndim == 3 else image >= cfg.saturation_level
        fraction = float(np.mean(saturated)); detail={'image_index':n,'actual_xyz':pose.tolist(),'saturated_fraction':fraction}
        if fraction > cfg.max_saturated_fraction:
            cv2.putText(overlay,f'{n} xyz={tuple(np.round(pose,2))} sat={fraction:.3%}',(10,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,0),2)
            return self._fail('CAMCAL-OVEREXPOSED',f'第 {n} 张饱和比例 {fraction:.3%} 超过 {cfg.max_saturated_fraction:.3%}；请降低照明。',detail,overlay)
        try:
            corners, ids, _, _ = self._detector.detectBoard(gray)
        except cv2.error:
            corners=ids=None
        ids = np.asarray(ids,dtype=int).ravel() if ids is not None else np.empty(0,int)
        pts = np.asarray(corners,dtype=float).reshape(-1,2) if corners is not None else np.empty((0,2))
        predicted = None
        if len(ids) >= 4:
            try:
                ok, rv, tv = cv2.solvePnP(self._board_points[ids], pts, self.nominal.K, self.nominal.distortion)
                if ok:
                    predicted=cv2.projectPoints(self._board_points,rv,tv,self.nominal.K,self.nominal.distortion)[0].reshape(-1,2)
                    if self._rb is None:
                        self._rb=self.nominal.rotation @ _rot(rv.ravel()); self._tb=pose+self.nominal.rotation @ tv.ravel()
            except cv2.error: pass
        if predicted is None and self._rb is not None:
            rc=self.nominal.rotation.T @ self._rb; tc=self.nominal.rotation.T @ (self._tb-pose)
            rv, tv = cv2.Rodrigues(rc)[0], tc
            predicted=cv2.projectPoints(self._board_points,rv,tv,self.nominal.K,self.nominal.distortion)[0].reshape(-1,2)
        if predicted is None:
            predicted=np.empty((0,2)); visible=np.empty(0, dtype=bool)
        else:
            visible=_visible_mask(self._board_points,cfg.square_mm,rv,tv,self.nominal,self.image_size)
        missing = [i for i in np.flatnonzero(visible) if i not in set(ids.tolist())]
        mask=np.zeros(gray.shape,np.uint8)
        if len(predicted):
            polygon=cv2.convexHull(np.round(predicted).astype(np.int32)); cv2.fillConvexPoly(mask,polygon,1)
        elif len(pts)>=3: cv2.fillConvexPoly(mask,cv2.convexHull(np.round(pts).astype(np.int32)),1)
        num, labels, stats, _=cv2.connectedComponentsWithStats((saturated & (mask>0)).astype(np.uint8),8)
        blob=None
        for idx in range(1,num):
            x,y,bw,bh,area=map(int,stats[idx]);
            if area < cfg.glare_blob_min_px: continue
            margin=_glare_margin(predicted, cfg)
            lost=[i for i in missing if x-margin <= predicted[i,0] <= x+bw+margin and y-margin <= predicted[i,1] <= y+bh+margin]
            if len(lost) >= cfg.glare_min_lost_corners:
                blob={'bbox':[x,y,x+bw-1,y+bh-1],'area_px':area,'missing_corners':len(lost)}; cv2.rectangle(overlay,(x,y),(x+bw-1,y+bh-1),(255,0,255),2); break
        for u,v in pts: cv2.circle(overlay,(round(u),round(v)),4,(0,255,0),-1)
        for i in missing:
            u,v=np.round(predicted[i]).astype(int); cv2.drawMarker(overlay,(u,v),(0,0,255),cv2.MARKER_TILTED_CROSS,10,2)
        detail.update({'detected_corners':len(ids),'predicted_corners':int(visible.sum()),'missing_corners':len(missing)})
        cv2.putText(overlay,f'{n}/{len(self._plan)+1 if self._plan else "?"} xyz={tuple(np.round(pose,2))} corners={len(ids)} sat={fraction:.3%}',(10,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,0),2)
        if blob:
            detail.update(blob); x1,y1,x2,y2=blob['bbox']
            return self._fail('CAMCAL-GLARE',f'第 {n} 张（位姿 {tuple(pose)}）画面 ({x1},{y1})-({x2},{y2}) 处有亮斑 面积 {blob["area_px"]} 像素，导致 {blob["missing_corners"]} 个角点丢失；请调整顶光角度后重新标定',detail,overlay)
        expected=int(visible.sum())
        if len(ids)<4 or expected == 0 or len(ids) < cfg.min_corner_ratio*expected:
            return self._fail('CAMCAL-NO-BOARD' if n==1 else 'CAMCAL-COVERAGE',f'第 {n} 张检出 {len(ids)} 个角点，预计可见 {expected} 个；请检查失焦、遮挡或光照。',detail,overlay)
        return self._accept_observation(pose, ids, pts, detail, overlay)

    def _accept_observation(self, pose, ids, pts, detail, overlay):
        cfg = self.cfg
        n = len(self._obs) + 1
        if n == 1:
            center_board=np.array([cfg.cols*cfg.square_mm/2,cfg.rows*cfg.square_mm/2,0.])
            center=self._rb @ center_board+self._tb
            principal=np.asarray(pixel_to_module(self.nominal.K[0,2],self.nominal.K[1,2],tuple(pose),self.nominal))
            target=np.asarray(self.start_xyz).copy(); target[:2]+=center[:2]-principal[:2]
            try: self._plan=_plan(cfg,self.start_xyz,target)
            except ValueError as exc: return self._fail('CAMCAL-OFFCENTER',str(exc),detail,overlay)
        repeat = n>1 and n-2 in (0,26,51)
        self._obs.append({'actual_xyz':pose.tolist(),'ids':ids.tolist(),'pixels':pts.tolist(),'detail':detail,'repeat':repeat})
        if len(self._obs) <= len(self._plan): return Step('move',target_xyz=self._plan[len(self._obs)-1],detail=detail,overlay=overlay)
        try: result=solve_offline(self.session_record(),cfg,self.nominal,self.image_size)
        except (ValueError,cv2.error,np.linalg.LinAlgError) as exc: return self._fail('CAMCAL-SOLVE',f'标定求解失败：{exc}；请检查图像与运动。',detail,overlay)
        report=result.report
        checks=[('CAMCAL-REPEAT',report['repeat_px'],cfg.max_repeat_px,'中心重拍位移')]
        for code,value,limit,label in checks:
            if value>limit: return self._fail(code,f'{label} {value:.4f}px 超过 {limit:.4f}px；请检查漂移、松动或图像质量。',report,overlay)
        if report['coverage']<cfg.min_coverage: return self._fail('CAMCAL-COVERAGE',f'画面覆盖率 {report["coverage"]:.1%} 低于 {cfg.min_coverage:.1%}；请检查视野。',report,overlay)
        for axis in ('x', 'y'):
            scale = report['axis_scale'][axis]
            error=abs(scale-1)
            if error>cfg.max_axis_scale_error: return self._fail('CAMCAL-AXIS-SCALE',f'{axis.upper()} 轴比例误差 {error:.3%} 超过 {cfg.max_axis_scale_error:.3%}，25mm 行程对应误差 {error*25:.4f}mm；请检查丝杠导程、电子齿轮比或标定板比例。',report,overlay)
        for value,limit,label in [(report['rms_px'],cfg.max_rms_px,'整体重投影误差'),(max(report['per_image_rms']),cfg.max_image_rms_px,'单图重投影误差')]:
            if value>limit: return self._fail('CAMCAL-RMS',f'{label} {value:.4f}px 超过 {limit:.4f}px；请检查图像质量。',report,overlay)
        self._terminal=Step('done',result=result,detail=report,overlay=overlay); return self._terminal

    def add_observation(self, corners, ids, actual_xyz):
        """录入已检测角点，供离线数据及合成测试使用；沿用会话规划与质量检查。"""
        if self._terminal is not None:
            return self._terminal
        n = len(self._obs) + 1
        detail = {'image_index': n}
        try:
            pose = np.asarray(actual_xyz, dtype=float)
            pts = np.asarray(corners, dtype=float).reshape(-1, 2)
            raw_ids = np.asarray(ids).reshape(-1)
            if (pose.shape != (3,) or not np.isfinite(pose).all() or
                not np.isfinite(pts).all() or len(pts) != len(raw_ids) or len(pts) < 4 or
                not np.issubdtype(raw_ids.dtype, np.integer) or
                np.any(raw_ids < 0) or np.any(raw_ids >= len(self._board_points)) or
                len(np.unique(raw_ids)) != len(raw_ids)):
                raise ValueError('角点、编号或坐标无效')
            ids = raw_ids.astype(int)
            expected = self.start_xyz if n == 1 else self._plan[n - 2]
            if np.linalg.norm(pose - np.asarray(expected)) > self.cfg.position_tolerance_mm:
                raise ValueError('实际位置与目标位置偏差超限')
            if n == 1:
                ok, rv, tv = cv2.solvePnP(self._board_points[ids], pts, self.nominal.K, self.nominal.distortion)
                if not ok:
                    raise ValueError('第一张图的板位姿求解失败')
                self._rb = self.nominal.rotation @ _rot(rv.ravel())
                self._tb = pose + self.nominal.rotation @ tv.ravel()
        except (ValueError, TypeError, IndexError, cv2.error) as exc:
            return self._fail('CAMCAL-SOLVE', f'观测无效：{exc}；请检查角点和运动。', detail, None)
        detail.update(actual_xyz=pose.tolist(), detected_corners=len(ids), source='observation')
        return self._accept_observation(pose, ids, pts, detail, None)
