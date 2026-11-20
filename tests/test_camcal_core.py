from dataclasses import replace
from sim.charuco_scene import sample_board
import time
import cv2
import numpy as np
import pytest
from nest_clear.camcal import CameraCalibConfig, CameraCalibrationSession, CalibResult, save_calibration, solve_offline
from nest_clear.camcal.session import _board, _plan, _visible_mask
from nest_clear.vision.geometry import Calibration, load_calibration, pixel_to_module

CFG=CameraCalibConfig()
K=np.array([[4650.,0,2020],[0,4655.,1510],[0,0,1.]])
D=np.array([-.06,.03,.0002,-.0001,0.])
R=cv2.Rodrigues(np.array([.008,.011,.026]))[0]@np.diag([1.,-1.,-1.])
B=cv2.Rodrigues(np.array([.002,.0028,.01]))[0]
T=np.array([-52.5,-40.,0.])
N=Calibration(np.array([[4664.,0,2012],[0,4664.,1518],[0,0,1.]]),np.zeros(5),np.diag([1.,-1.,-1.]),np.zeros(3),0.)

def observations(scale=1.):
    pts=np.asarray(_board(CFG).getChessboardCorners(),float); rng=np.random.default_rng(3); start=(0.,0.,110.); rows=[]
    for pose in [start]+_plan(CFG,start,start):
        actual=np.array(pose); true=actual*np.array([scale,1,1]); pc=(pts@B.T+T-true)@R
        uv=cv2.projectPoints(pc,np.zeros(3),np.zeros(3),K,D)[0].reshape(-1,2)
        visible=(uv[:,0]>5)&(uv[:,0]<4019)&(uv[:,1]>5)&(uv[:,1]<3031); ids=np.flatnonzero(visible)
        rows.append({'actual_xyz':list(pose),'ids':ids.tolist(),'pixels':(uv[visible]+rng.normal(0,.05,(len(ids),2))).tolist()})
    return {'observations':rows}

def test_synthetic_accuracy():
    started=time.monotonic(); result=solve_offline(observations(),CFG,N,(4024,3036)); elapsed=time.monotonic()-started
    assert elapsed<30
    assert np.all(np.abs(result.calibration.K.diagonal()[:2]/K.diagonal()[:2]-1)<.0005)
    assert np.max(abs(result.calibration.K[:2,2]-K[:2,2]))<1
    assert abs(result.report['camera_tilt_deg']-np.degrees(np.arccos(abs(R[2,2]))))<.02
    truth=Calibration(K,D,R,np.zeros(3),0.,False,T,B[:,2]); pose=(0.,0.,110.)
    for u,v in [(2020,1510),(0,0),(4023,0),(0,3035),(4023,3035)]:
        assert np.linalg.norm(np.array(pixel_to_module(u,v,pose,result.calibration))-pixel_to_module(u,v,pose,truth)) < (.005 if u==2020 else .03)
    for z in (-3., 3.):
        estimate=replace(result.calibration, plane_point=result.calibration.plane_point+[0,0,z])
        reference=replace(truth, plane_point=truth.plane_point+[0,0,z])
        for u,v in [(2020,1510),(0,0),(4023,0),(0,3035),(4023,3035)]:
            assert np.linalg.norm(np.asarray(pixel_to_module(u,v,pose,estimate))-pixel_to_module(u,v,pose,reference))<.03
    assert result.report['rms_px']<.1
    assert result.report['param_std']['fx']>0

@pytest.mark.parametrize('scale', [1.003,1.0005])
def test_axis_scale(scale):
    result=solve_offline(observations(scale),CFG,N,(4024,3036))
    assert abs(result.report['axis_scale']['x']-scale)<.0001
    assert (abs(result.report['axis_scale']['x']-1)>CFG.max_axis_scale_error)==(scale==1.003)

def test_plan_and_geometry(tmp_path):
    start=(0.,0.,110.); plan=_plan(CFG,start,(4.,-2.,110.)); a=np.array(plan)
    assert len(plan)==52 and plan[0]==plan[26]==plan[-1] == (4.,-2.,110.)
    assert np.all(abs(a[:,:2])<=25) and np.all(abs(a[:,2]-110)<=5)
    assert plan[1][0]<plan[2][0] and plan[6][0]>plan[7][0]
    old=load_calibration('config/calibration/camera.yaml'); assert np.allclose(old.plane_normal,[0,0,1])
    c=replace(N,plane_point=np.array([0.,0.,0.]),plane_normal=np.array([.1,0.,1.]))
    point=np.array(pixel_to_module(2100,1600,start,c)); assert abs(c.plane_normal@(point-c.plane_point))<1e-9
    path=tmp_path/'camera.yaml'; path.write_text('old',encoding='utf-8'); backup=save_calibration(CalibResult(c,{}),path,tmp_path/'history'); assert backup.read_text()=='old'; assert np.allclose(load_calibration(path).plane_normal,c.plane_normal)

def test_overexposure():
    session=CameraCalibrationSession(CFG,N,(100,100),(0.,0.,110.)); step=session.add(np.full((100,100,3),255,np.uint8),(0.,0.,110.))
    assert step.code=='CAMCAL-OVEREXPOSED' and step.detail['saturated_fraction']==1

def rendered_board(offset=0.):
    board=_board(CFG); texture=(board.generateImage((1050,800)).astype(float)*.8).astype('uint8')
    k=np.array([[800.,0,320],[0,800,240],[0,0,1.]])
    nominal=Calibration(k,np.zeros(5),np.diag([1.,-1.,-1.]),np.zeros(3),0.)
    obj=np.array([[0,0,0],[105,0,0],[105,-80,0],[0,-80,0]],float)+[-52.5+offset,40,0]
    pc=(obj-np.array([0,0,110]))@nominal.rotation
    uv=cv2.projectPoints(pc,np.zeros(3),np.zeros(3),k,np.zeros(5))[0].reshape(-1,2).astype('float32')
    transform=cv2.getPerspectiveTransform(np.array([[0,0],[1049,0],[1049,799],[0,799]],'float32'),uv)
    return cv2.warpPerspective(texture,transform,(640,480),borderValue=180),nominal

def test_visible_mask_requires_complete_surrounding_squares():
    points = np.asarray(_board(CFG).getChessboardCorners(), float)
    k = np.array([[100., 0, 50], [0, 100., 50], [0, 0, 1.]])
    nominal = replace(N, K=k, distortion=np.zeros(5))
    rv = np.zeros(3); tv = np.array([-50., -35., 100.])
    uv = cv2.projectPoints(points, rv, tv, k, nominal.distortion)[0].reshape(-1, 2)
    old = (uv[:, 0] >= 8) & (uv[:, 0] < 92) & (uv[:, 1] >= 8) & (uv[:, 1] < 92)
    visible = _visible_mask(points, CFG.square_mm, rv, tv, nominal, (100, 100))
    edge = np.flatnonzero(np.all(points == [10, 35, 0], axis=1))[0]
    inside = np.flatnonzero(np.all(points == [50, 35, 0], axis=1))[0]
    assert old[edge] and not visible[edge]
    assert old[inside] and visible[inside]
    assert 0 < visible.sum() < old.sum()

def test_rendered_detection_glare_and_offcenter():
    image, nominal=rendered_board(); session=CameraCalibrationSession(CFG,nominal,(640,480),(0,0,110))
    step=session.add(image,(0,0,110)); assert step.kind=='move' and step.detail['detected_corners']>100
    glare=image.copy(); cv2.ellipse(glare,(320,240),(30,24),0,0,360,255,-1)
    step=CameraCalibrationSession(CFG,nominal,(640,480),(0,0,110)).add(glare,(0,0,110))
    assert step.code=='CAMCAL-GLARE' and step.detail['bbox']==[290,216,350,264] and step.detail['missing_corners']>0
    image,nominal=rendered_board(24.); step=CameraCalibrationSession(CFG,nominal,(640,480),(0,0,110)).add(image,(0,0,110))
    assert step.code=='CAMCAL-OFFCENTER'

def test_rendered_multiview_solve():
    board=_board(CFG); tex=(board.generateImage((2100,1600)).astype(float)*.8).astype('uint8')
    k=np.array([[800.,0,320],[0,801.,240],[0,0,1.]])
    rot=np.diag([1.,-1.,-1.]); nominal=Calibration(k,np.zeros(5),rot,np.zeros(3),0.)
    obj=np.array([[0,0,0],[105,0,0],[105,-80,0],[0,-80,0]],float)+[-52.5,40,0]
    det=cv2.aruco.CharucoDetector(board); rows=[]
    distortion=np.array([-.06,.03,.0002,-.0001,0.])
    xx,yy=np.meshgrid(np.arange(640),np.arange(480)); grid=np.stack((xx,yy),axis=-1).astype('float32')
    rays=cv2.undistortPoints(grid.reshape(-1,1,2),k,distortion).reshape(480,640,2)
    for x,y,z in [(0,0,110),(-15,-15,110),(15,-15,110),(-15,15,110),(15,15,110),(0,0,105),(12,8,105),(0,0,115),(-12,-8,115)]:
        pose=np.array([x,y,z]); pc=(obj-pose)@rot
        uv=cv2.projectPoints(pc,np.zeros(3),np.zeros(3),k,np.zeros(5))[0].reshape(-1,2).astype('float32')
        # 每个像素去畸变后与板面求交，再采样原始板图。
        module_x=x+rays[:,:,0]*z; module_y=y-rays[:,:,1]*z
        map_x=((module_x+52.5)*2099/105).astype('float32')
        map_y=((40-module_y)*1599/80).astype('float32')
        image=sample_board(tex,map_x,map_y)
        if z!=110: image=cv2.GaussianBlur(image,(3,3),.5)
        corners,ids,_,_=det.detectBoard(image)
        assert ids is not None and len(ids)>35
        rows.append({'actual_xyz':pose.tolist(),'ids':ids.ravel().tolist(),'pixels':corners.reshape(-1,2).tolist()})
    result=solve_offline({'observations':rows},CFG,nominal,(640,480))
    assert result.report['rms_px'] < .5
    assert abs(result.calibration.K[0,0]/800-1)<.02

def test_repeat_drift_measurement():
    record=observations()
    for index, displacement in [(1,0.),(27,1.),(52,2.)]:
        row=record['observations'][index]; row['repeat']=True
        row['pixels']=(np.asarray(row['pixels'])+[displacement,0]).tolist()
    result=solve_offline(record,CFG,N,(4024,3036))
    assert result.report['repeat_px'] > CFG.max_repeat_px


def run_synthetic_session(*, x_scale=1., z_scale=1., repeat_drift=0.):
    """按会话实时 plan 生成角点，轴误差作用于相对起始位的运动。"""
    start = np.array([0., 0., 110.])
    session = CameraCalibrationSession(CFG, N, (4024, 3036), start)
    points = np.asarray(_board(CFG).getChessboardCorners(), float)
    pose = start.copy()
    for index in range(53):
        true_pose = start + (pose - start) * [x_scale, 1., z_scale]
        pc = (points @ B.T + T - true_pose) @ R
        uv = cv2.projectPoints(pc, np.zeros(3), np.zeros(3), K, D)[0].reshape(-1, 2)
        visible = (uv[:, 0] > 8) & (uv[:, 0] < 4016) & (uv[:, 1] > 8) & (uv[:, 1] < 3028)
        ids = np.flatnonzero(visible)
        if index == 52:
            uv[visible] += [repeat_drift, 0.]
        step = session.add_observation(uv[visible], ids, pose)
        if step.kind != 'move':
            return session, step
        pose = np.asarray(step.target_xyz)
    pytest.fail('会话未结束')


@pytest.mark.parametrize('z_scale', [1., 1.005])
def test_session_normal_and_z_scale_unobservable(z_scale):
    session, step = run_synthetic_session(z_scale=z_scale)
    assert step.kind == 'done'
    report = step.result.report
    assert {'rms_px', 'per_image_rms', 'axis_scale', 'axis_scale_note', 'param_std', 'coverage',
            'repeat_px', 'um_per_px_at_principal', 'focus_z', 'n_images', 'board_tilt_deg', 'camera_tilt_deg'} <= report.keys()
    assert report['n_images'] == 53
    assert report['axis_scale']['z'] == '不可观测（与焦距/距离耦合），未检查'
    truth = Calibration(K, D, R, np.zeros(3), 0., False, T, B[:, 2])
    pose = (0., 0., 110.)
    errors = [np.linalg.norm(np.asarray(pixel_to_module(u, v, pose, step.result.calibration))[:2] -
                             np.asarray(pixel_to_module(u, v, pose, truth))[:2])
              for u in np.linspace(0, 4023, 9) for v in np.linspace(0, 3035, 7)]
    print(f'Z 轴比例 {z_scale:.4f}：全视野 63 点 XY 最大误差 {max(errors):.9f} mm')
    assert max(errors) < .005
    assert len(session.session_record()['observations']) == 53


def test_session_x_scale_failure():
    _, step = run_synthetic_session(x_scale=1.003)
    assert step.kind == 'fail' and step.code == 'CAMCAL-AXIS-SCALE'
    assert abs(step.detail['axis_scale']['x'] - 1.003) < .0001


def test_session_third_repeat_drift_failure():
    _, step = run_synthetic_session(repeat_drift=1.)
    assert step.kind == 'fail' and step.code == 'CAMCAL-REPEAT'
    assert step.detail['repeat_px'] > .9
