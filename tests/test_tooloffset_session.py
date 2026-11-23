import math
from types import SimpleNamespace

import numpy as np
import pytest
from nest_clear.tooloffset.session import Session, fit_similarity, nominal_targets
from nest_clear.vision.geometry import Calibration


def calibration():
    return Calibration(np.array([[2000.,0,2000],[0,2000.,1500],[0,0,1]]), np.zeros(5), np.diag([1.,-1.,-1.]), np.zeros(3), -105.)


def world_to_uv(xy, pose):
    scale = 2000 / 99.9
    return (2000 + scale*(xy[0]-pose[0]), 1500 - scale*(xy[1]-pose[1]))


def target(index, xy, pose, quality=0.):
    return SimpleNamespace(index=index, ring_uv=world_to_uv(xy, pose), cross_error_mm=quality)


def ready(mode='center', cfg=None, scale=1.005):
    sess = Session(calibration(), (130.,70.,-5.), ((80,180),(-10,150),(-15,0)), cfg, mode, 1)
    # 可见卡面上两点精确生成模组坐标。
    nominal = nominal_targets()
    world = {i: np.array([130.,70.]) + scale*p for i,p in nominal.items()}
    observed = [target(i, world[i], sess.start) for i in (2,5)]
    return sess, world, sess.image(observed, sess.start, (4000,3000))


def measure(sess, world, noise=(0,0)):
    idx, pose = sess.current, sess.target_pose
    assert sess.image([target(idx, world[idx], pose)], pose, (4000,3000)) == 'tool'
    p = sess.measured[idx]
    return sess.add_sample((p[0]+2.2+noise[0], p[1]-1.3+noise[1], -5))


def test_similarity_rotation_and_scale():
    src = np.array([[0,0],[30,0],[0,30]])
    a = .1; expected = 1.005*np.array([[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]])
    matrix, trans, scale = fit_similarity(src, src@expected.T + [80,20])
    assert matrix == pytest.approx(expected)
    assert trans == pytest.approx([80,20])
    assert scale == pytest.approx(1.005)


def test_outlier_remeasured_and_converges():
    sess, world, step = ready(cfg={'point_order':['T02','T05','T26','T29','T12','T19'], 'min_points':5})
    assert step == 'continue'
    assert measure(sess,world,(.01,-.01)) == 'continue'
    assert measure(sess,world,(.3,0)) == 'continue'
    assert sess.samples[-1]['outlier']
    while step != 'confirm':
        step = measure(sess,world)
        assert step != 'failed', sess.failure
    assert [s['target'] for s in sess.samples].count(sess.samples[1]['target']) == 2
    assert sess.result()['n'] >= 5
    assert sess.result()['offset_xy_mm'] == pytest.approx([2.2,-1.3], abs=.01)


def test_edge_four_images_no_tool():
    sess, world, step = ready('center_edge')
    sess.travel = ((-500,500),(-500,500),(-15,0))
    while sess.phase == 'center':
        step = measure(sess,world)
    assert sess.phase == 'edge'
    while sess.phase == 'edge':
        pose, index = sess.target_pose, sess.current
        step = sess.image([target(index,world[index],pose)], pose, (4000,3000))
    assert step == 'confirm' and len(sess.edges) == 4
    assert max(x['distance_mm'] for x in sess.edges) < 1e-8


def test_bad_overview_and_scale_and_quality():
    sess, world, _ = ready()
    sess.phase = 'overview'
    assert sess.image([], sess.start, (4000,3000)) == 'failed'
    sess, world, step = ready(scale=1.05)
    assert step == 'failed' and '尺寸不符' in sess.failure
    sess, world, _ = ready()
    assert sess.image([target(sess.current,world[sess.current],sess.target_pose,.2)],sess.target_pose,(4000,3000)) == 'failed'
    assert '识别质量差' in sess.failure


def test_max_points_not_converged():
    sess, world, _ = ready(cfg={'max_points':5, 'min_points':5, 'se_tol_mm':.00001, 'outlier_mm':1.})
    for n in range(5):
        step = measure(sess,world,(.02*(-1)**n,0))
    assert step == 'failed' and '多点不一致' in sess.failure


def test_gaussian_noise_with_large_outlier():
    sess, world, step = ready(cfg={'outlier_mm':.1, 'min_points':5, 'max_points':12})
    rng = np.random.default_rng(43)
    while step not in ('confirm', 'failed'):
        noise = rng.normal(0, .03, 2)
        if len(sess.samples) == 2: noise += [.3, 0]
        step = measure(sess, world, noise)
    assert step == 'confirm', sess.failure
    assert any(sample['outlier'] for sample in sess.samples)
    assert all(value <= .02 for value in sess.result()['se'])


@pytest.mark.parametrize('board,card,camera_z,normal_sign', [(3., .1, -5., 1), (3.15, .1, -5., 1), (.6, .25, 12., -1), (1.5, .2, 30., 1)])
def test_card_plane_height_at_image_edges(board, card, camera_z, normal_sign):
    from dataclasses import replace
    from nest_clear.vision.geometry import pixel_to_module
    normal = np.array([.04, -.03, 1.]); normal /= np.linalg.norm(normal)
    tray = np.array([0., 0., -110.])
    base = replace(calibration(), plane_point=tray + board * normal, plane_normal=normal * normal_sign)
    sess = Session(base, (0, 0, camera_z), ((-500,500),)*3, {'card_thickness_mm':card}, board_thickness_mm=board)
    truth = replace(base, plane_point=tray + card * normal)
    for uv in ((0,0),(3999,0),(0,2999),(3999,2999)):
        actual = pixel_to_module(*uv, (0,0,camera_z), sess.cal)
        expected = pixel_to_module(*uv, (0,0,camera_z), truth)
        assert np.linalg.norm(np.asarray(actual)-expected) < .005


def test_selection_margin_spread_and_three_points():
    sess, world, step = ready(cfg={'min_points':3, 'prior_offset_mm':[29.3,24.2]})
    first = sess.current
    def margin(i):
        x, y = sess.predicted[i] + [29.3,24.2]
        return min(x-80, 180-x, y+10, 150-y)
    accessible = [i for i in sess.predicted if i not in {int(s['target'][1:]) for s in sess.skipped}]
    assert margin(first) == pytest.approx(max(map(margin, accessible)))
    step = measure(sess, world)
    second = sess.current
    remaining = [i for i in sess.predicted if i != first and i not in {int(s['target'][1:]) for s in sess.skipped}]
    assert np.linalg.norm(sess.predicted[second]-sess.predicted[first]) == pytest.approx(max(np.linalg.norm(sess.predicted[i]-sess.predicted[first]) for i in remaining))
    assert measure(sess, world) == 'continue'
    assert measure(sess, world) == 'confirm'
    assert sess.result()['n'] == 3


def test_clamped_camera_and_actual_pixel():
    sess, world, _ = ready(cfg={'min_points':5})
    index = 0
    world[index] = np.array([75., 70.])
    sess.predicted = {index: world[index]}
    sess.samples = []; sess.skipped = []; sess.measured = {}; sess.pending_retests = set()
    assert sess._move_next() == 'continue'
    assert sess.camera_clamped and sess.target_pose[0] == 80
    pose = sess.target_pose
    assert sess.image([target(index,world[index],pose)], pose, (4000,3000)) == 'tool'
    assert sess.measured[index] == pytest.approx(world[index])
    sess.add_sample((world[index][0]+2.2,world[index][1]-1.3,-5))
    assert sess.samples[0]['camera_clamped']
    assert sess.samples[0]['target_uv'] == pytest.approx(world_to_uv(world[index],pose))


def test_insufficient_targets_message():
    sess, world, _ = ready(cfg={'min_points':3})
    sess.predicted = {0: np.array([-1000., -1000.])}
    sess.samples = []; sess.skipped = []
    assert sess._move_next() == 'failed'
    assert '可用目标不足：已测有效 0 个，需 3 个' in sess.failure
    assert sess.skipped[0]['reason'] == '相机不可达'


def test_field_offset_finds_five():
    sess = Session(calibration(), (130.,70.,-5.), ((80,180),(-10,150),(-15,0)), {'min_points':5, 'prior_offset_mm':[29.3,24.2]})
    world = {i: np.array([100.,0.]) + p for i,p in nominal_targets().items()}
    old = [2,5,26,29,12,19,3,28,10,21]
    assert sum(sess._within((world[i][0]+29.3, world[i][1]+24.2,-5)) and sess._within(sess._pose_for(world[i], (2000,1500))) for i in old) <= 3
    step = sess.image([target(i,world[i],sess.start) for i in (2,5)],sess.start,(4000,3000))
    while step == 'continue':
        i, pose = sess.current, sess.target_pose
        assert sess.image([target(i,world[i],pose)],pose,(4000,3000)) == 'tool'
        p = sess.measured[i]
        actual = (p[0]+29.3,p[1]+24.2,-5)
        assert sess._within(actual)
        step = sess.add_sample(actual)
    assert step == 'confirm', sess.failure
    assert sess.result()['n'] >= 5
