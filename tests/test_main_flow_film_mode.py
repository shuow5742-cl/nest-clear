"""菲林模式依赖的工具对准数据验证。"""
import yaml

from nest_clear.tooloffset.store import load_pick_z


def test_tool_z_uses_each_tools_own_all_samples(tmp_path):
    path = tmp_path / 'tool_offset.yaml'
    path.write_text(yaml.safe_dump({
        'tool1': {'samples': [{'tool_xyz_mm': [1, 2, -7]},
                              {'tool_xyz_mm': [1, 2, -5], 'outlier': True}]},
        'tool2': {'samples': [{'tool_xyz_mm': [1, 2, -3]},
                              {'tool_xyz_mm': [1, 2, -4]}]},
    }))
    assert load_pick_z(path, 1) == (-5, [-7, -5])
    assert load_pick_z(path, 2) == (-3, [-3, -4])


def test_missing_or_invalid_sample_is_not_safe(tmp_path):
    path = tmp_path / 'tool_offset.yaml'
    assert load_pick_z(path, 1) is None
    path.write_text(yaml.safe_dump({'tool1': {'samples': [
        {'tool_xyz_mm': [1, 2, -5]}, {'tool_xyz_mm': [1, 2, float('nan')]}]}}))
    assert load_pick_z(path, 1) is None


def test_mode_persistence(tmp_path):
    from nest_clear.film_mode import read_mode, write_mode
    assert read_mode(tmp_path) is False
    write_mode(tmp_path, True)
    assert read_mode(tmp_path) is True
    write_mode(tmp_path, False)
    assert read_mode(tmp_path) is False


def test_film_worker_detects_whole_image_and_matches_label(monkeypatch):
    from types import SimpleNamespace
    from nest_clear.workers.vision import VisionWorker
    from nest_clear.tooloffset import detection
    import numpy as np
    seen = []
    def detect(image):
        seen.append(image.shape)
        return [SimpleNamespace(ring_uv=(5., 6.), marker=[[0,0],[4,0],[4,4],[0,4]], label='T03'),
                SimpleNamespace(ring_uv=(30., 40.), marker=[[0,0],[4,0],[4,4],[0,4]], label='T02')]
    monkeypatch.setattr(detection, 'detect', detect)
    monkeypatch.setattr('nest_clear.workers.vision.pixel_to_module', lambda u,v,pose,cal: (u,v,0.))
    worker = VisionWorker(None, None, 1, 10)
    cycle = SimpleNamespace(film_mode=True, film_calibration=object(), film_non_pickable={'T03'},
                            current_target=lambda: SimpleNamespace(film_label='T02', identity=(30.,40.,0.)))
    image = np.zeros((200, 300), dtype='uint8')
    result = worker.execute({'image':image, 'pose':(0,0,0), 'cycle':cycle, 'work_action':'fine'})
    assert seen == [(200,300)]
    assert [o.status for o in result['observations']] == ['rejected', 'eligible']
    assert result['fine_result'].target_xyz == (30.,40.,0.)


def test_film_review_and_neighbor_dedup_and_no_tool_change():
    from types import SimpleNamespace
    from nest_clear.flow.work_cycle import WorkCycle
    from nest_clear.vision.fine import FineResult
    def obs():
        return SimpleNamespace(target=SimpleNamespace(xyz=(1., 2., 0.)), identity=(1.,2.,0.), status='eligible', reason=None, area=1.)
    params = SimpleNamespace(skip_radius_mm=.5, max_fine_targets=10, cap_ratio=1, tool_check_min_attempts=1, tool_check_ratio=.1, max_coarse_shots_per_view=1)
    c = WorkCycle([(0,0,0),(10,0,0)], lambda target,pose:pose, lambda xyz:(xyz,xyz), params, film_mode=True)
    c.start(); c.on_coarse((0,0,0), [obs()]); c.begin_fine('fine')
    assert c.on_fine((0,0,0), FineResult('ok', target_xyz=(1.,2.,0.), identity=(1.,2.,0.))).func_code == 3
    c.mark_reply(1,1,{}); c.mark_done(1,1); c.on_picked()
    assert c.on_next().event['reason'] == 'review'
    review = obs(); reply = c.on_coarse((0,0,0), [review])
    assert review.status == 'failed' and reply.func_code == 1 and reply.event['tool_check_disabled']
    neighbor = obs(); reply = c.on_coarse((10,0,0), [neighbor])
    assert neighbor.status == 'failed' and reply.func_code == 9


def test_mode_can_change_only_when_idle():
    from nest_clear.film_mode import may_change_mode
    assert may_change_mode({'pid': 1, 'child_state': '未启动'}, {})
    assert may_change_mode({'pid': 1, 'child_pid': 2}, {'mode': '空闲'})
    assert not may_change_mode({'pid': 1, 'child_pid': 2}, {'mode': '挑毛中'})
    assert not may_change_mode({'pid': 1, 'child_state': '启动中'}, {})
    assert not may_change_mode({}, {})


def test_film_work_request_independent_z_and_missing_z_raises(tmp_path):
    from dataclasses import replace
    from types import SimpleNamespace
    import numpy as np
    import pytest
    from nest_clear.config import load_config
    from nest_clear.film_mode import write_mode
    from nest_clear.flow.context import FlowContext, RequestTracker
    from nest_clear.flow.picking import WorkRequest
    from nest_clear.guard import GuardError
    from nest_clear.vision.fine import FineResult
    from nest_clear.vision.geometry import Calibration
    from test_hair_integration import Board, Events
    config = load_config('config/nest_clear.yaml')
    config = replace(config, record_directory=str(tmp_path), travel=((-100.,100.),)*3,
                     tools=((1.,2.,3.), (4.,5.,6.)),
                     tool_calibration=replace(config.tool_calibration, directory=str(tmp_path)))
    write_mode(tmp_path, True)
    path = tmp_path / 'tool_offset.yaml'
    def make_cycle():
        ctx = FlowContext(None, config, [(0,0,0)], RequestTracker(0), Events(),
                          Calibration(np.eye(3), np.zeros(5), np.eye(3), np.zeros(3), 10))
        bb = Board(work_cycle=None, request_func=1, task_id=1)
        req = WorkRequest(ctx); req._blackboard = bb; req.perform()
        cycle = bb.read('work_cycle')
        obs = SimpleNamespace(target=SimpleNamespace(xyz=(2.,3.,1.)), identity=(2.,3.,1.), status='eligible', reason=None, area=1., film_label='T02')
        cycle.on_coarse((0,0,0), [obs])
        return cycle
    path.write_text(yaml.safe_dump({'tool1': {'samples':[{'tool_xyz_mm':[0,0,-4]},{'tool_xyz_mm':[0,0,-2]}]},
                                    'tool2': {'samples':[{'tool_xyz_mm':[0,0,-8]},{'tool_xyz_mm':[0,0,-6]}]}}))
    cycle = make_cycle()
    reply = cycle.on_fine((0,0,0), FineResult('ok', target_xyz=(2.,3.,1.)))
    assert reply.func_code == 3
    assert reply.tool1_xyz == (3.,5.,-2.) and reply.tool2_xyz == (6.,8.,-6.)
    path.write_text(yaml.safe_dump({'tool1': {'samples':[{'tool_xyz_mm':[0,0,-4]}]}}))
    cycle = make_cycle()
    with pytest.raises(GuardError, match='工具2 缺少工具偏移标定下针 Z'):
        cycle.on_fine((0,0,0), FineResult('ok', target_xyz=(2.,3.,1.)))
    assert cycle.attempt_count == 0


def test_synthetic_full_image_detects_film():
    import cv2
    import numpy as np
    from types import SimpleNamespace
    from nest_clear.workers.vision import VisionWorker
    from nest_clear.vision.geometry import Calibration
    image = np.full((600,800),255,np.uint8)
    cv2.circle(image,(300,300),37,0,6,cv2.LINE_AA)
    cv2.line(image,(200,300),(400,300),0,2,cv2.LINE_AA)
    cv2.line(image,(300,200),(300,400),0,2,cv2.LINE_AA)
    image[120:200,400:480] = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 12, 80)
    cal = Calibration(np.eye(3),np.zeros(5),np.eye(3),np.zeros(3),1)
    cycle = SimpleNamespace(film_mode=True, film_calibration=cal, film_non_pickable={'T12'})
    result = VisionWorker(None, cal, 1, 10).execute({'image':image,'pose':(0,0,0),'cycle':cycle,'work_action':'coarse'})
    assert len(result['observations']) == 1
    assert result['observations'][0].film_label == 'T12' and result['observations'][0].status == 'rejected'


def test_missing_z_uses_existing_999_response():
    from types import SimpleNamespace
    from nest_clear.flow.leaves import PrepareError
    from test_hair_integration import Board, Events
    events = Events()
    bb = Board(request_func=20, task_id=8, error='工具2 缺少工具偏移标定下针 Z，请先完成工具偏移标定',
               error_type='GuardError', work_cycle=None)
    node = PrepareError(events); node._blackboard = bb; node.perform()
    assert bb.read('reply_func') == 999 and bb.read('reply_coordinates') == {}
    assert '工具2 缺少工具偏移标定下针 Z' in events.rows[-1][1]['reason']


def test_tray_marker_blocks_switch_until_end(tmp_path):
    import os
    from nest_clear.film_mode import start_tray, tray_active, end_tray, write_mode
    write_mode(tmp_path, True)
    assert start_tray(tmp_path)
    assert tray_active(tmp_path, os.getpid())
    end_tray(tmp_path)
    assert not tray_active(tmp_path, os.getpid())


def test_all_synthetic_targets_are_detected_and_gated():
    import cv2
    import numpy as np
    from types import SimpleNamespace
    from nest_clear.workers.vision import VisionWorker
    from nest_clear.vision.geometry import Calibration
    image = np.full((600, 1200), 255, np.uint8)
    for x, marker_id in ((300, 2), (800, 3)):
        y = 300
        cv2.circle(image, (x,y), 37, 0, 6, cv2.LINE_AA)
        cv2.line(image, (x-100,y), (x+100,y), 0, 2, cv2.LINE_AA)
        cv2.line(image, (x,y-100), (x,y+100), 0, 2, cv2.LINE_AA)
        image[y-180:y-100,x+100:x+180] = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), marker_id, 80)
    cal = Calibration(np.eye(3),np.zeros(5),np.eye(3),np.zeros(3),1)
    cycle = SimpleNamespace(film_mode=True, film_calibration=cal, film_non_pickable={'T03'})
    result = VisionWorker(None, cal, 1, 10).execute({'image':image,'pose':(0,0,0),'cycle':cycle,'work_action':'coarse'})
    assert {o.film_label:o.status for o in result['observations']} == {'T02':'eligible', 'T03':'rejected'}


def test_committed_film_pick_records_two_z_sources():
    from types import SimpleNamespace
    from nest_clear.flow.work_cycle import WorkCycle
    from nest_clear.vision.fine import FineResult
    rows = []
    params = SimpleNamespace(skip_radius_mm=.5, max_fine_targets=10, cap_ratio=1, tool_check_min_attempts=1, tool_check_ratio=.1, max_coarse_shots_per_view=30)
    c = WorkCycle([(0,0,0)], lambda t,p:p, lambda xyz:((1,2,-3),(4,5,-6)), params,
                  record=lambda event, **fields: rows.append((event, fields)), film_mode=True)
    c.start()
    target = SimpleNamespace(target=SimpleNamespace(xyz=(1.,2.,0.)), identity=(1.,2.,0.), status='eligible', reason=None, area=1.)
    c.on_coarse((0,0,0), [target])
    c.begin_fine('capture')
    c.on_fine((0,0,0), FineResult('ok', target_xyz=(1.,2.,0.)))
    c.film_pending = {'tray':'1', 'view':1, 'target':'T02', 'pickable':True, 'target_xy':(1.,2.),
                      'pick1_xyz':(1,2,-3), 'pick2_xyz':(4,5,-6),
                      'z_sources':{'tool1':{'sample_count':2, 'max_z':-3}, 'tool2':{'sample_count':3, 'max_z':-6}}}
    assert not any(e == 'film_pick_sent' for e, _ in rows)
    c.mark_reply(1,2,{'pick1_xyz':(1,2,-3), 'pick2_xyz':(4,5,-6)})
    event = next(row for name, row in rows if name == 'film_pick_sent')
    assert event['z_sources']['tool1']['max_z'] == -3 and event['z_sources']['tool2']['max_z'] == -6


def test_synthetic_film_through_cycle_and_fake_plc():
    import cv2
    import numpy as np
    from types import SimpleNamespace
    from nest_clear.flow.work_cycle import WorkCycle
    from nest_clear.vision.geometry import Calibration
    from nest_clear.workers.vision import VisionWorker
    image = np.full((600,1200),255,np.uint8)
    for x, marker_id in ((300,2),(800,3)):
        y=300
        cv2.circle(image,(x,y),37,0,6,cv2.LINE_AA)
        cv2.line(image,(x-100,y),(x+100,y),0,2,cv2.LINE_AA)
        cv2.line(image,(x,y-100),(x,y+100),0,2,cv2.LINE_AA)
        image[y-180:y-100,x+100:x+180] = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),marker_id,80)
    cal = Calibration(np.eye(3),np.zeros(5),np.eye(3),np.zeros(3),1)
    params = SimpleNamespace(skip_radius_mm=.5,max_fine_targets=10,cap_ratio=1,tool_check_min_attempts=1,tool_check_ratio=.1,max_coarse_shots_per_view=30)
    cycle = WorkCycle([(0,0,0),(1,0,0)],lambda target,pose:pose,lambda target:((1,2,-3),(4,5,-6)),params,film_mode=True)
    cycle.film_calibration=cal; cycle.film_non_pickable={'T03'}
    worker=VisionWorker(None,cal,1,10)
    def capture(action):
        return worker.execute({'image':image,'pose':(0,0,0),'cycle':cycle,'work_action':action})
    plc=[]
    plc.append(cycle.start())
    plc.append(cycle.on_coarse((0,0,0),capture('coarse')['observations']))
    cycle.begin_fine('fine')
    plc.append(cycle.on_fine((0,0,0),capture('fine')['fine_result']))
    cycle.mark_reply(1,1,{'pick1_xyz':plc[-1].tool1_xyz,'pick2_xyz':plc[-1].tool2_xyz})
    cycle.mark_done(2,2); plc.append(cycle.on_picked()); plc.append(cycle.on_next())
    review=capture('coarse')['observations']; plc.append(cycle.on_coarse((0,0,0),review))
    neighbor=capture('coarse')['observations']; plc.append(cycle.on_coarse((0,0,0),neighbor))
    assert [r.func_code for r in plc] == [1,1,3,60,1,1,9]
    assert next(o for o in review if o.film_label == 'T02').status == 'failed'
    assert next(o for o in neighbor if o.film_label == 'T02').status == 'failed'
    assert next(o for o in neighbor if o.film_label == 'T03').status == 'rejected'


def test_incomplete_film_configuration_fails_closed():
    from types import SimpleNamespace
    import numpy as np
    import pytest
    from nest_clear.guard import GuardError
    from nest_clear.workers.vision import VisionWorker
    worker = VisionWorker(None, None, 1, 10)
    with pytest.raises(GuardError, match='菲林模式缺少平面标定或不可去目标配置'):
        worker.execute({'cycle': SimpleNamespace(film_mode=True), 'work_action': 'coarse',
                        'image': np.zeros((10, 10), dtype='uint8'), 'pose': (0, 0, 0)})
