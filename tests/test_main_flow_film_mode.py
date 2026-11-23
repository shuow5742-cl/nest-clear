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


def test_film_worker_applies_production_view_and_fine_filters(monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace
    import numpy as np
    from nest_clear.config import load_config
    from nest_clear.vision.geometry import Calibration
    from nest_clear.workers.vision import VisionWorker

    films = [SimpleNamespace(ring_uv=(x, 0.), marker=[[x, 0], [x+4, 0], [x+4, 4], [x, 4]], label='T02')
             for x in (2., 8., 20.)]
    monkeypatch.setattr('nest_clear.tooloffset.detection.detect', lambda image: films)
    monkeypatch.setattr('nest_clear.workers.vision.pixel_to_module', lambda u, v, pose, cal: (u, v, 0.))
    config = replace(load_config('config/nest_clear.yaml').vision, responsibility_zone=True,
                     ignore_regions_px=((19., -1., 21., 1.),), fine_box_half_mm=4.)
    cal = Calibration(np.eye(3), np.zeros(5), np.eye(3), np.zeros(3), 1)
    cycle = SimpleNamespace(film_mode=True, film_calibration=cal, film_non_pickable=set(),
                            current_target=lambda: SimpleNamespace(film_label='T02', identity=(2., 0., 0.)),
                            is_attempted=lambda xyz: xyz[0] == 2.)
    worker = VisionWorker(None, cal, 1, 10, config)
    payload = {'image': np.zeros((40, 40), np.uint8), 'pose': (0, 0, 0), 'cycle': cycle,
               'scan': (0, 0, 0), 'scan_path': [(0, 0, 0), (10, 0, 0)]}
    result = worker.execute({**payload, 'work_action': 'coarse'})
    assert [obs.status for obs in result['observations']] == ['eligible', 'other_view']
    result = worker.execute({**payload, 'work_action': 'fine'})
    assert result['fine_result'].status == 'fine_empty'


def test_film_review_marks_failed_and_moves_to_next_view():
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
    assert c.on_next().event['reason'] == 'cap'
    review = obs(); reply = c.on_coarse((0,0,0), [review])
    assert review.status == 'failed' and reply.func_code == 1
    assert not reply.event['tool_change'] and reply.event['tool_check_disabled']


def test_film_failed_target_stays_red_in_neighbor_view():
    from types import SimpleNamespace
    from nest_clear.flow.work_cycle import WorkCycle
    from nest_clear.vision.fine import FineResult
    params = SimpleNamespace(skip_radius_mm=.5, max_fine_targets=10, cap_ratio=1,
                             tool_check_min_attempts=99, tool_check_ratio=.1,
                             max_coarse_shots_per_view=30)
    cycle = WorkCycle([(0, 0, 0), (10, 0, 0)], lambda target, pose: pose,
                      lambda target: (target, target), params, film_mode=True)
    def obs():
        return SimpleNamespace(target=SimpleNamespace(xyz=(1., 2., 0.)), identity=(1., 2., 0.),
                               status='eligible', reason=None, area=1., film_label='T02')
    cycle.start(); cycle.on_coarse((0, 0, 0), [obs()])
    cycle.on_fine((0, 0, 0), FineResult('ok', target_xyz=(1., 2., 0.)))
    cycle.mark_reply(1, 1, {}); cycle.mark_done(1, 1); cycle.on_picked(); cycle.on_next()
    review = obs(); assert cycle.on_coarse((0, 0, 0), [review]).func_code == 1
    neighbor = obs(); assert cycle.on_coarse((10, 0, 0), [neighbor]).func_code == 9
    assert review.status == neighbor.status == 'failed'
    assert len(cycle.attempts_ledger) == 1


def test_film_and_production_share_queue_and_cap():
    from types import SimpleNamespace
    from nest_clear.flow.work_cycle import WorkCycle
    from nest_clear.vision.fine import FineResult

    def run(film_mode, cap_ratio, coarse_limit, fine_status):
        params = SimpleNamespace(skip_radius_mm=.5, max_fine_targets=10, cap_ratio=cap_ratio,
                                 tool_check_min_attempts=1, tool_check_ratio=.1,
                                 max_coarse_shots_per_view=coarse_limit)
        cycle = WorkCycle([(0, 0, 0)], lambda target, pose: pose, lambda target: (target, target),
                          params, film_mode=film_mode)
        def observations():
            return [SimpleNamespace(target=SimpleNamespace(xyz=(x, 0, 0)), identity=(x, 0, 0),
                                    status='eligible', reason=None, area=1.) for x in (1., 3.)]
        cycle.start()
        coarse = cycle.on_coarse((0, 0, 0), observations())
        fine = cycle.on_fine((0, 0, 0), FineResult(fine_status, target_xyz=(1., 0, 0)
                                                   if fine_status == 'ok' else None))
        if fine.func_code == 3:
            cycle.mark_reply(1, 1, {}); cycle.mark_done(1, 1); cycle.on_picked()
            next_reply = cycle.on_next()
            if next_reply.event['reason'] == 'next_target':
                fine = cycle.on_fine((0, 0, 0), FineResult('ok', target_xyz=(3., 0, 0)))
                cycle.mark_reply(2, 2, {}); cycle.mark_done(2, 2); cycle.on_picked()
                next_after_second = cycle.on_next()
            else:
                next_after_second = None
        else:
            next_reply = next_after_second = None
        review = observations()
        final = cycle.on_coarse((0, 0, 0), review)
        return (coarse.func_code, fine.func_code,
                next_reply.event['reason'] if next_reply else None,
                next_after_second.event['reason'] if next_after_second else None,
                final.func_code, final.event['reason'], final.event['tool_change'],
                [item.status for item in review])

    for args in ((.5, 30, 'ok'), (2., 1, 'ok'), (.5, 30, 'fine_empty')):
        film, production = run(True, *args), run(False, *args)
        assert film[:4] == production[:4] and film[5] == production[5] and film[7] == production[7]
        assert (film[4], film[6]) == (9, False)
        assert (production[4], production[6]) == (10, True)
    assert run(True, .5, 30, 'ok')[2:7] == ('cap', None, 9, 'cap', False)
    assert run(True, 2., 1, 'ok')[2:7] == ('next_target', 'list_exhausted', 9, 'coarse_limit', False)
    assert run(True, .5, 30, 'fine_empty')[4:7] == (9, 'cap', False)


def test_film_fine_out_of_range_overlay_is_blue():
    from types import SimpleNamespace
    import numpy as np
    from nest_clear.config import load_config
    from nest_clear.flow.context import FlowContext, RequestTracker
    from nest_clear.flow.picking import FinishCapture
    from nest_clear.flow.work_cycle import WorkCycle
    from nest_clear.vision.fine import FineResult
    from nest_clear.vision.geometry import Calibration
    from test_hair_integration import Board, Events

    config = load_config('config/nest_clear.yaml')
    cycle = WorkCycle([(100., 50., -3.)], lambda target, pose: pose,
                      lambda target: (target, target), config.vision,
                      film_mode=True, reachable=lambda target: target[0] < 5)
    candidate = SimpleNamespace(bbox=(20, 20, 40, 40), pick_uv=(30, 30), identity_uv=(30, 30),
                                gray_ratio=None, dark_frac=None)
    def observation(x):
        return SimpleNamespace(target=SimpleNamespace(xyz=(x, 0., 0.)), identity=(x, 0., 0.),
                               candidate=candidate, status='eligible', reason=None,
                               area=1., length=1., film_label='T02')
    cycle.start(); cycle.on_coarse((100., 50., -3.), [observation(1.)])
    fine_obs = observation(10.)
    detection = {'observations': [fine_obs],
                 'fine_result': FineResult('ok', candidate=candidate, target_xyz=(10., 0., 0.))}
    class Archive:
        def rendered(self, image, *args, **kwargs): self.image = image
    events = Events(); events.archive = Archive(); events.run_directory = None
    calibration = Calibration(np.eye(3), np.zeros(5), np.eye(3), np.zeros(3), 1)
    ctx = FlowContext(None, config, [(100., 50., -3.)], RequestTracker(0), events, calibration)
    board = Board(work_cycle=cycle, work_action='fine', detection=detection, actual=(100., 50., -3.),
                  image=np.zeros((80, 80, 3), np.uint8), task_id=1, next_index=1)
    finish = FinishCapture(ctx); finish._blackboard = board; finish.perform()
    assert detection['fine_result'].status == 'fine_out_of_range'
    assert fine_obs.status == 'out_of_range' and tuple(events.archive.image[20, 20]) == (255, 0, 0)
    assert board.read('reply_func') == 1


def test_mode_can_change_only_when_idle():
    from nest_clear.film_mode import may_change_mode
    assert may_change_mode({'pid': 1, 'child_state': '未启动'}, {})
    assert may_change_mode({'pid': 1, 'child_pid': 2}, {'mode': '空闲'}, False)
    assert not may_change_mode({'pid': 1, 'child_pid': 2}, {'mode': '空闲'}, True)
    assert not may_change_mode({'pid': 1, 'child_pid': 2}, {'mode': '挑毛中'})
    assert may_change_mode({'pid': 1, 'child_pid': 2}, {'mode': '挑毛中'}, False)
    assert not may_change_mode({'pid': 1, 'child_pid': 2}, {'mode': '相机标定中'}, False)
    assert not may_change_mode({'pid': 1, 'child_state': '启动中'}, {})
    assert not may_change_mode({}, {})


def test_terminal_replies_release_switch(tmp_path):
    import os
    import time
    from nest_clear.film_mode import start_tray, tray_active, may_change_mode
    from nest_clear.flow.leaves import CommitReply
    from nest_clear.flow.modes import Mode
    from nest_clear.record.events import EventRecorder
    from test_hair_integration import Board
    events = EventRecorder(tmp_path, run=True)
    supervisor = {'pid': 1, 'child_pid': os.getpid()}
    for code in (9, 10):
        start_tray(tmp_path)
        events.status.observe({'event': 'mode_change', 'mode': '挑毛中'})
        assert not may_change_mode(supervisor, events.status.data, tray_active(tmp_path, os.getpid()))
        board = Board(committed_func=code, reply_func=code, mode=Mode.PICKING,
                      started=time.monotonic(), task_id=1, request_func=20, next_index=1)
        leaf = CommitReply(events); leaf._blackboard = board; leaf.perform()
        assert not tray_active(tmp_path, os.getpid())
        assert events.status.data['mode'] == '挑毛中'
        assert may_change_mode(supervisor, events.status.data, tray_active(tmp_path, os.getpid()))
        assert board.read('mode') == Mode.PICKING
        start_tray(tmp_path)
        assert not may_change_mode(supervisor, events.status.data, tray_active(tmp_path, os.getpid()))


def test_main_start_clears_stale_tray(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import pytest
    from nest_clear import main
    state = tmp_path / 'state'; state.mkdir()
    marker = state / 'film_tray.pid'; marker.write_text('-1')
    monkeypatch.setattr(main, 'load_config', lambda path: SimpleNamespace(record_directory=tmp_path))
    def construct(*args, **kwargs):
        assert not marker.exists()
        raise RuntimeError('stop after startup check')
    monkeypatch.setattr(main, 'Application', construct)
    with pytest.raises(RuntimeError, match='startup check'):
        main.run()


def test_main_exception_cleans_tray_even_if_close_fails(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import pytest
    from nest_clear import main
    from nest_clear.film_mode import start_tray, tray_active
    import os
    monkeypatch.setattr(main, 'load_config', lambda path: SimpleNamespace(record_directory=tmp_path))
    class Application:
        def __init__(self, *args, **kwargs): start_tray(tmp_path)
        def step(self, gate): raise RuntimeError('interrupted')
        def close(self): raise RuntimeError('close failed')
    monkeypatch.setattr(main, 'Application', Application)
    with pytest.raises(RuntimeError, match='close failed'):
        main.run()
    assert not tray_active(tmp_path, os.getpid())


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
    assert [r.func_code for r in plc] == [1,1,3,60,1,1]
    assert next(o for o in review if o.film_label == 'T02').status == 'failed'
    assert next(o for o in review if o.film_label == 'T03').status == 'rejected'


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


def test_site_geometry_full_tray_filters_unreachable(tmp_path, monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace
    import numpy as np
    from nest_clear.config import load_config
    from nest_clear.film_mode import write_mode
    from nest_clear.flow.context import FlowContext, RequestTracker
    from nest_clear.flow.picking import WorkRequest
    from nest_clear.vision.fine import FineResult
    from nest_clear.vision.geometry import Calibration
    from test_hair_integration import Board, Events

    travel = ((80., 180.), (-10., 150.), (-15., 0.))
    config = load_config('config/nest_clear.yaml')
    config = replace(config, record_directory=str(tmp_path), travel=travel,
                     tools=((29.330, 24.322, 0.), (30.092, -25.283, 0.)),
                     tool_calibration=replace(config.tool_calibration, directory=str(tmp_path)))
    (tmp_path / 'tool_offset.yaml').write_text(yaml.safe_dump({
        'tool1': {'samples': [{'tool_xyz_mm': [0, 0, -15]}]},
        'tool2': {'samples': [{'tool_xyz_mm': [0, 0, 0]}]}}))
    write_mode(tmp_path, True)
    monkeypatch.setattr('nest_clear.flow.picking.fine_camera_xy',
                        lambda target, pose, cal: (target[0] + .86, target[1] + .92, pose[2]))
    events = Events()
    ctx = FlowContext(None, config, [(100., 50., -3.)], RequestTracker(0), events,
                      Calibration(np.eye(3), np.zeros(5), np.eye(3), np.zeros(3), 10))
    bb = Board(work_cycle=None, request_func=1, task_id=1)
    req = WorkRequest(ctx); req._blackboard = bb; req.perform()
    cycle = bb.read('work_cycle')
    assert bb.read('scan') == (100., 50., -3.)
    points = [(49.1 + 30 * (i // 8), -14.2 + 30 * (i % 8), 0.) for i in range(32)]
    non_pickable = {i for i in range(32) if f'T{i:02}' in config.test_mode['non_pickable']}
    def observations():
        return [SimpleNamespace(target=SimpleNamespace(xyz=p), identity=p,
                                status='rejected' if i in non_pickable else 'eligible',
                                reason=None, area=1., film_label=f'T{i:02}')
                for i, p in enumerate(points)]
    def inside(xyz):
        return all(lo <= v <= hi for v, (lo, hi) in zip(xyz, travel))
    reachable = {p for p in points if all(
        inside(q) for q in ((p[0]+.86, p[1]+.92, -3.),
                            (p[0]+29.330, p[1]+24.322, -15.),
                            (p[0]+30.092, p[1]-25.283, 0.)))}
    expected = {p for i, p in enumerate(points) if i not in non_pickable and p in reachable}
    sent = []
    reviewed = False
    while True:
        obs = observations()
        reply = cycle.on_coarse((100., 50., -3.), obs)
        assert cycle.can_reach((109.1, 15.6, 0.))
        assert not cycle.can_reach((109.1, 15.4, 0.))
        for i, o in enumerate(obs):
            if points[i] not in reachable:
                assert o.status == 'out_of_range'
        if reply.scan_xyz: assert inside(reply.scan_xyz)
        if sent:
            reviewed = True
            assert all(o.status == 'failed' for o in obs if o.target.xyz in sent)
        assert reply.func_code != 10
        if reply.func_code == 9: break
        while cycle.state.value == 'WAIT_FINE':
            point = cycle.current_target().target.xyz
            assert point in expected and point not in sent
            reply = cycle.on_fine(reply.scan_xyz, FineResult('ok', target_xyz=point, identity=point))
            assert reply.func_code == 3
            assert inside(reply.tool1_xyz) and inside(reply.tool2_xyz)
            sent.append(point)
            cycle.mark_reply(1, 1, {}); cycle.mark_done(1, 1)
            cycle.on_picked(); reply = cycle.on_next()
            assert inside(reply.scan_xyz)
    assert reply.func_code == 9 and reviewed and set(sent) == expected and len(sent) == len(set(sent))


def test_production_work_request_filters_tool_and_camera_limits(tmp_path, monkeypatch):
    import pytest
    from dataclasses import replace
    from types import SimpleNamespace
    import numpy as np
    from nest_clear.config import load_config
    from nest_clear.flow.context import FlowContext, RequestTracker
    from nest_clear.flow.picking import WorkRequest
    from nest_clear.vision.geometry import Calibration
    from test_hair_integration import Board, Events
    cfg = load_config('config/nest_clear.yaml')
    cfg = replace(cfg, record_directory=str(tmp_path), travel=((80., 180.), (-10., 150.), (-15., 0.)),
                  tools=((29.330, 24.322, 0.), (30.092, -25.283, 0.)), pick_depth_z=-5.,
                  tool_calibration=replace(cfg.tool_calibration, directory=str(tmp_path)))
    (tmp_path / 'pick_compensation.yaml').write_text(yaml.safe_dump(
        {'tool2': {'type': 'constant', 'constant_xy': [0., -1.]}}))
    monkeypatch.setattr('nest_clear.flow.picking.fine_camera_xy',
                        lambda target, pose, cal: (target[0]+.86, target[1]+.92, pose[2]))
    ctx = FlowContext(None, cfg, [(100., 50., -3.)], RequestTracker(0), Events(),
                      Calibration(np.eye(3), np.zeros(5), np.eye(3), np.zeros(3), 10))
    bb = Board(work_cycle=None, request_func=1, task_id=1)
    req = WorkRequest(ctx); req._blackboard = bb; req.perform()
    cycle = bb.read('work_cycle')
    def obs(p):
        return SimpleNamespace(target=SimpleNamespace(xyz=p), identity=p, status='eligible', reason=None, area=1.)
    camera_bad, tool_bad, good = obs((49.1, 45.8, 0.)), obs((109.1, 15.8, 0.)), obs((109.1, 45.8, 0.))
    reply = cycle.on_coarse((100., 50., -3.), [camera_bad, tool_bad, good])
    assert [o.status for o in (camera_bad, tool_bad, good)] == ['out_of_range', 'out_of_range', 'eligible']
    assert reply.scan_xyz == pytest.approx((109.96, 46.72, -3.))
