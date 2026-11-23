"""异步相机标定采样与求解，避免阻塞行为树时钟。"""
from types import SimpleNamespace
import logging
import time

logger = logging.getLogger(__name__)

from autoweaver.worker import AsyncPoolConfig, AsyncRequestWorker

from nest_clear.record.images import record_image_error


class CamcalWorker(AsyncRequestWorker):
    name = 'camcal'
    # 求解线程若卡住，不得耗尽 PLC 所用的共享线程池。
    async_pool_config = AsyncPoolConfig(mode='dedicated', max_workers=1)
    max_retained_results = 2

    def execute(self, payload):
        ctx, session = payload['context'], payload['session']
        image, meta = payload['captured'] if isinstance(payload['captured'], tuple) else (payload['captured'], {})
        actual, task_id = payload['actual'], payload['task_id']
        session['shots'] += 1
        shot = f"s{session['shots']:02d}"
        probe = session.get('probes', [None])[-1]
        seq, capture_id = ctx.events.next_capture()
        archive = getattr(ctx.events, 'archive', None)
        raw = None
        if ctx.events.run_directory is not None:
            import cv2
            path = ctx.events.run_directory / 'camcal' / session['directory'] / f'camcal_{shot}_raw.png'
            path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(path), image):
                raise OSError(f'无法保存相机标定原图：{path}')
            raw = str(path.relative_to(ctx.events.directory))
        elif archive:
            try:
                raw = archive.capture_raw(image, sequence=seq, capture_id=capture_id, kind='camcal', tray='0', view=0, shot=shot, task_id=task_id)
            except Exception as exc:
                record_image_error(ctx.events, '相机标定原图', exc, task_id)
        ctx.events.emit('capture', kind='camcal', capture_id=capture_id, task_id=task_id,
                        tray=0, view=0, shot=shot, actual_xyz=actual, target_xyz=session['target'], raw_path=raw,
                        **{k: v for k, v in meta.items() if k != 'capture_seq'})
        if session['engine'] is None:
            if ctx.camcal_factory is None:
                from nest_clear.camcal.config import CameraCalibConfig
                from nest_clear.camcal.session import CameraCalibrationSession
                factory = CameraCalibrationSession
                cfg = CameraCalibConfig(**ctx.config.camera_calibration)
            else:
                factory = ctx.camcal_factory
                cfg = SimpleNamespace(**{'xy_limit_mm': 25., 'z_limit_mm': 5., 'position_tolerance_mm': .05,
                                         **ctx.config.camera_calibration})
            session['cfg'] = cfg
            session['engine'] = factory(cfg, ctx.calibration, (image.shape[1], image.shape[0]), session['start'])
        engine = session['engine']
        final = bool(getattr(engine, 'plan', [])) and session['shots'] == len(engine.plan) + 1
        started = time.monotonic()
        if final:
            logger.info("相机标定最后一张开始求解 Task_ID=%s sequence=%s elapsed_s=0", task_id, session['shots'])
            ctx.events.emit('camcal_solve_start', task_id=task_id, sequence=session['shots'], elapsed_s=0.0)
        try:
            step = engine.add(image, tuple(actual))
        finally:
            if final:
                elapsed = time.monotonic() - started
                logger.info("相机标定最后一张求解结束 Task_ID=%s sequence=%s elapsed_s=%.3f", task_id, session['shots'], elapsed)
                ctx.events.emit('camcal_solve_end', task_id=task_id, sequence=session['shots'], elapsed_s=elapsed)
        if session.get('timed_out'):
            return {'step': step, 'raw_path': raw}
        if step.overlay is not None:
            from nest_clear.record.live import write_live
            write_live(ctx.events.directory, "camcal_latest", step.overlay)
            if archive:
                archive._submit(step.overlay, [], 0, task_id, raw, capture_id=capture_id)
        # 叠加图已交给归档器，勿在请求结果缓存中保留整幅大图。
        step.overlay = None
        ctx.events.emit('camcal_step', task_id=task_id, sequence=session['shots'], target_xyz=step.target_xyz,
                        actual_xyz=actual, plc_actual_probe=probe, detail=step.detail, raw_path=raw, kind=step.kind, code=step.code, reason=step.reason, plan=getattr(session["engine"], "plan", []))
        if step.kind in ('move', 'done') and step.detail.get('glare_blobs'):
            ctx.events.emit('camcal_glare_warning', task_id=task_id, sequence=session['shots'],
                            image_index=session['shots'], bboxes=[b['bbox'] for b in step.detail['glare_blobs']],
                            excluded_corners=step.detail['glare_excluded_corners'])
        return {'step': step, 'raw_path': raw}
