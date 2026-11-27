"""160/161 下针深度标定第一阶段的 PLC 请求入口。"""

from autoweaver.motion_policy.nodes.node import Status, TreeNode

from nest_clear.guard import GuardError, diagnostic
from .calibration import change_mode
from .modes import Mode

RESERVED_CODES = frozenset((170, 171, 180, 181, 190, 191))
DEPTH_CODES = RESERVED_CODES | {160}


class DepthCalibration(TreeNode):
    def __init__(self, ctx):
        super().__init__('下针深度标定')
        self.ctx = ctx

    def reply(self, code, coords=None):
        bb = self._blackboard
        bb.write('reply_func', code, self.name)
        bb.write('reply_coordinates', coords or {}, self.name)
        bb.write('reply_tool', None, self.name)
        return Status.SUCCESS

    def fail(self, exc):
        bb = self._blackboard
        details = diagnostic(exc, code='DEPTH-FAILED', where='下针深度标定',
                             hint='检查菲林、相机和 PLC 实时坐标。')
        bb.write('reply_diagnostic', details, self.name)
        self.ctx.events.emit('depthcal_state', phase='failed', **details)
        if bb.read('mode') == Mode.DEPTH_CALIB:
            change_mode(bb, self.ctx, Mode.IDLE, self.name)
        bb.write('depth_session', None, self.name)
        return self.reply(999)

    def on_start(self):
        bb = self._blackboard
        bb.write('depth_request', True, self.name)
        func, mode = bb.read('request_func'), bb.read('mode')
        try:
            if func in RESERVED_CODES:
                raise GuardError('DEPTH-UNIMPLEMENTED', '下针深度标定 170/180/190 尚未实现',
                                 'PLC 请求', '本阶段使用 160 启动料盘对焦测试。')
            if func != 160 or mode not in (Mode.IDLE, Mode.DEPTH_CALIB):
                raise GuardError('DEPTH-MODE', f'当前模式 {mode.value} 不允许功能码 {func}',
                                 'PLC 请求', '空闲时从 160 发起料盘对焦测试。')
            if self.ctx.config.depth_calibration['camera_z_min_mm'] is None:
                raise GuardError('DEPTH-Z-MIN', '未配置相机 Z 下限 depth_calibration.camera_z_min_mm（用户现场给出）',
                                 '160 会话启动', '现场确认两把工具与料盘间隙后填写 Z 下限。')
            return self.start_request()
        except Exception as exc:
            return self.fail(exc)

    def start_request(self):
        raise ValueError('对焦扫描尚未安装')

    def on_running(self):
        return self.fail('对焦扫描尚未安装')
