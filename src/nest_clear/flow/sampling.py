"""稳定实时坐标采样和非阻塞设备调用，共用于相机标定流程。"""

import math
import time

import numpy as np
from autoweaver.motion_policy.nodes.node import Status, TreeNode

from .leaves import CallWorker


class DeviceSampling(TreeNode):
    def start_call(self, target, payload, result):
        """发起非阻塞 Worker 调用。"""
        self.call = CallWorker(self.ctx.board, target, payload, result, name=f'{self.name}设备调用')
        self.call.set_blackboard(self._blackboard)

    def tick_call(self):
        """推进一次 Worker 调用。"""
        status = self.call.tick(self.snapshot, self.tick_ctx)
        if status != Status.RUNNING:
            self.call = None
        return status

    def settle(self):
        """非阻塞等待 PLC 实时坐标在窗口内稳定，取均值。"""
        cfg = self.settle_config
        if self.call:
            status = self.tick_call()
            if status == Status.FAILURE:
                raise ValueError('读取实时坐标失败：' + str(self._blackboard.read('error')))
            if status == Status.RUNNING:
                return None
            value = self._blackboard.read('stable_actual')
            if len(value) != 3 or not all(math.isfinite(x) for x in value):
                raise ValueError('实时坐标无效')
            self.samples.append(value)
            self.next_read = time.monotonic() + cfg['settle_window_s'] / (cfg['settle_samples'] - 1)
        if len(self.samples) >= cfg['settle_samples']:
            a = np.asarray(self.samples)
            if np.all(np.ptp(a, axis=0) <= cfg['jitter_tol_mm']):
                return tuple(float(x) for x in a.mean(axis=0))
            self.round += 1
            if self.round >= 3:
                return False
            self.samples = []
        if self.call is None and time.monotonic() >= self.next_read:
            self.start_call('plc', {'operation': 'actual'}, 'stable_actual')
        return None
