"""Single-slot preview frames and bounded, explicitly requested capture batches."""
from dataclasses import dataclass
import threading
import time


@dataclass(frozen=True)
class CapturedFrame:
    frame: object
    seq: int
    captured_mono: float
    captured_at: str
    exposure_ms: float | None
    generation: int


class FrameBroker:
    def __init__(self):
        self.condition = threading.Condition()
        self.latest = None
        self.seq = 0
        self.generation = 0
        self.closed = False
        self.error = None
        self.waiters = []

    def publish(self, frame, captured_mono, captured_at, exposure_ms, preview=True):
        with self.condition:
            self.seq += 1
            item = CapturedFrame(frame.copy(), self.seq, captured_mono, captured_at, exposure_ms, self.generation)
            if preview:
                self.latest = item
            for waiter in self.waiters:
                if captured_mono > waiter['after'] and len(waiter['frames']) < waiter['count']:
                    waiter['frames'].append(item)
            self.condition.notify_all()
            return item

    def invalidate(self):
        with self.condition:
            self.generation += 1
            self.latest = None
            self.condition.notify_all()

    def finish(self, error=None):
        with self.condition:
            self.closed = True
            if error is not None:
                self.error = error
            self.condition.notify_all()

    def next_latest(self, after_seq, stop):
        with self.condition:
            while not stop.is_set() and not self.closed:
                if self.latest is not None and self.latest.seq > after_seq:
                    return self.latest
                self.condition.wait(.1)
            if self.error is not None:
                raise self.error
            return None

    def capture_after(self, t_mono, count, timeout):
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError('count 必须为正整数')
        if timeout <= 0:
            raise ValueError('timeout 必须大于零')
        deadline = time.monotonic() + timeout
        waiter = {'after': t_mono, 'count': count, 'frames': []}
        with self.condition:
            if self.latest is not None and self.latest.captured_mono > t_mono:
                waiter['frames'].append(self.latest)
            self.waiters.append(waiter)
            try:
                while len(waiter['frames']) < count:
                    if self.closed:
                        raise RuntimeError(f'预览采集已停止: {self.error or "相机已关闭"}')
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(f'等待 {count} 帧超时，已采集 {len(waiter["frames"])} 帧')
                    self.condition.wait(remaining)
                return [CapturedFrame(item.frame.copy(), item.seq, item.captured_mono, item.captured_at, item.exposure_ms, item.generation) for item in waiter['frames']]
            finally:
                self.waiters.remove(waiter)
