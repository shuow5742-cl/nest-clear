"""轮询 PLC、发布暂停状态并管理主程序及其看门狗。

每拍检查子进程是否退出，读取设备状态并执行急停或初始化边沿动作，发布
暂停状态，按秒写心跳，最后检查活性。PLC 读失败时保留上次有效状态。
启动后等待首个活性点不超过启动宽限；打点后按文件更新时间检查超时。
看门狗杀掉卡死进程，意外退出或看门狗处置后检查 PLC 是否有未应答的
Task_ID，有则通过握手层回 999，通知 PLC 停机并要求人工处理。急停、
初始化重启及父程序退出属于有意停止，不代答。所有异常停止都等待新的
bit7 上升沿，因为设备初始化边沿是唯一启动入口，不能自行重启。
"""
import json
import logging
import os
import subprocess
import time
from typing import Any

from ..comm.handshake import Handshake
from ..config import Config
from ..paths import main_alive_file, supervisor_state_file
from .device_state import (
    ACTION_ESTOP,
    ACTION_RESTART,
    ACTION_START,
    BIT_FAULT,
    DeviceStateTracker,
    bit,
    conflicting_bits,
    describe,
)

logger = logging.getLogger("nest_clear.supervisor.runner")


class Supervisor:
    """父程序常驻循环；只有设备初始化边沿允许启动子进程。"""

    def __init__(self, config: Config, plc: Any) -> None:
        """绑定配置与具名 PLC 客户端，供入口和模拟测试使用。"""
        self.config = config
        self.plc = plc
        self.tracker = DeviceStateTracker()
        self.child: subprocess.Popen[Any] | None = None
        self.word: int | None = None
        self.paused = False
        self._heartbeat = 0
        self._next_heartbeat = 0.0
        self._failures = 0
        self._alive = main_alive_file(config.record_directory)
        self._state = supervisor_state_file(config.record_directory)
        self._started_at: float | None = None

    def _reply_after_failure(self, reason: str) -> None:
        """主程序非正常死亡时，为 PLC 未完成的请求提交错误应答。"""
        task_id: int | None = None
        try:
            task_id = self.plc.read("task_id")
            result_id = self.plc.read("result_id")
            if task_id == 0 or task_id == result_id:
                return
            Handshake(self.plc, self.plc.contract).write_reply(task_id, 999)
            logger.error(
                "错误码：SUPERVISOR-CHILD-FAILED；主程序%s；已给 Task_ID=%s 回 999；"
                "位置：父程序握手；排查建议：PLC 会报错停机，人工处理后重新初始化。",
                reason, task_id,
            )
        except Exception as exc:
            logger.error(
                "错误码：SUPERVISOR-REPLY；主程序%s；检查待应答任务或给 "
                "Task_ID=%s 回 999 失败：%s；位置：父程序握手；"
                "排查建议：查网线/IP/PLC 是否上电，人工处理后重新初始化。",
                reason, task_id, exc,
            )

    def _running(self) -> bool:
        """检查子进程并记录意外退出，等待下一个初始化边沿。"""
        if self.child is None:
            return False
        code = self.child.poll()
        if code is None:
            return True
        reason = f"意外退出，退出码 {code}"
        logger.error("主程序%s；等待下一次初始化信号", reason)
        self.child = None
        self._started_at = None
        self._reply_after_failure(reason)
        return False

    def _stop(self, reason: str) -> bool:
        """先请求子进程退出，超时后强制结束并记录原因。"""
        if not self._running():
            return False
        assert self.child is not None
        logger.warning("停止主程序：%s", reason)
        try:
            self.child.terminate()
            self.child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.child.kill()
            self.child.wait()
        except OSError as exc:
            logger.error("错误码：SUPERVISOR-STOP；停止主程序失败：%s；位置：子进程；排查建议：检查进程权限", exc)
            return False
        self.child = None
        self._started_at = None
        return True

    def _start(self) -> None:
        """清除旧活性点并启动配置的命令，失败后等待下一条边沿。"""
        try:
            self._alive.unlink(missing_ok=True)
            self.child = subprocess.Popen(self.config.supervisor.child_command)
            self._started_at = time.monotonic()
            logger.info("主程序已启动，PID %s，命令 %s", self.child.pid, self.config.supervisor.child_command)
        except OSError as exc:
            logger.error("错误码：SUPERVISOR-START；启动主程序失败：%s；位置：子进程；排查建议：检查命令、运行目录和权限", exc)

    def _publish(self) -> bool:
        """原子发布状态，避免主程序读取到半个 JSON。"""
        try:
            self._state.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._state.with_suffix(".json.tmp")
            state = {"word": self.word, "bits": describe(self.word or 0), "paused": self.paused}
            temporary.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
            os.replace(temporary, self._state)
            return True
        except OSError as exc:
            logger.error("错误码：SUPERVISOR-STATE；写状态文件失败：%s；位置：%s；排查建议：检查目录权限和磁盘空间", exc, self._state)
            return False

    def tick(self) -> None:
        """执行一次读状态、心跳和看门狗；通信失败保留上次状态。"""
        running = self._running()
        try:
            word = int(self.plc.read("status"))
        except Exception as exc:
            self._failures += 1
            if self._failures % 20 == 1:
                logger.warning(
                    "错误码：SUPERVISOR-PLC-READ；PLC 连不上或状态读取失败"
                    "（连续 %s 次）：%s；位置：设备状态；排查建议：查网线/IP/PLC 是否上电；"
                    "保留上次状态并继续重试", self._failures, exc,
                )
        else:
            if self._failures:
                logger.info("PLC 状态读取恢复，之前连续失败 %s 次", self._failures)
                self._failures = 0
            if word != self.word:
                level = logging.ERROR if bit(word, BIT_FAULT) else logging.INFO
                previous = "未知" if self.word is None else f"0x{self.word:04X}"
                logger.log(
                    level, "设备状态变化：%s → %s（%s → 0x%04X）",
                    describe(self.word or 0), describe(word), previous, word,
                )
                if conflicting_bits(word):
                    logger.warning("循环启动与暂停或停止同时为 1，按暂停处理")
                self.word = word
            decision = self.tracker.decide(word, child_running=running)
            self.paused = decision.paused
            published = self._publish()
            stopped = True
            if decision.action in (ACTION_ESTOP, ACTION_RESTART):
                stopped = self._stop(decision.reason)
            if decision.action in (ACTION_START, ACTION_RESTART):
                if published and (stopped or not self._running()):
                    self._start()
                elif not published:
                    logger.error("错误码：SUPERVISOR-STATE；无法发布暂停状态，主程序未启动；位置：初始化；排查建议：修复状态文件后重新初始化")
        now = time.monotonic()
        if now >= self._next_heartbeat:
            self._next_heartbeat = now + 1.0
            try:
                self._heartbeat = (self._heartbeat + 1) % 65536
                self.plc.write("heartbeat", self._heartbeat)
            except Exception as exc:
                logger.warning("错误码：SUPERVISOR-HEARTBEAT；PLC 心跳写入失败：%s；位置：心跳；排查建议：查网线/IP/PLC 是否上电", exc)
        if self._running():
            try:
                age = time.time() - self._alive.stat().st_mtime
            except FileNotFoundError:
                started_at = self._started_at if self._started_at is not None else now
                elapsed = time.monotonic() - started_at
                if elapsed > self.config.supervisor.startup_grace_s:
                    reason = (
                        f"启动后 {elapsed:.1f} 秒仍未打出活性点，可能卡在预检"
                        "（相机/PLC）——查 main.log"
                    )
                    logger.error("主程序%s", reason)
                    if self._stop(reason):
                        self._reply_after_failure(f"被看门狗停止：{reason}")
            except OSError as exc:
                logger.error(
                    "错误码：SUPERVISOR-ALIVE；读取活性文件失败：%s；位置：%s；"
                    "排查建议：检查目录权限和磁盘。", exc, self._alive,
                )
            else:
                if age > self.config.supervisor.watchdog_s:
                    reason = (
                        f"活性打点超时 {age:.1f} 秒，超过 "
                        f"{self.config.supervisor.watchdog_s:.1f} 秒；查 main.log"
                    )
                    logger.error("主程序%s", reason)
                    if self._stop(reason):
                        self._reply_after_failure(f"被看门狗停止：{reason}")

    def run(self) -> None:
        """持续运行直至中断，并在退出时释放子进程与连接。"""
        try:
            while True:
                self.tick()
                time.sleep(self.config.supervisor.poll_s)
        except KeyboardInterrupt:
            logger.info("收到退出信号")
        finally:
            self._stop("父程序退出")
            self.plc.close()
