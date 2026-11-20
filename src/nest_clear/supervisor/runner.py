"""轮询 PLC、发布暂停状态并管理主程序及其看门狗。

每拍检查子进程是否退出，读取设备状态并执行急停或初始化边沿动作，发布
暂停状态，按秒写心跳，最后检查活性。PLC 读失败时保留上次有效状态。
启动后等待首个活性点不超过启动宽限；打点后按文件更新时间检查超时。
看门狗杀掉卡死进程，意外退出或看门狗处置后检查 PLC 是否有未应答的
Task_ID，有则通过握手层回 999，通知 PLC 停机并要求人工处理。急停、
初始化重启由 PLC 放弃在途请求，不代答并记录任务；父程序退出时仍为未完成
请求代答。异常代答失败后重试，遇初始化、急停或任务变化即放弃。所有异常
停止都等待新的 bit7 上升沿，因为设备初始化边沿是唯一启动入口。
"""

import json
import logging
import os
import subprocess
from pathlib import Path
from datetime import datetime
import time
from typing import Any

from ..comm.handshake import Handshake
from ..config import Config
from ..guard import diagnostic
from ..paths import main_alive_file, supervisor_state_file
from ..record.events import EventRecorder
from .device_state import (
    ACTION_ESTOP,
    ACTION_RESTART,
    ACTION_START,
    BIT_FAULT,
    BIT_ESTOP,
    BIT_INITIALIZING,
    DeviceStateTracker,
    bit,
    conflicting_bits,
    describe,
)

logger = logging.getLogger("nest_clear.supervisor.runner")


class Supervisor:
    """父程序常驻循环；只有设备初始化边沿允许启动子进程。"""

    def __init__(self, config: Config, plc: Any, *, metadata: dict[str, Any] | None = None) -> None:
        """绑定配置与具名 PLC 客户端，供入口和模拟测试使用。"""
        self.config = config
        self.metadata = metadata or {}
        self.events = EventRecorder(config.record_directory, "supervisor")
        self.plc = plc
        self.tracker = DeviceStateTracker()
        self.child: subprocess.Popen[Any] | None = None
        self.word: int | None = None
        self.paused = False
        self._heartbeat = 0
        self._next_heartbeat = 0.0
        self._heartbeat_failures = 0
        self._heartbeat_failure_since = None
        self._heartbeat_failure_wall = None
        self._failures = 0
        self._failure_since = None
        self._pending_reply = None
        self._alive = main_alive_file(config.record_directory)
        self._state = supervisor_state_file(config.record_directory)
        self._started_at: float | None = None
        self._wall_start = datetime.now().astimezone().isoformat()
        self._child_start_wall = None
        self._last_exit = None
        self._plc_last_read = None
        self._heartbeat_last_write = None
        self._ready = None
        self.ui_process = None
        self._ui_failures = []
        self._ui_next = 0.0
        self._ui_disabled = False

    def _abandon_reply(self, reason: str, task_id: int | None = None, func: int | None = None) -> None:
        """记录 PLC 主动放弃的请求；读不到上下文时保留可用信息。"""
        if self._pending_reply is not None:
            if task_id is None:
                task_id = self._pending_reply["task_id"]
            if func is None and task_id == self._pending_reply["task_id"]:
                func = self._pending_reply.get("func")
        try:
            if task_id is None:
                task_id = self.plc.read("task_id")
            if func is None:
                func = self.plc.read("request_func")
        except Exception as exc:
            logger.warning("读取被放弃请求的上下文失败：%s", exc)
        self.events.emit("request_abandoned", task_id=task_id, request_func=func, reason=reason)
        logger.info("放弃代答：Task_ID=%s，功能码=%s，原因=%s", task_id, func, reason)
        self._pending_reply = None

    def _reply_after_failure(self, reason: str, *, retry: bool = True) -> None:
        """仅在 PLC 仍等待同一请求时代答；异常退出可在后续循环重试。"""
        pending = self._pending_reply if retry else None
        if pending is None:
            pending = {"reason": reason, "task_id": None, "func": None, "unidentified": False}
            if retry:
                self._pending_reply = pending
        task_id = pending["task_id"]
        try:
            word = int(self.plc.read("status"))
            if bit(word, BIT_ESTOP) or bit(word, BIT_INITIALIZING):
                self._abandon_reply("PLC 急停或初始化", task_id)
                return
            if pending["unidentified"]:
                self.events.emit("request_abandoned", task_id=None, request_func=None, reason="首次代答时无法读取 Task_ID，无法确认仍是同一请求")
                logger.error("首次代答时无法读取 Task_ID，放弃无法关联原请求的代答")
                self._pending_reply = None
                return
            current = self.plc.read("task_id")
            if task_id is not None and current != task_id:
                self._abandon_reply(f"Task_ID 已变化为 {current}", task_id)
                return
            task_id = current
            pending["task_id"] = task_id
            if task_id and pending["func"] is None:
                pending["func"] = self.plc.read("request_func")
            result_id = self.plc.read("result_id")
            if task_id == 0 or task_id == result_id:
                if retry:
                    self._pending_reply = None
                return
            Handshake(self.plc, self.plc.contract, self.config.plc.write_readback, self.events).write_reply(task_id, 999, failure=diagnostic(reason, code="SUPERVISOR-CHILD", where="父程序代答", hint="检查主程序退出原因、看门狗和设备状态后重新初始化。"))
            if retry:
                self._pending_reply = None
            self.events.emit("supervisor_reply_999", task_id=task_id, request_func=pending.get("func"), reply_func=999, reason=reason, **diagnostic(reason, code="SUPERVISOR-CHILD", where="父程序代答", hint="检查主程序退出原因、看门狗和设备状态后重新初始化。"))
            logger.error("主程序%s；已给 Task_ID=%s 回 999", reason, task_id)
        except Exception as exc:
            if retry and task_id is None:
                pending["unidentified"] = True
            self.events.emit("error", task_id=task_id, operation="supervisor_reply", reason=str(exc))
            logger.error("错误码：SUPERVISOR-REPLY；主程序%s；Task_ID=%s 代答失败：%s", reason, task_id, exc)

    def _running(self) -> bool:
        """检查子进程并记录意外退出，等待下一个初始化边沿。"""
        if self.child is None:
            return False
        code = self.child.poll()
        if code is None:
            return True
        reason = f"意外退出，退出码 {code}"
        self._last_exit = reason
        logger.error("主程序%s；等待下一次初始化信号", reason)
        self.events.emit("child_exit", reason=reason, code=code)
        self.child = None
        self._started_at = None
        self._clear_ready()
        self._reply_after_failure(reason)
        return False

    def _stop(self, reason: str) -> bool:
        """先请求子进程退出，超时后强制结束并记录原因。"""
        if not self._running():
            return False
        assert self.child is not None
        self._last_exit = reason
        logger.warning("停止主程序：%s", reason)
        self.events.emit("child_stop", reason=reason, child_pid=self.child.pid)
        try:
            self.child.terminate()
            self.child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.child.kill()
            self.child.wait()
        except OSError as exc:
            logger.error(
                "错误码：SUPERVISOR-STOP；停止主程序失败：%s；位置：子进程；排查建议：检查进程权限", exc
            )
            return False
        self.child = None
        self._started_at = None
        self._clear_ready()
        return True

    def _clear_ready(self) -> bool:
        """清除主程序就绪标志；通信失败时记录明确原因。"""
        try:
            self.plc.write("algorithm_ready", 0)
            return True
        except Exception as exc:
            logger.error("算法启动完成标志清零失败：%s；请检查 PLC 连接", exc)
            self.events.emit("error", operation="clear_algorithm_ready", reason=str(exc))
            return False

    def _start(self) -> None:
        """清除旧活性点并启动配置的命令，失败后等待下一条边沿。"""
        if not self._clear_ready():
            return
        try:
            self._alive.unlink(missing_ok=True)
            self.child = subprocess.Popen(self.config.supervisor.child_command)
            self._started_at = time.monotonic()
            self._child_start_wall = datetime.now().astimezone().isoformat()
            logger.info("主程序已启动，PID %s，命令 %s", self.child.pid, self.config.supervisor.child_command)
            self.events.emit(
                "child_start",
                child_pid=self.child.pid,
                reason="初始化中上升沿",
                command=self.config.supervisor.child_command,
            )
        except OSError as exc:
            logger.error(
                "错误码：SUPERVISOR-START；启动主程序失败：%s；位置：子进程；排查建议：检查命令、运行目录和权限",
                exc,
            )

    def _publish(self) -> bool:
        """原子发布状态，避免主程序读取到半个 JSON。"""
        try:
            self._state.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._state.with_suffix(".json.tmp")
            state = {"word": self.word, "bits": describe(self.word or 0), "paused": self.paused, "pid": os.getpid(), "start_time": self._wall_start, "plc_connected": None if self._plc_last_read is None and not self._failures else not bool(self._failures), "plc_last_read": self._plc_last_read, "heartbeat_last_write": self._heartbeat_last_write, "heartbeat_interrupted": bool(self._heartbeat_failures), "child_pid": self.child.pid if self.child else None, "child_state": ("就绪" if self._ready else "启动中") if self.child else ("退出" if self._last_exit else "未启动"), "child_start_time": self._child_start_wall, "child_last_exit_reason": self._last_exit, "algorithm_ready": self._ready}
            temporary.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
            os.replace(temporary, self._state)
            return True
        except OSError as exc:
            logger.error(
                "错误码：SUPERVISOR-STATE；写状态文件失败：%s；位置：%s；排查建议：检查目录权限和磁盘空间",
                exc,
                self._state,
            )
            return False

    def tick(self) -> None:
        """执行一次读状态、心跳和看门狗；通信失败保留上次状态。"""
        running = self._running()
        try:
            word = int(self.plc.read("status"))
        except Exception as exc:
            self._failures += 1
            if self._failures == 1:
                self._failure_since = time.monotonic()
                self.events.emit("plc_disconnected", reason=str(exc))
            if self._failures % 20 == 1:
                logger.warning(
                    "错误码：SUPERVISOR-PLC-READ；PLC 连不上或状态读取失败"
                    "（连续 %s 次）：%s；位置：设备状态；排查建议：查网线/IP/PLC 是否上电；"
                    "保留上次状态并继续重试",
                    self._failures,
                    exc,
                )
        else:
            self._plc_last_read = datetime.now().astimezone().isoformat()
            try:
                self._ready = int(self.plc.read("algorithm_ready"))
            except Exception:
                self._ready = None
            if self._failures:
                logger.info("PLC 状态读取恢复，之前连续失败 %s 次", self._failures)
                self.events.emit("plc_restored", failures=self._failures, duration_s=time.monotonic() - self._failure_since)
                self._failures = 0
            if word != self.word:
                level = logging.ERROR if bit(word, BIT_FAULT) else logging.INFO
                previous = "未知" if self.word is None else f"0x{self.word:04X}"
                logger.log(
                    level,
                    "设备状态变化：%s → %s（%s → 0x%04X）",
                    describe(self.word or 0),
                    describe(word),
                    previous,
                    word,
                )
                if conflicting_bits(word):
                    logger.warning("循环启动与暂停或停止同时为 1，按暂停处理")
                self.events.emit(
                    "device_state",
                    previous=self.word,
                    word=word,
                    bits=describe(word),
                    previous_bits=describe(self.word or 0),
                )
                self.word = word
            abandoned_pending = False
            if bit(word, BIT_ESTOP) or bit(word, BIT_INITIALIZING):
                if self._pending_reply is not None:
                    abandoned_pending = True
                    self._abandon_reply("PLC 急停或初始化")
            elif self._pending_reply is not None:
                self._reply_after_failure(self._pending_reply["reason"])
            decision = self.tracker.decide(word, child_running=running)
            self.paused = decision.paused
            published = self._publish()
            stopped = True
            if decision.action == ACTION_RESTART:
                self.events.emit("child_restart", reason=decision.reason)
            if decision.action in (ACTION_ESTOP, ACTION_RESTART):
                if not abandoned_pending:
                    self._abandon_reply(decision.reason)
                stopped = self._stop(decision.reason)
            if decision.action in (ACTION_START, ACTION_RESTART):
                if published and (stopped or not self._running()):
                    self._start()
                elif not published:
                    logger.error(
                        "错误码：SUPERVISOR-STATE；无法发布暂停状态，主程序未启动；位置：初始化；排查建议：修复状态文件后重新初始化"
                    )
        now = time.monotonic()
        if now >= self._next_heartbeat:
            self._next_heartbeat = now + 1.0
            try:
                next_heartbeat = 1 - self._heartbeat
                self.plc.write("heartbeat", next_heartbeat)
                self._heartbeat = next_heartbeat
                self._heartbeat_last_write = datetime.now().astimezone().isoformat()
                if self._heartbeat_failures:
                    self.events.emit("heartbeat_restored", address=40931, failures=self._heartbeat_failures, duration_s=now - self._heartbeat_failure_since)
                    self._heartbeat_failures = 0
                    self._heartbeat_failure_since = None
            except Exception as exc:
                if not self._heartbeat_failures:
                    self._heartbeat_failure_since = now
                    self._heartbeat_failure_wall = self.events.clock().isoformat()
                self._heartbeat_failures += 1
                self.events.emit("heartbeat_failed", address=40931, failures=self._heartbeat_failures, since=self._heartbeat_failure_wall, reason=str(exc))
                logger.warning(
                    "错误码：SUPERVISOR-HEARTBEAT；PLC 心跳写入失败：%s；位置：心跳；排查建议：查网线/IP/PLC 是否上电",
                    exc,
                )
        if self._running():
            try:
                age = time.time() - self._alive.stat().st_mtime
            except FileNotFoundError:
                started_at = self._started_at if self._started_at is not None else now
                elapsed = time.monotonic() - started_at
                if elapsed > self.config.supervisor.startup_grace_s:
                    reason = f"启动后 {elapsed:.1f} 秒仍未打出活性点，可能卡在预检（相机/PLC）——查 main.log"
                    logger.error("主程序%s", reason)
                    self.events.emit("watchdog", reason=reason)
                    if self._stop(reason):
                        self._reply_after_failure(f"被看门狗停止：{reason}")
            except OSError as exc:
                logger.error(
                    "错误码：SUPERVISOR-ALIVE；读取活性文件失败：%s；位置：%s；"
                    "排查建议：检查目录权限和磁盘。",
                    exc,
                    self._alive,
                )
            else:
                if age > self.config.supervisor.watchdog_s:
                    reason = (
                        f"活性打点超时 {age:.1f} 秒，超过 "
                        f"{self.config.supervisor.watchdog_s:.1f} 秒；查 main.log"
                    )
                    logger.error("主程序%s", reason)
                    self.events.emit("watchdog", reason=reason)
                    if self._stop(reason):
                        self._reply_after_failure(f"被看门狗停止：{reason}")

    def _ui_tick(self):
        if self._ui_disabled or not self.config.ui.enabled:
            return
        now = time.monotonic()
        if self.ui_process is not None:
            code = self.ui_process.poll()
            if code is None:
                return
            self.events.emit("ui_exit", code=code)
            self.ui_process = None
            self._ui_failures = [t for t in self._ui_failures if now - t < 60] + [now]
            if len(self._ui_failures) >= 5:
                logger.error("显示界面一分钟内连续退出五次，停止重启")
                self.events.emit("error", operation="ui_restart", reason="一分钟内连续退出五次，停止重启")
                self._ui_disabled = True
                return
            self._ui_next = now + 5
        if now < self._ui_next:
            return
        try:
            source = str(Path(__file__).resolve().parents[2])
            env = os.environ.copy()
            env["PYTHONPATH"] = source + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            command = [self.config.ui.python, "-m", "nest_clear.ui", "--root", self.config.record_directory, "--refresh-ms", str(self.config.ui.refresh_ms)]
            if self.config.ui.fullscreen:
                command.append("--fullscreen")
            self.ui_process = subprocess.Popen(command, env=env)
            self.events.emit("ui_start", pid=self.ui_process.pid)
        except OSError as exc:
            self.events.emit("error", operation="ui_start", reason=str(exc))
            self._ui_failures = [t for t in self._ui_failures if now-t < 60] + [now]
            self._ui_next = now + 5
            if len(self._ui_failures) >= 5:
                self._ui_disabled = True
                logger.error("显示界面启动失败五次，停止重启")

    def run(self) -> None:
        """持续运行直至中断，并在退出时释放子进程与连接。"""
        self.events.emit("process_start", **self.metadata)
        if self.config.ui.enabled and not os.environ.get("DISPLAY"):
            logger.warning("无 DISPLAY，显示界面不会启动")
            self._ui_disabled = True
        try:
            while True:
                try:
                    self._ui_tick()
                except Exception:
                    logger.exception("显示界面管理失败，父程序继续运行")
                self.tick()
                self._publish()
                time.sleep(self.config.supervisor.poll_s)
        except KeyboardInterrupt:
            logger.info("收到退出信号")
        finally:
            if self.ui_process is not None:
                try:
                    self.ui_process.terminate()
                    self.ui_process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.ui_process.kill()
                    self.ui_process.wait()
                except OSError:
                    logger.exception("停止显示界面失败")
            self._stop("父程序退出")
            self._reply_after_failure("父程序退出", retry=False)
            self.plc.close()
            self.events.emit("process_stop")
