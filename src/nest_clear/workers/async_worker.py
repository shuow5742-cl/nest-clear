"""给耗时操作提供不会提前确认的异步 note 协议。"""
from typing import Any

from autoweaver.worker.base import Worker


class AsyncWorker(Worker):
    """用框架线程池执行命令，结果和错误在 tick 线程提交。"""
    def __init__(self, name: str) -> None:
        """设置 Worker 命名空间，保证状态写入归属明确。"""
        super().__init__()
        self._name = name

    @property
    def name(self) -> str:
        """返回 WorldBoard 命名空间。"""
        return self._name

    def on_attach(self) -> None:
        """注册异步命令和结果状态，供树按请求编号等待。"""
        self.declare_state(f"{self.name}.result", dict)
        self.write_state(f"{self.name}.result", {})
        self._board.accept_notes(namespace=self.name, name="execute", payload_type=dict, on_receive=self._receive)

    def _receive(self, payload: dict[str, Any]) -> None:
        """只提交后台工作；同步 Modbus 和推理不能阻塞全局 tick。"""
        request = dict(payload)
        rid = request.pop("__request_id__")
        self.write_state(f"{self.name}.last_request_id", rid)
        self.write_state(f"{self.name}.last_error", "")

        def done(result: Any) -> None:
            """先发布结果，再确认编号，使树不会读到旧结果。"""
            self.write_state(f"{self.name}.result", {"id": rid, "value": result})
            self.write_state(f"{self.name}.last_completed_id", rid)

        def error(exc: BaseException) -> None:
            """失败也完成等待，但单独保存错误供树检查。"""
            self.write_state(f"{self.name}.last_error", str(exc))
            self.write_state(f"{self.name}.result", {"id": rid, "error": str(exc)})
            self.write_state(f"{self.name}.last_completed_id", rid)

        self.run_async(lambda: self.execute(request), done, error)

    def execute(self, payload: dict[str, Any]) -> Any:
        """由子类执行实际设备操作。"""
        raise NotImplementedError
