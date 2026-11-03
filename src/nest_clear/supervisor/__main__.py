"""提供父程序命令行入口，配置按天日志并记录版本。"""

import argparse
import hashlib
from datetime import date
import logging
from pathlib import Path
import subprocess

from ..comm import PLCClient, load_contract
from ..config import load_config
from ..guard import GuardError
from ..paths import supervisor_log_file
from .runner import Supervisor

ROOT = Path(__file__).resolve().parents[3]


class DailyHandler(logging.Handler):
    """按本地日期切换文件，确保跨日日志进入当天目录。"""

    def __init__(self, root: str) -> None:
        """保存记录根目录，并延迟到首条日志时打开文件。"""
        super().__init__()
        self.root = root
        self.day: date | None = None
        self.handler: logging.FileHandler | None = None

    def emit(self, record: logging.LogRecord) -> None:
        """按当前日期切换文件，供标准 logging 调用。"""
        try:
            day = date.today()
            if self.handler is None or self.day != day:
                if self.handler is not None:
                    self.handler.close()
                path = supervisor_log_file(self.root, day)
                path.parent.mkdir(parents=True, exist_ok=True)
                self.handler = logging.FileHandler(path, encoding="utf-8")
                self.handler.setFormatter(self.formatter)
                self.day = day
            self.handler.emit(record)
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        """关闭当前日期文件，供 logging 关闭时调用。"""
        if self.handler is not None:
            self.handler.close()
        super().close()


def _commit(path: Path) -> str:
    """查询仓库当前提交，查询失败时返回可诊断说明。"""
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL, timeout=5
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"无法读取（{exc}）"


def main() -> None:
    """加载配置与契约，记录两个仓库版本后启动父程序。"""
    parser = argparse.ArgumentParser(description="燕窝挑毛父程序")
    parser.add_argument("--config", default=str(ROOT / "config/nest_clear.yaml"))
    parser.add_argument("--contract", default=str(ROOT / "config/plc_contract.yaml"))
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        contract = load_contract(args.contract)
        handler = DailyHandler(config.record_directory)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
            handlers=[logging.StreamHandler(), handler],
        )
        logger = logging.getLogger("nest_clear.supervisor")
        logger.info(
            "父程序启动：nest-clear commit %s；Auto_Weaver commit %s",
            _commit(ROOT),
            _commit(Path("/home/shuow/Auto_Weaver")),
        )
        metadata = {
            "nest_commit": _commit(ROOT),
            "framework_commit": _commit(ROOT.parent / "Auto_Weaver"),
            "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
            "plc": f"{config.plc.ip}:{config.plc.port}",
            "camera_sn": config.camera_sn,
        }
        Supervisor(
            config,
            PLCClient(config.plc.ip, config.plc.port, config.plc.timeout_s, contract),
            metadata=metadata,
        ).run()
    except GuardError as exc:
        logging.getLogger("nest_clear.supervisor").error("启动失败：%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
