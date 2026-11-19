"""定义统一守卫错误，供配置、通信及后续流程报告可定位的问题。"""


class GuardError(Exception):
    """保存错误码、现象、位置和排查建议，供调用方记录和展示。"""

    def __init__(self, code: str, what: str, where: str, hint: str) -> None:
        """保存四项诊断信息，便于日志和人工排查使用。"""
        self.code = code
        self.what = what
        self.where = where
        self.hint = hint
        super().__init__(code, what, where, hint)

    def __str__(self) -> str:
        """用中文分行展示诊断信息，便于现场人员直接阅读。"""
        return (
            f"错误码：{self.code}\n发生了什么：{self.what}\n"
            f"位置：{self.where}\n排查建议：{self.hint}"
        )


def diagnostic(error, *, code="INTERNAL-ERROR", where="算法流程", hint="检查相关设备和程序日志后重试。"):
    """将 GuardError 或工作线程传回的错误文本转为可序列化诊断。"""
    if isinstance(error, GuardError):
        return {"reason_code": error.code, "what": error.what, "where": error.where, "hint": error.hint}
    value = str(error or "未知错误")
    parsed = {}
    for line in value.splitlines():
        for prefix, field in (("错误码：", "reason_code"), ("发生了什么：", "what"), ("位置：", "where"), ("排查建议：", "hint")):
            if line.startswith(prefix):
                parsed[field] = line[len(prefix):].strip()
    return {"reason_code": parsed.get("reason_code") or code, "what": parsed.get("what") or value,
            "where": parsed.get("where") or where, "hint": parsed.get("hint") or hint}
