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
