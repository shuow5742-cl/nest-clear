"""换针对刀界面共享的数值显示规则。"""


def format_number(value, digits=3, *, signed=False):
    """四舍五入后将负零显示为正零。"""
    return format(float(value), f'{"+" if signed else ""}z.{digits}f')
