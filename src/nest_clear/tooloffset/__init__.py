"""工具偏移结果文件的读写；命令通道无需加载 YAML。"""

__all__ = ['load_offset', 'save_result']


def __getattr__(name):
    if name in __all__:
        from . import store
        return getattr(store, name)
    raise AttributeError(name)
