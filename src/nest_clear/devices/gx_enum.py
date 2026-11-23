"""Normalize gxipy enum feature readbacks before comparison or restoration."""


def enum_value(value):
    if isinstance(value, (tuple, list)):
        value = value[0]
    return int(value)
