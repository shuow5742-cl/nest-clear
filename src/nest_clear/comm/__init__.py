"""PLC 通信组件，衔接地址契约、Modbus 传输与握手。"""
from .contract import load_contract
from .client import PLCClient
from .handshake import Handshake
