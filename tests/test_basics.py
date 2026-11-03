"""验证配置、地址契约、编解码和握手顺序。"""
from pathlib import Path
import dataclasses
import yaml
import pytest
from nest_clear.config import load_config
from nest_clear.guard import GuardError
from nest_clear.comm import load_contract, Handshake
from nest_clear.comm.codec import encode_real32, decode_real32, encode_u16

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('value', [0, 1, -1, -123.25, 3.14159])
def test_real_roundtrip(value):
    assert decode_real32(encode_real32(value)) == pytest.approx(value)

def test_cdab():
    assert encode_real32(1.0) == [0, 0x3f80]
    assert decode_real32([0, 0xbf80]) == -1.0
    with pytest.raises(GuardError): encode_u16(-1)

def test_contract():
    contract = load_contract(ROOT / 'config/plc_contract.yaml')
    assert contract['task_id'].address == 40903
    assert contract['calibration2_z'].type == 'REAL32'
    assert contract['result_id'].direction == 'PC→PLC'
    with pytest.raises(GuardError): contract['missing']

def test_config():
    config = load_config(ROOT / 'config/nest_clear.yaml')
    assert len(config.scan.corners) == 3
    with pytest.raises(dataclasses.FrozenInstanceError): config.plc.port = 1

@pytest.mark.parametrize('change', ['missing', 'type', 'outside'])
def test_invalid_config(tmp_path, change):
    data = yaml.safe_load((ROOT / 'config/nest_clear.yaml').read_text())
    if change == 'missing': del data['camera']['sn']
    elif change == 'type': data['plc']['port'] = True
    else: data['scan']['corners'][0][0] = 99999
    path = tmp_path / 'bad.yaml'; path.write_text(yaml.safe_dump(data))
    with pytest.raises(GuardError, match='配置项'): load_config(path)

def test_order():
    calls = []
    class IO:
        def write_real32_block(self, address, values): calls.append((address, values))
        def write_u16(self, address, value): calls.append((address, value))
    c = load_contract(ROOT / 'config/plc_contract.yaml')
    Handshake(IO(), c).write_reply(42, 1, {'scan_xyz': [1, 2, 3]})
    assert [c['scan_x'].address, c['reply_func'].address, c['result_id'].address] == [v[0] for v in calls]
