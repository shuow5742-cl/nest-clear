from dataclasses import replace
from nest_clear.comm import Handshake, load_contract
from nest_clear.comm.codec import encode_real32
from nest_clear.config import load_config
from nest_clear.record.events import EventRecorder
from nest_clear.supervisor.runner import Supervisor
from nest_clear.ui.model import classify_event

class Registers:
    def __init__(self, contract): self.words={}; self.contract=contract; self.corrupt=True
    def write_u16(self,a,v): self.words[a]=v
    def write_real32_block(self,a,values):
        for i,w in enumerate(sum((encode_real32(v) for v in values),[])): self.words[a+i]=w
    def read_u16(self,a): return self.words.get(a,0)+self.corrupt
    def read(self,name): return {'status':0,'task_id':9,'result_id':0,'request_func':20}.get(name,0)

def check(row, code):
    assert row['reply_func']==code
    assert all(row.get(k) for k in ('reason_code','what','where','hint'))
    model=classify_event(row)
    assert (model['phenomenon'],model['location'],model['suggestion']) == (row['what'],row['where'],row['hint'])

def test_readback_999_contains_diagnostic(tmp_path):
    contract=load_contract('config/plc_contract.yaml'); io=Registers(contract); events=EventRecorder(tmp_path); rows=[]; events.listener=rows.append
    Handshake(io,contract,events=events).write_reply(9,1,{'scan_xyz':[1,2,3]})
    row=next(r for r in rows if r['event']=='reply_error'); check(row,999); assert row['reason_code']=='COMM-READBACK'

def test_supervisor_999_contains_diagnostic(tmp_path):
    contract=load_contract('config/plc_contract.yaml'); io=Registers(contract); io.corrupt=False
    s=Supervisor(replace(load_config('config/nest_clear.yaml'),record_directory=str(tmp_path)),io); rows=[]; s.events.listener=rows.append
    s._reply_after_failure('看门狗超时')
    row=next(r for r in rows if r['event']=='supervisor_reply_999'); check(row,999); assert '看门狗超时' in row['what']

def test_actual_reply_event_keeps_readback_cause(tmp_path):
    import time
    from nest_clear.flow.leaves import CommitReply
    from nest_clear.flow.modes import Mode
    contract=load_contract('config/plc_contract.yaml'); io=Registers(contract); events=EventRecorder(tmp_path); rows=[]; events.listener=rows.append
    Handshake(io,contract,events=events).write_reply(9,1,{'scan_xyz':[1,2,3]})
    class Blackboard:
        def __init__(self): self.data={'task_id':9,'committed_func':999,'reply_func':1,'mode':Mode.IDLE,'started':time.monotonic(),'request_func':20}
        def read(self,key,default=None): return self.data.get(key,default)
        def write(self,key,value,*args): self.data[key]=value
    leaf=CommitReply(events); leaf._blackboard=Blackboard(); leaf.perform()
    row=next(r for r in rows if r['event']=='reply')
    assert row['func']==999 and row['reason_code']=='COMM-READBACK'
    assert all(row.get(k) for k in ('what','where','hint'))
