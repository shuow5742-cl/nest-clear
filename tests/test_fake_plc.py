"""通过真实 TCP 验证客户端、模拟 PLC、写入顺序及整盘驱动。"""
import threading
from sim.fake_plc import FakePLC
from nest_clear.comm import PLCClient, Handshake

def test_exchange_and_wrap():
    with FakePLC(wrap=2) as plc:
        client = PLCClient(plc.host, plc.port, 1, plc.contract)
        hand = Handshake(client, plc.contract)
        plc.set_status(3)
        assert client.read('status') == 3
        assert plc.request(1) == 1
        assert hand.read_request() == (1, 1, None)
        plc.writes.clear()
        hand.write_reply(1, 1, {'scan_xyz': [1, -2, 3]})
        assert plc.wait_reply(1)['scan_xyz'] == [1, -2, 3]
        assert [entry[0] for entry in plc.writes] == [plc.contract[n].address for n in ('scan_x', 'reply_func', 'result_id')]
        assert plc.request(60) == 2
        assert plc.request(1) == 1
        client.close()

def test_tray():
    """模拟工作模式，验证首次请求、到位和挑取完成的功能码。"""
    with FakePLC() as plc:
        client = PLCClient(plc.host, plc.port, 1, plc.contract)
        hand = Handshake(client, plc.contract)
        stop = threading.Event()
        errors = []

        def responder():
            """有限等待请求，异常交给测试线程报告。"""
            last = 0
            try:
                while not stop.is_set():
                    task, func, _ = hand.read_request()
                    if not task or task == last:
                        stop.wait(0.005)
                        continue
                    last = task
                    if task == 1:
                        assert func == 1
                        hand.write_reply(task, 1, {"scan_xyz": [1, 2, 3]})
                    elif task == 2:
                        assert func == 20
                        assert client.read_real32_block(plc.contract["actual_x"].address, 3) == [1, 2, 3]
                        hand.write_reply(task, 3, {"pick1_xyz": [4, 5, 6], "pick2_xyz": [7, 8, 9]})
                    elif task == 3:
                        assert func == 60
                        hand.write_reply(task, 60)
                    else:
                        assert func == 1
                        hand.write_reply(task, 9)
                        return
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=responder, name="fake-plc-responder", daemon=True)
        thread.start()
        try:
            records = plc.drive_tray()
            assert [r["func_code"] for r in records] == [1, 3, 60, 9]
            assert [r["request_func"] for r in records] == [1, 20, 60, 1]
        finally:
            stop.set()
            thread.join(3)
            client.close()
            if thread.is_alive():
                thread.join(3)
        assert not thread.is_alive()
        assert not errors


def test_plc_exception_and_reconnect(caplog):
    """真实服务器返回非法地址异常，并验证关闭连接后的下一次访问会重连。"""
    import logging
    import pytest
    from nest_clear.guard import GuardError

    with FakePLC() as plc:
        client = PLCClient(plc.host, plc.port, 1, plc.contract)
        caplog.set_level(logging.INFO, logger="nest_clear.comm")
        assert client.read("task_id") == 0
        with pytest.raises(GuardError) as captured:
            client.read_u16(50000)
        assert captured.value.code == "COMM-PLC-EXCEPTION"
        assert "异常码 2" in captured.value.what
        client.close()
        assert client.read("task_id") == 0
        assert "重连成功" in caplog.text
        client.close()


def test_tool_reply_write_order_and_skip_40935():
    """通过内存 Modbus 假设备验证工具应答与非工具应答的寄存器写入顺序。"""
    with FakePLC() as plc:
        client = PLCClient(plc.host, plc.port, 1, plc.contract)
        hand = Handshake(client, plc.contract)
        try:
            for tool, addr_name in ((1, "calibration1_x"), (2, "calibration2_x")):
                task = plc.request(155, tool_id=tool)
                assert hand.read_request() == (task, 155, tool)
                plc.writes.clear()
                hand.write_reply(task, 155, {f"calibration{tool}_xyz": [10., 20., -10.], "pc_tool_id": tool})
                assert [a for a, _ in plc.writes] == [plc.contract[n].address for n in (addr_name, "pc_tool_id", "reply_func", "result_id")]
                reply = plc.wait_reply(task, 5)
                assert reply["func_code"] == 155 and reply["pc_tool_id"] == tool

            task = plc.request(100, tool_id=9)
            assert hand.read_request() == (task, 100, 9)
            plc.writes.clear()
            hand.write_reply(task, 999, {"pc_tool_id": 9})
            assert [a for a, _ in plc.writes] == [plc.contract[n].address for n in ("pc_tool_id", "reply_func", "result_id")]

            task = plc.request(80)
            assert hand.read_request() == (task, 80, None)
            plc.writes.clear()
            hand.write_reply(task, 80, {"scan_xyz": [1., 2., 3.]})
            assert [a for a, _ in plc.writes] == [plc.contract[n].address for n in ("scan_x", "reply_func", "result_id")]
        finally:
            client.close()
