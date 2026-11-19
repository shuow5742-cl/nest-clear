# T26 通信联调模式（commtest）任务书

## 目标
配合电气测试**挑毛主流程**的 PLC 通信。除"交给状态机/PLC 的目标坐标"是假点外，其余一切都是真的：
真 Modbus（192.168.10.10:502）、真握手/心跳/40932/Task_ID、真相机（真打开、真设曝光增益白平衡并回读、真拍、真关闭）、
真检测器在真图上真跑、真守卫（到位检查 arrival_tolerance_mm、行程、取图守卫）、真记录/日报/界面。
三轴真的会动。只测挑毛主流程；相机标定(80)和换针对刀(100–131)不测。
这是临时联调代码，后续会全部改回真实情况：**假的部分必须集中、低耦合，删掉即可回到生产**。

## 设计（必须遵守）
1. 新目录 `src/nest_clear/commtest/`，所有假逻辑放这里（假世界、结果替换适配器、假坐标函数）。
   生产代码只允许在"组装处"（main.py 组装 worker / flow/picking.py 注入 fine_coord、pick_coords 处）各加一个
   `if config.run_mode == "commtest"` 分支调用 commtest 模块的工厂函数；不得把假逻辑散落进 work_cycle、handshake、vision 等模块。
   config.py 增加顶层 `run_mode: production|commtest`（缺省 production）和 `commtest:` 配置段（仅 commtest 时必填）。
2. **结果替换适配器**：包住真实 VisionWorker。每次请求先完整执行真实 VisionWorker（真图真检测）。**用户 11-18 裁决：料盘上不放任何东西，真图检测结果必然"有问题"——图像处理层的任何结果或异常（检测为空、检测器抛异常、select_fine_target 失败、尺度/参数告警等）都不得导致 999**，一律捕获后记入 real_detection（status=ok/empty/error，error 时附异常类型、消息、traceback 摘要），照常返回假结果；**相机硬件层（打开失败、取图失败/超时、曝光增益白平衡回读不符、图像尺寸不符）仍按现有真规则回 999**——这些在 CameraWorker 层，不在适配器里捕获。
   把真实结果记为事件 `real_detection`（capture_id、阶段 coarse/fine、候选数、每个候选像素与模组坐标、耗时、fine 阶段 select_fine_target 的真实状态），
   然后返回假世界生成的结果（与真实结果同类型/同字段，状态机无感知）。
3. **假世界**（固定随机种子，同场景每次结果完全一致）：每个扫描视野 3–5 根假毛，全盘总挑取约 25 次；假毛模组 XY 位于该视野责任区内，
   且距安全范围边界 ≥5mm（保证两把工具坐标都在范围内）；粗拍返回当前视野剩余假毛；精定位返回同一根毛加 ≤0.3mm 抖动（固定种子）；
   一次挑取（收到该目标的 60/复拍）后按场景决定消失与否。场景 `commtest.scenario`：
   - `normal`：挑一次即消失 → 最终回 9
   - `tool_change`：部分毛永不消失 → 触发换工具判定回 10
   - `fine_fail`：部分精定位返回失败状态（fine_empty / fine_point_outside 轮换）→ 回 1 重拍
   - `mixed`：以上混合，仍能走到 9 或 10 之一（写清楚哪一个）
4. **假坐标函数**：精定位相机位置 = 目标 XY、Z = 扫描 Z（不经标定）；工具坐标 = 目标 + config.tools 补偿，Z = pick_depth_z + 补偿 Z，
   仍走现有行程守卫。commtest 下跳过"占位标定拒绝下发"守卫（仅 commtest 分支内，生产不变）。
5. 新配置文件 `config/commtest.yaml`（从 nest_clear.yaml 复制后修改）：run_mode: commtest；plc 192.168.10.10:502；
   travel x[80,180] y[-10,150] z[0,0]；scan 用 corners 模式蛇形 6 视野：(160,0)→(160,70)→(160,140)→(100,140)→(100,70)→(100,0)，Z=0
   （若现有 corners/step 不能精确生成此序列，就在 commtest 配置里给显式视野列表，并实现最小改动的 scan.mode: list）；
   pick_depth_z 0；工具1 补偿 (0,0,0)、工具2 (+2,+2,0)（看 tooltip store/load_tools 怎么提供补偿，commtest 用独立的工具文件，不改生产工具文件）；
   tool_calibration.enabled false；camera_backend daheng；camera.sn: auto；record.directory runs_commtest；scenario mixed。
   **启动守卫**：commtest 下 travel 必须落在 X[80,180]/Y[-10,150]/Z[0,0] 以内、所有扫描位与假点都在内，否则启动失败（GuardError，写 startup_error.log）。
6. **相机 SN**：新工具 `python -m nest_clear.tools.list_cameras` 枚举大恒相机（SN、型号、接口），无 gxipy 时给明确报错。
   `camera.sn: auto` 只允许 run_mode=commtest：恰好 1 台→使用并把实际 SN 写进 meta.json；0 台或多台→启动失败并说明。生产仍须写死 SN。
7. **80 与 100–131**：commtest 下收到这些功能码 → 999，reason_code `COMMTEST-UNSUPPORTED`，what/where/hint 写"通信联调模式不支持相机标定/换针对刀"。
8. **标识防混淆**：meta.json 写 run_mode=commtest、scenario、seed；每条应答事件带 `commtest: true`；
   界面（ui）在 main_status.json 中有 run_mode 时顶部显示醒目横幅"通信联调模式：下发坐标为假点"；日报标题同样标注；
   粗拍/精定位标注图：画真实检测框（按现有六态风格中可用的），另以品红十字+文字标出本次下发的假点（用标定把模组坐标投影到像素，投影失败则只在信息栏写坐标）。
9. 文档：`docs/commtest.md`——用途、启动命令（父程序/主程序如何用 commtest.yaml 启动）、场景说明、预期应答序列、如何删除（列出所有接缝位置），
   给电气看的"预期通信时序"一节（每视野 1→20→3→60…、最后 9 或 10、999 情形）。

## 不许做
- 不改 Auto_Weaver（vendor/依赖）任何东西；不改生产 config/nest_clear.yaml 的行为（可加 run_mode: production 一行）。
- 不改 work_cycle 状态机逻辑、握手协议逻辑。
- 不做 git 分支/提交（.git 只读），不删已有测试。

## 测试（必须新增并全部通过）
在 tests/ 新增 commtest 测试（不开 TCP，用 tests/test_fake_plc_no_tcp.py 的方式驱动；相机用 mock 后端但走真实 CameraWorker 与真实 HairDetector 或现有测试用的检测器，确认真实 VisionWorker 确实被调用、real_detection 落账）：
- 四个场景各跑完一盘：normal→9、tool_change→10、fine_fail 出现回 1、mixed 按文档结局；所有下发坐标（扫描/工具1/工具2）在安全范围内；工具2 = 工具1 + (2,2,0)。
- 同种子两次运行下发坐标序列完全一致。
- 真实检测器抛异常 → **不回 999**，流程照常下发假点并走完，real_detection 记 status=error 与异常信息；mock 相机取图失败 → 999（硬件层规则不变）。
- 80、100 → 999 COMMTEST-UNSUPPORTED。
- 启动守卫：travel 超范围或 sn: auto 用于 production → 启动失败。
- 生产模式行为不变（现有测试全过）。

自测命令（必须用项目 .venv；沙箱禁 TCP，脚本会把被拦的文件标 SANDBOX，不算失败，不要去修它们）：
```
bash /tmp/claude-1000/-home-shuow/507ae54d-774c-46b6-9ca9-d506ff74d00b/scratchpad/selftest.sh
```
每个测试文件只输出一行。要求：没有 FAIL 行；你新增的 commtest 测试必须是 OK（不得依赖 TCP）。
单独调试某个文件：`cd /home/shuow/nest-clear && TMPDIR=$PWD/.pytest_tmp PYTHONPATH=src .venv/bin/python -m pytest -q -x tests/<文件> 2>&1 | tail -30`，只看 tail。

## 结束条件
**全部做完、上面自测全部通过才结束，不要分阶段交差。** 最后输出总结：新增/修改文件清单、每个接缝位置（文件:行）、四个场景的应答序列摘要（功能码计数）、测试结果。
