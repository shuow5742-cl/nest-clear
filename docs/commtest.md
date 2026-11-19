# 通信联调模式

此模式用于空料盘上的挑毛主流程通信联调；三轴会真实运动。PLC、握手、心跳、到位检查、相机采图及真实检测仍执行；下发的挑取目标由固定种子假世界生成。确认设备和人员安全后使用，禁止将运行数据当作生产数据。

在仓库根目录，用项目环境启动：

```bash
PYTHONPATH=src .venv/bin/python -m nest_clear.tools.list_cameras
PYTHONPATH=src .venv/bin/python -m nest_clear.supervisor --config config/commtest.yaml
# 或不经父程序直接启动：
PYTHONPATH=src .venv/bin/python -m nest_clear.main --config config/commtest.yaml
```

父程序配置已指定子程序使用同一配置。`camera.sn: auto` 要求恰好一台大恒相机；实际 SN 写入 `meta.json`。曝光、增益和白平衡配置会写入相机并回读，现场使用前须按设备核对 `camera` 参数。行程仅允许 X [80,180]、Y [-10,150]、Z [0,0]，工具补偿使用 `config/commtest_calibration/tools.yaml`，记录放在 `runs_commtest`。不得改生产工具文件来联调。

`commtest.seed` 保证相同场景、路径和请求顺序下的假坐标可复现；每视野四根。`normal` 挑取后消失，结束 9；`tool_change` 首视野有三根不会消失，结束 10；`fine_fail` 部分精定位交替返回空框和点在框外，回 1 重拍，结束 9；`mixed` 混合精定位失败和末视野不消失，结束 10。真实视觉的空结果或异常均写 `real_detection`，不阻止假点；相机硬件故障仍回 999。标注图使用真实检测框，品红十字与文字为假点；无法投影时仅写假坐标。

## 预期通信时序

PLC 先发 1，PC 回 1 和首扫描位；PLC 到位后发 20，PC 真实拍照与检测，再回 1 和假目标精拍位；PLC 到位再发 20，PC 回 3 和两把工具假坐标；PLC 挑取完成发 60。PC 若有下一目标回 1 和下一精拍位，否则回 1 和当前或下一视野扫描位。循环直至回 9（整盘完成）或 10（换工具判定）。精定位失败回 1 和当前视野，PLC 到位后发 20 重拍。相机、到位、行程及通信等真实故障可回 999；80 和 100–131 回 999，诊断 `COMMTEST-UNSUPPORTED`。

以默认种子和配置，PC 应答计数如下（包含最初扫描位）：

| 场景 | 1 | 3 | 9 | 10 |
|---|---:|---:|---:|---:|
| normal | 36 | 24 | 1 | 0 |
| tool_change | 6 | 4 | 0 | 1 |
| fine_fail | 39 | 21 | 1 | 0 |
| mixed | 39 | 21 | 0 | 1 |

## 移除联调模式

删除 `src/nest_clear/commtest/`、`config/commtest.yaml`、`config/commtest_calibration/`、联调测试及本文。移除 `main.py` 中联调视觉工厂、相机自动选择和假点预检接缝，`flow/picking.py` 中坐标工厂接缝，`main.py` 中不支持功能码的分发表接缝，以及 `config.py` 中模式、联调配置和行程校验。移除 `record/events.py`、`record/report.py`、`ui/status.py`、`ui/app.py` 的联调标识；相机枚举工具可独立保留。恢复时保留生产行程、占位标定、相机和 PLC 守卫，并运行完整回归。

## 真实视觉影子运行（11-18 验收时补充）
真实 VisionWorker 在独立单线程里"影子运行"：假结果立即交给状态机/PLC，真检测对每张图完整跑并记 `real_detection`（status=ok/empty/error、elapsed_ms）。
上一张未算完时，新图记 `status=skipped_busy`（不排队）。真检测耗时不会拖慢应答；现场看 `real_detection` 的 elapsed_ms 与 skipped_busy 数量评估图像处理速度。
标注图只画下发的假点（品红），真检测结果以事件为准。
