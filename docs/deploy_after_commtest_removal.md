# 设备迁移步骤（操作说明，未执行）

1. 停止设备程序，确认 PLC 和机械设备处于安全状态。设备上的配置和标定目录有 Git 跟踪文件，现场可能修改过。先将 `~/nest-clear/config/commtest_calibration/` 和 `~/nest-clear/runs_commtest/` 完整备份到 `~/nest-clear-backup-<日期>/`；同时备份现场修改的其他配置、生产标定文件和工作区差异。
2. 核实备份完整后，按现场变更记录恢复干净工作区，再执行 ff-only 合并。不要在备份之前覆盖或清理被跟踪的现场文件。
3. 检查备份中的标定产物，只有确认非占位且属于本设备的文件才复制到 `~/nest-clear/config/calibration/` 中对应位置，包括 camera.yaml、tools.yaml、tool1/2_baseline.yaml、tool_offset.yaml、pick_compensation.yaml、camera_exposure.yaml 等；保留备份。缺失或占位的产物需重新标定。
4. 将 `~/.config/nest-clear/env` 里的配置路径改为 `config/nest_clear.yaml`（按服务的工作目录使用绝对路径更稳妥）。核实 PLC 地址、相机连接、扫描覆盖、设备行程及标定参数，然后按部署说明启动并验证。新运行记录位于 `runs/`。
