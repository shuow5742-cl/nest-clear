# 挑取精度测试菲林片 V2（2026-11-23）

现场摆放为旋转 90°：卡片 +X 对应设备 Y+，卡片 −Y 对应设备 X+。V2 按此关系标注方向箭头；卡片自身的坐标定义、原点、32 个目标和 ArUco 编号均不变。V1 文件保留供追溯。

生产文件：燕窝二代_挑取精度测试菲林片_V2_245x120.pdf；A4 版：燕窝二代_挑取测试菲林片_A4打印版_20261123.pdf。verify_a4.py 同时将 V1 与 V2 每个目标（含编号及 ArUco）按 508 dpi 渲染逐像素比对。

在本目录依次运行项目 Python 执行 make_film.py、make_film.py --a4、verify_film.py、verify_a4.py；需要 ReportLab、OpenCV 和 Poppler。

定制文件（发厂家）：~/文档/CNN/燕窝二代_挑取测试菲林片_20261118/（菲林片定制图案_245x120.pdf + 菲林片定制要求.pdf）。
本目录为工程资料：targets.csv（32 目标设计坐标，原点=片中心，X 沿 245 边向右，Y 向上；ArUco DICT_4X4_50 id=index，码中心=目标+(7,7)mm）、make_film.py（生成）、verify_film.py（自检）、自检结果.txt、预览图、同版 PDF。
规格：白色哑光 PET 0.10±0.01mm（厂家逐张千分尺测 5 点，代码按实测厚度扣除）；8×4 目标 @30mm；环 Ø4/线宽0.3 + 十字 10mm/0.1。

## A4 纸张打印版（V2，2026-11-23）

`燕窝二代_挑取测试菲林片_A4打印版_20261123.pdf` 将原 245 × 120 mm 图案居中放入 A4 横向页面，图案保持 1:1；四周有 X、Y 方向的 100 mm 和 50 mm 校验尺。打印时选择“实际大小/100%”，关闭适合页面或自动缩放；用钢尺检查 100 mm 校验尺为 100.0 ± 0.3 mm。贴纸仅贴图案外四边，保持平整，避免针尖压动。普通打印机可能无法打印页面边缘，请检查四条校验尺是否完整。

在本目录用项目 Python 运行 `../../.venv/bin/python make_film.py --a4` 生成打印版（需要 ReportLab 和 OpenCV；本机若虚拟环境缺少 ReportLab，可在命令前加 `PYTHONPATH=/home/shuow/.local/lib/python3.13/site-packages`），再运行 `../../.venv/bin/python verify_a4.py` 更新 `自检结果.txt` 中的打印版自检。生成打印版不会改写原菲林 PDF 或 `targets.csv`。预览图仅供查看，不用于打印。

## 主流程测试纸（2026-11-23）

`燕窝二代_主流程测试纸_A4打印版_20261123.pdf` 用于主流程测试模式：245 × 120 mm 图案居中置于横向 A4，15 列 × 7 行共 105 个目标，间距 15 mm。编号 T00–T31 循环重复；主流程按模组坐标去重，精定位时在同编号目标中取离粗拍位置最近的目标，因此支持重复编号。默认不可去目标为 T03、T06、…、T30，包含这些编号重复出现的所有副本，约占三分之一。

打印方法同上面的 A4 版：选择实际大小/100%，关闭自动缩放，用钢尺检查校验尺并保持纸面平整。该纸仅用于主流程测试模式，不可用于工具偏移或挑取精度标定。`test_sheet_targets.csv` 是设计坐标，`test_sheet_preview.png` 仅供预览，不能用于打印。

在本目录生成和自检：

```sh
PYTHONPATH=/home/shuow/.local/lib/python3.13/site-packages ../../.venv/bin/python make_test_sheet.py
PYTHONPATH=/home/shuow/.local/lib/python3.13/site-packages ../../.venv/bin/python verify_test_sheet.py
```
