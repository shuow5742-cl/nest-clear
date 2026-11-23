# 主相机标定板（定稿，2026-11-18）

加工图纸：`charuco_21x16_5mm_DICT5X5_250_glass_V1_DFOPTIX.pdf`（东莞市大凡光学科技有限公司 DFOPTIX，V1 首次下发 2026-11-18）。标定板按此图案加工。

| 项 | 值 | 代码对应（config `camera_calibration` / `CameraCalibConfig`） |
|---|---|---|
| 图案 | ChArUco，OpenCV ≥4.6 标准布局（非 legacy），左上角为黑格 | `cv2.aruco.CharucoBoard((21,16), 5.0, 3.75, DICT_5X5_250)` |
| 列 × 行 | 21 × 16 格 | `cols: 21`, `rows: 16` |
| 格 / 码 | 5 mm / 3.75 mm | `square_mm: 5.0`, `marker_mm: 3.75` |
| 字典 | DICT_5X5_250，使用 ID 0–167（168 个码） | `dictionary: DICT_5X5_250` |
| 内角点 | 300 个（20 × 15） | — |
| 外形 / 图案 | 115 × 90 mm / 105 × 80 mm，居中留边 5 mm | — |
| 厚度 | 3.15 mm（11-22 实测） | `board_thickness_mm: 3.15`（相机工作平面为板面；工具标定按板厚及卡厚换算） |
| 基材 | 玻璃，低反射铬，图案在上表面 | — |
| 精度 / 平面度 | ±5 µm / ≤0.02 mm | — |

## 一致性验证（Claude，2026-11-18）
厂家 PDF 与原设计 PDF（`~/文档/CNN/燕窝二代_ChArUco标定板图案_20261116/charuco_21x16_5mm_DICT5X5_250.pdf`）均以 600 dpi 渲染：
- OpenCV CharucoDetector：两份均检出码 168/168、角点 300/300，非镜像；厂家图比例 23.6223 px/mm（理论 23.6220，1:1），旋转 0°。
- 两份按角点单应对齐到 20 px/mm 后逐像素比较：差异 0.96%，全部为码元边缘栅格化；5×5 腐蚀后差异 0 像素 ⇒ **图案逐码元一致**。
