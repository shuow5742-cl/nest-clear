# 界面部署

界面进程默认使用 `/usr/bin/python3`，父程序会将本项目 `src` 加入界面进程的 `PYTHONPATH`。如系统 Python 安装在其他路径，可在配置文件的 `ui.python` 中指定该解释器。算法主程序仍使用原来的运行环境。

在工控机安装系统 Tk、Pillow、YAML 和中文字体（以 Debian/Ubuntu 为例）：

```bash
sudo apt install python3-tk python3-pil python3-pil.imagetk python3-yaml fonts-noto-cjk
```

在可用的图形桌面会话中验证字体后端和字体族：

```bash
/usr/bin/python3 - <<'PY'
import tkinter as tk
from tkinter import font
root = tk.Tk()
print('fontsystem:', root.tk.call('::tk::pkgconfig', 'get', 'fontsystem'))
print('Noto Sans CJK SC:', 'Noto Sans CJK SC' in font.families(root))
root.destroy()
PY
```

期望分别看到 `xft` 和 `True`。若字体后端不是 Xft，界面顶部会显示黄色提示，并向界面进程日志记录 WARNING。

从项目根目录直接预览：

```bash
PYTHONPATH=src /usr/bin/python3 -m nest_clear.ui.demo --root /tmp/nest-clear-demo --page camcal
```

无人值守截图可增加 `--screenshot reports/ui/camcal.png --screenshot-after-ms 3000`；运行命令的环境需要可用的 `DISPLAY` 和访问权限。

相机标定页可打开实时预览，用模组坐标、板中心偏移和清晰度确定起始位置；标定会话中按钮禁用。界面向 `<root>/state/ui_command.json` 原子写入 `preview`、递增 `seq` 和 ISO `time`；主程序在 `main_status.json` 的 `preview` 回显处理结果，并将图像写入 `live/preview_latest.jpg`。主程序状态超过 3 秒未刷新时按钮不可用。`ui_preview.interval_s` 可设置采图间隔，默认 0.5 秒；预览与业务共用相机，请求到达时最多等待正在进行的一帧。演示运行 `--page camcal --preview-state on`，会话禁用效果使用 `--preview-state blocked`。
