# 标定文件迁移

此次部署会使 Git 从工作树移除原先跟踪的四个文件：`camera.yaml`、`tools.yaml`、`tool1_baseline.yaml`、`tool2_baseline.yaml`。设备上的文件可能包含现场标定结果；必须先完整备份并核对，再更新。以下命令在设备上执行，备份目录必须位于仓库之外，且不得覆盖已有备份。

1. 停止设备程序，备份整个目录并记录所有文件的 SHA-256：

   ```sh
   backup="$HOME/nest-clear-calibration-backup-$(date +%Y%m%d-%H%M%S)"
   mkdir "$backup"
   cp -a "$HOME/nest-clear/config/calibration" "$backup/calibration"
   (cd "$backup/calibration" && find . -type f -print0 | sort -z | xargs -0 -r sha256sum) > "$backup/SHA256SUMS"
   ```

2. 确认备份及校验文件完整且可读后，在仓库中仅对 Git 显示已修改的上述四个文件执行 `git checkout -- config/calibration/<文件名>`。此操作会丢弃工作树中的对应内容，只为允许快进，**绝不能在备份前执行**。
3. 执行 `git merge --ff-only <目标分支>`。
4. 将备份目录中的全部内容原样复制回 `config/calibration/`，包括四个文件、其他标定文件和 `history/`；保留新版本的 `examples/`：

   ```sh
   cd "$HOME/nest-clear"
   find "$backup/calibration" -mindepth 1 -maxdepth 1 ! -name examples -exec cp -a -t config/calibration/ -- {} +
   ```

5. 逐文件核对备份的 SHA-256，全部必须一致，然后检查 `git status --short`：这些现场标定文件应被忽略，不再出现。

   ```sh
   (cd config/calibration && sha256sum -c "$backup/SHA256SUMS")
   git status --short
   ```

缺失的四个文件会在启动时从 `examples/` 复制占位模板并输出警告；已有文件不会被引导程序覆盖。
