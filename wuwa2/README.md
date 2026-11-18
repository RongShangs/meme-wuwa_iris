# wuwa2：鸣潮角色虹膜转场

《鸣潮》角色虹膜转场短片的合成工程。包含八个角色的分层图像、Q 版角色素材、动画与音轨脚本、当前配置和导出的视频。角色顺序：菲比、弗洛洛、达尼亚、今汐、爱弥斯、赞妮、椿、卡卡罗。

[观看当前成片](deliverables/new-八角色-音频循环修正版/new-八角色-音频循环修正版.mp4) · [两遍循环音频试听](deliverables/new-八角色-音频循环修正版/首尾循环试听-两遍.wav) · [工作流](workflow/README.md)

视频为 1080 × 1576、30 fps、277 帧，约 9.23 秒。片头声痕保持形状缩小，接近最终大小时变成菲比的瞳孔；片尾回到声痕。音乐使用同一首曲目中按约 130 BPM 对齐的片段，首尾做了 120 毫秒交接。

## 使用

需要 Python 3.10+，以及已加入 PATH 的 `ffmpeg` 和 `ffprobe`。在 `wuwa2/` 目录运行：

```bash
python -m pip install -r workflow/requirements.txt
python workflow/render.py workflow/new-八角色-音频循环修正版.json --validate
python workflow/render.py workflow/new-八角色-音频循环修正版.json --output output.mp4
```

要从角色素材重新生成配置、分段和成片，在 Windows 运行 `python workflow/new_build.py --render --assemble`，之后可运行 `python workflow/new_check.py` 生成检查图和两遍循环试听。角色逐帧检查和总览目前使用 Windows 自带字体路径。直接运行 `python workflow/new_build.py` 可只重建各角色配置和循环音轨。

## 目录

- `workflow/`：渲染器、转场、音乐与节拍处理、项目清单和测试。
- `角色/`：角色的分层素材、Q 版素材和配置。
- `素材/`、`assets/`、`new/`：原始参考、音轨和工程使用的输入素材。
- `deliverables/`：当前导出的视频与试听文件。
- `research/`、`tools/`：制作研究和辅助脚本。

为便于重建，保留了素材及部分早期配置；大量重复的历史视频和检查图没有放入此目录，重新渲染会生成新的分段和检查图。

## 许可与素材

项目代码采用 [MIT License](LICENSE)。角色、游戏画面、音乐、参考视频、参考图片及其衍生内容可能涉及第三方权利；代码许可不授予这些素材的使用权。项目为非官方粉丝创作，再利用素材前请确认相应授权。
