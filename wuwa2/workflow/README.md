# 合成工作流

在 `wuwa2/` 目录运行。依赖 Python、FFmpeg 和 `requirements.txt` 中的库。

- `new-角色清单.json`：角色输入、音轨和输出设置。
- `new_build.py`：读取素材，重建角色配置；`--render --assemble` 同时生成角色片段、逐帧检查和整片。
- `render.py`：逐帧合成角色分层；`--validate` 只检查配置和素材。
- `loop_transition.py`：声痕、虹膜牙纹及首尾转场。
- `new_music.py` / `new_timing.py`：准备循环音轨并按音乐节拍安排镜头。
- `new_check.py`：检查整片、音画和首尾循环，输出逐帧预览与两遍音频试听；需先生成角色检查帧。

当前配置为 `new-八角色-音频循环修正版.json`。从已提供素材直接合成：

```bash
python workflow/render.py workflow/new-八角色-音频循环修正版.json --validate
python workflow/render.py workflow/new-八角色-音频循环修正版.json --output output.mp4
```

在 Windows 上完整重建并检查：

```bash
python workflow/new_build.py --render --assemble
python workflow/new_check.py
python -m unittest discover -s workflow -p test_*.py
```

部分预览和检查使用 Windows 字体文件路径。素材中的第三方内容与代码许可分开，详见上级 README。
