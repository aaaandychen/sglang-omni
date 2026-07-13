# 本地开发环境配置

## 背景

机器驱动为 CUDA 12.9（`nvidia-smi` 显示 560.35.03），不支持 sglang 0.5.11+ 的 CUDA 13.0 编译依赖。本仓库当前源码基于 sglang 0.5.12.post1，保留了所有 API 适配代码不改动，仅通过降级依赖栈适配 CUDA 12.9 驱动。

## 环境隔离

- 所有 Python 操作在本目录 `.venv` 内，不污染系统 Python。
- 不触碰 `/usr/`、`/sgl-workspace/` 及 `/mnt/cephfs/chenzhenyang/` 以外任何路径。

## 创建和安装

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
UV_INDEX_URL=https://pypi.corp.kuaishou.com/kuaishou/prod/+simple/ uv pip install -e . --prerelease=allow
```

## 与上游 pyproject.toml 的差异

依赖栈从 CUDA 13.0 降级到 CUDA 12.9，具体变更见 `pyproject.toml`。核心变化：

- **sglang**: 0.5.12.post1 → 0.5.10.post1（最后一个 CUDA 12.9 版本）
- **torch**: 2.11.0+cu130 → 2.9.1+cu128
- **nixl/mooncake**: cu13 专用 → cu12/通用
- **transformers**: 5.6.0 → 5.3.0
- **kernels**: 移除（sglang 0.5.10 内置 sglang-kernel）

其余版本以 `pyproject.toml` 中实际声明的为准。

## 已知风险

源码内的 sglang 0.5.12.post1 API 适配代码未回退，涉及 ~47 个文件（主要是各模型实现层）。启动特定模型时可能遇到 API 不兼容错误，需逐案修复。核心 multi-stage 架构的变更仅集中在 `model_runner/sglang_model_runner.py` 和 `scheduling/dllm_scheduler.py`。

## 验证

```bash
source .venv/bin/activate
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"  # True 8
python -c "import sglang; print(sglang.__version__)"  # 0.5.10.post1
sgl-omni --help
```
