# LongCat-Next Multimodal Pipeline Debug Notes

## Environment

| Machine | OS | gcc | Python | CUDA Driver | GPU |
|---------|-----|-----|--------|-------------|-----|
| A (最初的) | Ubuntu 20.04 | 9.4.0 | 3.10 (system) / 3.12 (uv) | 595.58.03 (CUDA 13.2) | 8x H200 |
| B (SFT 机器) | Ubuntu 24.04 | 13.3.0 | 3.12.3 (/opt/venv) | — | 8x H200 |

`.venv` 在 CephFS 上共享。

---

## 问题清单

### 1. .venv symlink 断裂

- **现象**: `.venv/bin/python → /usr/bin/python3.12` 不存在
- **原因**: `.venv` 由旧版 uv 用系统 Python 创建，系统 Python 被移除后 symlink 断裂
- **修复**: 用 `uv python install 3.12` 下载 uv 托管的 Python，重新指向

### 2. flash-attn v4 API 迁移

#### 2a. 顶层函数路径变动

- **现象**: `ImportError: cannot import name 'flash_attn_varlen_func' from 'flash_attn'`
- **根因**: `flash-attn-4` (4.0.0b15) 是 CUTE 重写版，`flash_attn` 变为 namespace package，`flash_attn_func` / `flash_attn_varlen_func` 移到了 `flash_attn.cute`
- **修复文件**: `sglang_omni/models/longcat_next/components/dynamic.py`
- **修复方式**: 在模块顶层将 `flash_attn.cute` 的函数注入 `flash_attn` 命名空间

#### 2b. bert_padding 模块移除

- **现象**: `ModuleNotFoundError: No module named 'flash_attn.bert_padding'`
- **根因**: `flash_attn.bert_padding` 在 v4 中移除，`index_first_axis` / `pad_input` / `unpad_input` 移到 `flash_attn.cute.testing`
- **修复文件**: 同上 `dynamic.py`
- **修复方式**: 创建 `flash_attn.bert_padding` 伪模块，注册到 `sys.modules`

#### 2c. flash_attn_varlen_func v4 新增 qv 参数

- **现象**: `RuntimeError: cu_seqlens_k must have shape (batch_size + 1,)`
- **根因**: v4 在第 4 个位置插入了 `qv` 参数，LongCat-Next audio model 用 positional args 调用，导致 `cu_len` 被解读为 `qv`
- **修复文件**: `models/LongCat-Next/modular_longcat_next_audio.py`
- **修复方式**: 改用 keyword arguments: `cu_seqlens_q=`, `cu_seqlens_k=`, `max_seqlen_q=`, `max_seqlen_k=`

### 3. transformers 5.6.0 API 迁移

#### 3a. Qwen2RMSNorm 导入路径变更

- **现象**: `ImportError: cannot import name 'Qwen2RMSNorm' from 'transformers.models.qwen2_5_vl.modeling_qwen2_5_vl'`
- **根因**: transformers 5.6.0 将 `Qwen2RMSNorm` 移到了 `transformers.models.qwen2.modeling_qwen2`
- **修复文件**: `models/LongCat-Next/modular_longcat_next_visual.py`
- **修复方式**: 改为 `from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm`

#### 3b. video_processor 类型强制校验

- **现象**: `TypeError: Received a Qwen2VLImageProcessor for argument video_processor, but a BaseVideoProcessor was expected`
- **根因**: transformers 5.6.0 新增 `check_argument_for_proper_class()`，要求 `video_processor` 必须是 `BaseVideoProcessor` 子类
- **修复文件**: `models/LongCat-Next/processing_longcat_next.py`
- **修复方式**: 在 `LongcatNextProcessor` 中 override `check_argument_for_proper_class`，对 `video_processor` 跳过校验

#### 3c. tokenizer.init_kwargs 不再保留自定义 key

- **现象**: `ValueError: text input must be of type 'str'...` → `token_str` 为 None
- **根因**: transformers 5.6.0 的 tokenizer 不再把 `tokenizer_config.json` 中的自定义 key 暴露在 `init_kwargs` 里
- **修复文件**: 同上 `processing_longcat_next.py`
- **修复方式**: `init_kwargs.get()` 返回 None 时，fallback 到直接读取 `tokenizer_config.json`

#### 3d. flash_attention_forward s_aux None 检查缺失

- **现象**: `AttributeError: 'NoneType' object has no attribute 'to'` (image_encoder)
- **根因**: `transformers/integrations/flash_attention.py:84` — `s_aux.to(query.dtype)` 未对 `s_aux=None` 做保护
- **修复文件**: `.venv/lib/python3.12/site-packages/transformers/integrations/flash_attention.py`
- **修复方式**: `s_aux.to(query.dtype)` → `s_aux.to(query.dtype) if s_aux is not None else None`

### 4. Runtime 问题

#### 4a. 空 tensor 在 boolean context 求值

- **现象**: `RuntimeError: Boolean value of Tensor with no values is ambiguous`
- **位置**: `sglang_omni/models/longcat_next/model_runner.py:48`
- **根因**: `getattr(req, "prefix_indices", []) or []` — 空 tensor 不能用于 boolean context
- **修复方式**: 显式检查 `numel() > 0`

#### 4b. Token usage 字段名不兼容

- **现象**: 返回 `prompt_tokens:0, completion_tokens:0, total_tokens:0`
- **位置**: `sglang_omni/models/longcat_next/request_builders.py`
- **根因**: result adapter 用 `output_tokens`，但 OpenAI 协议层期望 `completion_tokens`；`prompt_tokens` 缺失
- **修复方式**: 改用 `completion_tokens`，从 `data.req.origin_input_ids` 提取 `prompt_tokens`

### 5. 环境问题（非代码）

#### 5a. gcc 版本过旧（Ubuntu 20.04）

- **现象**: `fatal error: concepts: No such file or directory`
- **根因**: sglang JIT CUDA kernel 需要 C++20 (`<concepts>`, `<bit>`, `<source_location>`)，gcc 9.4 不支持
- **方案**: 通过 micromamba 在 `.venv/tools/` 安装 gcc-12，`gpu_compat.py` 自动注入 PATH
- **注意**: 此方案已回滚。Ubuntu 24.04 自带 gcc 13 无需处理

#### 5b. Python dev headers 不完整

- **现象**: `fatal error: x86_64-linux-gnu/python3.12/pyconfig.h: No such file or directory`
- **原因**: `python3.12-dev` 装了但架构相关头文件路径异常
- **方案**: `apt install libpython3.12-dev` 或检查 `/usr/include/x86_64-linux-gnu/python3.12/`

#### 5c. CephFS I/O 波动

- **现象**: shard 加载间隙极大（59s / 95s），总体 15 个 shard 耗时 ~2 分钟
- **原因**: 模型文件在 CephFS 网络存储，多用户竞争带宽

---

## 修改文件清单

| 文件 | 改动类型 |
|------|----------|
| `sglang_omni/models/longcat_next/components/dynamic.py` | flash-attn v4 桥接 |
| `sglang_omni/models/longcat_next/model_runner.py` | 空 tensor boolean fix |
| `sglang_omni/models/longcat_next/request_builders.py` | token usage 字段名 |
| `models/LongCat-Next/modular_longcat_next_visual.py` | Qwen2RMSNorm import + flash_attn v4 keyword args + tuple return |
| `models/LongCat-Next/modular_longcat_next_audio.py` | flash_attn v4 keyword args + tuple return |
| `models/LongCat-Next/processing_longcat_next.py` | video_processor + init_kwargs |
| `.venv/lib/python3.12/site-packages/transformers/integrations/flash_attention.py` | s_aux None check |

## E2E 验证

### Case 1: 纯文本

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/mnt/cephfs/chenzhenyang/models/LongCat-Next","messages":[{"role":"user","content":"Hello, who are you?"}],"max_tokens":50}'
```

**结果**:
```json
{
  "choices": [{"message": {"content": "I'm looking for someone to help me navigate..."}}],
  "usage": {"prompt_tokens": 6, "completion_tokens": 50, "total_tokens": 56}
}
```

✅ Pipeline 路径: `preprocessing` → `mm_aggregate` → `text_ar` 正常。

### Case 2: 图片

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/mnt/cephfs/chenzhenyang/models/LongCat-Next","messages":[{"role":"user","content":"Describe this image."}],"images":["/mnt/cephfs/chenzhenyang/czy/sglang-omni/tests/data/cars.jpg"],"max_tokens":100}'
```

**结果**:
```json
{
  "choices": [{"message": {"content": "The image is a composite of two distinct scenes: 1. Left Section ... a close-up shot of a person's hand ... 2. Right Section ... a red, vintage-style convertible car ..."}}],
  "usage": {"prompt_tokens": 4011, "completion_tokens": 269, "total_tokens": 4280}
}
```

✅ Pipeline 路径: `preprocessing` → `image_encoder` → `mm_aggregate` → `text_ar` 正常。`prompt_tokens: 4011` 包含视觉 token，确认图片被正确编码。

### Case 3: 音频

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/mnt/cephfs/chenzhenyang/models/LongCat-Next","messages":[{"role":"user","content":"What is this audio about?"}],"audios":["/mnt/cephfs/chenzhenyang/czy/sglang-omni/tests/data/query_to_cars.wav"],"max_tokens":100}'
```

**结果**:
```json
{
  "choices": [{"message": {"content": "Well, this audio clip includes two different parts. In the first part, a person is asking a question: \"How many cars are there in the picture?\" ..."}}],
  "usage": {"prompt_tokens": 69, "completion_tokens": 118, "total_tokens": 187}
}
```

✅ Pipeline 路径: `preprocessing` → `audio_encoder` → `mm_aggregate` → `text_ar` 正常。模型正确转写了音频内容。
```
