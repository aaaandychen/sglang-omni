# LongCat-Next Phase 2.5 详细设计

## Milestone A：可观测打点 ✅ 已完成

### 设计

环境变量 `SGLANG_OMNI_LONGCAT_DEBUG_TIMING=1` 控制全局开关。关闭时零开销（context manager 直接 yield，不记录时间）。

核心实现在 `payload_types.py`：

- `longcat_timing(event, **metadata)` — context manager，自动记录 `elapsed_ms`
- `longcat_log_timing(event, **metadata)` — 点状日志，不记录耗时

Logger 名称：`sglang_omni.longcat_next.timing`，INFO 级别。所有 stage worker 进程在 `stage_process_main` 中执行 `logging.basicConfig(level=INFO, stream=stdout)`（`pipeline/stage_workers.py:365`），timing logger 向上传播到 root，自动输出。

### 覆盖范围

**启动链路**：

| 事件 | 位置 |
|---|---|
| `preprocessing_executor_init` | `stages.py:create_preprocessing_executor` |
| `preprocessor_config_init` | `preprocessor.py:__init__` |
| `preprocessor_processor_init` | `preprocessor.py:__init__` |
| `image_encoder_executor_init` | `stages.py:create_image_encoder_executor` |
| `image_encoder_config_init` | `encoders.py:LongcatNextImageEncoder.__init__` |
| `image_encoder_remote_class_init` | `encoders.py:LongcatNextImageEncoder.__init__` |
| `codebook_embedding_load` | `encoders.py:_OffsetCodebookEmbedding.__init__` |
| `image_encoder_load_module` | `encoders.py:LongcatNextImageEncoder.__init__` |
| `audio_encoder_executor_init` | `stages.py:create_audio_encoder_executor` |
| `audio_encoder_config_init` | `encoders.py:LongcatNextAudioEncoder.__init__` |
| `audio_encoder_remote_class_init` | `encoders.py:LongcatNextAudioEncoder.__init__` |
| `audio_encoder_load_module` | `encoders.py:LongcatNextAudioEncoder.__init__` |
| `text_ar_tokenizer_init` | `stages.py:create_longcat_next_text_executor` |
| `text_ar_server_args_build` | `stages.py:create_longcat_next_text_executor` |
| `text_ar_infrastructure_init` | `stages.py:create_longcat_next_text_executor` |
| `text_ar_cuda_graph_init` | `stages.py:create_longcat_next_text_executor` |
| `text_ar_output_processor_init` | `stages.py:create_longcat_next_text_executor` |
| `text_ar_request_adapters_init` | `stages.py:create_longcat_next_text_executor` |

**请求链路**：

| 事件 | 位置 |
|---|---|
| `preprocessing_request` | `stages.py:_preprocess` |
| `preprocessor_build_text` | `preprocessor.py:__call__` |
| `preprocessor_processor_call` | `preprocessor.py:__call__` |
| `image_encoder_request` | `stages.py:_encode` |
| `image_encoder_h2d` | `encoders.py:LongcatNextImageEncoder.forward` |
| `image_encoder_tokenize` | `encoders.py:LongcatNextImageEncoder.forward` |
| `image_encoder_codebook_embedding` | `encoders.py:LongcatNextImageEncoder.forward` |
| `image_encoder_projection` | `encoders.py:LongcatNextImageEncoder.forward` |
| `audio_encoder_request` | `stages.py:_encode` |
| `audio_encoder_h2d` | `encoders.py:LongcatNextAudioEncoder.forward` |
| `audio_encoder_tokenize` | `encoders.py:LongcatNextAudioEncoder.forward` |
| `audio_encoder_codebook_embedding` | `encoders.py:LongcatNextAudioEncoder.forward` |
| `mm_aggregate_request` | `stages.py:_identity` |
| `merge_for_text_ar` | `merge.py:merge_for_text_ar` |
| `text_ar_request_build` | `request_builders.py:request_builder` |
| `text_ar_execute` (phase=prefill/decode, batch_size) | `model_runner.py:execute` |
| `text_ar_before_prefill_mm_injection` | `model_runner.py:before_prefill` |
| `build_input_embeds_pad_zero` | `sglang_model.py:_build_longcat_input_embeds` |
| `build_input_embeds_ngram` | `sglang_model.py:_build_longcat_input_embeds` |
| `build_input_embeds_scatter` | `sglang_model.py:_build_longcat_input_embeds` |
| `text_ar_mm_replacements_attached` | `model_runner.py:_attach_multimodal_replacements` |
| `encoder_cache` (stage, action, cache_key) | `stages.py:_encode` |

---

## Milestone B：P0 Encoder exact-match cache ✅ 已完成

### 概述

同一图片/音频被不同 prompt 重复请求时，跳过 GPU encoder 前向计算，直接复用缓存的 encoder 输出。

设计原则：

- 复用仓库已有 `StageOutputCache`（`scheduling/stage_cache.py`），不造新轮子
- 缓存逻辑在 executor 层（`stages.py`），encoder model（`encoders.py`）保持纯计算
- cache_key 在 preprocessor 层生成，通过 `encoder_inputs` dict 的 `"cache_key"` 字段随 pipeline state 传递
- 全部日志受 `SGLANG_OMNI_LONGCAT_DEBUG_TIMING` 控制，不新增独立的 env var

### 数据流

```
客户端请求
  → preprocessor.__call__
      _extract_media_paths(inputs) → ["/path/img.jpg"], []
      _build_media_cache_key(paths) → "/path/img.jpg:12345:1722000000000000000"
      ── 注入 encoder_inputs["image_encoder"]["cache_key"] ──
  → relay
  → image_encoder executor._encode
      cache_key = inputs.get("cache_key")
      cache.get(cache_key) → hit? 返回 cached → 跳过 model()
      miss? model() 执行 → cache.put(cache_key, result)
  → relay
  → mm_aggregate → merge_for_text_ar（不感知 cache）
  → text_ar（不感知 cache）
```

### 改动文件

#### `components/preprocessor.py`

**新增 `_extract_media_paths`**（第 146-160 行）：从 `inputs["images"]` / `inputs["audios"]` 提取字符串路径列表。非字符串（bytes、base64、URL）跳过，返回空列表 → cache_key 为 None → 不走缓存。

**新增 `_build_media_cache_key`**（第 163-180 行）：对每个路径调用 `os.stat`，取 `st_size` + `st_mtime_ns` 拼接。任何 `OSError` 返回 None（文件不存在/不可读 → 安全跳过缓存）。

**修改 `__call__`**（第 41-42、84-89、97-103 行）：在 processor 调用前提取 media paths，在构建 `encoder_inputs` 时注入 `"cache_key"`。

#### `stages.py`

**新增模块级缓存常量**（第 11-12 行）：
- `_ENCODER_CACHE_MAX_SIZE = 256`（最多 256 个不同输入）
- `_ENCODER_CACHE_MAX_BYTES = 1024 * 1024 * 1024`（1 GiB 显存上限）

**修改 `create_image_encoder_executor`**（第 59-63 行创建 cache，第 73-110 行缓存逻辑）：
- Model 初始化后创建 `StageOutputCache(max_size=256, max_bytes=1GiB, cache_device=None)`
- `cache_device=None`：cached tensor 保留在 encoder GPU 上，hit 时零拷贝
- `_encode` 中：先查缓存 → hit 直接返回 → miss 执行 model → store 结果

**修改 `create_audio_encoder_executor`**（第 133-137 行创建 cache，第 147-185 行缓存逻辑）：同上。

### Cache key 策略

| 场景 | cache_key | 行为 |
|---|---|---|
| 本地文件路径 | `path:size:mtime_ns` | 同一文件内容不变 → key 不变 → hit |
| 文件被覆盖 | `path:new_size:new_mtime_ns` | key 变化 → miss（正确） |
| 多图/多音频 | 各 key 用 `\|` 拼接 | 任何一张图变了整体 key 不同 |
| 非文件输入（bytes/URL） | `None` | 不走缓存 |
| 文件不存在 | `None`（OSError） | 不走缓存 |
| 纯文本请求 | encoder_inputs 为空 | 走 `if not inputs: return {}` 分支 |

### 日志示例

首次请求（`SGLANG_OMNI_LONGCAT_DEBUG_TIMING=1`）：

```
longcat_timing event=encoder_cache stage=image_encoder action=miss cache_key=...
longcat_timing event=image_encoder_h2d ...
longcat_timing event=image_encoder_tokenize ...
longcat_timing event=image_encoder_codebook_embedding ...
longcat_timing event=image_encoder_projection ...
longcat_timing event=encoder_cache stage=image_encoder action=store cache_key=...
```

第二次同图请求：

```
longcat_timing event=encoder_cache stage=image_encoder action=hit cache_key=...
longcat_timing event=image_encoder_request elapsed_ms=0.12
```

hit 时所有 encoder 子步骤（h2d/tokenize/codebook_embedding/projection）不会出现，因为 `model()` 根本没调用。

### 边界情况

| 场景 | 行为 |
|---|---|
| 多图请求 | cache key = 各图 key 用 `\|` 拼接，部分图变化 → 整体 key 不同 → 全部重新计算 |
| 超过 1GB 上限 | `StageOutputCache._evict_over_budget` 自动 LRU 淘汰，`eviction_count` 递增 |
| 不同 prompt 引用同一图片 | key 相同 → hit |
| 生产环境不设 `DEBUG_TIMING` | 日志完全静默，`longcat_log_timing` 在 env var 关闭时直接 return |
| cache_key 为 None | 跳过缓存逻辑，每次都正常执行 encoder |
| `cache_device=None` | 缓存 tensor 保留在原 device，hit 时 relay 仍会序列化，但省了 encoder forward |
| 进程重启 | 缓存丢失，首次请求全部 miss（预期行为） |

### 与后续优化的关系

- **TensorRef（P1）**：cache hit 后，cached tensor 通过 TensorRef lazy relay 到 text_ar，减少中间序列化开销。缓存逻辑本身不感知 TensorRef。
- **RadixCache 联动**：第一版不做。后续可在 `merge_for_text_ar` 中注入 `media_cache_keys`，让 RadixCache 对同媒体 prefix 更友好。但 LongCat 的 NgramEmbedding 使 placeholder token 处理更敏感，需要独立验证。

---

## Milestone C：TensorRef / lazy relay 🔜 待开发

（见 `开发文档_phase2.5_milestones.md` Step 2）

---

## Milestone D：Overlap schedule 🔜 待开发

（见 `开发文档_phase2.5_milestones.md` Step 3）

---

## Milestone E：CUDA Graph / async decode 🔜 待开发

（见 `开发文档_phase2.5_milestones.md` Step 4-5）
