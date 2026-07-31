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

## Milestone C：TensorRef / lazy relay ✅ 已完成（含 relay 层修复）

### 目标

让 `mm_aggregate` 不 materialize 大 tensor（`visual_embeds` / `audio_embeds`），只传 TensorRef dict（几十字节的 `{__sglang_omni_tensor_ref__, ref_id, shape, dtype, blob_key, blob_metadata}`），由最终消费者 `text_ar` 的 Stage runtime 调用 `materialize_payload_tensor_refs` 还原成真实 tensor。

### 设计

#### 机制来源

TensorRef 是 `sglang-omni` 框架已有的通用机制（`pipeline/tensor_ref.py`），Qwen3-Omni 和 Ming-Omni 已在生产中使用。设计意图：

> *"Lazy tensor handoff for large multimodal tensors crossing pipeline stages. Intermediate stages that only forward the value (e.g. mm_aggregate) never materialize it; only the declared consumer_stage resolves it back into a real tensor."*

#### 配置方式

通过 `PipelineConfig.env_defaults` 注入环境变量，框架 `mp_runner` 在 spawn 子进程前自动设置：

```python
env_defaults: dict[str, str] = Field(
    default_factory=lambda: {
        "SGLANG_OMNI_ENABLE_TENSOR_REFS": "1",
        "SGLANG_OMNI_TENSOR_REF_EDGES": (
            "image_encoder:mm_aggregate:text_ar,"
            "audio_encoder:mm_aggregate:text_ar"
        ),
        "SGLANG_OMNI_TENSOR_REF_PATHS": "visual_embeds,audio_embeds",
    }
)
```

- `SGLANG_OMNI_ENABLE_TENSOR_REFS=1`：全局开关
- `SGLANG_OMNI_TENSOR_REF_EDGES`：边格式 `from_stage:to_stage:consumer_stage`。encoder→mm_aggregate 的 hop 上创建 ref，consumer 设为 text_ar
- `SGLANG_OMNI_TENSOR_REF_PATHS`：tensor leaf name 白名单，`visual_embeds,audio_embeds`

#### 数据流

```
image_encoder (GPU 0)
  visual_embeds [N, 3072]
  ↓ write_blob → relay blob (SHM, 一次性写入)
  ↓ 生成 TensorRef dict {blob_key, shm_name, shape, dtype}
  ↓ write_payload → payload SHM（不含 visual_embeds，只含 ref dict + 其他小 tensor）
  ↓ relay → mm_aggregate
mm_aggregate (CPU)
  收到 payload → read_payload → tensor_dict 不含 visual_embeds
  Stage runtime: materialize_payload_tensor_refs(current_stage="mm_aggregate")
    → ref.consumer_stage="text_ar" ≠ "mm_aggregate" → 跳过，ref 保持为 dict
  merge_fn: longcat_mm_inputs["image_embeds"] = <TensorRef dict>
  ↓ write_payload → relay → text_ar
text_ar (GPU 2-5)
  Stage runtime: materialize_payload_tensor_refs(current_stage="text_ar")
    → ref.consumer_stage="text_ar" == "text_ar" → read_blob → SHM → 还原为 GPU tensor
  ↓ OmniScheduler inbox 收到的是真实 tensor
```

#### merge.py 适配

```python
# merge.py — 参照 Qwen3-Omni 的 merge.py 模式
from sglang_omni.pipeline.tensor_ref import is_tensor_ref_dict, tensor_ref_numel

def _non_empty(value: Any) -> bool:
    """True when *value* is a non-empty tensor or TensorRef."""
    if value is None:
        return False
    if is_tensor_ref_dict(value):
        return tensor_ref_numel(value) > 0  # 不需要 materialize 就能判断
    if hasattr(value, "numel"):
        return value.numel() > 0
    return bool(value)
```

`is not None` 对 TensorRef dict 同样成立（dict 是 truthy），`_non_empty` 额外提供空 ref 的安全判断。

### 问题与修复：SHM blob 提前销毁

#### 现象

```
FileNotFoundError: [Errno 2] No such file or directory: '/psm_b4692304'
RuntimeError: SHM block psm_b4692304 not found.
```

text_ar 的 `materialize_payload_tensor_refs` → `read_blob` → `ShmGetOperation.wait_for_completion` 尝试通过 `shm_name` 打开 SHM block 时，block 已不存在。

#### 根因

SHM relay 的 blob 生命周期中，sender 进程仅持有 **一个** `SharedMemory` handle（`create=True`）。当该 handle 的 fd 被关闭时（无论是显式 `close()` 还是 GC 触发 `__del__`），Python 3.12 的 `resource_tracker` 检测到 sender 进程中对这个 SHM 的最后一个引用消失，误判为「进程不再需要」，触发 `shm_unlink()`。此时 consumer（text_ar）尚未调用 `shm_open`，SHM 已不存在。

TensorRef 路径中 `_track_background_op` 创建后台 task 调用 `wait_for_completion`。task 完成后 `ShmPutOperation` 无引用 → GC → `SharedMemory.__del__` → `close()`。GC 时机不可控，若发生在 consumer 读取之前，SHM 被提前 unlink。

#### 修复（最终版本）：双 handle 保活

**方案演进**：

- **v1**（初次尝试）：去掉 `wait_for_completion` 中的显式 `close()`。**不足**：GC 触发 `__del__` 同样会 close → unlink，只是时机随机。
- **v2**（最终）：`shm_create_from_tensor` 返回双 handle —— `creator`（`create=True`）和 `keeper`（`SharedMemory(name=shm.name)`）。`wait_for_completion` 关闭 creator 释放写 fd，keeper 存活在 `ShmPutOperation` 中，resource tracker 始终看到至少一个活跃引用，不会 unlink。consumer 读取+unlink 后，`ShmPutOperation` 被 GC → `keeper.__del__` → `close()` 仅清理已消失的 SHM 残留。

```python
# sglang_omni/relay/shm.py

def shm_create_from_tensor(tensor: torch.Tensor) -> tuple[SharedMemory, SharedMemory]:
    shm = SharedMemory(create=True, size=size)
    keeper = SharedMemory(name=shm.name)  # 第二个 handle 防止 tracker 提前 unlink
    # ... 写入数据 ...
    return shm, keeper

class ShmPutOperation:
    def __init__(self, metadata, shm_obj, keeper=None):
        self._shm_obj = shm_obj   # creator
        self._keeper = keeper     # 保活 handle

    async def wait_for_completion(self, timeout=30.0):
        # 关 creator fd，keeper 拦住 tracker 的 unlink
        if not self._completed:
            self._shm_obj.close()
            self._completed = True
```

**生命周期**：

```
image_encoder (sender 进程):
  shm_create_from_tensor → (creator, keeper)
  → publish_tensor_ref → ShmPutOperation(creator, keeper)
  → _track_background_op → wait_for_completion → creator.close()
  → keeper 仍存活（tracker 看到 1 个活跃引用）→ 不会 unlink

mm_aggregate:
  透传 TensorRef dict（不解包 SHM）

text_ar (consumer 进程):
  materialize_payload_tensor_refs → read_blob
  → ShmGetOperation.wait_for_completion
  → SharedMemory(name=shm_name) → 打开成功
  → 读取 → close() + unlink()

sender 进程 ShmPutOperation GC:
  → keeper.__del__ → close() → 只清理已 unlink 的残留 fd
```

## Milestone D：Overlap schedule ✅ 已完成

### 改动

`stages.py` 中 `create_longcat_next_text_executor`：

```python
# 改前
disable_overlap_schedule=True,

# 改后
disable_overlap_schedule=False,
```

### overlap schedule 是什么

SGLang 的 overlap schedule 让 CPU 的 batch 调度和 GPU 的 forward 执行重叠：

```
# 不开启（串行）
CPU: schedule → [等 GPU] → process → schedule → ...
GPU: [等 CPU] → forward    → [等 CPU] → forward → ...

# 开启（overlap）
CPU: schedule_1 → schedule_2 → process_1 → schedule_3 → ...
GPU: forward_1 ───────────→ forward_2 ───────────→ ...
```

decode 阶段每个 token 的 forward 很快（几 ms），调度延迟占比高，overlap 收益最明显。

### 安全适配：ForwardBatch 防残留

overlap 模式下 SGLang 可能池化复用 `ForwardBatch` 对象。如果不做清理，上一轮 prefill batch 的 `longcat_replace_embeds` 会残留到下一轮 decode batch，decode 时 `model.forward` 读到非 None 的旧值就会错误进入多模态路径。

修复（`model_runner.py`）：

```python
def before_prefill(self, forward_batch, schedule_batch, requests):
    del requests
    if not schedule_batch.forward_mode.is_extend():
        # overlap schedule 下 ForwardBatch 可能被池化复用，
        # 必须显式清空以避免 prefill 残留泄漏到 decode batch
        forward_batch.longcat_replace_embeds = None
        forward_batch.longcat_replace_positions = None
        return
    ...
```

EXTEND 路径在 `_attach_multimodal_replacements` 末尾也会显式设置（有值时设为 cat 结果，无值时 `else` 分支设为 None），两端都覆盖。

### 与后续 CUDA Graph 的关系

当前 `disable_cuda_graph=True`，只测 overlap。CUDA Graph 在 Milestone E 独立验证后再叠加。

---

## Milestone E：CUDA Graph / async decode 🔜 待开发

（见 `开发文档_phase2.5_milestones.md` Step 4-5）

---

## Debug 记录

### Debug-004：TensorRef SHM blob 生命周期问题 ✅ 已修复

- **时间**：2026-07-31
- **环境**：H200，`SGLANG_OMNI_ENABLE_TENSOR_REFS=1`
- **现象**：text_ar 在 `materialize_payload_tensor_refs` → `read_blob` 时报 `SHM block psm_b4692304 not found`
- **根因**：三个独立子问题叠加 —— SHM GC 竞态、TP 多 rank 竞争、merge 重复引用。

#### 子问题 1：SHM GC 竞态

sender 进程仅持有一个 `SharedMemory` handle。`_track_background_op` task 完成后 `ShmPutOperation` 无引用 → GC → `SharedMemory.__del__` → `close()` → Python 3.12 resource tracker 检测到 sender 进程中最后一个引用消失，预判「进程不再需要」，`shm_unlink()`。此时 consumer 尚未 `shm_open`，SHM 已不存在。

**修复**：双 handle 保活 + 注册表防 GC。

```python
# shm_create_from_tensor 返回 (creator, keeper)
shm = SharedMemory(create=True, size=size)       # creator
keeper = SharedMemory(name=shm.name)             # 第二个 handle 防止 tracker unlink

# _PENDING_PUTS 注册表阻止 ShmPutOperation 被 GC
_PENDING_PUTS: dict[str, ShmPutOperation] = {}
```

- creator：用于写入数据，`wait_for_completion` 中关闭释放写 fd
- keeper：第二个 `shm_open`，存活在 `ShmPutOperation` 中。resource tracker 始终看到至少一个活跃引用，不会 unlink
- `_PENDING_PUTS`：模块级 dict，持有 `ShmPutOperation` 强引用，阻止 GC 回收 `keeper`

#### 子问题 2：TP 多 rank 竞争

text_ar 是 TP=4 的 stage，4 个 rank 的 `_execute` 各自调用 `materialize_payload_tensor_refs`。所有 rank 指向同一个 SHM block。rank 0 先读并 unlink，rank 1/2/3 再读时 SHM 已消失。

**修复**：`stage/runtime.py` 中只有 `role="leader"` 的 rank 执行 materialize。follower rank 通过 OmniScheduler 的 TP broadcast 接收已解析的 payload。

```python
if tensor_refs_enabled():
    if self.role == "leader":
        payload_for_scheduler = await relay_io.materialize_payload_tensor_refs(...)
```

#### 子问题 3：merge 重复引用

`merge.py` 中 `merge_for_text_ar` 把同一个 tensor_ref dict 同时放进 `encoder_outs` 和 `text_ar_inputs.longcat_mm_inputs`。`materialize_tensor_refs` 解析 `encoder_outs` 中的 ref 时创建新 dict（SHM 被 read + unlink），但 `text_ar_inputs` 中的旧 ref dict 未更新。第二次遇到同一个 ref 时 SHM 已不存在。

**修复**：`merge.py` 中不向 `encoder_outs` 传递 encoder output（text_ar 只消费 `text_ar_inputs`，`encoder_outs` 在合并后 payload 中无用）。

#### 验证

H200 端到端测试：纯文本、图片、音频、图片+音频四种请求全部通过。tokens/s 从 0.5 提升至 50~1000。

### Debug-005：Overlap schedule 启用 + ForwardBatch 防残留

- **时间**：2026-07-31
- **环境**：H200，`disable_overlap_schedule=False`
- **现象**：启用 overlap 后需要确保 decode batch 不会读到 prefill 残留的 `longcat_replace_embeds`
- **修复**：`model_runner.py` `before_prefill` 中 decode 分支显式设置 `forward_batch.longcat_replace_embeds = None` 和 `longcat_replace_positions = None`
- **验证**：静态分析通过。EXTEND 路径在 `_attach_multimodal_replacements` 末尾 `else` 分支也显式设为 None，两端覆盖无遗漏

---

## 附录：API 请求格式

### 纯文本

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/path/to/LongCat-Next",
    "messages": [{"role": "user", "content": "Hello, who are you?"}],
    "max_tokens": 1000
  }'
```

### 图片

图片通过顶级 `images` 字段传入（本地文件路径列表），prompt 中描述图片内容：

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/path/to/LongCat-Next",
    "messages": [{"role": "user", "content": "Describe this image."}],
    "images": ["/absolute/path/to/cars.jpg"],
    "max_tokens": 1000
  }'
```

### 音频

音频通过顶级 `audios` 字段传入（本地文件路径列表）：

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/path/to/LongCat-Next",
    "messages": [{"role": "user", "content": "What do you hear?"}],
    "audios": ["/absolute/path/to/audio.wav"],
    "max_tokens": 1000
  }'
```

### 图片 + 音频混合

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/path/to/LongCat-Next",
    "messages": [{"role": "user", "content": "Describe the image and audio together."}],
    "images": ["/absolute/path/to/cars.jpg"],
    "audios": ["/absolute/path/to/cough.wav"],
    "max_tokens": 1000
  }'
```

### 关键说明

- `images` 和 `audios` 是 **ChatCompletionRequest 的顶级字段**，不是消息 content 中的嵌套类型
- Content 中的 `image_url` / `audio_url` 类型仅参与 prompt 文本拼接，**不会触发实际的媒体编解码**
- 路径必须是服务进程可访问的本地文件路径
- 多图/多音频在数组中列出即可，各文件分别处理

---

## 附录：CUDA 12.9 兼容性适配（gpu_compat.py）

### 问题 1：kernels HF hub cu129 回退

`transformers==5.6.0` 通过 `kernels` 包从 HuggingFace Hub 拉取预编译的 flash-attn2 kernel，但 `kernels-community/flash-attn2` 仓库没有 `torch211-cxx11-cu129` 构建。CUDA 12.8 kernel 在 12.9 运行时上二进制兼容。

**修复**：`_patch_kernels_cu129_fallback()` monkey-patch `kernels.utils.build_variants()`，为每个 cu129 variant 额外生成 cu128 回退项。

### 问题 2：pip 传递依赖 `kernels` 版本漂移

`sglang==0.5.12.post1` 对 `kernels` 无版本约束，`transformers==5.6.0` 需要 `kernels<0.13`。`uv sync` 时解析器安装了 `kernels==0.16.0`，其中 `LayerRepository.__init__` 新增了 `revision/version` 必填参数，导致 `hub_kernels.py` 在 import 时崩溃。

**修复**：`pyproject.toml` `override-dependencies` 中添加 `kernels<0.13`。

### 问题 3：`sgl-deep-gemm` CUDA 13→12.9 兼容

`sglang==0.5.12.post1` 的硬依赖 `sgl-deep-gemm==0.1.0` 的 PyPI 版本 `_C.so` 链接 `libcudart.so.13`，在 CUDA 12.9 系统上无法加载。

**修复**：`pyproject.toml` 中指定 `sgl-deep-gemm = { index = "sglang-cu129" }` 从 `https://docs.sglang.ai/whl/cu129` 获取 `0.1.0+cu129` 版本，链接 CUDA 12。

### 问题 4：transformers 5.6.0 flash_attention_forward s_aux=None

Qwen2.5-VL 视觉模型不使用 sliding window attention，`flash_attention_forward` 的 `s_aux` 参数为 None，但 transformers 5.6.0 未做 None 检查直接调用 `.to()`。

**修复**：`_patch_flash_attn_s_aux_none()` monkey-patch，将 None 替换为空的 dummy tensor。
