# LongCat-Next Phase 2：多模态 Encoder 接入

## 概述

Phase 1 实现了纯文本 AR backbone。Phase 2 的目标是接入图像和音频 encoder，使模型能够理解多模态输入。

核心思路：新增独立的 encoder stage（图像和音频各一个），encoder 输出的 embedding 通过 `replace_embeds` 机制注入到 AR backbone 的指定 token 位置。架构从单 stage 扩展为多 stage fan-out + fan-in 拓扑。

---

## 步骤一：`replace_embeds` — 多模态 embedding 注入机制

### 1.1 为什么需要这个机制

Phase 1 的文本生成链路中，SGLang 内部自动完成 embedding：

```
input_ids → NgramEmbedding(word embedding + 12个ngram投影平均) → embeddings → Transformer
```

外部不干预，每个 token 的 embedding 都由 NgramEmbedding 查表产生。

加入图像后，输入序列变成：

```
["你好", "描述", "这张图", <img_pad>, <img_pad>, ..., "请", "回答"]
```

`<img_pad>` 是图像内容的占位 token。它需要的 embedding 不在 NgramEmbedding 表里，而是由外部的 VisualEmbeddingBridge 产生。

VisualEmbeddingBridge 的工作方式：图像经过 ViT 编码、残差 VQ 量化后得到离散 visual token IDs（8 个 codebook 各一个 id），8 个 Embedding 层分别查表求和，再经过一个 DecoderLayer（MLP + LayerNorm + residual），输出 `[num_visual_tokens, hidden_size]` 的连续 embedding。

### 1.2 官方方案（`input_embeds` 整段替换）为什么不能用

官方 LongCat-Next 推理代码的做法：

```
PreProcessor:
  文本 token → 普通 embedding 查表（无 ngram）
  图像 → ViT → VQ → VisualEmbeddingBridge → visual_embeddings
  get_multimodal_embed() → 在 <img_pad> 位置 scatter visual_embeddings
  → 整体 input_embeds 传给 LLM

LLM (NmmFlashForCausalLM.forward):
  发现 forward_batch 里有 input_embeds → 跳过自己的 embedding 查表
  → 直接 input_embeds 进 Transformer
```

整个 embedding 计算在 PreProcessor 里完成，LLM 内部不做任何 embedding。

sglang-omni 不能用这个方案。原因：LongCat-Next 使用 NgramEmbedding（word embedding + 12 个 n-gram 投影取平均），文本 token **必须**经过 SGLang 内部的 NgramEmbedding 流程。跳过它等于 ngram 信息全部丢失，输出乱码（Phase 1 问题 3 已验证）。

### 1.3 `replace_embeds` 方案

把流程拆成两步：

1. SGLang 正常对所有 token（包括 `<img_pad>`）做 NgramEmbedding
2. 用外部传入的 embedding 覆盖 `<img_pad>` 位置

```
SGLang 内部:
  input_ids → NgramEmbedding → embeddings [seq_len, hidden_size]
                                    │
  replace_embeds [N, hidden_size] ──→ scatter 覆盖 embeddings[replace_positions]
  replace_positions [N]
                                    │
                                    ▼
                             fused embeddings → Transformer
```

`replace_embeds` 和 `replace_positions` 是两个 Tensor：

- **`replace_embeds`**：shape `[num_multimodal_tokens, hidden_size]`，来自 image_encoder 或 audio_encoder 的输出。image_encoder 产生 visual_embeddings，audio_encoder 产生 audio_embeddings。两个 encoder 的输出在 merge_fn 里拼接成一份 replace_embeds。
- **`replace_positions`**：shape `[num_multimodal_tokens]`，long 类型。标识 replace_embeds 中每个 embedding 要放到序列的哪个位置。值是 `<img_pad>` 和 `<audio_pad>` token 在 input_ids 中的下标。

### 1.4 在 model forward 中的实现位置

当前 `LongcatNextTextForCausalLM.forward` 继承自 SGLang 的 `LongcatFlashForCausalLM`：

```python
# 当前流程（简化）
def forward(self, input_ids, positions, forward_batch):
    hidden_states = self.model(input_ids, positions, forward_batch, input_embeds=None)
    return self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch)
```

`self.model()` 内部会调用 `self.embed_tokens(input_ids)` 做 NgramEmbedding，然后逐层 Transformer。

改动位置在 NgramEmbedding 之后、Transformer 之前。需要确认 SGLang 的 `LongcatFlashModel.forward` 是否已有类似的注入点，如果没有，需要在 `LongcatNextTextForCausalLM.forward` 中覆盖父类实现：

```python
# 改后流程
def forward(self, input_ids, positions, forward_batch):
    # forward_batch 携带 encoder 传来的数据
    replace_embeds = getattr(forward_batch, 'replace_embeds', None)
    replace_positions = getattr(forward_batch, 'replace_positions', None)

    if replace_embeds is not None:
        # SGLang 正常做 NgramEmbedding
        hidden_states = self.model.embed_tokens(input_ids)
        # 覆盖多模态位置
        hidden_states[replace_positions] = replace_embeds.to(hidden_states.dtype)
        # Transformer 正常执行
        for layer in self.model.layers:
            hidden_states = layer(hidden_states, positions, forward_batch)
        hidden_states = self.model.norm(hidden_states)
    else:
        # 纯文本路径，走父类逻辑
        hidden_states = self.model(input_ids, positions, forward_batch)

    return self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch)
```

### 1.5 需要确认的事

SGLang 的 `ForwardBatch` 已有 `token_ids`、`positions`、`out_cache_loc` 等字段。需要查 SGLang 多模态模型（Qwen2-VL、Qwen3-Omni）的 `ForwardBatch` 是否已经有 `replace_embeds` 这类字段。如果有，直接复用字段名；如果没有，新增字段并通过 custom model forward 读取。

另外，`replace_embeds` 的数据需要从 request builder 传递到 ForwardBatch。SGLang 的 Req 对象和 ForwardBatch 初始化之间有一个数据传递路径，需要确认具体在哪个环节注入（预测在 `ForwardBatch.init_new` 或类似的 batch 准备阶段）。

### 1.6 做完步骤一后的状态

text_ar 的 model forward 具备接收任意外部 embedding 的能力。图像 encoder 的 visual_embeddings 和音频 encoder 的 audio_embeddings 共享完全相同的注入路径——区别仅在 `replace_positions` 指向 `<img_pad>` 还是 `<audio_pad>`。

---

## 步骤二：多 Stage PipelineConfig

### 2.1 Phase 1 的拓扑

```python
# config.py 当前
class LongcatNextTextPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "LongcatNextTextForCausalLM"
    model_path: str
    entry_stage: str = "text"
    stages: list[StageConfig] = [
        StageConfig(
            name="text",
            factory="...stages.create_longcat_next_text_executor",
            gpu=[0,1,2,3],
            tp_size=4,
            terminal=True,
        )
    ]
```

请求直接到达 text stage，没有上游。

### 2.2 Phase 2 的拓扑

请求先到 preprocessing（CPU），拆成三路分别发给 image_encoder、audio_encoder、mm_aggregate。encoder 完成后把结果也发给 mm_aggregate，由 mm_aggregate 统一 fan-in 合并，再把最终 AR 输入发给 text_ar。

这比“text_ar 直接 wait_for 多路上游”更贴近 `sglang-omni` 里 Qwen3-Omni 的成熟实现：

1. `text_ar` 保持纯 AR stage，只负责 SGLang continuous batching / KV cache / decode loop。
2. 多模态聚合、空模态跳过、encoder cache 命中、payload 结构整理都放在轻量 `mm_aggregate` stage。
3. 后续要加入 talker / image generation / audio generation 时，aggregate 可以继续作为统一分发点，而不用污染 AR stage。

```
Request
  │
  ▼
preprocessing (CPU)
  ├─ tokenize 文本 → input_ids
  ├─ 加载图像 → pixel_values, grid_thw, image_positions
  ├─ 加载音频 → audio_waveform / audio_features, audio_positions
  │
  ├─ project ──→ image_encoder: {pixel_values, grid_thw, cache_key}
  ├─ project ──→ audio_encoder: {audio_waveform/audio_features, lengths, cache_key}
  └─ project ──→ mm_aggregate:  {input_ids, sampling_params, positions, metadata}
  │
  ├─▶ image_encoder (GPU 0)
  │     ViT → VQ/RVQ → VisualEmbeddingBridge → visual_embeddings
  │     结果 relay → mm_aggregate
  │
  ├─▶ audio_encoder (GPU 1)
  │     AudioTokenizer/Quantizer/Bridge → audio_embeddings
  │     结果 relay → mm_aggregate
  │
  ▼
mm_aggregate (CPU / lightweight)
  wait_for: preprocessing, image_encoder, audio_encoder
  wait_for_fn: 按请求实际模态跳过空 encoder
  merge_fn: 合并 input_ids + replace_embeds + replace_positions
  │
  ▼
text_ar (GPU 2-5, TP=4)
  LongCat-Next AR backbone + NgramEmbedding + replace scatter
```

### 2.3 框架机制：fan-out 和 fan-in

StageConfig 用以下字段控制拓扑：

**fan-out（一对多分发）：**

`next` 字段指定当前 stage 完成后把结果发给哪些下游 stage。框架的 Stage runtime 在请求处理完成后，对 `next` 中的每个目标 stage 调用 relay 发送 payload。

`project_payload` 可选，用于按需分发。preprocessing 产出的 payload 包含 `{input_ids, pixel_values, audio_waveform, ...}` 全部数据，但 image_encoder 只需要 `pixel_values`，audio_encoder 只需要 `audio_waveform`。`project_payload` 是一个 `{stage_name: dotted_fn_path}` 的映射，对应的函数接收完整 payload，返回该 stage 需要的子集。不指定 `project_payload` 的 target 收到完整 payload。

**fan-in（多等一合并）：**

`wait_for` 字段列出当前 stage 启动前需要等待哪些上游 stage 都完成。框架的 merge 逻辑在收到所有 `wait_for` 中列出的 stage 的 payload 后，调用 `merge_fn` 合并。

`merge_fn` 是 dotted function path，接收 `dict[str, StagePayload]`（key 是 stage name，value 是该 stage 发来的 payload），返回一个合并后的 `StagePayload`。

### 2.4 具体配置

```python
_PKG = "sglang_omni.models.longcat_next"

class LongcatNextPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "LongcatNextTextForCausalLM"
    model_path: str
    entry_stage: str = "preprocessing"

    stages: list[StageConfig] = [
        # Stage 1: CPU 预处理
        StageConfig(
            name="preprocessing",
            process="preprocessing",
            factory=f"{_PKG}.stages.create_preprocessing_executor",
            next=["image_encoder", "audio_encoder", "mm_aggregate"],
            route_fn=f"{_PKG}.request_builders.resolve_preprocessing_next_stages",
            project_payload={
                "image_encoder": f"{_PKG}.request_builders.project_to_image_encoder",
                "audio_encoder": f"{_PKG}.request_builders.project_to_audio_encoder",
                "mm_aggregate": f"{_PKG}.request_builders.project_to_mm_aggregate",
            },
        ),

        # Stage 2: 图像编码器
        StageConfig(
            name="image_encoder",
            process="image_encoder",
            factory=f"{_PKG}.stages.create_image_encoder_executor",
            gpu=[0],
            next="mm_aggregate",
            project_payload={
                "mm_aggregate": f"{_PKG}.request_builders.project_encoder_to_mm_aggregate",
            },
        ),

        # Stage 3: 音频编码器
        StageConfig(
            name="audio_encoder",
            process="audio_encoder",
            factory=f"{_PKG}.stages.create_audio_encoder_executor",
            gpu=[1],
            next="mm_aggregate",
            project_payload={
                "mm_aggregate": f"{_PKG}.request_builders.project_encoder_to_mm_aggregate",
            },
        ),

        # Stage 4: 多模态聚合
        StageConfig(
            name="mm_aggregate",
            process="mm_aggregate",
            factory=f"{_PKG}.stages.create_aggregate_executor",
            wait_for=["preprocessing", "image_encoder", "audio_encoder"],
            wait_for_fn=f"{_PKG}.request_builders.resolve_mm_aggregate_wait_sources",
            merge_fn=f"{_PKG}.merge.merge_for_text_ar",
            next="text_ar",
        ),

        # Stage 5: AR backbone
        StageConfig(
            name="text_ar",
            process="text_ar",
            factory=f"{_PKG}.stages.create_longcat_next_text_executor",
            factory_args={
                "device": "cuda:0",
                "max_running_requests": 32,
                "enable_torch_compile": False,
                "mem_fraction_static": 0.85,
            },
            gpu=[2, 3, 4, 5],
            tp_size=4,
            terminal=True,
        ),
    ]
```

Qwen3-Omni 的 `config.py` 是此模式的直接参考——preprocessing fan-out 到 image_encoder + audio_encoder + mm_aggregate，encoder 结果统一 relay 到 mm_aggregate，mm_aggregate fan-in 后再到 thinker。LongCat-Next Phase 2 应采用同一拓扑：把多模态合并逻辑放在 aggregate stage，而不是让 AR stage 直接管理多路上游。

### 2.5 与原方案的差异

原文档把 `text_ar` 设计成直接 `wait_for=[preprocessing, image_encoder, audio_encoder]`。这个方案可行，但会让 AR stage 同时承担两类职责：

1. SGLang AR serving：continuous batching、KV cache、decode loop、ngram token table。
2. 多模态编排：等待上游、跳过空 encoder、合并 payload、处理 encoder cache metadata。

Phase1 的经验表明，LongCat-Next 的 AR 路径已经有很多模型特异约束（NgramEmbedding、zero-expert patch、vocab size、MLA config）。Phase2 应尽量保持 AR stage 单纯，避免把 pipeline orchestration 混进 AR scheduler。因此更新为 `mm_aggregate -> text_ar`：

- `mm_aggregate` 负责 fan-in / merge / wait_for_fn。
- `text_ar` 只接收一份已经整理好的 AR payload。
- 未来接入生成侧 `visual_head/audio_head` 时，aggregate 也可以作为统一路由点。

### 2.6 `mm_aggregate` 的资源定位：控制面，不占 GPU

`mm_aggregate` 的职责是控制面和元数据合并，而不是模型计算。它不执行 ViT、audio encoder、MoE、attention、logits 或 sampling，因此不应配置 `gpu`，也不应参与 CUDA device 选择。

它应该做：

1. 根据 `wait_for_fn` 判断本请求需要等待哪些上游。
2. 合并 `input_ids`、`sampling_params`、`image_positions`、`audio_positions` 等小元数据。
3. 组织 `replace_embeds` 的来源信息，形成 text_ar 可消费的 payload contract。
4. 尽量不读取、不 `.cpu()`、不 `.cuda()` 大 tensor。

需要区分两条路径：

| 层面 | 位置 | 职责 |
|---|---|---|
| 控制面 | `mm_aggregate` | wait/merge/route/metadata |
| 数据面 | encoder relay + text_ar materialize | 大 tensor 的传输和落到目标 device |

#### 传输开销风险

如果 `image_encoder/audio_encoder` 输出的 CUDA tensor 被 `mm_aggregate` 反序列化到 CPU，或者在 `merge_fn` 中执行：

```python
visual_embeddings = visual_embeddings.cpu()
replace_embeds = torch.cat([visual_embeddings, audio_embeddings], dim=0)
```

就会产生额外 `GPU -> CPU -> GPU`，即 D2H/H2D overhead。图像 token 多、音频长时，这个开销会很明显。

因此，Phase2 的实现原则是：

- MVP 阶段可以接受 CPU/shm 中转以验证正确性，但要把它标记为未优化路径。
- `mm_aggregate` 不应主动做大 tensor copy；如果必须拼接，优先在 `text_ar` 所在进程、目标 GPU 上完成。
- 优化阶段应把 `replace_embeds` 改成 tensor ref / shm ref / cache key 形式，由 text_ar materialize。

推荐 payload contract：

```python
{
    "input_ids": input_ids,
    "sampling_params": sampling_params,
    "replace_positions": replace_positions,
    # MVP: 可以是 CPU tensor 或 relay 后 tensor
    "replace_embeds": replace_embeds,
    # 优化版: 用 refs 延迟 materialize
    "replace_embed_refs": [
        {"stage": "image_encoder", "key": image_cache_key, "shape": ..., "dtype": ...},
        {"stage": "audio_encoder", "key": audio_cache_key, "shape": ..., "dtype": ...},
    ],
}
```

最终目标是：`mm_aggregate` 只拼 metadata/ref，真正的大 tensor 拼接和 H2D 放在 `text_ar` request_builder 或 model runner 侧完成。

---

## 步骤三：Encoder Stage 工厂函数

### 3.1 Encoder 和 AR Stage 的本质区别

AR stage（text_ar）返回 `OmniScheduler`：管理 KV cache、token-level batch 调度、CUDA graph、continuous batching。这是完整的 SGLang serving 栈。

Encoder stage（image_encoder、audio_encoder）返回 `SimpleScheduler`：接收 StagePayload，调用编码函数，返回 StagePayload。没有 KV cache，没有 token 级别调度，没有 CUDA graph。请求级处理——一个请求进来，编码完，发出去。

```python
# SimpleScheduler 的核心
class SimpleScheduler:
    def __init__(self, compute_fn, batch_compute_fn=None, max_batch_size=1, ...):
        self.inbox = Queue()   # 接收 StagePayload
        self.outbox = Queue()  # 发出处理后的 StagePayload
        self._fn = compute_fn  # payload → payload
```

框架的 Stage runtime 负责把 inbox 的 payload 喂给 compute_fn，把 outbox 的结果 relay 到下游 stage。

### 3.2 工厂函数骨架

图像和音频 encoder 的工厂函数骨架完全一致。参照 Qwen3-Omni 的 `create_image_encoder_executor`：

```python
def create_image_encoder_executor(model_path, *, device="cuda", dtype=None):
    from sglang_omni.models.longcat_next.components.visual_model import (
        LongcatNextVisualTokenizer,
    )
    from sglang_omni.models.weight_loader import load_module
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    # 1. 实例化模型
    model = LongcatNextVisualTokenizer(model_path)

    # 2. 从 checkpoint 只加载 visual tokenizer 权重
    load_module(model, model_path, prefix="model.visual_tokenizer.",
                dtype=torch.bfloat16, device=device, strict=True)

    # 3. 单请求编码
    def _encode(payload):
        pixel_values = payload.data["pixel_values"]
        grid_thw = payload.data["grid_thw"]
        visual_embeddings = model.encode(pixel_values, grid_thw)
        payload.data["visual_embeddings"] = visual_embeddings
        return payload

    # 4. 批量编码
    def _encode_batch(payloads):
        pixel_values = torch.cat([p.data["pixel_values"] for p in payloads])
        grid_thw = torch.cat([p.data["grid_thw"] for p in payloads])
        visual_embeddings = model.encode(pixel_values, grid_thw)
        # 按 token counts 拆分回各请求
        ...
        return payloads

    return SimpleScheduler(
        _encode,
        batch_compute_fn=_encode_batch,
        max_batch_size=32,
        max_batch_wait_ms=50,
    )
```

### 3.3 Visual Tokenizer 模型类的移植

从官方 LongCat-Next 推理代码搬到 `sglang_omni/models/longcat_next/components/`：

| 文件 | 内容 |
|------|------|
| `visual_encoder.py` | `LongcatNextViT`：包装 Qwen2.5-VL 的 `Qwen2_5_VisionTransformerPretrainedModel`，做 ViT 编码 + 2D RoPE + window attention |
| `visual_bridge.py` | `LongcatNextVisualBridge`：`RMSNorm` + `Linear→GELU→Linear`，merge_size=2 时把 2×2 相邻 patch 合并 |
| `visual_quantize.py` | `ResidualVectorQuantizer`：残差向量量化器，8 层 depth，每层 16384 个 code，每层量化前一层的残差。输出 `indices [num_tokens, 8]` |
| `visual_embedding.py` | `VisualEmbeddingBridge`：8 个 `nn.Embedding(16385, hidden_size)` 按 codebook 求和 + `DecoderLayer`（MLP + LayerNorm + residual）。输入 `indices [N, 8]`，输出 `embeddings [N, hidden_size]` |
| `visual_model.py` | `LongcatNextVisualTokenizer`：包装以上四个的顶层类。`encode(pixel_values, grid_thw) → visual_embeddings` |

移植量约 800-900 行。关键依赖：`LongcatNextViT` 继承自 transformers 的 `Qwen2_5_VisionTransformerPretrainedModel`，需要确认 transformers 版本兼容性。

### 3.4 权重加载详解

Checkpoint 的 `model.safetensors.index.json` 记录了所有权重的 key 到 shard 文件的映射：

```json
{
  "model.embed_tokens.weight": "model-00001-of-00005.safetensors",
  "model.layers.0.self_attn.q_a_proj.weight": "model-00001-of-00005.safetensors",
  ...
  "model.visual_tokenizer.visual_model.patch_embed.proj.weight": "model-00004.safetensors",
  "model.visual_tokenizer.visual_model.blocks.0.attn.qkv.weight": "model-00004.safetensors",
  "model.visual_tokenizer.visual_bridge_model.bridge.mlp.0.weight": "model-00004.safetensors",
  "model.visual_tokenizer.visual_bridge_model.quantizer.codebooks.0.weight": "model-00004.safetensors",
  ...
  "model.audio_tokenizer.encoder.conv1.weight": "model-00005.safetensors",
  ...
}
```

`load_module(model, model_path, prefix="model.visual_tokenizer.")` 做的事：

1. 遍历 weight_map，只取 `key.startswith("model.visual_tokenizer.")` 的条目
2. 加载对应的 shard 文件，只读匹配的 tensor
3. 去掉 prefix（`"model.visual_tokenizer.visual_model.patch_embed.proj.weight"` → `"visual_model.patch_embed.proj.weight"`）
4. `model.load_state_dict(state_dict, strict=True)` —— 确保 model 的每个参数都有对应权重，一个不多一个不少

图像 encoder 只加载 `model.visual_tokenizer.*`，音频 encoder 只加载 `model.audio_tokenizer.*`。text_ar 在 `load_weights` 里过滤掉这两个前缀。同一份 checkpoint，三个 stage 各取所需。

### 3.5 音频 Encoder

音频 encoder 骨架和图像完全一致，差异在模型结构和 weight prefix：

- 模型类：`LongcatNextAudioTokenizer`（官方代码中的 `LongcatAudioTokenizer`）
- weight prefix：`"model.audio_tokenizer."`
- 输入：`audio_waveform` + `encoder_length` + `bridge_length`
- 输出：`audio_embeddings [num_audio_tokens, hidden_size]`

---

## 步骤四：Preprocessing 和 Merge

### 4.1 Preprocessing Stage

```
接收 HTTP 请求（messages 格式）
  │
  ├─ tokenizer.apply_chat_template(messages)
  │    ├─ <img_start>...<img_end> 之间的图像引用 → 提取图像路径
  │    ├─ <audio_start>...<audio_end> 之间的音频引用 → 提取音频路径
  │    └─ 按图像/音频尺寸填充 <img_pad> / <audio_pad> token
  │
  ├─ 图像处理（OmniMMProcessor）
  │    ├─ smart_resize(h, w) → 对齐到 factor=28 的尺寸
  │    ├─ resize → normalize → pixel_values [N_patches, 1176]
  │    └─ grid_thw [[1, h_bar//merge, w_bar//merge]]
  │
  ├─ 音频处理
  │    └─ 加载 waveform → feature extraction（具体依赖官方 audio tokenizer）
  │
  └─ fan-out
       ├─→ image_encoder: {pixel_values, grid_thw, image_positions, cache_key, request_id}
       ├─→ audio_encoder: {audio_waveform/features, encoder_length, bridge_length, audio_positions, cache_key, request_id}
       └─→ mm_aggregate:  {input_ids, sampling_params, image_positions, audio_positions, request_id}
```

图像尺寸决定了 `<img_pad>` 的数量：图像经过 `smart_resize` 得到 `(h_bar, w_bar)`，ViT 的 `patch_size=14, merge_size=2` 下，visual token 数 = `(h_bar/14) × (w_bar/14) / 4` = `(h_bar × w_bar) / 784`。

这些 `<img_pad>` token 的 embedding 不会来自 NgramEmbedding 查表，而是在步骤一中被 replace_embeds 覆盖。

### 4.2 Merge：多路上游合并

text_ar 收到来自 mm_aggregate 的单份 payload，不再直接等待 preprocessing / image_encoder / audio_encoder。`merge_fn` 在 mm_aggregate 内合并三份上游 payload：

```python
def merge_for_text_ar(payloads: dict[str, StagePayload]) -> StagePayload:
    """
    payloads = {
        "preprocessing":  StagePayload(data={input_ids, sampling_params, positions, ...}),
        "image_encoder":  StagePayload(data={visual_embeddings, ...}) or absent,
        "audio_encoder":  StagePayload(data={audio_embeddings, ...}) or absent,
    }
    """
    pre = payloads["preprocessing"].data
    input_ids = pre["input_ids"]

    # MVP 写法：如果 encoder 输出已经是可 relay 的 tensor，可以直接组织；
    # 注意不要在 mm_aggregate 中主动 .cpu() / .cuda() 大 tensor。
    replace_embeds_list = []
    replace_positions_list = []
    replace_embed_refs = []

    ie = payloads.get("image_encoder")
    if ie is not None:
        img_positions = pre.get("image_positions") or (input_ids == IMG_PAD_TOKEN_ID).nonzero(as_tuple=True)[0]
        if "visual_embeddings" in ie.data:
            replace_embeds_list.append(ie.data["visual_embeddings"])
        if "visual_embed_ref" in ie.data:
            replace_embed_refs.append(ie.data["visual_embed_ref"])
        replace_positions_list.append(img_positions)

    ae = payloads.get("audio_encoder")
    if ae is not None:
        audio_positions = pre.get("audio_positions") or (input_ids == AUDIO_PAD_TOKEN_ID).nonzero(as_tuple=True)[0]
        if "audio_embeddings" in ae.data:
            replace_embeds_list.append(ae.data["audio_embeddings"])
        if "audio_embed_ref" in ae.data:
            replace_embed_refs.append(ae.data["audio_embed_ref"])
        replace_positions_list.append(audio_positions)

    return StagePayload(data={
        "input_ids": input_ids,
        "sampling_params": pre["sampling_params"],
        "replace_embeds": torch.cat(replace_embeds_list, dim=0) if replace_embeds_list else None,
        "replace_embed_refs": replace_embed_refs or None,
        "replace_positions": torch.cat(replace_positions_list, dim=0) if replace_positions_list else None,
    })
```

merge_fn 的输出进入 text_ar 的 request builder，构造 SGLang Req 时把 `replace_embeds` / `replace_embed_refs` 和 `replace_positions` 挂到 Req 上，后续由 ForwardBatch 带入 model forward。

需要额外实现 `wait_for_fn`：根据 preprocessing 判断本请求实际是否包含 image/audio。纯文本请求只等待 preprocessing；只有图像时等待 preprocessing + image_encoder；只有音频时等待 preprocessing + audio_encoder；图音混合才等待三路。这样可以避免空 encoder stage 造成不必要等待。

实现注意：如果走 `replace_embed_refs` 优化路径，`mm_aggregate` 只传引用，text_ar 侧负责把引用解析成目标 GPU 上的 tensor；如果走 MVP 直接 tensor 路径，允许一次 CPU/shm 中转，但后续需要通过 profiler 验证是否成为瓶颈。

---

## 步骤五：端到端集成

### 5.1 集成路径

```
curl → API Server
         │
         ▼
       Coordinator → preprocessing
                       ├─→ image_encoder (shm relay)
                       ├─→ audio_encoder (shm relay)
                       └─→ mm_aggregate (shm relay)
                                ▲
                                ├─ image_encoder result
                                └─ audio_encoder result
                                │
                           merge_fn 合并三路
                                │
                                ▼
                            text_ar
                                │
                           request_builder 构造 Req + replace_embeds
                                │
                           OmniScheduler → AR 生成
                                │
                           result_adapter → 文本响应
```

### 5.2 验证策略

1. **纯文本请求**：不携带图像/音频时，`replace_embeds` 为 None，走 Phase 1 的原始路径。输出应与 Phase 1 完全一致（回归测试）。
2. **图像+文本请求**：对比官方 LongCat-Next 推理代码，相同 image + prompt → 相同的 output logits。
3. **音频+文本请求**：同上。
4. **混合请求**：同时有图像和音频输入，验证两个 encoder 的 embedding 都正确注入。

### 5.3 GPU 分配（单机 8 卡 H800）

| GPU | Stage | 说明 |
|-----|-------|------|
| CPU | preprocessing + mm_aggregate | 解析请求、预处理、控制面聚合；`mm_aggregate` 不占 GPU，避免大 tensor copy |
| 0 | image_encoder | ViT+VQ+Bridge，~1-1.5B 参数，bf16 ~2-3GB 显存 |
| 1 | audio_encoder | AudioEncoder+Quantizer，~0.5B，bf16 ~1GB |
| 2-5 | text_ar (TP=4) | AR backbone 68B MoE，KV cache 主要消费者 |
| 6-7 | 预留 | 未来扩展（talker AR、code2wav 等） |

---

## 文件清单（预计新增/修改）

```
sglang_omni/models/longcat_next/
├── config.py                          # 改：单 stage → preprocessing/image/audio/mm_aggregate/text_ar
├── stages.py                          # 改：新增 create_preprocessing/image_encoder/audio_encoder/aggregate_executor
├── sglang_model.py                    # 改：model forward 支持 replace_embeds
├── request_builders.py                # 改：project_payload、wait_for_fn、replace_embeds 请求构造
├── merge.py                           # 新增：merge_for_text_ar
├── components/
│   ├── visual_encoder.py              # 新增：LongcatNextViT
│   ├── visual_bridge.py               # 新增：LongcatNextVisualBridge
│   ├── visual_quantize.py             # 新增：ResidualVectorQuantizer
│   ├── visual_embedding.py            # 新增：VisualEmbeddingBridge
│   ├── visual_model.py                # 新增：LongcatNextVisualTokenizer（顶层包装）
│   └── audio_encoder.py               # 新增：LongcatNextAudioTokenizer
└── 开发文档_phase2.md                  # 本文档
```

---

## 附录 A：HF 权重命名表

Phase2 开发前需要先用 HF checkpoint 的 `model.safetensors.index.json` 生成权重 key 表，避免移植 encoder 时靠猜测命名。当前已从 HuggingFace 下载元数据到：

```text
artifacts/longcat_next/hf_meta/
├── config.json                         # HF 原始文件
├── model.safetensors.index.json         # HF 原始文件，包含 weight_map
├── tokenizer_config.json                # HF 原始文件
└── generation_config.json               # HF 原始文件
```

基于 `weight_map` 派生的本地分析表在：

```text
artifacts/longcat_next/
├── all_weights.tsv
├── ar_backbone_weights.tsv
├── visual_tokenizer_weights.tsv
├── audio_tokenizer_weights.tsv
├── visual_head_weights.tsv
├── audio_head_weights.tsv
├── audio_embed_layers_weights.tsv
└── summary.tsv
```

当前统计：

| component | keys | shards | 说明 |
|---|---:|---:|---|
| all | 13450 | 15 | 全量 checkpoint key |
| ar_backbone | 11143 | 15 | AR backbone / ngram / lm_head |
| visual_tokenizer | 425 | 3 | Phase2 图像 encoder 主体 |
| audio_tokenizer | 1740 | 3 | Phase2 音频 encoder 主体 |
| visual_head | 71 | 3 | 后续图像生成侧 |
| audio_head | 71 | 3 | 后续音频生成侧 |
| audio_embed_layers | 0 | 0 | HF key 中未命中该前缀，需要从 `audio_tokenizer` 或 `all` 表确认真实命名 |

开发要求：

1. 移植 visual/audio encoder 前，先对照 `*_weights.tsv` 确认 `state_dict()` 名称与 checkpoint prefix-stripped 名称一致。
2. `load_module(..., strict=True)` 必须作为阶段性验收；不能长期使用非 strict 加载掩盖缺失权重。
3. 如果官方实现中的模块名与 HF key 不一致，以 HF `model.safetensors.index.json` 为准，必要时在移植类中做命名适配。

---

## 附录 B：性能优化策略

Phase2 不应一开始追求完整性能优化，应分三阶段推进。

### B.1 正确性优先

目标：

1. 纯文本回归完全不退化。
2. 单图 + 文本理解跑通。
3. 单音频 + 文本理解跑通。
4. 相同输入下，encoder 输出 embedding / AR logits 尽量对齐官方实现。

此阶段可以暂时不做：encoder batching、CUDA graph、cache、同卡显存压榨、生成侧 image/audio head。

原因：Phase1 的经验表明，LongCat-Next 的主要风险不是“慢”，而是“能跑但数值语义错”：例如 ngram 未启用、hash base 错、vocab size 错、zero-expert `-1` 越界等。

### B.2 利用 multi-stage 的基础优化

正确性稳定后，再启用 `sglang-omni` 的通用优化：

1. encoder 与 AR 解耦：encoder 是请求级计算，AR 是 token-level continuous batching，不能互相阻塞。
2. image/audio encoder 并行 fan-out。
3. `SimpleScheduler(batch_compute_fn, max_batch_size=32, max_batch_wait_ms=50)` 做 encoder micro-batching。
4. `StageOutputCache` 缓存重复图片/音频的 encoder 输出。
5. 若 encoder 与 AR 同卡，参考 Qwen3-Omni 的 `encoder_mem_reserve` / `mem_fraction_static` 做显存契约；初期建议独立 GPU 更稳。

### B.3 后续激进优化

在端到端稳定后再考虑：

- encoder CUDA graph；
- image/audio tokenizer torch.compile；
- shared-memory tensor relay，减少大 tensor 序列化；
- pinned CPU media embedding cache；
- AR async decode；
- prefix cache + media cache 协同。

这些优化都不应阻塞 Phase2 MVP。MVP 的核心是：**保留 NgramEmbedding，正确注入 replace_embeds，完成图像/音频理解链路。**
