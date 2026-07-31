# LongCat-Next Phase 3 开发文档：语音输出

## 概述

Phase 3 的目标是实现语音输出（image + audio + text → text + speech），使模型具备"说话"能力。

LongCat-Next 的语音输出基于 DiNA 范式下的 discrete audio codebook token——AR backbone 在 decode 阶段交替产出文本 token 和 audio codebook token（8 个 codebook，每步同时产出）。audio token 经 audio de-tokenizer（flow matching）和 vocoder（HiFi-GAN）解码为 PCM 波形。

### 和 Phase 2 的差异

| | Phase 2 | Phase 3 |
|---|---|---|
| AR 输出 | 仅文本 token | 文本 token + audio codebook token |
| model forward | decode 时 `super().forward()` 标准路径 | decode 时覆盖 forward：双头 logits + embedding 融合 |
| 采样 | 单头 `lm_head` sample | 双头 `lm_head` + `audio_head` sample，状态机驱动 |
| 新增 stage | image_encoder + audio_encoder | + code2wav（audio de-tokenizer + vocoder） |
| 流式输出 | 文本 stream to client | 文本 stream + audio frame stream_to → code2wav |

### 数据流

```
preprocessing ─┬─→ image_encoder ────┐
               ├─→ audio_encoder ────┤
               └─→ mm_aggregate ─────┤
                                     ▼
                                  text_ar (GPU 2-5, TP=4)
                                     │
                                     ├─→ text tokens → stream to client
                                     │
                                     └─→ audio tokens → stream_to → code2wav (GPU 6)
                                                                        │
                                                                        ├─ audio de-tokenizer (flow matching)
                                                                        ├─ vocoder (HiFi-GAN)
                                                                        └─→ PCM audio stream to client
```

---

## 步骤一：audio_head 集成（✅ 已完成）

### 1.1 架构

audio_head 是独立于 AR backbone 的 nn.Module，在 text_ar stage 内部和 lm_head 共享同一个 hidden state，实现双头输出。

**设计决策：重新实现而非导入官方代码。**

官方 `OmniAudioHead`（`LongCat-Next-inference/modules/image_head.py`）包含完整的 TP 支持代码，依赖 `sglang.srt.distributed`、`sglang.srt.layers.linear` 等 SGLang 内部模块，且 flash_attn 调用直接使用 v2 API。为了：
- 避免对 inference repo 的运行时文件路径依赖
- 去除我们不需要的 TP 分支（`enable_tp=False`）
- 保持代码自包含、可维护

我们在 [components/audio_head.py](sglang_omni/models/longcat_next/components/audio_head.py) 中重新实现了非 TP 路径，和官方实现逐层对齐。

### 1.2 组件层次

```
LongcatNextAudioHead           ← 顶层 wrapper，外部调用入口
  ├── _OmniAudioHead           ← audio head 主体
  │     ├── _RMSNorm           ← hidden_norm, headnorm
  │     ├── nn.Linear          ← hidden_proj
  │     ├── _CasualDepthTransformerLayer × 4
  │     │     ├── _FlashVarLenAttention  ← causal flash attention over 8 codebook positions
  │     │     ├── _RMSNorm × 2           ← layernorm1, layernorm2
  │     │     └── nn.Linear × 2          ← FFN (einsum depth-aware path)
  │     └── heads (nn.ModuleList[Linear]) × 8  ← 每 codebook 一个预测头
  └── audio_emb_layers (nn.ModuleList[Embedding]) × 7  ← 从 embed_tokens 切片
```

### 1.3 _OmniAudioHead 前向逻辑

```python
def forward(x, audio_tokens, audio_emb_layers, batch_size, codebook_id):
    # Step 1: 对前 7 个 codebook 的历史预测做 embedding + cumsum
    cumsum_audio_embed = cumsum(stack([emb_layers[i](audio_tokens[:, i]) for i in 0..6]))

    # Step 2: 将 LLM hidden state 和 cumsum embedding 沿 depth 维度拼接
    hidden_states = concat([x, cumsum_audio_embed], dim=1)  # [B, 8, hidden_size]
    #   pos 0: LLM hidden state（当前 step 的语义信息）
    #   pos 1..7: 前 i 个 codebook 的累计 embedding（音色/韵律 context）

    # Step 3: norm → proj → 4 层 DepthTransformer → headnorm
    hidden_states = headnorm(transformer_layers(hidden_proj(hidden_norm(hidden_states))))

    # Step 4: 取 codebook_id 位置的 hidden，过对应的预测头
    return heads[codebook_id](hidden_states[:, codebook_id])
    # 输出 shape: [B, codebook_size + 1]
    # +1 是 audio_head 的标准设计：多一个 logit 表示"无音频"
```

### 1.4 DepthTransformer 因果注意力

`_CasualDepthTransformerLayer` 在 8 个 codebook position 上做 causal flash attention：

- **causal mask**：codebook `i` 只能 attend codebook `0..i`。这意味着预测 codebook 3 时，attention 可以看到 codebook 0/1/2 的 embedding 上下文，但不能看到 codebook 4..7
- **FFN**：`depth > 1` 时使用 einsum 路径——将 `linear1.weight` reshape 为 `[ffn//depth, depth, dims]`，对每个 codebook position 做 depth-aware 变换
- **4 层**：和官方一致，`audio_head_transformer_layers=4`

### 1.5 8 步因果预测循环

```python
# LongcatNextAudioHead.forward
next_codes = zeros(B, 8)
for codebook_id in 0..7:
    logits = audio_head(hidden_state, next_codes, audio_emb_layers, B, codebook_id)
    next_codes[:, codebook_id] = argmax(logits)
return next_codes
```

- 循环 8 次，每次预测一个 codebook
- 后一次预测能看到前一次的预测结果（通过 `audio_tokens` 参数传入）
- 使用 `argmax`（greedy decoding），避免 `torch.multinomial` 的 host sync → CUDA Graph 兼容

### 1.6 audio_emb_layers 权重来源

`audio_emb_layers` 是 7 个 `nn.Embedding` 层（codebook 0..6，不含最后一个），权重从 `model.embed_tokens.weight` 切片得到：

| 层 | embed_tokens 切片范围 | vocab 大小 |
|----|----------------------|-----------|
| Layer 0 | `[131125 : 131125+8192]` | 8192 |
| Layer 1 | `[139317 : 139317+4096]` | 4096 |
| Layer 2 | `[143413 : 143413+2048]` | 2048 |
| Layer 3 | `[145461 : 145461+1024]` | 1024 |
| Layer 4 | `[146485 : 146485+1024]` | 1024 |
| Layer 5 | `[147509 : 147509+1024]` | 1024 |
| Layer 6 | `[148533 : 148533+1024]` | 1024 |

- 起始偏移 = `audio_offset = 131125`
- 最后一个 codebook（index 7）不需要自己的 embedding，因为它之后没有更高 index 的 codebook 需要 aggregate 它的信息
- 这 7 个 layer 和 Phase 2 encoder 中的 `_OffsetCodebookEmbedding` 逻辑一致，区别在：encoder 是 8 层 sum all，audio_head 是 7 层按需索引

### 1.7 权重加载

| 组件 | 加载方式 | Keys 数量 | 来源 |
|------|---------|----------|------|
| `_OmniAudioHead` | `load_module(prefix="audio_head.")` | 71 | safetensors |
| `audio_emb_layers` | `load_weights_by_prefix(prefix="model.embed_tokens.")` → 切片 | 1（共享 embed_tokens） | safetensors |

SGLang weight loading（`sglang_model.py` 的 `load_weights`）仍然过滤 `audio_head.` 前缀，避免重复加载。

### 1.8 模型集成

在 `stages.py` 的 `create_longcat_next_text_executor` 中：

```
create_sglang_infrastructure_defer_cuda_graph
  → model 创建，权重加载
  → vocab_size fix（ngram 扩展后同步）
  → _attach_audio_head()          ← 新增：创建并挂载 audio_head
  → init_device_graphs()          ← CG capture（目前仍只捕获 text-only 路径）
  → SGLangOutputProcessor
  → OmniScheduler
```

`sglang_model.py` 变更：
- `__init__` 末尾新增 `self.audio_head = None`
- 新增 `set_audio_head()` 方法
- `forward()` 当前不变（text-only），audio 模式的 forward 覆盖在下一步实现

### 1.9 当前行为

- **Text-only 模式**：完全不受影响。`self.audio_head` 已加载但未被调用，forward 仍走 `super().forward()`
- **显存**：audio_head 约 71 keys，占几十 MB（远小于 AR backbone 的 68B MoE）
- **CG 兼容**：audio_head 在 CG capture 之前完成初始化，后续 capture 双头 forward 时可以直接包含

### 1.10 文件变更清单

```
sglang_omni/models/longcat_next/
├── components/
│   └── audio_head.py                  ← 新增（360 行）
├── sglang_model.py                    ← 改：+ self.audio_head, + set_audio_head()
└── stages.py                          ← 改：+ _attach_audio_head(), 插入调用点
```

---

## 步骤二：输入端 — decode 阶段的 embedding 融合（🔜 待实现）

### 2.1 目标

Parallel 模式下每个 decode step 同时产出文本 token 和 audio codebook token。下一步的 input embedding 需要融合两者：

```python
input_embeds = embed_tokens(prev_text_token) + audio_embed(prev_audio_codes)
```

### 2.2 audio_embed 来源

`audio_embed` 是 `_OffsetCodebookEmbedding`（Phase 2 encoder 中已有的组件），从 `model.embed_tokens.weight` 的 `audio_offset` 处切片，对 8 个 codebook id 分别查表再求和。

### 2.3 和 replace_embeds 的关系

- Phase 2 的 `replace_embeds`：**prefill 阶段**注入外部 encoder 输出（视觉/音频理解），在 `_build_longcat_input_embeds` 中通过 scatter 实现
- Phase 3 的 embedding 融合：**decode 阶段**当前 step 的输入构造，通过 add 实现
- 两者互不影响，可以共存

### 2.4 实现位置

在 `sglang_model.py` 中修改 `_build_longcat_input_embeds` 的 decode 路径，或新增独立方法。

### 2.5 和 CUDA Graph 的兼容性

embedding 融合使用纯 tensor 操作（查表 + add），无动态 shape、无 host sync，天然兼容 CUDA Graph。

---

## 步骤三：输出端 — 双头 forward（🔜 待实现）

### 3.1 目标

同一个 hidden state 同时过两个 head：

```python
hidden = Transformer(input_embeds)       # [B, 1, 3072]
text_logits = lm_head(hidden[:, -1])      # [B, vocab_size]
audio_codes = audio_head(hidden[:, -1])   # [B, 8]
```

### 3.2 实现方式

覆盖 `sglang_model.py` 的 `forward()` 方法：

```python
def forward(self, input_ids, positions, forward_batch):
    # Phase 2: multimodal prefill
    if forward_batch.longcat_replace_embeds is not None:
        return self._multimodal_prefill_forward(...)

    # Phase 3: audio decode (detect via forward_batch flag)
    if self.audio_head is not None and self._is_audio_mode(forward_batch):
        return self._audio_decode_forward(...)

    # Phase 1: text-only
    return super().forward(input_ids, positions, forward_batch)
```

### 3.3 CUDA Graph 注意事项

当前 CG 只捕获了 text-only decode 路径。双头 forward 的 CG capture 需要在 `init_device_graphs()` 之前完成所有 buffer 预分配，并在 capture 时走 audio mode 的 forward 路径。详见步骤五。

---

## 步骤四：采样端 — 状态机（🔜 待实现）

### 4.1 Parallel 模式

同一个 forward step 中 `lm_head` 和 `audio_head` 同时采样。推理时 `delay=0`：

```
Step 0: text_token_0 + audio_codes_0    ← 同时产出
Step 1: text_token_1 + audio_codes_1
...
Step N: text_token_N + audio_codes_N
```

### 4.2 状态机设计

```
TEXT_MODE ──→ 检测到 <audio_gen_start> → AUDIO_MODE
AUDIO_MODE:
  lm_head 采样 text token
  audio_head 采样 8 个 codebook token（argmax，GPU 原生操作）
  如果 text_token == <audio_gen_end> → TEXT_MODE（或 EOS）
```

### 4.3 采样策略

- **文本 token**：沿用 SGLang 的 top-k/top-p 采样（在 `process_batch_result` 中）
- **Audio codebook token**：`argmax`（greedy），原因：
  - `torch.multinomial` 需要 CPU sync，破坏 CUDA Graph replay
  - 和其他 CG 文章分析一致：graph 内部的采样必须是 GPU 原生确定性操作
  - 官方 TTS 场景使用 greedy decoding

---

## 步骤五：CUDA Graph 兼容性（🔜 待实现）

### 5.1 核心问题

Phase 2 的 decode 路径是干净的 `super().forward()`——SGLang 标准 LLM decode，CG 捕获无额外适配成本。

Phase 3 的 decode 路径变了：embedding 融合、audio_head forward（含 8 步 codebook 因果循环）、双头输出。这三项必须出现在 capture 的 graph 里。

### 5.2 统一 Graph 策略

**始终运行双头，通过 persistent buffer 的值控制有效输出。**

```python
def forward(input_ids, positions, forward_batch):
    # Step 1: embedding 融合（始终执行）
    text_emb = embed_tokens(input_ids)
    audio_codes = forward_batch.longcat_audio_codes  # persistent buffer
    audio_emb = audio_codebook_embedding(audio_codes) # text-only 时全零 → 无贡献
    input_embeds = text_emb + audio_emb

    # Step 2: Transformer（始终执行）
    hidden = self.model(input_embeds)

    # Step 3: 双头（始终执行）
    text_logits = self.lm_head(hidden)
    audio_logits = self.audio_head(hidden, audio_codes)  # text-only 时结果被丢弃

    return text_logits, audio_logits
```

Graph 永远是同一张——embedding 融合 + Transformer + lm_head + audio_head 的完整 DAG。text-only 模式下 audio_head 仍执行但结果在 `process_batch_result` 中忽略。

### 5.3 静态约束

| 约束 | 状态 |
|------|------|
| `range(8)` 在 capture 时固化为常量 | ✅ 静态循环 |
| `argmax` 是 GPU 原生操作 | ✅ 无 host sync |
| `prev_codes` in-place 更新 | ✅ 不分配新 tensor |
| codebook 间因果依赖 | ✅ DepthTransformer 内部 causal mask |

### 5.4 Deferred Capture

```
Step 1: create_sglang_infrastructure_defer_cuda_graph → 模型创建
Step 2: _attach_audio_head()                           → 加载 audio_head（✅ 已实现）
Step 3: model.setup_audio_buffers(max_bs)              → 预分配 persistent buffer（🔜）
Step 4: init_device_graphs()                           → capture 完整 graph
```

### 5.5 显存影响

audio_head 参数约几十 MB。DepthTransformer 中间激活被 graph pool 锁定，但 4 层小 transformer + 8 个 head 的峰值显存远小于 AR backbone 单层 MLA/MoE 的激活。和当前已启用的 overlap schedule + CG 的 memory pool 共享，pool 只按最大 graph 的 high-water mark 分配。

---

## 步骤六：code2wav — audio token 解码为波形（🔜 待实现）

### 6.1 解码链路

```
audio codebook tokens [N, 8]
  ↓ audio_tokenizer.decode(codes, bridge_length)
  → flow matching mel spectrogram
  ↓ vocoder.decode(mel)  [HiFi-GAN]
  → PCM waveform 24kHz
```

### 6.2 权重加载

| 组件 | 加载方式 | 来源 |
|------|---------|------|
| audio de-tokenizer（flow matching） | `load_module(prefix="model.audio_tokenizer.audio_decoder.")` | safetensors 内 1740 keys |
| vocoder（HiFi-GAN） | `torch.load(path)` | `cosy24k_vocoder/hift.pt` |

### 6.3 流式架构

text_ar 每产出一个 audio frame（8 个 codebook token），通过 `stream_to` 推给 code2wav。code2wav 维护 ring buffer，攒够 K 帧后调用 de-tokenizer + vocoder 产出 PCM，通过 cross-fade 拼接。

### 6.4 可复用组件

- `stream_to` 基础设施（`StageConfig.stream_to` + `can_accept_stream_before_payload`）
- `StreamingSimpleScheduler` 的 inbox/outbox 模式
- `BatchVocoderBase`（`sglang_omni/scheduling/vocoder_base.py`）——vocoder 标准基类

### 6.5 和 Qwen3-Omni code2wav 的关系

可复用：
- `stream_to` 基础设施
- `StreamingSimpleScheduler`

不可复用（需重写/新建）：
- `_decode_incremental` 内部逻辑：Qwen3-Omni 接收单 codebook codec code 直接过 vocoder；LongCat-Next 需要先过 flow matching de-tokenizer（8 codebook → mel），再过 HiFi-GAN vocoder（mel → waveform）

---

## 步骤七：PipelineConfig 扩展（🔜 待实现）

```python
stages: list[StageConfig] = [
    # Phase 2 stages（保持不变）
    StageConfig(name="preprocessing", ...),
    StageConfig(name="image_encoder", ...),
    StageConfig(name="audio_encoder", ...),
    StageConfig(name="mm_aggregate", ...),

    # Phase 3: text_ar 新增 stream_to
    StageConfig(
        name="text_ar",
        factory=...,
        gpu=[2, 3, 4, 5],
        tp_size=4,
        stream_to=["code2wav"],   # 新增
        terminal=True,
    ),

    # Phase 3: 新增 code2wav
    StageConfig(
        name="code2wav",
        factory=...,
        gpu=6,
        can_accept_stream_before_payload=True,
        terminal=True,
    ),
]
```

---

## 文件清单

```
sglang_omni/models/longcat_next/
├── config.py                          # 改：新增 code2wav stage + text_ar stream_to
├── stages.py                          # 改：_attach_audio_head（✅） + create_code2wav_executor
├── sglang_model.py                    # 改：audio_head attr（✅） + forward 双头覆盖
├── model_runner.py                    # 改：采样状态机 + audio codebook buffer 管理
├── request_builders.py                # 改：新增 code2wav 结果适配
├── merge.py                           # 改：传递 stream metadata
├── components/
│   ├── audio_head.py                  # 新增（✅）
│   └── code2wav.py                    # 新增：LongCatNextCode2Wav
└── 开发文档_phase3_开发文档.md          # 本文档
```

---

## 关键 checkpoint 参数

来源：`config.json` → `audio_config`

| 参数 | 值 |
|------|----|
| `hidden_size` | 3072 |
| `audio_head_transformer_dims` | 3072 |
| `audio_head_transformer_ffn_scale` | 16 |
| `audio_head_transformer_layers` | 4 |
| `codebook_sizes` | `[8192, 4096, 2048, 1024, 1024, 1024, 1024, 1024]` |
| `audio_offset` | 131125 |
| `audio_head.*` keys | 71 |
| `model.audio_tokenizer.*` keys | 1740 |

---

## 参考

- 论文 3.2.4 节：Parallel and Serial Audio Generation
- 官方实现：`LongCat-Next-inference/modules/image_head.py`（OmniAudioHead）、`modules/output_processor.py`（depth_transformer_forward）
- sglang-omni 通用组件：`sglang_omni/scheduling/vocoder_base.py`（BatchVocoderBase）、`sglang_omni/scheduling/streaming_vocoder.py`（StreamingVocoderBase）
