# LongCat-Next Phase 3：语音输出

## 概述

Phase 2 实现了多模态输入理解（image + audio + text → text）。Phase 3 的目标是实现语音输出（image + audio + text → text + speech），使模型具备"说话"能力。

LongCat-Next 的语音输出基于 DiNA 范式下的 discrete audio codebook token——AR backbone 在 decode 阶段交替产出文本 token 和 audio codebook token。audio token 经 audio de-tokenizer（flow matching）和 vocoder（HiFi-GAN）解码为 PCM 波形。

---

## 步骤一：架构总览

### 1.1 和 Phase 2 的差异

| | Phase 2 | Phase 3 |
|---|---|---|
| AR 输出 | 仅文本 token | 文本 token + audio codebook token |
| model forward | decode 时 `super().forward()` 标准路径 | decode 时覆盖 forward：双头 logits + embedding 融合 |
| 采样 | 单头 `lm_head` sample | 双头 `lm_head` + `audio_head` sample，状态机驱动 |
| 新增 stage | image_encoder + audio_encoder | + code2wav（audio de-tokenizer + vocoder） |
| 流式输出 | 文本 stream to client | 文本 stream + audio frame stream_to → code2wav |

### 1.2 数据流

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

## 步骤二：输入端 — decode 阶段的 embedding 融合

### 2.1 当前行为

Phase 2 decode 每步的 embedding 来源：

```python
# 当前
input_embeds = embed_tokens(prev_text_token)
```

### 2.2 Phase 3 行为

Parallel 模式下每个 decode step 同时产出文本 token 和 audio codebook token。下一步的 input embedding 需要融合两者：

```python
# Phase 3
input_embeds = embed_tokens(prev_text_token) + audio_embed(prev_audio_codes)
```

`audio_embed` 将 8 个 codebook id 分别从 `embed_tokens` 的 audio 区间查表再求和。和 Phase 2 encoder 中 `_OffsetCodebookEmbedding` 逻辑完全一致——区别仅调用时机从 prefill 移到了 decode 循环。

### 2.3 和 replace_embeds 的关系

Phase 2 的 `replace_embeds` 用于 **prefill 阶段**注入外部 encoder 输出（视觉/音频理解）。Phase 3 的 embedding 融合用于 **decode 阶段**当前 step 的输入构造。两者互不影响。

### 2.4 实现位置

在 `LongcatNextTextForCausalLM` 中新增一个方法，替代 `_build_longcat_input_embeds` 中 decode 路径的纯文本 embedding：

```python
def _build_decode_input_embeds(self, input_ids, forward_batch):
    # 纯文本 token embedding
    input_embeds = self.model.embed_tokens(input_ids, forward_batch)

    # 如果是 audio 生成模式，融合 audio codebook embedding
    audio_codes = getattr(forward_batch, "longcat_audio_codes", None)
    if audio_codes is not None:
        audio_embeds = self._audio_codebook_embedding(audio_codes)
        input_embeds = input_embeds + audio_embeds.to(input_embeds.dtype)

    return input_embeds
```

---

## 步骤三：输出端 — 双头 logits 和 audio_head

### 3.1 audio_head 结构

`audio_head` 的权重在 checkpoint 中（71 keys），结构如下：

```
hidden_state [B, 3072]
  ↓ concat(cumsum(prev_audio_embeddings)) → [B, 8, transformer_dims]
  ↓ 4× CasualDepthTransformerLayer（codebook 维度因果注意力）
  ↓ headnorm
  ↓ heads[codebook_id](hidden[:, codebook_id]) → [B, codebook_size + 1]
```

`CasualDepthTransformerHead` 一次只输出一个 codebook 的 logits，外部循环 8 次：

```python
for level in range(num_codebooks):
    logits = audio_head(hidden, prev_audio_codes, audio_emb_layers, bs, level)
    next_token_ids[:, level] = argmax(logits)
    # 下一个 level 的 attention 能看到 level 0..level-1 的预测结果
```

对于 CUDA Graph 兼容性，这个循环需要封装在 model forward 内部（作为 graph 的一部分），不能放在 `process_batch_result` 中逐次调用。

### 3.2 双头输出

同一个 hidden state 同时过两个 head：

```python
hidden = Transformer(input_embeds)     # [B, 1, 3072]
text_logits = lm_head(hidden[:, -1])    # [B, 131125]

# audio_head: 8 步因果采样，封装在 model forward 内部
for level in range(8):
    audio_logits_level = audio_head(hidden, prev_codes, emb_layers, bs, level)
    next_codes[:, level] = argmax(audio_logits_level)
```

### 3.3 模型加载

官方实现中，视觉和音频 head 共用 `CasualDepthTransformerHead` 类（`modules/image_head.py` 中的 `OmniImageHead` / `OmniAudioHead`），仅参数不同。Phase 3 新建 `LongCatNextAudioHead` wrapper 封装 8 步循环和 persistent buffer 管理：

```python
# 权重加载
audio_head = OmniAudioHead(  # 来自官方 modules/image_head.py
    hidden_size=hidden_size,
    codebook_sizes=audio_codebook_sizes,
    audio_head_transformer_ffn_scale=...,
    audio_head_transformer_dims=...,
    audio_head_transformer_layers=...,
    audio_head_enable=True,
)
load_module(audio_head, model_path, prefix="audio_head.")
```

---

## 步骤四：采样端 — 延迟状态机

### 4.1 Parallel 模式（论文默认）

同一个 forward step 中 `lm_head` 和 `audio_head` 同时采样。论文 3.2.4 节：

> *"In parallel internal text-guided generation, each decoding step simultaneously generates both text and audio outputs."*

官方实现通过 `delay` 参数控制，推理时 `delay=0` 即为 parallel：

```
GEN_AUDIO_STAGE:
  Step 0: text_token_0 + audio_codes_0    ← 同时产出
  Step 1: text_token_1 + audio_codes_1
  ...
  Step N: text_token_N + audio_codes_N
```

### 4.2 状态机

```
TEXT_MODE ──→ 检测到 <audio_gen_start> → AUDIO_MODE
AUDIO_MODE:
  lm_head 采样 text token
  audio_head 采样 8 个 codebook token（argmax，避免 host sync）
  如果 token == <audio_gen_end> → TEXT_MODE（或 EOS）
```

### 4.3 为什么用 argmax 而不是 top-k/top-p

和其他 CG 文章分析一致：音频 token 采样在 CUDA Graph 内部执行，必须避免 `torch.multinomial`（需要 CPU sync）。官方 TTS 场景使用 `torch.argmax(biased_logits, dim=-1)` greedy decoding。

### 4.4 audio codebook 的因果结构

audio_head 内部的 DepthTransformer 在 codebook 深度维度做因果自注意力。外部调用不需要 8 步循环——一次 `audio_head(hidden, prev_audio_codes)` 返回全部 8 个 codebook 的 logits，内部 in-place 填充。

```python
# audio_head 内部（简化）
for i in range(num_codebooks):
    logits = heads[i](transformer_output[:, i])
    next_token_ids[:, i] = argmax(logits)
    # 下一个 codebook 的 attention 能看到 codebook_0..i
```

---

## 步骤五：CUDA Graph 兼容性

Phase 2 的 decode 路径是干净的 `super().forward()`——SGLang 标准 LLM decode，CUDA Graph 捕获无额外适配成本。

Phase 3 的 decode 路径变了：embedding 融合、audio_head forward（含 8 步 codebook 因果循环）、双头输出。这三项必须出现在 capture 的 graph 里，否则 graph replay 时执行的是不包含 audio 输出的残缺 forward。这引入了一组需要明确解决的问题。

### 5.1 统一 Graph 策略：始终双头，buffer 区分模式

Graph 要求静态控制流——capture 时录了什么，replay 时就执行什么。不能按模式（text-only / audio）切换 graph。

策略：**始终运行双头，通过 persistent buffer 的值控制有效输出**。

```python
def forward(input_ids, positions, forward_batch):
    # Step 1: embedding 融合（始终执行）
    text_emb = embed_tokens(input_ids, forward_batch)
    audio_codes = forward_batch.longcat_audio_codes  # persistent buffer
    audio_emb = audio_codebook_embedding(audio_codes) # text-only 时 audio_codes 全零 → audio_emb 为零
    input_embeds = text_emb + audio_emb

    # Step 2: Transformer（始终执行）
    hidden = self.model(input_ids, positions, forward_batch, input_embeds)

    # Step 3: 双头（始终执行）
    text_logits = self.lm_head(hidden)
    audio_logits = self.audio_head(hidden, audio_codes)  # text-only 时输出被丢弃

    return text_logits, audio_logits
```

text-only 模式下 `longcat_audio_codes` buffer 被 `before_prefill` 清零，audio_head 仍执行但结果在 `process_batch_result` 里被忽略。graph 永远是同一张——embedding 融合 + Transformer + lm_head + audio_head 的完整 DAG。

### 5.2 Persistent Buffer 设计

参照 S2-Pro 文章的 pattern。所有跨 step 变动的数据通过 `copy_()` / `fill_()` 就地更新，不分配新 tensor：

```python
# 在 setup 阶段预分配（model.__init__ 或专门的 setup_audio_buffers）
self.register_buffer("_audio_codes",
    torch.zeros(max_bs, num_codebooks, dtype=torch.long))
self.register_buffer("_audio_mask",
    torch.zeros(max_bs, dtype=torch.bool))
self._audio_codebook_embedding = _OffsetCodebookEmbedding(
    model_path=model_path,
    offset=audio_offset,
    codebook_sizes=audio_codebook_sizes,
    ...
)
```

Buffer 生命周期：

| 步骤 | 函数 | 操作 |
|------|------|------|
| forward 前 | `before_prefill` / `_update_audio_buffers` | `_audio_codes[:bs].copy_(prev_codes)`，`_audio_mask[:bs].fill_(is_audio_step)` |
| forward 内 | model.forward | 读取 `_audio_codes`、`_audio_mask`（地址不变，值变了） |
| forward 后 | `process_batch_result` | argmax `audio_logits`，`copy_` 回 `_audio_codes` 供下一步 |

### 5.3 audio_head 内部的静态约束

`audio_head.forward` 每次调用只输出一个 codebook 的 logits，8 步循环封装在 model forward 内部（作为 graph 的一部分）：

```python
# model forward 内（CUDA Graph 捕获此循环）
prev_codes = forward_batch.longcat_audio_codes  # persistent buffer
for level in range(num_codebooks):               # 静态循环（capture 时固化 = 8）
    logits = audio_head(hidden, prev_codes, audio_emb_layers, bs, level)
    prev_codes[:, level] = torch.argmax(logits, dim=-1)  # in-place 更新，无 host sync
    # 下一个 level 的 DepthTransformer attention 看到 codebook_0..level
```

- `range(8)` 在 capture 时固化为常量——"静态控制流"约束满足
- `argmax` 是 GPU 原生操作，不涉及 host-device sync——第三条约满足
- `prev_codes` 的 in-place 更新不分配新 tensor——"指针稳定"约束满足
- codebook 之间的因果依赖：DepthTransformer 内部使用 `causal=True` 的 flash attention mask 实现

### 5.4 Deferred Capture

audio_head 的权重加载和 buffer 分配必须在 CUDA Graph capture 之前完成：

```
create_longcat_next_text_executor:
  Step 1: create_sglang_infrastructure_defer_cuda_graph → 模型创建（disable_cuda_graph=True）
  Step 2: load_module(audio_head, prefix="audio_head.")  → 加载 audio_head 权重
  Step 3: model.setup_audio_buffers(max_bs)               → 预分配 persistent buffer
  Step 4: model.setup_audio_codebook_embedding(...)       → 挂载 codebook embedding
  Step 5: init_device_graphs()                            → capture 完整 graph
```

Step 2-4 必须在 Step 5 之前完成。当前 `create_sglang_infrastructure_defer_cuda_graph` 已经返回了 `want_cuda_graph` 和 model，在 `init_device_graphs()` 之前插入初始化即可。Phase 2 的 vocab_size fix 就是在这个位置执行的——模式一致。

### 5.5 显存影响

audio_head 参数约 71 keys（几十 MB）。DepthTransformer 中间激活被 graph pool 锁定，但 4 层小 transformer + 8 个 head 的峰值显存远小于 AR backbone 单层 MLA/MoE 的激活。和当前已启用的 overlap schedule + CG 的 memory pool 共享，pool 只按最大 graph 的 high-water mark 分配一次，audio_head 的额外激活不会导致 pool 膨胀。

### 5.6 和 Phase 2.5 CG + overlap + async_decode 的关系

当前已启用：
- `disable_cuda_graph=False`（SGLang 标准 LLM decode graph）
- `disable_overlap_schedule=False`
- `enable_async_decode=True`（OmniScheduler 层）

Phase 3 的改动集中在 model forward 内部（步骤 2-4），不接触 OmniScheduler 事件循环。overlap 和 async_decode 不受影响。唯一变化是 CG capture 的内容从单头变成了双头——需要在 capture 完成后重新验证 `init_device_graphs()` 成功，确认新增的 audio_head 路径无动态操作导致 capture 失败。

---

## 步骤六：code2wav — audio token 解码为波形

### 5.1 解码链路

```
audio codebook tokens [N, 8]
  ↓ audio_tokenizer.decode(codes, bridge_length)
  → flow matching mel spectrogram
  ↓ vocoder.decode(mel)  [Cosy24kVocoder / HiFi-GAN]
  → PCM waveform 24kHz
```

### 5.2 权重加载

| 组件 | 加载方式 | 来源 |
|------|---------|------|
| audio de-tokenizer（flow matching） | `load_module(prefix="model.audio_tokenizer.audio_decoder.")` | safetensors 内 1740 keys |
| vocoder（HiFi-GAN） | `Cosy24kVocoder.from_pretrained(path)` | `cosy24k_vocoder/hift.pt` |

两个文件都在 checkpoint 目录内，不需要额外下载。

### 5.3 流式架构

text_ar 每产出一个 audio frame，通过 `stream_to` 推给 code2wav：

```python
# code2wav 内部维护 ring buffer
buffer.append(audio_frame)     # 每个 frame = 8 个 codebook id
if len(buffer) >= min_frames:  # 攒够 K 帧（建议 ≥ 8 帧 ≈ 640ms）
    segment = cat(buffer)      # [K, 8]
    mel = audio_tokenizer.decode(segment, bridge_length=[K])
    wave = vocoder.decode(mel)
    stream_to_client(wave, overlap=1200)  # 50ms cross-fade
    buffer = buffer[K - overlap_frames:]
```

### 5.4 TTFA vs 音质 trade-off

| 缓冲帧数 | TTFA | 音质 |
|---------|------|------|
| 1-3 帧（80-240ms） | 极低 | 不可用，flow matching 无足够 context |
| 4-8 帧（320-640ms） | 低 | 勉强可用，段边界不自然 |
| 10-20 帧（0.8-1.6s） | 中等 | 接近全量解码质量 |
| 全量 | 无流式 | 最优 |

`wave_concat_overlap` cross-fade 可以在波形层缓解段边界的不连续，但无法弥补 flow matching 缺乏全局 context 导致的 mel 质量下降。

### 5.5 和 Qwen3-Omni code2wav 的关系

可复用部分：
- `stream_to` 基础设施（`StageConfig.stream_to` + `can_accept_stream_before_payload`）
- `StreamingSimpleScheduler` 的 inbox/outbox 模式

不可复用部分（需重写/新建）：
- `_decode_incremental` 内部逻辑：Qwen3-Omni 接收单 codebook codec code 直接过 vocoder；LongCat-Next 需要先过 flow matching de-tokenizer（8 codebook → mel spectrogram），再过 HiFi-GAN vocoder（mel → waveform），是两个阶段
- 工厂函数命名：建议使用 `create_longcat_next_code2wav_executor`，区别于 Qwen3-Omni 的 `create_code2wav_scheduler`

---

## 步骤七：PipelineConfig 扩展

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
        factory=f"{_PKG}.stages.create_code2wav_executor",
        gpu=6,
        can_accept_stream_before_payload=True,
        terminal=True,
    ),
]
```

---

## 步骤八：GPU 分配（单机 8 卡 H800）

| GPU | Stage | 说明 |
|-----|-------|------|
| 0 | image_encoder | ViT+VQ+Bridge，~1-1.5B 参数 |
| 1 | audio_encoder | AudioEncoder+Quantizer，~0.5B |
| 2-5 | text_ar (TP=4) | AR backbone 68B MoE + audio_head |
| 6 | code2wav | flow matching + HiFi-GAN vocoder |
| 7 | 预留 | 未来扩展 |

---

## 文件清单（预计新增/修改）

```
sglang_omni/models/longcat_next/
├── config.py                          # 改：新增 code2wav stage + text_ar stream_to
├── stages.py                          # 改：新增 create_code2wav_executor
├── sglang_model.py                    # 改：model forward 支持双头输出 + embedding 融合
├── model_runner.py                    # 改：采样状态机 + audio codebook buffer 管理
├── request_builders.py                # 改：新增 code2wav 结果适配
├── merge.py                           # 改：mm_aggregate 负责传递 stream metadata
├── components/
│   ├── audio_head.py                  # 新增：LongCatNextAudioHead（包装 OmniAudioHead）
│   └── code2wav.py                    # 新增：LongCatNextCode2Wav（de-tokenizer + vocoder）
└── 开发文档_phase3.md                  # 本文档
```

---

## 附录：关键论文依据

论文 3.2.4 节 — Semantic Comparison of Parallel and Serial Audio Generation：

> *"In parallel internal text-guided generation, each decoding step simultaneously generates both text and audio outputs. To this end, we propose a random delay-based unified modeling paradigm for internal language guidance."*

- **Parallel**：每个 decode step 同时产出 text + audio codebook token。同一个 hidden state 过两个 head。
- **Serial**：先完整生成文本，再切换到 audio 模式生成全部音频 token。
- **随机延迟训练**：训练时在文本结束到音频开始间插入随机延迟（`delay`），模型学会两种模式。推理时 `delay=0` 即为 parallel（流式语音）。
- **统一 checkpoint**：同一个权重同时支持两种模式，不需要独立训练。

---

## 实施记录（2026-08-01）

### 当前状态

✅ 端到端语音输出已跑通。支持 `modalities: ["text", "audio"]` 请求，并行模式（delay=0）从第一个 decode step 同时产出文本和音频 codebook token，经 code2wav 解码为 PCM 波形返回客户端。

⏳ 流式音频输出尚未实现——当前为 batch 模式，音频 codes 全部攒完后一次性送 code2wav 解码。

### 遇到的问题与修复

#### 问题 1：`_longcat_audio_state` 未初始化

**现象**：`code2wav _decode: audio_codes=None` — 音频 token 从未生成。

**根因**：`post_decode` 中 `_longcat_audio_state` 初始化为 `{"mode": "text"}`，模型始终在 text mode 跑，不会产出 audio token。LongCat-Next 的 parallel 模式（delay=0）要求从第一个 decode step 就同时产出文本+音频 token。

**修复**：`request_builders.py` 中读取 `params["output_modalities"]`，若包含 `"audio"` 则初始化 `req._longcat_audio_state = {"mode": "audio", "prev_codes": None}` 和 `req._longcat_audio_codes_list = []`。

#### 问题 2：`output_modalities` 未到达 text_ar

**现象**：`output_modalities=['text']` 而非 `['text', 'audio']`。

**根因**：API 将 `output_modalities` 放入 `OmniRequest.metadata`，但 `request_builders.py` 从 `payload.request.params` 读取——`params` 只含 sampling 参数，不含 `output_modalities`。

**修复**：`client.py` `_build_omni_request` 中同时将 `output_modalities` 写入 `params["output_modalities"]`。

#### 问题 3：`post_decode` 无法读取 output_ids

**现象**：`output_ids=False` — `ScheduleBatch.output_ids` 为 None。

**根因**：`output_ids` 是 SGLang `Req` 对象的属性（per-request），不在 `ScheduleBatch` 上。

**修复**：`model_runner.py` 中从 `req.output_ids[-1]` 读取当前 step 的 token。

#### 问题 4：`outputs[rid].extra` 为 None

**现象**：`'NoneType' object does not support item assignment`。

**根因**：`post_process_outputs` 中 `outputs[rid].extra["audio_codes"]` 赋值时 `extra` 可能为 None。

**修复**：None guard，先初始化空 dict。

#### 问题 5：code2wav tensor 维度不匹配

**现象**：`too many indices for tensor of dimension 2`。

**根因**：`audio_codes` 是 `[N, 8]`，`code2wav.decode()` 期望 `[batch, N, 8]`。

**修复**：`stages.py` 中 `unsqueeze(0)` 加 batch 维度。

#### 问题 6：CPU/GPU device 不匹配

**现象**：`Expected all tensors to be on the same device, but found at least two devices, cuda:6 and cpu!`。

**根因**：`audio_codes` 在 CPU 上，`code2wav` 在 GPU 6。

**修复**：`audio_codes.to(device=device)`。

#### 问题 7：code2wav dtype 不匹配（float32 vs bfloat16）

**现象**：`Input type (float) and bias type (c10::BFloat16) should be the same`。

**根因**：code2wav 模型权重是 bfloat16，但 `audio_tokenizer.decode` 和 `vocoder.decode` 的输入为 float32。

**修复**：`code2wav.decode` 整体包 `torch.amp.autocast("cuda", dtype=self._dtype)`，mel 输入也改为 `.to(self._dtype)` 而非 `.to(torch.float32)`。

#### 问题 8：msgpack 无法序列化 Tensor

**现象**：`TypeError: can not serialize 'Tensor' object` — code2wav 返回时 relay 序列化失败。

**根因**：(a) waveform 是 GPU tensor，msgpack 无法序列化；(b) `result = dict(state)` 把原始 `audio_codes` tensor 也带进了返回 dict。

**修复**：(a) waveform 转 `numpy → tobytes()`，附带 `dtype` + `shape`；(b) result 只取 `text/modality/usage` 三个安全字段。

#### 问题 9：vocoder 输出缩进错误

**现象**：`results.append(wav.cpu())` 在 for 循环外执行。

**根因**：autocast 重构时缩进错位。

**修复**：修正缩进。

### 改动文件清单

| 文件 | 改动 |
|------|------|
| `client/client.py` | `_build_omni_request` 将 `output_modalities` 写入 `params` |
| `request_builders.py` | 读取 `params["output_modalities"]` 初始化 `_longcat_audio_state` |
| `model_runner.py` | `post_decode` 从 `req.output_ids[-1]` 读 token；`extra` None guard |
| `stages.py` | `_decode` 完整重写：dim fix、device fix、tensor→bytes 序列化 |
| `code2wav.py` | autocast 包裹解码；mel dtype fix；缩进修复 |

### 当前 API 用法

```bash
# Parallel 模式（delay=0）：文本+语音同时输出
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/path/to/LongCat-Next","messages":[{"role":"user","content":"Say hello."}],"modalities":["text","audio"],"max_tokens":64}'

# 流式版本
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/path/to/LongCat-Next","messages":[{"role":"user","content":"Say hello."}],"modalities":["text","audio"],"max_tokens":64,"stream":true}'
```

### 待完成

- **OmniScheduler 发射 audio stream item**：当前 `stream_to` 配置已就绪，但 OmniScheduler 的 `_emit_stream_output` 回调尚未接入——text_ar 每步 decode 产出的 audio codes 没有推送到 code2wav 的 stream inbox。需要实现 `_stream_output_builder` 回调，在 `post_decode` 后将每帧 audio codes 打包成 `StreamItem` 通过 outbox 发送。

---

## 流式音频输出设计

### 架构

```
text_ar (GPU 2-5)                             code2wav (GPU 6)
  │                                               │
  │ decode step 1: text_tok + audio_codes[0]     │
  │ ──── stream_to ──── StreamItem ────────────→ │ buffer.append(codes[0])
  │                                               │ [frames < min_frames: skip]
  │ decode step 2: text_tok + audio_codes[1]     │
  │ ──── stream_to ──── StreamItem ────────────→ │ buffer.append(codes[1])
  │                                               │ [frames >= min_frames: decode → stream waveform]
  │ ...                                           │ ...
  │ decode step N: text_tok + audio_codes[N]     │
  │ ──── batch payload ────────────────────────→ │ final decode remaining → complete
```

### Config 改动

```python
# text_ar: 新增 stream_to
stream_to=["code2wav"]  # 每帧 audio codes 实时推送

# code2wav: 新增 can_accept_stream_before_payload
can_accept_stream_before_payload=True  # 支持在收到完整 payload 前处理 stream items
```

### code2wav 流式 Scheduler

`stages.py` 中 `create_code2wav_executor` 改用 `StreamingSimpleScheduler`：
- `compute_fn`：处理单个 stream item（audio frame），增量解码
- `batch_compute_fn`：处理最终 payload，解码剩余帧

非流式请求照旧走 batch 路径，框架通过 `is_streaming_payload` 自动区分。

- **统一 checkpoint**：同一个权重同时支持两种模式，不需要独立训练。
