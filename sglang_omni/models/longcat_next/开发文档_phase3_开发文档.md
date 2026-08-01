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
| 流式输出 | 文本 stream to client | 文本 + audio codes → code2wav |

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
                                     └─→ audio codes → code2wav (GPU 6)
                                                          │
                                                          ├─ flow matching de-tokenizer
                                                          ├─ HiFi-GAN vocoder
                                                          └─→ PCM waveform
```

---

## 如何开启/关闭音频输出

音频输出通过环境变量控制，默认关闭（零额外开销）：

```bash
# 开启音频输出
export SGLANG_OMNI_LONGCAT_ENABLE_AUDIO_OUTPUT=1

# 关闭音频输出（默认）
unset SGLANG_OMNI_LONGCAT_ENABLE_AUDIO_OUTPUT
```

**关闭时（默认）**：
- `stages.py` 不创建 audio_head，`self.audio_head` 保持 `None`
- `sglang_model.py` forward 走 `super().forward()` 标准路径
- pipeline config 不注册 code2wav stage，不占用 GPU 6
- 和 Phase 2 行为字节级一致

**开启时**：
- audio_head 在 text_ar init 时创建并加载权重（~几十 MB 显存）
- decode 阶段状态机检测 `<audiogen_start>` token（131123），进入音频生成模式
- 每步同时产出文本 token + 8 个 audio codebook token
- code2wav stage 在 GPU 6 上按需解码 audio codes → PCM 波形

---

## 实现总览

### 关键组件

| 组件 | 文件 | 功能 |
|------|------|------|
| `_OmniAudioHead` | `components/audio_head.py` | audio head 主体（4 层 DepthTransformer + 8 个预测头） |
| `LongcatNextAudioHead` | `components/audio_head.py` | 顶层 wrapper：权重加载 + 8 步因果预测 + input embedding |
| `LongcatNextTextForCausalLM` | `sglang_model.py` | AR backbone：forward 增加 audio decode 路径 |
| `LongcatNextModelRunner` | `model_runner.py` | 状态机：检测 audio gen 起止、管理 prev_codes、融合 embedding |
| `LongcatNextCode2Wav` | `components/code2wav.py` | 音频解码：flow matching de-tokenizer + HiFi-GAN vocoder |
| Pipeline config | `config.py` | 条件注册 code2wav stage + text_ar stream_to |

### 每步解码流程

```
Step N: 文本生成中
  before_decode: longcat_audio_codes = None
  model.forward: super().forward()  (standard text-only)
  post_decode: text_token = "hello"
  → 检测到 text_token == audiogen_start_token_id (131123)
  → 切换到 AUDIO_MODE

Step N+1: 第一个音频步
  before_decode: longcat_audio_codes = zeros(8)  (首次，无历史 codes)
  model.forward:
    1. text_emb = embed_tokens(input_ids)
    2. audio_emb = audio_head.build_input_embedding(zeros)  → 0 contribution
    3. input_embeds = text_emb + audio_emb
    4. hidden = LLM(input_embeds)
    5. text_logits = lm_head(hidden)
    6. forward_batch.longcat_new_audio_codes = audio_head(hidden[:, -1])
  sample: text_token = "<audiotext_pad>"
  post_decode: 存储 codes，准备下一步

Step N+2: 第二个音频步
  before_decode: longcat_audio_codes = codes_from_step_N+1
  model.forward:
    1. audio_emb = audio_head.build_input_embedding(codes_N+1)  → 非零贡献
    2. input_embeds = text_emb + audio_emb  → LLM 获得声学上下文
    3-6. 同 N+1
  ...

Step N+K: 检测到 text_token == audiogen_end_token_id (131124)
  → 切换回 TEXT_MODE
  → 累计的 audio_codes_list 在 result_adapter 中输出给 code2wav
```

---

## 步骤一：audio_head 集成（✅ 已完成）

### 1.1 架构

audio_head 是独立于 AR backbone 的 nn.Module，在 text_ar stage 内部和 lm_head 共享同一个 hidden state，实现双头输出。

**设计决策：重新实现而非导入官方代码。** 官方 `OmniAudioHead` 包含 TP 支持代码，依赖 SGLang 内部模块。在非 TP 路径下，核心逻辑可以自包含实现，避免对 inference repo 的路径依赖。

### 1.2 组件层次

```
LongcatNextAudioHead           ← 顶层 wrapper
  ├── input_codebook_embedding  ← _OffsetCodebookEmbedding (8 codebook, sum → LLM input)
  ├── audio_emb_layers          ← 7 nn.Embedding (codebook 0..6, per-layer → audio_head internal)
  └── _OmniAudioHead            ← audio head 主体
        ├── _RMSNorm × 2        ← hidden_norm, headnorm
        ├── nn.Linear           ← hidden_proj
        ├── _CasualDepthTransformerLayer × 4
        │     ├── _FlashVarLenAttention  ← causal flash attention over 8 codebook positions
        │     ├── _RMSNorm × 2           ← layernorm1, layernorm2
        │     └── nn.Linear × 2          ← FFN (einsum depth-aware path)
        └── heads (nn.ModuleList[Linear]) × 8  ← 每 codebook 一个预测头
```

### 1.3 _OmniAudioHead 前向逻辑

```python
def forward(x, audio_tokens, audio_emb_layers, batch_size, codebook_id):
    # Step 1: 前 7 个 codebook 的历史预测 → embedding + cumsum
    cumsum_audio_embed = cumsum(stack([emb_layers[i](audio_tokens[:, i]) for i in 0..6]))

    # Step 2: LLM hidden state + cumsum embedding → [B, 8, hidden_size]
    #   pos 0: LLM hidden state（当前 step 的语义信息）
    #   pos 1..7: 前 i 个 codebook 的累计 embedding（音色/韵律 context）
    hidden_states = concat([x, cumsum_audio_embed], dim=1)

    # Step 3: norm → proj → 4 层 DepthTransformer → headnorm
    hidden_states = headnorm(transformer_layers(hidden_proj(hidden_norm(hidden_states))))

    # Step 4: 取 codebook_id 位置的 hidden，过对应的预测头
    return heads[codebook_id](hidden_states[:, codebook_id])  # [B, codebook_size + 1]
```

### 1.4 8 步因果预测循环

```python
# LongcatNextAudioHead.forward
next_codes = zeros(B, 8)
for codebook_id in 0..7:
    logits = audio_head(hidden_state, next_codes, audio_emb_layers, B, codebook_id)
    next_codes[:, codebook_id] = argmax(logits)
return next_codes
```

- 使用 `argmax`（greedy decoding），避免 `torch.multinomial` 的 host sync → CUDA Graph 兼容
- 后一次预测能看到前一次的预测结果（codebook 间因果依赖）

### 1.5 audio_emb_layers 权重来源

7 个 `nn.Embedding` 从 `model.embed_tokens.weight` 切片，起始偏移 `audio_offset = 131125`：

| 层 | 切片范围 | vocab 大小 |
|----|---------|-----------|
| Layer 0 | `[131125 : 131125+8192]` | 8192 |
| Layer 1 | `[139317 : 139317+4096]` | 4096 |
| ... | ... | ... |
| Layer 6 | `[148533 : 148533+1024]` | 1024 |

最后一个 codebook（index 7）不需要 embedding——它之后没有更高 index 的 codebook 需要 aggregate。

### 1.6 权重加载

| 组件 | 加载方式 | Keys |
|------|---------|------|
| `_OmniAudioHead` | `load_module(prefix="audio_head.")` | 71 |
| `audio_emb_layers` | `load_weights_by_prefix(prefix="model.embed_tokens.")` → 切片 | 1（共享） |
| `input_codebook_embedding` | 同 audio_emb_layers，切 8 层 | 1（共享） |

---

## 步骤二：输入端 embedding 融合（✅ 已完成）

`model_runner.before_decode` 为 audio 模式的请求设置 `forward_batch.longcat_audio_codes`。

`sglang_model.py` 的 `forward()` 检测到此 flag 后，调用 `audio_head.build_input_embedding(codes)` 将 8 codebook embedding 求和，加到 text embedding 上：

```python
if prev_audio_codes is not None and self.audio_head is not None:
    text_emb = self.model.embed_tokens(input_ids, forward_batch)
    audio_emb = self.audio_head.build_input_embedding(prev_audio_codes)
    input_embeds = text_emb + audio_emb.to(text_emb.dtype)
    hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
```

首次音频步的 `prev_audio_codes` 为 zeros → audio_emb 贡献为 0，行为等价于 text-only。

---

## 步骤三：双头 forward（✅ 已完成）

同上，在 embedding 融合之后：

```python
    # lm_head
    text_logits = self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch)
    # audio_head（side channel）
    forward_batch.longcat_new_audio_codes = self.audio_head(hidden_states[:, -1])
    return text_logits
```

audio codes 通过 `forward_batch.longcat_new_audio_codes` side channel 传递，model_runner 在 `post_decode` 中读取。

---

## 步骤四：状态机（✅ 已完成）

`model_runner.py` 中的简化状态机：

**状态转换：**
```
TEXT_MODE ── text_token == audiogen_start (131123) ──→ AUDIO_MODE
AUDIO_MODE:
  每步产出 text_token + 8 audio codes
  ── text_token == audiogen_end (131124) ──→ TEXT_MODE
```

**关键 hooks：**
- `before_decode`：为 audio 模式的请求设置 `forward_batch.longcat_audio_codes`
- `post_decode`：读取 `forward_batch.longcat_new_audio_codes`，更新状态，累积 codes 到 `req._longcat_audio_codes_list`
- `post_process_outputs`：附加累积的 audio codes 到最终输出

**混合 batch 支持：** text-only 请求的 `longcat_audio_codes` 设为 zeros（embedding 贡献为 0），audio_head 仍运行但结果被丢弃。

**结果传递：** `request_builders.result_adapter` 将 `req._longcat_audio_codes_list` 写为 `StagePayload.data["audio_codes"]`，供 code2wav stage 使用。

---

## 步骤五：CUDA Graph 兼容性（🔜 待实现）

当前 audio decode 路径走标准 forward（非 CG）。后续需改为始终双头 + persistent buffer 的方案以支持 CG capture。

---

## 步骤六：code2wav（✅ 已完成）

### 解码链路

```
audio codebook tokens [N, 8]
  ↓ audio_tokenizer.decode(codes, bridge_length=valid_len)
  → ret.flow_matching_mel → mel spectrogram [n_frames, 80]
  ↓ vocoder.decode(mel.transpose(0,1).float().unsqueeze(0))
  → PCM waveform [1, samples] 24kHz
```

### 权重加载

| 组件 | 加载方式 | 来源 |
|------|---------|------|
| audio de-tokenizer | `load_module(prefix="model.audio_tokenizer.")` | safetensors (1740 keys) |
| vocoder (HiFi-GAN) | `Cosy24kVocoder.from_pretrained(path)` | `cosy24k_vocoder/hift.pt` |

### 当前行为

- `create_code2wav_executor` 创建 `SimpleScheduler`，接收含 `audio_codes` 字段的 payload
- 批量解码所有 audio frames → 返回 `audio_waveforms` 列表
- 不支持流式解码（后续迁移到 `StreamingVocoderBase`）

---

## 步骤七：PipelineConfig 扩展（✅ 已完成）

`config.py` 中 `_audio_output_enabled()` 条件控制：

```python
# 开启时
text_ar.stream_to = ["code2wav"]
stages += [StageConfig(
    name="code2wav",
    factory="...stages.create_code2wav_executor",
    gpu=6,
    can_accept_stream_before_payload=True,
    terminal=True,
)]
```

---

## 文件变更清单

```
sglang_omni/models/longcat_next/
├── components/
│   ├── audio_head.py       ← 新增（370 行）
│   └── code2wav.py         ← 新增（155 行）
├── sglang_model.py         ← 改：audio_head attribute + forward audio decode 路径
├── model_runner.py         ← 改：状态机 before_decode/post_decode/post_process_outputs
├── stages.py               ← 改：audio 开关 + _attach_audio_head() + create_code2wav_executor()
├── config.py               ← 改：条件 code2wav stage + text_ar stream_to
├── request_builders.py     ← 改：result_adapter 输出 audio_codes
└── 开发文档_phase3_开发文档.md   ← 本文档
```

---

## 关键参数

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
| `audiogen_start_token_id` | 131123 |
| `audiogen_end_token_id` | 131124 |

---

## 参考

- 论文 3.2.4 节：Parallel and Serial Audio Generation
- 官方实现：`LongCat-Next-inference/modules/image_head.py`（OmniAudioHead）、`modules/output_processor.py`（depth_transformer_forward）、`processor/decoder/audio_decode.py`（decode_wave_vocoder）
- sglang-omni 通用组件：`sglang_omni/scheduling/vocoder_base.py`（BatchVocoderBase）、`sglang_omni/scheduling/streaming_vocoder.py`（StreamingVocoderBase）
