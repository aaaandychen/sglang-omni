# sglang-omni

基于 [SGLang](https://github.com/sgl-project/sglang) 的原生多模态模型推理框架，支持文本、图像、音频的离散 token 统一自回归生成。

## 核心模型：LongCat-Next

首个完整支持 LongCat-Next 68B MoE 的开源推理框架，覆盖从多模态理解到实时语音输出的全链路。

### 能力矩阵

| 能力 | 状态 |
|---|---|
| AR backbone 文本生成 | ✅ |
| 图像理解（ViT + VQ + Bridge） | ✅ |
| 音频理解（Audio Encoder + Quantizer） | ✅ |
| 实时语音输出（双头 decode + HiFi-GAN vocoder） | ✅ |
| 性能优化（Encoder Cache / TensorRef / Overlap / CUDA Graph / Async Decode） | ✅ |
| 全双工对话 | 🔜 |

### 架构

```
preprocessing ─┬─ image_encoder(GPU0) ──┐
               ├─ audio_encoder(GPU1) ──┤
               └─ mm_aggregate ─────────┘
                         │
                    text_ar(GPU2-5, TP=4)
                    68B MoE + lm_head ──→ text tokens ──→ client
                              audio_head ──→ audio codes ──→ code2wav(GPU6)
                                                         flow matching + HiFi-GAN
                                                                  │
                                                            24kHz PCM ──→ client
```

### 实现过程

**Phase 1** — 将 LongCat-Next 的 68B MoE 骨架（`LongcatFlashLite A3B`, 14 层 3072-dim）封装为 SGLang 原生模型。解决 ngram embedding 参数推导（hash base 必须用 text_vocab 而非 full_vocab）、HF/SGLang 配置字段映射、多模态权重过滤等问题。

**Phase 2** — 实现图像和音频 encoder 的独立 GPU 进程隔离，通过 `replace_embeds` / `replace_positions` 机制在 prefill 阶段将 encoder 输出注入 AR backbone 的指定位置，支持 chunked prefill。

**Phase 2.5** — 性能优化：per-stage timing 可观测性、encoder 输出 LRU 缓存（path+size+mtime 指纹）、TensorRef SHM 大张量零拷贝 relay、overlap schedule CPU/GPU 并行、CUDA Graph decode 静态图、async decode 一步流水线延迟隐藏。

**Phase 3** — 端到端语音输出：重写官方 `OmniAudioHead`（4 层 `CasualDepthTransformer`，8 个 codebook position 的 causal flash attention，每步 argmax 采样），实现双头 forward（text embedding + audio codebook embedding 融合 → LLM → lm_head + audio_head 并行输出），状态机驱动 text↔audio 模式切换，code2wav stage（flow matching de-tokenizer 1740 keys + HiFi-GAN vocoder）解码为 24kHz PCM 波形。通过 `SGLANG_OMNI_LONGCAT_ENABLE_AUDIO_OUTPUT=1` 控制开关，关闭时零开销。

### 快速启动

```bash
# 纯文本
sgl-omni serve --config examples/configs/longcat_next_text.yaml

# 多模态输入理解
sgl-omni serve --config examples/configs/longcat_next_multimodal.yaml

# 多模态 → 实时语音输出
export SGLANG_OMNI_LONGCAT_ENABLE_AUDIO_OUTPUT=1
sgl-omni serve --config examples/configs/longcat_next_multimodal.yaml
```

### 展望：全双工

论文为全双工提供了完整的理论基础：DiNA 统一 discrete codebook space（输入输出音频共享同一 RVQ 词表）、pure audio modality（无文本引导的实时语音输入）、parallel generation（每步同时产出 text + audio tokens）、any-to-any interleaved generation（Section 6.2 未来方向）。模型权重无需重新训练。

工程上需将 pipeline 从「encoder 一次 → decode 全部」重构为「encoder 持续 streaming → decode 期间 interleaved prefill」，支持 barge-in 打断和低延迟闭环。纯推理工程问题。
