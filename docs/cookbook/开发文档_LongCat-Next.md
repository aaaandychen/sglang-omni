# LongCat-Next on sglang-omni 开发文档

> 本文是 LongCat-Next 接入 sglang-omni 的唯一权威开发文档，合并并取代原
> Phase 1–5 系列文档。内容以当前代码为准；涉及性能数字处标注验证状态。

---

## 目录

- [1. 概述](#1-概述)
- [2. 快速开始](#2-快速开始)
- [3. 总体架构](#3-总体架构)
- [4. AR Backbone（Phase 1）](#4-ar-backbonephase-1)
- [5. 多模态输入（Phase 2）](#5-多模态输入phase-2)
- [6. 性能优化（Phase 2.5 / 4）](#6-性能优化phase-25--4)
- [7. 语音输出（Phase 3）](#7-语音输出phase-3)
- [8. 实时语音对话（Phase 4/5）](#8-实时语音对话phase-45)
- [9. 环境适配与已知坑](#9-环境适配与已知坑)
- [10. 环境变量速查](#10-环境变量速查)
- [11. 测试与验证](#11-测试与验证)
- [12. 已知限制与后续方向](#12-已知限制与后续方向)
- [13. 文件索引](#13-文件索引)

---

## 1. 概述

LongCat-Next 是美团 LongCat 团队的原生多模态离散 token 模型（DiNA 范式）：
文本、图像、音频统一为离散 token，由单一 AR backbone 做 Next Token
Prediction，另挂双生成头（visual_head / audio_head）做模态输出。

关键架构事实：

| 项 | 值 |
|---|---|
| 总参数量 | ~68.5B（A3B MoE，激活 ~3B/token） |
| Backbone | 与 SGLang 原生 LongCat-Flash 同构（MLA + MoE），仅维度不同 |
| hidden_size / 层数 | 3072 / 14（每层 2×MLA + 2×Dense MLP + 1×MoE） |
| MoE | 256 routed + 128 zero experts，top-12 |
| 上下文 | 131072 |
| 音频输出 | 8 个 codebook 的因果 DepthTransformer（audio_head）+ code2wav（flow matching + HiFi-GAN） |
| 对话模式 | **turn-based**：输出状态机无「听」状态，架构上不支持真全双工（官方 Omni-Flow 同样只有 cancel 式 barge-in） |

接入分五个 Phase 完成：纯文本 AR（P1）→ 多模态输入（P2）→ 性能（P2.5）→
语音输出（P3）→ offload 与全双工对话（P4）→ 伪全双工延迟优化（P5）。

---

## 2. 快速开始

```bash
# 多模态 pipeline（文/图/音输入，文本输出）
sgl-omni serve \
    --config examples/configs/longcat_next_multimodal.yaml \
    --model-path /path/to/LongCat-Next \
    --host 0.0.0.0 --port 8100

# 开启语音输出（audio_head + code2wav，默认关闭）
SGLANG_OMNI_LONGCAT_ENABLE_AUDIO_OUTPUT=1 sgl-omni serve ...

# 纯文本单 stage 模式
sgl-omni serve --config examples/configs/longcat_next_text.yaml ...
```

冒烟验证（三条路径都应返回连贯内容，usage 非零）：

```bash
# 文本
curl -s localhost:8100/v1/chat/completions -H 'Content-Type: application/json' -d \
  '{"model":"...","messages":[{"role":"user","content":"你好"}],"max_tokens":50}'
# 图片 / 音频：在请求中加 "images":[...] 或 "audios":[...]（本地路径或 data URI）
```

实时语音对话走 `/v1/realtime` WebSocket（OpenAI Realtime 协议子集），前端参考
`playground/qwen-omni/realtime/app.js`。

---

## 3. 总体架构

```
Request
  │
  ▼
preprocessing (CPU)           tokenize + 图像/音频特征抽取 + cache_key 生成
  ├─ project ──→ image_encoder (GPU0)   ViT → VQ/RVQ → bridge → visual_embeds
  ├─ project ──→ audio_encoder (GPU1)   audio tokenizer → bridge → audio_embeds
  └─ project ──→ mm_aggregate (CPU)     input_ids + positions + metadata
                  │  wait_for + merge_fn 做 fan-in（按实际模态跳过空 encoder）
                  ▼
text_ar (GPU2-5, TP=4)        AR backbone；prefill 时按 replace_positions
                              scatter 注入多模态 embedding；decode 双头采样
                  │ stream_to（开启语音输出时）
                  ▼
code2wav (GPU6)               攒帧 → flow matching → HiFi-GAN → PCM 24kHz
```

设计原则：

- **AR stage 保持单纯**：只做 continuous batching / KV cache / decode loop；
  多模态编排全部放在轻量 CPU stage（mm_aggregate），不占 GPU。
- **fan-out / fan-in 由框架驱动**：`StageConfig.next` + `project_payload`
  （按需分发子集），`wait_for` + `wait_for_fn` + `merge_fn`（合并上游）。
- **各组件按需从同一 checkpoint 加载**，不做权重抽取。各组件权重前缀：
  `model.layers.*`（AR）、`model.visual_tokenizer.*`、`model.audio_tokenizer.*`、
  `visual_head.*`、`audio_head.*`、`model.ngram_embeddings.*`。

---

## 4. AR Backbone（Phase 1）

### 4.1 词表不对称（最关键的适配）

| 尺寸 | 名称 | 用途 |
|---|---|---|
| 131072 | `text_vocab_size` | 纯文本 token；**ngram 哈希基数** |
| 131125 | text + 53 个多模态特殊标记 | `embed_tokens` / `lm_head` 行数 |
| 282624 | `vocab_size` | 完整词表（`audio_offset=131125`，`visual_offset=150581`） |

多模态**内容** token（≥131125）不走 embed_tokens 查表，由 encoder 编码后以
`replace_embeds` 注入。`LongcatNextTextForCausalLM` 在 `super().__init__()` 前
把 `config.vocab_size = 131125`，`load_weights` 对 embed_tokens/lm_head 做
`[:131125]` 防御性截断，并过滤多模态前缀权重。

### 4.2 NgramEmbedding（三处适配，缺一不可）

输入 embedding 是 word embedding + 12 个 n-gram 投影共 13 项取均值，**不是**
普通 embedding。缺任何一处适配都会乱码或崩溃：

1. **config 派生**（`model_worker._apply_arch_override`）：在 ModelConfig 之后、
   model_runner 构造之前派生 `use_ngram_embedding=True`、`m = int(78 × 131072)
   = 10223616`、`n=4`、`k=4`。`m` 必须用 text_vocab=131072，否则形状断言失败。
2. **oe_weights 哈希基数修正**（`sglang_model.py`）：遍历 NgramEmbedding 实例，
   用 base=131072 重算 `oe_weights`（词表行数与哈希基数解耦，与官方
   `FusedOverEmbedding.num_embeddings_text` 一致）。
3. **OmniScheduler 运行时管线**（`omni_scheduler.py`）：补
   `_maybe_prepare_ngram_embedding(batch)` 填 `batch.ne_token_table`。
   **陷阱：该方法每个分支都必须 `return batch`**——漏写会让 stock
   `get_next_batch_to_run` 拿到 None，batch 被静默丢弃，症状是请求永远卡
   waiting_queue、GPU 0%、无任何报错。

### 4.3 MoE zero-expert 越界修复

router 输出 384 维（256 routed + 128 zero-expert），SGLang 把 zero-expert 的
`topk_idx` 置 -1，但 `fused_moe_kernel` 的 -1 保护在纯 TP 下被
`filter_expert=False` 短路 → 越界崩溃。**修复**：`sglang_model.py` 在模块加载时
patch `LongcatFlashMoE.forward`，把 -1 clamp 为 0（这些位置 combine 权重已为
0，数值等价）。

### 4.4 其他适配

- **config 字段映射**：`ffn_hidden_size→intermediate_size`、
  `expert_ffn_hidden_size→moe_intermediate_size`、`num_layers→num_hidden_layers`，
  必须无条件覆盖（SGLang 默认值会让 `hasattr` 误判）。
- **多模态 token 抑制**：纯文本推理时把 token 131072–131124 的 logits 置 -inf。
- TP 通过 `StageConfig.gpu=[...] + tp_size` 配置。

---

## 5. 多模态输入（Phase 2）

### 5.1 replace_embeds 注入机制

不能用官方「input_embeds 整段替换」：ngram embedding 需要先对文本部分算
ngram 嵌入，再把多模态 embedding **按位置 scatter 覆盖**。因此约定：

- preprocessing 计算 `image_positions` / `audio_positions`（pad token 占位区间）；
- encoder 产出 `visual_embeds` / `audio_embeds`（AR hidden_size 维）；
- `model_runner.before_prefill` 把二者组装成
  `forward_batch.longcat_replace_embeds / longcat_replace_positions`，
  支持 chunked prefill 按 chunk 区间切片（`_longcat_mm_consumed` 记录进度）；
- model forward 内先算文本 ngram embedding，再按 positions 覆盖。

嵌入计数严格校验：chunk 内 embeds 数量与 positions 数不一致直接抛错。

### 5.2 数据结构

- `LongcatNextPipelineState`：跨 stage 的 payload 状态（`encoder_inputs` /
  `encoder_outs` / positions / metadata），见 `payload_types.py`。
- `longcat_mm_inputs`：relay 给 text_ar 的最终多模态输入。

### 5.3 不加载 `model.audio_embed_layers.*`

audio_head 的 codebook embedding 直接从 `model.embed_tokens.weight` 按
`audio_offset` + `codebook_sizes` 切片构建（`_OffsetCodebookEmbedding`），
官方 checkpoint 的 `audio_embed_layers` 是冗余副本，跳过以省显存。

---

## 6. 性能优化（Phase 2.5 / 4）

### 6.1 Encoder exact-match cache

preprocessing 为每段媒体生成 `cache_key`；image/audio executor 查
`StageOutputCache`（LRU，256 条 / 1 GiB 双约束）。多轮同媒体追问直接命中，
跳过整个 encoder forward。

### 6.2 Encoder 跨请求 micro-batching

encoder stage 默认把等待窗口内的多个请求合并成**一次** ViT / audio forward
（Qwen2-VL 式 packed pixel_values + grid_thw；音频零填充堆叠），再按请求切回。
每请求预期 token 数来自 preprocessing 的 positions（与下游注入路径同一 ground
truth），计数不匹配自动回退逐请求串行，批处理不会损坏请求。`MAX_BATCH_SIZE=1`
恢复旧行为。

### 6.3 TensorRef lazy relay

`visual_embeds` / `audio_embeds` 走 TensorRef：encoder→mm_aggregate 一跳只传
引用，CPU 侧 aggregate 从不物化大张量，仅 text_ar 端解析。由
`LongcatNextPipelineConfig.env_defaults` 自动注入开关。

### 6.4 多模态 cache CPU offload（pinned + 真异步）

`SGLANG_OMNI_LONGCAT_ENCODER_CACHE_DEVICE=cpu` 时，encoder 输出 offload 到
**pinned host memory**，释放 encoder GPU 显存：

- D2H 在共享侧流上**异步**发出，`put()` 立即返回不阻塞 producer；
- 每个缓存条目携带 `ready_event`，`get()` / 淘汰 / clear 时才等待（实际
  场景下届时拷贝早已完成）——真正达成 D2H 与后续计算的 overlap；
- 源张量 `record_stream` 防内存提前复用；text_ar 侧
  `chunk.to(device, non_blocking=True)` 仅在源为 pinned 时真异步，
  `SGLANG_OMNI_LONGCAT_DEBUG_PINNED=1` 可观测静默退化。

### 6.5 text_ar CUDA Graph / async decode

decode 回到 SGLang 标准 `super().forward()`，CUDA Graph 与 async decode 均已
启用（audio 输出模式下双头始终执行，靠 persistent buffer 值区分有效输出，
保证 graph 控制流静态）。

### 6.6 启动加速要点

- CephFS 上 shard 加载波动大（实测单 shard 间隙可达 59–95s）；建议模型拷本地盘；
- JIT cache 放本地：`TRITON_CACHE_DIR` / `CUDA_CACHE_PATH` /
  `TORCHINDUCTOR_CACHE_DIR` 指向 /tmp；
- image/audio encoder 各要读一次完整 `embed_tokens.weight` 做 codebook 切片，
  启动慢时可预处理成独立小文件。

---

## 7. 语音输出（Phase 3）

默认关闭，`SGLANG_OMNI_LONGCAT_ENABLE_AUDIO_OUTPUT=1` 开启。

### 7.1 每步解码流程（parallel 模式，delay=0）

1. 上一步的 8 个 audio codes 经 `input_codebook_embedding` 求和，与文本
   embedding 相加融合为 LLM 输入（text-only 时 codes 全零、贡献为零）；
2. AR forward 得 hidden_states；
3. `lm_head` 采样文本 token；`audio_head` 对 `hidden[:, -1]` 做 **8 步因果
   codebook 预测**（DepthTransformer 在 codebook 深度维做因果注意力，
   codebook k 只依赖 0..k-1，循环内零初始化、argmax 采样——无 host sync）；
4. 采样端状态机：`GEN_TEXT → 检测到 <audio_gen_start>(131123) → GEN_AUDIO
   → <audio_gen_end>(131124) → 回到文本或 EOS`。

### 7.2 audio_head CUDA Graph（batch 分档）

8 步循环纯 GPU、无数据相关分支，可整段图捕获。为避免在线服务 bs 分布散导致
显存无界增长，图按 **2 的幂 batch 分档**（1–128，共 ≤8 张）捕获：输入零填充到
最近档位、输出裁回原 bs（行间独立互不影响），超上限回退 eager。默认关，
`SGLANG_OMNI_LONGCAT_AUDIO_HEAD_CUDA_GRAPH=1` 开启。

### 7.3 code2wav 流式解码

- text_ar 每步产出的 8 codes 经 `stream_to` 推到 code2wav 的 stream inbox；
- **渐进式窗口**：首 chunk 5 帧抢 TTFA，之后每次 emit ×2 直至稳态 20 帧
  （三参数均可 env 调）；窗口阈值 per-request 维护；
- flow matching de-tokenizer + HiFi-GAN vocoder 输出 24kHz PCM；
- 请求 abort 时 `clear_stream_state` 统一清理 `_buffers` / `_stream_thresholds`，
  长会话不泄漏。

---

## 8. 实时语音对话（Phase 4/5）

`/v1/realtime` 提供 OpenAI Realtime 风格的全双工体感语音对话。
**真全双工不可达**（turn-based 模型，见 §1），工程目标是把伪全双工（极低延迟
快速轮转）做到产品级体感。实现位置：`sglang_omni/serve/realtime/`。

### 8.1 每轮流程

1. 服务端 VAD（silero，自适应端点，默认静音 300ms）判定说话开始/结束；
2. `speech_stopped` → auto-commit → 入 `response_queue`（FIFO 串行化）；
3. `run_turn`：先占 pending 用户历史槽位 → `run_response` 流式下发
   text/audio delta → 转写走后台任务回填槽位（P3，不拖慢下一轮 TTFA）；
4. 历史拼接固定顺序、跳过 pending/空条目，保证前缀逐字稳定 → RadixCache
   命中，prefill 只付增量（P4）。

### 8.2 Barge-in（打断）

- VAD `speech_started` 瞬间**先发** `speech_started` + `response.audio.flush`
  事件（前端立即清空播放缓冲），abort 引擎请求在后台进行（P0/P1）；
- abort 屏障在第一个 await 之前同步建立，下一轮开始前必须等待旧请求释放
  （防 KV 竞争）；队列中已提交的用户语句**保留不丢**；
- **上下文保真**：即使打断发生在响应中途，已流出的半截回复由
  `run_response` 暂存、用户语音由 finally 中兜底启动的后台转写回填——
  该轮对话历史完整保留，pending 槽位不会滞留；
- 后台转写不复用 `active_request_id`（避免 abort 错请求），被取消时主动
  abort 自己的引擎请求。

### 8.3 打断续接的代价

采用降级方案：cancel 当前请求 + 开新请求 + RadixCache 前缀复用历史 KV。
代价是每次打断后付一次增量 prefill；序列内 mid-stream 增量续接未实现
（需改 scheduler，留作后续里程碑）。

---

## 9. 环境适配与已知坑

| 依赖 | 坑 | 解法 |
|---|---|---|
| flash-attn 4.x | v4 是 CUTE 重写：`flash_attn_varlen_func` 移到 `flash_attn.cute`，`bert_padding` 移除，新增 `qv` 位置参数 | `components/dynamic.py` 顶层注入命名空间 + 伪模块；调用一律用 keyword args |
| transformers 5.6 | `Qwen2RMSNorm` 移到 qwen2；`video_processor` 强类型校验；tokenizer 不再暴露自定义 init_kwargs；`flash_attention.py` 缺 `s_aux=None` 保护 | `processing_longcat_next.py` override 校验 + 回读 tokenizer_config.json；site-packages 打 None-check 补丁 |
| gcc | Ubuntu 20.04 gcc 9.4 不支持 sglang JIT 需要的 C++20 | 用 Ubuntu 24.04（自带 gcc 13） |
| CephFS | shard 加载 I/O 波动（59–95s 间隙） | 模型拷本地；见 §6.6 |
| python dev headers | `pyconfig.h` 缺失 | `apt install libpython3.12-dev` |

注意：flash-attn / transformers 的兼容层是运行时 monkey-patch，上游升级后需
重新验证。

---

## 10. 环境变量速查

| 变量 | 默认 | 作用 |
|---|---|---|
| `SGLANG_OMNI_LONGCAT_ENABLE_AUDIO_OUTPUT` | 关 | 开启语音输出（audio_head + code2wav stage） |
| `SGLANG_OMNI_LONGCAT_STREAM_FRAMES` | 20 | code2wav 稳态攒帧窗口 |
| `SGLANG_OMNI_LONGCAT_STREAM_FRAMES_FIRST` | 5 | 首包窗口（≤稳态） |
| `SGLANG_OMNI_LONGCAT_STREAM_FRAMES_GROWTH` | 2 | 窗口增长倍数（1=关闭） |
| `SGLANG_OMNI_LONGCAT_AUDIO_HEAD_CUDA_GRAPH` | 关 | audio_head 8 步循环图捕获 |
| `SGLANG_OMNI_LONGCAT_AUDIO_GRAPH_MAX_BS` | 128 | 图分档上限，超出回退 eager |
| `SGLANG_OMNI_LONGCAT_ENCODER_CACHE_DEVICE` | 关 | `=cpu` 时 encoder cache offload 到 pinned 内存 |
| `SGLANG_OMNI_LONGCAT_ENCODER_CACHE_MAX_BYTES` | 1 GiB | cache 字节上限 |
| `SGLANG_OMNI_LONGCAT_{IMAGE,AUDIO}_ENCODER_MAX_BATCH_SIZE` | 4 / 8 | encoder micro-batch 上限（1=串行） |
| `SGLANG_OMNI_LONGCAT_{IMAGE,AUDIO}_ENCODER_MAX_BATCH_WAIT_MS` | 10 | micro-batch 等待窗口 |
| `SGLANG_OMNI_LONGCAT_DEBUG_TIMING` | 关 | longcat 全链路计时打点 |
| `SGLANG_OMNI_LONGCAT_DEBUG_PINNED` | 关 | pageable 源导致 H2D 退化时记日志 |
| `SGLANG_OMNI_REALTIME_TIMING` | 关 | realtime TTFA / barge-in 打点 |
| `SGLANG_OMNI_REALTIME_STREAMING_ASR` | 关 | 边说边转写（额外 ASR 算力，仅 UI） |
| `SGLANG_OMNI_ENABLE_TENSOR_REFS` 等 | 由 config 注入 | TensorRef lazy relay，一般无需手设 |

---

## 11. 测试与验证

### 11.1 单元测试

| 文件 | 覆盖 |
|---|---|
| `tests/unit_test/serve/test_realtime_session.py` | barge-in 上下文保真、abort 屏障时序、request-id 隔离、response.done 契约、teardown |
| `tests/unit_test/scheduling/test_stage_cache.py` | cache LRU/字节预算/offload 退化路径 |
| `tests/unit_test/test_longcat_audio_head_graph.py` | graph batch 分档映射（需 flash_attn，GPU CI） |
| `tests/unit_test/test_longcat_encoder_batch.py` | encoder micro-batching 合并/切分/回退 |

### 11.2 E2E 冒烟（GPU 机器）

纯文本、图片、音频三条 chat/completions 用例均已在 8×H200 实机验证通过
（文本正常、图片描述准确、音频转写正确，usage 字段齐全）。

### 11.3 待 GPU 实测项

- 多模态 offload 的 D2H/H2D overlap 实测收益；
- audio_head graph 分档后的显存占用与 replay 正确性；
- 实时对话端到端 TTFA / barge-in 感知延迟绝对值（开
  `SGLANG_OMNI_REALTIME_TIMING=1` 采集）；
- RadixCache 命中率随对话轮次曲线；VAD 300ms 端点的误切/漏切率。

---

## 12. 已知限制与后续方向

| 项 | 状态 |
|---|---|
| 真全双工（边说边听双流并行） | ❌ 模型架构限制，需换模型（如 Moshi）或重训 |
| 序列内 mid-stream 增量续接 | 未实现，打断后走 cancel + 新 req + 前缀复用（付一次增量 prefill） |
| vocoder D2H 与下一次 launch 的 overlap | 暂缓（需 GPU 验证 stream/event 双缓冲正确性） |
| vocoder CUDA Graph | 暂缓（渐进窗口使 shape 不固定，需多图复用） |
| 流式 relay 消息合并 | 明确不做（每消息仅 64B，消费端已攒帧，收益边际） |
| 视觉输出（visual_head） | 未接入 |
| ngram embedder 表加载慢 | ~40GB，4 卡约 310s，可预处理优化 |
| Phase 5 延迟目标 | 全部编排层优化已落地，绝对值待 GPU 实测 |

---

## 13. 文件索引

```
sglang_omni/models/longcat_next/
├── config.py                # 两套 PipelineConfig（纯文本 / 多模态）+ env 注入
├── stages.py                # stage 工厂：preprocessing/encoders/aggregate/text_ar/code2wav
├── sglang_model.py          # nn.Module：词表截断、zero-expert patch、双头 forward
├── model_runner.py          # prefill 注入（replace_embeds）+ decode 音频钩子
├── request_builders.py      # fan-out 投影 / wait_for 解析 / 结果适配
├── merge.py                 # mm_aggregate fan-in 合并
├── payload_types.py         # LongcatNextPipelineState + 计时工具
└── components/
    ├── preprocessor.py      # tokenize + 媒体特征 + cache_key + positions
    ├── encoders.py          # ViT / audio tokenizer 移植 + _OffsetCodebookEmbedding
    ├── audio_head.py        # 8 步因果 codebook 预测 + graph 分档
    ├── code2wav.py          # flow matching + HiFi-GAN 流式解码
    └── dynamic.py           # flash-attn v4 兼容桥

sglang_omni/serve/realtime/  # /v1/realtime 语音对话（session/vad/events/audio_buffer）
examples/configs/longcat_next_{text,multimodal}.yaml
artifacts/longcat_next/      # checkpoint 权重清单与 HF 元数据快照
```
