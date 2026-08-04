# LongCat-Next on sglang-omni 开发文档

## 概述

LongCat-Next 是美团 LongCat 团队发布的原生多模态离散 token 模型（DiNA 范式），
将文本、图像、音频统一为离散 token，由单一 AR backbone 做 Next Token Prediction。

sglang-omni 按阶段接入：
- **Phase 1**（当前）：纯文本 AR backbone 启动

关键架构事实：LongCat-Next 的 `LongcatNextForCausalLM` 继承自
`LongcatFlashForCausalLM`，底层 Transformer（MLA + MoE）与 SGLang 原生支持的
LongCat-Flash 完全一致，仅维度不同。

---

## Phase 1：纯文本 AR Backbone

### 目标

从完整 LongCat-Next checkpoint 中启动 AR backbone，提供标准 chat/completions API。

### 架构参数

| 参数 | 值 |
|---|---|
| 总参数量 | ~68.5B（A3B MoE，激活 ~3B/token） |
| hidden_size | 3072 |
| 层数 | 14（每层：2× MLA + 2× Dense MLP + 1× MoE） |
| 注意力 | MLA（kv_lora_rank=512, q_lora_rank=1536） |
| MoE | 256 routed + 128 zero experts，top-12 |
| 上下文长度 | 131072 |
| text_vocab_size | 131072 |
| 完整词表 | 131125（含 53 个多模态特殊 token） |

### 权重加载策略

完整 checkpoint 约 140GB，各组件按前缀分布：

```
checkpoint 所有权重
├── model.layers.*              → AR backbone，SGLang 直接加载
├── model.embed_tokens.weight   → [131072, 3072]，正常加载
├── model.norm.*                → 正常加载
├── model.ngram_embeddings.*    → SGLang NgramEmbedding 原生支持
├── lm_head.weight              → [131125, 3072]，自定义 head 加载
├── model.visual_tokenizer.*    → Phase 1 过滤（load_weights 中 skip）
├── model.audio_tokenizer.*     → Phase 1 过滤
├── visual_head.*               → Phase 1 过滤
└── audio_head.*                → Phase 1 过滤
```

**不做权重抽取**——各 Phase 按需从同一 checkpoint 加载各自组件。

### 关键适配：词表不对称

LongCat-Next 的三个词表尺寸,用途不同:

| 尺寸 | 名称 | 用途 |
|---|---|---|
| 131072 | `text_vocab_size` | 纯文本 token;**ngram 哈希基数** |
| 131125 | `text_vocab_plus_multimodal_special_token_size` | 文本 + 53 个多模态特殊标记;**embed_tokens / lm_head / word_embeder 行数** |
| 282624 | `vocab_size` | 完整词表(含视觉/音频内容 token 区,`audio_offset=131125`、`visual_offset=150581`) |

多模态**内容** token(≥131125)不走 embed_tokens 查表,而是经各自 encoder 编码后以 `input_embeds` / `replace_embeds`+`replace_positions` 注入。所以 embed_tokens 只需覆盖文本+特殊标记 = 131125 行。

`LongcatNextTextForCausalLM` 在 `super().__init__()` 前把 `config.vocab_size = 131125`,使父类同时用 131125 创建 embed_tokens 与 lm_head(两者形状一致,`_tied_weights_keys = ["lm_head.weight"]` 正常绑定);`load_weights` 对 embed_tokens/lm_head 做 `[:131125]` 截断。

### 关键适配：load_weights

参照官方 `NmmFlashForCausalLM.load_weights`（`modules/nmm_flash.py`）：

1. 过滤多模态前缀（`audio_head.`、`visual_head.`、`model.audio_tokenizer.`、
   `model.visual_tokenizer.`、`model.audio_embed_layers.`）
2. 对 `model.embed_tokens.weight` 和 `lm_head.weight` 做防御性 `[:131125]` 截断
3. 其余委托父类处理（MLA fused QKV、MoE stacked/expert mapping、ngram embeddings）

### Ngram Embedding（关键：见"问题 4"）

LongCat-Next 的输入 embedding 是 `NgramEmbedding`（word embedding + 12 个 n-gram
投影，共 13 项取均值），**不是普通 embedding**。SGLang 的 `NgramEmbedding.load_weight`
原生支持 `model.ngram_embeddings.embedders.*` / `post_projs.*` 命名,但**启用它需要
三处适配**(config 派生 + oe_weights 哈希基数修正 + OmniScheduler 运行时管线),详见
"问题 4"。缺任何一处都会导致乱码或 forward 崩溃。

### 多模态 token 抑制

纯文本推理时，通过 `req._codec_suppress_tokens` 将 token 131072–131124 的
logits 置为 `-inf`，防止模型输出视觉/音频 token。

### 文件清单

```
sglang_omni/models/longcat_next/
├── __init__.py              # 包标记
├── config.py                # PipelineConfig
├── sglang_model.py          # nn.Module（lm_head + load_weights）
├── stages.py                # Stage 工厂函数
├── request_builders.py      # 请求/结果适配器
└── 开发文档.md              # 本文档

sglang_omni/model_runner/sglang_model_runner.py  # +1 行注册
```

### 启动方式

```bash
sgl-omni serve \
    --config examples/configs/longcat_next_text.yaml \
    --model-path /mnt/cephfs/chenzhenyang/models/LongCat-Next \
    --host 0.0.0.0 --port 8100

curl http://localhost:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "longcat-next",
    "messages": [{"role": "user", "content": "你好，请用一句话介绍自己"}],
    "max_tokens": 128
  }'
```

TP=2 示例：

```yaml
stages:
  - name: text
    gpu: [0, 1]
    tp_size: 2
```

### 已知问题与修复

#### 问题 1：SGLang Config 字段名不兼容

LongCat-Next HF config 与 SGLang `LongcatFlashConfig` 字段名不一致：

| SGLang 期望 | LongCat-Next 实际 | 值 |
|---|---|---|
| `intermediate_size` | `ffn_hidden_size` | 6144 |
| `moe_intermediate_size` | `expert_ffn_hidden_size` | 1024 |
| `num_hidden_layers` | `num_layers` | 14 |

`ModelConfig` 对象已有这些属性（来自 SGLang 默认值），导致 `hasattr` 判断为 True 跳过映射。

**修复**：在 `sglang_model.py.__init__` 中无条件覆盖（不依赖 `hasattr` 判断）。

#### 问题 2：MoE fused kernel 非法内存访问（真因见问题 4）

```
Triton Error [CUDA]: an illegal memory access was encountered
  at moe_runner/triton_utils/fused_moe_triton_kernels.py fused_moe_kernel
```

**最初误判**为 Triton kernel 在 H800/H200 SM90 上的硬件兼容问题,试图用
`server_args.moe_runner_backend = "flashinfer_cutlass"` 绕过。但实测两种 backend
崩溃路径完全相同(未量化 MoE 都走 `fused_experts_none_to_triton` → triton
`fused_moe_kernel`),证明 backend 不是病根。**真正根因是 zero-expert 索引越界,
详见问题 4。** `moe_runner_backend` 设置对未量化 MoE 实际不生效,保留 `flashinfer_cutlass`
仅为与既有代码一致。

#### 问题 3（误报，已排除）：`kv_a_proj_with_mqa` 缺失

最初怀疑 checkpoint 缺少 `kv_a_proj_with_mqa` 权重导致 SGLang Q/KV 融合失败。远程检查 safetensors 后确认 checkpoint **包含全部 MLA 权重**（`q_a_proj`、`q_b_proj`、`kv_a_proj_with_mqa`、`kv_b_proj`、`o_proj`），SGLang 融合逻辑正常工作。`_q_a_buffers` 手动加载代码已回退。

#### 问题 4：MoE zero-expert 索引越界（问题 2 的真因）

LongCat MoE = 256 routed + 128 个 `identity` zero-expert,router 输出 384 维,top-12。
SGLang 的 `LongcatFlashMoE.forward` 调 `zero_experts_compute_triton`,把 `topk_idx`
中 ≥256 的 zero-expert 索引**原地改成 `-1`**(同时把其 combine 权重清零),再交给
只建了 256 个权重的 `fused_moe`。

`fused_moe_kernel` 跳过 `-1` 的保护是 `if filter_expert and off_experts == -1`,
而 `filter_expert = (num_experts != num_local_experts)`——**纯 TP 下二者都是 256 →
`filter_expert = False`**,保护被短路 → kernel 用 `-1` 索引 `w13_weight[-1]` →
illegal memory access。(EP 分片场景才 `filter_expert=True`,故上游未暴露此 bug。)

官方 0.4.3 fork 的 `TopK` 内建 `zero_expert_num` 处理,不依赖 `filter_expert`,故能跑。

**修复**（`sglang_model.py`,模块加载时 patch `LongcatFlashMoE.forward`）:在 `topk_idx`
传给 `self.experts` 前把 `-1` clamp 成 0(`topk_idx.masked_fill(topk_idx < 0, 0)`)。
这些位置 combine 权重已为 0,贡献恒为 0 → 与"跳过 -1"数值等价,消除越界。

#### 问题 5：Ngram embedding 未启用导致乱码

`LongcatNextConfig`(trust_remote_code)只带原始字段
(`ngram_vocab_size_ratio`/`emb_neighbor_num`/`emb_split_num`),不带 SGLang 期望的派生
字段(`ngram_embedding_m/k/n`),导致原 `__init__` fallback 到普通 `VocabParallelEmbedding`:
①24 个 ngram 权重被丢弃(`not found in params_dict`);②输入 embedding 幅度约为训练时
的 13 倍且缺 n-gram 项 → **数值从根本上错误 → 乱码**。

启用 NgramEmbedding 需**三处**适配:

1. **config 派生**(`model_worker.py._apply_arch_override` + `sglang_model.py.__init__`):
   派生 `use_ngram_embedding=True`、`ngram_embedding_m = int(ratio × text_vocab)
   = int(78×131072) = 10223616`、`n=4`、`k=4`。**关键**:`m` 用 `text_vocab=131072`
   (非 full_vocab 131125),否则 embedder 形状断言失败。派生必须在 ModelConfig 之后、
   model_runner 构造之前完成(在 `_apply_arch_override` 里同时设 `model_config.use_ngram_embedding`
   与 `hf_config.*`),否则 stock `maybe_init_ngram_embedding` / `ForwardBatch.init_new`
   读到的仍是 False → token_table 不分配 → forward 崩。

2. **oe_weights 哈希基数修正**(`sglang_model.py`,`super().__init__()` 后):venv 的
   `NgramEmbedding` 把 word 表行数(131125)与 n-gram 哈希基数(应为 131072)绑成单一
   `num_embeddings`。遍历 NgramEmbedding 实例,用 `base=text_vocab(131072)` 重算
   `oe_weights`(`pow(131072, delta, mod)`)。base 131072 vs 131125 产生完全不同的
   哈希权重 → 用错则 n-gram id 全错。此设计与官方 `FusedOverEmbedding` 的
   `num_embeddings_text` 参数一致,天然兼容未来多模态(哈希基数与 word 表解耦)。

3. **OmniScheduler ngram 运行时管线**(`omni_scheduler.py`):OmniScheduler 不继承 stock
   Scheduler(用 `__getattr__` 委托)。补:①`_init_upstream_compat_flags` 里从 `model_config`
   派生 `use_ngram_embedding` + 抓 `token_table` / `ngram_embedding_n/k`;
   ②覆盖 stock `_maybe_prepare_ngram_embedding(batch)` 填 `batch.ne_token_table`(EXTEND 模式)。
   decode 步的 token table 更新由 stock `ModelRunner.sample → maybe_update_ngram_token_table`
   自动完成,无需改。全部用 `use_ngram_embedding` 守卫,其他模型不受影响。

   **关键陷阱(务必 `return batch`)**:OmniScheduler 经 `__getattr__` 走 stock
   `get_next_batch_to_run`,后者内部 `ret = self._maybe_prepare_ngram_embedding(ret)`。
   因为我们在 OmniScheduler 上**定义**了这个方法(正常 MRO 优先于 `__getattr__`),该调用命中
   我们的版本。stock 版本返回 `batch`;若我们的版本漏写 `return batch`(隐式返回 None),
   则 `ret=None` → batch 被静默丢弃 → 事件循环空转、GPU 0%、请求永远卡在 waiting_queue、
   无任何报错。**症状极具迷惑性**(submitted 后静默、health 200、进程存活)。所以本方法
   每个分支都必须 `return batch`。也正因 stock 已在 `get_next_batch_to_run` 调用它,
   **不要**在 `_run_batch` 里再显式调一次(冗余)。

**验证状态(已通过)**:启动日志确认 config 派生(`m=10223616`)、oe_weights 重算
(`hash_base=131072`)、24 个 ngram 权重全部加载(0 个 not-found);curl 实测输出通顺:
"我是 LongCat，由美团研发的人工智能……"、量子纠缠解释、"3加5=8" 等多 prompt 均正确。
从此前乱码彻底修复。

### 当前限制

- 仅文本生成，无视觉/音频输入输出
- ngram 启用后权重加载慢(embedder 表 ~40GB,4 卡加载 ~310s)
- 未来多模态:ngram 兼容的 embedding 注入路径是 `replace_embeds`/`replace_positions`
  (先对文本算 ngram embedding,再 scatter 覆盖多模态位置),不能用整段替换的 `input_embeds`

2026-07-14 10:01:46,564 [WARNING] sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config: Using MoE kernel config with down_moe=False. Performance might be sub-optimal! Config file not found at /mnt/cephfs/chenzhenyang/czy/sglang-omni/.venv/lib/python3.12/site-packages/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/E=256,N=256,device_name=NVIDIA_H800_down.json, you can create them with https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton