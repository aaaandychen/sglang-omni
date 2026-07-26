# LongCat-Next Phase2 详细设计与 Debug 记录

> 本文档专门记录 LongCat-Next Phase2（图像/音频/文本输入 → 文本输出）的详细设计、当前实现、H200 验证方法和后续 debug 过程。  
> Phase1 文本 AR 的历史问题与结论见 `开发文档_phase1.md`；Phase2 高层规划见 `开发文档_phase2.md`。

---

## 1. 阶段目标

Phase2 的目标是把 LongCat-Next 的图像/音频输入理解能力接入 `sglang-omni`，让服务支持：

```text
image + audio + text input -> LongCat-Next AR backbone -> text output
```

本阶段暂不接入：

- 图像生成输出；
- 音频生成输出；
- `visual_head` / `audio_head` 采样；
- vocoder / image refiner；
- 高阶 CUDA graph / tensor ref relay 优化。

Phase2 MVP 的优先级：

1. 保持 Phase1 纯文本能力不退化。
2. 保留 LongCat-Next 必需的 `NgramEmbedding` 路径。
3. 正确把 image/audio encoder 输出 embedding 注入 AR prefill。
4. 在 H200 上完成图像+音频+文本输入的文本输出正确性验证。

---

## 2. 为什么不能照搬官方 `input_embeds` 整段替换

官方 LongCat-Next inference 中，`NmmFlashForCausalLM.forward()` 从：

```python
forward_batch.request_cache_input["input_embedding"]
```

读取整段 embedding，然后直接进入 LLM。

但 `sglang-omni` Phase1 已经验证：LongCat-Next 文本 token 必须经过 SGLang 内部 `NgramEmbedding`。如果外部整段构造 `input_embeds` 并绕开模型内部 embedding，会丢失 n-gram 信息，出现语义错误或乱码。

因此 Phase2 使用：

```text
input_ids -> SGLang LongCat NgramEmbedding -> text embeddings
                                             │
encoder visual/audio embeddings ------------┤ scatter/replace pad positions
                                             ▼
                                      Transformer / MLA / MoE
```

核心原则：

> 多模态 embedding 只覆盖 `<img_pad>` / `<audio_pad>` 位置，普通文本 token 仍走 Phase1 已验证过的 NgramEmbedding 管线。

---

## 3. Multi-stage 拓扑

当前 Phase2 拆成 5 个 stage：

```text
preprocessing
  ├── image_encoder
  ├── audio_encoder
  └── mm_aggregate
          └── text_ar
```

### 3.1 Stage 职责

| Stage | 资源 | 职责 |
|---|---|---|
| `preprocessing` | CPU | 解析请求，调用官方 processor，生成 `input_ids`、encoder 输入和 pad positions |
| `image_encoder` | GPU 0 | 图像 → visual ids → visual embeddings |
| `audio_encoder` | GPU 1 | 音频特征 → audio ids → audio embeddings |
| `mm_aggregate` | CPU | 等待需要的上游，合并 metadata 和 encoder outputs，构造 text_ar payload |
| `text_ar` | GPU 2-5, TP=4 | Phase1 AR backbone，prefill 时注入 replace embeddings，decode 输出文本 |

### 3.2 为什么需要 `mm_aggregate`

原本可以设计成：

```text
preprocessing + image_encoder + audio_encoder -> text_ar
```

但这会让 AR stage 直接承担 pipeline 编排职责，包括：

- 等待不同上游；
- 跳过空模态；
- 合并 payload；
- 处理 encoder cache metadata；
- 后续扩展 generation head 路由。

Phase1 的 AR 路径已经有很多模型特异逻辑：

- config 字段映射；
- vocab size 临时改写；
- ngram 派生与 token table；
- zero-expert patch；
- MoE/MLA 权重加载。

所以 Phase2 让 `text_ar` 保持纯 AR stage，把多模态 fan-in 放到 `mm_aggregate`。

### 3.3 `mm_aggregate` 不占 GPU

`mm_aggregate` 是控制面 stage：

- `wait_for_fn`；
- metadata merge；
- route；
- 构造 payload contract。

它不应该执行 `.cpu()` / `.cuda()` / 大 tensor 拼接。MVP 阶段允许 shm/CPU 中转大 tensor 来优先验证正确性；优化阶段应改成 tensor ref / cache key，由 `text_ar` 侧 materialize 到目标 GPU。

---

## 4. 当前资源分配

当前默认配置适配用户计划：4 卡跑 AR，图像/音频 encoder 各 1 卡。

```text
CPU: preprocessing + mm_aggregate
GPU 0: image_encoder
GPU 1: audio_encoder
GPU 2-5: text_ar TP=4
```

配置文件：

```text
examples/configs/longcat_next_multimodal.yaml
```

内容：

```yaml
config_cls: LongcatNextPipelineConfig
model_path: /mnt/cephfs/chenzhenyang/models/LongCat-Next
relay_backend: shm
```

---

## 5. 实现文件清单

### 5.1 修改文件

```text
sglang_omni/models/longcat_next/config.py
sglang_omni/models/longcat_next/stages.py
sglang_omni/models/longcat_next/request_builders.py
sglang_omni/models/longcat_next/sglang_model.py
```

### 5.2 新增文件

```text
sglang_omni/models/longcat_next/payload_types.py
sglang_omni/models/longcat_next/merge.py
sglang_omni/models/longcat_next/model_runner.py
sglang_omni/models/longcat_next/components/__init__.py
sglang_omni/models/longcat_next/components/dynamic.py
sglang_omni/models/longcat_next/components/encoders.py
sglang_omni/models/longcat_next/components/preprocessor.py
examples/configs/longcat_next_multimodal.yaml
```

---

## 5.3 Pipeline 注册方式

`config.py` 中保留：

```python
EntryClass = LongcatNextTextPipelineConfig
```

这是为了不破坏 Phase1 默认行为。Phase2 多模态 pipeline 通过 YAML 显式指定：

```yaml
config_cls: LongcatNextPipelineConfig
```

也就是说，H200 验证时必须使用：

```text
examples/configs/longcat_next_multimodal.yaml
```

`Variants = {"text": ..., "multimodal": ...}` 仅作为兼容/发现辅助；是否被 CLI 直接读取取决于外层 config loader。当前确定可用路径是通过 `config_cls: LongcatNextPipelineConfig` 指定。

---

## 6. 关键数据结构

### 6.1 `LongcatNextPipelineState`

文件：`payload_types.py`

```python
@dataclass
class LongcatNextPipelineState:
    prompt: dict[str, Any]
    mm_inputs: dict[str, Any]
    encoder_inputs: dict[str, dict[str, Any]]
    encoder_outs: dict[str, dict[str, Any]]
    text_ar_inputs: dict[str, Any]
```

用途：在各 stage 之间传递结构化状态。

### 6.2 `longcat_mm_inputs`

`mm_aggregate` 输出到 `text_ar` 的核心字段：

```python
longcat_mm_inputs = {
    "image_embeds": Tensor[num_img_tokens, hidden_size],
    "image_positions": Tensor[num_img_tokens],
    "audio_embeds": Tensor[num_audio_tokens, hidden_size],
    "audio_positions": Tensor[num_audio_tokens],
}
```

`positions` 是单条请求内的 token 下标，进入 batch 后由 `LongcatNextModelRunner` 加上 batch offset。

---

## 7. Preprocessing 设计

文件：`components/preprocessor.py`

核心类：

```python
LongcatNextPreprocessor
```

它复用官方 HF processor：

```python
AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
```

输入支持：

1. 纯字符串；
2. `messages`；
3. dict 里的 `images` / `audios`；
4. OpenAI-style message content。

输出：

```python
prompt = {
    "text": text,
    "input_ids": input_ids,
    "sampling_params": request.params,
}

mm_inputs = {
    "input_ids": input_ids,
    "image_positions": image_positions,
    "audio_positions": audio_positions,
    "audiotext_start_positions": ...,
    "audiotext_pad_positions": ...,
}

encoder_inputs = {
    "image_encoder": {
        "pixel_values": ...,
        "visual_grid_thw": ...,
        "image_positions": ...,
    },
    "audio_encoder": {
        "audio": ...,
        "encoder_length": ...,
        "bridge_length": ...,
        "audio_positions": ...,
    },
}
```

### 7.1 输入文本构造

官方 processor 需要文本中显式包含：

```text
<image_start>{image_path}<image_end>
<audio_start>{audio_path}<audio_end>
```

当前 preprocessor 会把 `inputs["images"]` / `inputs["audios"]` 追加成官方格式。

---

## 8. Encoder 设计

文件：`components/encoders.py`

当前实现不手工复制官方 1000+ 行 encoder，而是动态加载 HF remote code。

### 8.1 动态加载

文件：`components/dynamic.py`

```python
get_class_from_dynamic_module(
    "modular_longcat_next_visual.LongcatNextVisualTokenizer",
    model_path,
    trust_remote_code=True,
)
```

音频同理：

```python
"modular_longcat_next_audio.LongcatNextAudioTokenizer"
```

要求模型目录中包含 HF remote code 文件。

### 8.2 Image encoder

流程：

```text
pixel_values + visual_grid_thw
  -> LongcatNextVisualTokenizer.encode()
  -> visual_ids [N, num_codebooks]
  -> model.embed_tokens.weight 按 visual_offset/codebook_sizes 切片求和
  -> visual_tokenizer.visual_embedding_layer
  -> visual_embeds [N, hidden_size]
```

加载权重：

```text
model.visual_tokenizer.*
model.embed_tokens.weight  # 只切 visual codebook 区间
```

接口来源：当前代码依赖 HF remote code `modular_longcat_next_visual.LongcatNextVisualTokenizer` 同时具备：

```python
encode(pixel_values, visual_grid_thw)
visual_embedding_layer
```

这个接口已通过从 HF 下载的 `modular_longcat_next_visual.py` 确认：`LongcatNextVisualTokenizer.__init__` 中定义了 `self.visual_embedding_layer = VisualEmbeddingBridge(config)`，`encode()` 返回 visual ids。若 H200 checkpoint 目录中的 remote code 版本不同，可能需要按实际类结构调整 wrapper。

### 8.3 Audio encoder

流程：

```text
audio + encoder_length + bridge_length
  -> LongcatNextAudioTokenizer.encode()
  -> audio_ids [N, num_codebooks]
  -> model.embed_tokens.weight 按 audio_offset/codebook_sizes 切片求和
  -> audio_embeds [N, hidden_size]
```

加载权重：

```text
model.audio_tokenizer.*
model.embed_tokens.weight  # 只切 audio codebook 区间
```

### 8.4 为什么不加载 `model.audio_embed_layers.*`

HF `model.safetensors.index.json` 中未命中 `model.audio_embed_layers.` 前缀。HF 官方实现使用 `embed_tokens` 中 audio codebook 区间求和：

```python
audio_embeddings = self.embed_tokens(audio_ids).sum(dim=1)
```

所以当前实现也采用 `embed_tokens` 切片方式。

---

## 9. Merge 设计

文件：`merge.py`

核心函数：

```python
merge_for_text_ar(payloads: dict[str, StagePayload]) -> StagePayload
```

输入：

```text
preprocessing payload
image_encoder payload, optional
audio_encoder payload, optional
```

输出：

```python
text_ar_inputs = {
    "input_ids": input_ids,
    "sampling_params": sampling_params,
    "longcat_mm_inputs": {
        "image_embeds": ...,
        "image_positions": ...,
        "audio_embeds": ...,
        "audio_positions": ...,
    },
}
```

---

## 10. Routing / wait_for 设计

文件：`request_builders.py`

### 10.1 `resolve_preprocessing_next_stages`

根据 preprocessing 结果决定实际下游：

```python
return [active_encoder_stages..., "mm_aggregate"]
```

纯文本请求不会发给 encoder。

### 10.2 `resolve_mm_aggregate_wait_sources`

根据本请求实际模态决定 aggregate 等待哪些 stage：

```text
纯文本: preprocessing
图像: preprocessing + image_encoder
音频: preprocessing + audio_encoder
图音: preprocessing + image_encoder + audio_encoder
```

---

## 11. AR 注入设计

### 11.1 Request builder

文件：`request_builders.py`

`text_ar` request builder 如果收到 `text_ar_inputs.input_ids`，则直接使用 preprocessing 生成的 input_ids，不再重新 tokenize。

同时把：

```python
longcat_mm_inputs
```

挂到：

```python
SGLangARRequestData.longcat_mm_inputs
```

进入 scheduler 后，框架会把 request data 挂到：

```python
req._omni_data
```

### 11.2 ModelRunner hook

文件：`model_runner.py`

`LongcatNextModelRunner.before_prefill()`：

1. 遍历 `schedule_batch.reqs`；
2. 从 `req._omni_data.longcat_mm_inputs` 读取 embeddings 和 positions；
3. 根据当前 EXTEND chunk 的 `prefix_indices` / `extend_seq_lens_cpu`，把 preprocessing 阶段的全局 positions 过滤到本次 chunk 内；
4. 将全局 positions 映射为本次 `forward_batch.input_ids` 的局部下标，再加 batch 内 offset；
5. 写入：

```python
forward_batch.longcat_replace_embeds
forward_batch.longcat_replace_positions
```

### 11.3 Model forward

文件：`sglang_model.py`

`LongcatNextTextForCausalLM.forward()`：

1. 如果没有 `longcat_replace_embeds`，走 Phase1 原始路径。
2. 如果有：
   - 先计算 NgramEmbedding；
   - 将 multimodal pad token 位置先置 0 避免越界/错误 hash；
   - 用 encoder embeddings 覆盖这些位置；
   - 调用 LongCat-Flash backbone。

### 11.4 NgramEmbedding 与多模态 pad token 的取舍

当前实现会在计算 NgramEmbedding 前，把 `<img_pad>` / `<audio_pad>` 位置的 token id 替换为 0，然后再覆盖这些位置的 embedding。

这会带来一个理论影响：NgramEmbedding 的 n-gram hash 会参考相邻 token，因此靠近多模态 pad 的文本 token，其 n-gram 可能受到 `pad -> 0` 的影响。

这里是有意取舍：

1. 官方多模态 inference 路径是整段 `input_embeds`，实际上完全跳过了 ngram。
2. Phase1 已验证 LongCat-Next 文本路径如果不走 ngram，输出质量会明显异常。
3. 在 sglang-omni 中，保留绝大多数文本 token 的 ngram 信息，比让整段文本退化为普通 embedding 更重要。
4. 多模态 pad 附近少量文本 token 的 ngram 偏差是 MVP 可接受风险。

后续如果需要更接近训练/官方行为，可以考虑：

- 只在多模态请求中完全禁用 ngram，与官方 input_embeds 路径对齐；
- 或让 NgramEmbedding 支持 ignore multimodal pad token，不让 pad token 参与相邻文本 hash；
- 或在 tokenizer 模板上通过分隔符减少 pad 与关键文本直接相邻。

---

## 12. H200 验证方式

### 12.1 启动

```bash
sgl-omni serve \
  --config examples/configs/longcat_next_multimodal.yaml \
  --model-path /mnt/cephfs/chenzhenyang/models/LongCat-Next \
  --host 0.0.0.0 \
  --port 8100
```

### 12.2 纯文本回归

```bash
curl http://localhost:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "longcat-next",
    "messages": [{"role": "user", "content": "你好，请用一句话介绍自己"}],
    "max_tokens": 128
  }'
```

预期：与 Phase1 输出质量一致。

### 12.3 图像+文本

客户端输入需要在 metadata 或 inputs 中携带 image path。当前 preprocessor 支持：

```json
{
  "messages": [{"role": "user", "content": "请描述这张图片"}],
  "images": ["/path/to/image.jpg"]
}
```

### 12.4 音频+文本

```json
{
  "messages": [{"role": "user", "content": "请总结这段音频"}],
  "audios": ["/path/to/audio.wav"]
}
```

### 12.5 图像+音频+文本

```json
{
  "messages": [{"role": "user", "content": "结合图片和音频回答问题"}],
  "images": ["/path/to/image.jpg"],
  "audios": ["/path/to/audio.wav"]
}
```

---

## 13. 当前已完成检查

本地已完成 Python 语法编译检查：

```bash
python3 -m py_compile \
  sglang_omni/models/longcat_next/config.py \
  sglang_omni/models/longcat_next/payload_types.py \
  sglang_omni/models/longcat_next/components/__init__.py \
  sglang_omni/models/longcat_next/components/dynamic.py \
  sglang_omni/models/longcat_next/components/encoders.py \
  sglang_omni/models/longcat_next/components/preprocessor.py \
  sglang_omni/models/longcat_next/model_runner.py \
  sglang_omni/models/longcat_next/merge.py \
  sglang_omni/models/longcat_next/request_builders.py \
  sglang_omni/models/longcat_next/stages.py \
  sglang_omni/models/longcat_next/sglang_model.py
```

本地 linter 检查：

```text
No linter errors found in sglang_omni/models/longcat_next
```

本地限制：当前机器没有安装 `sglang`，无法启动端到端服务；需在 H200 环境验证。

---

## 14. 已知风险点

### 14.1 HF remote code 文件缺失

动态加载依赖模型目录中的 remote code，例如：

```text
configuration_longcat_next.py
processing_longcat_next.py
modular_longcat_next_visual.py
modular_longcat_next_audio.py
modeling_longcat_ngram.py
modular_longcat_next.py
```

如果模型目录缺这些小文件，需要从 HF 同步；无需重新下载大权重。

### 14.2 Audio tokenizer strict load

当前实现整体实例化：

```python
LongcatNextAudioTokenizer
```

并 strict 加载：

```text
model.audio_tokenizer.*
```

如果 H200 上报 missing/unexpected keys，说明 HF remote code 与 checkpoint key 有版本差异。修复方向：拆成 encode-only 子模块，只加载 `audio_model` + `audio_bridge_model`。

### 14.3 `LongcatFlashModel.forward` 签名差异

当前多模态路径调用：

```python
self.model(input_ids, positions, forward_batch, input_embeds)
```

这与官方 `NmmFlashForCausalLM.forward()` 风格一致。但如果 H200 的 SGLang 版本签名不同，需要根据报错调整。

### 14.4 Tensor relay overhead

MVP 可能会让 encoder embeddings 经 shm/CPU 中转，再到 text_ar GPU。正确性验证可接受；如果性能成为瓶颈，后续改为 tensor ref / cache key，由 text_ar 侧 materialize。

### 14.5 Processor 输入格式

当前 preprocessor 会把 `inputs["images"]` / `inputs["audios"]` 追加成官方特殊 token 格式。如果请求本身已经包含特殊 token，需要确认不会重复追加。

---

## 15. Debug 记录

> 后续 H200 上的每次运行问题都记录在这里。建议按“现象 → 日志 → 初判 → 修复 → 验证”格式追加。

### Debug-000：Phase2 MVP 本地静态检查

- 时间：2026-07-26
- 环境：本地 macOS，无 `sglang` 运行环境
- 现象：无法端到端启动，仅能做静态检查
- 操作：执行 `python3 -m py_compile` 和 linter
- 结果：通过
- 结论：代码语法层面可进入 H200 验证

### Debug-001：Code review 修复记录

- 时间：2026-07-26
- 环境：本地静态修复
- 现象：review 发现 5 个问题/疑点：
  1. chunked prefill 下 `replace_positions` 使用全局位置可能越界；
  2. `visual_embedding_layer` 依赖 HF remote code 接口；
  3. `audio_text_ids` 未写入，model_runner 中相关逻辑为死代码；
  4. 多模态 pad 置 0 后对相邻文本 token 的 NgramEmbedding 有理论影响；
  5. `EntryClass` 仍是 Phase1 text pipeline，Phase2 如何选择需说明。
- 修复：
  1. `model_runner.py` 已按当前 chunk 的 `[len(req.prefix_indices), + extend_seq_len)` 过滤 positions，并映射成局部下标；
  2. 已在本文档 8.2 说明 `visual_embedding_layer` 接口来源和版本风险；
  3. 已移除 `audio_text_ids` 死代码；该逻辑属于后续音频生成/并行解码，不属于 Phase2 输入理解 MVP；
  4. 已在本文档 11.4 说明 NgramEmbedding 取舍；
  5. 已在本文档 5.3 说明 Phase2 通过 YAML `config_cls: LongcatNextPipelineConfig` 显式启用。
- 验证：`python3 -m py_compile` 通过；`read_lints` 无新增 linter 错误。

### Debug-002：第二轮 code review 结论

- 时间：2026-07-26
- 环境：本地静态修复
- 现象：review 发现/确认：
  1. `merge.py::_cat_optional` 未使用；
  2. `visual_embedding_layer` 仍需明确是否在目标 checkpoint 上验证；
  3. `Field(default_factory=...)` 用法无问题，撤回；
  4. NgramEmbedding 多模态边界影响目前缺少实测。
- 修复/结论：
  1. 已删除 `_cat_optional` 和随之无用的 `torch` import；
  2. `visual_embedding_layer` 目前只通过 HF 官方 remote code 文件确认，未在 H200 checkpoint 上实际 import/instantiate 验证。需要 H200 首次启动时验证目标模型目录的 remote code 是否与 HF main 一致；
  3. `config.py` 的 `Field(default_factory=...)` 保持不变；
  4. NgramEmbedding 的取舍目前是基于 Phase1 经验和官方多模态路径差异的理论判断，尚无实测验证。H200 上建议做 A/B：
     - A：当前实现，保留文本 ngram，pad 置 0 后覆盖；
     - B：多模态 prefill 完全绕过/禁用 ngram 或让 pad 保持真实 id 后观察质量/稳定性；
     - 对比纯文本邻近 pad 的输出质量和是否出现乱码。
- 验证：`python3 -m py_compile` 通过；`read_lints` 无新增 linter 错误。

### Debug-003：待 H200 首次启动记录

- 时间：待补充
- 命令：

```bash
sgl-omni serve \
  --config examples/configs/longcat_next_multimodal.yaml \
  --model-path /mnt/cephfs/chenzhenyang/models/LongCat-Next \
  --host 0.0.0.0 \
  --port 8100
```

- 现象：待补充
- 日志：待补充
- 初判：待补充
- 修复：待补充
- 验证：待补充

---

## 16. 后续优化方向

正确性通过后再考虑：

1. encoder batching：`SimpleScheduler(batch_compute_fn=...)`；
2. encoder output cache：图片/音频重复输入缓存；
3. tensor ref relay：避免 D2H/H2D；
4. encoder 与 AR 共卡时的 `encoder_mem_reserve`；
5. image/audio generation head 接入；
6. CUDA graph / torch.compile。

当前阶段不建议让这些优化阻塞 MVP 正确性验证。
