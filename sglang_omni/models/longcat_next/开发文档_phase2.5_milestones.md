## Phase2.5 开发分步计划

### Step 0：先做观测能力，不改性能逻辑

目标：先把“启动慢”和“0.6 tokens/s”定位清楚。

改动：

- 给启动链路加耗时日志：
  - `create_preprocessing_executor`
  - `create_image_encoder_executor`
  - `create_audio_encoder_executor`
  - `create_longcat_next_text_executor`
  - `LongcatNextImageEncoder.__init__`
  - `LongcatNextAudioEncoder.__init__`
- 给请求链路加轻量耗时统计：
  - preprocessing 耗时
  - image/audio encoder 耗时
  - `mm_aggregate` 耗时
  - `text_ar` prefill / decode 粗粒度耗时
- 在文档里记录 H200 baseline：
  - 启动总耗时
  - `15/15` 后到 ready 耗时
  - 纯文本第二条 steady-state tokens/s
  - 图片/音频 E2E latency

这一步最重要，因为它决定后面优化是否真的有效。

---

### Step 1：实现 P0 Encoder exact-match cache

目标：同一图片 / 同一音频重复请求时跳过 GPU encoder。

改动：

- 新增 cache key 生成：
  - 本地文件：`path + size + mtime_ns`
  - 后续再支持 bytes/base64 hash
- `LongcatNextPreprocessor` 给 encoder input 增加：
  - `cache_key`
- `image_encoder` / `audio_encoder` executor 持有 `StageOutputCache`
- encode 前：
  - hit：直接返回 cached `visual_embeds/audio_embeds`
  - miss：跑 encoder，写 cache
- 加 cache trace 日志：
  - `hit`
  - `miss`
  - `store`
  - `evict`

验证：

- 同图不同 prompt：第二次 hit。
- 同音频不同 prompt：第二次 hit。
- 修改文件后 key 变化。
- 输出与无 cache 基本一致。

这是第一项真正性能优化，收益高、风险最低。

---

### Step 2：实现 P1 TensorRef / lazy relay

目标：让 `mm_aggregate` 不 materialize 大 embedding，只传 ref / metadata。

改动：

- 配置或代码中启用 TensorRef：
  - `visual_embeds`
  - `audio_embeds`
- 设置 edge：
  - `image_encoder -> mm_aggregate -> text_ar`
  - `audio_encoder -> mm_aggregate -> text_ar`
- 首版不强制改 `merge.py`：
  - 当前 `is not None` 对 TensorRef dict 已经成立。
  - `_non_empty` 后续再补。
- 在日志中确认：
  - `stage_hop_sent` 出现 `ref_count/ref_bytes`
  - `mm_aggregate` 不读取大 tensor
  - `text_ar` 正常 materialize

注意边界：

- 这不是零拷贝。
- 仍然有 `encoder/cache -> SHM/blob -> text_ar GPU` 搬运。
- 它省的是 `mm_aggregate` 中间序列化/反序列化/materialize。

验证：

- 图片 / 音频 E2E 输出正常。
- 大图 relay 时间下降。
- cache hit + TensorRef 组合路径正常。

---

### Step 3：独立验证 overlap schedule

目标：在 CUDA Graph 前先验证 overlap 本身。

配置实验：

```text
disable_cuda_graph=True
disable_overlap_schedule=False
enable_async_decode=False
```

为什么先做这个：

- overlap 有自己的 CPU/GPU 同步复杂性。
- 多模态 prefill + decode 混跑可能暴露调度问题。
- 不应该和 CUDA Graph 同时首次开启。

验证 case：

- 纯文本短 prompt
- 纯文本长 prompt
- 图片 + 文本
- 音频 + 文本
- batch size > 1

指标：

- 输出正确性
- 是否卡死
- 是否有 batch/retract/chunked prefill 异常
- TPOT 是否改善
- GPU util 是否更平滑

如果这一步不稳定，先不要开 CUDA Graph。

---

### Step 4：验证 CUDA Graph decode

目标：恢复 `text_ar` decode CUDA Graph。

分两层：

#### 4.1 CUDA Graph-only

```text
disable_cuda_graph=False
disable_overlap_schedule=True
enable_async_decode=False
```

先测纯文本，再测图片/音频。

重点确认：

- prefill 仍走普通 extend path。
- decode 阶段 `replace_embeds is None`。
- `LongcatNextTextForCausalLM.forward()` decode 回到 `super().forward`。
- MoE zero-expert patch 和 ngram 路径 graph-safe。

#### 4.2 CUDA Graph + overlap

```text
disable_cuda_graph=False
disable_overlap_schedule=False
enable_async_decode=False
```

这一步在 Step 3 和 4.1 都稳定后再做。

验证：

- 纯文本 batch > 1
- 图片/音频混跑
- 长生成
- 多轮重复请求

---

### Step 5：最后验证 async decode

目标：进一步隐藏 D2H / CPU resolve 开销。

配置：

```text
disable_cuda_graph=False
disable_overlap_schedule=False
enable_async_decode=True
```

前提：

- overlap-only 已稳定。
- CUDA Graph-only 已稳定。
- graph + overlap 已稳定。

重点验证：

- batch size = 1 时是否自动退回同步路径。
- batch size > 1 时是否收益明显。
- sampling 参数：
  - repetition penalty
  - frequency/presence penalty
  - min_new_tokens
- 输出是否和 sync 路径一致。

这一步风险最高，应该最后做。

---

## 推荐实际里程碑

### Milestone A：可观测 baseline

产出：

- 启动耗时日志
- 请求阶段耗时日志
- H200 baseline 表格

### Milestone B：P0 cache 可用

产出：

- image/audio encoder cache
- cache hit/miss 日志
- 同媒体多 prompt benchmark

### Milestone C：P1 TensorRef 可用

产出：

- `visual_embeds/audio_embeds` lazy relay
- `mm_aggregate` 不 materialize 大 tensor
- cache + TensorRef 组合验证

### Milestone D：overlap schedule 稳定

产出：

- overlap-only 验证结果
- 多模态混跑无异常

### Milestone E：CUDA Graph / async decode

产出：

- CUDA Graph-only
- graph + overlap
- async decode
- 最终 throughput / latency 对比

---

## 一句话总结

Phase2.5 不应该直接等于“三个优化一起做”。更稳妥的开发顺序是：

```text
观测打点
 -> Encoder Cache
 -> TensorRef lazy relay
 -> overlap-only
 -> CUDA Graph-only
 -> CUDA Graph + overlap
 -> async decode
```

这样每一步都有独立收益，也能在 H200 上明确定位问题来源。