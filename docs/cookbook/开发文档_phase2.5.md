## 0. 当前最新代码基线

`65b4d15` 主要修了 H200 实跑问题：

- `dynamic.py`：兼容 `flash-attn v4`，补旧 API shim。
- `preprocessor.py`：messages 走 `apply_chat_template`，保证 OpenAI chat 输入更正确。
- `model_runner.py`：修复空 tensor boolean context。
- `request_builders.py`：修 usage 字段。
- 新增 `longcat_next_debug.md`，记录 H200 E2E 验证。

当前已经验证：

- 纯文本：`preprocessing -> mm_aggregate -> text_ar`
- 图片：`preprocessing -> image_encoder -> mm_aggregate -> text_ar`
- 音频：`preprocessing -> audio_encoder -> mm_aggregate -> text_ar`

所以现在优化的前提是比较好的：**功能路径已经跑通，可以开始做性能闭环**。

---

## 1. Encoder exact-match cache：我建议第一优先级

### 当前状态

LongCat-Next 当前的 encoder stage 是每次请求都执行 GPU encoder：

```48:63:sglang_omni/models/longcat_next/stages.py
    model = LongcatNextImageEncoder(model_path, device=device, dtype=dtype)

    def _encode(payload):
        state = LongcatNextPipelineState.from_dict(payload.data)
        inputs = state.encoder_inputs.get(IMAGE_STAGE) or {}
        if not inputs:
            state.encoder_outs[IMAGE_STAGE] = {}
            return payload_with_state(payload, state)
        result = model(
            pixel_values=inputs["pixel_values"],
            visual_grid_thw=inputs["visual_grid_thw"],
        )
        state.encoder_outs[IMAGE_STAGE] = result
        return payload_with_state(payload, state)
```

```82:98:sglang_omni/models/longcat_next/stages.py
    model = LongcatNextAudioEncoder(model_path, device=device, dtype=dtype)

    def _encode(payload):
        state = LongcatNextPipelineState.from_dict(payload.data)
        inputs = state.encoder_inputs.get(AUDIO_STAGE) or {}
        if not inputs:
            state.encoder_outs[AUDIO_STAGE] = {}
            return payload_with_state(payload, state)
        result = model(
            audio=inputs["audio"],
            encoder_length=inputs["encoder_length"],
            bridge_length=inputs["bridge_length"],
        )
        state.encoder_outs[AUDIO_STAGE] = result
        return payload_with_state(payload, state)
```

也就是说，现在同一张图、同一段音频被不同 prompt 重复使用时，会重复跑：

- image tokenizer / visual tokenizer
- audio tokenizer
- codebook embedding
- projection / visual embedding layer

这正好对应 PDF 里的 **Encoder LRU Caching**。PDF 的核心逻辑是：encoder 输出对相同输入是 deterministic 的，所以可以用内容哈希做 key，命中后跳过 GPU encoder。

当前仓库已经有通用的 `StageOutputCache`：

```41:87:sglang_omni/scheduling/stage_cache.py
class StageOutputCache:
    """Small in-memory LRU cache for non-AR stage outputs."""

    def __init__(
        self,
        max_size: int | None = None,
        max_bytes: int | None = None,
        cache_device: torch.device | str | None = None,
        size_fn: Callable[[Any], int] | None = None,
    ) -> None:
        ...
    def get(self, key: str | None) -> Any | None:
        ...
    def put(self, key: str | None, data: Any) -> None:
        ...
```

所以 LongCat 不需要从零写缓存结构，应该直接复用它。

### 关键设计点

#### cache key 应该在哪里生成？

最合理位置是 `LongcatNextPreprocessor`，因为它最早能看到原始用户输入路径：

```126:129:sglang_omni/models/longcat_next/components/preprocessor.py
        for image in inputs.get("images") or []:
            text += f"\n{self.processor.image_start_token}{image}{self.processor.image_end_token}"
        for audio in inputs.get("audios") or []:
            text += f"\n{self.processor.audio_start_token}{audio}{self.processor.audio_end_token}"
```

但当前 `encoder_inputs` 只放了 tensor 和 positions，没有放 `cache_key`：

```67:89:sglang_omni/models/longcat_next/components/preprocessor.py
        if visual_inputs is not None and image_positions.numel() > 0:
            vi = dict(visual_inputs)
            visual_grid_thw = vi.get("visual_grid_thw")
            if visual_grid_thw is None:
                visual_grid_thw = vi.get("image_grid_thw")
            encoder_inputs[IMAGE_STAGE] = {
                "pixel_values": vi.get("pixel_values"),
                "visual_grid_thw": visual_grid_thw,
                "image_positions": image_positions,
            }

        if audio_inputs is not None and audio_positions.numel() > 0:
            ai = dict(audio_inputs)
            audio = ai.get("audio")
            encoder_length = ai.get("encoder_length")
            bridge_length = ai.get("bridge_length")
            if audio is not None:
                encoder_inputs[AUDIO_STAGE] = {
                    "audio": audio,
                    "encoder_length": encoder_length,
                    "bridge_length": bridge_length,
                    "audio_positions": audio_positions,
                }
```

建议新增：

- `encoder_inputs["image_encoder"]["cache_key"]`
- `encoder_inputs["audio_encoder"]["cache_key"]`

key 生成策略可以分两级：

| 场景 | key 策略 | 说明 |
|---|---|---|
| 本地文件路径 | `path + size + mtime_ns` | 快，适合 benchmark / 生产本地文件 |
| bytes / base64 / URL 下载后内容 | `sha256(bytes)` | 最准确 |
| tensor-only fallback | hash tensor bytes / shape / dtype | 成本高，只作为 fallback |

我建议 MVP 先支持本地路径，因为当前 LongCat debug case 就是本地路径：

```147:150:sglang_omni/models/longcat_next/longcat_next_debug.md
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/mnt/cephfs/chenzhenyang/models/LongCat-Next","messages":[{"role":"user","content":"Describe this image."}],"images":["/mnt/cephfs/chenzhenyang/czy/sglang-omni/tests/data/cars.jpg"],"max_tokens":100}'
```

```165:168:sglang_omni/models/longcat_next/longcat_next_debug.md
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/mnt/cephfs/chenzhenyang/models/LongCat-Next","messages":[{"role":"user","content":"What is this audio about?"}],"audios":["/mnt/cephfs/chenzhenyang/czy/sglang-omni/tests/data/query_to_cars.wav"],"max_tokens":100}'
```

#### cache value 放什么？

建议 cache value 就是 encoder stage result：

- image:
  - `visual_ids`
  - `visual_embeds`
- audio:
  - `audio_ids`
  - `audio_embeds`

当前 encoder 输出是：

```115:118:sglang_omni/models/longcat_next/components/encoders.py
        return {
            "visual_ids": visual_ids.detach(),
            "visual_embeds": visual_embeds.reshape(-1, visual_embeds.shape[-1]).detach(),
        }
```

```170:173:sglang_omni/models/longcat_next/components/encoders.py
        return {
            "audio_ids": audio_ids.detach(),
            "audio_embeds": audio_embeds.reshape(-1, audio_embeds.shape[-1]).detach(),
        }
```

这非常适合直接缓存。

### 风险

低风险，但有几个注意点：

1. **路径 key 不能只用 path**  
   CephFS 上文件可能被覆盖，同 path 不同内容会污染 cache。至少加 `mtime_ns + size`。

2. **多图片 / 多音频**  
   当前 LongCat 代码看起来把多个 image/audio 合并进一个 processor 文本。cache key 需要支持列表组合，例如：
   - 单图：`image:<file_key>`
   - 多图：`image-list:<key1>|<key2>|...`
   audio 同理。

3. **cache device 选择**  
   如果 image encoder 在 GPU0，audio encoder 在 GPU1，cache 放 GPU 可以避免命中后 H2D，但占显存。建议：
   - 默认先 `cache_device=None`：保留原 device。
   - 配 `max_bytes`，避免 H200 显存被 cache 占满。
   - 后续如果显存紧张，再考虑 cache 到 CPU pinned memory。

### 预期收益

最明显的场景：

- benchmark 同一媒体多 prompt
- 用户围绕同一图片追问
- 同一音频多轮问答 / 多任务

收益大概来自跳过：

- image/audio encoder 前向
- codebook embedding
- projection
- encoder stage GPU 排队

我认为这是最适合先做的，因为**和 AR 路径解耦，不影响生成正确性**。

---

## 2. 避免 `mm_aggregate` D2H/H2D：第二优先级

### 当前风险点

当前 LongCat 的 `mm_aggregate` 是逻辑上的 CPU/control-plane stage，但它会携带 encoder output：

```42:48:sglang_omni/models/longcat_next/merge.py
    longcat_mm_inputs: dict[str, Any] = {}
    if image_out.get("visual_embeds") is not None and image_positions is not None:
        longcat_mm_inputs["image_embeds"] = image_out["visual_embeds"]
        longcat_mm_inputs["image_positions"] = image_positions
    if audio_out.get("audio_embeds") is not None and audio_positions is not None:
        longcat_mm_inputs["audio_embeds"] = audio_out["audio_embeds"]
        longcat_mm_inputs["audio_positions"] = audio_positions
```

然后 `text_ar` 的 `before_prefill` 再把 embedding 搬到 AR device：

```54:78:sglang_omni/models/longcat_next/model_runner.py
            for key in ("image", "audio"):
                embeds = mm.get(f"{key}_embeds")
                positions = mm.get(f"{key}_positions")
                ...
                chunk = embeds[offset : offset + selected_count]
                ...
                replace_embeds_parts.append(chunk.to(device=device))
                replace_positions_parts.append(local_positions.to(device=device))
```

这意味着只要 `embeds` 到达 `text_ar` 时不在目标 GPU，就会发生 H2D / GPU-to-GPU / CPU-to-GPU 拷贝。

更关键的是，stage 间 relay 默认会把 payload tensor 抽出来再通过 relay 传：

```429:506:sglang_omni/pipeline/relay_io.py
async def write_payload(
    relay: Relay,
    request_id: str,
    payload: StagePayload,
    *,
    from_stage: str | None = None,
    to_stage: str | None = None,
    tensor_ref_policy: TensorRefPolicy | None = None,
) -> tuple[dict[str, Any], Any]:
    ...
    if tensor_ref_policy is not None:
        ...
    else:
        modified_data, tensor_dict = extract_tensors(payload.data)
    ...
    if tensor_dict:
        ...
        for path, tensor in tensor_dict.items():
            flat = tensor.contiguous().view(torch.uint8).reshape(-1)
            if flat.device != transport_device:
                flat = flat.to(device=transport_device)
```

如果 relay backend 是 CPU/shm，那 GPU encoder output 就可能变成：

```text
image_encoder GPU0 / audio_encoder GPU1
  -> relay CPU/shm
  -> mm_aggregate CPU
  -> relay CPU/shm
  -> text_ar GPU2
```

这就是你说的 `encoder GPU -> CPU/shm -> text_ar GPU`。

### 好消息：仓库已经有 TensorRef 机制

当前通用 pipeline 里已经有 lazy tensor handoff：

```1:8:sglang_omni/pipeline/tensor_ref.py
"""Lazy tensor handoff for large multimodal tensors crossing pipeline stages.

A ``TensorRef`` stands in for a tensor that has been externalized to the
relay rather than inlined into a ``StagePayload``. Intermediate stages that
only forward the value (e.g. ``mm_aggregate``) never materialize it; only
the declared ``consumer_stage`` resolves it back into a real tensor.
"""
```

runtime 会在当前 stage 是 consumer 时才 materialize：

```711:715:sglang_omni/pipeline/stage/runtime.py
        payload_for_scheduler = payload
        if tensor_refs_enabled():
            payload_for_scheduler = await relay_io.materialize_payload_tensor_refs(
                self.relay, payload, current_stage=self.name
            )
```

发送 stage payload 时也已经会按环境变量生成 policy：

```1034:1044:sglang_omni/pipeline/stage/runtime.py
        tensor_ref_policy = TensorRefPolicy.from_env(
            from_stage=self.name, to_stage=target
        )
        metadata, op = await relay_io.write_payload(
            self.relay,
            request_id,
            projected_payload,
            from_stage=self.name,
            to_stage=target,
            tensor_ref_policy=tensor_ref_policy,
        )
```

但默认 allowlist 里没有 LongCat 的字段：

```18:25:sglang_omni/pipeline/tensor_ref.py
TENSOR_REF_MARKER = "__sglang_omni_tensor_ref__"
DEFAULT_TENSOR_REF_THRESHOLD_MB = 2.0
DEFAULT_TENSOR_REF_PATHS = (
    "video_embeds",
    "deepstack_visual_embeds_image",
    "deepstack_visual_embeds_video",
)
```

所以当前 LongCat 的 `visual_embeds` / `audio_embeds` 大概率还没有走 lazy ref。

### 建议的落地方式

这里我建议分两步，而不是直接做复杂的 GPU-to-GPU 零拷贝。

#### Step 2.1：先启用已有 TensorRef lazy materialization

目标不是完全消灭 D2H/H2D，而是先避免 `mm_aggregate` materialize 大 tensor，减少中间 stage 的重复搬运。

配置大概是这种意图：

```text
SGLANG_OMNI_ENABLE_TENSOR_REFS=1
SGLANG_OMNI_TENSOR_REF_EDGES=image_encoder:mm_aggregate:text_ar,audio_encoder:mm_aggregate:text_ar
SGLANG_OMNI_TENSOR_REF_PATHS=visual_embeds,audio_embeds
SGLANG_OMNI_TENSOR_REF_THRESHOLD_MB=1
```

逻辑：

```text
image_encoder/audio_encoder
  -> 对 visual_embeds/audio_embeds 生成 TensorRef
  -> mm_aggregate 只看 ref，不读 tensor
  -> text_ar 是 consumer，才 materialize
```

这和 PDF 里的思想一致：**中间控制面只传 metadata / ref，不碰大 tensor**。

#### Step 2.2：LongCat merge 的首版适配边界

当前 `merge.py` 里判断 `image_out.get("visual_embeds") is not None` / `audio_out.get("audio_embeds") is not None`，对 TensorRef dict 同样成立：TensorRef 是一个 dict，不是 `None`；encoder 无输入时 `.get()` 返回 `None`，仍会按现有逻辑跳过。

因此，首版启用 TensorRef 时**不必须**引入 `_non_empty` helper。只有在未来出现 encoder 可能产出 shape=0 tensor / 空 TensorRef 的场景时，才需要补充类似 Qwen3-Omni 的 TensorRef-aware non-empty 判断，作为后续健壮性优化即可。

Qwen3-Omni 中可参考的 helper 是：

```30:37:sglang_omni/models/qwen3_omni/merge.py
def _non_empty(value: Any) -> bool:
    if value is None:
        return False
    if is_tensor_ref_dict(value):
        return tensor_ref_numel(value) > 0
    if isinstance(value, torch.Tensor):
        return value.numel() > 0
    return False
```

但 LongCat 第一版可以先不引入，避免扩大改动范围。

#### Step 2.3：结合 encoder cache 的收益边界

第二项优化和第一项优化可以组合：

- cache miss：encoder 输出 `visual_embeds/audio_embeds`，通过 TensorRef lazy relay。
- cache hit：直接返回 cached embedding，再通过 TensorRef 或本地 payload 送到 `text_ar`。
- `mm_aggregate` 永远只做 metadata fan-in，不主动 `.to()` / `.cpu()`。

需要明确的是：**cache hit + TensorRef 并不等于零拷贝**。当前 relay 语义下仍然存在首尾搬运：

```text
encoder/cache 所在进程/GPU -> relay SHM/blob -> text_ar 目标 GPU
```

TensorRef 主要节省的是 `mm_aggregate` 这个中间控制面阶段对大 embedding 的序列化、反序列化和 materialize；它不会自动消除 encoder 到 text_ar 的最终跨设备传输。即便如此，在同媒体多 prompt 的 benchmark / 调试场景中，P0 跳过 encoder GPU 计算，P1 减少中间 payload 搬运，组合收益仍然显著。

### 风险

中等风险，主要是：

1. **TensorRef 不是零拷贝**  
   它避免 `mm_aggregate` 中间 materialize 大 tensor，但 encoder/cache 到 `text_ar` 之间仍然会经过 relay blob/SHM，再由 `text_ar` materialize 到目标 GPU。首版目标应定义为“减少中间 stage 搬运”，不是“完全 GPU 零拷贝”。

2. **位置 tensor 不应 externalize**  
   `image_positions/audio_positions` 很小，没必要 ref，直接 inline 即可。

3. **跨 GPU materialize 仍可能有一次搬运**  
   即便 TensorRef 避免了 `mm_aggregate` 中间读取，最后 `text_ar` 还是要把 embedding 放到 GPU2/3/4/5 的对应 worker。它优化的是“不在中间 stage 反复搬”，不是完全零拷贝。

> 修正说明：这里不再把 TP fanout / multi-reader 作为 LongCat P1 风险。`OmniScheduler.requires_tp_work_fanout = False`，`text_ar` 不走 pipeline runtime 的 payload fanout；TensorRef blob 只会被 leader 单次读取，不存在 multi-reader 竞争。`runtime.py` 中关于 unresolved refs fanout 的注释针对需要 TP work fanout 的 SimpleScheduler / ThreadedSimpleScheduler 场景，不适用于当前 LongCat `text_ar`。

### 预期收益

对图片尤其明显。debug 文档里图片 prompt token 是 4011：

```153:158:sglang_omni/models/longcat_next/longcat_next_debug.md
{
  "choices": [{"message": {"content": "The image is a composite of two distinct scenes: 1. Left Section ... a close-up shot of a person's hand ... 2. Right Section ... a red, vintage-style convertible car ..."}}],
  "usage": {"prompt_tokens": 4011, "completion_tokens": 269, "total_tokens": 4280}
}
```

4000 级视觉 token 的 embedding 很大，反复走 CPU/shm relay 会很亏。这个优化在大图、多图时收益更明显。

---

## 3. 恢复 `text_ar` CUDA Graph / async decode：第三优先级

### 当前状态

LongCat 当前显式关闭了 CUDA Graph 和 overlap schedule：

```139:149:sglang_omni/models/longcat_next/stages.py
    overrides = build_generation_batch_overrides(
        max_running_requests=max_running_requests,
        server_args_overrides=server_args_overrides,
        disable_cuda_graph=True,
        disable_overlap_schedule=True,
        enable_torch_compile=enable_torch_compile,
        mem_fraction_static=mem_fraction_static,
        max_prefill_tokens=16384,
        chunked_prefill_size=16384,
        sampling_backend="pytorch",
        dtype=dtype,
    )
```

但下面已经接了 deferred CUDA graph 初始化逻辑：

```165:183:sglang_omni/models/longcat_next/stages.py
    want_cuda_graph, (
        model_worker,
        tree_cache,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        prefill_mgr,
        decode_mgr,
        model_config,
    ) = create_sglang_infrastructure_defer_cuda_graph(
        server_args,
        gpu_id,
        tp_rank=tp_rank,
        nccl_port=nccl_port,
        model_arch_override="LongcatNextTextForCausalLM",
    )

    if want_cuda_graph:
        model_worker.model_runner.init_device_graphs()
```

说明基础设施层面不是不能开，而是当前为了 Phase2 稳定性先关了。

### 为什么 LongCat decode 理论上适合 CUDA Graph？

PDF 里讲的核心是：

- AR decode 每步有很多小 kernel。
- eager 模式每步都有 CPU dispatch 开销。
- 每步采样后还要 D2H 读 token / done 状态。
- batch size 变大后，CPU/GPU 同步开销会堆积。
- CUDA Graph 可以把固定形状 decode step capture/replay。
- async decode 可以把 D2H 和下一步 GPU compute overlap。

LongCat-Next 是 68B MoE，decode 阶段很重。虽然 prefill 有 multimodal embedding injection，但 decode 阶段实际上会走普通 token 输入。

当前 `sglang_model.py` 的 forward 也支持这个判断：

```294:307:sglang_omni/models/longcat_next/sglang_model.py
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: "ForwardBatch",
    ) -> torch.Tensor:
        replace_embeds = getattr(forward_batch, "longcat_replace_embeds", None)
        if replace_embeds is None:
            return super().forward(input_ids, positions, forward_batch)

        input_embeds = self._build_longcat_input_embeds(input_ids, forward_batch)
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
```

这意味着：

- multimodal prefill：`replace_embeds != None`，走自定义 input embedding patch。
- decode：通常 `replace_embeds is None`，回到 `super().forward`。

所以我判断：**decode-only CUDA Graph 是可尝试的**，比“prefill + decode 全量 graph”安全很多。

### 建议的分阶段验证

#### Step 3.1：先独立验证 overlap schedule

在开启 CUDA Graph 之前，先单独验证 overlap schedule：

```text
disable_cuda_graph=True
disable_overlap_schedule=False
```

目标：

- 验证 overlap schedule 在 LongCat 多模态 prefill + decode 混跑下是否稳定。
- 单独观察 CPU/GPU 同步、running batch 切换、chunked prefill 与 decode 交错是否有问题。
- 避免首次实验同时引入 CUDA Graph 和 overlap，导致问题定位困难。

这一步需要分别覆盖：

- 纯文本短 prompt；
- 纯文本长 prompt；
- 图片 + 文本；
- 音频 + 文本；
- batch size > 1。

#### Step 3.2：纯文本再开 CUDA Graph

在 overlap-only 稳定后，再启动 text-only 或 multimodal pipeline 的纯文本请求：

```text
disable_cuda_graph=False
disable_overlap_schedule=True
```

目标：

- 验证 LongCat-Next text AR + ngram + MoE zero-expert patch 是否 graph-safe。
- 对比输出是否正常。
- 对比 decode latency。

#### Step 3.3：多模态 prefill + decode graph

再测图片/音频请求：

- prefill 仍然走非 graph / extend path。
- decode 阶段 graph replay。

这里主要验证 `forward_batch.longcat_replace_embeds` 是否只在 prefill 存在，decode 中能稳定为 `None`。当前 `model_runner.before_prefill` 只在 extend 时设置 replacement：

```16:19:sglang_omni/models/longcat_next/model_runner.py
    def before_prefill(self, forward_batch: Any, schedule_batch: Any, requests: list) -> None:
        del requests
        if not schedule_batch.forward_mode.is_extend():
            return
```

这对 decode graph 是有利的：decode 阶段不应该被 multimodal replacement 污染。

#### Step 3.4：最后考虑 CUDA Graph + overlap / async decode

我不建议一开始就同时开：

```text
disable_cuda_graph=False
disable_overlap_schedule=False
enable_async_decode=True
```

因为 LongCat 有几类额外风险：

- ngram embedding 的 forward-batch 依赖。
- MoE zero-expert patch。
- `flashinfer_cutlass` MoE backend。
- TP=4。
- multimodal prefill replacement。
- chunked prefill。

应该逐个打开：

1. CUDA Graph off，overlap on：先独立验证 overlap schedule。
2. CUDA Graph on，overlap off：验证 decode graph 本身。
3. CUDA Graph on，overlap on：验证 graph 与 overlap 组合。
4. async decode on：最后验证 one-step lookahead。
5. batch size > 1 压测。

### 风险

中高风险，主要不是概念问题，而是工程验证问题：

1. **LongCat MoE + CUDA Graph**  
   MoE routing shape稳定，但 expert 分布动态。一般 graph 可以支持，但具体 backend 是否 capture-safe 要实测。

2. **sampling backend**  
   当前是 `sampling_backend="pytorch"`。async decode 和 graph 的收益可能受 sampling 实现影响。

3. **chunked prefill**  
   prefill 不一定 graph，但 decode graph 不能被 chunked prefill 状态污染。

4. **多模态 replacement buffer**  
   必须确保 `forward_batch.longcat_replace_embeds` decode 阶段为 `None`，否则 graph capture 可能看到动态 input_embeds 路径。

### 预期收益

这个优化对单请求 latency 可能不一定最明显，但对并发 decode 阶段会重要。尤其 LongCat-Next：

- completion tokens 较长；
- decode step 多；
- MoE 模型 kernel 数多；
- H200 上 GPU 很快，CPU launch/sync overhead 更容易暴露。

---

## 4. 我建议的实际落地路线

### 第一阶段：只做 Encoder Cache

目标：不改 AR，不碰 TensorRef，不碰 CUDA Graph。

改动范围：

- `components/preprocessor.py`
  - 为 image/audio encoder_inputs 增加 `cache_key`。
- `stages.py`
  - image/audio executor 持有 `StageOutputCache`。
  - encode 前查 cache，miss 后执行，执行完 put。
- 可能新增一个小工具：
  - `components/cache_key.py` 或复用已有 `preprocessing/cache_key.py`。

验证指标：

| case | 预期 |
|---|---|
| 同图片 + 不同 prompt 连续请求 | 第一次 miss，第二次 hit |
| 同音频 + 不同 prompt 连续请求 | 第一次 miss，第二次 hit |
| 修改文件内容后再请求 | key 改变，miss |
| 多图/多音频 | key 组合稳定 |
| 输出文本 | 与无 cache 一致或近似一致 |

日志建议：

```text
encoder_cache stage=image_encoder action=miss/hit/store key=...
encoder_cache stage=audio_encoder action=miss/hit/store key=...
```

Qwen3-Omni 已有类似 trace，可以参考：

```288:312:sglang_omni/models/qwen3_omni/stages.py
def _trace_encoder_cache(
    stage_name: str,
    action: str,
    *,
    request_id: str,
    cache_key: str | None,
    input_bytes: int | None = None,
    output_bytes: int | None = None,
    detail: str | None = None,
) -> None:
    ...
    logger.info("encoder_cache %s", " ".join(parts))
```

---

### 第二阶段：LongCat TensorRef / lazy relay

目标：`mm_aggregate` 不 materialize 大 embedding。

改动范围可能很小：

- 环境变量配置：
  - enable tensor refs
  - edge policy
  - path allowlist 加 `visual_embeds,audio_embeds`
- `merge.py`
  - 首版不必新增 `_non_empty`；现有 `is not None` 判断已兼容 TensorRef dict。
  - 如未来支持 shape=0 encoder 输出，再补 TensorRef-aware non-empty helper。
- 视情况文档化 `longcat_next_multimodal.yaml` 启动参数。

验证指标：

| 指标 | 预期 |
|---|---|
| stage_hop_sent metadata | 出现 `ref_count/ref_bytes` |
| mm_aggregate 显存/CPU 拷贝 | 不 materialize large embeds |
| text_ar 输入 | 能正确 materialize |
| E2E 图片/音频输出 | 正常 |
| 大图 latency | relay 时间下降 |

这里我建议优先用已有 `TensorRefPolicy.from_env`，不要先写 LongCat 专用复杂 relay。

---

### 第三阶段：CUDA Graph / async decode

目标：先 decode graph，再 async。

建议配置实验矩阵：

| 实验 | cuda graph | overlap | async decode | 请求 |
|---|---:|---:|---:|---|
| A | off | off | off | 当前 baseline |
| B | off | on | off | 纯文本 + 图片/音频，先独立验证 overlap |
| C | on | off | off | 纯文本，验证 decode graph |
| D | on | off | off | 图片/音频，验证多模态 prefill 后 decode graph |
| E | on | on | off | 并发纯文本 + 多模态混跑 |
| F | on | on | on | batch size > 1 |

关键指标：

- TTFT
- TPOT / decode token latency
- end-to-end latency
- GPU utilization
- CPU utilization
- CUDA graph capture 是否失败
- 输出是否异常
- illegal memory access / NCCL / flashinfer 错误

---

## 5. 一个重要补充：Encoder Cache 和 RadixCache 可以联动，但别第一版就做复杂

Qwen3-Omni 里有 `media_cache_keys`，用媒体 cache key 影响 placeholder token 的 hashed pad value，从而让 RadixCache 对“同媒体 prefix”更友好：

```158:172:sglang_omni/models/qwen3_omni/merge.py
    media_cache_keys: dict[str, str] = {}
    encoder_inputs = state.encoder_inputs or {}
    image_ck = (encoder_inputs.get("image_encoder") or {}).get("cache_key")
    audio_ck = (encoder_inputs.get("audio_encoder") or {}).get("cache_key")
    if image_ck:
        media_cache_keys["image"] = f"image:{image_ck}"
        ...
    if audio_ck:
        media_cache_keys["audio"] = f"audio:{audio_ck}"

    result: dict[str, Any] = {"model_inputs": thinker_model_inputs}
    if media_cache_keys:
        result["media_cache_keys"] = media_cache_keys
```

LongCat 当前还没做这个。我的建议是：

- 第一版只做 encoder output cache。
- 等稳定后再考虑 “media cache key -> prefix/RadixCache namespace”。
- 因为 LongCat 有 `NgramEmbedding`，placeholder token 的处理比 Qwen3 更敏感，不宜一开始就改 cache key token 化逻辑。

---

## 6. 总结结论

我建议你按这个优先级推进：

### P0：Encoder exact-match cache

**最值得先做。**

- 复用 `StageOutputCache`。
- preprocessor 生成 `cache_key`。
- image/audio executor 查 cache。
- 风险最低，收益最确定。
- 对 benchmark 和多轮同媒体追问非常有价值。

### P1：TensorRef / lazy relay

**第二做。**

- 当前 pipeline 已经有机制。
- LongCat 需要把 `visual_embeds/audio_embeds` 加入 allowlist。
- `mm_aggregate` 只传 ref，不碰大 tensor。
- 能减少中间 stage 的 CPU/shm 搬运。
- 但它不是零拷贝；encoder/cache 到 `text_ar` 的首尾 relay 搬运仍然存在。

### P2：CUDA Graph / async decode

**第三做。**

- LongCat decode 理论适合 CUDA Graph。
- 当前代码结构也支持 decode 回到 `super().forward`。
- 但 MoE、ngram、TP、多模态 prefill 注入都需要逐步验证。
- 验证顺序应为：先 overlap-only，再 CUDA Graph-only，再组合 overlap + graph，最后 async decode。

---

## 7. 启动慢与单请求 0.6 tokens/s 的诊断

### 7.1 现象拆解

当前观察到两个启动卡点：

1. **启动初期长时间没有日志**。
2. **模型 shard 加载进度到 `15/15` 后仍然等待很久**。

这两个卡点通常不是同一个问题：

- “启动初期无日志”更可能发生在 stage 进程创建、Python import、HF remote code / processor 初始化、flash-attn 兼容 shim、image/audio encoder 构造阶段。
- “`15/15` 后很久”说明 safetensors shard 已经读完，但后续还在做 CUDA/NCCL 初始化、KV cache memory profiling、memory pool / RadixCache 创建、Triton/FlashInfer/CUTLASS kernel JIT、ngram buffer 初始化等 post-load 工作。

当前 LongCat multimodal pipeline 是 5-stage：

```text
preprocessing
  -> image_encoder
  -> audio_encoder
  -> mm_aggregate
  -> text_ar(TP=4)
```

因此启动时不是只启动一个 AR worker，而是同时启动 preprocessing、image encoder、audio encoder、aggregate，以及 `text_ar` 的 4 个 TP rank。任何一个阶段缺少细粒度日志，都会表现成“黑屏等待”。

### 7.2 启动慢的主要怀疑点

#### A. CephFS 模型 I/O 波动

`longcat_next_debug.md` 已记录 CephFS shard 加载间隙可能达到 59s / 95s，总体 15 个 shard 耗时约 2 分钟。当前 YAML 使用：

```yaml
model_path: /mnt/cephfs/chenzhenyang/models/LongCat-Next
relay_backend: shm
```

这会让启动时间高度依赖 CephFS 带宽和并发负载。

更重要的是，`text_ar` 是 TP=4，多个 rank 可能并行读取 checkpoint；同时 image/audio encoder 也会加载各自 tokenizer 权重和 codebook embedding 切片，进一步放大 I/O 压力。

#### B. image/audio encoder 额外读取 `embed_tokens.weight`

当前 image/audio encoder 都会构造 `_OffsetCodebookEmbedding`，其中会通过 `load_weights_by_prefix(model_path, prefix="model.embed_tokens.")` 读取 `model.embed_tokens.weight` 并切片：

```49:58:sglang_omni/models/longcat_next/components/encoders.py
        state = load_weights_by_prefix(model_path, prefix="model.embed_tokens.")
        weight = state["weight"]
        layers = []
        start = int(offset)
        for size in codebook_sizes:
            stop = start + int(size)
            emb_weight = weight[start:stop].to(dtype=dtype)
            layers.append(nn.Embedding.from_pretrained(emb_weight, freeze=True))
            start = stop
        self.layers = nn.ModuleList(layers).to(device=device, dtype=dtype)
```

这意味着：

```text
image_encoder 读取一次 embed_tokens
 audio_encoder 读取一次 embed_tokens
 text_ar TP ranks 读取 AR 权重
```

如果这些都从 CephFS 读，会显著拉长启动时间。后续可以考虑将 codebook embedding 切片预处理成独立小文件，或只读取需要的 row range，减少重复扫描完整 embedding 权重。

#### C. JIT cache 可能落在共享文件系统

`15/15` 后的等待也可能来自 Triton / CUDA / TorchInductor JIT。如果 JIT cache 默认落在 home 或 CephFS，会受到共享文件系统锁和网络 I/O 影响。

建议启动前设置本地 cache：

```bash
export TRITON_CACHE_DIR=/tmp/$USER/triton_cache
export CUDA_CACHE_PATH=/tmp/$USER/cuda_cache
export TORCHINDUCTOR_CACHE_DIR=/tmp/$USER/torchinductor_cache
mkdir -p "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$TORCHINDUCTOR_CACHE_DIR"
```

#### D. post-load memory profile / KV pool 初始化缺少日志

SGLang `ModelRunner` 在 shard 读取完成后，还会做 memory profiling、KV pool 分配、tree cache 创建等工作。当前 LongCat 通过 `create_sglang_infrastructure_defer_cuda_graph()` 构建 AR 基础设施，即使 `disable_cuda_graph=True`，也仍然需要完成这些初始化。

建议在以下位置补启动耗时日志：

- `create_preprocessing_executor` 前后；
- `create_image_encoder_executor` 前后；
- `create_audio_encoder_executor` 前后；
- `LongcatNextImageEncoder.__init__` 的 remote class、load_module、codebook embedding 三段；
- `LongcatNextAudioEncoder.__init__` 的 remote class、load_module、codebook embedding 三段；
- `create_longcat_next_text_executor` 中 build server args、create infrastructure、memory pool ready、scheduler ready。

这类日志不会改变功能，但能把“无日志黑洞”拆成可定位的耗时区间。

### 7.3 单请求 0.6 tokens/s 的判断边界

需要先区分三类吞吐：

| 指标 | 计算方式 | 含义 |
|---|---|---|
| E2E throughput | `completion_tokens / request_total_time` | 包含 preprocessing、encoder、relay、prefill、decode、首轮 JIT |
| decode throughput | `completion_tokens / decode_time_only` | 只看 AR decode 阶段 |
| steady-state throughput | warmup 后第 2/3 条请求的 throughput | 排除首条 JIT/warmup 污染 |

如果 `0.6 tokens/s` 是首条图片/音频请求的 E2E throughput，它可能混入了：

```text
preprocessing
+ image/audio encoder
+ stage relay
+ long multimodal prefill
+ first-request JIT
+ eager decode
```

这种测法不能直接说明 decode 本身只有 0.6 tokens/s。必须至少区分：

- 首条 vs 第二条请求；
- 纯文本 vs 图片 vs 音频；
- E2E vs decode-only；
- GPU 是否在持续计算，还是 CPU/I/O/JIT 阻塞。

### 7.4 当前低吞吐的主要怀疑点

#### A. 当前显式关闭了 CUDA Graph 和 overlap schedule

LongCat `text_ar` 当前为了稳定性关闭了两个关键优化：

```139:149:sglang_omni/models/longcat_next/stages.py
    overrides = build_generation_batch_overrides(
        max_running_requests=max_running_requests,
        server_args_overrides=server_args_overrides,
        disable_cuda_graph=True,
        disable_overlap_schedule=True,
        enable_torch_compile=enable_torch_compile,
        mem_fraction_static=mem_fraction_static,
        max_prefill_tokens=16384,
        chunked_prefill_size=16384,
        sampling_backend="pytorch",
        dtype=dtype,
    )
```

这意味着 decode 路径目前是 conservative baseline：

```text
每 token 一轮 Python scheduler
每 token eager forward
每 token sampling
每 token 至少一次 next_token_ids materialize / D2H
无 CUDA Graph replay
无 overlap schedule
无 async decode
```

对 LongCat-Next 68B MoE，单请求 eager decode 的 CPU launch / D2H 同步会被放大，但单独这一点通常不应低到 0.6 tokens/s；如果 warmup 后纯文本仍然这么低，应继续排查 backend 是否退化。

#### B. 首条请求包含 JIT warmup

首次 prefill/decode 可能触发：

- LongCat MLA attention kernel；
- `flashinfer_cutlass` MoE kernel；
- ngram embedding kernel；
- sampling kernel；
- Triton/CUDA JIT。

如果用首条请求总耗时计算 throughput，会严重低估 steady-state 性能。建议固定做 warmup，并丢弃首条统计。

#### C. MoE backend 需要确认是否真走 `flashinfer_cutlass`

LongCat stage 中手动设置：

```158:158:sglang_omni/models/longcat_next/stages.py
    server_args.moe_runner_backend = "flashinfer_cutlass"
```

但需要在 H200 日志中确认最终生效。重点 grep：

```bash
grep -n "Configured SGLang backend policy\|moe_runner_backend\|flashinfer\|cutlass" server.log
```

期望看到：

```text
arch=LongcatNextTextForCausalLM ... moe_runner_backend=flashinfer_cutlass
```

如果实际 fallback 到慢 backend，或者出现 flashinfer/cutlass 初始化异常，`0.6 tokens/s` 就可能来自 MoE backend 退化。

#### D. NgramEmbedding 是 LongCat 独有额外开销

LongCat-Next 必须启用 `NgramEmbedding`，否则输出会明显错误。当前代码在 model config 和 scheduler runtime 中都显式启用了 ngram token table。若 pure text steady-state 仍极慢，可以做一次仅用于定位的 A/B：临时禁用 ngram，观察 tokens/s 是否暴涨。若暴涨，说明瓶颈集中在 ngram token table 更新或 ngram embedding forward 路径。

注意：该 A/B 只用于性能定位，禁用 ngram 后输出不可信，不能作为最终方案。

#### E. AR 当前只使用 4 张 H200

Phase2 默认资源布局是：

```text
GPU0: image_encoder
GPU1: audio_encoder
GPU2-5: text_ar TP=4
```

LongCat-Next AR 只用 4 张 H200。若官方或参考实现使用 8 卡 TP/EP，当前 4 卡 TP 的单请求 decode 会更慢。但 0.6 tokens/s 仍偏异常，更像是首轮 JIT、backend fallback、CPU sync 或测量口径导致。

### 7.5 建议的排查矩阵

#### Step 1：分离 warmup 与 steady-state

先跑一条短请求 warmup，丢弃统计：

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/mnt/cephfs/chenzhenyang/models/LongCat-Next","messages":[{"role":"user","content":"Hello"}],"max_tokens":8}'
```

再跑正式测量：

```bash
time curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/mnt/cephfs/chenzhenyang/models/LongCat-Next","messages":[{"role":"user","content":"Write a short story about a cat."}],"max_tokens":128}'
```

#### Step 2：拆分纯文本 / 图片 / 音频

| case | 目的 |
|---|---|
| 纯文本短 prompt | 测纯 decode |
| 纯文本长 prompt | 测 prefill + decode |
| 图片 + 短 prompt | 测 image encoder + visual prefill + decode |
| 音频 + 短 prompt | 测 audio encoder + prefill + decode |

如果纯文本第二条以后仍然只有 0.6 tokens/s，问题在 `text_ar`。如果只有图片/音频慢，优先看 encoder、relay、multimodal prefill。

#### Step 3：观察 GPU/CPU/I/O

请求期间运行：

```bash
nvidia-smi dmon -s pucm
```

判断：

| 现象 | 可能原因 |
|---|---|
| GPU util 高 | 真正在算，可能 MoE/TP/backend 慢 |
| GPU util 低、CPU 高 | Python scheduler、D2H、tokenizer、relay 同步 |
| GPU util 低、I/O 高 | CephFS 或 JIT cache 阻塞 |
| 某些 TP rank idle | TP/NCCL/进程绑定异常 |

#### Step 4：确认 backend 与关键日志

建议收集：

```bash
grep -n "Configured SGLang backend policy\|moe_runner_backend\|flashinfer\|cutlass\|cuda graph\|memory profile\|KV" server.log
```

重点看：

- `moe_runner_backend` 是否是 `flashinfer_cutlass`；
- 是否有 backend fallback；
- 是否有 JIT 编译长时间卡顿；
- 是否有 NCCL / illegal memory access / flashinfer 异常；
- `15/15` 后到 ready 之间到底是哪一步耗时。

### 7.6 建议的优化/修复顺序

1. **先补启动耗时日志**：只加打点，不改功能，定位“无日志”和 `15/15` 后的黑洞。
2. **把模型和 JIT cache 放本地盘**：优先排除 CephFS 和 shared cache 干扰。
3. **warmup 后重新测纯文本 steady-state**：确认 0.6 tokens/s 是否真实发生在 decode。
4. **确认 MoE backend 生效**：确保 `flashinfer_cutlass` 没有 fallback。
5. **按本文 P2 顺序逐步开启性能开关**：先 overlap-only，再 CUDA Graph-only，再 graph+overlap，最后 async decode。
6. **若仍然慢，再做 ngram A/B 定位**：确认 ngram token table / embedding 是否为主要瓶颈。

