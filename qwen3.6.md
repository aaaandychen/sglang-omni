# Qwen3.6 多模态 RL 训练支持：技术深潜（面试版）

> 仓库：krl（fork 自 THUDM/slime，Megatron 训练 + SGLang rollout + Ray）
> 分支：qwen3.6-tianmu-rl（HEAD = cd365742）
> 覆盖提交：2b8ffbf5 → dd67e5ee → b96d83e2 → cd365742（2026-08-02 ~ 2026-08-12）
> 文中所有技术细节均来自上述提交的 diff 与当前源码，标注了文件路径与函数名。

---

## 1. 背景与问题定义

### 1.1 Qwen3.6-35B-A3B 是什么架构

从 `scripts/models/qwen3.6-35B-A3B.sh`（raw/spec 路径）和 `scripts/models/qwen3.6-35B-A3B-bridge.sh`（bridge 路径）可以完整读出模型配置：

- **MoE**：40 层（`NLAYERS=40`），**全部层都是 MoE**（`FIRST_K_DENSE_REPLACE=0`，`MOE_LAYER_FREQ` 全 1），256 个专家、top-8 路由（`--num-experts 256 --moe-router-topk 8`），带 1 个共享专家（`--moe-shared-expert-intermediate-size 512`，且 `--moe-shared-expert-gate` 共享专家带门）。每个专家 FFN 维度 512（`--moe-ffn-hidden-size 512`），hidden size 2048。命名 "35B-A3B" 即总参数约 35B、每 token 激活约 3B 量级。
- **GQA + 门控注意力**：16 个 attention head、2 个 KV group（`--group-query-attention --num-query-groups 2`），head_dim = 256（`--kv-channels 256`），QK-LayerNorm（`--qk-layernorm`），注意力输出带门（`--use-gated-attention --attention-output-gate`）。
- **混合线性注意力**：`slime/backends/megatron_utils/actor.py` 中从 HF `text_config.layer_types` 检测 `"linear_attention"` 并设置 `gdn_layer_types` / `linear_num_value_heads` 等 GDN（gated delta net）参数；`slime_plugins/models/qwen3_5.py` 的 spec 里用 `fla.ops.gated_delta_rule.chunk_gated_delta_rule` 实现线性注意力层。即 Qwen3.5/3.6 是线性注意力与全注意力混合的 hybrid 架构。
- **MTP**：1 层 Multi-Token Prediction（`--mtp-num-layers 1`）。
- **RoPE**：`--rotary-percent 0.25`（部分旋转），`--rotary-base 10000000`。
- **VL**：HF checkpoint 内含 `vision_config` + `model.visual.*` 视觉塔权重（Qwen-VL 系列结构：patch_embed → decoder blocks → merger/projector），词表 248320。

### 1.2 为什么开源 slime 不能直接跑

开源 slime 的 Megatron 后端只覆盖了语言模型部分。对 Qwen3.6-VL 这个具体模型，断点是一整条链：

1. **权重进不来**：HF → torch_dist 转换时，视觉塔在 mcore 模型里没有对应模块，权重被静默丢弃（`tools/convert_hf_to_torch_dist.py` 原注释明确写了这一点）。
2. **权重回不去**：torch_dist → HF 回转时无法重建视觉塔，SGLang 加载时按 VLM 架构建视觉塔却拿到随机初始化权重，**decode CUDA graph capture 阶段直接 NaN**（`visual_passthrough.py` docstring 原文）。
3. **MTP/专家布局不匹配**：Qwen3.6 HF 侧 MTP 层专家是 fused 布局（`mtp.layers.*.mlp.experts.gate_up_proj`），而旧的 mbridge 映射假定 per-expert；EP>1 时还有 local/global expert id 索引 bug。
4. **训练侧没有视觉塔 forward**：即使权重带上了，Megatron 侧的模型也只有语言模型，无法做真正的 VLM 训练。
5. **数据管线不通**：rollout 侧（SGLang）吃图片走 processor，训练侧（Megatron）只认 token id 序列，pixel_values / image_grid_thw 没有通道送进训练 batch；且 VLM 场景下"训练用 prompt ids"（processor 展开图像占位符后）与"rollout 用 prompt ids"（SGLang 内部自己展开）长度不一致，CP 切分、logprob 对齐全都会错。
6. **生态版本错位**：能解决 4 的官方方案 megatron-bridge 0.4.0 是针对更新的 megatron-core 写的，内部 megatron 0.16 缺符号、MTP 模块命名不同，直接 import 就炸。

这次开发就是把这条链逐环接通，最终落地为 4 个提交。

---

## 2. 开发链路时间线与依赖关系

### 提交 1：2b8ffbf5 — Adapt Qwen3.6 offline weight conversion（08-02，12 文件 +710）

**解决的问题**：先让权重"进得来、回得去"。

- 新增 `scripts/models/qwen3.6-35B-A3B.sh`：raw/spec 路径的完整模型配置（复用 qwen3_5 spec）。
- 新增 `slime/backends/megatron_utils/vision_model.py`（325 行）：手写一套**只做 checkpoint 载体、不做 forward** 的视觉塔模块（`QwenVisionCheckpointModel`），结构镜像 Qwen-VL 视觉塔（`VisionPatchEmbed` / `VisionDecoder`×depth / `VisionMerger`），所有参数 `requires_grad=False`，并正确实现 TP 切分（`TensorParallelLinear` 按 partition_dim 切 weight/bias）与 `sharded_state_dict`（支持 megatron dist checkpoint 存取）。提供双向名字映射 `mcore_to_hf_vision_name()`（`vision_model.decoder.layers.N.self_attention.linear_qkv.weight` ↔ `model.visual.blocks.N.attn.qkv.weight` 等）和 QKV 交织格式互转 `vision_qkv_to_mcore()` / `vision_qkv_to_hf()`（HF 是 `[3, heads, head_dim]` 按 qkv 优先，mcore 是 head 优先，互相 permute）。
- `model_provider.py`：`--vision-weights megatron` 时在 `pre_process` 阶段 `attach_vision_model()` 把视觉塔挂到 GPTModel 上（只挂首个 pipeline stage）。
- `tools/convert_hf_to_torch_dist.py`：`has_vision_weights()` 自动检测 HF checkpoint 是否含 `model.visual.*`，有则自动打开 `vision_weights="megatron"`——视觉权重从此**以 Megatron 组织的冻结参数形式随 torch_dist checkpoint 一起走**。
- `tools/convert_torch_dist_to_hf.py` / `_parallel.py`：回转时 `convert_to_hf()` 先走 `mcore_to_hf_vision_name()` 分支直接吐 HF 视觉权重名；新增 `--expert-format auto/fused/per-expert`（qwen3_5 系默认 fused，经 `normalize_fused_expert_name()` 输出 `gate_up_proj` 形态）；新增 `--assert-hf-parity`（`hf_parity.assert_hf_metadata_parity()`：与参考 HF checkpoint 逐 tensor 比对名字/shape/dtype，不一致直接 fail——把"转换对不对"从人工抽查变成硬断言）。
- `slime_plugins/mbridge/qwen3_5.py`（mbridge 路径）：MTP 专家映射拆成 `_MTP_MLP_MAPPING_PER_EXPERT` / `_MTP_MLP_MAPPING_FUSED` 两套，`_mtp_mlp_mapping` property 用 `hf_mtp_experts_are_fused()` 扫 HF index 的 key **自动探测布局**（懒加载 + 缓存，依赖 `load_weights()` 初始化 `safetensor_io`）；修复 EP>1 时 fused 专家取数的 bug——原来用 local expert id 直接索引完整 fused 张量，改为 `global_expert_id()`（`ep_rank * experts_per_rank + local_id`，带整除与越界检查）。

### 提交 2：dd67e5ee — Fix Qwen3.6 vision weight sync（08-04，5 文件 +55）

**解决的问题**：提交 1 让视觉权重以参数形式进入了 Megatron 模型，于是它们会进入**在线训推权重同步**（update_weight）的参数枚举，这条路径立刻暴露两个 bug：

- **GLU 重排误伤视觉塔**：`update_weight/common.py` 的 `all_gather_param()` / `all_gather_params_async()` 里，凡名字含 `linear_fc1.weight` 的参数都会做 GLU 切半重排（`chunk(2, dim=0)` 后 gate/up 重新拼接——因为 Megatron 把 gated MLP 的 fc1 按 [gate;up] 在 TP 维度交错切分，HF 是分开的两个矩阵）。但视觉塔的 `linear_fc1` 是**非门控**普通 MLP，重排会把权重直接改错。修复：`_needs_glu_reorder()` 显式排除 `vision_model.` 前缀。
- **参数枚举的静默错位**：`hf_weight_iterator_direct.py` 原来到处是 `zip(..., strict=False)`，视觉参数引入后 PP/EP broadcast 的参数 info 一旦错位会被静默吞掉。修复：全部改 `strict=True`，并增加跨 rank 参数数量/name/shape/dtype/**attrs** 一致性断言（TP 属性不一致会在 all-gather 时产生无声错误）。
- `vision_model.py` 重构：抽出 `VisionCheckpointModule` 基类统一处理"自身参数 + 子模块"的 `sharded_state_dict`（原先只有部分模块正确处理 TP 分片 checkpoint 元数据）。
- `actor.py`：把 `hf_config.vision_config` 挂到 `args.vision_config`（回转时视觉塔层数 `vision_config.depth` 需要它，见提交 1 的 `get_layer_param`）。

### 提交 3：b96d83e2 — Support Qwen3.6 vision training（08-07，23 文件 +1635，核心提交）

**解决的问题**：让视觉塔真正参与 Megatron 训练 forward。这里做了关键架构决策——**引入 megatron-bridge 0.4.0 作为第二条建模/转换路径**（`--megatron-to-hf-mode bridge`），而不是继续在手写 spec 路线上堆代码：

- `requirements.txt`：pin `megatron-bridge==0.4.0`、`transformers>=5.2.0,<=5.3.0`。
- 新增 `slime_plugins/megatron_shims/__init__.py`：让 bridge 0.4.0 能在内部 megatron 0.16 上跑的兼容层（详见 3.1）。
- 新增 `slime_plugins/megatron_bridge/qwen35_vl.py`：bridge 入口工厂 `create_auto_bridge()` / `create_qwen35vl_provider()`，带两个 monkey patch（MTP 映射前缀、CP 视觉 embedding all-gather，详见 3.4/3.5）和严格版本校验 `validate_bridge_environment()`。
- `model_provider.py::_get_bridge_provider()`：从 HF checkpoint 直接派生 `Qwen35VLMoEModelProvider`，把运行时并行度字段（TP/PP/CP/EP/ETP/SP/recompute/dtype 等）从命令行 args 覆盖进 provider，强制 `calculate_per_token_loss=True`、关 router aux loss、`vision_dp_when_cp=False`、`use_hf_vision_model=False`、`freeze_vision_model/freeze_vision_projection` 跟随 `--freeze-vision-encoder/--freeze-vision-projection`（默认都 True）。有一个很细的坑：bridge 的 `finalize()` 会校验/改写 CP，所以先记下请求的 CP → 置 1 → `finalize()` → 再写回，并用两个 RuntimeError 兜底校验。
- 数据/损失链路 bshd 化：bridge 的 VLM 模型吃 padded `[batch, seq]` + bool attention mask，不吃 slime 原来的 packed thd 序列。`data.py::get_batch()` 在 bridge 模式下构造 `input_ids_bshd` / `attention_mask_bshd`（pad 到 `tp*2cp` 对齐的 `s_pad`），拼接 `pixel_values`/`image_grid_thw` 并校验二者成对出现；`cp_utils.py` 全套函数（`get_logits_and_tokens_offset_with_cp` / `slice_log_prob_with_cp` / `all_gather_with_cp` / `get_sum_of_sample_mean`）新增 `padded_length` 参数，让 CP 偏移计算基于**统一 padded 长度**而不是逐样本实际长度；`loss.py::get_responses()` 新增 bshd 分支，从 `[B, S, V]` logits 里按 CP chunk 偏移抠出每个样本 response 段的 logits/tokens。
- rollout 侧：`sglang_rollout.py` 新增 `_ensure_multimodal_train_inputs()`，把 processor 调用从 `generate()` 里解耦——自定义 generate 函数（多智能体）也能拿到训练用多模态输入；`processing_utils.py::prepare_model_inputs()` 返回 `multimodal_train_inputs`（pixel_values、image_grid_thw、bool attention_mask）；`utils/data.py::_build_messages()` 增加**占位符/媒体数量硬校验**（prompt 里 `<image>` 个数 ≠ images 字段个数直接 raise）。
- `model.py::_audit_bridge_freeze()`：建模后硬校验冻结是否生效（vision encoder/projection 参数 `requires_grad` 必须为 False，找不到视觉参数直接 RuntimeError）；bridge 模式显式禁止 combined-1f1b schedule plan。
- `arguments.py::slime_validate_args()`：bridge 模式的边界条件——与 `--spec`、`--vision-weights megatron`、`--enable-tree`、`--enable-mtp-training` 互斥，必须 `--colocate`，必须有 `--hf-checkpoint`，暂不支持 critic。
- 新增两个训练脚本：`run-qwen3.6-35b-kwaienv-use-rollout-logprobs-x40.sh`（文本 agent RL）和 `run-qwen3.6-35b-mathvision-vlm-rl-x40.sh`（MathVision 图文 RL，含数据集解压/转 parquet/校验的完整前置步骤）。

### 提交 4：cd365742 — Support multimodal RL training（08-12，16 文件 +1173）

**解决的问题**：从"能训图片"补全到"完整多模态 RL"，并补上可观测性：

- **视频支持**：`prepare_model_inputs()` 改用 `qwen_vl_utils.extract_vision_info` + `process_vision_info(return_video_kwargs=True, return_video_metadata=True)`；`Sample` 拆出 `rollout_prompt_ids` / `train_prompt_ids` / `rollout_response_ids` 三段 token；rollout payload 支持 `video_data`（字符串路径）；`data.py::get_batch()` 的多模态拼接从只认 `(pixel_values, image_grid_thw)` 推广到图像+视频两组 modality。
- **训推 token 一致性**：`generate()` 里训练侧用 processor 展开后的 `train_prompt_ids`，SGLang 侧用 tokenizer 纯文本的 `rollout_prompt_ids`（SGLang 内部自己展开图像 token），并用 SGLang 返回的 `meta_info.prompt_tokens` 与训练侧上下文长度**对账**，不等直接 raise；resume 场景下 prompt ids 变化也直接 raise。`ray/rollout.py` 增加 `tokens == train_prompt_ids + rollout_response_ids`、rollout logprobs 全有或全无、长度等于 response_length 的断言。
- **VLM forward audit 工具**：新增 `vlm_forward_audit.py`（274 行，详见 3.7），训练脚本里留了 `--vlm-forward-audit-*` 开关（注释态，排查时打开）。
- 新增 MMMU-Pro、VideoMMMU 两个训练脚本；`rm_hub/gpqa.py` 兼容 numpy 类型的 choices/valid_letters。

### 为什么必须是这个顺序

1. **先 offline 转换**：一切训练的前提是有一个 bit 级正确的 Megatron checkpoint，以及能转回 HF 给 SGLang/评测用。这步不成立后面无从谈起，所以这步还顺手建了 `--assert-hf-parity` 这种硬校验工具。
2. **再 weight sync**：提交 1 让视觉参数进了模型，在线同步路径立刻被污染（GLU 误重排会**改错**视觉权重再推给 SGLang）。不修这个，第一轮训练结束同步权重时视觉塔就坏了——这是"引入新参数后必须审一遍所有遍历参数的代码路径"的典型教训。
3. **再 vision 训练**：权重静态正确之后，才能谈 forward 正确。bridge 路径一次性解决建模、CP、冻结、数据布局四个问题。
4. **最后完整多模态 RL**：视频、训推 token 对账、audit 工具都是在前三步能跑通的基础上做正确性加固和覆盖面扩展。依赖关系是严格的：没有 1 没有 checkpoint，没有 2 同步即损坏，没有 3 视觉不进 forward，没有 4 训推对不齐。

---

## 3. 核心机制逐模块讲透

### 3.1 megatron-bridge 双路径与 megatron 0.16 shim

**问题**。slime 原有建模路线是"手写 spec + mbridge/自写映射转权重"（下称 raw 路径，`--megatron-to-hf-mode raw`）。对 Qwen3.6-VL 这条路线意味着要自己手写整个 VLM 的 Megatron 模型（含视觉塔 forward、CP 下视觉 embedding 处理、MTP）——工作量和风险都不可控。NVIDIA 官方 megatron-bridge 0.4.0 自带 `Qwen35VLMoEBridge` / `Qwen35VLMoEModelProvider`，但它是针对新版 megatron-core 开发的，内部 megatron 0.16 直接 import 就缺符号。

**设计**。保留 raw 路径（offline 转换工具仍在用），新增 bridge 路径用于真实 VLM 训练，二者通过 `--megatron-to-hf-mode {raw,bridge}` 切换（`slime/utils/arguments.py:139`）。两条路径的边界在 `slime_validate_args()` 里写死：bridge 与 `--spec` / `--vision-weights megatron` 互斥，必须 `--colocate` + `--hf-checkpoint`（因为 provider 直接从 HF config 派生，权重同步走 `export_hf_weights`）。

**shim 怎么工作**（`slime_plugins/megatron_shims/__init__.py`）：

- bridge 0.4.0 缺三个符号：`megatron.core.ssm.mamba_hybrid_layer_allocation.parse_hybrid_pattern`、`Symbols.MTP_SEPARATOR`、`experimental_attention_variant_module_specs.get_transformer_block_with_experimental_attention_variant_spec`。
- 前两个只是因为 `megatron/bridge/models/__init__.py` 会 eager import Mamba provider，**Qwen 路径永远不会调用**，所以 shim 只需让它们"存在"（`parse_hybrid_pattern` 从 Megatron-LM 0.20 移植了一个 faithful 实现，不是空壳）。
- 第三个是真实被调用的：内部 0.16 的 `get_gpt_decoder_block_spec` 已经实现了同等语义（解析 `moe_layer_freq` / `linear_attention_freq` 并按 pipeline stage 切片），所以 shim 只做薄委托。
- `apply_megatron_bridge_shims()` 幂等、必须在首次 `import megatron.bridge` 之前调用（挂在 `slime/backends/megatron_utils/__init__.py` 顶部），返回实际注入的符号列表——注释里明确说"assert 返回值，别信 docstring"，这是防未来 megatron 升级后 shim 静默失效的设计。

**版本钉死**：`validate_bridge_environment()`（`slime_plugins/megatron_bridge/qwen35_vl.py:62`）强制 megatron-bridge 恰好 0.4.0、transformers ∈ [5.2.0, 5.3.0]。monkey patch 只对单一版本负责，版本漂移直接 fail-fast 而不是行为漂移。

### 3.2 视觉塔 attach 与冻结策略

**raw 路径的 attach**（提交 1，`vision_model.py`）：`QwenVisionCheckpointModel` 是一个**纯权重载体**——模块树镜像 Qwen-VL 视觉塔（patch_embed 3D conv 投影、pos_embed、depth 层 `VisionLayer`（attn: qkv+proj；mlp: fc1+fc2，注意 qkv 和 fc1 各带一对 `layer_norm_weight/bias`，对应 HF 的 norm1/norm2）、merger（patch_norm + 两层 linear）），但没有任何 forward。它的全部意义是：

1. 让视觉权重以 Megatron 组织方式（TP 切分、`sharded_state_dict` 元数据）随 torch_dist checkpoint 存取；
2. 回转 HF 时经 `mcore_to_hf_vision_name()` + `vision_qkv_to_hf()` 还原成 HF 命名与 QKV 布局；
3. `ReplicatedParameter`/`ReplicatedWeightBias`（不切分）与 `TensorParallelLinear`（按 partition_dim 切）区分哪些视觉参数走 TP——merger 的 fc1/fc2、attention、MLP 走 TP，patch_embed/pos_embed/norm 全复制。

**为什么 VLM RL 通常冻结视觉塔**：`--freeze-vision-encoder` / `--freeze-vision-projection` 默认均为 True（`arguments.py:145-153`，BooleanOptionalAction）。动机是 RLVR 场景下 reward 信号只来自文本答案对错，视觉塔的梯度信号极其稀疏且噪声大；冻结视觉塔可以 (a) 保住预训练视觉表征不被 RL 初期的噪声梯度破坏；(b) 省下视觉塔的优化器状态和梯度 all-reduce 显存/通信；(c) 让"训推视觉表征一致"这个不变量天然成立， rollout 侧视觉编码与训练侧不会因为视觉塔更新而产生额外 mismatch。**梯度路径**：冻结不等于断路——梯度仍然穿过视觉 embedding 回传到语言模型的 embedding 层（image token 位置的 embedding 由冻结视觉塔产生，对语言模型参数照常反传），只是视觉塔自身参数 `requires_grad=False` 不更新。

**bridge 路径的冻结**：`_get_bridge_provider()` 把 `args.freeze_vision_encoder/projection` 透传给 provider 的 `freeze_vision_model/freeze_vision_projection`；`model.py::_audit_bridge_freeze()` 在 `setup_model_and_optimizer()` 里**建模后硬校验**：遍历所有参数，名字含 `vision_model` 的按是否含 `merger`/`projection` 分两类，冻结开关开了但还有 `requires_grad=True` 的直接 RuntimeError；一个视觉参数都找不到也 RuntimeError（防 provider 静默建错模型）。这是"配置可能被上游默认值覆盖，那就在建模后用事实校验"的防御式写法。

### 3.3 视觉权重透传（visual_passthrough）与 NaN 因果链

**完整因果链**（`megatron_to_hf/visual_passthrough.py` 的 docstring 是权威描述）：

1. slime/Megatron 只拥有 VLM checkpoint 的语言模型部分；
2. HF → torch_dist 转换把视觉塔权重全部丢弃（mcore 侧无对应模块）；
3. torch_dist → HF 回转自然也无法重建这些权重；
4. 但 SGLang 加载模型时看 config 是 VLM 架构，**照样会建视觉塔**——缺的 key 留在随机初始化状态；
5. bf16 随机未初始化显存里可能含 NaN/Inf 位型，decode 阶段 CUDA graph capture 一跑 forward 就输出 NaN。

**解法**有两代：

- **透传方案**（本模块，更早存在、本次开发中保留为 `--visual` 后备路径）：`copy_visual_weights()` 在主转换完成后，从**原始 HF 目录**把名字匹配 `("visual", "vision")` 前缀的 tensor **原样复制**进输出目录，新写 `model-visual-NNNNN.safetensors` 分片并就地更新 `model.safetensors.index.json` 的 weight_map 和 `metadata.total_size`。细节设计：`matches_any_prefix()` 用点边界匹配（`visual.` 开头或含 `.visual.`），避免误伤 `vocab_size` 这种 key；幂等（已在输出 weight_map 里的 key 跳过）；按源分片分组、每文件只开一次；5GB 分片上限对齐 HF 默认 `max_shard_size`；单 tensor 超限也单独成片而不是 crash。
- **载体方案**（提交 1 主线）：视觉权重直接以冻结 Megatron 参数形式随 checkpoint 走（3.2 节），回转时从参数重建，不再需要"从原始目录抄一份"。提交 1 把 `convert_hf_to_torch_dist.py` 改成检测到视觉权重就自动开启，docstring 同步改写为"视觉权重转成 Megatron 组织的冻结参数模块，不参与 forward/优化/在线同步"。

面试时如果被问"为什么有两个方案"：透传方案要求原始 HF 目录永远可得且和输出配对管理，载体方案让 checkpoint 自包含、可独立分发和续训；后者是前者的替代，前者保留作兜底。

### 3.4 MTP 权重映射 patch 与 FusedGatedExpertMapping

**问题**。bridge 0.4.0 的 `Qwen35VLMoEBridge.mapping_registry` 里 MTP 层的 Megatron 参数名用 `.mtp_model_layer.` 前缀，但内部 megatron 0.16 的 MTP 模块把这一层命名为 `.transformer_layer.`——名字对不上，MTP 权重加载/导出全部落空。另外 bridge 默认的 MTP 专家映射假定 per-expert HF 布局，而 Qwen3.6 HF checkpoint 的 MTP 专家是 fused 的（`mtp.layers.*.mlp.experts.gate_up_proj` / `down_proj`）。

**解法**（`qwen35_vl.py::_patch_mtp_mapping_prefix()`）：monkey patch `mapping_registry` property——先调原实现拿到 registry，然后逐条 mapping 把 `.mtp_model_layer.` 替换为 `.transformer_layer.`；对 MTP 专家的两条映射整体替换 mapping 类型：

- `linear_fc1.weight*` → `FusedGatedExpertMapping(hf_param="mtp.layers.*.mlp.experts.gate_up_proj")`（fused + 门控：gate/up 合并存储，导入时要拆、导出时要并）；
- `linear_fc2.weight*` → `FusedExpertMapping(hf_param="mtp.layers.*.mlp.experts.down_proj", transpose_on_export=True)`（fused 非门控，导出时需转置）。

幂等标记 `_slime_mtp_mapping_patched` 防重复 patch。

**mbridge（raw）路径的对应处理**（`slime_plugins/mbridge/qwen3_5.py`，提交 1）：同一问题的另一条路线解法——`_MTP_MLP_MAPPING_PER_EXPERT` / `_MTP_MLP_MAPPING_FUSED` 双映射 + `hf_mtp_experts_are_fused()` 扫 HF safetensors index 的 key 自动判别（fused 与 per-expert 必须恰居其一，否则 ValueError），结果缓存在 `_mtp_fused_cache`。两个路线共用 `expert_layout.py` 的 `global_expert_id()`——修掉 EP>1 时用本地 expert id 索引完整 fused 张量的越界型错误（本地 id 必须加 `ep_rank * experts_per_rank` 偏移）。

### 3.5 CP（context parallel）下视觉 embedding 的 all-gather patch 与 bshd 数据改造

**问题**。CP 把序列按"首尾对称 chunk"切到 cp_size 个 rank 上。VLM 场景里图像 token 集中在 prompt 的少数位置，视觉塔算出的 image embedding 必须准确 scatter 到各 CP rank 的本地序列片段里；bridge 的实现用 `AllGatherVisionEmbeddings` 这个 autograd Function 在 CP group 上 all-gather 视觉 embedding（`provider.vision_dp_when_cp = False` 即选择"all-gather 而不是视觉 DP"策略）。但 bridge 0.4.0 调用 `AllGatherVisionEmbeddings.apply(..., cp_group=...)` 用关键字传参，而当前 torch/megatron 组合的 `autograd.Function.apply` 只接受位置参数，直接 TypeError。

**patch**（`qwen35_vl.py::_patch_vision_all_gather_apply()`）：包一层 `apply`，把 `cp_group` 从 kwargs 弹出追加到位置参数，其余 kwargs 出现即 TypeError（显式拒绝未知签名漂移），幂等标记防重复 patch。

**训练数据的 bshd 改造**（提交 3，`data.py` / `cp_utils.py` / `loss.py`）。bridge 的 VLM 模型不吃 slime 的 packed 序列，吃 `[B, S]` padded + bool mask：

- `get_batch()` bridge 分支：把变长 token 列表 pad 成 `input_ids_bshd [B, s_pad]` 与 `attention_mask_bshd [B, s_pad]`（bool），`s_pad` 对齐到 `tp_size * 2*cp_size`（CP 要求序列能被 2*cp 整除做首尾对称切分，TP/sequence-parallel 要求能被 tp 整除）；`bshd_padded_length` 随 batch 下传。
- **CP 偏移计算的口径统一**：原 `get_logits_and_tokens_offset_with_cp(total_length, response_length)` 内部按逐样本实际长度自己算 pad；bshd 模式下整 batch 用同一个 `s_pad`，所以全套函数加 `padded_length` 参数，断言 `padded_length >= total_length` 且 `% (2*cp_size) == 0`。同一偏移函数被三处复用：loss 侧从 logits 抠 response 段、rollout logprob 切片、loss_mask 求和（`get_sum_of_sample_mean`）——**三处必须用同一个 padded 口径，否则 CP 下 logprob 和 mask 会对错位置**，这是这次改造最核心的不变量。
- `loss.py::get_responses()` bshd 分支：logits 从 `[1, packed_len, V]` 变 `[B, s_pad, V]`，CP>1 时按 `chunk_size = s_pad // (2*cp)` 找到本 rank 的首尾两个 chunk，再按 logits_offset/tokens_offset 在 chunk 内抠出 response 对应区间，cat 起来 yield；并断言抠出的 logits 与 tokens 等长。
- bridge 模式下 rollout_log_probs 的处理顺序也变了：`actor.py` 里不再预先按样本 CP 切片（保留全量 fp32 张量），而是到 `get_batch()` 里用 `slice_log_prob_with_cp(value, total_length, response_length, s_pad)` 按统一 padded 长度切——因为切片口径必须和训练侧 logits 的 padded 布局一致。

### 3.6 多模态 rollout 输入准备与训练侧 data pipeline

**rollout 侧**：

- 数据集侧（`scripts/run-qwen3.6-35b-mathvision-vlm-rl-x40.sh` 的内嵌 python）：MathVision parquet → 逐行重组 prompt（剥离 `<imageN>`、追加选项文本、首部插 `<image>` 占位符、要求 `\boxed{}` 输出），产出 `{prompt, images, answer}` 三列，并**离线校验占位符数量与图片数量逐行一致**。
- 加载侧（`slime/utils/data.py::_build_messages()`）：`--multimodal-keys '{"image":"images"}'` 声明占位符与数据列的映射；把 `<image>` 占位符切分 prompt、组装成 Qwen-VL chat 格式的 content list；提交 3 加入占位符/媒体数量硬校验（不一致直接 ValueError，把数据错误挡在训练前），提交 4 兼容 numpy ndarray 列。
- 处理侧（`processing_utils.py::prepare_model_inputs()`）：`qwen_vl_utils.process_vision_info` 抽图片/视频 → processor 产出 `input_ids`（图像占位符已按 grid 展开成视觉 token）+ `multimodal_train_inputs`（pixel_values、image_grid_thw、bool attention_mask）。提交 4 的关键拆分：`train_prompt_ids`（processor 展开后，训练用）与 `rollout_prompt_ids`（tokenizer 纯文本编码，给 SGLang，由引擎内部展开视觉 token）分离，`Sample` 增加 `train_prompt_ids/rollout_prompt_ids/rollout_response_ids` 三个字段，`tokens = train_prompt_ids + rollout_response_ids` 成为唯一事实源，并在 `ray/rollout.py` 断言这个等式。`generate()` 里用 SGLang 返回的 `meta_info.prompt_tokens` 与训练上下文长度对账（含图像/视频时启用），不一致直接 raise——**训推 token 错位从"静默 reward 异常"变成"当场 crash"**。
- 解耦（`_ensure_multimodal_train_inputs()`，提交 3）：processor 调用从 `generate()` 挪到生成之后统一补齐。原因：自定义 generate 函数（多智能体 agent rollout）不走 `generate()` 内部逻辑，但训练侧同样需要 pixel_values；统一在 `generate_and_rm()` 里对任意来源的 sample 补 `multimodal_train_inputs`。

**训练侧**（`data.py::get_batch()` bridge 分支 + `actor.py`）：

- `actor.py::_get_rollout_data()` 把每条样本的 `multimodal_train_inputs` 里的张量挪上 GPU（non_blocking），并注入逐样本 `bridge_mode` 标志随数据管线流动。
- `get_batch()` 把 micro-batch 内各样本的 `pixel_values` 沿 dim 0 `torch.cat`（视觉塔一次 forward 处理整个 batch 的所有图）、`image_grid_thw` 同样 cat，并校验二者成对出现；`num_images_per_sample` 记录归属。提交 4 推广为 `(pixel_values, image_grid_thw)` 与 `(pixel_values_videos, video_grid_thw)` 两组 modality 同一套逻辑。
- `model.py` 的 `forward_only` / `train_one_step` bridge 分支把 `pixel_values / image_grid_thw / pixel_values_videos / video_grid_thw` 连同 `input_ids_bshd / attention_mask_bshd` 一起传给模型；提交 4 起统一经过 `audited_vlm_forward()` 包装（见 3.7）。
- `ray/rollout.py`：`multimodal_train_inputs` 替代旧 `multimodal_inputs` 进入 train_data 并参与 partition；rollout logprobs 增加"全有或全无 + 长度等于 response_length"断言（GSPO 等 use-rollout-logprobs 路径的数据完整性）。

### 3.7 VLM forward audit：训推一致性对拍工具

`vlm_forward_audit.py`（提交 4 新增，274 行）解决的是"VLM 训练里图像到底进没进模型、进得对不对"这类问题很难从 loss 曲线看出来的痛点。核心类 `VLMForwardAuditor`，通过 `audited_vlm_forward(model, kwargs, args, context)` 包在每次 forward 外面（仅 bridge 模式 + 开了 `--vlm-forward-audit-log` 才生效），四层检查：

1. **输入不变量校验**（`_validate_inputs()`，每次审计 forward 都做）：attention_mask 必须 bool；`pixel_values` 与 `image_grid_thw` 必须成对；pixel 行数必须等于 `prod(t,h,w)` 求和；**图像 token 数必须等于 `prod(t,h,w) / spatial_merge_size²`**（视觉 token 与输入图像块的数值对账）；pixel_values 必须全部 finite；input_ids 里不能有"有图像 token 但没给 pixel"或反之。
2. **视觉塔确实被调且只调一次**：给 `vision_model` 注册 forward hook 计数，有媒体时要求恰好 1 次调用且能捕获到视觉输出，否则 RuntimeError（抓"图像 embedding 根本没进语言模型"或 CP/PP 下被重复调用）。
3. **因果探针**（`_causal_probe()`，设计最巧的部分）：把 `pixel_values` 沿 patch 维 `flip(0)` 反转后重跑一次 forward，比较两次 logits——如果图像真的影响了输出，打乱图像内容后 logits 必须变；`logits_max_abs <= vlm_forward_audit_min_logit_diff`（默认 1e-6）即判定"视觉输入被静默丢弃/错接"，直接 raise。对比在 TP+CP group 上 all-reduce（sum/count/max），恢复 RNG 状态和 train/eval 模式避免污染训练。
4. **结构化落盘**：每次审计写一行 JSONL（`_write()`，os.open+fsync 防缓冲丢失）：input_ids/mask shape、媒体 token 数、pixel/vision 输出/logits 的 shape/dtype/finite/mean/abs_max/l2_norm、causal 探针结果、phase（forward_only/train）、rollout_id/step_id。rank 0 写。审计频率由 `--vlm-forward-audit-first-n`（前 N 次必审）+ `--vlm-forward-audit-interval` 控制。

`actor.py` 里还在训练前用 `forward_only` 单独跑一轮 causal audit（前 `causal_first_n` 个 rollout）。**定位方法论**：VLM 训推不一致时，先用 audit 的 tensor stats 确认"训练侧图像确实进了模型且数值正常"，再对比 SGLang 侧 logprob——把"图像链路断了"和"语言模型数值差异"两类问题快速二分。

### 3.8 训推权重同步（update_weight）对视觉权重的处理

- **raw/direct 路径**（`hf_weight_iterator_direct.py` + `common.py`）：视觉参数作为普通参数进入枚举，TP 参数先 `all_gather_param` 拼全量，再 `convert_to_hf()` 走 `mcore_to_hf_vision_name()` 分支转 HF 命名（`linear_qkv` 还要 `vision_qkv_to_hf()` 做 QKV 反交织）。提交 2 的修复（`_needs_glu_reorder` 排除视觉塔、全链路 `strict=True` zip、跨 rank name/shape/dtype/attrs 四重断言）保证这条路径对视觉权重是"显式正确或显式报错"，不再静默出错。视觉塔冻结意味着值不变，但同步是无差别全量推，正确性仍必须有保证。
- **bridge 路径**（`hf_weight_iterator_bridge.py`）：`create_auto_bridge()` 建 bridge，`export_hf_weights()` 直接从 Megatron 模型导出 HF 命名权重（视觉塔因为是 provider 建的真模块，自然在导出范围内），再经 `postprocess_hf_param()` 后按 `update_weight_buffer_size` 分 chunk 推给 SGLang。bridge 要求 `--colocate`（`UpdateWeightFromTensor` 走显存直传而非分布式 RPC）——这是 `slime_validate_args()` 里 bridge 必须 colocate 的原因。提交 3 还适配了 `export_hf_weights` 返回 2 元组/3 元组两种形态。

---

## 4. 面试问答预演

**Q1：为什么用 megatron-bridge 而不是自己写 provider？**
A：手写意味着要重新实现整个 Qwen3.6-VL 的 Megatron 模型——视觉塔 forward、CP 下视觉 embedding 的 all-gather/scatter、MTP 层、混合线性注意力，每一项都是独立的大坑，且要和 HF 权重布局、SGLang 行为逐一对齐。bridge 0.4.0 自带官方维护的 `Qwen35VLMoEModelProvider` 和双向权重映射，我们实际需要补的只有三块：内部 megatron 0.16 的符号 shim（3 个符号）、MTP 命名/布局 patch、CP all-gather 的签名 patch。用 patch 面最小化换取整个模型实现的正确性，是明显的工程量/风险权衡。代价是版本钉死（bridge==0.4.0 + transformers [5.2,5.3]）和 colocate 限定，我们用 `validate_bridge_environment()` fail-fast 锁住。

**Q2：冻结视觉塔后梯度怎么传？视觉 embedding 的梯度去哪了？**
A：冻结只是把视觉塔参数 `requires_grad=False`，计算图没有断：图像经冻结视觉塔产生 image token 位置的 embedding，后续语言模型层的参数梯度照常反传，只是梯度流到视觉塔参数处不累积、不更新。optimizer 也不为这些参数建状态（省显存）。建模后还有 `_audit_bridge_freeze()` 硬校验冻结真的生效。

**Q3：vision token 在 CP 下怎么切分？**
A：序列按首尾对称 chunk 切到 CP rank（chunk_size = padded_len/(2*cp)），图像 token 落在哪个 rank 由位置决定。视觉塔输出的 embedding 通过 `AllGatherVisionEmbeddings` 在 CP group 上 all-gather（`vision_dp_when_cp=False` 选的这条策略），每个 rank 拿到全部视觉 embedding 后按本地序列片段里的图像 token 位置 scatter 进去。我们 patch 了它的 `apply` 解决 cp_group 关键字/位置参数签名不兼容。loss 侧三处 CP 偏移计算（logits 抠取、rollout logprob 切片、mask 求和）统一用 `bshd_padded_length` 口径，保证对齐。

**Q4：训推 logprob 不一致怎么排查？**
A：分层二分：(1) 先跑 VLM forward audit 的因果探针——反转 pixel patch 顺序看重算 logits 是否变化，确认训练侧图像真的进了模型；audit JSONL 里的 tensor stats（finite/mean/abs_max）确认视觉输出数值正常。(2) 确认 token 对齐：`generate()` 里有 SGLang `prompt_tokens` 与训练上下文长度的对账断言，训推 prompt 展开方式（processor vs 引擎内部）不一致会在第一步就 crash 而不是静默错。(3) 再看语言模型数值：两边 dtype、router fp32、attention softmax fp32 等配置是否一致，use-rollout-logprobs 时用 rollout 侧 logprob 做 GSPO 修正本身就容忍一定差异，关注量级而非零差。

**Q5：NaN 问题怎么定位到的？**
A：因果链是"Megatron 只训语言模型 → HF 回转丢视觉权重 → SGLang 按 VLM config 建视觉塔但 key 缺失 → 随机/未初始化显存（可能本身就是 NaN 位型）→ decode CUDA graph capture 跑 forward 出 NaN"。定位手段：现象是 SGLang 侧 NaN 而 Megatron 侧完全正常，说明问题在权重交接而非训练；对比回转后 HF 目录与原始目录的 tensor 清单即可发现视觉 key 缺失（后来这个对比固化成了 `--assert-hf-parity` 工具）。修复分两层：短期 `copy_visual_weights()` 从原始 HF 目录原样透传视觉权重；长期把视觉权重做成 Megatron 冻结参数模块随 checkpoint 走（`--vision-weights megatron`）。

**Q6：MTP 权重映射为什么要 patch？patch 了什么？**
A：两个不匹配：命名上 bridge 用 `.mtp_model_layer.`，内部 megatron 0.16 叫 `.transformer_layer.`；布局上 bridge 默认 MTP 专家是 per-expert，Qwen3.6 HF 是 fused（`gate_up_proj`/`down_proj`）。patch 在 `mapping_registry` 上逐条改前缀，并把两条专家映射换成 `FusedGatedExpertMapping`（fc1，门控 fused，导入拆 gate/up）和 `FusedExpertMapping`（fc2，`transpose_on_export=True`）。raw/mbridge 路线则是扫 HF index 自动判别布局（`hf_mtp_experts_are_fused`），两条路线解同一个问题。

**Q7：EP>1 时 fused 专家权重怎么取数？踩过什么坑？**
A：HF fused 张量第一维是全部 256 个专家，EP 下每个 rank 只持有 `256/ep_size` 个。原来的代码用本地 expert id 直接索引完整张量——ep_rank=0 碰巧正确，其他 rank 全取错。修复是 `global_expert_id()`：local_id + ep_rank × experts_per_rank，并带 num_experts 整除、local id 越界两个显式校验。这类"单卡/单 EP 测试碰巧正确"的 bug 只能靠多并行度配置暴露，所以我们把相关断言都做成了硬校验。

**Q8：bridge 模式为什么要 bshd？原来的 packed 序列怎么了？**
A：bridge 的 VLM provider 实现吃标准 `[B, S]` padded batch + bool attention mask（视觉 embedding scatter、CP all-gather 都按这个布局写），不吃 slime 原来的 packed thd + packed_seq_params。所以 `get_batch()` 在 bridge 模式构造 `input_ids_bshd/attention_mask_bshd`，pad 到 `tp*2cp` 对齐；`loss.py::get_responses()` 加 bshd 分支按 CP 偏移从 `[B, s_pad, V]` logits 里抠 response 段。代价是 padding 算力浪费和 combined-1f1b schedule plan 不支持（代码里显式 raise），换来的是不修改上游 bridge 模型实现。

**Q9：`_get_bridge_provider` 里为什么要把 CP 先置 1、finalize、再写回？**
A：bridge 的 `finalize()` 会基于配置做校验和派生计算，其中对 CP 的处理与我们想要的运行时 CP 语义冲突。绕过方式：先存请求的 CP → 置 1 让 finalize 按无 CP 走完 → 写回真实 CP → 再用两个 RuntimeError 校验 `calculate_per_token_loss` 和 `context_parallel_size` 确实是最终值。这是"第三方库的初始化校验与运行时需求冲突"的标准处理：顺着它的校验走，再覆盖，并用断言证明覆盖生效。

**Q10：rollout logprob 在 bridge 模式下为什么不在 actor 里预切？**
A：原来（raw 路径）`actor.py::_get_rollout_data()` 里按逐样本实际长度做 CP 切片；但 bridge 训练侧 logits 是按 batch 统一 `s_pad` 布局切的，两个口径不同会错位。所以 bridge 模式把 fp32 全量 logprob 保留到 `get_batch()`，用 `slice_log_prob_with_cp(..., s_pad)` 按训练侧同一 padded 口径切。原则：**凡是需要两处对齐的切片，必须用同一份长度元数据在同一点计算**。

**Q11：为什么 VLM RL 默认冻结视觉塔？什么情况下应该解冻？**
A：RLVR 的 reward 只从文本答案来，视觉塔梯度信号稀疏且方差大；冻结保住预训练视觉表征、省 optimizer state 与梯度通信、并天然维持训推视觉表征一致。解冻的合理场景：视觉 encoder 本身欠训（比如换了分辨率/patch 策略）、或 reward 明确依赖细粒度视觉理解且数据量足够。我们的实现里解冻只是把 `--freeze-vision-encoder/projection` 关掉，audit 校验会自动跟上。

**Q12：数据错误怎么防？**
A：三道闸：离线——数据准备脚本逐行校验 `<image>` 占位符与 images 数量一致；加载——`_build_messages()` 占位符/媒体数量不一致直接 ValueError；运行时——audit 校验 pixel 行数 == `prod(thw)`、图像 token 数 == `prod(thw)/merge²`、pixel 与 grid 成对出现。原则是多模态数据错误在训练里表现为静默的 reward 噪声，必须在进模型前全部变成显式异常。

**Q13：visual_passthrough 和 vision-weights megatron 两个方案什么关系？**
A：前者是存量方案：回转时从原始 HF 目录把视觉 tensor 原样抄进输出，要求原始目录永远可得且与输出配对。后者是这次的主线：视觉权重作为冻结 Megatron 参数随 torch_dist checkpoint 自包含流转，回转时从参数重建。前者保留为 `--visual` 兜底。面试可以展开：自包含 checkpoint 对断点续训、集群间分发、版本管理都更友好。

**Q14：bridge 导出权重给 SGLang 的流程？为什么必须 colocate？**
A：`HfWeightIteratorBridge` 用 `create_auto_bridge()` 建 bridge，`export_hf_weights(model)` 直接按 HF 命名导出（含视觉塔，因为 provider 建的就是完整 VLM），`postprocess_hf_param` 处理后按 buffer size 分 chunk 推给引擎。`slime_validate_args` 要求 bridge 必须 `--colocate`，即 Stage 1 只支持 `UpdateWeightFromTensor`（训推同卡显存直传），还没接 `UpdateWeightFromDistributed` 的 RPC 路径——这是明确的阶段性边界，代码里用 ValueError 写死。

**Q15：为什么版本要钉死 megatron-bridge=\=0.4.0？**
A：我们有三处 monkey patch（shim 符号、MTP mapping、AllGather apply），每处都依赖该版本的内部实现细节（mapping registry 的参数名、apply 的调用签名）。版本漂移后 patch 可能静默失效或打错位置。钉版本 + `validate_bridge_environment()` 启动即校验，把"依赖内部实现"的风险显性化。升级路径也很清楚：跑一遍 `--assert-hf-parity` + VLM audit + 小规模训推对拍，即可验证 patch 是否还需要。

**Q16：CP 下 loss_mask 求和为什么要传 padded_length？**
A：`get_sum_of_sample_mean()` 要按 CP chunk 偏移把全量 loss_mask 切成"本 rank 实际负责的部分"再求和，偏移来自 `get_logits_and_tokens_offset_with_cp()`。bshd 模式下这个偏移必须以 batch 统一的 `s_pad` 算，否则切出来的 mask 段与 logits 段错位，per-token loss 归一会错。这是 3.5 节"三处共用同一 padded 口径"不变量的一部分。

**Q17：训推两侧 prompt tokenization 为什么不一致？怎么处理的？**
A：训练侧要显式知道每个图像占位符展开成多少个视觉 token（loss 对齐、CP 切分都要），所以用 processor 展开后的 `train_prompt_ids`；SGLang 内部有自己的多模态展开逻辑，给它文本 tokenizer 编码的 `rollout_prompt_ids` + 原始图片即可。两套 ids 长度不同但各自闭合，`Sample.tokens = train_prompt_ids + rollout_response_ids` 作为训练唯一事实源，response ids 来自 SGLang 返回的 token 流；再用 `meta_info.prompt_tokens` 对账确认引擎实际处理的 prompt 长度与训练侧一致。

**Q18：如果让你支持一个新的 VLM，这套东西的迁移路径是什么？**
A：(1) 看 megatron-bridge 是否已有对应 provider——有则复用 bridge 路线，重点检查 shim 是否需要扩、mapping 是否要 patch；(2) 没有则走 raw 路线：写 `vision_model.py` 式载体模块 + `mcore_to_hf_*` 名字映射 + QKV 布局互转，并审一遍 update_weight 的参数遍历路径（GLU 重排这类名字启发式是最大的坑）；(3) 数据侧改 `--multimodal-keys` 和占位符约定即可复用整条管线；(4) 验证三板斧：`--assert-hf-parity` 比转换、VLM audit（含因果探针）比 forward、小规模训推 logprob 对拍。

**Q19：这套开发里你觉得最能体现工程判断的决策是什么？**
A（参考）：三个。(a) 引入 bridge 时用 shim + patch 而不是 fork——修改面最小，升级路径清晰，代价（版本钉死）显性化；(b) 所有"可能静默错"的地方都做成硬断言——HF parity、freeze audit、token 对账、strict zip、占位符计数，多模态 RL 的调试成本主要在静默错误，把隐式假设全部显式化是这次最重要的工程质量来源；(c) 分四个提交按依赖关系推进，每一步都有独立的正确性校验工具，下一步建立在上一步"已被证明正确"的基础上。

**Q20：bridge 模式目前还有哪些限制？**
A（代码里写死的边界）：不支持 critic（`get_model_provider_func` 直接 ValueError）、不支持 combined-1f1b schedule plan、与 `--enable-tree` / `--enable-mtp-training` / `--spec` / `--vision-weights megatron` 互斥、必须 colocate、版本钉死。这些限制都在 `slime_validate_args()` 和 `model.py` 里是显式异常，不是隐式行为。

---

## 5. 贡献表述建议（结对开发场景）

背景事实：提交 1/2 作者为 root（实习生机时），提交 3/4 作者为 sunxianda（正职 mentor）。诚实且有力的表述方式是把"我做了什么"说成**可验证的具体技术点**，把"我们怎么协作"说成**正常的工程分工**：

**推荐表述框架（STAR 变体）：**

1. **问题定义归自己**："我负责 Qwen3.6 多模态 RL 在内部 slime 框架上的落地。这个模型的难点是 MoE + 混合线性注意力 + MTP + 视觉塔四个非标准特性叠加，开源框架一条链路都不通。"——这是事实，且体现你理解全局。

2. **关键实现点归自己**（挑你真能讲清细节的，本篇第 3 节任何一个模块都可以）：
   - "我实现了视觉权重的 Megatron 载体模块和双向转换，把 SGLang 视觉塔随机初始化导致 CUDA graph NaN 的问题从根上修掉"（提交 1，`vision_model.py` / `convert_to_hf` 视觉分支）；
   - "我修掉了在线权重同步里 GLU 重排误伤视觉塔的 bug，并把参数枚举全链路改成 strict + 四重一致性断言"（提交 2）；
   - "我写了 VLM forward audit 工具，用反转图像 patch 顺序的因果探针证明视觉输入真的影响了 logits"（提交 4，`vlm_forward_audit.py`）；
   - "我落地了多模态数据管线：占位符校验、processor 输入准备、训推双 prompt ids 分离与对账"（提交 3/4）。

3. **架构决策说协作、说理由**："整体路线（引入 megatron-bridge 而不是手写 provider）是我和 mentor 一起定的，我负责论证工程量对比并完成适配层——shim、MTP mapping patch、CP all-gather patch 都是我写的。"——结对开发中"参与决策 + 独立实现"是标准且受认可的贡献形态，面试官追问细节时你能答（本篇第 4 节）才是决定性证据。

4. **被追问"这是你想的还是 mentor 想的"时**：不要抢也不要推。标准答法："方案方向是我们讨论定的，我提的实现候选里被采纳的是 X；从设计文档到代码、调试、验证工具是我独立完成的，比如 NaN 的定位过程和 audit 工具的设计我可以完整讲一遍。"——然后真的完整讲一遍。**能讲清"为什么"的地方才是你的贡献，背下来"是什么"不算**。

5. **绝对不要说的**：不要说"整个 Qwen3.6 支持是我做的"（git author 一查即穿）；也不要过度自谦成"我只是打下手"（你能讲透 CP 偏移口径统一这种细节，就不是打下手）。最稳的姿态是：**模块级 ownership 清晰 + 全链路理解到位 + 对 mentor 主导的决策能复述其 trade-off**——这恰恰是高级工程师的画像。


# Qwen3.6 多模态代码逐模块深度分析（面试版·下篇）

> 仓库：krl（fork 自 THUDM/slime），分支 qwen3.6-tianmu-rl，HEAD = cd365742。
> 本文是 `qwen36_multimodal_deep_dive.md` 的姊妹篇：上篇讲全链路架构，本篇**只讲多模态相关代码**，逐文件、逐函数、带行号。
> 所有结论均来自 HEAD 处源码与 4 个提交（2b8ffbf5 / dd67e5ee / b96d83e2 / cd365742）的 diff；代码里找不到直接证据的推断，一律显式标注「合理推测」。

---

## 模块 1：slime/backends/megatron_utils/vision_model.py（355 行，全文精读）

### 职责

raw（spec/mbridge）路径下的**视觉塔权重载体**：一套镜像 Qwen-VL 视觉塔参数结构、但**没有任何 forward** 的 torch.nn.Module 树。它存在的唯一目的是让视觉权重以 Megatron 组织方式（TP 切分 + dist-checkpoint 元数据）随训练 checkpoint 流转，并能无损转回 HF 命名。由提交 2b8ffbf5 新建（325 行），dd67e5ee 重构 checkpoint 基类（+30 行）。

### 关键数据结构

- `VISION_MCORE_PREFIX = "vision_model"`、`HF_VISION_ROOT = "model.visual"`（L18-19）：两套命名空间的根。
- `VisionConfig`（L28-56）：frozen dataclass，10 个字段（depth / hidden_size / intermediate_size / num_heads / num_position_embeddings / out_hidden_size / patch_size / temporal_patch_size / in_channels / spatial_merge_size），`from_hf_checkpoint()` 直接读 HF 目录 `config.json` 的 `vision_config` 节。**注意它不从 Megatron args 取配置**——视觉塔结构的事实源永远是 HF checkpoint。

### 核心流程

**1. `has_vision_weights(hf_dir)`（L22-25）**
读 `model.safetensors.index.json` 的 `weight_map`，任意 key 以 `model.visual.` 开头即判定为 VLM。调用点在 `tools/convert_hf_to_torch_dist.py::get_args()`：检测到就自动置 `args.vision_weights = "megatron"`——离线转换时视觉权重自动转为 Megatron 冻结参数，不需要人工开关。

**2. 模块树与 TP 语义（L59-265）**

| 类 | 行号 | 切分方式 |
|---|---|---|
| `VisionCheckpointModule`（基类） | L59-86 | dd67e5ee 引入；`sharded_state_dict` = 自身参数（`make_sharded_tensors_for_checkpoint`）+ 逐子模块 `sharded_state_dict_default`，TP group 与 `dp_cp_group`（由 `ensure_metadata_has_dp_cp_group` 补齐）都进元数据 |
| `ReplicatedParameter` / `ReplicatedWeightBias` | L89-102 | 不切分，`requires_grad=False`，只设默认 TP 属性 |
| `TensorParallelLinear` | L105-154 | `partition_dim=0`（列切）：weight `[out/tp, in]`、bias `[out/tp]`（bias 也带 TP 属性 dim 0）；`partition_dim=1`（行切）：weight `[out, in/tp]`、bias **完整复制**；可选挂一对 replicated 的 `layer_norm_weight/bias`（LN 参数寄生在 Linear 模块名下，见下文映射） |
| `VisionAttention` | L157-172 | `linear_qkv`：hidden→3·hidden，dim 0，带 LN；`linear_proj`：dim 1 |
| `VisionMLP` | L175-190 | `linear_fc1`：hidden→intermediate，dim 0，带 LN；`linear_fc2`：dim 1。**非门控**（单个 fc1，不是 GLU/SwiGLU 的 gate+up 两份）——这是 dd67e5ee GLU bug 的根源 |
| `VisionDecoder` | L200-223 | `depth` 层 `VisionLayer`；自定义 `sharded_state_dict` 给每层加 `((0, layer_idx, depth), ...)` 偏移，支持 dist checkpoint 跨并行度 reshard |
| `VisionPatchEmbed` | L226-239 | `proj` 是 **5 维 conv 权重** `[hidden, in_channels, temporal_patch_size, patch_size, patch_size]`（3D patch conv），replicated |
| `VisionMerger` | L242-252 | `merged_size = hidden × spatial_merge_size²`；`patch_norm`（replicated）+ `linear_fc1`（merged→merged，dim 0）+ `linear_fc2`（merged→**out_hidden_size**，dim 1，即投影到语言模型 hidden） |
| `QwenVisionCheckpointModel` | L255-281 | 四个子模块 + `self.requires_grad_(False)`（L265，整树冻结的第二道保险） |

**3. `attach_vision_model(model, hf_dir)`（L284-287）**
从 HF config 建 `QwenVisionCheckpointModel`，`model.add_module("vision_model", ...)` 挂到 GPTModel 上。调用时机（`model_provider.py::get_model_provider_func` 内的 `model_provider`，2b8ffbf5 diff）：仅 `pre_process=True`（pipeline 第一段）且 `args.vision_weights == "megatron"` 时；且强制要求 `--hf-checkpoint`（否则 ValueError）。**冻结实现是双层的**：每个 Parameter 创建时 `requires_grad=False`（L92/99-100/122-123/130-136），顶层再 `requires_grad_(False)`（L265）。

**4. `mcore_to_hf_vision_name(mcore_key)`（L290-331）**
双向映射的出口方向（mcore → HF）。先 `strip_param_name_prefix()` 剥掉 DDP 的 `module.module.` 前缀，再分两类：

- 非层参数走 `direct` 字典（L296-306）。注意唯一的**改名**：`merger.patch_norm.*` → `merger.norm.*`（mcore 侧为了不和 Linear 的 LN 属性撞名，用了 patch_norm）。
- 层参数（L309-331）：`vision_model.decoder.layers.{N}.{suffix}` → `model.visual.blocks.{N}.{hf_suffix}`。`layer_mapping` 里最要紧的两条：寄生在 `linear_qkv` 上的 LN 参数 → HF 的 `norm1.*`（attention 前 LN）；寄生在 `linear_fc1` 上的 LN → `norm2.*`（MLP 前 LN）。匹配不上返回 `None`，调用方（`megatron_to_hf/__init__.py::convert_to_hf`）据此 fall through 到语言模型转换逻辑。

**5. QKV 重排（L334-355）**

```python
# vision_qkv_to_mcore (L338-343): HF -> mcore
tensor.view(3, num_heads, head_dim, *trailing).permute(1, 0, 2, ...)
# vision_qkv_to_hf (L350-354): mcore -> HF
tensor.view(num_heads, 3, head_dim, *trailing).permute(1, 0, 2, ...)
```

HF 的 fused `attn.qkv.weight` 布局是 **qkv 优先**（全部 q 头 → 全部 k 头 → 全部 v 头），`view(3, H, D)` 成立；mcore 侧为了 TP 沿 dim 0 切分后**每个分片内每个 head 的 q/k/v 仍然连续**，用 **head 优先**（每个 head 内 qkv 相邻），`view(H, 3, D)` 成立。两个方向都是 `permute(1,0,2)`，区别只在第一步 view 的解释。`*trailing` 的设计让同一函数同时适用于 weight `[3H·D, in]` 和 bias `[3H·D]`。`num_heads` 来源：导出侧 `convert_to_hf` 从 `args.vision_config.num_heads`（默认 16）取；导入侧 mbridge `Qwen3_5Bridge._weight_to_mcore_format` 从 `hf_config.vision_config.num_heads` 取。

### 边界条件与坑

- **GLU 重排误伤**（dd67e5ee 修复）：视觉 MLP 的 `linear_fc1` 名字与语言模型 gated MLP 的 fc1 同名，但**不是门控结构**。`update_weight/common.py` 原来对一切 `linear_fc1.weight` 做 `chunk(2, dim=0)` 的 gate/up 重排，会直接改错视觉权重。修复是 `_needs_glu_reorder()` 显式排除 `vision_model.`（common.py L15-16）。
- `VisionCheckpointModule` 基类是 dd67e5ee 才抽出来的：此前部分模块的 `sharded_state_dict` 没正确处理"自身参数 + TP 分片元数据"，checkpoint 存取会丢/错。
- `attach_vision_model` 只挂 `pre_process` 段：PP=1 时无所谓，PP>1 时只有第一段持有视觉参数（该模型实际配置 PP=1）。

---

## 模块 2：slime_plugins/megatron_bridge/qwen35_vl.py（86 行）

### 职责

megatron-bridge 路径的唯一入口工厂 + 两个 monkey patch + 版本闸。由 b96d83e2 新建。

### 核心流程

**1. `_patch_vision_all_gather_apply()`（L6-21）**

```python
def apply(*args, **kwargs):
    if "cp_group" in kwargs:
        args = (*args, kwargs.pop("cp_group"))
    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}")
    return original_apply(*args)
```

被 patch 对象：`megatron.bridge.models.qwen_vl.modelling_qwen3_vl.utils.AllGatherVisionEmbeddings`——CP 下视觉 embedding 的 all-gather autograd Function（视觉塔在各 rank 上算完 embedding 后，在 CP group 上汇聚，使每个 CP rank 都能把图像 embedding scatter 到自己持有的序列片段；配合 `_get_bridge_provider` 里 `provider.vision_dp_when_cp = False` 即"不走视觉 DP、走 CP all-gather"策略）。

**不兼容点是什么**：patch 的行为是把 `cp_group` 关键字参数转成位置参数再调原 `apply`。代码证据到此为止；本地环境未安装 megatron-bridge/megatron，无法读两侧源码确认签名归属。**合理推测**：`torch.autograd.Function.apply` 在该 torch 版本不接受关键字参数（或 bridge 0.4.0 的 `apply` 定义为纯位置参数），而调用方（bridge 的 Qwen3-VL modelling 代码）以 `cp_group=...` 关键字调用，直接 TypeError。patch 同时用「多余 kwargs 直接 TypeError」防未来签名漂移被静默吞掉，并用 `_slime_apply_patched` 标记幂等。

**2. `_patch_mtp_mapping_prefix()`（L24-59）**

patch `Qwen35VLMoEBridge.mapping_registry`：先调原实现，然后逐条 mapping 把 Megatron 参数名的 `.mtp_model_layer.` 替换为 `.transformer_layer.`（内部 megatron 0.16 的 MTP 模块命名与 bridge 0.4.0 假设不同），并对两条 MTP 专家映射整体换类型：

- `...transformer_layer.mlp.experts.linear_fc1.weight*` → `FusedGatedExpertMapping(hf_param="mtp.layers.*.mlp.experts.gate_up_proj")`：Qwen3.6 HF 的 MTP 专家是 **fused + 门控**布局，导入时拆 gate/up、导出时合并；
- `...linear_fc2.weight*` → `FusedExpertMapping(hf_param="mtp.layers.*.mlp.experts.down_proj", transpose_on_export=True)`。

`_slime_mtp_mapping_patched` 幂等标记。

**3. `validate_bridge_environment()`（L62-68）**
硬钉 `megatron-bridge == 0.4.0`、`transformers ∈ [5.2.0, 5.3.0]`，不符直接 RuntimeError。因为上面两个 patch 都依赖该版本内部实现细节，版本漂移必须 fail-fast。调用点：`create_auto_bridge()` 内（L74）和 `slime/utils/arguments.py::slime_validate_args` 的 bridge 分支（L1498-1500）——**启动期就炸，而不是训练三小时后权重错位**。

**4. 两个工厂（L71-86）**
`create_auto_bridge(hf_checkpoint)`：校验版本 → 打两个 patch → `AutoBridge.from_hf_pretrained(..., trust_remote_code=True)`。
`create_qwen35vl_provider(hf_checkpoint)`：`bridge.to_megatron_provider(load_weights=False)`，并用 `isinstance` 断言拿到的是 `Qwen35VLMoEModelProvider`（防 HF config 识别错模型族静默建错模型）。

### 边界条件与坑

- patch 顺序无关但**必须先于首次 `import megatron.bridge` 之后、首次使用之前**；shim（`slime_plugins/megatron_shims`）则更严格，必须在首次 import 前（挂在 `slime/backends/megatron_utils/__init__.py` 顶部，b96d83e2 diff）。
- `load_weights=False`：provider 只负责建模，权重加载走 `bridge.load_hf_weights()`（`checkpoint.py::_load_checkpoint_hf`），职责分离。

---

## 模块 3：slime/backends/megatron_utils/megatron_to_hf/visual_passthrough.py（229 行，全文精读）

### 职责

torch_dist → HF 回转时，把视觉塔权重从**原始 HF 目录**逐字节透传进输出目录。是 NaN 问题的第一代解法，当前保留为 `--visual` 兜底路径（主线已由模块 1 的载体方案替代）。

### NaN 因果链的代码证据

文件 docstring（L1-22，原文照录要点）：

> slime/Megatron only owns the language-model portion of VLM checkpoints. HF → torch_dist drops every key whose module does not exist on the mcore side (i.e. the vision tower) ... SGLang still builds the vision tower at load time when the config advertises a VLM architecture; **missing weights leave it randomly initialised and produce NaN during decode CUDA-graph capture.**

另一条证据在 2b8ffbf5 的 diff：`tools/convert_hf_to_torch_dist.py` 的旧注释写着 "Without that step SGLang will build the vision tower from uninitialised memory and emit NaN during decode"，该提交把它改写为载体方案的说明。完整链条：**Megatron 只持语言模型 → 回转丢视觉 key → SGLang 按 VLM config 建视觉塔但权重缺失 → 未初始化显存（可能含 NaN 位型）→ decode CUDA graph capture 跑 forward → NaN**。

### 核心流程（`copy_visual_weights`，L91-224）

1. **前置校验**：输出目录必须已有 `model.safetensors.index.json`（L128-133，否则 FileNotFoundError——必须排在主转换之后）；`prefixes` 非空。
2. **发现**：`build_weight_map(origin_hf_dir)`（L43-62）优先读 index.json，单文件模型则扫描所有 `.safetensors` 自建映射；`matches_any_prefix()`（L65-75）做**点边界匹配**（`visual.` 开头或含 `.visual.`），注释明确说为了防 `vocab_size` 这类误命中。
3. **幂等**：已在输出 `weight_map` 里的 key 跳过（L150-157）——重跑转换不会产生重复条目。
4. **拷贝**：按源分片分组（每文件只 `safe_open` 一次，L160-162、190-204）；缓冲区超 `max_shard_bytes`（默认 5GB，对齐 HF `max_shard_size`，L40）先 flush；**单 tensor 超限也单独成片而不是 crash**（L200 注释）。
5. **落盘**：新分片命名 `model-visual-{NNNNN}.safetensors`（`next_shard_name`，L78-88，避开已有名）；就地更新输出 index.json 的 `weight_map` 并把 `metadata.total_size` 加上新增字节数（L207-212，维持 HF 契约）。
6. 返回 `{copied, skipped, bytes_added, new_shards}` 统计。

### 边界条件与坑

- 透传要求**原始 HF 目录永远可得**且与输出配对管理——这正是载体方案（模块 1）要替代它的原因：checkpoint 自包含、可独立分发续训。
- 2b8ffbf5 把 `_build_weight_map` 等三个私有函数改为公开（保留别名兼容），因为 `hf_parity.py` 的 `read_tensor_metadata()` 要复用 `build_weight_map` 做逐 tensor parity 校验。

---

## 模块 4：slime/backends/megatron_utils/vlm_forward_audit.py（274 行，全文精读）

### 职责

bridge 模式下的 **VLM forward 运行时审计器**：回答"图像/视频到底有没有进模型、数值正不正常、训推不一致是不是视觉链路的锅"。cd365742 新建。

### 触发条件（参数设计）

`slime/utils/arguments.py`（cd365742 diff，`--vlm-forward-audit-*` 一组）：

| 参数 | 默认 | 作用 |
|---|---|---|
| `--vlm-forward-audit-log` | None | JSONL 输出路径；不设则整个机制关闭（`audited_vlm_forward` L268 直接透传 `model(**kwargs)`） |
| `--vlm-forward-audit-first-n` | 4 | 前 N 次 forward 必审（`_should_audit` L83-87：`index <= first_n`） |
| `--vlm-forward-audit-interval` | 0 | 之后每隔 interval 审一次；0 = 只审前 N 次 |
| `--vlm-forward-audit-causal` | False | 开因果探针 |
| `--vlm-forward-audit-causal-first-n` | 1 | 因果探针只在前 1 次做 |
| `--vlm-forward-audit-min-logit-diff` | 1e-6 | 因果判定阈值 |

另有 `slime_validate_args` 的约束：audit 只在 bridge 模式 + megatron 后端可用（L1483-1485）。训练脚本（mathvision）里这组参数是**注释态**——排查时打开。

### 核心流程

**入口**：`audited_vlm_forward(model, kwargs, args, context)`（L267-274）——没开 log 或非 bridge 模式直接透传；否则按 model 从 `WeakKeyDictionary` 取/建 `VLMForwardAuditor`（不持有强引用，模型销毁 auditor 随之回收）。model.py 的两处 forward（`forward_only` L286、`train_one_step` L526）都经过它。

**`_core_model()`（L27-41）**：unwrap 后沿 `.module` 链爬，找到同时有 `vision_model` 和 `language_model` 属性的核心模块；找不到直接 RuntimeError（L66）。

**第一层：输入不变量（`_validate_inputs` L92-145，每次审计都做）**
- `attention_mask` 必须 bool（L100-101）——bridge 模型按 bool mask 实现；
- `pixel_values` ↔ `image_grid_thw`、`pixel_values_videos` ↔ `video_grid_thw` 必须成对（L111-112）；
- **数值对账**：`expected_rows = grid.prod(-1).sum()`（L124），pixel 行数必须等于它（L125-128）；视觉 token 数必须等于 `expected_rows // spatial_merge_size²`（L129-132）——即 `image_grid_thw` 每行是 (t, h, w) 的 patch 网格，每张图占 `t*h*w` 个 patch 行、展开成 `t*h*w/merge²` 个视觉 token；
- 有 token 没 pixel（或反之）直接 raise（L120-121）；
- pixel 必须全 finite（L133-134）。

**第二层：视觉塔调用计数（`_vision_hook` L74-78 + `forward` L235-245）**
给 `vision_model` 注册 forward hook 计数；有媒体时要求一次 forward 里**恰好调用 1 次**视觉塔（L242-243），且能捕获输出（L244-245）。抓两类故障：图像 embedding 根本没进语言模型（0 次）、CP/PP 下被重复计算（>1 次）。

**第三层：因果探针（`_causal_probe` L155-198）——最巧妙的设计**
1. 取 `pixel_values`（或视频），至少 2 行才做；
2. 保存 CUDA RNG 状态、切 eval 模式，先跑一次真实 forward（`_run_probe`，`torch.no_grad`）；
3. 恢复 RNG，把 pixel 沿 patch 维 **`flip(0)`**（反转 patch 顺序）后再跑一次；
4. 比较两次 logits：在 TP+CP group 上 all-reduce sum/count/max（L179-181）；
5. `logits_max_abs <= min_logit_diff`（默认 1e-6）→ **图像内容变化没有引起输出任何变化 → 视觉输入被静默丢弃/错接** → RuntimeError（L195-196）；
6. 同时记录视觉塔输出本身的 diff（`vision_output_diff`）。

逻辑要点：它不对比"SGLang vs Megatron"（跨系统对拍），而是**单系统内的因果干预实验**——打乱图像必须改变输出，否则视觉链路必然是断的。`actor.py`（cd365742 diff）还在训练前用 `forward_only` 单独跑一轮 causal audit（前 `causal_first_n` 个 rollout，`timer("vlm_audit_causal")`）。

**第四层：结构化落盘（`_write` L200-223）**
rank 0 写 JSONL：`os.open(O_APPEND) + os.write + fsync`（防 Python 缓冲在崩溃时丢最后几行——审计日志的价值恰恰在崩溃现场）。记录内容（L246-259）：phase（forward_only/train）、rollout_id/step_id、forward_index、输入形状、媒体 token 数、pixel/vision 输出/logits 的 `_tensor_stats`（shape/dtype/finite/mean/abs_max/l2_norm/sum，L44-57）、因果探针结果。同时 `logger.warning` 打一行摘要。

### 边界条件与坑

- 因果探针 = **每次额外 2 次完整 forward**，这就是它默认关闭、且只在前 1 次做的原因（参数默认值的直接证据）。
- 审计器对视频同样生效（modalities 二元组 L94-97 同时列了 image/video 两组 key）。
- flip 探针要求 `pixel_values.shape[0] >= 2`（L163-164），单行 patch 的极小图跳过。

---

## 模块 5：rollout 侧多模态（slime/rollout/sglang_rollout.py + slime/utils/processing_utils.py）

### 职责

把"含图/视频的 prompt"变成：给 SGLang 的 rollout 请求（图走 base64/路径，token 未展开）+ 给 Megatron 训练侧的多模态张量（pixel_values/grid_thw，token 已展开），并保证两侧 token 数对账一致。

### 关键数据结构

`Sample` 新增字段（`slime/utils/types.py`，cd365742 diff）：`train_prompt_ids`（processor 展开后，训练用）、`rollout_prompt_ids`（纯文本 tokenizer 编码，给 SGLang）、`rollout_response_ids`（引擎返回的 response token）。不变量：`sample.tokens == train_prompt_ids + rollout_response_ids`（`ray/rollout.py` L262-263 有硬断言）。

### 核心流程

**1. `prepare_model_inputs()`（processing_utils.py L77-123，多模态分支）**
- `extract_vision_info(prompt)`（L81）抽出 `rollout_video_data`：视频**路径字符串列表**，给 SGLang 用（引擎自己解码采样帧）；
- `process_vision_info(prompt, return_video_kwargs=True, return_video_metadata=True, image_patch_size=...)`（L83-88）读图/解码视频帧，`image_patch_size` 从 `processor.video_processor.patch_size` 取（缺省 14）；
- 视频元数据（fps 等，`video_metadata`）从 videos 二元组里拆出塞进 `videos_kwargs`（L90-93、96），再进 processor——**训练侧的帧采样参数由 processor + video_kwargs 决定**；
- `build_processor_kwargs()`（L16-30）：text 强制 `return_tensors=None`（拿 list），模态输出强制 `return_tensors="pt"`（拿张量）；
- **两套 ids 同点产出**：`rollout_input_ids = tokenizer.encode(text_prompt)`（L103，纯文本，图像占位符不展开）；`input_ids = processor_output["input_ids"][0]`（L104-105，processor 把 `<image>` 展开成视觉 token 序列）；
- `multimodal_train_inputs`（L106-112）：processor 输出除 `input_ids` 外全部保留，`attention_mask` 转 bool，非张量转张量。`extra_info` 同时带 `images / videos / rollout_prompt_ids / rollout_video_data / multimodal_inputs / multimodal_train_inputs`（L114-121）。

**2. `generate()`（sglang_rollout.py L200-363）多模态段**
- L211-232：三段 token 的初始化与**resume 防护**——`train_prompt_ids`/`rollout_prompt_ids` 已有值且与新计算不一致，直接 `RuntimeError("... changed while resuming rollout")`（L223-228）；有 response 但缺 response ids 时从 `tokens` 尾部切出（L229-231）；最后统一 `sample.tokens = train + rollout_response`（L232）。
- 图片：`payload["image_data"] = [encode_image_for_rollout_engine(img) ...]`（L266-267）——RGB 转换后 **PNG** base64（processing_utils L126-132；注意 docstring 写的是 JPEG，代码实际存 PNG，文档/实现不一致的小瑕疵）；视频：`payload["video_data"] = video_data` 路径字符串列表，且逐元素断言是 str（L270-273）。
- **发给引擎的是未展开 ids**：`payload["input_ids"] = rollout_prompt_ids + rollout_response_ids`（L276）——图像 token 由 SGLang 服务端随 image_data 自行展开。
- **对账**（L280-287）：有媒体时，要求 SGLang 返回的 `meta_info.prompt_tokens == len(train_prompt_ids) + len(rollout_response_ids)`，不等直接 RuntimeError。这验证的是"引擎展开后的 prompt 长度 == 训练侧展开后的 prompt 长度（含已生成 response 的续写场景）"——两套展开规则（引擎 vs processor）一旦不一致，第一轮就炸而不是静默错位。
- response 累积走 `rollout_response_ids.extend(...)` 再重建 `tokens`（L323-325）；slime-router 分支则整串取回后按 response_length 切三段（L301-303）。

**3. `_ensure_multimodal_train_inputs()`（L366-382，b96d83e2 新增）**
递归处理 sample 列表；若 `multimodal_train_inputs` 已有或 `multimodal_inputs` 为 None 则跳过；否则重新跑 `prepare_model_inputs` 补上训练侧张量。调用点在 `generate_and_rm()` L411——**在 generate 之后、对任意来源的 sample 统一补齐**。动机：自定义 generate 函数（`--custom-generate-function-path`，多智能体 rollout）不走 `generate()` 内部，训练侧张量必须在统一收口处补齐。

**4. 回传与校验（slime/ray/rollout.py）**
L262-263：`tokens == train_prompt_ids + rollout_response_ids` 硬断言；rollout logprobs"全有或全无 + 长度 == response_length"断言；L308-309：`multimodal_train_inputs` 进 `train_data`；L351：参与 DP partition 的 key 列表。

### 边界条件与坑

- resume 场景三段 ids 的任何漂移都是 RuntimeError 而不是警告——partial rollout 续写时 token 边界错一位，loss 全错且无报错，所以必须硬炸。
- `max_new_tokens` 扣减用 `len(rollout_response_ids)`（L241）——续写场景已生成部分不计入预算。
- 视频进引擎只传路径（字符串），训练侧才解码成帧张量——两侧解码实现不同（SGLang vs qwen_vl_utils+processor），靠 prompt_tokens 对账兜住一致性。

---

## 模块 6：训练侧数据管线（data.py / model.py / actor.py / cp_utils.py / processing_utils.py / utils/data.py）

### 职责

把 rollout 回传的逐样本多模态数据组装成 bridge 模型能吃的 `[B, S]` padded batch，并在 CP 下保持 logits / logprob / mask 三处切片口径一致。

### 核心流程

**1. 数据集加载期校验（slime/utils/data.py `_build_messages`，b96d83e2 diff）**
`--multimodal-keys '{"image":"images"}'` 声明占位符↔数据列映射；按占位符切 prompt、组装 Qwen-VL chat content list；**占位符个数 ≠ 媒体个数直接 ValueError**（含列名与两侧计数）。cd365742 兼容 numpy ndarray 列（`.tolist()`）。mathvision 脚本内嵌 python 还做了离线逐行校验——数据错误三道闸（离线 / 加载 / 运行时 audit）。

**2. actor 侧上卡（actor.py `_get_rollout_data`，b96d83e2 diff）**
逐样本把 `multimodal_train_inputs` 里的张量 `.to(cuda, non_blocking=True)`；注入逐样本 `bridge_mode` 标志（`megatron_to_hf_mode == "bridge"`）；**bridge 模式下 rollout_log_probs 不做预切**（保留 fp32 全量，推迟到 get_batch 用统一 `s_pad` 口径切）。

**3. `get_batch()`（data.py L59-177）——多模态组装核心**

```python
# L60-73: bshd 化
align_size = tp_size * (2 * cp_size if cp_size > 1 else 1)
s_pad = ceil(max_length / align_size) * align_size
input_ids_bshd      = full((B, s_pad), pad_token_id)   # 逐样本填入前缀
attention_mask_bshd = zeros((B, s_pad), bool)          # 有效位 True
batch["bshd_padded_length"] = s_pad

# L128-148: 两组 modality 统一拼接
modality_pairs = (
    ("pixel_values",        "image_grid_thw", "num_images_per_sample"),
    ("pixel_values_videos", "video_grid_thw", "num_videos_per_sample"),
)
# 每组：成对校验 -> torch.cat(dim=0) 跨样本拼接 -> 记录逐样本图/视频数

# L150-177: 响应侧张量按 s_pad 口径做 CP 切片
for key in ("log_probs","ref_log_probs","rollout_log_probs","values","advantages","returns","entropy"):
    if len(value) == response_length:
        value = slice_log_prob_with_cp(value, total_length, response_length, s_pad)
```

要点：
- `pixel_values`/`image_grid_thw` **沿 dim 0 直接 cat 跨样本**——Qwen-VL 约定下视觉塔消费的就是"所有图的 patch 大拼盘 + grid 元数据"，逐图边界由 `image_grid_thw` 的行承载，不需要按样本分开。audit 的不变量（行数 == `prod(thw)` 求和）正是校验这个拼盘完整性。
- `num_images_per_sample`/`num_videos_per_sample` 写入 batch 但 slime 侧**无下游消费**（grep 全仓库仅 data.py 一处）——合理推测为调试/预留字段，真正的边界信息由 grid_thw 承载。
- **pixel_values 不做 CP 切分**：`slice_with_cp` 只作用于 token 序列（L76），视觉张量整批进模型。CP 下视觉 embedding 的汇聚由模型内部 `AllGatherVisionEmbeddings` 完成（见模块 2）。
- 序列的 CP 切分发生在 **Megatron 模型内部**：data.py 给 bridge 模型的是**未切片**的完整 `input_ids_bshd`；证据是 loss.py 按 CP chunk 偏移从输出 logits 抠 response 段（见下），说明模型输出已是本 CP rank 的片段。

**4. forward 注入（model.py L270-290 / L511-531，cd365742 形态）**
bridge 分支构造 kwargs：`input_ids=input_ids_bshd, attention_mask=attention_mask_bshd, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw, position_ids=None, labels=None, packed_seq_params=None`，经 `audited_vlm_forward(model, kwargs, args, {"phase": ...})` 调用。视觉 embedding 的"注入"（图像 embedding 替换 input_ids 中视觉 token 位置的 text embedding）发生在 bridge 的 `Qwen35VLMoEModelProvider` 所建模型内部——slime 侧只负责把张量喂到签名上。

**5. loss 对齐（loss.py `get_responses` bshd 分支 L61-87）**
logits 形状 `[B, s_pad, V]`（本 CP rank 片段）。CP=1：直接 `[row, total-response-1 : total-1]`；CP>1：`chunk_size = s_pad // (2·cp)`，按 `get_logits_and_tokens_offset_with_cp(total, response, s_pad)` 算出的 logits_offset/tokens_offset 从首尾两个 chunk 抠 response 区间再 cat，并断言抠出的 logits/tokens 等长。`cp_utils.py` 全套函数（b96d83e2 diff）新增 `padded_length` 参数并断言 `padded_length >= total_length`、`% (2·cp) == 0`——**logits 抠取、rollout logprob 切片、loss_mask 求和（`get_sum_of_sample_mean`）三处共用同一 s_pad 口径**，这是 CP 正确性的核心不变量。

### 边界条件与坑

- bridge 模式显式不支持 combined-1f1b schedule plan（model.py 直接 raise）、critic、tree、MTP 训练（arguments.py `slime_validate_args`）。
- pad 长度按 batch 内最长样本对齐到 `tp·2cp`，长短样本混排时 padding 算力浪费明显——靠 `--balance-data` 和 dynamic batch size 缓解（配置证据：mathvision 脚本 PERF_ARGS/ROLLOUT_ARGS）。
- `log_rollout_data` 对 bridge 模式的逐样本均值用了另一条求和路径（b96d83e2 diff：bridge 且 `numel == sum(response_lengths)` 时按 chunk 直接加权），因为 logprob 已是切片后形态，不能再走 `get_sum_of_sample_mean` 的 CP 还原逻辑。

---

## 模块 7：端到端数据流图（含形状与字段名）

以 MathVision 单图样本为例（视频同理，key 换 `pixel_values_videos`/`video_grid_thw`）。记号：`P` = 单图 patch 数 `t·h·w`；`V` = 视觉 token 数 `P/merge²`；`L_txt` = 纯文本 token 数；`R` = response 长度；`B` = micro-batch 样本数；`s_pad` = 对齐后 batch 序列长。

```
数据集 parquet {prompt, images:[path], answer}
   │  scripts/run-...-mathvision-...sh 内嵌 python：组 prompt、插 <image>、离线校验占位符数==图片数
   ▼
slime/utils/data.py::_build_messages          ── 占位符/media 数量硬校验（ValueError）
   │  Sample{prompt: chat 格式 str, metadata}
   ▼
────────────── rollout 侧（SGLang）──────────────
processing_utils.py::prepare_model_inputs
   ├─ train_prompt_ids   = processor(...)["input_ids"][0]   [L_txt + V]   ← 图像占位符已展开
   ├─ rollout_prompt_ids = tokenizer.encode(text_prompt)    [L_txt]       ← 未展开
   ├─ multimodal_train_inputs = {pixel_values [P, patch_dim],
   │                             image_grid_thw [n_img, 3],   # 每行 (t,h,w)
   │                             attention_mask(bool)}
   └─ rollout_video_data / images（路径字符串 / PIL）
sglang_rollout.py::generate
   │  payload: input_ids = rollout_prompt_ids [L_txt]        ← 引擎内部展开
   │          image_data = [PNG base64] / video_data = [path str]
   │  SGLang /generate ──► output.meta_info.prompt_tokens    ← 引擎展开后的实际 prompt 长度
   │  对账: prompt_tokens == len(train_prompt_ids)+len(rollout_response_ids)，不等即 raise
   ▼  Sample{train_prompt_ids, rollout_prompt_ids, rollout_response_ids [R],
            tokens = train+rollout_response [L_txt+V+R], rollout_log_probs [R],
            multimodal_train_inputs}
generate_and_rm → _ensure_multimodal_train_inputs            ← 自定义 generate 的统一补齐点
   ▼
ray/rollout.py：tokens==train+response 断言 → train_data["multimodal_train_inputs"] → DP partition
   ▼
────────────── 训练侧（Megatron, bridge 模式）──────────────
actor.py::_get_rollout_data：张量上卡(non_blocking)，注入 bridge_mode 标志
   ▼
data.py::get_batch
   ├─ input_ids_bshd      [B, s_pad]   int64, pad_token_id=0
   ├─ attention_mask_bshd [B, s_pad]   bool
   ├─ pixel_values        [ΣP_i, patch_dim]   ← 跨样本 cat(dim=0)，不切 CP
   ├─ image_grid_thw      [Σn_img_i, 3]       ← 跨样本 cat(dim=0)
   └─ rollout_log_probs   逐样本 slice_log_prob_with_cp(..., s_pad) → 本 rank 片段
   ▼
model.py::train_one_step / forward_only
   audited_vlm_forward ──► VLMForwardAuditor（可选）：不变量校验 + 因果探针 + JSONL
   ▼
bridge 模型内部（megatron-bridge Qwen35VLMoE）：
   视觉塔 forward(pixel_values, grid_thw) → 视觉 embedding
   → AllGatherVisionEmbeddings 在 CP group 汇聚（slime patch 过 apply 签名）
   → 替换 input_ids_bshd 中视觉 token 位置的 embedding
   → 语言模型（CP 内部切分序列）→ logits [B, s_pad 的本 rank 片段, vocab]
   ▼
loss.py::get_responses(bshd 分支)：按 CP 偏移抠 response 段 logits/tokens
   → policy_loss_function（GSPO/GRPO + use_rollout_logprobs）→ 反传（视觉塔冻结，requires_grad=False）
   ▼
update_weight（bridge）：export_hf_weights → postprocess_hf_param → colocate 直传 SGLang
```

---

## 模块 8：设计决策与权衡（证据分级）

**D1：为什么训练侧和 rollout 侧用两套 prompt ids？**
代码证据：`prepare_model_inputs` 同点产出两套（processing_utils.py L103-105）；payload 发未展开 ids + 原始图（sglang_rollout.py L266-276）；对账逻辑（L280-287）。直接证据到此。合理推测：SGLang `/generate` 的多模态接口契约就是"文本 ids + image_data，引擎内部展开"；若发 processor 已展开的 ids，引擎会二次展开或无法对齐 image 占位。而训练侧必须显式知道每个占位符展开的 token 数（loss 对齐、CP 切分、logprob 长度都依赖），所以必须自持一份展开版。两套各服务一个系统，用 `prompt_tokens` 对账弥合。

**D2：为什么视觉权重一度走 passthrough 而不是让 Megatron 持有？**
代码证据：visual_passthrough docstring（L1-22）+ 2b8ffbf5 对 `convert_hf_to_torch_dist.py` docstring 的改写。时间线证据：passthrough 是先存在的兜底；2b8ffbf5 才引入载体方案并设为自动默认。即**不是"选 passthrough 而不是持有"，而是"先有 passthrough 应急，后用持有方案替代，passthrough 留作 `--visual` 兜底"**。持有的好处（自包含 checkpoint）在模块 3 已述。
另有一处诚实的张力需要说明：载体方案的 docstring 称视觉权重"不参与在线 Megatron→SGLang 同步"，但 dd67e5ee 修的恰恰是 update_weight 路径对视觉参数的处理（GLU 排除、strict zip）——说明实际上视觉参数会进入同步枚举，该路径必须对它们正确。docstring 表述与代码现实有出入，以代码为准。

**D3：为什么审计工具做成可选开关而非常开？**
直接证据：参数默认全关（log=None、interval=0、causal=False）；mathvision 脚本里这组参数是注释态；`audited_vlm_forward` 未配置时零开销透传（L268-269）；因果探针每次 = 2 次额外完整 forward（L155-198），审计 forward 还带 fsync 的同步 IO（L200-213）。合理推测：常开的算力/IO 成本不可接受，且审计价值集中在"链路刚打通"和"出问题排查"两个时刻，所以做成默认关闭、按需打开、前 N 次必审的采样式设计。

**D4：为什么数据校验全部做成 raise 而不是 warning？**
直接证据：`_build_messages` 的 ValueError、generate 的 RuntimeError 对账、audit 的一串 RuntimeError、ray/rollout.py 的 assert。合理推测（上篇也论述过）：多模态数据错误在 RL 里的表现是 reward 噪声/loss 微偏，几乎不可回溯；把隐式假设全部变成显式异常是这套代码贯穿始终的质量策略。

---

## 模块 9：面试追问预演（多模态代码细节专项）

**Q1：`image_grid_thw` 三个维度是什么？**
A：每张图一行 `(t, h, w)`，是 **patch 网格**尺寸：t = 时间维 patch 数（图片为 1，视频为帧方向 patch 数），h/w = 空间维 patch 数。代码证据：`vlm_forward_audit.py` L124 用 `grid.prod(-1).sum()` 算总 patch 行数并与 `pixel_values.shape[0]` 对账；L129-132 用 `prod(t,h,w) // spatial_merge_size²` 算应展开的视觉 token 数。即一张图占 `t·h·w` 行 pixel、展开成 `t·h·w/merge²` 个视觉 token。

**Q2：视频的 fps/元数据怎么传？**
A：训练侧：`process_vision_info(return_video_kwargs=True, return_video_metadata=True)` 返回的 `video_kwargs` + 逐视频 `video_metadata` 被合入 `build_processor_kwargs()["videos_kwargs"]` 再进 processor（processing_utils.py L83-96）——帧采样参数由 processor 侧决定。rollout 侧不传元数据，只传视频路径字符串列表 `payload["video_data"]`（sglang_rollout.py L270-273），帧采样由 SGLang 引擎内部完成；两侧一致性靠 `prompt_tokens` 对账兜底。

**Q3：CP 下 `pixel_values` 切不切？**
A：不切。`get_batch()` 只对 token 序列做 `slice_with_cp`（data.py L76），`pixel_values`/`image_grid_thw` 是整 micro-batch `torch.cat(dim=0)`（L143-144）后原样进模型。视觉塔在每个 rank 上对完整 patch 集合 forward，embedding 经 `AllGatherVisionEmbeddings` 在 CP group 汇聚（`vision_dp_when_cp=False` 选的就是这条策略），再按序列位置 scatter 到各 CP rank 的片段——汇聚逻辑在 megatron-bridge 包内部（本地无源码，此句为基于 provider 标志与 patch 对象的合理推断）；slime 侧的证据是：输入模型的 `input_ids_bshd` 是**未切片**的完整 batch，而 loss 侧按 CP 偏移抠 logits，说明序列 CP 切分发生在模型内部。

**Q4：如果 SGLang 和 processor 的图像 token 展开规则不一致会怎样？**
A：第一轮 rollout 就 RuntimeError。`generate()` L280-287：有媒体时强制 `meta_info.prompt_tokens == len(train_prompt_ids) + len(rollout_response_ids)`。processor 展开长度（训练侧）与引擎展开长度（rollout 侧）的任何分歧——比如 patch_size、merge_size、最小/最大像素约束版本不一致——都会在这个对账点爆炸，而不是变成静默的 logprob 错位。

**Q5：`multimodal_inputs` 和 `multimodal_train_inputs` 两个字段有什么区别？**
A：`multimodal_inputs` 面向 rollout 引擎（图像原始对象/base64 等，决定 `Sample.multimodal_inputs` 是否需要往引擎带）；`multimodal_train_inputs` 是 processor 产出的**训练张量字典**（pixel_values/grid_thw/bool mask），随 train_data 进 Megatron batch。types.py 两个字段并存（b96d83e2 diff），`_ensure_multimodal_train_inputs` 的短路条件（L371）正是"训练张量已有或引擎输入根本没有多模态"。

**Q6：视觉塔的 LN 参数为什么挂在 `linear_qkv`/`linear_fc1` 名字下？**
A：这是对齐 Megatron 语言模型的命名惯例（pre-LN 寄生在随后的 Linear 模块上）。`TensorParallelLinear(layer_norm_size=...)` 创建 `layer_norm_weight/bias`（vision_model.py L129-139），映射时在 `mcore_to_hf_vision_name` 里还原为 HF 的 `norm1.*`（attn 前）/`norm2.*`（mlp 前）（L315-316、L321-322）。面试加分点：merger 的 LN 因为模块名冲突改叫 `patch_norm` → HF `merger.norm`（L300-301）。

**Q7：QKV 重排为什么必须做？不做会怎样？**
A：HF fused qkv 是 qkv 优先布局（`view(3, H, D)`），mcore 为 TP dim-0 切分正确性用 head 优先（`view(H, 3, D)`）。不做重排直接切分：每个 TP 分片里会混入"前半个 q 头 + 后半个 k 头"这类断头，注意力计算全错但**不报错**（形状完全合法）。两个方向的函数只差第一步 view（L334-355），bias 走同一函数靠 `*trailing` 兼容。

**Q8：dd67e5ee 的 GLU bug 具体是什么？**
A：`update_weight/common.py` 的 all-gather 路径对一切名字含 `linear_fc1.weight` 的参数做 `chunk(2, dim=0)` 的 gate/up 重排（Megatron gated MLP 的 fc1 在 TP 维度上 gate/up 交错，HF 是分离矩阵，同步时要重排）。视觉塔的 fc1 是**非门控**普通 MLP（VisionMLP L175-190 只有一份 fc1），重排会把权重数值改错再推给 SGLang。修复：`_needs_glu_reorder = "linear_fc1.weight" in name and "vision_model." not in name`（common.py L15-16）。教训：**按名字子串做的启发式转换，每引入一类新参数都必须重审**。

**Q9：resume / partial rollout 时三段 token 怎么保证不错位？**
A：四层防护：`generate()` 开头已有 `train_prompt_ids`/`rollout_prompt_ids` 且与新计算不一致 → RuntimeError（L221-228）；有 response 无 response_ids 时按 `len(train_prompt_ids)` 切出（L229-231）；每轮生成后 `tokens` 由 `train + rollout_response` 重建而不是就地 append（L232、L324）；回传时 `ray/rollout.py` L262-263 再断言等式成立。

**Q10：`attention_mask` 为什么强制 bool？**
A：bridge 的 VLM 模型按 bool mask 实现（内部 `~mask` 或 masked_fill 语义）。audit `_validate_inputs` L100-101 对非 bool 直接 RuntimeError；产生侧 `prepare_model_inputs` L112 把 processor 的 mask 转 bool，`get_batch` L67 直接建 bool 张量。两处生产 + 一处校验，类型在链路上被钉死。

**Q11：为什么 `pixel_values` 可以跨样本直接 cat？逐图边界丢了怎么办？**
A：没丢——边界由 `image_grid_thw` 的行承载（每行一张图的 (t,h,w)）。这是 Qwen-VL 系列的既定约定：视觉塔消费"所有图的 patch 拼盘 + grid 元数据"，内部按 grid 切回逐图。audit 的不变量校验（总行数 == Σprod(thw)，L124-128）正是保证 cat 过程没把拼盘弄乱。代价是 `num_images_per_sample` 这类冗余字段在 slime 侧无人消费（grep 证据），只作调试信息。

**Q12：审计工具为什么不直接对比 SGLang 和 Megatron 的 logprob？**
A：那是跨系统对拍，数值路径差异（kernel、dtype、attention 实现）会淹没信号。`_causal_probe` 用的是**单系统因果干预**：同一模型、同一输入，只把 pixel patch 顺序 flip，logits 必须变（阈值 1e-6，L195-196）。它回答的是"视觉链路通不通"这个二值问题，先把它排除，再谈训推数值对齐——排障上这是严格更强的二分。文件里 `_tensor_stats`（finite/mean/abs_max/l2_norm）才是留给数值对比的素材。

**Q13：bridge 模式下 rollout logprob 为什么推迟到 get_batch 才切 CP？**
A：raw 路径在 actor 侧按逐样本实际长度预切；bridge 训练侧 logits 按 batch 统一 `s_pad` 布局切。两个口径不同 → 错位。所以 bridge 模式 actor 只上卡不切片（b96d83e2 actor.py diff 的 bridge 分支），切片在 `get_batch` L169 用 `slice_log_prob_with_cp(..., s_pad)` 完成，与 logits 抠取、mask 求和共享同一 `padded_length` 口径。原则：**需要对齐的多处切片必须共用一份长度元数据、在同一点计算**。

**Q14：冻结视觉塔后，在线权重同步还会推视觉权重吗？**
A：bridge 路径：`export_hf_weights` 从完整 VLM 模型导出，视觉塔自然在导出范围内（frozen，值不变，推送无害）。raw 路径：dd67e5ee 的修复（GLU 排除、strict zip、四重断言）恰恰说明视觉参数会进入同步枚举，所以该路径必须对它们显式正确——注意这与 2b8ffbf5 docstring 的"不参与在线同步"表述存在张力，以代码行为为准（模块 8-D2 已述）。

**Q15：如果新增一种模态（比如 audio），要动哪些地方？**
A：沿着现有图像/视频的平行结构复制即可：① `modality_pairs`（data.py L133-136）加一组三元组；② audit 的 `modalities` 二元组（vlm_forward_audit.py L94-97）加一行；③ `MultimodalTypes` 与 `--multimodal-keys` 支持新占位符（utils/data.py）；④ `prepare_model_inputs` 的 extra_info 加对应 key；⑤ model.py 的 forward kwargs 加对应参数（前提是 bridge provider 的模型签名支持）。这套"每组模态 = (values, grid, count) 三元组"的同构设计（cd365742 从单组图像推广到图+视频两组时确立）就是为了让第 N 种模态是纯增量。

---

*本文所有行号对应 HEAD=cd365742 的工作区文件；标注「合理推测」的条目为代码中无直接证据、基于上下文的工程推断。*


# VLM Audit 与训推 Token 一致性：5 层递进追问自问自答

> 仓库：krl（fork 自 THUDM/slime），HEAD = cd365742。
> 主题一：`slime/backends/megatron_utils/vlm_forward_audit.py`（274 行）
> 主题二：训推 token 一致性（`slime/rollout/sglang_rollout.py` 三段 token + prompt_tokens 对账；`slime/utils/processing_utils.py`；`slime/ray/rollout.py` 断言）
> 所有行号对应 HEAD 工作区文件；标注「推测」的条目为代码中无直接证据的工程推断。

---

## 使用说明

**怎么练**：每一组先遮住【参考回答】，自己用口语答一遍，再对照。答的时候强迫自己说出具体文件名和函数名——说不出来就是没真读过。

**每层答不上来意味着什么**：

- **L1（概念层）答不上 = 没做过**。这两个机制"解决什么问题"必须能一句话说清。
- **L2（机制层）答不上 = 没读过代码**。流程要能脱稿讲，不需要背行号。
- **L3（权衡层）答不上 = 只做过没思考过**。这是"执行者"和"工程师"的分水岭，面试官从这里开始区分人。
- **L4（细节层）答不上 = 正常，答上是加分**。行号、默认值、边界条件，答出来证明代码是你亲手写/亲手调过的。
- **L5（施压层）答不上 = 完全正常，答上是强加分**。故障注入和反事实问题没有标准答案，面试官看的是你能不能把系统行为从代码里推出来。本篇 L5 的回答全部给出了代码证据，作为推理范例。

---

## 主题一：VLM forward audit 工具

### Q1.1（L1 概念层）

【面试官问】你提到你们写了个 VLM forward audit，这是个什么东西？解决什么问题？

【参考回答】
这是我们在 Megatron 训练侧加的一个运行时审计器，解决的是多模态 RL 里一类特别坑的问题：**图像到底进没进模型，从 loss 曲线上是看不出来的**。纯文本 RL 里数据错了 loss 会明显不对，但 VLM 里如果图像 embedding 因为 CP 切分、scatter 逻辑或者数据管线的 bug 被静默丢掉，语言模型照样能对文本部分算出 logprob，loss 看起来完全正常，reward 只是"莫名其妙地差"。我们遇到过这类问题之后，就写了这个工具：它包在每次 forward 外面，做输入不变量校验、视觉塔调用计数、还有一个反转图像内容看输出变不变的因果探针，把"视觉链路断了"从隐式假设变成显式异常。默认是关的，排查时通过 `--vlm-forward-audit-log` 打开。

【代码证据】
`vlm_forward_audit.py:267-274`：`audited_vlm_forward()` 入口，未配置 log 或非 bridge 模式时直接 `return model(**kwargs)` 透传。
`arguments.py`（cd365742 diff）：`--vlm-forward-audit-log` 默认 None，整个机制默认关闭。

【回答要点点评】
考察点：能否一句话说清"为什么需要它"而不是"它有什么功能"。好回答的标志是点出问题的特殊性——**多模态错误的静默性**（loss 不报错、reward 缓慢变差），这说明答者真的被这类 bug 咬过。

### Q1.2（L2 机制层）

【面试官问】那它具体是怎么实现的？讲讲工作流程。

【参考回答】
核心是 `VLMForwardAuditor` 类，通过 `audited_vlm_forward` 包在模型调用外面，每个 model 实例用 `WeakKeyDictionary` 挂一个 auditor。四层机制：第一层，每次被审计的 forward 先做输入校验——attention mask 必须 bool、pixel_values 和 grid_thw 必须成对、pixel 行数必须等于 grid 里 t·h·w 的乘积求和、input_ids 里视觉 token 数必须等于 patch 数除以 merge 尺寸的平方、pixel 必须全 finite。第二层，给 vision_model 注册 forward hook 计数，有媒体时要求一次 forward 恰好调用一次视觉塔，并且能捕获到视觉输出。第三层是因果探针，把 pixel_values 沿 patch 维反转后重跑 forward，比较两次 logits，没变就说明图像根本不影响输出。第四层落盘，rank 0 把每次审计的 tensor 统计、对账结果、因果探针结果写成 JSONL，用 os.open 加 fsync 保证崩溃现场不丢日志。审计频率是前 N 次必审、之后按 interval 采样，由参数控制。

【代码证据】
`vlm_forward_audit.py:60-72`：`__init__` 里 `_core_model()` 找到同时有 `vision_model`/`language_model` 的核心模块并注册 hook。
`vlm_forward_audit.py:225-261`：`forward()` 主流程——`_validate_inputs` → 可选 `_causal_probe` → 真实 forward + 调用计数校验 → `_write` 落盘。

【回答要点点评】
考察点：能否把 274 行代码压缩成结构化的四层机制。注意参考回答的顺序就是代码里 `forward()` 的执行顺序——按代码结构讲，比按功能分类讲更显得亲手写过。

### Q1.3（L3 权衡层）

【面试官问】为什么不直接对比 SGLang 和 Megatron 的 logprob 来查训推一致性，而要搞个 flip 图像的因果探针？还有，为什么做成开关而不是常开？

【参考回答】
两个问题其实是同一个设计哲学：**信号的特异性**。跨系统对拍的问题是噪声太大——SGLang 和 Megatron 的 kernel、dtype、attention 实现、router 精度全都不一样，logprob 本来就有合理差异，你设什么阈值都是拍脑袋，图像链路断没断这个二值信号会被数值噪声淹没。因果探针换了个思路：同一个模型、同一份输入，只把图像 patch 顺序 flip，logits **必须**变。这是单系统内的干预实验，回答的是"视觉链路通不通"这一个二值问题，几乎零误报。先把这个排除掉，再去谈数值对齐——排障上是严格更强的二分。至于为什么做成开关：因果探针每次是两次额外的完整 forward，审计本身还带 fsync 的同步磁盘 IO，常开的算力和 IO 成本不可接受；而且它的价值集中在"链路刚打通"和"出问题排查"两个时刻，所以默认全关、前几次必审、按需打开。

【代码证据】
`vlm_forward_audit.py:170-172`：`real_output, real_vision = self._run_probe(kwargs, "causal_real")` 后 `shuffled_kwargs[values_key] = pixel_values.flip(0).contiguous()` 重跑。
`vlm_forward_audit.py:195-196`：`if result["logits_max_abs"] <= self.args.vlm_forward_audit_min_logit_diff: raise RuntimeError(...)`——阈值默认 1e-6（arguments.py）。
`vlm_forward_audit.py:165-174`：探针内 `torch.no_grad` + eval 模式 + RNG 保存恢复，零梯度开销。

【回答要点点评】
考察点：调试方法论。关键词是**特异性**——"用一个几乎零误报的二值实验替代高噪声的连续量对比"。能讲出"为什么跨系统对拍不行"比能讲出"探针怎么做"更值钱。

### Q1.4（L4 细节层）

【面试官问】问几个细的：flip(0) 翻的是什么？为什么要求 pixel_values 至少两行？审计日志为什么要 fsync？频率控制的具体逻辑是什么？

【参考回答】
flip(0) 翻的是 pixel_values 的第 0 维，也就是 patch 行维——Qwen-VL 约定下每张图占 t·h·w 行 patch，flip(0) 把整个 batch 所有图的 patch 顺序整体反转，图像内容必然改变但张量形状、dtype 完全不变，是最小侵入的干预。要求 `shape[0] >= 2` 是因为 flip 一个单行张量是恒等操作，探针会假性通过，所以直接跳过。fsync 是因为审计日志的最大价值在崩溃现场——如果用普通 buffered write，进程崩的时候最后几行还躺在 Python 缓冲里，恰恰是最接近事故现场的记录丢了，所以用 os.open 加 O_APPEND 加 write 加 fsync 的系统调用序列，还检查了 short write。频率控制：`_should_audit` 里是 `index <= first_n or (interval > 0 and index % interval == 0)`，forward_count 在进入时先自增，默认 first_n=4、interval=0，也就是默认只审前 4 次 forward；因果探针单独由 causal_first_n 控制，默认只做 1 次，因为它最贵。

【代码证据】
`vlm_forward_audit.py:163-164`：`if pixel_values.shape[0] < 2: return None`。
`vlm_forward_audit.py:206-212`：`os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)` → `os.write` → `os.fsync(fd)`，且 `written != len(payload)` 时 raise short write。
`vlm_forward_audit.py:83-90`：`_should_audit` 与 `_should_run_causal`（`causal_count < causal_first_n`）。
`arguments.py`（cd365742 diff）：`--vlm-forward-audit-first-n` 默认 4、`--vlm-forward-audit-interval` 默认 0、`--vlm-forward-audit-causal-first-n` 默认 1、`--vlm-forward-audit-min-logit-diff` 默认 1e-6。

【回答要点点评】
考察点：是不是亲手逐行读过。"flip 单行是恒等操作所以跳过"、"崩溃时 buffer 里丢的恰恰是最后几行"这种级别的细节，靠背是背不出来的，面试官一听就知道真假。

### Q1.5（L5 极限施压层）

【面试官问】我给你几个场景：第一，CP 模式下如果视觉塔因为 bug 被调了两次，你的 audit 在哪一步发现？第二，如果某张图的 patch 数不能被 spatial_merge_size² 整除会怎样？第三，因果探针切了 eval 模式又切回来，会不会影响 dropout 之类的状态？两次额外 forward 对训练有什么副作用？

【参考回答】
第一个场景：在第二层"调用计数"被发现，不用等因果探针。每次审计 forward 前记下 `vision_calls` 基数，forward 后取差值，有媒体时 `calls != 1` 直接 RuntimeError，报错信息里带实际次数。调用两次和零次都在同一个检查点炸。第二个场景：audit 里 `expected_tokens = expected_rows // merge²` 是整除，如果 patch 数整除不了，那 input_ids 里实际的视觉 token 数必然不等于这个floor值，`token invariant failed` 直接 raise——不过更真实的情况是 processor 侧保证了 grid 和 token 展开的一致性，audit 抓的是**两侧约定被破坏**（比如自研 processor 改了 merge 逻辑），而不是 processor 自己的内部错误。第三个场景：探针保存了 `model.training`，做完恢复；CUDA RNG 状态也保存恢复了，所以后续训练 forward 的 dropout 随机序列不被污染；而且 Megatron 这套模型没有 BatchNorm，eval 切换影响的只有 dropout 类层，跑完就切回来了。副作用上，两次额外 forward 是 no_grad 的，不产生梯度图，代价是纯算力和一次显存峰值；而且因果探针被限制在 forward_only 阶段和训练前的专项 audit pass 里、默认只做 1 次，对训练吞吐的影响被参数设计压到最小。

【代码证据】
`vlm_forward_audit.py:235-245`：`calls_before = self.vision_calls` … `if input_record["has_media"] and calls != 1: raise RuntimeError(f"VLM audit expected exactly one vision forward, got {calls}")`。
`vlm_forward_audit.py:124-132`：`expected_rows = int(grid.prod(-1).sum().item())` … `expected_tokens = expected_rows // (merge_size**2)` … 不等即 raise。
`vlm_forward_audit.py:165-174`：`was_training = self.model.training; self.model.eval(); rng_state = torch.cuda.get_rng_state()` … `torch.cuda.set_rng_state(rng_state)` … `if was_training: self.model.train()`。
`vlm_forward_audit.py:230-233`：因果探针仅在 `context.get("phase") == "forward_only"` 且有媒体时触发；`actor.py:495-508` 训练前专项 audit pass（`timer("vlm_audit_causal")`，仅前 `causal_first_n` 个 rollout）。

【回答要点点评】
考察点：从代码推系统行为的能力。三个子问题分别落在"检查点位置"、"整除语义的失败模式"、"状态恢复完备性"上。注意回答里承认了"processor 内部一致时整除问题不会出现"——施压层问题敢于给出边界条件，比硬撑"都能抓到"更可信。

---

## 主题二：训推 token 一致性

### Q2.1（L1 概念层）

【面试官问】多模态 RL 里"训推 token 一致性"具体指什么问题？纯文本场景没有这个问题吗？

【参考回答】
指的是同一个样本，推理引擎（SGLang）看到的 token 序列和训练框架（Megatron）算 loss 用的 token 序列必须完全一致，否则 rollout logprob 和训练 logprob 对的是不同的 token，importance sampling 修正、KL、优势估计全错。纯文本场景这个问题基本不存在，因为两边都是 tokenizer 编码同一段文本，结果确定性一致。多模态引入了分歧点：图像在文本里只是一个 `<image>` 占位符，要展开成几百上千个视觉 token，**展开规则**——patch 怎么切、merge 怎么算、token 数多少——训练侧的 HF processor 和推理侧的 SGLang 引擎各有一份实现。只要两份实现有任何版本或配置差异，两侧序列长度就不一样了，而且不报错，表现为 reward 信号的静默劣化。我们的解法是三段 token 拆分加运行时对账。

【代码证据】
`processing_utils.py:103-105`：`rollout_input_ids = tokenizer.encode(text_prompt, ...)`（纯文本）与 `input_ids = processor_output["input_ids"][0]`（processor 展开后）在同一点分别产出。
`sglang_rollout.py:280-287`：`meta_info.prompt_tokens` 与训练侧长度的对账 raise。

【回答要点点评】
考察点：能否讲清问题在多模态下的**新增性**——不是老问题变严重，而是图像 token 展开规则存在两份实现这个结构性分歧点。能说出"静默劣化"这个后果特征是关键。

### Q2.2（L2 机制层）

【面试官问】三段 token 拆分和对账具体怎么实现的？走一遍流程。

【参考回答】
Sample 上加了三个字段：`train_prompt_ids`、`rollout_prompt_ids`、`rollout_response_ids`。流程是：`prepare_model_inputs` 里同一点产出两套 prompt ids——processor 展开图像占位符后的叫 train_prompt_ids，训练用；tokenizer 纯文本编码的叫 rollout_prompt_ids，给 SGLang。发给引擎的 payload 是 rollout_prompt_ids 拼上已生成的 rollout_response_ids，图像走 image_data 的 base64 或 video_data 的路径字符串，由引擎内部展开。引擎返回后，`meta_info.prompt_tokens`——也就是引擎实际处理的 prompt 长度——必须等于 len(train_prompt_ids) 加 len(rollout_response_ids)，不等直接 RuntimeError。response token 从引擎返回的 logprobs 里提取，extend 进 rollout_response_ids，`sample.tokens` 每次都由 train 加 response 重建。数据回传时 ray/rollout.py 再断言一遍 tokens 等于三段拼接、response_length 等于 response ids 长度、loss_mask 长度等于 response_length。整条链上任何一环漂了都会当场炸。

【代码证据】
`types.py`（cd365742 diff）：`rollout_prompt_ids / train_prompt_ids / rollout_response_ids` 三个新字段。
`sglang_rollout.py:276`：`payload["input_ids"] = sample.rollout_prompt_ids + sample.rollout_response_ids`。
`sglang_rollout.py:323-325`：`rollout_response_ids.extend(...)`；`tokens = train_prompt_ids + rollout_response_ids`；`response_length = len(rollout_response_ids)`。
`ray/rollout.py:262-267`：`assert sample.tokens == expected_tokens` + response_length 断言。

【回答要点点评】
考察点：流程的完整性。注意回答里明确了"哪个 id 给谁用"和"不变量在哪几处被断言"——机制和不变量一起讲，说明理解的是设计而不只是代码路径。

### Q2.3（L3 权衡层）

【面试官问】为什么不统一用一套 ids，两边都用 processor 展开后的，非要拆两套？还有，对账失败为什么是 raise 而不是 warn 记个日志继续跑？

【参考回答】
统一一套听上去干净，但不可行：SGLang `/generate` 的多模态接口契约就是"文本 ids 加 image_data，引擎内部展开"，你把 processor 展开后的 ids 发过去，引擎要么二次展开、要么图像占位和 image_data 对不上——引擎侧根本没有"跳过展开"的选项（推测：这是 SGLang 接口的固有限制，代码里没有注释明说，但 payload 同时携带未展开 ids 和 image_data 说明引擎需要原始占位符来定位图像插入点）。所以两套 ids 不是选择，是两个系统各自的接口契约，我们能做的只是同点产出加对账。为什么 raise 不 warn：因为 token 错位的后果不是"这条样本废了"，而是**整个 batch 的 logprob 对齐全错、梯度被污染但训练继续跑**——warn 之后继续跑等于用错误数据训练还自以为正常。RL 里 reward 噪声本来就大，这种污染混进去根本追不回来。当场 crash 的代价是重启一次任务，warn 的代价可能是几天训练白费，这个账很好算。

【代码证据】
`sglang_rollout.py:266-276`：`payload["image_data"] = [...]` / `payload["video_data"] = video_data` 与未展开的 `input_ids` 同时存在于 payload。
`sglang_rollout.py:283-287`：`raise RuntimeError(f"SGLang processed prompt length ... does not match training context length ...")`。
`ray/rollout.py:290-299`：rollout logprobs"全有或全无"断言 + 长度等于 response_length 断言——同为 fail-fast 风格。

【回答要点点评】
考察点：识别"设计自由度"——哪些是我们选的（对账、raise），哪些是外部契约逼的（两套 ids）。raise vs warn 的回答要算清"错误继续传播的代价"，这是工程判断而非风格偏好。

### Q2.4（L4 细节层）

【面试官问】对账公式里为什么要把 rollout_response_ids 也算进去？prompt_tokens 不是只算 prompt 吗？还有 resume 的时候你们怎么防止 ids 漂移？

【参考回答】
好问题，这正是多轮和 partial rollout 场景的关键。我们发给引擎的 input_ids 是 rollout_prompt_ids 拼 rollout_response_ids——partial rollout 续写时，已生成的 response 对新一次引擎调用来说就是 prompt 的一部分，引擎的 `prompt_tokens` 统计的是它这次实际 prefill 处理的全部 token 数，所以期望值必须是 len(train_prompt_ids) 加 len(rollout_response_ids)。注意等式左边用 train 长度——因为引擎会把图像占位符展开，展开后的 prompt 长度应该等于训练侧展开后的长度，这正是对账要验证的东西；右边加 rollout response 是因为那些 token 是引擎原样吃进 prefill 的。还有个细节：只有存在图像或视频、且 `prompt_tokens` 字段不为 None 时才对账——纯文本样本不查，因为纯文本两套 ids 本来就相等，查了是浪费。resume 防护是 generate 开头的两组检查：sample 上已有 train_prompt_ids 或 rollout_prompt_ids 且与本次重算不一致，直接 RuntimeError("... changed while resuming rollout")；有 response 但缺 response ids 的恢复场景，按 len(train_prompt_ids) 从 tokens 尾部切出来补齐。

【代码证据】
`sglang_rollout.py:280-287`：`processed_prompt_tokens = output["meta_info"].get("prompt_tokens")`；`if (image_data or video_data) and processed_prompt_tokens is not None:` → `expected = len(train_prompt_ids) + len(rollout_response_ids)` → 不等即 raise。
`sglang_rollout.py:221-228`：`elif sample.train_prompt_ids != train_prompt_ids: raise RuntimeError("Train prompt IDs changed while resuming rollout")`（rollout 侧同理）。
`sglang_rollout.py:229-231`：`if sample.response and not sample.rollout_response_ids: response_start = len(sample.train_prompt_ids); sample.rollout_response_ids = sample.tokens[response_start:]`。

【回答要点点评】
考察点：对公式每一项的语义理解。"prompt_tokens 为什么含 response"答不出来就说明没真正理解 partial rollout 的引擎视角；`.get(...)` 加 `is not None` 守卫、纯文本跳过对账这种条件细节是加分项。

### Q2.5（L5 极限施压层）

【面试官问】施压一下：如果 SGLang 升级后 prompt_tokens 的统计口径变了——比如用了 prefix cache 只统计新算的 token——你们的代码会怎样？如果 meta_info 里这个 key 干脆没了呢？再一个，图像如果出现在多轮对话的中间轮次而不是第一轮，你们这套还对吗？

【参考回答】
前两个问题其实是同一个失效模式的两面，而且代码行为不一样，值得分开说。如果口径变了、key 还在：对账会从"永远通过"变成"永远炸"——prefix cache 命中时 prompt_tokens 只算新 token，必然小于我们的期望值，第一轮 rollout 就 RuntimeError。这是**保守方向的失效**，任务起不来、人立刻来查，不会静默出错，我个人能接受这种 trade-off。如果 key 没了：麻烦一点——`meta_info.get("prompt_tokens")` 拿到 None，`is not None` 守卫让整个对账被**静默跳过**，一致性保障凭空消失而且没有任何报警。这是这套机制的真实盲区，要堵的话应该改成"有媒体时 key 缺失也 raise"，现在的写法是兼容老引擎的妥协（推测：当时需要兼容不返回该字段的引擎版本；代码中无注释说明）。第三个问题：多轮中间轮次的图像——我们的三段拆分按"train prompt + rollout response"两段式切分，prompt 内部的结构（第几轮、图在哪轮）对机制是透明的，因为 `prepare_model_inputs` 处理的是完整 message list，processor 会把所有轮的图像都展开，对账验证的是总长度，所以机制上成立；真正的风险是多轮里某轮图像被自定义 generate 函数改写过导致重算的 prompt ids 漂移——这正是 resume 那两条 RuntimeError 守的场景。

【代码证据】
`sglang_rollout.py:280-281`：`output["meta_info"].get("prompt_tokens")` + `processed_prompt_tokens is not None` 守卫——key 缺失时对账整体跳过。
`processing_utils.py:58-68`：prompt 为 list（多轮 messages）时 `apply_chat_template` 处理完整对话，`process_vision_info(prompt, ...)`（L83）对全量消息抽视觉信息。
`sglang_rollout.py:223-228`：resume 漂移 raise。

【回答要点点评】
考察点：能不能诚实地区分"保守失效"（口径变、当场炸）和"危险失效"（key 没了、静默跳过），并指出后者是真实盲区。施压层的最佳策略不是防御系统完美，而是展示你对失效模式的分析比面试官的问题还细——主动说出 key 缺失的盲区并给出修复方向，是把施压问题变成加分题的打法。

---

## 两个主题的串联故事（面试收尾用，140 字）

多模态 RL 的训推一致性，我们的保障体系是"三道闸加一个探针"：数据进系统前，占位符与媒体数量校验把脏数据挡在门外；rollout 时，三段 token 拆分加 prompt_tokens 对账保证两侧序列逐 token 一致，错了当场 crash；训练时，forward audit 的不变量对账守住 pixel 与 token 的数值关系，flip 因果探针证明图像真的影响了输出。四层都遵循同一个原则：多模态错误是静默的，所以每一层防护都必须 fail-fast。

---

*本文所有行号对应 HEAD=cd365742 的工作区文件；标注「推测」处为代码中无直接证据的工程推断。*


