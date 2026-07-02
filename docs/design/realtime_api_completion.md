# Realtime API 补全设计

> **Owner:** Chenyang Zhao | **工期:** 120–170h（~150h）/ 8–10 周 | **代码:** ~450 行核心 + ~650 行测试

---

## 1. 背景与范围

`/v1/realtime` 已实现 WebSocket 建连、VAD、两轮推理、对话历史、文本流式输出。**不能出语音。** 此外多处协议缺口——打断无取消事件、VAD 参数不可热更新、session 无生命周期管理、缺少模块级单元/Mock 测试。

**范围：** 补全到协议完整、有基本容错、可测试。主体改动集中在 `sglang_omni/serve/realtime/` 和测试目录，不碰 pipeline/scheduler/relay 的核心执行逻辑。

**范围修订（基于代码阅读）：** 下列接入点不可避免地会越过 `serve/realtime/`，但应保持“小而薄”：

- `sglang_omni/serve/openai_api.py`：WebSocket 握手前 max sessions 拒绝、FastAPI shutdown 时关闭 realtime sessions、可选 `/metrics` 注册。
- `sglang_omni/client/client.py`：若实现 token-aware conversation trim，可能需要暴露只读 tokenizer/property；若无法稳定暴露，则先使用字符数估算。
- `pyproject.toml`：若引入 `prometheus-client`，需要确认其是否已存在；当前依赖列表未包含该包，不能称为“零额外依赖”。

**测试现状修订：** 不是完全“零测试”。当前已有 `tests/test_model/test_qwen3_omni_realtime.py` 的 GPU/真实模型集成测试，覆盖 VAD → response → transcription 与断连健康检查；缺的是 `serve/realtime/` 的轻量单元测试、mock session 事件序列测试、故障路径测试、cancel 测试与压测。

**详细设计入口：** 本文保留为功能补全路线图；更细的状态机、事件 schema、任务/request 生命周期、buffer 生命周期、session.update 校验矩阵和测试分层见同级文档：`docs/design/realtime_api_detailed_design.md`。

### 1.1 缺口

| # | 缺口 | 现状 |
|---|---|---|
| 1 | 音频输出 | `modalities=["text"]` 写死，talker+code2wav 从未触发 |
| 2 | VAD 热更新 + 多 Turn 状态一致性 | `session.update` 不触达 VAD；abort 路径 `buffer_origin_samples` 可能未推进 |
| 3 | 打断协议 | cancel 后不发取消事件、已产出内容直接丢弃 |
| 4 | VAD 中断策略 | 新 speech 只能排队，不能打断当前 turn |
| 5 | Session 生命周期 | 无超时、无限流、无优雅关闭 |
| 6 | 对话历史 | 无限增长，超出上下文窗口 |
| 7 | Error 事件 | `send_error()` 存在但从未调用 |
| 8 | 测试 | 已有 GPU 集成测试；缺少 realtime 模块级单元/Mock 测试 |

### 1.2 诚实定性

8 个 Feature 中，**7 个是读 spec → 翻译成代码 + 边界处理。** 这些工作正确、必要，但不难——价值在于你需要在 6 个文件、3 层 asyncio Task 嵌套、跨进程数据链路中定位和验证，这要求理解整个系统的运作方式。

**F3（打断协议）的"部分输出保留"是唯一涉及设计决策的 ambiguity：** OpenAI spec 未明确定义 cancel 时已生成但未 complete 的内容是否应发给客户端。本方案选择"补发"——理由是客户端已有对应的 delta 事件，补发 done 让客户端能正确关闭 content item 状态。

### 1.3 代码量

| 模块 | 核心 | 测试 |
|---|---|---|
| F1 音频输出 | ~50 行 | ~60 行 |
| F2 VAD + 状态一致性 | ~50 行 | ~80 行 |
| F3 打断协议 | ~100 行 | ~80 行 |
| F4 Session 生命周期 | ~50 行 | ~40 行 |
| F5 对话裁剪 | ~60 行 | ~50 行 |
| F6 Error 协议 | ~60 行 | ~50 行 |
| F7 可观测性 | ~30 行 | ~30 行 |
| F8 集成测试 + 压测 | — | ~260 行 |
| 文档 | ~50 行 | — |
| **合计** | **~450 行** | **~650 行** |

---

## 2. 现状

### 2.1 已有基础设施

WebSocket `/v1/realtime`、PCM16 buffer + base64 解码、Silero ONNX VAD（逐帧检测 speech_started/stopped）、`CompletionStreamChunk(modality="audio", audio_b64)`、abort 广播链（Coordinator → ZMQ → Stage → Scheduler → OmniScheduler 标记 + batch 边界跳过）、Thinker→Talker→Code2Wav pipeline（`/v1/audio/speech` 已验证）。

### 2.2 链路验证

`session.py:build_response_request` 的 `output_modalities` 从 `["text"]` 改为 `["text", "audio"]` 后：

```
build_response_request() → Client._build_omni_request()
  → metadata["output_modalities"] = ["text", "audio"]
  → should_generate_audio_output() → True → resolve → [thinker, talker]
  → Talker → Code2WavScheduler(stream_chunk_size=10) → PCM → base64
  → CompletionStreamChunk(modality="audio", audio_b64)
  → session.py: async for chunk in completion_stream()
```

`should_generate_audio_output()` 是运行时开关。`/v1/audio/speech` 已验证同一条链路产出有效音频。

### 2.3 事件覆盖

已发送 11 种。缺失：`response.audio.delta`、`response.audio.done`、`response.done(status=cancelled)`、`error`。

### 2.4 Turn 模型

```
PCM → vad.process() → speech_started → recording → speech_stopped
  → auto_commit_utterance() → response_queue(FIFO)
  → drain_queue() → run_turn() → run_response() + run_transcription()
```

active turn 期间新 speech 排队，不打断。

---

## 3. Feature 1：音频输出（12–18h）

三处改动。核心代码短，耗时在实际调试。

```python
# ① __init__ — 默认策略需谨慎
# 方案 A：保守默认 text，允许 session.update(modalities=["text", "audio"]) 开启
# 方案 B：仅在确认 pipeline 支持 talker/code2wav 时默认 text+audio
self.session_object = SessionObject(modalities=["text"], ...)

# 后续若产品决定默认语音输出，再切换为：
# self.session_object = SessionObject(modalities=["text", "audio"], ...)

# ② build_response_request — 透传而非写死
output_modalities=self.session_object.modalities

# ③ run_response — 消费 audio chunk，构建完整 output array
# 累加器设计为实例属性（不是局部变量），为 Phase 3 预留
self._partial_text: list[str] = []
self._partial_audio: list[str] = []
self._partial_response_id: str | None = None
self._partial_item_id: str | None = None

async for chunk in self.client.completion_stream(...):
    if chunk.modality == "text" and chunk.text:
        self._partial_text.append(chunk.text)
        await self.send(make_event("response.text.delta", ...))
    if chunk.modality == "audio" and chunk.audio_b64:
        self._partial_audio.append(chunk.audio_b64)
        await self.send(make_event("response.audio.delta", ...))

# 正常路径：发送 done 事件 + response.done(completed)
# response.done 的 output array 维持一个 assistant message item；
# content 数组按 modalities 顺序包含 text/audio content part。
# 例如 modalities=["text", "audio"] 时 text content_index=0，audio content_index=1。
```

**耗时分布：** 理解 `completion_stream()` 的 async generator 行为 + 四跳链路追踪（5-8h）；`response.done` output array 与 delta 事件的 content_index 对齐（2-3h）；端到端验证（3-5h）；搭 session mock 测试框架（2-3h，为后续 Phase 铺路）。

**伴随测试（Phase 1 交付）：** mock `completion_stream` 返回固定 text + audio chunk → 验证事件类型序列和 content_index 正确。

**验收标准：** `wscat` 连接 `/v1/realtime`，发送 3 秒 PCM16 音频，并通过 `session.update(modalities=["text", "audio"])` 或服务端默认语音配置开启音频输出 → 收到 `response.audio.delta` 事件 → base64 解码后可播放（实际编码由 `Client.completion_stream(audio_format=...)` 决定，默认 WAV）。

---

## 4. Feature 2：VAD 配置 + 状态一致性（12–18h）

### 4.1 热更新

```python
# vad.py
def update_config(self, config: VADConfig) -> None:
    self.config = config
    self.silence_run_samples = 0  # 重置：保守策略，宁延迟不误判

# session.py:handle_session_update 末尾
if candidate.turn_detection is not None:
    self.vad.update_config(VADConfig(...))
```

### 4.2 多 Turn 状态一致性

两个问题：① `speech_stopped` → `drop_buffer_and_reset_vad()` 之间的窗口期新 PCM 被误清；② abort 路径若跳过 `drop_buffer_and_reset_vad()`，`buffer_origin_samples` 未推进 → 下一 turn 时间戳偏移。

**解决方向：** 统一清理语义，但不把所有 cleanup 混成一个无条件清 buffer 的函数。正常 commit 清 input buffer；response cancel 清 response partial/active request state；只有明确 turn boundary 才 reset VAD：

```python
async def _cleanup_committed_input(self) -> None:
    self.buffer_origin_samples += self.audio_buffer.num_samples
    self.audio_buffer.clear()
    self.utterance_start_byte = None
    self.utterance_item_id = None
    self.vad.reset()

async def _cleanup_response_state(self) -> None:
    self.active_response_request_id = None
    self.active_response_id = None
    self._partial_text.clear()
    self._partial_audio.clear()
```

### 4.3 中断策略

`handle_vad_emit` 中 `SPEECH_STARTED` 分支：若 `active_task` 运行中且 `turn_detection` 不为 None，触发 `handle_response_cancel()` 打断当前 turn。当前默认保守（排队），后续可配置化。

**伴随测试（Phase 2 交付）：** 参数化 VAD 状态转移表（已有 `emits_for_test` 助手）；100 个连续 turn 后验证 `audio_start_ms` 偏移 < 10ms；config 变更 + 已累积 silences 场景的确定性验证。

**验收标准：** `session.update(turn_detection={threshold: 0.8})` → 后续 VAD 检测使用新阈值；100 turn 循环后时间戳偏移 < 10ms。

---

## 5. Feature 3：打断协议（20–30h）

唯一有设计 ambiguity 的功能。

### 5.1 核心决策：cancel 时是否补发部分输出

**问题：** cancel 到达时 async generator 可能已 yield N 个 delta。客户端已收到这些 delta，但没收到对应的 `*.done` 事件。应该补发吗？

**决策：补发。** 理由：OpenAI Realtime 的事件模型是"delta → done"配对——每个 content item 的完整语义由这两个事件共同构成。客户端 SDK 在收到 delta 后等待 done 来关闭 item 状态。不补发会导致客户端状态机卡住。这与其他 streaming 协议（SSE、gRPC stream）的"cancel 即丢弃"惯例不同——因为这里的 delta 已经在客户端了，不能假装没发过。

### 5.2 实现

累加器已由 Phase 1 设计为实例属性（`_partial_text`、`_partial_audio`、`_partial_response_id`、`_partial_item_id`），Phase 3 只加 cancel 路径：

```python
async def handle_response_cancel(self, event: ResponseCancel) -> None:
    # 详细实现见 realtime_api_detailed_design.md：
    # - 使用 ResponseState 保存 partial text/audio
    # - 拆分 active_response_request_id / active_transcription_request_id
    # - response.cancel 只取消 assistant response
    # - cancel 不清 input audio buffer
    # - 先 emit partial text/audio done，再 emit response.done(cancelled)
    await self._cancel_active_response(reason="cancelled")
```

### 5.3 真正的难点：CancelledError 时机与累加器一致性

```
run_response() 执行序列：
  for chunk in generator:       ← 1. yield chunk（text delta 已发送）
    text_acc.append(chunk.text) ← 2. 累加
                                ← CancelledError 可能在 1 和 2 之间抛出！
```

如果 cancel 恰好在 delta 已发送但未累加时到达，`_emit_partial_output()` 发出的 `text.done` 会少最后一条。当前 Python 的 async generator 在每次 `await` 时检查 cancel——而 `self.send(make_event(...))` 是 await 调用。所以实际危险区域在 `send` 调用前后。**缓解：** 在 `run_response()` 中用 `try/finally` 确保即使 cancel，最后一个已发送 chunk 也被累加。

### 5.4 中断策略联动

Feature 2 的 `SPEECH_STARTED` 分支调用 `handle_response_cancel()`，cancel 完成后 drain_queue 立即处理新 turn。

### 5.5 Request 生命周期补充

当前 `active_request_id` 同时被 response pass 和 transcription pass 使用。实现 cancel 前必须拆分为 `active_response_request_id` 与 `active_transcription_request_id`，并明确 `response.cancel` 默认只取消 assistant response；transcription 是否继续写入 history 作为独立策略处理。否则 cancel 可能误 abort transcription，或在 transcription 阶段收到 cancel 时发出错误的 `response.done(status="cancelled")`。

### 5.6 Partial output 事件结构

cancel 前已经发出的 delta 必须被对应 done 闭合，但 `response.done(status="cancelled")` 的 `output` 是否携带 partial content 需要保持一致。详细设计选择：

1. 先发送 `response.text.done` / `response.audio.done` 闭合已发送 content part。
2. 再发送 `response.done(status="cancelled")`。
3. `response.done.response.output` 携带同一个 assistant message item 的 partial content，便于无状态客户端一次性恢复最终 cancelled response；若后续与 OpenAI SDK 兼容性测试发现冲突，再切换为 output 为空。

**伴随测试（Phase 3 交付）：** mock generator 在 yield N 个 chunk 后抛出 `CancelledError` → 验证 `_emit_partial_output` 发出 N 个 done + cancel 事件。100 次连续 cancel 循环 → `torch.cuda.memory_summary()` 前后差值 < 50MB。

**验收标准：** cancel 发送 → < 200ms 内收到 `response.done(status=cancelled)`；如果 cancel 前已有 delta，先收到 `*.done`（partial）再收到 cancel；100 次 cancel 后显存差值 < 50MB。

---

## 6. Feature 4–6：基础设施（合并，共 25–35h）

三个关联较紧的功能合并实施。

### 6.1 Session 生命周期（8–12h）

Idle timeout（默认 5min，30s 周期扫描 `_last_event_time`）、max sessions 限制（超限拒绝 WebSocket 握手）、graceful shutdown（遍历关闭，单 session 超时 5s 强制断开）。

### 6.2 对话裁剪（10–15h）

`_trim_conversation(max_tokens=8000)` —— 从最新到最旧累计 token 数，超出阈值时丢弃旧项。集成 tokenizer 做精确计数，不可用时回退字符数估算。至少保留 1 轮（2 条）。在 `run_turn()` 末尾调用。

Token 计数访问路径：`self.client.tokenizer`（需要在 Client 类加一个 2 行的 `@property`，是独立于 pipeline 的公共 API，不违反 scope 约束）。

### 6.3 Error 协议（8–12h）

5 条路径：畸形 base64 → `error(type="invalid_request_error")`；buffer 超限 → error + auto clear + `buffer.cleared` 通知；推理错误 → `error(type="server_error")` → session 保持；`session.update` 非法参数 → error；所有 error 路径后 session 需保持可用。

**伴随测试（Phase 4-6 交付）：** 裁剪逻辑参数化（1/10/50/100 轮对话 → 验证裁剪后 token 数 ≤ max_tokens）；超时清扫单元测试（mock 时间）；5 种 error 场景 → 验证 error 事件格式 + session 后续可用。

**验收标准：** session idle 5min → 自动关闭，GPU 显存释放；50 轮对话后 conversation tokens ≤ 8000；畸形 base64 → error 事件 + session 仍可处理后续正常音频。

---

## 7. Feature 7：可观测性（8–12h）

使用 `prometheus-client` 库（Python 生态标准，但当前 `pyproject.toml` 未声明该依赖；若采用需新增依赖，或先落地内置轻量 counters 后续再接 Prometheus）。6 个指标：

- `realtime_sessions_active`（gauge）
- `realtime_turns_total{status}`（counter）
- `realtime_vad_events_total{type}`（counter）
- `realtime_audio_chunks_total`（counter）
- `realtime_turn_duration_seconds`（histogram）
- `realtime_conversation_trims_total`（counter）

复用 `/metrics` 端点（或新增）。

---

## 8. Feature 8：集成测试 + 压测（12–18h）

| 测试 | 内容 |
|---|---|
| `tests/unit_test/realtime/test_vad.py` | 参数化 VAD 状态转移表（已有 `emits_for_test`） |
| `tests/unit_test/realtime/test_session.py` | mock client → 验证事件序列、cancel、error、trim |
| `tests/test_model/test_qwen3_omni_realtime.py` | 复用现有真实 pipeline 单 session 完整对话，并扩展 audio/cancel 场景 |
| `benchmarks/realtime/bench.py` | TTFT/TTFA/E2E/abort 延迟；1-16 并发 sweep |
| `benchmarks/realtime/concurrency.py` | 16 路 10 turn 循环 → p50/p95/p99 + GPU 显存曲线 |

---

## 9. 实施计划

```
Phase 0:  环境搭建                4h     GPU + text-only + speech 链路验证
Phase 1:  音频输出               12–18h     实例属性设计 + audio delta + mock 测试
Phase 2:  VAD + 状态一致性       12–18h     update_config + _cleanup_turn + 参数化测试
Phase 3:  打断协议               20–30h     cancel 决策 + 部分输出 + 100 次循环验证
Phase 4:  基础设施（生命周期）     8–12h     idle timeout + max sessions + shutdown
Phase 5:  基础设施（裁剪）        10–15h     token-aware trim + 边界测试
Phase 6:  基础设施（Error）        8–12h     5 路径 + session 恢复测试
Phase 7:  可观测性                8–12h     prometheus-client + /metrics
Phase 8:  集成测试 + 压测         12–18h     3 类测试 + 2 个 benchmark
Phase 9:  收尾                    5–10h     README + 冒烟测试
─────────────────────────────────────────
合计                            107–175h    ~450 行核心 + ~650 行测试
```

P50: **~145h。** 测试不再集中在 Phase 8，而是每个 Feature 交付时带测试。

---

## 10. 风险

| 风险 | 概率 | 缓解 |
|---|---|---|
| GPU 环境不可用 | 高 | 提前确认；Mac 本地无法开发 |
| 四跳链路调试超预期 | 中 | `/v1/audio/speech` 已验证同链路；Phase 1 预留 5-8h |
| CancelledError 时机导致累加器丢最后一条 | 中 | `run_response()` 加 try/finally；Phase 3 测试覆盖 |
| VAD 配置切换触发误判 | 中 | `silence_run_samples=0` 保守策略；参数化测试 |
| cancel 后显存残留 | 中 | 拆分 response/transcription request id；response cancel 只 abort response request；100 次循环验证 |
| 16 路并发 GPU OOM | 中 | max_sessions 兜底；压测确认实际上限 |
| 上游版本升级破坏 realtime | 中 | 锁定依赖版本 |

---

## 11. 设计原则

1. **Scope 限定但允许薄接入。** 主体改动在 `sglang_omni/serve/realtime/` + 测试目录；允许 `openai_api.py` 做握手/生命周期/metrics 接入，允许 `Client` 暴露 tokenizer 等只读公共 API。不修改 pipeline/scheduler/relay 核心执行逻辑。
2. **cancel 阻止下一轮 kernel launch，不中断当前 CUDA kernel。**
3. **VAD 保守重置。** 宁延迟一个窗口，不产生虚假 speech_stopped。
4. **Session always recoverable。** 任一 error 后 session 可用（WebSocket 断开除外）。
5. **每个 Phase 带测试交付。** 不等 Phase 8。
