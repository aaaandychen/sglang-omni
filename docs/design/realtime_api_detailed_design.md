# Realtime API 详细设计

> 本文是 `realtime_api_completion.md` 的落地设计补充。前者描述功能路线图，本文约束具体状态机、事件 schema、任务/request 生命周期、buffer 生命周期、错误边界、测试分层与实施顺序。

---

## 1. 当前代码事实

### 1.1 Realtime 入口与会话

- WebSocket 入口：`sglang_omni/serve/openai_api.py::_register_realtime()`。
- Session manager：`sglang_omni/serve/realtime/manager.py::RealtimeSessionManager`。
- Session 主体：`sglang_omni/serve/realtime/session.py::RealtimeSession`。
- 事件模型：`sglang_omni/serve/realtime/events.py`。
- VAD：`sglang_omni/serve/realtime/vad.py::StreamingVAD`。
- 音频输入 buffer：`sglang_omni/serve/realtime/audio_buffer.py::RealtimeAudioBuffer`。

当前 `RealtimeSession` 的 per-turn 逻辑：

```text
WebSocket receive
  -> input_audio_buffer.append
  -> audio_buffer.append_b64()
  -> vad.process()
  -> speech_started / speech_stopped
  -> auto_commit_utterance()
  -> response_queue.put((item_id, wav_data_uri))
  -> drain_queue()
  -> run_turn()
     -> run_response()
     -> run_transcription()
     -> append conversation history
```

关键限制：

- `SessionObject.modalities` 默认 `['text']`。
- `build_response_request()` 写死 `output_modalities=['text']`。
- `handle_response_cancel()` 只 abort + cancel task，不 emit protocol completion/cancel events。
- `active_request_id` 同时被 response pass 和 transcription pass 复用。
- `send_error()` 已存在，但 parse/validation/handler/generation 异常都没有统一接入。

### 1.2 Client 与音频输出链路

`Client.completion_stream()` 已能把 `GenerateChunk(modality='audio', audio_data=...)` 转成 `CompletionStreamChunk(modality='audio', audio_b64=...)`。

`GenerateRequest.output_modalities` 会在 `Client._build_omni_request()` 中写入 `OmniRequest.metadata['output_modalities']`。

Qwen3-Omni 的 `should_generate_audio_output()` 语义是：

```text
metadata.output_modalities is None -> 生成 audio
metadata.output_modalities contains 'audio' -> 生成 audio
otherwise -> text only
```

因此 Realtime 只要正确透传 `session_object.modalities`，pipeline 层理论上即可打开 thinker+talker/code2wav 音频链路。

### 1.3 Abort 语义

`Client.abort()` 调用 `Coordinator.abort()`。Coordinator 行为：

1. 若 request 不存在或已完成，返回 `False`。
2. 广播 abort 给所有 stage。
3. 将 request state 置为 `ABORTED`。
4. 对 streaming queue 注入 failed `CompleteMessage(error='aborted')`。
5. 清理 request tracking。

注意：abort 不能中断已经运行中的 CUDA kernel，只能阻止后续 stage/后续 batch boundary 的继续执行。因此 Realtime 层必须把 cancel 设计为“尽快停止流式输出和后续 kernel launch”，不能承诺硬中断当前 kernel。

---

## 2. 目标与非目标

### 2.1 目标

1. 支持 `response.audio.delta` / `response.audio.done`。
2. `session.update` 能更新 modalities、sampling、server VAD 参数，并对非法字段返回 `error`。
3. `response.cancel` 具有完整协议闭合：partial content done + `response.done(status='cancelled')`。
4. VAD barge-in 可打断当前 assistant response，但不误清新 speech input buffer。
5. Session 有 idle timeout、max sessions、graceful shutdown。
6. Conversation history 有 bounded trim。
7. 所有可恢复错误走 `error` 事件，session 保持可用。
8. 建立轻量 mock 测试体系，真实 GPU 集成测试只覆盖端到端烟测和性能。

### 2.2 非目标

1. 不改 pipeline/scheduler/relay 核心执行语义。
2. 不保证 cancel 可硬中断当前 CUDA kernel。
3. 不在第一版支持 `semantic_vad`。
4. 不在第一版支持 `input_audio_format` 除 `pcm16` 以外的格式。
5. 不在第一版实现完整 OpenAI Realtime 所有事件类型，只补齐当前链路必须事件。

---

## 3. Response 事件 schema

### 3.1 content index 规则

Realtime response 使用一个 assistant message item：

```json
{
  "id": "item_xxx",
  "object": "realtime.item",
  "type": "message",
  "role": "assistant",
  "content": [ ... ]
}
```

`content_index` 按最终启用的 response modalities 顺序分配：

| modalities | text content_index | audio content_index |
|---|---:|---:|
| `["text"]` | 0 | N/A |
| `["audio"]` | N/A | 0 |
| `["text", "audio"]` | 0 | 1 |
| `["audio", "text"]` | 1 | 0 |

实现建议：每个 response 创建时生成 `content_indexes: dict[str, int]`，不要在发送事件处硬编码。

### 3.2 `response.created`

```json
{
  "type": "response.created",
  "response": {
    "id": "resp_xxx",
    "object": "realtime.response",
    "status": "in_progress",
    "output": []
  }
}
```

### 3.3 `response.text.delta`

仅当 `text` modality 被启用且 chunk 有文本时发送：

```json
{
  "type": "response.text.delta",
  "response_id": "resp_xxx",
  "item_id": "item_xxx",
  "output_index": 0,
  "content_index": 0,
  "delta": "hello"
}
```

### 3.4 `response.audio.delta`

仅当 `audio` modality 被启用且 chunk 有 `audio_b64` 时发送：

```json
{
  "type": "response.audio.delta",
  "response_id": "resp_xxx",
  "item_id": "item_xxx",
  "output_index": 0,
  "content_index": 1,
  "delta": "base64-wav-or-configured-audio-chunk"
}
```

第一版沿用 `Client.completion_stream(audio_format='wav')` 默认编码。若后续要降低延迟，可将 Realtime audio format 独立配置为 PCM chunk，但那是第二阶段优化。

### 3.5 `response.text.done`

```json
{
  "type": "response.text.done",
  "response_id": "resp_xxx",
  "item_id": "item_xxx",
  "output_index": 0,
  "content_index": 0,
  "text": "complete or partial text"
}
```

发送条件：

- text modality 启用；并且
- 已经至少发送过 text delta，或正常完成时希望显式闭合空文本 content。

第一版建议：只要 text modality 启用，正常完成总是发送；cancel 时仅在已发送过 text delta 后发送。

### 3.6 `response.audio.done`

```json
{
  "type": "response.audio.done",
  "response_id": "resp_xxx",
  "item_id": "item_xxx",
  "output_index": 0,
  "content_index": 1
}
```

第一版不在 `audio.done` 里重复完整 audio，完整 audio 通过 `response.done.response.output[*].content[*].audio` 汇总。若客户端 SDK 要求 `audio.done` 携带字段，再按兼容性测试调整。

### 3.7 `response.done(status='completed')`

```json
{
  "type": "response.done",
  "response": {
    "id": "resp_xxx",
    "object": "realtime.response",
    "status": "completed",
    "status_details": {"reason": "stop"},
    "output": [
      {
        "id": "item_xxx",
        "object": "realtime.item",
        "type": "message",
        "role": "assistant",
        "content": [
          {"type": "text", "text": "..."},
          {"type": "audio", "audio": "concatenated-base64"}
        ]
      }
    ],
    "usage": null
  }
}
```

`content` 只包含启用且实际生成的 modalities。audio 汇总策略第一版使用字符串拼接 `audio_b64`；注意这只在每个 delta 都是可拼接的 raw base64 PCM 时严格正确。当前 `Client.completion_stream(audio_format='wav')` 每个 chunk 可能都是独立 WAV 编码，直接拼接不一定得到合法 WAV。因此第一版有两个可选策略：

- **策略 A（推荐先落地）：** `response.done` 的 audio content 仅放空字符串或省略完整 audio，只依赖 delta；`audio.done` 表示流结束。
- **策略 B（需要 client 支持）：** Realtime 调用 `completion_stream(audio_format='pcm')` 或新增 raw PCM chunk 编码，保证可拼接。

详细实现前必须确认 `audio_to_base64(..., output_format='wav')` 的 chunk 是否为独立容器。若是独立 WAV，不能把多个 base64 WAV 直接拼成一个完整 WAV。

### 3.8 `response.done(status='cancelled')`

cancel 路径事件顺序：

```text
response.text.done?   # 若已发 text delta
response.audio.done?  # 若已发 audio delta
response.done(status='cancelled')
```

`response.done` 示例：

```json
{
  "type": "response.done",
  "response": {
    "id": "resp_xxx",
    "object": "realtime.response",
    "status": "cancelled",
    "status_details": {"reason": "cancelled"},
    "output": [
      {
        "id": "item_xxx",
        "object": "realtime.item",
        "type": "message",
        "role": "assistant",
        "content": [
          {"type": "text", "text": "partial text"}
        ]
      }
    ],
    "usage": null
  }
}
```

如果 SDK 兼容性测试发现 cancelled response 不应带 output，则降级为空 output，但 delta/done 闭合仍保留。

---

## 4. Response state machine

### 4.1 状态定义

```text
IDLE
  -> CREATING
  -> STREAMING
  -> COMPLETING
  -> COMPLETED

STREAMING
  -> CANCELLING
  -> CANCELLED

STREAMING / COMPLETING
  -> FAILED_RECOVERABLE
  -> IDLE
```

### 4.2 状态与允许事件

| 状态 | 允许动作 | 禁止/特殊处理 |
|---|---|---|
| `IDLE` | 开始新 response | `response.cancel` no-op 或 error（二选一，第一版 no-op） |
| `CREATING` | 发送 `response.created` | cancel 需要等待 `response_id/item_id` 初始化后闭合 |
| `STREAMING` | 发送 text/audio delta | cancel 进入 `CANCELLING` |
| `COMPLETING` | 发送 text/audio done + response.done | cancel 视为 no-op，避免重复 done |
| `CANCELLING` | abort engine，cancel task，emit partial done/cancelled | 禁止重复 cancel |
| `COMPLETED` | 清 response state | N/A |
| `FAILED_RECOVERABLE` | send error，清 response state | session 继续可用 |

### 4.3 实现状态对象

建议新增轻量 dataclass：

```python
@dataclass
class ResponseState:
    response_id: str
    item_id: str
    request_id: str
    modalities: list[str]
    content_indexes: dict[str, int]
    text_parts: list[str] = field(default_factory=list)
    audio_parts: list[str] = field(default_factory=list)
    text_delta_sent: bool = False
    audio_delta_sent: bool = False
    text_done_sent: bool = False
    audio_done_sent: bool = False
    response_done_sent: bool = False
    status: Literal['creating', 'streaming', 'completing', 'cancelling'] = 'creating'
```

`RealtimeSession` 只保留：

```python
self.active_response: ResponseState | None
self.active_response_task: asyncio.Task | None
self.active_transcription_task: asyncio.Task | None
self.active_turn_task: asyncio.Task | None
```

不要再用一个 `active_request_id` 表示所有 pass。

---

## 5. Task / request 生命周期

### 5.1 拆分 active ids

当前 `active_request_id` 同时覆盖 response 与 transcription。需要改为：

```python
self.active_response_request_id: str | None = None
self.active_transcription_request_id: str | None = None
```

### 5.2 `run_turn()` 推荐结构

第一版保守结构：response 和 transcription 仍顺序执行，但 id 分离。

```text
run_turn(item_id, audio_payload)
  -> run_response(audio_payload)
     active_response_request_id = rt-...-resp
  -> run_transcription(item_id, audio_payload)
     active_transcription_request_id = rt-...-tr
  -> append conversation
  -> trim conversation
```

这样改动小，兼容当前“用户先看到回复，再补 transcription”的设计。

第二版可优化为：response 与 transcription 并发执行，response 先流式给用户，transcription 后台补 history。但这会增加 cancel/history 一致性复杂度，不建议第一版做。

### 5.3 `response.cancel` 语义

第一版定义：

- 只取消 assistant response。
- 如果 response 已完成、正在 transcription，则 no-op。
- cancel 不清 input audio buffer。
- cancel 不主动取消 queued future turns。
- VAD barge-in 触发的 cancel 完成后，新 utterance 继续按 VAD buffer 正常 commit。

伪代码：

```python
async def handle_response_cancel(self, event):
    if self.active_response is None:
        return
    if self.active_response.status in {'completing'}:
        return
    await self._cancel_active_response(reason='cancelled')
```

`_cancel_active_response()`：

1. 标记 response state `cancelling`，防重复。
2. 若 `active_response_request_id` 存在，调用 `client.abort()`。
3. cancel `active_response_task` 或 `active_turn_task` 中的 response 子任务。
4. absorb `CancelledError`。
5. emit partial done。
6. emit `response.done(status='cancelled')`。
7. 清 response state。

### 5.4 是否保留 transcription

当前顺序结构下，response cancel 会取消整个 `run_turn()` 时，transcription 尚未开始，用户本轮 spoken audio 不会进 history。第一版可接受，但要明确：

- 显式 `response.cancel`：取消 assistant response 后，不再转写该 turn，不追加 assistant history。
- VAD barge-in：被打断 turn 不追加 assistant history；新 turn 重新开始。

如果产品要求保留被打断前用户输入 transcript，需要将 transcription 与 response 解耦，这是第二版。

---

## 6. Input audio buffer 与 VAD 生命周期

### 6.1 当前风险

当前 `auto_commit_utterance()` 在生成 wav payload 后立即 `drop_buffer_and_reset_vad()`。这对单 turn 正常路径可以工作，但 barge-in/cancel 后如果复用无条件 cleanup，容易清掉新 speech 的前缀。

### 6.2 Cleanup 分类

必须拆分：

#### 6.2.1 commit cleanup

仅在 `speech_stopped` 并成功切出 payload 后调用：

```python
def _cleanup_committed_input(self):
    self.buffer_origin_samples += self.audio_buffer.num_samples
    self.audio_buffer.clear()
    self.utterance_start_byte = None
    self.utterance_item_id = None
    self.vad.reset()
```

#### 6.2.2 response cleanup

仅清 response state，不动 input buffer：

```python
def _cleanup_response_state(self):
    self.active_response = None
    self.active_response_request_id = None
```

#### 6.2.3 session teardown cleanup

WebSocket 断开/manager shutdown 才可以 cancel tasks 并关闭 socket。

### 6.3 VAD hot update

新增：

```python
def update_config(self, config: VADConfig) -> None:
    self.config = config
    self.silence_run_samples = 0
```

是否 reset `leftover_pcm` / `samples_consumed`：

- 配置热更新时不 reset `samples_consumed`，否则 timestamp 会跳。
- 不清 `leftover_pcm`，避免丢正在输入的半帧。
- 重置 `silence_run_samples`，避免旧 silence 阈值残留导致立刻 speech_stopped。
- 若 threshold 改动非常大，当前 frame 后状态自然收敛。

### 6.4 Barge-in 策略

第一版：

```text
on SPEECH_STARTED:
  emit input_audio_buffer.speech_started
  if active_response exists and active_response.status == streaming:
      cancel active response
```

但为了降低误触发，建议加两个保护：

1. 只在已经发送过至少一个 response delta 后触发 cancel。
2. 若 `turn_detection` 为 `None`，不启用自动 VAD 与 barge-in。

后续可加配置字段：`interrupt_response: bool = True`，但当前事件模型没有该字段，第一版可内部默认 true。

---

## 7. `session.update` 校验矩阵

当前代码用 `assert` 校验 `input_audio_format`，不适合协议层。应改为显式 validation + error event。

| 字段 | 第一版支持 | 非法处理 |
|---|---|---|
| `modalities` | `['text']`, `['audio']`, `['text','audio']`, `['audio','text']` | `error(type='invalid_request_error', code='invalid_modalities')` |
| `instructions` | string | 非 string 由 Pydantic 捕获并转 error |
| `input_audio_format` | `pcm16` | `error(code='unsupported_audio_format')` |
| `turn_detection` | object or `null` | unsupported type error |
| `turn_detection.type` | `server_vad` | `semantic_vad` 返回 unsupported error |
| `threshold` | `0.0 <= x <= 1.0` | `error(code='invalid_vad_threshold')` |
| `prefix_padding_ms` | `>= 0` | `error(code='invalid_vad_prefix_padding_ms')` |
| `silence_duration_ms` | `> 0` | `error(code='invalid_vad_silence_duration_ms')` |
| `temperature` | 建议 `0.0 <= x <= 2.0` | `error(code='invalid_temperature')` |
| `max_response_output_tokens` | positive int or `'inf'` | `error(code='invalid_max_response_output_tokens')` |

`turn_detection=null` 语义：

- 禁用 server-side VAD auto commit。
- 第一版没有 manual commit 事件，因此禁用 VAD 后 session 只接收 audio 但不会自动生成 response；建议返回 `unsupported_turn_detection`，除非同时实现 manual commit。
- 因当前 `CLIENT_EVENT_TYPES` 没有 `input_audio_buffer.commit` / `response.create`，第一版建议不允许 `turn_detection=null`，返回 error，避免 session 进入不可用交互状态。

---

## 8. Error 边界设计

### 8.1 可恢复错误类型

新增内部异常：

```python
class RealtimeProtocolError(Exception):
    def __init__(self, code: str, message: str, type_: str = 'invalid_request_error'):
        ...
```

### 8.2 `run()` loop 必须 catch recoverable errors

当前 `json.loads()`、`parse_client_event()`、Pydantic validation、base64 decode、buffer overflow、handler assert 都可能打断 WebSocket loop。应在每条 message 级别包住：

```python
while not self.closed:
    message = await receive()
    try:
        payload = parse json
        await self.dispatch(payload)
    except RealtimeProtocolError as exc:
        await self.send_error(exc.type_, exc.code, exc.message)
    except ValidationError as exc:
        await self.send_error('invalid_request_error', 'invalid_event', str(exc))
    except Exception as exc:
        logger.exception(...)
        await self.send_error('server_error', 'internal_error', 'Internal server error')
```

WebSocket disconnect、send failure、session teardown 不应伪装为 recoverable error。

### 8.3 base64 和 buffer overflow

`audio_buffer.append_b64()` 当前使用 `base64.b64decode(validate=False)`。为协议严格性，建议改为 `validate=True`，并捕获 `binascii.Error` 转为：

```json
{
  "type": "error",
  "error": {
    "type": "invalid_request_error",
    "code": "invalid_audio_base64",
    "message": "input_audio_buffer.append.audio must be valid base64"
  }
}
```

Buffer overflow 行为：

1. send error `audio_buffer_overflow`。
2. clear buffer。
3. send `input_audio_buffer.cleared`。
4. session 保持可用。

---

## 9. Conversation trim

### 9.1 预算定义

`conversation_history_max_tokens=8000` 表示历史文本预算，不等于完整 prompt 上限。system instructions、当前 turn audio、固定 user prompt 都需要额外 headroom。

### 9.2 计数策略

优先级：

1. 若 `Client` 暴露 tokenizer：使用 tokenizer encode 计数。
2. 否则字符估算：英文/符号约 `len(text) // 4`，中文约 `len(text)`，第一版可用保守 `ceil(len(text) / 2)` 或 `len(text)`。

### 9.3 裁剪算法

- 从最新到最旧保留。
- 至少保留最近一轮 user+assistant，如果存在。
- 裁剪只发生在 `run_turn()` 成功追加 history 后。
- cancel/failed response 不追加 assistant history。

伪代码：

```python
def _trim_conversation(self):
    kept = []
    total = 0
    for item in reversed(self.conversation):
        cost = count(item.text)
        if kept and total + cost > max_tokens and len(kept) >= min_items:
            break
        kept.append(item)
        total += cost
    self.conversation = list(reversed(kept))
```

---

## 10. Session lifecycle

### 10.1 Manager state

`RealtimeSessionManager` 增加：

```python
max_sessions: int
idle_timeout_s: float
idle_scan_interval_s: float
_scanner_task: asyncio.Task | None
```

### 10.2 max sessions

严格 handshake reject 需要在 `openai_api.py` 中 `websocket.accept()` 前判断：

```python
if not manager.can_open():
    await websocket.close(code=1013, reason='Too many realtime sessions')
    return
await websocket.accept()
```

如果 FastAPI/Starlette 要求 accept 前 close 行为不稳定，可 accept 后立即 close，但文档要承认这是 accept-then-close。

### 10.3 idle timeout

`RealtimeSession` 增加 `_last_event_time`，每次收到合法客户端 event 后更新。manager 定期扫描：

```text
now - session.last_event_time > idle_timeout_s -> session.teardown(code=1000, reason='idle timeout')
```

生成中的 response 是否算 idle：第一版只看 client input，长时间生成不应被 idle timeout 断开；因此 session 需要 `is_busy()`：active response/transcription/turn 时不 idle close。

### 10.4 graceful shutdown

`openai_api.py` 注册 shutdown hook：

```python
@app.on_event('shutdown')
async def shutdown_realtime():
    await manager.shutdown(timeout_per_session=5)
```

若项目后续改 FastAPI lifespan，再迁移到 lifespan。

---

## 11. Observability

### 11.1 指标

| 指标 | 类型 | 标签 | 触发点 |
|---|---|---|---|
| `realtime_sessions_active` | gauge | none | manager open/close |
| `realtime_turns_total` | counter | `status=completed,cancelled,error` | turn end |
| `realtime_vad_events_total` | counter | `type=speech_started,speech_stopped` | `handle_vad_emit` |
| `realtime_audio_chunks_total` | counter | `direction=input,output` | append/audio delta |
| `realtime_turn_duration_seconds` | histogram | `status` | run_turn duration |
| `realtime_conversation_trims_total` | counter | none | trim occurred |

### 11.2 依赖策略

当前 `pyproject.toml` 未包含 `prometheus-client`。两种方案：

- 方案 A：新增依赖并在 `openai_api.py` 注册 `/metrics`。
- 方案 B：先实现内部 no-op/轻量 metrics facade，后续统一接入项目级 metrics。

第一版推荐方案 B，避免为 realtime 单独引入导出端点带来的部署决策。

---

## 12. 测试设计

### 12.1 单元测试目录

建议新增：

```text
tests/unit_test/realtime/
  test_audio_buffer.py
  test_vad.py
  test_session_events.py
  test_session_cancel.py
  test_session_update.py
  test_conversation_trim.py
  test_manager_lifecycle.py
```

### 12.2 FakeWebSocket

Mock session 测试需要 fake websocket：

```python
class FakeWebSocket:
    application_state = WebSocketState.CONNECTED
    client_state = WebSocketState.CONNECTED
    sent: list[dict]
    recv_queue: asyncio.Queue[dict]
    async def receive(self): ...
    async def send_text(self, text): self.sent.append(json.loads(text))
    async def close(self, *args, **kwargs): ...
```

### 12.3 FakeClient

```python
class FakeClient:
    def __init__(self, chunks): self.chunks = chunks; self.aborted = []
    async def completion_stream(self, request, *, request_id, audio_format='wav'):
        for chunk in self.chunks:
            await asyncio.sleep(0)
            yield chunk
    async def abort(self, request_id):
        self.aborted.append(request_id)
        return AbortResult(success=True, level_applied=AbortLevel.SOFT)
```

### 12.4 必测场景

#### 音频输出

- `modalities=['text']`：只发送 text delta/done。
- `modalities=['audio']`：只发送 audio delta/done。
- `modalities=['text','audio']`：content_index text=0 audio=1。
- `modalities=['audio','text']`：content_index audio=0 text=1。

#### cancel

- 无 active response：no-op。
- text delta 后 cancel：text.done + response.done(cancelled)。
- audio delta 后 cancel：audio.done + response.done(cancelled)。
- text/audio delta 后 cancel：两个 done 顺序稳定。
- cancel 重入：只发一次 cancelled。
- abort 返回 False：仍发 cancelled，但记录日志。

#### error

- invalid JSON。
- unsupported event type。
- invalid base64。
- buffer overflow。
- invalid session.update。
- client.completion_stream 抛异常后 session 仍可继续处理下一条合法消息。

#### VAD

- update_config threshold 生效。
- silence_run_samples reset。
- timestamp 100 turns drift < 10ms。
- barge-in speech_started cancel response，但不清新 input buffer。

#### lifecycle

- max sessions。
- idle close ignores busy session。
- shutdown closes all sessions and absorbs task errors。

### 12.5 真实集成测试

扩展现有 `tests/test_model/test_qwen3_omni_realtime.py`：

- 保留 text-only baseline，防止默认 audio 破坏 thinker-only 环境。
- 新增 audio-capable 环境标记，只有确认 talker/code2wav 可用时跑。
- 新增 cancel latency smoke test，但不要在普通 CI 强制 GPU memory threshold。

---

## 13. 推荐实施顺序

### Phase 0.5：测试支架与 error boundary（新增）

1. FakeWebSocket / FakeClient。
2. `RealtimeProtocolError`。
3. `run()` message-level try/except。
4. invalid JSON / invalid event / invalid base64 测试。

这一步先做，因为它降低后续所有 feature 的调试成本。

### Phase 1：modalities 透传 + audio events

1. `SessionObject.modalities` 保守默认 `['text']`。
2. `session.update(modalities=...)` 校验。
3. `build_response_request()` 使用 session modalities。
4. `run_response()` 支持 text/audio delta/done。
5. mock tests 覆盖 content_index。

### Phase 2：ResponseState + cancel

1. 引入 `ResponseState`。
2. 拆 `active_response_request_id` / `active_transcription_request_id`。
3. 实现 partial done + cancelled response。
4. cancel tests。

### Phase 3：VAD hot update + barge-in

1. `StreamingVAD.update_config()`。
2. `session.update(turn_detection=...)`。
3. speech_started 触发 response cancel。
4. 不清新 input buffer 的测试。

### Phase 4：Conversation trim + lifecycle

1. `_trim_conversation()`。
2. manager max sessions / idle scanner。
3. shutdown hook。

### Phase 5：Observability + benchmarks

1. metrics facade。
2. 可选 Prometheus 接入。
3. latency/concurrency benchmark。

---

## 14. Open questions

1. `response.audio.done` 是否必须携带完整 audio 或 transcript？需要用 OpenAI SDK/客户端做兼容性验证。
2. 当前 `Client.completion_stream(audio_format='wav')` 每个 audio delta 是否是独立 WAV？如果是，`response.done` 不能简单拼接 audio base64。
3. `response.cancel` 后是否必须保留用户 transcript？若必须，需要 response/transcription 并发化或 cancel 后继续 transcription。
4. `turn_detection=null` 是否要支持？如果支持，必须同时实现 manual commit / response.create。
5. 默认是否开启 `['text','audio']` 应该由产品和部署能力决定，不应在 thinker-only 环境下硬切。
