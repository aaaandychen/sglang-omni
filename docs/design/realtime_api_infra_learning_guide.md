# Realtime API AI Infra 学习指南（30 小时）

> 目标读者：已有 SGLang / RL infra / LLM serving 基础，但对 `sglang-omni` 代码库还不熟的人。  
> 目标：用约 30 小时掌握实现 `/v1/realtime` 补全所需的 AI infra 知识：多模态 serving 链路、WebSocket session、streaming 状态机、cancel/abort、VAD turn、测试与可观测性。  
> 使用方式：每章 3–5 小时，按“读代码 → 跑/观察 → 画图/思考 → 小产出”推进。不要只通读，要边执行边记录。

---

## 0. 总览：30 小时路线图

| 章节 | 主题 | 预计时间 | 你要掌握的核心问题 |
|---|---:|---:|---|
| Chapter 1 | 服务入口与 Realtime session 主循环 | 4h | WebSocket event 如何进入 session，session 状态在哪里 |
| Chapter 2 | Client / Coordinator / pipeline streaming 链路 | 5h | Realtime 如何变成模型请求，stream chunk 如何回来 |
| Chapter 3 | 多模态 audio output 链路 | 4h | `output_modalities` 如何触发 talker/code2wav，audio chunk 如何表达 |
| Chapter 4 | VAD、audio buffer 与 turn boundary | 4h | speech_started/stopped、buffer offset、commit 如何工作 |
| Chapter 5 | Response state machine 与 cancel/abort | 5h | cancel 为什么难，如何保证 partial output 和资源清理一致 |
| Chapter 6 | Error recovery、session lifecycle、conversation trim | 4h | 长连接服务如何保持可恢复、可控、不会无限增长 |
| Chapter 7 | 测试、可观测性与工程验收 | 4h | 如何用 mock/integration/benchmark 证明功能正确 |
| **合计** |  | **30h** |  |

建议每章结束后产出一页笔记，格式固定：

```text
本章我理解的链路：
关键状态变量：
失败/取消/边界情况：
需要实现时注意的坑：
我能写出的测试：
```

相关设计文档：

- `docs/design/realtime_api_completion.md`：功能路线图。
- `docs/design/realtime_api_detailed_design.md`：详细设计与实现约束。
- 本文：学习路径。

---

## Chapter 1：服务入口与 Realtime session 主循环（4h）

### 1.1 学习目标

掌握 `/v1/realtime` 的入口、session 创建/关闭、WebSocket receive loop、client event dispatch、server event send 的基本结构。

你需要回答：

1. WebSocket 在哪里注册？
2. 一个 client 连接对应哪个对象？
3. session 的核心状态变量有哪些？
4. 收到 JSON event 后如何分发到 handler？
5. server event 如何补 `event_id` 并发送？
6. 当前错误为什么会导致 session 不可恢复？

### 1.2 代码阅读路径

按顺序阅读：

1. `sglang_omni/serve/openai_api.py`
   - 重点：`create_app()`、`_register_realtime()`。
   - 关注：`await websocket.accept()`、`manager.open(websocket)`、`session.run()`、`manager.close()`。

2. `sglang_omni/serve/realtime/manager.py`
   - 重点：`RealtimeSessionManager.open()`、`close()`、`active_sessions()`。
   - 关注：当前 manager 只是 sessions dict，还没有 max sessions / idle scanner / shutdown。

3. `sglang_omni/serve/realtime/session.py`
   - 重点：`RealtimeSession.__init__()`、`run()`、`dispatch()`、`send()`、`send_error()`、`teardown()`。
   - 关注：
     - `self.session_object`
     - `self.audio_buffer`
     - `self.conversation`
     - `self.active_request_id`
     - `self.active_task`
     - `self.response_queue`
     - `self.queue_drainer`
     - `self.vad`
     - `self.buffer_origin_samples`

4. `sglang_omni/serve/realtime/events.py`
   - 重点：`SessionObject`、`SessionUpdate`、`InputAudioBufferAppend`、`ResponseCancel`、`parse_client_event()`、`make_event()`。

### 1.3 关键代码片段定位

WebSocket 注册：

```text
sglang_omni/serve/openai_api.py::_register_realtime
```

Session 主循环：

```text
sglang_omni/serve/realtime/session.py::RealtimeSession.run
```

事件分发：

```text
sglang_omni/serve/realtime/session.py::RealtimeSession.dispatch
sglang_omni/serve/realtime/session.py::HANDLERS
```

事件发送：

```text
sglang_omni/serve/realtime/session.py::RealtimeSession.send
sglang_omni/serve/realtime/events.py::make_event
```

### 1.4 执行/观察任务

#### 任务 A：静态画图

画出当前连接生命周期：

```text
FastAPI app
  -> _register_realtime
  -> websocket.accept
  -> manager.open
  -> session.run
  -> receive JSON
  -> dispatch handler
  -> send event
  -> websocket.disconnect
  -> manager.close
  -> session.teardown
```

#### 任务 B：观察 session.created

如果你已有服务启动方式，连接 `/v1/realtime`，只观察首个事件：

```text
session.created
```

记录其中：

- `session.id`
- `session.modalities`
- `session.input_audio_format`
- `session.turn_detection`
- `temperature`
- `max_response_output_tokens`

#### 任务 C：构造非法 event

手动发一个不支持的事件，例如：

```json
{"type":"unknown.event"}
```

观察当前行为：

- 是返回 `error`？
- 是 WebSocket 断开？
- server log 是否有 traceback？

这会帮助你理解为什么需要 error boundary。

### 1.5 思考题

1. 为什么 `run()` 里不能让 `json.loads()` / `assert` / Pydantic validation 直接冒泡？
2. `send_error()` 已存在但没有被调用，应该从哪些层 catch error？
3. `manager.close(session.session_id)` 如果 session id 不存在会怎样？是否需要幂等？
4. max sessions 应该在 `websocket.accept()` 前还是后判断？为什么？

### 1.6 本章产出

- 一张 session lifecycle 图。
- 一张 `RealtimeSession` 状态变量表。
- 一段“当前 error 为什么不可恢复”的说明。

---

## Chapter 2：Client / Coordinator / pipeline streaming 链路（5h）

### 2.1 学习目标

掌握 Realtime 的 `run_response()` 如何通过 `Client.completion_stream()` 进入 coordinator/pipeline，以及 stream chunk 如何回到 WebSocket event。

你需要回答：

1. `GenerateRequest` 是什么？
2. `Client.completion_stream()` 和 `Client.generate()` 的关系是什么？
3. `Coordinator.stream()` 如何管理 stream queue？
4. `StreamMessage` 如何变成 `GenerateChunk` / `CompletionStreamChunk`？
5. request id 在哪些层出现？
6. abort 时 coordinator 做了什么？

### 2.2 代码阅读路径

1. `sglang_omni/serve/realtime/session.py`
   - 重点：`run_turn()`、`run_response()`、`run_transcription()`、`build_response_request()`、`build_transcription_request()`。

2. `sglang_omni/client/types.py`
   - 重点：
     - `GenerateRequest`
     - `GenerateChunk`
     - `CompletionStreamChunk`
     - `AbortResult`
     - `UsageInfo`

3. `sglang_omni/client/client.py`
   - 重点：
     - `Client.generate()`
     - `Client.completion_stream()`
     - `Client.abort()`
     - `_build_omni_request()`
     - `_default_stream_builder()`
     - `_set_audio_data()`

4. `sglang_omni/pipeline/coordinator.py`
   - 重点：
     - `Coordinator.stream()`
     - `_submit_request()`
     - `abort()`
     - `_handle_stream()`
     - `_handle_completion()`

5. `sglang_omni/proto.py`
   - 重点：
     - `OmniRequest`
     - `StagePayload`
     - `StreamMessage`
     - `CompleteMessage`
     - `AbortMessage`
     - `RequestInfo`
     - `RequestState`

### 2.3 主链路图

你需要最终能画出：

```text
RealtimeSession.run_response(audio_payload)
  -> build_response_request()
  -> Client.completion_stream(request, request_id)
     -> Client.generate()
        -> Client._build_omni_request()
        -> Coordinator.stream(request_id, omni_request)
           -> _submit_request()
           -> control_plane.submit_to_stage(entry_stage)
           -> stream queue receives StreamMessage / CompleteMessage
        -> _default_stream_builder()
     -> CompletionStreamChunk
  -> response.text.delta / response.done
```

### 2.4 执行/观察任务

#### 任务 A：追踪 request_id

在代码中记录这些 request id 的生成点和生命周期：

- `run_response()` 里的 request id。
- `run_transcription()` 里的 request id。
- `Coordinator._requests[request_id]`。
- `Coordinator._stream_queues[request_id]`。
- `Client.abort(request_id)`。

回答：当前为什么一个 `active_request_id` 不够？

#### 任务 B：阅读 abort 行为

重点看：

```text
sglang_omni/pipeline/coordinator.py::Coordinator.abort
```

记录：

- request 不存在时返回什么？
- abort 后是否向 stream queue 塞消息？
- completion future 如何处理？
- `_requests` 什么时候被 pop？

#### 任务 C：对比 response pass 和 transcription pass

在 `session.py` 中对比：

```text
run_response()
run_transcription()
build_response_request()
build_transcription_request()
```

记录差异：

| 项目 | response pass | transcription pass |
|---|---|---|
| prompt |  |  |
| conversation history |  |  |
| output_modalities |  |  |
| event 类型 |  |  |
| 是否写 conversation |  |  |

### 2.5 思考题

1. 为什么 `response.cancel` 不能复用一个全局 `active_request_id`？
2. 如果 response 已完成、transcription 正在跑，此时收到 `response.cancel` 应该怎样？
3. `Coordinator.abort()` 返回 `False` 时，Realtime 层还要不要发 `response.done(cancelled)`？
4. `Coordinator.stream()` 的 `finally` 会清理哪些结构？这对 cancel 有什么影响？

### 2.6 本章产出

- 一张 request lifecycle 图。
- 一张 response/transcription pass 对比表。
- 一段对 `active_response_request_id` / `active_transcription_request_id` 拆分必要性的说明。

---

## Chapter 3：多模态 audio output 链路（4h）

### 3.1 学习目标

掌握 Realtime audio output 如何从 API 层 `modalities` 传到多模态 pipeline，并最终作为 `CompletionStreamChunk(modality='audio', audio_b64=...)` 回来。

你需要回答：

1. 当前为什么 `/v1/realtime` 不能出语音？
2. `output_modalities` 是在哪里写入 metadata 的？
3. Qwen3-Omni 如何根据 metadata 决定是否走 talker/code2wav？
4. `CompletionStreamChunk.audio_b64` 是如何生成的？
5. 为什么 audio delta 的 base64 不能随便拼接？

### 3.2 代码阅读路径

1. `sglang_omni/serve/realtime/session.py`
   - `SessionObject(modalities=['text'])`
   - `build_response_request(output_modalities=['text'])`
   - `run_response()` 当前只处理 text chunk。

2. `sglang_omni/client/client.py`
   - `_build_omni_request()`：`metadata['output_modalities'] = request.output_modalities`
   - `completion_stream()`：`audio_to_base64(chunk.audio_data, output_format=audio_format)`
   - `_set_audio_data()`：将 dict/memoryview/numpy waveform 转成 `GenerateChunk.audio_data`。

3. `sglang_omni/client/audio.py`
   - 重点：`audio_to_base64()`、`encode_audio()`、格式和采样率处理。

4. `sglang_omni/models/qwen3_omni/request_builders.py`
   - `output_modalities()`
   - `should_generate_audio_output()`
   - `resolve_mm_aggregate_next_stages()`
   - `resolve_thinker_stream_done_targets()`
   - `resolve_terminal_stages()`

5. 如果涉及 Ming-Omni，也看：
   - `sglang_omni/models/ming_omni/components/talker_executor.py`
   - `should_generate_audio()`
   - `_output_modalities()`

6. 对照已有 HTTP audio/chat stream：
   - `sglang_omni/serve/openai_api.py::_chat_stream()`
   - `sglang_omni/serve/openai_api.py::_speech_stream()`
   - `sglang_omni/serve/openai_api.py::build_speech_generate_request()`

### 3.3 主链路图

```text
session.update(modalities=['text','audio'])
  -> self.session_object.modalities
  -> build_response_request(output_modalities=session.modalities)
  -> Client._build_omni_request()
  -> OmniRequest.metadata['output_modalities']
  -> should_generate_audio_output()
  -> resolve stages include talker/code2wav
  -> StreamMessage(modality='audio', chunk.audio_data)
  -> Client._default_stream_builder()
  -> Client.completion_stream()
  -> CompletionStreamChunk(audio_b64=...)
  -> response.audio.delta
```

### 3.4 执行/观察任务

#### 任务 A：确认 text-only 当前状态

阅读并记录当前两处写死：

```text
SessionObject.modalities = ['text']
build_response_request().output_modalities = ['text']
```

说明这两处分别影响：

- 对客户端展示的 session 配置；
- 实际 pipeline 执行路径。

#### 任务 B：对比 ChatCompletion audio streaming

阅读 `openai_api.py::_chat_stream()`，记录它如何处理：

- `requested_modalities`
- `chunk.modality == 'text'`
- `chunk.modality == 'audio'`
- `chunk.audio_b64`
- finish chunk。

思考：哪些逻辑可以借鉴到 Realtime？哪些不能直接复用？

#### 任务 C：确认 audio chunk 格式风险

阅读 `client/audio.py` 之后回答：

- `audio_to_base64(..., output_format='wav')` 输出的是完整 WAV 容器还是 raw audio？
- 如果每个 delta 是完整 WAV，多个 base64 字符串能否直接拼接？
- `response.done` 是否应该放完整 audio？如果不能拼接，第一版如何设计？

### 3.5 思考题

1. Realtime 默认是否应该开启 `['text','audio']`？为什么 thinker-only 环境是风险？
2. `content_index` 应该由什么决定？为什么不能硬编码 text=0/audio=1？
3. 如果用户设置 `modalities=['audio']`，还要不要发 `response.text.done`？
4. audio output 的真实验收标准应该是什么？仅收到 `audio.delta` 够不够？

### 3.6 本章产出

- 一张 audio output 链路图。
- 一张 `modalities` 与 `content_index` 对照表。
- 一段对 audio base64 拼接风险的说明。

---

## Chapter 4：VAD、audio buffer 与 turn boundary（4h）

### 4.1 学习目标

掌握 server VAD 如何把连续 audio stream 切成 turn，audio buffer 如何维护样本偏移，为什么 cleanup 不能乱做。

你需要回答：

1. `input_audio_buffer.append` 的 base64 如何变成 PCM bytes？
2. VAD 是按多大 frame 工作？
3. `speech_started` / `speech_stopped` 的 sample offset 如何计算？
4. `utterance_start_byte` 和 `buffer_origin_samples` 分别是什么？
5. `auto_commit_utterance()` 什么时候切 wav payload？
6. 为什么 barge-in cancel 不能清 input buffer？

### 4.2 代码阅读路径

1. `sglang_omni/serve/realtime/audio_buffer.py`
   - `RealtimeAudioBuffer.append_b64()`
   - `tail()`
   - `to_sliced_wav_data_uri()`
   - `num_samples`
   - `DEFAULT_MAX_BUFFER_BYTES`

2. `sglang_omni/serve/realtime/vad.py`
   - `VADConfig`
   - `StreamingVAD.process()`
   - `reset()`
   - `offsets_to_ms()`
   - `emits_for_test()`

3. `sglang_omni/serve/realtime/session.py`
   - `handle_audio_append()`
   - `handle_vad_emit()`
   - `auto_commit_utterance()`
   - `drop_buffer_and_reset_vad()`
   - `handle_audio_clear()`

### 4.3 主链路图

```text
input_audio_buffer.append(audio_b64)
  -> audio_buffer.append_b64()
  -> new_bytes = audio_buffer.tail(decoded_len)
  -> vad.process(new_bytes)
     -> emits speech_started / speech_stopped
  -> handle_vad_emit()
     speech_started:
       utterance_start_byte = sample_offset * 2
       utterance_item_id = new item
       send input_audio_buffer.speech_started
     speech_stopped:
       send input_audio_buffer.speech_stopped
       auto_commit_utterance(end_sample_offset)
         -> slice audio_buffer[start_byte:end_byte]
         -> wav data uri
         -> drop_buffer_and_reset_vad
         -> send input_audio_buffer.committed
         -> response_queue.put
```

### 4.4 执行/观察任务

#### 任务 A：手算 offset

用 16kHz PCM16 mono 计算：

- 1 sample = 2 bytes。
- 512 samples = 32ms。
- 300ms prefix padding = 4800 samples。

回答：

- `speech_started` 的 sample offset 为什么要减 prefix padding？
- `audio_start_ms = offsets_to_ms(buffer_origin_samples + emit.sample_offset)` 中 `buffer_origin_samples` 的意义是什么？

#### 任务 B：看 VAD 状态机

阅读 `StreamingVAD.process()`，列出这些状态何时变化：

| 状态变量 | 变化时机 | 用途 |
|---|---|---|
| `leftover_pcm` |  |  |
| `samples_consumed` |  |  |
| `is_speech` |  |  |
| `silence_run_samples` |  |  |
| `last_speech_offset` |  |  |

#### 任务 C：分析 cleanup 风险

基于当前 `drop_buffer_and_reset_vad()`，分析这个场景：

```text
旧 response 正在 streaming
用户开始说新话 -> speech_started
系统决定 cancel 旧 response
如果 cancel cleanup 调用 audio_buffer.clear()
新话的前缀会怎样？
```

写出你认为应该拆分的 cleanup：

- committed input cleanup；
- response state cleanup；
- teardown cleanup。

### 4.5 思考题

1. `session.update(turn_detection.threshold=0.8)` 后是否应该 reset `samples_consumed`？为什么？
2. VAD hot update 为什么只重置 `silence_run_samples` 比较保守？
3. `turn_detection=null` 在当前没有 manual commit 的情况下应不应该支持？
4. buffer overflow 后应该 clear buffer 吗？应该发哪些事件？

### 4.6 本章产出

- 一张 audio buffer / VAD turn boundary 图。
- 一张 VAD 状态变量表。
- 一段 cleanup 分类说明。

---

## Chapter 5：Response state machine 与 cancel/abort（5h）

### 5.1 学习目标

掌握为什么 `response.cancel` 是整个任务的核心难点，以及如何从 AI infra 角度设计 request lifecycle、partial output、abort、task cleanup。

你需要回答：

1. cancel 到达时可能处于哪些状态？
2. delta 已发但 done 未发时客户端会怎样？
3. Python task cancel 和 engine abort 是什么关系？
4. 为什么 cancel 不等于硬中断 CUDA kernel？
5. 如何避免重复 cancel / 重复 done？
6. cancel 后是否保留 transcript？这是产品决策还是技术必然？

### 5.2 代码阅读路径

1. `sglang_omni/serve/realtime/session.py`
   - `handle_response_cancel()` 当前实现。
   - `run_response()` 中 `async for chunk in completion_stream()`。
   - `run_turn()` 中 response/transcription 顺序。
   - `_cancel_and_abort()` 和 `teardown()`。

2. `sglang_omni/client/client.py`
   - `Client.abort()`。

3. `sglang_omni/pipeline/coordinator.py`
   - `Coordinator.abort()`。
   - `Coordinator.stream()` 的 `finally` 清理。

4. `sglang_omni/models/qwen3_omni/components/streaming_detokenizer.py`
   - `abort()`。

5. 如果你要进一步理解 scheduler 层，可快速浏览：
   - `sglang_omni/scheduling/omni_scheduler.py::abort()`
   - `sglang_omni/scheduling/simple_scheduler.py::abort()`
   - 不要求深入实现，只需理解“标记 aborted / batch boundary 跳过”的语义。

### 5.3 目标状态机

你需要能画出：

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

并说明每个状态下允许的事件。

### 5.4 执行/观察任务

#### 任务 A：分析当前 cancel 缺陷

阅读当前 `handle_response_cancel()`，列出它做了什么、没做什么：

| 项目 | 当前是否做了 | 问题 |
|---|---|---|
| abort engine request |  |  |
| cancel Python task |  |  |
| await task result |  |  |
| emit text/audio done |  |  |
| emit response.done(cancelled) |  |  |
| 清 response state |  |  |
| 区分 response/transcription |  |  |

#### 任务 B：构造 cancel race 时间线

写出至少 3 条时间线：

```text
Case 1: response.created 后、任何 delta 前 cancel
Case 2: text.delta 已发送、text.done 前 cancel
Case 3: response.done 已发送、transcription 正在跑时 cancel
Case 4: VAD speech_started 触发 barge-in cancel
```

对每条说明应该发哪些事件。

#### 任务 C：设计 ResponseState

基于 `docs/design/realtime_api_detailed_design.md`，写出你自己的 `ResponseState` 字段：

- response id；
- item id；
- request id；
- modalities；
- content indexes；
- partial text/audio；
- delta/done sent flags；
- status。

说明每个字段解决什么问题。

### 5.5 思考题

1. cancel 前已经发给客户端的 delta 能不能“当没发生”？为什么？
2. `asyncio.CancelledError` 可能在哪些 await 点出现？
3. 为什么应该先标记 `status='cancelling'` 再做 abort/cancel？
4. `client.abort()` 返回 False 时，协议层应该怎么处理？
5. 如果 generator abort 后又吐出一个 stream chunk，Realtime 层应如何防御？
6. 显式 `response.cancel` 后是否继续 transcription？你会如何取舍？

### 5.6 本章产出

- 一张 response/cancel state machine。
- 一张 cancel race case 表。
- 一份 ResponseState 字段设计说明。

---

## Chapter 6：Error recovery、session lifecycle、conversation trim（4h）

### 6.1 学习目标

掌握 Realtime 长连接服务如何做到“错误可恢复、资源有边界、历史不无限增长”。

你需要回答：

1. 哪些错误应该发 `error` 后继续 session？
2. 哪些错误应该关闭 WebSocket？
3. max sessions 应该在哪里做？
4. idle timeout 如何避免误杀正在生成的 session？
5. conversation trim 应该按什么预算裁剪？
6. shutdown 如何 cancel active tasks 并吸收异常？

### 6.2 代码阅读路径

1. Error boundary：
   - `sglang_omni/serve/realtime/session.py::run`
   - `dispatch()`
   - `send_error()`
   - `handle_session_update()`
   - `handle_audio_append()`

2. Pydantic event：
   - `sglang_omni/serve/realtime/events.py`

3. Audio buffer error：
   - `sglang_omni/serve/realtime/audio_buffer.py::append_b64`
   - `DEFAULT_MAX_BUFFER_BYTES`

4. Session lifecycle：
   - `sglang_omni/serve/realtime/manager.py`
   - `sglang_omni/serve/openai_api.py::_register_realtime`
   - `sglang_omni/serve/realtime/session.py::teardown`
   - `_cancel_and_abort()`

5. Conversation：
   - `ConversationItem`
   - `self.conversation`
   - `build_response_request()`
   - `run_turn()` append history。

### 6.3 执行/观察任务

#### 任务 A：错误分类

列出当前可能发生的错误，并分类：

| 错误 | 位置 | 当前行为 | 目标行为 |
|---|---|---|---|
| invalid JSON | `run()` |  | error + keep session |
| unsupported event | `dispatch()` |  | error + keep session |
| invalid base64 | `audio_buffer.append_b64()` |  | error + keep session |
| buffer overflow | `append_b64()` |  | error + clear + keep session |
| invalid session.update | `handle_session_update()` |  | error + old config remains |
| inference exception | `run_response()` |  | server_error + keep session |
| websocket disconnect | `run()` |  | teardown |

#### 任务 B：设计 lifecycle state

给 `RealtimeSessionManager` 设计字段：

```text
sessions: dict[str, RealtimeSession]
max_sessions
idle_timeout_s
idle_scan_interval_s
scanner_task
```

给 `RealtimeSession` 设计字段：

```text
last_event_time
closed
is_busy()
```

说明它们如何配合。

#### 任务 C：conversation trim 手算

假设有 50 轮对话，每条 user/assistant 平均 200 token，history 总量约 20k token。设计一个从最新往旧保留的算法，至少保留最近一轮。

回答：

- system prompt 算不算 history budget？
- 当前 turn audio 算不算？
- tokenizer 不可用时如何估算？

### 6.4 思考题

1. `assert candidate.input_audio_format == 'pcm16'` 为什么不适合协议层？
2. invalid `session.update` 为什么不能部分写入 live state？
3. idle timeout 是否应该在 response 正在生成时触发？
4. max sessions 是拒绝握手还是 accept 后 close？各有什么差异？
5. conversation trim 后如何保证 role 顺序仍合理？

### 6.5 本章产出

- 一张 error 分类表。
- 一张 manager/session lifecycle 设计表。
- 一段 conversation trim 算法说明。

---

## Chapter 7：测试、可观测性与工程验收（4h）

### 7.1 学习目标

掌握如何验证 Realtime serving infra 的正确性：不是只看 happy path，而是建立 mock、集成、压测、metrics 分层。

你需要回答：

1. 哪些测试必须是 unit/mock？
2. 哪些测试必须跑真实 GPU？
3. fake websocket/client 怎么设计？
4. cancel race 如何稳定复现？
5. metrics 应该记录哪些关键现象？
6. 最终验收如何证明系统可用？

### 7.2 代码阅读路径

1. 测试组织说明：
   - `tests/README.md`

2. 现有 Realtime 集成测试：
   - `tests/test_model/test_qwen3_omni_realtime.py`

3. 现有 unit test/fake 风格：
   - `tests/unit_test/fixtures/pipeline_fakes.py`
   - `tests/unit_test/pipeline/test_coordinator.py`
   - `tests/unit_test/qwen3_omni/test_streaming.py`
   - `tests/unit_test/qwen3_omni/test_code2wav.py`
   - `tests/unit_test/serve/test_openai_api.py`

4. Benchmark 目录：
   - `benchmarks/README.md`
   - 后续计划新增 `benchmarks/realtime/bench.py`、`benchmarks/realtime/concurrency.py`。

5. 配置：
   - `pyproject.toml` 中 pytest markers 和 dependencies。

### 7.3 测试分层

| 层级 | 位置 | 用途 | 是否需要 GPU |
|---|---|---|---|
| unit: audio buffer | `tests/unit_test/realtime/test_audio_buffer.py` | base64、overflow、slice | 否 |
| unit: VAD | `tests/unit_test/realtime/test_vad.py` | config/state/timestamp | 否，必要时 monkeypatch infer |
| unit: session mock | `tests/unit_test/realtime/test_session_events.py` | event sequence | 否 |
| unit: cancel | `tests/unit_test/realtime/test_session_cancel.py` | partial done/cancel race | 否 |
| unit: manager | `tests/unit_test/realtime/test_manager_lifecycle.py` | max sessions / idle / shutdown | 否 |
| integration | `tests/test_model/test_qwen3_omni_realtime.py` | 真实 server + websocket | 是 |
| benchmark | `benchmarks/realtime/*` | latency/concurrency/memory | 是 |

### 7.4 FakeWebSocket / FakeClient 设计

FakeWebSocket 需要模拟：

```python
receive()
send_text()
close()
application_state
client_state
sent events list
```

FakeClient 需要模拟：

```python
completion_stream(request, request_id, audio_format='wav')
abort(request_id)
health()
```

为了复现 cancel race，FakeClient 的 `completion_stream()` 应支持：

- yield text chunk；
- yield audio chunk；
- 在某个 await 点阻塞；
- 在收到 cancel 后抛 `CancelledError`；
- 记录 request id；
- 模拟 abort 返回 True/False。

### 7.5 可观测性指标

第一版至少设计这些指标，即使先用 no-op facade：

| 指标 | 价值 |
|---|---|
| `realtime_sessions_active` | 是否 session 泄漏 |
| `realtime_turns_total{status}` | completed/cancelled/error 比例 |
| `realtime_vad_events_total{type}` | VAD 是否异常频繁触发 |
| `realtime_audio_chunks_total{direction}` | input/output chunk 规模 |
| `realtime_turn_duration_seconds` | E2E turn latency |
| `realtime_conversation_trims_total` | history 是否持续增长 |
| cancel latency | barge-in 体验 |
| GPU memory drift | cancel 是否泄漏资源 |

### 7.6 执行/观察任务

#### 任务 A：阅读现有 Realtime 集成测试

阅读：

```text
tests/test_model/test_qwen3_omni_realtime.py
```

记录它覆盖了什么、没覆盖什么：

| 能力 | 是否覆盖 |
|---|---|
| session.created |  |
| VAD speech_started/stopped |  |
| response.text.delta/done |  |
| transcription delta/completed |  |
| disconnect teardown |  |
| audio output |  |
| cancel |  |
| error recovery |  |
| session.update |  |

#### 任务 B：设计 5 个最小 mock 测试

写出测试名和预期事件序列：

1. text-only response。
2. text+audio response。
3. text delta 后 cancel。
4. invalid base64 后继续正常 append。
5. session.update invalid threshold 不污染旧 config。

#### 任务 C：定义最终验收 checklist

至少包括：

- text-only baseline 不回归；
- audio-capable 环境收到 audio delta；
- cancel < 200ms 收到 cancelled；
- invalid input 后 session 可继续；
- 50 轮后 history bounded；
- max sessions 生效；
- 100 次 cancel 后显存无明显增长。

### 7.7 思考题

1. 为什么 cancel race 不能只靠真实 GPU 集成测试？
2. 为什么 fake client 比 monkeypatch coordinator 更适合 session 事件测试？
3. 哪些 metrics 是 debug 必需，哪些只是 dashboard 好看？
4. benchmark 结果应该作为 CI gate 还是人工报告？为什么？
5. 如何避免新增测试过度依赖实现细节？

### 7.8 本章产出

- 一张测试分层表。
- 5 个 mock test case 的事件序列。
- 一份最终验收 checklist。

---

## 8. 30 小时结束后的自检清单

完成 30 小时学习后，你应该能做到：

### 8.1 代码理解

- [ ] 能从 WebSocket event 讲到 `RealtimeSession.dispatch()`。
- [ ] 能从 `run_response()` 讲到 `Coordinator.stream()`。
- [ ] 能解释 `GenerateRequest.output_modalities` 如何影响 pipeline。
- [ ] 能解释 `CompletionStreamChunk.audio_b64` 从哪里来。
- [ ] 能解释 VAD 的 `speech_started` / `speech_stopped` offset。
- [ ] 能解释 `buffer_origin_samples` 的作用。
- [ ] 能解释当前 `active_request_id` 为什么需要拆分。

### 8.2 设计理解

- [ ] 能画出 response state machine。
- [ ] 能说明 delta/done/response.done 的事件闭合关系。
- [ ] 能说明 cancel 为什么需要 partial output done。
- [ ] 能说明 cancel 为什么不能清 input audio buffer。
- [ ] 能说明 error boundary 应该放在 message loop 内。
- [ ] 能说明 session lifecycle 的 max/idle/shutdown 设计。

### 8.3 实现准备

- [ ] 能设计 `ResponseState`。
- [ ] 能设计 `StreamingVAD.update_config()`。
- [ ] 能设计 `session.update` validation matrix。
- [ ] 能设计 fake websocket/client。
- [ ] 能写出至少 5 个 mock event sequence tests。
- [ ] 能列出 GPU 集成测试验收项。

---

## 9. 建议的个人笔记结构

建议在本地另建一份个人笔记，不提交，按这个模板记录：

```text
# Chapter X Notes

## 我读过的文件
- path: 结论

## 我画出的链路
用缩进文本或 Mermaid 记录链路。

## 关键状态变量
| 变量 | 所属对象 | 生命周期 | 风险 |
|---|---|---|---|

## 失败/取消场景
| 场景 | 当前行为 | 目标行为 | 测试方法 |
|---|---|---|---|

## 我仍不确定的问题
1. ...
```

这份笔记会直接服务后续实现和 PR review。

---

## 10. 推荐阅读顺序总结

如果只能记一条主线，就按下面顺序读：

```text
openai_api.py::_register_realtime
  -> manager.py::RealtimeSessionManager
  -> session.py::RealtimeSession.run / dispatch / handlers
  -> audio_buffer.py / vad.py
  -> session.py::auto_commit_utterance / drain_queue / run_turn
  -> session.py::run_response / build_response_request
  -> client/types.py::GenerateRequest / CompletionStreamChunk
  -> client/client.py::completion_stream / _build_omni_request
  -> coordinator.py::stream / abort
  -> qwen3_omni/request_builders.py::should_generate_audio_output
  -> tests/test_model/test_qwen3_omni_realtime.py
  -> tests/README.md
```

这条线读懂后，再回头看详细设计文档，基本就能进入实现阶段。
