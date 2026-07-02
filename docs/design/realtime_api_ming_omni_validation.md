# Realtime API 验证方案：基于 Ming-Omni

## 结论：Ming-Omni 完全可以用于 Realtime API 的开发和验证

Realtime 模块（`sglang_omni/serve/realtime/`）**与具体模型解耦**——它只通过
`Client.completion_stream()` 消费通用的 `CompletionStreamChunk`（含 `.text`、`.modality`、
`.audio_b64`、`.finish_reason` 字段），不依赖任何模型特定的类或接口。

Ming-Omni 支持 `output_modalities`、有 thinker 对话能力、有 talker 音频输出能力（speech
pipeline），满足 Realtime API 全部功能验证的需求。

## 与 Qwen3-Omni 的对比

| | Qwen3-Omni | Ming-Omni |
|---|---|---|
| Realtime 入口 | 已有 `--enable-realtime` | 已添加（`run_ming_omni_server.py` +5 行） |
| 模型规模 | 30B 总 / 3B 激活（MoE） | ~200B（MoE） |
| 音频输出链路 | talker_ar → code2wav（vocoder） | talker（LLM+CFM+DiT+AudioVAE） |
| 流式 TTS | 无 | 有（`MingOmniStreamingSpeechPipelineConfig`） |
| 文本管线 | thinker → decode | thinker → decode |

## 硬件需求

- **文本验证**（当前 Realtime 功能）：1× H200 即可
- **音频输出验证**（F1 目标）：建议 2-4× H200（talker 额外显存）
- 用户当前：8× H200，绰绰有余

## 启动方式

### Step 1：文本链路（当前 Realtime 功能）

```bash
python examples/run_ming_omni_server.py \
  --model-path inclusionAI/Ming-flash-omni-2.0 \
  --port 8080 \
  --enable-realtime \
  --cpu-offload-gb 0
```

验证：

```bash
# WebSocket 连通性
wscat -c ws://localhost:8080/v1/realtime
# 预期：收到 {"type":"session.created",...}

# Chat API
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"ming-omni","messages":[{"role":"user","content":"你好"}],"max_tokens":64,"stream":true}'
```

### Step 2：音频输出链路（F1 实现时）

`run_ming_omni_speech_server.py` 使用 `MingOmniSpeechPipelineConfig`（含 talker）或
`MingOmniStreamingSpeechPipelineConfig`（含 segmenter + talker_stream 流式 TTS）。
需要同样加上 `--enable-realtime` 参数。

## 执行流追踪要点

启动服务后，追踪以下路径确认理解：

```
WebSocket /v1/realtime
  → RealtimeSession.run()                        # session.py:117
  → receive JSON → dispatch(payload)             # session.py:137
  → handle_audio_append                          # session.py:158
    → audio_buffer.append_b64                    # audio_buffer.py:39
    → vad.process(new_bytes)                     # vad.py:57
    → handle_vad_emit                            # session.py:165
      → speech_started / speech_stopped
      → auto_commit_utterance                    # session.py:196
        → drop_buffer_and_reset_vad              # session.py:189
        → response_queue.put                     # session.py:210
  → drain_queue                                  # session.py:225
    → run_turn                                   # session.py:232
      → run_response                             # session.py:246
        → build_response_request                 # session.py:373
        → client.completion_stream               # session.py:269
        → response.text.delta × N               # session.py:275
        → response.text.done                     # session.py:296
        → response.done                          # session.py:306
      → run_transcription                        # session.py:330
        → conversation.item...transcription.*
      → conversation.append                      # session.py:239-244
```

观察要点：
1. `chunk.modality` 的值（当前只有 `"text"`）
2. `completion_stream()` 返回的 `CompletionStreamChunk` 结构
3. VAD 的 `speech_started` / `speech_stopped` 时序
4. response 和 transcription 的顺序关系

## 当前限制

1. Realtime 写死 `output_modalities=["text"]`，即使启动 speech pipeline 也不会出音频（F1 待实现）
2. Realtime 没有 `response.cancel` 的协议闭合（F3 待实现）
3. Realtime 没有 session 生命周期管理（F4 待实现）
4. 所有其他缺口见 `realtime_api_completion.md` 第 1.1 节

## 代码改动记录

`examples/run_ming_omni_server.py`：新增 `--enable-realtime` CLI 参数 + 透传至 `launch_server()`。
共 5 行，无行为影响（默认 `False`）。
