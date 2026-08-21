# 📋 智能审查报告

| 🟠 P1 建议修复 | 🟢 P2 可选优化 | ⚪ P3 仅供参考 | 💬 讨论 |
|:-------------:|:-------------:|:-------------:|:-------:|
| 1 | 3 | 2 | 2 |

## 摘要

本次审查覆盖最新 3 个 commit（`c631fe4` 多模态 cache offload、`7beecbd` 流式音频输出、`8fa42b3` 全双工），涉及 `session.py`、`stage_cache.py`、`stages.py`、`audio_head.py`、`model_runner.py`。整体设计清晰、注释详尽、env-gated 特性开关的向后兼容做得好。主要值得关注的点是：barge-in 打断会导致该轮对话上下文（用户转写 + 助手回复）整体丢失；以及 pinned-host offload 的 side-stream 在 `synchronize()` 后实际退化为同步拷贝，未达成预期的 D2H/H2D overlap。

---

## 🟠 P1 问题

> 建议修复

---

### Barge-in 打断会丢失整轮对话上下文

<sub>`LOGIC` · `Issue-001/kl9dvkrau22x63b4s007`</sub>

📍 `sglang_omni/serve/realtime/session.py:L165-L192` · `L237-L249` · 🟠 待解决

`run_turn` 采用「先 `run_response` 再 `run_transcription`，两者都完成后再一次性 append 到 `self.conversation`」的顺序：

**证据**:
```python
async def run_turn(self, item_id: str, audio_payload: str) -> None:
    response_text = await self.run_response(audio_payload)
    transcript = await self.run_transcription(item_id, audio_payload)
    # 只有走到这里才写入历史
    if transcript:
        self.conversation.append(ConversationItem(role="user", text=transcript))
    if response_text:
        self.conversation.append(ConversationItem(role="assistant", text=response_text))
```

而 barge-in 在 `SPEECH_STARTED` 时会 `cancel` 掉整个 `active_task`（即 `run_turn`）：

```python
if emit.event_type == VADEvent.SPEECH_STARTED:
    await self._cancel_and_abort(self.active_task, self.active_request_id)
```

问题在于：如果打断发生在 `run_response` 已完成、`run_transcription` 进行中的时刻，`run_turn` 被取消后 append 两行都不会执行 → 本轮**用户转写和助手回复都不会进入 `conversation`**。后续轮次因此丢失该轮上下文，多轮连贯性受损（尤其打断是语音交互的高频场景）。

**建议**: 考虑将「已经完成的部分」尽早落库，而不是等两个 pass 全部结束。例如把 `response_text` 在 `run_response` 返回后立即 append；或在 `_cancel_and_abort` 前保存已产出的 `response_text`/`transcript`，在 finally 中补写。至少建议评估「打断即丢弃整轮历史」是否是期望语义。

---

## 🟢 P2 问题

> 可选优化

---

### side-stream 在 `synchronize()` 后退化为同步拷贝，未达成 overlap

<sub>`MAINT` · `Issue-002/kl9dvkrau22x63b4s007`</sub>

📍 `sglang_omni/scheduling/stage_cache.py:L31-L47` · 🟢 待解决

注释目标是「D2H 在 side stream 上跑以便和后续 relay H2D overlap」，但代码在 side stream 上拷贝后立即 `stream.synchronize()`：

```python
if stream is not None:
    stream.wait_stream(torch.cuda.current_stream(value.device))
    with torch.cuda.stream(stream):
        staging.copy_(value, non_blocking=True)
    value.record_stream(stream)
    stream.synchronize()   # ← 这里阻塞，直到 D2H 完成
```

`put()` 是同步调用，`synchronize()` 会让当前 Python 线程阻塞至拷贝结束，等价于同步 D2H，overlap 收益基本为零，反而多了创建 side stream 的开销。

**建议**: 若要真正 overlap，考虑改为记录 `torch.cuda.Event` 并在**取用（`get`）时**才 wait，或在下一次 stage 边界统一同步；若暂不追求 overlap，可以简化为直接同步拷贝并在注释中说明现状，避免误导。

---

### 流式 buffer / threshold 状态在请求异常中断时可能泄漏

<sub>`EDGE` · `Issue-003/kl9dvkrau22x63b4s007`</sub>

📍 `sglang_omni/models/longcat_next/stages.py:L861-L900` · 🟢 待解决

`_stream_thresholds` 与 `_buffers` 仅在 `finalize`（`rid in _buffers` 分支）时清理：

```python
if rid in _buffers:
    remaining = _buffers.pop(rid, [])
    _stream_thresholds.pop(rid, None)
```

如果请求被 abort/cancel（如上面的 barge-in）而没有走到正常 finalize，这两个 dict 的条目不会被清除，长时间运行下会累积。`_buffers` 本身在改动前就有此问题，本次新增的 `_stream_thresholds` 沿用了相同生命周期。

**建议**: 评估是否需要一个按 request_id 的统一清理入口（abort 回调）来同时清理 `_buffers` 和 `_stream_thresholds`，避免长会话下的 dict 膨胀。

---

### `non_blocking=True` 的 H2D 仅在源为 pinned 内存时才真正异步

<sub>`EDGE` · `Issue-004/kl9dvkrau22x63b4s007`</sub>

📍 `sglang_omni/models/longcat_next/model_runner.py:L100-L108` · 🟢 待解决

```python
replace_embeds_parts.append(chunk.to(device=device, non_blocking=True))
replace_positions_parts.append(local_positions.to(device=device, non_blocking=True))
```

注释已正确说明「non_blocking lets a pinned (offloaded) source H2D overlap; no-op otherwise」。补充一个隐性契约：只有当 `chunk` 来自 pinned host（即前述 offload cache）时 overlap 才成立；若来源是普通 pageable CPU tensor，`non_blocking=True` 会退化为同步拷贝——功能上安全，但需保证 append 后到实际 `cat`/使用之间不会在**未同步**的情况下被跨流读取。当前路径看起来是同一默认流内消费，风险低。

**建议**: 保持现状即可，建议在依赖 overlap 的路径上加一处断言/日志确认 `chunk.is_pinned()`，让性能假设可观测。

---

## ⚪ P3 问题

> 仅供参考

---

### `audio_head.forward` 移除 `prev_audio_codes` 使用

<sub>`API` · `Issue-005/kl9dvkrau22x63b4s007`</sub>

📍 `sglang_omni/models/longcat_next/components/audio_head.py:L379-L397`

`forward` 现在 `del prev_audio_codes`，改为每步在 `_decode_loop` 内零初始化 `codes`，理由是「因果掩码使 codebook k 只 attend 0..k-1」。查证唯一调用点 `sglang_model.py:L338` 本就只传 `hidden_state`（`prev_audio_codes` 为 None → 原逻辑也是零初始化），故行为等价、无回归。保留参数做 API 兼容也合理。仅提示：该参数已成为死参，未来可考虑彻底移除以减少认知负担。

---

### `response.done` 的 audio content 缺少实际音频字段

<sub>`API` · `Issue-006/kl9dvkrau22x63b4s007`</sub>

📍 `sglang_omni/serve/realtime/session.py:L339-L341`

```python
content = [{"type": "text", "text": response_text}]
if audio_emitted:
    content.append({"type": "audio", "transcript": response_text})
```

audio content item 仅带 `transcript`（复用文本），无 `audio` 数据字段。考虑到音频已通过 `response.audio.delta` 流式下发，这里不重复携带二进制是合理的。仅提示与 OpenAI Realtime 规范对齐时，部分客户端可能期望该 item 带上 `audio` 或至少 `content_index` 对应关系，建议核对目标客户端契约。

---

## 💬 待讨论

> 一些观察和思考，供参考

### Q-1: barge-in 后 audio buffer 与响应队列的时序

打断时先 `_cancel_and_abort(active_task)`，随后设置新的 `utterance_start_byte`。此时 `drain_queue` 会继续消费队列中**已提交**的后续 utterance。若打断瞬间队列里仍有上一句的残留（例如快速连续说话），是否可能出现「取消了当前轮，但队列里还排着刚才那句」的错位？

**Agent 分析**: 从代码看 `auto_commit_utterance` 是在 `SPEECH_STOPPED` 才入队，barge-in 的 `SPEECH_STARTED` 只取消在途任务、不动队列，逻辑上是自洽的。但在「A 说话→停→引擎开始响应→A 立刻打断说 B」的链路里，A 那句已经在队列/在途，B 尚未提交。取消在途后 drain_queue 会不会又把 A 重新拉起跑一遍？

这是否是期望行为，还是应该在 barge-in 时同时清空 `response_queue` 中尚未开始的旧轮？

### Q-2: CUDA graph 按 batch size 缓存的显存占用

`_decode_loop_graphed` 为每个出现过的 `bs` 各捕获一张图并常驻 `static_in/static_out`。在线服务 batch size 分布较散时，`_graphs` 会累积多张图与静态缓冲。

**Agent 分析**: 8 步 decode 的图本身不大，但若 bs 取值范围广（1..N 各来一次），常驻缓冲会线性增长。是否需要限制只对少数「热点 bs」建图，或对 bs 做 padding 归一到固定档位（类似 SGLang 主体的 cuda graph bs 分档）？

---

*📝 本报告由 Code Review Skill 生成*