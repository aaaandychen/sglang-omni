from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Env-gated latency timing (Phase 5). When enabled, per-turn latency markers
# (TTFA, server-side barge-in stop) are logged. Zero overhead when off.
_TIMING = os.environ.get("SGLANG_OMNI_REALTIME_TIMING", "0") == "1"


def _now_ms() -> float:
    return time.monotonic() * 1000.0


def _log_timing(label: str, start_ms: float, **extra: Any) -> None:
    if not _TIMING:
        return
    dt = _now_ms() - start_ms
    tail = " ".join(f"{k}={v}" for k, v in extra.items())
    logger.info("[realtime-timing] %s=%.1fms %s", label, dt, tail)

from fastapi import WebSocket
from starlette.websockets import WebSocketState

from sglang_omni.client import Client, GenerateRequest, Message, SamplingParams
from sglang_omni.serve.realtime.audio_buffer import RealtimeAudioBuffer
from sglang_omni.serve.realtime.events import (
    InputAudioBufferAppend,
    InputAudioBufferClear,
    ResponseCancel,
    SessionObject,
    SessionUpdate,
    make_event,
    parse_client_event,
)
from sglang_omni.serve.realtime.vad import (
    StreamingVAD,
    VADConfig,
    VADEvent,
    offsets_to_ms,
)

DEFAULT_INSTRUCTIONS = (
    "You are a helpful realtime voice assistant. Respond conversationally."
)

# Hardcoded — transcription must be verbatim regardless of session instructions.
_TRANSCRIPTION_PROMPT = (
    "You are a speech-to-text engine. Transcribe the user's spoken audio "
    "verbatim into the same language they spoke. Output ONLY the transcript "
    "— no descriptions, no refusals, no explanations."
)

HANDLERS: dict[type, str] = {
    SessionUpdate: "handle_session_update",
    InputAudioBufferAppend: "handle_audio_append",
    InputAudioBufferClear: "handle_audio_clear",
    ResponseCancel: "handle_response_cancel",
}


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


@dataclass
class ConversationItem:
    role: str  # "user" | "assistant"
    text: str
    # P3/P4: a user item is created (empty, pending=True) at turn start to
    # preserve chronological ordering, then filled by the background
    # transcription task. Pending items are skipped when building request
    # history until their text arrives.
    pending: bool = False


class RealtimeSession:
    """Owns one WebSocket and one OpenAI-Realtime audio-in / text-out session.

    Per turn (VAD ``speech_stopped`` → auto-commit):
      1. ``run_response`` consumes the audio + prior conversation, streams
         ``response.*`` events to the client. User sees their reply fast.
      2. A pending user history slot is reserved BEFORE the response so
         chronological ordering is preserved (P3).
      3. ``run_transcription`` runs as a BACKGROUND task (off the TTFA
         critical path, P3), re-consuming the audio with a verbatim-transcribe
         prompt and filling the reserved user slot when done.

    Latency optimizations (Phase 5 pseudo-full-duplex):
      * P0: barge-in emits ``speech_started`` + ``response.audio.flush``
        immediately, aborting the prior response in the background; the next
        turn waits on that abort via a barrier in ``drain_queue``.
      * P2: adaptive VAD end-pointing (see ``vad.py``).
      * P4: deterministic history prefix so RadixCache prefill hits.
      * P5: optional env-gated streaming partial transcription for live UI.
    """

    def __init__(
        self,
        websocket: WebSocket,
        *,
        client: Client,
        model_name: str,
        session_id: str | None = None,
    ) -> None:
        self.websocket = websocket
        self.client = client
        self.model_name = model_name
        self.session_id = session_id or new_id("sess")

        self.session_object = SessionObject(
            id=self.session_id,
            model=model_name,
            modalities=["text"],
            instructions=DEFAULT_INSTRUCTIONS,
            input_audio_format="pcm16",
        )

        self.audio_buffer = RealtimeAudioBuffer(source_sr=16000, target_sr=16000)
        # (role, text) records — fed back as message history on the next turn.
        self.conversation: list[ConversationItem] = []
        self.closed = False

        self.active_request_id: str | None = None
        self.active_task: asyncio.Task | None = None
        # VAD may emit speech_stopped while engine is still busy on an
        # earlier utterance — serialize via FIFO.
        self.response_queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        self.queue_drainer: asyncio.Task | None = None
        # P0 barge-in: cancel/abort of the previous in-flight response runs in
        # the background so ``speech_started`` is emitted immediately. The next
        # turn must wait on this barrier before starting (avoid new/old req
        # racing for KV). ``None`` when no abort is pending.
        self._pending_abort: asyncio.Task | None = None
        # P3: transcription runs off the TTFA critical path as a background
        # task. Tracked so teardown can cancel any still-running transcribe.
        self._bg_tasks: set[asyncio.Task] = set()
        # P5 (optional, env-gated): streaming ASR — emit partial transcripts of
        # the CURRENT utterance while the user is still speaking, purely for UI
        # (does not feed the response). Off by default; extra ASR compute.
        self._streaming_asr = (
            os.environ.get("SGLANG_OMNI_REALTIME_STREAMING_ASR", "0") == "1"
        )
        self._partial_interval_ms = 400.0  # throttle partial ASR passes
        self._last_partial_ms = 0.0
        self._partial_task: asyncio.Task | None = None

        # VAD is created once with default config; session.update doesn't
        # touch it. Reconnect to change VAD params.
        self.vad = StreamingVAD(VADConfig())
        # Session-wall-clock sample offset of buffer byte 0; advances on
        # commit so speech timestamps stay correct after a buffer drop.
        self.buffer_origin_samples = 0
        self.utterance_start_byte: int | None = None
        # speech_started.item_id predicts the eventual committed id so
        # clients can align live VAD events to the transcript.
        self.utterance_item_id: str | None = None

    async def run(self) -> None:
        """Drive the WebSocket loop; ``websocket.disconnect`` arrives in-band."""
        await self.send(
            make_event(
                "session.created",
                session=self.session_object.model_dump(exclude_none=True),
            )
        )

        while not self.closed:
            message = await self.websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message["type"] != "websocket.receive":
                continue
            raw = message["text"]
            payload = json.loads(raw)
            assert isinstance(payload, dict), "Top-level payload must be a JSON object"
            await self.dispatch(payload)

    async def dispatch(self, payload: dict[str, Any]) -> None:
        event = parse_client_event(payload)
        assert event is not None, f"Unsupported event type: {payload.get('type')!r}"
        method_name = HANDLERS[type(event)]
        await getattr(self, method_name)(event)

    async def handle_session_update(self, event: SessionUpdate) -> None:
        # Validate a candidate first so a rejected update never lands in live state.
        update = event.session.model_dump(exclude_none=True, exclude_unset=True)
        candidate = SessionObject.model_validate(
            self.session_object.model_dump() | update
        )
        assert candidate.input_audio_format == "pcm16", "Only pcm16 is supported"
        self.session_object = candidate
        # P2: apply turn_detection tuning to the live VAD config. Only the
        # fields the client set are overridden; adaptive end-pointing stays on.
        td = candidate.turn_detection
        if td is not None:
            cfg = self.vad.config
            if td.threshold is not None:
                cfg.threshold = td.threshold
            if td.prefix_padding_ms is not None:
                cfg.prefix_padding_ms = td.prefix_padding_ms
            if td.silence_duration_ms is not None:
                cfg.silence_duration_ms = td.silence_duration_ms
                # Reset the adaptive baseline to the newly requested value.
                self.vad.effective_silence_ms = td.silence_duration_ms
        await self.send(
            make_event(
                "session.updated",
                session=self.session_object.model_dump(exclude_none=True),
            )
        )

    async def handle_audio_append(self, event: InputAudioBufferAppend) -> None:
        decoded_len = self.audio_buffer.append_b64(event.audio)
        new_bytes = self.audio_buffer.tail(decoded_len)
        emits = await asyncio.to_thread(self.vad.process, new_bytes)
        for emit in emits:
            await self.handle_vad_emit(emit)
        # P5 streaming ASR: while the user is mid-utterance, periodically kick a
        # partial transcription of the audio-so-far for live UI. Throttled and
        # single-flight so it never piles up or blocks ingestion.
        if self._streaming_asr and self.vad.is_speech:
            await self._maybe_emit_partial_transcript()

    async def _maybe_emit_partial_transcript(self) -> None:
        now = _now_ms()
        if now - self._last_partial_ms < self._partial_interval_ms:
            return
        if self._partial_task is not None and not self._partial_task.done():
            return  # single-flight: previous partial still running
        self._last_partial_ms = now
        start_byte = self.utterance_start_byte or 0
        end_byte = self.audio_buffer.num_bytes
        if end_byte <= start_byte:
            return
        item_id = self.utterance_item_id or new_id("item")
        payload = self.audio_buffer.to_sliced_wav_data_uri(
            start_byte=start_byte, end_byte=end_byte
        )
        self._partial_task = asyncio.create_task(
            self._run_partial_transcript(item_id, payload)
        )
        self._bg_tasks.add(self._partial_task)
        self._partial_task.add_done_callback(self._bg_tasks.discard)

    async def _run_partial_transcript(self, item_id: str, payload: str) -> None:
        """Fire-and-forget partial ASR pass; emits a delta for live UI only.
        Uses a throwaway request id so it never collides with the response /
        final-transcription request tracking."""
        request_id = f"rt-partial-{self.session_id}-{uuid.uuid4().hex}"
        text_acc: list[str] = []
        try:
            async for chunk in self.client.completion_stream(
                self.build_transcription_request(payload),
                request_id=request_id,
            ):
                if chunk.modality == "text" and chunk.text:
                    text_acc.append(chunk.text)
                if chunk.finish_reason is not None:
                    break
        except asyncio.CancelledError:
            await self.client.abort(request_id)
            raise
        partial = "".join(text_acc)
        if partial and not self.closed:
            await self.send(
                make_event(
                    "conversation.item.input_audio_transcription.delta",
                    item_id=item_id,
                    content_index=0,
                    delta=partial,
                    partial=True,
                )
            )

    async def handle_vad_emit(self, emit: Any) -> None:
        timestamp_ms = offsets_to_ms(self.buffer_origin_samples + emit.sample_offset)
        if emit.event_type == VADEvent.SPEECH_STARTED:
            # P0 Barge-in: user started speaking while we're still responding.
            # Emit speech_started + audio flush IMMEDIATELY so the client stops
            # playback without waiting for the engine abort round-trip; the
            # abort itself is done in the background. The next turn waits on
            # ``self._pending_abort`` (barrier in drain_queue) so new/old reqs
            # don't race for KV.
            self.utterance_start_byte = min(
                max(0, emit.sample_offset * 2), self.audio_buffer.num_bytes
            )
            self.utterance_item_id = new_id("item")
            barge = self.active_task is not None and not self.active_task.done()
            await self.send(
                make_event(
                    "input_audio_buffer.speech_started",
                    audio_start_ms=timestamp_ms,
                    item_id=self.utterance_item_id,
                )
            )
            if barge:
                # P1: tell the client to drop any buffered assistant audio now.
                await self.send(
                    make_event(
                        "response.audio.flush",
                        reason="barge_in",
                    )
                )
                # P0: abort previous response in background (do not block).
                prev_task, prev_rid = self.active_task, self.active_request_id
                self._pending_abort = asyncio.create_task(
                    self._cancel_and_abort(prev_task, prev_rid)
                )
                if _TIMING:
                    logger.info("[realtime-timing] barge_in_signal_sent")
        elif emit.event_type == VADEvent.SPEECH_STOPPED:
            await self.send(
                make_event(
                    "input_audio_buffer.speech_stopped",
                    audio_end_ms=timestamp_ms,
                    item_id=self.utterance_item_id or new_id("item"),
                )
            )
            await self.auto_commit_utterance(emit.sample_offset)

    def drop_buffer_and_reset_vad(self) -> None:
        self.buffer_origin_samples += self.audio_buffer.num_samples
        self.audio_buffer.clear()
        self.utterance_start_byte = None
        self.utterance_item_id = None
        self.vad.reset()

    async def auto_commit_utterance(self, end_sample_offset: int) -> None:
        if self.audio_buffer.is_empty():
            return
        start_byte = self.utterance_start_byte or 0
        end_byte = min(end_sample_offset * 2, self.audio_buffer.num_bytes)
        if end_byte <= start_byte:
            return
        payload = self.audio_buffer.to_sliced_wav_data_uri(
            start_byte=start_byte, end_byte=end_byte
        )
        item_id = self.utterance_item_id or new_id("item")
        self.drop_buffer_and_reset_vad()

        await self.send(make_event("input_audio_buffer.committed", item_id=item_id))
        await self.response_queue.put((item_id, payload))
        if self.queue_drainer is None or self.queue_drainer.done():
            self.queue_drainer = asyncio.create_task(self.drain_queue())

    async def handle_audio_clear(self, event: InputAudioBufferClear) -> None:
        self.drop_buffer_and_reset_vad()
        await self.send(make_event("input_audio_buffer.cleared"))

    async def handle_response_cancel(self, event: ResponseCancel) -> None:
        if self.active_task is None or self.active_task.done():
            return
        if self.active_request_id is not None:
            await self.client.abort(self.active_request_id)
        self.active_task.cancel()

    async def drain_queue(self) -> None:
        while not self.closed:
            item_id, payload = await self.response_queue.get()
            # P0 barrier: if a barge-in triggered a background abort of the
            # previous response, wait for it to finish releasing the engine
            # request before starting the next turn (avoid KV races).
            if self._pending_abort is not None:
                await asyncio.gather(self._pending_abort, return_exceptions=True)
                self._pending_abort = None
            self.active_task = asyncio.create_task(self.run_turn(item_id, payload))
            await asyncio.gather(self.active_task, return_exceptions=True)
            self.active_task = None

    async def run_turn(self, item_id: str, audio_payload: str) -> None:
        """Pass 1 (critical path): response — streams audio/text to the user.
        Pass 2 (background): transcription — fills history/UI, off the TTFA
        path so the next turn is not delayed by it (P3).

        Chronological history is preserved by inserting an empty pending user
        item BEFORE the assistant reply; the background transcription fills its
        text in place. Pending items are skipped by request-building until
        filled, so a not-yet-transcribed turn never corrupts the next prompt.
        """
        # Reserve the user slot first (spoke first), then the assistant slot.
        user_item = ConversationItem(role="user", text="", pending=True)
        self.conversation.append(user_item)
        response_text = ""
        try:
            response_text = await self.run_response(audio_payload)
        finally:
            if response_text:
                self.conversation.append(
                    ConversationItem(role="assistant", text=response_text)
                )
        # P3: transcription off the critical path — next turn can start now.
        task = asyncio.create_task(
            self._background_transcribe(item_id, audio_payload, user_item)
        )
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    async def _background_transcribe(
        self, item_id: str, audio_payload: str, user_item: ConversationItem
    ) -> None:
        """Run the verbatim transcription pass off the TTFA critical path and
        fill in the reserved user history slot (P3)."""
        transcript = ""
        try:
            transcript = await self.run_transcription(item_id, audio_payload)
        finally:
            if transcript:
                user_item.text = transcript
            user_item.pending = False

    async def run_response(self, audio_payload: str) -> str:
        """Emit response.created → text.delta / audio.delta × N → done.

        Audio-out is gated on the session having "audio" in its modalities.
        Audio chunks are streamed as ``response.audio.delta`` (base64 PCM16),
        text chunks as ``response.text.delta`` — both may interleave.
        """
        response_id = new_id("resp")
        request_id = f"rt-{self.session_id}-{uuid.uuid4().hex}"
        self.active_request_id = request_id
        turn_start_ms = _now_ms()
        first_audio_logged = False

        try:
            await self.send(
                make_event(
                    "response.created",
                    response={
                        "id": response_id,
                        "object": "realtime.response",
                        "status": "in_progress",
                        "output": [],
                    },
                )
            )

            resp_item_id = new_id("item")
            text_acc: list[str] = []
            audio_emitted = False
            finish_reason = "stop"
            usage: dict[str, Any] | None = None
            async for chunk in self.client.completion_stream(
                self.build_response_request(audio_payload),
                request_id=request_id,
                audio_format="pcm",
            ):
                if chunk.modality == "text" and chunk.text:
                    text_acc.append(chunk.text)
                    await self.send(
                        make_event(
                            "response.text.delta",
                            response_id=response_id,
                            item_id=resp_item_id,
                            output_index=0,
                            content_index=0,
                            delta=chunk.text,
                        )
                    )
                elif chunk.modality == "audio" and chunk.audio_b64:
                    audio_emitted = True
                    if not first_audio_logged:
                        first_audio_logged = True
                        _log_timing("ttfa", turn_start_ms, resp=response_id)
                    await self.send(
                        make_event(
                            "response.audio.delta",
                            response_id=response_id,
                            item_id=resp_item_id,
                            output_index=0,
                            content_index=0,
                            delta=chunk.audio_b64,
                        )
                    )
                if chunk.finish_reason is not None:
                    finish_reason = chunk.finish_reason
                    usage = (
                        dataclasses.asdict(chunk.usage)
                        if chunk.usage is not None
                        else None
                    )
                    break

            response_text = "".join(text_acc)
            if audio_emitted:
                await self.send(
                    make_event(
                        "response.audio.done",
                        response_id=response_id,
                        item_id=resp_item_id,
                        output_index=0,
                        content_index=0,
                    )
                )
            await self.send(
                make_event(
                    "response.text.done",
                    response_id=response_id,
                    item_id=resp_item_id,
                    output_index=0,
                    content_index=0,
                    text=response_text,
                )
            )
            content = [{"type": "text", "text": response_text}]
            if audio_emitted:
                content.append({"type": "audio", "transcript": response_text})
            await self.send(
                make_event(
                    "response.done",
                    response={
                        "id": response_id,
                        "object": "realtime.response",
                        "status": "completed",
                        "status_details": {"reason": finish_reason},
                        "output": [
                            {
                                "id": resp_item_id,
                                "object": "realtime.item",
                                "type": "message",
                                "role": "assistant",
                                "content": content,
                            }
                        ],
                        "usage": usage,
                    },
                )
            )
            return response_text
        finally:
            self.active_request_id = None

    async def run_transcription(self, item_id: str, audio_payload: str) -> str:
        request_id = f"rt-{self.session_id}-{uuid.uuid4().hex}"
        self.active_request_id = request_id
        try:
            text_acc: list[str] = []
            async for chunk in self.client.completion_stream(
                self.build_transcription_request(audio_payload),
                request_id=request_id,
            ):
                if chunk.modality == "text" and chunk.text:
                    text_acc.append(chunk.text)
                    await self.send(
                        make_event(
                            "conversation.item.input_audio_transcription.delta",
                            item_id=item_id,
                            content_index=0,
                            delta=chunk.text,
                        )
                    )
                if chunk.finish_reason is not None:
                    break

            transcript = "".join(text_acc)
            await self.send(
                make_event(
                    "conversation.item.input_audio_transcription.completed",
                    item_id=item_id,
                    content_index=0,
                    transcript=transcript,
                )
            )
            return transcript
        finally:
            self.active_request_id = None

    def _sampling(self) -> SamplingParams:
        max_tokens = self.session_object.max_response_output_tokens
        return SamplingParams(
            temperature=self.session_object.temperature,
            top_p=1.0,
            max_new_tokens=max_tokens if isinstance(max_tokens, int) else None,
        )

    def build_response_request(self, audio_payload: str) -> GenerateRequest:
        """Response pass: session instructions + conversation history + current audio.

        A trailing user message anchors the audio as *this turn's* user input.
        Without it Qwen3-Omni treats audio as background context and ignores it
        once any prior conversation exists, falling back to greeting on every
        turn.
        """
        messages: list[Message] = [
            Message(
                role="system",
                content=self.session_object.instructions or DEFAULT_INSTRUCTIONS,
            )
        ]
        # P4 RadixCache determinism: emit history in a fixed, deterministic
        # order and SKIP pending (not-yet-transcribed) user items. This keeps
        # the request prefix byte-identical as the conversation grows, so the
        # prefix tree hits and prefill only pays for the new suffix. Any
        # non-determinism here (ids, timestamps, reordering) would evict the
        # cache and force a full recompute.
        for item in self.conversation:
            if item.pending or not item.text:
                continue
            messages.append(Message(role=item.role, content=item.text))
        messages.append(
            Message(
                role="user",
                content="Listen to the spoken audio above and respond to it.",
            )
        )
        return GenerateRequest(
            model=self.model_name,
            messages=messages,
            sampling=self._sampling(),
            stream=True,
            output_modalities=list(self.session_object.modalities or ["text"]),
            metadata={"audios": [audio_payload]},
        )

    def build_transcription_request(self, audio_payload: str) -> GenerateRequest:
        """Transcription pass: hardcoded verbatim prompt + current audio only."""
        return GenerateRequest(
            model=self.model_name,
            messages=[
                Message(role="system", content=_TRANSCRIPTION_PROMPT),
                Message(role="user", content="Transcribe the spoken audio."),
            ],
            sampling=self._sampling(),
            stream=True,
            output_modalities=["text"],
            metadata={"audios": [audio_payload]},
        )

    async def send(self, event: dict[str, Any]) -> None:
        if self.closed:
            return
        if self.websocket.application_state != WebSocketState.CONNECTED:
            return
        event.setdefault("event_id", new_id("evt"))
        await self.websocket.send_text(json.dumps(event))

    async def send_error(self, type_: str, code: str, message: str) -> None:
        await self.send(
            make_event(
                "error",
                error={"type": type_, "code": code, "message": message},
            )
        )

    async def _cancel_and_abort(
        self, task: asyncio.Task | None, request_id: str | None
    ) -> None:
        """Abort engine request, cancel task, absorb result.

        ``asyncio.gather(..., return_exceptions=True)`` is used instead of
        ``.exception()`` because the latter re-raises ``CancelledError`` on a
        cancelled task, turning a normal disconnect into a handler exception.
        """
        if task is None or task.done():
            return
        if request_id is not None:
            await self.client.abort(request_id)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def teardown(self) -> None:
        self.closed = True
        await self._cancel_and_abort(self.active_task, self.active_request_id)
        await self._cancel_and_abort(self.queue_drainer, None)
        # P0/P3: absorb any background barge-in abort and transcription tasks.
        if self._pending_abort is not None:
            await asyncio.gather(self._pending_abort, return_exceptions=True)
            self._pending_abort = None
        for task in list(self._bg_tasks):
            task.cancel()
        if self._bg_tasks:
            await asyncio.gather(*self._bg_tasks, return_exceptions=True)
            self._bg_tasks.clear()
        if self.websocket.client_state == WebSocketState.CONNECTED:
            await self.websocket.close()
