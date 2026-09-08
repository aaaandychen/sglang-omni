# SPDX-License-Identifier: Apache-2.0
"""Unit tests for realtime session barge-in/context handling (CPU-only).

Covers the P1 fix (barge-in must not drop the interrupted turn's user
transcription or leave a pending slot stuck), the background-transcription
request-id isolation, the barge-in abort-barrier ordering (Q-1), and the
response.done audio content contract.
"""

from __future__ import annotations

import asyncio
import json
import types

import pytest

import sglang_omni.serve.realtime.session as session_mod
from sglang_omni.serve.realtime.session import RealtimeSession
from sglang_omni.serve.realtime.vad import VADEvent


class _FakeWebSocket:
    """Minimal WebSocket stand-in recording every sent event."""

    def __init__(self) -> None:
        from starlette.websockets import WebSocketState

        self.application_state = WebSocketState.CONNECTED
        self.client_state = WebSocketState.CONNECTED
        self.events: list[dict] = []

    async def send_text(self, text: str) -> None:
        self.events.append(json.loads(text))

    async def close(self) -> None:
        from starlette.websockets import WebSocketState

        self.client_state = WebSocketState.DISCONNECTED


def _chunk(text=None, audio=None, finish=None):
    return types.SimpleNamespace(
        modality="audio" if audio is not None else "text",
        text=text,
        audio_b64=audio,
        finish_reason=finish,
        usage=None,
    )


class _FakeClient:
    """Routes completion_stream by prompt: transcription vs response."""

    def __init__(self, *, response_chunks, transcript_chunks) -> None:
        self._response_chunks = response_chunks
        self._transcript_chunks = transcript_chunks
        self.aborted: list[str] = []

    async def completion_stream(self, req, request_id=None, audio_format=None):
        is_transcription = "speech-to-text" in str(req.messages[0].content)
        chunks = self._transcript_chunks if is_transcription else self._response_chunks
        for chunk in chunks:
            await asyncio.sleep(0)  # yield so cancellation can land mid-stream
            yield chunk

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)
        await asyncio.sleep(0)


def _make_session(client) -> RealtimeSession:
    """Build a session without loading the silero VAD model."""

    class _DummyVAD:
        def __init__(self, config=None) -> None:
            self.config = types.SimpleNamespace(
                threshold=None, prefix_padding_ms=None, silence_duration_ms=None
            )
            self.effective_silence_ms = 300
            self.is_speech = False

        def reset(self) -> None:
            pass

    original = session_mod.StreamingVAD
    session_mod.StreamingVAD = _DummyVAD
    try:
        return RealtimeSession(
            _FakeWebSocket(), client=client, model_name="fake-model"
        )
    finally:
        session_mod.StreamingVAD = original


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# P1: barge-in mid-response must not lose the turn's context
# ---------------------------------------------------------------------------


def test_barge_in_mid_response_preserves_user_transcript_and_partial_reply():
    async def scenario():
        # Response streams forever (never finishes) so we can cancel mid-way.
        response_chunks = [_chunk(text=f"w{i}") for i in range(50)]
        client = _FakeClient(
            response_chunks=response_chunks,
            transcript_chunks=[_chunk(text="hello ", ), _chunk(text="world", finish="stop")],
        )
        sess = _make_session(client)

        turn = asyncio.create_task(sess.run_turn("item_1", "data:audio/wav;base64,AAA"))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        # Barge-in: cancel the in-flight turn (as _cancel_and_abort does).
        turn.cancel()
        await asyncio.gather(turn, return_exceptions=True)
        # Let the background transcription finish.
        for task in list(sess._bg_tasks):
            await asyncio.gather(task, return_exceptions=True)

        return sess

    sess = _run(scenario())
    roles = [(item.role, item.text, item.pending) for item in sess.conversation]
    assert ("user", "hello world", False) in roles, roles
    assistant = [r for r in roles if r[0] == "assistant"]
    assert assistant and assistant[0][1].startswith("w0"), roles
    # No slot may stay pending forever.
    assert all(not item.pending for item in sess.conversation)


def test_normal_turn_still_fills_history():
    async def scenario():
        client = _FakeClient(
            response_chunks=[_chunk(text="hi", finish="stop")],
            transcript_chunks=[_chunk(text="question", finish="stop")],
        )
        sess = _make_session(client)
        await sess.run_turn("item_1", "data:audio/wav;base64,AAA")
        for task in list(sess._bg_tasks):
            await asyncio.gather(task, return_exceptions=True)
        return sess

    sess = _run(scenario())
    roles = [(item.role, item.text, item.pending) for item in sess.conversation]
    assert roles == [("user", "question", False), ("assistant", "hi", False)]


# ---------------------------------------------------------------------------
# Background transcription must not clobber the live response's request id
# ---------------------------------------------------------------------------


def test_transcription_does_not_clobber_active_request_id_and_aborts_own():
    async def scenario():
        client = _FakeClient(
            response_chunks=[],
            transcript_chunks=[_chunk(text=f"t{i}") for i in range(50)],
        )
        sess = _make_session(client)
        sess.active_request_id = "live-response-req"

        task = asyncio.create_task(sess.run_transcription("item_1", "payload"))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return sess, client

    sess, client = _run(scenario())
    assert sess.active_request_id == "live-response-req"
    # The cancelled transcription aborted its OWN engine request.
    assert len(client.aborted) == 1
    assert client.aborted[0] != "live-response-req"


# ---------------------------------------------------------------------------
# Q-1: barge-in installs the abort barrier before yielding to the loop
# ---------------------------------------------------------------------------


def test_barge_in_sets_abort_barrier_before_events():
    async def scenario():
        client = _FakeClient(response_chunks=[], transcript_chunks=[])
        sess = _make_session(client)
        # Pretend a response is in flight.
        sess.active_task = asyncio.create_task(asyncio.sleep(60))
        sess.active_request_id = "req-old"

        emit = types.SimpleNamespace(event_type=VADEvent.SPEECH_STARTED, sample_offset=0)
        await sess.handle_vad_emit(emit)

        barrier_set = sess._pending_abort is not None
        event_types = [e["type"] for e in sess.websocket.events]
        # Cleanup.
        if sess._pending_abort is not None:
            await asyncio.gather(sess._pending_abort, return_exceptions=True)
        return barrier_set, event_types

    barrier_set, event_types = _run(scenario())
    assert barrier_set, "abort barrier must exist right after handle_vad_emit"
    assert event_types[:2] == [
        "input_audio_buffer.speech_started",
        "response.audio.flush",
    ]


# ---------------------------------------------------------------------------
# response.done audio content contract (OpenAI Realtime alignment)
# ---------------------------------------------------------------------------


def test_response_done_audio_item_carries_audio_field():
    async def scenario():
        client = _FakeClient(
            response_chunks=[_chunk(text="hello"), _chunk(audio="QUJD"), _chunk(finish="stop")],
            transcript_chunks=[],
        )
        sess = _make_session(client)
        sess.session_object.modalities = ["text", "audio"]
        await sess.run_response("data:audio/wav;base64,AAA")
        return sess.websocket.events

    events = _run(scenario())
    done = next(e for e in events if e["type"] == "response.done")
    content = done["response"]["output"][0]["content"]
    audio_parts = [c for c in content if c["type"] == "audio"]
    assert audio_parts, content
    assert "audio" in audio_parts[0], audio_parts
    assert audio_parts[0]["transcript"] == "hello"


def test_teardown_marks_pending_slots_done_without_new_engine_work():
    async def scenario():
        response_chunks = [_chunk(text=f"w{i}") for i in range(50)]
        client = _FakeClient(response_chunks=response_chunks, transcript_chunks=[])
        sess = _make_session(client)
        turn = asyncio.create_task(sess.run_turn("item_1", "payload"))
        await asyncio.sleep(0)
        sess.closed = True  # teardown began before the cancel lands
        turn.cancel()
        await asyncio.gather(turn, return_exceptions=True)
        return sess

    sess = _run(scenario())
    assert all(not item.pending for item in sess.conversation)
    assert not sess._bg_tasks


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
