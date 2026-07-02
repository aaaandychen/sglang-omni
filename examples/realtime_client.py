#!/usr/bin/env python3
"""Minimal realtime voice client — mic in, text/audio out.

Connects to ``/v1/realtime``, streams mic audio (PCM16 16kHz mono),
and prints text responses as they arrive. When the server sends audio
deltas they are queued for playback.

Usage::

    # Install deps (once):
    pip install pyaudio websockets sounddevice numpy

    # Run against local or remote server:
    python examples/realtime_client.py --url ws://10.0.0.5:8080/v1/realtime

    # Speak into your mic. Press Ctrl-C to stop.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import signal
import sys
import threading
from collections import deque
from typing import Any

# ---------------------------------------------------------------------------
# Audio capture (pyaudio) — runs in a thread, pushes chunks to a queue.
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_MS = 60  # send a chunk every 60 ms
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("realtime_client")


class Microphone:
    """Non-blocking mic capture. Call ``start()``, read from ``queue``."""

    def __init__(self) -> None:
        import pyaudio

        self.pa = pyaudio.PyAudio()
        self.queue: deque[bytes] = deque()
        self._running = False
        self._stream: Any = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._running = True
        self._stream = self.pa.open(
            format=self.pa.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SAMPLES,
        )
        self._thread = threading.Thread(target=self._capture, daemon=True)
        self._thread.start()
        logger.info("🎤 Microphone started (16kHz mono)")

    def _capture(self) -> None:
        while self._running:
            try:
                data = self._stream.read(CHUNK_SAMPLES, exception_on_overflow=False)
                self.queue.append(data)
            except Exception:
                break

    def stop(self) -> None:
        self._running = False
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
        self.pa.terminate()
        logger.info("🎤 Microphone stopped")


def play_audio_pcm16(audio_b64: str) -> None:
    """Play a base64 WAV audio chunk. Best-effort, non-blocking."""
    try:
        import io
        import wave

        import sounddevice as sd

        raw = base64.b64decode(audio_b64)
        with wave.open(io.BytesIO(raw)) as wf:
            data = wf.readframes(wf.getnframes())
            sd.play(data, samplerate=wf.getframerate(), blocking=False)
    except Exception:
        pass  # playback is best-effort


# ---------------------------------------------------------------------------
# WebSocket session
# ---------------------------------------------------------------------------
async def realtime_session(url: str, verbose: bool = False) -> None:
    import websockets

    mic = Microphone()
    shutdown_event = asyncio.Event()

    def _on_sigint() -> None:
        logger.info("Shutting down...")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, _on_sigint)

    async with websockets.connect(url, ping_interval=20) as ws:  # type: ignore[union-attr]
        logger.info("🔗 Connected to %s", url)

        # ---- task: mic → server ----
        async def send_audio() -> None:
            mic.start()
            try:
                while not shutdown_event.is_set():
                    try:
                        chunk = mic.queue.popleft()
                    except IndexError:
                        await asyncio.sleep(0.01)
                        continue
                    payload = {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode(),
                    }
                    await ws.send(json.dumps(payload))
            finally:
                mic.stop()

        sender = asyncio.create_task(send_audio())

        # ---- task: server → terminal / speakers ----
        text_buffer: list[str] = []
        audio_chunks: list[str] = []

        try:
            while not shutdown_event.is_set():
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue

                evt = json.loads(raw)
                etype = evt.get("type", "?")

                if verbose:
                    # compact dump for debugging
                    brief = {k: v for k, v in evt.items() if k not in ("audio",)}
                    logger.debug("← %s", json.dumps(brief, ensure_ascii=False))

                # --- session lifecycle ---
                if etype == "session.created":
                    sid = evt["session"]["id"]
                    logger.info("✅ Session created: %s", sid)

                elif etype == "session.updated":
                    mods = evt["session"].get("modalities", [])
                    logger.info("📝 Session updated, modalities=%s", mods)

                # --- VAD ---
                elif etype == "input_audio_buffer.speech_started":
                    logger.info("🎙️  Speech started")

                elif etype == "input_audio_buffer.speech_stopped":
                    logger.info("🎙️  Speech stopped (committing...)")

                # --- response text ---
                elif etype == "response.created":
                    logger.info("🤖 Assistant is thinking...")

                elif etype == "response.text.delta":
                    delta = evt.get("delta", "")
                    print(delta, end="", flush=True)
                    text_buffer.append(delta)

                elif etype == "response.text.done":
                    print()  # newline after streaming
                    full = evt.get("text", "") or "".join(text_buffer)
                    logger.info("📝 Full response: %s", full[:200])
                    text_buffer.clear()

                # --- response audio ---
                elif etype == "response.audio.delta":
                    audio_b64 = evt.get("delta", "")
                    if audio_b64:
                        audio_chunks.append(audio_b64)
                        play_audio_pcm16(audio_b64)
                        logger.debug("🔊 Audio delta: %d chars", len(audio_b64))

                elif etype == "response.audio.done":
                    logger.info("🔊 Audio done (%d chunks)", len(audio_chunks))
                    audio_chunks.clear()

                # --- response final ---
                elif etype == "response.done":
                    status = evt.get("response", {}).get("status", "?")
                    logger.info("✅ Response done (status=%s)", status)

                # --- transcription ---
                elif etype == "conversation.item.input_audio_transcription.completed":
                    transcript = evt.get("transcript", "")
                    logger.info("📋 Transcript: %s", transcript)

                # --- error ---
                elif etype == "error":
                    err = evt.get("error", {})
                    logger.error("❌ Error: %s", err.get("message", str(err)))

                # --- catch-all ---
                else:
                    if verbose:
                        logger.debug("← %s", etype)

        except websockets.exceptions.ConnectionClosed as exc:
            logger.info("🔌 Connection closed: %s", exc)
        finally:
            shutdown_event.set()
            sender.cancel()
            try:
                await sender
            except asyncio.CancelledError:
                pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime voice client")
    parser.add_argument(
        "--url",
        default="ws://localhost:8080/v1/realtime",
        help="WebSocket URL (default: ws://localhost:8080/v1/realtime)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Log every received event",
    )
    args = parser.parse_args()

    print(f"Connecting to {args.url} ...")
    print("Speak into your microphone. Press Ctrl-C to stop.\n")

    try:
        asyncio.run(realtime_session(args.url, verbose=args.verbose))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
