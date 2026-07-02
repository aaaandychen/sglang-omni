#!/usr/bin/env python3
"""Test the realtime endpoint with a pre-recorded WAV file.

Runs entirely on the server — no microphone needed. Reads a 16kHz mono
WAV, streams it to ``/v1/realtime`` chunk-by-chunk (like a real client
would), and prints every server event.

Usage::

    python examples/realtime_file_test.py \
        --url ws://localhost:8080/v1/realtime \
        --wav tests/data/query_to_draw.wav

If you don't have a WAV file, any 16kHz mono PCM16 WAV works, e.g.:

    # record 5 seconds on your laptop and upload:
    ffmpeg -f avfoundation -t 5 -ar 16000 -ac 1 -sample_fmt s16 test.wav

    # or generate a test tone:
    python -c "
    import wave,struct,math
    with wave.open('/tmp/test.wav','wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(b''.join(int(16000*math.sin(2*math.pi*440*i/16000)).to_bytes(2,'little',signed=True) for i in range(16000*3)))
    "
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import wave
from pathlib import Path

CHUNK_MS = 60
SAMPLE_RATE = 16000


def load_pcm16(path: Path) -> bytes:
    """Load a 16kHz mono PCM16 WAV file as raw bytes."""
    with wave.open(str(path)) as wf:
        assert wf.getnchannels() == 1, f"Must be mono, got {wf.getnchannels()} channels"
        assert wf.getframerate() == 16000, f"Must be 16kHz, got {wf.getframerate()} Hz"
        assert wf.getsampwidth() == 2, f"Must be PCM16, got {wf.getsampwidth()} bytes"
        return wf.readframes(wf.getnframes())


async def test_realtime(
    url: str,
    pcm: bytes,
    *,
    modalities: list[str] | None = None,
    extra_silence_ms: int = 1000,
) -> None:
    """Stream PCM audio to the realtime endpoint and print all events."""
    import websockets

    chunk_bytes = SAMPLE_RATE * CHUNK_MS // 1000 * 2  # PCM16 = 2 bytes/sample
    audio = pcm + b"\x00\x00" * (SAMPLE_RATE * extra_silence_ms // 1000)

    async with websockets.connect(url, ping_interval=20) as ws:  # type: ignore[union-attr]

        # Receive session.created
        evt = json.loads(await ws.recv())
        assert evt["type"] == "session.created", f"Expected session.created, got {evt}"
        session_id = evt["session"]["id"]
        print(f"✅ Session: {session_id}")
        print(f"   modalities: {evt['session']['modalities']}")
        print(f"   input_audio_format: {evt['session']['input_audio_format']}")
        print(f"   temperature: {evt['session']['temperature']}")

        # Optionally update modalities (F1 — won't take effect until implemented)
        if modalities:
            await ws.send(json.dumps({
                "type": "session.update",
                "session": {"modalities": modalities},
            }))
            evt = json.loads(await ws.recv())
            if evt["type"] == "session.updated":
                print(f"📝 Updated modalities → {evt['session']['modalities']}")
            elif evt["type"] == "error":
                print(f"❌ Update failed: {evt.get('error', {}).get('message')}")
            else:
                print(f"⚠️  Unexpected: {evt['type']}")

        # Stream audio chunks
        print(f"\n🎵 Streaming {len(audio)} bytes of audio...")
        for i in range(0, len(audio), chunk_bytes):
            chunk = audio[i:i + chunk_bytes]
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(chunk).decode(),
            }))
            await asyncio.sleep(CHUNK_MS / 1000 * 0.8)  # simulate real-time pacing

        print("✅ Audio sent. Waiting for response...\n")

        # Receive all server events
        response_text: list[str] = []
        transcript = ""

        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=30)
            except asyncio.TimeoutError:
                print("⏰ No more events for 30s, stopping.")
                break

            evt = json.loads(raw)
            etype = evt["type"]

            # --- VAD ---
            if etype == "input_audio_buffer.speech_started":
                ms = evt.get("audio_start_ms", "?")
                print(f"🎙️  speech_started  @ {ms}ms")

            elif etype == "input_audio_buffer.speech_stopped":
                ms = evt.get("audio_end_ms", "?")
                print(f"🎙️  speech_stopped  @ {ms}ms")

            elif etype == "input_audio_buffer.committed":
                print(f"📦 buffer committed  item={evt.get('item_id', '')[:20]}...")

            # --- Response ---
            elif etype == "response.created":
                rid = evt["response"]["id"]
                print(f"\n🤖 response.created  id={rid}")

            elif etype == "response.text.delta":
                delta = evt.get("delta", "")
                print(delta, end="", flush=True)
                response_text.append(delta)

            elif etype == "response.text.done":
                full = evt.get("text", "") or "".join(response_text)
                print(f"\n📝 response.text.done  ({len(full)} chars)")

            elif etype == "response.audio.delta":
                b64 = evt.get("delta", "")
                print(f"🔊 audio.delta  {len(b64)} base64 chars")

            elif etype == "response.audio.done":
                print(f"🔊 audio.done")

            elif etype == "response.done":
                status = evt.get("response", {}).get("status", "?")
                output = evt.get("response", {}).get("output", [])
                for item in output:
                    for c in item.get("content", []):
                        if c.get("type") == "text":
                            print(f"📄 output text: {c['text'][:200]}...")
                        elif c.get("type") == "audio":
                            print(f"📄 output audio: {len(c.get('audio', ''))} chars")
                print(f"✅ response.done  status={status}")
                # For text-only mode, response.done is the last response event.
                # Wait for transcription to arrive next.

            # --- Transcription ---
            elif etype == "conversation.item.input_audio_transcription.delta":
                delta = evt.get("delta", "")
                print(f"📋 transcript delta: {delta}", end="", flush=True)

            elif etype == "conversation.item.input_audio_transcription.completed":
                transcript = evt.get("transcript", "")
                print(f"\n📋 transcript completed: {transcript}")

                # transcription.completed is the last event of the turn.
                print("\n🏁 Turn complete!")
                break

            # --- Error ---
            elif etype == "error":
                err = evt.get("error", {})
                print(f"❌ ERROR  type={err.get('type')} code={err.get('code')}")
                print(f"   message: {err.get('message')}")

            # --- Other ---
            elif etype == "session.updated":
                print(f"📝 {etype}")

            else:
                print(f"❓ {etype}  {json.dumps(evt, ensure_ascii=False)[:120]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test realtime with a WAV file")
    parser.add_argument("--url", default="ws://localhost:8080/v1/realtime")
    parser.add_argument("--wav", type=Path, default=None,
                        help="16kHz mono PCM16 WAV file")
    parser.add_argument("--modalities", nargs="*", default=None,
                        help="Session modalities, e.g. --modalities text audio")
    parser.add_argument("--silence", type=int, default=1000,
                        help="Extra trailing silence in ms (helps VAD detect speech_stopped)")
    args = parser.parse_args()

    if args.wav:
        pcm = load_pcm16(args.wav)
    else:
        # Look for the test fixture
        test_fixture = Path(__file__).parent.parent / "tests" / "data" / "query_to_draw.wav"
        if test_fixture.exists():
            print(f"Using test fixture: {test_fixture}")
            pcm = load_pcm16(test_fixture)
        else:
            print("No --wav specified and test fixture not found.")
            print("Generating a 3-second 440Hz test tone instead (won't produce meaningful response).")
            import math
            pcm = b"".join(
                int(16000 * math.sin(2 * math.pi * 440 * i / 16000))
                .to_bytes(2, "little", signed=True)
                for i in range(16000 * 3)
            )

    asyncio.run(test_realtime(
        args.url,
        pcm,
        modalities=args.modalities,
        extra_silence_ms=args.silence,
    ))


if __name__ == "__main__":
    main()
