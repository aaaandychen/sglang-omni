#!/usr/bin/env python3
"""Save streaming audio chunks from sglang-omni text+audio output.

Usage:
  curl -s -N http://localhost:8000/v1/chat/completions ... | python save_audio_stream.py
  python save_audio_stream.py --input response.json  # non-streaming mode
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import wave
from pathlib import Path


def _sanitize_filename(text: str, max_len: int = 60) -> str:
    """Clean text for use as a filename."""
    text = text.strip()
    # Remove non-filename-safe characters
    text = re.sub(r'[<>:"/\\|?*\n\r\t]', "", text)
    text = re.sub(r"\s+", "_", text)
    if len(text) > max_len:
        text = text[:max_len]
    return text or "audio"


def _save_wav(data_b64: str, path: str) -> None:
    """Decode base64-encoded WAV data and save to file."""
    raw = base64.b64decode(data_b64)
    with open(path, "wb") as f:
        f.write(raw)


def process_stream(stream_lines, output_dir: str = "."):
    """Process SSE stream chunks.  Each ``audio`` delta is saved as a
    separate WAV file named by the text content that accompanies it."""
    os.makedirs(output_dir, exist_ok=True)
    chunk_idx = 0
    pending_audio = None

    for line in stream_lines:
        line = line.strip()
        if not line.startswith("data: "):
            continue
        data_str = line[6:]
        if data_str == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        choices = chunk.get("choices", [])
        if not choices:
            continue
        delta = choices[0].get("delta", {})
        text = delta.get("content", "")
        audio = delta.get("audio")

        if audio and audio.get("data"):
            # New audio delta — if we already have one pending, flush it.
            if pending_audio:
                _save_wav(pending_audio["data"], pending_audio["path"])
            fname = _sanitize_filename(text) if text else f"chunk_{chunk_idx:04d}"
            path = os.path.join(output_dir, f"{fname}.wav")
            pending_audio = {"data": audio["data"], "path": path}
            chunk_idx += 1

    # Flush the last pending audio.
    if pending_audio:
        _save_wav(pending_audio["data"], pending_audio["path"])

    print(f"Saved {chunk_idx} audio file(s) to {os.path.abspath(output_dir)}")


def process_non_stream(response_file: str, output_dir: str = "."):
    """Process a non-streaming JSON response."""
    with open(response_file) as f:
        data = json.load(f)
    choices = data.get("choices", [])
    if not choices:
        print("No audio in response.")
        return
    msg = choices[0].get("message", {})
    audio = msg.get("audio")
    text = msg.get("content", "")
    if not audio or not audio.get("data"):
        print("No audio in response.")
        return
    os.makedirs(output_dir, exist_ok=True)
    fname = _sanitize_filename(text) if text else "audio_output"
    path = os.path.join(output_dir, f"{fname}.wav")
    _save_wav(audio["data"], path)
    print(f"Saved to {os.path.abspath(path)}")


def main():
    parser = argparse.ArgumentParser(description="Save sglang-omni audio stream chunks")
    parser.add_argument(
        "--input", "-i",
        help="Non-streaming JSON response file",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="audio_outputs",
        help="Output directory (default: audio_outputs)",
    )
    args = parser.parse_args()

    if args.input:
        process_non_stream(args.input, args.output_dir)
    else:
        process_stream(sys.stdin, args.output_dir)


if __name__ == "__main__":
    main()
