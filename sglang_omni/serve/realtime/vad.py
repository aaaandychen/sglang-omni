from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from silero_vad import load_silero_vad

logger = logging.getLogger(__name__)

# silero-vad operates on 512-sample windows @ 16 kHz (32 ms each).
VAD_FRAME_SAMPLES = 512
VAD_SAMPLE_RATE = 16000


@dataclass
class VADConfig:
    """Mirrors OpenAI Realtime ``turn_detection`` (server_vad mode)."""

    # Probs greater than threshold are considered speech
    threshold: float = 0.5
    # Prefix padding in milliseconds
    prefix_padding_ms: int = 300
    # Silence duration in milliseconds before declaring end-of-turn. Lowered
    # from 500 → 300 (P2): silero-vad end-pointing is accurate enough at 300ms
    # in quiet conditions, directly cutting ~200ms off TTFA. Adaptive tuning
    # (see ``adaptive``) nudges this between the bounds below at runtime.
    silence_duration_ms: int = 300
    # P2 adaptive end-pointing: when enabled, the effective silence duration
    # auto-adjusts within [min, max] based on recent false-cut rate. A false
    # cut = speech resumes very soon after a speech_stopped (we cut mid-turn).
    adaptive: bool = True
    silence_min_ms: int = 200
    silence_max_ms: int = 700
    # A speech_started within this window after a speech_stopped counts as a
    # false cut and pushes the silence duration up.
    false_cut_window_ms: int = 400
    # Step by which the effective silence duration moves per adaptation.
    adapt_step_ms: int = 50


class VADEvent:
    SPEECH_STARTED = "speech_started"
    SPEECH_STOPPED = "speech_stopped"


@dataclass
class Emit:
    event_type: str
    sample_offset: int


class StreamingVAD:
    """Per-session frame-by-frame VAD state machine.

    Callers feed raw PCM16 LE mono @ 16 kHz via :meth:`process`. The
    wrapper buffers up to one frame's worth of leftover bytes between
    calls so the caller doesn't have to align to 32 ms.
    """

    def __init__(self, config: VADConfig | None = None) -> None:
        self.config = config or VADConfig()
        self.vad_model = load_silero_vad(onnx=True)
        self.leftover_pcm = bytearray()
        self.samples_consumed = 0
        self.is_speech = False
        self.silence_run_samples = 0
        self.last_speech_offset = 0
        # P2 adaptive end-pointing state. ``effective_silence_ms`` starts at the
        # configured value and drifts within [min, max]. ``last_stop_sample``
        # records where we last declared end-of-turn so a quick speech resume
        # can be detected as a false cut.
        self.effective_silence_ms = self.config.silence_duration_ms
        self.last_stop_sample: int | None = None

    def process(self, pcm_bytes: bytes) -> list[Emit]:
        """Feed PCM16 LE mono @ 16 kHz; return any state transitions."""
        if not pcm_bytes:
            return []
        self.leftover_pcm.extend(pcm_bytes)
        emits: list[Emit] = []

        while len(self.leftover_pcm) >= VAD_FRAME_SAMPLES * 2:
            frame_bytes = bytes(self.leftover_pcm[: VAD_FRAME_SAMPLES * 2])
            del self.leftover_pcm[: VAD_FRAME_SAMPLES * 2]
            frame = np.frombuffer(frame_bytes, dtype="<i2").astype(np.float32) / 32768.0

            prob = self.infer(frame)
            self.samples_consumed += VAD_FRAME_SAMPLES
            speech = prob >= self.config.threshold

            if speech:
                self.silence_run_samples = 0
                self.last_speech_offset = self.samples_consumed
                if not self.is_speech:
                    self.is_speech = True
                    # P2 adaptive: if speech resumes very soon after we cut the
                    # previous turn, that was a false cut — lengthen the silence
                    # window so we stop chopping mid-utterance.
                    if self.config.adaptive and self.last_stop_sample is not None:
                        gap_ms = (
                            (self.samples_consumed - self.last_stop_sample)
                            * 1000
                            // VAD_SAMPLE_RATE
                        )
                        if gap_ms <= self.config.false_cut_window_ms:
                            self.effective_silence_ms = min(
                                self.config.silence_max_ms,
                                self.effective_silence_ms
                                + self.config.adapt_step_ms,
                            )
                        else:
                            # Clean turn boundary — we can afford to be snappier.
                            self.effective_silence_ms = max(
                                self.config.silence_min_ms,
                                self.effective_silence_ms
                                - self.config.adapt_step_ms,
                            )
                    # OpenAI's contract: speech_started reports the start
                    # offset *minus* prefix_padding so the caller includes
                    # a leading prefix in the committed audio.
                    pad = self.config.prefix_padding_ms * VAD_SAMPLE_RATE // 1000
                    started_at = max(0, self.samples_consumed - VAD_FRAME_SAMPLES - pad)
                    emits.append(
                        Emit(
                            event_type=VADEvent.SPEECH_STARTED, sample_offset=started_at
                        )
                    )
            else:
                self.silence_run_samples += VAD_FRAME_SAMPLES
                if self.is_speech:
                    silence_threshold = (
                        self.effective_silence_ms * VAD_SAMPLE_RATE // 1000
                    )
                    if self.silence_run_samples >= silence_threshold:
                        self.is_speech = False
                        self.last_stop_sample = self.samples_consumed
                        emits.append(
                            Emit(
                                event_type=VADEvent.SPEECH_STOPPED,
                                sample_offset=self.last_speech_offset,
                            )
                        )

        return emits

    def infer(self, frame: np.ndarray) -> float:
        with torch.inference_mode():
            tensor = torch.from_numpy(frame).unsqueeze(0)
            prob = self.vad_model(tensor, VAD_SAMPLE_RATE).item()
        return float(prob)

    def reset(self) -> None:
        self.leftover_pcm.clear()
        self.samples_consumed = 0
        self.is_speech = False
        self.silence_run_samples = 0
        self.last_speech_offset = 0
        # Keep the learned ``effective_silence_ms`` across resets (it's a
        # session-level adaptation), but drop the stop marker since sample
        # offsets restart after a buffer drop.
        self.last_stop_sample = None
        if hasattr(self.vad_model, "reset_states"):
            self.vad_model.reset_states()  # type: ignore[union-attr]


def offsets_to_ms(samples: int) -> int:
    return samples * 1000 // VAD_SAMPLE_RATE


def emits_for_test(pcm_bytes: bytes, **cfg) -> list[tuple[str, int]]:
    """Test helper: drive the VAD on a complete byte buffer."""
    vad = StreamingVAD(VADConfig(**cfg))
    emits = vad.process(pcm_bytes)
    return [(e.event_type, offsets_to_ms(e.sample_offset)) for e in emits]
