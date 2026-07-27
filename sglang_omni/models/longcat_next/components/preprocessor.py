# SPDX-License-Identifier: Apache-2.0
"""LongCat-Next Phase 2 preprocessing."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.models.longcat_next.components.dynamic import (
    get_longcat_config,
    get_longcat_processor,
)
from sglang_omni.models.longcat_next.payload_types import (
    AUDIO_STAGE,
    IMAGE_STAGE,
    LongcatNextPipelineState,
    longcat_timing,
    payload_with_state,
)


class LongcatNextPreprocessor:
    """Build text ids and encoder inputs from user payloads.

    The official LongCatNextProcessor expects a single string where image/audio
    paths are wrapped by model-specific special-token strings.  sglang-omni
    client payloads usually carry media paths in ``inputs['images']`` /
    ``inputs['audios']``; this preprocessor normalizes both forms.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        with longcat_timing("preprocessor_config_init"):
            self.config = get_longcat_config(model_path)
        with longcat_timing("preprocessor_processor_init"):
            self.processor = get_longcat_processor(model_path)
        self.tokenizer = self.processor.tokenizer

    def __call__(self, payload):
        with longcat_timing("preprocessor_build_text", request_id=payload.request_id):
            text = self._build_processor_text(payload.request.inputs)
        with longcat_timing(
            "preprocessor_processor_call",
            request_id=payload.request_id,
            text_len=len(text),
        ):
            text_inputs, visual_inputs, audio_inputs = self.processor(
                text,
                return_tensors="pt",
            )
        input_ids = text_inputs["input_ids"][0].to(torch.long)

        image_positions = (
            input_ids == int(self.config.visual_config.image_pad_token_id)
        ).nonzero(as_tuple=True)[0]
        audio_positions = (
            input_ids == int(self.config.audio_config.audio_pad_token_id)
        ).nonzero(as_tuple=True)[0]
        audiotext_start_positions = (
            input_ids == int(self.config.audio_config.audiotext_start_token_id)
        ).nonzero(as_tuple=True)[0]
        audiotext_pad_positions = (
            input_ids == int(self.config.audio_config.audiotext_pad_token_id)
        ).nonzero(as_tuple=True)[0]

        encoder_inputs: dict[str, dict[str, Any]] = {}
        mm_inputs: dict[str, Any] = {
            "input_ids": input_ids,
            "image_positions": image_positions,
            "audio_positions": audio_positions,
            "audiotext_start_positions": audiotext_start_positions,
            "audiotext_pad_positions": audiotext_pad_positions,
        }

        if visual_inputs is not None and image_positions.numel() > 0:
            vi = dict(visual_inputs)
            visual_grid_thw = vi.get("visual_grid_thw")
            if visual_grid_thw is None:
                visual_grid_thw = vi.get("image_grid_thw")
            encoder_inputs[IMAGE_STAGE] = {
                "pixel_values": vi.get("pixel_values"),
                "visual_grid_thw": visual_grid_thw,
                "image_positions": image_positions,
            }

        if audio_inputs is not None and audio_positions.numel() > 0:
            ai = dict(audio_inputs)
            audio = ai.get("audio")
            encoder_length = ai.get("encoder_length")
            bridge_length = ai.get("bridge_length")
            if audio is not None:
                encoder_inputs[AUDIO_STAGE] = {
                    "audio": audio,
                    "encoder_length": encoder_length,
                    "bridge_length": bridge_length,
                    "audio_positions": audio_positions,
                }

        state = LongcatNextPipelineState(
            prompt={
                "text": text,
                "input_ids": input_ids,
                "sampling_params": dict(payload.request.params or {}),
            },
            mm_inputs=mm_inputs,
            encoder_inputs=encoder_inputs,
        )
        return payload_with_state(payload, state)

    def _build_processor_text(self, inputs: Any) -> str:
        if isinstance(inputs, str):
            return inputs
        if isinstance(inputs, list):
            return self._messages_to_text(inputs)
        if not isinstance(inputs, dict):
            return str(inputs)

        if isinstance(inputs.get("text"), str):
            text = inputs["text"]
        elif isinstance(inputs.get("prompt"), str):
            text = inputs["prompt"]
        elif isinstance(inputs.get("messages"), list):
            try:
                text = self.tokenizer.apply_chat_template(
                    inputs["messages"],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except (AttributeError, ValueError):
                text = self._messages_to_text(inputs["messages"])
        else:
            text = ""

        for image in inputs.get("images") or []:
            text += f"\n{self.processor.image_start_token}{image}{self.processor.image_end_token}"
        for audio in inputs.get("audios") or []:
            text += f"\n{self.processor.audio_start_token}{audio}{self.processor.audio_end_token}"
        return text

    @staticmethod
    def _messages_to_text(messages: list[Any]) -> str:
        parts: list[str] = []
        for msg in messages:
            if isinstance(msg, str):
                parts.append(msg)
                continue
            if not isinstance(msg, dict):
                parts.append(str(msg))
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict):
                        if isinstance(item.get("text"), str):
                            parts.append(item["text"])
                        elif item.get("type") in {"image", "image_url"}:
                            value = item.get("image") or item.get("image_url")
                            if isinstance(value, dict):
                                value = value.get("url")
                            if value:
                                parts.append(str(value))
                        elif item.get("type") in {"audio", "input_audio"}:
                            value = item.get("audio") or item.get("input_audio")
                            if isinstance(value, dict):
                                value = value.get("url") or value.get("path")
                            if value:
                                parts.append(str(value))
            else:
                parts.append(str(content))
        return "\n".join(parts)
