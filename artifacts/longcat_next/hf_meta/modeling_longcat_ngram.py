# -*- coding: utf-8 -*-
# Copyright (c) 2026 Meituan
# This code is licensed under the MIT License, for details, see the ./LICENSE file.

from typing import Optional, Tuple, Dict, List

import torch
from torch import nn
import torch.nn.functional as F

from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import auto_docstring, logging
from transformers.models.longcat_flash.modeling_longcat_flash import (
    LongcatFlashForCausalLM,
    LongcatFlashModel,
    LongcatFlashRMSNorm,
    LongcatFlashRotaryEmbedding,
    LongcatFlashDecoderLayer,
    LongcatFlashPreTrainedModel,
)
from .configuration_longcat_ngram import LongcatFlashNgramConfig

logger = logging.get_logger(__name__)


@auto_docstring
class LongcatFlashNgramPreTrainedModel(LongcatFlashPreTrainedModel):
    pass


class NgramCache(DynamicCache):
    """
    Extended DynamicCache for storing N-gram context alongside KV cache.
    """
    def __init__(self, config=None):
        super().__init__()
        self.ngram_context = None
        # Keep only n-1 tokens (minimum needed for N-gram computation)
        self.max_context_len = config.emb_neighbor_num - 1
        self.oe_ignored_token_ids = torch.tensor(config.oe_ignored_token_ids, dtype=torch.long)


    def update_ngram_context(self, new_tokens: torch.Tensor) -> None:
        """
        Update N-gram context with window management.

        Args:
            new_tokens: New tokens to append, shape (batch_size, seq_len)
        """
        new_tokens = new_tokens.clone()
        new_tokens[torch.isin(new_tokens, self.oe_ignored_token_ids.to(new_tokens.device))] = 0

        if self.ngram_context is None:
            self.ngram_context = new_tokens
        else:
            self.ngram_context = torch.cat([self.ngram_context, new_tokens], dim=-1)

        # Truncate to maintain constant memory footprint
        if self.ngram_context.size(-1) > self.max_context_len:
            self.ngram_context = self.ngram_context[..., -self.max_context_len:]

    def reorder_cache(self, beam_idx: torch.LongTensor) -> "Cache":
        """Reorder cache for beam search."""
        # Reorder parent's KV cache
        super().reorder_cache(beam_idx)

        # Reorder N-gram context
        if self.ngram_context is not None:
            self.ngram_context = self.ngram_context.index_select(0, beam_idx.to(self.ngram_context.device))

        return self


class EmbeddingWithMask(nn.Embedding):
    def forward(self, input: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input indices of shape (batch_size, seq_len)
            mask (torch.Tensor): Boolean mask of shape (batch_size, seq_len).
                                 True means compute, False means skip and return 0.
        Returns:
            torch.Tensor: Embeddings of shape (batch_size, seq_len, embedding_dim)
        """
        if mask is not None:
            # Ensure mask is boolean
            mask = mask.bool()
        else:
            mask = torch.ones_like(input, dtype=torch.bool)

        batch_size, seq_len = input.shape
        embedding_dim = self.embedding_dim

        # 1. Initialize the output tensor with zeros on the correct device
        output = torch.zeros(
            (batch_size, seq_len, embedding_dim),
            device=input.device,
            dtype=self.weight.dtype
        )

        # 2. Filter out the valid indices using the mask
        # valid_indices is a 1D tensor containing only the elements where mask is True
        valid_indices = input[mask]

        # 3. Only perform the embedding lookup if there is at least one valid index
        if valid_indices.numel() > 0:
            # Look up only the necessary embeddings (saves compute/memory bandwidth)
            valid_embeddings = F.embedding(
            valid_indices, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)

            # 4. Scatter the valid embeddings back to their original positions in the output tensor
            output[mask] = valid_embeddings

        return output


class NgramEmbedding(nn.Module):
    """
    Computes embeddings enriched with N-gram features without maintaining internal state.
    """
    def __init__(self, config, base_embeddings):
        super().__init__()
        self.config = config
        self.word_embeddings = base_embeddings

        # self.m = config.ngram_vocab_size_ratio * config.vocab_size
        self.m = config.ngram_vocab_size_ratio * config.text_vocab_size
        self.k = config.emb_split_num
        self.n = config.emb_neighbor_num
        self.oe_ignored_token_ids = torch.tensor(config.oe_ignored_token_ids)

        self._init_ngram_embeddings()
        self._vocab_mods_cache = None

    def _init_ngram_embeddings(self) -> None:
        """Initialize N-gram embedding and projection layers."""
        num_embedders = self.k * (self.n - 1)
        emb_dim = self.config.hidden_size // num_embedders

        embedders = []
        post_projs = []

        for i in range(num_embedders):
            vocab_size = int(self.m + i * 2 + 1)
            emb = EmbeddingWithMask(vocab_size, emb_dim, padding_idx=self.config.pad_token_id)
            proj = nn.Linear(emb_dim, self.config.hidden_size, bias=False)
            embedders.append(emb)
            post_projs.append(proj)

        self.embedders = nn.ModuleList(embedders)
        self.post_projs = nn.ModuleList(post_projs)

    def _shift_right_ignore_eos(self, tensor: torch.Tensor, n: int, eos_token_id: int = 2) -> torch.Tensor:
        p, q = tensor.shape
        # special_token / modal set 0
        special_tokens = 0

        if n == 0:
            return tensor.clone()

        if n >= q:
            return torch.zeros_like(tensor)

        result = torch.zeros_like(tensor)

        # Find all special_token/modal/EOS locations
        special_mask = (tensor == special_tokens)
        total_mask = (tensor == eos_token_id) | special_mask

        # Calculate the segment ID to which each position belongs
        eos_cumsum = total_mask.long().cumsum(dim=1)
        # Shift right by 1, so that the first EOS position still belongs to segment 0, and the second EOS position belongs to segment 1
        segment_ids = torch.cat([
            torch.zeros(p, 1, dtype=torch.long, device=tensor.device),
            eos_cumsum[:, :-1]
        ], dim=1)

        col_indices = torch.arange(q, device=tensor.device).unsqueeze(0).expand(p, q)
        # Number of segments
        max_segments = segment_ids.max().item() + 1
        segment_starts = torch.full((p, max_segments), q, dtype=torch.long, device=tensor.device)
        # Calculate the starting position of each segment
        segment_starts.scatter_reduce_(1, segment_ids, col_indices, reduce='amin', include_self=False)

        # Get the start position of the segment to which each position belongs
        segment_start_per_pos = torch.gather(segment_starts, 1, segment_ids)

        # Calculate the offset of each position within the segment
        offset_in_segment = col_indices - segment_start_per_pos

        # Data for each position should be taken from the position offset -n within the segment
        source_offset = offset_in_segment - n
        valid_mask = source_offset >= 0

        # Calculate the actual source index
        source_indices = segment_start_per_pos + torch.clamp(source_offset, min=0)

        # Data is collected by source_indices
        result = torch.gather(tensor, 1, source_indices)

        # Set invalid position to zero
        result = result * valid_mask * (~special_mask)

        return result

    def _precompute_vocab_mods(self) -> Dict[Tuple[int, int], List[int]]:
        """Precompute modular arithmetic values for vocabulary."""
        if self._vocab_mods_cache is not None:
            return self._vocab_mods_cache

        vocab_mods = {}
        vocab_size = self.config.text_vocab_size

        for i in range(2, self.n + 1):
            for j in range(self.k):
                index = (i - 2) * self.k + j
                emb_vocab_dim = int(self.m + index * 2 + 1)

                mods = []
                power_mod = 1
                for _ in range(i - 1):
                    power_mod = (power_mod * vocab_size) % emb_vocab_dim
                    mods.append(power_mod)

                vocab_mods[(i, j)] = mods

        self._vocab_mods_cache = vocab_mods
        return vocab_mods

    def _get_ngram_ids(
        self,
        input_ids: torch.Tensor,
        shifted_ids: Dict[int, torch.Tensor],
        vocab_mods: List[int],
        ngram: int
    ) -> torch.Tensor:
        """Compute N-gram hash IDs using polynomial rolling hash."""
        ngram_ids = input_ids.clone()
        for k in range(2, ngram + 1):
            ngram_ids = ngram_ids + shifted_ids[k] * vocab_mods[k - 2]
        return ngram_ids

    def forward(
        self,
        input_ids: torch.Tensor,
        ngram_context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Stateless forward pass.

        Args:
            input_ids: Current input token IDs of shape (batch_size, seq_len)
            ngram_context: Optional historical context of shape (batch_size, context_len)

        Returns:
            Embedding tensor of shape (batch_size, seq_len, hidden_size)
        """
        seq_len = input_ids.size(-1)

        # Determine complete context
        if ngram_context is not None:
            context = torch.cat([ngram_context[..., -(self.n-1):], input_ids], dim=-1)
        else:
            context = input_ids.clone()

        # Skip N-gram look-up for oe_ignored_token_ids
        oe_ignored_mask = torch.isin(input_ids, self.oe_ignored_token_ids.to(device=input_ids.device))
        context[torch.isin(context, self.oe_ignored_token_ids.to(device=context.device))] = 0

        # Base word embeddings
        device = self.word_embeddings.weight.device
        x = self.word_embeddings(input_ids.to(device)).clone()

        # Precompute modular values
        vocab_mods = self._precompute_vocab_mods()

        # Compute shifted IDs
        shifted_ids = {}
        for i in range(2, self.n + 1):
            shifted_ids[i] = self._shift_right_ignore_eos(
                context, i - 1, eos_token_id=self.config.eos_token_id
            )

        # Add N-gram embeddings
        for i in range(2, self.n + 1):
            for j in range(self.k):
                index = (i - 2) * self.k + j
                emb_vocab_dim = int(self.m + index * 2 + 1)

                ngram_ids = self._get_ngram_ids(context, shifted_ids, vocab_mods[(i, j)], ngram=i)
                new_ids = (ngram_ids % emb_vocab_dim)[..., -seq_len:]
                text_mask = new_ids > 0

                embedder_device = self.embedders[index].weight.device
                x_ngram = self.embedders[index](new_ids.to(embedder_device), text_mask)

                proj_device = self.post_projs[index].weight.device
                x_proj = self.post_projs[index](x_ngram.to(proj_device))
                x = x + x_proj.to(x.device)

        # Normalize
        x[~oe_ignored_mask] /= (1 + self.k * (self.n - 1))

        return x


class LongcatFlashNgramModel(LongcatFlashModel):
    """LongcatFlash model with N-gram enhanced embeddings."""
    _keys_to_ignore_on_load_unexpected = [r"model\.mtp.*"]
    config_class = LongcatFlashNgramConfig

    def __init__(self, config):
        super().__init__(config)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.ngram_embeddings = NgramEmbedding(config, self.embed_tokens)

        self.layers = nn.ModuleList(
            [LongcatFlashDecoderLayer(config, layer_idx) for layer_idx in range(config.num_layers)]
        )

        self.head_dim = config.head_dim
        self.config.num_hidden_layers = 2 * config.num_layers
        self.norm = LongcatFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LongcatFlashRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # Extract N-gram context if available
        ngram_context = None
        if isinstance(past_key_values, NgramCache) and past_key_values.ngram_context is not None:
            ngram_context = past_key_values.ngram_context

        if inputs_embeds is None:
            inputs_embeds = self.ngram_embeddings(input_ids, ngram_context=ngram_context)

        # Initialize NgramCache if needed
        if use_cache and past_key_values is None:
            past_key_values = NgramCache(config=self.config)

        # Update N-gram context
        if use_cache and isinstance(past_key_values, NgramCache) and input_ids is not None:
            past_key_values.update_ngram_context(input_ids)

        # Prepare cache position
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                inputs_embeds.shape[1], device=inputs_embeds.device
            ) + past_seen_tokens

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Create causal mask
        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        # Forward through decoder layers
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )


class LongcatFlashNgramForCausalLM(LongcatFlashForCausalLM):
    """LongcatFlash model for causal language modeling with N-gram embeddings."""
    _keys_to_ignore_on_load_unexpected = [r"model\.mtp.*"]
    config_class = LongcatFlashNgramConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = LongcatFlashNgramModel(config)

    @torch.no_grad()
    def generate(self, inputs=None, generation_config=None, **kwargs):
        """Override to ensure NgramCache is used."""

        if "past_key_values" not in kwargs or kwargs["past_key_values"] is None:
            kwargs["past_key_values"] = NgramCache(config=self.config)

        return super().generate(inputs=inputs, generation_config=generation_config, **kwargs)

__all__ = ["LongcatFlashNgramPreTrainedModel", "LongcatFlashNgramModel", "LongcatFlashNgramForCausalLM"]
