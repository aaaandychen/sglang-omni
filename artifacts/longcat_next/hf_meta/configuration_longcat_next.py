from transformers.configuration_utils import PretrainedConfig
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
from transformers.models.whisper.configuration_whisper import WhisperConfig

from .configuration_longcat_ngram import LongcatFlashNgramConfig

class LongcatNextConfig(LongcatFlashNgramConfig):
    model_type = "longcat_next"
    def __init__(
        self,
        vocab_size=131072,
        hidden_size=6144,
        num_hidden_layers=56,
        num_layers=28,
        num_attention_heads=64,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=131072,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=10000000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        ffn_hidden_size=12288,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        head_dim=64,
        v_head_dim=128,
        qk_head_dim=None,
        moe_topk=12,
        n_routed_experts=512,
        zero_expert_num=256,
        expert_ffn_hidden_size=2048,
        routed_scaling_factor=6.0,
        emb_neighbor_num=None,
        emb_split_num=None,
        ngram_vocab_size_ratio=None,
        oe_ignored_token_ids=[],
        text_vocab_size=131072, # text vocab size (vocab_size = text_vocab_size + audio_token + visual_token + multimodal_special_token_list)
        text_vocab_plus_multimodal_special_token_size=131125,
        visual_embedding_layer_intermediate_size=8192,
        visual_embedding_layer_hidden_act="silu",
        visual_offset=150581,
        audio_offset=131125,
        visual_config={},
        audio_config={},
        **kwargs,
    ):
        self.text_vocab_size = text_vocab_size
        self.text_vocab_plus_multimodal_special_token_size = text_vocab_plus_multimodal_special_token_size
        self.visual_embedding_layer_intermediate_size = visual_embedding_layer_intermediate_size
        self.visual_embedding_layer_hidden_act = visual_embedding_layer_hidden_act
        self.visual_offset = visual_offset
        self.audio_offset = audio_offset
        self.visual_config = LongcatNextVisualConfig(**visual_config)
        self.audio_config = LongcatNextAudioConfig(**audio_config)
        oe_ignored_token_ids = oe_ignored_token_ids or list(range(self.text_vocab_size, self.text_vocab_plus_multimodal_special_token_size))

        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            ffn_hidden_size=ffn_hidden_size,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            head_dim=head_dim,
            v_head_dim=v_head_dim,
            qk_head_dim=qk_head_dim,
            moe_topk=moe_topk,
            n_routed_experts=n_routed_experts,
            zero_expert_num=zero_expert_num,
            expert_ffn_hidden_size=expert_ffn_hidden_size,
            routed_scaling_factor=routed_scaling_factor,
            emb_neighbor_num=emb_neighbor_num,
            emb_split_num=emb_split_num,
            ngram_vocab_size_ratio=ngram_vocab_size_ratio,
            oe_ignored_token_ids=oe_ignored_token_ids,
            **kwargs,
        )

class LongcatNextVisualConfig(Qwen2_5_VLVisionConfig):
    model_type = "longcat_next_visual"
    base_config_key = ""

    def __init__(
        self,
        image_start_token_id=131106,
        image_end_token_id=131107,
        image_pad_token_id=131108,
        image_newline_token_id=131109,
        vq_config={},
        visual_decoder_config={},
        **kwargs,
    ):
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.image_pad_token_id = image_pad_token_id
        self.image_newline_token_id = image_newline_token_id
        self.vq_config = PretrainedConfig(**vq_config)
        self.visual_decoder_config = PretrainedConfig(**visual_decoder_config)
        self.visual_decoder_config.image_decoder_config = PretrainedConfig(**getattr(self.visual_decoder_config, "image_decoder_config", {}))
        self.visual_decoder_config.transformer_config = PretrainedConfig(**getattr(self.visual_decoder_config, "transformer_config", {}))
        self.visual_decoder_config.vae_config = PretrainedConfig(**getattr(self.visual_decoder_config, "vae_config", {}))
        self.visual_decoder_config.scheduler_config = PretrainedConfig(**getattr(self.visual_decoder_config, "scheduler_config", {}))
        super().__init__(**kwargs)

class LongcatNextAudioConfig(WhisperConfig):
    model_type = "longcat_next_audio"
    base_config_key = ""

    def __init__(
        self,
        vq_config={},
        vocoder_config={},
        flow_matching_config={},
        cosy24kvocoder_config={},
        **kwargs
    ):
        self.vq_config = PretrainedConfig(**vq_config)
        self.vocoder_config = PretrainedConfig(**vocoder_config)
        self.flow_matching_config = PretrainedConfig(**flow_matching_config)
        self.flow_matching_config.cfm_params = PretrainedConfig(**getattr(self.flow_matching_config, "cfm_params", {}))
        self.cosy24kvocoder_config = PretrainedConfig(**cosy24kvocoder_config)
        super().__init__(**kwargs)


__all__ = ["LongcatNextConfig", "LongcatNextVisualConfig", "LongcatNextAudioConfig"]
