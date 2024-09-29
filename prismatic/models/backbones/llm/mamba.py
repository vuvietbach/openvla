"""
llama2.py

Class definition for all LLMs derived from LlamaForCausalLM.
"""

from typing import Optional, Sequence, Type

import torch
from torch import nn as nn
from transformers.models.mamba2.modeling_mamba2 import Mamba2Model, Mamba2Block
from transformers.models.mamba.modeling_mamba import MambaModel, MambaBlock

from transformers import Mamba2ForCausalLM, MambaForCausalLM

from prismatic.models.backbones.llm.base_llm import HFCausalLLMBackbone
from prismatic.models.backbones.llm.prompting import MistralInstructPromptBuilder, PromptBuilder, PurePromptBuilder
# Registry =>> Support LLaMa-2 Models (from HF Transformers)
# fmt: off
MAMBA_MODELS = {
    # === Pure Meta LLaMa-2 (non-instruct/chat-tuned) Models ===
    "mamba-codestral-7b": {
        "llm_family": "mamba", "llm_cls": Mamba2ForCausalLM, "hf_hub_path": "mistralai/Mamba-Codestral-7B-v0.1"
    },
    "mamba": {
        "llm_family": "mamba", "llm_cls": MambaForCausalLM, "hf_hub_path": "state-spaces/mamba-1.4b-hf"
    },
}
# fmt: on


class MambaLLMBackbone(HFCausalLLMBackbone):
    def __init__(
        self,
        llm_backbone_id: str,
        llm_max_length: int = 2048,
        hf_token: Optional[str] = None,
        inference_mode: bool = False,
        use_flash_attention_2: bool = False,
    ) -> None:
        super().__init__(
            llm_backbone_id,
            llm_max_length=llm_max_length,
            hf_token=hf_token,
            inference_mode=inference_mode,
            use_flash_attention_2=use_flash_attention_2,
            **MAMBA_MODELS[llm_backbone_id],
        )

        # [Special Case] LLaMa-2 PAD Token Handling --> for clarity, we add an extra token (and resize)
        self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        self.llm.config.pad_token_id = self.tokenizer.pad_token_id
        self.llm.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=64)
        
    @property
    def prompt_builder_fn(self) -> Type[PromptBuilder]:
        # if self.identifier.endswith("-pure"):
        #     return PurePromptBuilder

        # elif self.identifier.endswith("-instruct"):
        return MistralInstructPromptBuilder

        # raise ValueError(f"No PromptBuilder defined for LLM Backbone `{self.identifier}`")

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        if self.identifier == "mamba-codestral-7b":
            return Mamba2Block
        else:
            return MambaBlock


    @property
    def last_layer_finetune_modules(self) -> Sequence[nn.Module]:
        return None
    
    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16
    
