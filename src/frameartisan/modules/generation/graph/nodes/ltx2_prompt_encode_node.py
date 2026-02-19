from __future__ import annotations

import logging
from typing import ClassVar

from frameartisan.modules.generation.graph.node_error import NodeError
from frameartisan.modules.generation.graph.nodes.node import Node


logger = logging.getLogger(__name__)


class LTX2PromptEncodeNode(Node):
    PRIORITY = 1
    REQUIRED_INPUTS: ClassVar[list[str]] = [
        "tokenizer",
        "text_encoder",
        "connectors",
        "prompt",
    ]
    OPTIONAL_INPUTS: ClassVar[list[str]] = ["negative_prompt", "guidance_scale"]
    OUTPUTS: ClassVar[list[str]] = ["prompt_embeds", "audio_prompt_embeds", "attention_mask"]

    SERIALIZE_INCLUDE: ClassVar[set[str]] = set()

    def __call__(self):
        import torch

        from frameartisan.app.model_manager import get_model_manager
        from frameartisan.modules.generation.graph.nodes.ltx2_utils import pack_text_embeds

        tokenizer = self.tokenizer
        text_encoder = self.text_encoder
        connectors = self.connectors
        prompt = self.prompt
        negative_prompt = self.negative_prompt or ""

        if tokenizer is None or text_encoder is None or connectors is None:
            raise NodeError("tokenizer, text_encoder, and connectors must be connected", self.__class__.__name__)

        device = self.device
        dtype = text_encoder.dtype
        max_sequence_length = 1024

        guidance_scale = self.guidance_scale
        do_cfg = bool(negative_prompt) and (guidance_scale is None or guidance_scale > 1.0)
        mm = get_model_manager()

        def _encode(text: str) -> tuple[torch.Tensor, torch.Tensor]:
            text = text.strip()
            tokenizer.padding_side = "left"
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            text_inputs = tokenizer(
                [text],
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            input_ids = text_inputs.input_ids.to(device)
            attention_mask = text_inputs.attention_mask.to(device)

            outputs = text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = torch.stack(outputs.hidden_states, dim=-1)
            del outputs
            sequence_lengths = attention_mask.sum(dim=-1)

            embeds = pack_text_embeds(
                hidden_states,
                sequence_lengths,
                padding_side=tokenizer.padding_side,
            )
            del hidden_states
            embeds = embeds.to(dtype=dtype)
            return embeds, attention_mask

        with mm.use_components("text_encoder", "connectors", device=device):
            logger.info("Encoding prompt")
            prompt_embeds, prompt_attention_mask = _encode(prompt)

            if do_cfg:
                logger.info("Encoding negative prompt")
                negative_embeds, negative_attention_mask = _encode(negative_prompt)
                # CFG: [negative, positive] along batch dim
                prompt_embeds = torch.cat([negative_embeds, prompt_embeds], dim=0)
                prompt_attention_mask = torch.cat([negative_attention_mask, prompt_attention_mask], dim=0)

            # Convert attention mask to additive format and run through connectors
            additive_mask = (1 - prompt_attention_mask.to(prompt_embeds.dtype)) * -1000000.0
            connector_video_embeds, connector_audio_embeds, connector_mask = connectors(
                prompt_embeds,
                additive_mask,
                additive_mask=True,
            )

        # Store on CPU for VRAM efficiency
        self.values = {
            "prompt_embeds": connector_video_embeds.cpu(),
            "audio_prompt_embeds": connector_audio_embeds.cpu(),
            "attention_mask": connector_mask.cpu(),
        }
        return self.values
