from __future__ import annotations

from .boolean_node import BooleanNode
from .image_load_node import ImageLoadNode
from .ltx2_decode_node import LTX2DecodeNode
from .ltx2_latent_upsample_node import LTX2LatentUpsampleNode
from .ltx2_lora_node import LTX2LoraNode
from .ltx2_audio_encode_node import LTX2AudioEncodeNode
from .ltx2_condition_encode_node import LTX2ConditionEncodeNode
from .ltx2_denoise_node import LTX2DenoiseNode
from .ltx2_image_encode_node import LTX2ImageEncodeNode
from .ltx2_latents_node import LTX2LatentsNode
from .ltx2_model_node import LTX2ModelNode
from .ltx2_prompt_encode_node import LTX2PromptEncodeNode
from .ltx2_second_pass_latents_node import LTX2SecondPassLatentsNode
from .ltx2_video_send_node import LTX2VideoSendNode
from .number_node import NumberNode
from .text_node import TextNode
from .video_load_node import VideoLoadNode


NODE_CLASSES = {
    "BooleanNode": BooleanNode,
    "ImageLoadNode": ImageLoadNode,
    "LTX2AudioEncodeNode": LTX2AudioEncodeNode,
    "LTX2ConditionEncodeNode": LTX2ConditionEncodeNode,
    "LTX2DecodeNode": LTX2DecodeNode,
    "LTX2DenoiseNode": LTX2DenoiseNode,
    "LTX2ImageEncodeNode": LTX2ImageEncodeNode,
    "LTX2LatentUpsampleNode": LTX2LatentUpsampleNode,
    "LTX2LoraNode": LTX2LoraNode,
    "LTX2LatentsNode": LTX2LatentsNode,
    "LTX2ModelNode": LTX2ModelNode,
    "LTX2PromptEncodeNode": LTX2PromptEncodeNode,
    "LTX2SecondPassLatentsNode": LTX2SecondPassLatentsNode,
    "LTX2VideoSendNode": LTX2VideoSendNode,
    "NumberNode": NumberNode,
    "TextNode": TextNode,
    "VideoLoadNode": VideoLoadNode,
}
