from diffusers import (
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    FlowMatchEulerDiscreteScheduler,
    FlowMatchHeunDiscreteScheduler,
    FlowMatchLCMScheduler,
    SASolverScheduler,
    UniPCMultistepScheduler,
)


MODEL_TYPES = {
    1: "LTX-2",
    2: "LTX-2 Distilled",
    3: "LTX-2.3",
    4: "LTX-2.3 Distilled",
}

LORA_MODEL_TYPES = {
    0: "LTX-2",
    1: "LTX-2.3",
}


def get_default_granular_weights(model_type: int) -> dict:
    return {f"transformer_blocks.{i}": 1.0 for i in range(48)}


# Default generation parameters per model type (num_inference_steps, guidance_scale).
MODEL_TYPE_DEFAULTS: dict[int, dict[str, int | float]] = {
    1: {"num_inference_steps": 40, "guidance_scale": 4.0},
    2: {"num_inference_steps": 8, "guidance_scale": 1.0},
    3: {"num_inference_steps": 40, "guidance_scale": 4.0},
    4: {"num_inference_steps": 8, "guidance_scale": 1.0},
}

# Defaults for the 2nd pass (stage 2). Distilled uses a 3-step sigma schedule.
SECOND_PASS_MODEL_TYPE_DEFAULTS: dict[int, dict[str, int | float]] = {
    1: {"num_inference_steps": 10, "guidance_scale": 4.0},
    2: {"num_inference_steps": 3, "guidance_scale": 1.0},
    3: {"num_inference_steps": 10, "guidance_scale": 4.0},
    4: {"num_inference_steps": 3, "guidance_scale": 1.0},
}

# Fixed sigma schedules for the LTX2 distilled model — imported from diffusers
# and re-exported for use by graph nodes.
from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES as LTX2_DISTILLED_SIGMAS  # noqa: E402, F401
from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES as LTX2_STAGE2_DISTILLED_SIGMAS  # noqa: E402, F401

# Latent upsampler model constants.
LTX2_LATENT_UPSAMPLER_REPO = "OzzyGT/LTX2_latent_upsampler"
LTX2_LATENT_UPSAMPLER_DIR = "LTX2_latent_upsampler"

DEFAULT_NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted"

ADVANCED_GUIDANCE_DEFAULTS: dict[str, float | int | str] = {
    "stg_scale": 1.0,
    "stg_blocks": "29",
    "rescale_scale": 0.7,
    "modality_scale": 3.0,
    "guidance_skip_step": 0,
}


SCHEDULER_CLASS_MAPPING = {
    "DEISMultistepScheduler": DEISMultistepScheduler,
    "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
    "DPMSolverSinglestepScheduler": DPMSolverSinglestepScheduler,
    "FlowMatchEulerDiscreteScheduler": FlowMatchEulerDiscreteScheduler,
    "FlowMatchHeunDiscreteScheduler": FlowMatchHeunDiscreteScheduler,
    "FlowMatchLCMScheduler": FlowMatchLCMScheduler,
    "SASolverScheduler": SASolverScheduler,
    "UniPCMultistepScheduler": UniPCMultistepScheduler,
}

SCHEDULER_NAME_CLASS_MAPPING = {
    "DEIS": "DEISMultistepScheduler",
    "DPM++ 2M": "DPMSolverMultistepScheduler",
    "DPM++ 2S": "DPMSolverSinglestepScheduler",
    "Euler": "FlowMatchEulerDiscreteScheduler",
    "Heun": "FlowMatchHeunDiscreteScheduler",
    "LCM": "FlowMatchLCMScheduler",
    "SA": "SASolverScheduler",
    "UniPC": "UniPCMultistepScheduler",
}

SCHEDULER_NAMES = list(SCHEDULER_NAME_CLASS_MAPPING.keys())

OFFLOAD_STRATEGIES: dict[str, str] = {
    "auto": "Auto",
    "no_offload": "No Offload",
    "model_offload": "Model Offload",
    "sequential_group_offload": "Sequential Group Offload",
    "group_offload": "Group Offload",
}
