from __future__ import annotations

import logging
from typing import ClassVar

from frameartisan.modules.generation.graph.node_error import NodeError
from frameartisan.modules.generation.graph.nodes.node import Node


logger = logging.getLogger(__name__)


class LTX2AudioEncodeNode(Node):
    """Encode an audio file into clean audio latents for audio conditioning.

    Follows the pipeline_ltx2_audio2video.py logic:
    load audio → mel spectrogram → audio VAE encode → normalize → pack.
    """

    PRIORITY = 0
    REQUIRED_INPUTS: ClassVar[list[str]] = ["audio_vae", "num_frames", "frame_rate"]
    OPTIONAL_INPUTS: ClassVar[list[str]] = []
    OUTPUTS: ClassVar[list[str]] = ["clean_audio_latents", "audio_conditioning_mask", "audio_path"]
    SERIALIZE_INCLUDE: ClassVar[set[str]] = {"audio_path", "trim_start_s", "trim_end_s"}

    def __init__(self):
        super().__init__()
        self.audio_path: str | None = None
        self.trim_start_s: float | None = None
        self.trim_end_s: float | None = None

    def update_path(self, path: str | None) -> None:
        self.audio_path = path
        self.set_updated()

    def update_trim(self, start_s: float | None, end_s: float | None) -> None:
        self.trim_start_s = start_s
        self.trim_end_s = end_s
        self.set_updated()

    def __call__(self):
        import torch
        import torchaudio

        from frameartisan.app.model_manager import get_model_manager
        from frameartisan.modules.generation.graph.nodes.ltx2_utils import (
            normalize_audio_latents,
            pack_audio_latents,
        )

        audio_vae = self.audio_vae
        num_frames = int(self.num_frames)
        frame_rate = float(self.frame_rate)

        if audio_vae is None:
            raise NodeError("audio_vae input is not connected", self.__class__.__name__)
        if not self.audio_path:
            raise NodeError("audio_path is not set", self.__class__.__name__)

        device = self.device
        mm = get_model_manager()

        # Audio VAE properties
        sample_rate = getattr(audio_vae.config, "sample_rate", 16000)
        mel_hop_length = getattr(audio_vae.config, "mel_hop_length", 160)
        mel_bins = getattr(audio_vae.config, "mel_bins", 64)
        audio_temporal_ratio = getattr(audio_vae, "temporal_compression_ratio", 4)
        # Load audio
        waveform, sr = torchaudio.load(self.audio_path)

        # Resample to target sample rate
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
            waveform = resampler(waveform)

        # Ensure stereo (2 channels)
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2]

        # Apply source trim if set (skip when range covers full audio or would produce empty waveform)
        if self.trim_start_s is not None and self.trim_end_s is not None and self.trim_end_s > self.trim_start_s:
            trim_start_sample = int(self.trim_start_s * sample_rate)
            trim_end_sample = int(self.trim_end_s * sample_rate)
            trim_start_sample = max(0, min(trim_start_sample, waveform.shape[1]))
            trim_end_sample = max(trim_start_sample + 1, min(trim_end_sample, waveform.shape[1]))
            if not (trim_start_sample == 0 and trim_end_sample >= waveform.shape[1]):
                waveform = waveform[:, trim_start_sample:trim_end_sample]

        # Compute target duration and trim/pad audio
        num_frames_snapped = 8 * ((num_frames - 1) // 8) + 1
        duration_s = num_frames_snapped / frame_rate
        target_samples = int(duration_s * sample_rate)

        # Track original audio length before padding for conditioning mask
        original_samples = waveform.shape[1]

        if waveform.shape[1] > target_samples:
            waveform = waveform[:, :target_samples]
            original_samples = target_samples  # trimmed to fit, no mask needed
        elif waveform.shape[1] < target_samples:
            pad = target_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))

        # Add batch dimension: [channels, samples] -> [batch, channels, samples]
        waveform = waveform.unsqueeze(0)

        # Compute mel spectrogram (matching reference pipeline parameters exactly)
        n_fft = 1024
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=mel_hop_length,
            f_min=0.0,
            f_max=sample_rate / 2.0,
            n_mels=mel_bins,
            window_fn=torch.hann_window,
            center=True,
            pad_mode="reflect",
            power=1.0,
            mel_scale="slaney",
            norm="slaney",
        )
        # mel_spec shape: [batch, channels, mel_bins, time]
        mel_spec = mel_transform(waveform)

        # Log scale (matching reference: log(clamp(x, min=1e-5)))
        mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))

        # Permute to [batch, channels, time, mel_bins] as expected by VAE
        mel_spec = mel_spec.permute(0, 1, 3, 2).contiguous()

        # Audio VAE encode
        with mm.use_components("audio_vae", device=device, strategy_override="model_offload"):
            mel_spec = mel_spec.to(device=device, dtype=audio_vae.dtype)
            latents = audio_vae.encode(mel_spec).latent_dist.sample()

        # Normalize: patchify → normalize → unpatchify
        packed = pack_audio_latents(latents)
        packed = normalize_audio_latents(packed, audio_vae.latents_mean, audio_vae.latents_std)

        # Compute target temporal length
        audio_latents_per_second = sample_rate / mel_hop_length / float(audio_temporal_ratio)
        target_length = round(duration_s * audio_latents_per_second)

        # Compute how many latent tokens the original (pre-pad) audio maps to
        original_duration_s = original_samples / sample_rate
        original_latent_length = round(original_duration_s * audio_latents_per_second)

        # Pad or trim temporal dim
        current_length = packed.shape[1]
        if current_length > target_length:
            packed = packed[:, :target_length, :]
        elif current_length < target_length:
            pad_size = target_length - current_length
            packed = torch.nn.functional.pad(packed, (0, 0, 0, pad_size))

        # Create conditioning mask when original audio is shorter than target
        audio_conditioning_mask = None
        if original_latent_length < target_length:
            original_latent_length = max(0, min(original_latent_length, target_length))
            audio_conditioning_mask = torch.zeros(1, target_length, device=packed.device, dtype=torch.float32)
            audio_conditioning_mask[:, :original_latent_length] = 1.0

        logger.info(
            "Audio encoded: latents %s from %s (conditioning_mask=%s)",
            packed.shape,
            self.audio_path,
            audio_conditioning_mask.shape if audio_conditioning_mask is not None else None,
        )

        self.values = {
            "clean_audio_latents": packed.float(),
            "audio_conditioning_mask": audio_conditioning_mask,
            "audio_path": self.audio_path,
        }
        return self.values
