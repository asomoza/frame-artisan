"""
Tiny AutoEncoder for Hunyuan Video (TAEHV)
DNN for encoding / decoding videos to various latent spaces.

Vendored from https://github.com/madebyollin/taehv (MIT License).
Modified to support loading from safetensors files.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple

TWorkItem = namedtuple("TWorkItem", ("input_tensor", "block_index"))


def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class MemBlock(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in * 2, n_out), nn.ReLU(inplace=True), conv(n_out, n_out), nn.ReLU(inplace=True), conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, past):
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))


class TPool(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f * stride, n_f, 1, bias=False)

    def forward(self, x):
        _NT, C, H, W = x.shape
        return self.conv(x.reshape(-1, self.stride * C, H, W))


class TGrow(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f * stride, 1, bias=False)

    def forward(self, x):
        _NT, C, H, W = x.shape
        x = self.conv(x)
        return x.reshape(-1, C, H, W)


def apply_model_with_memblocks_parallel(model, x):
    assert x.ndim == 5, f"TAEHV operates on NTCHW tensors, but got {x.ndim}-dim tensor"
    N, T, C, H, W = x.shape
    x = x.reshape(N * T, C, H, W)

    for b in model:
        if isinstance(b, MemBlock):
            NT, C, H, W = x.shape
            T = NT // N
            _x = x.reshape(N, T, C, H, W)
            block_memory = F.pad(_x, (0, 0, 0, 0, 0, 0, 1, 0), value=0)[:, :T].reshape(x.shape)
            x = b(x, block_memory)
        else:
            x = b(x)
    NT, C, H, W = x.shape
    T = NT // N
    return x.view(N, T, C, H, W)


class TAEHV(nn.Module):
    _DEFAULT_DECODER_TIME_UPSCALE = (False, True, True)

    def __init__(
        self,
        checkpoint_path=None,
        encoder_time_downscale=(True, True, False),
        decoder_time_upscale=None,
        decoder_space_upscale=(True, True, True),
        patch_size=1,
        latent_channels=16,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.latent_channels = latent_channels
        self.image_channels = 3
        _decoder_time_upscale_default = decoder_time_upscale is None
        if decoder_time_upscale is None:
            decoder_time_upscale = self._DEFAULT_DECODER_TIME_UPSCALE
        if len(decoder_time_upscale) == 2:
            decoder_time_upscale = (False, *decoder_time_upscale)

        if checkpoint_path is not None and "taeltx" in checkpoint_path:
            self.patch_size, self.latent_channels = 4, 128
            encoder_time_downscale = (True, True, True)
            if _decoder_time_upscale_default:
                decoder_time_upscale = (True, True, True)

        self.encoder = nn.Sequential(
            conv(self.image_channels * self.patch_size**2, 64), nn.ReLU(inplace=True),
            TPool(64, 2 if encoder_time_downscale[0] else 1), conv(64, 64, stride=2, bias=False), MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64),
            TPool(64, 2 if encoder_time_downscale[1] else 1), conv(64, 64, stride=2, bias=False), MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64),
            TPool(64, 2 if encoder_time_downscale[2] else 1), conv(64, 64, stride=2, bias=False), MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64),
            conv(64, self.latent_channels),
        )
        n_f = [256, 128, 64, 64]
        self.decoder = nn.Sequential(
            Clamp(), conv(self.latent_channels, n_f[0]), nn.ReLU(inplace=True),
            MemBlock(n_f[0], n_f[0]), MemBlock(n_f[0], n_f[0]), MemBlock(n_f[0], n_f[0]), nn.Upsample(scale_factor=2 if decoder_space_upscale[0] else 1), TGrow(n_f[0], 2 if decoder_time_upscale[0] else 1), conv(n_f[0], n_f[1], bias=False),
            MemBlock(n_f[1], n_f[1]), MemBlock(n_f[1], n_f[1]), MemBlock(n_f[1], n_f[1]), nn.Upsample(scale_factor=2 if decoder_space_upscale[1] else 1), TGrow(n_f[1], 2 if decoder_time_upscale[1] else 1), conv(n_f[1], n_f[2], bias=False),
            MemBlock(n_f[2], n_f[2]), MemBlock(n_f[2], n_f[2]), MemBlock(n_f[2], n_f[2]), nn.Upsample(scale_factor=2 if decoder_space_upscale[2] else 1), TGrow(n_f[2], 2 if decoder_time_upscale[2] else 1), conv(n_f[2], n_f[3], bias=False),
            nn.ReLU(inplace=True), conv(n_f[3], self.image_channels * self.patch_size**2),
        )

        self.t_downscale = 2 ** sum(t.stride == 2 for t in self.encoder if isinstance(t, TPool))
        self.t_upscale = 2 ** sum(t.stride == 2 for t in self.decoder if isinstance(t, TGrow))
        self.frames_to_trim = self.t_upscale - 1

        if checkpoint_path is not None:
            sd = self._load_checkpoint(checkpoint_path)
            self.load_state_dict(self.patch_tgrow_layers(sd))

    @staticmethod
    def _load_checkpoint(path: str) -> dict:
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file
            return load_file(path)
        return torch.load(path, map_location="cpu", weights_only=True)

    def patch_tgrow_layers(self, sd):
        new_sd = self.state_dict()
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, TGrow):
                key = f"decoder.{i}.conv.weight"
                if sd[key].shape[0] > new_sd[key].shape[0]:
                    sd[key] = sd[key][-new_sd[key].shape[0]:]
        return sd

    def postprocess_output_frames(self, x):
        if self.patch_size > 1:
            x = F.pixel_shuffle(x, self.patch_size)
        return x.clamp_(0, 1)

    def decode_video(self, x, parallel=True):
        """Decode a sequence of latent frames.

        Args:
            x: NTCHW latent tensor (C=self.latent_channels) with ~Gaussian values.
            parallel: if True, all frames processed at once (faster, more memory).

        Returns NTCHW RGB tensor with values in [0, 1].
        """
        x = apply_model_with_memblocks_parallel(self.decoder, x)
        x = self.postprocess_output_frames(x)
        return x[:, self.frames_to_trim:]
