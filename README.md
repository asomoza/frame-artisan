# Frame Artisan

A desktop application for AI video generation using the LTX 2.0 and 2.3 video models, built with PyQt6 and powered by Diffusers.

## Features

- Node-based video generation pipeline
- Text-to-video and image-to-video generation
- Audio conditioning support
- LoRA adapter support with per-LoRA controls
- Multi-condition visual conditioning (images, video clips, keyframes)
- Two-stage generation with spatial upsampling
- VRAM-optimized with multiple offload strategies
- Built-in video editor for source preprocessing

## Requirements

- Python 3.12+
- NVIDIA GPU with CUDA support (recommended)
- 8GB+ VRAM

## Installation

See the [installer releases](https://github.com/asomoza/frame-artisan/releases) for a one-click setup.

### Manual installation

You will need Python 3.12+ and [uv](https://docs.astral.sh/uv/) installed.

```bash
git clone https://github.com/asomoza/frame-artisan.git
cd frame-artisan
uv sync
uv run frameartisan
```

The first run will download all required packages. PyTorch with CUDA 13.0 is installed automatically via the index configured in `pyproject.toml`.

To use a different CUDA version (e.g. `cu126`), edit the `[[tool.uv.index]]` URL in `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
```

And update the index name in the `[tool.uv.sources]` entries to match.

For AMD GPUs, install [ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/) first, then use the ROCm index:

```toml
[[tool.uv.index]]
name = "pytorch-rocm"
url = "https://download.pytorch.org/whl/rocm6.4"
```

See [PyTorch's install page](https://pytorch.org/get-started/locally/) for all available CUDA and ROCm versions.

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
