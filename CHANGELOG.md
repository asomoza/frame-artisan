# Changelog

## 0.1.2 — 2026-03-22

- Lazy-load tokenizer from component registry for prompt token counting (removes bundled 32MB config)
- Include theme assets (icons, images, stylesheet) in package
- Installer: add `--index-strategy unsafe-best-match` to fix setuptools resolution from PyTorch index
- Installer: shorter error messages with log file reference

## 0.1.1 — 2026-03-22

- Fix `setuptools>=77` build requirement for PEP 639 license field
- Fix Python version requirement: `>=3.12` (was `>=3.13`)

## 0.1.0 — 2026-03-22

Initial release.

- Node-based video generation pipeline with LTX-Video
- Text-to-video and image-to-video generation
- Two-stage generation with spatial upsampling
- Audio conditioning
- LoRA support with granular weight controls and spatiotemporal masking
- Multi-condition visual conditioning (images, video clips, keyframes)
- Multiple VRAM offload strategies (group offload, model offload, sequential)
- Feed-forward chunking and streaming VAE decode for low-VRAM setups
- Built-in video editor for source preprocessing
- Model and LoRA management with component deduplication
