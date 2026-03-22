# Changelog

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
