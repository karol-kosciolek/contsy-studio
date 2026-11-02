# Model and tool sources

Weights are downloaded separately, not included with the application. `config/models.json` records exact repository revisions, filenames, sizes and SHA-256 hashes for large model files. Quantized/repacked files are obtained from their named converters; the original model authors' terms still apply. Settings lists download providers and license notes.

| Component | Upstream source | Catalog license |
| --- | --- | --- |
| Z-Image Turbo | https://huggingface.co/Tongyi-MAI/Z-Image-Turbo | Apache 2.0 |
| Wan 2.2 | https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B | Apache 2.0 |
| Qwen Image Edit | https://huggingface.co/Qwen/Qwen-Image-Edit-2511 | Apache 2.0 |
| Qwen3.5 9B | https://huggingface.co/Qwen/Qwen3.5-9B | Apache 2.0 |
| Qwen GGUF conversion | https://huggingface.co/unsloth/Qwen3.5-9B-GGUF | Apache 2.0 |
| Wan 2.2 GGUF conversions (Q6_K full, Q4_K_M Lite) | https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF, https://huggingface.co/QuantStack/Wan2.2-T2V-A14B-GGUF | Apache 2.0 |
| UMT5-XXL GGUF conversion (Lite) | https://huggingface.co/city96/umt5-xxl-encoder-gguf | Apache 2.0 |
| Qwen Image Edit GGUF conversion (Lite) | https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF | Apache 2.0 |
| VoxCPM2 | https://huggingface.co/openbmb/VoxCPM2 | Apache 2.0 |
| SeedVR2 | https://huggingface.co/ByteDance-Seed/SeedVR2-3B | Apache 2.0 |
| LatentSync 1.6 | https://huggingface.co/ByteDance/LatentSync-1.6 | OpenRAIL++ usage restrictions |
| Stable Audio Open 1.0 | https://huggingface.co/stabilityai/stable-audio-open-1.0 | Stability AI Community License (Powered by Stability AI) |
| Training adapter | https://huggingface.co/ostris/zimage_turbo_training_adapter | Apache 2.0 |
| OpenCV SFace / YuNet | https://github.com/opencv/opencv_zoo | Per-model Apache 2.0 / MIT |
| CLIP ViT-B/32 (face tracking) | https://huggingface.co/openai/clip-vit-base-patch32 | MIT |
| MediaPipe Face Landmarker | https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker | Apache 2.0 (linked from the model cards) |

Read the exact license linked by the provider before downloading or using a model commercially. Stable Audio's terms are conditional; it is not an unconditional Apache/MIT component. It is free below USD 1 million in annual revenue, commercial users must register with Stability AI, and its required notice is in [NOTICE](NOTICE). LatentSync's CreativeML Open RAIL++-M license adds use restrictions that also apply to people you let use it. Gated access must be accepted by the user and is never bypassed.

The engine and extensions are separate third-party projects that the installer fetches from their upstream repositories and the PyTorch package index. [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) lists each one with its pinned revision and license. Contsy publishes no third-party weights.

Contsy's own code is licensed under the [GNU GPL v3.0 only](LICENSE). That license covers only Contsy's own code. It does not change the terms of any model above, so whether you may use an output commercially depends on the models that produced it. [docs/LICENSING.md](docs/LICENSING.md) explains how the licenses fit together.
