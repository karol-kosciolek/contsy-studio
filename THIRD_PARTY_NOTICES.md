# Third-party notices

Contsy Studio's own code is licensed under the [GNU GPL v3.0 only](LICENSE) (SPDX: `GPL-3.0-only`). Parts that come from other projects keep their own licenses, listed below. This repository includes no third-party model weights, engine binaries or Python packages. `install.bat` and the in-app model manager download them from their upstream sources at pinned revisions. [docs/LICENSING.md](docs/LICENSING.md) explains how the parts fit together.

If you build a bundle that redistributes any downloaded component, you must meet that component's license terms yourself. Several are GPL-licensed: ComfyUI, ComfyUI-VideoHelperSuite and the FFmpeg build.

The installer leaves out funasr, which VoxCPM declares but never imports, and with it kaldiio, whose license allows evaluation only. Earlier installations lose both on their next `install.bat` run.

## Included in this repository

| Path | Origin | License | Notes |
| --- | --- | --- | --- |
| `app/lipsync_worker.py` | `scripts/inference.py` from [LatentSync](https://github.com/bytedance/LatentSync), Copyright (c) 2024 Bytedance Ltd. and/or its affiliates | Apache-2.0 ([text](LICENSES/Apache-2.0.txt)) for the original parts; Contsy's changes are GPL-3.0-only | Changes: Whisper and the VAE load from local files, MediaPipe Face Landmarker replaces InsightFace, it runs as a short-lived worker process, and it has a low-VRAM mode. |
| `patches/ComfyUI-Frame-Interpolation/` | [ComfyUI-Frame-Interpolation](https://github.com/Fannovel16/ComfyUI-Frame-Interpolation), Copyright (c) 2023 Fannovel16 | MIT ([text](patches/ComfyUI-Frame-Interpolation/LICENSE)) | Modified `vfi_models/rife/__init__.py` and a new `frame_rate.py` (exact source/target frame rates). `install.bat` installs them over the pinned node. |
| `workflows/*.json` | Structure of ComfyUI's official [workflow templates](https://github.com/Comfy-Org/workflow_templates), Copyright (c) 2023-present Comfy Org | MIT ([text](LICENSES/MIT.txt)) for template-derived structure; Contsy's adaptations are GPL-3.0-only | Node graphs that Contsy fills in and sends to ComfyUI. |
| Sidebar icons in `app/ui_theme.py` | Geometry of [Lucide](https://lucide.dev) icons, Copyright (c) 2026 Lucide Icons and Contributors | ISC ([notice](LICENSES/ISC-Lucide.txt)) | Inline SVG paths, adjusted. |
| `assets/voices/*.wav` | Generated during development with VoxCPM2 (Apache-2.0) from text alone: no reference recording and no voice description | GPL-3.0-only, like the rest of the repository | Four synthetic voices, not recordings of real people. Chosen from 24 random voices by pitch and a quality score. |

## Downloaded by the installer

| Component | Source (pinned) | License | How Contsy uses it |
| --- | --- | --- | --- |
| ComfyUI (engine) | [Comfy-Org/ComfyUI](https://github.com/Comfy-Org/ComfyUI) @ `7cebca5` | GPL-3.0 | A separate process that Contsy starts and drives through its local HTTP/WebSocket API. Contsy does not copy, import or modify it. |
| ComfyUI-GGUF | [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) @ `6ea2651` | Apache-2.0 | Loaded by ComfyUI. |
| ComfyUI-VideoHelperSuite | [Kosinkadink/ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) @ `4d907be` | GPL-3.0 | Loaded by ComfyUI. |
| ComfyUI-Frame-Interpolation | [Fannovel16/ComfyUI-Frame-Interpolation](https://github.com/Fannovel16/ComfyUI-Frame-Interpolation) @ `26545cc` | MIT | Loaded by ComfyUI; patched as described above. |
| RIFE 4.9 weights (`rife49.pth`) | Fetched by ComfyUI-Frame-Interpolation from its configured GitHub release mirrors | RIFE code is MIT; the weight file's license on the mirrors is **not stated** and needs checking | Frame interpolation inside ComfyUI. |
| LatentSync code (`vendor/LatentSync`) | [ShmuelRonen/ComfyUI-LatentSyncWrapper](https://github.com/ShmuelRonen/ComfyUI-LatentSyncWrapper) @ `360d528` | ByteDance LatentSync code (Apache-2.0) with small changes. The wrapper's README states Apache-2.0, but the repository has **no license file**. | Imported by the lip-sync worker process. |
| musubi-tuner | [kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner) @ `f8a1b03` | Apache-2.0 according to its README (no license file). Some subdirectories follow the licenses of the models they adapt. | Separate processes for LoRA training. |
| llama.cpp (Vulkan build `b11900`) | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) releases, SHA-256 pinned | MIT | Separate executables for the assistant and safety checks. |
| PyTorch 2.14.1, torchvision, torchaudio (CUDA 13.0) | download.pytorch.org | BSD-style. The bundled NVIDIA CUDA libraries are under NVIDIA's license terms. | Imported. |
| Gradio 6.30 (including its web frontend) | PyPI | Apache-2.0 | Imported; serves the local web UI. |
| Other Python packages (about 190 in a fresh install) | PyPI | Mostly Apache-2.0, MIT or BSD, for example torchcodec (BSD-3-Clause). Weak copyleft, used unmodified: easydict (LGPL-3.0), soxr (LGPL-2.1+), certifi, tqdm and orjson (MPL-2.0 parts). comfyui-embedded-docs and comfy-aimdo (GPL-3.0) are ComfyUI dependencies. | Imported, or used by ComfyUI. |
| FFmpeg | Windows Package Manager `Gyan.FFmpeg` | GPL-3.0 build | Separate executable. |
| Git, Python | Windows Package Manager | GPL-2.0, PSF-2.0 | Separate programs. |

## Models

Models download the first time a feature needs them, or all at once through `install.bat`. Contsy does not redistribute any weights. The table summarizes their terms; [MODEL_LICENSES.md](MODEL_LICENSES.md) has the pinned sources.

| Model | License | Commercial use and hosting |
| --- | --- | --- |
| Z-Image Turbo (and its training adapter) | Apache-2.0 | Allowed |
| Wan 2.2 A14B, QuantStack GGUF conversions, LightX2V LoRAs, Wan 2.1 VAE, UMT5 encoder (incl. city96 GGUF) | Apache-2.0 | Allowed |
| Qwen Image Edit 2511 (incl. Unsloth GGUF), Lightning LoRA, Qwen2.5-VL 7B encoder, Qwen Image VAE | Apache-2.0 | Allowed |
| Qwen3.5 9B (Unsloth GGUF) | Apache-2.0 | Allowed |
| SeedVR2 3B | Apache-2.0 (Comfy-Org repository card) | Allowed |
| VoxCPM2 | Apache-2.0 | Allowed |
| LatentSync 1.6 | CreativeML Open RAIL++-M | Allowed, with use-based restrictions that must be passed on to your users |
| Whisper tiny, sd-vae-ft-mse (used by LatentSync) | MIT | Allowed |
| Stable Audio Open 1.0 | Stability AI Community License | Conditional: free below USD 1 million annual revenue; commercial users must register with Stability AI; above the threshold an enterprise license is needed. Requires the notice in [NOTICE](NOTICE) and "Powered by Stability AI" |
| T5-base (Stable Audio text encoder) | Apache-2.0 | Allowed |
| OpenCV YuNet / SFace (face check) | MIT (Copyright (c) 2020 Shiqi Yu) / Apache-2.0 | Allowed |
| CLIP ViT-B/32 (face tracking) | MIT | Allowed |
| MediaPipe Face Landmarker (face detector, face mesh and blendshape models) | Apache-2.0 (each of the three model cards links the Apache License 2.0) | Allowed |

The Qwen2.5-VL encoder used to download from `Comfy-Org/HunyuanVideo_1.5_repackaged`, whose card names the Tencent Hunyuan Community License. That license excludes the EU, the UK and South Korea. The identical file (same SHA-256) now downloads from `Comfy-Org/Qwen-Image_ComfyUI`, which is Apache-2.0. Existing installations keep their file unchanged.
