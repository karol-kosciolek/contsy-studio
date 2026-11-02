# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Budowanie workflowow obrazow ComfyUI w formacie API (slownik wezlow).

Model opisuje slownik `spec`:
  {"family": "zimage", "unet": ..., "clip": ..., "vae": ...}  - Z-Image-Turbo (osobne pliki)
  {"family": "flux_schnell" | "flux" | "sdxl" | "sd", "ckpt": ...}  - checkpoint all-in-one

Rodziny:
  - zimage: graf z oficjalnego szablonu ComfyUI (image_z_image_turbo): CLIPLoader typu
    lumina2, ModelSamplingAuraFlow shift 3, res_multistep/simple, CFG 1, 8 krokow.
  - flux_schnell / flux: CFG 1 i wyzerowany negatyw; flux (dev-like) dodatkowo FluxGuidance.
  - sdxl / sd: klasyczny KSampler z pozytywem i negatywem.
LoRA wstawiane sa jako lancuch LoraLoader miedzy loaderami a enkoderem i samplerem.
"""
from __future__ import annotations

from typing import Dict, List, Optional

SAMPLERS = {
    "zimage": ("res_multistep", "simple"),
    "flux_schnell": ("euler", "simple"),
    "flux": ("euler", "simple"),
    "sdxl": ("dpmpp_2m", "karras"),
    "sd": ("dpmpp_2m", "karras"),
}

# Zalecane ustawienia: preset -> (kroki, cfg, szerokosc, wysokosc)
PRESETS = {
    "zimage": {"szybki": (6, 1.0, 1024, 1024), "jakosc": (8, 1.0, 1024, 1024)},
    "flux_schnell": {"szybki": (4, 1.0, 768, 768), "jakosc": (4, 1.0, 1024, 1024)},
    "flux": {"szybki": (20, 1.0, 1024, 1024), "jakosc": (28, 1.0, 1024, 1024)},
    "sdxl": {"szybki": (20, 6.5, 1024, 1024), "jakosc": (35, 6.5, 1024, 1024)},
    "sd": {"szybki": (20, 7.0, 512, 512), "jakosc": (30, 7.0, 768, 768)},
}


def detect_family(model_name: str) -> str:
    n = model_name.lower()
    if "z_image" in n or "zimage" in n:
        return "zimage"
    if "schnell" in n:
        return "flux_schnell"
    if "flux" in n:
        return "flux"
    if "xl" in n:
        return "sdxl"
    return "sd"


def recommended(family: str, preset: str) -> dict:
    steps, cfg, width, height = PRESETS[family].get(preset, PRESETS[family]["jakosc"])
    return {"steps": steps, "cfg": cfg, "width": width, "height": height}


def _loaders(g: Dict, spec: dict, loras: Optional[List[dict]]):
    """Wezly ladujace model. Zwraca (model_src, clip_src, vae_src)."""
    if spec["family"] == "zimage":
        g["28"] = {"class_type": "UNETLoader",
                   "inputs": {"unet_name": spec["unet"], "weight_dtype": "default"}}
        g["30"] = {"class_type": "CLIPLoader",
                   "inputs": {"clip_name": spec["clip"], "type": "lumina2", "device": "default"}}
        g["29"] = {"class_type": "VAELoader", "inputs": {"vae_name": spec["vae"]}}
        model_src, clip_src, vae_src = ["28", 0], ["30", 0], ["29", 0]
    else:
        g["4"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": spec["ckpt"]}}
        model_src, clip_src, vae_src = ["4", 0], ["4", 1], ["4", 2]

    for i, lora in enumerate(loras or []):
        key = str(50 + i)
        g[key] = {"class_type": "LoraLoader",
                  "inputs": {"lora_name": lora["name"],
                             "strength_model": float(lora.get("strength_model", 1.0)),
                             "strength_clip": float(lora.get("strength_clip", 1.0)),
                             "model": model_src, "clip": clip_src}}
        model_src, clip_src = [key, 0], [key, 1]

    if spec["family"] == "zimage":
        g["11"] = {"class_type": "ModelSamplingAuraFlow", "inputs": {"shift": 3, "model": model_src}}
        model_src = ["11", 0]
    return model_src, clip_src, vae_src


def _conditioning(g: Dict, family: str, clip_src: list, positive: str, negative: str,
                  guidance: float):
    """Wezly pozytywu/negatywu. Zwraca (pos_src, neg_src)."""
    g["6"] = {"class_type": "CLIPTextEncode", "inputs": {"text": positive, "clip": clip_src}}
    pos_src = ["6", 0]
    if family in ("zimage", "flux_schnell", "flux"):
        if family == "flux":
            g["61"] = {"class_type": "FluxGuidance",
                       "inputs": {"guidance": float(guidance), "conditioning": pos_src}}
            pos_src = ["61", 0]
        g["7"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["6", 0]}}
    else:
        g["7"] = {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": clip_src}}
    return pos_src, ["7", 0]


def _sample_decode_save(g: Dict, family: str, model_src: list, pos_src: list, neg_src: list,
                        latent_src: list, vae_src: list, seed: int, steps: int, cfg: float,
                        denoise: float = 1.0, prefix: str = "aistudio"):
    sampler, scheduler = SAMPLERS[family]
    g["3"] = {"class_type": "KSampler",
              "inputs": {"seed": int(seed), "steps": int(steps), "cfg": float(cfg),
                         "sampler_name": sampler, "scheduler": scheduler,
                         "denoise": float(denoise), "model": model_src,
                         "positive": pos_src, "negative": neg_src, "latent_image": latent_src}}
    g["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": vae_src}}
    g["9"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["8", 0]}}


def txt2img(spec: dict, positive: str, negative: str, width: int, height: int, seed: int,
            steps: int, cfg: float, loras: Optional[List[dict]] = None,
            batch_size: int = 1, guidance: float = 3.5) -> Dict:
    family = spec["family"]
    g: Dict = {}
    model_src, clip_src, vae_src = _loaders(g, spec, loras)
    pos_src, neg_src = _conditioning(g, family, clip_src, positive, negative, guidance)
    latent_node = "EmptyLatentImage" if family in ("sdxl", "sd") else "EmptySD3LatentImage"
    g["5"] = {"class_type": latent_node,
              "inputs": {"width": int(width), "height": int(height), "batch_size": int(batch_size)}}
    _sample_decode_save(g, family, model_src, pos_src, neg_src, ["5", 0], vae_src,
                        seed, steps, cfg)
    return g


def img2img(spec: dict, image_name: str, positive: str, negative: str, denoise: float,
            seed: int, steps: int, cfg: float, loras: Optional[List[dict]] = None,
            guidance: float = 3.5) -> Dict:
    family = spec["family"]
    g: Dict = {}
    model_src, clip_src, vae_src = _loaders(g, spec, loras)
    pos_src, neg_src = _conditioning(g, family, clip_src, positive, negative, guidance)
    g["10"] = {"class_type": "LoadImage", "inputs": {"image": image_name}}
    g["12"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["10", 0], "vae": vae_src}}
    _sample_decode_save(g, family, model_src, pos_src, neg_src, ["12", 0], vae_src,
                        seed, steps, cfg, denoise=denoise)
    return g


def inpaint(spec: dict, image_name: str, mask_name: str, positive: str, negative: str,
            denoise: float, seed: int, steps: int, cfg: float,
            loras: Optional[List[dict]] = None, guidance: float = 3.5) -> Dict:
    """Inpainting dla zwyklych modeli: VAEEncode + SetLatentNoiseMask (dziala z denoise < 1)."""
    family = spec["family"]
    g: Dict = {}
    model_src, clip_src, vae_src = _loaders(g, spec, loras)
    pos_src, neg_src = _conditioning(g, family, clip_src, positive, negative, guidance)
    g["10"] = {"class_type": "LoadImage", "inputs": {"image": image_name}}
    g["13"] = {"class_type": "LoadImageMask", "inputs": {"image": mask_name, "channel": "red"}}
    g["12"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["10", 0], "vae": vae_src}}
    g["14"] = {"class_type": "SetLatentNoiseMask",
               "inputs": {"samples": ["12", 0], "mask": ["13", 0]}}
    _sample_decode_save(g, family, model_src, pos_src, neg_src, ["14", 0], vae_src,
                        seed, steps, cfg, denoise=denoise)
    return g


def upscale(image_name: str, upscale_model: str, prefix: str = "aistudio_up") -> Dict:
    """Powiekszanie obrazu modelem ESRGAN (niezalezne od modelu obrazu)."""
    return {
        "20": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "21": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": upscale_model}},
        "22": {"class_type": "ImageUpscaleWithModel",
               "inputs": {"upscale_model": ["21", 0], "image": ["20", 0]}},
        "23": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["22", 0]}},
    }
