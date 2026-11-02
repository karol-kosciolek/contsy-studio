# SPDX-FileCopyrightText: 2024 Bytedance Ltd. and/or its affiliates
# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only AND Apache-2.0
# Adapted from scripts/inference.py of LatentSync (https://github.com/bytedance/LatentSync),
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates, licensed under the Apache License 2.0
# (LICENSES/Apache-2.0.txt). Modified for Contsy: local Whisper/VAE loading, MediaPipe face
# detection and a short-lived worker process. See NOTICE and THIRD_PARTY_NOTICES.md.
"""Proces roboczy lip-sync (LatentSync 1.6), uruchamiany przez app/lipsync.py.

LatentSync dziala w osobnym, krotkotrwalym procesie zamiast jako wezel ComfyUI: po
zakonczeniu proces konczy sie, a system odzyskuje cala pamiec GPU. Jako wezel zostawial
w ComfyUI ok. 5 GB VRAM (DeepCache), na stale ograniczal proces ComfyUI do 80% VRAM
i juz przy starcie podmienial mu folder TEMP - dlatego kod ComfyUI-LatentSyncWrapper
lezy w vendor/LatentSync, poza custom_nodes.

Wzgledem scripts/inference.py z ComfyUI-LatentSyncWrapper:
  - Whisper i VAE laduja sie z lokalnych plikow (bez pobierania z internetu),
  - twarz wykrywa MediaPipe Face Landmarker (Apache 2.0) zamiast InsightFace, ktorego
    modele sa wylacznie do niekomercyjnych badan.

Pipeline LatentSync tworzy i kasuje folder "temp" w biezacym katalogu - app/lipsync.py
uruchamia ten proces w osobnym katalogu tymczasowym.

Uzycie: python lipsync_worker.py <wideo> <narracja.wav> <wynik.mp4> [--steps N] [--seed N]
Postep: linie "PROGRESS <ulamek 0-1> <opis>" na stdout.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WRAPPER_DIR = os.path.join(ROOT, "vendor", "LatentSync")
# Pakiet latentsync z wrappera zamiast katalogu app/ (jego moduly moglyby przeslaniac biblioteki)
sys.path[0] = WRAPPER_DIR

import argparse  # noqa: E402
import math  # noqa: E402
import types  # noqa: E402

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

import numpy as np  # noqa: E402
import soundfile  # noqa: E402
import torch  # noqa: E402

CKPT_DIR = os.path.join(WRAPPER_DIR, "checkpoints")
LANDMARKER_PATH = os.path.join(ROOT, "models", "mediapipe", "face_landmarker.task")
sys.path.append(ROOT)
from app import lowvram  # noqa: E402
from app.aux_models import ensure_landmarker
# Punkty MediaPipe Face Mesh: obie brwi (gorny i dolny obrys) i dolna czesc nosa
_BROW_A = [70, 63, 105, 66, 107, 46, 53, 52, 65, 55]
_BROW_B = [300, 293, 334, 296, 336, 276, 283, 282, 295, 285]
_NOSE = [1, 2, 98, 327]
_FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
              400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
              54, 103, 67, 109]


def progress(frac, desc):
    print(f"PROGRESS {frac:.3f} {desc}", flush=True)


class MediaPipeFaceDetector:
    """Zamiennik latentsync.utils.face_detector.FaceDetector (bbox + uklad 106 punktow).

    LatentSync czyta z ukladu InsightFace tylko punkty [43, 48, 49, 51, 50] (brew po lewej
    stronie obrazu), [101:106] (brew po prawej) i [74, 77, 83, 86] (nos) - wypelniamy je
    odpowiednikami z MediaPipe; reszta ukladu nie jest uzywana.
    """

    def __init__(self, device="cuda"):
        import mediapipe as mp

        ensure_landmarker()
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=LANDMARKER_PATH),
            running_mode=mp.tasks.vision.RunningMode.IMAGE, num_faces=1)
        self._mp = mp
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self._last = (None, None)
        self._misses = 0

    def __call__(self, frame, threshold=0.5):
        h, w = frame.shape[:2]
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                               data=np.ascontiguousarray(frame, dtype=np.uint8))
        result = self.landmarker.detect(image)
        if not result.face_landmarks:
            # pojedyncze klatki bez twarzy (ruch, rozmycie, obrot glowy) - poprzednia pozycja
            # zamiast przerwania calego ruchu ust ("Face not detected")
            self._misses += 1
            return self._last if self._misses <= 12 else (None, None)
        self._misses = 0
        pts = np.array([[p.x * w, p.y * h] for p in result.face_landmarks[0]])
        brow_a, brow_b = pts[_BROW_A].mean(axis=0), pts[_BROW_B].mean(axis=0)
        left, right = (brow_a, brow_b) if brow_a[0] < brow_b[0] else (brow_b, brow_a)
        lmk = np.zeros((106, 2))
        lmk[[43, 48, 49, 51, 50]] = left
        lmk[101:106] = right
        lmk[[74, 77, 83, 86]] = pts[_NOSE]
        oval = pts[_FACE_OVAL]
        bbox = (max(0, int(oval[:, 0].min())), max(0, int(oval[:, 1].min())),
                min(w, int(oval[:, 0].max())), min(h, int(oval[:, 1].max())))
        self._last = (bbox, lmk)
        return bbox, lmk


_face_detector_module = types.ModuleType("latentsync.utils.face_detector")
_face_detector_module.FaceDetector = MediaPipeFaceDetector
sys.modules["latentsync.utils.face_detector"] = _face_detector_module

from DeepCache import DeepCacheSDHelper  # noqa: E402
from diffusers import AutoencoderKL, DDIMScheduler  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from latentsync.models.unet import UNet3DConditionModel  # noqa: E402
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline  # noqa: E402
from latentsync.whisper.audio2feature import Audio2Feature  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("audio")
    parser.add_argument("output")
    parser.add_argument("--steps", type=int, default=20)
    # 2.5 zamiast domyslnego 1.5: w tescie ruch ust przy mowie +48%, bez drgan w ciszy
    # (usmiechnieta twarz przy 1.5 ledwo artykulowala), ok. 20% dluzej
    parser.add_argument("--guidance", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=42)
    # GPUs under 15 GB: the VAE encodes and decodes one frame at a time instead of 16 at once
    parser.add_argument("--low-vram", action="store_true")
    args = parser.parse_args()

    config = OmegaConf.load(os.path.join(WRAPPER_DIR, "configs", "unet", "stage2_512.yaml"))
    dtype = torch.float16
    torch.backends.cudnn.benchmark = True

    progress(0.02, "Lip sync: loading the model")
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                              clip_sample=False, num_train_timesteps=1000,
                              prediction_type="epsilon", set_alpha_to_one=False, steps_offset=1)
    audio_encoder = Audio2Feature(model_path=os.path.join(CKPT_DIR, "whisper", "tiny.pt"),
                                  device="cuda", num_frames=config.data.num_frames,
                                  audio_feat_length=config.data.audio_feat_length)
    vae = AutoencoderKL.from_pretrained(os.path.join(CKPT_DIR, "vae"), torch_dtype=dtype,
                                        local_files_only=True)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0
    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model), os.path.join(CKPT_DIR, "latentsync_unet.pt"),
        device="cpu")
    pipeline = LipsyncPipeline(vae=vae, audio_encoder=audio_encoder, unet=unet.to(dtype=dtype),
                               scheduler=scheduler).to("cuda")
    if args.low_vram:
        pipeline.enable_vae_slicing()
        lowvram.chunk_feed_forward(unet)
        lowvram.split_cfg(unet)
    else:
        helper = DeepCacheSDHelper(pipe=pipeline)
        helper.set_params(cache_interval=3, cache_branch_id=0)
        helper.enable()
    torch.manual_seed(args.seed)

    # Postep krokow odszumiania: partie po num_frames klatek (25 fps) na dlugosc narracji
    frames = math.ceil(soundfile.info(args.audio).duration * 25)
    total = max(1, math.ceil(frames / config.data.num_frames) * args.steps)
    done = [0]

    def on_step(step, timestep, latents):
        done[0] += 1
        progress(0.1 + 0.85 * min(done[0] / total, 1.0), f"Lip sync: step {done[0]}/{total}")

    progress(0.08, "Lip sync: analyzing faces")
    pipeline(video_path=args.video, audio_path=args.audio, video_out_path=args.output,
             num_frames=config.data.num_frames, num_inference_steps=args.steps,
             guidance_scale=args.guidance, weight_dtype=dtype,
             width=config.data.resolution, height=config.data.resolution,
             mask_image_path=os.path.join(WRAPPER_DIR, "latentsync", "utils", "mask.png"),
             callback=on_step, callback_steps=1)
    if not os.path.exists(args.output):
        sys.exit("LatentSync did not save an output")
    progress(1.0, "Lip sync: done")


if __name__ == "__main__":
    main()
