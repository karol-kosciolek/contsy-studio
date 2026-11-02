# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Proces roboczy syntezy mowy (VoxCPM2), uruchamiany przez app/tts.py.

VoxCPM2 (Apache 2.0, 48 kHz) dziala w osobnym, krotkotrwalym procesie - po zakonczeniu
cala pamiec GPU wraca do systemu (jak lip-sync). Barwe glosu daje probka z
assets/voices (wygenerowana tym samym modelem - nie nagranie prawdziwej osoby), wiec
postac brzmi tak samo w kazdym filmie. Glosy z opisu slownego ("Voice Design")
wypadaly w ocenie jakosci (UTMOS) wyraznie gorzej, dlatego z nich nie korzystamy.

Uzycie: tts_worker.py <zadanie.json> - lista {"text", "reference", "output"}; model
laduje sie raz na wszystkie kwestie (dialog kilku postaci).
Postep: linie "PROGRESS <ulamek 0-1> <opis>" na stdout.
"""
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "voxcpm2")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402


def progress(frac, desc):
    print(f"PROGRESS {frac:.3f} {desc}", flush=True)


def main():
    with open(sys.argv[1], encoding="utf-8") as fh:
        items = json.load(fh)

    from voxcpm import VoxCPM

    progress(0.1, "Voice: loading the model")
    model = VoxCPM.from_pretrained(MODEL_DIR, load_denoiser=False, optimize=False)
    for i, item in enumerate(items):
        progress(0.3 + 0.7 * i / len(items), f"Voice: line {i + 1}/{len(items)}")
        # ta wersja voxcpm nie przyjmuje seeda - ustawiamy go globalnie (powtarzalny wynik)
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        # 20 krokow: najlepsza srednia ocena jakosci (UTMOS) w testach 10/20/30 krokow i cfg 1,5-3
        wav = model.generate(text=item["text"], reference_wav_path=item["reference"],
                             cfg_value=2.0, inference_timesteps=20)
        sf.write(item["output"], wav, model.tts_model.sample_rate)
    progress(1.0, "Voice: done")


if __name__ == "__main__":
    main()
