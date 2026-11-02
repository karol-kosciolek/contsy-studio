# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Download model bundles in advance instead of on first use.

    venv\\Scripts\\python.exe -m app.download_models                 list bundles and sizes
    venv\\Scripts\\python.exe -m app.download_models all             every model, full quality
    venv\\Scripts\\python.exe -m app.download_models all --lite      every model, Lite set
    venv\\Scripts\\python.exe -m app.download_models images|video    one bundle

Without --full/--lite the model set chosen in Settings (or suited to this GPU) is used.
Downloads resume after interruptions and are verified like downloads started from the app.
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from . import config, model_manager

BUNDLES = {"images": "obrazy", "video": "wideo"}  # set keys in config/models.json


def model_ids(bundle: str, tier_name: str = "full") -> List[str]:
    """Catalog entries of a bundle in the given model set."""
    if bundle == "all":
        base = [m["id"] for m in config.CATALOG.get("models", []) if not m["id"].endswith("-lite")]
    else:
        base = list(config.CATALOG.get("sets", {}).get(BUNDLES[bundle], {}).get("models", []))
    return list(dict.fromkeys(model_manager.tiered(i, tier_name) for i in base))


def size_gb(ids: List[str]) -> float:
    """Download size; files shared by several entries (VAE, LoRAs) count once."""
    files = {}
    for model_id in ids:
        for spec in (config.get_model(model_id) or {}).get("files", []):
            files[(spec["dest"], spec.get("save_as") or spec["filename"])] = spec.get("size_bytes") or 0
    return sum(files.values()) / 1e9


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Download Contsy model bundles in advance.")
    parser.add_argument("bundle", nargs="?", choices=[*BUNDLES, "all"])
    chosen = parser.add_mutually_exclusive_group()
    chosen.add_argument("--full", action="store_const", const="full", dest="tier",
                        help="full-quality models (16 GB+ VRAM)")
    chosen.add_argument("--lite", action="store_const", const="lite", dest="tier",
                        help="Lite models (8-12 GB VRAM)")
    args = parser.parse_args(argv)
    if not args.bundle:
        for tier_name in model_manager.TIERS:
            for bundle in (*BUNDLES, "all"):
                ids = model_ids(bundle, tier_name)
                print(f"{tier_name:5s} {bundle:7s} {size_gb(ids):6.1f} GB  ({len(ids)} models)")
        print(f"This PC uses the '{model_manager.tier()}' set. Run again with a bundle name "
              "(and --full or --lite) to download it.")
        return 0

    tier_name = args.tier or model_manager.tier()

    def progress(frac: float, desc: str) -> None:
        print(f"\r{frac * 100:5.1f}%  {desc[:100]:100s}", end="", flush=True)

    error = model_manager.ensure(model_ids(args.bundle, tier_name), progress, exact=True)
    print()
    if error:
        print(error, file=sys.stderr)
        return 1
    print(f"Ready: every model in '{args.bundle}' ({tier_name}) is installed and verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
