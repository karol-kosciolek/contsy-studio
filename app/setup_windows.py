# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Windows installation and read-only environment checks. Run via install.bat."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parent.parent
REPOS = [
    ('ComfyUI', 'https://github.com/Comfy-Org/ComfyUI.git', '7cebca52227c63b4180ceed448cbd7ad22903b6d'),
    ('ComfyUI/custom_nodes/ComfyUI-GGUF', 'https://github.com/city96/ComfyUI-GGUF.git', '6ea2651e7df66d7585f6ffee804b20e92fb38b8a'),
    ('ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite', 'https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git', '4d907bee61e92c2e65af3bd6383a4e4d356126d1'),
    ('ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation', 'https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git', '26545cc2dd95bc3d27f056016300673bdeee78f5'),
    ('vendor/LatentSync', 'https://github.com/ShmuelRonen/ComfyUI-LatentSyncWrapper.git', '360d5283d7276aee68b4237b1387e594e4ce640e'),
    ('vendor/musubi-tuner', 'https://github.com/kohya-ss/musubi-tuner.git', 'f8a1b03794a49239a3539015075f5123d6c07d66'),
]
# Contsy's changes to pinned extensions, stored under patches/<folder name>/. Each file maps to
# the SHA-256 of the pinned upstream file with LF line endings (None for a new file). A file
# that matches neither the upstream nor the patched version is preserved and reported.
PATCHES = ROOT / 'patches'
OVERLAYS = {
    'ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation': {  # exact source/target fps for RIFE VFI
        'vfi_models/rife/__init__.py': '416616e737e6bafb62b6df83eeb9829377fa45a2e2c403f4d818d8ce9dd9b11d',
        'vfi_models/rife/frame_rate.py': None,
    },
}
# VoxCPM declares funasr but never imports it. funasr pulls in kaldiio, whose license allows
# evaluation only (docs/LICENSING.md), so VoxCPM is installed without its own dependency list
# and both packages are removed from earlier installations.
VOXCPM = 'voxcpm==2.0.3'
UNUSED_PACKAGES = ('funasr', 'kaldiio')
_UNUSED_REQUIREMENT = re.compile(r'voxcpm \S+ requires funasr, which is not installed\.')


def run(args, **kwargs):
    return subprocess.run([str(a) for a in args], check=True, cwd=ROOT, **kwargs)


def prepare_venv():
    """Repair an incompatible environment by preserving it before recreation."""
    environment = ROOT / "venv"
    interpreter = environment / "Scripts/python.exe"
    if interpreter.is_file():
        try:
            result = subprocess.run([str(interpreter), "-c",
                "import sys; assert sys.version_info[:2] in ((3,11),(3,12)) and sys.maxsize>2**32"],
                cwd=ROOT, capture_output=True, timeout=30, check=False)
            if result.returncode == 0:
                return
        except (OSError, subprocess.TimeoutExpired):
            pass
    if environment.exists() or environment.is_symlink():
        expected = ROOT.resolve() / "venv"
        if environment.resolve() != expected or environment.is_symlink():
            raise RuntimeError("Refusing to replace a redirected venv folder. Preserve it manually and retry.")
        preserved = ROOT / ("venv.preserved_" + uuid.uuid4().hex)
        environment.rename(preserved)
        print(f"[OK] Previous environment preserved at {preserved}")
    run([sys.executable, "-m", "venv", environment])


def prepare_repository(folder, url, revision):
    destination = ROOT / folder
    if destination.exists():
        # Only a clone with no working files can be safely resumed automatically.
        if (destination / ".git").is_dir() and not any(p.name != ".git" for p in destination.iterdir()):
            remote = run(["git", "-C", destination, "remote", "get-url", "origin"],
                         capture_output=True, text=True).stdout.strip()
            if remote.rstrip("/").removesuffix(".git") != url.rstrip("/").removesuffix(".git"):
                raise RuntimeError(f"The empty checkout at {folder} has a different origin; it was preserved.")
            run(["git", "-C", destination, "fetch", "origin", revision])
            run(["git", "-C", destination, "checkout", "--detach", revision])
        else:
            print(f"[OK] Preserving existing {folder}. Expected revision: {revision}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="." + destination.name + ".install_", dir=destination.parent))
    try:
        run(["git", "clone", "--no-checkout", url, stage])
        run(["git", "-C", stage, "checkout", "--detach", revision])
        stage.rename(destination)
    except BaseException:
        print(f"[ERROR] Incomplete checkout preserved at {stage}. Retry install.bat.")
        raise
    return destination


def _lf(data: bytes) -> bytes:
    return data.replace(b'\r\n', b'\n')


def overlay_state(folder):
    """Paths of Contsy's patched files that are not in place (empty when fully applied)."""
    missing = []
    for relative in OVERLAYS.get(folder, {}):
        target = ROOT / folder / relative
        patched = (PATCHES / Path(folder).name / relative).read_bytes()
        if not target.is_file() or _lf(target.read_bytes()) != _lf(patched):
            missing.append(relative)
    return missing


def apply_overlays(folder, destination):
    """Install Contsy's changes to a pinned extension without overwriting other local edits."""
    for relative, upstream in OVERLAYS.get(folder, {}).items():
        patched = (PATCHES / Path(folder).name / relative).read_bytes()
        target = destination / relative
        current = _lf(target.read_bytes()) if target.is_file() else None
        if current == _lf(patched):
            continue
        if current is not None and hashlib.sha256(current).hexdigest() != upstream:
            raise RuntimeError(f'{folder}/{relative} has local changes, so it was preserved. '
                               'Restore the pinned version and run install.bat again.')
        target.write_bytes(patched)
    if folder in OVERLAYS:
        print(f'[OK] Contsy changes applied to {folder}')


def voxcpm_requirements():
    """VoxCPM's declared dependencies, without funasr and the development extras."""
    listed = run([sys.executable, '-c',
                  'import importlib.metadata as m; print(*(m.requires("voxcpm") or []), sep="\\n")'],
                 capture_output=True, text=True).stdout.splitlines()
    return [line.strip() for line in listed if line.strip() and 'extra ==' not in line
            and re.split(r'[\s;<>=!~\[(]', line.strip(), maxsplit=1)[0].lower() not in UNUSED_PACKAGES]


def broken_requirements():
    """Problems reported by pip check, except the funasr requirement that VoxCPM never uses."""
    output = subprocess.run([sys.executable, '-m', 'pip', 'check'], cwd=ROOT, capture_output=True,
                            text=True, check=False).stdout
    return [line.strip() for line in output.splitlines() if line.strip()
            and not line.startswith('No broken requirements')
            and not _UNUSED_REQUIREMENT.fullmatch(line.strip())]


def check_environment(cuda=True):
    errors = []
    if sys.version_info[:2] not in ((3, 11), (3, 12)) or sys.maxsize <= 2**32:
        errors.append('Use Python 3.11 or 3.12 (64-bit).')
    for exe in ('git', 'ffmpeg', 'ffprobe'):
        if not shutil.which(exe):
            errors.append(f'{exe} is missing. Run install.bat to install prerequisites.')
    if not (ROOT / 'ComfyUI/main.py').is_file():
        errors.append('ComfyUI is missing. Run install.bat.')
    else:
        for folder in OVERLAYS:
            if (ROOT / folder).is_dir() and overlay_state(folder):
                errors.append(f'Contsy changes to {folder} are missing. Run install.bat.')
    for package in ('gradio', 'requests', 'Pillow', 'huggingface_hub', 'voxcpm', 'opencv-python'):
        try:
            print(f'[OK] {package} {importlib.metadata.version(package)}')
        except importlib.metadata.PackageNotFoundError:
            errors.append(f'{package} is missing. Run install.bat.')
    broken = broken_requirements()
    errors += [f'{line} Run install.bat.' for line in broken]
    if not broken:
        print('[OK] Python dependencies are consistent.')
    if cuda:
        try:
            import torch
            if not torch.cuda.is_available():
                errors.append('CUDA is unavailable. Install an NVIDIA driver and run install.bat.')
            else:
                torch.ones(1, device='cuda')
                torch.cuda.synchronize()
                print(f'[OK] CUDA {torch.version.cuda}: {torch.cuda.get_device_name(0)}')
        except Exception as exc:
            errors.append(f'CUDA check failed: {exc}')
    if shutil.disk_usage(ROOT).free < 40 * 1024**3:
        print('[WARNING] Less than 40 GB free. Video model bundles may need more space.')
    for error in errors:
        print('[ERROR]', error)
    return not errors


def install():
    if sys.version_info[:2] not in ((3, 11), (3, 12)) or sys.maxsize <= 2**32:
        raise RuntimeError('Python 3.11 or 3.12 (64-bit) is required.')
    pip = [sys.executable, '-m', 'pip', 'install']
    run(pip + ['--upgrade', 'pip', 'wheel', 'setuptools'])
    run(pip + ['torch==2.14.1+cu130', 'torchvision==0.29.1+cu130', 'torchaudio==2.11.0+cu130',
               '--index-url', 'https://download.pytorch.org/whl/cu130'])
    constraints = ROOT / 'venv/torch-constraints.txt'
    constraints.parent.mkdir(exist_ok=True)
    constraints.write_text(''.join(n+'=='+importlib.metadata.version(n)+'\n' for n in
                                  ('torch', 'torchvision', 'torchaudio')), encoding='utf-8')
    os.environ['PIP_CONSTRAINT'] = str(constraints)
    for folder, url, revision in REPOS:
        destination = prepare_repository(folder, url, revision)
        apply_overlays(folder, destination)
        req = destination / ('requirements-no-cupy.txt' if 'Frame-Interpolation' in folder else 'requirements.txt')
        if req.is_file() and 'musubi-tuner' not in folder:
            run(pip + ['-r', req])
    run(pip + ['--no-deps', VOXCPM])
    run(pip + ['-r', 'requirements.txt', *voxcpm_requirements(), 'opencv-python==5.0.0.93',
               'accelerate', 'bitsandbytes', 'voluptuous', 'ftfy', 'easydict'])
    run([sys.executable, '-m', 'pip', 'uninstall', '-y', *UNUSED_PACKAGES])
    if not check_environment():
        raise RuntimeError('Environment verification failed. See the errors above.')
    from . import config
    config.ensure_dirs()
    print('[OK] The app and its engine are installed.')


TIER_LABELS = {'lite': 'Lite', 'full': 'Full quality'}


def _missing_gb(model_ids) -> float:
    """Catalog files not on disk yet, judged by name and size (hashes are checked while installing)."""
    from . import config, model_manager
    missing = {}
    for model_id in model_ids:
        for spec in (config.get_model(model_id) or {}).get('files', []):
            target = model_manager._file_target(spec)
            size = spec.get('size_bytes') or 0
            if not (target.is_file() and (not size or target.stat().st_size == size)):
                missing[str(target)] = size
    return sum(missing.values()) / 1e9


def choose_tier(answer=None) -> str:
    """'lite' or 'full': the installer argument, else the user's pick, else the default."""
    from . import download_models, model_manager, prefs, vram
    if answer in model_manager.TIERS:
        return answer
    recommended = model_manager.default_tier()
    current = prefs.load().get('tier')
    default = current if current in model_manager.TIERS else recommended
    gpu = vram.get_gpu_info()
    card = f'{gpu.name}, {gpu.total_gb:.0f} GB' if gpu.total_gb else 'not detected'
    size = {t: download_models.size_gb(download_models.model_ids('all', t)) for t in model_manager.TIERS}
    print()
    print('Choose the model set. Both have the same features; models keep their own licenses (MODEL_LICENSES.md).')
    print(f'  1) Lite          for graphics cards with 8-12 GB of VRAM, about {size["lite"]:.0f} GB to download')
    print(f'  2) Full quality  for graphics cards with 16 GB or more, about {size["full"]:.0f} GB to download')
    print(f'Your graphics card: {card}. Recommended: {TIER_LABELS[recommended]}.')
    try:
        reply = input(f'Type 1 or 2 and press Enter, or just press Enter for {TIER_LABELS[default]}: ').strip()
    except EOFError:
        reply = ''
    return {'1': 'lite', '2': 'full'}.get(reply, default)


def install_models(answer=None) -> bool:
    """Download and verify every model of the chosen set. A rerun resumes and skips finished files."""
    if answer == 'skip':
        print('[OK] Model downloads skipped: each feature downloads its models the first time you use it.')
        return True
    from . import download_models, model_manager, prefs
    tier = choose_tier(answer)
    ids = download_models.model_ids('all', tier)
    need, free = _missing_gb(ids), shutil.disk_usage(ROOT).free / 1e9
    if need + 10 > free:  # room for temporary video and training files
        print(f'[ERROR] The {TIER_LABELS[tier]} models need {need:.0f} GB more, but only {free:.0f} GB is free here.')
        print('Free up space or choose Lite, then run install.bat again.')
        return False
    prefs.save(tier=tier)
    if need:
        print(f'Downloading the {TIER_LABELS[tier]} models ({need:.0f} GB). Leave this window open.')
        print('If the download stops, run install.bat again: finished files are kept.')
    else:
        print(f'Checking the {TIER_LABELS[tier]} models...')

    def progress(frac: float, desc: str) -> None:
        print(f'\r  {frac * 100:5.1f}%  {desc[:90]:90s}', end='', flush=True)

    error = model_manager.ensure(ids, progress, exact=True)
    print()
    if error:
        print('[ERROR]', error)
        return False
    print(f'[OK] All {TIER_LABELS[tier]} models are installed and verified.')
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--install', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--prepare-venv', action='store_true')
    parser.add_argument('--models', nargs='?', const='', default=None, metavar='lite|full|skip',
                        help='download every model of a set (asks when no set is given)')
    args = parser.parse_args()
    try:
        if args.prepare_venv:
            prepare_venv()
            return 0
        if args.models is not None:
            return 0 if install_models(args.models.lower() or None) else 1
        if args.plan:
            print(json.dumps(REPOS, indent=2))
            return 0
        if args.install:
            install()
            return 0
        return 0 if check_environment() else 1
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f'[ERROR] Installation stopped: {exc}')
        print('No user outputs, models or character data have been removed. Fix the error and run install.bat again.')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
