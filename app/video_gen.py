# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Serwis generowania wideo: ultra realistyczne 1080p, tryb Jakosc, Fast albo Podglad (MODES).

Pipeline: dialogi (VoxCPM2, osobny proces) -> klatki startowe z obrazow pomocniczych
(Qwen-Image-Edit-2511, opcjonalnie) -> wideo 720p albo 480p (ComfyUI: Wan 2.2 A14B + LoRA lightx2v,
interpolacja RIFE) -> dzwiek tla (Stable Audio Open) -> ruch ust (LatentSync, osobny proces)
-> 1080p (SeedVR2; Podglad bez tego etapu - zostaje 480p) -> miks i montaz z metadanymi
(FFmpeg). Etapy audio sa opcjonalne - brak modelu/FFmpeg nie przerywa generacji.

Wan A14B generuje 16 fps i jest uczony na 5-sekundowych ujeciach w 720p, wiec dluzsze wideo
powstaje z kolejnych czesci po 5 s, z ktorych kazda startuje od ostatniej klatki poprzedniej
(obraz->wideo), a 1080p dorysowuje SeedVR2. Prompt moze miec kilka ujec ("Ujęcie 2: ...") -
kazde z wlasnym opisem i wlasnymi obrazami pomocniczymi (app/refs.py).
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image

from . import cancel, characters, config, content, dialogue, face_fix, ffmpeg_utils, history, llm
from . import model_manager, qc, refs, sound, templates, tts
from . import lipsync as lipsync_proc
from . import upscale
from .comfy_client import ComfyClient
from .vram import low_vram_hint

ProgressCb = Callable[[float, str], None]

NATIVE_FPS = 16          # Wan 2.2 A14B
SEGMENT_SECONDS = 5      # dlugosc, na ktorej model byl uczony (81 klatek)
MAX_SECONDS = 30  # 6 czesci po 5 s; kolejna czesc startuje z bezstratnej klatki z modelu
FPS = 30                 # docelowy plik, rowniez po LatentSync (natywnie 25 fps)
ASPECTS = {"Landscape 16:9": 16 / 9, "Portrait 9:16": 9 / 16, "Square 1:1": 1.0,
           "Portrait 3:4": 3 / 4}
ASPECT_ALIASES = {"Poziomo 16:9": "Landscape 16:9", "Pionowo 9:16": "Portrait 9:16",
                  "Kwadrat 1:1": "Square 1:1", "Portret 3:4": "Portrait 3:4"}
# Tryb (zakladka Wideo, Auto-plan, persona autopilota). Jakosc: Wan liczy 720p (najwyzsza
# rozdzielczosc, na ktorej byl uczony) w 6 krokach - w testach ostrzejsze twarze niz 4.
# Fast: 480p w 4 krokach (na tylu uczona jest LoRA lightx2v); 1080p i tak dorysowuje SeedVR2,
# ale twarze w szerszych kadrach sa miekksze. Pomiar 31.10 (5 s, RTX 4070 Ti Super, ta sama
# klatka startowa): Jakosc 13,9 min (Wan 585 s), Fast 6,5 min (Wan 143 s) - SeedVR2 to
# ok. 226 s w obu trybach. Twarz (jak w app/qc.py) srednio 0,67 w obu.
# Podglad: Fast bez SeedVR2 - film zostaje w 480p (do sprawdzania pomyslow, nie do publikacji:
# autopilot i paczki go nie biora).
QUALITY, FAST, PREVIEW = "quality", "fast", "preview"  # nazwy zna tez queue_manager (tempo ETA)
MODES = {QUALITY: {"pixels": 1280 * 720, "steps": 6}, FAST: {"pixels": 832 * 480, "steps": 4},
         PREVIEW: {"pixels": 832 * 480, "steps": 4, "upscale": False}}
MODE_LABELS = {QUALITY: "Quality", FAST: "Fast", PREVIEW: "Preview"}
PUBLISH_MODES = (QUALITY, FAST)  # tryby z gotowym 1080p - autopilot (persona) tylko te
FACE_KEEP = 0.6   # sila poprawki twarzy postaci na granicy czesci tej samej sceny (test:
                  # 0.35 prawie nic nie zmienialo, 0.6 przywracalo rysy bez zmiany miny)
OVERLAP_FRAMES = 4  # zakladka czesci (klatki Wan, 0,25 s) - dlugosc 81+4 = 4n+1 jak chce Wan
STEPS_MAX = 40
FAST_MAX_STEPS = 8   # do tylu krokow: LoRA przyspieszajaca lightx2v (cfg 1)


def sampling_plan(steps: int) -> Dict[str, float]:
    """Ustawienia probkowania dla liczby krokow. Ekspert high-noise robi pierwsza polowe.

    Do 8 krokow: LoRA lightx2v (uczona na 4 krokach), cfg 1, shift 5 - szybko.
    Powyzej: pelny model bez LoRA jak w oficjalnym szablonie ComfyUI (cfg 3,5, shift 8) -
    najwyzsza jakosc, ale cfg > 1 liczy kazdy krok dwa razy (ok. 2x dluzej na krok).
    """
    steps = min(max(int(steps), 2), STEPS_MAX)
    fast = steps <= FAST_MAX_STEPS
    return {"steps": steps, "split": max(1, steps // 2), "cfg": 1.0 if fast else 3.5,
            "shift": 5.0 if fast else 8.0, "lora": 1.0 if fast else 0.0}

# Oficjalny negatyw Wan (dodawany zawsze; przy cfg 1 dziala slabo, ale nie szkodzi)
WAN_NEGATIVE = ("oversaturated colors, overexposure, static scene, blurry details, subtitles, "
                "stylized artwork, painting, still image, gray cast, worst quality, low quality, "
                "JPEG artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly "
                "drawn face, deformed, disfigured, malformed limbs, fused fingers, motionless "
                "frame, cluttered background, three legs, crowded background, walking backward")

_OOM_HINTS = ("out of memory", "allocation on device", "oom")


@dataclass
class VideoOutput:
    ok: bool
    files: List[str] = field(default_factory=list)
    seed: int = -1
    error: str = ""
    notes: List[str] = field(default_factory=list)


def _client() -> ComfyClient:
    cfg = config.SETTINGS["comfyui"]
    return ComfyClient(cfg["host"], cfg["port"])


def _resolve_seed(seed: int) -> int:
    return random.randint(0, 2 ** 31 - 1) if seed is None or seed < 0 else int(seed)


def _friendly(exc: Exception) -> str:
    msg = str(exc)
    if any(h in msg.lower() for h in _OOM_HINTS):
        return ("Not enough GPU memory. Close other apps using the GPU (such as games or OBS) "
                "and try again.\n" + low_vram_hint(15.0))
    return f"Video generation failed: {msg}"


def frame_size(aspect: float, pixels: int, step: int = 16) -> Tuple[int, int]:
    """Rozmiar klatki o danych proporcjach i liczbie pikseli (wielokrotnosci `step`)."""
    aspect = min(max(aspect, 0.5), 2.0)
    w = round(math.sqrt(pixels * aspect) / step) * step
    h = round(w / aspect / step) * step   # z proporcji, zeby 1:1 bylo kwadratem
    return int(w), int(h)


def sizes(ratio: float, speed: str) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """(klatka liczona przez Wan w tym trybie, klatka trybu Jakosc). Ta druga to rozmiar klatek
    startowych (postac, obrazy pomocnicze: z wiekszej Wan ma wyrazniejsza twarz, sam ja
    zmniejszy) i podstawa 1080p - gotowy film ma ten sam rozmiar w obu trybach."""
    return frame_size(ratio, MODES[speed]["pixels"]), frame_size(ratio, MODES[QUALITY]["pixels"])


def mode_tag(speed: str) -> str:
    """Dopisek do nazwy zadania w kolejce - przy trybie innym niz Jakosc (domyslna)."""
    return {FAST: " · fast", PREVIEW: " · preview"}.get(speed, "")


def upscales(speed: str) -> bool:
    """Czy tryb konczy sie dorysowaniem 1080p (SeedVR2) - Podglad nie."""
    return MODES.get(speed, MODES[QUALITY]).get("upscale", True)


def rife_plan(native_fps: int, target_fps: int) -> int:
    """Mnoznik RIFE: najmniejszy, ktory daje FPS podzielny przez docelowy (bez szarpania);
    gdy takiego nie ma do x4 - najmniejszy, ktory go przekracza (nadmiar klatek odrzucamy)."""
    for mult in range(1, 5):
        if native_fps * mult >= target_fps and (native_fps * mult) % target_fps == 0:
            return mult
    return max(1, math.ceil(target_fps / native_fps))


def _lipsync_fps_workflow(video: str) -> dict:
    """Ten sam RIFE co przy Wan, z klatkami na dokladnej osi czasu 30 FPS.

    Model i ustawienia jakosci pochodza z istniejacego szablonu. Posredni zapis
    jest bezstratny, zeby dodatkowy etap nie dokladal kompresji obrazu.
    """
    rife = templates.load_raw("wan14b_t2v")["50"]
    rife["inputs"].update({"frames": ["2", 0], "multiplier": 2,
                           "source_fps": 25, "target_fps": FPS})
    return {
        "1": {"class_type": "LoadVideo", "inputs": {"file": video}},
        "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
        "50": rife,
        "90": {"class_type": "VHS_VideoCombine", "inputs": {
            "images": ["50", 0], "frame_rate": FPS, "loop_count": 0,
            "filename_prefix": "aistudio_lipsync_rife", "format": "video/h264-mp4",
            "pix_fmt": "yuv420p", "crf": 0, "save_metadata": False,
            "pingpong": False, "save_output": True,
        }},
    }


def _lipsync_to_output_fps(client: ComfyClient, source: Path, output: Path,
                          progress_cb: Optional[ProgressCb] = None) -> Tuple[bool, str]:
    """Interpoluje ruch ust i zachowuje czas audio przy konwersji 25 -> 30 fps."""
    duration = ffmpeg_utils.probe_duration(source)
    if not duration:
        return False, "Could not read video duration after lip sync."
    intermediate = output.with_name(output.stem + "_rife.mp4")
    try:
        client.prepare_pipeline("video_interpolation")
        result = client.run_and_wait(_lipsync_fps_workflow(client.upload_image(str(source))),
                                     progress_cb=progress_cb)
        item = next((it for it in result.files
                     if it.get("filename", "").startswith("aistudio_lipsync_rife")
                     and it["filename"].lower().endswith(".mp4")), None)
        if item is None:
            return False, "RIFE returned no video after lip sync."
        intermediate.write_bytes(client.download_output(item))
        return ffmpeg_utils.convert_fps(intermediate, output, FPS, duration=duration)
    except Exception as exc:
        return False, f"Interpolation to {FPS} FPS failed: {exc}"
    finally:
        intermediate.unlink(missing_ok=True)


def _models(kind: str) -> Dict[str, Optional[str]]:
    """Pliki paczki A14B z aktywnego zestawu modeli (full/Lite): kind = 't2v' lub 'i2v'."""
    model_id = f"wan22-a14b-{kind}"

    def pick(dest: str, *subs: str) -> Optional[str]:
        return model_manager.model_file(model_id, dest, *subs)

    return {"%UNET_HIGH%": pick("diffusion_models", "highnoise"),
            "%UNET_LOW%": pick("diffusion_models", "lownoise"),
            "%LORA_HIGH%": pick("loras", "high_noise"), "%LORA_LOW%": pick("loras", "low_noise"),
            "%CLIP%": pick("text_encoders"), "%VAE%": pick("vae")}


def _segments(seconds: int) -> List[int]:
    """Podzial na czesci po maks. 5 s, np. 12 s -> [5, 5, 2]."""
    full, rest = divmod(int(seconds), SEGMENT_SECONDS)
    return [SEGMENT_SECONDS] * full + ([rest] if rest else [])


def _shot_refs(shots: List[str], count: int, has_photo: bool) -> List[List[int]]:
    """Numery obrazow pomocniczych dla kazdego ujecia (do refs.MAX_REFS; wiecej niz miesci
    jeden przebieg modelu sklada refs.compose_refs). Bez numerow w prompcie wszystkie obrazy
    ida do pierwszego ujecia."""
    marked = [[n for n in refs.numbers(s) if 1 <= n <= count] for s in shots]
    if count and not any(marked):
        marked[0] = list(range(1, count + 1))
    return [used[:refs.MAX_REFS] for used in marked]


def generate(*args, **kwargs) -> VideoOutput:
    """Patrz _generate. Przy przerwaniu przez uzytkownika (cancel.Cancelled) sprzata pliki
    posrednie i przekazuje wyjatek dalej - do kolejki."""
    temp_files: List[Path] = []
    try:
        return _generate(*args, _temp=temp_files, **kwargs)
    except cancel.Cancelled:
        for tmp in temp_files:
            Path(tmp).unlink(missing_ok=True)
        raise
    except Exception as exc:
        for tmp in temp_files:
            Path(tmp).unlink(missing_ok=True)
        return VideoOutput(ok=False, error=_friendly(exc))


def _generate(prompt: str, seconds: int, aspect: str, seed: int = -1,
              image_path: Optional[str] = None, ref_images: Optional[List[str]] = None,
              sound_on: bool = True, style: str = "", steps: Optional[int] = None,
              progress_cb: Optional[ProgressCb] = None,
              _temp: Optional[List[Path]] = None,
              character: Optional[dict] = None,
              speed: str = QUALITY) -> VideoOutput:
    """Kwestie postaci w cudzyslowie w prompcie (np. says „Cześć!”) staja sie dialogiem:
    glos dobrany do plci mowiacego i ruch ust kazdej mowiacej postaci. Obrazy pomocnicze
    wskazane w prompcie numerem ([1], [2]...) trafiaja do klatki startowej ujecia.
    character - postac z LoRA (characters): klatka startowa kazdego ujecia to jej obraz
    z Z-Image + LoRA (zdjecia pomocnicze dokladaja do niego rzeczy), dalej Wan obraz -> wideo;
    naga klatka startowa (straz SFW, app/qc.py) konczy zadanie bledem.
    speed - tryb (MODES): Jakosc, Fast albo Podglad (480p bez SeedVR2); steps - wymuszona
    liczba krokow (domyslnie z trybu). sound_on=False - film bez dzwieku. Brakujace modele
    pobieraja sie same."""
    refusal = content.check(prompt, style)
    if refusal or llm.blocked(prompt, style):
        return VideoOutput(ok=False, error=refusal or llm.BLOCK_MSG)
    if not str(prompt or "").strip():
        return VideoOutput(ok=False, error="Enter a description for the video.")
    aspect = ASPECT_ALIASES.get(aspect, aspect)
    seed = _resolve_seed(seed)
    speed = speed if speed in MODES else QUALITY
    upscale_on = upscales(speed)
    plan = sampling_plan(steps or MODES[speed]["steps"])
    ref_images = [p for p in (ref_images or []) if p and Path(p).exists()]
    shots = refs.split_shots(prompt)
    seconds = min(max(int(seconds), 1, 2 * len(shots) if len(shots) > 1 else 1), MAX_SECONDS)
    notes: List[str] = []
    _, sound_prompt, lines = dialogue.parse(refs.plain(prompt))
    if not sound_on:
        lines = []
    background = sound_on
    use_lipsync = bool(sound_on and lines)

    def dcb(frac: float, desc: str) -> None:
        if progress_cb:
            progress_cb(0.01 * frac, desc)

    needed_models = ((["seedvr2-3b"] if upscale_on else [])
                     + (["qwen-image-edit-2511"] if ref_images else [])
                     + (["z-image-turbo"] if character else [])
                     + (["voxcpm2", "latentsync-16"] if lines else [])
                     + (["stable-audio-open"] if sound_on else []))
    err = model_manager.ensure(needed_models, dcb)
    if err:
        return VideoOutput(ok=False, error=err)
    timings: List[Tuple[float, float]] = []
    # Udzialy etapow w pasku postepu (i ETA kolejki): glos, wideo, tlo, ruch ust, 1080p
    # (Podglad bez 1080p - wideo i ruch ust zajmuja cala reszte paska)
    p_start = 0.05 if lines else 0.02
    if upscale_on:
        p_video, p_bg, p_ls = (0.50, 0.53, 0.55) if use_lipsync else (0.70, 0.74, 0.0)
        p_up = 0.78
    else:
        p_video, p_bg, p_ls = (0.60, 0.63, 0.65) if use_lipsync else (0.88, 0.94, 0.0)
        p_up = 0.95

    client = _client()
    if not client.system_stats():
        return VideoOutput(ok=False, error=(
            "ComfyUI is not responding. Start the app with start.bat "
            "or check comfyui.log."))

    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    temp_files: List[Path] = _temp if _temp is not None else []

    def fail(error: str) -> VideoOutput:
        for tmp in temp_files:
            Path(tmp).unlink(missing_ok=True)
        return VideoOutput(ok=False, error=error)

    # --- 1. Dialog (VoxCPM2 w osobnym procesie) - najpierw, bo moze wydluzyc wideo ---
    voice_file: Optional[str] = None
    if lines:
        if progress_cb:
            progress_cb(0.01, "Voice: preparing...")
        client.free_memory()  # model glosu dostaje karte bez modeli z ComfyUI
        line_files = [config.OUTPUTS_DIR / f"video_{stamp}_line{i + 1}.wav"
                      for i in range(len(lines))]
        voice_path = config.OUTPUTS_DIR / f"video_{stamp}_voice.wav"
        temp_files += [*line_files, voice_path]

        def tcb(frac: float, desc: str) -> None:
            if progress_cb:
                progress_cb(0.01 + (p_start - 0.01) * frac, desc)

        voices = dialogue.voices_for(lines)
        ok, msg = tts.synthesize_many(
            [(line.text, v, str(f)) for line, v, f in zip(lines, voices, line_files)], tcb)
        if ok:
            length, timings = dialogue.assemble([str(f) for f in line_files], voice_path)
            voice_file = str(voice_path)
            # dluzszy dialog -> dluzsze wideo zamiast zatrzymanej ostatniej klatki
            needed = math.ceil(length + 0.3)
            if needed > seconds:
                notes.append(f"Video extended from {seconds} to {min(needed, MAX_SECONDS)} s "
                             "to fit the dialogue.")
                seconds = min(needed, MAX_SECONDS)
        else:
            notes.append(msg)
            use_lipsync = False

    # --- Plan ujec: dlugosc, obrazy pomocnicze, czesci po 5 s ---
    shot_secs = refs.split_seconds(seconds, len(shots))
    if len(shot_secs) < len(shots):
        notes.append(f"The video is too short for {len(shots)} shots; using {len(shot_secs)}.")
        shots = shots[:len(shot_secs)]
    shot_refs = _shot_refs(shots, len(ref_images), bool(image_path))
    plan_parts = [(i, s) for i, secs in enumerate(shot_secs) for s in _segments(secs)]
    first_kind = "i2v" if (image_path or shot_refs[0] or character) else "t2v"
    kinds = [first_kind] + ["i2v"] * (len(plan_parts) - 1)
    err = model_manager.ensure([f"wan22-a14b-{k}" for k in dict.fromkeys(kinds)], dcb)
    if err:
        return fail(err)
    models = {k: _models(k) for k in set(kinds)}

    if image_path:  # proporcje jak na zdjeciu - bez obcinania kadru
        with Image.open(image_path) as im:
            ratio = im.width / im.height
    else:
        ratio = ASPECTS.get(aspect, 16 / 9)
    (width, height), (full_w, full_h) = sizes(ratio, speed)
    mult = rife_plan(NATIVE_FPS, FPS)

    def with_style(text: str) -> str:
        return f"{text.strip().rstrip('.')}, {style}" if style else text

    parts: List[str] = []
    overlaps: List[float] = []  # zakladka (s) kazdej czesci z poprzednia - join_segments
    overlap = 0.0
    start_image = image_path
    looks: Dict[int, str] = {}  # wyglad osob ze zdjec pomocniczych (llm.describe_person)
    done_shot = -1

    # --- 2. Wideo: kolejne ujecia i ich czesci ---
    for k, (shot_i, seg_seconds) in enumerate(plan_parts):
        lo = p_start + (p_video - p_start) * k / len(plan_parts)
        hi = p_start + (p_video - p_start) * (k + 1) / len(plan_parts)
        label = f"Part {k + 1}/{len(plan_parts)}: " if len(plan_parts) > 1 else "Video: "
        shot_visual = dialogue.parse(shots[shot_i])[0]
        if shot_i != done_shot:  # poczatek ujecia: klatka startowa z obrazow pomocniczych
            done_shot = shot_i
            used = shot_refs[shot_i]
            char_base = False
            if character and not (shot_i == 0 and image_path):
                ck = config.OUTPUTS_DIR / f"video_{stamp}_char{shot_i + 1}.png"
                temp_files.append(ck)
                if progress_cb:
                    progress_cb(lo, f"{label}starting frame - character {character['name']}...")
                # straz SFW i kontrola klatki startowej - zanim Wan zacznie ja ozywiac (app/qc.py)
                ok, msg = qc.render_checked(client, character, with_style(shot_visual), full_w,
                                            full_h, ck, seed + 2000 + shot_i, notes)
                if ok:
                    start_image, char_base = msg, True
                else:
                    return fail(f"Shot {shot_i + 1}: {msg}")
            if used:
                key = config.OUTPUTS_DIR / f"video_{stamp}_key{shot_i + 1}.png"
                temp_files.append(key)
                def kcb(frac: float, desc: str, lo=lo, label=label) -> None:
                    if progress_cb:
                        progress_cb(lo, f"{label}reference frame: {desc}")

                for n in used:  # wyglad osoby ze zdjecia (raz na zdjecie, zapamietany)
                    if n not in looks:
                        looks[n] = llm.describe_person(ref_images[n - 1], kcb)
                ok, msg = refs.compose_refs(
                    client, start_image, [(n, ref_images[n - 1]) for n in used],
                    with_style(shot_visual), full_w, full_h, key, seed + 1000 + shot_i,
                    continuing=shot_i > 0 and not char_base, progress_cb=kcb,
                    looks={n: looks[n] for n in used},
                    base_lead=refs.CHARACTER if char_base else "")
                if ok:
                    start_image = msg
                    if char_base:  # skladanie upraszczalo rysy - twarz z LoRA postaci
                        fk = config.OUTPUTS_DIR / f"video_{stamp}_key{shot_i + 1}_face.png"
                        temp_files.append(fk)
                        if face_fix.fix(client, character, start_image, fk,
                                        seed=seed + 3000 + shot_i)[0]:
                            start_image = str(fk)
                else:
                    return fail(f"Shot {shot_i + 1}: {msg}")
        kind = "i2v" if start_image else "t2v"
        if kind not in models:
            err = model_manager.ensure([f"wan22-a14b-{kind}"], dcb)
            if err:
                return fail(err)
            models[kind] = _models(kind)
        mapping = dict(models[kind])
        person_looks = ([character.get("look", "")] if character else []) + list(looks.values())
        # ta sama scena z postacia w nastepnej czesci - start z zakladki (patrz nizej)
        next_same = (bool(character) and k + 1 < len(plan_parts)
                     and plan_parts[k + 1][0] == shot_i)
        n_frames = NATIVE_FPS * seg_seconds + 1 + (OVERLAP_FRAMES if overlap else 0)
        mapping.update({
            # przy osobie ze zdjecia bez opisu skory/wlosow/wieku - inaczej Wan zmienial ja
            # w trakcie ujecia (np. opalal blada modelke ze zdjecia)
            "%POSITIVE%": with_style(
                (refs.neutral_look(refs.strip(shot_visual))
                 + "".join(f". The person has {lk}" for lk in dict.fromkeys(person_looks) if lk))
                if (any(shot_refs) or character) else refs.strip(shot_visual)),
            "%NEGATIVE%": WAN_NEGATIVE,
            "%WIDTH%": width, "%HEIGHT%": height,
            # czesc z zakladka jest o nia dluzsza - dlugosc filmu bez zmian
            "%LENGTH%": n_frames,
            # klatka startowa nastepnej czesci zapisana bezstratnie (PNG prosto z VAE) -
            # z pliku MP4 kazda czesc dziedziczylaby kompresje poprzedniej (do 6 czesci)
            "%LAST_INDEX%": n_frames - 1 - (OVERLAP_FRAMES if next_same else 0),
            "%STEPS%": plan["steps"], "%SPLIT%": plan["split"], "%SHIFT%": plan["shift"],
            "%CFG%": plan["cfg"], "%LORA_STRENGTH%": plan["lora"], "%SEED%": seed + k,
            "%RIFE_MULT%": mult, "%FPS%": FPS,
        })

        def vcb(node: str, value: int, maximum: int, lo=lo, hi=hi, label=label) -> None:
            """Wspolny licznik krokow obu ekspertow (wezly 30 i 31), potem RIFE (50)."""
            if not progress_cb:
                return
            if node in ("30", "31"):
                done = value + (plan["split"] if node == "31" else 0)
                frac, desc = 0.9 * done / plan["steps"], f"step {done}/{plan['steps']}"
            else:
                frac, desc = 0.9 + 0.1 * value / maximum, "frame interpolation (RIFE)"
            progress_cb(lo + (hi - lo) * frac, label + desc)

        if progress_cb:
            progress_cb(lo, label + "generating in the engine...")
        try:
            client.prepare_pipeline("video")
            if kind == "i2v":
                mapping["%IMAGE%"] = client.upload_image(start_image)
            result = client.run_and_wait(templates.build(f"wan14b_{kind}", mapping),
                                         node_progress_cb=vcb)
        except Exception as e:  # noqa: BLE001
            return fail(_friendly(e))

        item = next((it for it in result.files
                     if (it.get("filename") or "").lower().endswith(".mp4")), None)
        if item is None:
            return fail("The engine returned no video. Check comfyui.log.")
        part = config.OUTPUTS_DIR / f"video_{stamp}_part{k + 1}.mp4"
        part.write_bytes(client.download_output(item))
        parts.append(str(part))
        temp_files.append(part)
        overlaps.append(overlap)
        overlap = 0.0
        if k + 1 < len(plan_parts):
            frame = config.OUTPUTS_DIR / f"video_{stamp}_last{k + 1}.png"
            temp_files.append(frame)
            # postac i ta sama scena dalej: start z klatki chwile przed koncem, z twarza
            # poprawiona LoRA (bez tego po kilku czesciach postac odplywala); ten odcinek
            # przenika sie przy sklejaniu, wiec poprawka nie skacze w jednej klatce
            same_scene = next_same
            if result.images:  # bezstratna klatka z silnika (wezel 61 szablonu)
                frame.write_bytes(result.images[-1])
                ok, msg = True, str(frame)
            elif same_scene:
                ok, msg = ffmpeg_utils.frame_before_end(part, frame, OVERLAP_FRAMES / NATIVE_FPS)
            else:
                ok, msg = ffmpeg_utils.last_frame(part, frame)
            if not ok:
                notes.append(f"Video shortened to {k + 1} parts because the last frame "
                             f"could not be extracted ({msg}).")
                break
            start_image = str(frame)
            if same_scene:
                overlap = OVERLAP_FRAMES / NATIVE_FPS
                fixed = config.OUTPUTS_DIR / f"video_{stamp}_last{k + 1}_face.png"
                temp_files.append(fixed)
                if face_fix.fix(client, character, frame, fixed, denoise=FACE_KEEP,
                                seed=seed + 4000 + k, text=refs.strip(shot_visual))[0]:
                    start_image = str(fixed)

    # Sklejenie czesci i docelowy FPS (bez przekodowania, gdy nie ma czego zmieniac)
    raw_video = config.OUTPUTS_DIR / f"video_{stamp}_raw.mp4"
    temp_files.append(raw_video)
    if len(parts) == 1:
        ok, msg = ffmpeg_utils.convert_fps(parts[0], raw_video, FPS,
                                           duration=float(plan_parts[0][1]))
        if not ok:
            return fail(f"Could not save the 30 FPS video: {msg}")
    else:
        ok, msg = ffmpeg_utils.join_segments(parts, raw_video, FPS, overlaps,
                                             duration=sum(s for _, s in plan_parts[:len(parts)]))
        if not ok:
            return fail(f"FFmpeg could not assemble the video: {msg}")

    # --- 3. Dzwiek tla (Stable Audio Open w ComfyUI) - z opisu sceny ---
    bg_file: Optional[str] = None
    if background:
        if progress_cb:
            progress_cb(p_video, "Background audio...")
        bg_path = config.OUTPUTS_DIR / f"video_{stamp}_bg.flac"
        temp_files.append(bg_path)

        def bcb(frac: float, desc: str) -> None:
            if progress_cb:
                progress_cb(p_video + (p_bg - p_video) * frac, "Background audio: " + desc)

        ok, msg = sound.generate(client, sound_prompt, "",
                                 ffmpeg_utils.probe_duration(raw_video) or seconds, seed,
                                 bg_path, bcb)
        if ok:
            bg_file = str(bg_path)
        else:
            notes.append(msg)

    # --- 4. Ruch ust (LatentSync w osobnym procesie) - kazda mowiaca postac w kadrze ---
    video_track = raw_video
    track_fps = FPS
    lipsync_out = config.OUTPUTS_DIR / f"video_{stamp}_lipsync.mp4"
    temp_files.append(lipsync_out)
    if use_lipsync and voice_file:
        if progress_cb:
            progress_cb(p_ls, "Lip sync: preparing...")
        client.free_memory()  # LatentSync dostaje karte bez modeli wideo z ComfyUI

        def lcb(frac: float, desc: str) -> None:
            if progress_cb:
                progress_cb(p_ls + (p_up - p_ls) * frac, desc)

        ok, msg, ls_notes = lipsync_proc.run_dialogue(
            str(raw_video), voice_file, timings, [line.who for line in lines], str(lipsync_out),
            lcb)
        notes += ls_notes
        if ok:
            # LatentSync musi pracowac w 25 FPS. Interpolujemy przed SeedVR2,
            # aby wynik mial 30 FPS bez przyspieszenia filmu i rozjazdu dzwieku.
            fps_path = config.OUTPUTS_DIR / f"video_{stamp}_lipsync30.mp4"
            temp_files.append(fps_path)
            if progress_cb:
                progress_cb(p_up, "Interpolating lip sync to 30 FPS (RIFE)...")
            ok, msg = _lipsync_to_output_fps(client, lipsync_out, fps_path)
            if not ok:
                return fail(msg)
            video_track, track_fps = fps_path, FPS
        else:
            notes.append(f"Lip sync skipped ({msg}). Dialogue added as an audio track.")

    # --- 5. 1080p (SeedVR2) - po ruchu ust, zeby usta tez dostaly ostre szczegoly ---
    upscaled = False
    if upscale_on:
        if progress_cb:
            progress_cb(p_up, "1080p (SeedVR2)...")
        hd_path = config.OUTPUTS_DIR / f"video_{stamp}_1080p.mp4"
        temp_files.append(hd_path)

        def ucb(frac: float, desc: str) -> None:
            if progress_cb:
                progress_cb(p_up + (0.95 - p_up) * frac, desc)

        upscaled, msg = upscale.video(client, str(video_track), hd_path, full_w, full_h,
                                      track_fps, ucb)
        if upscaled:
            video_track = hd_path
        if msg:
            notes.append(msg)
    else:
        notes.append(f"Preview {width}×{height} — without 1080p upscaling (SeedVR2). For "
                     "publishing, generate a video in Quality or Fast mode.")
    out_w, out_h = upscale.target_size(full_w, full_h) if upscaled else (width, height)

    # --- 6. Montaz (FFmpeg) ---
    # Pod wideo (takze po lip-sync, ktory oddaje dzwiek 16 kHz) trafia oryginalna
    # narracja 48 kHz zmiksowana z tlem (tlo przycisza sie pod glosem).
    if progress_cb:
        progress_cb(0.96, "Assembling video (FFmpeg)...")
    final_path = config.OUTPUTS_DIR / f"video_{stamp}.mp4"
    temp_files.append(final_path)
    mixed: Optional[str] = None
    if voice_file or bg_file:
        mixed_path = config.OUTPUTS_DIR / f"video_{stamp}_audio.wav"
        temp_files.append(mixed_path)
        # dlugosc dzwieku = dlugosc obrazu (narracja jest juz w niej zmieszczona - wyjatek:
        # narracja dluzsza niz limit 15 s, wtedy montaz przedluzy ostatnia klatke)
        v_len = ffmpeg_utils.probe_duration(video_track) or float(seconds)
        a_len = (ffmpeg_utils.probe_duration(voice_file) or 0.0) + 0.3 if voice_file else 0.0
        okm, _ = ffmpeg_utils.mix_voice_and_background(voice_file, bg_file, mixed_path,
                                                       max(v_len, a_len))
        mixed = str(mixed_path) if okm else None
        if not okm:
            notes.append("Could not mix the audio; saving a silent video.")
    # bez 1080p obraz jest jeszcze bezstratnym plikiem posrednim - kodujemy go jak wynik SeedVR2
    ok, msg = ffmpeg_utils.combine_video_audio(video_track, mixed, final_path,
                                               compress=not upscaled)
    if not ok:
        notes.append(f"FFmpeg assembly failed ({msg}). Saved the video without final assembly.")
        ffmpeg_utils.tag_video_metadata(video_track)
        final_path = Path(video_track)

    actual_fps = ffmpeg_utils.video_fps(final_path)
    if actual_fps is None or abs(actual_fps - FPS) > 0.001:
        return fail(f"The output does not have the required {FPS} FPS. Check FFmpeg and ffprobe.")

    if progress_cb:
        progress_cb(0.98, "Verifying SFW safety across the video...")
    verdict = qc.check_video(str(final_path), character, quality=bool(character) and qc.enabled())
    if not verdict.safe:
        qc.reject(str(final_path), verdict, f"video: {prompt[:80]}")
        return fail("Output failed the safety check: " + verdict.summary())
    if not verdict.ok:
        notes.append("Quality check: " + verdict.summary())

    # Sprzatanie plikow posrednich - w outputs/ zostaje tylko gotowe wideo
    for tmp in temp_files:
        tmp = Path(tmp)
        if tmp != final_path and tmp.exists():
            tmp.unlink()

    if progress_cb:
        progress_cb(1.0, "Done")

    info = {"mode": "video_i2v" if (image_path or shot_refs[0]) else "video_t2v",
            "engine": "Wan 2.2 A14B" + (" + SeedVR2" if upscaled else ""), "positive": prompt,
            "width": out_w,
            "height": out_h, "seconds": seconds, "fps": track_fps, "steps": plan["steps"],
            "speed": speed,
            "seed": seed, "shots": len(shots), "ref_images": len(ref_images),
            "dialogue": [line.text for line in lines],
            "background": bool(bg_file), "files": [str(final_path)], "qc": verdict.record()}
    if character:
        info["character"] = character["name"]
    history.record(info)
    return VideoOutput(ok=True, files=[str(final_path)], seed=seed, notes=notes)
