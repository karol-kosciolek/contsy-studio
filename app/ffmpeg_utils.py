# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Narzedzia FFmpeg: laczenie wideo z audio, miks sciezek, metadane, dlugosc.

FFmpeg musi byc w PATH (patrz README). Kazda funkcja zwraca (ok, komunikat),
a bledy nie wywracaja pipeline'u - montaz audio jest etapem opcjonalnym.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import List, Optional, Tuple

AI_COMMENT = "AI-generated; Contsy Studio"


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _run(args: List[str], cwd: Optional[str] = None) -> Tuple[bool, str]:
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=600, cwd=cwd)
    except Exception as e:  # noqa: BLE001
        return False, str(e)
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout or "unknown FFmpeg error")[-600:]
    return True, ""


def probe_duration(path: str | Path) -> Optional[float]:
    """Czas trwania pliku w sekundach (przez ffprobe) lub None."""
    if shutil.which("ffprobe") is None:
        return None
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        data = json.loads(out.stdout or "{}")
        return float(data.get("format", {}).get("duration", 0)) or None
    except Exception:
        return None


def video_size(path: str | Path) -> Optional[Tuple[int, int]]:
    """(szerokosc, wysokosc) pierwszej sciezki wideo lub None."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "v:0", "-show_entries",
             "stream=width,height", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        stream = json.loads(out.stdout or "{}")["streams"][0]
        return int(stream["width"]), int(stream["height"])
    except Exception:
        return None


def video_fps(path: str | Path) -> Optional[float]:
    """Rzeczywisty sredni FPS pierwszej sciezki wideo (ffprobe)."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "v:0", "-show_entries",
             "stream=avg_frame_rate", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        stream = json.loads(out.stdout or "{}")["streams"][0]
        return float(Fraction(stream["avg_frame_rate"])) or None
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError,
            ZeroDivisionError):
        return None


def convert_fps(video: str | Path, out_path: str | Path, fps: int,
                duration: Optional[float] = None) -> Tuple[bool, str]:
    """Przelicza klatki wg czasu, zachowujac dlugosc z dokladnoscia do 1 klatki.

    Wejscie jest juz interpolowane RIFE. Filtr fps wybiera klatki po znacznikach
    czasu; nie zmieniamy szybkosci odtwarzania przez nadpisanie frame_rate.
    RIFE zapisuje ostatni punkt czasowy, bez jego pelnego interwalu: dopelnienie
    konca zachowuje czas oryginalnego klipu i synchronizacje audio. Plik posredni
    jest kodowany bezstratnie, przed docelowym powiekszaniem i kompresja.
    """
    duration = duration if duration is not None else probe_duration(video)
    if not duration or duration <= 0 or fps <= 0:
        return False, "Could not determine video duration for FPS conversion."
    frames = max(1, int(duration * fps + 0.5))
    filters = (f"fps={fps},tpad=stop_mode=clone:stop_duration=1,"
               f"trim=end_frame={frames},setpts=PTS-STARTPTS")
    ok, msg = _run(["ffmpeg", "-y", "-i", str(video), "-vf", filters,
                    "-an", "-c:v", "libx264", "-crf", "0", "-preset", "fast",
                    "-pix_fmt", "yuv420p", str(out_path)])
    if not ok:
        return False, msg
    actual_fps = video_fps(out_path)
    actual_duration = probe_duration(out_path)
    if actual_fps is None or abs(actual_fps - fps) > 0.001:
        return False, f"The converted video does not have {fps} FPS."
    if actual_duration is None or abs(actual_duration - frames / fps) > 0.01:
        return False, "FPS conversion changed the clip duration."
    return True, str(out_path)


def mix_audio(tracks: List[str], out_path: str | Path) -> Tuple[bool, str]:
    """Miksuje kilka sciezek audio w jedna (amix). Jedna sciezka = kopia."""
    tracks = [t for t in tracks if t and Path(t).exists()]
    if not tracks:
        return False, "No audio tracks are available to mix."
    if not has_ffmpeg():
        return False, "FFmpeg was not found in PATH."
    if len(tracks) == 1:
        ok, msg = _run(["ffmpeg", "-y", "-i", tracks[0], "-c", "copy", str(out_path)])
        if not ok:  # kopia moze nie przejsc przy roznych formatach - przekoduj
            ok, msg = _run(["ffmpeg", "-y", "-i", tracks[0], str(out_path)])
        return ok, msg or str(out_path)
    args = ["ffmpeg", "-y"]
    for t in tracks:
        args += ["-i", t]
    args += ["-filter_complex",
             f"amix=inputs={len(tracks)}:duration=longest:normalize=0",
             str(out_path)]
    ok, msg = _run(args)
    return ok, msg or str(out_path)


def last_frame(video: str | Path, out_png: str | Path) -> Tuple[bool, str]:
    """Zapisuje ostatnia klatke wideo jako PNG (pierwsza klatka kolejnego ujecia)."""
    ok, msg = _run(["ffmpeg", "-y", "-sseof", "-1", "-i", str(video), "-update", "1",
                    str(out_png)])
    return ok and Path(out_png).exists(), msg or str(out_png)


def frame_before_end(video: str | Path, out_png: str | Path,
                     seconds: float) -> Tuple[bool, str]:
    """Zapisuje klatke `seconds` przed koncem wideo (start kolejnej czesci z zakladka)."""
    ok, msg = _run(["ffmpeg", "-y", "-sseof", f"-{seconds:.3f}", "-i", str(video),
                    "-frames:v", "1", str(out_png)])
    return ok and Path(out_png).exists(), msg or str(out_png)


def join_segments(parts: List[str], out_path: str | Path, fps: int,
                  overlaps: Optional[List[float]] = None,
                  duration: Optional[float] = None) -> Tuple[bool, str]:
    """Skleja ujecia w jedno wideo o podanym FPS (H.264, CRF 17).

    Kazde kolejne ujecie zaczyna sie od ostatniej klatki poprzedniego - ta powtorzona
    klatka jest pomijana, wiec ruch przechodzi plynnie. overlaps[i] > 0: czesc i zaczyna
    sie tyle sekund przed koncem poprzedniej (poprawiona twarz postaci) - ten odcinek
    przenika sie, zamiast skoku twarzy w jednej klatce.
    """
    if not has_ffmpeg():
        return False, "FFmpeg was not found in PATH."
    overlaps = list(overlaps or [])
    durations = [probe_duration(p) for p in parts]
    if any(overlaps[1:]) and all(durations):
        return _join_overlapped(parts, durations, overlaps, out_path, fps, duration)
    args = ["ffmpeg", "-y"]
    chains, labels = [], ""
    for i, part in enumerate(parts):
        args += ["-i", str(part)]
        trim = "trim=start_frame=1,setpts=PTS-STARTPTS" if i else "null"
        chains.append(f"[{i}:v]{trim}[v{i}]")
        labels += f"[v{i}]"
    graph = ";".join(chains) + f";{labels}concat=n={len(parts)}:v=1:a=0,fps={fps}[out]"
    length_args = ["-t", str(duration)] if duration is not None else []
    ok, msg = _run([*args, "-filter_complex", graph, "-map", "[out]", *length_args, "-c:v", "libx264",
                    "-crf", "17", "-preset", "slow", "-pix_fmt", "yuv420p", str(out_path)])
    return ok, msg or str(out_path)


def _join_overlapped(parts: List[str], durations: List[float], overlaps: List[float],
                     out_path: str | Path, fps: int,
                     duration: Optional[float] = None) -> Tuple[bool, str]:
    """join_segments z przenikaniem (xfade) na zakladkach i zwyklym sklejeniem w reszcie."""
    args, chains = ["ffmpeg", "-y"], []
    for i, part in enumerate(parts):
        args += ["-i", str(part)]
        over = overlaps[i] if i < len(overlaps) else 0.0
        trim = "trim=start_frame=1," if i and not over else ""
        chains.append(f"[{i}:v]{trim}setpts=PTS-STARTPTS,fps={fps},settb=AVTB,setsar=1,"
                      f"format=yuv420p[v{i}]")
    cur, length = "v0", durations[0]
    for i in range(1, len(parts)):
        over = overlaps[i] if i < len(overlaps) else 0.0
        if over:
            # przenikanie tylko przez pierwsza polowe zakladki - tam obie wersje ruchu sa
            # jeszcze prawie takie same (na calej zakladce kontury lekko sie dublowaly)
            chains.append(f"[{cur}][v{i}]xfade=transition=fade:duration={over / 2:.3f}:"
                          f"offset={max(length - over, 0):.3f}[x{i}]")
            length += durations[i] - over
        else:
            chains.append(f"[{cur}][v{i}]concat=n=2:v=1:a=0[x{i}]")
            length += durations[i]
        cur = f"x{i}"
    length_args = ["-t", str(duration)] if duration is not None else []
    ok, msg = _run([*args, "-filter_complex", ";".join(chains), "-map", f"[{cur}]", *length_args,
                    "-c:v", "libx264", "-crf", "17", "-preset", "slow", "-pix_fmt", "yuv420p",
                    str(out_path)])
    return ok, msg or str(out_path)


def overlay_layers(base: str | Path, layers: List[Tuple[str, int, int, int, int]],
                   out_path: str | Path, workdir: str | Path) -> Tuple[bool, str]:
    """Wkleja wycinki (sciezka, x, y, szer, wys) na wideo bazowe z miekkimi krawedziami.

    Uzywane przy ruchu ust wielu postaci: kazda twarz przetworzona osobno wraca w swoje
    miejsce, a rozmyta maska ukrywa szew miedzy wycinkiem a reszta kadru.
    """
    from PIL import Image, ImageDraw, ImageFilter

    if not has_ffmpeg():
        return False, "FFmpeg was not found in PATH."
    length = probe_duration(base) or 0
    args = ["ffmpeg", "-y", "-i", str(base)]
    chains, prev = [], "0:v"
    for k, (path, x, y, w, h) in enumerate(layers):
        feather = max(8, int(min(w, h) * 0.12))
        mask = Image.new("L", (w, h), 0)
        ImageDraw.Draw(mask).rectangle([feather, feather, w - feather, h - feather], fill=255)
        mask_path = Path(workdir) / f"mask{k}.png"
        mask.filter(ImageFilter.GaussianBlur(feather / 2)).save(mask_path)
        args += ["-i", str(path), "-loop", "1", "-i", str(mask_path)]
        a, m = 1 + 2 * k, 2 + 2 * k
        chains.append(f"[{a}:v]format=rgba[l{k}];[{m}:v]format=gray[m{k}];"
                      f"[l{k}][m{k}]alphamerge[a{k}];"
                      f"[{prev}][a{k}]overlay={x}:{y}:eof_action=pass[t{k}]")
        prev = f"t{k}"
    graph = ";".join(chains) + f";[{prev}]format=yuv420p[out]"
    ok, msg = _run([*args, "-filter_complex", graph, "-map", "[out]", "-t", f"{length:.3f}",
                    "-c:v", "libx264", "-crf", "16", "-an", str(out_path)])
    return ok, msg or str(out_path)


def pad_audio(src: str | Path, out_path: str | Path, seconds: float) -> Tuple[bool, str]:
    """Dopelnia audio cisza do podanej dlugosci (dluzsze audio zostaje bez zmian)."""
    if not has_ffmpeg():
        return False, "FFmpeg was not found in PATH."
    ok, msg = _run(["ffmpeg", "-y", "-i", str(src), "-af", f"apad=whole_dur={seconds:.2f}",
                    str(out_path)])
    return ok, msg or str(out_path)


def mix_voice_and_background(voice: Optional[str], background: Optional[str],
                             out_path: str | Path, duration: float) -> Tuple[bool, str]:
    """Miks narracji z tlem dzwiekowym do WAV 48 kHz stereo o dlugosci `duration` s.

    Tlo jest ciszej, lagodnie wchodzi i schodzi, i samo przycisza sie, gdy mowi lektor
    (sidechain). Glosnosc ustawiamy w dwoch przebiegach (pomiar, potem stale wzmocnienie),
    bo jednoprzebiegowy loudnorm zaczyna za glosno: -16 LUFS z lektorem (standard
    serwisow internetowych), takze dla samego tla - przy -20 LUFS efekty byly za ciche.
    """
    if not has_ffmpeg():
        return False, "FFmpeg was not found in PATH."
    fmt = "aresample=48000,aformat=channel_layouts=stereo"
    fade = f"afade=t=in:d=0.5,afade=t=out:st={max(duration - 0.6, 0):.2f}:d=0.6"
    fit = f"apad,atrim=0:{duration:.3f}"
    # Stable Audio dokleja niesłyszalne dudnienie < 40 Hz (w ogniu i deszczu nawet 50-70%
    # energii) - zjada zapas glosnosci i przycisza to, co slychac; wycinamy je z tla
    low_cut = "highpass=f=40:poles=2"
    if voice and background:
        inputs = ["-i", str(voice), "-i", str(background)]
        # glos dopelniony cisza do dlugosci filmu: sidechaincompress konczy sie z krotszym
        # wejsciem, wiec po ostatnim slowie tlo znikalo i zostawala cisza
        graph = (f"[0:a]{fmt},{fit},asplit=2[v1][v2];"
                 f"[1:a]{fmt},{low_cut},{fit},{fade},volume=0.8[bg];"
                 "[bg][v1]sidechaincompress=threshold=0.05:ratio=3:attack=50:release=400[duck];"
                 f"[duck][v2]amix=inputs=2:duration=first:normalize=0,{fit}[out]")
    elif voice:
        inputs = ["-i", str(voice)]
        graph = f"[0:a]{fmt},{fit}[out]"
    elif background:
        inputs = ["-i", str(background)]
        graph = f"[0:a]{fmt},{low_cut},{fit},{fade}[out]"
    else:
        return False, "No audio tracks are available."
    premix = Path(out_path).with_name(Path(out_path).stem + "_premix.wav")
    ok, msg = _run(["ffmpeg", "-y", *inputs, "-filter_complex", graph, "-map", "[out]",
                    "-c:a", "pcm_s16le", str(premix)])
    if not ok:
        return ok, msg
    target = "I=-16:TP=-1.5:LRA=11"
    try:
        probe = subprocess.run(["ffmpeg", "-i", str(premix), "-af",
                                f"loudnorm={target}:print_format=json", "-f", "null", "-"],
                               capture_output=True, text=True, timeout=120).stderr
        m = json.loads(probe[probe.rindex("{"):probe.rindex("}") + 1])
        norm = (f"loudnorm={target}:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
                f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:"
                f"offset={m['target_offset']}:linear=true")
    except (ValueError, KeyError, subprocess.SubprocessError):
        norm = f"loudnorm={target}"
    ok, msg = _run(["ffmpeg", "-y", "-i", str(premix), "-af", norm, "-ar", "48000",
                    "-c:a", "pcm_s16le", str(out_path)])
    premix.unlink(missing_ok=True)
    return ok, msg or str(out_path)


def combine_video_audio(video: str | Path, audio: Optional[str | Path],
                        out_path: str | Path,
                        tag_ai: bool = True, compress: bool = False) -> Tuple[bool, str]:
    """Laczy wideo z audio (lub samo wideo) i dodaje metadane 'AI-generated'.

    Gdy audio (np. narracja) jest dluzsze od wideo, ostatnia klatka jest przedluzana,
    zeby wypowiedz nie zostala ucieta. W przeciwnym razie wideo jest kopiowane bez
    rekompresji w calej dlugosci, a krotsze audio dopelniane cisza.
    compress - obraz zawsze kodowany od nowa: bezstratny plik posredni (convert_fps) ma
    profil H.264 High 4:4:4, ktorego nie odtworzy np. telefon ani odtwarzacz Windows.
    """
    if not has_ffmpeg():
        return False, "FFmpeg was not found in PATH; saving video without audio assembly."
    meta = ["-metadata", f"comment={AI_COMMENT}"] if tag_ai else []
    reencode = ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p"]
    if not (audio and Path(str(audio)).exists()):
        video_opts = reencode if compress else ["-c", "copy"]
        ok, msg = _run(["ffmpeg", "-y", "-i", str(video), *video_opts, *meta, str(out_path)])
        return ok, msg or str(out_path)

    inputs = ["ffmpeg", "-y", "-i", str(video), "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0"]
    a_len, v_len = probe_duration(audio) or 0.0, probe_duration(video) or 0.0
    if a_len - v_len > 0.05:
        total = a_len + 0.3  # cala wypowiedz + krotki oddech na koncu
        video_opts = ["-vf", f"tpad=stop_mode=clone:stop_duration={total - v_len:.2f}", *reencode]
        length_opts = ["-af", "apad", "-t", f"{total:.2f}"]
    else:  # samo "-shortest" ucieloby wideo do dlugosci krotszej narracji
        video_opts = reencode if compress else ["-c:v", "copy"]
        length_opts = ["-af", "apad", "-t", f"{v_len:.3f}"] if v_len else ["-shortest"]
    ok, msg = _run([*inputs, *video_opts, "-c:a", "aac", "-b:a", "192k", *length_opts, *meta,
                    str(out_path)])
    if not ok and video_opts[-1] == "copy":  # kopia moze nie przejsc - przekoduj wideo
        ok, msg = _run([*inputs, *reencode, "-c:a", "aac", *length_opts, *meta, str(out_path)])
    return ok, msg or str(out_path)


def tag_video_metadata(path: str | Path) -> Tuple[bool, str]:
    """Dodaje metadane 'AI-generated' do istniejacego pliku wideo (remux)."""
    if not has_ffmpeg():
        return False, "FFmpeg is not installed."
    p = Path(path)
    tmp = p.with_name(p.stem + "_tagged" + p.suffix)
    ok, msg = _run(["ffmpeg", "-y", "-i", str(p), "-c", "copy",
                    "-metadata", f"comment={AI_COMMENT}", str(tmp)])
    if ok:
        tmp.replace(p)
    return ok, msg or str(p)
