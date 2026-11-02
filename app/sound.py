# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Dzwiek tla (ambient i efekty) z opisu sceny - Stable Audio Open 1.0 w ComfyUI.

Model nie "widzi" wideo i byl uczony na krotkich opisach w stylu Freesound ("dog barking,
rain, city traffic"). Dlugi opis obrazu (kto siedzi gdzie, kolory, "Pixar style") dawal
przypadkowe zawodzenie zamiast szczekania, dlatego z promptu sceny wyciagamy tylko to, co
slychac: wprost opisane dzwieki ("sound of rain", "cafe ambience") i zdarzenia ze slownika
(szczekanie, deszcz, kroki, ruch uliczny... po angielsku i po polsku). Negatyw jest krotki
("music, speech") - dlugi, z "noise, hiss", odpychal model od szumu deszczu czy ognia.
Generujemy zawsze pelne okno modelu (47.6 s, patrz workflow) i przycinamy do dlugosci wideo.

Test klasyfikatorem AudioSet (AST, 2 seedy): krotki latent + dlugi opis sceny - szczekanie 0.0,
deszcz 0.0-0.1, fale 0.0; pelne okno + krotki opis - szczekanie 0.6-0.7, deszcz 0.6-0.9,
fale 0.4-0.7, ptaki 0.7-0.8.
Licencja: Stability AI Community License (komercyjnie dla firm < 1 mln USD/rok).
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import soundfile as sf

from . import model_manager, templates
from .comfy_client import ComfyClient

MODEL_ID = "stable-audio-open"
MAX_SECONDS = 47  # dlugosc okna modelu
_STYLE = "high quality realistic sound effects recording"
_NEGATIVE = "music, speech"
_FALLBACK = "quiet natural ambience, subtle room tone"
_MAX_ITEMS = 6

# (wzorzec, opis dla modelu, czy to tlo miejsca). Zdarzenia (False) ida przed tlem (True).
# Wzorce sa waskie celowo: "wiejski" to nie wiatr, "kapibara" to nie kapanie wody,
# "explores" to nie wybuch, "swimsuit" to nie plusk, "depth of field" to nie laka.
_LEXICON: List[Tuple[str, str, bool]] = [
    # zwierzeta
    (r"bark\w*|woof\w*|szczek\w*|ujada\w*", "dog barking", False),
    (r"growl\w*|warcz\w*", "dog growling", False),
    (r"howl\w*|\bwyj[eą]\b|wycie|wyjąc\w*", "wolf howling", False),
    (r"meow\w*|miaucz\w*|\bmiau\b", "cat meowing", False),
    (r"\bpurr\w*|mrucz\w*", "cat purring", False),
    (r"\broar\w*|\bryczy\b|\bryk\w*", "loud animal roar", False),
    (r"rooster\w*|kogut\w*|\bpieje\b", "rooster crowing", False),
    (r"\bmoo(s|ing)?\b|\bcows?\b|\bkrow[aęy]\b|mucz\w*", "cow mooing", False),
    (r"\bneigh(s|ing|ed)?\b|\bhorses?\b|gallop\w*|\bkoń\b|\bkonie?\b|\bkoni\b|\brż[yą]\b|"
     r"galop\w*", "horse hooves and neighing", False),
    (r"quack\w*|\bducks?\b|\bkacz\w*", "duck quacking", False),
    (r"\bsheep\b|bleat\w*|\bowc[aey]\b|\bbecz[yą]\b|beczeni\w*", "sheep bleating", False),
    (r"\boink\w*|\bpigs?\b|piglets?\b|świni\w*|chrząk\w*", "pig oinking", False),
    (r"squeak\w*|\bpisk\w*|piszcz\w*", "squeaking", False),
    (r"chatter\w*|\bclick(s|ing)?\b|\bklika\w*", "clicking and chattering", False),
    (r"\bhiss\w*|\bsyczy\b|\bsyk\w*", "hissing", False),
    (r"\bbees?\b|\bbuzz(es|ing)?\b|pszcz[oe]\w*|bzycz\w*|brzęcz\w*", "bees buzzing", False),
    (r"\bfrogs?\b|croak\w*|\bżab[ayi]\w*|rechocz\w*", "frogs croaking", False),
    (r"crickets?\b|świerszcz\w*|cykad\w*", "crickets chirping", False),
    (r"\bowls?\b|\bhoot\w*|\bsow[ay]\b", "owl hooting", False),
    (r"seagull\w*|\bgulls?\b|\bmew[ya]\b", "seagulls calling", False),
    (r"elephant\w*|\bsłoń\b|\bsłoni\w*", "elephant trumpeting", False),
    (r"\bchirp\w*|\btweet(s|ing)?\b|\bbirds?\b|songbird\w*|\bptak\w*|\bptasz\w*|ćwierk\w*",
     "birds chirping", False),
    # ludzie i przedmioty
    (r"footsteps?\b|\bwalk(s|ing|ed)?\b|\bstroll\w*|\bkroki\b|\bkroków\b|\bidzie\b|spaceruj\w*",
     "footsteps", False),
    (r"\brun(s|ning)?\b|\bsprint\w*|biegn\w*|\bbiega\w*", "running footsteps", False),
    (r"\bknock\w*|\bpuka\w*|\bstuka\w*", "knocking on a door", False),
    (r"\bdoors?\b.{0,20}(open|close|slam)\w*|(otwiera|zamyka|trzaska)\w* drzwi\w*",
     "door opening and closing", False),
    (r"applau\w*|\bclapp\w*|oklask\w*|klaszcz\w*", "applause", False),
    (r"\bcheer(s|ing|ed)?\b|\bwiwat\w*|kibicuj\w*", "crowd cheering", False),
    (r"shatter\w*|glass break\w*|\btłucz\w*|rozbija\w*", "glass shattering", False),
    (r"explosion\w*|\bexplod\w*|wybuch\w*|eksplozj\w*", "explosion", False),
    (r"gunshot\w*|gunfire|\bstrzał[yu]?\b|\bstrzela\w*", "gunshots", False),
    (r"firework\w*|fajerwerk\w*|petard\w*", "fireworks", False),
    (r"\btyping\b|keyboard\w*|klawiatur\w*", "keyboard typing", False),
    (r"\bclocks?\b|\bticking\b|tick-tock|zegar\w*|\btyka\w*", "clock ticking", False),
    (r"phone ring\w*|telefon dzwoni", "phone ringing", False),
    (r"\bbells?\b|\bdzwon(y|ek|ki|u)\b", "bell ringing", False),
    (r"splash\w*|\bswim(s|ming)?\b|\bswam\b|\bdives?\b|\bplusk\w*|\bpływa\w*|nurkuj\w*",
     "water splashing", False),
    (r"sizzl\w*|\bfry(ing)?\b|\bcooking\b|skwiercz\w*|\bsmaży\w*|\bgotuj\w*",
     "sizzling frying pan", False),
    (r"crunch\w*|\bchrup\w*", "crunching", False),
    (r"\bdrip\w*|\bkapi[eą]\b|kapanie", "water dripping", False),
    (r"\bhammer\w*|\bmłot\w*", "hammering", False),
    (r"\bsawing\b|chainsaw\w*|piłuj\w*", "saw cutting wood", False),
    (r"\bswords?\b|\bmiecz\w*", "swords clashing", False),
    (r"vacuum\w*|odkurz\w*", "vacuum cleaner", False),
    (r"\bsirens?\b|ambulance\w*|police car\w*|karetk\w*|radiow[óo]z\w*", "siren", False),
    (r"\bhonk\w*|car horn\w*|klakson\w*|\btrąbi\w*", "car horn honking", False),
    (r"\bengines?\b|\brevv\w*|\bsilnik\w*", "engine revving", False),
    (r"motorcycl\w*|motorbike\w*|motocykl\w*|\bmotor\b", "motorcycle passing", False),
    (r"\btrains?\b|railway|pociąg\w*", "train passing", False),
    (r"airplane\w*|aeroplane\w*|\bjets?\b|\bplanes?\b|samolot\w*", "airplane flying overhead",
     False),
    (r"helicopter\w*|helikopter\w*|śmigłow\w*", "helicopter", False),
    # pogoda i miejsca
    (r"\brain(ing|y|drops?|fall|storm)?\b|\bdeszcz\w*|\bpada\b|\bulew\w*", "rain falling", True),
    (r"thunder\w*|lightning|\bstorm\w*|\bburz[ayę]\b|grzmo\w*|piorun\w*", "thunder rumbling",
     True),
    # "wind blowing" dawalo w testach gwizd jak syrena; "strong wind" - prawdziwy wiatr
    (r"\bwind(s|y)?\b|\bgust\w*|\bwiatr\b|\bwiatru\b|\bwieje\b|\bwiał\w*|wichur\w*", "strong wind",
     True),
    (r"breeze|wietrzyk\w*|bryz\w*", "light breeze rustling leaves", True),
    (r"blizzard|snowstorm|zamie[ćc]\w*|śnieżyc\w*", "strong cold wind", True),
    (r"ocean\w*|\bsea\b|seaside|beach\w*|\bsurf(ing|er|ers)?\b|shore\w*|crashing waves|"
     r"waves crash\w*|\bmorz\w*|\bplaż\w*|\bfal[ae]\b|\bfale\b", "ocean waves crashing", True),
    (r"underwater|pod wodą", "underwater bubbles", True),
    (r"\brivers?\b|\bstreams?\b|\bcreek\w*|\bbrook\w*|waterfall\w*|fountain\w*|\brzek[aiąęo]\b|"
     r"\brzece\b|strumie\w*|wodospad\w*|fontann\w*", "flowing water", True),
    (r"campfire\w*|fireplace\w*|bonfire\w*|\bfire\b|\bflames?\b|ognisk\w*|\bkomin(ek|ku)\b|"
     r"\bogień\b|\bognia\b|płomie\w*", "fire crackling", True),
    (r"\bforests?\b|\bjungle\w*|\bwoods\b|\blas\b|\blesie\b|\blasu\b|dżungl\w*",
     "forest ambience with birds and rustling leaves", True),
    (r"rustl\w*|falling leaves|autumn leaves|dry leaves|\bliści\w*|szeleszcz\w*|szelest\w*",
     "leaves rustling", True),
    (r"\bpark\b|\bgardens?\b|meadow\w*|orchard\w*|(?<!of )\bfields?\b|countryside|ogród\w*|"
     r"ogrodzie|\bsadzie\b|\błąk\w*|\bpolu\b|\bwsi\b", "birds chirping and light breeze", True),
    (r"traffic|\bstreets?\b|\bcars?\b|\btaxi\w*|\bbus(es)?\b|highway|\broad\b|ruch\w* uliczn\w*|"
     r"\bulic\w*|samoch\w*|\bauto\b|\baut[ao]\b|autobus\w*|\bszos\w*", "city street traffic",
     True),
    (r"\bcity\b|downtown|metropol\w*|\bmiast\w*", "busy city ambience", True),
    (r"\bcrowd\w*|\bbusy\b|\bmarkets?\b|bazaar|\btłum\w*|\btargu?\b|\bbazar\w*|\brynek\b|"
     r"\brynku\b", "crowd murmur", True),
    (r"stadium|\barena\b|stadion\w*", "stadium crowd", True),
    (r"\bcaf[eé]\w*|coffee shop|restaurant\w*|\bbar\b|\bpub\b|kawiarn\w*|restauracj\w*",
     "cafe ambience with cups and plates clinking", True),
    (r"kitchen\w*|\bkuchni\w*", "kitchen ambience", True),
    (r"\boffices?\b|\bbiur[ozea]\b|\bbiurze\b", "office ambience", True),
    (r"classroom\w*|\bschool\w*|\bklas(a|ie|y)\b|szkoł\w*|szkole", "quiet classroom room tone",
     True),
    (r"space ?ship\w*|statek kosmiczny|statku kosmicznym", "spaceship engine hum", True),
    (r"\bcaves?\b|\bjaskini\w*", "cave ambience with water drips", True),
    (r"\blakes?\b|\bponds?\b|jezior\w*|\bstaw(ie|u)?\b", "gentle water lapping", True),
    (r"\bboats?\b|\bship\b|\bharbou?r\b|\bport\b|\błód\w*|\błodzi\w*|\bstatek\b|\bporcie\b",
     "boat on water, harbor ambience", True),
    (r"\bmountains?\b|\bgór[ay]?\b|\bgórach\b|\bszczycie\b", "strong wind in the mountains", True),
    (r"\bdeserts?\b|pustyni\w*", "strong wind in the desert", True),
    (r"\bfarms?\b|\bbarn\b|gospodarstw\w*|\bstodo\w*", "farm ambience with animals", True),
    (r"\bairports?\b|lotnisk\w*", "airport terminal ambience", True),
    (r"\bsubway\b|\bmetro\b", "subway train passing", True),
    (r"\bfactory\b|warehouse|fabryk\w*|\bhal[ai] produkcyjn\w*", "factory machinery hum", True),
    (r"\bconstruction\b|\bbudow(a|ie|y)\b", "construction site ambience", True),
    (r"\bhospital\w*|szpital\w*", "hospital ambience with beeping monitor", True),
    (r"\blibrary\b|bibliotek\w*", "quiet library ambience", True),
    (r"\bbedroom\b|living room|\bsypialni\w*|\bsalon\w*|\bpok[oó]j\w*", "quiet indoor room tone",
     True),
    (r"\bnight\b|\bnoc[ąy]?\b", "night ambience", True),
]
_LEXICON_RE = [(re.compile(pat, re.I), tag, amb) for pat, tag, amb in _LEXICON]

# wprost opisany dzwiek: "sound of rain", "quiet cafe ambience", "chattering sound"
# (ale nie "ambient light" ani "film grain noise" - to opis obrazu)
_EXPLICIT = re.compile(r"\b(sounds?|noises?|ambien(?:ce|t)|ambiance)\b(?!\s+(?:light|lighting|"
                       r"glow|occlusion|colou?rs?|grain))", re.I)
# slowa, po ktorych zaczyna sie sam opis dzwieku ("it makes | a weird noise")
_CUT = {"and", "with", "while", "as", "if", "like", "of", "hear", "hears", "heard", "makes",
        "making", "made", "produces", "producing", "emits", "emitting", "there", "is", "are",
        "it", "its", "his", "her", "their", "a", "an", "the"}
_LOUD = re.compile(r"\b(loud\w*|głośn\w*)\b", re.I)
# slowa, ktore nie odrozniaja dzwiekow - przy szukaniu powtorzen ich nie liczymy
_COMMON = {"quiet", "ambience", "ambient", "sound", "sounds", "with", "and", "of", "room", "tone",
           "loud", "light", "gentle", "the", "a", "on", "in"}


def _words(item: str) -> set:
    return set(re.findall(r"\w+", item.lower())) - _COMMON


def available() -> bool:
    return model_manager.is_installed(MODEL_ID)


def _explicit_phrases(scene: str) -> List[str]:
    """Fragmenty zdan, w ktorych autor sam opisal dzwiek (tylko po angielsku)."""
    out = []
    for clause in re.split(r"[,.;:!?]", scene):
        m = _EXPLICIT.search(clause)
        if not m or re.search(r"[ąćęłńóśźż]", clause, re.I):
            continue
        before = clause[:m.start()].split()[-5:]
        cuts = [i for i, w in enumerate(before) if w.lower() in _CUT]
        if cuts:
            before = before[cuts[-1] + 1:]
        after = clause[m.end():].split()
        tail = after[:6] if after[:1] == ["of"] else []
        phrase = " ".join(before + [m.group(1)] + tail).lower()
        if len(phrase.split()) >= 2:
            out.append(phrase)
    return out


def prompt_from_scene(scene: str, custom: str = "") -> str:
    """Krotki opis dzwieku dla Stable Audio: wlasny opis albo dzwieki wyciagniete ze sceny."""
    if custom and custom.strip():
        return f"{custom.strip().rstrip('.')}, {_STYLE}"
    text = scene or ""
    hits = []  # (pozycja w tekscie, opis, czy tlo)
    for rx, tag, amb in _LEXICON_RE:
        m = rx.search(text)
        if m:
            # "barking loudly" -> "loud dog barking" (tylko gdy "glosno" stoi obok zdarzenia)
            if not amb and _LOUD.search(text[max(0, m.start() - 30):m.end() + 30]):
                tag = f"loud {tag}"
            hits.append((m.start(), tag, amb))
    events = [t for _, t, amb in sorted(hits) if not amb]
    ambience = [t for _, t, amb in sorted(hits) if amb]
    items: List[str] = []
    for item in _explicit_phrases(text) + events + ambience:
        if not any(_words(item) & _words(x) for x in items):  # bez powtorzen ("cafe", "birds")
            items.append(item)
    items = items[:_MAX_ITEMS] or [_FALLBACK]
    return f"{', '.join(items)}, {_STYLE}"


def generate(client: ComfyClient, scene: str, custom: str, seconds: float, seed: int,
             out_path: str | Path,
             progress_cb: Optional[Callable[[float, str], None]] = None) -> Tuple[bool, str]:
    """Generuje tlo dzwiekowe o dlugosci wideo. Zwraca (ok, sciezka_lub_komunikat)."""
    if not available():
        return False, ("Background audio skipped: install the Stable Audio Open model "
                       "(Video + Audio package).")
    mapping = {"%CKPT%": model_manager.find_file("checkpoints", "stable-audio-open"),
               "%CLIP%": model_manager.find_file("text_encoders", "t5_base"),
               "%POSITIVE%": prompt_from_scene(scene, custom), "%NEGATIVE%": _NEGATIVE,
               "%SECONDS%": float(min(math.ceil(seconds + 0.5), MAX_SECONDS)),
               "%SEED%": int(seed)}
    try:
        result = client.run_and_wait(templates.build("stable_audio_bg", mapping),
                                     progress_cb=progress_cb)
    except Exception as e:  # noqa: BLE001
        return False, f"Background audio skipped: {e}"
    item = next((it for it in result.files
                 if (it.get("filename") or "").lower().endswith((".flac", ".wav", ".mp3"))), None)
    if item is None:
        return False, "Background audio skipped: the engine returned no audio file."
    Path(out_path).write_bytes(client.download_output(item))
    try:  # model oddaje cale okno 47.6 s - zostawiamy dlugosc wideo (reszta to cisza)
        audio, rate = sf.read(str(out_path))
        sf.write(str(out_path), audio[:int((seconds + 0.5) * rate)], rate)
    except Exception:  # noqa: BLE001 - mikser i tak przytnie do dlugosci wideo
        pass
    return True, str(out_path)
