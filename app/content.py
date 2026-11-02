# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""SFW prompt preflight rules and legacy content-level compatibility.

Checks cover English, Polish, and selected explicit terms in Spanish, French, German,
Italian, Portuguese, Russian and Ukrainian, including look-alike letters, digit
substitutions and letters split by spaces or dots. This lexical guard is not comprehensive
multilingual semantic moderation. Generated media also needs mandatory visual validation
in app.qc.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Optional

REFUSAL = ("⛔ Contsy Studio creates SFW content only. Remove nudity and adult "
           "sexual content from the description.")
# Poziom zapisany w zadaniach i historii sprzed usuniecia trybu 18+ - takie wyniki nie trafiaja
# do Galerii ani do paczek
LEGACY_ADULT = "nsfw"
# LoRA postaci uczona tez na aktach rozbierala postac (testy 30.10, Virtual Mia, bikini na
# plazy): LoRA 1.0 - 2 z 3 nago; LoRA 0.7 - 6 z 6 ubrana, twarz nadal ta sama. CFG 2.5
# z negatywem tez trzymal stroj, ale obraz liczyl sie 2x dluzej i bywal z przebarwieniami.
LORA_STRENGTH = 0.7

# "nude" bywa kolorem (nude lipstick, w kolorze nude) - to nie nagosc
_NUDE = (r"(?<!colored )(?<!coloured )(?<!color )(?<!colour )(?<!kolorze )"
         r"(?<!labial )(?<!rossetto )(?<!batom )(?<!couleur )(?<!colore )(?<!cor )"
         r"(?<!farbe )(?<!à lèvres )(?<!vestido )(?<!robe )(?<!kleid )(?<!vestito )nude"
         r"(?![- ](?:lip\w*|make-?up|heels?|shoes?|pumps|sandals|nails?|polish|colou?r\w*|"
         r"tones?|beige|palette|shade\w*|dress\w*|tops?|bra|lingerie|bodysuit|leggings|"
         r"stockings|tights|gloss|lippenstift|labial|rossetto|batom))")
# Nagosc i tresci dla doroslych po polsku i po angielsku. Bikini, bielizna, "seksowna" - SFW.
_NUDITY = re.compile(
    rf"\b(?:{_NUDE}|nudes|n[4a]ked(?![ -]+eye\b)|nudity|topless|bottomless|undressed|unclothed|"
    r"no\s+cloth(?:es|ing)|without\s+(?:any\s+)?cloth(?:es|ing)|wearing\s+nothing|"
    r"bare (?:breasts?|butt|buttocks)|nipples?|areolas?|genital\w*|pussy|vagina|penis|"
    r"sex(?![yi])\w*|erotic\w*|porn\w*|nsfw|"
    r"nago|nag(?:a|ą|ie|i|ich|iej|im|imi|ość|ości|ością)|nagusieńk\w*|golas\w*|toples|"
    r"bez\s+(?:stanika|biustonosza|ubrania|ubrań|majtek|bielizny|niczego|"
    r"stroju\s+k[aą]pielowego|ciuch[oó]w|odzie[zż]y)|rozebran\w*|"
    r"rozbier\w*|(?:gołe|nagie|odsłonięte)\s+piersi|sutk\w*|akty?|aktów|erotyczn\w*|"
    r"seks(?!own|apil)\w*|dla\s+dorosłych|"
    r"undress\w*|striptease|strip(?:s|ped|ping)?\s+(?:naked|nude)|"
    r"(?:tak(?:e|es|ing)|took|strip(?:s|ped|ping)?|pull(?:s|ed|ing)?)\s+off\s+"
    r"(?:all\s+(?:of\s+)?)?(?:her|his|their|the|my)?\s*"
    r"(?:clothes|clothing|bra|underwear|panties|bikini|swimsuit)|"
    r"remov(?:e|es|ing|ed)\s+(?:all\s+(?:of\s+)?)?(?:her|his|their|the|my)\s+"
    r"(?:clothes|clothing|bra|underwear|panties|bikini|swimsuit)|"
    r"see[- ]?through\s+(?:lingerie|bra|underwear|panties|nightgown|nightie|bodysuit))\b"
    r"|(?<!\d)18 ?\+", re.I)
# Selected explicit terms; avoid ambiguous standalone words such as Spanish "nudo"
# (a knot), French "nu" (also a musical/scientific term), and Portuguese "pelada".
_MULTILINGUAL = re.compile(
    r"\b(?:desnud[oa]s?|desnudez|sin\s+(?:ropa|vestimenta|sujetador)|"
    r"sexo|sexuales?|pornograf[ií]a|pornogr[aá]fic[oa]s?|"
    r"nudit[eé]|sans\s+v[eê]tements?|seins\s+nus|rapports?\s+sexuels?|sexuel(?:le)?s?|"
    r"(?:femme|homme|personne|personnage)\s+nu(?:e|s|es)?|"
    r"(?:totalement|compl[eè]tement|enti[eè]rement|tout(?:e)?)\s+nu(?:e|s|es)?|"
    r"nacktheit|nackt(?:e|en|er|es)?|ohne\s+kleidung|geschlechtsverkehr|"
    r"sexuell(?:e|en|er|es)?|"
    r"nudit[aà]|senza\s+(?:vestiti|indumenti|reggiseno)|sesso|sessual[ei]|"
    r"(?:donna|uomo|persona|completamente)\s+nud[oaie]|"
    r"nudez|sem\s+(?:roupa|roupas|suti[aã])|sexua(?:l|is)|"
    r"(?:mulher|homem|pessoa|completamente|est[aá])\s+nu(?:a|as|s)?|"
    r"(?:mulher|modelo|menina|garota|mo[cç]a|homem|rapaz)\s+pelad[ao]s?|"
    r"oben\s+ohne|unbekleidet\w*|entbl(?:ö|oe|o)(?:ß|ss)t\w*|"
    # Russian/Ukrainian. "гол-" also means bare (голые деревья) or a goal (после гола), so it
    # counts only next to a person; "сексуальн-" (sexy) stays allowed like "sexy".
    r"гол(?:ая|ый|ые|ую|ой|ого|ым|ыми|ою|ий|і|а)\s+(?:женщин|девушк|девочк|мужчин|парн|"
    r"модел|жінк|дівчин|чоловік|хлопц|люд|тел)\w*|"
    r"(?:женщин|девушк|модел|жінк|дівчин)\w*\s+(?:полностью\s+|повністю\s+)?гол(?:ая|ые|а|і)|"
    r"обнаж[её]нн\w*|наг(?:ая|ой|ие|ую|ишом|ота|оту)|топлесс?|"
    r"без\s+(?:одежды|одягу|белья|білизни|лифчика|бюстгальтера)|"
    r"секс(?!уальн|и\b)\w*|порн\w*|эроти\w*|еротик\w*)\b", re.I)

# Look-alike letters (Cyrillic, Greek, Armenian) and digit/symbol stand-ins used to slip a
# word past the guard ("nսde", "nud3", "p0rn"). Applied to a lower-cased comparison copy.
_LOOKALIKE = str.maketrans({
    "а": "a", "в": "b", "е": "e", "к": "k", "м": "m", "н": "h", "о": "o", "р": "p", "с": "c",
    "т": "t", "у": "y", "х": "x", "і": "i", "ј": "j", "ѕ": "s", "ԁ": "d", "α": "a", "ε": "e",
    "ι": "i", "κ": "k", "ν": "v", "ο": "o", "ρ": "p", "τ": "t", "υ": "u", "ս": "u", "օ": "o",
    "ո": "n", "հ": "h", "ց": "g", "0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t",
    "@": "a", "$": "s"})
# Three or more single letters split by spaces or punctuation: "n u d e", "s.e.x"
_SPACED = re.compile(r"(?<!\w)(?:\w[ .\-_*·]+){2,}\w(?!\w)")
_SEPARATORS = re.compile(r"[ .\-_*·]+")


# "without exposing nipples ...", "not revealing too much ..." - przywoluje to, czego ma nie
# byc. Wycinamy do konca czlonu zdania. Tylko przy ciele - "without showing her face" zostaje.
_HIDING = re.compile(
    r"\s*\b(?:without|not|never)\s+(?:fully\s+|completely\s+|ever\s+)?(?:expos|reveal|show)\w*"
    r"\s+(?:any\s+|her\s+|his\s+|their\s+|the\s+)?(?:nipples?|areolas?|genital\w*|private|"
    r"intimate|breasts?|chest|nudity|too\s+much|anything|explicit)[^,.;]*", re.I)

# Regula dla modelu tekstowego (dopisywana do instrukcji systemowej)
RULE = (" Content level: SFW. Every person is clothed in an outfit that fits the scene "
        "(swimwear or lingerie only when the idea asks for it). No nudity and no sexual "
        "content. Never mention nipples, genitals or other intimate body parts, not even to "
        "say they are hidden or covered.")


def normalize_for_check(text: str) -> str:
    """Normalize comparison text without changing the prompt sent or saved."""
    normalized = unicodedata.normalize("NFKC", text or "")
    return "".join(char for char in normalized if unicodedata.category(char) != "Cf")


def _variants(text: str):
    """The normalized text, then a copy with look-alikes mapped and split letters joined."""
    normalized = normalize_for_check(text)
    yield normalized
    latin = normalized.lower().translate(_LOOKALIKE)
    yield _SPACED.sub(lambda m: _SEPARATORS.sub("", m.group()), latin)


def asks_nudity(*texts: str) -> bool:
    """Detect explicit terms in the supported lexical guard."""
    for text in texts:
        if text:
            for variant in _variants(text):
                if _NUDITY.search(variant) or _MULTILINGUAL.search(variant):
                    return True
    return False


def check(*texts: str) -> str:
    """Pusty napis - tekst SFW; inaczej komunikat odmowy dla uzytkownika."""
    return REFUSAL if asks_nudity(*texts) else ""


def legacy_adult(entry: Optional[dict]) -> bool:
    """Zadanie albo wpis historii oznaczony jeszcze jako 18+ (sprzed usuniecia tego trybu)."""
    return isinstance(entry, dict) and isinstance(entry.get("level"), str) and \
        normalize_for_check(entry["level"]).strip().casefold() == LEGACY_ADULT


def strip_hiding(text: str) -> str:
    out = _HIDING.sub("", text or "")
    out = re.sub(r"\s+([,.;])", r"\1", out)
    out = re.sub(r"([,;])\s*([,.;])", r"\2", out)
    return re.sub(r"\s{2,}", " ", out).strip()


def enforce(prompt: str) -> Optional[str]:
    """Prompt od modelu tekstowego bez zwrotow "without exposing..." albo None, gdy model
    dopisal nagosc (do odrzucenia)."""
    text = strip_hiding(prompt)
    return None if asks_nudity(text) else text
