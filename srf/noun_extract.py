#!/usr/bin/env python3
"""
Shared CLIP noun extraction for SRF autoresearch loops.

Each dataset requires a different extraction strategy because question formats differ:
  - "mmvp"    : fine-grained visual attribute questions (orientation, state, color)
                "Is the butterfly's wings open or closed?" → "butterfly"
  - "vlmbias" : counting questions with explicit counted objects
                "How many logos are on this image?" → "logos"
  - "pope"    : existence questions with a single object
                "Is there a cat in the image?" → "cat"

Usage:
    from noun_extract import extract_clip_noun
    noun = extract_clip_noun(question, mode="mmvp")
"""
from __future__ import annotations

import re

_GENERIC_NOUNS = {
    "thing", "things", "item", "items", "object", "objects",
    "image", "picture", "photo", "answer",
    "there", "taken", "angle", "likely", "blowing",
    # short words that could slip through fallback
    "any", "from", "which", "with", "this", "that", "some",
}


# ---------------------------------------------------------------------------
# MMVP mode
# ---------------------------------------------------------------------------
def _mmvp(question: str) -> str:
    """Extract the visual subject from an MMVP attribute-discrimination question.

    MMVP questions ask about a specific object's visual attribute (orientation,
    state, color, shape). We want the SUBJECT OBJECT so CLIP can localise it.

    Key design decisions:
    - All captures limited to ONE word (no two-word greedy) to avoid "shadow on",
      "hand using", "reflection of" style over-captures.
    - "of the/a X" uses the LAST match (findall) so "of a single ear of corn" → "corn".
    - "Is there a X", "Are there any X" etc. come before "of the X" to avoid
      "of the bicycle" firing on "Is there a reflection of the bicycle".
    """
    q = question.strip().lower().rstrip("?")

    # 1. Possessive — "the butterfly's wings" → "butterfly"
    m = re.search(r"(?:the|this|an?)\s+(\w+(?:\s+\w+)?)'s\b", q)
    if m:
        return m.group(1).strip()

    # 2. "Is there a/an X" — single word only, "shadow" not "shadow on"
    m = re.search(r"\bis there (?:an?\s+)?(\w+)\b", q)
    if m:
        noun = m.group(1).strip()
        if noun not in _GENERIC_NOUNS:
            return noun

    # 3. "Are there any X" — single word
    m = re.search(r"\bare there (?:any\s+)?(\w+)\b", q)
    if m:
        noun = m.group(1).strip()
        if noun not in _GENERIC_NOUNS:
            return noun

    # 4. "Can/Do you see the X" — "see the key" → "key"
    m = re.search(r"\bsee (?:the|a|an) (\w+)\b", q)
    if m:
        noun = m.group(1).strip()
        if noun not in _GENERIC_NOUNS:
            return noun

    # 5. "[V]ing the X" — "wind blowing the flag" → "flag" (last word)
    matches = re.findall(r"\b\w+ing (?:the|a|an) (\w+)\b", q)
    for noun in reversed(matches):
        if noun not in _GENERIC_NOUNS:
            return noun

    # 6. "captures a/an X" — "captures a woman running" → "woman"
    m = re.search(r"\bcaptures (?:an?\s+)?(\w+)\b", q)
    if m:
        noun = m.group(1).strip()
        if noun not in _GENERIC_NOUNS:
            return noun

    # 7a. "of X or" — bare "of" before a noun followed by "or" (e.g., "ear of corn or multiple")
    m = re.search(r"\bof (\w+)\s+or\b", q)
    if m:
        noun = m.group(1).strip()
        if noun not in _GENERIC_NOUNS:
            return noun

    # 7b. "of the/this/a X" — use LAST match so "of a single ear of corn" → "corn"
    matches = re.findall(r"\bof (?:the|this|an?)\s+(\w+)\b", q)
    for noun in reversed(matches):
        if noun not in _GENERIC_NOUNS:
            return noun

    # 8. "Is this [noun] going/facing/pointing/showing/more/look…"
    m = re.search(
        r"\bis this (?:an?\s+)?(\w+)\s+"
        r"(?:going|facing|pointing|showing|more|look)", q
    )
    if m:
        return m.group(1).strip()

    # 9. "Is the [noun] more/going/facing/…" — attribute verb follows (one word)
    m = re.search(
        r"\bis (?:the|this) (\w+)\s+"
        r"(?:more|going|facing|rotated|pointing|sit|stand|open|close|"
        r"rais|low|bend|stretch|look|wear|hold|face|visible|present|tilt|"
        r"blur|dark|light|bright|sharp|lean|curve|straight|mirror|flip|"
        r"invert|reflect|cast|shown|display|align|orient|extend|compress|"
        r"widen|narrow|tall|short|wide|thick|thin|full|empty|wet|dry)", q
    )
    if m:
        return m.group(1).strip()

    # 10. "Are the/these [noun] …" (plural subjects, one word)
    m = re.search(r"\bare (?:the|these) (\w+)\s+", q)
    if m:
        noun = m.group(1).strip()
        if noun not in _GENERIC_NOUNS:
            return noun

    # 11. "is it a/an X" — "is it a salmon fillet" → "salmon"
    m = re.search(r"\bis it (?:an?\s+)?(\w+)\b", q)
    if m:
        noun = m.group(1).strip()
        if noun not in _GENERIC_NOUNS:
            return noun

    # 12. "Is the/this/a [noun]" — simple subject (one word)
    m = re.search(r"\bis (?:the|this|an?) (\w+)\b", q)
    if m:
        noun = m.group(1).strip()
        if noun not in _GENERIC_NOUNS | {"it"}:
            return noun

    # 13. Fallback: first 4+ char word not in generic set
    words = re.findall(r'\b[a-z]{4,}\b', q)
    for w in words:
        if w not in _GENERIC_NOUNS:
            return w
    return "object"


# ---------------------------------------------------------------------------
# VLM Bias mode
# ---------------------------------------------------------------------------

_VLMBIAS_GENERIC = {
    "thing", "things", "item", "items", "object", "objects",
    "image", "picture", "photo",
}

def _vlmbias(question: str) -> str:
    """Extract the most CLIP-queryable noun from a VLM Bias counting question.

    Priority: specific counted object > scene container > fallback.
    "How many logos are on this image?" → "logos" (not "image").
    """
    q = question.split("Answer")[0].strip().lower()

    # 1. Explicit count target — "how many X are/is/…"
    m = re.search(r'how many (\w+(?:\s+\w+)?) (?:are|is|have|does)', q)
    if m:
        noun = m.group(1).strip()
        if noun not in _VLMBIAS_GENERIC:
            return noun

    # 2. "count the X pieces/on/in"
    m = re.search(r'count the (\w+(?:\s+\w+)?) (?:pieces|on|in)', q)
    if m:
        return m.group(1).strip()

    # 3. Scene/container (less specific)
    m = re.search(r'(?:on|in) this (\w+(?:\s+\w+)?)', q)
    if m:
        noun = m.group(1).strip()
        if noun not in _VLMBIAS_GENERIC:
            return noun

    m = re.search(r'(?:on|in) the (\w+)', q)
    if m:
        return m.group(1).strip()

    words = re.findall(r'\b[a-z]{4,}\b', q)
    return words[0] if words else "object"


# ---------------------------------------------------------------------------
# POPE mode
# ---------------------------------------------------------------------------
def _pope(question: str) -> str:
    """Extract the queried object from a POPE existence question.

    "Is there a cat in the image?" → "cat"
    "Is there a tennis racket in the image?" → "tennis racket"
    """
    q = question.strip().lower().rstrip("?")

    # "Is there a/an X in/on/at the image"
    m = re.search(r"\bis there (?:an? )?(.+?)\s+(?:in|on|at|visible)", q)
    if m:
        noun = m.group(1).strip()
        if noun not in _GENERIC_NOUNS:
            return noun

    # "Is there a/an X?" (no location phrase)
    m = re.search(r"\bis there (?:an? )?(\w+(?:\s+\w+)?)\b", q)
    if m:
        noun = m.group(1).strip()
        if noun not in _GENERIC_NOUNS:
            return noun

    # Fallback: first 4+ char word not in generic set
    words = re.findall(r'\b[a-z]{4,}\b', q)
    for w in words:
        if w not in _GENERIC_NOUNS:
            return w
    return "object"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _singularize(noun: str) -> str:
    """Convert plural to singular for CLIP query.

    Simple heuristic rules:
    - s → es (bus → buses): keep as is
    - ies → y (cities → city)
    - ses → s (classes → class)
    - ves → f (knives → knife)
    - Default: remove trailing 's'
    """
    noun = noun.strip()
    if not noun.endswith('s'):
        return noun

    # Special cases
    if noun.endswith('ies'):
        return noun[:-3] + 'y'
    elif noun.endswith('ses'):
        return noun[:-2]
    elif noun.endswith('ves'):
        return noun[:-3] + 'f'
    elif noun.endswith('ss'):
        return noun  # "glass" stays "glass"
    else:
        # Default: remove trailing 's'
        return noun[:-1]


def extract_clip_noun(question: str, mode: str = "mmvp", singular: bool = False) -> str:
    """Return the best CLIP query noun for a VLM benchmark question.

    Args:
        question: Raw question string (any casing).
        mode:     Dataset mode — "mmvp", "vlmbias", or "pope".
        singular: If True, convert plural nouns to singular for CLIP.

    Returns:
        A short noun string suitable for CLIP text query.
    """
    if mode == "mmvp":
        result = _mmvp(question)
    elif mode == "vlmbias":
        result = _vlmbias(question)
    elif mode == "pope":
        result = _pope(question)
    else:
        raise ValueError(f"Unknown mode {mode!r}. Use 'mmvp', 'vlmbias', or 'pope'.")

    if singular:
        result = _singularize(result)
    return result
