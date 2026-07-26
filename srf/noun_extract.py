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
# VLind-Bench mode
# ---------------------------------------------------------------------------

def _vlind(question: str) -> str:
    """Extract noun from VLind-Bench question.

    'What color is the banana?' → 'banana'
    'What is the dog doing?' → 'dog'
    'How many cars are there?' → 'cars'
    """
    q = question.strip().lower().rstrip("?")

    # "What is the X doing/wearing/etc" → X (handle before generic "the X")
    m = re.search(r'\bis the (\w+(?:\s+\w+)?)\s+(?:doing|wearing|holding|carrying)\b', q)
    if m:
        return m.group(1).strip()

    # "What [attr] is the X" → X
    m = re.search(r'\bthe\s+(\w+(?:\s+\w+)?)\b', q)
    if m:
        noun = m.group(1).strip()
        if noun not in _GENERIC_NOUNS:
            return noun

    # "How many X are/is/…" → X
    m = re.search(r'how many (\w+(?:\s+\w+)?) (?:are|is|have|does)', q)
    if m:
        noun = m.group(1).strip()
        if noun not in _GENERIC_NOUNS:
            return noun

    # Fallback: first 4+ char content word
    words = re.findall(r'\b[a-z]{4,}\b', q)
    for w in words:
        if w not in _GENERIC_NOUNS:
            return w
    return "object"


# ---------------------------------------------------------------------------
# WhatsUp mode
# ---------------------------------------------------------------------------

def _whatsup(caption: str) -> tuple:
    """Extract object nouns from a WhatsUp caption option.

    'A photo of a dining table on the bottom' → ('dining table', None)
    'A photo of a dining table to the left of a refrigerator' → ('dining table', 'refrigerator')
    'A beer bottle on a armchair' → ('beer bottle', 'armchair')

    Returns (noun_A, noun_B) where noun_B is None for single-object captions.
    """
    c = caption.strip().lower()
    # Remove "A photo of" prefix if present
    c = re.sub(r'^a photo of\s+', '', c)
    # Remove leading article
    c = re.sub(r'^an?\s+', '', c)

    # Two-object: "X [spatial_rel] [a/an] Y"
    spatial_rels = r'(?:to the left of|to the right of|in front of|behind|above|below|on top of|next to|near|beside)'
    m = re.search(rf'^(.+?)\s+{spatial_rels}\s+(?:an?\s+)?(.+?)(?:\s*$)', c)
    if m:
        return m.group(1).strip(), m.group(2).strip()

    # One-object: "X [spatial_rel]" (position descriptor at end)
    m = re.search(r'^(.+?)\s+(?:on|under|above|below|to the|in|at)\b', c)
    if m:
        return m.group(1).strip(), None

    return c.strip(), None


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


def extract_clip_noun(question: str, mode: str = "mmvp", singular: bool = False):
    """Return the best CLIP query noun for a VLM benchmark question.

    Args:
        question: Raw question string (any casing).
        mode:     Dataset mode — "mmvp", "vlmbias", "pope", "vlind", or "whatsup".
        singular: If True, convert plural nouns to singular for CLIP.

    Returns:
        A short noun string (or tuple for whatsup) suitable for CLIP text query.
    """
    if mode == "mmvp":
        result = _mmvp(question)
    elif mode == "vlmbias":
        result = _vlmbias(question)
    elif mode == "pope":
        result = _pope(question)
    elif mode == "vlind":
        result = _vlind(question)
    elif mode == "whatsup":
        result = _whatsup(question)  # Returns (noun_A, noun_B)
    else:
        raise ValueError(f"Unknown mode {mode!r}. Use 'mmvp', 'vlmbias', 'pope', 'vlind', or 'whatsup'.")

    if singular and isinstance(result, str):
        result = _singularize(result)
    return result
