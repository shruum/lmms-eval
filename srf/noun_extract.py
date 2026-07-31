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
    """Extract the queried object from a POPE / MME existence question.

    "Is there a cat in the image?" → "cat"
    "Is there a tennis racket in the image?" → "tennis racket"
    "Is there only one bottle in the image?" → "bottle"
    "Is the pineapple on the left?" → "pineapple"
    """
    q = question.strip().lower().rstrip("?")

    # "Is there only [one/a/an/...] X in/on/at"
    m = re.search(r"\bis there (?:only\s+)?(?:\w+\s+)?(\w+(?:\s+\w+)?)\s+(?:in|on|at|visible)", q)
    if m:
        noun = m.group(1).strip()
        if noun not in _GENERIC_NOUNS:
            return noun

    # "Is there a/an X in/on/at the image"
    m = re.search(r"\bis there (?:an? )?(.+?)\s+(?:in|on|at|visible)", q)
    if m:
        noun = m.group(1).strip()
        if noun not in _GENERIC_NOUNS:
            return noun

    # "Is the/a X [position/colour/...]" — MME position/color questions
    m = re.search(r"\bis (?:the|a|an) (\w+(?:\s+\w+)?)\s+(?:on|in|at|to|left|right|above|below|next)", q)
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

    words = re.findall(r'\b[a-z]{4,}\b', q)
    return words[0] if words else "object"


# ---------------------------------------------------------------------------
# MMBench / general MCQ mode
# ---------------------------------------------------------------------------
def _mmbench(question: str) -> str:
    """Extract the main visual subject from an MMBench MCQ question.

    MMBench questions are diverse (spatial, attribute, counting, scene).
    Strategy: strip options if present, then apply mmvp-style extraction
    with extra patterns for MCQ phrasing.
    """
    # Strip "A. ... B. ... C. ... D. ..." option block if present
    q = re.split(r'\n[A-D][\.\)]\s', question)[0].strip().lower().rstrip("?")

    # Common MCQ patterns
    # "what is the X in/of/on"
    m = re.search(r'what (?:is|are) the (\w+(?:\s+\w+)?)\s+(?:in|of|on|at|doing|shown)', q)
    if m:
        noun = m.group(1).strip()
        if noun not in _GENERIC_NOUNS:
            return noun

    # "what color/shape/size is the X"
    m = re.search(r'what (?:color|shape|size|type|kind)\s+(?:is|are)\s+(?:the|a|an)\s+(\w+)', q)
    if m:
        noun = m.group(1).strip()
        if noun not in _GENERIC_NOUNS:
            return noun

    # "where is the X" / "what is the position of the X"
    m = re.search(r'(?:where is|position of)\s+(?:the|a|an)\s+(\w+)', q)
    if m:
        noun = m.group(1).strip()
        if noun not in _GENERIC_NOUNS:
            return noun

    # "how many X"
    m = re.search(r'how many (\w+(?:\s+\w+)?)\s+(?:are|is|can|do)', q)
    if m:
        noun = m.group(1).strip()
        if noun not in _GENERIC_NOUNS:
            return noun

    # "which X" — "which animal", "which object"
    m = re.search(r'which (\w+)', q)
    if m:
        noun = m.group(1).strip()
        if noun not in _GENERIC_NOUNS | {"one", "of", "the"}:
            return noun

    # Fall back to mmvp extractor (handles most remaining cases)
    return _mmvp(question)


# ---------------------------------------------------------------------------
# VLind-Bench mode — dual noun extraction
# ---------------------------------------------------------------------------

_VLIND_SKIP = _GENERIC_NOUNS | {
    "early", "usual", "typical", "normal", "traditional", "original",
    "their", "they", "were", "have", "been", "does", "used", "found",
    "aquatic", "urban", "open", "flat", "natural", "wild", "bubbling",
}


def _vlind_subject(statement: str) -> str:
    """Extract the image subject (first content noun) from a VLIND statement.

    'The swans are found in desert sands.'     → 'swans'
    'A jungle surrounds the Eiffel Tower.'     → 'jungle'
    'The chameleon is eating tofu.'            → 'chameleon'
    'The feather is longer than the person.'   → 'feather'
    'Racers drive with views of bubbling lava.'→ 'racers'
    """
    q = statement.strip().lower().rstrip(".?!")

    # "The/A/An X ..." — single noun after article (not a verb/adjective)
    m = re.search(r'^(?:the|a|an) (\w+)\b', q)
    if m:
        noun = m.group(1)
        if noun not in _VLIND_SKIP and len(noun) > 2:
            return noun

    # No article: first non-skip word of 3+ chars
    words = re.findall(r'\b[a-z]{3,}\b', q)
    for w in words:
        if w not in _VLIND_SKIP:
            return w
    return "object"


def _vlind_context(statement: str) -> str:
    """Extract the counterfactual context/environment noun from a VLIND statement.

    This is the UNUSUAL element — what makes the image counterfactual.
    'The swans are found in desert sands.'              → 'desert'
    'A jungle surrounds the Eiffel Tower.'              → 'jungle'   (subject IS the context)
    'The chameleon is eating tofu.'                     → 'tofu'
    'The feather is longer than the person.'            → 'person'
    'Racers drive with views of bubbling lava.'         → 'lava'
    'Gladiators used communication devices during fights.' → 'devices'
    'The grain of rice is heavier than the basketball.' → 'basketball'
    'The swans are aquatic birds found in lakes.'       → 'lakes'
    'The Eiffel Tower is surrounded by urban landscape.'→ 'landscape'
    """
    q = statement.strip().lower().rstrip(".?!")

    # "X surrounds [the] Y" — X is the unusual environment (subject)
    m = re.search(r'^(?:the |a |an )?(\w+)\s+surrounds?\b', q)
    if m and m.group(1) not in _VLIND_SKIP:
        return m.group(1)

    # "surrounded by [its/the/a/usual/adj...] X" — X wraps the subject
    m = re.search(r'\bsurrounded by\s+([\w\s,]+?)(?:\.|$)', q)
    if m:
        cands = [w for w in re.findall(r'\b[a-z]{4,}\b', m.group(1)) if w not in _VLIND_SKIP]
        if cands:
            return cands[-1]  # last non-skip word = the actual environment noun

    # "found/located/living in/on/at [det] [adj...] X"
    m = re.search(r'\b(?:located|living|lives?)\s+(?:in|on|at|near)\s+([\w\s,]+?)(?:\.|,\s*(?:and|but|or)|$)', q)
    if m:
        cands = [w for w in re.findall(r'\b[a-z]{4,}\b', m.group(1)) if w not in _VLIND_SKIP]
        if cands:
            return cands[-1]

    # "eating/drinking/consuming [a/the] X"
    m = re.search(r'\b(?:eating|drinking|consuming)\s+(?:a |an |the )?(\w+)', q)
    if m and m.group(1) not in _VLIND_SKIP:
        return m.group(1)

    # "heavier/taller/longer than [the] X"
    m = re.search(r'\b(?:heavier|taller|longer|bigger|smaller|shorter|lighter)\s+than\s+(?:the |a |an )?(\w+)', q)
    if m and m.group(1) not in _VLIND_SKIP:
        return m.group(1)

    # "views of [adj] X"
    m = re.search(r'\bviews? of\s+(?:\w+\s+)?(\w+)', q)
    if m and m.group(1) not in _VLIND_SKIP:
        return m.group(1)

    # "used/using [adj] X" — e.g. "used communication devices"
    m = re.search(r'\b(?:used?|using|included?)\s+(?:\w+\s+)?(\w+)\b', q)
    if m and m.group(1) not in _VLIND_SKIP | {"during", "while", "when"}:
        return m.group(1)

    # "in/on/at/near [det] [adj...] X" — first non-skip noun after the preposition
    for prep_m in re.finditer(r'\b(?:in|on|at|near|inside)\s+([\w\s,]+?)(?:\.|,\s*(?:and|but|or)|$)', q):
        cands = [w for w in re.findall(r'\b[a-z]{4,}\b', prep_m.group(1)) if w not in _VLIND_SKIP]
        if cands:
            return cands[0]  # first = environment type (e.g. "desert" in "desert sands")

    # Fallback: second non-skip noun (skip the subject)
    words = re.findall(r'\b[a-z]{4,}\b', q)
    skip = 0
    for w in words:
        if w not in _VLIND_SKIP:
            skip += 1
            if skip > 1:
                return w
    return _vlind_subject(statement)


def extract_vlind_nouns(statement: str) -> tuple:
    """Return (subject_noun, context_noun) from a VLIND statement.

    subject_noun  — the image subject (used for saliency MAP localisation).
    context_noun  — the counterfactual context (used for presence GATE).

    'The swans are found in desert sands.'   → ('swans', 'desert')
    'A jungle surrounds the Eiffel Tower.'   → ('jungle', 'jungle')  ← jungle IS the subject
    'The chameleon is eating tofu.'          → ('chameleon', 'tofu')
    'Racers drive with views of bubbling lava.' → ('racers', 'lava')
    """
    subject = _vlind_subject(statement)
    context = _vlind_context(statement)
    return subject, context


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def extract_clip_noun(question: str, mode: str = "mmvp") -> str:
    """Return the best CLIP query noun for a VLM benchmark question.

    Args:
        question: Raw question string (any casing).
        mode:     Dataset mode — "mmvp", "vlmbias", "pope", or "mmbench".

    Returns:
        A short noun string suitable for CLIP text query.
    """
    if mode == "mmvp":
        return _mmvp(question)
    elif mode == "vlmbias":
        return _vlmbias(question)
    elif mode == "pope":
        return _pope(question)
    elif mode in ("mmbench", "hallusionbench"):
        return _mmbench(question)
    else:
        # Fallback to mmvp for unknown modes
        return _mmvp(question)
