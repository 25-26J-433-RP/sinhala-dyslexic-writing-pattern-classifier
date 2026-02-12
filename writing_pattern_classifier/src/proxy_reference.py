"""
Proxy reference generation for Sinhala dyslexic writing analysis.

Conservative, deterministic, Sinhala-aware.
This is NOT correction.
"""

import unicodedata
import regex as re  # IMPORTANT: grapheme-aware regex


# --------------------------------------------------
# Layer 1: Unicode & whitespace normalization
# --------------------------------------------------

def normalize_unicode_and_space(text: str) -> str:
    if not isinstance(text, str):
        return text

    text = unicodedata.normalize("NFC", text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)

    return text


# --------------------------------------------------
# Layer 2: Grapheme-aware repetition stabilization
# --------------------------------------------------

def stabilize_character_repetition(text: str) -> str:
    """
    Collapse 3+ repeated grapheme clusters → exactly 2.
    Preserve 1 or 2 repetitions.
    """

    if not isinstance(text, str):
        return text

    graphemes = re.findall(r"\X", text)

    result = []
    prev = None
    count = 0

    for g in graphemes:
        if g == prev:
            count += 1
        else:
            prev = g
            count = 1

        if count <= 2:
            result.append(g)

    return "".join(result)


# --------------------------------------------------
# Public API
# --------------------------------------------------

def generate_proxy_reference(raw_sentence: str) -> str:
    if not isinstance(raw_sentence, str):
        return raw_sentence

    proxy = normalize_unicode_and_space(raw_sentence)
    proxy = stabilize_character_repetition(proxy)

    return proxy
