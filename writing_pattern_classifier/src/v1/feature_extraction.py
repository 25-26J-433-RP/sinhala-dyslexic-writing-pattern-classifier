# feature_extraction.py
"""
Sentence-level surface feature extraction for Sinhala dyslexic writing analysis.

This module computes interpretable surface-level error signals
by comparing clean and dyslexic sentence pairs.
"""

import difflib

# Sinhala diacritic characters
SINHALA_DIACRITICS = set([
    "ා", "ැ", "ෑ", "ි", "ී", "ු", "ූ", "ෘ", "ෙ", "ේ", "ො", "ෝ", "ං", "ඃ"
])


def char_level_diff(clean: str, dyslexic: str) -> dict:
    """
    Compute character-level edit operations between clean and dyslexic sentences.
    """
    matcher = difflib.SequenceMatcher(None, clean, dyslexic)

    additions = omissions = substitutions = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "insert":
            additions += (j2 - j1)
        elif tag == "delete":
            omissions += (i2 - i1)
        elif tag == "replace":
            substitutions += max(i2 - i1, j2 - j1)

    return {
        "char_addition": additions,
        "char_omission": omissions,
        "char_substitution": substitutions,
        "has_addition": additions > 0,
        "has_omission": omissions > 0,
        "has_substitution": substitutions > 0,
    }


def spacing_diff(clean: str, dyslexic: str) -> dict:
    """
    Detect word boundary (spacing) inconsistencies.
    """
    diff = abs(len(clean.split()) - len(dyslexic.split()))
    return {
        "word_count_diff": diff,
        "has_spacing_issue": diff > 0,
    }


def diacritic_loss(clean: str, dyslexic: str) -> dict:
    """
    Detect diacritic loss in dyslexic writing.
    """
    clean_count = sum(1 for c in clean if c in SINHALA_DIACRITICS)
    dys_count = sum(1 for c in dyslexic if c in SINHALA_DIACRITICS)

    return {
        "has_diacritic_loss": clean_count > dys_count
    }


def extract_surface_features_offline(clean_sentence, dyslexic_sentence):
    """
    OFFLINE FEATURE EXTRACTOR
    Uses clean + dyslexic sentence pairs.
    Used ONLY for training / analysis.
    """
    ...

    features = {}

    features.update(char_level_diff(clean_sentence, dyslexic_sentence))
    features.update(spacing_diff(clean_sentence, dyslexic_sentence))
    features.update(diacritic_loss(clean_sentence, dyslexic_sentence))

    return features

import difflib
import regex as re


def extract_surface_features_runtime(raw_sentence: str, reference_sentence: str) -> dict:
    """
    Extract surface-level dyslexic features at runtime
    using RAW sentence vs PROXY reference.

    This is conservative and abstains from deep correction.
    """

    features = {
        "char_addition": 0,
        "char_omission": 0,
        "char_substitution": 0,
        "has_addition": False,
        "has_omission": False,
        "has_substitution": False,
        "word_count_diff": 0,
        "has_spacing_issue": False,
        "has_diacritic_loss": False,
    }

    if not isinstance(raw_sentence, str) or not isinstance(reference_sentence, str):
        return features

    # --- Character-level diff ---
    diff = difflib.ndiff(raw_sentence, reference_sentence)

    for d in diff:
        if d.startswith("- "):
            features["char_omission"] += 1
            features["has_omission"] = True

        elif d.startswith("+ "):
            features["char_addition"] += 1
            features["has_addition"] = True

        elif d.startswith("? "):
            # caret markers often indicate substitutions / diacritic instability
            features["char_substitution"] += 1
            features["has_substitution"] = True

    # --- Word boundary issues ---
    raw_words = raw_sentence.split()
    ref_words = reference_sentence.split()

    if len(raw_words) != len(ref_words):
        features["word_count_diff"] = abs(len(raw_words) - len(ref_words))
        features["has_spacing_issue"] = True

    # --- Diacritic loss (very conservative) ---
    # Sinhala diacritics Unicode range heuristic
    diacritic_pattern = re.compile(r"[\u0DCA-\u0DDF]")

    raw_diacritics = len(diacritic_pattern.findall(raw_sentence))
    ref_diacritics = len(diacritic_pattern.findall(reference_sentence))

    if raw_diacritics < ref_diacritics:
        features["has_diacritic_loss"] = True

    return features


