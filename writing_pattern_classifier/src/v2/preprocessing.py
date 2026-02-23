import regex
import unicodedata


# ─────────────────────────────────────────────────────────────────────────────
# Sinhala Unicode ranges
# ─────────────────────────────────────────────────────────────────────────────
SINHALA_RANGE = (0x0D80, 0x0DFF)

# Sinhala vowel diacritics (්, ා, ැ, ෑ, ි, ී, ු, ූ, ෙ, ේ, ෛ, ො, ෝ, ෞ, ෟ)
SINHALA_VOWEL_DIACRITICS = set(
    chr(c) for c in range(0x0DCF, 0x0DE0)
)

# Al-lakuna (virama) – used to cancel inherent vowel
AL_LAKUNA = "\u0DCA"

# Sinhala independent vowels
SINHALA_INDEPENDENT_VOWELS = set(
    chr(c) for c in range(0x0D85, 0x0D97)
)

# Sinhala consonants
SINHALA_CONSONANTS = set(
    chr(c) for c in range(0x0D9A, 0x0DC7)
)

# Visually / phonetically similar consonant pairs common in dyslexia
# (character pairs that are frequently confused)
SIMILAR_CONSONANT_PAIRS = [
    ("\u0D9A", "\u0D9B"),  # ක / ඛ
    ("\u0D9C", "\u0D9D"),  # ග / ඝ
    ("\u0DA0", "\u0DA1"),  # ච / ඡ
    ("\u0DA4", "\u0DA5"),  # ඤ / ඥ
    ("\u0DA7", "\u0DA8"),  # ට / ඨ
    ("\u0DAA", "\u0DAB"),  # ඩ / ඪ
    ("\u0DAD", "\u0DAE"),  # ත / ථ
    ("\u0DB0", "\u0DB1"),  # ද / ධ
    ("\u0DB3", "\u0DB4"),  # න / ණ
    ("\u0DB5", "\u0DB7"),  # ප / ඵ
    ("\u0DB6", "\u0DB8"),  # බ / භ
    ("\u0DB9", "\u0DBA"),  # ම / ය  (less common but occurs)
    ("\u0DC0", "\u0DC2"),  # ව / ශ
    ("\u0DC3", "\u0DC4"),  # ස / හ
]
SIMILAR_CONSONANT_SET = set()
for a, b in SIMILAR_CONSONANT_PAIRS:
    SIMILAR_CONSONANT_SET.add(a)
    SIMILAR_CONSONANT_SET.add(b)

# Short/long vowel diacritic pairs (common confusion in dyslexia)
SHORT_LONG_PAIRS = [
    ("\u0DCF", "\u0DD0"),  # ා / ැ
    ("\u0DD2", "\u0DD3"),  # ි / ී
    ("\u0DD4", "\u0DD6"),  # ු / ූ
    ("\u0DD9", "\u0DDA"),  # ෙ / ේ
    ("\u0DDC", "\u0DDD"),  # ො / ෝ
]
SHORT_VOWELS = {p[0] for p in SHORT_LONG_PAIRS}
LONG_VOWELS  = {p[1] for p in SHORT_LONG_PAIRS}


def _grapheme_clusters(text: str):
    """Return list of Unicode grapheme clusters using the \\X regex."""
    return regex.findall(r"\X", text)


def _sinhala_graphemes(text: str):
    """Return only grapheme clusters that contain at least one Sinhala character."""
    clusters = _grapheme_clusters(text)
    return [
        g for g in clusters
        if any(SINHALA_RANGE[0] <= ord(c) <= SINHALA_RANGE[1] for c in g)
    ]


def extract_structured_features(text: str) -> dict:
    """
    Extract a rich set of Sinhala-aware features for dyslexic pattern detection.

    Feature groups
    ──────────────
    1. Surface length features
    2. Grapheme-level features
    3. Vowel diacritic features  ← key for Phonetic / Visual errors
    4. Consonant features        ← key for Spelling / Visual errors
    5. Al-lakuna (virama) features
    6. Repetition / gemination features
    7. Short–long vowel confusion proxy
    8. Sentence-level ratios (length-normalised)
    """
    features = {}

    # ── 1. Surface length ──────────────────────────────────────────────────
    features["char_length"] = len(text)
    words = text.split()
    features["word_count"] = len(words)
    features["avg_word_length"] = (
        sum(len(w) for w in words) / len(words) if words else 0.0
    )

    # ── 2. Grapheme clusters ───────────────────────────────────────────────
    all_clusters   = _grapheme_clusters(text)
    sinh_clusters  = _sinhala_graphemes(text)
    n_clusters     = len(all_clusters)
    n_sinh         = len(sinh_clusters)

    features["grapheme_count"]       = n_clusters
    features["sinhala_grapheme_count"] = n_sinh
    features["sinhala_grapheme_ratio"] = n_sinh / n_clusters if n_clusters else 0.0

    # ── 3. Vowel diacritic features ────────────────────────────────────────
    all_chars = list(text)

    vowel_diac_count = sum(1 for c in all_chars if c in SINHALA_VOWEL_DIACRITICS)
    indep_vowel_count = sum(1 for c in all_chars if c in SINHALA_INDEPENDENT_VOWELS)

    features["vowel_diacritic_count"] = vowel_diac_count
    features["independent_vowel_count"] = indep_vowel_count
    features["vowel_diacritic_ratio"] = (
        vowel_diac_count / n_sinh if n_sinh else 0.0
    )

    # Short-vowel vs long-vowel counts
    short_v = sum(1 for c in all_chars if c in SHORT_VOWELS)
    long_v  = sum(1 for c in all_chars if c in LONG_VOWELS)
    features["short_vowel_count"] = short_v
    features["long_vowel_count"]  = long_v
    # Imbalance ratio: high value → many short or many long (possible confusion)
    total_v = short_v + long_v
    features["vowel_length_imbalance"] = (
        abs(short_v - long_v) / total_v if total_v else 0.0
    )

    # ── 4. Consonant features ──────────────────────────────────────────────
    consonant_chars = [c for c in all_chars if c in SINHALA_CONSONANTS]
    n_consonants = len(consonant_chars)
    features["consonant_count"] = n_consonants
    features["consonant_ratio"]  = n_consonants / n_sinh if n_sinh else 0.0

    # How many consonants are in the visually-similar confusable set?
    confusable_consonants = sum(
        1 for c in consonant_chars if c in SIMILAR_CONSONANT_SET
    )
    features["confusable_consonant_count"] = confusable_consonants
    features["confusable_consonant_ratio"] = (
        confusable_consonants / n_consonants if n_consonants else 0.0
    )

    # ── 5. Al-lakuna (virama / halant) features ────────────────────────────
    al_count = text.count(AL_LAKUNA)
    features["al_lakuna_count"] = al_count
    features["al_lakuna_ratio"] = al_count / n_sinh if n_sinh else 0.0
    # Consecutive al-lakunas can signal omission errors
    features["consecutive_al_lakuna"] = sum(
        1 for i in range(len(all_chars) - 1)
        if all_chars[i] == AL_LAKUNA and all_chars[i + 1] == AL_LAKUNA
    )

    # ── 6. Repetition / gemination features ───────────────────────────────
    # Repeated consecutive grapheme clusters
    repeated = sum(
        1 for i in range(1, len(all_clusters))
        if all_clusters[i] == all_clusters[i - 1]
    )
    features["repeated_graphemes"] = repeated

    # Repeated characters (non-grapheme level)
    repeated_chars = sum(
        1 for i in range(1, len(all_chars))
        if all_chars[i] == all_chars[i - 1]
    )
    features["repeated_chars"] = repeated_chars

    # Word-level repetitions (same word appearing consecutively)
    repeated_words = sum(
        1 for i in range(1, len(words))
        if words[i] == words[i - 1]
    )
    features["repeated_words"] = repeated_words

    # ── 7. Suffix / word-ending features ──────────────────────────────────
    # Sinhala verbs typically end with specific diacritics.
    # Missing these may indicate suffix omission.
    # We approximate by counting words whose last grapheme cluster is a pure
    # consonant (no diacritic attached) — a proxy for omitted suffix.
    bare_consonant_endings = 0
    for word in words:
        w_clusters = _sinhala_graphemes(word)
        if w_clusters:
            last_cluster = w_clusters[-1]
            # "bare" = cluster that is just a consonant with al-lakuna or raw
            if any(c in SINHALA_CONSONANTS for c in last_cluster) and not any(
                c in SINHALA_VOWEL_DIACRITICS for c in last_cluster
            ):
                bare_consonant_endings += 1

    features["bare_consonant_endings"] = bare_consonant_endings
    features["bare_consonant_ending_ratio"] = (
        bare_consonant_endings / len(words) if words else 0.0
    )

    # ── 8. Length-normalised ratios ────────────────────────────────────────
    features["vowel_per_word"]     = vowel_diac_count / len(words) if words else 0.0
    features["consonant_per_word"] = n_consonants / len(words) if words else 0.0

    return features
