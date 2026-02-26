"""
Vectorizer module — Improved v2.

Responsibility:
- Load the trained TF-IDF vectorizers (char + word)
- Extract 8 Sinhala-specific handcrafted features
- Return a combined sparse feature matrix for inference

Changes from v1:
  - Added word-level TF-IDF (1,2) stacked with char TF-IDF (2,5)
  - Added extract_sinhala_features() for 8 linguistic features
  - vectorize_sentence() now returns a combined matrix
"""

import re
import joblib
import os
import numpy as np
from scipy.sparse import hstack, csr_matrix

# ------------------------------------------------------------
# Resolve project root directory
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHAR_VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_char_vectorizer.pkl")
WORD_VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_word_vectorizer.pkl")

# Lazy-loaded vectorizer instances
_char_vectorizer = None
_word_vectorizer = None


def load_char_vectorizer():
    global _char_vectorizer
    if _char_vectorizer is None:
        _char_vectorizer = joblib.load(CHAR_VECTORIZER_PATH)
    return _char_vectorizer


def load_word_vectorizer():
    global _word_vectorizer
    if _word_vectorizer is None:
        _word_vectorizer = joblib.load(WORD_VECTORIZER_PATH)
    return _word_vectorizer


def extract_sinhala_features(sentence: str) -> csr_matrix:
    """
    Extracts 8 Sinhala-specific handcrafted features.

    These capture linguistic patterns that are characteristic of
    dyslexic Sinhala writing but are invisible to TF-IDF:

    1. hal_ratio         — fraction of hal kirima (්) chars
                           Dyslexic writers drop geminate markers: අම්මා → මමා
    2. diacritic_ratio   — vowel diacritic density
                           Wrong/missing diacritics are a core dyslexia marker
    3. avg_word_len      — mean word length in characters
    4. word_count        — total words in the sentence
    5. unique_char_ratio — distinct Sinhala chars / total Sinhala chars
                           Measures character-level variety / substitutions
    6. space_ratio       — spaces / total chars (word-boundary anomalies)
    7. has_english       — presence of Latin characters (0 or 1)
    8. repeat_char_ratio — consecutive repeated character ratio (perseveration)

    Sinhala Unicode reference:
      Block:     U+0D80–U+0DFF
      Hal kirima (virama):  U+0DCA (්)
      Dependent vowels:     U+0DCF–U+0DDF
    """
    chars  = list(sentence)
    n      = max(len(chars), 1)
    words  = sentence.split()
    nw     = max(len(words), 1)

    hal_count       = sentence.count('\u0DCA')
    hal_ratio       = hal_count / n

    diacritic_count = sum(1 for c in chars if '\u0DCF' <= c <= '\u0DDF')
    diacritic_ratio = diacritic_count / n

    avg_word_len    = sum(len(w) for w in words) / nw
    word_count      = float(nw)

    sinhala_chars   = [c for c in chars if '\u0D80' <= c <= '\u0DFF']
    unique_ratio    = len(set(sinhala_chars)) / max(len(sinhala_chars), 1)

    space_ratio     = sentence.count(' ') / n

    has_english     = float(bool(re.search(r'[a-zA-Z]', sentence)))

    repeats         = sum(1 for i in range(1, len(chars)) if chars[i] == chars[i - 1])
    repeat_ratio    = repeats / n

    feature_vector  = np.array([[
        hal_ratio,
        diacritic_ratio,
        avg_word_len,
        word_count,
        unique_ratio,
        space_ratio,
        has_english,
        repeat_ratio
    ]], dtype=np.float32)

    return csr_matrix(feature_vector)


def vectorize_sentence(sentence: str):
    """
    Converts a single Sinhala sentence into a combined feature vector.

    Feature composition:
      [char TF-IDF (2,5)] + [word TF-IDF (1,2)] + [8 handcrafted features]

    Returns:
        scipy sparse matrix: Combined feature representation
    """
    char_vec = load_char_vectorizer()
    word_vec = load_word_vectorizer()

    char_features = char_vec.transform([sentence])
    word_features = word_vec.transform([sentence])
    hand_features = extract_sinhala_features(sentence)

    return hstack([char_features, word_features, hand_features])
