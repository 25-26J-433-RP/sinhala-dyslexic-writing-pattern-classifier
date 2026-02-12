"""
Vectorizer module.

Responsibility:
- Load the trained TF-IDF vectorizer
- Convert raw Sinhala sentences into numerical feature vectors

This module is intentionally isolated so that:
- Feature extraction logic is reusable
- Vectorizer is loaded only once (lazy loading)
- Inference code remains clean and readable
"""

import joblib
import os

# ------------------------------------------------------------
# Resolve project root directory
# ------------------------------------------------------------
# BASE_DIR points to: binary_dyslexia_detector/
# This allows model paths to work regardless of where the app is run
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Absolute path to the saved TF-IDF vectorizer
VECTORIZER_PATH = os.path.join(
    BASE_DIR, "models", "tfidf_vectorizer.pkl"
)

# Cached vectorizer instance (loaded once)
_vectorizer = None


def load_vectorizer():
    """
    Loads the TF-IDF vectorizer from disk if not already loaded.

    Uses lazy loading to:
    - Avoid repeated disk I/O
    - Improve inference performance
    """
    global _vectorizer
    if _vectorizer is None:
        _vectorizer = joblib.load(VECTORIZER_PATH)
    return _vectorizer


def vectorize_sentence(sentence: str):
    """
    Converts a single Sinhala sentence into a TF-IDF feature vector.

    Args:
        sentence (str): Raw Sinhala sentence

    Returns:
        scipy sparse matrix: Vectorized sentence representation
    """
    vectorizer = load_vectorizer()
    return vectorizer.transform([sentence])
