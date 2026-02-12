"""
Sentence-level dyslexia classifier.

Responsibility:
- Load the trained binary classifier
- Predict dyslexia probability for a single sentence

This module performs ONLY sentence-level inference.
Essay-level logic is handled elsewhere.
"""

import joblib
import os
from src.vectorizer import vectorize_sentence

# ------------------------------------------------------------
# Resolve project root directory
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Absolute path to trained binary classifier
MODEL_PATH = os.path.join(
    BASE_DIR, "models", "dyslexia_binary_model.pkl"
)

# Cached model instance (lazy loaded)
_model = None


def load_model():
    """
    Loads the trained dyslexia classifier from disk if not already loaded.

    Lazy loading ensures:
    - Model is loaded only once
    - Faster inference for multiple requests
    """
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_sentence(sentence: str) -> float:
    """
    Predicts the probability that a sentence is dyslexic.

    Args:
        sentence (str): Sinhala sentence

    Returns:
        float: Probability of dyslexia (0.0 – 1.0)
    """
    model = load_model()

    # Convert sentence to TF-IDF features
    vec = vectorize_sentence(sentence)

    # Predict probability for class '1' (dyslexic)
    prob = model.predict_proba(vec)[0][1]

    return float(prob)
