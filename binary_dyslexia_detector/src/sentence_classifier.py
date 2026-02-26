"""
Sentence-level dyslexia classifier — Improved v2.

Responsibility:
- Load the trained calibrated ensemble models (LR + SVC)
- Predict dyslexia probability for a single sentence

Changes from v1:
  - Loads two calibrated models (LR + LinearSVC)
  - Averages their calibrated probability outputs (soft voting)
  - Calibrated probabilities → more reliable thresholding
"""

import joblib
import os
from binary_dyslexia_detector.src.vectorizer import vectorize_sentence


# ------------------------------------------------------------
# Resolve project root directory
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LR_MODEL_PATH  = os.path.join(BASE_DIR, "models", "dyslexia_binary_model_lr.pkl")
SVC_MODEL_PATH = os.path.join(BASE_DIR, "models", "dyslexia_binary_model_svc.pkl")

# Cached model instances (lazy loaded)
_lr_model  = None
_svc_model = None


def load_lr_model():
    """Loads the calibrated Logistic Regression model."""
    global _lr_model
    if _lr_model is None:
        _lr_model = joblib.load(LR_MODEL_PATH)
    return _lr_model


def load_svc_model():
    """Loads the calibrated LinearSVC model."""
    global _svc_model
    if _svc_model is None:
        _svc_model = joblib.load(SVC_MODEL_PATH)
    return _svc_model


def predict_sentence(sentence: str) -> float:
    """
    Predicts the probability that a sentence is dyslexic.

    Uses a soft-voting ensemble of two calibrated models:
      - Calibrated Logistic Regression (char+word TF-IDF + handcrafted features)
      - Calibrated LinearSVC           (same feature set)

    Soft voting averages their probability outputs, which:
      - Reduces prediction variance
      - Improves reliability on borderline cases (0.45–0.65 range)
      - Preserves calibrated probability meaning

    Args:
        sentence (str): Sinhala sentence

    Returns:
        float: Calibrated probability of dyslexia (0.0 – 1.0)
    """
    lr_model  = load_lr_model()
    svc_model = load_svc_model()

    # Convert sentence to combined feature vector
    vec = vectorize_sentence(sentence)

    # Predict from both calibrated models
    lr_prob  = lr_model.predict_proba(vec)[0][1]
    svc_prob = svc_model.predict_proba(vec)[0][1]

    # Soft voting: equal-weight average
    ensemble_prob = (lr_prob + svc_prob) / 2.0

    return float(ensemble_prob)
