import os
import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix
from .preprocessing import extract_structured_features


BASE_DIR           = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
ARTIFACT_DIR       = os.path.join(BASE_DIR, "artifacts")

MODEL_PATH         = os.path.join(ARTIFACT_DIR, "v2_pattern_model.pkl")
VECTORIZER_PATH    = os.path.join(ARTIFACT_DIR, "v2_pattern_vectorizer.pkl")
LABEL_ENCODER_PATH = os.path.join(ARTIFACT_DIR, "v2_pattern_label_encoder.pkl")
SCALER_PATH        = os.path.join(ARTIFACT_DIR, "v2_pattern_scaler.pkl")

model         = joblib.load(MODEL_PATH)
vectorizer    = joblib.load(VECTORIZER_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

# Scaler is optional — only present after running train_v2_improved.py
# Old models trained without scaling still work (scaler=None path below)
scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None


def _build_feature_matrix(sentences: list):
    """
    Build combined TF-IDF + structured feature matrix for a list of sentences.
    Applies StandardScaler to structured features if scaler artifact exists.
    """
    # Structured features (26 features per sentence)
    structured_list = [
        list(extract_structured_features(s).values())
        for s in sentences
    ]
    structured_array = np.array(structured_list, dtype=np.float32)

    # Apply scaler if available (from train_v2_improved.py)
    if scaler is not None:
        structured_array = scaler.transform(structured_array)

    # TF-IDF text features
    text_vec = vectorizer.transform(sentences)

    # Combine sparse text + structured
    return hstack([text_vec, csr_matrix(structured_array)])


def predict_sentence_pattern(text: str) -> dict:
    """Single sentence prediction. Returns {label: probability} dict."""
    combined = _build_feature_matrix([text])
    probs    = model.predict_proba(combined)[0]
    return dict(zip(label_encoder.classes_, probs))


def predict_batch_sentence_patterns(sentences: list) -> list:
    """
    Batch prediction — much faster than looping predict_sentence_pattern.
    Returns list of {label: probability} dicts, one per sentence.
    """
    combined = _build_feature_matrix(sentences)
    probs    = model.predict_proba(combined)
    return [
        dict(zip(label_encoder.classes_, probs[i]))
        for i in range(len(sentences))
    ]
