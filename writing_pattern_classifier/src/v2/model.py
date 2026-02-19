import os
import joblib
import numpy as np
from scipy.sparse import hstack
from .preprocessing import extract_structured_features


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

MODEL_PATH = os.path.join(ARTIFACT_DIR, "v2_pattern_model.pkl")
VECTORIZER_PATH = os.path.join(ARTIFACT_DIR, "v2_pattern_vectorizer.pkl")
LABEL_ENCODER_PATH = os.path.join(ARTIFACT_DIR, "v2_pattern_label_encoder.pkl")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)


def predict_sentence_pattern(text):
    structured = extract_structured_features(text)
    structured_array = np.array(list(structured.values())).reshape(1, -1)

    text_vec = vectorizer.transform([text])
    combined = hstack([text_vec, structured_array])

    probs = model.predict_proba(combined)[0]

    return dict(zip(label_encoder.classes_, probs))
