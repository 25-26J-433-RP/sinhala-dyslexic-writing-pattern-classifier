"""
train_v2_improved.py
────────────────────
Retrain the Sinhala dyslexic writing pattern classifier with:
  • Rich structured features from improved preprocessing.py
  • Character n-gram TF-IDF (same as before, but wider range)
  • Calibrated probability outputs (CalibratedClassifierCV)
  • Balanced + stratified training
  • Held-out evaluation with per-class F1
"""

import os
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack, csr_matrix
import joblib

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)


# ── Label normalisation ────────────────────────────────────────────────────
def normalize_label(label: str) -> str:
    label = str(label).lower()
    if "phonetic" in label:
        return "Phonetic"
    if "visual" in label or "scrambling" in label or "reversal" in label:
        return "Visual"
    if "grammar" in label or "tense" in label or "case" in label:
        return "Grammar"
    if "spelling" in label:
        return "Spelling"
    return "None"


# ── Feature extraction (must mirror preprocessing.py) ─────────────────────
import regex
import unicodedata

SINHALA_RANGE            = (0x0D80, 0x0DFF)
SINHALA_VOWEL_DIACRITICS = set(chr(c) for c in range(0x0DCF, 0x0DE0))
AL_LAKUNA                = "\u0DCA"
SINHALA_INDEPENDENT_VOWELS = set(chr(c) for c in range(0x0D85, 0x0D97))
SINHALA_CONSONANTS       = set(chr(c) for c in range(0x0D9A, 0x0DC7))

SIMILAR_CONSONANT_PAIRS = [
    ("\u0D9A", "\u0D9B"), ("\u0D9C", "\u0D9D"), ("\u0DA0", "\u0DA1"),
    ("\u0DA4", "\u0DA5"), ("\u0DA7", "\u0DA8"), ("\u0DAA", "\u0DAB"),
    ("\u0DAD", "\u0DAE"), ("\u0DB0", "\u0DB1"), ("\u0DB3", "\u0DB4"),
    ("\u0DB5", "\u0DB7"), ("\u0DB6", "\u0DB8"), ("\u0DB9", "\u0DBA"),
    ("\u0DC0", "\u0DC2"), ("\u0DC3", "\u0DC4"),
]
SIMILAR_CONSONANT_SET = {c for pair in SIMILAR_CONSONANT_PAIRS for c in pair}
SHORT_LONG_PAIRS = [
    ("\u0DCF", "\u0DD0"), ("\u0DD2", "\u0DD3"),
    ("\u0DD4", "\u0DD6"), ("\u0DD9", "\u0DDA"), ("\u0DDC", "\u0DDD"),
]
SHORT_VOWELS = {p[0] for p in SHORT_LONG_PAIRS}
LONG_VOWELS  = {p[1] for p in SHORT_LONG_PAIRS}


def _sinhala_graphemes(text):
    clusters = regex.findall(r"\X", text)
    return [g for g in clusters if any(SINHALA_RANGE[0] <= ord(c) <= SINHALA_RANGE[1] for c in g)]


def extract_structured_features(text: str) -> dict:
    features = {}
    words       = text.split()
    all_chars   = list(text)
    all_clusters = regex.findall(r"\X", text)
    sinh_clusters = _sinhala_graphemes(text)
    n_clusters   = len(all_clusters)
    n_sinh       = len(sinh_clusters)

    features["char_length"]          = len(text)
    features["word_count"]           = len(words)
    features["avg_word_length"]      = sum(len(w) for w in words) / len(words) if words else 0.0
    features["grapheme_count"]       = n_clusters
    features["sinhala_grapheme_count"] = n_sinh
    features["sinhala_grapheme_ratio"] = n_sinh / n_clusters if n_clusters else 0.0

    vowel_diac = sum(1 for c in all_chars if c in SINHALA_VOWEL_DIACRITICS)
    indep_v    = sum(1 for c in all_chars if c in SINHALA_INDEPENDENT_VOWELS)
    features["vowel_diacritic_count"]  = vowel_diac
    features["independent_vowel_count"] = indep_v
    features["vowel_diacritic_ratio"]  = vowel_diac / n_sinh if n_sinh else 0.0

    short_v = sum(1 for c in all_chars if c in SHORT_VOWELS)
    long_v  = sum(1 for c in all_chars if c in LONG_VOWELS)
    total_v = short_v + long_v
    features["short_vowel_count"]       = short_v
    features["long_vowel_count"]        = long_v
    features["vowel_length_imbalance"]  = abs(short_v - long_v) / total_v if total_v else 0.0

    consonant_chars = [c for c in all_chars if c in SINHALA_CONSONANTS]
    n_consonants = len(consonant_chars)
    confusable   = sum(1 for c in consonant_chars if c in SIMILAR_CONSONANT_SET)
    features["consonant_count"]           = n_consonants
    features["consonant_ratio"]           = n_consonants / n_sinh if n_sinh else 0.0
    features["confusable_consonant_count"] = confusable
    features["confusable_consonant_ratio"] = confusable / n_consonants if n_consonants else 0.0

    al_count = text.count(AL_LAKUNA)
    features["al_lakuna_count"]      = al_count
    features["al_lakuna_ratio"]      = al_count / n_sinh if n_sinh else 0.0
    features["consecutive_al_lakuna"] = sum(
        1 for i in range(len(all_chars) - 1)
        if all_chars[i] == AL_LAKUNA and all_chars[i + 1] == AL_LAKUNA
    )

    repeated_g = sum(1 for i in range(1, n_clusters) if all_clusters[i] == all_clusters[i - 1])
    repeated_c = sum(1 for i in range(1, len(all_chars)) if all_chars[i] == all_chars[i - 1])
    repeated_w = sum(1 for i in range(1, len(words)) if words[i] == words[i - 1])
    features["repeated_graphemes"] = repeated_g
    features["repeated_chars"]     = repeated_c
    features["repeated_words"]     = repeated_w

    bare = 0
    for word in words:
        wc = _sinhala_graphemes(word)
        if wc:
            lc = wc[-1]
            if any(c in SINHALA_CONSONANTS for c in lc) and not any(
                c in SINHALA_VOWEL_DIACRITICS for c in lc
            ):
                bare += 1
    features["bare_consonant_endings"]      = bare
    features["bare_consonant_ending_ratio"] = bare / len(words) if words else 0.0
    features["vowel_per_word"]     = vowel_diac / len(words) if words else 0.0
    features["consonant_per_word"] = n_consonants / len(words) if words else 0.0

    return features


# ── Training ───────────────────────────────────────────────────────────────
def train():
    print("Loading dataset...")
    dataset = load_dataset("SPEAK-ASR/akura-sinhala-dyslexia-corrected")
    df = dataset["train"].to_pandas()

    df["macro_label"] = df["error_type"].apply(normalize_label)
    df = df[df["macro_label"] != "None"].reset_index(drop=True)

    print(f"Dataset size after filtering: {len(df)}")
    print(df["macro_label"].value_counts())

    print("Extracting structured features...")
    structured_df = pd.DataFrame(
        list(df["dyslexic_sentence"].apply(extract_structured_features))
    )
    feature_names = list(structured_df.columns)
    print(f"Structured features: {len(feature_names)} → {feature_names}")

    print("Vectorizing text (char n-grams 2–5)...")
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),       # wider than before (was 2-4)
        max_features=30000,       # more features
        sublinear_tf=True,        # log-scale TF
        min_df=2,
    )
    X_text       = vectorizer.fit_transform(df["dyslexic_sentence"])
    X_structured = csr_matrix(structured_df.values.astype(np.float32))
    X_combined   = hstack([X_text, X_structured])

    le = LabelEncoder()
    y  = le.fit_transform(df["macro_label"])

    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training Logistic Regression with calibration...")
    base_model = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        C=1.0,
        solver="saga",          # faster for large sparse matrices
        #multi_class="multinomial",
    )

    # Wrap with CalibratedClassifierCV for better probability estimates
    model = CalibratedClassifierCV(base_model, cv=3, method="isotonic")
    model.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    print("Saving artifacts...")
    joblib.dump(model,      os.path.join(ARTIFACT_DIR, "v2_pattern_model.pkl"))
    joblib.dump(vectorizer, os.path.join(ARTIFACT_DIR, "v2_pattern_vectorizer.pkl"))
    joblib.dump(le,         os.path.join(ARTIFACT_DIR, "v2_pattern_label_encoder.pkl"))

    print("✓ Training complete.")


if __name__ == "__main__":
    train()
