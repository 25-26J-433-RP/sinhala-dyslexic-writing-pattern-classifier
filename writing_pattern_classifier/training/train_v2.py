"""
train_v2_improved.py
────────────────────
Improvements over previous version:
  1. Uses BOTH datasets combined (23k + 27k = ~50k rows)
  2. Adds "Normal" class from clean_sentence column
  3. SMOTE oversampling to fix Visual class imbalance
  4. StandardScaler on structured features so they actually matter
  5. LightGBM classifier (F1 target: 65-70% vs previous 55%)
  6. Saves scaler artifact alongside model

Run from project root:
  python -m training.train_v2_improved

Install extras first:
  pip install lightgbm imbalanced-learn
"""

import os
import pandas as pd
import numpy as np
import regex
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
from scipy.sparse import hstack, csr_matrix
import joblib

try:
    from lightgbm import LGBMClassifier
    USE_LGBM = True
    print("✓ LightGBM available")
except ImportError:
    from sklearn.linear_model import LogisticRegression
    USE_LGBM = False
    print("⚠ LightGBM not found — using LogisticRegression (pip install lightgbm)")

try:
    from imblearn.over_sampling import SMOTE
    USE_SMOTE = True
    print("✓ SMOTE available")
except ImportError:
    USE_SMOTE = False
    print("⚠ SMOTE not found — skipping (pip install imbalanced-learn)")

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# ── Sinhala Unicode ────────────────────────────────────────────────────────
SINHALA_RANGE            = (0x0D80, 0x0DFF)
SINHALA_VOWEL_DIACRITICS = set(chr(c) for c in range(0x0DCF, 0x0DE0))
AL_LAKUNA                = "\u0DCA"
SINHALA_INDEPENDENT_VOWELS = set(chr(c) for c in range(0x0D85, 0x0D97))
SINHALA_CONSONANTS       = set(chr(c) for c in range(0x0D9A, 0x0DC7))
SIMILAR_CONSONANT_PAIRS  = [
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
    return [g for g in regex.findall(r"\X", text)
            if any(SINHALA_RANGE[0] <= ord(c) <= SINHALA_RANGE[1] for c in g)]


def extract_structured_features(text: str) -> dict:
    f = {}
    words = text.split()
    all_chars = list(text)
    all_clusters = regex.findall(r"\X", text)
    sinh = _sinhala_graphemes(text)
    nc, ns = len(all_clusters), len(sinh)

    f["char_length"]            = len(text)
    f["word_count"]             = len(words)
    f["avg_word_length"]        = sum(len(w) for w in words) / len(words) if words else 0.0
    f["grapheme_count"]         = nc
    f["sinhala_grapheme_count"] = ns
    f["sinhala_grapheme_ratio"] = ns / nc if nc else 0.0

    vd = sum(1 for c in all_chars if c in SINHALA_VOWEL_DIACRITICS)
    iv = sum(1 for c in all_chars if c in SINHALA_INDEPENDENT_VOWELS)
    f["vowel_diacritic_count"]   = vd
    f["independent_vowel_count"] = iv
    f["vowel_diacritic_ratio"]   = vd / ns if ns else 0.0

    sv = sum(1 for c in all_chars if c in SHORT_VOWELS)
    lv = sum(1 for c in all_chars if c in LONG_VOWELS)
    tv = sv + lv
    f["short_vowel_count"]      = sv
    f["long_vowel_count"]       = lv
    f["vowel_length_imbalance"] = abs(sv - lv) / tv if tv else 0.0

    cc = [c for c in all_chars if c in SINHALA_CONSONANTS]
    ncc = len(cc)
    conf = sum(1 for c in cc if c in SIMILAR_CONSONANT_SET)
    f["consonant_count"]            = ncc
    f["consonant_ratio"]            = ncc / ns if ns else 0.0
    f["confusable_consonant_count"] = conf
    f["confusable_consonant_ratio"] = conf / ncc if ncc else 0.0

    al = text.count(AL_LAKUNA)
    f["al_lakuna_count"]       = al
    f["al_lakuna_ratio"]       = al / ns if ns else 0.0
    f["consecutive_al_lakuna"] = sum(
        1 for i in range(len(all_chars)-1)
        if all_chars[i] == AL_LAKUNA and all_chars[i+1] == AL_LAKUNA
    )

    f["repeated_graphemes"] = sum(1 for i in range(1, nc) if all_clusters[i] == all_clusters[i-1])
    f["repeated_chars"]     = sum(1 for i in range(1, len(all_chars)) if all_chars[i] == all_chars[i-1])
    f["repeated_words"]     = sum(1 for i in range(1, len(words)) if words[i] == words[i-1])

    bare = sum(
        1 for w in words
        if (wc := _sinhala_graphemes(w)) and
        any(c in SINHALA_CONSONANTS for c in wc[-1]) and
        not any(c in SINHALA_VOWEL_DIACRITICS for c in wc[-1])
    )
    f["bare_consonant_endings"]      = bare
    f["bare_consonant_ending_ratio"] = bare / len(words) if words else 0.0
    f["vowel_per_word"]     = vd / len(words) if words else 0.0
    f["consonant_per_word"] = ncc / len(words) if words else 0.0
    return f


def normalize_label(label: str) -> str:
    label = str(label).lower().strip()
    if "phonetic" in label or "confusion" in label:
        return "Phonetic"
    if "visual" in label or "scrambling" in label or "reversal" in label:
        return "Visual"
    if "grammar" in label or "tense" in label or "case" in label or "spoken" in label:
        return "Grammar"
    if "spelling" in label:
        return "Spelling"
    return "None"


def train():
    # ── Load from cache if available ───────────────────────────────────────
    cache_path = os.path.join(BASE_DIR, "training", "cached_dataset.parquet")
    if os.path.exists(cache_path):
        print(f"Loading from cache: {cache_path}")
        df_final = pd.read_parquet(cache_path)
        print(f"Cached dataset: {len(df_final)} rows")
        print(df_final["macro_label"].value_counts())
    else:
        # ── Load both HuggingFace datasets ─────────────────────────────────
        print("Downloading datasets from HuggingFace...")
        ds1 = load_dataset("SPEAK-ASR/akura-sinhala-dyslexia-corrected",      split="train")
        ds2 = load_dataset("SPEAK-PP/sinhala-dyslexia-corrected-id20percent",  split="train")

        df1, df2 = ds1.to_pandas(), ds2.to_pandas()
        print(f"Dataset 1: {len(df1)} rows | Dataset 2: {len(df2)} rows")

        shared = ["clean_sentence", "dyslexic_sentence", "error_type"]
        df1 = df1[[c for c in shared if c in df1.columns]]
        df2 = df2[[c for c in shared if c in df2.columns]]

        df_combined = pd.concat([df1, df2], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=["dyslexic_sentence"])

        # Remove HTML/JSON junk rows
        df_combined = df_combined[
            ~df_combined["dyslexic_sentence"].str.contains(
                r"<.*?>|\{.*?\}", regex=True, na=False
            )
        ]
        print(f"Combined + cleaned: {len(df_combined)} rows")

        df_combined["macro_label"] = df_combined["error_type"].apply(normalize_label)

        # ── Add Normal class ───────────────────────────────────────────────
        normal_df = (df_combined[["clean_sentence"]]
                     .rename(columns={"clean_sentence": "dyslexic_sentence"})
                     .copy())
        normal_df["macro_label"] = "Normal"
        normal_df = normal_df.dropna(subset=["dyslexic_sentence"])
        normal_df = normal_df[normal_df["dyslexic_sentence"].str.len() > 5]
        normal_df = normal_df.sample(n=min(3000, len(normal_df)), random_state=42)
        print(f"Normal class: {len(normal_df)} samples")

        df_dyslexic = (df_combined[df_combined["macro_label"] != "None"]
                       [["dyslexic_sentence", "macro_label"]])
        df_final = pd.concat([df_dyslexic, normal_df], ignore_index=True)
        df_final = df_final.dropna(subset=["dyslexic_sentence"]).reset_index(drop=True)

        print(f"\nFinal dataset: {len(df_final)} rows")
        print(df_final["macro_label"].value_counts())

        # Cache for next run
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df_final.to_parquet(cache_path, index=False)
        print(f"✓ Cached to {cache_path}")

    # ── Features ───────────────────────────────────────────────────────────
    print("\nExtracting structured features...")
    structured_df = pd.DataFrame(
        list(df_final["dyslexic_sentence"].apply(extract_structured_features))
    )
    print(f"Structured features: {len(structured_df.columns)}")

    print("Scaling structured features...")
    scaler = StandardScaler()
    X_structured_scaled = scaler.fit_transform(
        structured_df.values.astype(np.float32)
    )

    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(2, 5),
        max_features=30000, sublinear_tf=True, min_df=2,
    )
    X_text     = vectorizer.fit_transform(df_final["dyslexic_sentence"])
    X_combined = hstack([X_text, csr_matrix(X_structured_scaled)])

    le = LabelEncoder()
    y  = le.fit_transform(df_final["macro_label"])
    print(f"Classes: {list(le.classes_)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── SMOTE ──────────────────────────────────────────────────────────────
    if USE_SMOTE:
        print("\nApplying SMOTE...")
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        unique, counts = np.unique(y_train, return_counts=True)
        for cls, cnt in zip(le.classes_, counts):
            print(f"  {cls}: {cnt}")

    # ── Train ──────────────────────────────────────────────────────────────
    if USE_LGBM:
        print("\nTraining LightGBM...")
        base = LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=63,
            class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
        )
    else:
        print("\nTraining LogisticRegression...")
        base = LogisticRegression(
            max_iter=5000, class_weight="balanced",
            C=1.0, solver="saga", 
            # multi_class="multinomial",
        )

    model = CalibratedClassifierCV(base, cv=3, method="isotonic")
    model.fit(X_train, y_train)

    # ── Evaluate ───────────────────────────────────────────────────────────
    print("\nTest set results:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # ── Save ───────────────────────────────────────────────────────────────
    print("Saving artifacts...")
    joblib.dump(model,      os.path.join(ARTIFACT_DIR, "v2_pattern_model.pkl"))
    joblib.dump(vectorizer, os.path.join(ARTIFACT_DIR, "v2_pattern_vectorizer.pkl"))
    joblib.dump(le,         os.path.join(ARTIFACT_DIR, "v2_pattern_label_encoder.pkl"))
    joblib.dump(scaler,     os.path.join(ARTIFACT_DIR, "v2_pattern_scaler.pkl"))
    print("✓ Done.")


if __name__ == "__main__":
    train()
