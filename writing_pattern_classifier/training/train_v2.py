import os
import pandas as pd
import numpy as np
import regex
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from scipy.sparse import hstack
import joblib


import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

os.makedirs(ARTIFACT_DIR, exist_ok=True)


def normalize_label(label):
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


def extract_structured_features(text):
    features = {}

    features["char_length"] = len(text)
    features["word_count"] = len(text.split())

    graphemes = regex.findall(r"\X", text)

    repeated = sum(
        1 for i in range(1, len(graphemes))
        if graphemes[i] == graphemes[i - 1]
    )

    features["repeated_graphemes"] = repeated

    return features


def train():
    print("Loading dataset...")
    dataset = load_dataset("SPEAK-ASR/akura-sinhala-dyslexia-corrected")
    df = dataset["train"].to_pandas()

    df["macro_label"] = df["error_type"].apply(normalize_label)
    df = df[df["macro_label"] != "None"]

    print("Extracting structured features...")
    structured_df = pd.DataFrame(
        list(df["dyslexic_sentence"].apply(extract_structured_features))
    )

    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        max_features=20000
    )

    X_text = vectorizer.fit_transform(df["dyslexic_sentence"])
    X_structured = structured_df.values
    X_combined = hstack([X_text, X_structured])

    le = LabelEncoder()
    y = le.fit_transform(df["macro_label"])

    X_train, X_test, y_train, y_test = train_test_split(
        X_combined,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced"
    )

    print("Training model...")
    model.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = model.predict(X_test)

    print(classification_report(
        y_test,
        y_pred,
        target_names=le.classes_
    ))

    print("Saving artifacts...")
    joblib.dump(model, os.path.join(ARTIFACT_DIR, "v2_pattern_model.pkl"))
    joblib.dump(vectorizer, os.path.join(ARTIFACT_DIR, "v2_pattern_vectorizer.pkl"))
    joblib.dump(le, os.path.join(ARTIFACT_DIR, "v2_pattern_label_encoder.pkl"))

    print("Training complete.")


if __name__ == "__main__":
    train()
