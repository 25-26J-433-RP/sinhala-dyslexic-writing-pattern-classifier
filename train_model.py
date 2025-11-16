# train_model.py
# Simple baseline model for Sinhala dyslexic error classification

import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# -----------------------------
# 1. Load dataset
# -----------------------------
with open("data/sinhala_dyslexia_samples.json", "r", encoding="utf-8") as f:
    data = json.load(f)

texts, labels = [], []
for sample in data:
    for word, label in zip(sample["tokens"], sample["labels"]):
        texts.append(word)
        labels.append(label)

print(f"Loaded {len(texts)} Sinhala tokens")

# -----------------------------
# 2. Convert to character n-grams
# -----------------------------
vectorizer = CountVectorizer(analyzer="char", ngram_range=(2, 3))
X = vectorizer.fit_transform(texts)

# -----------------------------
# 3. Train simple classifier
# -----------------------------
clf = LogisticRegression(max_iter=500)
clf.fit(X, labels)

# -----------------------------
# 4. Evaluate on same data (for now)
# -----------------------------
preds = clf.predict(X)
print("\nModel performance (training data):")
print(classification_report(labels, preds, digits=3))

# -----------------------------
# 5. Save model and vectorizer
# -----------------------------
joblib.dump(clf, "dyslexia_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model & vectorizer saved!")
