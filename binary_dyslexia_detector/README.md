# Binary Dyslexia Detector (Sinhala)

This module implements a **binary dyslexia screening model** for Sinhala text.  
Its purpose is to determine whether a given Sinhala essay exhibits **dyslexic characteristics** before further pattern-level analysis is performed.

This component acts as the **first-stage filter** in the overall system.

---

## Purpose

- Identify whether an essay is **DYSLEXIC** or **NORMAL**
- Operate as a **screening mechanism**
- Forward only dyslexic essays to the writing-pattern classifier

This separation avoids unnecessary pattern analysis on non-dyslexic writing.

---

## Model Overview

- **Input level:** Sentence-level Sinhala text
- **Feature extraction:** Character-level TF-IDF (2–4 grams)
- **Classifier:** Logistic Regression
- **Aggregation:** Sentence-level predictions aggregated to essay-level decision

---

## Outputs

For a given essay, the detector produces:

- Essay label: `NORMAL ESSAY` or `DYSLEXIC ESSAY`
- Total number of sentences
- Number of dyslexic sentences
- Sentence-level predictions with probabilities

### Example Output

```json
{
  "essay_label": "DYSLEXIC ESSAY",
  "total_sentences": 3,
  "dyslexic_sentences": 1,
  "sentence_analysis": [
    {
      "sentence": "විවේක කලයෙදි මිතුරන් සමග කතාකර ගිය",
      "probability": 0.84,
      "label": "DYSLEXIC"
    }
  ]
}
```

## Evaluation

This component is evaluated using standard supervised learning metrics:

- Accuracy

- Precision

- Recall

- F1-score

- Confusion Matrix

Summary (Sentence-level)

- Accuracy ≈ 0.78

- Balanced precision and recall across classes

These metrics validate the model’s effectiveness as a screening detector, not a diagnostic tool.

## Model Files

Pre-trained models are stored in:

models/
├── dyslexia_binary_model.pkl
└── tfidf_vectorizer.pkl

The training notebook is not included; the model is treated as a pre-trained artifact within this system.

## Demo Application

A web-based demo (Gradio / Hugging Face) is available under:
demo/
└── app.py

This demo allows interactive testing with Sinhala essays.

## Limitations

- This component does not identify specific dyslexic writing patterns

- It serves only as a binary screening step

- False positives and false negatives are possible

For detailed pattern analysis, see the Writing-Pattern Classifier module.

## Role in the Overall System

Essay
↓
Binary Dyslexia Detector
↓
(if dyslexic)
↓
Writing-Pattern Classifier

## Disclaimer

This component is intended for research and educational use only and does not provide clinical diagnosis.

---
