# Binary Dyslexia Detector

Sentence-level and essay-level Sinhala dyslexia screening module.

This module is responsible for detecting dyslexic writing characteristics in Sinhala essays using a binary classification approach. It serves as the **screening pre-filter** in the overall Dyslexia Pattern Analysis System.

⚠️ This system is intended for educational and research screening purposes only. It is not a medical diagnostic tool.

---

# Table of Contents

- Overview
- System Role in Architecture
- Model Design
- Dataset
- Feature Engineering
- Training Pipeline
- Evaluation Results
- Essay-Level Aggregation Logic
- Inference Pipeline
- API Usage
- Project Structure
- Deployment
- Limitations
- License

---

# Overview

The Binary Dyslexia Detector performs:

1. Sentence-level dyslexia probability prediction
2. Essay-level aggregation based on sentence predictions
3. Threshold-based classification
4. Confidence scoring

It is designed as a **screening gate**.  
If an essay is flagged as dyslexic, it proceeds to the Writing Pattern Classifier module for deeper pattern analysis.

---

# System Role in Architecture

```
Essay Input
    │
    ▼
Binary Dyslexia Detector (This Module)
    │
    ├── NORMAL ESSAY → Return result
    └── DYSLEXIC ESSAY → Writing Pattern Classifier
```

This module does NOT perform multi-class pattern detection.  
It only determines whether dyslexic characteristics are present.

---

# Model Design

Algorithm: Logistic Regression  
Regularization: Default L2  
Max Iterations: 1000  
Hyperparameter Tuning: None (default scikit-learn parameters used)

The model was chosen because:

- Produces calibrated probability outputs
- Works well with sparse TF-IDF features
- Computationally lightweight
- Interpretable
- Stable for production deployment

---

# Dataset

Source:

```
SPEAK-ASR/sinhala-dyslexia-corrected-id20percent
```

Structure:

- clean_sentence
- dyslexic_sentence

Binary Dataset Construction:

Each pair was split into:

- Label 0 → Clean sentence
- Label 1 → Dyslexic sentence

Final Dataset Size:

- Total samples: 55,275 clean + 55,275 dyslexic
- Total = 110,550 sentences
- Balanced dataset (1:1 ratio)
- No oversampling
- No synthetic augmentation

Train/Test Split:

- 80% training
- 20% testing
- Stratified split
- Random state = 42

---

# Feature Engineering

Feature Type: Character-level TF-IDF  
Analyzer: "char"  
N-gram range: (2, 4)

Why character-level?

Because dyslexia manifests as:

- Spelling inconsistencies
- Diacritic omission
- Grapheme substitution
- Phonetic distortions

Word-level models fail when words are misspelled.

Character n-grams allow:

- Robust pattern capture
- Language-independent error detection
- Fine-grained orthographic modeling

---

# Training Pipeline

1. Load dataset from Hugging Face
2. Construct balanced binary dataset
3. Apply stratified train/test split
4. Fit TF-IDF vectorizer
5. Train Logistic Regression classifier
6. Evaluate using classification metrics
7. Save:
   - dyslexia_binary_model.pkl
   - tfidf_vectorizer.pkl

Artifacts are stored in:

```
binary_dyslexia_detector/models/
```

---

# Evaluation Results

Sentence-Level Performance:

Accuracy: 0.78  
Macro F1-score: 0.78

Classification Report:

```
              precision    recall  f1-score   support

           0       0.76      0.84      0.79      5528
           1       0.82      0.73      0.77      5527

    accuracy                           0.78     11055
   macro avg       0.79      0.78      0.78     11055
weighted avg       0.79      0.78      0.78     11055
```

Confusion Matrix:

```
[[4619  909]
 [1477 4050]]
```

Interpretation:

- High precision for dyslexic detection (0.82)
- Moderate recall for dyslexic class (0.73)
- Balanced performance across classes
- Model optimized for screening use case

Recall is especially important since missing dyslexic signals is costlier than over-flagging.

---

# Essay-Level Aggregation Logic

Sentence-level threshold (production):

```
threshold = 0.65
```

For each sentence:

- Compute probability of dyslexia
- If probability ≥ 0.65 → sentence labeled dyslexic

Essay-level rule:

```
If at least 1 sentence is dyslexic → Essay labeled "DYSLEXIC ESSAY"
Else → "NORMAL ESSAY"
```

Confidence Score:

```
Mean probability across all sentences
```

Returned fields:

```
{
  "essay_label": "DYSLEXIC ESSAY",
  "confidence": 0.45,
  "total_sentences": 24,
  "dyslexic_sentences": 3,
  "sentences": [...]
}
```

---

# Inference Pipeline

Sentence Classifier:

- Lazy-load model
- Vectorize sentence
- Predict probability

Essay Aggregator:

- Split sentences
- Apply classifier
- Aggregate counts
- Compute confidence
- Return structured response

Model loading is optimized via caching to avoid repeated disk I/O.

---

# API Usage

This module can be used:

1. Independently via demo/app.py
2. As a submodule inside the main FastAPI service

Main Endpoint:

```
POST /predict
```

Request:

```
{
  "essay": "Sinhala essay text..."
}
```

---

# Project Structure

```
binary_dyslexia_detector/
├── src/
│   ├── essay_aggregator.py
│   ├── sentence_classifier.py
│   └── vectorizer.py
├── models/
│   ├── dyslexia_binary_model.pkl
│   └── tfidf_vectorizer.pkl
├── notebooks/
│   └── 00_sinhala-dyslexia-binary-classifier.ipynb
├── demo/
│   └── app.py
├── Dockerfile
└── requirements.txt
```

---

# Deployment

Local:

```
uvicorn demo.app:app --reload
```

Docker:

```
docker build -t binary-dyslexia-detector .
docker run -p 8000:8000 binary-dyslexia-detector
```

Production deployment occurs via the main FastAPI service.

---

# Limitations

- Sentence-based detection may miss contextual dyslexia patterns
- Threshold rule may over-flag essays with isolated spelling errors
- No deep neural language modeling
- Limited to dataset domain characteristics

Future Improvements:

- Threshold optimization via ROC analysis
- Calibrated probability adjustment
- Ensemble modeling
- Context-aware sequence modeling

---

# License

MIT License
