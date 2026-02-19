# Writing Pattern Classifier (V2)

Multi-class Sinhala dyslexic writing pattern analysis module.

This module performs fine-grained classification of dyslexic writing into dominant linguistic error patterns. It is designed as the second-stage analytical component of the overall Sinhala Dyslexia Analysis System.

While the Binary Dyslexia Detector determines whether dyslexic characteristics are present, this module explains _how_ they manifest.

---

# Table of Contents

- Overview
- System Role in Architecture
- Error Taxonomy
- Model Design
- Feature Engineering
- Hybrid Feature Fusion
- Training Pipeline
- Inference Pipeline
- Essay-Level Aggregation Logic
- Dominance & Severity Logic
- Risk Scoring Formula
- Interpretability & Explainability
- Project Structure
- Deployment
- Limitations
- Future Work
- License

---

# Overview

The Writing Pattern Classifier (V2) is a multi-class classification system that identifies dominant dyslexic writing patterns in Sinhala text.

It operates at:

- Sentence-level (pattern probability estimation)
- Essay-level (dominance detection + risk scoring)

Unlike simple spelling checkers, this module detects structured cognitive error categories.

---

# System Role in Architecture

```
Essay Input
    │
    ▼
Binary Dyslexia Detector
    │
    ├── NORMAL → Stop
    └── DYSLEXIC → Writing Pattern Classifier (V2)
                      │
                      ▼
             Pattern Distribution + Risk Analysis
```

This module assumes the essay has already been flagged as dyslexic.

---

# Error Taxonomy (Macro Labels)

Original dataset labels are normalized into 4 macro categories:

- Grammar
- Phonetic
- Spelling
- Visual

Normalization Logic:

```
"phonetic" → Phonetic
"visual", "scrambling", "reversal" → Visual
"grammar", "tense", "case" → Grammar
"spelling" → Spelling
```

All other labels are discarded ("None" removed).

This ensures:

- Controlled label space
- Reduced noise
- Clear interpretability

---

# Model Design

Algorithm: Logistic Regression  
Max Iterations: 3000  
Class Weight: Balanced  
Regularization: Default L2

Why Logistic Regression?

- Produces calibrated probabilities
- Stable for multi-class classification
- Interpretable weight vectors
- Works well with sparse + dense hybrid features
- Lightweight and deployment-friendly

This is a 4-class classification problem.

---

# Feature Engineering

V2 uses a hybrid feature architecture:

### 1️⃣ Character-Level TF-IDF

- Analyzer: char
- N-gram range: (2, 4)
- Max features: 20,000

Captures:

- Orthographic instability
- Grapheme-level distortions
- Character misordering
- Diacritic loss

---

### 2️⃣ Structured Linguistic Features

Extracted via regex grapheme segmentation:

- char_length
- word_count
- repeated_graphemes

Repeated graphemes detect patterns like:

- Letter duplication
- Motor-control repetition artifacts

These features introduce linguistic structure beyond pure n-grams.

---

# Hybrid Feature Fusion

Final feature matrix:

```
X_combined = hstack([TF-IDF sparse matrix, structured dense matrix])
```

This produces:

- High-dimensional sparse lexical space
- Low-dimensional structured numeric features
- Combined representation for richer modeling

---

# Training Pipeline

1. Load dataset:
   SPEAK-ASR/akura-sinhala-dyslexia-corrected

2. Normalize labels into macro categories
3. Remove "None"
4. Extract structured features
5. Fit TF-IDF vectorizer
6. Combine sparse + dense features
7. Encode labels using LabelEncoder
8. Stratified 80/20 split
9. Train balanced Logistic Regression
10. Save artifacts:

```
artifacts/
 ├── v2_pattern_model.pkl
 ├── v2_pattern_vectorizer.pkl
 └── v2_pattern_label_encoder.pkl
```

---

# Inference Pipeline

Sentence-Level:

```
1. Extract structured features
2. Vectorize text
3. hstack features
4. model.predict_proba()
5. Return probability per class
```

Batch Inference:

`predict_batch_sentence_patterns()`  
Optimized to reduce repeated model calls.  
Single matrix transform → single predict call.

Significantly improves latency for long essays.

---

# Essay-Level Aggregation Logic

For an essay:

1. Split into sentences
2. Predict probability distribution per sentence
3. Sum probabilities across sentences
4. Normalize to distribution

```
normalized[label] = total_scores[label] / sum(total_scores)
```

This yields pattern distribution across the entire essay.

---

# Dominance & Severity Logic

Let:

- top_label = highest normalized score
- second_label = second highest

### Dominance Threshold

If:

```
top_score - second_score < 0.05
```

Then:

"No clear dominant pattern"

Otherwise:

"{top_label} Pattern Dominant"

---

### Severity Thresholds

```
if top_score > 0.45:
    High Pattern Dominance
elif top_score > 0.35:
    Moderate Pattern Presence
else:
    Mild Pattern Indicators
```

---

# Strong Pattern Sentence Analysis

Sentence considered strongly affected if:

```
probability > 0.45
```

Metrics computed:

- pattern_sentence_count
- pattern_sentence_examples
- pattern_density (% of essay strongly affected)

---

# Risk Scoring Formula

Risk score integrates:

1. Global distribution strength
2. Sentence-level density

Formula:

```
risk_score =
(
    normalized[top_label] * 0.6 +
    (pattern_sentence_count[top_label] / total_sentences) * 0.4
) * 100
```

Interpretation:

- 60% weight → overall dominance
- 40% weight → localized strong signals

Risk Levels:

```
> 60 → High Writing Pattern Risk
> 40 → Moderate Writing Pattern Risk
else → Low Writing Pattern Risk
```

---

# Interpretability & Explainability

The system provides:

- Full probability distribution
- Sentence-level probabilities
- Example sentences per pattern
- Natural language explanation (PATTERN_EXPLANATIONS)
- Structured diagnostic report (CLI mode)

This ensures transparency and auditability.

---

# Project Structure

```
writing_pattern_classifier/
├── artifacts/
├── src/
│   ├── v2/
│   │   ├── essay_profile.py
│   │   ├── model.py
│   │   ├── preprocessing.py
│   │   ├── pipeline.py
│   │   └── explanations.py
│   ├── router.py
│   └── training/
│       └── train_v2.py
├── test_v2_cli.py
└── README.md
```

---

# Deployment

Exposed via main FastAPI service:

```
POST /patterns
POST /analyze
```

Supports:

- Local execution
- Docker containerization
- Cloud Run deployment

---

# Limitations

- Sentence-level modeling ignores long-range discourse context
- Logistic Regression may not capture nonlinear interactions
- Dataset domain bias possible
- Structured features limited to 3 handcrafted metrics

---

# Future Work

- Context-aware sequence models (Transformer-based)
- Feature importance visualization
- Adaptive risk weighting
- Per-grade calibration
- Fairness-aware evaluation
- Neurodiversity-sensitive evaluation metrics

---

# License

MIT License
