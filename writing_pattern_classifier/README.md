# Dyslexic Writing-Pattern Classifier (Sinhala)

This module implements an **interpretable, rule-based dyslexic writing-pattern classifier** for Sinhala text.

Unlike traditional machine-learning classifiers, this component focuses on **pattern inference and explainability**, rather than predictive accuracy.  
It is designed to analyze _how_ dyslexic writing manifests, not merely _whether_ dyslexia is present.

---

## Purpose

- Identify **dominant dyslexic writing patterns** in Sinhala text
- Provide **explainable, linguistically grounded analysis**
- Support educational and research-oriented dyslexia-aware systems

This module is executed **only after** an essay has been identified as dyslexic by the Binary Dyslexia Detector.

---

## Core Design Principle

> Dyslexia is expressed through **consistent patterns of surface-level writing errors**, not isolated mistakes.

Therefore, this classifier infers patterns using **rule-based dominance of error signals**, rather than supervised learning.

---

## Writing Patterns Identified

The system currently identifies the following dyslexic writing patterns:

- **Orthographic Instability**  
  Frequent character omissions, additions, or diacritic loss

- **Phonetic Confusion**  
  Character substitutions reflecting phonetic similarity

- **Mixed Dyslexic Pattern**  
  Co-occurrence of multiple dominant error types

- **No Dominant Pattern**  
  Absence of consistent dyslexic error behavior

- **Word Boundary Confusion** (when applicable)  
  Spacing and word segmentation errors

These patterns are derived from dyslexia-related literature and adapted for Sinhala writing.

---

## Processing Pipeline

### 1. Sentence-Level Analysis

For each sentence:

- Clean and dyslexic versions are compared
- Surface error features are extracted:
  - Character addition
  - Character omission
  - Character substitution
  - Diacritic loss
  - Spacing issues
- A **rule-based inference engine** assigns a sentence-level writing pattern

### 2. Essay-Level Aggregation

Because the dataset does not provide explicit essay boundaries:

- Essays are approximated using **fixed-size sentence windows** (pseudo-essays)
- Sentence-level patterns are aggregated per essay

### 3. Dominant Pattern Classification

For each essay:

- The most frequent pattern is selected as the **dominant pattern**
- A **confidence score** is computed as:

\[
Confidence = \frac{\text{Number of sentences supporting dominant pattern}}
{\text{Total number of sentences in essay}}
\]

- Dominance strength is categorized as:
  - Strong Dominance
  - Moderate Dominance
  - Weak / Mixed

---

## Outputs

For each essay, the classifier produces:

- Dominant dyslexic writing pattern
- Pattern dominance confidence
- Dominance strength label
- Sentence-level pattern breakdown (for explainability)

### Example Output

```json
{
  "dominant_pattern": "Orthographic Instability",
  "confidence": 0.6,
  "dominance_strength": "Strong Dominance"
}

---

## Evaluation Strategy

This component does not use supervised evaluation metrics such as accuracy or F1-score.

Reason:

- Essay-level pattern labels are inferred, not manually annotated

- Reporting accuracy would result in label leakage

Instead, evaluation is performed using:

- Pattern distribution analysis

- Confidence distribution statistics

- Qualitative case studies with sentence-level evidence

This approach aligns with best practices in dyslexia-related linguistic analysis.

## Notebooks

notebooks/
├── 01_surface_feature_extraction_and_pattern_inference_v3.ipynb
└── 02_essay_level_dyslexic_pattern_profiling.ipynb

These notebooks document the full development and validation process.

## Limitations

Essay boundaries are approximated using fixed-size sentence windows

The system does not perform clinical diagnosis

Pattern definitions may evolve with expert validation

## Role in the Overall System

(Binary Dyslexia Detector)
          ↓
Dyslexic Essay
          ↓
Writing-Pattern Classifier
          ↓
Pattern Profile + Confidence

## Disclaimer

This module is intended for research and educational purposes only and should not be used for clinical diagnosis.

Generated CSV artifacts are intentionally excluded from version control and can be reproduced by executing the notebooks or pipeline.
```
