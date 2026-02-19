# Sinhala Dyslexic Writing Pattern Classifier

A research-grade FastAPI microservice for automated Sinhala dyslexia screening and writing pattern analysis.

This system performs:

- Binary essay-level dyslexia screening
- Multi-class writing pattern classification
- Sentence-level explainable diagnostics
- Risk scoring and severity estimation

Designed as a modular microservice architecture for integration into educational platforms, automated essay grading systems, and research environments.

---

## 📚 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Features](#features)
- [API Endpoints](#api-endpoints)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## 🔍 Overview

Sinhala dyslexia remains underrepresented in AI research due to limited linguistic resources. This project provides a machine learning–based screening and analysis system that:

1. Detects whether an essay exhibits dyslexic characteristics (Binary Classification)
2. Identifies dominant writing error patterns (Pattern Classification)
3. Provides sentence-level probabilities for explainability
4. Computes structured risk metrics and severity levels

⚠️ This system is intended for research and educational screening purposes. It is not a medical diagnostic tool.

---

## 🏗 Architecture

```
Essay Input
    │
    ▼
Binary Dyslexia Detector
    │
    ├── If NORMAL → Return screening result
    └── If DYSLEXIC → Writing Pattern Classifier (V2)
                │
                ▼
        Pattern Analysis + Risk Scoring
```

The system follows a modular microservice design, allowing independent development and testing of each component.

---

## 🛠 Tech Stack

- Python 3.11 / 3.12
- FastAPI
- Scikit-learn
- TF-IDF (Character n-grams)
- Logistic Regression
- Hugging Face Datasets
- Joblib
- Docker
- Google Cloud Run

---

## ✨ Features

### 1️⃣ Binary Dyslexia Detector

- Character-level TF-IDF (2–4 character n-grams)
- Logistic Regression classifier
- Sentence-level probability prediction
- Essay-level aggregation rule:
  - If ≥ 1 sentence exceeds threshold → Essay flagged as dyslexic
- Confidence score (mean sentence probability)
- Threshold-based sentence labeling
- Lazy-loaded model for optimized inference

---

### 2️⃣ Writing Pattern Classifier (V2)

Multi-class classification into:

- Grammar
- Phonetic
- Spelling
- Visual

Enhancements:

- Character-level TF-IDF vectorization
- Structured linguistic features:
  - Character length
  - Word count
  - Repeated grapheme count
- Balanced Logistic Regression
- Sentence-level probability breakdown
- Pattern dominance detection logic
- Pattern density computation
- Risk scoring system
- Severity estimation

---

### 3️⃣ Full Essay Analysis

Combined endpoint returns:

```json
{
  "binary": { ... },
  "patterns": { ... }
}
```

This enables seamless integration into front-end applications where the binary result determines whether deeper pattern analysis is required.

---

## 🚀 API Endpoints

### Health Check

```
GET /health
```

Response:

```json
{ "status": "ok" }
```

---

### Binary Essay Screening

```
POST /predict
```

Request:

```json
{
  "essay": "Sinhala essay text..."
}
```

Response Example:

```json
{
  "essay_label": "DYSLEXIC ESSAY",
  "confidence": 0.45,
  "total_sentences": 24,
  "dyslexic_sentences": 3,
  "sentences": [
    {
      "text": "...",
      "probability": 0.87,
      "label": "DYSLEXIC"
    }
  ]
}
```

---

### Writing Pattern Analysis

```
POST /patterns
```

Response Example:

```json
{
  "dominant": "Spelling Pattern Dominant (0.33)",
  "severity": "Mild Pattern Indicators",
  "distribution": {
    "Grammar": 0.19,
    "Phonetic": 0.26,
    "Spelling": 0.32,
    "Visual": 0.22
  },
  "pattern_density": {
    "Spelling": 28.5
  },
  "risk_score": 32.6,
  "risk_level": "Low Writing Pattern Risk"
}
```

---

### Full Analysis

```
POST /analyze
```

Response:

```json
{
  "binary": { ... },
  "patterns": { ... }
}
```

---

## 💻 Installation

### 1. Clone Repository

```bash
git clone https://github.com/<your-username>/sinhala-dyslexic-writing-pattern-classifier.git
cd sinhala-dyslexic-writing-pattern-classifier
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

(Windows)

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶ Usage

Run locally:

```bash
uvicorn app.main:app --reload --port 8002
```

Access Swagger UI:

```
http://localhost:8002/docs
```

---

## 🧠 Model Training

### Binary Dyslexia Model

- Dataset: SPEAK-ASR Sinhala Dyslexia Corrected Dataset
- Character-level TF-IDF (2–4 grams)
- Logistic Regression
- Stratified train/test split
- Probability-based threshold screening

Artifacts saved to:

```
binary_dyslexia_detector/models/
    dyslexia_binary_model.pkl
    tfidf_vectorizer.pkl
```

---

### Writing Pattern Classifier (V2)

- Dataset: akura-sinhala-dyslexia-corrected
- Structured + TF-IDF hybrid features
- Balanced Logistic Regression
- Label normalization pipeline

Artifacts saved to:

```
writing_pattern_classifier/artifacts/
    v2_pattern_model.pkl
    v2_pattern_vectorizer.pkl
    v2_pattern_label_encoder.pkl
```

---

## 🐳 Deployment

### Build Docker Image

```bash
docker build -t yourusername/sinhala-dyslexic-writing-pattern-classifier .
```

### Run Locally with Docker

```bash
docker run -p 8002:8002 yourusername/sinhala-dyslexic-writing-pattern-classifier
```

### Deploy to Google Cloud Run

```bash
gcloud run deploy dyslexic-pattern-detection-service \
  --image docker.io/yourusername/sinhala-dyslexic-writing-pattern-classifier:<tag> \
  --region europe-west1 \
  --allow-unauthenticated
```

---

## 📁 Project Structure

```
app/
 ├── binary_dyslexia_detector/
 │    ├── src/
 │    ├── models/
 │    └── training notebooks
 │
 ├── writing_pattern_classifier/
 │    ├── src/v2/
 │    ├── artifacts/
 │    └── training scripts
 │
 └── main.py
```

---

## 🤝 Contributing

Contributions are welcome.

Steps:

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Open a pull request

Please ensure:

- Code is formatted
- Tests pass
- Documentation is updated

---

## ⚖ License

MIT License
