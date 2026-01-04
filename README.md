# Sinhala Dyslexic Writing-Pattern Classifier — Training Repository

This repository contains the dataset, preprocessing pipeline, training scripts, and models for the Sinhala Dyslexic Writing-Pattern Classifier.  
It is part of the Automated Sinhala Essay Grading System.

## Project Objectives

### 1. Binary Dyslexia Detection

Predict whether a Sinhala essay is written by a dyslexic student.

### 2. Multi-Label Error Pattern Classification

Identify specific dyslexic writing error types such as:

- missing diacritics
- merged words
- letter omissions
- spacing issues
- irregular structures

These trained models are later used inside the Dyslexic Pattern Detection Service (FastAPI backend).

---

## Allowed Error Tags (Final List)

The classifier uses the following standardized error tags:

spelling_error  
missing_diacritic  
extra_diacritic  
letter_omission  
letter_addition  
order_swap  
merged_words  
fragmented_words  
spacing_issue  
repeated_letters  
irregular_structure

Defined in: src/utils/error_tags.py  
Schema: data/schemas/error_tags.json

---

## Repository Structure

DYSLEXIC-PATTERN-TRAINING/
│
├── data/
│ ├── raw/
│ ├── processed/
│ └── schemas/
│ └── error_tags.json
│
├── models/
│ ├── binary/
│ ├── patterns/
│ └── tokenizer/
│
├── notebooks/
│ ├── 01_binary_classifier_training.ipynb
│ ├── 02_pattern_classifier_training.ipynb
│ ├── 04_error_tag_cleaning.ipynb
│
│
├── src/
│ ├── utils/
│ │ ├── error_tags.py
│ │ ├── dataset_cleaner.py
│ │ ├── multilabel_converter.py
│ │ ├── tokenizer_loader.py
│ │ └── **init**.py
│ │
│ ├── training/
│ │ ├── train_binary.py
│ │ ├── train_patterns.py
│ │ └── **init**.py
│ │
│ ├── prediction/
│ │ ├── predict_binary.py
│ │ ├── predict_patterns.py
│ │ └── **init**.py
│ │
│ ├── evaluation/
│ │ ├── evaluate_binary.py
│ │ ├── evaluate_patterns.py
│ │ └── **init**.py
│ │
│ └── **init**.py
│
├── sample_essay.txt
├── requirements.txt
├── LICENSE
└── README.md

---

## Installation

pip install -r requirements.txt

---

## Training the Models

### Train Binary Dyslexia Classifier

python src/training/train_binary.py

### Train Multi-Label Error Pattern Classifier

python src/training/train_patterns.py

---

## Running Predictions

### Binary Dyslexia Prediction

python src/prediction/predict_binary.py sample_essay.txt

### Multi-Label Error Pattern Prediction

python src/prediction/predict_patterns.py sample_essay.txt

---

## Dataset Cleaning

python src/utils/dataset_cleaner.py

---

## Pipeline Overview

1. Dataset cleaning
2. Error-tag validation
3. Multi-label encoding
4. Tokenization (XLM-R)
5. Binary + multi-label training
6. Export to .pt model files
