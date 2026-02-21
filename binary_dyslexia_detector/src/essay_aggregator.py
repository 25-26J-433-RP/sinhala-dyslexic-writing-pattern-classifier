"""
Essay-level dyslexia analysis module.

Responsibility:
- Split an essay into sentences
- Apply sentence-level dyslexia detection
- Aggregate results into an essay-level decision

This module bridges sentence predictions → essay screening.
"""

import re
from binary_dyslexia_detector.src.sentence_classifier import predict_sentence



import sys





def split_sentences(text: str):
    if not text or not text.strip():
        return []

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Split by punctuation, Sinhala danda, or newline
    raw_sentences = re.split(r"[.!?।\n]+", text)

    cleaned = []
    for s in raw_sentences:
        s = s.strip()
        if len(s) >= 3:
            cleaned.append(s)

    # If still only 1 long paragraph, optionally chunk it
    if len(cleaned) == 1 and len(cleaned[0]) > 200:
        long_text = cleaned[0]
        cleaned = [long_text[i:i+120] for i in range(0, len(long_text), 120)]

    return cleaned








def analyze_essay(essay_text: str, threshold: float = 0.65):
    sentences = split_sentences(essay_text)

    if not sentences:
        return {"error": "No valid sentences found."}

    dyslexic_count = 0
    probabilities = []
    sentence_results = []

    for s in sentences:
        prob = predict_sentence(s)
        probabilities.append(prob)

        is_dyslexic = prob >= threshold
        if is_dyslexic:
            dyslexic_count += 1

        sentence_results.append({
            "text": s,
            "probability": round(float(prob), 2),
            "label": "DYSLEXIC" if is_dyslexic else "NORMAL"
        })
    # ---- Essay-level aggregation ----
    ratio = dyslexic_count / len(sentences)
    mean_prob = sum(probabilities) / len(probabilities)

    # Hybrid decision rule
    if ratio >= 0.2 and mean_prob >= 0.5:
        essay_label = "DYSLEXIC ESSAY"
    else:
        essay_label = "NORMAL ESSAY"

    return {
        "essay_label": essay_label,
        "confidence": round(mean_prob, 2),
        "dyslexic_ratio": round(ratio, 2),
        "max_sentence_probability": round(max(probabilities), 2),
        "total_sentences": len(sentences),
        "dyslexic_sentences": dyslexic_count,
        "sentences": sentence_results
    }
