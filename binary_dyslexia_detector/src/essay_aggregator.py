"""
Essay-level dyslexia analysis module — Improved v2.

Responsibility:
- Split an essay into sentences
- Apply sentence-level dyslexia detection
- Aggregate results into an essay-level decision

Changes from v1:
  - Weighted aggregation: longer sentences carry more weight
  - Composite score: 60% weighted mean + 40% peak probability
  - Three-tier sentence labeling: NORMAL / BORDERLINE / DYSLEXIC
  - Three-tier essay labeling: NORMAL / BORDERLINE / DYSLEXIC ESSAY
  - New output field: composite_score (transparent, rankable)
  - Minimum sentence length filter: skip fragments < 4 chars
"""

import re
from binary_dyslexia_detector.src.sentence_classifier import predict_sentence


def split_sentences(text: str):
    """
    Splits essay text into sentences.

    Handles: Sinhala punctuation (. ! ?), danda (।), and newlines.
    Filters out fragments shorter than 4 characters.
    Chunks long single-paragraph texts (> 200 chars) at 120-char intervals.
    """
    if not text or not text.strip():
        return []

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    raw_sentences = re.split(r"[.!?।\n]+", text)

    cleaned = [s.strip() for s in raw_sentences if len(s.strip()) >= 4]

    # Chunk long single-paragraph essays
    if len(cleaned) == 1 and len(cleaned[0]) > 200:
        long_text = cleaned[0]
        cleaned = [long_text[i:i+120] for i in range(0, len(long_text), 120)]

    return cleaned


def analyze_essay(essay_text: str, threshold: float = 0.65) -> dict:
    """
    Performs essay-level dyslexia analysis.

    Sentence-level thresholds:
      NORMAL     : prob < 0.50
      BORDERLINE : 0.50 <= prob < threshold (default 0.65)
      DYSLEXIC   : prob >= threshold

    Essay-level composite score:
      composite = 0.6 × weighted_mean_prob + 0.4 × peak_sentence_prob
      where weight = word count of each sentence

    Essay-level decision:
      DYSLEXIC ESSAY  : composite >= 0.55 OR (ratio >= 0.2 AND mean >= 0.5)
      BORDERLINE ESSAY: 0.45 <= composite < 0.55
      NORMAL ESSAY    : composite < 0.45

    Args:
        essay_text (str): Full essay text
        threshold  (float): Per-sentence dyslexia threshold (default 0.65)

    Returns:
        dict: Essay-level result with per-sentence breakdown
    """
    sentences = split_sentences(essay_text)

    if not sentences:
        return {"error": "No valid sentences found."}

    probabilities    = []
    sentence_results = []
    dyslexic_count   = 0
    borderline_count = 0

    for s in sentences:
        prob = predict_sentence(s)
        probabilities.append(prob)

        if prob >= threshold:
            label = "DYSLEXIC"
            dyslexic_count += 1
        elif prob >= 0.50:
            label = "BORDERLINE"
            borderline_count += 1
        else:
            label = "NORMAL"

        sentence_results.append({
            "text":        s,
            "probability": round(float(prob), 3),
            "label":       label
        })

    # ---- Weighted essay aggregation ----
    weights       = [max(len(s.split()), 1) for s in sentences]
    total_weight  = sum(weights)
    weighted_mean = sum(p * w for p, w in zip(probabilities, weights)) / total_weight
    peak_prob     = max(probabilities)

    # Composite score: weighted mean (60%) + peak signal (40%)
    composite_score = 0.6 * weighted_mean + 0.4 * peak_prob
    dyslexic_ratio  = dyslexic_count / len(sentences)

    # ---- Essay-level decision ----
    if composite_score >= 0.60 or (dyslexic_ratio >= 0.25 and weighted_mean >= 0.55):
        essay_label = "DYSLEXIC ESSAY"
    elif composite_score >= 0.57:
        essay_label = "BORDERLINE ESSAY"

    else:
        essay_label = "NORMAL ESSAY"

    return {
        "essay_label":          essay_label,
        "composite_score":      round(composite_score, 3),
        "weighted_mean_prob":   round(weighted_mean, 3),
        "peak_sentence_prob":   round(peak_prob, 3),
        "dyslexic_ratio":       round(dyslexic_ratio, 3),
        "total_sentences":      len(sentences),
        "dyslexic_sentences":   dyslexic_count,
        "borderline_sentences": borderline_count,
        "sentences":            sentence_results
    }
