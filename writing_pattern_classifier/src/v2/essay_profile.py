import re
from .model import predict_batch_sentence_patterns
from .explanations import PATTERN_EXPLANATIONS


# ─────────────────────────────────────────────────────────────────────────────
# Tuning constants
# ─────────────────────────────────────────────────────────────────────────────

MIN_WORDS_FOR_RELIABLE   = 3
# Calibrated to this model's natural confidence range (0.38–0.62).
# threshold=0.44, margin=0.10 produces ~17/33 dyslexic sentences on
# a known dyslexic essay — matching the binary classifier's output.
STRONG_PATTERN_THRESHOLD = 0.44
N_CLASSES        = 4
RANDOM_BASELINE  = 1.0 / N_CLASSES   # 0.25
RELIABLE_MARGIN  = 0.10


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_python(obj):
    """Recursively convert numpy types to native Python for JSON serialization."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _confidence_weight(probs: dict, word_count: int) -> float:
    max_prob = max(probs.values())
    margin = max_prob - RANDOM_BASELINE
    if margin <= 0:
        return 0.0
    margin_score  = min(margin / (1.0 - RANDOM_BASELINE), 1.0)
    length_factor = min(word_count / 8.0, 1.0)
    return 0.65 * margin_score + 0.35 * length_factor


def _is_dyslexic(probs: dict, word_count: int) -> bool:
    if word_count < MIN_WORDS_FOR_RELIABLE:
        return False
    max_prob = max(probs.values())
    margin   = max_prob - RANDOM_BASELINE
    return bool(margin >= RELIABLE_MARGIN and max_prob >= STRONG_PATTERN_THRESHOLD)


def _weighted_distribution(sentence_results: list) -> dict:
    total_scores = {}
    total_weight = 0.0

    for sr in sentence_results:
        probs      = sr["probabilities"]
        word_count = sr.get("word_count", len(sr["text"].split()))
        weight     = _confidence_weight(probs, word_count)
        if weight <= 0:
            continue
        for label, prob in probs.items():
            total_scores[label] = total_scores.get(label, 0.0) + float(prob) * weight
        total_weight += weight

    if total_weight == 0 or not total_scores:
        for sr in sentence_results:
            for label, prob in sr["probabilities"].items():
                total_scores[label] = total_scores.get(label, 0.0) + float(prob)

    total = sum(total_scores.values()) or 1.0
    return {k: float(v / total) for k, v in total_scores.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_full_essay(essay_text: str) -> dict:
    raw = re.split(r"[.\n\u0964]+", essay_text)
    sentences = [s.strip() for s in raw if s.strip()]

    if not sentences:
        return {"error": "No valid sentences detected."}

    prob_list = predict_batch_sentence_patterns(sentences)

    sentence_results        = []
    dyslexic_sentence_count = 0
    reliable_sentence_count = 0
    max_sentence_prob       = 0.0
    weighted_dyslexic_score = 0.0
    total_confidence_weight = 0.0

    for text, probs in zip(sentences, prob_list):
        word_count = len(text.split())
        weight     = _confidence_weight(probs, word_count)
        is_dys     = _is_dyslexic(probs, word_count)
        max_prob   = float(max(probs.values()))

        if max_prob > max_sentence_prob:
            max_sentence_prob = max_prob

        if word_count >= MIN_WORDS_FOR_RELIABLE:
            reliable_sentence_count += 1
            total_confidence_weight += weight
            if is_dys:
                dyslexic_sentence_count += 1
                weighted_dyslexic_score += weight

        sentence_results.append({
            "text":          text,
            "probabilities": {k: float(v) for k, v in probs.items()},
            "word_count":    int(word_count),
            "weight":        float(round(weight, 4)),
            "is_dyslexic":   bool(is_dys),
        })

    total_sentences = len(sentences)
    normalized      = _weighted_distribution(sentence_results)
    sorted_patterns = sorted(normalized.items(), key=lambda x: x[1], reverse=True)
    top_label,    top_score    = sorted_patterns[0]
    second_label, second_score = sorted_patterns[1]

    if top_score > 0.45:
        severity = "High Pattern Dominance"
    elif top_score > 0.35:
        severity = "Moderate Pattern Presence"
    else:
        severity = "Mild Pattern Indicators"

    if (top_score - second_score) < 0.08:
        dominant = (
            f"No clear dominant pattern detected. "
            f"The essay shows overlapping characteristics of {top_label} "
            f"and {second_label} patterns."
        )
        explanation = (
            f"The writing demonstrates mixed indicators, "
            f"with {top_label} ({top_score:.2f}) and "
            f"{second_label} ({second_score:.2f}) appearing closely."
        )
    else:
        dominant    = f"{top_label} Pattern Dominant ({top_score:.2f})"
        explanation = PATTERN_EXPLANATIONS.get(top_label, "")

    pattern_sentence_count    = {label: 0 for label in normalized}
    pattern_sentence_examples = {label: [] for label in normalized}

    for sr in sentence_results:
        for label, prob in sr["probabilities"].items():
            if prob > STRONG_PATTERN_THRESHOLD:
                pattern_sentence_count[label] += 1
                pattern_sentence_examples[label].append(sr["text"])

    pattern_density = {
        label: float(pattern_sentence_count[label] / total_sentences * 100)
        for label in normalized
    }

    if total_confidence_weight > 0:
        weighted_dyslexic_ratio = float(weighted_dyslexic_score / total_confidence_weight)
    else:
        weighted_dyslexic_ratio = 0.0

    dyslexic_ratio = float(
        dyslexic_sentence_count / reliable_sentence_count
        if reliable_sentence_count else 0.0
    )

    base_risk     = weighted_dyslexic_ratio * 100
    pattern_boost = float((top_score - 0.33) * 40) if top_score > 0.40 else 0.0

    if max_sentence_prob > 0.85:
        severity_boost = 8.0
    elif max_sentence_prob > 0.75:
        severity_boost = 4.0
    elif max_sentence_prob > 0.65:
        severity_boost = 2.0
    else:
        severity_boost = 0.0

    strong_pattern_count = sum(1 for d in pattern_density.values() if d > 10)
    multi_pattern_boost  = 5.0 if strong_pattern_count >= 2 else 0.0

    risk_score = float(min(
        base_risk + pattern_boost + severity_boost + multi_pattern_boost, 95.0
    ))

    if risk_score > 65:
        risk_level = "High Writing Pattern Risk"
    elif risk_score > 45:
        risk_level = "Moderate Writing Pattern Risk"
    elif risk_score > 25:
        risk_level = "Low-Moderate Writing Pattern Risk"
    else:
        risk_level = "Low Writing Pattern Risk"

    result = {
        "dominant":    dominant,
        "severity":    severity,
        "explanation": explanation,
        "risk_score":  float(round(risk_score, 2)),
        "risk_level":  risk_level,
        "distribution":              {k: float(v) for k, v in normalized.items()},
        "pattern_sentence_count":    {k: int(v) for k, v in pattern_sentence_count.items()},
        "pattern_sentence_examples": pattern_sentence_examples,
        "pattern_density":           {k: float(v) for k, v in pattern_density.items()},
        "sentences": [
            {
                "text":          sr["text"],
                "probabilities": {k: float(v) for k, v in sr["probabilities"].items()},
                "is_dyslexic":   bool(sr["is_dyslexic"]),
                "weight":        float(sr["weight"]),
            }
            for sr in sentence_results
        ],
        "dyslexic_ratio":           float(round(dyslexic_ratio, 4)),
        "weighted_dyslexic_ratio":  float(round(weighted_dyslexic_ratio, 4)),
        "dyslexic_sentence_count":  int(dyslexic_sentence_count),
        "reliable_sentence_count":  int(reliable_sentence_count),
        "total_sentence_count":     int(total_sentences),
        "max_sentence_probability": float(round(max_sentence_prob, 4)),
    }

    return _to_python(result)
