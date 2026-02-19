import re
from .model import predict_batch_sentence_patterns
from .explanations import PATTERN_EXPLANATIONS


def analyze_full_essay(essay_text):

    sentences = re.split(r"[.\n]+", essay_text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return {"error": "No valid sentences detected."}

    # 🔥 ONE BATCH CALL
    probs_list = predict_batch_sentence_patterns(sentences)

    total_scores = {}
    sentence_results = []

    # Aggregate + store sentence details
    for s, probs in zip(sentences, probs_list):
        sentence_results.append({
            "text": s,
            "probabilities": {k: float(v) for k, v in probs.items()}
        })

        for label, value in probs.items():
            total_scores[label] = total_scores.get(label, 0.0) + float(value)

    total = sum(total_scores.values())
    normalized = {k: float(v / total) for k, v in total_scores.items()}

    sorted_patterns = sorted(normalized.items(), key=lambda x: x[1], reverse=True)
    top_label, top_score = sorted_patterns[0]
    second_label, second_score = sorted_patterns[1]

    # Severity
    if top_score > 0.45:
        severity = "High Pattern Dominance"
    elif top_score > 0.35:
        severity = "Moderate Pattern Presence"
    else:
        severity = "Mild Pattern Indicators"

    # Dominance
    if top_score - second_score < 0.05:
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
        dominant = f"{top_label} Pattern Dominant ({top_score:.2f})"
        explanation = PATTERN_EXPLANATIONS.get(top_label, "")

    # -------------------------
    # Strong Pattern Sentences
    # -------------------------
    strong_threshold = 0.45
    pattern_sentence_count = {label: 0 for label in normalized.keys()}
    pattern_sentence_examples = {label: [] for label in normalized.keys()}

    for sentence in sentence_results:
        for label, prob in sentence["probabilities"].items():
            if prob > strong_threshold:
                pattern_sentence_count[label] += 1
                pattern_sentence_examples[label].append(sentence["text"])

    total_sentences = len(sentences)

    pattern_density = {
        label: float((pattern_sentence_count[label] / total_sentences) * 100)
        for label in normalized.keys()
    }

    # -------------------------
    # Risk Score (Original)
    # -------------------------
    risk_score = (
        normalized[top_label] * 0.6 +
        (pattern_sentence_count[top_label] / total_sentences) * 0.4
    ) * 100

    risk_score = float(risk_score)

    if risk_score > 60:
        risk_level = "High Writing Pattern Risk"
    elif risk_score > 40:
        risk_level = "Moderate Writing Pattern Risk"
    else:
        risk_level = "Low Writing Pattern Risk"

    return {
        "dominant": dominant,
        "severity": severity,
        "explanation": explanation,
        "distribution": normalized,
        "sentences": sentence_results,
        "pattern_sentence_count": pattern_sentence_count,
        "pattern_sentence_examples": pattern_sentence_examples,
        "pattern_density": pattern_density,
        "risk_score": risk_score,
        "risk_level": risk_level
    }
