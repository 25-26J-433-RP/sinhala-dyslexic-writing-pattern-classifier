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
    # sentence_results = []   # ❌ Temporarily disabled

    # ------------------------
    # Sentence Level Analysis (COMMENTED)
    # ------------------------
    # for s, probs in zip(sentences, probs_list):
    #     sentence_results.append({"text": s, "probabilities": probs})
    #
    #     for label, value in probs.items():
    #         total_scores[label] = total_scores.get(label, 0.0) + float(value)

    # ✅ Instead, just aggregate without storing sentences
    for probs in probs_list:
        for label, value in probs.items():
            total_scores[label] = total_scores.get(label, 0.0) + float(value)

    total = sum(total_scores.values())
    normalized = {k: float(v / total) for k, v in total_scores.items()}

    sorted_patterns = sorted(normalized.items(), key=lambda x: x[1], reverse=True)
    top_label, top_score = sorted_patterns[0]
    second_label, second_score = sorted_patterns[1]

    # ------------------------
    # SEVERITY LOGIC
    # ------------------------
    if top_score > 0.45:
        severity = "High Pattern Dominance"
    elif top_score > 0.35:
        severity = "Moderate Pattern Presence"
    else:
        severity = "Mild Pattern Indicators"

    # ------------------------
    # DOMINANCE LOGIC
    # ------------------------
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

    # ------------------------
    # RISK SCORE (Simplified)
    # ------------------------
    risk_score = float(normalized[top_label] * 100)

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
        # "sentences": sentence_results,  # ❌ Disabled for now
        "risk_score": risk_score,
        "risk_level": risk_level
    }
