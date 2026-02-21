import re
from .model import predict_sentence_pattern
from .explanations import PATTERN_EXPLANATIONS


def analyze_full_essay(essay_text):
    """
    CORRECTED VERSION with proper risk scoring
    """
    sentences = re.split(r"[.\n]+", essay_text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return {"error": "No valid sentences detected."}

    total_scores = {}
    sentence_results = []

    # NEW: Track binary classification for risk calculation
    dyslexic_binary_count = 0
    binary_threshold = 0.40  # LOWERED from 0.5
    max_sentence_prob = 0.0

    # ------------------------
    # Sentence Level Analysis
    # ------------------------
    for s in sentences:
        probs = predict_sentence_pattern(s)
        
        # Calculate if sentence is dyslexic (highest prob wins)
        max_prob = max(probs.values())
        max_label = max(probs.items(), key=lambda x: x[1])[0]
        
        # Track maximum probability seen
        if max_prob > max_sentence_prob:
            max_sentence_prob = max_prob
        
        # Binary classification: is this sentence dyslexic?
        # A sentence is dyslexic if its highest pattern probability > threshold
        # AND that pattern is not "Normal" (if you have that class)
        if max_prob > binary_threshold:
            dyslexic_binary_count += 1
        
        sentence_results.append({"text": s, "probabilities": probs})

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

    # ---------------------------------
    # STRONG PATTERN SENTENCE ANALYSIS
    # ---------------------------------
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

    # ========================================
    # NEW CORRECTED RISK SCORE CALCULATION
    # ========================================
    
    # Calculate dyslexic ratio from binary classification
    dyslexic_ratio = dyslexic_binary_count / total_sentences
    
    # Base risk from dyslexic ratio (most important factor)
    base_risk = dyslexic_ratio * 100
    
    # Boost if pattern distribution shows strong dominance
    pattern_boost = 0
    if top_score > 0.35:
        pattern_boost = (top_score - 0.25) * 50  # Up to +25% boost
    
    # Boost if maximum sentence probability is very high (severe errors)
    severity_boost = 0
    if max_sentence_prob > 0.85:
        severity_boost = 10
    elif max_sentence_prob > 0.75:
        severity_boost = 5
    
    # Boost if multiple strong pattern types present
    multi_pattern_boost = 0
    strong_patterns = sum(1 for density in pattern_density.values() if density > 15)
    if strong_patterns >= 2:
        multi_pattern_boost = 5
    
    # Calculate final risk score
    risk_score = base_risk + pattern_boost + severity_boost + multi_pattern_boost
    
    # Cap at 95% (never 100% certain)
    risk_score = min(float(risk_score), 95.0)
    
    # ========================================
    # CORRECTED RISK LEVEL CLASSIFICATION
    # ========================================
    
    if risk_score > 60:
        risk_level = "High Writing Pattern Risk"
    elif risk_score > 40:
        risk_level = "Moderate Writing Pattern Risk"
    elif risk_score > 20:
        risk_level = "Low-Moderate Writing Pattern Risk"
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
        "risk_level": risk_level,
        # NEW: Add these for debugging/transparency
        "dyslexic_ratio": dyslexic_ratio,
        "dyslexic_sentence_count": dyslexic_binary_count,
        "total_sentence_count": total_sentences,
        "max_sentence_probability": max_sentence_prob,
    }
