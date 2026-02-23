import re
from .model import predict_sentence_pattern, predict_batch_sentence_patterns
from .explanations import PATTERN_EXPLANATIONS


# ─────────────────────────────────────────────────────────────────────────────
# Tuning constants
# ─────────────────────────────────────────────────────────────────────────────

# Minimum word count for a sentence to be considered reliable
MIN_WORDS_FOR_RELIABLE = 3

# Threshold above which a single pattern probability counts as "strong"
STRONG_PATTERN_THRESHOLD = 0.50

# Entropy-based threshold: if a sentence's max-prob is only marginally above
# the random baseline (1/n_classes), down-weight its contribution.
N_CLASSES = 4
RANDOM_BASELINE = 1.0 / N_CLASSES           # 0.25 for 4 classes
RELIABLE_MARGIN  = 0.15                      # max_prob must exceed baseline by this


def _sentence_confidence_weight(probs: dict, word_count: int) -> float:
    """
    Return a weight in [0, 1] that reflects how much we trust this sentence's
    prediction.

    Two factors:
      • max_prob margin over random baseline (higher → more trustworthy)
      • word count (longer sentences → more reliable)
    """
    max_prob = max(probs.values())

    # Margin above random-class baseline
    margin = max_prob - RANDOM_BASELINE
    if margin <= 0:
        return 0.0  # no signal at all

    # Normalize margin to [0,1]: margin can go up to (1 - baseline)
    max_possible_margin = 1.0 - RANDOM_BASELINE
    margin_score = min(margin / max_possible_margin, 1.0)

    # Word-length factor: saturates at 8+ words
    length_factor = min(word_count / 8.0, 1.0)

    # Combined weight
    return 0.65 * margin_score + 0.35 * length_factor


def _is_dyslexic_sentence(probs: dict, word_count: int) -> bool:
    """
    Decide whether a sentence exhibits a dyslexic pattern.

    Rules (all must hold):
      1. The top pattern probability is meaningfully above random baseline.
      2. The sentence is long enough to carry signal.
      3. The top probability exceeds STRONG_PATTERN_THRESHOLD.
    """
    max_prob = max(probs.values())
    margin   = max_prob - RANDOM_BASELINE

    # Very short sentences (1-2 words) are not reliable; treat as neutral
    if word_count < MIN_WORDS_FOR_RELIABLE:
        return False

    return margin >= RELIABLE_MARGIN and max_prob >= STRONG_PATTERN_THRESHOLD


def _weighted_distribution(sentence_results: list) -> dict:
    """
    Compute a confidence-weighted pattern distribution across all sentences.

    Instead of a simple sum-then-normalise, each sentence contributes
    proportional to how confident its prediction is.
    """
    total_scores: dict = {}
    total_weight = 0.0

    for sr in sentence_results:
        probs      = sr["probabilities"]
        word_count = sr.get("word_count", len(sr["text"].split()))
        weight     = _sentence_confidence_weight(probs, word_count)

        if weight <= 0:
            continue

        for label, prob in probs.items():
            total_scores[label] = total_scores.get(label, 0.0) + prob * weight

        total_weight += weight

    if total_weight == 0 or not total_scores:
        # Fall back to unweighted
        for sr in sentence_results:
            for label, prob in sr["probabilities"].items():
                total_scores[label] = total_scores.get(label, 0.0) + float(prob)
        total = sum(total_scores.values()) or 1.0
        return {k: v / total for k, v in total_scores.items()}

    total = sum(total_scores.values()) or 1.0
    return {k: float(v / total) for k, v in total_scores.items()}


def analyze_full_essay(essay_text: str) -> dict:
    """
    Improved essay analysis with:
      • Batch sentence prediction (faster)
      • Confidence-weighted distribution
      • Sentence reliability filtering
      • More granular risk scoring
    """
    # ── Sentence splitting ─────────────────────────────────────────────────
    # Split on full stops, Sinhala danda (।), or newlines
    raw_sentences = re.split(r"[.\n।]+", essay_text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]

    if not sentences:
        return {"error": "No valid sentences detected."}

    # ── Batch prediction ───────────────────────────────────────────────────
    prob_list = predict_batch_sentence_patterns(sentences)

    # ── Build sentence result objects ──────────────────────────────────────
    sentence_results = []
    dyslexic_sentence_count   = 0
    reliable_sentence_count   = 0
    max_sentence_prob         = 0.0
    weighted_dyslexic_score   = 0.0  # sum of confidence weights for dyslexic sents
    total_confidence_weight   = 0.0

    for text, probs in zip(sentences, prob_list):
        word_count = len(text.split())
        weight     = _sentence_confidence_weight(probs, word_count)
        is_dys     = _is_dyslexic_sentence(probs, word_count)
        max_prob   = max(probs.values())

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
            "weight":        round(float(weight), 4),
            "is_dyslexic":   bool(is_dys),
        })

    total_sentences = len(sentences)

    # ── Weighted pattern distribution ──────────────────────────────────────
    normalized = _weighted_distribution(sentence_results)
    sorted_patterns = sorted(normalized.items(), key=lambda x: x[1], reverse=True)
    top_label,    top_score    = sorted_patterns[0]
    second_label, second_score = sorted_patterns[1]

    # ── Severity ───────────────────────────────────────────────────────────
    if top_score > 0.45:
        severity = "High Pattern Dominance"
    elif top_score > 0.35:
        severity = "Moderate Pattern Presence"
    else:
        severity = "Mild Pattern Indicators"

    # ── Dominance ──────────────────────────────────────────────────────────
    gap = top_score - second_score
    if gap < 0.08:
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

    # ── Strong-pattern sentence counts ────────────────────────────────────
    pattern_sentence_count    = {label: 0 for label in normalized}
    pattern_sentence_examples = {label: [] for label in normalized}

    for sr in sentence_results:
        for label, prob in sr["probabilities"].items():
            if prob > STRONG_PATTERN_THRESHOLD:
                pattern_sentence_count[label] += 1
                pattern_sentence_examples[label].append(sr["text"])

    pattern_density = {
        label: float((pattern_sentence_count[label] / total_sentences) * 100)
        for label in normalized
    }

    # ── Risk score (improved) ──────────────────────────────────────────────
    #
    # Primary signal: weighted dyslexic ratio
    #   = sum(confidence_weight of dyslexic sentences)
    #     / sum(confidence_weight of all reliable sentences)
    #
    # This is much better than simple binary count because:
    #   • Sentences with high-confidence predictions count more
    #   • Short / ambiguous sentences count less
    #
    if total_confidence_weight > 0:
        weighted_dyslexic_ratio = weighted_dyslexic_score / total_confidence_weight
    else:
        weighted_dyslexic_ratio = 0.0

    # Unweighted ratio (for transparency)
    dyslexic_ratio = (
        dyslexic_sentence_count / reliable_sentence_count
        if reliable_sentence_count else 0.0
    )

    # Base risk from weighted ratio
    base_risk = weighted_dyslexic_ratio * 100

    # Pattern dominance boost: reward strong single-pattern dominance
    pattern_boost = 0.0
    if top_score > 0.40:
        pattern_boost = (top_score - 0.33) * 40   # max +27 pts

    # High-confidence sentence boost
    severity_boost = 0.0
    if max_sentence_prob > 0.85:
        severity_boost = 8.0
    elif max_sentence_prob > 0.75:
        severity_boost = 4.0
    elif max_sentence_prob > 0.65:
        severity_boost = 2.0

    # Multi-pattern boost: if 2+ distinct pattern types have density > 10%
    strong_patterns     = sum(1 for d in pattern_density.values() if d > 10)
    multi_pattern_boost = 5.0 if strong_patterns >= 2 else 0.0

    risk_score = base_risk + pattern_boost + severity_boost + multi_pattern_boost
    risk_score = min(float(risk_score), 95.0)   # cap at 95

    # ── Risk level ─────────────────────────────────────────────────────────
    if risk_score > 65:
        risk_level = "High Writing Pattern Risk"
    elif risk_score > 45:
        risk_level = "Moderate Writing Pattern Risk"
    elif risk_score > 25:
        risk_level = "Low-Moderate Writing Pattern Risk"
    else:
        risk_level = "Low Writing Pattern Risk"

    return {
        # Summary
        "dominant":    dominant,
        "severity":    severity,
        "explanation": explanation,
        "risk_score":  round(risk_score, 2),
        "risk_level":  risk_level,

        # Pattern breakdown
        "distribution":             {k: float(v) for k, v in normalized.items()},
        "pattern_sentence_count":   {k: int(v) for k, v in pattern_sentence_count.items()},
        "pattern_sentence_examples": pattern_sentence_examples,
        "pattern_density":          {k: float(v) for k, v in pattern_density.items()},

        # Sentence-level detail
        "sentences": [
            {
                "text":          sr["text"],
                "probabilities": {k: float(v) for k, v in sr["probabilities"].items()},
                "is_dyslexic":   bool(sr["is_dyslexic"]),
                "weight":        float(sr["weight"]),
            }
            for sr in sentence_results
        ],

        # Diagnostics
        "dyslexic_ratio":            float(round(dyslexic_ratio, 4)),
        "weighted_dyslexic_ratio":   float(round(weighted_dyslexic_ratio, 4)),
        "dyslexic_sentence_count":   int(dyslexic_sentence_count),
        "reliable_sentence_count":   int(reliable_sentence_count),
        "total_sentence_count":      int(total_sentences),
        "max_sentence_probability":  float(round(max_sentence_prob, 4)),
    }
