import json
from pathlib import Path

# --------------------------------------------------
# Load ML-learned feature weights (OFFLINE artifact)
# --------------------------------------------------

_ARTIFACT_PATH = (
    Path(__file__).resolve().parent.parent
    / "artifacts"
    / "ml_learned_feature_weights.json"
)

with open(_ARTIFACT_PATH, "r", encoding="utf-8") as f:
    LEARNED_WEIGHTS = json.load(f)


# --------------------------------------------------
# Pattern explanations
# --------------------------------------------------

PATTERN_EXPLANATIONS = {
    "Orthographic Instability": (
        "Frequent character-level writing instability was detected, "
        "such as omissions, repetitions, or diacritic inconsistencies."
    ),
    "Phonetic Confusion": (
        "Phonetic-level confusion was observed due to substitutions "
        "between phonetically similar characters."
    ),
    "Word Boundary Confusion": (
        "Word boundary issues were detected, including incorrect spacing "
        "or merging/splitting of words."
    ),
    "Mixed Dyslexic Pattern": (
        "Multiple dyslexic writing behaviors were observed without a single "
        "clearly dominant pattern."
    ),
    "No Dominant Pattern": (
        "No strong dyslexic writing pattern was detected."
    ),
}


# --------------------------------------------------
# Scoring + inference
# --------------------------------------------------

def compute_weighted_scores(features: dict) -> dict:
    scores = {}

    for pattern, weights in LEARNED_WEIGHTS.items():
        score = 0.0
        for fname, weight in weights.items():
            value = int(features.get(fname, 0))
            score += value * weight
        scores[pattern] = score

    return scores


def infer_pattern(
    features: dict,
    min_score: float = 0.5,
    dominance_margin: float = 0.3
) -> str:
    scores = compute_weighted_scores(features)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top, top_score = ranked[0]
    second, second_score = ranked[1]

    if top_score < min_score:
        return "No Dominant Pattern"

    if (top_score - second_score) >= dominance_margin:
        return top

    return "Mixed Dyslexic Pattern"


def explain_pattern(pattern: str, scores: dict | None = None) -> str:
    explanation = PATTERN_EXPLANATIONS.get(pattern, "")

    if pattern == "Mixed Dyslexic Pattern" and scores:
        contributing = [
            p for p, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)
            if s > 0
        ][:2]

        if contributing:
            explanation += " Dominant contributing patterns: "
            explanation += ", ".join(contributing) + "."

    return explanation
