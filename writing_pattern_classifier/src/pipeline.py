from typing import List, Dict
import pandas as pd

from .proxy_reference import generate_proxy_reference
from .feature_extraction import extract_surface_features_runtime
from .pattern_rules import (
    compute_weighted_scores,
    infer_pattern,
    explain_pattern
)
from .essay_profile import profile_essays


def analyze_sentence(raw_sentence: str) -> Dict:
    proxy = generate_proxy_reference(raw_sentence)
    features = extract_surface_features_runtime(raw_sentence, proxy)

    scores = compute_weighted_scores(features)
    pattern = infer_pattern(features)
    explanation = explain_pattern(pattern, scores)

    return {
        "raw_sentence": raw_sentence,
        "proxy_reference": proxy,
        "features": features,
        "writing_pattern": pattern,
        "explanation": explanation
    }


def analyze_essay(raw_sentences: List[str], essay_id: int = 0) -> Dict:
    sentence_results = []

    for s in raw_sentences:
        if isinstance(s, str) and s.strip():
            r = analyze_sentence(s)
            r["essay_id"] = essay_id
            sentence_results.append(r)

    sentence_df = pd.DataFrame(sentence_results)
    essay_profile = profile_essays(sentence_df)

    return {
        "essay_id": essay_id,
        "sentence_analysis": sentence_results,
        "essay_profile": essay_profile
    }
