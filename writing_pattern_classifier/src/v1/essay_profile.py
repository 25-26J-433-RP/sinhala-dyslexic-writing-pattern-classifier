# essay_profile.py
"""
Essay-level dyslexic writing pattern profiling.

This module aggregates sentence-level dyslexic writing patterns
into dominance-based essay profiles.

NO sentence-level inference or ML logic lives here.
"""

import pandas as pd


# --------------------------------------------------
# Confidence normalization (UI-safe)
# --------------------------------------------------

def normalize_confidence(confidence: float) -> dict:
    """
    Normalize raw confidence into UI-safe representation.
    """

    percent = round(confidence * 100)

    if percent >= 70:
        level = "High"
    elif percent >= 40:
        level = "Medium"
    else:
        level = "Low"

    return {
        "confidence_raw": confidence,
        "confidence_percent": percent,
        "confidence_level": level
    }


# --------------------------------------------------
# Essay-level aggregation
# --------------------------------------------------

def profile_essays(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sentence-level patterns into essay-level dominance profiles.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain:
        - essay_id
        - writing_pattern

    Returns
    -------
    pd.DataFrame
        Essay-level pattern profiles with dominance and confidence.
    """

    if df.empty:
        return pd.DataFrame()

    # Count patterns per essay
    pattern_counts = (
        df
        .groupby("essay_id")["writing_pattern"]
        .value_counts()
        .unstack(fill_value=0)
    )

    essay_summary = pattern_counts.copy()

    # Dominant pattern
    essay_summary["dominant_pattern"] = essay_summary.idxmax(axis=1)

    # Compute dominance metrics
    pattern_columns = pattern_counts.columns
    essay_summary["max_count"] = essay_summary[pattern_columns].max(axis=1)
    essay_summary["total_sentences"] = essay_summary[pattern_columns].sum(axis=1)

    essay_summary["confidence"] = (
        essay_summary["max_count"] / essay_summary["total_sentences"]
    )

    # Dominance strength
    essay_summary["dominance_strength"] = essay_summary["confidence"].apply(
        dominance_strength
    )

    # UI-safe confidence
    essay_summary["confidence_info"] = essay_summary["confidence"].apply(
        normalize_confidence
    )

    return essay_summary.reset_index()


# --------------------------------------------------
# Dominance strength labeling
# --------------------------------------------------

def dominance_strength(confidence: float) -> str:
    """
    Categorize dominance strength based on confidence score.
    """

    if confidence >= 0.6:
        return "Strong"
    elif confidence >= 0.4:
        return "Moderate"
    else:
        return "Weak / Mixed"
