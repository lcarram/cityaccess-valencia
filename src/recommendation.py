import pandas as pd

from .preprocessing import CATEGORIES


def recommend_neighbourhoods(access, preferences, top_n=5):
    feature_map = {
        "healthcare": "healthcare_accessibility",
        "education": "education_accessibility",
        "libraries": "libraries_accessibility",
        "sports": "sports facilities_accessibility",
        "proximity": "proximity",
    }
    weights = pd.Series(preferences, dtype=float)
    weights = weights / (weights.sum() or 1)
    out = access[["neighbourhood"] + list(feature_map.values())].copy()
    for key, col in feature_map.items():
        out[f"{key}_contribution"] = out[col] * weights[key]
    contrib_cols = [f"{k}_contribution" for k in feature_map]
    out["recommendation_score"] = out[contrib_cols].sum(axis=1).round(2)
    out["explanation"] = out.apply(_explain, axis=1)
    return out.sort_values("recommendation_score", ascending=False).head(top_n)


def _explain(row):
    parts = {
        "healthcare": row["healthcare_contribution"],
        "education": row["education_contribution"],
        "libraries": row["libraries_contribution"],
        "sports": row["sports_contribution"],
        "proximity": row["proximity_contribution"],
    }
    best = max(parts, key=parts.get)
    return f"Ranked highly mainly because its {best} accessibility matches the selected preferences."

