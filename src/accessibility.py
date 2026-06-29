import numpy as np
import pandas as pd

from .preprocessing import CATEGORIES


def minmax(series, invert=False):
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    s = s.fillna(s.median() if s.notna().any() else 0)
    if s.max() == s.min():
        out = pd.Series(0.5, index=s.index)
    else:
        out = (s - s.min()) / (s.max() - s.min())
    return 1 - out if invert else out


def compute_accessibility(metrics, weights):
    out = metrics.copy()
    norm_cols = {}
    norm_cols["services_per_10k_norm"] = minmax(out["services_per_10k"])
    norm_cols["nearest_norm"] = minmax(out["mean_nearest_m"], invert=True)
    norm_cols["nearby_norm"] = minmax(out["nearby_services"])
    norm_cols["diversity_norm"] = minmax(out["service_diversity"])
    norm = pd.DataFrame(norm_cols)
    total_w = sum(weights.values()) or 1
    score = sum(norm[col] * weights[key] for key, col in {
        "services_per_10k": "services_per_10k_norm",
        "nearest_distance": "nearest_norm",
        "nearby_services": "nearby_norm",
        "diversity": "diversity_norm",
    }.items()) / total_w
    out = pd.concat([out, norm], axis=1)
    out["accessibility_score"] = (score * 100).round(2)
    for cat in CATEGORIES:
        cat_norm = pd.DataFrame(
            {
                "per_10k": minmax(out[f"{cat}_per_10k"]),
                "nearest": minmax(out[f"{cat}_nearest_m"], invert=True),
                "nearby": minmax(out[f"{cat}_within_radius"]),
                "density": minmax(out[f"{cat}_density_km2"]),
            }
        )
        out[f"{cat}_accessibility"] = (cat_norm.mean(axis=1) * 100).round(2)
    out["proximity"] = (minmax(out["mean_nearest_m"], invert=True) * 100).round(2)
    return out


def priority_score(access):
    out = access.copy()
    risk = pd.DataFrame(
        {
            "low_per_capita": minmax(out["services_per_capita"], invert=True),
            "far_distance": minmax(out["mean_nearest_m"]),
            "low_nearby": minmax(out["nearby_services"], invert=True),
            "low_diversity": minmax(out["service_diversity"], invert=True),
            "population": minmax(out["population"]),
        }
    )
    out["priority_score"] = (risk.mean(axis=1) * 100).round(2)
    return out.sort_values("priority_score", ascending=False), risk

