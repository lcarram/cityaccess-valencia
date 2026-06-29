import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .recommendation import recommend_neighbourhoods


def ranking_stability(access, preferences):
    base = recommend_neighbourhoods(access, preferences, top_n=len(access))
    base_rank = base.set_index("neighbourhood")["recommendation_score"].rank(ascending=False)
    rows = []
    for key in preferences:
        perturbed = preferences.copy()
        perturbed[key] = perturbed[key] * 1.2 + 1
        alt = recommend_neighbourhoods(access, perturbed, top_n=len(access))
        alt_rank = alt.set_index("neighbourhood")["recommendation_score"].rank(ascending=False)
        corr = spearmanr(base_rank.loc[alt_rank.index], alt_rank).correlation
        rows.append({"changed_weight": key, "spearman_correlation": round(float(corr), 3)})
    return pd.DataFrame(rows)


def anomaly_sensitivity(detector_func, access):
    rows = []
    for contamination in [0.05, 0.10, 0.15, 0.20, 0.25]:
        out = detector_func(access, contamination)
        rows.append({"contamination": contamination, "anomalies": int(out["is_anomaly"].sum())})
    return pd.DataFrame(rows)

