import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp


def psi(reference, latest, bins=10):
    ref = pd.Series(reference).dropna()
    new = pd.Series(latest).dropna()
    if ref.empty or new.empty:
        return np.nan
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    ref_counts = np.histogram(ref, edges)[0] / len(ref)
    new_counts = np.histogram(new, edges)[0] / len(new)
    ref_counts = np.where(ref_counts == 0, 0.001, ref_counts)
    new_counts = np.where(new_counts == 0, 0.001, new_counts)
    return float(np.sum((new_counts - ref_counts) * np.log(new_counts / ref_counts)))


def drift_report(reference, latest, category_ref=None, category_latest=None):
    rows = []
    numeric = ["total_services", "services_per_10k", "mean_nearest_m", "accessibility_score"]
    for col in numeric:
        if col in reference and col in latest:
            ks = ks_2samp(reference[col].dropna(), latest[col].dropna())
            delta_mean = latest[col].mean() - reference[col].mean()
            rows.append(
                {
                    "variable": col,
                    "mean_change": round(delta_mean, 3),
                    "std_change": round(latest[col].std() - reference[col].std(), 3),
                    "ks_p_value": round(float(ks.pvalue), 4),
                    "psi": round(psi(reference[col], latest[col]), 4),
                    "alert": abs(delta_mean) > reference[col].std() * 0.5 or ks.pvalue < 0.05,
                }
            )
    cat_result = None
    if category_ref is not None and category_latest is not None:
        cats = sorted(set(category_ref) | set(category_latest))
        table = np.array([[sum(category_ref == c) for c in cats], [sum(category_latest == c) for c in cats]])
        chi = chi2_contingency(table)
        cat_result = {"chi2_p_value": round(float(chi.pvalue), 4), "alert": chi.pvalue < 0.05}
    return pd.DataFrame(rows), cat_result

