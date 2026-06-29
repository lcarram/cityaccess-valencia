import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "healthcare_accessibility",
    "education_accessibility",
    "libraries_accessibility",
    "sports facilities_accessibility",
    "proximity",
    "services_per_capita",
]


def cluster_profiles(access, k=4, model_path="models/kmeans.joblib"):
    X = access[FEATURES].fillna(0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    k = max(2, min(int(k), len(access) - 1))
    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = model.fit_predict(Xs)
    score = silhouette_score(Xs, labels) if len(set(labels)) > 1 else np.nan
    out = access.copy()
    out["cluster"] = labels.astype(str)
    joblib.dump({"scaler": scaler, "model": model, "features": FEATURES}, model_path)
    return out, float(score), profile_clusters(out)


def profile_clusters(clustered):
    rows = []
    for cluster, group in clustered.groupby("cluster"):
        means = group[FEATURES].mean().sort_values(ascending=False)
        rows.append(
            {
                "cluster": cluster,
                "size": len(group),
                "profile": f"Strongest in {means.index[0].replace('_', ' ')}, weakest in {means.index[-1].replace('_', ' ')}.",
                "neighbourhoods": ", ".join(group["neighbourhood"].sort_values()),
            }
        )
    return pd.DataFrame(rows)


def silhouette_by_k(access, min_k=2, max_k=8):
    X = StandardScaler().fit_transform(access[FEATURES].fillna(0))
    rows = []
    for k in range(min_k, min(max_k, len(access) - 1) + 1):
        labels = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(X)
        rows.append({"k": k, "silhouette": silhouette_score(X, labels)})
    return pd.DataFrame(rows)

