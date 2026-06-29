import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .clustering import FEATURES


def detect_anomalies(access, contamination=0.15):
    X = access[FEATURES].fillna(0)
    Xs = StandardScaler().fit_transform(X)
    model = IsolationForest(contamination=contamination, random_state=42)
    pred = model.fit_predict(Xs)
    out = access.copy()
    out["anomaly_score"] = model.decision_function(Xs)
    out["is_anomaly"] = pred == -1
    global_mean = X.mean()
    explanations = []
    for _, row in out.iterrows():
        diffs = (row[FEATURES] - global_mean).sort_values(key=abs, ascending=False)
        top = diffs.index[0]
        direction = "high" if diffs.iloc[0] > 0 else "low"
        explanations.append(f"Unusual because {top.replace('_', ' ')} is especially {direction}.")
    out["anomaly_reason"] = explanations
    return out.sort_values("anomaly_score")

