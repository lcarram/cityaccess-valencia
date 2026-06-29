import pandas as pd

from .accessibility import minmax
from .facility_location import candidate_centroids


OBJECTIVE_WEIGHTS = {
    "Reach the largest population": {"low_coverage": 1.0, "far_distance": 1.0, "few_nearby": 1.0, "population": 2.3},
    "Reduce average distance": {"low_coverage": 1.0, "far_distance": 2.5, "few_nearby": 1.4, "population": 1.0},
    "Prioritise underserved neighbourhoods": {"low_coverage": 2.0, "far_distance": 1.6, "few_nearby": 2.0, "population": 1.0},
    "Balanced impact": {"low_coverage": 1.4, "far_distance": 1.4, "few_nearby": 1.4, "population": 1.4},
}


def priority_reason(row, service_type):
    coverage = row[f"{service_type}_per_10k"]
    nearby = row[f"{service_type}_within_radius"]
    distance = row[f"{service_type}_nearest_m"]
    if nearby == 0:
        return "few nearby facilities for residents"
    if distance > 1200:
        return "residents are far from the nearest facility"
    if coverage < 1:
        return "low coverage relative to population"
    return "access is weaker than other neighbourhoods"


def short_recommendation(row, service_type):
    if row[f"{service_type}_within_radius"] == 0:
        return "Test a candidate site close to this neighbourhood."
    if row["population"] > 15000:
        return "Prioritise a location that can serve nearby high-population areas."
    return "Compare this area with adjacent underserved neighbourhoods."


def service_priority(access, service_type, objective):
    score = pd.DataFrame(
        {
            "low_coverage": minmax(access[f"{service_type}_per_10k"], invert=True),
            "far_distance": minmax(access[f"{service_type}_nearest_m"]),
            "few_nearby": minmax(access[f"{service_type}_within_radius"], invert=True),
            "population": minmax(access["population"]),
        }
    )
    weights = OBJECTIVE_WEIGHTS[objective]
    weighted = sum(score[col] * weight for col, weight in weights.items()) / sum(weights.values())
    out = access.copy()
    out["planning_priority"] = (weighted * 100).round(1)
    out["priority_reason"] = out.apply(lambda row: priority_reason(row, service_type), axis=1)
    out["short_recommendation"] = out.apply(lambda row: short_recommendation(row, service_type), axis=1)
    return out.sort_values("planning_priority", ascending=False)


def scenario_candidates(boundaries, access, service_type, priority_data, max_candidates=5):
    candidate_names = priority_data.head(max_candidates)["neighbourhood"].tolist()
    candidates = candidate_centroids(boundaries, candidate_names)
    projected_neighbourhoods = boundaries.to_crs(25830).copy()
    projected_neighbourhoods["centroid"] = projected_neighbourhoods.geometry.centroid
    candidate_points = candidates.to_crs(25830)
    access_by_name = access.set_index("neighbourhood")
    rows = []
    current_nearest = access_by_name[f"{service_type}_nearest_m"].fillna(access_by_name[f"{service_type}_nearest_m"].median())
    current_coverage = access_by_name[f"{service_type}_within_radius"] > 0

    for idx, cand in candidates.iterrows():
        point = candidate_points.loc[idx].geometry
        distances = projected_neighbourhoods.set_index("neighbourhood")["centroid"].distance(point).reindex(access_by_name.index)
        after_nearest = pd.concat([current_nearest, distances], axis=1).min(axis=1)
        reduction = (current_nearest - after_nearest).clip(lower=0)
        improved = reduction > 25
        after_coverage = current_coverage | (distances <= 1000)
        coverage_gain = int(after_coverage.sum() - current_coverage.sum())
        distance_gain_share = (reduction / current_nearest.replace(0, pd.NA)).fillna(0).clip(lower=0, upper=1)
        proximity_share = (1 - (distances / 2500)).clip(lower=0, upper=1).fillna(0)
        benefit_share = (0.65 * distance_gain_share + 0.35 * proximity_share).clip(lower=0, upper=1)
        population_benefited = int((access_by_name["population"] * benefit_share).sum())
        directly_reached = int(access_by_name.loc[distances <= 1000, "population"].sum())
        candidate_population = int(access_by_name.loc[cand["candidate"], "population"]) if cand["candidate"] in access_by_name.index else 0
        population_benefited = max(population_benefited, min(candidate_population, directly_reached or candidate_population))
        rows.append(
            {
                "candidate": cand["candidate"],
                "latitude": cand["latitude"],
                "longitude": cand["longitude"],
                "mean_accessibility_gain": float((reduction / 100).mean()),
                "coverage_gain": coverage_gain,
                "neighbourhoods_improved": int(improved.sum()),
                "mean_distance_reduction_m": float(reduction.mean()),
                "population_benefited": population_benefited,
                "directly_reached_population": directly_reached,
                "candidate_population": candidate_population,
            }
        )
    out = pd.DataFrame(rows)
    out["overall_impact_score"] = (
        0.40 * minmax(out["population_benefited"])
        + 0.30 * minmax(out["mean_distance_reduction_m"])
        + 0.20 * minmax(out["neighbourhoods_improved"])
        + 0.10 * minmax(out["coverage_gain"])
    ) * 100
    out["overall_impact_score"] = out["overall_impact_score"].round(1)
    out["simulation_score"] = out["overall_impact_score"]
    return out.sort_values("overall_impact_score", ascending=False)
