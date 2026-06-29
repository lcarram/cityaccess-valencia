import geopandas as gpd
import pandas as pd

from .accessibility import compute_accessibility
from .geospatial_analysis import calculate_metrics, integrate_data


def candidate_centroids(neighbourhoods, candidate_names=None):
    if candidate_names is not None:
        neighbourhoods = neighbourhoods[neighbourhoods["neighbourhood"].isin(candidate_names)].copy()
    pts = neighbourhoods.to_crs(25830).copy()
    pts["geometry"] = pts.geometry.centroid
    pts = pts.to_crs(4326)
    return gpd.GeoDataFrame(
        {"candidate": pts["neighbourhood"], "longitude": pts.geometry.x, "latitude": pts.geometry.y},
        geometry=pts.geometry,
        crs=4326,
    )


def simulate_new_facility(services, boundaries, population, base_access, service_type, weights, radius_m, candidate_names=None):
    candidates = candidate_centroids(boundaries, candidate_names)
    rows = []
    base_mean = base_access["accessibility_score"].mean()
    base_coverage = int((base_access[f"{service_type}_within_radius"] > 0).sum())
    base_distance = base_access[f"{service_type}_nearest_m"].mean()
    for _, cand in candidates.iterrows():
        new_row = pd.DataFrame(
            [
                {
                    "service_name": f"Simulated {service_type}",
                    "category": service_type,
                    "latitude": cand["latitude"],
                    "longitude": cand["longitude"],
                }
            ]
        )
        sim_services = pd.concat([services, new_row], ignore_index=True)
        sj, neigh = integrate_data(sim_services, boundaries, population)
        metrics = calculate_metrics(sj, neigh, radius_m=radius_m)
        access = compute_accessibility(metrics, weights)
        after_mean = access["accessibility_score"].mean()
        after_coverage = int((access[f"{service_type}_within_radius"] > 0).sum())
        after_distance = access[f"{service_type}_nearest_m"].mean()
        improved = int((access["accessibility_score"] > base_access["accessibility_score"]).sum())
        rows.append(
            {
                "candidate": cand["candidate"],
                "latitude": cand["latitude"],
                "longitude": cand["longitude"],
                "mean_accessibility_gain": after_mean - base_mean,
                "coverage_gain": after_coverage - base_coverage,
                "neighbourhoods_improved": improved,
                "mean_distance_reduction_m": base_distance - after_distance,
            }
        )
    out = pd.DataFrame(rows)
    out["simulation_score"] = (
        out[["mean_accessibility_gain", "coverage_gain", "neighbourhoods_improved", "mean_distance_reduction_m"]]
        .rank(pct=True)
        .mean(axis=1)
        * 100
    ).round(2)
    return out.sort_values("simulation_score", ascending=False)
