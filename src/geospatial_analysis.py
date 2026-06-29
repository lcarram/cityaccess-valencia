import geopandas as gpd
import numpy as np
import pandas as pd

from .preprocessing import CATEGORIES, services_to_gdf


PROJECTED_CRS = 25830


def integrate_data(services, boundaries, population):
    services_gdf = services_to_gdf(services)
    joined = gpd.sjoin(services_gdf, boundaries[["neighbourhood", "geometry"]], how="left", predicate="within")
    joined["neighbourhood"] = joined["neighbourhood"].fillna("outside boundary")
    areas = boundaries.to_crs(PROJECTED_CRS).copy()
    areas["area_km2"] = areas.area / 1_000_000
    base = boundaries.merge(population, on="neighbourhood", how="left")
    base["population"] = base["population"].fillna(base["population"].median()).clip(lower=1)
    base = base.merge(areas[["neighbourhood", "area_km2"]], on="neighbourhood", how="left")
    return joined.drop(columns=[c for c in ["index_right"] if c in joined.columns]), base


def calculate_metrics(services_joined, neighbourhoods, radius_m=1000):
    n_proj = neighbourhoods.to_crs(PROJECTED_CRS).copy()
    s_proj = services_joined.to_crs(PROJECTED_CRS).copy()
    centroids = n_proj.geometry.centroid
    rows = []
    for idx, nrow in n_proj.iterrows():
        name = nrow["neighbourhood"]
        pop = float(nrow["population"])
        area = max(float(nrow["area_km2"]), 0.001)
        centroid = centroids.loc[idx]
        row = {
            "neighbourhood": name,
            "population": pop,
            "area_km2": area,
            "total_services": int((services_joined["neighbourhood"] == name).sum()),
        }
        nearby_total = 0
        diversity = 0
        nearest_values = []
        for cat in CATEGORIES:
            local_count = int(((services_joined["neighbourhood"] == name) & (services_joined["category"] == cat)).sum())
            cat_points = s_proj[s_proj["category"] == cat]
            if len(cat_points):
                distances = cat_points.geometry.distance(centroid)
                nearest = float(distances.min())
                nearby = int((distances <= radius_m).sum())
            else:
                nearest = np.nan
                nearby = 0
            row[f"{cat}_count"] = local_count
            row[f"{cat}_per_10k"] = local_count / pop * 10000
            row[f"{cat}_nearest_m"] = nearest
            row[f"{cat}_within_radius"] = nearby
            row[f"{cat}_density_km2"] = local_count / area
            nearby_total += nearby
            nearest_values.append(nearest)
            diversity += int(local_count > 0)
        row["services_per_10k"] = row["total_services"] / pop * 10000
        row["services_per_capita"] = row["total_services"] / pop
        row["nearby_services"] = nearby_total
        row["service_diversity"] = diversity
        row["mean_nearest_m"] = float(np.nanmean(nearest_values)) if any(~pd.isna(nearest_values)) else np.nan
        rows.append(row)
    metrics = pd.DataFrame(rows)
    for cat in CATEGORIES:
        city_avg = metrics[f"{cat}_per_10k"].mean()
        metrics[f"{cat}_vs_city_avg"] = metrics[f"{cat}_per_10k"] - city_avg
    return metrics

