import re
import unicodedata

import geopandas as gpd
import numpy as np
import pandas as pd


CATEGORY_MAP = {
    "health": "healthcare",
    "healthcare": "healthcare",
    "salud": "healthcare",
    "sanidad": "healthcare",
    "hospital": "healthcare",
    "centro de salud": "healthcare",
    "instalaciones sanitarias": "healthcare",
    "sanitarias": "healthcare",
    "education": "education",
    "educacion": "education",
    "educación": "education",
    "school": "education",
    "colegio": "education",
    "instalaciones educativas": "education",
    "library": "libraries",
    "libraries": "libraries",
    "biblioteca": "libraries",
    "bibliotecas": "libraries",
    "sport": "sports facilities",
    "sports": "sports facilities",
    "sports facilities": "sports facilities",
    "deporte": "sports facilities",
    "polideportivo": "sports facilities",
    "instalaciones deportivas": "sports facilities",
    "bienestar social": "social services",
    "servicios sociales": "social services",
    "social services": "social services",
    "museos": "culture",
    "museo": "culture",
    "teatros": "culture",
    "teatro": "culture",
    "archivos": "culture",
    "culture": "culture",
    "cultura": "culture",
}

CATEGORIES = ["healthcare", "education", "libraries", "sports facilities", "social services", "culture"]


def standardise_name(value):
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def harmonise_category(value):
    key = standardise_name(value)
    return CATEGORY_MAP.get(key, key if key in CATEGORIES else "other")


def clean_services(df, mapping):
    out = pd.DataFrame(
        {
            "service_name": df[mapping["service_name"]].fillna("Unnamed service").astype(str),
            "category": df[mapping["category"]].map(harmonise_category),
            "latitude": pd.to_numeric(df[mapping["latitude"]], errors="coerce"),
            "longitude": pd.to_numeric(df[mapping["longitude"]], errors="coerce"),
        }
    )
    out = out.drop_duplicates()
    out = out[out["category"].isin(CATEGORIES)]
    out = out[out["latitude"].between(38.0, 41.0) & out["longitude"].between(-2.0, 1.0)]
    return out.reset_index(drop=True)


def clean_real_facilities(gdf):
    """Clean Valencia facilities from the local GeoJSON shipped in data/."""
    out = gdf.copy()
    coords = gpd.GeoDataFrame(
        out,
        geometry=gpd.points_from_xy(pd.to_numeric(out["x"], errors="coerce"), pd.to_numeric(out["y"], errors="coerce")),
        crs=25830,
    ).to_crs(4326)
    cleaned = pd.DataFrame(
        {
            "service_name": out["equipamien"].fillna("Unnamed service").astype(str),
            "category": out["clase"].map(harmonise_category),
            "latitude": coords.geometry.y,
            "longitude": coords.geometry.x,
        }
    )
    cleaned = cleaned.dropna(subset=["latitude", "longitude"]).drop_duplicates()
    cleaned = cleaned[cleaned["category"].isin(CATEGORIES)]
    return cleaned.reset_index(drop=True)


def clean_population(df, mapping):
    out = pd.DataFrame(
        {
            "neighbourhood": df[mapping["neighbourhood"]].map(standardise_name),
            "population": pd.to_numeric(df[mapping["population"]], errors="coerce"),
        }
    )
    out = out.dropna(subset=["neighbourhood"])
    out["population"] = out["population"].fillna(out["population"].median()).clip(lower=1)
    return out.groupby("neighbourhood", as_index=False)["population"].sum()


def clean_boundaries(gdf, mapping):
    out = gdf.rename(columns={mapping["neighbourhood"]: "neighbourhood"}).copy()
    out["neighbourhood"] = out["neighbourhood"].map(standardise_name)
    out = out.dropna(subset=["geometry", "neighbourhood"])
    out["geometry"] = out.geometry.make_valid()
    out = out[~out.geometry.is_empty]
    if out.crs is None:
        out = out.set_crs(4326)
    return out[["neighbourhood", "geometry"]].to_crs(4326).reset_index(drop=True)


def services_to_gdf(services):
    return gpd.GeoDataFrame(
        services.copy(),
        geometry=gpd.points_from_xy(services["longitude"], services["latitude"]),
        crs=4326,
    )
