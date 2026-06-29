import os
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from .preprocessing import CATEGORIES


RNG = np.random.default_rng(42)


def read_table(uploaded_file):
    if uploaded_file is None:
        return None
    is_local_path = isinstance(uploaded_file, (str, os.PathLike, Path))
    name = str(uploaded_file if is_local_path else uploaded_file.name).lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    if name.endswith((".geojson", ".json")):
        return gpd.read_file(uploaded_file)
    if name.endswith(".zip"):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, uploaded_file.name)
            with open(path, "wb") as fh:
                fh.write(uploaded_file.getbuffer())
            return gpd.read_file(f"zip://{path}")
    raise ValueError("Unsupported file type. Use CSV, XLSX, GeoJSON or zipped shapefile.")


def make_demo_data():
    names = [
        "Benimaclet",
        "Ruzafa",
        "Campanar",
        "Cabanyal",
        "Patraix",
        "Malilla",
        "Orriols",
        "Natzaret",
        "Torrefiel",
        "La Seu",
        "Benicalap",
        "Quatre Carreres",
    ]
    base_lat, base_lon = 39.47, -0.376
    rows, pop_rows, poly_rows = [], [], []
    for i, name in enumerate(names):
        row, col = divmod(i, 4)
        lon = base_lon + (col - 1.5) * 0.027
        lat = base_lat + (1.2 - row) * 0.022
        size = 0.011
        poly_rows.append(
            {
                "neighbourhood": name,
                "geometry": Polygon(
                    [
                        (lon - size, lat - size),
                        (lon + size, lat - size),
                        (lon + size, lat + size),
                        (lon - size, lat + size),
                    ]
                ),
            }
        )
        population = int(RNG.integers(6500, 26000))
        pop_rows.append({"neighbourhood": name, "population": population})
        for cat in CATEGORIES:
            lam = {"healthcare": 1.5, "education": 2.4, "libraries": 0.9, "sports facilities": 1.4}[cat]
            if name in ["Natzaret", "Orriols"] and cat in ["healthcare", "libraries"]:
                lam *= 0.35
            if name in ["Ruzafa", "La Seu"] and cat in ["libraries", "education"]:
                lam *= 1.8
            count = max(0, int(RNG.poisson(lam)))
            for j in range(count):
                rows.append(
                    {
                        "service_name": f"{cat.title()} {j + 1} - {name}",
                        "category": cat,
                        "latitude": lat + RNG.normal(0, 0.006),
                        "longitude": lon + RNG.normal(0, 0.006),
                    }
                )
    return (
        pd.DataFrame(rows),
        gpd.GeoDataFrame(poly_rows, crs=4326),
        pd.DataFrame(pop_rows),
    )
