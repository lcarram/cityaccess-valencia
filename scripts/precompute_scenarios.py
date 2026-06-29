from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.accessibility import compute_accessibility
from src.data_loader import read_table
from src.geospatial_analysis import calculate_metrics, integrate_data
from src.preprocessing import CATEGORIES, clean_boundaries, clean_population, clean_real_facilities
from src.scenario_engine import OBJECTIVE_WEIGHTS, scenario_candidates, service_priority


BASE_WEIGHTS = {
    "services_per_10k": 2.0,
    "nearest_distance": 2.0,
    "nearby_services": 1.5,
    "diversity": 1.5,
}


def main():
    facilities = clean_real_facilities(read_table(Path("data/facilities_valencia.geojson")))
    boundaries = clean_boundaries(read_table(Path("data/neighbourhoods_valencia.geojson")), {"neighbourhood": "nombre"})
    population = clean_population(read_table(Path("data/population_valencia.xlsx")), {"neighbourhood": "neighbourhood_name", "population": "population"})

    services_joined, neighbourhoods = integrate_data(facilities, boundaries, population)
    metrics = calculate_metrics(services_joined, neighbourhoods, radius_m=1000)
    access = compute_accessibility(metrics, BASE_WEIGHTS)

    rows = []
    priority_rows = []
    for service_type in CATEGORIES:
        for objective in OBJECTIVE_WEIGHTS:
            priority = service_priority(access, service_type, objective)
            priority_export = priority.head(10)[
                ["neighbourhood", "population", "planning_priority", "priority_reason", "short_recommendation"]
            ].copy()
            priority_export["service_type"] = service_type
            priority_export["objective"] = objective
            priority_rows.append(priority_export)

            candidates = scenario_candidates(boundaries, access, service_type, priority, max_candidates=5)
            candidates["service_type"] = service_type
            candidates["objective"] = objective
            rows.append(candidates)

    scenarios = pd.concat(rows, ignore_index=True)
    priorities = pd.concat(priority_rows, ignore_index=True)
    Path("data").mkdir(exist_ok=True)
    scenarios.to_csv("data/precomputed_scenarios.csv", index=False)
    priorities.to_csv("data/precomputed_priorities.csv", index=False)
    print(f"Saved {len(scenarios)} scenario rows to data/precomputed_scenarios.csv")
    print(f"Saved {len(priorities)} priority rows to data/precomputed_priorities.csv")


if __name__ == "__main__":
    main()
