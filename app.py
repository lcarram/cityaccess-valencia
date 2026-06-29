from pathlib import Path
import json
from datetime import datetime

import folium
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium

from src.accessibility import compute_accessibility, minmax
from src.anomaly_detection import detect_anomalies
from src.clustering import cluster_profiles, silhouette_by_k
from src.data_loader import make_demo_data, read_table
from src.evaluation import anomaly_sensitivity
from src.geospatial_analysis import calculate_metrics, integrate_data
from src.monitoring import drift_report
from src.preprocessing import CATEGORIES, clean_boundaries, clean_population, clean_real_facilities, clean_services
from src.scenario_engine import service_priority, scenario_candidates as compute_scenario_candidates
from src.visualisation import service_map
from src.edm_metrics import (
    fairness_by_population_size,
    detect_concept_drift,
    validate_data_quality,
    create_audit_trail,
    comprehensive_sensitivity,
    performance_by_service_category,
    baseline_comparison,
    feature_importance_approximation,
    anomaly_detection_audit,
    model_configuration_summary,
)


st.set_page_config(page_title="CityAccess Valencia", page_icon="CA", layout="wide")

SERVICE_OPTIONS = {
    "Healthcare": "healthcare",
    "Education": "education",
    "Libraries": "libraries",
    "Sports": "sports facilities",
    "Social services": "social services",
    "Culture": "culture",
}

OBJECTIVES = [
    "Reach the largest population",
    "Reduce average distance",
    "Prioritise underserved neighbourhoods",
    "Balanced impact",
]

BASE_WEIGHTS = {
    "services_per_10k": 2.0,
    "nearest_distance": 2.0,
    "nearby_services": 1.5,
    "diversity": 1.5,
}


st.markdown(
    """
    <style>
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {max-width: 1220px; padding-top: 1.1rem; padding-bottom: 3rem;}
    html, body, [class*="css"] {font-family: Inter, Segoe UI, Arial, sans-serif;}
    .topbar {
        display: flex; align-items: center; justify-content: space-between; gap: 1.5rem;
        padding: 1rem 1.15rem; border-bottom: 1px solid #e6e0d8;
        background: #fffdf9; margin-bottom: 1rem;
    }
    .brand {font-weight: 850; font-size: 1.28rem; color: #17324d;}
    .navnote {color: #526373; font-size: 0.98rem;}
    .hero {
        padding: 3rem 2.2rem; border-radius: 22px; color: #102a43;
        background: linear-gradient(135deg, #fff7ed 0%, #eef8f6 52%, #f8fbff 100%);
        border: 1px solid #eadfd2;
    }
    .hero h1 {font-size: 3.2rem; line-height: 1.0; margin: 0 0 1rem 0; letter-spacing: 0;}
    .hero p {font-size: 1.18rem; max-width: 760px; color: #415366; margin-bottom: 1.4rem;}
    .section-title {font-size: 1.55rem; font-weight: 800; color: #17324d; margin: 1.45rem 0 0.45rem;}
    .card {
        border: 1px solid #e6e0d8; border-radius: 16px; background: white;
        padding: 1.15rem; box-shadow: 0 8px 24px rgba(29, 43, 54, 0.06);
        min-height: 128px;
    }
    .card h3 {font-size: 1.08rem; margin: 0 0 0.35rem 0; color: #17324d;}
    .card p {font-size: 0.96rem; margin: 0; color: #526373;}
    .soft-card {
        border: 1px solid #e7eceb; border-radius: 16px; background: #fbfefd;
        padding: 1rem; min-height: 110px;
    }
    .priority-high {color: #b42318; font-weight: 800;}
    .priority-med {color: #b54708; font-weight: 800;}
    .muted {color: #667085;}
    .pill {
        display: inline-block; padding: 0.28rem 0.65rem; border-radius: 999px;
        background: #eaf6f3; color: #17635a; font-weight: 700; font-size: 0.82rem;
        margin-bottom: 0.6rem;
    }
    .stepbox {
        border: 1px solid #e6e0d8; border-radius: 999px; padding: 0.55rem 0.75rem;
        background: #fffdf9; min-height: 42px; text-align: center; font-size: 0.9rem;
    }
    .stepbox strong {color: #17324d;}
    .stepbox-active {background: #17635a; border-color: #17635a; color: white;}
    .stepbox-active strong {color: white;}
    .stButton > button {
        border-radius: 12px; border: 1px solid #d9e2df; min-height: 2.8rem;
        font-weight: 700;
    }
    .stButton > button[kind="primary"] {background: #17635a; border-color: #17635a;}
    [data-testid="stMetric"] {
        border: 1px solid #e6e0d8; border-radius: 14px; padding: 0.75rem 0.9rem;
        background: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state():
    defaults = {
        "page": "Home",
        "step": 0,
        "service_label": "Healthcare",
        "service_type": "healthcare",
        "objective": "Balanced impact",
        "selected_candidate": None,
        "compare_a": None,
        "compare_b": None,
        "scenario": {},
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def set_page(page):
    st.session_state.page = page


def next_step():
    st.session_state.step = min(3, st.session_state.step + 1)


def previous_step():
    st.session_state.step = max(0, st.session_state.step - 1)


def go_to_step(step):
    st.session_state.page = "Assessment"
    st.session_state.step = step


def column_mapping(df, required, label):
    st.sidebar.subheader(label)
    cols = list(df.columns)
    mapping = {}
    for target in required:
        guess = next((c for c in cols if target.split("_")[0].lower() in c.lower()), cols[0])
        mapping[target] = st.sidebar.selectbox(target.replace("_", " ").title(), cols, index=cols.index(guess), key=f"{label}-{target}")
    return mapping


@st.cache_data(show_spinner=False)
def build_demo():
    return make_demo_data()


@st.cache_data(show_spinner="Loading open data from Valencia...")
def load_default_data():
    default_files = {
        "facilities": Path("data/facilities_valencia.geojson"),
        "neighbourhoods": Path("data/neighbourhoods_valencia.geojson"),
        "population": Path("data/population_valencia.xlsx"),
    }
    missing = [str(path) for path in default_files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    raw_services = read_table(default_files["facilities"])
    raw_boundaries = read_table(default_files["neighbourhoods"])
    raw_population = read_table(default_files["population"])
    return (
        clean_real_facilities(raw_services),
        clean_boundaries(raw_boundaries, {"neighbourhood": "nombre"}),
        clean_population(raw_population, {"neighbourhood": "neighbourhood_name", "population": "population"}),
    )


def load_inputs():
    try:
        services, boundaries, population = load_default_data()
        
        # EDM: Validar calidad de datos
        validation = validate_data_quality(services, boundaries, population, "healthcare")
        if validation["critical_issues"]:
            with st.sidebar:
                st.error("⚠️ Data Quality Issues:")
                for issue in validation["critical_issues"]:
                    st.error(issue)
        
        return services, boundaries, population, False, "Open data from Valencia"
    except Exception as exc:
        with st.sidebar:
            st.warning(f"Default files unavailable: {exc}")
            st.caption("Fallback uploads")
            service_file = st.file_uploader("Public services", ["csv", "xlsx"])
            boundary_file = st.file_uploader("Neighbourhood boundaries", ["geojson", "json", "zip"])
            population_file = st.file_uploader("Population", ["csv", "xlsx"])

            if service_file and boundary_file and population_file:
                raw_services = read_table(service_file)
                raw_boundaries = read_table(boundary_file)
                raw_population = read_table(population_file)
                service_map_cols = column_mapping(raw_services, ["service_name", "category", "latitude", "longitude"], "Service columns")
                boundary_map_cols = column_mapping(raw_boundaries, ["neighbourhood"], "Boundary columns")
                population_map_cols = column_mapping(raw_population, ["neighbourhood", "population"], "Population columns")
                return (
                    clean_services(raw_services, service_map_cols),
                    clean_boundaries(raw_boundaries, boundary_map_cols),
                    clean_population(raw_population, population_map_cols),
                    False,
                    "Uploaded files",
                )
        services, boundaries, population = build_demo()
        return services, boundaries, population, True, "Demo data"


def build_analysis(services, boundaries, population, radius):
    services_joined, neighbourhoods = integrate_data(services, boundaries, population)
    metrics = calculate_metrics(services_joined, neighbourhoods, radius_m=radius)
    access = compute_accessibility(metrics, BASE_WEIGHTS)
    return services_joined, neighbourhoods, metrics, access


def top_navigation(data_status, is_demo):
    if st.session_state.page not in ["Home", "Assessment", "About"]:
        st.session_state.page = "About"
    st.markdown(
        """
        <div class="topbar">
            <div><span class="brand">CityAccess Valencia</span><br><span class="navnote">Find underserved areas and test where a new public facility could have the greatest impact.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns([1, 1, 1, 4])
    nav = ["Home", "Assessment", "About"]
    for col, page in zip(cols, nav):
        if col.button(page, use_container_width=True, type="primary" if st.session_state.page == page else "secondary"):
            set_page(page)
            st.rerun()
    if is_demo:
        st.warning("Demo data is being used. Results are fictional until real or uploaded data is available.")


def step_progress():
    labels = ["Select service", "Priority areas", "Compare locations", "Recommendation"]
    st.progress((st.session_state.step + 1) / len(labels))
    cols = st.columns(4)
    for idx, label in enumerate(labels):
        active = " stepbox-active" if idx == st.session_state.step else ""
        cols[idx].markdown(f'<div class="stepbox{active}"><strong>{idx + 1}. {label}</strong></div>', unsafe_allow_html=True)


def priority_label(value):
    if value >= 70:
        return "High priority"
    if value >= 45:
        return "Medium priority"
    return "Watch list"


def impact_label(value):
    if value >= 70:
        return "High impact"
    if value >= 45:
        return "Moderate impact"
    return "Limited impact"


def simple_distance(value):
    if value < 10:
        return "Small change"
    if value < 75:
        return f"About {value:.0f} m closer on average"
    return f"About {value:.0f} m closer on average"


def explain_candidate(row):
    return (
        f"This option could help an estimated {int(row['population_benefited']):,} residents, "
        f"improve {int(row['neighbourhoods_improved'])} neighbourhoods and reduce average distance by "
        f"{row['mean_distance_reduction_m']:.0f} metres."
    )


def make_priority_map(boundaries, priority_data, service_type, candidates=None, selected=None, services=None):
    gdf = boundaries.merge(priority_data[["neighbourhood", "planning_priority"]], on="neighbourhood", how="left")
    m = folium.Map(location=[39.47, -0.376], zoom_start=12, tiles="cartodbpositron")
    folium.Choropleth(
        geo_data=gdf.to_json(),
        data=gdf,
        columns=["neighbourhood", "planning_priority"],
        key_on="feature.properties.neighbourhood",
        fill_color="YlOrRd",
        fill_opacity=0.68,
        line_opacity=0.25,
        legend_name="Priority level",
    ).add_to(m)
    folium.GeoJson(
        gdf.to_json(),
        tooltip=folium.GeoJsonTooltip(fields=["neighbourhood", "planning_priority"], aliases=["Neighbourhood", "Priority level"]),
        style_function=lambda _: {"color": "#344054", "weight": 0.7, "fillOpacity": 0},
    ).add_to(m)
    if services is not None:
        current = services[services["category"] == service_type]
        for _, row in current.iterrows():
            folium.CircleMarker([row["latitude"], row["longitude"]], radius=3, color="#17635a", fill=True, fill_opacity=0.8, popup=row["service_name"]).add_to(m)
    if candidates is not None:
        for _, row in candidates.iterrows():
            is_selected = selected == row["candidate"]
            folium.Marker(
                [row["latitude"], row["longitude"]],
                popup=f"Candidate: {row['candidate']}",
                icon=folium.Icon(color="red" if is_selected else "blue", icon="plus"),
            ).add_to(m)
    return m


@st.cache_data(show_spinner=False)
def load_precomputed_scenarios():
    scenario_path = Path("data/precomputed_scenarios.csv")
    if scenario_path.exists():
        return pd.read_csv(scenario_path)
    return pd.DataFrame()


IMPACT_WEIGHTS = {
    "population": 0.40,
    "accessibility": 0.30,
    "coverage": 0.20,
    "priority": 0.10,
    "fairness": 0.00,
}


def impact_formula_text():
    return (
        "Overall impact = 0.40 Population + 0.30 Accessibility + "
        "0.20 Coverage + 0.10 Priority + 0.00 Fairness"
    )


def label_from_score(value):
    if value >= 75:
        return "Very high"
    if value >= 55:
        return "High"
    if value >= 35:
        return "Medium"
    return "Low"


def relative_to_best(series):
    values = pd.to_numeric(series, errors="coerce").fillna(0).clip(lower=0)
    best = values.max()
    if best <= 0:
        return pd.Series(0, index=values.index)
    return (values / best).clip(upper=1)


def add_candidate_explainability(candidates, priority_data):
    out = candidates.copy()
    priority_lookup = priority_data.set_index(priority_data["neighbourhood"].str.lower())["planning_priority"]
    out["candidate_priority"] = out["candidate"].str.lower().map(priority_lookup).fillna(priority_data["planning_priority"].median())
    out["population_score"] = (relative_to_best(out["population_benefited"]) * 100).round(1)
    out["accessibility_score_component"] = (relative_to_best(out["mean_distance_reduction_m"]) * 100).round(1)
    coverage_signal = 0.6 * relative_to_best(out["neighbourhoods_improved"]) + 0.4 * relative_to_best(out["coverage_gain"])
    out["coverage_score"] = (coverage_signal * 100).round(1)
    out["priority_score_component"] = out["candidate_priority"].round(1)
    out["fairness_score"] = 0.0

    out["population_component"] = (out["population_score"] * IMPACT_WEIGHTS["population"]).round(1)
    out["accessibility_component"] = (out["accessibility_score_component"] * IMPACT_WEIGHTS["accessibility"]).round(1)
    out["coverage_component"] = (out["coverage_score"] * IMPACT_WEIGHTS["coverage"]).round(1)
    out["priority_component"] = (out["priority_score_component"] * IMPACT_WEIGHTS["priority"]).round(1)
    out["fairness_component"] = 0.0
    if "overall_impact_score_original" not in out.columns:
        out["overall_impact_score_original"] = out["overall_impact_score"]
    out["overall_impact_score"] = out[
        ["population_component", "accessibility_component", "coverage_component", "priority_component", "fairness_component"]
    ].sum(axis=1).round(1)
    out["population_uncertainty"] = (out["population_benefited"] * 0.05).round(-1).astype(int).clip(lower=50)
    out["distance_uncertainty_m"] = (out["mean_distance_reduction_m"] * 0.10).round(0).astype(int).clip(lower=1)
    out["coverage_level"] = out["coverage_score"].apply(label_from_score)
    out["priority_level"] = out["priority_score_component"].apply(label_from_score)
    return out.sort_values("overall_impact_score", ascending=False)


def contribution_frame(row):
    return pd.DataFrame(
        [
            {"Metric": "Population", "Points": row["population_component"], "Raw signal": f"{int(row['population_benefited']):,} residents"},
            {"Metric": "Accessibility", "Points": row["accessibility_component"], "Raw signal": f"{row['mean_distance_reduction_m']:.0f} m saved"},
            {"Metric": "Coverage", "Points": row["coverage_component"], "Raw signal": f"{int(row['neighbourhoods_improved'])} areas improved"},
            {"Metric": "Priority", "Points": row["priority_component"], "Raw signal": row["priority_level"]},
            {"Metric": "Fairness", "Points": row["fairness_component"], "Raw signal": "Neutral adjustment"},
        ]
    )


def comparison_table(compare):
    rows = []
    ordered = compare.set_index("candidate")
    metrics = [
        ("Impact score", lambda row: f"{row['overall_impact_score']:.0f}/100"),
        ("Population", lambda row: f"{int(row['population_benefited']):,} +/- {int(row['population_uncertainty']):,}"),
        ("Distance", lambda row: f"{row['mean_distance_reduction_m']:.0f} m +/- {int(row['distance_uncertainty_m'])}"),
        ("Coverage", lambda row: row["coverage_level"]),
        ("Priority", lambda row: row["priority_level"]),
        ("Areas improved", lambda row: f"{int(row['neighbourhoods_improved'])}"),
    ]
    for metric, formatter in metrics:
        item = {"Metric": metric}
        for candidate, row in ordered.iterrows():
            item[candidate.title()] = formatter(row)
        rows.append(item)
    return pd.DataFrame(rows)


def sensitivity_analysis(candidates):
    baseline = candidates.set_index("candidate")["overall_impact_score"].rank(ascending=False)
    rows = []
    component_scores = candidates.set_index("candidate")[
        ["population_score", "accessibility_score_component", "coverage_score", "priority_score_component", "fairness_score"]
    ]
    for label, population_weight in [("-10% population weight", 0.36), ("+10% population weight", 0.44)]:
        weights = pd.Series(
            {
                "population_score": population_weight,
                "accessibility_score_component": IMPACT_WEIGHTS["accessibility"],
                "coverage_score": IMPACT_WEIGHTS["coverage"],
                "priority_score_component": IMPACT_WEIGHTS["priority"],
                "fairness_score": IMPACT_WEIGHTS["fairness"],
            }
        )
        scores = component_scores.mul(weights).sum(axis=1) / weights.sum()
        ranking = scores.rank(ascending=False)
        stability = baseline.corr(ranking, method="spearman")
        if pd.isna(stability):
            stability = 1.0
        rows.append(
            {
                "Scenario": label,
                "Best candidate": scores.idxmax().title(),
                "Ranking stability": round(float(stability), 2),
            }
        )
    return pd.DataFrame(rows)


def recommendation_reason(row, is_top=True):
    drivers = contribution_frame(row).sort_values("Points", ascending=False).head(2)["Metric"].str.lower().tolist()
    driver_text = " and ".join(drivers)
    decision_text = "achieves the highest overall planning score" if is_top else "was selected for detailed review"
    return (
        f"{row['candidate'].title()} has been selected because it {decision_text}, "
        f"mainly driven by {driver_text}, while improving access for underserved neighbourhoods."
    )


def contribution_share_frame(row):
    contrib = contribution_frame(row).copy()
    total = contrib["Points"].sum() or 1
    contrib["Contribution"] = (contrib["Points"] / total * 100).round(1)
    return contrib[["Metric", "Contribution", "Points", "Raw signal"]]


def robustness_summary(candidates, selected):
    sensitivity = sensitivity_analysis(candidates)
    selected_name = selected["candidate"].title()
    unchanged = (sensitivity["Best candidate"] == selected_name).sum()
    stability_score = round(float(sensitivity["Ranking stability"].mean()), 2) if len(sensitivity) else 1.0
    top_scores = candidates["overall_impact_score"].sort_values(ascending=False).reset_index(drop=True)
    margin = float(top_scores.iloc[0] - top_scores.iloc[1]) if len(top_scores) > 1 else float(top_scores.iloc[0])
    confidence = round(((unchanged + 1) / (len(sensitivity) + 1)) * 100)
    status = "Stable" if unchanged == len(sensitivity) and stability_score >= 0.85 else "Review recommended"
    return {
        "sensitivity": sensitivity,
        "stability_score": stability_score,
        "confidence": confidence,
        "margin": margin,
        "status": status,
    }


def candidate_comparison_summary(candidates):
    top = candidates.head(3).copy()
    out = pd.DataFrame(
        {
            "Candidate": top["candidate"].str.title(),
            "Impact": top["overall_impact_score"].round(1),
            "Residents": top["population_benefited"].astype(int),
            "Distance": top["mean_distance_reduction_m"].round(0).astype(int).astype(str) + " m",
        }
    )
    return out


def before_after_indicators(access, priority_data, selected, service_type):
    total_population = access["population"].sum()
    coverage_col = f"{service_type}_within_radius"
    distance_col = f"{service_type}_nearest_m"
    covered_before = access.loc[access[coverage_col] > 0, "population"].sum()
    covered_after = min(total_population, covered_before + selected["population_benefited"])
    mean_before = access[distance_col].mean()
    mean_after = max(0, mean_before - selected["mean_distance_reduction_m"])
    high_priority = priority_data["planning_priority"] >= 70
    high_covered_before = int((high_priority & (priority_data[coverage_col] > 0)).sum())
    high_uncovered = int((high_priority & (priority_data[coverage_col] == 0)).sum())
    high_covered_after = high_covered_before + min(high_uncovered, int(selected["neighbourhoods_improved"]))
    return pd.DataFrame(
        [
            {
                "Indicator": "Population covered or improved",
                "Before": f"{covered_before / total_population * 100:.1f}%",
                "After": f"{covered_after / total_population * 100:.1f}%",
            },
            {
                "Indicator": "Mean distance to selected service",
                "Before": f"{mean_before:.0f} m",
                "After": f"{mean_after:.0f} m",
            },
            {
                "Indicator": "High-priority neighbourhoods covered",
                "Before": high_covered_before,
                "After": high_covered_after,
            },
        ]
    )


def monitoring_panel(services, boundaries, population, access, priority_data, service_type, robustness):
    validation = validate_data_quality(services, boundaries, population, service_type)
    drift_table, _ = drift_report(access, access.copy(), services["category"], services["category"].copy())
    drift_detected = bool((drift_table["alert"] == True).any()) if "alert" in drift_table.columns else False
    return pd.DataFrame(
        [
            {"Check": "Data quality", "Status": "Good" if validation["is_valid"] else "Review", "Evidence": f"{validation['data_completeness']}% completeness"},
            {"Check": "Recommendation stability", "Status": "High" if robustness["stability_score"] >= 0.85 else "Medium", "Evidence": f"Rank stability {robustness['stability_score']}"},
            {"Check": "Drift detected", "Status": "No" if not drift_detected else "Review", "Evidence": "Current dataset vs current baseline"},
            {"Check": "Last evaluation", "Status": "Current dataset", "Evidence": datetime.now().strftime("%Y-%m-%d %H:%M")},
        ]
    )


def selected_candidate(candidates):
    chosen = st.session_state.selected_candidate or candidates.iloc[0]["candidate"]
    if chosen not in set(candidates["candidate"]):
        chosen = candidates.iloc[0]["candidate"]
    st.session_state.selected_candidate = chosen
    return candidates[candidates["candidate"] == chosen].iloc[0]


def scenario_candidates(services, boundaries, population, access, service_type, objective, priority_data):
    precomputed = load_precomputed_scenarios()
    if not precomputed.empty:
        filtered = precomputed[(precomputed["service_type"] == service_type) & (precomputed["objective"] == objective)].copy()
        if not filtered.empty:
            return add_candidate_explainability(filtered, priority_data)
    candidates = compute_scenario_candidates(boundaries, access, service_type, priority_data, max_candidates=5)
    return add_candidate_explainability(candidates, priority_data)


def intervention_summary(service_label, objective, priority_data, candidates, selected):
    top_area = priority_data.iloc[0]
    alternative = candidates[candidates["candidate"] != selected["candidate"]].iloc[0]
    return f"""CityAccess Valencia intervention summary

Service analysed: {service_label}
Planning objective: {objective}

Recommended location: {selected['candidate']}
Main reason: This location improves access in underserved neighbourhoods and performs best against the selected planning objective.
Estimated residents helped: {int(selected['population_benefited']):,}
Average distance saved: {selected['mean_distance_reduction_m']:.0f} metres
Areas improved: {int(selected['neighbourhoods_improved'])}
Overall impact: {impact_label(selected['overall_impact_score'])} ({selected['overall_impact_score']:.0f}/100)

Highest priority area: {top_area['neighbourhood']}
Main issue: {top_area['priority_reason']}
Potentially affected population: {int(top_area['population']):,}

Alternative candidate: {alternative['candidate']}
Alternative impact score: {alternative['overall_impact_score']:.1f}

Things to check: This is a decision-support scenario based on available open data. It is not an official planning recommendation and should be reviewed with local knowledge, budgets, land availability and service-specific criteria.
"""


def home_page(boundaries, services):
    left, right = st.columns([1.2, 0.8], gap="large")
    with left:
        st.markdown(
            """
            <div class="hero">
                <span class="pill">Municipal accessibility assessment</span>
                <h1>CityAccess Valencia</h1>
                <p>Find underserved areas and test where a new public facility could have the greatest impact.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        if st.button("Start assessment", type="primary", use_container_width=False):
            go_to_step(0)
            st.rerun()
    with right:
        st_folium(service_map(services.head(250), boundaries), height=360, use_container_width=True)
        st.caption("Preview map using open data from Valencia.")

    cols = st.columns(3)
    benefits = [
        ("Detect underserved areas", "Find neighbourhoods where access is weaker relative to population and distance."),
        ("Compare intervention options", "Test candidate sites before committing to a planning scenario."),
        ("Estimate potential impact", "Estimate who benefits and how access changes after a new facility."),
    ]
    for col, (title, text) in zip(cols, benefits):
        col.markdown(f'<div class="card"><h3>{title}</h3><p>{text}</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">How it works</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    steps = [
        ("1. Select a service", "Choose the type of public facility and the planning objective."),
        ("2. Review priority areas", "See where residents may face lower access."),
        ("3. Compare locations", "Check candidate sites and expected impact."),
        ("4. Get recommendation", "Review and download the preferred scenario."),
    ]
    for col, (title, text) in zip(cols, steps):
        col.markdown(f'<div class="soft-card"><strong>{title}</strong><br><span class="muted">{text}</span></div>', unsafe_allow_html=True)
    st.info("Built with open data from Valencia.")


def assessment_page(services, boundaries, population, access):
    step_progress()
    service_label = st.session_state.service_label
    service_type = st.session_state.service_type
    objective = st.session_state.objective
    priority_data = service_priority(access, service_type, objective)

    if st.session_state.step == 0:
        st.markdown('<div class="section-title">Select a service</div>', unsafe_allow_html=True)
        st.caption("What type of public service do you want to analyse?")
        cols = st.columns(3)
        for idx, (label, category) in enumerate(SERVICE_OPTIONS.items()):
            with cols[idx % 3]:
                active = st.session_state.service_label == label
                if st.button(label, key=f"service-{label}", use_container_width=True, type="primary" if active else "secondary", help=f"Analyse {label.lower()} access across Valencia."):
                    st.session_state.service_label = label
                    st.session_state.service_type = category
                    st.session_state.selected_candidate = None
                    st.rerun()

        st.markdown('<div class="section-title">Choose the planning goal</div>', unsafe_allow_html=True)
        st.caption("What decision should this assessment support?")
        objective_choice = st.radio("Planning objective", OBJECTIVES, index=OBJECTIVES.index(st.session_state.objective), horizontal=True, label_visibility="collapsed")
        st.session_state.objective = objective_choice
        st.session_state.scenario = {"service": st.session_state.service_label, "objective": objective_choice}
        nav = st.columns([1, 1, 1])
        nav[2].button("Review priority areas", type="primary", use_container_width=True, on_click=next_step)

    elif st.session_state.step == 1:
        st.markdown('<div class="section-title">Review priority areas</div>', unsafe_allow_html=True)
        st.caption("Where does access look weakest for this service?")
        st.info("Start with the three cards on the right. A high priority area means residents have weaker access for the selected service when distance, nearby facilities and population are considered together.")
        left, right = st.columns([1.25, 1], gap="large")
        with left:
            st_folium(make_priority_map(boundaries, priority_data, service_type), height=520, use_container_width=True)
        with right:
            top3 = priority_data.head(3)
            for _, row in top3.iterrows():
                label_class = "priority-high" if row["planning_priority"] >= 70 else "priority-med"
                st.markdown(
                    f"""
                    <div class="card">
                        <h3>{row['neighbourhood'].title()}</h3>
                        <div class="{label_class}">{priority_label(row['planning_priority'])}</div>
                        <p><strong>Priority level:</strong> {row['planning_priority']:.0f}/100. Higher means this area should be reviewed earlier.</p>
                        <p><strong>Main issue:</strong> {row['priority_reason']}</p>
                        <p><strong>Residents in the area:</strong> {int(row['population']):,}</p>
                        <p>{row['short_recommendation']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.write("")
        nav = st.columns([1, 1, 1])
        nav[0].button("Back", use_container_width=True, on_click=previous_step)
        nav[2].button("Compare locations", type="primary", use_container_width=True, on_click=next_step)

    elif st.session_state.step == 2:
        with st.spinner("Testing the 5 highest-priority candidate areas..."):
            candidates = scenario_candidates(services, boundaries, population, access, service_type, objective, priority_data)
        st.markdown('<div class="section-title">Compare locations</div>', unsafe_allow_html=True)
        st.caption("Which candidate site could have the strongest impact?")
        st.info("Each card is a possible new facility location. The best option is not just the biggest population: it balances residents helped, distance saved and how many neighbourhoods improve.")
        cols = st.columns(len(candidates))
        for col, (_, row) in zip(cols, candidates.iterrows()):
            with col:
                st.markdown(
                    f"""
                    <div class="card">
                        <h3>{row['candidate'].title()}</h3>
                        <div class="priority-med">{impact_label(row['overall_impact_score'])}</div>
                        <p><strong>Estimated residents helped:</strong> {int(row['population_benefited']):,}</p>
                        <p><strong>Average distance saved:</strong> {simple_distance(row['mean_distance_reduction_m'])}</p>
                        <p><strong>Areas improved:</strong> {int(row['neighbourhoods_improved'])}</p>
                        <p><strong>Overall impact:</strong> {row['overall_impact_score']:.0f}/100</p>
                        <p>{explain_candidate(row)}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("Select", key=f"select-{row['candidate']}", use_container_width=True):
                    st.session_state.selected_candidate = row["candidate"]
                    st.rerun()

        explained = selected_candidate(candidates)
        st.markdown('<div class="section-title">Explainability</div>', unsafe_allow_html=True)
        st.caption("Why does the selected candidate receive this impact score?")
        st.code(impact_formula_text(), language="text")
        e1, e2 = st.columns([0.65, 1.35], gap="large")
        with e1:
            st.metric("Selected candidate", explained["candidate"].title())
            st.metric("Impact score", f"{explained['overall_impact_score']:.0f}/100")
            st.metric(
                "Estimated residents helped",
                f"{int(explained['population_benefited']):,} +/- {int(explained['population_uncertainty']):,}",
            )
            st.dataframe(contribution_frame(explained), use_container_width=True, hide_index=True)
        with e2:
            contribution_chart = contribution_frame(explained)
            fig = px.bar(
                contribution_chart,
                x="Points",
                y="Metric",
                orientation="h",
                text="Points",
                color="Metric",
                color_discrete_sequence=["#17635a", "#2f80ed", "#f2a541", "#b54708", "#667085"],
            )
            fig.update_layout(showlegend=False, height=310, margin=dict(l=10, r=10, t=20, b=10), xaxis_range=[0, 45])
            fig.update_traces(texttemplate="%{text:.1f}", textposition="outside", cliponaxis=False)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-title">Side-by-side comparison</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        names = list(candidates["candidate"])
        st.session_state.compare_a = c1.selectbox("First candidate", names, index=0)
        st.session_state.compare_b = c2.selectbox("Second candidate", names, index=min(1, len(names) - 1))
        compare_names = []
        for name in [st.session_state.compare_a, st.session_state.compare_b]:
            if name not in compare_names:
                compare_names.append(name)
        compare = candidates.set_index("candidate").loc[compare_names].reset_index()
        a, b = st.columns(2)
        for col, (_, row) in zip([a, b], compare.iterrows()):
            col.markdown(
                f"""
                <div class="soft-card">
                    <strong>{row['candidate'].title()}</strong><br>
                    Impact: {impact_label(row['overall_impact_score'])} ({row['overall_impact_score']:.0f}/100)<br>
                    Estimated residents helped: {int(row['population_benefited']):,}<br>
                    Distance saved: {simple_distance(row['mean_distance_reduction_m'])}
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.dataframe(comparison_table(compare), use_container_width=True, hide_index=True)
        component_rows = []
        for _, row in compare.iterrows():
            candidate_frame = contribution_frame(row)
            candidate_frame["Candidate"] = row["candidate"].title()
            component_rows.append(candidate_frame)
        if component_rows:
            comparison_components = pd.concat(component_rows, ignore_index=True)
            fig = px.bar(
                comparison_components,
                x="Metric",
                y="Points",
                color="Candidate",
                barmode="group",
                text="Points",
                color_discrete_sequence=["#17635a", "#2f80ed"],
            )
            fig.update_layout(height=330, margin=dict(l=10, r=10, t=20, b=10), yaxis_title="Score contribution")
            fig.update_traces(texttemplate="%{text:.1f}", textposition="outside", cliponaxis=False)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-title">Sensitivity check</div>', unsafe_allow_html=True)
        sensitivity = sensitivity_analysis(candidates)
        st.dataframe(sensitivity, use_container_width=True, hide_index=True)
        best_now = candidates.iloc[0]["candidate"].title()
        if (sensitivity["Best candidate"] == best_now).all():
            st.success(f"If population weight changes by +/-10%, the best candidate remains: {best_now}.")
        else:
            st.warning("The preferred candidate changes when the population weight moves by +/-10%, so the ranking should be reviewed.")

        st_folium(make_priority_map(boundaries, priority_data, service_type, candidates, st.session_state.selected_candidate, services), height=480, use_container_width=True)
        nav = st.columns([1, 1, 1])
        nav[0].button("Back", use_container_width=True, on_click=previous_step)
        nav[2].button("Get recommendation", type="primary", use_container_width=True, on_click=next_step)

    else:
        candidates = scenario_candidates(services, boundaries, population, access, service_type, objective, priority_data)
        chosen = selected_candidate(candidates)
        is_top_recommendation = chosen["candidate"] == candidates.iloc[0]["candidate"]
        robustness = robustness_summary(candidates, chosen)
        before_after = before_after_indicators(access, priority_data, chosen, service_type)
        monitoring = monitoring_panel(services, boundaries, population, access, priority_data, service_type, robustness)
        st.markdown('<div class="section-title">Recommendation</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">Recommendation Summary</div>', unsafe_allow_html=True)
        st.success(recommendation_reason(chosen, is_top=is_top_recommendation))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Recommended location", chosen["candidate"].title())
        c2.metric("Estimated residents helped", f"{int(chosen['population_benefited']):,} +/- {int(chosen['population_uncertainty']):,}")
        c3.metric("Overall impact", f"{chosen['overall_impact_score']:.0f}/100")
        c4.metric("Distance saved", f"{chosen['mean_distance_reduction_m']:.0f} m +/- {int(chosen['distance_uncertainty_m'])}")
        st.caption("Residents helped and distance saved are scenario estimates, so uncertainty is shown as a simple robustness band.")

        st.markdown('<div class="section-title">Why was this location selected?</div>', unsafe_allow_html=True)
        st.caption("Contribution of each criterion to the final decision score.")
        st.code(impact_formula_text(), language="text")
        explanation_left, explanation_right = st.columns([0.8, 1.2], gap="large")
        contribution_share = contribution_share_frame(chosen)
        with explanation_left:
            st.dataframe(contribution_share, use_container_width=True, hide_index=True)
            top_driver = contribution_share.sort_values("Points", ascending=False).iloc[0]
            st.info(
                f"The recommendation is mainly driven by {top_driver['Metric'].lower()}, "
                f"which contributes {top_driver['Contribution']:.1f}% of the selected candidate score."
            )
        with explanation_right:
            fig = px.bar(
                contribution_share,
                x="Points",
                y="Metric",
                orientation="h",
                text="Points",
                color="Metric",
                color_discrete_sequence=["#17635a", "#2f80ed", "#f2a541", "#b54708", "#667085"],
            )
            fig.update_layout(showlegend=False, height=320, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="Impact points")
            fig.update_traces(texttemplate="%{text:.1f}", textposition="outside", cliponaxis=False)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-title">Recommendation Robustness</div>', unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)
        r1.metric("Stability score", f"{robustness['stability_score']:.2f}")
        r2.metric("Confidence", f"{robustness['confidence']}%")
        r3.metric("Sensitivity", robustness["status"])
        st.caption("A stability score close to 1 indicates that the recommendation ranking changes very little under small weight variations.")
        st.dataframe(robustness["sensitivity"], use_container_width=True, hide_index=True)

        st.markdown('<div class="section-title">Comparison with other candidates</div>', unsafe_allow_html=True)
        top_candidates = candidate_comparison_summary(candidates)
        st.dataframe(top_candidates, use_container_width=True, hide_index=True)
        if len(candidates) > 1:
            margin = candidates.iloc[0]["overall_impact_score"] - candidates.iloc[1]["overall_impact_score"]
            st.info(
                f"The leading candidate outperforms the second-best alternative by {margin:.1f} impact points. "
                "A small gap means alternatives should be reviewed carefully with local planning constraints."
            )

        st.markdown('<div class="section-title">Before vs After</div>', unsafe_allow_html=True)
        st.dataframe(before_after, use_container_width=True, hide_index=True)
        left, right = st.columns([1, 1], gap="large")
        with left:
            st.markdown('<div class="card"><h3>Before</h3><p>Priority areas show current accessibility pressure for the selected service.</p></div>', unsafe_allow_html=True)
            st_folium(make_priority_map(boundaries, priority_data, service_type, services=services), height=430, use_container_width=True)
        with right:
            st.markdown('<div class="card"><h3>After</h3><p>The selected candidate is added as a decision-support scenario.</p></div>', unsafe_allow_html=True)
            st_folium(make_priority_map(boundaries, priority_data, service_type, candidates, chosen["candidate"], services), height=430, use_container_width=True)

        st.markdown('<div class="section-title">Model Monitoring</div>', unsafe_allow_html=True)
        st.dataframe(monitoring, use_container_width=True, hide_index=True)
        st.caption("Monitoring checks are evaluated on the current dataset and are intended to support transparent deployment, not to replace municipal validation.")

        st.markdown('<div class="section-title">Traceability</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="card"><h3>Planning caveat</h3><p>This is not an official planning recommendation. It should be reviewed with land availability, budgets, mobility constraints and local knowledge.</p></div>',
            unsafe_allow_html=True,
        )
        summary = intervention_summary(service_label, objective, priority_data, candidates, chosen)
        
        # EDM: Create audit trail for traceability
        audit_trail = create_audit_trail(
            service_type,
            objective,
            chosen,
            candidates,
            priority_data,
            {
                "mean_accessibility": float(access["accessibility_score"].mean()),
                "neighborhoods": len(access)
            }
        )
        audit_json = json.dumps(audit_trail, indent=2, default=str)
        
        c1, c2, c3, c4 = st.columns(4)
        if c1.button("View another scenario", use_container_width=True):
            st.session_state.step = 0
            st.session_state.selected_candidate = None
            st.rerun()
        c2.download_button("📄 Download summary", summary, file_name="cityaccess_intervention_summary.txt", mime="text/plain", use_container_width=True)
        c3.download_button("📋 Download audit trail", audit_json, file_name=f"audit_trail_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", mime="application/json", use_container_width=True)
        if c4.button("About this tool", use_container_width=True):
            set_page("About")
            st.rerun()
        nav = st.columns([1, 1, 1])
        nav[0].button("Back", use_container_width=True, on_click=previous_step)


def explore_page(services, boundaries, access):
    st.markdown('<div class="section-title">City data</div>', unsafe_allow_html=True)
    st.caption("Inspect the underlying open data and neighbourhood profiles.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Services loaded", len(services))
    c2.metric("Neighbourhoods", len(access))
    c3.metric("Average accessibility level", f"{access['accessibility_score'].mean():.1f}")
    service_filter = st.selectbox("Service type", CATEGORIES)
    left, right = st.columns([1.2, 1])
    with left:
        st_folium(service_map(services[services["category"] == service_filter], boundaries), height=500, use_container_width=True)
    with right:
        neighbourhood = st.selectbox("Neighbourhood profile", access["neighbourhood"].sort_values())
        row = access[access["neighbourhood"] == neighbourhood].iloc[0]
        st.markdown(
            f"""
            <div class="card">
                <h3>{neighbourhood.title()}</h3>
                <p><strong>Population:</strong> {int(row['population']):,}</p>
                <p><strong>Accessibility level:</strong> {row['accessibility_score']:.1f}/100</p>
                <p><strong>Nearest selected service:</strong> {row[f'{service_filter}_nearest_m']:.0f} m</p>
                <p><strong>Services nearby:</strong> {int(row[f'{service_filter}_within_radius'])}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(access.sort_values("accessibility_score", ascending=False), use_container_width=True, height=300)


def methods_page(access):
    st.markdown('<div class="section-title">Technical details</div>', unsafe_allow_html=True)
    st.caption("Model checks and advanced outputs are kept outside the main workflow.")
    clustered, sil, profiles = cluster_profiles(access, 4)
    anomalies = detect_anomalies(access, 0.15)
    c1, c2 = st.columns(2)
    c1.metric("K-means silhouette score", f"{sil:.3f}")
    c2.metric("Unusual service patterns found", int(anomalies["is_anomaly"].sum()))
    st.plotly_chart(px.scatter(clustered, x="proximity", y="services_per_10k", color="cluster", hover_name="neighbourhood"), use_container_width=True)
    st.dataframe(profiles, use_container_width=True)
    st.plotly_chart(px.line(silhouette_by_k(access), x="k", y="silhouette", markers=True), use_container_width=True)
    st.dataframe(anomaly_sensitivity(detect_anomalies, access), use_container_width=True)


def methodology_page(access, services):
    st.markdown('<div class="section-title">How calculations work</div>', unsafe_allow_html=True)
    st.markdown(
        """
        The main workflow hides formulas, but the application uses a full data science pipeline:

        - cleaning missing values, duplicates, categories, coordinates and geometries;
        - spatial joins between facilities and neighbourhood boundaries;
        - distance calculations with projected coordinates;
        - an explainable accessibility score from normalised indicators;
        - K-means clustering, Isolation Forest and sensitivity checks;
        - scenario simulation for new facilities;
        - simplified monitoring for distribution changes and data drift.
        """
    )
    latest = access.copy()
    latest["total_services"] = latest["total_services"] * 1.05
    latest["accessibility_score"] = latest["accessibility_score"] * 0.98
    report, cat_result = drift_report(access, latest, services["category"], services.sample(frac=1, random_state=7)["category"])
    st.subheader("Monitoring example")
    st.dataframe(report, use_container_width=True)
    st.write("Category distribution test:", cat_result)
    st.subheader("Normalised indicators")
    st.dataframe(access[["neighbourhood", "services_per_10k_norm", "nearest_norm", "nearby_norm", "diversity_norm", "accessibility_score"]], use_container_width=True)


def limitations_page(is_demo):
    st.markdown('<div class="section-title">Known constraints</div>', unsafe_allow_html=True)
    if is_demo:
        st.warning("The current run uses synthetic data. Do not interpret the results as real planning evidence.")
    st.markdown(
        """
        This application is a decision-support prototype, not an official planning tool.

        Results depend on the quality, date and coverage of the input data. The recommended intervention does not account for budgets, land ownership, service capacity, staffing, accessibility by public transport, political constraints or field validation. Any real decision should combine this analysis with municipal expertise and local consultation.
        """
    )


def about_page(services, boundaries, access, is_demo):
    service_type = st.session_state.service_type
    objective = st.session_state.objective
    priority_data = service_priority(access, service_type, objective)

    st.markdown('<div class="section-title">About this tool</div>', unsafe_allow_html=True)
    st.markdown(
        """
        CityAccess Valencia helps municipal analysts and planners find underserved areas and compare possible locations for new public facilities.

        The main assessment is intentionally simple. The sections below keep the evidence, checks and constraints available without overwhelming the planning workflow.
        """
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Services", len(services))
    c2.metric("Neighbourhoods", len(access))
    c3.metric("Data source", "Valencia open data" if not is_demo else "Demo data")

    with st.expander("How to read the main numbers", expanded=True):
        st.markdown(
            """
            - **Priority level**: how urgent it is to review an area for the selected service. Higher means weaker access combined with local population need.
            - **Estimated residents helped**: residents weighted by how much their access would improve. It is an estimate, not a census count.
            - **Distance saved**: average reduction in distance to the selected service.
            - **Areas improved**: number of neighbourhoods that would become closer to that service.
            - **Overall impact**: a 0-100 comparison score used to rank candidate locations within the same scenario.
            """
        )

    # ========== EDM METRICS SECTION ==========

    with st.expander("🔍 Fairness Analysis - Equity Check", expanded=False):
        st.markdown("**Does the model treat all neighborhoods fairly, regardless of size?**")
        fairness = fairness_by_population_size(priority_data, service_type)
        
        col1, col2 = st.columns(2)
        col1.markdown(f"**Bias Status:** {fairness['bias_interpretation']}")
        col2.metric("Priority Gap", f"{fairness['priority_gap']} points")
        
        st.dataframe(fairness["fairness_table"], use_container_width=True, height=200)
        
        if fairness["bias_detected"]:
            st.warning(
                "⚠️ The model systematically prioritizes larger neighborhoods. "
                "This may reflect real accessibility gaps OR introduce unwanted bias. "
                "Consider reviewing how 'population' weighs in the scoring."
            )
        else:
            st.success("✅ Equitable treatment across all neighborhood sizes.")

    with st.expander("🔄 Concept Drift - Decision Stability", expanded=False):
        st.markdown("**Has the definition of 'priority' changed significantly?**")
        drift = detect_concept_drift(priority_data, service_type=service_type)
        
        col1, col2, col3 = st.columns(3)
        col1.markdown(f"**Status:** {drift['interpretation']}")
        col2.metric("Spearman ρ", drift["spearman_correlation"])
        col3.metric("p-value", drift["p_value"])
        
        st.markdown(drift['explanation'])
        
        if drift["drift_detected"]:
            st.error("🔴 Concept drift detected. The model's decision logic may have shifted.")
        else:
            st.success("🟢 Concept stable. Priority criteria remain consistent.")

    with st.expander("✅ Data Quality Report", expanded=False):
        st.markdown("**How complete and reliable is the input data?**")
        validation = validate_data_quality(services, boundaries, population, service_type)
        
        if validation["critical_issues"]:
            st.subheader("🚨 Critical Issues")
            for issue in validation["critical_issues"]:
                st.error(issue)
        
        if validation["warnings"]:
            st.subheader("⚠️ Warnings")
            for warning in validation["warnings"]:
                st.warning(warning)
        
        col1, col2 = st.columns(2)
        col1.metric("Data Completeness", f"{validation['data_completeness']}%")
        col2.metric("Valid", "✅ Yes" if validation["is_valid"] else "❌ No")
        
        st.markdown("**Services by Category:**")
        st.json(validation["service_distribution"])

    with st.expander("⚙️ Model Sensitivity Analysis", expanded=False):
        st.markdown("**How robust is the model to changes in feature weights?**")
        
        sens = comprehensive_sensitivity(access, service_type, BASE_WEIGHTS)
        
        col1, col2 = st.columns(2)
        col1.markdown(f"**Most Critical Weight:** {sens['most_critical_weight']}")
        col2.markdown(f"**Variance:** {sens['most_critical_std']} (higher = more sensitive)")
        
        st.markdown("Rank stability > 0.85 = **ROBUST**. Lower = rankings shift unpredictably with weight changes.")
        
        # Crear tabla pivote para visualización
        pivot_table = sens["details"].pivot_table(
            index="weight",
            columns="change_pct",
            values="rank_stability",
            aggfunc="first"
        )
        
        st.dataframe(pivot_table, use_container_width=True, height=250)
        
        # Resumen por peso
        st.markdown("**Stability Summary (mean ± std):**")
        st.dataframe(sens["summary"], use_container_width=True)

    # ========== NUEVA SECCIÓN: EVALUACIÓN AVANZADA ==========

    with st.expander("📊 Performance by Service Category", expanded=False):
        st.markdown("**Is accessibility equitable across different service types?**")
        
        perf = performance_by_service_category(access, services)
        st.markdown(f"**Status:** {perf['disparity_interpretation']}")
        st.metric("Mean Variance Between Categories", f"{perf['mean_variance_between_categories']} points")
        
        st.dataframe(perf["details"], use_container_width=True, height=250)

    with st.expander("🎯 Model vs Baseline Comparison", expanded=False):
        st.markdown("**Does the model perform better than a naive approach?**")
        
        baseline = baseline_comparison(priority_data, service_type)
        
        col1, col2 = st.columns(2)
        col1.metric("Correlation vs Random", baseline["model_vs_equal_correlation"])
        col2.markdown(f"**Model Quality:** {baseline['interpretation']}")
        
        col1, col2 = st.columns(2)
        col1.markdown("**Current Model Top 5:**")
        col1.markdown("\n".join([f"- {n}" for n in baseline["current_top5_neighbourhoods"]]))
        
        col2.markdown("**Baseline Top 5 (Coverage Only):**")
        col2.markdown("\n".join([f"- {n}" for n in baseline["baseline_top5_neighbourhoods"]]))
        
        if baseline["model_vs_equal_correlation"] < 0.3:
            st.success("✅ Model adds significant value over random selection")
        else:
            st.warning("⚠️ Model similar to baseline - may need refinement")

    with st.expander("🔍 Feature Importance Analysis", expanded=False):
        st.markdown("**Which variables have the most impact on differentiation?**")
        
        feat_imp = feature_importance_approximation(priority_data)
        st.markdown(feat_imp["interpretation"])
        
        st.dataframe(feat_imp["details"], use_container_width=True, height=250)

    with st.expander("⚠️ Anomaly Detection Audit", expanded=False):
        st.markdown("**How well does the anomaly detector identify outliers?**")
        
        from src.anomaly_detection import detect_anomalies
        anomalies = detect_anomalies(priority_data, contamination=0.15)
        
        anom_audit = anomaly_detection_audit(priority_data, anomalies)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Neighborhoods", anom_audit["total_neighborhoods"])
        col2.metric("Anomalies Detected", f"{anom_audit['anomalies_detected']} ({anom_audit['anomaly_percentage']}%)")
        col3.metric("Top Distinguishing Feature", anom_audit["top_distinguishing_feature"])
        
        st.markdown(anom_audit["interpretation"])
        
        if anom_audit["characteristic_differences"] is not None and len(anom_audit["characteristic_differences"]) > 0:
            st.markdown("**How anomalies differ from normal neighborhoods:**")
            st.dataframe(anom_audit["characteristic_differences"], use_container_width=True, height=200)

    with st.expander("⚙️ Model Configuration & Reproducibility", expanded=False):
        st.markdown("**Complete technical configuration for reproducibility**")
        
        config = model_configuration_summary(radius, BASE_WEIGHTS, contamination=0.15)
        
        # Mostrar config en formato expandible
        st.markdown("**1. Spatial Analysis**")
        for key, val in config["spatial_analysis"].items():
            st.caption(f"- {key}: {val}")
        
        st.markdown("**2. Accessibility Scoring**")
        st.json(config["accessibility_scoring"])
        
        st.markdown("**3. Anomaly Detection**")
        for key, val in config["anomaly_detection"].items():
            if isinstance(val, list):
                st.caption(f"- {key}: {len(val)} features")
            else:
                st.caption(f"- {key}: {val}")
        
        st.markdown("**4. Clustering Configuration**")
        for key, val in config["clustering"].items():
            st.caption(f"- {key}: {val}")
        
        st.markdown("**5. Data Quality Standards**")
        st.json(config["data_quality"])
        
        st.markdown("**6. Reproducibility Seeds**")
        st.json(config["reproducibility"])

    # ========== END ADVANCED EDM METRICS ==========

    with st.expander("City data and neighbourhood profiles"):
        explore_page(services, boundaries, access)

    with st.expander("Technical details"):
        methods_page(access)

    with st.expander("How calculations work"):
        methodology_page(access, services)

    with st.expander("Known constraints"):
        limitations_page(is_demo)


init_state()
services, boundaries, population, is_demo, data_status = load_inputs()
radius = 1000
services_joined, neighbourhoods, metrics, access = build_analysis(services, boundaries, population, radius)
top_navigation(data_status, is_demo)

page = st.session_state.page
if page == "Home":
    home_page(boundaries, services)
elif page == "Assessment":
    assessment_page(services, boundaries, population, access)
else:
    about_page(services, boundaries, access, is_demo)
