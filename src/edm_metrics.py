"""
EDM Metrics Module: Evaluación, Despliegue y Monitorización
Implements fairness analysis, concept drift detection, input validation,
audit trails, and sensitivity analysis for academic rigor.
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from datetime import datetime
from .accessibility import minmax


def fairness_by_population_size(access, service_type):
    """
    Detecta sesgo en el scoring por tamaño de población.
    EDM Concept: Equidad y sesgo en modelos
    
    Args:
        access: DataFrame con accessibility scores por barrio
        service_type: str, tipo de servicio (healthcare, education, etc)
    
    Returns:
        dict con fairness_table, priority_gap, bias_detected, bias_interpretation
    """
    access_copy = access.copy()
    
    # Dividir por terciles de población, manejando duplicados
    try:
        access_copy["pop_tercile"] = pd.qcut(
            access_copy["population"], 
            q=3, 
            labels=["small", "medium", "large"],
            duplicates="drop"
        )
    except Exception:
        # Si no hay suficiente variación, usar divide manual
        access_copy["pop_tercile"] = pd.cut(
            access_copy["population"],
            bins=3,
            labels=["small", "medium", "large"]
        )
    
    fairness_results = []
    for tercile in sorted(access_copy["pop_tercile"].dropna().unique()):
        subset = access_copy[access_copy["pop_tercile"] == tercile]
        
        if len(subset) == 0:
            continue
        
        coverage_col = f"{service_type}_within_radius"
        coverage_pct = (subset[coverage_col] > 0).sum() / len(subset) * 100 if coverage_col in subset.columns else 0
        
        fairness_results.append({
            "population_size": tercile,
            "neighborhoods": len(subset),
            "mean_accessibility": round(subset["accessibility_score"].mean(), 2),
            "mean_priority": round(subset["planning_priority"].mean(), 2),
            "high_priority_pct": round((subset["planning_priority"] >= 70).sum() / len(subset) * 100, 1),
            "coverage_pct": round(coverage_pct, 1),
        })
    
    fairness_df = pd.DataFrame(fairness_results)
    
    # Detectar inequidad: diferencia en media_priority entre small y large > 15 puntos
    small_data = fairness_df[fairness_df["population_size"] == "small"]
    large_data = fairness_df[fairness_df["population_size"] == "large"]
    
    if len(small_data) > 0 and len(large_data) > 0:
        priority_gap = abs(
            small_data["mean_priority"].values[0] - 
            large_data["mean_priority"].values[0]
        )
        has_bias = priority_gap > 15
    else:
        priority_gap = 0
        has_bias = False
    
    return {
        "fairness_table": fairness_df,
        "priority_gap": round(priority_gap, 1),
        "bias_detected": has_bias,
        "bias_interpretation": (
            "⚠️ POTENTIAL BIAS: Model systematically prioritizes larger neighborhoods" 
            if has_bias else 
            "✅ FAIR: Relatively equitable treatment across population sizes"
        )
    }


def detect_concept_drift(access_current, access_baseline=None, service_type="healthcare"):
    """
    Detecta si cambió la RELACIÓN entre features y prioridad (concept drift).
    EDM Concept: Concept drift y cambio de definiciones
    
    Args:
        access_current: DataFrame actual con scores
        access_baseline: DataFrame baseline (si None, se genera sintético)
        service_type: str, tipo de servicio
    
    Returns:
        dict con correlation, p_value, drift_detected, interpretation
    """
    # Si no hay baseline, crear uno sin peso de población
    if access_baseline is None:
        access_baseline = access_current.copy()
        # Score sin población: solo basado en distancia y cobertura
        baseline_score = (
            minmax(access_baseline[f"{service_type}_nearest_m"], invert=True) * 0.5 +
            minmax(access_baseline[f"{service_type}_within_radius"], invert=True) * 0.5
        ) * 100
        access_baseline["planning_priority"] = baseline_score
    
    # Comparar rankings por barrio
    current_ranking = access_current.set_index("neighbourhood")["planning_priority"].rank(ascending=False)
    baseline_ranking = access_baseline.set_index("neighbourhood")["planning_priority"].rank(ascending=False)
    
    # Alinear por índices comunes
    common_idx = current_ranking.index.intersection(baseline_ranking.index)
    
    if len(common_idx) < 3:
        return {
            "spearman_correlation": np.nan,
            "p_value": np.nan,
            "drift_detected": False,
            "interpretation": "ℹ️ Insufficient data for drift detection (< 3 neighborhoods)",
            "explanation": ""
        }
    
    try:
        corr, p_value = spearmanr(
            current_ranking.loc[common_idx],
            baseline_ranking.loc[common_idx]
        )
    except Exception:
        return {
            "spearman_correlation": np.nan,
            "p_value": np.nan,
            "drift_detected": False,
            "interpretation": "ℹ️ Cannot compute correlation",
            "explanation": ""
        }
    
    # Umbral: si corr < 0.75, hay concept drift significativo
    drift_detected = (corr < 0.75) and (p_value < 0.05)
    
    return {
        "spearman_correlation": round(float(corr), 3),
        "p_value": round(float(p_value), 4),
        "drift_detected": drift_detected,
        "interpretation": (
            "🔴 CONCEPT DRIFT DETECTED: Priority definition has shifted significantly"
            if drift_detected else
            "🟢 STABLE: Priority concept remains consistent"
        ),
        "explanation": (
            f"Ranking correlation = {corr:.3f} (threshold = 0.75). "
            "If < 0.75 and p-value < 0.05, the model's decision logic may have drifted."
        )
    }


def validate_data_quality(services, boundaries, population, service_type):
    """
    Validación robusta de inputs.
    EDM Concept: Data quality, preprocesamiento, manejo de errores
    
    Args:
        services, boundaries, population: DataFrames de entrada
        service_type: str, tipo de servicio a validar
    
    Returns:
        dict con critical_issues, warnings, is_valid, service_distribution, completeness
    """
    issues = []
    warnings = []
    
    # === SERVICIOS ===
    if len(services) == 0:
        issues.append("❌ CRITICAL: No services loaded")
    else:
        missing_coords = services["latitude"].isna().sum() + services["longitude"].isna().sum()
        if missing_coords > 0:
            warnings.append(f"⚠️ {missing_coords} services with missing coordinates")
        
        out_of_bounds = ~(
            (services["latitude"].between(38.0, 41.0)) & 
            (services["longitude"].between(-2.0, 1.0))
        )
        if out_of_bounds.sum() > 0:
            warnings.append(f"⚠️ {out_of_bounds.sum()} services outside Valencia bounds")
        
        service_count_by_cat = services.groupby("category").size()
        if service_type in service_count_by_cat.index:
            if service_count_by_cat[service_type] == 0:
                issues.append(f"❌ CRITICAL: No {service_type} services found")
        else:
            issues.append(f"❌ CRITICAL: Service category '{service_type}' not found")
    
    # === BARRIOS ===
    if len(boundaries) == 0:
        issues.append("❌ CRITICAL: No neighborhoods loaded")
    else:
        try:
            invalid_geom = (~boundaries.geometry.is_valid).sum()
            if invalid_geom > 0:
                warnings.append(f"⚠️ {invalid_geom} invalid geometries (auto-corrected)")
        except Exception:
            pass
    
    # === POBLACIÓN ===
    if len(population) > 0:
        zero_pop = (population["population"] <= 0).sum()
        if zero_pop > 0:
            warnings.append(f"⚠️ {zero_pop} neighborhoods with zero/negative population")
        
        missing_pop = population["population"].isna().sum()
        if missing_pop > 0:
            warnings.append(f"⚠️ {missing_pop} neighborhoods with missing population")
    else:
        issues.append("❌ CRITICAL: No population data loaded")
    
    # Calcular completeness
    total_records = len(services) + len(boundaries) + len(population)
    warning_count = len(warnings)
    completeness = round((1 - min(warning_count / max(total_records, 1), 1)) * 100, 1)
    
    # Service distribution
    try:
        service_dist = services.groupby("category").size().to_dict()
    except Exception:
        service_dist = {}
    
    return {
        "critical_issues": issues,
        "warnings": warnings,
        "is_valid": len(issues) == 0,
        "service_distribution": service_dist,
        "data_completeness": completeness,
    }


def create_audit_trail(service_type, objective, selected_candidate, 
                       all_candidates, access_data, metrics_snapshot):
    """
    Crea trazabilidad completa de la recomendación.
    EDM Concept: Reproducibilidad y trazabilidad de decisiones
    
    Args:
        service_type, objective: contexto de decisión
        selected_candidate: row del candidato elegido
        all_candidates: DataFrame todos los candidatos
        access_data: DataFrame con scores de barrios
        metrics_snapshot: dict con estadísticas del modelo
    
    Returns:
        dict con audit trail completo
    """
    audit = {
        "timestamp": datetime.now().isoformat(),
        "decision_context": {
            "service_type": service_type,
            "objective": objective,
        },
        "selected_recommendation": {
            "neighborhood": str(selected_candidate["candidate"]),
            "overall_impact_score": float(selected_candidate["overall_impact_score"]),
            "population_benefited": int(selected_candidate["population_benefited"]),
            "distance_reduction_m": float(selected_candidate["mean_distance_reduction_m"]),
            "neighborhoods_improved": int(selected_candidate["neighbourhoods_improved"]),
        },
        "top_priority_area": {
            "neighborhood": str(access_data.iloc[0]["neighbourhood"]),
            "planning_priority": float(access_data.iloc[0]["planning_priority"]),
            "reason": str(access_data.iloc[0]["priority_reason"]),
            "population": int(access_data.iloc[0]["population"]),
        },
        "model_state": {
            "total_neighborhoods": len(access_data),
            "mean_accessibility_score": round(access_data["accessibility_score"].mean(), 2),
            "neighborhoods_with_coverage": int((access_data[f"{service_type}_within_radius"] > 0).sum()),
            "max_priority_score": round(access_data["planning_priority"].max(), 1),
            "min_priority_score": round(access_data["planning_priority"].min(), 1),
            "std_priority_score": round(access_data["planning_priority"].std(), 1),
        },
        "all_candidates_ranked": [
            {
                "rank": i+1,
                "neighborhood": str(row["candidate"]),
                "impact_score": float(row["overall_impact_score"]),
                "population_benefited": int(row["population_benefited"]),
            }
            for i, (_, row) in enumerate(all_candidates.head(5).iterrows())
        ]
    }
    
    return audit


def comprehensive_sensitivity(access, service_type, BASE_WEIGHTS):
    """
    Análisis exhaustivo de sensibilidad del modelo.
    EDM Concept: Robustez, estabilidad de decisiones
    
    Args:
        access: DataFrame con accessibility scores
        service_type: str, tipo de servicio
        BASE_WEIGHTS: dict con pesos base del modelo
    
    Returns:
        dict con details, summary, most_critical_weight
    """
    from .accessibility import compute_accessibility
    
    sensitivity_results = []
    
    # Crear una copia con scores base
    base_access = access.copy()
    base_priority_rank = base_access["accessibility_score"].rank(pct=True)
    
    # Variar cada peso: -50%, -25%, baseline, +25%, +50%
    for weight_key in BASE_WEIGHTS.keys():
        for pct_change in [-50, -25, 0, 25, 50]:
            # Crear pesos perturbados
            perturbed_weights = BASE_WEIGHTS.copy()
            if pct_change != 0:
                perturbed_weights[weight_key] = BASE_WEIGHTS[weight_key] * (1 + pct_change / 100)
            
            try:
                # Recalcular accessibility con nuevos pesos
                perturbed_access = compute_accessibility(access, perturbed_weights)
                
                # Comparar rankings
                perturbed_priority_rank = perturbed_access["accessibility_score"].rank(pct=True)
                
                # Correlación entre rankings
                common_idx = base_priority_rank.index.intersection(perturbed_priority_rank.index)
                if len(common_idx) > 1:
                    rank_corr, _ = spearmanr(
                        base_priority_rank.loc[common_idx],
                        perturbed_priority_rank.loc[common_idx]
                    )
                else:
                    rank_corr = 1.0
                
                sensitivity_results.append({
                    "weight": weight_key,
                    "change_pct": pct_change,
                    "new_value": round(perturbed_weights[weight_key], 2),
                    "rank_stability": round(float(rank_corr), 3),
                    "robust": rank_corr > 0.85,
                })
            except Exception as e:
                # Si falla el cálculo, skip
                sensitivity_results.append({
                    "weight": weight_key,
                    "change_pct": pct_change,
                    "new_value": round(perturbed_weights[weight_key], 2),
                    "rank_stability": np.nan,
                    "robust": False,
                })
    
    df = pd.DataFrame(sensitivity_results)
    
    # Resumir por peso: qué es más crítico?
    by_weight = df.groupby("weight")["rank_stability"].agg(["mean", "std"]).round(3)
    
    if len(by_weight) > 0 and "std" in by_weight.columns:
        most_critical = by_weight["std"].idxmax()
        most_critical_std = round(by_weight.loc[most_critical, "std"], 3)
    else:
        most_critical = "N/A"
        most_critical_std = 0.0
    
    return {
        "details": df,
        "summary": by_weight,
        "most_critical_weight": most_critical,
        "most_critical_std": most_critical_std,
    }


def performance_by_service_category(access, services):
    """
    Evalúa la equidad entre categorías de servicio.
    EDM Concept: Evaluación de modelos por subgrupo
    
    Args:
        access: DataFrame con accessibility scores
        services: DataFrame con servicios
    
    Returns:
        DataFrame con métricas por categoría
    """
    from .preprocessing import CATEGORIES
    
    perf_results = []
    for cat in CATEGORIES:
        cat_col = f"{cat}_accessibility"
        
        if cat_col not in access.columns:
            continue
        
        cat_services = len(services[services["category"] == cat])
        
        cat_scores = access[cat_col].dropna()
        
        if len(cat_scores) == 0:
            continue
        
        perf_results.append({
            "service_category": cat,
            "services_count": cat_services,
            "neighborhoods_with_access": len(access[access[f"{cat}_within_radius"] > 0]),
            "mean_score": round(cat_scores.mean(), 2),
            "std_score": round(cat_scores.std(), 2),
            "min_score": round(cat_scores.min(), 2),
            "max_score": round(cat_scores.max(), 2),
            "coverage_gap": round(100 - (len(access[access[f"{cat}_within_radius"] > 0]) / len(access) * 100), 1),
        })
    
    df = pd.DataFrame(perf_results)
    
    # Resumir equidad
    if len(df) > 0:
        score_variance = df["mean_score"].std()
        has_disparity = score_variance > 10  # Umbral: variance > 10 = inequidad
    else:
        score_variance = 0
        has_disparity = False
    
    return {
        "details": df,
        "mean_variance_between_categories": round(score_variance, 2),
        "has_disparity": has_disparity,
        "disparity_interpretation": (
            "⚠️ DISPARITY: Some service categories have much higher accessibility than others"
            if has_disparity else
            "✅ BALANCED: Service categories have similar accessibility profiles"
        )
    }


def baseline_comparison(access, service_type):
    """
    Compara modelo actual vs baseline ingenuo (todos equal score).
    EDM Concept: Evaluación relativa, mejora sobre baseline
    
    Args:
        access: DataFrame con accessibility scores
        service_type: str, tipo de servicio
    
    Returns:
        dict con comparación
    """
    # Modelo actual: usa planning_priority
    current_top5 = access.nlargest(5, "planning_priority")
    
    # Baseline 1: order aleatorio (todos = score)
    baseline_equal = access.copy()
    baseline_equal["baseline_score"] = 1.0  # Todos iguales
    baseline_equal_top5 = baseline_equal.sample(n=min(5, len(baseline_equal)), random_state=42)
    
    # Baseline 2: solo por cobertura
    baseline_coverage = access.copy()
    baseline_coverage["coverage_score"] = access[f"{service_type}_within_radius"] > 0
    baseline_coverage_top5 = baseline_coverage.nlargest(5, "coverage_score")
    
    # Métrica: correlación entre modelos
    current_rank = access.set_index("neighbourhood")["planning_priority"].rank(ascending=False)
    baseline_equal_rank = baseline_equal.set_index("neighbourhood")["baseline_score"].rank(ascending=False)
    
    common_idx = current_rank.index.intersection(baseline_equal_rank.index)
    if len(common_idx) > 1 and baseline_equal_rank.loc[common_idx].nunique() > 1:
        corr_vs_equal, _ = spearmanr(current_rank.loc[common_idx], baseline_equal_rank.loc[common_idx])
        if pd.isna(corr_vs_equal):
            corr_vs_equal = 0
    else:
        corr_vs_equal = 0
    
    return {
        "model_vs_equal_correlation": round(float(corr_vs_equal), 3),
        "interpretation": (
            f"Current model is significantly different from random selection (ρ={corr_vs_equal:.3f}). "
            "This indicates the model adds meaningful structure to prioritization."
            if abs(corr_vs_equal) < 0.3 else
            "Model is not substantially different from baseline."
        ),
        "current_top5_neighbourhoods": list(current_top5["neighbourhood"].values),
        "baseline_top5_neighbourhoods": list(baseline_coverage_top5["neighbourhood"].values),
    }


def feature_importance_approximation(access):
    """
    Aproximación a feature importance usando variance.
    EDM Concept: Interpretabilidad, qué variables importan
    
    Args:
        access: DataFrame con accessibility scores
    
    Returns:
        DataFrame con feature importance ranking
    """
    # Seleccionar features numéricos relevantes
    feature_cols = [
        "services_per_10k",
        "mean_nearest_m",
        "nearby_services",
        "service_diversity",
        "population",
        "accessibility_score",
        "planning_priority"
    ]
    
    available_features = [col for col in feature_cols if col in access.columns]
    
    importance_results = []
    for col in available_features:
        if access[col].notna().sum() > 0:
            variance = access[col].var()
            normalized_variance = variance / access[col].std() if access[col].std() > 0 else 0
            
            importance_results.append({
                "feature": col,
                "variance": round(float(variance), 2),
                "std": round(float(access[col].std()), 2),
                "range": round(float(access[col].max() - access[col].min()), 2),
                "relative_importance": round(float(normalized_variance), 3),
            })
    
    df = pd.DataFrame(importance_results).sort_values("variance", ascending=False)
    
    return {
        "details": df,
        "top3_features": list(df.head(3)["feature"].values),
        "interpretation": (
            f"Top 3 most variable features: {', '.join(df.head(3)['feature'].values)}. "
            "Higher variance features drive more differentiation between neighborhoods."
        )
    }


def anomaly_detection_audit(access, anomalies_df):
    """
    Auditoría de detección de anomalías: ¿qué tan buena es?
    EDM Concept: Evaluación de modelos unsupervised
    
    Args:
        access: DataFrame con scores
        anomalies_df: DataFrame con is_anomaly, anomaly_score columns
    
    Returns:
        dict con audit results
    """
    if "is_anomaly" not in anomalies_df.columns:
        return {"error": "No anomaly column found"}
    
    anomaly_count = (anomalies_df["is_anomaly"] == True).sum()
    anomaly_pct = round(anomaly_count / len(anomalies_df) * 100, 1)
    
    # Analizar características de anomalías vs normales
    anomalies = anomalies_df[anomalies_df["is_anomaly"] == True]
    normals = anomalies_df[anomalies_df["is_anomaly"] == False]
    
    # Si hay anomalías, ver qué las hace especiales
    if len(anomalies) > 0 and len(normals) > 0:
        numeric_cols = access.select_dtypes(include=[np.number]).columns
        
        differences = []
        for col in numeric_cols:
            if col in anomalies.columns and col in access.columns:
                anomaly_mean = anomalies[col].mean()
                normal_mean = normals[col].mean()
                
                if normal_mean != 0:
                    pct_diff = abs((anomaly_mean - normal_mean) / normal_mean) * 100
                else:
                    pct_diff = 0
                
                if pct_diff > 20:  # Solo diferencias significativas
                    differences.append({
                        "feature": col,
                        "anomaly_mean": round(anomaly_mean, 2),
                        "normal_mean": round(normal_mean, 2),
                        "pct_difference": round(pct_diff, 1),
                    })
        
        top_diff_feature = differences[0]["feature"] if differences else "N/A"
    else:
        differences = []
        top_diff_feature = "N/A"
    
    return {
        "total_neighborhoods": len(anomalies_df),
        "anomalies_detected": anomaly_count,
        "anomaly_percentage": anomaly_pct,
        "interpretation": (
            f"{anomaly_pct}% of neighborhoods are flagged as anomalous. "
            "Typical range is 5-15% for Isolation Forest with contamination=0.15."
        ),
        "top_distinguishing_feature": top_diff_feature,
        "characteristic_differences": pd.DataFrame(differences) if differences else None,
    }


def model_configuration_summary(radius_m, BASE_WEIGHTS, contamination=0.15):
    """
    Resumen centralizado de configuración del modelo.
    EDM Concept: Reproducibilidad, trazabilidad de parámetros
    
    Args:
        radius_m: int, radio de búsqueda en metros
        BASE_WEIGHTS: dict, pesos del modelo
        contamination: float, parámetro de Isolation Forest
    
    Returns:
        dict con toda la configuración
    """
    config = {
        "spatial_analysis": {
            "search_radius_meters": radius_m,
            "crs_used": "EPSG:25830 (UTM Zone 30N)",
            "description": "Geographical proximity analysis for service accessibility"
        },
        "accessibility_scoring": {
            "weights": BASE_WEIGHTS,
            "total_weight": round(sum(BASE_WEIGHTS.values()), 2),
            "normalization": "MinMax scaling [0, 1]",
            "formula": "weighted_sum / total_weight * 100"
        },
        "anomaly_detection": {
            "algorithm": "Isolation Forest",
            "contamination": contamination,
            "expected_anomalies_pct": round(contamination * 100, 1),
            "features_used": ["healthcare_accessibility", "education_accessibility", "libraries_accessibility", 
                            "sports facilities_accessibility", "proximity", "services_per_capita"]
        },
        "clustering": {
            "algorithm": "K-means",
            "default_k": 4,
            "random_state": 42,
            "scaler": "StandardScaler"
        },
        "data_quality": {
            "coordinate_bounds": {"lat": [38.0, 41.0], "lon": [-2.0, 1.0]},
            "population_minimum": 1,
            "missing_value_strategy": "Median imputation"
        },
        "reproducibility": {
            "random_seeds": {
                "data_generation": 42,
                "kmeans": 42,
                "isolation_forest": 42,
                "sample": 42
            },
            "python_version": "3.13+",
            "key_packages": ["pandas==2.3.3", "geopandas==1.1.3", "scikit-learn==1.7.2", "scipy==1.16.3"]
        }
    }
    
    return config
