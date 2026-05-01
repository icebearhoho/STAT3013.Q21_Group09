from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from config import (
    BAYESIAN_NETWORK_FEATURES,
    BAYESIAN_NETWORK_MAX_ROWS,
    OUTPUT_REPORTS_DIR,
    OUTPUT_TABLES_DIR,
    RANDOM_STATE,
)

try:
    from pgmpy.estimators import BayesianEstimator, HillClimbSearch
    from pgmpy.inference import VariableElimination
    from pgmpy.models import DiscreteBayesianNetwork
except Exception:  # pragma: no cover
    BayesianEstimator = None
    HillClimbSearch = None
    VariableElimination = None
    DiscreteBayesianNetwork = None


def _discretize_numeric(series: pd.Series, labels: list[str]) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce")
    if clean.nunique(dropna=True) < 2:
        return clean.fillna(-1).astype(str)
    try:
        return pd.qcut(clean, q=len(labels), labels=labels, duplicates="drop").astype(str)
    except Exception:
        return pd.cut(clean, bins=len(labels), labels=labels, include_lowest=True).astype(str)


def prepare_bayesian_network_data(df: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in BAYESIAN_NETWORK_FEATURES if col in df.columns]
    data = df[cols].copy()
    if data.empty or "wealth_class" not in data.columns:
        return pd.DataFrame()

    numeric_3bin = {
        "HOUSEHOLD_SIZE": ["small", "medium", "large"],
        "AREA_PER_PERSON": ["low", "mid", "high"],
        "ROOMS_PER_PERSON": ["low", "mid", "high"],
    }
    for col, labels in numeric_3bin.items():
        if col in data.columns:
            data[col] = _discretize_numeric(data[col], labels)

    if "YEAR" in data.columns:
        data["YEAR"] = data["YEAR"].astype(str)
    if "GEO1_VN" in data.columns:
        top_regions = data["GEO1_VN"].astype(str).value_counts().head(12).index
        data["GEO1_VN"] = np.where(data["GEO1_VN"].astype(str).isin(top_regions), data["GEO1_VN"].astype(str), "OTHER")

    for col in data.columns:
        data[col] = data[col].astype(str).replace({"nan": "MISSING", "None": "MISSING", "<NA>": "MISSING"})

    return data.dropna(subset=["wealth_class"]).reset_index(drop=True)


def _cpd_summary(model) -> pd.DataFrame:
    rows = []
    for cpd in model.get_cpds():
        rows.append(
            {
                "variable": cpd.variable,
                "cardinality": cpd.variable_card,
                "evidence": " | ".join(cpd.variables[1:]) if len(cpd.variables) > 1 else "",
                "state_names": " | ".join(str(name) for name in cpd.state_names.get(cpd.variable, [])),
            }
        )
    return pd.DataFrame(rows)


def _scenario_analysis(inference, variables: list[str]) -> pd.DataFrame:
    scenarios = []
    if "wealth_class" not in variables:
        return pd.DataFrame()

    base_query = inference.query(["wealth_class"])
    base_probs = dict(zip(base_query.state_names["wealth_class"], base_query.values))
    scenarios.append(
        {
            "scenario": "baseline",
            "poor_prob": base_probs.get("poor", np.nan),
            "middle_prob": base_probs.get("middle", np.nan),
            "rich_prob": base_probs.get("rich", np.nan),
        }
    )

    evidence_specs = [
        ("urban_with_services", {"URBAN_H": "1", "ELECTRIC_H": "1.0", "WATSUP_H": "1.0"}),
        ("rural_without_services", {"URBAN_H": "0", "ELECTRIC_H": "0.0", "WATSUP_H": "0.0"}),
    ]
    for name, evidence in evidence_specs:
        filtered = {k: v for k, v in evidence.items() if k in variables}
        if not filtered:
            continue
        try:
            query = inference.query(["wealth_class"], evidence=filtered)
            probs = dict(zip(query.state_names["wealth_class"], query.values))
            scenarios.append(
                {
                    "scenario": name,
                    "evidence": " | ".join(f"{k}={v}" for k, v in filtered.items()),
                    "poor_prob": probs.get("poor", np.nan),
                    "middle_prob": probs.get("middle", np.nan),
                    "rich_prob": probs.get("rich", np.nan),
                }
            )
        except Exception:
            continue

    return pd.DataFrame(scenarios)


def run_bayesian_network_analysis(
    household_df: pd.DataFrame,
    output_dir: Path = OUTPUT_TABLES_DIR,
    report_dir: Path = OUTPUT_REPORTS_DIR,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    if HillClimbSearch is None:
        status = pd.DataFrame(
            [{"component": "bayesian_network", "status": "skipped", "reason": "pgmpy is not installed."}]
        )
        status.to_csv(report_dir / "bayesian_network_status.csv", index=False)
        return {}

    sample_n = min(BAYESIAN_NETWORK_MAX_ROWS, len(household_df))
    sample = household_df.sample(n=sample_n, random_state=RANDOM_STATE).copy() if len(household_df) > sample_n else household_df.copy()
    data = prepare_bayesian_network_data(sample)
    if data.empty:
        status = pd.DataFrame(
            [{"component": "bayesian_network", "status": "skipped", "reason": "No usable BN data after discretization."}]
        )
        status.to_csv(report_dir / "bayesian_network_status.csv", index=False)
        return {}

    train_df, test_df = train_test_split(
        data,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=data["wealth_class"],
    )

    search = HillClimbSearch(train_df)
    learned = search.estimate()
    model = DiscreteBayesianNetwork(learned.edges())
    model.fit(train_df)
    inference = VariableElimination(model)

    preds = []
    for _, row in test_df.iterrows():
        evidence = row.drop(labels=["wealth_class"]).to_dict()
        try:
            query = inference.query(["wealth_class"], evidence=evidence)
            labels = query.state_names["wealth_class"]
            pred = labels[int(np.argmax(query.values))]
        except Exception:
            pred = train_df["wealth_class"].mode().iloc[0]
        preds.append(pred)

    performance = {
        "model": "bayesian_network",
        "accuracy": accuracy_score(test_df["wealth_class"], preds),
        "macro_f1": f1_score(test_df["wealth_class"], preds, average="macro"),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "n_nodes": len(model.nodes()),
        "n_edges": len(model.edges()),
    }
    pd.DataFrame([performance]).to_csv(output_dir / "bayesian_network_prediction.csv", index=False)
    with open(output_dir / "bayesian_network_classification_report.txt", "w", encoding="utf-8") as handle:
        handle.write(classification_report(test_df["wealth_class"], preds))

    edges = pd.DataFrame(list(model.edges()), columns=["source", "target"])
    edges.to_csv(output_dir / "bayesian_network_structure_edges.csv", index=False)

    cpd_summary = _cpd_summary(model)
    cpd_summary.to_csv(output_dir / "bayesian_network_cpd_summary.csv", index=False)

    scenario = _scenario_analysis(inference, list(model.nodes()))
    scenario.to_csv(output_dir / "bayesian_network_scenario_analysis.csv", index=False)

    return {
        "performance": performance,
        "edges": edges,
        "cpd_summary": cpd_summary,
        "scenario": scenario,
    }

if __name__ == "__main__":
    print("Loading data...")
    # This automatically finds the correct folder no matter what
    data_path = Path(__file__).parent.parent / "data" / "processed" / "household_analysis_ready.csv"
    df = pd.read_csv(data_path)
    
    print("Running Bayesian Network...")
    run_bayesian_network_analysis(df)
    
    print("Finished successfully! Check your dashboard.")
