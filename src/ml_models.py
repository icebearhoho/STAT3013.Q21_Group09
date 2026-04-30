from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from config import (
    ANOMALY_MAX_ROWS,
    CLUSTERING_MAX_ROWS,
    HOUSEHOLD_MODEL_FEATURES,
    OUTPUT_TABLES_DIR,
    RANDOM_STATE,
    SVM_MAX_ROWS,
)

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - optional dependency
    LGBMClassifier = None


ENCODED_TARGET_MODELS = {"xgboost", "lightgbm"}


def build_household_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def prepare_household_classification_data(df: pd.DataFrame, target_col: str = "wealth_class") -> tuple[pd.DataFrame, pd.Series]:
    features = [column for column in HOUSEHOLD_MODEL_FEATURES if column in df.columns]
    model_df = df[features + [target_col]].dropna(subset=[target_col]).copy()
    return model_df[features], model_df[target_col]


def _build_model_specs() -> dict[str, object]:
    specs: dict[str, object] = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
        "svm_rbf": SVC(kernel="rbf", gamma="scale"),
    }
    if XGBClassifier is not None:
        specs["xgboost"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
        )
    if LGBMClassifier is not None:
        specs["lightgbm"] = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            verbosity=-1,
        )
    return specs


def _fit_predict_model(name: str, estimator, preprocessor, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame):
    pipe = Pipeline([("preprocess", preprocessor), ("model", estimator)])
    report_target = None

    if name in ENCODED_TARGET_MODELS:
        encoder = LabelEncoder()
        y_train_enc = encoder.fit_transform(y_train)
        if name == "lightgbm":
            pipe.fit(X_train, y_train_enc)
            preds = encoder.inverse_transform(pipe.predict(X_test).astype(int))
        else:
            pipe.fit(X_train, y_train_enc)
            preds = encoder.inverse_transform(pipe.predict(X_test).astype(int))
        report_target = y_train.dtype
    elif name == "svm_rbf" and len(X_train) > SVM_MAX_ROWS:
        subset = X_train.sample(n=SVM_MAX_ROWS, random_state=RANDOM_STATE)
        y_subset = y_train.loc[subset.index]
        pipe.fit(subset, y_subset)
        preds = pipe.predict(X_test)
    else:
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

    return preds, pipe


def train_wealth_models(
    df: pd.DataFrame,
    target_col: str = "wealth_class",
    output_dir: Path = OUTPUT_TABLES_DIR,
) -> dict[str, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    X, y = prepare_household_classification_data(df, target_col=target_col)
    preprocessor = build_household_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model_specs = _build_model_specs()
    results = {}
    comparison_rows = []

    for name, estimator in model_specs.items():
        preds, pipe = _fit_predict_model(name, estimator, preprocessor, X_train, y_train, X_test)
        results[name] = {
            "accuracy": accuracy_score(y_test, preds),
            "macro_f1": f1_score(y_test, preds, average="macro"),
            "report": classification_report(y_test, preds),
            "pipeline": pipe,
        }
        comparison_rows.append(
            {
                "model": name,
                "accuracy": results[name]["accuracy"],
                "macro_f1": results[name]["macro_f1"],
            }
        )

    pd.DataFrame(comparison_rows).sort_values("macro_f1", ascending=False).to_csv(
        output_dir / "traditional_model_comparison.csv",
        index=False,
    )
    return results


def temporal_wealth_validation(
    df: pd.DataFrame,
    target_col: str = "wealth_class",
    output_dir: Path = OUTPUT_TABLES_DIR,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    X, y = prepare_household_classification_data(df, target_col=target_col)
    if "YEAR" not in X.columns:
        return pd.DataFrame()

    model_df = X.copy()
    model_df[target_col] = y.values
    model_df["YEAR"] = pd.to_numeric(model_df["YEAR"], errors="coerce")
    model_df = model_df.dropna(subset=["YEAR", target_col]).copy()
    years = sorted(model_df["YEAR"].astype(int).unique().tolist())
    rows = []

    for test_year in years[1:]:
        train_df = model_df[model_df["YEAR"] < test_year].copy()
        test_df = model_df[model_df["YEAR"] == test_year].copy()
        
        if train_df.empty or test_df.empty or train_df[target_col].nunique() < 2:
            continue

        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]
        preprocessor = build_household_preprocessor(X_train)

        for name, estimator in _build_model_specs().items():
            preds, _ = _fit_predict_model(name, estimator, preprocessor, X_train, y_train, X_test)
            rows.append(
                {
                    "train_years": " | ".join(str(year) for year in sorted(train_df["YEAR"].astype(int).unique())),
                    "test_year": int(test_year),
                    "model": name,
                    "accuracy": accuracy_score(y_test, preds),
                    "macro_f1": f1_score(y_test, preds, average="macro"),
                    "n_train": len(train_df),
                    "n_test": len(test_df),
                }
            )

    results = pd.DataFrame(rows).sort_values(["test_year", "macro_f1"], ascending=[True, False]) if rows else pd.DataFrame()
    results.to_csv(output_dir / "temporal_model_validation.csv", index=False)
    return results


def model_feature_importance(
    df: pd.DataFrame,
    target_col: str = "wealth_class",
    output_dir: Path = OUTPUT_TABLES_DIR,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    X, y = prepare_household_classification_data(df, target_col=target_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    preprocessor = build_household_preprocessor(X_train)
    model = Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)),
        ]
    )
    model.fit(X_train, y_train)
    importance = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=5,
        random_state=RANDOM_STATE,
        scoring="f1_macro",
    )
    rows = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    rows.to_csv(output_dir / "model_feature_importance.csv", index=False)
    return rows


def subgroup_model_performance(
    df: pd.DataFrame,
    target_col: str = "wealth_class",
    output_dir: Path = OUTPUT_TABLES_DIR,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    X, y = prepare_household_classification_data(df, target_col=target_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    preprocessor = build_household_preprocessor(X_train)
    preds, _ = _fit_predict_model(
        "random_forest",
        RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
        preprocessor,
        X_train,
        y_train,
        X_test,
    )

    test_df = X_test.copy()
    test_df[target_col] = y_test.values
    test_df["prediction"] = preds

    subgroup_cols = [col for col in ["URBAN_H", "YEAR", "GEO1_VN"] if col in test_df.columns]
    rows = []
    for col in subgroup_cols:
        for group_value, grp in test_df.groupby(col):
            if len(grp) < 30:
                continue
            rows.append(
                {
                    "subgroup_dimension": col,
                    "subgroup_value": group_value,
                    "n_obs": len(grp),
                    "accuracy": accuracy_score(grp[target_col], grp["prediction"]),
                    "macro_f1": f1_score(grp[target_col], grp["prediction"], average="macro"),
                }
            )

    result = pd.DataFrame(rows).sort_values(["subgroup_dimension", "macro_f1"], ascending=[True, False]) if rows else pd.DataFrame()
    result.to_csv(output_dir / "subgroup_model_performance.csv", index=False)
    return result


def run_unsupervised_tabular_analysis(
    df: pd.DataFrame,
    output_dir: Path = OUTPUT_TABLES_DIR,
) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    X, y = prepare_household_classification_data(df, target_col="wealth_class")
    clustering_df = X.copy()
    clustering_df["wealth_class"] = y.values

    cluster_sample = clustering_df.sample(n=min(CLUSTERING_MAX_ROWS, len(clustering_df)), random_state=RANDOM_STATE).copy()
    X_cluster = cluster_sample.drop(columns=["wealth_class"])
    preprocessor = build_household_preprocessor(X_cluster)
    X_cluster_proc = preprocessor.fit_transform(X_cluster)
    if hasattr(X_cluster_proc, "toarray"):
        X_cluster_proc = X_cluster_proc.toarray()

    kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
    clusters = kmeans.fit_predict(X_cluster_proc)
    cluster_sample["cluster"] = clusters
    cluster_summary = (
        cluster_sample.groupby(["cluster", "wealth_class"], as_index=False)
        .size()
        .rename(columns={"size": "n_households"})
        .sort_values(["cluster", "wealth_class"])
    )
    cluster_metrics = pd.DataFrame(
        [
            {
                "model": "kmeans",
                "n_clusters": 3,
                "silhouette_score": float(silhouette_score(X_cluster_proc, clusters)),
                "sample_size": len(cluster_sample),
            }
        ]
    )
    cluster_summary.to_csv(output_dir / "kmeans_cluster_summary.csv", index=False)
    cluster_metrics.to_csv(output_dir / "kmeans_cluster_metrics.csv", index=False)

    anomaly_cols = [col for col in ["HOUSEHOLD_SIZE", "AREA_PER_PERSON", "ROOMS_PER_PERSON", "YEAR"] if col in clustering_df.columns]
    anomaly_base = df[[col for col in anomaly_cols + ["wealth_index", "URBAN_H"] if col in df.columns]].copy()
    anomaly_base = anomaly_base.sample(n=min(ANOMALY_MAX_ROWS, len(anomaly_base)), random_state=RANDOM_STATE).copy()
    numeric_cols = [col for col in anomaly_base.columns if pd.api.types.is_numeric_dtype(anomaly_base[col])]
    anomaly_matrix = anomaly_base[numeric_cols].copy()
    anomaly_matrix = anomaly_matrix.fillna(anomaly_matrix.median(numeric_only=True))
    detector = IsolationForest(contamination=0.03, random_state=RANDOM_STATE)
    anomaly_flags = detector.fit_predict(anomaly_matrix)
    anomaly_base["is_anomaly"] = (anomaly_flags == -1).astype(int)
    anomaly_summary = []
    for group_col in [col for col in ["YEAR", "URBAN_H"] if col in anomaly_base.columns]:
        grp = (
            anomaly_base.groupby(group_col, as_index=False)
            .agg(
                n_obs=("is_anomaly", "size"),
                anomaly_rate=("is_anomaly", "mean"),
            )
            .assign(group_dimension=group_col)
            .rename(columns={group_col: "group_value"})
        )
        anomaly_summary.append(grp)
    anomaly_summary_df = pd.concat(anomaly_summary, ignore_index=True) if anomaly_summary else pd.DataFrame()
    anomaly_records = anomaly_base.sort_values("is_anomaly", ascending=False).head(500)
    anomaly_summary_df.to_csv(output_dir / "isolation_forest_summary.csv", index=False)
    anomaly_records.to_csv(output_dir / "isolation_forest_top_records.csv", index=False)

    return {
        "kmeans_cluster_summary": cluster_summary,
        "kmeans_cluster_metrics": cluster_metrics,
        "isolation_forest_summary": anomaly_summary_df,
        "isolation_forest_top_records": anomaly_records,
    }
