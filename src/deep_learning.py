from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

from config import DEEP_LEARNING_MAX_ROWS, OUTPUT_REPORTS_DIR, OUTPUT_TABLES_DIR, RANDOM_STATE
from ml_models import build_household_preprocessor, prepare_household_classification_data

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


if torch is not None:
    class TorchMLP(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, hidden_sizes: tuple[int, int] = (128, 64)):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_sizes[0]),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                nn.ReLU(),
                nn.Dropout(0.10),
                nn.Linear(hidden_sizes[1], output_dim),
            )

        def forward(self, x):
            return self.network(x)


    class AutoEncoder(nn.Module):
        def __init__(self, input_dim: int, latent_dim: int = 3):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 16),
                nn.ReLU(),
                nn.Linear(16, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 64),
                nn.ReLU(),
                nn.Linear(64, input_dim),
            )

        def forward(self, x):
            latent = self.encoder(x)
            reconstructed = self.decoder(latent)
            return reconstructed, latent
else:
    TorchMLP = None
    AutoEncoder = None


def _prepare_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return prepare_household_classification_data(df, target_col="wealth_class")


def _sample_household_df(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) <= DEEP_LEARNING_MAX_ROWS:
        return df.copy()
    return df.sample(n=DEEP_LEARNING_MAX_ROWS, random_state=RANDOM_STATE).copy()


def _encode_target(y: pd.Series) -> tuple[np.ndarray, LabelEncoder]:
    encoder = LabelEncoder()
    return encoder.fit_transform(y), encoder


def _fit_preprocessor(X_train: pd.DataFrame):
    preprocessor = build_household_preprocessor(X_train)
    transformed = preprocessor.fit_transform(X_train)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    return preprocessor, transformed.astype(np.float32)


def _transform_features(preprocessor, X: pd.DataFrame) -> np.ndarray:
    transformed = preprocessor.transform(X)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    return transformed.astype(np.float32)


def _train_torch_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_sizes: tuple[int, int] = (128, 64),
    max_epochs: int = 40,
    patience: int = 6,
) -> tuple[object, list[dict[str, float]]]:
    model = TorchMLP(X_train.shape[1], len(np.unique(y_train)), hidden_sizes=hidden_sizes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=512,
        shuffle=True,
    )
    best_state = None
    best_val_loss = float("inf")
    bad_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(batch_x)
            train_n += len(batch_x)

        model.eval()
        with torch.no_grad():
            val_logits = model(torch.tensor(X_val))
            val_loss = float(loss_fn(val_logits, torch.tensor(y_val)).item())
            val_preds = val_logits.argmax(dim=1).numpy()
            val_macro_f1 = float(f1_score(y_val, val_preds, average="macro"))

        train_loss = train_loss_sum / max(train_n, 1)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_macro_f1": val_macro_f1})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def _predict_torch(model, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X))
        probs = torch.softmax(logits, dim=1).numpy()
        preds = probs.argmax(axis=1)
    return preds, probs


def _fit_mlp_model(X_train_df: pd.DataFrame, y_train: pd.Series) -> tuple[dict[str, object], object, LabelEncoder]:
    y_encoded, encoder = _encode_target(y_train)
    preprocessor, X_train = _fit_preprocessor(X_train_df)
    X_subtrain, X_val, y_subtrain, y_val = train_test_split(
        X_train,
        y_encoded,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )

    if torch is not None:
        model, history = _train_torch_mlp(X_subtrain, y_subtrain, X_val, y_val)
        bundle = {"backend": "torch", "model": model, "preprocessor": preprocessor, "history": pd.DataFrame(history)}
    else:
        model = MLPClassifier(hidden_layer_sizes=(128, 64), early_stopping=True, random_state=RANDOM_STATE, max_iter=120)
        model.fit(X_train, y_encoded)
        bundle = {"backend": "sklearn_fallback", "model": model, "preprocessor": preprocessor, "history": pd.DataFrame()}
    return bundle, preprocessor, encoder


def _predict_mlp(bundle: dict[str, object], encoder: LabelEncoder, X_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = _transform_features(bundle["preprocessor"], X_df)
    if bundle["backend"] == "torch":
        pred_idx, probs = _predict_torch(bundle["model"], X)
    else:
        probs = bundle["model"].predict_proba(X)
        pred_idx = bundle["model"].predict(X)
    preds = encoder.inverse_transform(pred_idx.astype(int))
    return preds, probs


def _manual_permutation_importance(
    bundle: dict[str, object],
    encoder: LabelEncoder,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    baseline_preds, _ = _predict_mlp(bundle, encoder, X_test)
    baseline = f1_score(y_test, baseline_preds, average="macro")
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for feature in X_test.columns:
        scores = []
        for _ in range(5):
            permuted = X_test.copy()
            permuted[feature] = rng.permutation(permuted[feature].to_numpy())
            preds, _ = _predict_mlp(bundle, encoder, permuted)
            scores.append(baseline - f1_score(y_test, preds, average="macro"))
        rows.append({"feature": feature, "importance_mean": float(np.mean(scores)), "importance_std": float(np.std(scores))})
    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False)


def _fit_autoencoder(X: np.ndarray, latent_dim: int = 3, max_epochs: int = 30) -> tuple[object, float, pd.DataFrame]:
    model = AutoEncoder(X.shape[1], latent_dim=latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.tensor(X)), batch_size=512, shuffle=True)
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        loss_sum = 0.0
        n = 0
        for (batch_x,) in loader:
            optimizer.zero_grad()
            reconstructed, _ = model(batch_x)
            loss = loss_fn(reconstructed, batch_x)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(batch_x)
            n += len(batch_x)
        history.append({"epoch": epoch, "reconstruction_loss": loss_sum / max(n, 1)})

    model.eval()
    with torch.no_grad():
        reconstructed, latent = model(torch.tensor(X))
        mse = float(loss_fn(reconstructed, torch.tensor(X)).item())
    latent_df = pd.DataFrame(latent.numpy(), columns=[f"latent_{i+1}" for i in range(latent.shape[1])])
    return model, mse, latent_df


def _evaluate_representation(rep_df: pd.DataFrame, prefix: str, wealth_index: pd.Series, wealth_class: pd.Series) -> dict[str, float]:
    eval_df = rep_df.copy()
    eval_df["wealth_index"] = wealth_index.to_numpy()
    eval_df["wealth_class"] = wealth_class.to_numpy()
    features = [col for col in eval_df.columns if col.startswith(prefix)]
    out = {
        "corr_with_wealth_index": float(eval_df[[features[0], "wealth_index"]].corr(method="spearman").iloc[0, 1]),
        "explained_variance_ratio": np.nan,
    }
    rep_nonmissing = eval_df.dropna(subset=["wealth_class"]).copy()
    X_train, X_test, y_train, y_test = train_test_split(
        rep_nonmissing[features],
        rep_nonmissing["wealth_class"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=rep_nonmissing["wealth_class"],
    )
    clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    preds = clf.predict(X_test)
    out["downstream_accuracy"] = float(accuracy_score(y_test, preds))
    out["downstream_macro_f1"] = float(f1_score(y_test, preds, average="macro"))
    out["cluster_silhouette"] = float(silhouette_score(rep_nonmissing[features], KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10).fit_predict(rep_nonmissing[features])))
    return out


def _temporal_mlp_validation(sample_df: pd.DataFrame) -> pd.DataFrame:
    X, y = _prepare_feature_frame(sample_df)
    if "YEAR" not in X.columns:
        return pd.DataFrame()
    model_df = X.copy()
    model_df["wealth_class"] = y.values
    model_df["YEAR"] = pd.to_numeric(model_df["YEAR"], errors="coerce")
    model_df = model_df.dropna(subset=["YEAR", "wealth_class"]).copy()
    years = sorted(model_df["YEAR"].astype(int).unique().tolist())
    rows = []

    for test_year in years[1:]:
        train_df = model_df[model_df["YEAR"] < test_year].copy()
        test_df = model_df[model_df["YEAR"] == test_year].copy()
        
        if train_df.empty or test_df.empty or train_df["wealth_class"].nunique() < 2:
            continue
        bundle, _, encoder = _fit_mlp_model(train_df.drop(columns=["wealth_class"]), train_df["wealth_class"])
        preds, _ = _predict_mlp(bundle, encoder, test_df.drop(columns=["wealth_class"]))
        rows.append(
            {
                "model": "mlp_classifier",
                "backend": bundle["backend"],
                "train_years": " | ".join(str(year) for year in sorted(train_df["YEAR"].astype(int).unique())),
                "test_year": int(test_year),
                "accuracy": accuracy_score(test_df["wealth_class"], preds),
                "macro_f1": f1_score(test_df["wealth_class"], preds, average="macro"),
                "n_train": len(train_df),
                "n_test": len(test_df),
            }
        )
    return pd.DataFrame(rows)


def run_deep_learning_analysis(
    household_df: pd.DataFrame,
    output_dir: Path = OUTPUT_TABLES_DIR,
    report_dir: Path = OUTPUT_REPORTS_DIR,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, object] = {}
    working = household_df.dropna(subset=["wealth_class"]).copy()
    if working.empty:
        pd.DataFrame([{"component": "deep_learning", "status": "skipped", "reason": "No non-missing wealth_class observations were available."}]).to_csv(
            report_dir / "deep_learning_status.csv",
            index=False,
        )
        return results

    sample_df = _sample_household_df(working)
    X_df, y = _prepare_feature_frame(sample_df)
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    bundle, preprocessor, encoder = _fit_mlp_model(X_train_df, y_train)
    preds, probs = _predict_mlp(bundle, encoder, X_test_df)
    accuracy = float(accuracy_score(y_test, preds))
    macro_f1 = float(f1_score(y_test, preds, average="macro"))
    mlp_result = {"model": "mlp_classifier", "backend": bundle["backend"], "accuracy": accuracy, "macro_f1": macro_f1}
    pd.DataFrame([mlp_result]).to_csv(output_dir / "deep_learning_model_comparison.csv", index=False)
    with open(output_dir / "mlp_classifier_report.txt", "w", encoding="utf-8") as handle:
        handle.write(classification_report(y_test, preds))
    pd.DataFrame(
        confusion_matrix(y_test, preds, labels=encoder.classes_),
        index=[f"true_{label}" for label in encoder.classes_],
        columns=[f"pred_{label}" for label in encoder.classes_],
    ).to_csv(output_dir / "mlp_confusion_matrix.csv")
    results["mlp_classifier"] = mlp_result

    if not bundle["history"].empty:
        bundle["history"].to_csv(output_dir / "mlp_training_history.csv", index=False)

    importance_df = _manual_permutation_importance(bundle, encoder, X_test_df, y_test)
    importance_df.to_csv(output_dir / "mlp_feature_importance.csv", index=False)
    results["mlp_feature_importance"] = importance_df

    subgroup_rows = []
    subgroup_eval = X_test_df.copy()
    subgroup_eval["wealth_class"] = y_test.values
    subgroup_eval["prediction"] = preds
    for col in [c for c in ["URBAN_H", "YEAR", "GEO1_VN"] if c in subgroup_eval.columns]:
        for group_value, grp in subgroup_eval.groupby(col):
            if len(grp) < 30:
                continue
            subgroup_rows.append(
                {
                    "model": "mlp_classifier",
                    "subgroup_dimension": col,
                    "subgroup_value": group_value,
                    "n_obs": len(grp),
                    "accuracy": accuracy_score(grp["wealth_class"], grp["prediction"]),
                    "macro_f1": f1_score(grp["wealth_class"], grp["prediction"], average="macro"),
                }
            )
    subgroup_df = pd.DataFrame(subgroup_rows).sort_values(["subgroup_dimension", "macro_f1"], ascending=[True, False]) if subgroup_rows else pd.DataFrame()
    subgroup_df.to_csv(output_dir / "mlp_subgroup_performance.csv", index=False)
    results["mlp_subgroup_performance"] = subgroup_df

    temporal_df = _temporal_mlp_validation(sample_df)
    temporal_df.to_csv(output_dir / "deep_learning_temporal_validation.csv", index=False)
    results["deep_learning_temporal_validation"] = temporal_df

    ablation_specs = {
        "assets_utilities_only": [c for c in X_df.columns if c in ["ELECTRIC_H", "WATSUP_H", "SEWAGE_H", "OWNERSHIP_H", "WALL_H", "ROOF_H"]],
        "space_time_only": [c for c in X_df.columns if c in ["URBAN_H", "YEAR", "GEO1_VN"]],
        "housing_density_only": [c for c in X_df.columns if c in ["HOUSEHOLD_SIZE", "AREA_PER_PERSON", "ROOMS_PER_PERSON"]],
        "all_features": list(X_df.columns),
    }
    ablation_rows = []
    for spec_name, cols in ablation_specs.items():
        if len(cols) < 2:
            continue
        Xa_train = X_train_df[cols].copy()
        Xa_test = X_test_df[cols].copy()
        spec_bundle, _, spec_encoder = _fit_mlp_model(Xa_train, y_train)
        spec_preds, _ = _predict_mlp(spec_bundle, spec_encoder, Xa_test)
        ablation_rows.append(
            {
                "specification": spec_name,
                "backend": spec_bundle["backend"],
                "n_features": len(cols),
                "accuracy": accuracy_score(y_test, spec_preds),
                "macro_f1": f1_score(y_test, spec_preds, average="macro"),
            }
        )
    ablation_df = pd.DataFrame(ablation_rows).sort_values("macro_f1", ascending=False) if ablation_rows else pd.DataFrame()
    ablation_df.to_csv(output_dir / "deep_learning_ablation.csv", index=False)
    results["deep_learning_ablation"] = ablation_df

    if torch is None:
        pd.DataFrame(
            [{"component": "autoencoder", "status": "skipped", "reason": "PyTorch not available in the environment. Install requirements to enable autoencoder training."}]
        ).to_csv(report_dir / "autoencoder_status.csv", index=False)
        return results

    full_preprocessor, X_processed = _fit_preprocessor(X_df)
    latent_sweep_rows = []
    latent_outputs: dict[int, pd.DataFrame] = {}
    wealth_index = sample_df["wealth_index"] if "wealth_index" in sample_df.columns else pd.Series(np.nan, index=sample_df.index)

    for latent_dim in [2, 3, 5]:
        _, recon_mse, latent_df = _fit_autoencoder(X_processed, latent_dim=latent_dim, max_epochs=30)
        latent_outputs[latent_dim] = latent_df
        latent_eval = _evaluate_representation(latent_df, "latent_", wealth_index, y)
        latent_sweep_rows.append({"latent_dim": latent_dim, "reconstruction_mse": recon_mse, **latent_eval})

    latent_sweep_df = pd.DataFrame(latent_sweep_rows).sort_values("downstream_macro_f1", ascending=False)
    latent_sweep_df.to_csv(output_dir / "autoencoder_latent_sweep.csv", index=False)
    best_latent_dim = int(latent_sweep_df.iloc[0]["latent_dim"])
    best_latent_df = latent_outputs[best_latent_dim].copy()
    best_latent_df["wealth_index"] = wealth_index.to_numpy()
    best_latent_df["wealth_class"] = y.to_numpy()
    best_latent_df.to_csv(output_dir / "autoencoder_latent_features.csv", index=False)

    autoencoder_eval = latent_sweep_df[latent_sweep_df["latent_dim"] == best_latent_dim].copy()
    autoencoder_eval.to_csv(output_dir / "autoencoder_evaluation.csv", index=False)
    results["autoencoder"] = autoencoder_eval

    pca = PCA(n_components=3, random_state=RANDOM_STATE)
    pca_features = pca.fit_transform(X_processed)
    pca_df = pd.DataFrame(pca_features, columns=["pca_1", "pca_2", "pca_3"])
    pca_eval = _evaluate_representation(pca_df, "pca_", wealth_index, y)
    pca_eval["explained_variance_ratio"] = float(pca.explained_variance_ratio_.sum())

    auto_eval_dict = _evaluate_representation(best_latent_df[[c for c in best_latent_df.columns if c.startswith("latent_")]], "latent_", wealth_index, y)
    auto_eval_dict["explained_variance_ratio"] = np.nan
    benchmark_df = pd.DataFrame(
        [
            {"representation": "autoencoder", "latent_dim": best_latent_dim, **auto_eval_dict},
            {"representation": "pca", "latent_dim": 3, **pca_eval},
        ]
    )
    benchmark_df.to_csv(output_dir / "representation_benchmark_pca_vs_autoencoder.csv", index=False)
    results["representation_benchmark"] = benchmark_df
    return results
