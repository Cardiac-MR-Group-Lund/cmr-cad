#!/usr/bin/env python3
"""
07_run_binary_rf_nested_cv.py

Deterministic nested-cross-validation pipeline for the revised binary Random
Forest analysis.

Primary binary endpoint
-----------------------
MACE during available follow-up from index CMR:
    mace_0d_event
Definition:
    cardiovascular death, acute myocardial infarction, or PCI/CABG.

Frozen inputs
-------------
- data/processed/cmr_revision_analysis.csv
- data/processed/outer_cv_folds.csv
- config/registry_vocabulary.json
- config/cv_plan.json

Model comparisons
-----------------
- demographics
- registry
- CMR
- registry + demographics
- demographics + CMR
- registry + demographics + CMR

The principal comparison is demographics + CMR versus registry + demographics
+ CMR.

Validation design
-----------------
- Frozen 5-fold outer cross-validation.
- Five-fold inner StratifiedKFold within each outer-training set.
- All imputation is fitted inside the model pipeline using training data only.
- Random Forest hyperparameters are selected by inner-fold AUROC.
- Each patient receives exactly one out-of-fold prediction.
- RF uses n_jobs=1 and fixed random states for deterministic fits.

Primary outputs
---------------
Private patient-level files:
    results/models/binary_rf/oof_predictions.csv

Aggregate outputs:
    results/models/binary_rf/metrics_summary.csv
    results/models/binary_rf/fold_metrics.csv
    results/models/binary_rf/selected_hyperparameters.csv
    results/models/binary_rf/primary_comparison.csv
    results/models/binary_rf/roc_binary_rf.png/.tiff/.pdf/.svg
    results/models/binary_rf/run_manifest.json
    results/models/binary_rf/run_summary.txt

Optional held-out permutation importance:
    python scripts/07_run_binary_rf_nested_cv.py --run-importance

Quick validation run using only the principal feature sets:
    python scripts/07_run_binary_rf_nested_cv.py \
        --feature-sets demo_cmr full

Full run:
    python scripts/07_run_binary_rf_nested_cv.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# Set thread limits before importing numerical libraries.
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline


SCRIPT_VERSION = "1.0.0"
GLOBAL_SEED = 42
INNER_BASE_SEED = 2026
N_INNER_SPLITS = 5
N_ESTIMATORS = 500
TARGET = "mace_0d_event"

DEFAULT_DATA = Path("data/processed/cmr_revision_analysis.csv")
DEFAULT_FOLDS = Path("data/processed/outer_cv_folds.csv")
DEFAULT_VOCAB = Path("config/registry_vocabulary.json")
DEFAULT_CV_PLAN = Path("config/cv_plan.json")
DEFAULT_OUTPUT_DIR = Path("results/models/binary_rf")

EXPECTED_DATA_SHA256 = "e94b8aebe4c4939b0b3def943879539f4f242ba7ef39fd7ac929407578f7db9c"
EXPECTED_VOCAB_SPEC_SHA256 = "1250d92ca3f5d5da2049f3965837a7cfa0b70daaa9ffae2e09d71f8235eac023"
EXPECTED_FOLD_SPEC_SHA256 = "69c4a862c97d96a7c29e21dcc9c3a1bb7b74f87f1d0baee6c26b4f87fda219eb"
EXPECTED_N = 2159
EXPECTED_EVENTS = 212

DEMO_COLS = ["age_years", "sex_male", "bmi"]
CMR_COLS = [
    "lvef_percent",
    "lvedvi_ml_m2",
    "lvesvi_ml_m2",
    "infarct_binary",
    "ischemia_binary",
]
CONTINUOUS_CLINICAL = [
    "age_years", "bmi", "lvef_percent", "lvedvi_ml_m2", "lvesvi_ml_m2"
]
BINARY_CLINICAL = ["sex_male", "infarct_binary", "ischemia_binary"]

FEATURE_SET_ORDER = [
    "demographics",
    "registry",
    "cmr",
    "registry_demo",
    "demo_cmr",
    "full",
]

FEATURE_SET_LABELS = {
    "demographics": "Demographics",
    "registry": "Registry",
    "cmr": "CMR",
    "registry_demo": "Registry + demographics",
    "demo_cmr": "Demographics + CMR",
    "full": "Registry + demographics + CMR",
}

PLOT_STYLES = {
    "demographics": {"color": "#8c8c8c", "linestyle": ":"},
    "registry": {"color": "#6f6f6f", "linestyle": "-."},
    "cmr": {"color": "#9467bd", "linestyle": "--"},
    "registry_demo": {"color": "#d95f02", "linestyle": "-."},
    "demo_cmr": {"color": "#1b9e77", "linestyle": "--"},
    "full": {"color": "#1f4e79", "linestyle": "-"},
}

PARAM_GRID = {
    "classifier__max_depth": [8, 12],
    "classifier__max_features": ["sqrt", "log2"],
    "classifier__min_samples_leaf": [2, 5, 10],
}


@dataclass(frozen=True)
class FeatureSet:
    key: str
    label: str
    columns: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic nested-CV binary Random Forest comparisons."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--vocabulary", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--cv-plan", type=Path, default=DEFAULT_CV_PLAN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        choices=FEATURE_SET_ORDER,
        default=FEATURE_SET_ORDER,
        help="Feature sets to run. Default: all six.",
    )
    parser.add_argument("--grid-jobs", type=int, default=-1,
                        help="Parallel workers across GridSearchCV candidates. RF itself remains single-threaded.")
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--run-importance", action="store_true")
    parser.add_argument("--importance-repeats", type=int, default=20)
    parser.add_argument("--allow-hash-mismatch", action="store_true")
    return parser.parse_args()


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        sample = handle.read(131072)
    try:
        return csv.Sniffer().sniff(sample, delimiters=";,\t|").delimiter
    except csv.Error:
        first = sample.splitlines()[0] if sample else ""
        return max([";", ",", "\t", "|"], key=first.count)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def clean_code_list(value: Any) -> set[str]:
    if value is None or pd.isna(value):
        return set()
    text = str(value).strip().upper()
    if not text:
        return set()
    return {part for part in re.split(r"[;,\s]+", text) if part}


def normalize_icd(code: str) -> str:
    compact = re.sub(r"[^A-Z0-9]", "", code.upper())
    return compact[:3] if len(compact) >= 3 else ""


def normalize_kva(code: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", code.upper())


def build_registry_matrix(df: pd.DataFrame, vocabulary: dict[str, Any]) -> pd.DataFrame:
    features = vocabulary["features"]
    feature_names = [item["feature_name"] for item in features]
    matrix = np.zeros((len(df), len(features)), dtype=np.uint8)

    icd_lookup = {
        item["normalized_code"]: idx
        for idx, item in enumerate(features)
        if item["code_system"] == "ICD10"
    }
    kva_lookup = {
        item["normalized_code"]: idx
        for idx, item in enumerate(features)
        if item["code_system"] == "KVA"
    }

    for row_idx, (icd_value, kva_value) in enumerate(
        zip(df["pre_icd_codes"], df["pre_kva_codes"])
    ):
        icd_codes = {normalize_icd(code) for code in clean_code_list(icd_value)}
        kva_codes = {normalize_kva(code) for code in clean_code_list(kva_value)}
        for code in icd_codes:
            col_idx = icd_lookup.get(code)
            if col_idx is not None:
                matrix[row_idx, col_idx] = 1
        for code in kva_codes:
            col_idx = kva_lookup.get(code)
            if col_idx is not None:
                matrix[row_idx, col_idx] = 1

    return pd.DataFrame(matrix, columns=feature_names, index=df.index)


def make_feature_sets(registry_cols: list[str]) -> dict[str, FeatureSet]:
    return {
        "demographics": FeatureSet("demographics", FEATURE_SET_LABELS["demographics"], list(DEMO_COLS)),
        "registry": FeatureSet("registry", FEATURE_SET_LABELS["registry"], list(registry_cols)),
        "cmr": FeatureSet("cmr", FEATURE_SET_LABELS["cmr"], list(CMR_COLS)),
        "registry_demo": FeatureSet("registry_demo", FEATURE_SET_LABELS["registry_demo"], list(registry_cols) + list(DEMO_COLS)),
        "demo_cmr": FeatureSet("demo_cmr", FEATURE_SET_LABELS["demo_cmr"], list(DEMO_COLS) + list(CMR_COLS)),
        "full": FeatureSet("full", FEATURE_SET_LABELS["full"], list(registry_cols) + list(DEMO_COLS) + list(CMR_COLS)),
    }


def make_pipeline(columns: list[str], outer_fold: int) -> Pipeline:
    continuous = [col for col in columns if col in CONTINUOUS_CLINICAL]
    binary_clinical = [col for col in columns if col in BINARY_CLINICAL]
    registry = [col for col in columns if col.startswith("pre_icd__") or col.startswith("pre_kva__")]

    transformers = []
    if continuous:
        transformers.append((
            "continuous",
            SimpleImputer(strategy="median", add_indicator=True),
            continuous,
        ))
    if binary_clinical:
        transformers.append((
            "binary_clinical",
            SimpleImputer(strategy="most_frequent", add_indicator=True),
            binary_clinical,
        ))
    if registry:
        transformers.append(("registry", "passthrough", registry))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )

    classifier = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight="balanced_subsample",
        criterion="gini",
        bootstrap=True,
        n_jobs=1,
        random_state=GLOBAL_SEED + outer_fold,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def calibration_intercept_slope(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    eps = 1e-6
    clipped = np.clip(p, eps, 1.0 - eps)
    logit = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    model = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=5000,
        random_state=GLOBAL_SEED,
    )
    model.fit(logit, y)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def metric_values(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    intercept, slope = calibration_intercept_slope(y, p)
    return {
        "auroc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def stratified_bootstrap_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    event_idx = np.flatnonzero(y == 1)
    nonevent_idx = np.flatnonzero(y == 0)
    sampled_events = rng.choice(event_idx, size=len(event_idx), replace=True)
    sampled_nonevents = rng.choice(nonevent_idx, size=len(nonevent_idx), replace=True)
    combined = np.concatenate([sampled_events, sampled_nonevents])
    rng.shuffle(combined)
    return combined


def bootstrap_metrics(
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
    repeats: int,
    seed: int,
) -> tuple[dict[str, dict[str, tuple[float, float]]], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    metric_names = ["auroc", "average_precision", "brier_score"]
    storage = {
        key: {metric: np.empty(repeats, dtype=float) for metric in metric_names}
        for key in predictions
    }
    delta_rows = []

    for repeat in range(repeats):
        idx = stratified_bootstrap_indices(y, rng)
        y_b = y[idx]
        for key, p in predictions.items():
            p_b = p[idx]
            storage[key]["auroc"][repeat] = roc_auc_score(y_b, p_b)
            storage[key]["average_precision"][repeat] = average_precision_score(y_b, p_b)
            storage[key]["brier_score"][repeat] = brier_score_loss(y_b, p_b)

    intervals: dict[str, dict[str, tuple[float, float]]] = {}
    for key in predictions:
        intervals[key] = {}
        for metric in metric_names:
            values = storage[key][metric]
            intervals[key][metric] = (
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            )

    if "demo_cmr" in predictions and "full" in predictions:
        for metric in metric_names:
            delta = storage["full"][metric] - storage["demo_cmr"][metric]
            delta_rows.append({
                "comparison": "Registry + demographics + CMR minus demographics + CMR",
                "metric": metric,
                "estimate": (
                    metric_values(y, predictions["full"])[metric]
                    - metric_values(y, predictions["demo_cmr"])[metric]
                ),
                "ci_lower": float(np.quantile(delta, 0.025)),
                "ci_upper": float(np.quantile(delta, 0.975)),
                "bootstrap_repeats": repeats,
                "bootstrap_seed": seed,
            })

    return intervals, pd.DataFrame(delta_rows)


def plot_roc(
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
    selected_keys: list[str],
    output_dir: Path,
) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(7.4, 7.0))
    for key in selected_keys:
        p = predictions[key]
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        style = PLOT_STYLES[key]
        ax.plot(
            fpr,
            tpr,
            linewidth=2.2,
            label=f"{FEATURE_SET_LABELS[key]} (AUROC {auc:.2f})",
            **style,
        )

    ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1.2,
            label="Reference (AUROC 0.50)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("1 - specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title("Binary Prediction of MACE During Available Follow-Up")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.grid(False)
    fig.tight_layout()

    paths = {
        "png": output_dir / "roc_binary_rf.png",
        "tiff": output_dir / "roc_binary_rf.tiff",
        "pdf": output_dir / "roc_binary_rf.pdf",
        "svg": output_dir / "roc_binary_rf.svg",
    }
    fig.savefig(paths["png"], dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(paths["tiff"], dpi=600, format="tiff", facecolor="white",
                bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(paths["pdf"], facecolor="white", bbox_inches="tight")
    fig.savefig(paths["svg"], facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return {key: str(path) for key, path in paths.items()}


def plot_importance(importance_df: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    top = importance_df.sort_values("importance_mean", ascending=False).head(20).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_sd"], color="#1f4e79")
    ax.set_xlabel("Mean decrease in held-out AUROC")
    ax.set_ylabel("")
    ax.set_title("Held-Out Permutation Importance: Full RF Model")
    ax.grid(False)
    fig.tight_layout()
    paths = {
        "png": output_dir / "permutation_importance_full_rf.png",
        "tiff": output_dir / "permutation_importance_full_rf.tiff",
        "pdf": output_dir / "permutation_importance_full_rf.pdf",
        "svg": output_dir / "permutation_importance_full_rf.svg",
    }
    fig.savefig(paths["png"], dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(paths["tiff"], dpi=600, format="tiff", facecolor="white",
                bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(paths["pdf"], facecolor="white", bbox_inches="tight")
    fig.savefig(paths["svg"], facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return {key: str(path) for key, path in paths.items()}


def write_csv(df: pd.DataFrame, path: Path, sep: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, sep=sep, encoding="utf-8", float_format="%.12g")


def main() -> int:
    args = parse_args()
    set_determinism(GLOBAL_SEED)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in [args.data, args.folds, args.vocabulary, args.cv_plan]:
        if not path.exists():
            print(f"[ERROR] Required input not found: {path}", file=sys.stderr)
            return 2

    data_hash = sha256_file(args.data)
    vocab = load_json(args.vocabulary)
    cv_plan = load_json(args.cv_plan)
    vocab_spec_hash = vocab.get("vocabulary_sha256")
    fold_spec_hash = cv_plan["outer_cross_validation"].get("assignment_specification_sha256")

    hash_checks = {
        "analysis_dataset": data_hash == EXPECTED_DATA_SHA256,
        "registry_vocabulary": vocab_spec_hash == EXPECTED_VOCAB_SPEC_SHA256,
        "outer_fold_assignment": fold_spec_hash == EXPECTED_FOLD_SPEC_SHA256,
    }
    if not all(hash_checks.values()) and not args.allow_hash_mismatch:
        raise RuntimeError(f"Frozen input hash mismatch: {hash_checks}")

    delimiter = detect_delimiter(args.data)
    df = pd.read_csv(args.data, sep=delimiter, encoding="utf-8-sig", low_memory=False)
    fold_delimiter = detect_delimiter(args.folds)
    folds = pd.read_csv(args.folds, sep=fold_delimiter, encoding="utf-8-sig")

    if len(df) != EXPECTED_N or len(folds) != EXPECTED_N:
        raise ValueError(f"Expected {EXPECTED_N} rows in data and folds.")
    if df["analysis_id"].isna().any() or df["analysis_id"].duplicated().any():
        raise ValueError("Invalid analysis_id in analysis dataset.")
    if folds["analysis_id"].isna().any() or folds["analysis_id"].duplicated().any():
        raise ValueError("Invalid analysis_id in fold map.")

    df = df.merge(folds[["analysis_id", "outer_fold"]], on="analysis_id", how="left", validate="one_to_one")
    if df["outer_fold"].isna().any():
        raise ValueError("At least one analysis row lacks an outer fold.")
    df = df.sort_values("analysis_id", kind="mergesort").reset_index(drop=True)

    y = pd.to_numeric(df[TARGET], errors="coerce")
    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError(f"{TARGET} is missing or nonbinary.")
    y_np = y.astype(int).to_numpy()
    if int(y_np.sum()) != EXPECTED_EVENTS:
        raise ValueError(f"Expected {EXPECTED_EVENTS} events; found {int(y_np.sum())}.")

    registry_matrix = build_registry_matrix(df, vocab)
    registry_cols = registry_matrix.columns.tolist()
    if len(registry_cols) != 62:
        raise ValueError(f"Expected 62 registry features; found {len(registry_cols)}.")
    df_model = pd.concat([df, registry_matrix], axis=1)

    feature_sets = make_feature_sets(registry_cols)
    run_keys = [key for key in FEATURE_SET_ORDER if key in args.feature_sets]

    missing_model_cols = sorted({col for key in run_keys for col in feature_sets[key].columns} - set(df_model.columns))
    if missing_model_cols:
        raise ValueError(f"Missing modelling columns: {missing_model_cols}")

    print(f"[INFO] Script version: {SCRIPT_VERSION}")
    print(f"[INFO] Target: {TARGET}; N={len(df_model)}; events={int(y_np.sum())} ({y_np.mean() * 100:.2f}%)")
    print(f"[INFO] Registry vocabulary: {len(registry_cols)} features")
    print(f"[INFO] Feature sets: {', '.join(run_keys)}")
    print(f"[INFO] RF grid: {math.prod(len(v) for v in PARAM_GRID.values())} combinations; inner folds={N_INNER_SPLITS}")

    oof_predictions: dict[str, np.ndarray] = {
        key: np.full(len(df_model), np.nan, dtype=float) for key in run_keys
    }
    fold_metric_rows = []
    parameter_rows = []
    importance_rows = []

    outer_folds = sorted(df_model["outer_fold"].astype(int).unique().tolist())
    for feature_key in run_keys:
        feature_set = feature_sets[feature_key]
        print(f"[INFO] Starting: {feature_set.label} ({len(feature_set.columns)} original predictors)")
        X_all = df_model[feature_set.columns].copy()

        for outer_fold in outer_folds:
            test_mask = df_model["outer_fold"].astype(int).eq(outer_fold).to_numpy()
            train_mask = ~test_mask
            X_train = X_all.loc[train_mask]
            X_test = X_all.loc[test_mask]
            y_train = y.loc[train_mask].astype(int)
            y_test = y.loc[test_mask].astype(int)

            inner_cv = StratifiedKFold(
                n_splits=N_INNER_SPLITS,
                shuffle=True,
                random_state=INNER_BASE_SEED + outer_fold,
            )
            pipeline = make_pipeline(feature_set.columns, outer_fold)
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=PARAM_GRID,
                scoring="roc_auc",
                cv=inner_cv,
                refit=True,
                n_jobs=args.grid_jobs,
                return_train_score=False,
                error_score="raise",
            )
            search.fit(X_train, y_train)
            p_test = search.predict_proba(X_test)[:, 1]
            oof_predictions[feature_key][test_mask] = p_test

            fold_metric_rows.append({
                "feature_set": feature_key,
                "feature_set_label": feature_set.label,
                "outer_fold": outer_fold,
                "train_n": int(train_mask.sum()),
                "train_events": int(y_train.sum()),
                "test_n": int(test_mask.sum()),
                "test_events": int(y_test.sum()),
                "auroc": float(roc_auc_score(y_test, p_test)),
                "average_precision": float(average_precision_score(y_test, p_test)),
                "brier_score": float(brier_score_loss(y_test, p_test)),
                "best_inner_auroc": float(search.best_score_),
            })
            parameter_rows.append({
                "feature_set": feature_key,
                "feature_set_label": feature_set.label,
                "outer_fold": outer_fold,
                "best_inner_auroc": float(search.best_score_),
                "max_depth": search.best_params_["classifier__max_depth"],
                "max_features": search.best_params_["classifier__max_features"],
                "min_samples_leaf": search.best_params_["classifier__min_samples_leaf"],
                "n_estimators": N_ESTIMATORS,
                "rf_random_state": GLOBAL_SEED + outer_fold,
                "inner_random_state": INNER_BASE_SEED + outer_fold,
            })

            if args.run_importance and feature_key == "full":
                result = permutation_importance(
                    search.best_estimator_,
                    X_test,
                    y_test,
                    scoring="roc_auc",
                    n_repeats=args.importance_repeats,
                    random_state=GLOBAL_SEED + 1000 + outer_fold,
                    n_jobs=args.grid_jobs,
                )
                for feature, mean_value, sd_value in zip(
                    feature_set.columns,
                    result.importances_mean,
                    result.importances_std,
                ):
                    importance_rows.append({
                        "outer_fold": outer_fold,
                        "feature": feature,
                        "importance_mean": float(mean_value),
                        "importance_sd_within_fold": float(sd_value),
                    })

            print(
                f"[INFO] {feature_key} | fold {outer_fold} | "
                f"AUROC={roc_auc_score(y_test, p_test):.4f} | best={search.best_params_}"
            )

        if np.isnan(oof_predictions[feature_key]).any():
            raise RuntimeError(f"Missing OOF predictions for {feature_key}.")

    intervals, primary_comparison = bootstrap_metrics(
        y_np,
        oof_predictions,
        repeats=args.bootstrap_repeats,
        seed=GLOBAL_SEED + 5000,
    )

    metrics_rows = []
    for key in run_keys:
        metrics = metric_values(y_np, oof_predictions[key])
        metrics_rows.append({
            "feature_set": key,
            "feature_set_label": FEATURE_SET_LABELS[key],
            "n_original_predictors": len(feature_sets[key].columns),
            "n": len(y_np),
            "events": int(y_np.sum()),
            **metrics,
            "auroc_ci_lower": intervals[key]["auroc"][0],
            "auroc_ci_upper": intervals[key]["auroc"][1],
            "average_precision_ci_lower": intervals[key]["average_precision"][0],
            "average_precision_ci_upper": intervals[key]["average_precision"][1],
            "brier_ci_lower": intervals[key]["brier_score"][0],
            "brier_ci_upper": intervals[key]["brier_score"][1],
            "bootstrap_repeats": args.bootstrap_repeats,
        })

    metrics_df = pd.DataFrame(metrics_rows).sort_values("auroc", ascending=False)
    fold_metrics_df = pd.DataFrame(fold_metric_rows)
    parameters_df = pd.DataFrame(parameter_rows)

    # Private OOF file: analysis_id plus endpoint and predictions only.
    oof_df = pd.DataFrame({
        "analysis_id": df_model["analysis_id"].astype(int),
        "outer_fold": df_model["outer_fold"].astype(int),
        TARGET: y_np,
    })
    for key in run_keys:
        oof_df[f"oof_rf_{key}"] = oof_predictions[key]

    write_csv(oof_df, output_dir / "oof_predictions.csv", sep=";")
    write_csv(metrics_df, output_dir / "metrics_summary.csv")
    write_csv(fold_metrics_df, output_dir / "fold_metrics.csv")
    write_csv(parameters_df, output_dir / "selected_hyperparameters.csv")
    if not primary_comparison.empty:
        write_csv(primary_comparison, output_dir / "primary_comparison.csv")

    roc_paths = plot_roc(y_np, oof_predictions, run_keys, output_dir)

    importance_paths = None
    if args.run_importance and importance_rows:
        raw_importance = pd.DataFrame(importance_rows)
        write_csv(raw_importance, output_dir / "permutation_importance_by_fold.csv")
        aggregated = (
            raw_importance.groupby("feature", as_index=False)
            .agg(
                importance_mean=("importance_mean", "mean"),
                importance_sd=("importance_mean", "std"),
                folds=("outer_fold", "nunique"),
            )
            .sort_values("importance_mean", ascending=False)
        )
        aggregated["importance_sd"] = aggregated["importance_sd"].fillna(0.0)
        write_csv(aggregated, output_dir / "permutation_importance_aggregated.csv")
        importance_paths = plot_importance(aggregated, output_dir)

    checks = [
        {"check": "Analysis dataset hash", "status": "PASS" if hash_checks["analysis_dataset"] else "REVIEW",
         "observed": data_hash, "expected": EXPECTED_DATA_SHA256},
        {"check": "Vocabulary hash", "status": "PASS" if hash_checks["registry_vocabulary"] else "REVIEW",
         "observed": vocab_spec_hash, "expected": EXPECTED_VOCAB_SPEC_SHA256},
        {"check": "Fold assignment hash", "status": "PASS" if hash_checks["outer_fold_assignment"] else "REVIEW",
         "observed": fold_spec_hash, "expected": EXPECTED_FOLD_SPEC_SHA256},
        {"check": "Cohort N", "status": "PASS" if len(df_model) == EXPECTED_N else "REVIEW",
         "observed": len(df_model), "expected": EXPECTED_N},
        {"check": "Event count", "status": "PASS" if int(y_np.sum()) == EXPECTED_EVENTS else "REVIEW",
         "observed": int(y_np.sum()), "expected": EXPECTED_EVENTS},
        {"check": "Registry feature count", "status": "PASS" if len(registry_cols) == 62 else "REVIEW",
         "observed": len(registry_cols), "expected": 62},
        {"check": "OOF prediction completeness", "status": "PASS" if all(not np.isnan(p).any() for p in oof_predictions.values()) else "REVIEW",
         "observed": sum(int(np.isnan(p).sum()) for p in oof_predictions.values()), "expected": 0},
        {"check": "OOF prediction range", "status": "PASS" if all(((p >= 0) & (p <= 1)).all() for p in oof_predictions.values()) else "REVIEW",
         "observed": "within [0,1]", "expected": "within [0,1]"},
    ]
    checks_df = pd.DataFrame(checks)
    write_csv(checks_df, output_dir / "run_checks.csv")
    review_count = int(checks_df["status"].eq("REVIEW").sum())
    if review_count and not args.allow_hash_mismatch:
        raise RuntimeError(f"Run completed but {review_count} frozen checks require review.")

    output_hashes = {}
    for path in output_dir.iterdir():
        if path.is_file():
            output_hashes[path.name] = sha256_file(path)

    manifest = {
        "run": {
            "script": Path(__file__).name,
            "script_version": SCRIPT_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "global_seed": GLOBAL_SEED,
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        },
        "frozen_inputs": {
            "analysis_dataset_sha256": data_hash,
            "registry_vocabulary_specification_sha256": vocab_spec_hash,
            "outer_fold_assignment_specification_sha256": fold_spec_hash,
        },
        "endpoint": {
            "column": TARGET,
            "definition": "CV death, AMI, or PCI/CABG during available follow-up from index CMR",
            "n": len(y_np),
            "events": int(y_np.sum()),
        },
        "feature_sets": {
            key: {
                "label": feature_sets[key].label,
                "original_predictor_count": len(feature_sets[key].columns),
                "columns": feature_sets[key].columns,
            }
            for key in run_keys
        },
        "validation": {
            "outer_folds": 5,
            "outer_assignment": "frozen",
            "inner_folds": N_INNER_SPLITS,
            "inner_base_seed": INNER_BASE_SEED,
            "grid_scoring": "roc_auc",
            "one_oof_prediction_per_patient": True,
            "preprocessing_fitted_within_training_data": True,
        },
        "random_forest": {
            "n_estimators": N_ESTIMATORS,
            "class_weight": "balanced_subsample",
            "n_jobs_per_forest": 1,
            "parameter_grid": PARAM_GRID,
            "grid_combinations": math.prod(len(v) for v in PARAM_GRID.values()),
        },
        "bootstrap": {
            "method": "stratified paired patient bootstrap",
            "repeats": args.bootstrap_repeats,
            "seed": GLOBAL_SEED + 5000,
            "confidence_level": 0.95,
        },
        "importance": {
            "run": bool(args.run_importance),
            "method": "held-out outer-fold permutation importance" if args.run_importance else None,
            "repeats_per_fold": args.importance_repeats if args.run_importance else None,
        },
        "outputs": output_hashes,
        "checks": {
            "pass": int(checks_df["status"].eq("PASS").sum()),
            "review": review_count,
        },
        "privacy": "OOF predictions are patient-level and must remain private. Figures and aggregate tables require disclosure review before publication.",
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    # Update manifest hash list after writing the manifest is intentionally avoided,
    # because a manifest cannot contain a stable hash of itself.
    summary_lines = [
        "Nested-CV binary Random Forest analysis",
        "=" * 39,
        f"Endpoint: {TARGET}",
        f"N={len(y_np):,}; events={int(y_np.sum()):,} ({y_np.mean() * 100:.2f}%)",
        f"Feature sets: {', '.join(run_keys)}",
        "Outer folds: frozen 5-fold assignment",
        f"Inner folds: {N_INNER_SPLITS}; base seed={INNER_BASE_SEED}",
        f"RF grid combinations: {math.prod(len(v) for v in PARAM_GRID.values())}",
        f"Bootstrap repeats: {args.bootstrap_repeats}",
        "",
        "Pooled out-of-fold performance:",
    ]
    for row in metrics_df.itertuples(index=False):
        summary_lines.append(
            f"- {row.feature_set_label}: AUROC {row.auroc:.4f} "
            f"(95% CI {row.auroc_ci_lower:.4f}-{row.auroc_ci_upper:.4f}); "
            f"AP {row.average_precision:.4f}; Brier {row.brier_score:.4f}; "
            f"calibration intercept {row.calibration_intercept:.4f}; "
            f"slope {row.calibration_slope:.4f}"
        )
    if not primary_comparison.empty:
        summary_lines.append("")
        summary_lines.append("Principal paired comparison:")
        for row in primary_comparison.itertuples(index=False):
            summary_lines.append(
                f"- {row.metric}: delta {row.estimate:.4f} "
                f"(95% CI {row.ci_lower:.4f}-{row.ci_upper:.4f})"
            )
    summary_lines.extend([
        "",
        f"Checks: {manifest['checks']['pass']} PASS; {manifest['checks']['review']} REVIEW",
        "ROC legend displays AUROC to two decimal places; machine-readable outputs retain full precision.",
    ])
    (output_dir / "run_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"[DONE] Outputs: {output_dir}")
    print(f"[INFO] Checks: {manifest['checks']['pass']} PASS; {manifest['checks']['review']} REVIEW")
    print("[INFO] ROC legend reports AUROC to 2 decimals; CSV outputs retain full precision.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
