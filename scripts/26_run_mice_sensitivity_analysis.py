#!/usr/bin/env python3
"""26_run_mice_sensitivity_analysis.py

Leakage-safe multiple-imputation sensitivity analysis for the principal binary
and time-to-event model comparisons.

Purpose
-------
Assess whether the principal conclusions are sensitive to replacing the primary
median/mode imputation with multiple imputation by chained equations (MICE).
The frozen cohort, endpoints, outer folds, feature definitions, model classes,
hyperparameter grids, and scoring rules are unchanged.

Scope
-----
Binary endpoint (mace_0d_event; 212 events):
    - Random Forest, demographics + CMR
    - Random Forest, registry + demographics + CMR

Primary survival endpoint (mace_30d_event / mace_30d_tte_years; 196 events):
    - Ridge Cox, demographics + CMR
    - Ridge Cox, registry + demographics + CMR
    - Random Survival Forest, demographics + CMR
    - Random Survival Forest, registry + demographics + CMR

MICE implementation
-------------------
- Five stochastic chained-equations imputations by default.
- IterativeImputer(sample_posterior=True) is inside each sklearn Pipeline.
- During inner cross-validation, the imputer is fitted separately in each inner
  training split by GridSearchCV.
- During outer evaluation, the refitted best pipeline learns imputation only
  from the outer-training data and transforms the held-out outer fold.
- Binary clinical variables are thresholded to 0/1 after imputation.
- Registry indicators are complete and pass through unchanged.
- One held-out prediction/risk score is produced per patient per imputation.
- Final sensitivity predictions are the patient-level means across imputations.

No primary result is overwritten. All outputs are written below:
    results/sensitivity/mice_imputation/

Run from repository root
------------------------
Quick smoke test (one imputation, reduced bootstrap; still fits all principal
models and should not be used for reporting):
    python scripts/26_run_mice_sensitivity_analysis.py \
        --n-imputations 1 --bootstrap-repeats 100

Final run:
    python scripts/26_run_mice_sensitivity_analysis.py

Dependencies
------------
    numpy, pandas, scikit-learn, scikit-survival
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import warnings

# Limit low-level numerical threads before importing numerical libraries.
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

# IterativeImputer is experimental in sklearn and must be enabled explicitly.
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored
    from sksurv.util import Surv
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'scikit-survival'. Install it in the active "
        "environment before running this script."
    ) from exc

warnings.filterwarnings("ignore", category=FutureWarning)

SCRIPT_VERSION = "1.0.0"
SEED = 42
INNER_BASE_SEED = 2026
BOOTSTRAP_SEED = 15042
N_INNER = 5
DEFAULT_N_IMPUTATIONS = 5
DEFAULT_BOOTSTRAP_REPEATS = 2000
BINARY_EVENTS = 212
SURVIVAL_EVENTS = 196
EXPECTED_N = 2159
EXPECTED_REGISTRY_FEATURES = 62

DEFAULT_BASE_SCRIPT = Path("scripts/07_run_binary_rf_nested_cv.py")
DEFAULT_DATA = Path("data/processed/cmr_revision_analysis.csv")
DEFAULT_FOLDS = Path("data/processed/outer_cv_folds.csv")
DEFAULT_VOCAB = Path("config/registry_vocabulary.json")
DEFAULT_PLAN = Path("config/cv_plan.json")
DEFAULT_OUTPUT = Path("results/sensitivity/mice_imputation")
DEFAULT_BINARY_PRIMARY = Path("results/models/binary_rf/metrics_summary.csv")
DEFAULT_SURVIVAL_PRIMARY = Path("results/models/survival_comparison/model_performance.csv")

DATA_SHA256 = "e94b8aebe4c4939b0b3def943879539f4f242ba7ef39fd7ac929407578f7db9c"
VOCAB_SPEC_SHA256 = "1250d92ca3f5d5da2049f3965837a7cfa0b70daaa9ffae2e09d71f8235eac023"
FOLD_SPEC_SHA256 = "69c4a862c97d96a7c29e21dcc9c3a1bb7b74f87f1d0baee6c26b4f87fda219eb"

BINARY_TARGET = "mace_0d_event"
SURVIVAL_EVENT = "mace_30d_event"
SURVIVAL_TIME = "mace_30d_tte_years"
FEATURE_KEYS = ["demo_cmr", "full"]
FEATURE_LABELS = {
    "demo_cmr": "Demographics + CMR",
    "full": "Registry + demographics + CMR",
}
MODEL_LABELS = {
    "binary_rf": "Random Forest",
    "ridge": "Ridge Cox",
    "rsf": "Random Survival Forest",
}

RF_N_ESTIMATORS = 500
RSF_N_ESTIMATORS = 800
RF_GRID = {
    "model__max_depth": [8, 12],
    "model__max_features": ["sqrt", "log2"],
    "model__min_samples_leaf": [2, 5, 10],
}
RIDGE_GRID = {"model__alpha": np.logspace(-4, 4, 17).tolist()}
RSF_GRID = {
    "model__max_depth": [None, 10],
    "model__max_features": ["sqrt", "log2"],
    "model__min_samples_leaf": [5, 10, 20],
}


class ThresholdBinaryColumns(BaseEstimator, TransformerMixin):
    """Threshold selected columns to 0/1 after chained-equations imputation."""

    def __init__(self, binary_indices: tuple[int, ...], threshold: float = 0.5):
        self.binary_indices = binary_indices
        self.threshold = threshold

    def fit(self, X, y=None):
        array = np.asarray(X)
        if array.ndim != 2:
            raise ValueError("Expected a 2-dimensional clinical predictor matrix")
        if any(index < 0 or index >= array.shape[1] for index in self.binary_indices):
            raise ValueError("At least one binary column index is out of range")
        return self

    def transform(self, X):
        array = np.asarray(X, dtype=float).copy()
        for index in self.binary_indices:
            array[:, index] = (array[:, index] >= self.threshold).astype(float)
        return array


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run leakage-safe MICE sensitivity analyses for principal models."
    )
    parser.add_argument("--base-script", type=Path, default=DEFAULT_BASE_SCRIPT)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--vocabulary", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--cv-plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--binary-primary", type=Path, default=DEFAULT_BINARY_PRIMARY)
    parser.add_argument("--survival-primary", type=Path, default=DEFAULT_SURVIVAL_PRIMARY)
    parser.add_argument("--n-imputations", type=int, default=DEFAULT_N_IMPUTATIONS)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--initial-strategy", default="median", choices=["mean", "median"])
    parser.add_argument("--grid-jobs", type=int, default=-1)
    parser.add_argument("--bootstrap-repeats", type=int, default=DEFAULT_BOOTSTRAP_REPEATS)
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


def read_csv_auto(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=detect_delimiter(path),
        encoding="utf-8-sig",
        low_memory=False,
        **kwargs,
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_base_module(path: Path):
    spec = importlib.util.spec_from_file_location("rf_shared_mice", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import shared definitions from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_csv(frame: pd.DataFrame, path: Path, sep: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        sep=sep,
        encoding="utf-8",
        float_format="%.12g",
    )


def make_survival(event: pd.Series, time: pd.Series):
    event_numeric = pd.to_numeric(event, errors="coerce")
    time_numeric = pd.to_numeric(time, errors="coerce")
    if event_numeric.isna().any() or not event_numeric.isin([0, 1]).all():
        raise ValueError("Invalid survival event indicator")
    if time_numeric.isna().any() or (time_numeric <= 0).any():
        raise ValueError("Invalid survival time")
    return Surv.from_arrays(event_numeric.astype(bool), time_numeric.astype(float))


def c_index(y, risk: np.ndarray) -> float:
    return float(concordance_index_censored(y["event"], y["time"], risk)[0])


def survival_scorer(estimator, X, y) -> float:
    risk = np.asarray(estimator.predict(X), dtype=float).reshape(-1)
    return c_index(y, risk)


def calibration_intercept_slope(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    logit = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    calibration = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=5000,
        random_state=SEED,
    )
    calibration.fit(logit, y)
    return float(calibration.intercept_[0]), float(calibration.coef_[0, 0])


def binary_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    intercept, slope = calibration_intercept_slope(y, probability)
    return {
        "auroc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "brier_score": float(brier_score_loss(y, probability)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def clinical_registry_columns(columns: list[str], base) -> tuple[list[str], list[str], tuple[int, ...]]:
    clinical = [
        column
        for column in columns
        if column in base.CONTINUOUS_CLINICAL or column in base.BINARY_CLINICAL
    ]
    registry = [
        column
        for column in columns
        if column.startswith("pre_icd__") or column.startswith("pre_kva__")
    ]
    binary_indices = tuple(
        index for index, column in enumerate(clinical) if column in base.BINARY_CLINICAL
    )
    if len(clinical) + len(registry) != len(columns):
        classified = set(clinical) | set(registry)
        raise ValueError(f"Unclassified model columns: {sorted(set(columns) - classified)}")
    return clinical, registry, binary_indices


def make_preprocessor(
    columns: list[str],
    base,
    imputation_seed: int,
    max_iter: int,
    initial_strategy: str,
    scale: bool,
) -> ColumnTransformer:
    clinical, registry, binary_indices = clinical_registry_columns(columns, base)
    transformers = []

    if clinical:
        clinical_steps: list[tuple[str, Any]] = [
            (
                "mice",
                IterativeImputer(
                    estimator=BayesianRidge(),
                    max_iter=max_iter,
                    sample_posterior=True,
                    initial_strategy=initial_strategy,
                    imputation_order="ascending",
                    skip_complete=True,
                    random_state=imputation_seed,
                ),
            ),
            ("threshold_binary", ThresholdBinaryColumns(binary_indices=binary_indices)),
        ]
        if scale:
            clinical_steps.append(("scaler", StandardScaler()))
        transformers.append(("clinical", Pipeline(clinical_steps), clinical))

    if registry:
        registry_transformer: Any = StandardScaler() if scale else "passthrough"
        transformers.append(("registry", registry_transformer, registry))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def make_binary_pipeline(
    columns: list[str],
    base,
    outer_fold: int,
    imputation_seed: int,
    max_iter: int,
    initial_strategy: str,
) -> Pipeline:
    preprocessor = make_preprocessor(
        columns=columns,
        base=base,
        imputation_seed=imputation_seed,
        max_iter=max_iter,
        initial_strategy=initial_strategy,
        scale=False,
    )
    model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        class_weight="balanced_subsample",
        criterion="gini",
        bootstrap=True,
        n_jobs=1,
        random_state=SEED + outer_fold,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def make_survival_pipeline(
    model_key: str,
    columns: list[str],
    base,
    outer_fold: int,
    imputation_seed: int,
    max_iter: int,
    initial_strategy: str,
) -> tuple[Pipeline, dict[str, list[Any]]]:
    preprocessor = make_preprocessor(
        columns=columns,
        base=base,
        imputation_seed=imputation_seed,
        max_iter=max_iter,
        initial_strategy=initial_strategy,
        scale=True,
    )
    if model_key == "ridge":
        model = CoxPHSurvivalAnalysis(alpha=1.0, ties="breslow", n_iter=500)
        grid = RIDGE_GRID
    elif model_key == "rsf":
        model = RandomSurvivalForest(
            n_estimators=RSF_N_ESTIMATORS,
            min_samples_split=10,
            min_samples_leaf=10,
            max_features="sqrt",
            n_jobs=1,
            random_state=SEED + outer_fold,
        )
        grid = RSF_GRID
    else:
        raise ValueError(f"Unsupported survival model: {model_key}")
    return Pipeline([("preprocessor", preprocessor), ("model", model)]), grid


def stratified_binary_bootstrap_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    event = np.flatnonzero(y == 1)
    nonevent = np.flatnonzero(y == 0)
    index = np.concatenate(
        [
            rng.choice(event, size=len(event), replace=True),
            rng.choice(nonevent, size=len(nonevent), replace=True),
        ]
    )
    rng.shuffle(index)
    return index


def stratified_survival_bootstrap_indices(y, rng: np.random.Generator) -> np.ndarray:
    event = np.flatnonzero(y["event"])
    censored = np.flatnonzero(~y["event"])
    index = np.concatenate(
        [
            rng.choice(event, size=len(event), replace=True),
            rng.choice(censored, size=len(censored), replace=True),
        ]
    )
    rng.shuffle(index)
    return index


def bootstrap_binary(
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
    repeats: int,
    seed: int,
) -> tuple[dict[str, tuple[float, float]], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    samples = {key: np.empty(repeats, dtype=float) for key in predictions}
    for repeat in range(repeats):
        index = stratified_binary_bootstrap_indices(y, rng)
        for key, probability in predictions.items():
            samples[key][repeat] = roc_auc_score(y[index], probability[index])
    intervals = {
        key: tuple(float(value) for value in np.quantile(sample, [0.025, 0.975]))
        for key, sample in samples.items()
    }
    delta = samples["full"] - samples["demo_cmr"]
    comparison = pd.DataFrame(
        [
            {
                "analysis": "binary_rf",
                "comparison": "Full minus demographics + CMR",
                "metric": "auroc",
                "estimate": roc_auc_score(y, predictions["full"])
                - roc_auc_score(y, predictions["demo_cmr"]),
                "ci_lower": float(np.quantile(delta, 0.025)),
                "ci_upper": float(np.quantile(delta, 0.975)),
                "bootstrap_repeats": repeats,
                "bootstrap_seed": seed,
            }
        ]
    )
    return intervals, comparison


def bootstrap_survival(
    y,
    risks: dict[tuple[str, str], np.ndarray],
    repeats: int,
    seed: int,
) -> tuple[dict[tuple[str, str], tuple[float, float]], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    samples = {key: np.empty(repeats, dtype=float) for key in risks}
    for repeat in range(repeats):
        index = stratified_survival_bootstrap_indices(y, rng)
        for key, risk in risks.items():
            samples[key][repeat] = c_index(y[index], risk[index])
    intervals = {
        key: tuple(float(value) for value in np.quantile(sample, [0.025, 0.975]))
        for key, sample in samples.items()
    }
    rows = []
    for model_key in ["ridge", "rsf"]:
        delta = samples[(model_key, "full")] - samples[(model_key, "demo_cmr")]
        rows.append(
            {
                "analysis": model_key,
                "comparison": "Full minus demographics + CMR",
                "metric": "c_index",
                "estimate": c_index(y, risks[(model_key, "full")])
                - c_index(y, risks[(model_key, "demo_cmr")]),
                "ci_lower": float(np.quantile(delta, 0.025)),
                "ci_upper": float(np.quantile(delta, 0.975)),
                "bootstrap_repeats": repeats,
                "bootstrap_seed": seed,
            }
        )
    for feature_key in FEATURE_KEYS:
        delta = samples[("rsf", feature_key)] - samples[("ridge", feature_key)]
        rows.append(
            {
                "analysis": "rsf_minus_ridge",
                "comparison": f"RSF minus Ridge Cox ({FEATURE_LABELS[feature_key]})",
                "metric": "c_index",
                "estimate": c_index(y, risks[("rsf", feature_key)])
                - c_index(y, risks[("ridge", feature_key)]),
                "ci_lower": float(np.quantile(delta, 0.025)),
                "ci_upper": float(np.quantile(delta, 0.975)),
                "bootstrap_repeats": repeats,
                "bootstrap_seed": seed,
            }
        )
    return intervals, pd.DataFrame(rows)


def compare_with_primary(
    binary_mice: pd.DataFrame,
    survival_mice: pd.DataFrame,
    binary_primary_path: Path,
    survival_primary_path: Path,
) -> pd.DataFrame:
    rows = []
    if binary_primary_path.exists():
        primary = read_csv_auto(binary_primary_path)
        for feature_key in FEATURE_KEYS:
            original = primary[primary["feature_set"].eq(feature_key)]
            sensitivity = binary_mice[binary_mice["feature_set"].eq(feature_key)]
            if len(original) == 1 and len(sensitivity) == 1:
                original_row = original.iloc[0]
                sensitivity_row = sensitivity.iloc[0]
                rows.append(
                    {
                        "task": "binary",
                        "model": "binary_rf",
                        "feature_set": feature_key,
                        "metric": "auroc",
                        "primary_estimate": float(original_row["auroc"]),
                        "mice_estimate": float(sensitivity_row["auroc"]),
                        "absolute_difference": float(sensitivity_row["auroc"])
                        - float(original_row["auroc"]),
                        "primary_ci_lower": float(original_row["auroc_ci_lower"]),
                        "primary_ci_upper": float(original_row["auroc_ci_upper"]),
                        "mice_ci_lower": float(sensitivity_row["auroc_ci_lower"]),
                        "mice_ci_upper": float(sensitivity_row["auroc_ci_upper"]),
                    }
                )
    if survival_primary_path.exists():
        primary = read_csv_auto(survival_primary_path)
        primary = primary[primary["endpoint"].eq("primary_30d")]
        for model_key in ["ridge", "rsf"]:
            for feature_key in FEATURE_KEYS:
                original = primary[
                    primary["model"].eq(model_key)
                    & primary["feature_set"].eq(feature_key)
                ]
                sensitivity = survival_mice[
                    survival_mice["model"].eq(model_key)
                    & survival_mice["feature_set"].eq(feature_key)
                ]
                if len(original) == 1 and len(sensitivity) == 1:
                    original_row = original.iloc[0]
                    sensitivity_row = sensitivity.iloc[0]
                    rows.append(
                        {
                            "task": "survival",
                            "model": model_key,
                            "feature_set": feature_key,
                            "metric": "c_index",
                            "primary_estimate": float(original_row["c_index"]),
                            "mice_estimate": float(sensitivity_row["c_index"]),
                            "absolute_difference": float(sensitivity_row["c_index"])
                            - float(original_row["c_index"]),
                            "primary_ci_lower": float(original_row["c_index_ci_lower"]),
                            "primary_ci_upper": float(original_row["c_index_ci_upper"]),
                            "mice_ci_lower": float(sensitivity_row["c_index_ci_lower"]),
                            "mice_ci_upper": float(sensitivity_row["c_index_ci_upper"]),
                        }
                    )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    if args.n_imputations < 1:
        raise ValueError("--n-imputations must be at least 1")
    if args.bootstrap_repeats < 1:
        raise ValueError("--bootstrap-repeats must be at least 1")
    set_determinism(SEED)

    required_paths = [args.base_script, args.data, args.folds, args.vocabulary, args.cv_plan]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base = load_base_module(args.base_script)
    vocabulary = read_json(args.vocabulary)
    cv_plan = read_json(args.cv_plan)
    hashes = {
        "analysis_dataset": sha256_file(args.data),
        "registry_vocabulary": vocabulary.get("vocabulary_sha256"),
        "outer_fold_assignment": cv_plan["outer_cross_validation"].get(
            "assignment_specification_sha256"
        ),
    }
    hash_checks = {
        "analysis_dataset": hashes["analysis_dataset"] == DATA_SHA256,
        "registry_vocabulary": hashes["registry_vocabulary"] == VOCAB_SPEC_SHA256,
        "outer_fold_assignment": hashes["outer_fold_assignment"] == FOLD_SPEC_SHA256,
    }
    if not all(hash_checks.values()) and not args.allow_hash_mismatch:
        raise RuntimeError(f"Frozen input hash mismatch: {hash_checks}")

    data = read_csv_auto(args.data)
    folds = read_csv_auto(args.folds)
    if len(data) != EXPECTED_N or len(folds) != EXPECTED_N:
        raise ValueError(f"Expected {EXPECTED_N} rows in both data and fold files")
    if data["analysis_id"].isna().any() or data["analysis_id"].duplicated().any():
        raise ValueError("Invalid analysis_id in processed dataset")
    if folds["analysis_id"].isna().any() or folds["analysis_id"].duplicated().any():
        raise ValueError("Invalid analysis_id in frozen fold map")

    data = data.merge(
        folds[["analysis_id", "outer_fold"]],
        on="analysis_id",
        how="left",
        validate="one_to_one",
    )
    if data["outer_fold"].isna().any():
        raise ValueError("At least one participant lacks a frozen outer-fold assignment")
    data = data.sort_values("analysis_id", kind="mergesort").reset_index(drop=True)
    outer_folds = sorted(data["outer_fold"].astype(int).unique().tolist())
    if outer_folds != [1, 2, 3, 4, 5]:
        raise ValueError(f"Expected frozen outer folds 1-5; observed {outer_folds}")

    registry = base.build_registry_matrix(data, vocabulary)
    if registry.shape[1] != EXPECTED_REGISTRY_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_REGISTRY_FEATURES} registry features; observed {registry.shape[1]}"
        )
    if registry.isna().any().any():
        raise ValueError("Registry indicator matrix unexpectedly contains missing values")
    model_data = pd.concat([data, registry], axis=1)
    feature_sets = base.make_feature_sets(registry.columns.tolist())

    binary_y_series = pd.to_numeric(model_data[BINARY_TARGET], errors="coerce")
    if binary_y_series.isna().any() or not binary_y_series.isin([0, 1]).all():
        raise ValueError(f"{BINARY_TARGET} is missing or nonbinary")
    binary_y = binary_y_series.astype(int).to_numpy()
    if int(binary_y.sum()) != BINARY_EVENTS:
        raise ValueError(f"Expected {BINARY_EVENTS} binary events; observed {int(binary_y.sum())}")

    survival_y = make_survival(model_data[SURVIVAL_EVENT], model_data[SURVIVAL_TIME])
    if int(survival_y["event"].sum()) != SURVIVAL_EVENTS:
        raise ValueError(
            f"Expected {SURVIVAL_EVENTS} survival events; observed {int(survival_y['event'].sum())}"
        )

    clinical_predictors = sorted(
        set(feature_sets["full"].columns) & set(base.CONTINUOUS_CLINICAL + base.BINARY_CLINICAL)
    )
    missingness_rows = []
    for column in clinical_predictors:
        missingness_rows.append(
            {
                "variable": column,
                "n": len(model_data),
                "missing": int(model_data[column].isna().sum()),
                "missing_percent": float(model_data[column].isna().mean() * 100.0),
            }
        )
    missingness = pd.DataFrame(missingness_rows).sort_values(
        ["missing_percent", "variable"], ascending=[False, True]
    )
    write_csv(missingness, output_dir / "mice_input_missingness.csv")

    print(f"[INFO] Script version: {SCRIPT_VERSION}")
    print(f"[INFO] Cohort N={len(model_data)}; binary events={int(binary_y.sum())}; survival events={int(survival_y['event'].sum())}")
    print(f"[INFO] MICE imputations={args.n_imputations}; max_iter={args.max_iter}")
    print(f"[INFO] Principal feature sets: {', '.join(FEATURE_KEYS)}")
    print("[INFO] Models: binary RF, Ridge Cox, RSF")

    n = len(model_data)
    binary_by_imputation: dict[int, dict[str, np.ndarray]] = {}
    survival_by_imputation: dict[int, dict[tuple[str, str], np.ndarray]] = {}
    fold_rows = []
    parameter_rows = []

    for imputation_index in range(1, args.n_imputations + 1):
        imputation_base_seed = SEED + 10000 * imputation_index
        print(f"[INFO] Starting imputation {imputation_index}/{args.n_imputations}")
        binary_predictions = {
            feature_key: np.full(n, np.nan, dtype=float) for feature_key in FEATURE_KEYS
        }
        survival_risks = {
            (model_key, feature_key): np.full(n, np.nan, dtype=float)
            for model_key in ["ridge", "rsf"]
            for feature_key in FEATURE_KEYS
        }

        for feature_key in FEATURE_KEYS:
            columns = feature_sets[feature_key].columns
            X = model_data[columns].copy()

            # Binary Random Forest.
            for outer_fold in outer_folds:
                test_mask = model_data["outer_fold"].astype(int).eq(outer_fold).to_numpy()
                train_mask = ~test_mask
                inner_cv = StratifiedKFold(
                    n_splits=N_INNER,
                    shuffle=True,
                    random_state=INNER_BASE_SEED + outer_fold,
                )
                mice_seed = imputation_base_seed + outer_fold
                estimator = make_binary_pipeline(
                    columns=columns,
                    base=base,
                    outer_fold=outer_fold,
                    imputation_seed=mice_seed,
                    max_iter=args.max_iter,
                    initial_strategy=args.initial_strategy,
                )
                search = GridSearchCV(
                    estimator=estimator,
                    param_grid=RF_GRID,
                    scoring="roc_auc",
                    cv=inner_cv,
                    refit=True,
                    n_jobs=args.grid_jobs,
                    return_train_score=False,
                    error_score="raise",
                )
                search.fit(X.loc[train_mask], binary_y[train_mask])
                probability = search.predict_proba(X.loc[test_mask])[:, 1]
                binary_predictions[feature_key][test_mask] = probability
                fold_rows.append(
                    {
                        "imputation": imputation_index,
                        "task": "binary",
                        "model": "binary_rf",
                        "feature_set": feature_key,
                        "outer_fold": outer_fold,
                        "test_n": int(test_mask.sum()),
                        "test_events": int(binary_y[test_mask].sum()),
                        "performance": float(roc_auc_score(binary_y[test_mask], probability)),
                        "performance_metric": "auroc",
                        "best_inner_performance": float(search.best_score_),
                    }
                )
                parameter_rows.append(
                    {
                        "imputation": imputation_index,
                        "task": "binary",
                        "model": "binary_rf",
                        "feature_set": feature_key,
                        "outer_fold": outer_fold,
                        "mice_seed": mice_seed,
                        "best_inner_performance": float(search.best_score_),
                        "best_params": json.dumps(search.best_params_, sort_keys=True),
                    }
                )
                print(
                    f"[INFO] MI={imputation_index} binary {feature_key} fold={outer_fold} "
                    f"AUROC={roc_auc_score(binary_y[test_mask], probability):.4f}"
                )

            # Survival Ridge and RSF.
            for model_key in ["ridge", "rsf"]:
                for outer_fold in outer_folds:
                    test_mask = model_data["outer_fold"].astype(int).eq(outer_fold).to_numpy()
                    train_mask = ~test_mask
                    inner_cv = StratifiedKFold(
                        n_splits=N_INNER,
                        shuffle=True,
                        random_state=INNER_BASE_SEED + outer_fold,
                    )
                    splits = list(
                        inner_cv.split(
                            np.zeros(int(train_mask.sum())),
                            survival_y[train_mask]["event"].astype(int),
                        )
                    )
                    mice_seed = imputation_base_seed + 1000 + outer_fold
                    estimator, grid = make_survival_pipeline(
                        model_key=model_key,
                        columns=columns,
                        base=base,
                        outer_fold=outer_fold,
                        imputation_seed=mice_seed,
                        max_iter=args.max_iter,
                        initial_strategy=args.initial_strategy,
                    )
                    search = GridSearchCV(
                        estimator=estimator,
                        param_grid=grid,
                        scoring=survival_scorer,
                        cv=splits,
                        refit=True,
                        n_jobs=args.grid_jobs,
                        return_train_score=False,
                        error_score="raise",
                    )
                    search.fit(X.loc[train_mask], survival_y[train_mask])
                    risk = np.asarray(search.predict(X.loc[test_mask]), dtype=float).reshape(-1)
                    survival_risks[(model_key, feature_key)][test_mask] = risk
                    fold_c = c_index(survival_y[test_mask], risk)
                    fold_rows.append(
                        {
                            "imputation": imputation_index,
                            "task": "survival",
                            "model": model_key,
                            "feature_set": feature_key,
                            "outer_fold": outer_fold,
                            "test_n": int(test_mask.sum()),
                            "test_events": int(survival_y[test_mask]["event"].sum()),
                            "performance": fold_c,
                            "performance_metric": "c_index",
                            "best_inner_performance": float(search.best_score_),
                        }
                    )
                    parameter_rows.append(
                        {
                            "imputation": imputation_index,
                            "task": "survival",
                            "model": model_key,
                            "feature_set": feature_key,
                            "outer_fold": outer_fold,
                            "mice_seed": mice_seed,
                            "best_inner_performance": float(search.best_score_),
                            "best_params": json.dumps(search.best_params_, sort_keys=True),
                        }
                    )
                    print(
                        f"[INFO] MI={imputation_index} {model_key} {feature_key} "
                        f"fold={outer_fold} C={fold_c:.4f}"
                    )

        if any(np.isnan(values).any() for values in binary_predictions.values()):
            raise RuntimeError(f"Incomplete binary OOF predictions for imputation {imputation_index}")
        if any(np.isnan(values).any() for values in survival_risks.values()):
            raise RuntimeError(f"Incomplete survival OOF risks for imputation {imputation_index}")
        binary_by_imputation[imputation_index] = binary_predictions
        survival_by_imputation[imputation_index] = survival_risks

    # Patient-level pooling: mean held-out probability/risk across imputations.
    pooled_binary = {
        feature_key: np.mean(
            np.vstack(
                [binary_by_imputation[index][feature_key] for index in binary_by_imputation]
            ),
            axis=0,
        )
        for feature_key in FEATURE_KEYS
    }
    pooled_survival = {
        (model_key, feature_key): np.mean(
            np.vstack(
                [
                    survival_by_imputation[index][(model_key, feature_key)]
                    for index in survival_by_imputation
                ]
            ),
            axis=0,
        )
        for model_key in ["ridge", "rsf"]
        for feature_key in FEATURE_KEYS
    }

    # Per-imputation performance for transparency.
    per_imputation_rows = []
    for index, predictions in binary_by_imputation.items():
        for feature_key, probability in predictions.items():
            metrics = binary_metrics(binary_y, probability)
            per_imputation_rows.append(
                {
                    "imputation": index,
                    "task": "binary",
                    "model": "binary_rf",
                    "feature_set": feature_key,
                    **metrics,
                }
            )
    for index, risks in survival_by_imputation.items():
        for (model_key, feature_key), risk in risks.items():
            per_imputation_rows.append(
                {
                    "imputation": index,
                    "task": "survival",
                    "model": model_key,
                    "feature_set": feature_key,
                    "c_index": c_index(survival_y, risk),
                }
            )
    per_imputation = pd.DataFrame(per_imputation_rows)

    binary_intervals, binary_comparison = bootstrap_binary(
        binary_y,
        pooled_binary,
        repeats=args.bootstrap_repeats,
        seed=BOOTSTRAP_SEED,
    )
    survival_intervals, survival_comparison = bootstrap_survival(
        survival_y,
        pooled_survival,
        repeats=args.bootstrap_repeats,
        seed=BOOTSTRAP_SEED + 1,
    )

    binary_rows = []
    for feature_key, probability in pooled_binary.items():
        metrics = binary_metrics(binary_y, probability)
        lo, hi = binary_intervals[feature_key]
        binary_rows.append(
            {
                "task": "binary",
                "endpoint": BINARY_TARGET,
                "n": len(binary_y),
                "events": int(binary_y.sum()),
                "model": "binary_rf",
                "model_label": MODEL_LABELS["binary_rf"],
                "feature_set": feature_key,
                "feature_set_label": FEATURE_LABELS[feature_key],
                "n_original_predictors": len(feature_sets[feature_key].columns),
                **metrics,
                "auroc_ci_lower": lo,
                "auroc_ci_upper": hi,
                "n_imputations": args.n_imputations,
            }
        )
    binary_performance = pd.DataFrame(binary_rows)

    survival_rows = []
    for (model_key, feature_key), risk in pooled_survival.items():
        lo, hi = survival_intervals[(model_key, feature_key)]
        survival_rows.append(
            {
                "task": "survival",
                "endpoint": "primary_30d",
                "n": len(survival_y),
                "events": int(survival_y["event"].sum()),
                "model": model_key,
                "model_label": MODEL_LABELS[model_key],
                "feature_set": feature_key,
                "feature_set_label": FEATURE_LABELS[feature_key],
                "n_original_predictors": len(feature_sets[feature_key].columns),
                "c_index": c_index(survival_y, risk),
                "c_index_ci_lower": lo,
                "c_index_ci_upper": hi,
                "n_imputations": args.n_imputations,
            }
        )
    survival_performance = pd.DataFrame(survival_rows)
    paired_comparisons = pd.concat(
        [binary_comparison, survival_comparison], ignore_index=True
    )

    # Private patient-level sensitivity outputs.
    binary_oof = pd.DataFrame(
        {
            "analysis_id": model_data["analysis_id"].astype(int),
            "outer_fold": model_data["outer_fold"].astype(int),
            BINARY_TARGET: binary_y,
            "mice_oof_rf_demo_cmr": pooled_binary["demo_cmr"],
            "mice_oof_rf_full": pooled_binary["full"],
        }
    )
    survival_oof = pd.DataFrame(
        {
            "analysis_id": model_data["analysis_id"].astype(int),
            "outer_fold": model_data["outer_fold"].astype(int),
            SURVIVAL_EVENT: survival_y["event"].astype(int),
            SURVIVAL_TIME: survival_y["time"].astype(float),
        }
    )
    for model_key in ["ridge", "rsf"]:
        for feature_key in FEATURE_KEYS:
            survival_oof[f"mice_risk__{model_key}__{feature_key}"] = pooled_survival[
                (model_key, feature_key)
            ]

    comparison_to_primary = compare_with_primary(
        binary_mice=binary_performance,
        survival_mice=survival_performance,
        binary_primary_path=args.binary_primary,
        survival_primary_path=args.survival_primary,
    )

    fold_performance = pd.DataFrame(fold_rows)
    selected_parameters = pd.DataFrame(parameter_rows)

    write_csv(binary_performance, output_dir / "mice_binary_model_performance.csv")
    write_csv(survival_performance, output_dir / "mice_survival_model_performance.csv")
    write_csv(paired_comparisons, output_dir / "mice_paired_comparisons.csv")
    write_csv(per_imputation, output_dir / "mice_per_imputation_performance.csv")
    write_csv(fold_performance, output_dir / "mice_fold_performance.csv")
    write_csv(selected_parameters, output_dir / "mice_selected_hyperparameters.csv")
    write_csv(comparison_to_primary, output_dir / "mice_vs_primary_comparison.csv")
    write_csv(binary_oof, output_dir / "mice_oof_binary_predictions.csv", sep=";")
    write_csv(survival_oof, output_dir / "mice_oof_survival_risks.csv", sep=";")

    checks = [
        {
            "check": "Analysis dataset hash",
            "status": "PASS" if hash_checks["analysis_dataset"] else "REVIEW",
            "observed": hashes["analysis_dataset"],
            "expected": DATA_SHA256,
        },
        {
            "check": "Vocabulary hash",
            "status": "PASS" if hash_checks["registry_vocabulary"] else "REVIEW",
            "observed": hashes["registry_vocabulary"],
            "expected": VOCAB_SPEC_SHA256,
        },
        {
            "check": "Fold specification hash",
            "status": "PASS" if hash_checks["outer_fold_assignment"] else "REVIEW",
            "observed": hashes["outer_fold_assignment"],
            "expected": FOLD_SPEC_SHA256,
        },
        {
            "check": "Cohort N",
            "status": "PASS" if len(model_data) == EXPECTED_N else "REVIEW",
            "observed": len(model_data),
            "expected": EXPECTED_N,
        },
        {
            "check": "Binary event count",
            "status": "PASS" if int(binary_y.sum()) == BINARY_EVENTS else "REVIEW",
            "observed": int(binary_y.sum()),
            "expected": BINARY_EVENTS,
        },
        {
            "check": "Survival event count",
            "status": "PASS" if int(survival_y["event"].sum()) == SURVIVAL_EVENTS else "REVIEW",
            "observed": int(survival_y["event"].sum()),
            "expected": SURVIVAL_EVENTS,
        },
        {
            "check": "Registry feature count",
            "status": "PASS" if registry.shape[1] == EXPECTED_REGISTRY_FEATURES else "REVIEW",
            "observed": registry.shape[1],
            "expected": EXPECTED_REGISTRY_FEATURES,
        },
        {
            "check": "Binary OOF completeness",
            "status": "PASS"
            if all(not np.isnan(value).any() for value in pooled_binary.values())
            else "REVIEW",
            "observed": sum(int(np.isnan(value).sum()) for value in pooled_binary.values()),
            "expected": 0,
        },
        {
            "check": "Survival OOF completeness",
            "status": "PASS"
            if all(not np.isnan(value).any() for value in pooled_survival.values())
            else "REVIEW",
            "observed": sum(int(np.isnan(value).sum()) for value in pooled_survival.values()),
            "expected": 0,
        },
        {
            "check": "Binary probability range",
            "status": "PASS"
            if all(((value >= 0) & (value <= 1)).all() for value in pooled_binary.values())
            else "REVIEW",
            "observed": "within [0,1]",
            "expected": "within [0,1]",
        },
        {
            "check": "Primary-comparison rows found",
            "status": "PASS" if len(comparison_to_primary) == 6 else "REVIEW",
            "observed": len(comparison_to_primary),
            "expected": 6,
        },
    ]
    checks_frame = pd.DataFrame(checks)
    write_csv(checks_frame, output_dir / "mice_run_checks.csv")
    review_count = int(checks_frame["status"].eq("REVIEW").sum())
    if review_count and not args.allow_hash_mismatch:
        raise RuntimeError(
            f"MICE sensitivity run completed, but {review_count} checks require review"
        )

    manifest = {
        "run": {
            "script": Path(__file__).name,
            "script_version": SCRIPT_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "global_seed": SEED,
        },
        "scope": {
            "binary_endpoint": BINARY_TARGET,
            "binary_events": int(binary_y.sum()),
            "survival_endpoint": [SURVIVAL_EVENT, SURVIVAL_TIME],
            "survival_events": int(survival_y["event"].sum()),
            "feature_sets": FEATURE_KEYS,
            "binary_models": ["binary_rf"],
            "survival_models": ["ridge", "rsf"],
        },
        "multiple_imputation": {
            "method": "stochastic chained equations using sklearn IterativeImputer",
            "estimator": "BayesianRidge",
            "n_imputations": args.n_imputations,
            "max_iter": args.max_iter,
            "sample_posterior": True,
            "initial_strategy": args.initial_strategy,
            "binary_postprocessing": "threshold at 0.5",
            "registry_features": "complete; passed through without imputation",
            "pooling": "mean patient-level held-out prediction/risk across imputations",
            "leakage_control": (
                "imputer is inside each Pipeline; fitted separately within inner-training "
                "splits during tuning and on outer-training data before held-out evaluation"
            ),
        },
        "validation": {
            "outer_folds": 5,
            "outer_assignment": "frozen",
            "inner_folds": N_INNER,
            "inner_base_seed": INNER_BASE_SEED,
        },
        "models": {
            "binary_rf": {
                "n_estimators": RF_N_ESTIMATORS,
                "class_weight": "balanced_subsample",
                "parameter_grid": RF_GRID,
            },
            "ridge": {"parameter_grid": {"alpha": RIDGE_GRID["model__alpha"]}},
            "rsf": {
                "n_estimators": RSF_N_ESTIMATORS,
                "parameter_grid": RSF_GRID,
            },
        },
        "bootstrap": {
            "repeats": args.bootstrap_repeats,
            "binary_seed": BOOTSTRAP_SEED,
            "survival_seed": BOOTSTRAP_SEED + 1,
            "confidence_level": 0.95,
        },
        "frozen_inputs": hashes,
        "checks": {
            "pass": int(checks_frame["status"].eq("PASS").sum()),
            "review": review_count,
        },
        "privacy": (
            "Patient-level MICE OOF prediction and risk files are private and must not "
            "be committed or shared publicly."
        ),
    }
    (output_dir / "mice_run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary_lines = [
        "MICE sensitivity analysis",
        "=" * 25,
        f"N={len(model_data)}",
        f"Imputations={args.n_imputations}",
        f"MICE iterations={args.max_iter}",
        f"Bootstrap repeats={args.bootstrap_repeats}",
        "",
        "Binary Random Forest:",
    ]
    for row in binary_performance.itertuples(index=False):
        summary_lines.append(
            f"- {row.feature_set_label}: AUROC {row.auroc:.4f} "
            f"(95% CI {row.auroc_ci_lower:.4f}-{row.auroc_ci_upper:.4f}); "
            f"AP {row.average_precision:.4f}; Brier {row.brier_score:.4f}"
        )
    summary_lines.append("")
    summary_lines.append("Primary survival models:")
    for row in survival_performance.itertuples(index=False):
        summary_lines.append(
            f"- {row.model_label} | {row.feature_set_label}: C-index "
            f"{row.c_index:.4f} (95% CI {row.c_index_ci_lower:.4f}-"
            f"{row.c_index_ci_upper:.4f})"
        )
    summary_lines.append("")
    summary_lines.append("Comparison with primary median/mode analysis:")
    if comparison_to_primary.empty:
        summary_lines.append("- Primary aggregate files were unavailable or incompatible.")
    else:
        for row in comparison_to_primary.itertuples(index=False):
            summary_lines.append(
                f"- {row.task} | {row.model} | {row.feature_set}: primary "
                f"{row.primary_estimate:.4f}; MICE {row.mice_estimate:.4f}; "
                f"difference {row.absolute_difference:+.4f}"
            )
    summary_lines.extend(
        [
            "",
            f"Checks: {manifest['checks']['pass']} PASS; {manifest['checks']['review']} REVIEW",
            "Interpretation must be based on the estimates and confidence intervals, not on a prespecified claim of equivalence.",
        ]
    )
    (output_dir / "mice_run_summary.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )

    print(f"[DONE] Outputs: {output_dir}")
    print(
        f"[INFO] Checks: {manifest['checks']['pass']} PASS; "
        f"{manifest['checks']['review']} REVIEW"
    )
    print("[INFO] Primary analysis files were not modified.")
    print("[INFO] Patient-level sensitivity outputs remain private.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
