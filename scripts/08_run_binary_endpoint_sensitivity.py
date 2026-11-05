#!/usr/bin/env python3
"""
08_run_binary_endpoint_sensitivity.py

Run deterministic nested-CV Random Forest sensitivity analyses for endpoints
that reduce or remove revascularization from MACE.

Endpoints
---------
1. mace_late_revasc_event
   CV death, AMI, or PCI/CABG occurring more than 6 months after index CMR.
   Expected events: 150.

2. hard_mace_event
   CV death or AMI; revascularization excluded entirely.
   Expected events: 83.

Feature configurations
----------------------
- demo_cmr: demographics + CMR
- full: registry + demographics + CMR

The script uses the same frozen dataset, registry vocabulary, outer folds,
preprocessing, RF tuning grid, seeds, and nested-CV design as
07_run_binary_rf_nested_cv.py.

Required files
--------------
- scripts/07_run_binary_rf_nested_cv.py
- data/processed/cmr_revision_analysis.csv
- data/processed/outer_cv_folds.csv
- config/registry_vocabulary.json
- config/cv_plan.json

Primary binary-model results are read from:
- results/models/binary_rf/metrics_summary.csv
- results/models/binary_rf/primary_comparison.csv

Outputs
-------
results/models/binary_rf_endpoint_sensitivity/
- sensitivity_metrics_long.csv
- sensitivity_primary_comparisons.csv
- supplemental_table_binary_endpoint_sensitivity.csv
- supplemental_table_binary_endpoint_sensitivity_formatted.csv
- selected_hyperparameters.csv
- fold_metrics.csv
- oof_predictions.csv                 [private patient-level output]
- sensitivity_roc_panels.png/.tiff/.pdf/.svg
- run_checks.csv
- run_manifest.json
- run_summary.txt

Run from repository root:
    python scripts/08_run_binary_endpoint_sensitivity.py
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
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold

SCRIPT_VERSION = "1.0.0"
GLOBAL_SEED = 42
INNER_BASE_SEED = 2026
N_INNER_SPLITS = 5
BOOTSTRAP_SEED_BASE = 8042

DEFAULT_BASE_SCRIPT = Path("scripts/07_run_binary_rf_nested_cv.py")
DEFAULT_DATA = Path("data/processed/cmr_revision_analysis.csv")
DEFAULT_FOLDS = Path("data/processed/outer_cv_folds.csv")
DEFAULT_VOCAB = Path("config/registry_vocabulary.json")
DEFAULT_CV_PLAN = Path("config/cv_plan.json")
DEFAULT_PRIMARY_DIR = Path("results/models/binary_rf")
DEFAULT_OUTPUT_DIR = Path("results/models/binary_rf_endpoint_sensitivity")

EXPECTED_DATA_SHA256 = "e94b8aebe4c4939b0b3def943879539f4f242ba7ef39fd7ac929407578f7db9c"
EXPECTED_VOCAB_SPEC_SHA256 = "1250d92ca3f5d5da2049f3965837a7cfa0b70daaa9ffae2e09d71f8235eac023"
EXPECTED_FOLD_SPEC_SHA256 = "69c4a862c97d96a7c29e21dcc9c3a1bb7b74f87f1d0baee6c26b4f87fda219eb"
EXPECTED_N = 2159

ENDPOINTS = {
    "late_revascularization": {
        "column": "mace_late_revasc_event",
        "label": "CV death, AMI, or PCI/CABG after 6 months",
        "short_label": "Late-revascularization composite",
        "expected_events": 150,
    },
    "hard_mace": {
        "column": "hard_mace_event",
        "label": "Hard MACE: CV death or AMI",
        "short_label": "Hard MACE",
        "expected_events": 83,
    },
}

FEATURE_KEYS = ["demo_cmr", "full"]
FEATURE_LABELS = {
    "demo_cmr": "Demographics + CMR",
    "full": "Registry + demographics + CMR",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run binary RF endpoint-sensitivity analyses."
    )
    parser.add_argument("--base-script", type=Path, default=DEFAULT_BASE_SCRIPT)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--vocabulary", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--cv-plan", type=Path, default=DEFAULT_CV_PLAN)
    parser.add_argument("--primary-results-dir", type=Path, default=DEFAULT_PRIMARY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--grid-jobs", type=int, default=-1)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--allow-hash-mismatch", action="store_true")
    return parser.parse_args()


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


def load_base_module(path: Path):
    spec = importlib.util.spec_from_file_location("binary_rf_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load base pipeline: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def stratified_bootstrap_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    events = np.flatnonzero(y == 1)
    nonevents = np.flatnonzero(y == 0)
    sampled = np.concatenate([
        rng.choice(events, size=len(events), replace=True),
        rng.choice(nonevents, size=len(nonevents), replace=True),
    ])
    rng.shuffle(sampled)
    return sampled


def bootstrap_endpoint(
    base,
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
    repeats: int,
    seed: int,
) -> tuple[dict[str, dict[str, tuple[float, float]]], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    metrics = ["auroc", "average_precision", "brier_score"]
    storage = {
        key: {metric: np.empty(repeats, dtype=float) for metric in metrics}
        for key in predictions
    }

    for i in range(repeats):
        idx = stratified_bootstrap_indices(y, rng)
        y_b = y[idx]
        for key, p in predictions.items():
            p_b = p[idx]
            storage[key]["auroc"][i] = roc_auc_score(y_b, p_b)
            storage[key]["average_precision"][i] = average_precision_score(y_b, p_b)
            storage[key]["brier_score"][i] = brier_score_loss(y_b, p_b)

    intervals = {}
    for key in predictions:
        intervals[key] = {}
        for metric in metrics:
            values = storage[key][metric]
            intervals[key][metric] = (
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            )

    comparison_rows = []
    observed_demo = base.metric_values(y, predictions["demo_cmr"])
    observed_full = base.metric_values(y, predictions["full"])
    for metric in metrics:
        delta_samples = storage["full"][metric] - storage["demo_cmr"][metric]
        comparison_rows.append({
            "metric": metric,
            "estimate": observed_full[metric] - observed_demo[metric],
            "ci_lower": float(np.quantile(delta_samples, 0.025)),
            "ci_upper": float(np.quantile(delta_samples, 0.975)),
            "bootstrap_repeats": repeats,
            "bootstrap_seed": seed,
        })

    return intervals, pd.DataFrame(comparison_rows)


def run_endpoint(
    base,
    df_model: pd.DataFrame,
    feature_sets: dict,
    endpoint_key: str,
    endpoint_info: dict[str, Any],
    grid_jobs: int,
    bootstrap_repeats: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    target = endpoint_info["column"]
    y = pd.to_numeric(df_model[target], errors="coerce")
    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError(f"{target} is missing or nonbinary.")
    y = y.astype(int)
    y_np = y.to_numpy()
    observed_events = int(y_np.sum())
    if observed_events != endpoint_info["expected_events"]:
        raise ValueError(
            f"{target}: expected {endpoint_info['expected_events']} events; observed {observed_events}."
        )

    print(
        f"[INFO] Endpoint: {endpoint_info['short_label']} | "
        f"N={len(y_np)} | events={observed_events} ({y_np.mean() * 100:.2f}%)"
    )

    predictions = {key: np.full(len(df_model), np.nan, dtype=float) for key in FEATURE_KEYS}
    fold_rows = []
    parameter_rows = []
    outer_folds = sorted(df_model["outer_fold"].astype(int).unique().tolist())

    for feature_key in FEATURE_KEYS:
        feature_set = feature_sets[feature_key]
        X_all = df_model[feature_set.columns].copy()
        print(f"[INFO] Starting {endpoint_key}: {feature_set.label} ({len(feature_set.columns)} predictors)")

        for outer_fold in outer_folds:
            test_mask = df_model["outer_fold"].astype(int).eq(outer_fold).to_numpy()
            train_mask = ~test_mask
            X_train = X_all.loc[train_mask]
            X_test = X_all.loc[test_mask]
            y_train = y.loc[train_mask]
            y_test = y.loc[test_mask]

            if int(y_train.sum()) < N_INNER_SPLITS:
                raise ValueError(f"Too few training events for inner CV in {endpoint_key}, fold {outer_fold}.")

            inner_cv = StratifiedKFold(
                n_splits=N_INNER_SPLITS,
                shuffle=True,
                random_state=INNER_BASE_SEED + outer_fold,
            )
            pipeline = base.make_pipeline(feature_set.columns, outer_fold)
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=base.PARAM_GRID,
                scoring="roc_auc",
                cv=inner_cv,
                refit=True,
                n_jobs=grid_jobs,
                return_train_score=False,
                error_score="raise",
            )
            search.fit(X_train, y_train)
            p_test = search.predict_proba(X_test)[:, 1]
            predictions[feature_key][test_mask] = p_test

            fold_rows.append({
                "endpoint": endpoint_key,
                "endpoint_label": endpoint_info["label"],
                "feature_set": feature_key,
                "feature_set_label": FEATURE_LABELS[feature_key],
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
                "endpoint": endpoint_key,
                "endpoint_label": endpoint_info["label"],
                "feature_set": feature_key,
                "feature_set_label": FEATURE_LABELS[feature_key],
                "outer_fold": outer_fold,
                "best_inner_auroc": float(search.best_score_),
                "max_depth": search.best_params_["classifier__max_depth"],
                "max_features": search.best_params_["classifier__max_features"],
                "min_samples_leaf": search.best_params_["classifier__min_samples_leaf"],
                "n_estimators": base.N_ESTIMATORS,
                "rf_random_state": GLOBAL_SEED + outer_fold,
                "inner_random_state": INNER_BASE_SEED + outer_fold,
            })
            print(
                f"[INFO] {endpoint_key} | {feature_key} | fold {outer_fold} | "
                f"AUROC={roc_auc_score(y_test, p_test):.4f} | best={search.best_params_}"
            )

        if np.isnan(predictions[feature_key]).any():
            raise RuntimeError(f"Missing OOF predictions for {endpoint_key}/{feature_key}.")

    seed = BOOTSTRAP_SEED_BASE + list(ENDPOINTS).index(endpoint_key)
    intervals, comparison = bootstrap_endpoint(
        base, y_np, predictions, bootstrap_repeats, seed
    )
    comparison.insert(0, "endpoint", endpoint_key)
    comparison.insert(1, "endpoint_label", endpoint_info["label"])
    comparison.insert(2, "comparison", "Full minus demographics + CMR")

    metric_rows = []
    for feature_key in FEATURE_KEYS:
        values = base.metric_values(y_np, predictions[feature_key])
        metric_rows.append({
            "endpoint": endpoint_key,
            "endpoint_label": endpoint_info["label"],
            "n": len(y_np),
            "events": observed_events,
            "event_percent": y_np.mean() * 100.0,
            "feature_set": feature_key,
            "feature_set_label": FEATURE_LABELS[feature_key],
            "n_original_predictors": len(feature_sets[feature_key].columns),
            **values,
            "auroc_ci_lower": intervals[feature_key]["auroc"][0],
            "auroc_ci_upper": intervals[feature_key]["auroc"][1],
            "average_precision_ci_lower": intervals[feature_key]["average_precision"][0],
            "average_precision_ci_upper": intervals[feature_key]["average_precision"][1],
            "brier_ci_lower": intervals[feature_key]["brier_score"][0],
            "brier_ci_upper": intervals[feature_key]["brier_score"][1],
            "bootstrap_repeats": bootstrap_repeats,
            "bootstrap_seed": seed,
        })

    oof = pd.DataFrame({
        "analysis_id": df_model["analysis_id"].astype(int),
        "outer_fold": df_model["outer_fold"].astype(int),
        "endpoint": endpoint_key,
        "event": y_np,
        "oof_rf_demo_cmr": predictions["demo_cmr"],
        "oof_rf_full": predictions["full"],
    })

    return (
        pd.DataFrame(metric_rows),
        comparison,
        pd.DataFrame(fold_rows),
        predictions,
        pd.DataFrame(parameter_rows),
        oof,
    )


def plot_sensitivity_roc(
    endpoint_results: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]],
    output_dir: Path,
) -> dict[str, str]:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.0), sharex=True, sharey=True)
    colors = {"demo_cmr": "#1b9e77", "full": "#1f4e79"}
    linestyles = {"demo_cmr": "--", "full": "-"}

    for ax, endpoint_key in zip(axes, ENDPOINTS):
        y, predictions = endpoint_results[endpoint_key]
        for feature_key in FEATURE_KEYS:
            fpr, tpr, _ = roc_curve(y, predictions[feature_key])
            auc = roc_auc_score(y, predictions[feature_key])
            ax.plot(
                fpr, tpr,
                color=colors[feature_key],
                linestyle=linestyles[feature_key],
                linewidth=2.2,
                label=f"{FEATURE_LABELS[feature_key]} (AUROC {auc:.2f})",
            )
        ax.plot([0, 1], [0, 1], color="#999999", linestyle=":", linewidth=1.2)
        ax.set_title(ENDPOINTS[endpoint_key]["short_label"])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.01)
        ax.set_xlabel("1 - specificity")
        ax.grid(False)
        ax.legend(loc="lower right", frameon=False, fontsize=8.5)

    axes[0].set_ylabel("Sensitivity")
    fig.suptitle("Binary Random Forest Endpoint Sensitivity Analyses", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    paths = {
        "png": output_dir / "sensitivity_roc_panels.png",
        "tiff": output_dir / "sensitivity_roc_panels.tiff",
        "pdf": output_dir / "sensitivity_roc_panels.pdf",
        "svg": output_dir / "sensitivity_roc_panels.svg",
    }
    fig.savefig(paths["png"], dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(paths["tiff"], dpi=600, format="tiff", facecolor="white",
                bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(paths["pdf"], facecolor="white", bbox_inches="tight")
    fig.savefig(paths["svg"], facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return {key: str(path) for key, path in paths.items()}


def load_primary_results(primary_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_path = primary_dir / "metrics_summary.csv"
    comparison_path = primary_dir / "primary_comparison.csv"
    if not metrics_path.exists() or not comparison_path.exists():
        raise FileNotFoundError(
            "Primary binary RF result files are missing. Run "
            "scripts/07_run_binary_rf_nested_cv.py first."
        )
    metrics = pd.read_csv(metrics_path)
    comparison = pd.read_csv(comparison_path)
    metrics = metrics[metrics["feature_set"].isin(FEATURE_KEYS)].copy()
    if set(metrics["feature_set"]) != set(FEATURE_KEYS):
        raise ValueError("Primary metrics do not contain both demo_cmr and full.")
    return metrics, comparison


def build_supplemental_table(
    primary_metrics: pd.DataFrame,
    primary_comparison: pd.DataFrame,
    sensitivity_metrics: pd.DataFrame,
    sensitivity_comparisons: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = primary_metrics.copy()
    primary.insert(0, "endpoint", "primary")
    primary.insert(1, "endpoint_label", "Primary composite: CV death, AMI, or PCI/CABG")
    primary["event_percent"] = primary["events"] / primary["n"] * 100.0

    common_cols = [
        "endpoint", "endpoint_label", "n", "events", "event_percent",
        "feature_set", "feature_set_label", "n_original_predictors",
        "auroc", "auroc_ci_lower", "auroc_ci_upper",
        "average_precision", "average_precision_ci_lower", "average_precision_ci_upper",
        "brier_score", "brier_ci_lower", "brier_ci_upper",
        "calibration_intercept", "calibration_slope",
    ]
    long_table = pd.concat([
        primary[common_cols],
        sensitivity_metrics[common_cols],
    ], ignore_index=True)

    all_comparisons = []
    primary_cmp = primary_comparison.copy()
    primary_cmp.insert(0, "endpoint", "primary")
    primary_cmp.insert(1, "endpoint_label", "Primary composite: CV death, AMI, or PCI/CABG")
    all_comparisons.append(primary_cmp)
    all_comparisons.append(sensitivity_comparisons)
    comparison = pd.concat(all_comparisons, ignore_index=True)

    delta_lookup = {
        (row.endpoint, row.metric): row
        for row in comparison.itertuples(index=False)
    }

    rows = []
    endpoint_order = ["primary", "late_revascularization", "hard_mace"]
    for endpoint in endpoint_order:
        subset = long_table[long_table["endpoint"].eq(endpoint)]
        for row in subset.sort_values("feature_set", key=lambda s: s.map({"demo_cmr": 0, "full": 1})).itertuples(index=False):
            auc_delta = delta_lookup.get((endpoint, "auroc")) if row.feature_set == "full" else None
            ap_delta = delta_lookup.get((endpoint, "average_precision")) if row.feature_set == "full" else None
            brier_delta = delta_lookup.get((endpoint, "brier_score")) if row.feature_set == "full" else None
            rows.append({
                "endpoint": endpoint,
                "endpoint_label": row.endpoint_label,
                "n": int(row.n),
                "events": int(row.events),
                "event_percent": row.event_percent,
                "feature_set": row.feature_set,
                "feature_set_label": row.feature_set_label,
                "auroc": row.auroc,
                "auroc_ci_lower": row.auroc_ci_lower,
                "auroc_ci_upper": row.auroc_ci_upper,
                "delta_auroc_vs_demo_cmr": auc_delta.estimate if auc_delta else np.nan,
                "delta_auroc_ci_lower": auc_delta.ci_lower if auc_delta else np.nan,
                "delta_auroc_ci_upper": auc_delta.ci_upper if auc_delta else np.nan,
                "average_precision": row.average_precision,
                "delta_average_precision_vs_demo_cmr": ap_delta.estimate if ap_delta else np.nan,
                "brier_score": row.brier_score,
                "delta_brier_vs_demo_cmr": brier_delta.estimate if brier_delta else np.nan,
                "calibration_intercept": row.calibration_intercept,
                "calibration_slope": row.calibration_slope,
            })
    numeric_table = pd.DataFrame(rows)

    formatted = numeric_table.copy()
    formatted["Events"] = formatted.apply(
        lambda r: f"{int(r['events'])}/{int(r['n'])} ({r['event_percent']:.1f}%)", axis=1
    )
    formatted["Model"] = formatted["feature_set_label"]
    formatted["AUROC (95% CI)"] = formatted.apply(
        lambda r: f"{r['auroc']:.2f} ({r['auroc_ci_lower']:.2f}-{r['auroc_ci_upper']:.2f})", axis=1
    )
    formatted["Delta AUROC (95% CI)"] = formatted.apply(
        lambda r: "Reference" if pd.isna(r["delta_auroc_vs_demo_cmr"]) else
        f"{r['delta_auroc_vs_demo_cmr']:.3f} ({r['delta_auroc_ci_lower']:.3f}-{r['delta_auroc_ci_upper']:.3f})",
        axis=1,
    )
    formatted["Average precision"] = formatted["average_precision"].map(lambda x: f"{x:.2f}")
    formatted["Brier score"] = formatted["brier_score"].map(lambda x: f"{x:.3f}")
    formatted["Calibration intercept"] = formatted["calibration_intercept"].map(lambda x: f"{x:.2f}")
    formatted["Calibration slope"] = formatted["calibration_slope"].map(lambda x: f"{x:.2f}")
    formatted = formatted[[
        "endpoint_label", "Events", "Model", "AUROC (95% CI)",
        "Delta AUROC (95% CI)", "Average precision", "Brier score",
        "Calibration intercept", "Calibration slope",
    ]].rename(columns={"endpoint_label": "Endpoint"})

    return numeric_table, formatted


def write_csv(df: pd.DataFrame, path: Path, sep: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, sep=sep, encoding="utf-8", float_format="%.12g")


def main() -> int:
    args = parse_args()
    set_determinism(GLOBAL_SEED)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    required_paths = [args.base_script, args.data, args.folds, args.vocabulary, args.cv_plan]
    for path in required_paths:
        if not path.exists():
            print(f"[ERROR] Required file not found: {path}", file=sys.stderr)
            return 2

    base = load_base_module(args.base_script)
    data_hash = sha256_file(args.data)
    vocabulary = load_json(args.vocabulary)
    cv_plan = load_json(args.cv_plan)
    vocab_hash = vocabulary.get("vocabulary_sha256")
    fold_hash = cv_plan["outer_cross_validation"].get("assignment_specification_sha256")

    hash_checks = {
        "analysis_dataset": data_hash == EXPECTED_DATA_SHA256,
        "registry_vocabulary": vocab_hash == EXPECTED_VOCAB_SPEC_SHA256,
        "outer_folds": fold_hash == EXPECTED_FOLD_SPEC_SHA256,
    }
    if not all(hash_checks.values()) and not args.allow_hash_mismatch:
        raise RuntimeError(f"Frozen input hash mismatch: {hash_checks}")

    data_delimiter = detect_delimiter(args.data)
    fold_delimiter = detect_delimiter(args.folds)
    df = pd.read_csv(args.data, sep=data_delimiter, encoding="utf-8-sig", low_memory=False)
    folds = pd.read_csv(args.folds, sep=fold_delimiter, encoding="utf-8-sig")

    if len(df) != EXPECTED_N or len(folds) != EXPECTED_N:
        raise ValueError(f"Expected {EXPECTED_N} rows in data and fold map.")
    df = df.merge(folds[["analysis_id", "outer_fold"]], on="analysis_id", how="left", validate="one_to_one")
    if df["outer_fold"].isna().any():
        raise ValueError("Missing outer-fold assignment.")
    df = df.sort_values("analysis_id", kind="mergesort").reset_index(drop=True)

    registry_matrix = base.build_registry_matrix(df, vocabulary)
    if registry_matrix.shape[1] != 62:
        raise ValueError(f"Expected 62 registry features; observed {registry_matrix.shape[1]}.")
    df_model = pd.concat([df, registry_matrix], axis=1)
    feature_sets = base.make_feature_sets(registry_matrix.columns.tolist())

    print(f"[INFO] Script version: {SCRIPT_VERSION}")
    print(f"[INFO] Base pipeline version: {base.SCRIPT_VERSION}")
    print(f"[INFO] Dataset N={len(df_model)}; registry features={registry_matrix.shape[1]}")
    print(f"[INFO] Feature sets: {', '.join(FEATURE_KEYS)}")
    print(f"[INFO] RF grid combinations: {math.prod(len(v) for v in base.PARAM_GRID.values())}")

    all_metrics = []
    all_comparisons = []
    all_fold_metrics = []
    all_parameters = []
    all_oof = []
    plot_results = {}

    for endpoint_key, endpoint_info in ENDPOINTS.items():
        metrics, comparison, fold_metrics, predictions, parameters, oof = run_endpoint(
            base=base,
            df_model=df_model,
            feature_sets=feature_sets,
            endpoint_key=endpoint_key,
            endpoint_info=endpoint_info,
            grid_jobs=args.grid_jobs,
            bootstrap_repeats=args.bootstrap_repeats,
        )
        all_metrics.append(metrics)
        all_comparisons.append(comparison)
        all_fold_metrics.append(fold_metrics)
        all_parameters.append(parameters)
        all_oof.append(oof)
        y = pd.to_numeric(df_model[endpoint_info["column"]], errors="raise").astype(int).to_numpy()
        plot_results[endpoint_key] = (y, predictions)

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    comparisons_df = pd.concat(all_comparisons, ignore_index=True)
    fold_metrics_df = pd.concat(all_fold_metrics, ignore_index=True)
    parameters_df = pd.concat(all_parameters, ignore_index=True)
    oof_df = pd.concat(all_oof, ignore_index=True)

    primary_metrics, primary_comparison = load_primary_results(args.primary_results_dir)
    supplemental_numeric, supplemental_formatted = build_supplemental_table(
        primary_metrics,
        primary_comparison,
        metrics_df,
        comparisons_df,
    )

    write_csv(metrics_df, output_dir / "sensitivity_metrics_long.csv")
    write_csv(comparisons_df, output_dir / "sensitivity_primary_comparisons.csv")
    write_csv(fold_metrics_df, output_dir / "fold_metrics.csv")
    write_csv(parameters_df, output_dir / "selected_hyperparameters.csv")
    write_csv(oof_df, output_dir / "oof_predictions.csv", sep=";")
    write_csv(supplemental_numeric, output_dir / "supplemental_table_binary_endpoint_sensitivity.csv")
    write_csv(supplemental_formatted, output_dir / "supplemental_table_binary_endpoint_sensitivity_formatted.csv")
    figure_paths = plot_sensitivity_roc(plot_results, output_dir)

    checks = [
        {"check": "Analysis dataset hash", "status": "PASS" if hash_checks["analysis_dataset"] else "REVIEW",
         "observed": data_hash, "expected": EXPECTED_DATA_SHA256},
        {"check": "Vocabulary hash", "status": "PASS" if hash_checks["registry_vocabulary"] else "REVIEW",
         "observed": vocab_hash, "expected": EXPECTED_VOCAB_SPEC_SHA256},
        {"check": "Outer-fold hash", "status": "PASS" if hash_checks["outer_folds"] else "REVIEW",
         "observed": fold_hash, "expected": EXPECTED_FOLD_SPEC_SHA256},
        {"check": "Cohort N", "status": "PASS" if len(df_model) == EXPECTED_N else "REVIEW",
         "observed": len(df_model), "expected": EXPECTED_N},
        {"check": "Late-revascularization events",
         "status": "PASS" if int(df_model["mace_late_revasc_event"].sum()) == 150 else "REVIEW",
         "observed": int(df_model["mace_late_revasc_event"].sum()), "expected": 150},
        {"check": "Hard-MACE events",
         "status": "PASS" if int(df_model["hard_mace_event"].sum()) == 83 else "REVIEW",
         "observed": int(df_model["hard_mace_event"].sum()), "expected": 83},
        {"check": "Sensitivity OOF completeness",
         "status": "PASS" if not oof_df[["oof_rf_demo_cmr", "oof_rf_full"]].isna().any().any() else "REVIEW",
         "observed": int(oof_df[["oof_rf_demo_cmr", "oof_rf_full"]].isna().sum().sum()), "expected": 0},
        {"check": "Supplemental table endpoints",
         "status": "PASS" if supplemental_numeric["endpoint"].nunique() == 3 else "REVIEW",
         "observed": int(supplemental_numeric["endpoint"].nunique()), "expected": 3},
        {"check": "Supplemental table rows",
         "status": "PASS" if len(supplemental_numeric) == 6 else "REVIEW",
         "observed": len(supplemental_numeric), "expected": 6},
    ]
    checks_df = pd.DataFrame(checks)
    write_csv(checks_df, output_dir / "run_checks.csv")
    review_count = int(checks_df["status"].eq("REVIEW").sum())
    if review_count and not args.allow_hash_mismatch:
        raise RuntimeError(f"Sensitivity run completed with {review_count} checks requiring review.")

    output_hashes = {
        path.name: sha256_file(path)
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "run_manifest.json"
    }
    manifest = {
        "run": {
            "script": Path(__file__).name,
            "script_version": SCRIPT_VERSION,
            "base_pipeline": str(args.base_script),
            "base_pipeline_version": base.SCRIPT_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "global_seed": GLOBAL_SEED,
        },
        "frozen_inputs": {
            "analysis_dataset_sha256": data_hash,
            "registry_vocabulary_specification_sha256": vocab_hash,
            "outer_fold_assignment_specification_sha256": fold_hash,
        },
        "endpoints": ENDPOINTS,
        "feature_sets": FEATURE_KEYS,
        "validation": {
            "outer_folds": 5,
            "inner_folds": N_INNER_SPLITS,
            "inner_base_seed": INNER_BASE_SEED,
            "grid_scoring": "roc_auc",
            "preprocessing_within_training_data": True,
        },
        "bootstrap": {
            "repeats": args.bootstrap_repeats,
            "endpoint_seed_base": BOOTSTRAP_SEED_BASE,
            "method": "stratified paired patient bootstrap",
        },
        "outputs": output_hashes,
        "checks": {
            "pass": int(checks_df["status"].eq("PASS").sum()),
            "review": review_count,
        },
        "privacy": "oof_predictions.csv is patient-level and must remain private.",
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    lines = [
        "Binary RF endpoint sensitivity analysis",
        "=" * 39,
        f"Dataset N: {len(df_model):,}",
        "Feature configurations: demographics + CMR; full registry + demographics + CMR",
        f"Bootstrap repeats: {args.bootstrap_repeats}",
        "",
    ]
    for endpoint_key in ENDPOINTS:
        lines.append(ENDPOINTS[endpoint_key]["short_label"] + ":")
        subset = metrics_df[metrics_df["endpoint"].eq(endpoint_key)]
        for row in subset.itertuples(index=False):
            lines.append(
                f"- {row.feature_set_label}: AUROC {row.auroc:.4f} "
                f"(95% CI {row.auroc_ci_lower:.4f}-{row.auroc_ci_upper:.4f}); "
                f"AP {row.average_precision:.4f}; Brier {row.brier_score:.4f}; "
                f"calibration intercept {row.calibration_intercept:.4f}; "
                f"slope {row.calibration_slope:.4f}"
            )
        cmp_subset = comparisons_df[
            comparisons_df["endpoint"].eq(endpoint_key)
            & comparisons_df["metric"].eq("auroc")
        ]
        if not cmp_subset.empty:
            row = cmp_subset.iloc[0]
            lines.append(
                f"- Paired AUROC difference: {row['estimate']:.4f} "
                f"(95% CI {row['ci_lower']:.4f}-{row['ci_upper']:.4f})"
            )
        lines.append("")

    lines.extend([
        "Supplemental table:",
        str(output_dir / "supplemental_table_binary_endpoint_sensitivity_formatted.csv"),
        "",
        f"Checks: {manifest['checks']['pass']} PASS; {manifest['checks']['review']} REVIEW",
        "ROC legends use two decimal places; machine-readable tables retain full precision.",
    ])
    (output_dir / "run_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[DONE] Outputs: {output_dir}")
    print(f"[INFO] Supplemental table: {output_dir / 'supplemental_table_binary_endpoint_sensitivity_formatted.csv'}")
    print(f"[INFO] Checks: {manifest['checks']['pass']} PASS; {manifest['checks']['review']} REVIEW")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
