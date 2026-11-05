#!/usr/bin/env python3
"""Lightweight fixed-parameter RSF sensitivity analysis for hard MACE.

This script fits only 10 forests:
- 5 frozen outer folds x 2 predictor sets.

The RSF specification is carried forward from the primary nested-CV analysis:
- 800 trees
- max_depth = 10
- max_features = "sqrt"
- min_samples_leaf = 20
- min_samples_split = 10

No hard-MACE-specific hyperparameter tuning is performed. This is intentional:
hard MACE is a supplementary sensitivity endpoint with only 83 events, and the
model specification was selected using the primary endpoint before evaluating
this sensitivity endpoint.

Outputs are compatible with 13_run_hard_mace_survival_sensitivity.py. After this
script completes, run:

    python scripts/13_run_hard_mace_survival_sensitivity.py --skip-fitting
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
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

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

SCRIPT_VERSION = "1.0.0"
SEED = 42
BOOTSTRAP_SEED = 14042
EXPECTED_N = 2159
EXPECTED_EVENTS = 83
EXPECTED_REGISTRY_FEATURES = 62

DATA_SHA = "e94b8aebe4c4939b0b3def943879539f4f242ba7ef39fd7ac929407578f7db9c"
VOCAB_SHA = "1250d92ca3f5d5da2049f3965837a7cfa0b70daaa9ffae2e09d71f8235eac023"
FOLD_SHA = "69c4a862c97d96a7c29e21dcc9c3a1bb7b74f87f1d0baee6c26b4f87fda219eb"

FEATURES = ["demo_cmr", "full"]
FEATURE_LABELS = {
    "demo_cmr": "Demographics + CMR",
    "full": "Registry + demographics + CMR",
}

FIXED_PARAMETERS = {
    "n_estimators": 800,
    "max_depth": 10,
    "max_features": "sqrt",
    "min_samples_leaf": 20,
    "min_samples_split": 10,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-script", type=Path,
        default=Path("scripts/07_run_binary_rf_nested_cv.py"),
    )
    parser.add_argument(
        "--data", type=Path,
        default=Path("data/processed/cmr_revision_analysis.csv"),
    )
    parser.add_argument(
        "--folds", type=Path,
        default=Path("data/processed/outer_cv_folds.csv"),
    )
    parser.add_argument(
        "--vocabulary", type=Path,
        default=Path("config/registry_vocabulary.json"),
    )
    parser.add_argument(
        "--cv-plan", type=Path,
        default=Path("config/cv_plan.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/models/hard_mace_survival_rsf"),
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--allow-hash-mismatch", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        sample = handle.read(131072)
    try:
        return csv.Sniffer().sniff(sample, delimiters=";,\t|").delimiter
    except csv.Error:
        first = sample.splitlines()[0] if sample else ""
        return max([";", ",", "\t", "|"], key=first.count)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_base(path: Path):
    spec = importlib.util.spec_from_file_location("shared_rf_features", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import shared feature definitions from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def cindex(y, risk) -> float:
    return float(concordance_index_censored(y["event"], y["time"], risk)[0])


def make_preprocessor(columns: list[str], base) -> ColumnTransformer:
    continuous = [column for column in columns if column in base.CONTINUOUS_CLINICAL]
    binary = [column for column in columns if column in base.BINARY_CLINICAL]
    registry = [
        column for column in columns
        if column.startswith("pre_icd__") or column.startswith("pre_kva__")
    ]
    transformers = []
    if continuous:
        transformers.append((
            "continuous",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            continuous,
        ))
    if binary:
        transformers.append((
            "binary",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("scaler", StandardScaler()),
            ]),
            binary,
        ))
    if registry:
        transformers.append(("registry", StandardScaler(), registry))
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
    )


def bootstrap_ci(y, risk, repeats: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    event_indices = np.flatnonzero(y["event"])
    censor_indices = np.flatnonzero(~y["event"])
    values = np.empty(repeats)
    for repeat in range(repeats):
        indices = np.concatenate([
            rng.choice(event_indices, len(event_indices), replace=True),
            rng.choice(censor_indices, len(censor_indices), replace=True),
        ])
        rng.shuffle(indices)
        values[repeat] = cindex(y[indices], risk[indices])
    lower, upper = np.quantile(values, [0.025, 0.975])
    return float(lower), float(upper)


def write_csv(frame: pd.DataFrame, path: Path, separator: str = ","):
    frame.to_csv(
        path, index=False, sep=separator,
        encoding="utf-8-sig", float_format="%.12g",
    )


def main():
    args = parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in [args.base_script, args.data, args.folds, args.vocabulary, args.cv_plan]:
        if not path.exists():
            raise FileNotFoundError(path)

    base = load_base(args.base_script)
    vocabulary = load_json(args.vocabulary)
    cv_plan = load_json(args.cv_plan)
    hashes = {
        "dataset": sha256(args.data),
        "vocabulary": vocabulary.get("vocabulary_sha256"),
        "folds": cv_plan["outer_cross_validation"].get("assignment_specification_sha256"),
    }
    hash_valid = {
        "dataset": hashes["dataset"] == DATA_SHA,
        "vocabulary": hashes["vocabulary"] == VOCAB_SHA,
        "folds": hashes["folds"] == FOLD_SHA,
    }
    if not all(hash_valid.values()) and not args.allow_hash_mismatch:
        raise RuntimeError(f"Frozen input mismatch: {hash_valid}")

    data = pd.read_csv(
        args.data,
        sep=detect_delimiter(args.data),
        encoding="utf-8-sig",
        low_memory=False,
    )
    fold_map = pd.read_csv(
        args.folds,
        sep=detect_delimiter(args.folds),
        encoding="utf-8-sig",
    )
    data = data.merge(
        fold_map[["analysis_id", "outer_fold"]],
        on="analysis_id", how="left", validate="one_to_one",
    ).sort_values("analysis_id").reset_index(drop=True)

    if len(data) != EXPECTED_N or data["outer_fold"].isna().any():
        raise ValueError("Cohort size or fold assignment is invalid")

    registry = base.build_registry_matrix(data, vocabulary)
    if registry.shape[1] != EXPECTED_REGISTRY_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_REGISTRY_FEATURES} registry features; "
            f"observed {registry.shape[1]}"
        )
    model_data = pd.concat([data, registry], axis=1)
    feature_sets = base.make_feature_sets(registry.columns.tolist())

    event = pd.to_numeric(model_data["hard_mace_event"], errors="raise").astype(bool)
    time = pd.to_numeric(model_data["hard_mace_tte_years"], errors="raise").astype(float)
    if int(event.sum()) != EXPECTED_EVENTS or (time <= 0).any():
        raise ValueError("Hard-MACE endpoint validation failed")
    y = Surv.from_arrays(event=event, time=time)

    print(f"[INFO] Script {SCRIPT_VERSION}; N={len(model_data)}; hard-MACE events={int(event.sum())}")
    print(f"[INFO] Fixed RSF parameters: {FIXED_PARAMETERS}")

    fold_rows = []
    parameter_rows = []
    risks = {}
    outer_folds = sorted(model_data["outer_fold"].astype(int).unique())

    for feature in FEATURES:
        columns = feature_sets[feature].columns
        X = model_data[columns].copy()
        oof_risk = np.full(len(model_data), np.nan)
        print(f"[INFO] Starting RSF | {FEATURE_LABELS[feature]} ({len(columns)} predictors)")

        for outer_fold in outer_folds:
            test = model_data["outer_fold"].astype(int).eq(outer_fold).to_numpy()
            train = ~test
            pipeline = Pipeline([
                ("preprocessor", make_preprocessor(columns, base)),
                ("model", RandomSurvivalForest(
                    n_estimators=FIXED_PARAMETERS["n_estimators"],
                    max_depth=FIXED_PARAMETERS["max_depth"],
                    max_features=FIXED_PARAMETERS["max_features"],
                    min_samples_leaf=FIXED_PARAMETERS["min_samples_leaf"],
                    min_samples_split=FIXED_PARAMETERS["min_samples_split"],
                    n_jobs=1,
                    random_state=SEED + int(outer_fold),
                )),
            ])
            pipeline.fit(X.loc[train], y[train])
            prediction = np.asarray(pipeline.predict(X.loc[test]), dtype=float).reshape(-1)
            oof_risk[test] = prediction
            fold_cindex = cindex(y[test], prediction)
            fold_rows.append({
                "endpoint": "hard_mace",
                "model": "rsf",
                "feature_set": feature,
                "outer_fold": int(outer_fold),
                "train_n": int(train.sum()),
                "train_events": int(y[train]["event"].sum()),
                "test_n": int(test.sum()),
                "test_events": int(y[test]["event"].sum()),
                "c_index": fold_cindex,
            })
            parameter_rows.append({
                "endpoint": "hard_mace",
                "model": "rsf",
                "feature_set": feature,
                "outer_fold": int(outer_fold),
                **FIXED_PARAMETERS,
                "random_state": SEED + int(outer_fold),
                "selection_source": "Carried forward from primary-endpoint nested CV",
            })
            print(f"[INFO] {feature} | fold={outer_fold} | C-index={fold_cindex:.4f}")

        if np.isnan(oof_risk).any():
            raise RuntimeError(f"Missing OOF risk values for {feature}")
        risks[feature] = oof_risk

    performance_rows = []
    for index, feature in enumerate(FEATURES):
        risk = risks[feature]
        lower, upper = bootstrap_ci(
            y, risk, args.bootstrap_repeats, BOOTSTRAP_SEED + index
        )
        performance_rows.append({
            "endpoint": "hard_mace",
            "endpoint_label": "Hard MACE",
            "n": EXPECTED_N,
            "events": EXPECTED_EVENTS,
            "model": "rsf",
            "model_label": "Random Survival Forest",
            "feature_set": feature,
            "feature_set_label": FEATURE_LABELS[feature],
            "n_original_predictors": len(feature_sets[feature].columns),
            "c_index": cindex(y, risk),
            "c_index_ci_lower": lower,
            "c_index_ci_upper": upper,
        })
    performance = pd.DataFrame(performance_rows)

    oof = pd.DataFrame({
        "analysis_id": model_data["analysis_id"].astype(int),
        "outer_fold": model_data["outer_fold"].astype(int),
        "risk__hard_mace__rsf__demo_cmr": risks["demo_cmr"],
        "risk__hard_mace__rsf__full": risks["full"],
    })

    write_csv(performance, output_dir / "model_performance.csv")
    write_csv(pd.DataFrame(fold_rows), output_dir / "fold_performance.csv")
    write_csv(pd.DataFrame(parameter_rows), output_dir / "selected_hyperparameters.csv")
    write_csv(oof, output_dir / "oof_predictions.csv", separator=";")

    checks = pd.DataFrame([
        {"check": "Analysis dataset hash", "status": "PASS" if hash_valid["dataset"] else "REVIEW"},
        {"check": "Vocabulary hash", "status": "PASS" if hash_valid["vocabulary"] else "REVIEW"},
        {"check": "Fold assignment hash", "status": "PASS" if hash_valid["folds"] else "REVIEW"},
        {"check": "Cohort N", "status": "PASS" if len(model_data) == EXPECTED_N else "REVIEW"},
        {"check": "Hard-MACE events", "status": "PASS" if int(event.sum()) == EXPECTED_EVENTS else "REVIEW"},
        {"check": "Registry features", "status": "PASS" if registry.shape[1] == EXPECTED_REGISTRY_FEATURES else "REVIEW"},
        {"check": "Two performance rows", "status": "PASS" if len(performance) == 2 else "REVIEW"},
        {"check": "OOF completeness", "status": "PASS" if not oof.isna().any().any() else "REVIEW"},
    ])
    write_csv(checks, output_dir / "run_checks.csv")
    review_count = int(checks["status"].eq("REVIEW").sum())

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": SCRIPT_VERSION,
        "analysis": "fixed-parameter hard-MACE RSF sensitivity",
        "n": EXPECTED_N,
        "events": EXPECTED_EVENTS,
        "feature_sets": FEATURES,
        "fixed_parameters": FIXED_PARAMETERS,
        "parameter_selection": (
            "Specification carried forward from the primary-endpoint nested-CV analysis; "
            "no hard-MACE-specific tuning"
        ),
        "seeds": {"base": SEED, "bootstrap_base": BOOTSTRAP_SEED},
        "hashes": hashes,
        "checks": {
            "pass": int(checks["status"].eq("PASS").sum()),
            "review": review_count,
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    summary = [
        "Fixed-parameter hard-MACE RSF sensitivity",
        "=" * 41,
        f"N={EXPECTED_N}; events={EXPECTED_EVENTS}",
        f"Parameters: {FIXED_PARAMETERS}",
        "",
    ]
    for row in performance.itertuples(index=False):
        summary.append(
            f"{row.feature_set_label}: C-index {row.c_index:.4f} "
            f"(95% CI {row.c_index_ci_lower:.4f} to {row.c_index_ci_upper:.4f})"
        )
    summary.append(
        f"Checks: {int(checks['status'].eq('PASS').sum())} PASS; {review_count} REVIEW"
    )
    (output_dir / "run_summary.txt").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )

    if review_count:
        raise RuntimeError(f"Completed with {review_count} REVIEW checks")
    print(f"[DONE] {output_dir}")
    print(f"[INFO] Checks: {int(checks['status'].eq('PASS').sum())} PASS; 0 REVIEW")


if __name__ == "__main__":
    main()
