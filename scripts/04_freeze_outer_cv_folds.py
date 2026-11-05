#!/usr/bin/env python3
"""
04_freeze_outer_cv_folds.py

Freeze one shared 5-fold outer cross-validation partition for all directly
comparable revised CMR-MACE models.

Input:
    data/processed/cmr_revision_analysis.csv

Private patient-level output:
    data/processed/outer_cv_folds.csv

Aggregate/configuration outputs:
    config/cv_plan.json
    results/audit/cv_folds/outer_fold_summary.csv
    results/audit/cv_folds/outer_fold_checks.csv
    results/audit/cv_folds/outer_fold_manifest.json
    results/audit/cv_folds/outer_fold_summary.txt

Design:
- 5 outer folds, shuffled, random seed 42.
- Stratification uses the principal survival endpoint, mace_30d_event.
- The same outer fold assignment is intended for all feature configurations and
  model classes, enabling paired out-of-fold comparisons.
- The script verifies event balance for mace_0d, mace_30d, mace_60d,
  mace_90d, mace_late_revasc, and hard_mace.
- Inner cross-validation is not stored here. Each model-training script should
  construct its inner folds using training data only and the settings recorded
  in config/cv_plan.json.
- No predictors, outcomes, dates, or identifiers other than analysis_id and
  outer_fold are written to the private fold file.

Run from repository root:
    python scripts/04_freeze_outer_cv_folds.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


SCRIPT_VERSION = "1.0.0"
DEFAULT_INPUT = Path("data/processed/cmr_revision_analysis.csv")
DEFAULT_FOLD_OUTPUT = Path("data/processed/outer_cv_folds.csv")
DEFAULT_CONFIG_OUTPUT = Path("config/cv_plan.json")
DEFAULT_AUDIT_DIR = Path("results/audit/cv_folds")

EXPECTED_INPUT_SHA256 = "e94b8aebe4c4939b0b3def943879539f4f242ba7ef39fd7ac929407578f7db9c"
EXPECTED_VOCABULARY_SHA256 = "1250d92ca3f5d5da2049f3965837a7cfa0b70daaa9ffae2e09d71f8235eac023"
EXPECTED_N = 2159
OUTER_SPLITS = 5
OUTER_SEED = 42
INNER_SPLITS = 5
INNER_SEED = 2026
STRATIFY_COLUMN = "mace_30d_event"

ENDPOINT_COLUMNS = [
    "mace_0d_event",
    "mace_30d_event",
    "mace_60d_event",
    "mace_90d_event",
    "mace_late_revasc_event",
    "hard_mace_event",
]

EXPECTED_EVENT_COUNTS = {
    "mace_0d_event": 212,
    "mace_30d_event": 196,
    "mace_60d_event": 179,
    "mace_90d_event": 163,
    "mace_late_revasc_event": 150,
    "hard_mace_event": 83,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze shared outer cross-validation folds."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--fold-output", type=Path, default=DEFAULT_FOLD_OUTPUT)
    parser.add_argument("--config-output", type=Path, default=DEFAULT_CONFIG_OUTPUT)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--outer-splits", type=int, default=OUTER_SPLITS)
    parser.add_argument("--outer-seed", type=int, default=OUTER_SEED)
    parser.add_argument("--inner-splits", type=int, default=INNER_SPLITS)
    parser.add_argument("--inner-seed", type=int, default=INNER_SEED)
    parser.add_argument("--expected-sha256", default=EXPECTED_INPUT_SHA256)
    parser.add_argument("--allow-input-hash-mismatch", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_canonical_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        sample = handle.read(131072)
    try:
        return csv.Sniffer().sniff(sample, delimiters=";,\t|").delimiter
    except csv.Error:
        first = sample.splitlines()[0] if sample else ""
        return max([";", ",", "\t", "|"], key=first.count)


def write_csv(df: pd.DataFrame, path: Path, sep: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, sep=sep, encoding="utf-8", float_format="%.12g")


def validate_binary(series: pd.Series, name: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any():
        raise ValueError(f"{name} contains {int(values.isna().sum())} missing values.")
    invalid = ~values.isin([0, 1])
    if invalid.any():
        raise ValueError(f"{name} contains {int(invalid.sum())} nonbinary values.")
    return values.astype("int8")


def main() -> int:
    args = parse_args()
    source = args.input.resolve()
    fold_output = args.fold_output.resolve()
    config_output = args.config_output.resolve()
    audit_dir = args.audit_dir.resolve()

    if not source.exists():
        print(f"[ERROR] Input not found: {source}", file=sys.stderr)
        return 2
    if args.outer_splits < 3:
        print("[ERROR] --outer-splits must be at least 3.", file=sys.stderr)
        return 2
    if args.inner_splits < 3:
        print("[ERROR] --inner-splits must be at least 3.", file=sys.stderr)
        return 2

    source_hash = sha256_file(source)
    hash_match = source_hash.lower() == args.expected_sha256.lower()
    if not hash_match and not args.allow_input_hash_mismatch:
        print("[ERROR] Processed dataset SHA-256 mismatch.", file=sys.stderr)
        print(f"[ERROR] Expected: {args.expected_sha256}", file=sys.stderr)
        print(f"[ERROR] Observed: {source_hash}", file=sys.stderr)
        return 3

    delimiter = detect_delimiter(source)
    required_columns = ["analysis_id", *ENDPOINT_COLUMNS]
    df = pd.read_csv(
        source,
        sep=delimiter,
        usecols=required_columns,
        dtype={"analysis_id": "Int64"},
        encoding="utf-8-sig",
        low_memory=False,
    )

    if len(df) != EXPECTED_N:
        raise ValueError(f"Expected N={EXPECTED_N}; observed N={len(df)}")
    if df["analysis_id"].isna().any() or df["analysis_id"].duplicated().any():
        raise ValueError("analysis_id contains missing or duplicate values.")

    # Stable row order independent of source-file ordering.
    df = df.sort_values("analysis_id", kind="mergesort").reset_index(drop=True)
    endpoints = {col: validate_binary(df[col], col) for col in ENDPOINT_COLUMNS}

    for col, expected in EXPECTED_EVENT_COUNTS.items():
        observed = int(endpoints[col].sum())
        if observed != expected:
            raise ValueError(f"{col}: expected {expected} events; observed {observed}")

    y = endpoints[STRATIFY_COLUMN]
    if int(y.sum()) < args.outer_splits:
        raise ValueError("Too few principal-endpoint events for the requested outer folds.")

    splitter = StratifiedKFold(
        n_splits=args.outer_splits,
        shuffle=True,
        random_state=args.outer_seed,
    )

    # Store folds as 1..K for manuscript/reporting readability.
    fold_assignment = np.zeros(len(df), dtype=np.int8)
    for fold_zero, (_, validation_idx) in enumerate(splitter.split(np.zeros(len(df)), y)):
        fold_assignment[validation_idx] = fold_zero + 1

    if np.any(fold_assignment == 0):
        raise RuntimeError("At least one patient did not receive an outer-fold assignment.")

    fold_df = pd.DataFrame({
        "analysis_id": df["analysis_id"].astype("int64"),
        "outer_fold": fold_assignment,
    })

    # Aggregate fold summary for all important endpoints.
    summary_rows: list[dict[str, Any]] = []
    for fold in range(1, args.outer_splits + 1):
        mask = fold_assignment == fold
        row: dict[str, Any] = {
            "outer_fold": fold,
            "n": int(mask.sum()),
        }
        for col in ENDPOINT_COLUMNS:
            events = int(endpoints[col][mask].sum())
            row[f"{col}_events"] = events
            row[f"{col}_percent"] = round(events / int(mask.sum()) * 100.0, 6)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, observed: Any, expected: Any, detail: str) -> None:
        checks.append({
            "check": name,
            "status": "PASS" if passed else "REVIEW",
            "observed": observed,
            "expected": expected,
            "detail": detail,
        })

    check("Input SHA-256", hash_match, source_hash, args.expected_sha256,
          "Frozen processed-dataset identity.")
    check("Cohort size", len(df) == EXPECTED_N, len(df), EXPECTED_N,
          "Rows included in fold assignment.")
    check("Unique analysis_id", fold_df["analysis_id"].nunique() == len(fold_df),
          int(fold_df["analysis_id"].nunique()), len(fold_df), "One assignment per patient.")
    check("All patients assigned", bool((fold_df["outer_fold"] > 0).all()),
          int((fold_df["outer_fold"] > 0).sum()), len(fold_df), "No unassigned patients.")
    check("Valid fold labels", set(fold_df["outer_fold"]) == set(range(1, args.outer_splits + 1)),
          sorted(fold_df["outer_fold"].unique().tolist()), list(range(1, args.outer_splits + 1)),
          "Expected contiguous fold labels.")

    fold_sizes = summary_df["n"]
    check("Fold-size balance", int(fold_sizes.max() - fold_sizes.min()) <= 1,
          int(fold_sizes.max() - fold_sizes.min()), "<=1", "StratifiedKFold size balance.")

    for col, expected in EXPECTED_EVENT_COUNTS.items():
        event_col = f"{col}_events"
        total = int(summary_df[event_col].sum())
        minimum = int(summary_df[event_col].min())
        maximum = int(summary_df[event_col].max())
        check(f"{col} total preserved", total == expected, total, expected,
              "Events summed across outer folds.")
        check(f"{col} represented in every fold", minimum > 0, minimum, ">0",
              "Each validation fold contains events.")
        # Principal endpoint should be almost exactly balanced; other endpoints
        # are checked descriptively without demanding identical counts.
        if col == STRATIFY_COLUMN:
            check(f"{col} fold-event balance", maximum - minimum <= 1,
                  maximum - minimum, "<=1", "Endpoint used for stratification.")

    checks_df = pd.DataFrame(checks)
    review_count = int(checks_df["status"].eq("REVIEW").sum())

    audit_dir.mkdir(parents=True, exist_ok=True)
    write_csv(checks_df, audit_dir / "outer_fold_checks.csv")
    write_csv(summary_df, audit_dir / "outer_fold_summary.csv")

    if review_count:
        raise RuntimeError(
            f"Fold build stopped: {review_count} validation check(s) require review. "
            f"See {audit_dir / 'outer_fold_checks.csv'}"
        )

    # The fold specification hash is independent of timestamps and file paths.
    canonical_assignment = {
        "outer_splits": args.outer_splits,
        "outer_seed": args.outer_seed,
        "stratify_column": STRATIFY_COLUMN,
        "assignments": [
            {"analysis_id": int(row.analysis_id), "outer_fold": int(row.outer_fold)}
            for row in fold_df.itertuples(index=False)
        ],
    }
    assignment_hash = sha256_canonical_json(canonical_assignment)

    # Private patient-level fold map.
    write_csv(fold_df, fold_output, sep=";")
    fold_file_hash = sha256_file(fold_output)

    cv_plan = {
        "schema_version": "1.0",
        "script_version": SCRIPT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_dataset": {
            "path": str(args.input),
            "sha256": source_hash,
            "cohort_n": len(df),
        },
        "registry_vocabulary": {
            "path": "config/registry_vocabulary.json",
            "vocabulary_specification_sha256": EXPECTED_VOCABULARY_SHA256,
            "total_features": 62,
        },
        "outer_cross_validation": {
            "method": "StratifiedKFold",
            "n_splits": args.outer_splits,
            "shuffle": True,
            "random_state": args.outer_seed,
            "stratify_column": STRATIFY_COLUMN,
            "fold_labels": list(range(1, args.outer_splits + 1)),
            "assignment_path": str(args.fold_output),
            "assignment_file_sha256": fold_file_hash,
            "assignment_specification_sha256": assignment_hash,
            "one_out_of_fold_prediction_per_patient": True,
        },
        "inner_cross_validation": {
            "method": "StratifiedKFold for binary models; model-appropriate event-stratified K-fold for survival models",
            "n_splits": args.inner_splits,
            "shuffle": True,
            "base_random_state": args.inner_seed,
            "construction": (
                "Created separately within each outer-training set. "
                "The inner seed for outer fold k is base_random_state + k."
            ),
            "purpose": "Hyperparameter optimisation only",
        },
        "comparison_rules": {
            "shared_outer_folds": True,
            "paired_out_of_fold_predictions": True,
            "preprocessing_fit_on_outer_training_data_only": True,
            "hyperparameter_tuning_within_outer_training_data_only": True,
            "processed_dataset_not_used_as_final_test_set": True,
            "external_validation": False,
        },
        "endpoint_event_counts": EXPECTED_EVENT_COUNTS,
    }

    config_output.parent.mkdir(parents=True, exist_ok=True)
    with config_output.open("w", encoding="utf-8") as handle:
        json.dump(cv_plan, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    config_file_hash = sha256_file(config_output)

    manifest = {
        "build": {
            "script": Path(__file__).name,
            "script_version": SCRIPT_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
        "input": {
            "path": str(args.input),
            "sha256": source_hash,
            "delimiter": delimiter,
            "rows": len(df),
        },
        "fold_output": {
            "path": str(args.fold_output),
            "sha256": fold_file_hash,
            "assignment_specification_sha256": assignment_hash,
            "rows": len(fold_df),
        },
        "config_output": {
            "path": str(args.config_output),
            "sha256": config_file_hash,
        },
        "settings": {
            "outer_splits": args.outer_splits,
            "outer_seed": args.outer_seed,
            "stratify_column": STRATIFY_COLUMN,
            "inner_splits": args.inner_splits,
            "inner_base_seed": args.inner_seed,
        },
        "checks": {
            "pass": int(checks_df["status"].eq("PASS").sum()),
            "review": review_count,
        },
        "privacy": (
            "outer_cv_folds.csv is a private patient-level linkage file containing only "
            "analysis_id and outer_fold. Aggregate fold summaries contain no row-level data."
        ),
    }
    with (audit_dir / "outer_fold_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    lines = [
        "Frozen outer cross-validation folds",
        "=" * 35,
        f"Input dataset: {args.input}",
        f"Input SHA-256: {source_hash}",
        f"Input hash matched expected: {hash_match}",
        f"Cohort N: {len(df):,}",
        "",
        f"Outer method: StratifiedKFold",
        f"Outer folds: {args.outer_splits}",
        f"Outer seed: {args.outer_seed}",
        f"Stratification endpoint: {STRATIFY_COLUMN}",
        f"Inner folds planned: {args.inner_splits}",
        f"Inner base seed: {args.inner_seed}",
        "",
        f"Assignment specification SHA-256: {assignment_hash}",
        f"Fold file SHA-256: {fold_file_hash}",
        f"CV plan SHA-256: {config_file_hash}",
        "",
        "Per-fold event counts:",
    ]
    for row in summary_df.itertuples(index=False):
        lines.append(
            f"- Fold {row.outer_fold}: N={row.n}; "
            f"MACE 30d={getattr(row, 'mace_30d_event_events')}; "
            f"MACE 0d={getattr(row, 'mace_0d_event_events')}; "
            f"hard MACE={getattr(row, 'hard_mace_event_events')}"
        )
    lines.extend([
        "",
        f"Validation checks: {manifest['checks']['pass']} PASS; 0 REVIEW",
        "The same outer folds must be used for all paired primary model comparisons.",
        "The patient-level fold file must remain private.",
    ])
    (audit_dir / "outer_fold_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print(f"[INFO] Outer folds: {args.outer_splits}; seed={args.outer_seed}")
    print(f"[INFO] Stratified on: {STRATIFY_COLUMN}")
    print(f"[INFO] Assignment SHA-256: {assignment_hash}")
    print(f"[INFO] Wrote private fold map: {fold_output}")
    print(f"[INFO] Wrote CV plan: {config_output}")
    print(f"[INFO] Checks: {manifest['checks']['pass']} PASS; 0 REVIEW")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
