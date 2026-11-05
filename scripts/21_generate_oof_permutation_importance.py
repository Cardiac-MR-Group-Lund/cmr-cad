#!/usr/bin/env python3
"""Generate revised Figure 2 from held-out permutation importance.

Only the five final full-model Random Forests are refitted. Hyperparameters are
read from the completed nested-CV analysis, and permutation importance is
calculated on each corresponding held-out outer fold. No hyperparameter search
is repeated.

Default inputs:
- data/processed/cmr_revision_analysis.csv
- data/processed/outer_cv_folds.csv
- config/registry_vocabulary.json
- results/models/binary_rf/selected_hyperparameters.csv
- scripts/07_run_binary_rf_nested_cv.py

Outputs:
- results/models/binary_rf/figure2_permutation_importance.png
- results/models/binary_rf/figure2_permutation_importance.tiff
- results/models/binary_rf/figure2_permutation_importance.pdf
- results/models/binary_rf/figure2_permutation_importance.svg
- results/models/binary_rf/figure2_permutation_importance.csv
- results/models/binary_rf/fold_permutation_importance.csv
- results/models/binary_rf/figure2_permutation_importance_summary.txt
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

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
from sklearn.pipeline import Pipeline

EXPECTED_N = 2159
EXPECTED_EVENTS = 212
EXPECTED_FEATURES = 70
SEED = 21042

CLINICAL_LABELS = {
    "age_years": "Age",
    "age_at_cmr": "Age",
    "sex_male": "Sex",
    "sex_numerical": "Sex",
    "bmi": "BMI",
    "bmi_at_cmr": "BMI",
    "lvef_percent": "LVEF",
    "lvef": "LVEF",
    "lvedvi_ml_m2": "LVEDV",
    "lvesvi_ml_m2": "LVESV",
    "ischemia_binary": "Stress-induced\nischemia",
    "mr_ischemia": "Stress-induced\nischemia",
    "infarct_binary": "Infarction on LGE",
    "mr_infarct": "Infarction on LGE",
}

# Common ICD descriptions can be shown immediately. Unmapped ICD and KVA codes
# receive a clean code-system label and can later be replaced from an external
# description dictionary without rerunning model fitting.
COMMON_CODE_LABELS = {
    "pre_icd__I10": "Hypertension",
    "pre_icd__E10": "Type 1 diabetes mellitus",
    "pre_icd__E11": "Type 2 diabetes mellitus",
    "pre_icd__I20": "Angina pectoris",
    "pre_icd__I21": "Acute myocardial infarction",
    "pre_icd__I25": "Chronic ischemic heart disease",
    "pre_icd__I48": "Atrial fibrillation and flutter",
    "pre_icd__I50": "Heart failure",
    "pre_icd__R07": "Chest pain",
    "pre_kva__FNG05": "PCI with stent",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-script", type=Path, default=Path("scripts/07_run_binary_rf_nested_cv.py"))
    p.add_argument("--data", type=Path, default=Path("data/processed/cmr_revision_analysis.csv"))
    p.add_argument("--folds", type=Path, default=Path("data/processed/outer_cv_folds.csv"))
    p.add_argument("--vocabulary", type=Path, default=Path("config/registry_vocabulary.json"))
    p.add_argument("--hyperparameters", type=Path, default=Path("results/models/binary_rf/selected_hyperparameters.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("results/models/binary_rf"))
    p.add_argument("--repeats", type=int, default=40)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument(
        "--label-map",
        type=Path,
        default=None,
        help="Optional semicolon/comma CSV with columns feature_name and display_name.",
    )
    return p.parse_args()


def detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        sample = handle.read(131072)
    try:
        return csv.Sniffer().sniff(sample, delimiters=";,\t|").delimiter
    except csv.Error:
        first = sample.splitlines()[0] if sample else ""
        return max([";", ",", "\t", "|"], key=first.count)


def read_csv_auto(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=detect_delimiter(path), encoding="utf-8-sig", low_memory=False)


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("binary_rf_shared", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def make_preprocessor(columns: list[str], base) -> ColumnTransformer:
    continuous = [c for c in columns if c in base.CONTINUOUS_CLINICAL]
    binary = [c for c in columns if c in base.BINARY_CLINICAL]
    registry = [c for c in columns if c.startswith("pre_icd__") or c.startswith("pre_kva__")]
    transformers = []
    if continuous:
        transformers.append(("continuous", SimpleImputer(strategy="median"), continuous))
    if binary:
        transformers.append(("binary", SimpleImputer(strategy="most_frequent"), binary))
    if registry:
        transformers.append(("registry", "passthrough", registry))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)


def load_optional_labels(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    frame = read_csv_auto(path)
    required = {"feature_name", "display_name"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Label map must contain {sorted(required)}")
    return dict(zip(frame["feature_name"].astype(str), frame["display_name"].astype(str)))


def pretty_name(feature: str, optional_labels: dict[str, str]) -> str:
    if feature in optional_labels:
        return optional_labels[feature]
    if feature in CLINICAL_LABELS:
        return CLINICAL_LABELS[feature]
    if feature in COMMON_CODE_LABELS:
        return COMMON_CODE_LABELS[feature]
    if feature.startswith("pre_icd__"):
        return f"ICD-10 {feature.split('__', 1)[1]}"
    if feature.startswith("pre_kva__"):
        return f"KVÅ {feature.split('__', 1)[1]}"
    return feature.replace("_", " ")


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in [args.base_script, args.data, args.folds, args.vocabulary, args.hyperparameters]:
        if not path.exists():
            raise FileNotFoundError(path)

    base = load_module(args.base_script)
    vocabulary = load_json(args.vocabulary)
    data = read_csv_auto(args.data)
    folds = read_csv_auto(args.folds)
    hyper = read_csv_auto(args.hyperparameters)

    if "outer_fold" in data.columns:
        data = data.drop(columns=["outer_fold"])
    data = data.merge(folds[["analysis_id", "outer_fold"]], on="analysis_id", validate="one_to_one")
    data = data.sort_values("analysis_id").reset_index(drop=True)
    if len(data) != EXPECTED_N:
        raise ValueError(f"Expected N={EXPECTED_N}; observed {len(data)}")

    registry = base.build_registry_matrix(data, vocabulary)
    model_data = pd.concat([data, registry], axis=1)
    feature_sets = base.make_feature_sets(registry.columns.tolist())
    full_columns = list(feature_sets["full"].columns)
    if len(full_columns) != EXPECTED_FEATURES:
        raise ValueError(f"Expected {EXPECTED_FEATURES} full-model predictors; observed {len(full_columns)}")

    outcome_column = "mace_0d_event"
    if outcome_column not in model_data.columns:
        raise ValueError(f"Missing outcome: {outcome_column}")
    y = pd.to_numeric(model_data[outcome_column], errors="raise").astype(int)
    if int(y.sum()) != EXPECTED_EVENTS:
        raise ValueError(f"Expected {EXPECTED_EVENTS} events; observed {int(y.sum())}")

    selected = hyper[hyper["feature_set"].eq("full")].copy()
    if set(selected["outer_fold"].astype(int)) != {1, 2, 3, 4, 5}:
        raise ValueError("Expected one full-model hyperparameter row for each outer fold")

    X = model_data[full_columns].copy()
    fold_rows = []
    print(f"[INFO] Refitting five full RF models; {args.repeats} permutations per feature per fold")

    for fold in range(1, 6):
        test = model_data["outer_fold"].astype(int).eq(fold).to_numpy()
        train = ~test
        params = selected[selected["outer_fold"].astype(int).eq(fold)].iloc[0]

        max_depth = int(params["max_depth"]) if pd.notna(params["max_depth"]) else None
        max_features_raw = str(params["max_features"])
        max_features = max_features_raw if max_features_raw in {"sqrt", "log2"} else float(max_features_raw)

        pipeline = Pipeline([
            ("preprocessor", make_preprocessor(full_columns, base)),
            ("model", RandomForestClassifier(
                n_estimators=int(params["n_estimators"]),
                max_depth=max_depth,
                max_features=max_features,
                min_samples_leaf=int(params["min_samples_leaf"]),
                class_weight="balanced",
                n_jobs=1,
                random_state=int(params["rf_random_state"]),
            )),
        ])
        pipeline.fit(X.loc[train], y.loc[train])

        result = permutation_importance(
            pipeline,
            X.loc[test],
            y.loc[test],
            scoring="roc_auc",
            n_repeats=args.repeats,
            random_state=SEED + fold,
            n_jobs=args.jobs,
        )
        for feature, mean_value, sd_value in zip(
            full_columns, result.importances_mean, result.importances_std
        ):
            fold_rows.append({
                "outer_fold": fold,
                "feature_name": feature,
                "importance_mean": float(mean_value),
                "importance_sd_within_fold": float(sd_value),
                "test_n": int(test.sum()),
                "test_events": int(y.loc[test].sum()),
            })
        print(f"[INFO] fold={fold} complete; test N={int(test.sum())}, events={int(y.loc[test].sum())}")

    fold_importance = pd.DataFrame(fold_rows)
    aggregate = (
        fold_importance.groupby("feature_name", as_index=False)
        .agg(
            mean_decrease_auroc=("importance_mean", "mean"),
            sd_between_folds=("importance_mean", "std"),
            min_fold_importance=("importance_mean", "min"),
            max_fold_importance=("importance_mean", "max"),
        )
        .sort_values("mean_decrease_auroc", ascending=False)
        .reset_index(drop=True)
    )
    aggregate.insert(0, "rank", np.arange(1, len(aggregate) + 1))

    optional_labels = load_optional_labels(args.label_map)
    aggregate["display_name"] = aggregate["feature_name"].map(
        lambda value: pretty_name(value, optional_labels)
    )

    fold_path = output_dir / "fold_permutation_importance.csv"
    aggregate_path = output_dir / "figure2_permutation_importance.csv"
    fold_importance.to_csv(fold_path, index=False, sep=";", encoding="utf-8-sig", float_format="%.9f")
    aggregate.to_csv(aggregate_path, index=False, sep=";", encoding="utf-8-sig", float_format="%.9f")

    top = aggregate.head(10).sort_values("mean_decrease_auroc", ascending=True)
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    positions = np.arange(len(top))
    ax.barh(positions, top["mean_decrease_auroc"], color="#1f77b4", alpha=0.95)
    ax.set_yticks(positions)
    ax.set_yticklabels(top["display_name"])
    ax.set_xlabel("Decrease in AUROC (permutation importance)")
    # ax.set_title("Top 10 Features by Permutation Importance")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(False)
    fig.tight_layout()

    output_files = {
        "png": output_dir / "figure2_permutation_importance.png",
        "tiff": output_dir / "figure2_permutation_importance.tiff",
        "pdf": output_dir / "figure2_permutation_importance.pdf",
        "svg": output_dir / "figure2_permutation_importance.svg",
    }
    fig.savefig(output_files["png"], dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(
        output_files["tiff"], dpi=600, format="tiff", facecolor="white",
        bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(output_files["pdf"], facecolor="white", bbox_inches="tight")
    fig.savefig(output_files["svg"], facecolor="white", bbox_inches="tight")
    plt.close(fig)

    lines = [
        "Held-out permutation importance for the full binary RF model",
        "=" * 59,
        f"Outer folds: 5; permutation repeats per fold: {args.repeats}",
        "Importance was calculated on each held-out outer fold and averaged across folds.",
        "",
        "Top 10 features:",
    ]
    for row in aggregate.head(10).itertuples(index=False):
        lines.append(
            f"{row.rank}. {row.display_name} [{row.feature_name}]: "
            f"mean decrease in AUROC={row.mean_decrease_auroc:.4f}"
        )
    (output_dir / "figure2_permutation_importance_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print(f"[DONE] {aggregate_path}")
    print(f"[DONE] {output_files['tiff']}")
    print("[INFO] Top feature: " + aggregate.iloc[0]["display_name"])


if __name__ == "__main__":
    main()
