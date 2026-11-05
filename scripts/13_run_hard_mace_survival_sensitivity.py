#!/usr/bin/env python3
"""Hard-MACE survival sensitivity analysis.

Runs only:
- Elastic-net Cox: demographics + CMR; full registry model
- Random Survival Forest: demographics + CMR; full registry model

The script reuses scripts/09_run_survival_model_comparison.py so that the hard-
MACE analysis uses the same frozen data, folds, preprocessing, tuning framework,
and model definitions as the primary survival analysis.

Completed sub-runs are cached and skipped on rerun. Thus, the expensive RSF run
will not be repeated if its output directory already contains a valid completed
run with 0 REVIEW checks.

After fitting, the script merges the saved OOF risk scores and calculates:
- C-index with 95% paired-bootstrap CI for each model/configuration
- registry increment within elastic-net Cox and RSF
- RSF versus elastic-net Cox for each predictor configuration
- two-sided null-centered paired-bootstrap P values

No hard-MACE results are placed in the main manuscript; outputs are intended for
Supplemental Table Y and the response letter.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

EXPECTED_N = 2159
EXPECTED_EVENTS = 83
SEED = 13042
MODELS = ["elastic_net", "rsf"]
FEATURES = ["demo_cmr", "full"]
MODEL_LABELS = {
    "elastic_net": "Elastic-net Cox regression",
    "rsf": "Random Survival Forest",
}
FEATURE_LABELS = {
    "demo_cmr": "Demographics + CMR",
    "full": "Registry + demographics + CMR",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-script",
        type=Path,
        default=Path("scripts/09_run_survival_model_comparison.py"),
    )
    parser.add_argument(
        "--analysis-data",
        type=Path,
        default=Path("data/processed/cmr_revision_analysis.csv"),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/models"),
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--rsf-trees", type=int, default=800)
    parser.add_argument(
        "--skip-fitting",
        action="store_true",
        help="Only merge/evaluate existing completed sub-runs.",
    )
    return parser.parse_args()


def read_csv_auto(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8-sig") as handle:
        first = handle.readline()
    separator = ";" if first.count(";") > first.count(",") else ","
    return pd.read_csv(path, sep=separator, encoding="utf-8-sig", low_memory=False)


def run_is_complete(directory: Path, model: str) -> bool:
    checks_path = directory / "run_checks.csv"
    oof_path = directory / "oof_predictions.csv"
    performance_path = directory / "model_performance.csv"
    if not (checks_path.exists() and oof_path.exists() and performance_path.exists()):
        return False
    checks = read_csv_auto(checks_path)
    if checks["status"].eq("REVIEW").any():
        return False
    performance = read_csv_auto(performance_path)
    expected = {(model, feature) for feature in FEATURES}
    observed = set(zip(performance["model"], performance["feature_set"]))
    return observed == expected


def run_model(base_script: Path, directory: Path, model: str, rsf_trees: int):
    if run_is_complete(directory, model):
        print(f"[SKIP] Using completed frozen run: {directory}")
        return

    directory.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(base_script),
        "--endpoints", "hard_mace",
        "--models", model,
        "--feature-sets", "demo_cmr", "full",
        "--grid-jobs", "1",
        "--output-dir", str(directory),
    ]
    if model == "rsf":
        command.extend(["--rsf-trees", str(rsf_trees)])

    print("[RUN] " + " ".join(command))
    subprocess.run(command, check=True)
    if not run_is_complete(directory, model):
        raise RuntimeError(f"Run did not produce a complete valid result: {directory}")


def cindex(y, risk) -> float:
    return float(concordance_index_censored(y["event"], y["time"], risk)[0])


def load_risks(directory: Path, model: str) -> pd.DataFrame:
    frame = read_csv_auto(directory / "oof_predictions.csv")
    columns = ["analysis_id", "outer_fold"]
    rename = {}
    for feature in FEATURES:
        source = f"risk__hard_mace__{model}__{feature}"
        if source not in frame.columns:
            raise ValueError(f"Missing {source} in {directory / 'oof_predictions.csv'}")
        columns.append(source)
        rename[source] = f"risk__{model}__{feature}"
    return frame[columns].rename(columns=rename)


def p_value_null_centered(draws: np.ndarray, observed: float) -> float:
    centered = draws - observed
    count = int(np.count_nonzero(np.abs(centered) >= abs(observed)))
    return float((count + 1) / (len(draws) + 1))


def format_p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def bootstrap(y, risks, repeats: int):
    rng = np.random.default_rng(SEED)
    event_indices = np.flatnonzero(y["event"])
    censor_indices = np.flatnonzero(~y["event"])
    metric_draws = {key: np.empty(repeats) for key in risks}

    comparisons = [
        {
            "comparison_type": "registry_increment",
            "model": "elastic_net",
            "comparison": "Elastic-net Cox: full minus demographics + CMR",
            "left": ("elastic_net", "full"),
            "right": ("elastic_net", "demo_cmr"),
        },
        {
            "comparison_type": "registry_increment",
            "model": "rsf",
            "comparison": "RSF: full minus demographics + CMR",
            "left": ("rsf", "full"),
            "right": ("rsf", "demo_cmr"),
        },
        {
            "comparison_type": "rsf_vs_elastic_net",
            "model": "rsf_minus_elastic_net",
            "comparison": "RSF minus elastic-net Cox (Demographics + CMR)",
            "left": ("rsf", "demo_cmr"),
            "right": ("elastic_net", "demo_cmr"),
        },
        {
            "comparison_type": "rsf_vs_elastic_net",
            "model": "rsf_minus_elastic_net",
            "comparison": "RSF minus elastic-net Cox (Full predictors)",
            "left": ("rsf", "full"),
            "right": ("elastic_net", "full"),
        },
    ]
    delta_draws = {item["comparison"]: np.empty(repeats) for item in comparisons}

    for repeat in range(repeats):
        indices = np.concatenate([
            rng.choice(event_indices, len(event_indices), replace=True),
            rng.choice(censor_indices, len(censor_indices), replace=True),
        ])
        rng.shuffle(indices)
        y_boot = y[indices]
        values = {}
        for key, risk in risks.items():
            value = cindex(y_boot, risk[indices])
            values[key] = value
            metric_draws[key][repeat] = value
        for item in comparisons:
            delta_draws[item["comparison"]][repeat] = (
                values[item["left"]] - values[item["right"]]
            )

    intervals = {
        key: tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))
        for key, draws in metric_draws.items()
    }
    comparison_rows = []
    for item in comparisons:
        observed = cindex(y, risks[item["left"]]) - cindex(y, risks[item["right"]])
        draws = delta_draws[item["comparison"]]
        lower, upper = np.quantile(draws, [0.025, 0.975])
        p_value = p_value_null_centered(draws, observed)
        comparison_rows.append({
            "comparison_type": item["comparison_type"],
            "model": item["model"],
            "comparison": item["comparison"],
            "estimate": observed,
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "p_value_two_sided": p_value,
            "p_value_formatted": format_p(p_value),
            "bootstrap_repeats": repeats,
            "bootstrap_seed": SEED,
        })
    return intervals, pd.DataFrame(comparison_rows)


def main():
    args = parse_args()
    root = args.results_root.resolve()
    elastic_dir = root / "hard_mace_survival_elastic_net"
    rsf_dir = root / "hard_mace_survival_rsf"
    output_dir = root / "hard_mace_survival_sensitivity"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_fitting:
        # Fast model first. If RSF is interrupted later, elastic-net remains saved.
        run_model(args.base_script, elastic_dir, "elastic_net", args.rsf_trees)
        run_model(args.base_script, rsf_dir, "rsf", args.rsf_trees)

    if not run_is_complete(elastic_dir, "elastic_net"):
        raise RuntimeError(f"Missing completed elastic-net run: {elastic_dir}")
    if not run_is_complete(rsf_dir, "rsf"):
        raise RuntimeError(f"Missing completed RSF run: {rsf_dir}")

    analysis = read_csv_auto(args.analysis_data).sort_values("analysis_id").reset_index(drop=True)
    if len(analysis) != EXPECTED_N:
        raise ValueError(f"Expected N={EXPECTED_N}; observed {len(analysis)}")
    event = pd.to_numeric(analysis["hard_mace_event"], errors="raise").astype(bool).to_numpy()
    time = pd.to_numeric(analysis["hard_mace_tte_years"], errors="raise").astype(float).to_numpy()
    if int(event.sum()) != EXPECTED_EVENTS:
        raise ValueError(f"Expected {EXPECTED_EVENTS} hard-MACE events; observed {int(event.sum())}")
    y = Surv.from_arrays(event=event, time=time)

    merged = load_risks(elastic_dir, "elastic_net").merge(
        load_risks(rsf_dir, "rsf"),
        on=["analysis_id", "outer_fold"],
        how="inner",
        validate="one_to_one",
    ).sort_values("analysis_id").reset_index(drop=True)

    if len(merged) != EXPECTED_N or not np.array_equal(
        merged["analysis_id"].to_numpy(), analysis["analysis_id"].to_numpy()
    ):
        raise ValueError("Hard-MACE OOF rows do not align with the frozen analysis dataset")

    risks = {}
    for model in MODELS:
        for feature in FEATURES:
            column = f"risk__{model}__{feature}"
            risk = pd.to_numeric(merged[column], errors="coerce").to_numpy(float)
            if np.isnan(risk).any():
                raise ValueError(f"Missing OOF risks in {column}")
            risks[(model, feature)] = risk

    intervals, comparisons = bootstrap(y, risks, args.bootstrap_repeats)
    performance_rows = []
    for model in MODELS:
        for feature in FEATURES:
            risk = risks[(model, feature)]
            lower, upper = intervals[(model, feature)]
            performance_rows.append({
                "endpoint": "hard_mace",
                "n": EXPECTED_N,
                "events": EXPECTED_EVENTS,
                "model": model,
                "model_label": MODEL_LABELS[model],
                "feature_set": feature,
                "feature_set_label": FEATURE_LABELS[feature],
                "c_index": cindex(y, risk),
                "c_index_ci_lower": lower,
                "c_index_ci_upper": upper,
            })
    performance = pd.DataFrame(performance_rows)

    increment = comparisons[
        comparisons["comparison_type"].eq("registry_increment")
    ].set_index("model")
    formatted_rows = []
    for model in MODELS:
        for feature in FEATURES:
            row = performance[
                performance["model"].eq(model)
                & performance["feature_set"].eq(feature)
            ].iloc[0]
            if feature == "full":
                delta = increment.loc[model]
                delta_text = (
                    f"{delta['estimate']:.3f} "
                    f"({delta['ci_lower']:.3f} to {delta['ci_upper']:.3f})"
                )
                p_text = str(delta["p_value_formatted"])
            else:
                delta_text = "Reference"
                p_text = ""
            formatted_rows.append({
                "Model": MODEL_LABELS[model],
                "Predictors": FEATURE_LABELS[feature],
                "C-index (95% CI)": (
                    f"{row['c_index']:.2f} "
                    f"({row['c_index_ci_lower']:.2f} to {row['c_index_ci_upper']:.2f})"
                ),
                "Delta C-index (95% CI)": delta_text,
                "P value": p_text,
            })

    performance.to_csv(output_dir / "hard_mace_model_performance.csv", index=False, sep=";", encoding="utf-8-sig")
    comparisons.to_csv(output_dir / "hard_mace_paired_comparisons.csv", index=False, sep=";", encoding="utf-8-sig")
    pd.DataFrame(formatted_rows).to_csv(
        output_dir / "supplemental_table_hard_mace_survival.csv",
        index=False,
        sep=";",
        encoding="utf-8-sig",
    )
    merged.to_csv(
        output_dir / "hard_mace_oof_predictions_merged.csv",
        index=False,
        sep=";",
        encoding="utf-8-sig",
    )

    checks = pd.DataFrame([
        {"check": "Cohort N", "status": "PASS" if len(merged) == EXPECTED_N else "REVIEW"},
        {"check": "Hard-MACE events", "status": "PASS" if int(event.sum()) == EXPECTED_EVENTS else "REVIEW"},
        {"check": "Four performance rows", "status": "PASS" if len(performance) == 4 else "REVIEW"},
        {"check": "OOF completeness", "status": "PASS" if not merged.isna().any().any() else "REVIEW"},
    ])
    checks.to_csv(output_dir / "run_checks.csv", index=False)

    summary = [
        "Hard-MACE survival sensitivity analysis",
        "=" * 39,
        f"N={EXPECTED_N}; events={EXPECTED_EVENTS}",
        "",
    ]
    for row in performance.itertuples(index=False):
        summary.append(
            f"{row.model_label} | {row.feature_set_label}: C-index {row.c_index:.4f} "
            f"(95% CI {row.c_index_ci_lower:.4f} to {row.c_index_ci_upper:.4f})"
        )
    summary.append("")
    for row in comparisons.itertuples(index=False):
        summary.append(
            f"{row.comparison}: delta {row.estimate:.4f} "
            f"(95% CI {row.ci_lower:.4f} to {row.ci_upper:.4f}); "
            f"P {row.p_value_formatted}"
        )
    review_count = int(checks["status"].eq("REVIEW").sum())
    summary.append(f"Checks: {int(checks['status'].eq('PASS').sum())} PASS; {review_count} REVIEW")
    (output_dir / "run_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": "hard MACE: cardiovascular death or acute myocardial infarction",
        "n": EXPECTED_N,
        "events": EXPECTED_EVENTS,
        "models": MODELS,
        "feature_sets": FEATURES,
        "bootstrap_repeats": args.bootstrap_repeats,
        "bootstrap_seed": SEED,
        "rsf_trees": args.rsf_trees,
        "cached_subruns": {
            "elastic_net": str(elastic_dir),
            "rsf": str(rsf_dir),
        },
        "checks": {
            "pass": int(checks["status"].eq("PASS").sum()),
            "review": review_count,
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    if review_count:
        raise RuntimeError(f"Hard-MACE sensitivity completed with {review_count} REVIEW checks")
    print(f"[DONE] {output_dir}")
    print(f"[INFO] Checks: {int(checks['status'].eq('PASS').sum())} PASS; 0 REVIEW")


if __name__ == "__main__":
    main()
