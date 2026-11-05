#!/usr/bin/env python3
"""Calculate two-sided paired-bootstrap P values for frozen survival-model comparisons.

No model is refitted. The script reads the merged out-of-fold risk scores created by
10_merge_survival_model_results.py and computes:

1. Registry increment within each model:
   full model minus demographics + CMR.
2. RSF versus each penalized Cox model for each predictor configuration.

The P value uses a null-centered paired bootstrap:
    P = proportion(|delta_bootstrap - delta_observed| >= |delta_observed|)
with the finite-sample correction (count + 1) / (B + 1).

Default: 10,000 stratified bootstrap repetitions, fixed seed 11042.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

SEED = 11042
EXPECTED_N = 2159
EXPECTED_EVENTS = 196
MODEL_ORDER = ["ridge", "lasso", "elastic_net", "rsf"]
FEATURE_ORDER = ["demo_cmr", "full"]
MODEL_LABELS = {
    "ridge": "Ridge Cox",
    "lasso": "LASSO Cox",
    "elastic_net": "Elastic-net Cox",
    "rsf": "Random Survival Forest",
}
FEATURE_LABELS = {
    "demo_cmr": "Demographics + CMR",
    "full": "Registry + demographics + CMR",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--analysis-data",
        type=Path,
        default=Path("data/processed/cmr_revision_analysis.csv"),
    )
    p.add_argument(
        "--merged-oof",
        type=Path,
        default=Path("results/models/survival_comparison_merged/oof_predictions_merged.csv"),
    )
    p.add_argument(
        "--comparisons",
        type=Path,
        default=Path("results/models/survival_comparison_merged/paired_comparisons_merged.csv"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/models/survival_comparison_merged"),
    )
    p.add_argument("--bootstrap-repeats", type=int, default=10000)
    return p.parse_args()


def read_csv_auto(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8-sig") as handle:
        first = handle.readline()
    sep = ";" if first.count(";") > first.count(",") else ","
    return pd.read_csv(path, sep=sep, encoding="utf-8-sig", low_memory=False)


def cindex(y, risk) -> float:
    return float(concordance_index_censored(y["event"], y["time"], risk)[0])


def two_sided_null_centered_p(delta_boot: np.ndarray, delta_observed: float) -> float:
    centered = delta_boot - delta_observed
    count = int(np.count_nonzero(np.abs(centered) >= abs(delta_observed)))
    return float((count + 1) / (len(delta_boot) + 1))


def format_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def main():
    args = parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    analysis = read_csv_auto(args.analysis_data).sort_values("analysis_id").reset_index(drop=True)
    oof = read_csv_auto(args.merged_oof).sort_values("analysis_id").reset_index(drop=True)
    prior = read_csv_auto(args.comparisons)

    if len(analysis) != EXPECTED_N or len(oof) != EXPECTED_N:
        raise ValueError(f"Expected N={EXPECTED_N} in analysis and OOF files")
    if not np.array_equal(analysis["analysis_id"].to_numpy(), oof["analysis_id"].to_numpy()):
        raise ValueError("Analysis and OOF rows do not align by analysis_id")

    event = pd.to_numeric(analysis["mace_30d_event"], errors="raise").astype(bool).to_numpy()
    time = pd.to_numeric(analysis["mace_30d_tte_years"], errors="raise").astype(float).to_numpy()
    if int(event.sum()) != EXPECTED_EVENTS:
        raise ValueError(f"Expected {EXPECTED_EVENTS} events; observed {int(event.sum())}")
    y = Surv.from_arrays(event=event, time=time)

    risks = {}
    for model in MODEL_ORDER:
        for feature in FEATURE_ORDER:
            col = f"risk__{model}__{feature}"
            if col not in oof.columns:
                raise ValueError(f"Missing OOF column: {col}")
            risk = pd.to_numeric(oof[col], errors="coerce").to_numpy(float)
            if np.isnan(risk).any():
                raise ValueError(f"Missing values in {col}")
            risks[(model, feature)] = risk

    comparisons = []
    for model in MODEL_ORDER:
        comparisons.append({
            "comparison_type": "registry_increment",
            "model": model,
            "comparison": f"{MODEL_LABELS[model]}: full minus demographics + CMR",
            "left": (model, "full"),
            "right": (model, "demo_cmr"),
        })
    for feature in FEATURE_ORDER:
        for cox in ["ridge", "lasso", "elastic_net"]:
            comparisons.append({
                "comparison_type": "rsf_vs_cox",
                "model": f"rsf_minus_{cox}",
                "comparison": f"RSF minus {MODEL_LABELS[cox]} ({FEATURE_LABELS[feature]})",
                "left": ("rsf", feature),
                "right": (cox, feature),
            })

    observed = {}
    for item in comparisons:
        observed[item["comparison"]] = (
            cindex(y, risks[item["left"]]) - cindex(y, risks[item["right"]])
        )

    rng = np.random.default_rng(SEED)
    event_idx = np.flatnonzero(event)
    censor_idx = np.flatnonzero(~event)
    draws = {item["comparison"]: np.empty(args.bootstrap_repeats) for item in comparisons}

    for repeat in range(args.bootstrap_repeats):
        idx = np.concatenate([
            rng.choice(event_idx, size=len(event_idx), replace=True),
            rng.choice(censor_idx, size=len(censor_idx), replace=True),
        ])
        rng.shuffle(idx)
        y_b = y[idx]
        c_values = {key: cindex(y_b, risk[idx]) for key, risk in risks.items()}
        for item in comparisons:
            draws[item["comparison"]][repeat] = c_values[item["left"]] - c_values[item["right"]]

    rows = []
    for item in comparisons:
        name = item["comparison"]
        delta = observed[name]
        values = draws[name]
        p_value = two_sided_null_centered_p(values, delta)
        rows.append({
            "comparison_type": item["comparison_type"],
            "model": item["model"],
            "comparison": name,
            "estimate": delta,
            "ci_lower": float(np.quantile(values, 0.025)),
            "ci_upper": float(np.quantile(values, 0.975)),
            "p_value_two_sided": p_value,
            "p_value_formatted": format_p(p_value),
            "bootstrap_method": "null-centered paired stratified bootstrap",
            "bootstrap_repeats": args.bootstrap_repeats,
            "bootstrap_seed": SEED,
        })
    results = pd.DataFrame(rows)

    # Check that estimates agree with the previously merged comparison file.
    check = results.merge(
        prior[["comparison", "estimate"]].rename(columns={"estimate": "prior_estimate"}),
        on="comparison",
        how="left",
        validate="one_to_one",
    )
    check["absolute_difference"] = (check["estimate"] - check["prior_estimate"]).abs()
    if check["prior_estimate"].isna().any() or (check["absolute_difference"] > 1e-10).any():
        raise ValueError("Comparison estimates do not match the frozen merged analysis")

    results.to_csv(out / "paired_comparisons_with_pvalues.csv", index=False, float_format="%.12g")
    check[["comparison", "estimate", "prior_estimate", "absolute_difference"]].to_csv(
        out / "pvalue_consistency_checks.csv", index=False, float_format="%.12g"
    )

    summary_lines = [
        "Two-sided paired-bootstrap P values",
        "=" * 37,
        f"Bootstrap repetitions: {args.bootstrap_repeats:,}",
        f"Seed: {SEED}",
        "",
    ]
    for row in results.itertuples(index=False):
        summary_lines.append(
            f"{row.comparison}: delta {row.estimate:.4f} "
            f"(95% CI {row.ci_lower:.4f} to {row.ci_upper:.4f}); "
            f"P {row.p_value_formatted}"
        )
    (out / "paired_comparisons_with_pvalues_summary.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "null-centered paired stratified bootstrap",
        "bootstrap_repeats": args.bootstrap_repeats,
        "bootstrap_seed": SEED,
        "n": EXPECTED_N,
        "events": EXPECTED_EVENTS,
        "comparison_count": len(results),
        "consistency_check_max_absolute_difference": float(check["absolute_difference"].max()),
    }
    (out / "pvalue_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"[DONE] {out / 'paired_comparisons_with_pvalues.csv'}")
    print(f"[INFO] Comparisons: {len(results)}; consistency checks PASS")


if __name__ == "__main__":
    main()
