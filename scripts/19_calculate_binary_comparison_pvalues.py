#!/usr/bin/env python3
"""Calculate final P values for frozen binary Random Forest comparisons.

No model is fitted. The script reads the final out-of-fold probabilities from
``results/models/binary_rf/oof_predictions.csv`` and calculates:

* Correlated DeLong P values for differences in AUROC.
* Two-sided null-centered paired-bootstrap P values for differences in AUROC,
  average precision, and Brier score.
* Paired-bootstrap 95% confidence intervals for all metric differences.

The manuscript's principal comparison is:
    Registry + demographics + CMR minus demographics + CMR

DeLong is retained for AUROC because the manuscript already specifies DeLong's
test for correlated ROC curves. Bootstrap inference is additionally retained as
a consistency check and is used for average precision and Brier score.

Outputs are semicolon-delimited for Swedish Excel.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

SEED = 19042
EXPECTED_N = 2159
EXPECTED_EVENTS = 212

FEATURE_ORDER = [
    "demographics",
    "registry",
    "registry_demo",
    "cmr",
    "demo_cmr",
    "full",
]

FEATURE_LABELS = {
    "demographics": "Demographics",
    "registry": "Registry",
    "registry_demo": "Registry + demographics",
    "cmr": "CMR",
    "demo_cmr": "Demographics + CMR",
    "full": "Registry + demographics + CMR",
}

# Comparisons needed for the clinical narrative and figure/table annotations.
COMPARISONS = [
    ("registry_demo", "demographics", "Registry + demographics minus demographics"),
    ("demo_cmr", "demographics", "Demographics + CMR minus demographics"),
    ("full", "registry_demo", "Full model minus registry + demographics"),
    ("full", "demo_cmr", "Full model minus demographics + CMR"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--oof-predictions",
        type=Path,
        default=Path("results/models/binary_rf/oof_predictions.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/models/binary_rf"),
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument(
        "--outcome-column",
        type=str,
        default=None,
        help="Optional explicit binary outcome column.",
    )
    return parser.parse_args()


def detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        sample = handle.read(131072)
    try:
        return csv.Sniffer().sniff(sample, delimiters=";,\t|").delimiter
    except csv.Error:
        first = sample.splitlines()[0] if sample else ""
        return max([";", ",", "\t", "|"], key=first.count)


def read_csv_auto(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=detect_delimiter(path),
        encoding="utf-8-sig",
        low_memory=False,
    )


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def resolve_outcome_column(frame: pd.DataFrame, explicit: str | None) -> str:
    if explicit is not None:
        if explicit not in frame.columns:
            raise ValueError(f"Requested outcome column not found: {explicit}")
        return explicit

    candidates = [
        "mace_0d_event",
        "event",
        "y",
        "outcome",
        "mace_event",
        "y_mace",
    ]
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    lookup = {normalize(column): column for column in frame.columns}
    for candidate in candidates:
        if normalize(candidate) in lookup:
            return lookup[normalize(candidate)]
    raise ValueError(
        "Could not identify the binary outcome column. "
        f"Available columns: {list(frame.columns)}"
    )


def resolve_prediction_column(frame: pd.DataFrame, key: str) -> str:
    exact_candidates = {
        "demographics": [
            "probability__demographics", "oof_probability__demographics",
            "oof_probability_demographics", "oof_rf_demographics",
            "oof_RF_demographics_only", "oof_probs_RF_demographics_only",
        ],
        "registry": [
            "probability__registry", "oof_probability__registry",
            "oof_probability_registry", "oof_rf_registry",
            "oof_RF_codes_only", "oof_probs_RF_codes_only",
        ],
        "registry_demo": [
            "probability__registry_demo", "oof_probability__registry_demo",
            "oof_probability_registry_demo", "oof_rf_registry_demo",
            "oof_RF_codes_plus_demo", "oof_probs_RF_codes_plus_demo",
        ],
        "cmr": [
            "probability__cmr", "oof_probability__cmr",
            "oof_probability_cmr", "oof_rf_cmr",
            "oof_RF_cmr_only", "oof_probs_RF_cmr_only",
        ],
        "demo_cmr": [
            "probability__demo_cmr", "oof_probability__demo_cmr",
            "oof_probability_demo_cmr", "oof_rf_demo_cmr",
            "oof_RF_demo_plus_cmr", "oof_probs_RF_demo_plus_cmr",
        ],
        "full": [
            "probability__full", "oof_probability__full",
            "oof_probability_full", "oof_rf_full",
            "oof_RF_codes_plus_demo_cmr", "oof_probs_RF_codes_plus_demo_cmr",
        ],
    }
    for candidate in exact_candidates[key]:
        if candidate in frame.columns:
            return candidate

    aliases = {
        "demographics": ["demographics", "demographicsonly", "demoonly"],
        "registry": ["registry", "codesonly"],
        "registry_demo": ["registrydemo", "codesplusdemo", "codesdemo"],
        "cmr": ["cmronly"],
        "demo_cmr": ["democmr", "demopluscmr"],
        "full": ["full", "codesplusdemocmr", "codesdemocmr", "registrydemographicscmr"],
    }
    matches = []
    for column in frame.columns:
        low = normalize(column)
        prediction_like = any(token in low for token in ("prob", "pred", "oof"))
        excluded = any(token in low for token in ("fold", "event", "outcome", "label"))
        alias_match = any(alias in low for alias in aliases[key])
        if prediction_like and alias_match and not excluded:
            matches.append(column)

    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(
            f"Could not identify the prediction column for {key}. "
            f"Available columns: {list(frame.columns)}"
        )
    raise ValueError(
        f"Multiple possible columns for {key}: {matches}. "
        "Rename the desired column to probability__" + key + "."
    )


# ---------- Correct fast DeLong implementation for correlated ROC curves ----------
def compute_midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    n = len(values)
    ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    result = np.empty(n, dtype=float)
    result[order] = ranks
    return result


def fast_delong(predictions_sorted: np.ndarray, positive_count: int):
    classifier_count, total_count = predictions_sorted.shape
    m = positive_count
    n = total_count - m
    positive = predictions_sorted[:, :m]
    negative = predictions_sorted[:, m:]

    tx = np.empty((classifier_count, m), dtype=float)
    ty = np.empty((classifier_count, n), dtype=float)
    tz = np.empty((classifier_count, total_count), dtype=float)
    for classifier in range(classifier_count):
        tx[classifier] = compute_midrank(positive[classifier])
        ty[classifier] = compute_midrank(negative[classifier])
        tz[classifier] = compute_midrank(predictions_sorted[classifier])

    aucs = (tz[:, :m].sum(axis=1) - m * (m + 1) / 2.0) / (m * n)
    v10 = (tz[:, :m] - tx) / n
    v01 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v10, bias=True))
    sy = np.atleast_2d(np.cov(v01, bias=True))
    covariance = sx / m + sy / n
    return aucs, covariance


def delong_test(y: np.ndarray, reference: np.ndarray, comparison: np.ndarray):
    positive_indices = np.flatnonzero(y == 1)
    negative_indices = np.flatnonzero(y == 0)
    order = np.concatenate([positive_indices, negative_indices])
    predictions = np.vstack([reference[order], comparison[order]])
    aucs, covariance = fast_delong(predictions, len(positive_indices))

    delta = float(aucs[1] - aucs[0])
    variance = float(covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1])
    variance = max(variance, 1e-15)
    standard_error = float(np.sqrt(variance))
    z_value = delta / standard_error
    p_value = float(2.0 * norm.sf(abs(z_value)))
    return {
        "reference_auroc": float(aucs[0]),
        "comparison_auroc": float(aucs[1]),
        "delta": delta,
        "standard_error": standard_error,
        "z_value": z_value,
        "p_value": p_value,
    }


def metric_values(y: np.ndarray, probabilities: dict[str, np.ndarray]):
    return {
        key: {
            "auroc": float(roc_auc_score(y, values)),
            "average_precision": float(average_precision_score(y, values)),
            "brier_score": float(brier_score_loss(y, values)),
        }
        for key, values in probabilities.items()
    }


def stratified_bootstrap_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    indices = np.concatenate([
        rng.choice(positive, size=len(positive), replace=True),
        rng.choice(negative, size=len(negative), replace=True),
    ])
    rng.shuffle(indices)
    return indices


def bootstrap_comparisons(
    y: np.ndarray,
    probabilities: dict[str, np.ndarray],
    repeats: int,
):
    rng = np.random.default_rng(SEED)
    observed_metrics = metric_values(y, probabilities)
    metrics = ["auroc", "average_precision", "brier_score"]
    draws = {
        (comparison_name, metric): np.empty(repeats)
        for _, _, comparison_name in COMPARISONS
        for metric in metrics
    }

    for repeat in range(repeats):
        indices = stratified_bootstrap_indices(y, rng)
        y_boot = y[indices]
        boot_values = metric_values(
            y_boot,
            {key: values[indices] for key, values in probabilities.items()},
        )
        for comparison_key, reference_key, comparison_name in COMPARISONS:
            for metric in metrics:
                draws[(comparison_name, metric)][repeat] = (
                    boot_values[comparison_key][metric]
                    - boot_values[reference_key][metric]
                )

    rows = []
    for comparison_key, reference_key, comparison_name in COMPARISONS:
        delong = delong_test(y, probabilities[reference_key], probabilities[comparison_key])
        for metric in metrics:
            observed_delta = (
                observed_metrics[comparison_key][metric]
                - observed_metrics[reference_key][metric]
            )
            values = draws[(comparison_name, metric)]
            centered = values - observed_delta
            count = int(np.count_nonzero(np.abs(centered) >= abs(observed_delta)))
            bootstrap_p = float((count + 1) / (repeats + 1))
            lower, upper = np.quantile(values, [0.025, 0.975])
            rows.append({
                "comparison": comparison_name,
                "comparison_feature_set": comparison_key,
                "reference_feature_set": reference_key,
                "metric": metric,
                "comparison_value": observed_metrics[comparison_key][metric],
                "reference_value": observed_metrics[reference_key][metric],
                "estimate": observed_delta,
                "ci_lower": float(lower),
                "ci_upper": float(upper),
                "bootstrap_p_value_two_sided": bootstrap_p,
                "delong_p_value_two_sided": delong["p_value"] if metric == "auroc" else np.nan,
                "delong_z": delong["z_value"] if metric == "auroc" else np.nan,
                "bootstrap_repeats": repeats,
                "bootstrap_seed": SEED,
            })
    return pd.DataFrame(rows)


def format_p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.oof_predictions.exists():
        raise FileNotFoundError(args.oof_predictions)
    frame = read_csv_auto(args.oof_predictions)

    outcome_column = resolve_outcome_column(frame, args.outcome_column)
    prediction_columns = {
        key: resolve_prediction_column(frame, key)
        for key in FEATURE_ORDER
    }

    outcome = pd.to_numeric(frame[outcome_column], errors="coerce")
    if outcome.isna().any() or not outcome.isin([0, 1]).all():
        raise ValueError(f"Outcome column {outcome_column} is not binary")
    y = outcome.astype(int).to_numpy()
    if len(y) != EXPECTED_N or int(y.sum()) != EXPECTED_EVENTS:
        raise ValueError(
            f"Expected N={EXPECTED_N}, events={EXPECTED_EVENTS}; "
            f"observed N={len(y)}, events={int(y.sum())}"
        )

    probabilities = {}
    for key, column in prediction_columns.items():
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"Invalid prediction probabilities in {column}")
        probabilities[key] = values.to_numpy(float)

    results = bootstrap_comparisons(y, probabilities, args.bootstrap_repeats)
    results["preferred_p_value"] = np.where(
        results["metric"].eq("auroc"),
        results["delong_p_value_two_sided"],
        results["bootstrap_p_value_two_sided"],
    )
    results["preferred_p_value_formatted"] = results["preferred_p_value"].map(format_p)

    output_path = output_dir / "binary_paired_comparisons_with_pvalues.csv"
    results.to_csv(
        output_path,
        index=False,
        sep=";",
        encoding="utf-8-sig",
        float_format="%.12g",
    )

    principal = results[
        results["comparison"].eq("Full model minus demographics + CMR")
    ].copy()
    principal.to_csv(
        output_dir / "binary_principal_comparison_with_pvalues.csv",
        index=False,
        sep=";",
        encoding="utf-8-sig",
        float_format="%.12g",
    )

    lines = [
        "Final binary paired-comparison P values",
        "=" * 39,
        f"N={len(y)}; events={int(y.sum())}",
        f"Bootstrap repetitions={args.bootstrap_repeats}; seed={SEED}",
        "AUROC P values: correlated DeLong test",
        "Average precision and Brier P values: two-sided null-centered paired bootstrap",
        "",
        "Resolved columns:",
        f"  outcome: {outcome_column}",
    ]
    for key in FEATURE_ORDER:
        lines.append(f"  {key}: {prediction_columns[key]}")
    lines.append("")

    for comparison_name in [item[2] for item in COMPARISONS]:
        lines.append(comparison_name)
        subset = results[results["comparison"].eq(comparison_name)]
        for row in subset.itertuples(index=False):
            p_value = row.preferred_p_value
            lines.append(
                f"  {row.metric}: {row.reference_value:.4f} to {row.comparison_value:.4f}; "
                f"delta {row.estimate:.4f} "
                f"(95% CI {row.ci_lower:.4f} to {row.ci_upper:.4f}); "
                f"P {format_p(p_value)}"
            )
        lines.append("")

    summary_path = output_dir / "binary_paired_comparisons_with_pvalues_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(args.oof_predictions),
        "n": len(y),
        "events": int(y.sum()),
        "outcome_column": outcome_column,
        "prediction_columns": prediction_columns,
        "bootstrap_repeats": args.bootstrap_repeats,
        "bootstrap_seed": SEED,
        "auroc_p_value_method": "correlated DeLong test",
        "other_p_value_method": "two-sided null-centered paired stratified bootstrap",
    }
    (output_dir / "binary_pvalue_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"[DONE] {output_path}")
    print(f"[DONE] {summary_path}")
    print("[INFO] Principal full-versus-demographics+CMR comparison:")
    for row in principal.itertuples(index=False):
        print(
            f"[INFO]   {row.metric}: delta={row.estimate:.4f}; "
            f"95% CI {row.ci_lower:.4f} to {row.ci_upper:.4f}; "
            f"P={format_p(row.preferred_p_value)}"
        )


if __name__ == "__main__":
    main()
