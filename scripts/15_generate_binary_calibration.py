#!/usr/bin/env python3
"""Generate primary binary-model calibration figure and Supplemental Table 5.

Uses frozen out-of-fold predictions from the primary nested-CV RF analysis.
No model is fitted or recalibrated.

Expected input directory:
    results/models/binary_rf/

Outputs:
    calibration_primary_models.png
    calibration_primary_models.tiff
    calibration_primary_models.pdf
    calibration_primary_models.svg
    supplemental_table_5_calibration.csv
    calibration_points.csv
    calibration_run_summary.txt
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

MODEL_KEYS = ["demo_cmr", "full"]
MODEL_LABELS = {
    "demo_cmr": "Demographics + CMR",
    "full": "Registry + demographics + CMR",
}
COLORS = {
    "demo_cmr": "#1f77b4",
    "full": "crimson",
}
MARKERS = {
    "demo_cmr": "o",
    "full": "s",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/models/binary_rf"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/models/binary_rf"),
    )
    parser.add_argument("--bins", type=int, default=10)
    return parser.parse_args()


def read_csv_auto(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        first = handle.readline()
    separator = ";" if first.count(";") > first.count(",") else ","
    return pd.read_csv(path, sep=separator, encoding="utf-8-sig", low_memory=False)


def find_column(columns, exact_candidates, contains_candidates=()):
    for candidate in exact_candidates:
        if candidate in columns:
            return candidate
    lowered = {column.lower(): column for column in columns}
    for candidate in exact_candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    for column in columns:
        low = column.lower()
        if all(token.lower() in low for token in contains_candidates):
            return column
    return None


def calibration_intercept_slope(y: np.ndarray, probabilities: np.ndarray):
    """Estimate weak calibration from logit predictions.

    Slope: logistic regression of outcome on prediction logit.
    Intercept: logistic regression offset-style estimate with slope fixed at 1.
    """
    eps = 1e-6
    p = np.clip(probabilities, eps, 1 - eps)
    logit = np.log(p / (1 - p))

    slope_model = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=10000,
    )
    slope_model.fit(logit.reshape(-1, 1), y)
    slope = float(slope_model.coef_[0, 0])

    # Solve sum(y - expit(intercept + logit(p))) = 0 by bisection.
    low, high = -20.0, 20.0
    for _ in range(200):
        mid = (low + high) / 2
        fitted = 1.0 / (1.0 + np.exp(-(mid + logit)))
        score = float(np.sum(y - fitted))
        if score > 0:
            low = mid
        else:
            high = mid
    intercept = float((low + high) / 2)
    return intercept, slope


def prediction_column(frame: pd.DataFrame, key: str):
    candidates = {
        "demo_cmr": [
            "oof_rf_demo_cmr",
            "oof_demo_cmr",
            "pred_demo_cmr",
            "prob_demo_cmr",
            "oof_probability_demo_cmr",
        ],
        "full": [
            "oof_rf_full",
            "oof_full",
            "pred_full",
            "prob_full",
            "oof_probability_full",
            "oof_rf_codes_plus_demo_cmr",
        ],
    }
    column = find_column(frame.columns, candidates[key])
    if column is not None:
        return column

    # Robust fallback: choose one prediction-like column containing the key.
    tokens = ["oof", key]
    matches = [
        column for column in frame.columns
        if all(token in column.lower() for token in tokens)
        and "event" not in column.lower()
        and "fold" not in column.lower()
    ]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(
        f"Could not uniquely identify the {key} prediction column. "
        f"Available columns: {list(frame.columns)}"
    )


def main():
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    oof_path = input_dir / "oof_predictions.csv"
    if not oof_path.exists():
        raise FileNotFoundError(oof_path)
    frame = read_csv_auto(oof_path)

    target_column = find_column(
        frame.columns,
        ["event", "y", "outcome", "mace_event", "y_mace", "mace_0d_event"],
    )
    if target_column is None:
        raise ValueError(
            "Could not identify the binary outcome column in oof_predictions.csv. "
            f"Available columns: {list(frame.columns)}"
        )

    y = pd.to_numeric(frame[target_column], errors="coerce")
    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError(f"Outcome column {target_column} is missing or nonbinary")
    y_array = y.astype(int).to_numpy()

    prediction_columns = {
        key: prediction_column(frame, key)
        for key in MODEL_KEYS
    }

    metric_rows = []
    point_rows = []
    curves = {}
    for key in MODEL_KEYS:
        column = prediction_columns[key]
        probabilities = pd.to_numeric(frame[column], errors="coerce")
        if probabilities.isna().any():
            raise ValueError(f"Missing predictions in {column}")
        p = probabilities.to_numpy(float)
        if np.any((p < 0) | (p > 1)):
            raise ValueError(f"Predictions outside [0,1] in {column}")

        intercept, slope = calibration_intercept_slope(y_array, p)
        brier = float(brier_score_loss(y_array, p))
        observed, predicted = calibration_curve(
            y_array,
            p,
            n_bins=args.bins,
            strategy="quantile",
        )
        curves[key] = (predicted, observed)

        metric_rows.append({
            "Model": MODEL_LABELS[key],
            "Brier score": brier,
            "Calibration intercept": intercept,
            "Calibration slope": slope,
        })
        for bin_number, (predicted_value, observed_value) in enumerate(
            zip(predicted, observed), start=1
        ):
            point_rows.append({
                "model": key,
                "model_label": MODEL_LABELS[key],
                "bin": bin_number,
                "mean_predicted_probability": float(predicted_value),
                "observed_event_proportion": float(observed_value),
            })

    metrics = pd.DataFrame(metric_rows)
    points = pd.DataFrame(point_rows)

    # Excel-compatible semicolon-delimited files.
    metrics.to_csv(
        output_dir / "supplemental_table_5_calibration.csv",
        index=False,
        sep=";",
        encoding="utf-8-sig",
        float_format="%.3f",
    )
    points.to_csv(
        output_dir / "calibration_points.csv",
        index=False,
        sep=";",
        encoding="utf-8-sig",
        float_format="%.6f",
    )

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1.2,
            label="Ideal calibration")
    for key in MODEL_KEYS:
        predicted, observed = curves[key]
        metric = metrics[metrics["Model"].eq(MODEL_LABELS[key])].iloc[0]
        label = (
            f"{MODEL_LABELS[key]} "
            f"(slope {metric['Calibration slope']:.2f}, "
            f"Brier {metric['Brier score']:.3f})"
        )
        ax.plot(
            predicted,
            observed,
            color=COLORS[key],
            marker=MARKERS[key],
            linewidth=2,
            markersize=5,
            label=label,
        )

    # Focus the axes on the observed prediction range while retaining the origin.
    max_value = max(
        0.20,
        max(float(points["mean_predicted_probability"].max()),
            float(points["observed_event_proportion"].max())) + 0.05,
    )
    max_value = min(1.0, max_value)
    ax.set_xlim(0, max_value)
    ax.set_ylim(0, max_value)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed event proportion")
    ax.set_title("Calibration of Binary Random Forest Models")
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    ax.grid(False)
    fig.tight_layout()

    outputs = {
        "png": output_dir / "calibration_primary_models.png",
        "tiff": output_dir / "calibration_primary_models.tiff",
        "pdf": output_dir / "calibration_primary_models.pdf",
        "svg": output_dir / "calibration_primary_models.svg",
    }
    fig.savefig(outputs["png"], dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(
        outputs["tiff"],
        dpi=600,
        format="tiff",
        facecolor="white",
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(outputs["pdf"], facecolor="white", bbox_inches="tight")
    fig.savefig(outputs["svg"], facecolor="white", bbox_inches="tight")
    plt.close(fig)

    lines = [
        "Primary binary-model calibration",
        "=" * 32,
        f"Input: {oof_path}",
        f"Outcome column: {target_column}",
        f"Prediction columns: {prediction_columns}",
        f"N={len(y_array)}; events={int(y_array.sum())}",
        f"Quantile bins: {args.bins}",
        "",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(
            f"{row[0]}: Brier {row[1]:.4f}; "
            f"calibration intercept {row[2]:.4f}; slope {row[3]:.4f}"
        )
    (output_dir / "calibration_run_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print(f"[DONE] Calibration outputs written to {output_dir}")
    print(f"[INFO] Outcome: {target_column}; predictions: {prediction_columns}")


if __name__ == "__main__":
    main()
