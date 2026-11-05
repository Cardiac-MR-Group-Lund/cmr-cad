#!/usr/bin/env python3
"""Plot the final binary RF ROC curves without refitting any models.

The script reads the frozen pooled out-of-fold predictions produced by the
completed nested cross-validation analysis. It plots only the four models that
follow the clinical workflow:

1. Demographics
2. Demographics + Registry
3. Demographics + CMR
4. Demographics + CMR + Registry

Registry-only and CMR-only models remain in the analysis outputs but are not
shown in this figure. The diagonal reference line is retained without a legend
entry. No models are fitted and no performance estimates are changed.

Default input:
    results/models/binary_rf/oof_predictions.csv

Default outputs:
    results/models/binary_rf/roc_binary_rf_clinical_workflow.png
    results/models/binary_rf/roc_binary_rf_clinical_workflow.tiff
    results/models/binary_rf/roc_binary_rf_clinical_workflow.pdf
    results/models/binary_rf/roc_binary_rf_clinical_workflow.svg
    results/models/binary_rf/roc_binary_rf_clinical_workflow_values.csv
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

EXPECTED_N = 2159
EXPECTED_EVENTS = 212

# Edit labels, colors, line styles, or order here when iterating on the figure.
ROC_MODELS = [
    {
        "key": "demographics",
        "label": "Demographics",
        "ci": "0.62–0.69",
        "color": "#E3E700",
        "linestyle": "-",
        "linewidth": 2.2,
    },
    {
        "key": "registry",
        "label": "Registry",
        "ci": "0.63–0.71",
        "color": "#585858",
        "linestyle": "--",
        "linewidth": 1.8,
    },
    {
        "key": "registry_demo",
        "label": "Demographics + Registry",
        "ci": "0.68–0.75",
        "color": "#009E73",
        "linestyle": "-",
        "linewidth": 2.4,
    },
    {
        "key": "demo_cmr",
        "label": "Demographics + CMR",
        "ci": "0.78–0.84",
        "color": "#E60000",
        "linestyle": "-",
        "linewidth": 2.4,
    },
    {
        "key": "full",
        "label": "Demographics + CMR + Registry",
        "ci": "0.79–0.85",
        "color": "#0072B2",
        "linestyle": "-",
        "linewidth": 2.6,
    },
]

PREDICTION_ALIASES = {
    "demographics": [
        "oof_rf_demographics",
        "probability__demographics",
        "oof_probability__demographics",
        "oof_probability_demographics",
    ],
    "registry": [
    "oof_rf_registry",
    "probability__registry",
    "oof_probability__registry",
    "oof_probability_registry",
    ],
    "registry_demo": [
        "oof_rf_registry_demo",
        "probability__registry_demo",
        "oof_probability__registry_demo",
        "oof_probability_registry_demo",
    ],
    "demo_cmr": [
        "oof_rf_demo_cmr",
        "probability__demo_cmr",
        "oof_probability__demo_cmr",
        "oof_probability_demo_cmr",
    ],
    "full": [
        "oof_rf_full",
        "probability__full",
        "oof_probability__full",
        "oof_probability_full",
    ],
}

OUTCOME_ALIASES = [
    "mace_0d_event",
    "event",
    "outcome",
    "y",
    "mace_event",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot clinical-workflow ROC curves from frozen OOF predictions."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/models/binary_rf/oof_predictions.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/models/binary_rf"),
    )
    parser.add_argument(
        "--basename",
        default="roc_binary_rf_clinical_workflow",
        help="Output filename without extension.",
    )
    parser.add_argument(
        "--legend-location",
        default="lower right",
        choices=["lower right", "lower left", "upper right", "upper left"],
    )
    parser.add_argument(
        "--legend-fontsize",
        type=float,
        default=8.8,
    )
    parser.add_argument(
        "--figure-size",
        nargs=2,
        type=float,
        metavar=("WIDTH", "HEIGHT"),
        default=(7.2, 6.4),
    )
    parser.add_argument(
        "--show-title",
        action="store_true",
        help="Add an internal plot title. Default is no title for journal submission.",
    )
    parser.add_argument(
        "--overwrite-original-name",
        action="store_true",
        help="Also write roc_binary_rf.* after the new files have been validated.",
    )
    return parser.parse_args()


def detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        sample = handle.read(131072)
    try:
        return csv.Sniffer().sniff(sample, delimiters=";,\t|").delimiter
    except csv.Error:
        first_line = sample.splitlines()[0] if sample else ""
        return max([";", ",", "\t", "|"], key=first_line.count)


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def resolve_column(frame: pd.DataFrame, aliases: list[str], label: str) -> str:
    for alias in aliases:
        if alias in frame.columns:
            return alias
    normalized = {normalize(column): column for column in frame.columns}
    for alias in aliases:
        if normalize(alias) in normalized:
            return normalized[normalize(alias)]
    raise ValueError(
        f"Could not identify {label}. Tried {aliases}. "
        f"Available columns: {list(frame.columns)}"
    )


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(
        args.input,
        sep=detect_delimiter(args.input),
        encoding="utf-8-sig",
        low_memory=False,
    )

    outcome_column = resolve_column(frame, OUTCOME_ALIASES, "binary outcome")
    prediction_columns = {
        model["key"]: resolve_column(
            frame,
            PREDICTION_ALIASES[model["key"]],
            f"prediction column for {model['label']}",
        )
        for model in ROC_MODELS
    }

    outcome = pd.to_numeric(frame[outcome_column], errors="coerce")
    if outcome.isna().any() or not outcome.isin([0, 1]).all():
        raise ValueError(f"Outcome column {outcome_column} is not binary")
    y = outcome.astype(int)

    if len(y) != EXPECTED_N or int(y.sum()) != EXPECTED_EVENTS:
        raise ValueError(
            f"Expected N={EXPECTED_N} and events={EXPECTED_EVENTS}; "
            f"observed N={len(y)} and events={int(y.sum())}"
        )

    curves = []
    metric_rows = []
    for model in ROC_MODELS:
        column = prediction_columns[model["key"]]
        probability = pd.to_numeric(frame[column], errors="coerce")
        if probability.isna().any() or ((probability < 0) | (probability > 1)).any():
            raise ValueError(f"Invalid probabilities in {column}")

        fpr, tpr, thresholds = roc_curve(y, probability)
        auc_value = float(roc_auc_score(y, probability))
        curves.append((model, fpr, tpr, thresholds, auc_value))
        metric_rows.append(
            {
                "model_key": model["key"],
                "model_label": model["label"],
                "prediction_column": column,
                "n": len(y),
                "events": int(y.sum()),
                "auroc": auc_value,
                "color": model["color"],
                "linestyle": model["linestyle"],
            }
        )

    # Sanity check against the frozen final analysis values.
    expected_aurocs = {
        "demographics": 0.656199426307,
        "registry": 0.663604384103,
        "registry_demo": 0.715387000804,
        "demo_cmr": 0.807863573374,
        "full": 0.821796474499,
    }
    for row in metric_rows:
        difference = abs(row["auroc"] - expected_aurocs[row["model_key"]])
        if difference > 1e-10:
            raise ValueError(
                f"AUROC mismatch for {row['model_key']}: observed {row['auroc']:.12f}, "
                f"expected {expected_aurocs[row['model_key']]:.12f}"
            )

    fig, ax = plt.subplots(figsize=tuple(args.figure_size))

    # Reference line without a legend entry.
    ax.plot(
        [0, 1],
        [0, 1],
        color="#A6A6A6",
        linewidth=1.1,
        linestyle=(0, (4, 4)),
        zorder=1,
    )

    for model, fpr, tpr, _thresholds, auc_value in curves:
        ci_text = f" (95% CI, {model['ci']})" if "ci" in model else ""
        ax.plot(
            fpr,
            tpr,
            color=model["color"],
            linestyle=model["linestyle"],
            linewidth=model["linewidth"],
            label=f"{model['label']}: AUROC {auc_value:.2f}{ci_text}",
            zorder=2,
        )

    ax.set_xlabel("1-Specificity", fontsize=11)
    ax.set_ylabel("Sensitivity", fontsize=11)
    if args.show_title:
        ax.set_title("Binary Prediction of MACE", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(
        loc=args.legend_location,
        frameon=False,
        fontsize=args.legend_fontsize,
        handlelength=2.5,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    paths = {
        "png": output_dir / f"{args.basename}.png",
        "tiff": output_dir / f"{args.basename}.tiff",
        "pdf": output_dir / f"{args.basename}.pdf",
        "svg": output_dir / f"{args.basename}.svg",
    }
    fig.savefig(paths["png"], dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(
        paths["tiff"],
        dpi=600,
        format="tiff",
        facecolor="white",
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(paths["pdf"], facecolor="white", bbox_inches="tight")
    fig.savefig(paths["svg"], facecolor="white", bbox_inches="tight")
    plt.close(fig)

    values_path = output_dir / f"{args.basename}_values.csv"
    pd.DataFrame(metric_rows).to_csv(
        values_path,
        index=False,
        sep=";",
        encoding="utf-8-sig",
        float_format="%.12g",
    )

    # Validate that all outputs were created and are non-empty.
    for path in [*paths.values(), values_path]:
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Output validation failed: {path}")

    if args.overwrite_original_name:
        import shutil

        for extension, source_path in paths.items():
            target_path = output_dir / f"roc_binary_rf.{extension}"
            shutil.copy2(source_path, target_path)
            if not target_path.exists() or target_path.stat().st_size == 0:
                raise RuntimeError(f"Original-name copy failed: {target_path}")
            print(f"[DONE] {target_path}")

    print(f"[DONE] {paths['png']}")
    print(f"[DONE] {paths['tiff']}")
    print(f"[DONE] {paths['pdf']}")
    print(f"[DONE] {paths['svg']}")
    print(f"[DONE] {values_path}")
    print("[INFO] No models were fitted; frozen pooled OOF predictions were used.")
    for row in metric_rows:
        print(f"[INFO] {row['model_label']}: AUROC={row['auroc']:.4f}")


if __name__ == "__main__":
    main()
