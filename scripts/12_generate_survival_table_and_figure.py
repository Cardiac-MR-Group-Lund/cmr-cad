#!/usr/bin/env python3
"""Generate Supplemental Table X and an annotated Ridge Cox vs RSF Figure 3.

No models are fitted. The plot follows the original grouped-bar design:
- blue Ridge Cox bars and crimson RSF bars
- 95% CI error bars
- within-feature-set Ridge-vs-RSF brackets
- within-model brackets comparing addition of registry data
- significance stars with a compact legend
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
import pandas as pd

MODEL_ORDER = ["ridge", "lasso", "elastic_net", "rsf"]
MODEL_LABELS = {
    "ridge": "Ridge Cox regression",
    "lasso": "LASSO Cox regression",
    "elastic_net": "Elastic-net Cox regression",
    "rsf": "Random Survival Forest",
}
FEATURE_ORDER = ["demo_cmr", "full"]
FEATURE_LABELS = {
    "demo_cmr": "Demographics + CMR",
    "full": "Registry + Demographics + CMR",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/models/survival_comparison_merged"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/models/survival_comparison_merged"),
    )
    return parser.parse_args()


def format_p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def stars(value: float) -> str:
    if value < 0.001:
        return "***"
    if value < 0.01:
        return "**"
    if value < 0.05:
        return "*"
    return "ns"


def add_bracket(ax, x1: float, x2: float, y: float, text: str, pad: float = 0.004):
    ax.plot([x1, x1, x2, x2], [y - pad, y, y, y - pad], lw=1.2, c="black", clip_on=False)
    ax.text((x1 + x2) / 2, y + pad, text, ha="center", va="bottom", fontsize=11)


def main():
    args = parse_args()
    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    performance = pd.read_csv(results_dir / "model_performance_merged.csv")
    comparisons = pd.read_csv(results_dir / "paired_comparisons_with_pvalues.csv")

    expected = {(model, feature) for model in MODEL_ORDER for feature in FEATURE_ORDER}
    observed = set(zip(performance["model"], performance["feature_set"]))
    if observed != expected:
        raise ValueError(f"Unexpected model combinations: {observed}")

    registry_comparisons = comparisons[
        comparisons["comparison_type"].eq("registry_increment")
    ].set_index("model")

    # Supplemental Table X: all four model specifications.
    numeric_rows = []
    formatted_rows = []
    for model in MODEL_ORDER:
        for feature in FEATURE_ORDER:
            row = performance[
                performance["model"].eq(model)
                & performance["feature_set"].eq(feature)
            ].iloc[0]

            if feature == "full":
                delta = registry_comparisons.loc[model]
                delta_value = float(delta["estimate"])
                delta_low = float(delta["ci_lower"])
                delta_high = float(delta["ci_upper"])
                p_value = float(delta["p_value_two_sided"])
                delta_text = f"{delta_value:.3f} ({delta_low:.3f} to {delta_high:.3f})"
                p_text = format_p(p_value)
            else:
                delta_value = delta_low = delta_high = p_value = np.nan
                delta_text = "Reference"
                p_text = ""

            numeric_rows.append({
                "model": model,
                "model_label": MODEL_LABELS[model],
                "predictors": feature,
                "predictor_label": FEATURE_LABELS[feature],
                "c_index": float(row["c_index"]),
                "c_index_ci_lower": float(row["c_index_ci_lower"]),
                "c_index_ci_upper": float(row["c_index_ci_upper"]),
                "delta_c_index": delta_value,
                "delta_ci_lower": delta_low,
                "delta_ci_upper": delta_high,
                "p_value": p_value,
            })
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

    pd.DataFrame(numeric_rows).to_csv(
        output_dir / "supplemental_table_x_survival_models_numeric.csv", index=False, sep=";", encoding="utf-8-sig"
    )
    pd.DataFrame(formatted_rows).to_csv(
        output_dir / "supplemental_table_x_survival_models_formatted.csv", index=False, sep=";", encoding="utf-8-sig"
    )

    # Figure 3: Ridge Cox and RSF in the original annotated grouped-bar style.
    plot_models = ["ridge", "rsf"]
    plot_data = {}
    for model in plot_models:
        values = []
        for feature in FEATURE_ORDER:
            row = performance[
                performance["model"].eq(model)
                & performance["feature_set"].eq(feature)
            ].iloc[0]
            values.append((
                float(row["c_index"]),
                float(row["c_index_ci_lower"]),
                float(row["c_index_ci_upper"]),
            ))
        plot_data[model] = values

    p_between = {}
    for feature_index, feature in enumerate(FEATURE_ORDER):
        comparison_name = (
            f"RSF minus Ridge Cox ({'Demographics + CMR' if feature == 'demo_cmr' else 'Registry + demographics + CMR'})"
        )
        match = comparisons[comparisons["comparison"].eq(comparison_name)]
        if match.empty:
            raise ValueError(f"Missing comparison: {comparison_name}")
        p_between[feature_index] = float(match.iloc[0]["p_value_two_sided"])

    p_within = {
        "ridge": float(registry_comparisons.loc["ridge", "p_value_two_sided"]),
        "rsf": float(registry_comparisons.loc["rsf", "p_value_two_sided"]),
    }

    groups = [FEATURE_LABELS[feature] for feature in FEATURE_ORDER]
    x = np.arange(len(groups))
    width = 0.38

    def yerr(values):
        return np.array([
            [value[0] - value[1] for value in values],
            [value[2] - value[0] for value in values],
        ])

    ridge_values = plot_data["ridge"]
    rsf_values = plot_data["rsf"]
    ridge_points = [value[0] for value in ridge_values]
    rsf_points = [value[0] for value in rsf_values]
    ridge_errors = yerr(ridge_values)
    rsf_errors = yerr(rsf_values)

    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=150)
    ridge_bars = ax.bar(
        x - width / 2,
        ridge_points,
        width=width,
        yerr=ridge_errors,
        capsize=3,
        label="Ridge Cox",
        alpha=0.95,
        color="#1f77b4",
    )
    rsf_bars = ax.bar(
        x + width / 2,
        rsf_points,
        width=width,
        yerr=rsf_errors,
        capsize=3,
        label="RSF",
        alpha=0.95,
        color="crimson",
    )

    # Within-feature-set brackets: Ridge Cox versus RSF.
    within_group_levels = []
    for index in range(len(groups)):
        ridge_top = ridge_bars[index].get_height() + ridge_errors[1, index]
        rsf_top = rsf_bars[index].get_height() + rsf_errors[1, index]
        level = max(ridge_top, rsf_top) + 0.014
        within_group_levels.append(level)
        add_bracket(
            ax,
            x[index] - width / 2,
            x[index] + width / 2,
            level,
            stars(p_between[index]),
        )

    # Within-model brackets: effect of adding registry predictors.
    ridge_top_0 = ridge_bars[0].get_height() + ridge_errors[1, 0]
    ridge_top_1 = ridge_bars[1].get_height() + ridge_errors[1, 1]
    ridge_pair_level = max(ridge_top_0, ridge_top_1, *within_group_levels) + 0.026
    add_bracket(
        ax,
        x[0] - width / 2,
        x[1] - width / 2,
        ridge_pair_level,
        stars(p_within["ridge"]),
    )

    rsf_top_0 = rsf_bars[0].get_height() + rsf_errors[1, 0]
    rsf_top_1 = rsf_bars[1].get_height() + rsf_errors[1, 1]
    rsf_pair_level = max(rsf_top_0, rsf_top_1, ridge_pair_level + 0.022)
    add_bracket(
        ax,
        x[0] + width / 2,
        x[1] + width / 2,
        rsf_pair_level,
        stars(p_within["rsf"]),
    )

    ax.set_ylabel("C-index")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    required_top = max(ridge_pair_level, rsf_pair_level) + 0.05
    ax.set_ylim(0.70, max(0.92, required_top))
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.text(
        0.98,
        0.98,
        "* P<0.05, ** P<0.01, *** P<0.001; ns, not significant",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color="dimgray",
    )
    fig.tight_layout()

    output_paths = {
        "png": output_dir / "figure3_cindex_bars_annotated.png",
        "tiff": output_dir / "figure3_cindex_bars_annotated.tiff",
        "pdf": output_dir / "figure3_cindex_bars_annotated.pdf",
        "svg": output_dir / "figure3_cindex_bars_annotated.svg",
    }
    fig.savefig(output_paths["png"], dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(
        output_paths["tiff"],
        dpi=600,
        format="tiff",
        facecolor="white",
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(output_paths["pdf"], facecolor="white", bbox_inches="tight")
    fig.savefig(output_paths["svg"], facecolor="white", bbox_inches="tight")
    plt.close(fig)

    footnote = (
        "Supplemental Table X. Time-to-event model performance\n\n"
        "Performance was estimated from pooled out-of-fold predictions generated "
        "through nested 5-fold cross-validation. Delta C-index represents the "
        "paired difference after addition of registry predictors to demographics "
        "and CMR. P values were calculated using two-sided paired bootstrap "
        "resampling with 10,000 repetitions.\n"
    )
    (output_dir / "supplemental_table_x_title_and_footnote.txt").write_text(
        footnote, encoding="utf-8"
    )

    print(f"[DONE] Supplemental Table X and annotated Figure 3 written to {output_dir}")
    print(f"[INFO] Between-model P values: {p_between}")
    print(f"[INFO] Within-model P values: {p_within}")


if __name__ == "__main__":
    main()
