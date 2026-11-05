#!/usr/bin/env python3
"""Proportional-hazards diagnostics for the demographics + CMR Cox model.

Uses lifelines' implementation of the Grambsch-Therneau proportional-hazards
test based on scaled Schoenfeld residuals, with Kaplan-Meier transformed time.
Also produces visual checks of scaled Schoenfeld residuals against transformed
time with LOWESS smooths.

The diagnostic model contains the eight demographics + CMR predictors. Missing
continuous variables are median-imputed and missing binary variables are
mode-imputed. This is an assumption-diagnostic model, not a performance model.

Outputs
-------
results/models/proportional_hazards/
- supplemental_table_7_grambsch_therneau.csv
- supplemental_figure_3_schoenfeld_residuals.png/.tiff/.pdf/.svg
- cox_model_summary.csv
- run_summary.txt
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
from scipy.stats import chi2

try:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.statistics import proportional_hazard_test
except ImportError as exc:
    raise SystemExit(
        "Install lifelines in the active environment first:\n"
        "    pip install lifelines"
    ) from exc

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
except ImportError as exc:
    raise SystemExit(
        "Install statsmodels in the active environment first:\n"
        "    pip install statsmodels"
    ) from exc

EXPECTED_N = 2159
EXPECTED_EVENTS = 196

DISPLAY_LABELS = {
    "age": "Age",
    "sex": "Sex",
    "bmi": "Body mass index",
    "lvef": "Left ventricular ejection fraction",
    "lvedv_indexed": "LVEDV indexed to body surface area",
    "lvesv_indexed": "LVESV indexed to body surface area",
    "infarction_lge": "Infarction on LGE",
    "stress_ischemia": "Stress-induced ischemia",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/cmr_revision_analysis.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/models/proportional_hazards"),
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


def pick_column(frame: pd.DataFrame, candidates: list[str], label: str) -> str:
    """Resolve a column using exact names, normalized names, then safe fuzzy aliases."""
    import re

    def normalize(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value).lower())

    # 1. Exact and case-insensitive matches.
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    lower_lookup = {str(column).lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lower_lookup:
            return lower_lookup[candidate.lower()]

    # 2. Match after removing punctuation and underscores.
    normalized_lookup = {normalize(column): column for column in frame.columns}
    for candidate in candidates:
        if normalize(candidate) in normalized_lookup:
            return normalized_lookup[normalize(candidate)]

    # 3. Conservative semantic fallbacks. A match must be unique.
    fuzzy_tokens = {
        "age": [("age",), ("alder",)],
        "sex": [("sex",), ("gender",), ("kon",)],
        "BMI": [("bmi",), ("body", "mass", "index")],
        "LVEF": [("lvef",)],
        "indexed LVEDV": [("lvedv", "m2"), ("lvedvi",)],
        "indexed LVESV": [("lvesv", "m2"), ("lvesvi",)],
        "infarction on LGE": [("infarct",), ("lge", "infarct")],
        "stress-induced ischemia": [("ischemia",), ("ischemi",)],
        "survival time": [("mace", "30d", "tte"), ("mace", "tte", "year")],
        "event indicator": [("mace", "30d", "event")],
    }
    normalized_columns = {column: normalize(column) for column in frame.columns}
    for token_group in fuzzy_tokens.get(label, []):
        matches = [
            column for column, normalized in normalized_columns.items()
            if all(normalize(token) in normalized for token in token_group)
        ]
        if len(matches) == 1:
            return matches[0]

    relevant = [
        column for column in frame.columns
        if any(normalize(token) in normalize(column)
               for token in ("age", "alder", "sex", "bmi", "lvef", "lvedv", "lvesv", "infarct", "ischemi"))
    ]
    raise ValueError(
        f"Could not identify {label}; tried {candidates}. "
        f"Potential clinical columns were: {relevant}"
    )


def median_impute(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.fillna(values.median())


def mode_impute(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.dropna().empty:
        raise ValueError(f"No observed values in {series.name}")
    return values.fillna(values.mode(dropna=True).iloc[0])


def km_transform(event_times: np.ndarray, all_times: np.ndarray, events: np.ndarray) -> np.ndarray:
    """Return KM-transformed event times, matching the GT test time scale."""
    km = KaplanMeierFitter()
    km.fit(all_times, event_observed=events)
    survival = km.survival_function_at_times(event_times).to_numpy(dtype=float)
    return 1.0 - survival


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(
        args.data,
        sep=detect_delimiter(args.data),
        encoding="utf-8-sig",
        low_memory=False,
    )

    columns = {
    "duration": pick_column(
        raw,
        ["mace_30d_tte_years"],
        "survival time",
    ),
    "event": pick_column(
        raw,
        ["mace_30d_event"],
        "event indicator",
    ),
    "age": pick_column(
        raw,
        ["age_years", "age_at_cmr", "age"],
        "age",
    ),
    "sex": pick_column(
        raw,
        ["sex_male", "sex_numerical", "sex"],
        "sex",
    ),
    "bmi": pick_column(
        raw,
        ["bmi", "bmi_at_cmr"],
        "BMI",
    ),
    "lvef": pick_column(
        raw,
        ["lvef_percent", "lvef"],
        "LVEF",
    ),
    "lvedv_indexed": pick_column(
        raw,
        ["lvedvi_ml_m2"],
        "indexed LVEDV",
    ),
    "lvesv_indexed": pick_column(
        raw,
        ["lvesvi_ml_m2"],
        "indexed LVESV",
    ),
    "infarction_lge": pick_column(
        raw,
        ["infarct_binary", "mr_infarct", "infarct_raw"],
        "infarction on LGE",
    ),
    "stress_ischemia": pick_column(
        raw,
        ["ischemia_binary", "mr_ischemia", "ischemia_raw"],
        "stress-induced ischemia",
    ),
    }

    print("[INFO] Resolved input columns:")
    for semantic_name, source_column in columns.items():
        print(f"[INFO]   {semantic_name}: {source_column}")

    analysis = pd.DataFrame({
        "duration": pd.to_numeric(raw[columns["duration"]], errors="coerce"),
        "event": pd.to_numeric(raw[columns["event"]], errors="coerce"),
        "age": median_impute(raw[columns["age"]]),
        "sex": mode_impute(raw[columns["sex"]]),
        "bmi": median_impute(raw[columns["bmi"]]),
        "lvef": median_impute(raw[columns["lvef"]]),
        "lvedv_indexed": median_impute(raw[columns["lvedv_indexed"]]),
        "lvesv_indexed": median_impute(raw[columns["lvesv_indexed"]]),
        "infarction_lge": mode_impute(raw[columns["infarction_lge"]]),
        "stress_ischemia": mode_impute(raw[columns["stress_ischemia"]]),
    })

    if len(analysis) != EXPECTED_N:
        raise ValueError(f"Expected N={EXPECTED_N}; observed {len(analysis)}")
    if analysis[["duration", "event"]].isna().any().any() or (analysis["duration"] <= 0).any():
        raise ValueError("Invalid survival endpoint")
    if not analysis["event"].isin([0, 1]).all():
        raise ValueError("Event indicator is not binary")
    analysis["event"] = analysis["event"].astype(int)
    if int(analysis["event"].sum()) != EXPECTED_EVENTS:
        raise ValueError(
            f"Expected {EXPECTED_EVENTS} events; observed {int(analysis['event'].sum())}"
        )

    cph = CoxPHFitter(baseline_estimation_method="breslow")
    cph.fit(analysis, duration_col="duration", event_col="event", show_progress=False)

    # lifelines implements the Grambsch-Therneau family of tests. KM-transformed
    # time is used to align with the conventional cox.zph default.
    ph_test = proportional_hazard_test(cph, analysis, time_transform="km")
    results = ph_test.summary.reset_index()
    first_column = results.columns[0]
    results = results.rename(columns={
        first_column: "Variable",
        "test_statistic": "Chi-square",
        "p": "P value",
        "-log2(p)": "-log2(P)",
    })
    results["Variable"] = results["Variable"].map(DISPLAY_LABELS).fillna(results["Variable"])
    results.insert(2, "Degrees of freedom", 1)
    results = results[["Variable", "Chi-square", "Degrees of freedom", "P value", "-log2(P)"]]
    results.to_csv(
        output_dir / "supplemental_table_7_grambsch_therneau.csv",
        index=False,
        sep=";",
        encoding="utf-8-sig",
        float_format="%.6g",
    )

    cox_summary = cph.summary.reset_index().rename(columns={"covariate": "Variable"})
    cox_summary["Variable"] = cox_summary["Variable"].map(DISPLAY_LABELS).fillna(
        cox_summary["Variable"]
    )
    cox_summary.to_csv(
        output_dir / "cox_model_summary.csv",
        index=False,
        sep=";",
        encoding="utf-8-sig",
        float_format="%.6g",
    )

    # Visual inspection: scaled Schoenfeld residuals versus KM-transformed time.
    residuals = cph.compute_residuals(analysis, kind="scaled_schoenfeld")
    event_rows = residuals.index.to_numpy()
    event_times = analysis.loc[event_rows, "duration"].to_numpy(float)
    transformed_time = km_transform(
        event_times,
        analysis["duration"].to_numpy(float),
        analysis["event"].to_numpy(int),
    )

    fig, axes = plt.subplots(3, 3, figsize=(11.0, 9.0))
    axes = axes.flatten()
    for axis, variable in zip(axes[:8], DISPLAY_LABELS):
        values = residuals[variable].to_numpy(float)
        order = np.argsort(transformed_time)
        x = transformed_time[order]
        y = values[order]
        smooth = lowess(y, x, frac=0.35, return_sorted=True)
        axis.scatter(x, y, s=10, alpha=0.28, color="#6b6b6b", edgecolors="none")
        axis.plot(smooth[:, 0], smooth[:, 1], color="#1f77b4", linewidth=2)
        axis.axhline(0, color="black", linestyle="--", linewidth=0.8)
        test_row = results[results["Variable"].eq(DISPLAY_LABELS[variable])].iloc[0]
        p_value = float(test_row["P value"])
        p_text = "P<0.001" if p_value < 0.001 else f"P={p_value:.3f}"
        axis.set_title(f"{DISPLAY_LABELS[variable]}\n{p_text}", fontsize=9.5)
        axis.set_xlabel("KM-transformed time", fontsize=8.5)
        axis.set_ylabel("Scaled Schoenfeld residual", fontsize=8.5)
        axis.grid(False)

    axes[8].axis("off")
    significant = results.loc[results["P value"] < 0.05, "Variable"].tolist()
    axes[8].text(
        0.05, 0.78,
        "Grambsch-Therneau tests",
        fontsize=11, fontweight="bold", transform=axes[8].transAxes,
    )
    axes[8].text(
        0.05, 0.60,
        f"N={EXPECTED_N}; events={EXPECTED_EVENTS}",
        fontsize=10, transform=axes[8].transAxes,
    )
    axes[8].text(
        0.05, 0.44,
        "P<0.05: " + (", ".join(significant) if significant else "None"),
        fontsize=9, wrap=True, transform=axes[8].transAxes,
    )
    fig.suptitle("Scaled Schoenfeld Residual Diagnostics", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    outputs = {
        "png": output_dir / "supplemental_figure_3_schoenfeld_residuals.png",
        "tiff": output_dir / "supplemental_figure_3_schoenfeld_residuals.tiff",
        "pdf": output_dir / "supplemental_figure_3_schoenfeld_residuals.pdf",
        "svg": output_dir / "supplemental_figure_3_schoenfeld_residuals.svg",
    }
    fig.savefig(outputs["png"], dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(
        outputs["tiff"], dpi=600, format="tiff", facecolor="white",
        bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(outputs["pdf"], facecolor="white", bbox_inches="tight")
    fig.savefig(outputs["svg"], facecolor="white", bbox_inches="tight")
    plt.close(fig)

    # lifelines exposes coefficient-specific GT tests, not cox.zph's joint global
    # test. A clearly labelled omnibus sum is included only as a descriptive
    # screen and must not be described as the exact cox.zph global test.
    omnibus_chisq = float(results["Chi-square"].sum())
    omnibus_df = int(len(results))
    omnibus_p = float(chi2.sf(omnibus_chisq, omnibus_df))

    lines = [
        "Proportional-hazards diagnostics",
        "=" * 32,
        f"N={EXPECTED_N}; events={EXPECTED_EVENTS}",
        "Model: demographics + CMR Cox proportional hazards model",
        "Method: lifelines proportional_hazard_test with KM-transformed time",
        "Visual check: scaled Schoenfeld residual plots with LOWESS smooths",
        "",
        "Variables with P<0.05: " + (", ".join(significant) if significant else "None"),
        f"Descriptive omnibus sum: chi-square={omnibus_chisq:.4f}, df={omnibus_df}, P={omnibus_p:.4g}",
        "Note: the omnibus sum is descriptive and is not the exact joint cox.zph global test.",
    ]
    (output_dir / "run_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[DONE] {output_dir}")
    print(
        "[INFO] Variable-specific P<0.05: "
        + (", ".join(significant) if significant else "None")
    )
    print(
        "[INFO] Python/lifelines provides coefficient-specific GT tests and visual checks; "
        "the exact cox.zph global test is R-specific."
    )


if __name__ == "__main__":
    main()
