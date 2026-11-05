#!/usr/bin/env python3
"""Replot the frozen Kaplan-Meier figure in the original manuscript style.

This is a plotting-only revision of 06_generate_kaplan_meier.py.

Preserved from the original figure:
- lifelines Kaplan-Meier curves and 95% confidence intervals;
- censoring marks;
- original blue/red colors;
- original title, axis labels, legend position, and P-value annotation;
- x-axis range 0 to 4.5 years and y-axis range 0 to 1.01.

Changed only for layout:
- risk-table row labels ("No ischemia" and "Ischemia") are removed;
- no duplicate year header is drawn in the risk-table panel;
- frozen risk counts are centered exactly below the main-axis ticks 0, 1, 2, 3,
  and 4;
- risk counts use the original blue and red group colors;
- the wider original 9.2-inch figure is retained for horizontal spacing.

Frozen numerical inputs are read from the existing analysis outputs:
- results/figures/kaplan_meier/km_statistics.csv
- results/figures/kaplan_meier/km_number_at_risk.csv

The processed dataset is read only to redraw the already-defined survival
curves. The script does not rerun the log-rank test, regenerate risk counts,
change the endpoint, refit a prediction model, tune a model, or resample data.
It verifies all frozen cohort/event counts and checks the plotted KM estimates
against the saved 1- to 4-year survival estimates before saving the figure.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from lifelines import KaplanMeierFitter
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'lifelines'. Install it in the active environment with:\n"
        "    pip install lifelines"
    ) from exc

EXPECTED_INPUT_SHA256 = "e94b8aebe4c4939b0b3def943879539f4f242ba7ef39fd7ac929407578f7db9c"
EXPECTED_N = 2159
EXPECTED_ANALYSIS_N = 2070
EXPECTED_GROUPS = {
    "No stress-induced ischemia": {"value": 0, "n": 1645, "events": 67, "color": "#1f77b4"},
    "Stress-induced ischemia": {"value": 1, "n": 425, "events": 129, "color": "#d62728"},
}
RISK_TIMES = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
TIME_COL = "mace_0d_tte_years"
EVENT_COL = "mace_0d_event"
GROUP_COL = "ischemia_binary"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replot the frozen KM figure in the original style."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/cmr_revision_analysis.csv"),
    )
    parser.add_argument(
        "--statistics",
        type=Path,
        default=Path("results/figures/kaplan_meier/km_statistics.csv"),
    )
    parser.add_argument(
        "--risk-counts",
        type=Path,
        default=Path("results/figures/kaplan_meier/km_number_at_risk.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures/kaplan_meier"),
    )
    parser.add_argument(
        "--basename",
        default="km_primary_endpoint_by_ischemia_replotted",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--overwrite-original-name",
        action="store_true",
        help="Copy validated outputs to km_primary_endpoint_by_ischemia.*.",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        sample = handle.read(131072)
    try:
        return csv.Sniffer().sniff(sample, delimiters=";,\t|").delimiter
    except csv.Error:
        first = sample.splitlines()[0] if sample else ""
        return max([";", ",", "\t", "|"], key=first.count)


def read_csv_auto(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=detect_delimiter(path),
        encoding="utf-8-sig",
        low_memory=False,
        **kwargs,
    )


def fmt_p(p_value: float) -> str:
    return "P < 0.001" if p_value < 0.001 else f"P = {p_value:.3f}"


def normalize_risk_group(value: str) -> str:
    text = str(value).strip().lower()
    if text in {"no ischemia", "no stress-induced ischemia"}:
        return "No stress-induced ischemia"
    if text in {"ischemia", "stress-induced ischemia"}:
        return "Stress-induced ischemia"
    return str(value)


def main() -> None:
    args = parse_args()
    for path in [args.input, args.statistics, args.risk_counts]:
        if not path.exists():
            raise FileNotFoundError(path)

    observed_hash = sha256_file(args.input)
    if observed_hash.lower() != EXPECTED_INPUT_SHA256.lower():
        raise ValueError(
            "Frozen processed dataset checksum mismatch.\n"
            f"Expected: {EXPECTED_INPUT_SHA256}\nObserved: {observed_hash}"
        )

    required = ["analysis_id", GROUP_COL, EVENT_COL, TIME_COL]
    data = read_csv_auto(args.input, usecols=required)
    if len(data) != EXPECTED_N:
        raise ValueError(f"Expected N={EXPECTED_N}; observed {len(data)}")

    group = pd.to_numeric(data[GROUP_COL], errors="coerce")
    event = pd.to_numeric(data[EVENT_COL], errors="coerce")
    time = pd.to_numeric(data[TIME_COL], errors="coerce")
    if event.isna().any() or not event.isin([0, 1]).all():
        raise ValueError("Frozen event indicator is missing or nonbinary")
    if time.isna().any() or (time <= 0).any():
        raise ValueError("Frozen event/censoring time is missing or nonpositive")

    analysis_mask = group.isin([0, 1])
    analysis = pd.DataFrame(
        {
            "group": group[analysis_mask].astype("int8"),
            "event": event[analysis_mask].astype("int8"),
            "time": time[analysis_mask].astype(float),
        }
    ).reset_index(drop=True)
    if len(analysis) != EXPECTED_ANALYSIS_N:
        raise ValueError(
            f"Expected stratified KM N={EXPECTED_ANALYSIS_N}; observed {len(analysis)}"
        )

    statistics = read_csv_auto(args.statistics)
    risk_counts = read_csv_auto(args.risk_counts)
    risk_counts["group"] = risk_counts["group"].map(normalize_risk_group)
    risk_counts["time_years"] = pd.to_numeric(risk_counts["time_years"], errors="raise")
    risk_counts["number_at_risk"] = pd.to_numeric(
        risk_counts["number_at_risk"], errors="raise"
    ).astype(int)

    if "logrank_p_value" not in statistics.columns:
        raise ValueError("Saved km_statistics.csv lacks logrank_p_value")
    p_values = pd.to_numeric(statistics["logrank_p_value"], errors="coerce").dropna()
    if len(p_values) != 1:
        raise ValueError(f"Expected one frozen log-rank P value; observed {len(p_values)}")
    frozen_p = float(p_values.iloc[0])

    # Original figure geometry and style.
    fig = plt.figure(figsize=(9.2, 7.8))
    grid = fig.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=[5.2, 1.15],
        hspace=0.22,
    )
    ax = fig.add_subplot(grid[0])
    risk_ax = fig.add_subplot(grid[1], sharex=ax)

    validation_rows = []
    for label, specification in EXPECTED_GROUPS.items():
        mask = analysis["group"].eq(specification["value"])
        durations = analysis.loc[mask, "time"]
        events = analysis.loc[mask, "event"]
        observed_n = int(mask.sum())
        observed_events = int(events.sum())
        if observed_n != specification["n"] or observed_events != specification["events"]:
            raise ValueError(
                f"Frozen group mismatch for {label}: N={observed_n}, events={observed_events}"
            )

        kmf = KaplanMeierFitter(label=f"{label} (n={observed_n})")
        kmf.fit(durations=durations, event_observed=events)
        kmf.plot_survival_function(
            ax=ax,
            ci_show=True,
            ci_alpha=0.16,
            color=specification["color"],
            linewidth=2.2,
            show_censors=True,
            censor_styles={"marker": "+", "ms": 4.5, "mew": 0.9},
        )

        saved = statistics[statistics["group"].eq(label)]
        if len(saved) != 1:
            raise ValueError(f"Expected one saved statistics row for {label}; observed {len(saved)}")
        saved = saved.iloc[0]
        for year, column in [
            (1.0, "survival_1_year"),
            (2.0, "survival_2_years"),
            (3.0, "survival_3_years"),
            (4.0, "survival_4_years"),
        ]:
            plotted_value = float(kmf.predict(year))
            saved_value = float(saved[column])
            if not np.isclose(plotted_value, saved_value, rtol=0, atol=1e-10):
                raise ValueError(
                    f"Frozen KM estimate mismatch for {label} at {year:g} years: "
                    f"plotted={plotted_value:.12f}, saved={saved_value:.12f}"
                )
            validation_rows.append(
                {
                    "group": label,
                    "time_years": year,
                    "saved_survival": saved_value,
                    "plotted_survival": plotted_value,
                    "difference": plotted_value - saved_value,
                    "status": "PASS",
                }
            )

    ax.set_xlim(0, 4.5)
    ax.set_ylim(0.0, 1.01)
    ax.set_ylabel("Event-free survival probability")
    ax.set_xlabel("Time since index CMR, years")
    ax.set_title("Primary Endpoint by Stress-Induced Ischemia")
    ax.grid(False)
    ax.legend(loc="lower left", bbox_to_anchor=(0.02, 0.04), frameon=False)
    ax.text(
        0.98,
        0.96,
        fmt_p(frozen_p),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
    )
    ax.set_xticks(RISK_TIMES)
    ax.set_xticklabels([str(int(value)) for value in RISK_TIMES])
    ax.tick_params(axis="x", labelbottom=True, length=4, pad=5)

    # Plot-only risk panel. No repeated group labels and no duplicate year row.
    # The shared x-axis guarantees exact alignment with years 0-4 above.
    risk_ax.set_xlim(0, 4.5)
    risk_ax.set_ylim(-0.25, 1.55)
    risk_ax.set_xticks(RISK_TIMES)
    risk_ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    risk_ax.set_yticks([])
    risk_ax.set_frame_on(False)
    risk_ax.text(
        0.0,
        0.96,
        "Number at risk",
        transform=risk_ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
        fontsize=9.5,
    )

    row_y = {
        "No stress-induced ischemia": 0.73,
        "Stress-induced ischemia": 0.18,
    }
    for label in EXPECTED_GROUPS:
        subset = (
            risk_counts[risk_counts["group"].eq(label)]
            .set_index("time_years")
            .reindex(RISK_TIMES)
        )
        if subset["number_at_risk"].isna().any():
            raise ValueError(f"Saved risk counts are incomplete for {label}")
        for risk_time, count in zip(RISK_TIMES, subset["number_at_risk"].astype(int)):
            risk_ax.text(
                float(risk_time),
                row_y[label],
                str(int(count)),
                ha="center",
                va="center",
                fontsize=9.5,
                color=EXPECTED_GROUPS[label]["color"],
                clip_on=False,
            )

    fig.subplots_adjust(left=0.12, right=0.98, top=0.93, bottom=0.08)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "png": output_dir / f"{args.basename}.png",
        "pdf": output_dir / f"{args.basename}.pdf",
        "svg": output_dir / f"{args.basename}.svg",
        "tiff": output_dir / f"{args.basename}.tiff",
    }
    fig.savefig(paths["png"], dpi=args.dpi, bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    fig.savefig(paths["svg"], bbox_inches="tight")
    fig.savefig(
        paths["tiff"],
        dpi=600,
        format="tiff",
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)

    validation_path = output_dir / f"{args.basename}_validation.csv"
    pd.DataFrame(validation_rows).to_csv(
        validation_path,
        index=False,
        sep=";",
        encoding="utf-8-sig",
        float_format="%.12g",
    )

    for path in [*paths.values(), validation_path]:
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Output validation failed: {path}")

    if args.overwrite_original_name:
        for extension, source in paths.items():
            target = output_dir / f"km_primary_endpoint_by_ischemia.{extension}"
            shutil.copy2(source, target)
            if not target.exists() or target.stat().st_size == 0:
                raise RuntimeError(f"Failed to write final-name output: {target}")
            print(f"[DONE] {target}")

    print(f"[DONE] {paths['png']}")
    print(f"[DONE] {paths['pdf']}")
    print(f"[DONE] {paths['svg']}")
    print(f"[DONE] {paths['tiff']}")
    print(f"[DONE] {validation_path}")
    print("[INFO] Original plot style retained.")
    print("[INFO] Frozen log-rank P value and frozen number-at-risk counts reused.")
    print("[INFO] Risk-row labels and duplicate risk-table year header removed.")
    print("[INFO] Risk counts aligned to shared 0-4 year ticks and colored by group.")


if __name__ == "__main__":
    main()
