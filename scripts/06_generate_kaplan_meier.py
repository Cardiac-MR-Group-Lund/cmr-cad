#!/usr/bin/env python3
"""
06_generate_kaplan_meier.py

Generate the Kaplan-Meier figure requested by Reviewer 1, stratified by
stress-induced ischemia.

Primary analysis:
- Endpoint: mace_0d_event / mace_0d_tte_years
- Time zero: index stress-perfusion CMR
- Groups: observed ischemia_binary = 0 versus 1
- Patients with indeterminate ischemia are excluded from this stratified plot
- Unadjusted log-rank test
- 95% confidence intervals, censoring marks, and number-at-risk table

Input:
    data/processed/cmr_revision_analysis.csv

Outputs:
    results/figures/kaplan_meier/
        km_primary_endpoint_by_ischemia.png
        km_primary_endpoint_by_ischemia.pdf
        km_primary_endpoint_by_ischemia.svg
        km_statistics.csv
        km_number_at_risk.csv
        km_checks.csv
        km_manifest.json
        km_summary.txt

Dependencies:
    pandas, numpy, matplotlib, lifelines

Run from repository root:
    python scripts/06_generate_kaplan_meier.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'lifelines'. Install it in the active environment with:\n"
        "    pip install lifelines\n"
        "Then rerun this script."
    ) from exc


SCRIPT_VERSION = "1.2.0"
DEFAULT_INPUT = Path("data/processed/cmr_revision_analysis.csv")
DEFAULT_OUTPUT_DIR = Path("results/figures/kaplan_meier")
EXPECTED_INPUT_SHA256 = "e94b8aebe4c4939b0b3def943879539f4f242ba7ef39fd7ac929407578f7db9c"
EXPECTED_N = 2159
EXPECTED_ANALYSIS_N = 2070
EXPECTED_NO_ISCHEMIA_N = 1645
EXPECTED_ISCHEMIA_N = 425
EXPECTED_NO_ISCHEMIA_EVENTS = 67
EXPECTED_ISCHEMIA_EVENTS = 129
EXPECTED_INDETERMINATE_N = 89
EXPECTED_INDETERMINATE_EVENTS = 16

TIME_COL = "mace_0d_tte_years"
EVENT_COL = "mace_0d_event"
GROUP_COL = "ischemia_binary"
RISK_TIMES = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
GROUPS = [
    (0, "No stress-induced ischemia", "#1f77b4"),
    (1, "Stress-induced ischemia", "#d62728"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate ischemia-stratified Kaplan-Meier figure.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--expected-sha256", default=EXPECTED_INPUT_SHA256)
    p.add_argument("--allow-input-hash-mismatch", action="store_true")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


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


def fmt_p(p: float) -> str:
    if p < 0.001:
        return "P < 0.001"
    return f"P = {p:.3f}"


def at_risk_counts(times: pd.Series, grid: np.ndarray) -> list[int]:
    # At risk immediately before each displayed time: observed duration >= t.
    return [int((times >= float(t)).sum()) for t in grid]


def survival_at(kmf: KaplanMeierFitter, time: float) -> float:
    return float(kmf.predict(time))


def median_or_none(kmf: KaplanMeierFitter) -> float | None:
    value = float(kmf.median_survival_time_)
    return None if math.isinf(value) else value


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8", float_format="%.12g")


def main() -> int:
    args = parse_args()
    source = args.input.resolve()
    output_dir = args.output_dir.resolve()

    if not source.exists():
        print(f"[ERROR] Input not found: {source}", file=sys.stderr)
        return 2

    input_hash = sha256_file(source)
    hash_match = input_hash.lower() == args.expected_sha256.lower()
    if not hash_match and not args.allow_input_hash_mismatch:
        print("[ERROR] Frozen analysis dataset checksum mismatch.", file=sys.stderr)
        print(f"[ERROR] Expected: {args.expected_sha256}", file=sys.stderr)
        print(f"[ERROR] Observed: {input_hash}", file=sys.stderr)
        return 3

    delimiter = detect_delimiter(source)
    required = ["analysis_id", GROUP_COL, EVENT_COL, TIME_COL]
    df = pd.read_csv(
        source,
        sep=delimiter,
        usecols=required,
        encoding="utf-8-sig",
        low_memory=False,
    )

    if len(df) != EXPECTED_N:
        raise ValueError(f"Expected N={EXPECTED_N}; observed N={len(df)}")
    if df["analysis_id"].isna().any() or df["analysis_id"].duplicated().any():
        raise ValueError("analysis_id contains missing or duplicate values.")

    group = pd.to_numeric(df[GROUP_COL], errors="coerce")
    event = pd.to_numeric(df[EVENT_COL], errors="coerce")
    time = pd.to_numeric(df[TIME_COL], errors="coerce")

    if event.isna().any() or ~event.isin([0, 1]).all():
        raise ValueError("Primary endpoint indicator is missing or nonbinary.")
    if time.isna().any() or (time <= 0).any():
        raise ValueError("Primary endpoint TTE is missing or nonpositive.")
    if not (group.isna() | group.isin([0, 1])).all():
        raise ValueError("Ischemia grouping variable contains unexpected values.")

    analysis_mask = group.isin([0, 1])
    analysis = pd.DataFrame({
        "group": group[analysis_mask].astype("int8"),
        "event": event[analysis_mask].astype("int8"),
        "time": time[analysis_mask].astype(float),
    }).reset_index(drop=True)

    indeterminate_mask = group.isna()
    indeterminate_events = int(event[indeterminate_mask].sum())

    checks: list[dict[str, Any]] = []
    def check(name: str, passed: bool, observed: Any, expected: Any, detail: str) -> None:
        checks.append({
            "check": name,
            "status": "PASS" if passed else "REVIEW",
            "observed": observed,
            "expected": expected,
            "detail": detail,
        })

    check("Input SHA-256", hash_match, input_hash, args.expected_sha256, "Frozen dataset identity.")
    check("Total cohort N", len(df) == EXPECTED_N, len(df), EXPECTED_N, "All MR patients.")
    check("KM analysis N", len(analysis) == EXPECTED_ANALYSIS_N, len(analysis), EXPECTED_ANALYSIS_N,
          "Patients with observed ischemia status.")
    check("No-ischemia N", int((analysis["group"] == 0).sum()) == EXPECTED_NO_ISCHEMIA_N,
          int((analysis["group"] == 0).sum()), EXPECTED_NO_ISCHEMIA_N, "Observed status.")
    check("Ischemia N", int((analysis["group"] == 1).sum()) == EXPECTED_ISCHEMIA_N,
          int((analysis["group"] == 1).sum()), EXPECTED_ISCHEMIA_N, "Observed status.")
    check("Indeterminate N", int(indeterminate_mask.sum()) == EXPECTED_INDETERMINATE_N,
          int(indeterminate_mask.sum()), EXPECTED_INDETERMINATE_N, "Excluded from stratified KM only.")
    check("No-ischemia events", int(analysis.loc[analysis["group"] == 0, "event"].sum()) == EXPECTED_NO_ISCHEMIA_EVENTS,
          int(analysis.loc[analysis["group"] == 0, "event"].sum()), EXPECTED_NO_ISCHEMIA_EVENTS, "Primary endpoint.")
    check("Ischemia events", int(analysis.loc[analysis["group"] == 1, "event"].sum()) == EXPECTED_ISCHEMIA_EVENTS,
          int(analysis.loc[analysis["group"] == 1, "event"].sum()), EXPECTED_ISCHEMIA_EVENTS, "Primary endpoint.")
    check("Indeterminate events", indeterminate_events == EXPECTED_INDETERMINATE_EVENTS,
          indeterminate_events, EXPECTED_INDETERMINATE_EVENTS, "Not displayed in stratified KM.")
    check("All primary events reconcile",
          EXPECTED_NO_ISCHEMIA_EVENTS + EXPECTED_ISCHEMIA_EVENTS + indeterminate_events == int(event.sum()),
          EXPECTED_NO_ISCHEMIA_EVENTS + EXPECTED_ISCHEMIA_EVENTS + indeterminate_events,
          int(event.sum()), "Total primary events.")

    output_dir.mkdir(parents=True, exist_ok=True)
    checks_df = pd.DataFrame(checks)
    write_csv(checks_df, output_dir / "km_checks.csv")
    review_count = int(checks_df["status"].eq("REVIEW").sum())
    if review_count:
        raise RuntimeError(f"KM generation stopped: {review_count} checks require review.")

    # Log-rank test.
    no_isch = analysis["group"].eq(0)
    yes_isch = analysis["group"].eq(1)
    logrank = logrank_test(
        analysis.loc[no_isch, "time"],
        analysis.loc[yes_isch, "time"],
        event_observed_A=analysis.loc[no_isch, "event"],
        event_observed_B=analysis.loc[yes_isch, "event"],
    )

    # Figure with dedicated axes for plot and risk table.
    fig = plt.figure(figsize=(9.2, 7.8))
    grid = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[5.2, 1.15], hspace=0.22)
    ax = fig.add_subplot(grid[0])
    risk_ax = fig.add_subplot(grid[1], sharex=ax)

    fitters: dict[int, KaplanMeierFitter] = {}
    stats_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []

    for value, label, color in GROUPS:
        mask = analysis["group"].eq(value)
        durations = analysis.loc[mask, "time"]
        events = analysis.loc[mask, "event"]
        kmf = KaplanMeierFitter(label=f"{label} (n={int(mask.sum())})")
        kmf.fit(durations=durations, event_observed=events)
        fitters[value] = kmf
        kmf.plot_survival_function(
            ax=ax,
            ci_show=True,
            ci_alpha=0.16,
            color=color,
            linewidth=2.2,
            show_censors=True,
            censor_styles={"marker": "+", "ms": 4.5, "mew": 0.9},
        )

        counts = at_risk_counts(durations, RISK_TIMES)
        for t, n_at_risk in zip(RISK_TIMES, counts):
            risk_rows.append({
                "group": label,
                "time_years": float(t),
                "number_at_risk": n_at_risk,
            })

        stats_rows.append({
            "group": label,
            "n": int(mask.sum()),
            "events": int(events.sum()),
            "censored": int(mask.sum() - events.sum()),
            "survival_1_year": survival_at(kmf, 1.0),
            "survival_2_years": survival_at(kmf, 2.0),
            "survival_3_years": survival_at(kmf, 3.0),
            "survival_4_years": survival_at(kmf, 4.0),
            "median_event_free_survival_years": median_or_none(kmf),
        })

    ax.set_xlim(0, 4.5)
    ax.set_ylim(0.0, 1.01)
    ax.set_ylabel("Event-free survival probability")
    ax.set_xlabel("Time since index CMR, years")
    ax.set_title("Primary Endpoint by Stress-Induced Ischemia")
    ax.grid(False)
    ax.legend(loc="lower left", bbox_to_anchor=(0.02, 0.04), frameon=False)
    ax.text(
        0.98, 0.96, fmt_p(float(logrank.p_value)),
        transform=ax.transAxes, ha="right", va="top", fontsize=10,
    )
    ax.set_xticks(RISK_TIMES)
    ax.set_xticklabels([str(int(t)) for t in RISK_TIMES])
    ax.tick_params(axis="x", labelbottom=True, length=4, pad=5)

    # Risk table on a separate, axis-free panel. Using a real Matplotlib table
    # prevents row labels from overlapping or clipping the main x-axis.
    risk_ax.axis("off")
    risk_df = pd.DataFrame(risk_rows)
    table_values = []
    row_labels = []
    row_colors = []
    for value, label, color in GROUPS:
        subset = (
            risk_df[risk_df["group"].eq(label)]
            .set_index("time_years")
            .reindex(RISK_TIMES)
        )
        table_values.append([str(int(x)) for x in subset["number_at_risk"].tolist()])
        row_labels.append("No ischemia" if value == 0 else "Ischemia")
        row_colors.append(color)

    risk_table = risk_ax.table(
        cellText=table_values,
        rowLabels=row_labels,
        colLabels=[str(int(t)) for t in RISK_TIMES],
        cellLoc="center",
        rowLoc="right",
        colLoc="center",
        loc="center",
        bbox=[0.08, 0.05, 0.88, 0.88],
    )
    risk_table.auto_set_font_size(False)
    risk_table.set_fontsize(9.5)
    risk_table.scale(1.0, 1.20)

    # Remove table borders for a clean journal-style risk table.
    for (row, col), cell in risk_table.get_celld().items():
        cell.set_linewidth(0)
        cell.set_facecolor("white")
        if row == 0:
            cell.get_text().set_fontweight("normal")
        if col == -1 and row > 0:
            cell.get_text().set_color(row_colors[row - 1])
            cell.get_text().set_ha("right")

    risk_ax.text(
        0.0, 0.96, "Number at risk",
        transform=risk_ax.transAxes,
        ha="left", va="top", fontweight="bold", fontsize=9.5,
    )

    fig.subplots_adjust(left=0.12, right=0.98, top=0.93, bottom=0.08)

    png_path = output_dir / "km_primary_endpoint_by_ischemia.png"
    pdf_path = output_dir / "km_primary_endpoint_by_ischemia.pdf"
    svg_path = output_dir / "km_primary_endpoint_by_ischemia.svg"
    tiff_path = output_dir / "km_primary_endpoint_by_ischemia.tiff"
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(tiff_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    stats_df = pd.DataFrame(stats_rows)
    stats_df["logrank_chi_square"] = np.nan
    stats_df["logrank_p_value"] = np.nan
    stats_df.loc[0, "logrank_chi_square"] = float(logrank.test_statistic)
    stats_df.loc[0, "logrank_p_value"] = float(logrank.p_value)
    write_csv(stats_df, output_dir / "km_statistics.csv")
    write_csv(risk_df, output_dir / "km_number_at_risk.csv")

    manifest = {
        "build": {
            "script": Path(__file__).name,
            "version": SCRIPT_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
        "input": {"path": str(args.input), "sha256": input_hash, "rows": len(df)},
        "analysis": {
            "endpoint_event": EVENT_COL,
            "endpoint_time": TIME_COL,
            "time_zero": "Date of index stress-perfusion CMR",
            "groups": GROUP_COL,
            "included_n": len(analysis),
            "excluded_indeterminate_n": int(indeterminate_mask.sum()),
            "method": "Kaplan-Meier event-free survival",
            "comparison": "Two-sided unadjusted log-rank test",
            "logrank_chi_square": float(logrank.test_statistic),
            "logrank_p_value": float(logrank.p_value),
            "risk_table_times_years": RISK_TIMES.tolist(),
        },
        "outputs": {
            "png": str(png_path), "pdf": str(pdf_path), "svg": str(svg_path),
            "png_sha256": sha256_file(png_path),
            "pdf_sha256": sha256_file(pdf_path),
            "svg_sha256": sha256_file(svg_path),
            "tiff_sha256": sha256_file(tiff_path),
        },
        "checks": {"pass": int(checks_df["status"].eq("PASS").sum()), "review": 0},
        "privacy": "Aggregate statistics and figure only; no patient-level rows are exported.",
    }
    with (output_dir / "km_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    result_sentence = (
        "Kaplan-Meier event-free survival differed between patients with and "
        f"without stress-induced ischemia (log-rank {fmt_p(float(logrank.p_value))})."
    )
    summary_lines = [
        "Kaplan-Meier analysis complete",
        "=" * 31,
        f"Input SHA-256: {input_hash}",
        f"Included N: {len(analysis):,}",
        f"Excluded indeterminate ischemia: {int(indeterminate_mask.sum()):,}",
        "",
        f"No ischemia: N={int(no_isch.sum()):,}; events={int(analysis.loc[no_isch, 'event'].sum()):,}",
        f"Ischemia: N={int(yes_isch.sum()):,}; events={int(analysis.loc[yes_isch, 'event'].sum()):,}",
        "",
        f"Log-rank chi-square: {float(logrank.test_statistic):.6f}",
        f"Log-rank P value: {float(logrank.p_value):.12g}",
        f"Manuscript format: {fmt_p(float(logrank.p_value))}",
        "",
        "Suggested Results sentence:",
        result_sentence,
        "",
        "Checks: 10 PASS; 0 REVIEW",
        "Y-axis displayed from 0 to 1 to avoid visually exaggerating group differences.",
        "Number-at-risk values are displayed in a separate table panel below the x-axis.",
    ]
    (output_dir / "km_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"[INFO] KM analysis N={len(analysis)}; no ischemia={int(no_isch.sum())}; ischemia={int(yes_isch.sum())}")
    print(f"[INFO] Events: no ischemia={int(analysis.loc[no_isch, 'event'].sum())}; ischemia={int(analysis.loc[yes_isch, 'event'].sum())}")
    print(f"[INFO] Log-rank chi-square={float(logrank.test_statistic):.6f}; {fmt_p(float(logrank.p_value))}")
    print(f"[INFO] Figure written: {png_path}")
    print("[INFO] Checks: 10 PASS; 0 REVIEW")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
