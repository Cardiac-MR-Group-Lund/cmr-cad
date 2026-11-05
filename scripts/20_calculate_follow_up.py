#!/usr/bin/env python3
"""Calculate follow-up summaries for the final manuscript cohort.

This script separates three concepts that should not be mixed:

1. Potential administrative follow-up:
   Time from index CMR to the common administrative censoring date for every
   patient. This is the appropriate quantity for a sentence such as
   "potential administrative follow-up was ...".

2. Observed time under the binary 0-day MACE endpoint:
   Time from index CMR to MACE or administrative censoring.

3. Reverse Kaplan-Meier follow-up:
   A censoring-aware estimate of the follow-up distribution, obtained by
   reversing event and censoring indicators.

Default input:
    data/processed/cmr_revision_analysis.csv

Outputs:
    results/tables/follow_up_summary.csv
    results/tables/follow_up_summary.txt
    results/tables/follow_up_manifest.json

CSV output is semicolon-delimited with UTF-8 BOM for Swedish Excel.
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

DAYS_PER_YEAR = 365.2425
EXPECTED_N = 2159
EXPECTED_EVENTS_0D = 212
EXPECTED_EVENTS_30D = 196
DEFAULT_ADMIN_DATE = "2023-12-31"


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
        default=Path("results/tables"),
    )
    parser.add_argument(
        "--administrative-date",
        type=str,
        default=DEFAULT_ADMIN_DATE,
        help="Common administrative censoring date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--index-date-column",
        type=str,
        default=None,
        help="Optional explicit index CMR date column.",
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


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def resolve_column(frame: pd.DataFrame, aliases: list[str], label: str) -> str:
    for alias in aliases:
        if alias in frame.columns:
            return alias
    lookup = {normalize(column): column for column in frame.columns}
    for alias in aliases:
        if normalize(alias) in lookup:
            return lookup[normalize(alias)]
    raise ValueError(
        f"Could not identify {label}. Tried {aliases}. "
        f"Available columns: {list(frame.columns)}"
    )


def summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("No finite values available for summary")
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)),
        "median": float(np.median(values)),
        "q1": float(np.quantile(values, 0.25)),
        "q3": float(np.quantile(values, 0.75)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def reverse_km_quantile(
    time: np.ndarray,
    original_event: np.ndarray,
    probability: float,
) -> float:
    """Quantile of follow-up from the reverse Kaplan-Meier estimator.

    Original censorings are treated as events; original clinical events are
    treated as censored. The returned quantile is the earliest time at which
    the reverse-KM survival estimate is <= 1 - probability.
    """
    time = np.asarray(time, dtype=float)
    original_event = np.asarray(original_event, dtype=int)
    reverse_event = 1 - original_event

    order = np.argsort(time)
    time = time[order]
    reverse_event = reverse_event[order]

    unique_times = np.unique(time)
    at_risk = len(time)
    survival = 1.0
    target = 1.0 - probability

    for current_time in unique_times:
        mask = time == current_time
        d = int(reverse_event[mask].sum())
        total_at_time = int(mask.sum())
        if at_risk > 0 and d > 0:
            survival *= 1.0 - d / at_risk
            if survival <= target:
                return float(current_time)
        at_risk -= total_at_time

    return float("nan")


def add_summary(rows, measure: str, values: np.ndarray):
    summary = summarize(values)
    rows.append({"Measure": measure, **summary})


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.data.exists():
        raise FileNotFoundError(args.data)

    frame = pd.read_csv(
        args.data,
        sep=detect_delimiter(args.data),
        encoding="utf-8-sig",
        low_memory=False,
    )
    if len(frame) != EXPECTED_N:
        raise ValueError(f"Expected N={EXPECTED_N}; observed {len(frame)}")

    event_0d_col = resolve_column(
        frame, ["mace_0d_event"], "0-day MACE event indicator"
    )
    tte_0d_col = resolve_column(
        frame, ["mace_0d_tte_years"], "0-day MACE time-to-event"
    )
    event_30d_col = resolve_column(
        frame, ["mace_30d_event"], "30-day MACE event indicator"
    )
    tte_30d_col = resolve_column(
        frame, ["mace_30d_tte_years"], "30-day MACE time-to-event"
    )

    if args.index_date_column is not None:
        if args.index_date_column not in frame.columns:
            raise ValueError(f"Index-date column not found: {args.index_date_column}")
        index_date_col = args.index_date_column
    else:
        index_date_col = resolve_column(
            frame,
            [
                "cmr_index_date",
                "index_exam_date",
                "index_cmr_date",
                "cmr_date",
                "exam_date",
            ],
            "index CMR date",
        )

    event_0d = pd.to_numeric(frame[event_0d_col], errors="coerce")
    tte_0d = pd.to_numeric(frame[tte_0d_col], errors="coerce")
    event_30d = pd.to_numeric(frame[event_30d_col], errors="coerce")
    tte_30d = pd.to_numeric(frame[tte_30d_col], errors="coerce")
    index_date = pd.to_datetime(frame[index_date_col], errors="coerce")
    admin_date = pd.Timestamp(args.administrative_date)

    for label, event, expected in [
        ("0-day MACE", event_0d, EXPECTED_EVENTS_0D),
        ("30-day MACE", event_30d, EXPECTED_EVENTS_30D),
    ]:
        if event.isna().any() or not event.isin([0, 1]).all():
            raise ValueError(f"Invalid event values for {label}")
        if int(event.sum()) != expected:
            raise ValueError(
                f"Expected {expected} events for {label}; observed {int(event.sum())}"
            )
    if tte_0d.isna().any() or tte_30d.isna().any():
        raise ValueError("Missing time-to-event values")
    if (tte_0d <= 0).any() or (tte_30d <= 0).any():
        raise ValueError("Non-positive time-to-event values")
    if index_date.isna().any():
        raise ValueError(f"Missing or invalid dates in {index_date_col}")
    if (index_date > admin_date).any():
        raise ValueError("An index date occurs after administrative censoring")

    potential_admin_years = (admin_date - index_date).dt.days.to_numpy(float) / DAYS_PER_YEAR
    tte_0d_array = tte_0d.to_numpy(float)
    tte_30d_array = tte_30d.to_numpy(float)
    event_0d_array = event_0d.astype(int).to_numpy()
    event_30d_array = event_30d.astype(int).to_numpy()

    rows = []
    add_summary(rows, "Potential administrative follow-up, all patients", potential_admin_years)
    add_summary(rows, "Observed event/censoring time, 0-day MACE", tte_0d_array)
    add_summary(rows, "Observed event/censoring time, 30-day MACE", tte_30d_array)
    add_summary(
        rows,
        "Administrative follow-up among patients without 0-day MACE",
        tte_0d_array[event_0d_array == 0],
    )
    add_summary(
        rows,
        "Time to event among patients with 0-day MACE",
        tte_0d_array[event_0d_array == 1],
    )

    summary_table = pd.DataFrame(rows)

    reverse_km = {
        "median": reverse_km_quantile(tte_0d_array, event_0d_array, 0.50),
        "q1": reverse_km_quantile(tte_0d_array, event_0d_array, 0.25),
        "q3": reverse_km_quantile(tte_0d_array, event_0d_array, 0.75),
    }

    summary_path = output_dir / "follow_up_summary.csv"
    summary_table.to_csv(
        summary_path,
        index=False,
        sep=";",
        encoding="utf-8-sig",
        float_format="%.6f",
    )

    admin = summary_table.iloc[0]
    censored = summary_table.iloc[3]
    lines = [
        "Follow-up summary",
        "=" * 17,
        f"N={len(frame)}; 0-day MACE events={int(event_0d.sum())}; 30-day MACE events={int(event_30d.sum())}",
        f"Index-date column: {index_date_col}",
        f"Administrative censoring date: {admin_date.date().isoformat()}",
        "",
        "Potential administrative follow-up (index CMR to 2023-12-31):",
        f"  Mean +/- SD: {admin['mean']:.2f} +/- {admin['sd']:.2f} years",
        f"  Median (IQR): {admin['median']:.2f} ({admin['q1']:.2f} to {admin['q3']:.2f}) years",
        f"  Range: {admin['minimum']:.2f} to {admin['maximum']:.2f} years",
        "",
        "Reverse Kaplan-Meier follow-up for the 0-day endpoint:",
        f"  Median: {reverse_km['median']:.2f} years",
        f"  IQR: {reverse_km['q1']:.2f} to {reverse_km['q3']:.2f} years",
        "",
        "Follow-up among patients without 0-day MACE:",
        f"  Mean +/- SD: {censored['mean']:.2f} +/- {censored['sd']:.2f} years",
        f"  Median (IQR): {censored['median']:.2f} ({censored['q1']:.2f} to {censored['q3']:.2f}) years",
        f"  Range: {censored['minimum']:.2f} to {censored['maximum']:.2f} years",
        "",
        "Recommended manuscript wording:",
        (
            f"Potential administrative follow-up was {admin['mean']:.2f} +/- "
            f"{admin['sd']:.2f} years."
        ),
        (
            f"Median follow-up estimated using the reverse Kaplan-Meier method was "
            f"{reverse_km['median']:.2f} years "
            f"(IQR, {reverse_km['q1']:.2f}-{reverse_km['q3']:.2f})."
        ),
    ]
    text_path = output_dir / "follow_up_summary.txt"
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data": str(args.data),
        "n": len(frame),
        "index_date_column": index_date_col,
        "administrative_censoring_date": admin_date.date().isoformat(),
        "days_per_year": DAYS_PER_YEAR,
        "event_columns": {
            "0_day": event_0d_col,
            "30_day": event_30d_col,
        },
        "time_columns": {
            "0_day": tte_0d_col,
            "30_day": tte_30d_col,
        },
        "reverse_km_0_day": reverse_km,
    }
    (output_dir / "follow_up_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"[DONE] {summary_path}")
    print(f"[DONE] {text_path}")
    print(
        f"[INFO] Potential administrative follow-up: "
        f"{admin['mean']:.2f} +/- {admin['sd']:.2f} years; "
        f"median {admin['median']:.2f} years"
    )
    print(
        f"[INFO] Reverse-KM median follow-up: {reverse_km['median']:.2f} years "
        f"(IQR {reverse_km['q1']:.2f}-{reverse_km['q3']:.2f})"
    )


if __name__ == "__main__":
    main()
