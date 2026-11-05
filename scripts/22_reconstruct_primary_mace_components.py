#!/usr/bin/env python3
"""Reconstruct mutually exclusive first-event components of revised MACE.

The source patient_v15.csv contains:
- primary MACE event/time pairs for 0-, 30-, 60-, and 90-day windows;
- a validated hierarchy label for cardiovascular death (1), acute myocardial
  infarction (2), and late PCI/CABG (3);
- the corresponding hard-event/late-revascularization event time.

For each primary endpoint window, this script assigns the first qualifying MACE
component as follows:
1. Cardiovascular death when hierarchy label == 1 and the primary endpoint time
   agrees with the hierarchy event time within one day.
2. Acute myocardial infarction when hierarchy label == 2 and the times agree
   within one day.
3. PCI/CABG for every other primary MACE event, because the primary composite
   contains only cardiovascular death, AMI, and PCI/CABG.

This correctly handles patients who had PCI/CABG before a later death or AMI:
they are classified by the first qualifying primary event, not by any-event
history. Only aggregate counts are written; no identifiers or patient rows are
exported.
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

DAYS_PER_YEAR = 365.25
EXPECTED_N = 2159
EXPECTED_EVENTS = {0: 212, 30: 196, 60: 179, 90: 163}
WINDOWS = (0, 30, 60, 90)
BLANKS = {"", "nan", "none", "null", "na", "n/a", "nat", "<na>"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("data/raw/patient_v15.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("results/audit/mace_components"))
    p.add_argument("--tolerance-days", type=float, default=1.0)
    return p.parse_args()


def detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        sample = handle.read(131072)
    try:
        return csv.Sniffer().sniff(sample, delimiters=";,\t|").delimiter
    except csv.Error:
        first = sample.splitlines()[0] if sample else ""
        return max([";", ",", "\t", "|"], key=first.count)


def normalize_name(value: str) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[ .\-/]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def clean(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip()
    return out.mask(out.str.lower().isin(BLANKS))


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(clean(series).str.replace(",", ".", regex=False), errors="coerce")


def fmt(n: int, denominator: int) -> str:
    return f"{n} ({100*n/denominator:.1f}%)"


def main():
    args = parse_args()
    outdir = args.output_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(
        args.input,
        sep=detect_delimiter(args.input),
        dtype=str,
        encoding="utf-8-sig",
        keep_default_na=False,
        na_filter=False,
        low_memory=False,
    )
    raw.columns = [normalize_name(c) for c in raw.columns]
    mr = raw.loc[clean(raw["modality"]).str.upper().eq("MR")].copy().reset_index(drop=True)
    if len(mr) != EXPECTED_N:
        raise ValueError(f"Expected MR N={EXPECTED_N}; observed {len(mr)}")

    required = {
        "comp_cv_event_late_revasc_diff",
        "comp_cv_tte_late_revasc_5y",
        "ischemia_dicho",
    }
    for window in WINDOWS:
        required |= {
            f"comp_cv_event_blind{window}_5y",
            f"comp_cv_tte_blind{window}_5y",
        }
    missing = sorted(required - set(mr.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    hierarchy = numeric(mr["comp_cv_event_late_revasc_diff"])
    hierarchy_tte = numeric(mr["comp_cv_tte_late_revasc_5y"])
    ischemia = numeric(mr["ischemia_dicho"])

    aggregate_rows = []
    stratified_rows = []
    check_rows = []

    for window in WINDOWS:
        event = numeric(mr[f"comp_cv_event_blind{window}_5y"])
        primary_tte = numeric(mr[f"comp_cv_tte_blind{window}_5y"])
        if event.isna().any() or primary_tte.isna().any():
            raise ValueError(f"Missing endpoint values for {window}-day window")
        observed_events = int(event.eq(1).sum())
        if observed_events != EXPECTED_EVENTS[window]:
            raise ValueError(
                f"Expected {EXPECTED_EVENTS[window]} events at {window} days; observed {observed_events}"
            )

        time_agrees = (
            (primary_tte - hierarchy_tte).abs() * DAYS_PER_YEAR
            <= args.tolerance_days
        )
        cv_death = event.eq(1) & hierarchy.eq(1) & time_agrees
        ami = event.eq(1) & hierarchy.eq(2) & time_agrees
        intervention = event.eq(1) & ~(cv_death | ami)

        components = {
            "Cardiovascular death": cv_death,
            "Acute myocardial infarction": ami,
            "Coronary intervention (PCI/CABG)": intervention,
        }

        # Audit cases with a hard-event hierarchy but an earlier primary event.
        earlier_intervention_before_hard = (
            event.eq(1)
            & hierarchy.isin([1, 2])
            & (primary_tte < hierarchy_tte - args.tolerance_days / DAYS_PER_YEAR)
        )
        primary_after_hard = (
            event.eq(1)
            & hierarchy.isin([1, 2])
            & (primary_tte > hierarchy_tte + args.tolerance_days / DAYS_PER_YEAR)
        )

        component_total = sum(int(mask.sum()) for mask in components.values())
        check_rows.extend([
            {
                "window_days": window,
                "check": "Components sum to composite",
                "status": "PASS" if component_total == observed_events else "REVIEW",
                "observed": component_total,
                "expected": observed_events,
            },
            {
                "window_days": window,
                "check": "Primary event after hard hierarchy time",
                "status": "PASS" if int(primary_after_hard.sum()) == 0 else "REVIEW",
                "observed": int(primary_after_hard.sum()),
                "expected": 0,
            },
        ])

        aggregate_rows.append({
            "window_days": window,
            "total_mace": observed_events,
            "cardiovascular_death": int(cv_death.sum()),
            "acute_myocardial_infarction": int(ami.sum()),
            "coronary_intervention_pci_cabg": int(intervention.sum()),
            "patients_with_intervention_before_later_hard_event": int(
                earlier_intervention_before_hard.sum()
            ),
            "formatted_total": fmt(observed_events, EXPECTED_N),
            "formatted_cv_death": fmt(int(cv_death.sum()), EXPECTED_N),
            "formatted_ami": fmt(int(ami.sum()), EXPECTED_N),
            "formatted_intervention": fmt(int(intervention.sum()), EXPECTED_N),
        })

        for group_label, group_mask in [
            ("Total cohort", pd.Series(True, index=mr.index)),
            ("No stress-induced ischemia", ischemia.eq(0)),
            ("Stress-induced ischemia", ischemia.eq(1)),
        ]:
            denominator = int(group_mask.sum())
            for component_label, component_mask in {
                "Total MACE": event.eq(1),
                **components,
            }.items():
                count = int((group_mask & component_mask).sum())
                stratified_rows.append({
                    "window_days": window,
                    "group": group_label,
                    "group_n": denominator,
                    "component": component_label,
                    "count": count,
                    "percent": 100 * count / denominator,
                    "formatted": fmt(count, denominator),
                })

    aggregate = pd.DataFrame(aggregate_rows)
    stratified = pd.DataFrame(stratified_rows)
    checks = pd.DataFrame(check_rows)

    # Cross-tab provides a transparent aggregate audit of hierarchy labels among
    # each primary endpoint's events.
    crosstab_rows = []
    for window in WINDOWS:
        event = numeric(mr[f"comp_cv_event_blind{window}_5y"])
        primary_tte = numeric(mr[f"comp_cv_tte_blind{window}_5y"])
        for label in [0, 1, 2, 3]:
            mask = event.eq(1) & hierarchy.eq(label)
            agreeing = mask & (
                (primary_tte - hierarchy_tte).abs() * DAYS_PER_YEAR <= args.tolerance_days
            )
            earlier = mask & (
                primary_tte < hierarchy_tte - args.tolerance_days / DAYS_PER_YEAR
            )
            crosstab_rows.append({
                "window_days": window,
                "hierarchy_label": label,
                "hierarchy_definition": {
                    0: "No late-hierarchy event",
                    1: "Cardiovascular death",
                    2: "Acute myocardial infarction",
                    3: "PCI/CABG after 6 months",
                }[label],
                "primary_mace_events": int(mask.sum()),
                "primary_time_agrees_with_hierarchy": int(agreeing.sum()),
                "primary_event_earlier_than_hierarchy": int(earlier.sum()),
            })
    crosstab = pd.DataFrame(crosstab_rows)

    for frame, name in [
        (aggregate, "primary_mace_component_summary.csv"),
        (stratified, "primary_mace_components_by_ischemia.csv"),
        (crosstab, "primary_mace_hierarchy_crosstab.csv"),
        (checks, "primary_mace_component_checks.csv"),
    ]:
        frame.to_csv(outdir / name, index=False, sep=";", encoding="utf-8-sig")

    review_count = int(checks["status"].eq("REVIEW").sum())
    lines = [
        "Primary MACE component reconstruction",
        "=" * 37,
        f"Tolerance for matching event times: {args.tolerance_days:.1f} day",
        "Components are mutually exclusive first qualifying events.",
        "",
    ]
    for row in aggregate.itertuples(index=False):
        lines.extend([
            f"{row.window_days}-day window: total {row.total_mace}",
            f"  Cardiovascular death: {row.cardiovascular_death}",
            f"  Acute myocardial infarction: {row.acute_myocardial_infarction}",
            f"  PCI/CABG: {row.coronary_intervention_pci_cabg}",
            f"  Intervention before later hard event: {row.patients_with_intervention_before_later_hard_event}",
        ])
    lines.append("")
    lines.append(
        f"Checks: {int(checks['status'].eq('PASS').sum())} PASS; {review_count} REVIEW"
    )
    (outdir / "primary_mace_component_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(args.input),
        "n": EXPECTED_N,
        "tolerance_days": args.tolerance_days,
        "component_definition": (
            "Mutually exclusive first qualifying event. CV death or AMI requires "
            "matching primary and validated hierarchy event times; all remaining "
            "primary events are PCI/CABG."
        ),
        "checks": {
            "pass": int(checks["status"].eq("PASS").sum()),
            "review": review_count,
        },
    }
    (outdir / "primary_mace_component_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(f"[DONE] {outdir}")
    for row in aggregate.itertuples(index=False):
        print(
            f"[INFO] {row.window_days}d: total={row.total_mace}; "
            f"CV death={row.cardiovascular_death}; AMI={row.acute_myocardial_infarction}; "
            f"PCI/CABG={row.coronary_intervention_pci_cabg}"
        )
    print(
        f"[INFO] Checks: {int(checks['status'].eq('PASS').sum())} PASS; "
        f"{review_count} REVIEW"
    )
    if review_count:
        raise RuntimeError("One or more endpoint reconstruction checks require review")


if __name__ == "__main__":
    main()
