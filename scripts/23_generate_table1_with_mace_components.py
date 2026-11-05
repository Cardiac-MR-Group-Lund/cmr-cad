#!/usr/bin/env python3
"""Regenerate Table 1 and add an audited MACE component breakdown.

This script first runs the existing ``scripts/05_generate_table1.py`` so that
all current Table 1 definitions, checks, missingness outputs, and formatting are
preserved. It then reconstructs mutually exclusive first qualifying MACE events
from ``data/raw/patient_v15.csv`` and inserts the component rows directly below
Total MACE.

Default displayed component window: 0 days (MACE from index CMR).
Optional alternatives: 30, 60, or 90 days via ``--component-window``.

Primary MACE components:
- cardiovascular death;
- acute myocardial infarction;
- coronary intervention with PCI/CABG.

For cardiovascular death or AMI, the primary endpoint time must agree with the
validated hard-event hierarchy time within one day. Remaining primary MACE
first events are classified as PCI/CABG. This correctly classifies patients
whose first event was PCI/CABG and who later experienced death or AMI.

No patient-level data or identifiers are written.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365.25
EXPECTED_N = 2159
EXPECTED_GROUPS = {
    "Total cohort": 2159,
    "No stress-induced ischemia": 1645,
    "Stress-induced ischemia": 425,
}
EXPECTED_EVENTS = {0: 212, 30: 196, 60: 179, 90: 163}
WINDOWS = (0, 30, 60, 90)
BLANKS = {"", "nan", "none", "null", "na", "n/a", "nat", "<na>"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-script",
        type=Path,
        default=Path("scripts/05_generate_table1.py"),
        help="Existing validated Table 1 generator.",
    )
    parser.add_argument(
        "--processed-data",
        type=Path,
        default=Path("data/processed/cmr_revision_analysis.csv"),
    )
    parser.add_argument(
        "--raw-data",
        type=Path,
        default=Path("data/raw/patient_v15.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/tables/table1"),
    )
    parser.add_argument(
        "--component-window",
        type=int,
        choices=WINDOWS,
        default=0,
        help="Window whose MACE components are displayed beneath total MACE.",
    )
    parser.add_argument("--tolerance-days", type=float, default=1.0)
    parser.add_argument(
        "--skip-base-generation",
        action="store_true",
        help="Use existing Table 1 files instead of rerunning 05_generate_table1.py.",
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


def normalize_name(value: str) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[ .\-/]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def clean(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip()
    return out.mask(out.str.lower().isin(BLANKS))


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(clean(series).str.replace(",", ".", regex=False), errors="coerce")


def fmt_cat(count: int, denominator: int) -> str:
    return f"{count} ({100.0 * count / denominator:.1f}%)"


def load_mr_raw(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        sep=detect_delimiter(path),
        dtype=str,
        encoding="utf-8-sig",
        keep_default_na=False,
        na_filter=False,
        low_memory=False,
    )
    frame.columns = [normalize_name(column) for column in frame.columns]
    if "modality" not in frame.columns:
        raise ValueError("Raw data do not contain modality")
    mr = frame.loc[clean(frame["modality"]).str.upper().eq("MR")].copy()
    mr = mr.reset_index(drop=True)
    if len(mr) != EXPECTED_N:
        raise ValueError(f"Expected MR N={EXPECTED_N}; observed {len(mr)}")
    return mr


def reconstruct_components(mr: pd.DataFrame, window: int, tolerance_days: float):
    required = {
        "ischemia_dicho",
        "comp_cv_event_late_revasc_diff",
        "comp_cv_tte_late_revasc_5y",
        f"comp_cv_event_blind{window}_5y",
        f"comp_cv_tte_blind{window}_5y",
    }
    missing = sorted(required - set(mr.columns))
    if missing:
        raise ValueError(f"Missing component source columns: {missing}")

    event = numeric(mr[f"comp_cv_event_blind{window}_5y"])
    primary_tte = numeric(mr[f"comp_cv_tte_blind{window}_5y"])
    hierarchy = numeric(mr["comp_cv_event_late_revasc_diff"])
    hierarchy_tte = numeric(mr["comp_cv_tte_late_revasc_5y"])
    ischemia = numeric(mr["ischemia_dicho"])

    if event.isna().any() or primary_tte.isna().any():
        raise ValueError("Primary endpoint contains missing status or time")
    if int(event.eq(1).sum()) != EXPECTED_EVENTS[window]:
        raise ValueError(
            f"Expected {EXPECTED_EVENTS[window]} events for {window}-day window; "
            f"observed {int(event.eq(1).sum())}"
        )

    time_agrees = (
        (primary_tte - hierarchy_tte).abs() * DAYS_PER_YEAR <= tolerance_days
    )
    cv_death = event.eq(1) & hierarchy.eq(1) & time_agrees
    ami = event.eq(1) & hierarchy.eq(2) & time_agrees
    intervention = event.eq(1) & ~(cv_death | ami)

    component_masks = {
        "Cardiovascular death, n (%)": cv_death,
        "Acute myocardial infarction, n (%)": ami,
        "Coronary intervention (PCI/CABG), n (%)": intervention,
    }

    group_masks = {
        "Total cohort": pd.Series(True, index=mr.index),
        "No stress-induced ischemia": ischemia.eq(0),
        "Stress-induced ischemia": ischemia.eq(1),
    }

    for name, expected in EXPECTED_GROUPS.items():
        observed = int(group_masks[name].sum())
        if observed != expected:
            raise ValueError(f"Expected {name} N={expected}; observed {observed}")

    rows = []
    long_rows = []
    for label, component_mask in component_masks.items():
        formatted = {
            "section": "Follow-up and outcomes",
            "characteristic": f"    {label}",
        }
        for group_name, group_mask in group_masks.items():
            denominator = int(group_mask.sum())
            count = int((group_mask & component_mask).sum())
            formatted[group_name] = fmt_cat(count, denominator)
            long_rows.append({
                "section": "Follow-up and outcomes",
                "characteristic": f"    {label}",
                "variable": {
                    "Cardiovascular death, n (%)": f"mace_{window}d_first_cv_death",
                    "Acute myocardial infarction, n (%)": f"mace_{window}d_first_ami",
                    "Coronary intervention (PCI/CABG), n (%)": f"mace_{window}d_first_pci_cabg",
                }[label],
                "type": "categorical",
                "group": group_name,
                "group_n": denominator,
                "available_n": denominator,
                "missing_n": 0,
                "mean": np.nan,
                "sd": np.nan,
                "count": count,
                "percent": 100.0 * count / denominator,
            })
        rows.append(formatted)

    component_counts = {label: int(mask.sum()) for label, mask in component_masks.items()}
    component_total = sum(component_counts.values())
    if component_total != EXPECTED_EVENTS[window]:
        raise ValueError(
            f"Component total {component_total} does not equal composite total "
            f"{EXPECTED_EVENTS[window]}"
        )

    # Informational audit: early qualifying hard events can be blanked while a
    # later event remains eligible. This is expected for windows > 0.
    primary_after_hard = int((
        event.eq(1)
        & hierarchy.isin([1, 2])
        & (primary_tte > hierarchy_tte + tolerance_days / DAYS_PER_YEAR)
    ).sum())

    audit = {
        "window_days": window,
        "total_mace": EXPECTED_EVENTS[window],
        "cardiovascular_death": component_counts["Cardiovascular death, n (%)"],
        "acute_myocardial_infarction": component_counts["Acute myocardial infarction, n (%)"],
        "coronary_intervention_pci_cabg": component_counts[
            "Coronary intervention (PCI/CABG), n (%)"
        ],
        "components_sum": component_total,
        "later_eligible_event_after_blanked_hard_event": primary_after_hard,
        "status": "PASS",
    }
    return rows, long_rows, audit


def insert_after_characteristic(frame: pd.DataFrame, characteristic: str, new_rows: list[dict]):
    matches = frame.index[frame["characteristic"].eq(characteristic)].tolist()
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one '{characteristic}' row; found {len(matches)}"
        )
    position = frame.index.get_loc(matches[0]) + 1
    before = frame.iloc[:position]
    after = frame.iloc[position:]
    added = pd.DataFrame(new_rows).reindex(columns=frame.columns)
    return pd.concat([before, added, after], ignore_index=True)


def main():
    args = parse_args()
    outdir = args.output_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    for path in [args.base_script, args.processed_data, args.raw_data]:
        if not path.exists():
            raise FileNotFoundError(path)

    if not args.skip_base_generation:
        command = [
            sys.executable,
            str(args.base_script),
            "--input",
            str(args.processed_data),
            "--output-dir",
            str(outdir),
        ]
        print("[RUN] " + " ".join(command))
        subprocess.run(command, check=True)

    formatted_path = outdir / "table1_formatted.csv"
    numeric_path = outdir / "table1_numeric_long.csv"
    if not formatted_path.exists() or not numeric_path.exists():
        raise FileNotFoundError("Base Table 1 outputs were not generated")

    formatted = pd.read_csv(formatted_path, encoding="utf-8-sig")
    numeric_long = pd.read_csv(numeric_path, encoding="utf-8-sig")

    # Standardize the two composite row labels.
    formatted["characteristic"] = formatted["characteristic"].replace({
        "Primary MACE from index, n (%)": "Total MACE from index, n (%)",
        "Primary MACE after 30-day window, n (%)":
            "MACE after 30-day exclusion window, n (%)",
    })
    numeric_long["characteristic"] = numeric_long["characteristic"].replace({
        "Primary MACE from index, n (%)": "Total MACE from index, n (%)",
        "Primary MACE after 30-day window, n (%)":
            "MACE after 30-day exclusion window, n (%)",
    })

    mr = load_mr_raw(args.raw_data)
    component_rows, component_long_rows, audit = reconstruct_components(
        mr, args.component_window, args.tolerance_days
    )

    # Display components beneath the selected composite row. By default this is
    # Total MACE from index. Alternative windows can be generated without
    # changing the endpoint reconstruction logic.
    target_label = {
        0: "Total MACE from index, n (%)",
        30: "MACE after 30-day exclusion window, n (%)",
        60: "MACE after 60-day exclusion window, n (%)",
        90: "MACE after 90-day exclusion window, n (%)",
    }[args.component_window]

    if args.component_window in (60, 90) and target_label not in set(formatted["characteristic"]):
        # Add the requested composite row if it is not part of the standard
        # minimal Table 1. Values are generated directly from the raw endpoint.
        event = numeric(mr[f"comp_cv_event_blind{args.component_window}_5y"])
        ischemia = numeric(mr["ischemia_dicho"])
        composite_row = {
            "section": "Follow-up and outcomes",
            "characteristic": target_label,
            "Total cohort": fmt_cat(int(event.eq(1).sum()), EXPECTED_N),
            "No stress-induced ischemia": fmt_cat(
                int((event.eq(1) & ischemia.eq(0)).sum()), EXPECTED_GROUPS["No stress-induced ischemia"]
            ),
            "Stress-induced ischemia": fmt_cat(
                int((event.eq(1) & ischemia.eq(1)).sum()), EXPECTED_GROUPS["Stress-induced ischemia"]
            ),
        }
        hard_index = formatted.index[formatted["characteristic"].eq("Hard MACE, n (%)")].tolist()
        if len(hard_index) != 1:
            raise ValueError("Could not locate Hard MACE row for insertion")
        position = formatted.index.get_loc(hard_index[0])
        formatted = pd.concat([
            formatted.iloc[:position],
            pd.DataFrame([composite_row]).reindex(columns=formatted.columns),
            formatted.iloc[position:],
        ], ignore_index=True)

    formatted = insert_after_characteristic(formatted, target_label, component_rows)

    # Insert long-form component rows immediately after the selected composite's
    # group rows.
    matching = numeric_long.index[numeric_long["characteristic"].eq(target_label)].tolist()
    if matching:
        last_position = numeric_long.index.get_loc(matching[-1]) + 1
        numeric_long = pd.concat([
            numeric_long.iloc[:last_position],
            pd.DataFrame(component_long_rows).reindex(columns=numeric_long.columns),
            numeric_long.iloc[last_position:],
        ], ignore_index=True)
    else:
        numeric_long = pd.concat(
            [numeric_long, pd.DataFrame(component_long_rows).reindex(columns=numeric_long.columns)],
            ignore_index=True,
        )

    # Preserve the established comma-delimited machine outputs from the base
    # generator and additionally write Excel-friendly semicolon copies.
    formatted.to_csv(formatted_path, index=False, encoding="utf-8")
    numeric_long.to_csv(numeric_path, index=False, encoding="utf-8", float_format="%.12g")
    formatted.to_csv(
        outdir / "table1_formatted_excel.csv",
        index=False,
        sep=";",
        encoding="utf-8-sig",
    )

    pd.DataFrame([audit]).to_csv(
        outdir / "table1_mace_component_audit.csv",
        index=False,
        sep=";",
        encoding="utf-8-sig",
    )

    footnote = (
        "MACE comprised cardiovascular death, acute myocardial infarction, or "
        "coronary intervention with PCI or CABG. Component counts represent the "
        "mutually exclusive first qualifying event during follow-up. The "
        "time-to-event analysis applied a 30-day exclusion window. Hard MACE "
        "comprised cardiovascular death or acute myocardial infarction. "
        "Stress-induced ischemia status was indeterminate in 89 patients; these "
        "patients were included in the total cohort but excluded from the "
        "ischemia-stratified columns."
    )
    (outdir / "table1_title_and_footnote.txt").write_text(
        "Table 1. Baseline Characteristics and Clinical Outcomes\n\n"
        + footnote
        + "\n",
        encoding="utf-8",
    )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_script": str(args.base_script),
        "processed_data": str(args.processed_data),
        "raw_data": str(args.raw_data),
        "component_window_days": args.component_window,
        "tolerance_days": args.tolerance_days,
        "component_audit": audit,
        "privacy": "Aggregate Table 1 outputs only; no identifiers or patient rows written.",
    }
    (outdir / "table1_mace_component_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"[DONE] {formatted_path}")
    print(f"[DONE] {outdir / 'table1_formatted_excel.csv'}")
    print(
        f"[INFO] {args.component_window}d components: "
        f"CV death={audit['cardiovascular_death']}; "
        f"AMI={audit['acute_myocardial_infarction']}; "
        f"PCI/CABG={audit['coronary_intervention_pci_cabg']}; "
        f"total={audit['components_sum']}"
    )
    print("[INFO] MACE component checks: PASS")


if __name__ == "__main__":
    main()
