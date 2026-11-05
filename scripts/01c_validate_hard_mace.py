#!/usr/bin/env python3
"""
01c_validate_hard_mace.py

Small, read-only aggregate validation of the proposed hard-MACE derivation from
patient_v15.csv.

Proposed derivation to validate:
- comp_cv_event_late_revasc_diff == 1: cardiovascular death
- comp_cv_event_late_revasc_diff == 2: acute myocardial infarction
- comp_cv_event_late_revasc_diff == 3: late PCI/CABG
- comp_cv_event_late_revasc_diff == 0: no qualifying event

Hard MACE is cardiovascular death or AMI, therefore labels 1 or 2 are events.
Patients labelled 0 or 3 are hard-MACE non-events and must be followed to the
administrative censoring date rather than censored at revascularization.

No patient-level rows, identifiers, or dates are exported.

Run from repository root:
    python scripts/01c_validate_hard_mace.py

Outputs:
    results/audit/hard_mace_validation/
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = "1.0.0"
DEFAULT_INPUT = Path("data/raw/patient_v15.csv")
DEFAULT_OUTPUT_DIR = Path("results/audit/hard_mace_validation")
BLANK_STRINGS = {"", "nan", "none", "null", "na", "n/a", "nat", "<na>"}

LABEL_COL = "comp_cv_event_late_revasc_diff"
COMPOSITE_EVENT_COL = "comp_cv_event_late_revasc_5y"
COMPOSITE_TTE_COL = "comp_cv_tte_late_revasc_5y"
INDEX_DATE_COL = "index_exam_date"
ADMIN_DATE_COL = "admin_censor_date"
CV_DEATH_TTE_COL = "time_to_cv_death"
DAYS_PER_YEAR = 365.25

LABEL_NAMES = {
    0: "No qualifying composite event",
    1: "Cardiovascular death",
    2: "Acute myocardial infarction",
    3: "Late PCI/CABG",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate validation of hard-MACE derivation from patient_v15.csv."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-mr-n", type=int, default=2159)
    parser.add_argument(
        "--tolerance-days", type=float, default=1.0,
        help="Maximum allowed difference when comparing two TTE values."
    )
    return parser.parse_args()


def normalize_name(value: str) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[ .\-/]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        sample = handle.read(131072)
    try:
        return csv.Sniffer().sniff(sample, delimiters=";,\t|").delimiter
    except csv.Error:
        first = sample.splitlines()[0] if sample else ""
        return max([";", ",", "\t", "|"], key=first.count)


def clean_string(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip()
    return out.mask(out.str.lower().isin(BLANK_STRINGS))


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        clean_string(series).str.replace(",", ".", regex=False),
        errors="coerce",
    )


def parse_date(series: pd.Series) -> tuple[pd.Series, str]:
    """Parse ISO/text dates or numeric seconds since 1601 used in patient_v15."""
    s = clean_string(series)
    nonmissing = s.notna()
    if not nonmissing.any():
        return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]"), "all_missing"

    nums = pd.to_numeric(s, errors="coerce")
    numeric_share = nums.notna().sum() / nonmissing.sum()
    if numeric_share >= 0.90:
        finite = nums[np.isfinite(nums)]
        magnitude = float(finite.abs().median()) if len(finite) else np.nan
        if 5e9 <= magnitude < 5e10:
            parsed = pd.Timestamp("1601-01-01") + pd.to_timedelta(nums, unit="s", errors="coerce")
            return parsed, "seconds_since_1601-01-01"
        if 1e9 <= magnitude < 5e9:
            return pd.to_datetime(nums, unit="s", origin="unix", errors="coerce"), "unix_seconds"
        if 1e12 <= magnitude < 5e13:
            return pd.to_datetime(nums, unit="ms", origin="unix", errors="coerce"), "unix_milliseconds"
        if 1e7 <= magnitude < 1e9:
            return pd.to_datetime(s, format="%Y%m%d", errors="coerce"), "yyyymmdd"

    return pd.to_datetime(s, format="mixed", errors="coerce"), "text_mixed"


def load_mr(path: Path) -> tuple[pd.DataFrame, str]:
    delimiter = detect_delimiter(path)
    df = pd.read_csv(
        path,
        sep=delimiter,
        dtype=str,
        encoding="utf-8-sig",
        quotechar='"',
        low_memory=False,
        keep_default_na=False,
        na_filter=False,
    )
    normalized = [normalize_name(c) for c in df.columns]
    duplicates = sorted({c for c in normalized if normalized.count(c) > 1})
    if duplicates:
        raise ValueError(f"Column normalization created duplicates: {duplicates}")
    df.columns = normalized
    if "modality" not in df.columns:
        raise ValueError("Expected a 'modality' column.")
    mr = df.loc[clean_string(df["modality"]).str.upper().eq("MR")].copy().reset_index(drop=True)
    if mr.empty:
        raise ValueError("No MR rows found.")
    return mr, delimiter


def safe_float(value: Any, digits: int = 6) -> float | None:
    if value is None or pd.isna(value) or not np.isfinite(value):
        return None
    return round(float(value), digits)


def describe(series: pd.Series, prefix: str) -> dict[str, Any]:
    s = pd.to_numeric(series, errors="coerce")
    x = s.dropna()
    return {
        f"{prefix}_n_available": int(x.size),
        f"{prefix}_n_missing": int(s.isna().sum()),
        f"{prefix}_min": safe_float(x.min()) if len(x) else None,
        f"{prefix}_q1": safe_float(x.quantile(0.25)) if len(x) else None,
        f"{prefix}_median": safe_float(x.median()) if len(x) else None,
        f"{prefix}_q3": safe_float(x.quantile(0.75)) if len(x) else None,
        f"{prefix}_max": safe_float(x.max()) if len(x) else None,
        f"{prefix}_mean": safe_float(x.mean()) if len(x) else None,
    }


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8", float_format="%.6g")


def main() -> int:
    args = parse_args()
    source = args.input.resolve()
    output_dir = args.output_dir.resolve()

    if not source.exists():
        print(f"[ERROR] File not found: {source}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Script version: {SCRIPT_VERSION}")
    print(f"[INFO] Reading: {source}")
    mr, delimiter = load_mr(source)
    print(f"[INFO] MR cohort: N={len(mr):,}; delimiter={delimiter!r}")

    required = [
        LABEL_COL, COMPOSITE_EVENT_COL, COMPOSITE_TTE_COL,
        INDEX_DATE_COL, ADMIN_DATE_COL,
    ]
    missing = [c for c in required if c not in mr.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    label = numeric(mr[LABEL_COL])
    composite_event = numeric(mr[COMPOSITE_EVENT_COL])
    composite_tte = numeric(mr[COMPOSITE_TTE_COL])
    cv_death_tte = (
        numeric(mr[CV_DEATH_TTE_COL])
        if CV_DEATH_TTE_COL in mr.columns
        else pd.Series(np.nan, index=mr.index, dtype=float)
    )

    index_date, index_method = parse_date(mr[INDEX_DATE_COL])
    admin_date, admin_method = parse_date(mr[ADMIN_DATE_COL])
    admin_followup = (admin_date - index_date).dt.total_seconds() / (DAYS_PER_YEAR * 24 * 60 * 60)

    valid_labels = label.isin([0, 1, 2, 3])
    hard_event = label.isin([1, 2])
    hard_tte = pd.Series(
        np.where(hard_event, composite_tte, admin_followup),
        index=mr.index,
        dtype=float,
    )

    # Aggregate distribution and time summaries by hierarchy label.
    label_rows: list[dict[str, Any]] = []
    for value in [0, 1, 2, 3]:
        mask = label.eq(value)
        row = {
            "label_value": value,
            "label_name": LABEL_NAMES[value],
            "n": int(mask.sum()),
            "percent_of_mr_cohort": round(float(mask.mean() * 100.0), 3),
            "composite_event_ones": int((mask & composite_event.eq(1)).sum()),
            "composite_event_zeros": int((mask & composite_event.eq(0)).sum()),
            "hard_mace_event_ones": int((mask & hard_event).sum()),
        }
        row.update(describe(composite_tte[mask], "composite_tte_years"))
        row.update(describe(admin_followup[mask], "admin_followup_years"))
        row.update(describe(hard_tte[mask], "derived_hard_tte_years"))
        label_rows.append(row)

    unknown_mask = label.notna() & ~valid_labels
    if unknown_mask.any():
        row = {
            "label_value": "OTHER",
            "label_name": "Unexpected label",
            "n": int(unknown_mask.sum()),
            "percent_of_mr_cohort": round(float(unknown_mask.mean() * 100.0), 3),
            "composite_event_ones": int((unknown_mask & composite_event.eq(1)).sum()),
            "composite_event_zeros": int((unknown_mask & composite_event.eq(0)).sum()),
            "hard_mace_event_ones": 0,
        }
        row.update(describe(composite_tte[unknown_mask], "composite_tte_years"))
        row.update(describe(admin_followup[unknown_mask], "admin_followup_years"))
        row.update(describe(hard_tte[unknown_mask], "derived_hard_tte_years"))
        label_rows.append(row)

    distribution = pd.DataFrame(label_rows)

    # Consistency checks.
    tolerance_years = args.tolerance_days / DAYS_PER_YEAR
    cv_mask = label.eq(1)
    cv_both = cv_mask & composite_tte.notna() & cv_death_tte.notna()
    cv_abs_diff = (composite_tte[cv_both] - cv_death_tte[cv_both]).abs()

    late_mask = label.eq(3)
    late_both = late_mask & composite_tte.notna() & admin_followup.notna()
    late_before_admin = composite_tte[late_both] <= admin_followup[late_both] + tolerance_years

    hard_mask = label.isin([1, 2])
    nonhard_mask = label.isin([0, 3])

    checks: list[dict[str, Any]] = []

    def add_check(name: str, passed: bool, observed: Any, expected: Any, detail: str) -> None:
        checks.append({
            "check": name,
            "status": "PASS" if passed else "REVIEW",
            "observed": observed,
            "expected": expected,
            "detail": detail,
        })

    add_check("MR cohort size", len(mr) == args.expected_mr_n, len(mr), args.expected_mr_n,
              "MR-only rows in patient_v15.csv.")
    add_check("Hierarchy labels complete", label.notna().all(), int(label.isna().sum()), 0,
              f"Missing values in {LABEL_COL}.")
    add_check("Hierarchy labels valid", bool((label.isna() | valid_labels).all()), int(unknown_mask.sum()), 0,
              "Expected only labels 0, 1, 2, and 3.")
    add_check("Composite event complete", composite_event.notna().all(), int(composite_event.isna().sum()), 0,
              f"Missing values in {COMPOSITE_EVENT_COL}.")
    add_check("Composite TTE complete", composite_tte.notna().all(), int(composite_tte.isna().sum()), 0,
              f"Missing values in {COMPOSITE_TTE_COL}.")
    add_check("Index dates parsed", index_date.notna().all(), int(index_date.isna().sum()), 0,
              f"Parse method: {index_method}.")
    add_check("Administrative dates parsed", admin_date.notna().all(), int(admin_date.isna().sum()), 0,
              f"Parse method: {admin_method}.")
    add_check("Administrative follow-up positive", bool((admin_followup.dropna() > 0).all()),
              int((admin_followup.notna() & (admin_followup <= 0)).sum()), 0,
              "Administrative censoring must occur after the index MR.")
    add_check("Labels 1-3 imply composite event", bool(composite_event[label.isin([1, 2, 3])].eq(1).all()),
              int((label.isin([1, 2, 3]) & ~composite_event.eq(1)).sum()), 0,
              "CV death, AMI, and late revascularization labels should be composite events.")
    add_check("Label 0 implies no composite event", bool(composite_event[label.eq(0)].eq(0).all()),
              int((label.eq(0) & ~composite_event.eq(0)).sum()), 0,
              "No-event labels should have composite event=0.")
    add_check("Hard events have composite TTE", composite_tte[hard_mask].notna().all(),
              int(composite_tte[hard_mask].isna().sum()), 0,
              "Labels 1 and 2 require a hard-event time.")
    add_check("Non-hard events have administrative follow-up", admin_followup[nonhard_mask].notna().all(),
              int(admin_followup[nonhard_mask].isna().sum()), 0,
              "Labels 0 and 3 require administrative censoring for hard MACE.")
    add_check("Late revascularization occurs by administrative censoring", bool(late_before_admin.all()),
              int((~late_before_admin).sum()), 0,
              f"Compared with tolerance of {args.tolerance_days:g} day(s).")

    if cv_mask.any():
        add_check("CV-death TTE available for label 1", cv_death_tte[cv_mask].notna().all(),
                  int(cv_death_tte[cv_mask].isna().sum()), 0,
                  "Each label-1 patient should have time_to_cv_death.")
        add_check("CV-death TTE agrees with composite TTE", bool((cv_abs_diff <= tolerance_years).all()),
                  int((cv_abs_diff > tolerance_years).sum()), 0,
                  f"Agreement tolerance: {args.tolerance_days:g} day(s); overlapping N={int(cv_both.sum())}.")
    else:
        add_check("CV-death labels present", False, 0, ">0", "No label-1 events were found.")

    checks_df = pd.DataFrame(checks)

    # Proposed hard-MACE endpoint aggregate summary.
    hard_summary = pd.DataFrame([{
        "endpoint": "Hard MACE: cardiovascular death or AMI",
        "n": len(mr),
        "events": int(hard_event.sum()),
        "event_percent": round(float(hard_event.mean() * 100.0), 3),
        "non_events": int((~hard_event).sum()),
        "cv_death_events_label_1": int(label.eq(1).sum()),
        "ami_events_label_2": int(label.eq(2).sum()),
        "late_revascularization_only_label_3": int(label.eq(3).sum()),
        "no_composite_event_label_0": int(label.eq(0).sum()),
        **describe(hard_tte, "hard_tte_years_all"),
        **describe(hard_tte[hard_event], "hard_tte_years_events"),
        **describe(hard_tte[~hard_event], "hard_tte_years_nonevents"),
        "hard_events_missing_tte": int((hard_event & hard_tte.isna()).sum()),
        "hard_nonevents_missing_tte": int((~hard_event & hard_tte.isna()).sum()),
        "nonpositive_hard_tte": int((hard_tte.notna() & (hard_tte <= 0)).sum()),
    }])

    # Difference between late-revascularization TTE and admin follow-up by label.
    delta_rows: list[dict[str, Any]] = []
    for value in [0, 1, 2, 3]:
        mask = label.eq(value) & composite_tte.notna() & admin_followup.notna()
        delta = admin_followup[mask] - composite_tte[mask]
        delta_rows.append({
            "label_value": value,
            "label_name": LABEL_NAMES[value],
            "n_compared": int(mask.sum()),
            **describe(delta, "admin_minus_composite_tte_years"),
            "n_composite_tte_before_admin": int((delta > tolerance_years).sum()),
            "n_approximately_equal": int((delta.abs() <= tolerance_years).sum()),
            "n_composite_tte_after_admin": int((delta < -tolerance_years).sum()),
        })
    delta_df = pd.DataFrame(delta_rows)

    write_csv(distribution, output_dir / "hard_mace_label_distribution_and_times.csv")
    write_csv(checks_df, output_dir / "hard_mace_validation_checks.csv")
    write_csv(hard_summary, output_dir / "proposed_hard_mace_summary.csv")
    write_csv(delta_df, output_dir / "composite_tte_vs_admin_by_label.csv")

    pass_count = int(checks_df["status"].eq("PASS").sum())
    review_count = int(checks_df["status"].eq("REVIEW").sum())

    summary = {
        "script_version": SCRIPT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(args.input),
        "mr_n": len(mr),
        "date_parsing": {
            INDEX_DATE_COL: index_method,
            ADMIN_DATE_COL: admin_method,
        },
        "label_counts": {
            str(value): int(label.eq(value).sum()) for value in [0, 1, 2, 3]
        },
        "proposed_hard_mace": {
            "definition": "comp_cv_event_late_revasc_diff in {1, 2}",
            "events": int(hard_event.sum()),
            "non_events": int((~hard_event).sum()),
            "event_percent": round(float(hard_event.mean() * 100.0), 3),
            "event_time": COMPOSITE_TTE_COL,
            "non_event_time": f"({ADMIN_DATE_COL} - {INDEX_DATE_COL}) / 365.25",
        },
        "checks": {"pass": pass_count, "review": review_count},
        "privacy": "Aggregate outputs only; no identifiers, patient rows, or raw dates exported.",
    }
    with (output_dir / "hard_mace_validation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    text = [
        "Hard-MACE validation summary",
        "=" * 29,
        f"MR cohort N: {len(mr):,}",
        "",
        "Hierarchy-label counts:",
        f"- 0, no qualifying composite event: {int(label.eq(0).sum()):,}",
        f"- 1, cardiovascular death: {int(label.eq(1).sum()):,}",
        f"- 2, acute myocardial infarction: {int(label.eq(2).sum()):,}",
        f"- 3, late PCI/CABG: {int(label.eq(3).sum()):,}",
        "",
        "Proposed hard MACE:",
        "- Definition: labels 1 or 2",
        f"- Events: {int(hard_event.sum()):,} ({hard_event.mean() * 100.0:.1f}%)",
        f"- Non-events: {int((~hard_event).sum()):,}",
        f"- Event TTE source: {COMPOSITE_TTE_COL}",
        f"- Non-event TTE source: administrative censoring minus index date",
        "",
        f"Validation checks: {pass_count} PASS; {review_count} REVIEW",
        "Review hard_mace_validation_checks.csv before using the endpoint.",
        "",
        "No patient-level data were exported.",
    ]
    (output_dir / "hard_mace_validation_summary.txt").write_text("\n".join(text) + "\n", encoding="utf-8")

    print(f"[INFO] Label counts: 0={int(label.eq(0).sum())}, 1={int(label.eq(1).sum())}, "
          f"2={int(label.eq(2).sum())}, 3={int(label.eq(3).sum())}")
    print(f"[INFO] Proposed hard-MACE events: {int(hard_event.sum())} ({hard_event.mean() * 100.0:.1f}%)")
    print(f"[INFO] Validation checks: {pass_count} PASS; {review_count} REVIEW")
    print(f"[INFO] Outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
