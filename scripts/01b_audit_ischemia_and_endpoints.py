#!/usr/bin/env python3
"""
01b_audit_ischemia_and_endpoints.py

Focused aggregate audit for:
1. Reconciliation of `ischemia` and `ischemia_dicho` in the MR cohort.
2. Identification of available endpoint/component columns in patient_v15.csv.
3. Aggregate cross-tabs of legacy event type against the validated composite
   endpoints at 0, 30, 60, and 90 days.
4. Aggregate comparison of CV and CV+UA composite variants.

No patient-level rows, identifiers, raw dates, or raw code lists are exported.

Expected layout:
    data/raw/patient_v15.csv
    scripts/01b_audit_ischemia_and_endpoints.py

Run from repository root:
    python scripts/01b_audit_ischemia_and_endpoints.py
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
DEFAULT_OUTPUT_DIR = Path("results/audit/ischemia_endpoints")
BLANK_STRINGS = {"", "nan", "none", "null", "na", "n/a", "nat", "<na>"}

WINDOWS = (0, 30, 60, 90)

# Search terms are deliberately broad because patient_v15 versions may use
# dates, event flags, composite variables, or time-to-event fields.
COMPONENT_PATTERNS = {
    "cardiovascular_death": [
        r"cv.*death", r"death.*cv", r"cardiovascular.*death",
        r"cause_of_death", r"time_to_cv_death",
    ],
    "myocardial_infarction": [
        r"(^|_)mi($|_)", r"myocardial", r"infarct", r"(^|_)ami($|_)",
    ],
    "unstable_angina": [
        r"(^|_)ua($|_)", r"unstable.*angina", r"angina",
    ],
    "pci": [r"(^|_)pci($|_)"],
    "cabg": [r"(^|_)cabg($|_)"],
    "revascularization": [r"revasc", r"coronary.*intervention"],
    "angiography": [r"angio"],
    "event_type": [r"event.*type", r"type.*event"],
    "composite": [r"composite", r"comp_cv", r"mace"],
    "follow_up_or_censoring": [r"censor", r"follow.*up", r"tte", r"time_to", r"tt_"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate audit of ischemia definitions and endpoint fields."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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
        raise ValueError(f"Normalized duplicate columns: {duplicates}")
    df.columns = normalized

    if "modality" not in df.columns:
        raise ValueError("Expected a 'modality' column.")
    mr = df.loc[clean_string(df["modality"]).str.upper().eq("MR")].copy()
    mr = mr.reset_index(drop=True)
    if mr.empty:
        raise ValueError("No MR rows found.")
    return mr, delimiter


def label_binary(value: Any) -> str:
    if pd.isna(value):
        return "Missing"
    value = float(value)
    if np.isclose(value, 0.0):
        return "0"
    if np.isclose(value, 1.0):
        return "1"
    return f"Other ({value:g})"


def aggregate_crosstab(
    left: pd.Series,
    right: pd.Series,
    left_name: str,
    right_name: str,
) -> pd.DataFrame:
    a = numeric(left).map(label_binary)
    b = numeric(right).map(label_binary)
    order = ["0", "1", "Missing"]
    extra_a = sorted(set(a.dropna()) - set(order))
    extra_b = sorted(set(b.dropna()) - set(order))
    a = pd.Categorical(a, categories=order + extra_a, ordered=True)
    b = pd.Categorical(b, categories=order + extra_b, ordered=True)
    table = pd.crosstab(a, b, dropna=False).reset_index()
    table = table.rename(columns={table.columns[0]: left_name})
    table.columns.name = None
    table.insert(0, "row_variable", left_name)
    table.insert(1, "column_variable", right_name)
    return table


def ischemia_group_summary(mr: pd.DataFrame) -> pd.DataFrame:
    if "ischemia_dicho" not in mr.columns:
        raise ValueError("Expected 'ischemia_dicho' in patient_v15.csv.")

    group = numeric(mr["ischemia_dicho"])
    rows: list[dict[str, Any]] = []

    for value, label in [(0.0, "No ischemia"), (1.0, "Ischemia"), (np.nan, "Missing")]:
        mask = group.isna() if pd.isna(value) else group.eq(value)
        row: dict[str, Any] = {
            "ischemia_group": label,
            "n": int(mask.sum()),
            "percent_of_mr_cohort": round(mask.mean() * 100.0, 3),
        }
        for col in ["ischemia", "ischemia_percent", "lvef", "infarct_dicho"]:
            if col not in mr.columns:
                continue
            values = numeric(mr.loc[mask, col])
            row[f"{col}_available_n"] = int(values.notna().sum())
            row[f"{col}_mean"] = round(float(values.mean()), 6) if values.notna().any() else np.nan
            row[f"{col}_median"] = round(float(values.median()), 6) if values.notna().any() else np.nan
            row[f"{col}_min"] = round(float(values.min()), 6) if values.notna().any() else np.nan
            row[f"{col}_max"] = round(float(values.max()), 6) if values.notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def missingness_pattern(mr: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["ischemia", "ischemia_dicho", "infarct", "infarct_dicho"] if c in mr.columns]
    if not cols:
        return pd.DataFrame()
    flags = pd.DataFrame({f"{c}_missing": numeric(mr[c]).isna() for c in cols})
    out = flags.value_counts(dropna=False).rename("n").reset_index()
    out["percent"] = (out["n"] / len(mr) * 100.0).round(3)
    return out


def identify_candidate_columns(columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for domain, patterns in COMPONENT_PATTERNS.items():
        for col in columns:
            if any(re.search(pattern, col, flags=re.IGNORECASE) for pattern in patterns):
                rows.append({"domain": domain, "column": col})
    return pd.DataFrame(rows).drop_duplicates().sort_values(["domain", "column"])


def summarize_candidate_columns(mr: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in candidates.itertuples(index=False):
        col = item.column
        raw = clean_string(mr[col])
        values = numeric(mr[col])
        numeric_share = values.notna().sum() / raw.notna().sum() if raw.notna().any() else 0.0
        row: dict[str, Any] = {
            "domain": item.domain,
            "column": col,
            "n_total": len(mr),
            "n_raw_available": int(raw.notna().sum()),
            "n_raw_missing": int(raw.isna().sum()),
            "numeric_share_of_available": round(float(numeric_share), 4),
            "n_numeric_available": int(values.notna().sum()),
            "numeric_min": round(float(values.min()), 6) if values.notna().any() else np.nan,
            "numeric_max": round(float(values.max()), 6) if values.notna().any() else np.nan,
            "numeric_mean": round(float(values.mean()), 6) if values.notna().any() else np.nan,
            "n_equal_zero": int(values.eq(0).sum()),
            "n_equal_one": int(values.eq(1).sum()),
            "n_unique_raw": int(raw.nunique(dropna=True)),
        }
        # Do not export raw categories or dates. Only aggregate metadata are written.
        rows.append(row)
    return pd.DataFrame(rows)


def endpoint_window_summary(mr: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family, prefix in [("CV", "comp_cv"), ("CV+UA", "comp_cv_ua")]:
        for window in WINDOWS:
            event_col = f"{prefix}_event_blind{window}_5y"
            time_col = f"{prefix}_tte_blind{window}_5y"
            if event_col not in mr.columns and time_col not in mr.columns:
                continue
            event = numeric(mr[event_col]) if event_col in mr.columns else pd.Series(np.nan, index=mr.index)
            time = numeric(mr[time_col]) if time_col in mr.columns else pd.Series(np.nan, index=mr.index)
            event_times = time[event.eq(1)]
            nonevent_times = time[event.eq(0)]
            rows.append({
                "family": family,
                "window_days": window,
                "event_column": event_col,
                "time_column": time_col,
                "n": len(mr),
                "n_event_status_available": int(event.notna().sum()),
                "events": int(event.eq(1).sum()),
                "event_percent": round(float(event.eq(1).sum() / event.notna().sum() * 100.0), 3)
                    if event.notna().any() else np.nan,
                "n_time_available": int(time.notna().sum()),
                "time_min_all": round(float(time.min()), 6) if time.notna().any() else np.nan,
                "time_max_all": round(float(time.max()), 6) if time.notna().any() else np.nan,
                "time_min_events": round(float(event_times.min()), 6) if event_times.notna().any() else np.nan,
                "time_max_events": round(float(event_times.max()), 6) if event_times.notna().any() else np.nan,
                "time_min_nonevents": round(float(nonevent_times.min()), 6) if nonevent_times.notna().any() else np.nan,
                "time_max_nonevents": round(float(nonevent_times.max()), 6) if nonevent_times.notna().any() else np.nan,
            })
    return pd.DataFrame(rows)


def compare_cv_vs_cvua(mr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, Any]] = []
    cross_tabs: list[pd.DataFrame] = []
    for window in WINDOWS:
        cv_col = f"comp_cv_event_blind{window}_5y"
        ua_col = f"comp_cv_ua_event_blind{window}_5y"
        if cv_col not in mr.columns or ua_col not in mr.columns:
            continue
        cv = numeric(mr[cv_col])
        ua = numeric(mr[ua_col])
        both = cv.notna() & ua.notna()
        summaries.append({
            "window_days": window,
            "n_with_both": int(both.sum()),
            "cv_events": int(cv.eq(1).sum()),
            "cv_plus_ua_events": int(ua.eq(1).sum()),
            "additional_events_in_cv_plus_ua": int((cv.eq(0) & ua.eq(1)).sum()),
            "events_in_cv_but_not_cv_plus_ua": int((cv.eq(1) & ua.eq(0)).sum()),
            "disagreements_total": int((both & cv.ne(ua)).sum()),
        })
        tab = aggregate_crosstab(cv, ua, cv_col, ua_col)
        tab.insert(0, "window_days", window)
        cross_tabs.append(tab)
    return pd.DataFrame(summaries), pd.concat(cross_tabs, ignore_index=True) if cross_tabs else pd.DataFrame()


def event_type_tables(mr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "type_of_event" not in mr.columns:
        return pd.DataFrame(), pd.DataFrame()

    event_type = numeric(mr["type_of_event"])
    counts = (
        event_type.fillna(-999999)
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("type_of_event_numeric")
        .rename("n")
        .reset_index()
    )
    counts["type_of_event"] = counts["type_of_event_numeric"].map(
        lambda x: "Missing" if x == -999999 else f"{x:g}"
    )
    counts = counts[["type_of_event", "n"]]
    counts["percent_of_mr_cohort"] = (counts["n"] / len(mr) * 100.0).round(3)

    tabs: list[pd.DataFrame] = []
    type_label = event_type.map(lambda x: "Missing" if pd.isna(x) else f"{x:g}")
    for family, prefix in [("CV", "comp_cv"), ("CV+UA", "comp_cv_ua")]:
        for window in WINDOWS:
            event_col = f"{prefix}_event_blind{window}_5y"
            if event_col not in mr.columns:
                continue
            event_label = numeric(mr[event_col]).map(label_binary)
            table = pd.crosstab(type_label, event_label, dropna=False).reset_index()
            table = table.rename(columns={table.columns[0]: "type_of_event"})
            table.columns.name = None
            table.insert(0, "family", family)
            table.insert(1, "window_days", window)
            table.insert(2, "event_column", event_col)
            tabs.append(table)
    return counts, pd.concat(tabs, ignore_index=True) if tabs else pd.DataFrame()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
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

    required = ["ischemia", "ischemia_dicho"]
    missing_required = [c for c in required if c not in mr.columns]
    if missing_required:
        raise ValueError(f"Missing required ischemia columns: {missing_required}")

    ischemia_cross = aggregate_crosstab(
        mr["ischemia"], mr["ischemia_dicho"], "ischemia", "ischemia_dicho"
    )
    ischemia_summary = ischemia_group_summary(mr)
    ischemia_missing = missingness_pattern(mr)

    candidates = identify_candidate_columns(list(mr.columns))
    candidate_summary = summarize_candidate_columns(mr, candidates)
    window_summary = endpoint_window_summary(mr)
    cvua_summary, cvua_cross = compare_cv_vs_cvua(mr)
    event_type_counts, event_type_cross = event_type_tables(mr)

    write_csv(ischemia_cross, output_dir / "ischemia_vs_ischemia_dicho_crosstab.csv")
    write_csv(ischemia_summary, output_dir / "ischemia_dicho_group_summary.csv")
    write_csv(ischemia_missing, output_dir / "ischemia_infarction_missingness_patterns.csv")
    write_csv(candidates, output_dir / "endpoint_component_candidate_columns.csv")
    write_csv(candidate_summary, output_dir / "endpoint_component_candidate_summary.csv")
    write_csv(window_summary, output_dir / "endpoint_window_summary.csv")
    write_csv(cvua_summary, output_dir / "cv_vs_cvua_summary.csv")
    write_csv(cvua_cross, output_dir / "cv_vs_cvua_crosstabs.csv")
    write_csv(event_type_counts, output_dir / "type_of_event_counts.csv")
    write_csv(event_type_cross, output_dir / "type_of_event_by_composite.csv")

    dicho = numeric(mr["ischemia_dicho"])
    raw = numeric(mr["ischemia"])
    overlap = dicho.notna() & raw.notna()
    disagreement = overlap & dicho.ne(raw)

    summary = {
        "script_version": SCRIPT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(args.input),
        "mr_n": len(mr),
        "ischemia_dicho": {
            "zero": int(dicho.eq(0).sum()),
            "one": int(dicho.eq(1).sum()),
            "missing": int(dicho.isna().sum()),
            "invalid_nonbinary": int((dicho.notna() & ~dicho.isin([0, 1])).sum()),
        },
        "ischemia_raw": {
            "zero": int(raw.eq(0).sum()),
            "one": int(raw.eq(1).sum()),
            "missing_or_non_numeric": int(raw.isna().sum()),
            "invalid_nonbinary": int((raw.notna() & ~raw.isin([0, 1])).sum()),
        },
        "comparison": {
            "both_available": int(overlap.sum()),
            "disagreements": int(disagreement.sum()),
            "raw_0_dicho_1": int((raw.eq(0) & dicho.eq(1)).sum()),
            "raw_1_dicho_0": int((raw.eq(1) & dicho.eq(0)).sum()),
            "raw_missing_dicho_available": int((raw.isna() & dicho.notna()).sum()),
            "raw_available_dicho_missing": int((raw.notna() & dicho.isna()).sum()),
            "both_missing": int((raw.isna() & dicho.isna()).sum()),
        },
        "files_written": sorted(p.name for p in output_dir.glob("*.csv")),
        "privacy": "Aggregate outputs only; no lopnr, patient rows, raw dates, or raw code lists exported.",
    }
    with (output_dir / "focused_audit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    text = [
        "Focused ischemia and endpoint audit",
        "=" * 37,
        f"MR cohort N: {len(mr):,}",
        "",
        "ischemia_dicho:",
        f"- 0: {summary['ischemia_dicho']['zero']:,}",
        f"- 1: {summary['ischemia_dicho']['one']:,}",
        f"- missing: {summary['ischemia_dicho']['missing']:,}",
        f"- invalid nonbinary: {summary['ischemia_dicho']['invalid_nonbinary']:,}",
        "",
        "ischemia vs ischemia_dicho:",
        f"- both available: {summary['comparison']['both_available']:,}",
        f"- disagreements: {summary['comparison']['disagreements']:,}",
        f"- raw 0 / dicho 1: {summary['comparison']['raw_0_dicho_1']:,}",
        f"- raw 1 / dicho 0: {summary['comparison']['raw_1_dicho_0']:,}",
        f"- raw missing / dicho available: {summary['comparison']['raw_missing_dicho_available']:,}",
        f"- raw available / dicho missing: {summary['comparison']['raw_available_dicho_missing']:,}",
        "",
        "Review these files first:",
        "- ischemia_vs_ischemia_dicho_crosstab.csv",
        "- ischemia_dicho_group_summary.csv",
        "- endpoint_component_candidate_columns.csv",
        "- endpoint_component_candidate_summary.csv",
        "- cv_vs_cvua_summary.csv",
        "- type_of_event_counts.csv",
        "",
        "No patient-level data were exported.",
    ]
    (output_dir / "focused_audit_summary.txt").write_text("\n".join(text) + "\n", encoding="utf-8")

    print(f"[INFO] ischemia_dicho: 0={summary['ischemia_dicho']['zero']}, "
          f"1={summary['ischemia_dicho']['one']}, missing={summary['ischemia_dicho']['missing']}")
    print(f"[INFO] Ischemia disagreements: {summary['comparison']['disagreements']}")
    print(f"[INFO] Outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
