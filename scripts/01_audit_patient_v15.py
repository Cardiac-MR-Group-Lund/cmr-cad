#!/usr/bin/env python3
"""
01_audit_patient_v15.py

Read-only, aggregate audit of patient_v15.csv for the JACC: Advances revision.

The script:
- never modifies the source file;
- never writes patient-level rows or identifiers;
- audits cohort composition, MR filtering, duplicates, dates, predictors,
  registry code lists, follow-up variables, endpoint pairs, and component events;
- writes only aggregate outputs to results/audit/.

Expected repository layout:
    ./data/raw/patient_v15.csv
    ./scripts/01_audit_patient_v15.py

Run from repository root:
    python scripts/01_audit_patient_v15.py

Optional:
    python scripts/01_audit_patient_v15.py \
        --input data/raw/patient_v15.csv \
        --output-dir results/audit \
        --min-code-count 10

Security:
- Keep data/raw/ and results/audit/ outside public version control until every
  generated file has been reviewed.
- The audit reports hashes and aggregate counts only. It does not export lopnr,
  dates at patient level, raw code lists, or sample rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SCRIPT_VERSION = "1.0.0"
DEFAULT_INPUT = Path("data/raw/patient_v15.csv")
DEFAULT_OUTPUT_DIR = Path("results/audit")

# Expected source variables. Missing variables are reported, not silently created.
PREDICTOR_SPECS: dict[str, list[str]] = {
    "age": ["age", "age_at_cmr"],
    "sex": ["sex_num", "sex_numerical", "sex"],
    "height_cm": ["length_cm", "height_cm", "length"],
    "weight_kg": ["weight_kg", "weight"],
    "bsa_m2": ["bsa_m2", "bsa"],
    "bmi": ["bmi", "bmi_at_cmr"],
    "lvef": ["lvef", "lvef_mps"],
    "lvedv_ml": ["lvedv", "lvedv_ml"],
    "lvesv_ml": ["lvesv", "lvesv_ml"],
    "lv_mass_g": ["lv_mass", "lv_mass_g"],
    "infarction": ["infarct_dicho", "mr_infarct", "infarct"],
    "stress_induced_ischemia": ["ischemia_dicho", "mr_ischemia", "ischemia"],
    "ischemia_percent": ["ischemia_percent", "ischemi_percent"],
}

PLAUSIBILITY_RANGES: dict[str, tuple[float, float]] = {
    "age": (18.0, 110.0),
    "height_cm": (100.0, 230.0),
    "weight_kg": (25.0, 300.0),
    "bsa_m2": (0.8, 3.5),
    "bmi": (10.0, 80.0),
    "lvef": (0.0, 100.0),
    "lvedv_ml": (10.0, 800.0),
    "lvesv_ml": (0.0, 700.0),
    "lv_mass_g": (10.0, 600.0),
    "ischemia_percent": (0.0, 100.0),
}

DATE_CANDIDATES = [
    "index_exam_date", "cmr_index_date", "date_of_birth", "death_date",
    "mi_date", "ua_date", "pci_date", "cabg_date", "angio_date",
    "angina_date", "cvd_date", "admin_censor_date",
    "visit_first_admission_date", "visit_last_discharge_date",
]

# Known event/time pairs in patient_v15. Additional pairs are discovered by pattern.
KNOWN_EVENT_TIME_PAIRS: list[tuple[str, str, str]] = [
    ("event_5years", "tte_5years", "Submitted/legacy available-follow-up endpoint"),
    ("event_years", "tte_years", "Open-ended/legacy endpoint"),
    ("comp_cv_event_blind0_5y", "comp_cv_tte_blind0_5y", "Composite CV, 0-day window"),
    ("comp_cv_event_blind30_5y", "comp_cv_tte_blind30_5y", "Composite CV, 30-day window"),
    ("comp_cv_event_blind60_5y", "comp_cv_tte_blind60_5y", "Composite CV, 60-day window"),
    ("comp_cv_event_blind90_5y", "comp_cv_tte_blind90_5y", "Composite CV, 90-day window"),
    ("comp_cv_ua_event_blind0_5y", "comp_cv_ua_tte_blind0_5y", "Composite CV+UA, 0-day window"),
    ("comp_cv_ua_event_blind30_5y", "comp_cv_ua_tte_blind30_5y", "Composite CV+UA, 30-day window"),
    ("comp_cv_ua_event_blind60_5y", "comp_cv_ua_tte_blind60_5y", "Composite CV+UA, 60-day window"),
    ("comp_cv_ua_event_blind90_5y", "comp_cv_ua_tte_blind90_5y", "Composite CV+UA, 90-day window"),
    ("comp_cv_event_late_revasc_5y", "comp_cv_tte_late_revasc_5y", "Late-revascularization composite"),
    ("composite_event_0", "tte_0", "Alternative composite, 0-month/day label"),
    ("composite_event_6", "tte_6", "Alternative composite, 6-month label"),
    ("composite_event_9", "tte_9", "Alternative composite, 9-month label"),
    ("composite_event_12", "tte_12", "Alternative composite, 12-month label"),
]

CODE_COLUMNS = {
    "pre_icd": "pre_icd_codes",
    "pre_kva": "pre_kva_codes",
    "post_icd": "post_icd_codes",
    "post_kva": "post_kva_codes",
}

BLANK_STRINGS = {"", "nan", "none", "null", "na", "n/a", "nat", "<na>"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an aggregate, non-patient-level audit of patient_v15.csv."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-code-count", type=int, default=10)
    parser.add_argument(
        "--expected-mr-n", type=int, default=2159,
        help="Expected unique MR cohort size; used only as an audit check."
    )
    parser.add_argument(
        "--expected-index-start", default="2019-01-01",
        help="Expected earliest eligible MR index date."
    )
    parser.add_argument(
        "--expected-index-end", default="2022-12-31",
        help="Expected latest eligible MR index date."
    )
    return parser.parse_args()


def normalize_name(value: str) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[ .\-/]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


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
        dialect = csv.Sniffer().sniff(sample, delimiters=";,\t|")
        return dialect.delimiter
    except csv.Error:
        first_line = sample.splitlines()[0] if sample else ""
        counts = {sep: first_line.count(sep) for sep in [";", ",", "\t", "|"]}
        return max(counts, key=counts.get)


def load_source(path: Path) -> tuple[pd.DataFrame, str, list[str]]:
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
    original_columns = [str(c) for c in df.columns]
    normalized = [normalize_name(c) for c in original_columns]
    duplicates = sorted({c for c in normalized if normalized.count(c) > 1})
    if duplicates:
        raise ValueError(f"Column-name normalization created duplicates: {duplicates}")
    df.columns = normalized
    return df, delimiter, original_columns


def clean_string(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip()
    return out.mask(out.str.lower().isin(BLANK_STRINGS))


def to_numeric(series: pd.Series) -> pd.Series:
    s = clean_string(series).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def parse_datetime_multi(series: pd.Series) -> tuple[pd.Series, str]:
    """Parse ISO/text dates and large Unix-like numeric timestamps."""
    s = clean_string(series)
    nonmissing = s.notna()
    if not nonmissing.any():
        return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]"), "all_missing"

    numeric = pd.to_numeric(s, errors="coerce")
    numeric_share = float(numeric.notna().sum() / nonmissing.sum())

    if numeric_share >= 0.90:
        finite = numeric[np.isfinite(numeric)]
        magnitude = float(finite.abs().median()) if len(finite) else math.nan
        # patient_v15 examples around 1.3e10 are consistent with seconds from 1601-01-01.
        if 5e9 <= magnitude < 5e10:
            base = pd.Timestamp("1601-01-01")
            parsed = base + pd.to_timedelta(numeric, unit="s", errors="coerce")
            return parsed, "seconds_since_1601-01-01"
        if 1e9 <= magnitude < 5e9:
            return pd.to_datetime(numeric, unit="s", origin="unix", errors="coerce"), "unix_seconds"
        if 1e12 <= magnitude < 5e13:
            return pd.to_datetime(numeric, unit="ms", origin="unix", errors="coerce"), "unix_milliseconds"
        if 1e7 <= magnitude < 1e9:
            parsed = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
            return parsed, "yyyymmdd_numeric"

    parsed = pd.to_datetime(s, errors="coerce", format="mixed", dayfirst=False)
    return parsed, "text_mixed"


def first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    available = set(columns)
    return next((name for name in candidates if name in available), None)


def safe_float(value: Any, digits: int | None = 6) -> float | None:
    if value is None or pd.isna(value) or not np.isfinite(value):
        return None
    value = float(value)
    return round(value, digits) if digits is not None else value


def safe_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def value_counts_table(series: pd.Series, variable: str) -> pd.DataFrame:
    cleaned = clean_string(series).fillna("<MISSING>")
    counts = cleaned.value_counts(dropna=False)
    total = len(series)
    return pd.DataFrame({
        "variable": variable,
        "value": counts.index.astype(str),
        "n": counts.values.astype(int),
        "percent": np.round(counts.values / total * 100.0, 3) if total else np.nan,
    })


def normalize_modality(series: pd.Series) -> pd.Series:
    return clean_string(series).str.upper()


def choose_mr_mask(df: pd.DataFrame) -> tuple[pd.Series, str]:
    if "modality" in df.columns:
        values = normalize_modality(df["modality"])
        return values.eq("MR"), "modality == 'MR'"
    if "modality_num" in df.columns:
        values = normalize_modality(df["modality_num"])
        # Do not guess silently; report the conventional legacy mapping only.
        return values.isin({"MR", "1"}), "modality_num in {'MR', '1'}"
    raise ValueError("Neither 'modality' nor 'modality_num' exists; cannot define MR cohort.")


def binary_audit(series: pd.Series) -> dict[str, Any]:
    numeric = to_numeric(series)
    observed = numeric.dropna()
    invalid_mask = numeric.notna() & ~numeric.isin([0, 1])
    return {
        "n_total": len(series),
        "n_observed_numeric": int(numeric.notna().sum()),
        "n_missing_or_unparseable": int(numeric.isna().sum()),
        "n_zero": int(numeric.eq(0).sum()),
        "n_one": int(numeric.eq(1).sum()),
        "n_invalid_nonbinary": int(invalid_mask.sum()),
        "observed_min": safe_float(observed.min()) if len(observed) else None,
        "observed_max": safe_float(observed.max()) if len(observed) else None,
    }


def numeric_summary(series: pd.Series) -> dict[str, Any]:
    values = to_numeric(series)
    observed = values.dropna()
    if observed.empty:
        return {
            "n_total": len(series), "n_available": 0, "n_missing": len(series),
            "percent_missing": 100.0 if len(series) else None,
            "mean": None, "sd": None, "median": None,
            "q1": None, "q3": None, "min": None, "max": None,
        }
    return {
        "n_total": len(series),
        "n_available": int(observed.size),
        "n_missing": int(values.isna().sum()),
        "percent_missing": safe_float(values.isna().mean() * 100.0, 3),
        "mean": safe_float(observed.mean()),
        "sd": safe_float(observed.std(ddof=1)) if observed.size > 1 else None,
        "median": safe_float(observed.median()),
        "q1": safe_float(observed.quantile(0.25)),
        "q3": safe_float(observed.quantile(0.75)),
        "min": safe_float(observed.min()),
        "max": safe_float(observed.max()),
    }


def audit_predictors(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for concept, candidates in PREDICTOR_SPECS.items():
        source = first_existing(df.columns, candidates)
        if source is None:
            rows.append({
                "concept": concept, "source_column": "<NOT FOUND>",
                "n_total": len(df), "n_available": 0, "n_missing": len(df),
                "percent_missing": 100.0 if len(df) else np.nan,
                "mean": np.nan, "sd": np.nan, "median": np.nan,
                "q1": np.nan, "q3": np.nan, "min": np.nan, "max": np.nan,
                "plausible_min": PLAUSIBILITY_RANGES.get(concept, (np.nan, np.nan))[0],
                "plausible_max": PLAUSIBILITY_RANGES.get(concept, (np.nan, np.nan))[1],
                "n_outside_plausible_range": np.nan,
                "n_binary_zero": np.nan, "n_binary_one": np.nan,
                "n_invalid_nonbinary": np.nan,
            })
            continue

        summary = numeric_summary(df[source])
        values = to_numeric(df[source])
        low, high = PLAUSIBILITY_RANGES.get(concept, (np.nan, np.nan))
        outside = (
            int(((values < low) | (values > high)).sum())
            if np.isfinite(low) and np.isfinite(high) else np.nan
        )
        binary = binary_audit(df[source]) if concept in {
            "infarction", "stress_induced_ischemia"
        } else {}
        rows.append({
            "concept": concept,
            "source_column": source,
            **summary,
            "plausible_min": low,
            "plausible_max": high,
            "n_outside_plausible_range": outside,
            "n_binary_zero": binary.get("n_zero", np.nan),
            "n_binary_one": binary.get("n_one", np.nan),
            "n_invalid_nonbinary": binary.get("n_invalid_nonbinary", np.nan),
        })
    return pd.DataFrame(rows)


def audit_dates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    rows: list[dict[str, Any]] = []
    parsed_dates: dict[str, pd.Series] = {}
    for col in DATE_CANDIDATES:
        if col not in df.columns:
            rows.append({
                "column": col, "present": False, "parse_method": None,
                "n_raw_nonmissing": 0, "n_parsed": 0, "n_failed": 0,
                "min_date": None, "max_date": None,
            })
            continue
        raw_nonmissing = clean_string(df[col]).notna()
        parsed, method = parse_datetime_multi(df[col])
        parsed_dates[col] = parsed
        rows.append({
            "column": col,
            "present": True,
            "parse_method": method,
            "n_raw_nonmissing": int(raw_nonmissing.sum()),
            "n_parsed": int(parsed.notna().sum()),
            "n_failed": int((raw_nonmissing & parsed.isna()).sum()),
            "min_date": parsed.min().date().isoformat() if parsed.notna().any() else None,
            "max_date": parsed.max().date().isoformat() if parsed.notna().any() else None,
        })
    return pd.DataFrame(rows), parsed_dates


def infer_event_time_pairs(columns: Iterable[str]) -> list[tuple[str, str, str]]:
    cols = set(columns)
    pairs: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()

    for event_col, time_col, label in KNOWN_EVENT_TIME_PAIRS:
        if event_col in cols or time_col in cols:
            pairs.append((event_col, time_col, label))
            seen.add((event_col, time_col))

    for event_col in sorted(c for c in cols if "event" in c):
        candidates = [
            event_col.replace("event", "tte", 1),
            event_col.replace("event", "time", 1),
            event_col.replace("_event_", "_tte_", 1),
        ]
        for time_col in candidates:
            if time_col in cols and (event_col, time_col) not in seen:
                pairs.append((event_col, time_col, "Auto-discovered event/time pair"))
                seen.add((event_col, time_col))
                break
    return pairs


def audit_event_pair(df: pd.DataFrame, event_col: str, time_col: str, label: str) -> dict[str, Any]:
    event_present = event_col in df.columns
    time_present = time_col in df.columns
    row: dict[str, Any] = {
        "label": label,
        "event_column": event_col,
        "time_column": time_col,
        "event_present": event_present,
        "time_present": time_present,
        "n_total": len(df),
    }
    if not event_present:
        row.update({
            "n_event_observed": np.nan, "n_events": np.nan, "event_percent": np.nan,
            "n_event_missing_or_unparseable": len(df), "n_event_invalid_nonbinary": np.nan,
        })
        event = pd.Series(np.nan, index=df.index)
    else:
        event = to_numeric(df[event_col])
        row.update({
            "n_event_observed": int(event.notna().sum()),
            "n_events": int(event.eq(1).sum()),
            "event_percent": safe_float(event.eq(1).sum() / event.notna().sum() * 100.0, 3)
                if event.notna().any() else np.nan,
            "n_event_missing_or_unparseable": int(event.isna().sum()),
            "n_event_invalid_nonbinary": int((event.notna() & ~event.isin([0, 1])).sum()),
        })

    if not time_present:
        row.update({
            "n_time_observed": np.nan, "n_time_missing_or_unparseable": len(df),
            "time_min_all": np.nan, "time_max_all": np.nan,
            "time_mean_all": np.nan, "time_median_all": np.nan,
            "time_min_events": np.nan, "time_max_events": np.nan,
            "time_mean_events": np.nan, "time_median_events": np.nan,
            "time_min_nonevents": np.nan, "time_max_nonevents": np.nan,
            "n_nonpositive_time": np.nan, "n_event_missing_time": np.nan,
            "n_nonevent_missing_time": np.nan,
        })
        return row

    time = to_numeric(df[time_col])
    all_time = time.dropna()
    event_time = time[event.eq(1)].dropna()
    nonevent_time = time[event.eq(0)].dropna()
    row.update({
        "n_time_observed": int(time.notna().sum()),
        "n_time_missing_or_unparseable": int(time.isna().sum()),
        "time_min_all": safe_float(all_time.min()) if len(all_time) else np.nan,
        "time_max_all": safe_float(all_time.max()) if len(all_time) else np.nan,
        "time_mean_all": safe_float(all_time.mean()) if len(all_time) else np.nan,
        "time_median_all": safe_float(all_time.median()) if len(all_time) else np.nan,
        "time_min_events": safe_float(event_time.min()) if len(event_time) else np.nan,
        "time_max_events": safe_float(event_time.max()) if len(event_time) else np.nan,
        "time_mean_events": safe_float(event_time.mean()) if len(event_time) else np.nan,
        "time_median_events": safe_float(event_time.median()) if len(event_time) else np.nan,
        "time_min_nonevents": safe_float(nonevent_time.min()) if len(nonevent_time) else np.nan,
        "time_max_nonevents": safe_float(nonevent_time.max()) if len(nonevent_time) else np.nan,
        "n_nonpositive_time": int(time.notna().mul(time <= 0).sum()),
        "n_event_missing_time": int((event.eq(1) & time.isna()).sum()),
        "n_nonevent_missing_time": int((event.eq(0) & time.isna()).sum()),
    })
    return row


def audit_event_pairs(df: pd.DataFrame) -> pd.DataFrame:
    pairs = infer_event_time_pairs(df.columns)
    return pd.DataFrame([audit_event_pair(df, *pair) for pair in pairs])


def split_code_list(value: Any) -> list[str]:
    if value is None or pd.isna(value):
        return []
    text = str(value).strip().upper()
    if text.lower() in BLANK_STRINGS:
        return []
    # patient_v15 has used commas and semicolons in different versions.
    return sorted({part for part in re.split(r"[;,\s]+", text) if part})


def normalize_code(code: str, code_system: str) -> str:
    code = code.strip().upper()
    if code_system == "ICD":
        return re.sub(r"[^A-Z0-9]", "", code)[:3]
    return re.sub(r"[^A-Z0-9]", "", code)


def audit_code_column(
    df: pd.DataFrame,
    column: str,
    domain: str,
    min_count: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if column not in df.columns:
        return ({
            "domain": domain, "column": column, "present": False,
            "n_patients": len(df), "n_nonempty_patients": 0,
            "n_empty_patients": len(df), "n_unique_normalized_codes": 0,
            "n_codes_meeting_threshold": 0, "minimum_patient_count": min_count,
            "max_codes_per_patient": np.nan, "median_codes_per_patient": np.nan,
        }, pd.DataFrame())

    code_system = "ICD" if "icd" in domain else "KVA"
    counter: Counter[str] = Counter()
    codes_per_patient: list[int] = []
    raw = df[column]
    for value in raw:
        normalized = {
            normalize_code(code, code_system)
            for code in split_code_list(value)
        }
        normalized.discard("")
        codes_per_patient.append(len(normalized))
        counter.update(normalized)

    vocabulary_rows = [
        {
            "domain": domain,
            "code_system": code_system,
            "normalized_code": code,
            "patient_count": count,
            "patient_percent": round(count / len(df) * 100.0, 4) if len(df) else np.nan,
            "meets_minimum_count": bool(count >= min_count),
        }
        for code, count in sorted(counter.items())
        if count >= min_count  # Do not export rare code counts below threshold.
    ]
    summary = {
        "domain": domain,
        "column": column,
        "present": True,
        "n_patients": len(df),
        "n_nonempty_patients": int(sum(n > 0 for n in codes_per_patient)),
        "n_empty_patients": int(sum(n == 0 for n in codes_per_patient)),
        "n_unique_normalized_codes": len(counter),
        "n_codes_meeting_threshold": int(sum(v >= min_count for v in counter.values())),
        "minimum_patient_count": min_count,
        "max_codes_per_patient": max(codes_per_patient) if codes_per_patient else 0,
        "median_codes_per_patient": safe_float(np.median(codes_per_patient)) if codes_per_patient else None,
    }
    return summary, pd.DataFrame(vocabulary_rows)


def audit_registry_codes(
    df: pd.DataFrame, min_count: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, Any]] = []
    vocabularies: list[pd.DataFrame] = []
    for domain, column in CODE_COLUMNS.items():
        summary, vocab = audit_code_column(df, column, domain, min_count)
        summaries.append(summary)
        if not vocab.empty:
            vocabularies.append(vocab)
    combined = pd.concat(vocabularies, ignore_index=True) if vocabularies else pd.DataFrame()
    return pd.DataFrame(summaries), combined


def audit_component_events(df: pd.DataFrame, parsed_dates: dict[str, pd.Series]) -> pd.DataFrame:
    index_col = first_existing(df.columns, ["index_exam_date", "cmr_index_date"])
    index_date = parsed_dates.get(index_col) if index_col else None
    rows: list[dict[str, Any]] = []
    components = [
        ("cardiovascular_death", "time_to_cv_death", None),
        ("myocardial_infarction", None, "mi_date"),
        ("unstable_angina", None, "ua_date"),
        ("pci", "tte_any_pci", "pci_date"),
        ("cabg", "tte_any_cabg", "cabg_date"),
        ("angiography", "tte_any_angiography", "angio_date"),
    ]
    for component, tte_col, date_col in components:
        row: dict[str, Any] = {
            "component": component,
            "tte_column": tte_col,
            "date_column": date_col,
            "tte_present": bool(tte_col and tte_col in df.columns),
            "date_present": bool(date_col and date_col in df.columns),
        }
        if tte_col and tte_col in df.columns:
            tte = to_numeric(df[tte_col])
            row.update({
                "n_tte_available": int(tte.notna().sum()),
                "tte_min": safe_float(tte.dropna().min()) if tte.notna().any() else np.nan,
                "tte_max": safe_float(tte.dropna().max()) if tte.notna().any() else np.nan,
            })
        else:
            row.update({"n_tte_available": 0, "tte_min": np.nan, "tte_max": np.nan})

        if date_col and date_col in parsed_dates:
            dates = parsed_dates[date_col]
            row["n_dates_available"] = int(dates.notna().sum())
            if index_date is not None:
                post = dates.notna() & index_date.notna() & (dates >= index_date)
                pre = dates.notna() & index_date.notna() & (dates < index_date)
                row["n_post_index_dates"] = int(post.sum())
                row["n_pre_index_dates"] = int(pre.sum())
                delta_days = (dates[post] - index_date[post]).dt.total_seconds() / 86400.0
                row["post_index_days_min"] = safe_float(delta_days.min()) if len(delta_days) else np.nan
                row["post_index_days_max"] = safe_float(delta_days.max()) if len(delta_days) else np.nan
            else:
                row.update({
                    "n_post_index_dates": np.nan, "n_pre_index_dates": np.nan,
                    "post_index_days_min": np.nan, "post_index_days_max": np.nan,
                })
        else:
            row.update({
                "n_dates_available": 0, "n_post_index_dates": np.nan,
                "n_pre_index_dates": np.nan, "post_index_days_min": np.nan,
                "post_index_days_max": np.nan,
            })
        rows.append(row)
    return pd.DataFrame(rows)


def add_derived_consistency_checks(df: pd.DataFrame) -> pd.DataFrame:
    checks: list[dict[str, Any]] = []

    def add(name: str, status: str, observed: Any, expected: Any, detail: str) -> None:
        checks.append({
            "check": name, "status": status, "observed": observed,
            "expected": expected, "detail": detail,
        })

    # Compare synonymous fields without outputting row-level discrepancies.
    synonyms = [
        ("ischemia", "ischemia_dicho"),
        ("infarct", "infarct_dicho"),
        ("lvef", "lvef_mps"),
        ("event_years", "event_5years"),
        ("tte_years", "tte_5years"),
    ]
    for left, right in synonyms:
        if left not in df.columns or right not in df.columns:
            continue
        a = to_numeric(df[left])
        b = to_numeric(df[right])
        overlap = a.notna() & b.notna()
        differences = int((overlap & ~np.isclose(a.fillna(0), b.fillna(0), rtol=1e-8, atol=1e-8)).sum())
        add(
            f"agreement:{left}_vs_{right}",
            "PASS" if differences == 0 else "REVIEW",
            differences,
            0,
            f"Disagreements among {int(overlap.sum())} rows with both values available.",
        )

    if {"lvedv", "lvesv"}.issubset(df.columns):
        edv = to_numeric(df["lvedv"])
        esv = to_numeric(df["lvesv"])
        valid = edv.notna() & esv.notna()
        invalid = int((valid & (esv > edv)).sum())
        add("LVESV_not_greater_than_LVEDV", "PASS" if invalid == 0 else "REVIEW", invalid, 0,
            "Count where LVESV exceeds LVEDV.")

    return pd.DataFrame(checks)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8", float_format="%.6g")


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def main() -> int:
    args = parse_args()
    source = args.input.resolve()
    output_dir = args.output_dir.resolve()

    if not source.exists():
        print(f"[ERROR] Source file not found: {source}", file=sys.stderr)
        return 2
    if args.min_code_count < 1:
        print("[ERROR] --min-code-count must be >= 1", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Audit script version: {SCRIPT_VERSION}")
    print(f"[INFO] Reading source: {source}")

    df, delimiter, original_columns = load_source(source)
    source_hash = sha256_file(source)
    print(f"[INFO] Loaded {len(df):,} rows x {len(df.columns):,} columns; delimiter={delimiter!r}")

    # Overall modality audit and MR filtering.
    if "modality" in df.columns:
        modality_counts = value_counts_table(normalize_modality(df["modality"]), "modality")
    elif "modality_num" in df.columns:
        modality_counts = value_counts_table(normalize_modality(df["modality_num"]), "modality_num")
    else:
        modality_counts = pd.DataFrame([{
            "variable": "<NOT FOUND>", "value": "<NOT FOUND>",
            "n": len(df), "percent": 100.0,
        }])

    mr_mask, mr_rule = choose_mr_mask(df)
    mr = df.loc[mr_mask].copy().reset_index(drop=True)
    print(f"[INFO] MR rule: {mr_rule}")
    print(f"[INFO] MR rows: {len(mr):,}")

    id_col = first_existing(mr.columns, ["lopnr"])
    if id_col:
        ids = clean_string(mr[id_col])
        n_missing_ids = int(ids.isna().sum())
        n_unique_ids = int(ids.nunique(dropna=True))
        n_duplicate_rows = int(ids.notna().sum() - n_unique_ids)
        duplicate_group_sizes = ids.dropna().value_counts()
        n_ids_with_multiple_rows = int((duplicate_group_sizes > 1).sum())
        max_rows_per_id = int(duplicate_group_sizes.max()) if len(duplicate_group_sizes) else 0
    else:
        n_missing_ids = len(mr)
        n_unique_ids = 0
        n_duplicate_rows = np.nan
        n_ids_with_multiple_rows = np.nan
        max_rows_per_id = np.nan

    date_audit, parsed_dates = audit_dates(mr)
    index_col = first_existing(mr.columns, ["index_exam_date", "cmr_index_date"])
    index_date = parsed_dates.get(index_col) if index_col else None

    predictor_audit = audit_predictors(mr)
    endpoint_audit = audit_event_pairs(mr)
    registry_summary, eligible_code_counts = audit_registry_codes(mr, args.min_code_count)
    component_audit = audit_component_events(mr, parsed_dates)
    consistency_checks = add_derived_consistency_checks(mr)

    # Core checks.
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, observed: Any, expected: Any, detail: str) -> None:
        checks.append({
            "check": name,
            "status": "PASS" if passed else "REVIEW",
            "observed": observed,
            "expected": expected,
            "detail": detail,
        })

    check("MR cohort unique N", n_unique_ids == args.expected_mr_n,
          n_unique_ids, args.expected_mr_n, "Based on nonmissing lopnr values after MR filtering.")
    check("Missing lopnr in MR rows", n_missing_ids == 0,
          n_missing_ids, 0, "No patient identifier is written to audit outputs.")
    check("Duplicate MR rows by lopnr", n_duplicate_rows == 0,
          n_duplicate_rows, 0, "Duplicates require an explicit index-exam selection rule.")

    if index_date is not None and index_date.notna().any():
        observed_min = index_date.min().date().isoformat()
        observed_max = index_date.max().date().isoformat()
        expected_start = pd.Timestamp(args.expected_index_start)
        expected_end = pd.Timestamp(args.expected_index_end)
        out_of_period = int((index_date.notna() & ((index_date < expected_start) | (index_date > expected_end))).sum())
        check("MR index dates within expected period", out_of_period == 0,
              out_of_period, 0,
              f"Observed index range {observed_min} to {observed_max}; expected {args.expected_index_start} to {args.expected_index_end}.")
    else:
        check("MR index date available", False, 0, len(mr), "Could not identify or parse an index date.")

    schema = pd.DataFrame({
        "position": range(1, len(df.columns) + 1),
        "original_column": original_columns,
        "normalized_column": df.columns,
        "present_in_mr_table": [c in mr.columns for c in df.columns],
    })

    core_checks = pd.DataFrame(checks)
    if not consistency_checks.empty:
        core_checks = pd.concat([core_checks, consistency_checks], ignore_index=True)

    write_csv(schema, output_dir / "schema.csv")
    write_csv(modality_counts, output_dir / "modality_counts.csv")
    write_csv(predictor_audit, output_dir / "predictor_missingness_and_ranges.csv")
    write_csv(date_audit, output_dir / "date_audit.csv")
    write_csv(endpoint_audit, output_dir / "endpoint_pair_audit.csv")
    write_csv(component_audit, output_dir / "component_event_audit.csv")
    write_csv(registry_summary, output_dir / "registry_code_summary.csv")
    if not eligible_code_counts.empty:
        write_csv(eligible_code_counts, output_dir / "eligible_registry_codes_min_count.csv")
    write_csv(core_checks, output_dir / "audit_checks.csv")

    manifest = {
        "audit": {
            "script": Path(__file__).name,
            "script_version": SCRIPT_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "aggregate_outputs_only": True,
        },
        "source": {
            "path": str(args.input),
            "filename": source.name,
            "sha256": source_hash,
            "size_bytes": source.stat().st_size,
            "delimiter": delimiter,
            "rows": len(df),
            "columns": len(df.columns),
        },
        "mr_cohort": {
            "filter_rule": mr_rule,
            "rows": len(mr),
            "unique_nonmissing_lopnr": n_unique_ids,
            "missing_lopnr": n_missing_ids,
            "duplicate_rows_beyond_first": safe_int(n_duplicate_rows),
            "identifiers_with_multiple_rows": safe_int(n_ids_with_multiple_rows),
            "maximum_rows_per_identifier": safe_int(max_rows_per_id),
            "expected_unique_n": args.expected_mr_n,
            "index_date_column": index_col,
            "index_date_min": index_date.min().date().isoformat()
                if index_date is not None and index_date.notna().any() else None,
            "index_date_max": index_date.max().date().isoformat()
                if index_date is not None and index_date.notna().any() else None,
        },
        "registry_feature_rule": {
            "minimum_patient_count": args.min_code_count,
            "icd_representation": "first three alphanumeric characters",
            "kva_representation": "full alphanumeric code without punctuation",
            "source_columns": CODE_COLUMNS,
            "note": "eligible_registry_codes_min_count.csv exports only codes meeting the threshold; rare code counts are suppressed.",
        },
        "outputs": sorted(p.name for p in output_dir.glob("*")),
    }
    with (output_dir / "audit_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, default=json_default)
        handle.write("\n")

    review_count = int(core_checks["status"].eq("REVIEW").sum()) if not core_checks.empty else 0
    pass_count = int(core_checks["status"].eq("PASS").sum()) if not core_checks.empty else 0

    summary_lines = [
        "patient_v15 aggregate audit",
        "=" * 30,
        f"Script version: {SCRIPT_VERSION}",
        f"Source: {args.input}",
        f"SHA-256: {source_hash}",
        f"Source dimensions: {len(df):,} rows x {len(df.columns):,} columns",
        f"MR filter: {mr_rule}",
        f"MR rows: {len(mr):,}",
        f"Unique MR identifiers: {n_unique_ids:,}",
        f"Missing MR identifiers: {n_missing_ids:,}",
        f"Duplicate rows beyond first: {n_duplicate_rows}",
        f"Checks: {pass_count} PASS; {review_count} REVIEW",
        "",
        "Important:",
        "- No patient-level rows or identifiers were written.",
        "- REVIEW does not necessarily mean an error; inspect audit_checks.csv.",
        "- Do not commit generated outputs until disclosure risk has been reviewed.",
        "",
        "Generated files:",
    ]
    summary_lines.extend(f"- {p.name}" for p in sorted(output_dir.glob("*")))
    (output_dir / "audit_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"[INFO] Audit outputs written to: {output_dir}")
    print(f"[INFO] Checks: {pass_count} PASS; {review_count} REVIEW")
    print("[INFO] Inspect audit_summary.txt and audit_checks.csv first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
