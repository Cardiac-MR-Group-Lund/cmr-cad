#!/usr/bin/env python3
"""
02_build_analysis_dataset.py

Build the frozen MR-only revision analysis dataset from patient_v15.csv.

This script is deliberately limited to deterministic cohort construction,
variable standardisation, endpoint derivation, and validation. It performs no
imputation, scaling, feature selection, cross-validation, or model fitting.

Active source:
    data/raw/patient_v15.csv

Outputs (local/private):
    data/processed/cmr_revision_analysis.csv
    results/audit/analysis_dataset/analysis_dataset_manifest.json
    results/audit/analysis_dataset/analysis_data_dictionary.csv
    results/audit/analysis_dataset/analysis_dataset_checks.csv
    results/audit/analysis_dataset/analysis_dataset_summary.txt
    results/audit/analysis_dataset/endpoint_summary.csv
    results/audit/analysis_dataset/predictor_missingness.csv

The processed dataset contains an internal sequential analysis_id but no lopnr.
Do not commit data/raw/, data/processed/, or patient-level outputs to Git.

Run from repository root:
    python scripts/02_build_analysis_dataset.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

SCRIPT_VERSION = "1.0.1"
DEFAULT_INPUT = Path("data/raw/patient_v15.csv")
DEFAULT_OUTPUT = Path("data/processed/cmr_revision_analysis.csv")
DEFAULT_AUDIT_DIR = Path("results/audit/analysis_dataset")
EXPECTED_SOURCE_SHA256 = "4767b39b32542b7890af258b4ecb9e8f586e574571010f8df8a1d1783b236e76"
EXPECTED_MR_N = 2159
DAYS_PER_YEAR = 365.25
BLANK_STRINGS = {"", "nan", "none", "null", "na", "n/a", "nat", "<na>"}

PRIMARY_SOURCE_COLUMNS = {
    "lopnr", "modality", "index_exam_date", "admin_censor_date",
    "age", "sex_num", "bmi", "bsa_m2", "length_cm", "weight_kg",
    "lvef", "lvedv", "lvesv", "infarct", "infarct_dicho",
    "ischemia", "ischemia_dicho", "pre_icd_codes", "pre_kva_codes",
    "comp_cv_event_blind0_5y", "comp_cv_tte_blind0_5y",
    "comp_cv_event_blind30_5y", "comp_cv_tte_blind30_5y",
    "comp_cv_event_blind60_5y", "comp_cv_tte_blind60_5y",
    "comp_cv_event_blind90_5y", "comp_cv_tte_blind90_5y",
    "comp_cv_ua_event_blind0_5y", "comp_cv_ua_tte_blind0_5y",
    "comp_cv_ua_event_blind30_5y", "comp_cv_ua_tte_blind30_5y",
    "comp_cv_ua_event_blind60_5y", "comp_cv_ua_tte_blind60_5y",
    "comp_cv_ua_event_blind90_5y", "comp_cv_ua_tte_blind90_5y",
    "comp_cv_event_late_revasc_5y", "comp_cv_tte_late_revasc_5y",
    "comp_cv_event_late_revasc_diff", "time_to_cv_death", "death_date",
}

ENDPOINT_MAP = {
    "mace_0d": ("comp_cv_event_blind0_5y", "comp_cv_tte_blind0_5y",
                "CV death, AMI, or PCI/CABG; no blanking window"),
    "mace_30d": ("comp_cv_event_blind30_5y", "comp_cv_tte_blind30_5y",
                 "CV death, AMI, or PCI/CABG; 30-day window"),
    "mace_60d": ("comp_cv_event_blind60_5y", "comp_cv_tte_blind60_5y",
                 "CV death, AMI, or PCI/CABG; 60-day window"),
    "mace_90d": ("comp_cv_event_blind90_5y", "comp_cv_tte_blind90_5y",
                 "CV death, AMI, or PCI/CABG; 90-day window"),
    "mace_ua_0d": ("comp_cv_ua_event_blind0_5y", "comp_cv_ua_tte_blind0_5y",
                   "CV death, AMI, PCI/CABG, or UA; no blanking window"),
    "mace_ua_30d": ("comp_cv_ua_event_blind30_5y", "comp_cv_ua_tte_blind30_5y",
                    "CV death, AMI, PCI/CABG, or UA; 30-day window"),
    "mace_ua_60d": ("comp_cv_ua_event_blind60_5y", "comp_cv_ua_tte_blind60_5y",
                    "CV death, AMI, PCI/CABG, or UA; 60-day window"),
    "mace_ua_90d": ("comp_cv_ua_event_blind90_5y", "comp_cv_ua_tte_blind90_5y",
                    "CV death, AMI, PCI/CABG, or UA; 90-day window"),
    "mace_late_revasc": ("comp_cv_event_late_revasc_5y", "comp_cv_tte_late_revasc_5y",
                         "CV death, AMI, or PCI/CABG occurring after 6 months"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build frozen MR-only revision analysis dataset.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    p.add_argument("--expected-sha256", default=EXPECTED_SOURCE_SHA256)
    p.add_argument("--expected-mr-n", type=int, default=EXPECTED_MR_N)
    p.add_argument("--allow-source-hash-mismatch", action="store_true",
                   help="Allow a different source hash; mismatch remains a REVIEW check.")
    return p.parse_args()


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
        return csv.Sniffer().sniff(sample, delimiters=";,\t|").delimiter
    except csv.Error:
        first = sample.splitlines()[0] if sample else ""
        return max([";", ",", "\t", "|"], key=first.count)


def clean_string(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip()
    return out.mask(out.str.lower().isin(BLANK_STRINGS))


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(clean_string(series).str.replace(",", ".", regex=False), errors="coerce")


def parse_date(series: pd.Series) -> tuple[pd.Series, str]:
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
            return pd.Timestamp("1601-01-01") + pd.to_timedelta(nums, unit="s", errors="coerce"), "seconds_since_1601-01-01"
        if 1e9 <= magnitude < 5e9:
            return pd.to_datetime(nums, unit="s", origin="unix", errors="coerce"), "unix_seconds"
        if 1e12 <= magnitude < 5e13:
            return pd.to_datetime(nums, unit="ms", origin="unix", errors="coerce"), "unix_milliseconds"
        if 1e7 <= magnitude < 1e9:
            return pd.to_datetime(s, format="%Y%m%d", errors="coerce"), "yyyymmdd"
    return pd.to_datetime(s, format="mixed", errors="coerce"), "text_mixed"


def load_source(path: Path) -> tuple[pd.DataFrame, str]:
    delimiter = detect_delimiter(path)
    df = pd.read_csv(path, sep=delimiter, dtype=str, encoding="utf-8-sig", quotechar='"',
                     low_memory=False, keep_default_na=False, na_filter=False)
    cols = [normalize_name(c) for c in df.columns]
    duplicates = sorted({c for c in cols if cols.count(c) > 1})
    if duplicates:
        raise ValueError(f"Normalised duplicate columns: {duplicates}")
    df.columns = cols
    return df, delimiter


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required source columns: {missing}")


def normalize_code_list(series: pd.Series) -> pd.Series:
    def one(value: Any) -> str:
        if value is None or pd.isna(value):
            return ""
        text = str(value).strip().upper()
        if text.lower() in BLANK_STRINGS:
            return ""
        codes = {re.sub(r"[^A-Z0-9.]", "", x) for x in re.split(r"[;,\s]+", text) if x}
        codes.discard("")
        return ";".join(sorted(codes))
    return series.map(one).astype("string")


def count_codes(series: pd.Series) -> pd.Series:
    return series.fillna("").map(lambda x: 0 if x == "" else len(set(str(x).split(";")))).astype("int64")


def safe_float(value: Any, digits: int = 8) -> float | None:
    if value is None or pd.isna(value) or not np.isfinite(value):
        return None
    return round(float(value), digits)


def write_csv(df: pd.DataFrame, path: Path, sep: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, sep=sep, encoding="utf-8", float_format="%.12g", date_format="%Y-%m-%d")


def main() -> int:
    args = parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    audit_dir = args.audit_dir.resolve()
    audit_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    if not source.exists():
        print(f"[ERROR] Source not found: {source}", file=sys.stderr)
        return 2

    source_hash = sha256_file(source)
    hash_match = source_hash.lower() == args.expected_sha256.lower()
    if not hash_match and not args.allow_source_hash_mismatch:
        print("[ERROR] Source SHA-256 does not match the frozen patient_v15.csv.", file=sys.stderr)
        print(f"[ERROR] Expected: {args.expected_sha256}", file=sys.stderr)
        print(f"[ERROR] Observed: {source_hash}", file=sys.stderr)
        return 3

    print(f"[INFO] Script version: {SCRIPT_VERSION}")
    print(f"[INFO] Source SHA-256 match: {hash_match}")
    raw, delimiter = load_source(source)
    require_columns(raw, PRIMARY_SOURCE_COLUMNS)

    mr = raw.loc[clean_string(raw["modality"]).str.upper().eq("MR")].copy()
    ids = clean_string(mr["lopnr"])
    if len(mr) != args.expected_mr_n or ids.nunique(dropna=True) != args.expected_mr_n:
        raise ValueError(f"Expected {args.expected_mr_n} unique MR patients; got rows={len(mr)}, unique={ids.nunique(dropna=True)}")
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("MR cohort has missing or duplicated lopnr.")

    mr = mr.assign(_sort_id=ids).sort_values("_sort_id", kind="mergesort").reset_index(drop=True)
    analysis_id = pd.Series(np.arange(1, len(mr) + 1), index=mr.index, dtype="int64")

    index_date, index_method = parse_date(mr["index_exam_date"])
    admin_date, admin_method = parse_date(mr["admin_censor_date"])
    death_date, death_method = parse_date(mr["death_date"])
    admin_followup_years = (admin_date - index_date).dt.total_seconds() / (DAYS_PER_YEAR * 86400.0)

    age = numeric(mr["age"])
    sex = numeric(mr["sex_num"])
    height = numeric(mr["length_cm"])
    weight = numeric(mr["weight_kg"])
    bmi_source = numeric(mr["bmi"])
    bsa_source = numeric(mr["bsa_m2"])

    bmi_calc = weight / (height / 100.0) ** 2
    bsa_calc = np.sqrt((height * weight) / 3600.0)
    bmi = bmi_source.where(bmi_source.notna(), bmi_calc)

    # NA-safe validity mask for pandas nullable numeric dtypes.
    bsa_source_valid = (
        bsa_source.notna() & bsa_source.gt(0).fillna(False)
    ).astype(bool)
    bsa = bsa_source.where(bsa_source_valid, bsa_calc)

    lvef = numeric(mr["lvef"])
    lvedv = numeric(mr["lvedv"])
    lvesv = numeric(mr["lvesv"])
    with np.errstate(divide="ignore", invalid="ignore"):
        lvedvi = (lvedv / bsa).replace([np.inf, -np.inf], np.nan)
        lvesvi = (lvesv / bsa).replace([np.inf, -np.inf], np.nan)

    infarct_raw = numeric(mr["infarct"])
    infarct = numeric(mr["infarct_dicho"])
    ischemia_raw = numeric(mr["ischemia"])
    ischemia = numeric(mr["ischemia_dicho"])
    pre_icd = normalize_code_list(mr["pre_icd_codes"])
    pre_kva = normalize_code_list(mr["pre_kva_codes"])

    final = pd.DataFrame({
        "analysis_id": analysis_id,
        "index_exam_date": index_date,
        "admin_censor_date": admin_date,
        "admin_followup_years": admin_followup_years,
        "age_years": age,
        "sex_male": sex,
        "height_cm": height,
        "weight_kg": weight,
        "bmi": bmi,
        "bmi_source_observed": bmi_source.notna().astype("int8"),
        "bsa_m2": bsa,
        "bsa_source_observed": bsa_source_valid.astype("int8"),
        "lvef_percent": lvef,
        "lvedv_ml": lvedv,
        "lvesv_ml": lvesv,
        "lvedvi_ml_m2": lvedvi,
        "lvesvi_ml_m2": lvesvi,
        "infarct_raw": infarct_raw,
        "infarct_binary": infarct,
        "ischemia_raw": ischemia_raw,
        "ischemia_binary": ischemia,
        "pre_icd_codes": pre_icd,
        "pre_icd_n": count_codes(pre_icd),
        "pre_kva_codes": pre_kva,
        "pre_kva_n": count_codes(pre_kva),
    })

    endpoint_definitions: list[dict[str, Any]] = []
    for target, (event_source, time_source, definition) in ENDPOINT_MAP.items():
        final[f"{target}_event"] = numeric(mr[event_source])
        final[f"{target}_tte_years"] = numeric(mr[time_source])
        endpoint_definitions.append({"endpoint": target, "event_source": event_source,
                                     "time_source": time_source, "definition": definition})

    late_label = numeric(mr["comp_cv_event_late_revasc_diff"])
    hard_event = late_label.isin([1, 2]).astype("int8")
    hard_tte = pd.Series(np.where(hard_event.eq(1), numeric(mr["comp_cv_tte_late_revasc_5y"]),
                                  admin_followup_years), index=mr.index, dtype=float)
    final["late_revasc_hierarchy_label"] = late_label
    final["hard_mace_event"] = hard_event
    final["hard_mace_tte_years"] = hard_tte
    endpoint_definitions.append({
        "endpoint": "hard_mace",
        "event_source": "derived: comp_cv_event_late_revasc_diff in {1,2}",
        "time_source": "event: comp_cv_tte_late_revasc_5y; non-event: admin censoring",
        "definition": "Cardiovascular death or acute myocardial infarction",
    })

    checks: list[dict[str, Any]] = []
    def check(name: str, passed: bool, observed: Any, expected: Any, detail: str) -> None:
        checks.append({"check": name, "status": "PASS" if passed else "REVIEW",
                       "observed": observed, "expected": expected, "detail": detail})

    check("Source SHA-256", hash_match, source_hash, args.expected_sha256, "Frozen source identity.")
    check("MR cohort rows", len(final) == args.expected_mr_n, len(final), args.expected_mr_n, "MR-only cohort.")
    check("Unique analysis_id", final["analysis_id"].nunique() == len(final), final["analysis_id"].nunique(), len(final), "Internal ID.")
    check("Index dates complete", index_date.notna().all(), int(index_date.isna().sum()), 0, f"Parse: {index_method}.")
    check("Admin dates complete", admin_date.notna().all(), int(admin_date.isna().sum()), 0, f"Parse: {admin_method}.")
    check("Admin follow-up positive", bool((admin_followup_years > 0).all()), int((admin_followup_years <= 0).sum()), 0, "Censor after index.")
    check("Sex binary", bool((sex.isna() | sex.isin([0, 1])).all()), int((sex.notna() & ~sex.isin([0, 1])).sum()), 0, "0=female; 1=male.")
    check("Infarct binary", bool((infarct.isna() | infarct.isin([0, 1])).all()), int((infarct.notna() & ~infarct.isin([0, 1])).sum()), 0, "Indeterminate missing.")
    check("Ischemia binary", bool((ischemia.isna() | ischemia.isin([0, 1])).all()), int((ischemia.notna() & ~ischemia.isin([0, 1])).sum()), 0, "Raw 1/2 positive; 99 missing.")
    check("Ischemia positive count", int(ischemia.eq(1).sum()) == 425, int(ischemia.eq(1).sum()), 425, "Frozen count.")
    check("Ischemia missing count", int(ischemia.isna().sum()) == 89, int(ischemia.isna().sum()), 89, "Indeterminate.")
    check("LV volume pairs available", int((lvedv.notna() & lvesv.notna()).sum()) > 0,
          int((lvedv.notna() & lvesv.notna()).sum()), ">0", "Rows with both LVEDV and LVESV.")
    check("No LVESV greater than LVEDV", int((lvedv.notna() & lvesv.notna() & (lvesv > lvedv)).sum()) == 0,
          int((lvedv.notna() & lvesv.notna() & (lvesv > lvedv)).sum()), 0, "Physiological consistency.")
    check("Hard-MACE count", int(hard_event.sum()) == 83, int(hard_event.sum()), 83, "37 CV deaths + 46 AMIs.")
    check("Hard-MACE TTE complete", hard_tte.notna().all(), int(hard_tte.isna().sum()), 0, "Event/censor time.")
    check("Hard-MACE TTE positive", bool((hard_tte > 0).all()), int((hard_tte <= 0).sum()), 0, "Positive TTE.")

    expected_events = {
        "mace_0d": 212, "mace_30d": 196, "mace_60d": 179, "mace_90d": 163,
        "mace_ua_0d": 216, "mace_ua_30d": 200, "mace_ua_60d": 182,
        "mace_ua_90d": 167, "mace_late_revasc": 150,
    }
    for endpoint, expected in expected_events.items():
        event = final[f"{endpoint}_event"]
        tte = final[f"{endpoint}_tte_years"]
        check(f"{endpoint} event count", int(event.eq(1).sum()) == expected,
              int(event.eq(1).sum()), expected, "Frozen endpoint count.")
        check(f"{endpoint} event/TTE complete", event.notna().all() and tte.notna().all(),
              int(event.isna().sum() + tte.isna().sum()), 0, "Complete status and TTE.")
        check(f"{endpoint} TTE positive", bool((tte > 0).all()), int((tte <= 0).sum()), 0, "Positive TTE.")

    cv_label = late_label.eq(1)
    death_tte = (death_date - index_date).dt.total_seconds() / (DAYS_PER_YEAR * 86400.0)
    common = cv_label & death_tte.notna()
    diff_days = (death_tte[common] - numeric(mr.loc[common, "comp_cv_tte_late_revasc_5y"])).abs() * DAYS_PER_YEAR
    check("CV-death hierarchy TTE agrees with death_date", bool((diff_days <= 1.0).all()),
          int((diff_days > 1.0).sum()), 0, f"Tolerance 1 day; N={int(common.sum())}; parse={death_method}.")

    checks_df = pd.DataFrame(checks)
    review_count = int(checks_df["status"].eq("REVIEW").sum())
    write_csv(checks_df, audit_dir / "analysis_dataset_checks.csv")
    if review_count:
        raise RuntimeError(f"Build stopped: {review_count} validation check(s) require review.")

    write_csv(final, output, sep=";")
    output_hash = sha256_file(output)

    endpoint_rows = []
    for item in endpoint_definitions:
        endpoint = item["endpoint"]
        event = final[f"{endpoint}_event"]
        tte = final[f"{endpoint}_tte_years"]
        endpoint_rows.append({**item, "n": len(final), "events": int(event.eq(1).sum()),
                              "event_percent": safe_float(event.mean() * 100.0, 4),
                              "tte_min_all": safe_float(tte.min()),
                              "tte_median_all": safe_float(tte.median()),
                              "tte_max_all": safe_float(tte.max()),
                              "tte_min_events": safe_float(tte[event.eq(1)].min()),
                              "tte_max_events": safe_float(tte[event.eq(1)].max())})
    endpoint_summary = pd.DataFrame(endpoint_rows)

    predictor_cols = ["age_years", "sex_male", "bmi", "bsa_m2", "lvef_percent",
                      "lvedvi_ml_m2", "lvesvi_ml_m2", "infarct_binary", "ischemia_binary"]
    missingness = pd.DataFrame([{
        "variable": col, "n": len(final), "available": int(final[col].notna().sum()),
        "missing": int(final[col].isna().sum()),
        "missing_percent": safe_float(final[col].isna().mean() * 100.0, 4),
    } for col in predictor_cols])

    dictionary_rows = [
        ("analysis_id", "Derived", "Stable sequential ID; lopnr is not exported."),
        ("index_exam_date", "index_exam_date", "Index MR date."),
        ("admin_censor_date", "admin_censor_date", "Administrative censor date."),
        ("admin_followup_years", "Derived", "Years from index to administrative censoring."),
        ("age_years", "age", "Age at index MR."),
        ("sex_male", "sex_num", "0=female; 1=male."),
        ("bmi", "bmi/height/weight", "Observed BMI; deterministic backfill if missing."),
        ("bsa_m2", "bsa_m2/height/weight", "Observed BSA; Mosteller backfill if missing."),
        ("lvef_percent", "lvef", "Left ventricular ejection fraction."),
        ("lvedvi_ml_m2", "lvedv/bsa_m2", "Indexed LV end-diastolic volume."),
        ("lvesvi_ml_m2", "lvesv/bsa_m2", "Indexed LV end-systolic volume."),
        ("infarct_binary", "infarct_dicho", "0=no; 1=yes; indeterminate missing."),
        ("ischemia_binary", "ischemia_dicho", "0=no; 1=yes; raw 99 missing."),
        ("pre_icd_codes", "pre_icd_codes", "Normalised pre-index ICD code list."),
        ("pre_kva_codes", "pre_kva_codes", "Normalised pre-index KVA code list."),
        ("late_revasc_hierarchy_label", "comp_cv_event_late_revasc_diff", "0=none; 1=CV death; 2=AMI; 3=late PCI/CABG."),
        ("hard_mace_event", "Derived", "1 for hierarchy label 1 or 2."),
        ("hard_mace_tte_years", "Derived", "Hard-event TTE or administrative censoring."),
    ]
    for item in endpoint_definitions:
        dictionary_rows.append((f"{item['endpoint']}_event", item["event_source"], item["definition"] + "; event indicator."))
        dictionary_rows.append((f"{item['endpoint']}_tte_years", item["time_source"], item["definition"] + "; time in years."))
    dictionary = pd.DataFrame(dictionary_rows, columns=["analysis_variable", "source_or_derivation", "definition"])

    write_csv(endpoint_summary, audit_dir / "endpoint_summary.csv")
    write_csv(missingness, audit_dir / "predictor_missingness.csv")
    write_csv(dictionary, audit_dir / "analysis_data_dictionary.csv")

    manifest = {
        "build": {"script": Path(__file__).name, "version": SCRIPT_VERSION,
                  "created_utc": datetime.now(timezone.utc).isoformat()},
        "source": {"path": str(args.input), "sha256": source_hash, "delimiter": delimiter,
                   "rows": len(raw), "columns": len(raw.columns)},
        "cohort": {"filter": "modality == MR", "n": len(final),
                   "index_date_min": index_date.min().date().isoformat(),
                   "index_date_max": index_date.max().date().isoformat()},
        "processed_output": {"path": str(args.output), "sha256": output_hash,
                             "rows": len(final), "columns": len(final.columns), "delimiter": ";"},
        "date_parsing": {"index_exam_date": index_method, "admin_censor_date": admin_method,
                         "death_date": death_method},
        "decisions": {"lv_mass": "Excluded because 51.6% was missing.",
                      "imputation": "None in dataset build.",
                      "active_source": "patient_v15.csv only; PAR_UT not read."},
        "backfills": {"bmi": int((bmi_source.isna() & bmi.notna()).sum()),
                      "bsa": int((~bsa_source_valid & bsa.notna()).sum())},
        "checks": {"pass": int(checks_df["status"].eq("PASS").sum()), "review": 0},
    }
    with (audit_dir / "analysis_dataset_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    summary = [
        "Frozen MR analysis dataset build", "=" * 32,
        f"Source SHA-256: {source_hash}", f"Source hash matched expected: {hash_match}",
        f"MR cohort: {len(final):,}", f"Processed file: {args.output}",
        f"Processed SHA-256: {output_hash}",
        f"Processed dimensions: {len(final):,} rows x {len(final.columns)} columns",
        f"BMI values backfilled: {manifest['backfills']['bmi']}",
        f"BSA values backfilled: {manifest['backfills']['bsa']}",
        "LV mass: excluded", "Imputation: none", "", "Endpoint event counts:",
    ]
    summary.extend(f"- {r.endpoint}: {r.events}" for r in endpoint_summary.itertuples(index=False))
    summary.extend(["", f"Validation checks: {manifest['checks']['pass']} PASS; 0 REVIEW",
                    "No lopnr is present in the processed dataset."])
    (audit_dir / "analysis_dataset_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

    print(f"[INFO] Built: {output}")
    print(f"[INFO] Processed SHA-256: {output_hash}")
    print(f"[INFO] Dimensions: {len(final):,} rows x {len(final.columns)} columns")
    print(f"[INFO] Checks: {manifest['checks']['pass']} PASS; 0 REVIEW")
    print(f"[INFO] Audit outputs: {audit_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
