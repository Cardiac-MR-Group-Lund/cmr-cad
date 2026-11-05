#!/usr/bin/env python3
"""
05_generate_table1.py

Generate revised Table 1 stratified by stress-induced ischemia from the frozen
MR analysis dataset.

Input:
    data/processed/cmr_revision_analysis.csv

Outputs:
    results/tables/table1/table1_formatted.csv
    results/tables/table1/table1_numeric_long.csv
    results/tables/table1/table1_missingness.csv
    results/tables/table1/table1_endpoint_sensitivity.csv
    results/tables/table1/table1_checks.csv
    results/tables/table1/table1_manifest.json
    results/tables/table1/table1_summary.txt

Conventions:
- Total cohort includes all 2,159 patients.
- Ischemia-stratified columns include only observed ischemia status:
  1,645 without ischemia and 425 with ischemia.
- The 89 clinically indeterminate ischemia assessments remain in the total
  cohort and are excluded from the two stratified columns.
- Continuous variables: mean +/- sample SD.
- Categorical variables: n (%), with percentage based on nonmissing values.
- No between-group P values are generated.
- LV mass is excluded.
- Historical diabetes, hypertension, and IHD are deterministically derived from
  frozen pre-index ICD/KVA lists using the documented definitions.
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
from typing import Any, Callable

import numpy as np
import pandas as pd

SCRIPT_VERSION = "1.0.0"
DEFAULT_INPUT = Path("data/processed/cmr_revision_analysis.csv")
DEFAULT_OUTPUT_DIR = Path("results/tables/table1")
EXPECTED_INPUT_SHA256 = "e94b8aebe4c4939b0b3def943879539f4f242ba7ef39fd7ac929407578f7db9c"
EXPECTED_N = 2159
EXPECTED_NO_ISCHEMIA = 1645
EXPECTED_ISCHEMIA = 425
EXPECTED_ISCHEMIA_MISSING = 89

GROUPS = [
    ("Total cohort", None),
    ("No stress-induced ischemia", 0),
    ("Stress-induced ischemia", 1),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate revised ischemia-stratified Table 1.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--expected-sha256", default=EXPECTED_INPUT_SHA256)
    p.add_argument("--allow-input-hash-mismatch", action="store_true")
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


def split_codes(value: Any) -> set[str]:
    if value is None or pd.isna(value):
        return set()
    text = str(value).strip().upper()
    if not text:
        return set()
    return {re.sub(r"[^A-Z0-9]", "", x) for x in re.split(r"[;,\s]+", text) if x}


def derive_registry_flags(df: pd.DataFrame) -> pd.DataFrame:
    diabetes = []
    hypertension = []
    ihd = []
    for icd_value, kva_value in zip(df["pre_icd_codes"], df["pre_kva_codes"]):
        icd = split_codes(icd_value)
        kva = split_codes(kva_value)
        icd3 = {code[:3] for code in icd if len(code) >= 3}

        diabetes.append(any(code in {"E10", "E11", "E12", "E13", "E14"} for code in icd3))
        hypertension.append(any(code in {"I10", "I11", "I12", "I13", "I15"} for code in icd3))

        prior_ihd_icd = any(code in {"I21", "I22", "I23", "I24", "I25"} for code in icd3)
        prior_pci = any(code.startswith("FNG") for code in kva)
        prior_cabg = any(code.startswith(prefix) for code in kva for prefix in ("FNA", "FNB", "FNC", "FND", "FNE"))
        ihd.append(prior_ihd_icd or prior_pci or prior_cabg)

    return pd.DataFrame({
        "diabetes_preindex_derived": pd.Series(diabetes, dtype="int8"),
        "hypertension_preindex_derived": pd.Series(hypertension, dtype="int8"),
        "ihd_preindex_derived": pd.Series(ihd, dtype="int8"),
    }, index=df.index)


def fmt_cont(mean: float, sd: float, digits: int) -> str:
    if pd.isna(mean):
        return ""
    return f"{mean:.{digits}f} +/- {sd:.{digits}f}"


def fmt_cat(n: int, denominator: int, digits: int = 1) -> str:
    if denominator == 0:
        return ""
    return f"{n} ({100.0 * n / denominator:.{digits}f}%)"


def group_mask(df: pd.DataFrame, value: int | None) -> pd.Series:
    if value is None:
        return pd.Series(True, index=df.index)
    return df["ischemia_binary"].eq(value)


def continuous_row(df: pd.DataFrame, variable: str, label: str, section: str, digits: int = 1) -> tuple[dict, list[dict]]:
    formatted = {"section": section, "characteristic": label}
    long_rows = []
    for group_name, group_value in GROUPS:
        mask = group_mask(df, group_value)
        values = pd.to_numeric(df.loc[mask, variable], errors="coerce")
        available = int(values.notna().sum())
        missing = int(values.isna().sum())
        mean = float(values.mean()) if available else np.nan
        sd = float(values.std(ddof=1)) if available > 1 else np.nan
        formatted[group_name] = fmt_cont(mean, sd, digits)
        long_rows.append({
            "section": section, "characteristic": label, "variable": variable,
            "type": "continuous", "group": group_name,
            "group_n": int(mask.sum()), "available_n": available, "missing_n": missing,
            "mean": mean, "sd": sd, "count": np.nan, "percent": np.nan,
        })
    return formatted, long_rows


def categorical_row(df: pd.DataFrame, variable: str, label: str, section: str, positive: Any = 1) -> tuple[dict, list[dict]]:
    formatted = {"section": section, "characteristic": label}
    long_rows = []
    for group_name, group_value in GROUPS:
        mask = group_mask(df, group_value)
        values = df.loc[mask, variable]
        available = int(values.notna().sum())
        missing = int(values.isna().sum())
        count = int(values.eq(positive).sum())
        percent = 100.0 * count / available if available else np.nan
        formatted[group_name] = fmt_cat(count, available)
        long_rows.append({
            "section": section, "characteristic": label, "variable": variable,
            "type": "categorical", "group": group_name,
            "group_n": int(mask.sum()), "available_n": available, "missing_n": missing,
            "mean": np.nan, "sd": np.nan, "count": count, "percent": percent,
        })
    return formatted, long_rows


def n_row(df: pd.DataFrame) -> tuple[dict, list[dict]]:
    formatted = {"section": "Population", "characteristic": "Patients, n"}
    long_rows = []
    for group_name, group_value in GROUPS:
        n = int(group_mask(df, group_value).sum())
        formatted[group_name] = str(n)
        long_rows.append({
            "section": "Population", "characteristic": "Patients, n", "variable": "n",
            "type": "count", "group": group_name, "group_n": n, "available_n": n,
            "missing_n": 0, "mean": np.nan, "sd": np.nan, "count": n, "percent": np.nan,
        })
    return formatted, long_rows


def add_row(formatted_rows: list[dict], long_rows: list[dict], result: tuple[dict, list[dict]]) -> None:
    formatted, long_part = result
    formatted_rows.append(formatted)
    long_rows.extend(long_part)


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
        print("[ERROR] Frozen processed dataset checksum mismatch.", file=sys.stderr)
        print(f"[ERROR] Expected: {args.expected_sha256}", file=sys.stderr)
        print(f"[ERROR] Observed: {input_hash}", file=sys.stderr)
        return 3

    delimiter = detect_delimiter(source)
    df = pd.read_csv(source, sep=delimiter, encoding="utf-8-sig", low_memory=False)

    required = {
        "analysis_id", "age_years", "sex_male", "bmi", "lvef_percent",
        "lvedvi_ml_m2", "lvesvi_ml_m2", "infarct_binary", "ischemia_binary",
        "pre_icd_codes", "pre_kva_codes", "admin_followup_years",
        "mace_0d_event", "mace_30d_event", "mace_90d_event",
        "mace_late_revasc_event", "hard_mace_event",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if len(df) != EXPECTED_N:
        raise ValueError(f"Expected N={EXPECTED_N}; observed N={len(df)}")
    if df["analysis_id"].isna().any() or df["analysis_id"].duplicated().any():
        raise ValueError("analysis_id contains missing or duplicate values.")

    flags = derive_registry_flags(df)
    df = pd.concat([df, flags], axis=1)

    formatted_rows: list[dict] = []
    long_rows: list[dict] = []

    add_row(formatted_rows, long_rows, n_row(df))
    add_row(formatted_rows, long_rows, continuous_row(df, "age_years", "Age, years", "Demographics", 1))
    add_row(formatted_rows, long_rows, categorical_row(df, "sex_male", "Women, n (%)", "Demographics", positive=0))
    add_row(formatted_rows, long_rows, continuous_row(df, "bmi", "BMI, kg/m2", "Demographics", 1))

    add_row(formatted_rows, long_rows, categorical_row(df, "ischemia_binary", "Stress-induced ischemia, n (%)", "CMR findings"))
    add_row(formatted_rows, long_rows, categorical_row(df, "infarct_binary", "Myocardial infarction by LGE, n (%)", "CMR findings"))
    add_row(formatted_rows, long_rows, continuous_row(df, "lvef_percent", "LVEF, %", "CMR findings", 1))
    add_row(formatted_rows, long_rows, continuous_row(df, "lvedvi_ml_m2", "LVEDVi, mL/m2", "CMR findings", 1))
    add_row(formatted_rows, long_rows, continuous_row(df, "lvesvi_ml_m2", "LVESVi, mL/m2", "CMR findings", 1))

    add_row(formatted_rows, long_rows, categorical_row(df, "ihd_preindex_derived", "Prior ischemic heart disease, n (%)", "Historical registry data"))
    add_row(formatted_rows, long_rows, categorical_row(df, "hypertension_preindex_derived", "Hypertension, n (%)", "Historical registry data"))
    add_row(formatted_rows, long_rows, categorical_row(df, "diabetes_preindex_derived", "Diabetes mellitus, n (%)", "Historical registry data"))

    add_row(formatted_rows, long_rows, continuous_row(df, "admin_followup_years", "Available follow-up, years", "Follow-up and outcomes", 2))
    add_row(formatted_rows, long_rows, categorical_row(df, "mace_0d_event", "Primary MACE from index, n (%)", "Follow-up and outcomes"))
    add_row(formatted_rows, long_rows, categorical_row(df, "mace_30d_event", "Primary MACE after 30-day window, n (%)", "Follow-up and outcomes"))
    add_row(formatted_rows, long_rows, categorical_row(df, "hard_mace_event", "Hard MACE, n (%)", "Follow-up and outcomes"))

    formatted_df = pd.DataFrame(formatted_rows)
    long_df = pd.DataFrame(long_rows)

    # Predictor missingness before any modelling imputation.
    missing_vars = [
        ("age_years", "Age"), ("sex_male", "Sex"), ("bmi", "BMI"),
        ("lvef_percent", "LVEF"), ("lvedvi_ml_m2", "LVEDVi"),
        ("lvesvi_ml_m2", "LVESVi"), ("infarct_binary", "Infarction"),
        ("ischemia_binary", "Stress-induced ischemia"),
        ("ihd_preindex_derived", "Prior ischemic heart disease"),
        ("hypertension_preindex_derived", "Hypertension"),
        ("diabetes_preindex_derived", "Diabetes mellitus"),
    ]
    missingness_rows = []
    for variable, label in missing_vars:
        s = df[variable]
        missingness_rows.append({
            "variable": variable, "label": label, "n": len(df),
            "available_n": int(s.notna().sum()), "missing_n": int(s.isna().sum()),
            "missing_percent": round(float(s.isna().mean() * 100.0), 4),
        })
    missingness_df = pd.DataFrame(missingness_rows)

    # Supplementary endpoint sensitivity summary by ischemia status.
    endpoint_variables = [
        ("mace_0d_event", "Primary composite, 0-day window"),
        ("mace_30d_event", "Primary composite, 30-day window"),
        ("mace_90d_event", "Primary composite, 90-day window"),
        ("mace_late_revasc_event", "CV death/AMI/late PCI-CABG"),
        ("hard_mace_event", "Hard MACE: CV death or AMI"),
    ]
    sensitivity_rows = []
    for variable, label in endpoint_variables:
        for group_name, group_value in GROUPS:
            mask = group_mask(df, group_value)
            s = df.loc[mask, variable]
            available = int(s.notna().sum())
            events = int(s.eq(1).sum())
            sensitivity_rows.append({
                "endpoint": label, "variable": variable, "group": group_name,
                "group_n": int(mask.sum()), "available_n": available,
                "events": events,
                "event_percent": round(events / available * 100.0, 4) if available else np.nan,
            })
    sensitivity_df = pd.DataFrame(sensitivity_rows)

    checks: list[dict[str, Any]] = []
    def check(name: str, passed: bool, observed: Any, expected: Any, detail: str) -> None:
        checks.append({"check": name, "status": "PASS" if passed else "REVIEW",
                       "observed": observed, "expected": expected, "detail": detail})

    check("Input SHA-256", hash_match, input_hash, args.expected_sha256, "Frozen analysis dataset identity.")
    check("Cohort N", len(df) == EXPECTED_N, len(df), EXPECTED_N, "Total MR cohort.")
    check("No-ischemia N", int(df["ischemia_binary"].eq(0).sum()) == EXPECTED_NO_ISCHEMIA,
          int(df["ischemia_binary"].eq(0).sum()), EXPECTED_NO_ISCHEMIA, "Observed ischemia status.")
    check("Ischemia N", int(df["ischemia_binary"].eq(1).sum()) == EXPECTED_ISCHEMIA,
          int(df["ischemia_binary"].eq(1).sum()), EXPECTED_ISCHEMIA, "Observed ischemia status.")
    check("Indeterminate ischemia N", int(df["ischemia_binary"].isna().sum()) == EXPECTED_ISCHEMIA_MISSING,
          int(df["ischemia_binary"].isna().sum()), EXPECTED_ISCHEMIA_MISSING, "Excluded from stratified columns only.")
    check("Ischemia groups reconcile", EXPECTED_NO_ISCHEMIA + EXPECTED_ISCHEMIA + EXPECTED_ISCHEMIA_MISSING == len(df),
          EXPECTED_NO_ISCHEMIA + EXPECTED_ISCHEMIA + EXPECTED_ISCHEMIA_MISSING, len(df), "All patients accounted for.")
    check("Primary MACE count", int(df["mace_0d_event"].sum()) == 212,
          int(df["mace_0d_event"].sum()), 212, "CV death, AMI, or PCI/CABG from index.")
    check("30-day MACE count", int(df["mace_30d_event"].sum()) == 196,
          int(df["mace_30d_event"].sum()), 196, "Primary composite after 30-day window.")
    check("Hard-MACE count", int(df["hard_mace_event"].sum()) == 83,
          int(df["hard_mace_event"].sum()), 83, "CV death or AMI.")
    check("No LV mass row", not formatted_df["characteristic"].str.contains("mass", case=False).any(),
          "absent", "absent", "LV mass deliberately excluded.")
    check("No P-value column", not any("p" == str(c).lower().strip() or "p value" in str(c).lower() for c in formatted_df.columns),
          "absent", "absent", "Descriptive table only.")

    checks_df = pd.DataFrame(checks)
    review_count = int(checks_df["status"].eq("REVIEW").sum())

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(checks_df, output_dir / "table1_checks.csv")
    if review_count:
        raise RuntimeError(f"Table 1 generation stopped: {review_count} checks require review.")

    write_csv(formatted_df, output_dir / "table1_formatted.csv")
    write_csv(long_df, output_dir / "table1_numeric_long.csv")
    write_csv(missingness_df, output_dir / "table1_missingness.csv")
    write_csv(sensitivity_df, output_dir / "table1_endpoint_sensitivity.csv")

    manifest = {
        "build": {"script": Path(__file__).name, "version": SCRIPT_VERSION,
                  "created_utc": datetime.now(timezone.utc).isoformat()},
        "input": {"path": str(args.input), "sha256": input_hash, "rows": len(df)},
        "groups": {
            "total": len(df),
            "no_ischemia": int(df["ischemia_binary"].eq(0).sum()),
            "ischemia": int(df["ischemia_binary"].eq(1).sum()),
            "indeterminate": int(df["ischemia_binary"].isna().sum()),
        },
        "registry_flag_definitions": {
            "diabetes": "Any pre-index ICD-10 category E10-E14",
            "hypertension": "Any pre-index ICD-10 category I10, I11, I12, I13, or I15",
            "prior_ihd": "Any pre-index ICD-10 category I21-I25, PCI FNG*, or CABG FNA-FNE",
        },
        "format": {
            "continuous": "mean +/- sample SD",
            "categorical": "n (%) among nonmissing observations",
            "p_values": False,
        },
        "checks": {"pass": int(checks_df["status"].eq("PASS").sum()), "review": 0},
        "privacy": "Aggregate tables only; no patient-level rows or identifiers are written.",
    }
    with (output_dir / "table1_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    summary_lines = [
        "Revised Table 1 generated",
        "=" * 25,
        f"Input SHA-256: {input_hash}",
        f"Total cohort: {len(df):,}",
        f"No stress-induced ischemia: {int(df['ischemia_binary'].eq(0).sum()):,}",
        f"Stress-induced ischemia: {int(df['ischemia_binary'].eq(1).sum()):,}",
        f"Indeterminate ischemia: {int(df['ischemia_binary'].isna().sum()):,}",
        "",
        "Formatting:",
        "- Continuous variables: mean +/- SD",
        "- Categorical variables: n (%) among nonmissing observations",
        "- No between-group P values",
        "- LV mass excluded",
        "",
        "Footnote to use:",
        "Stress-induced ischemia status was indeterminate in 89 patients. These patients are included in the total cohort but excluded from the ischemia-stratified columns.",
        "",
        f"Checks: {manifest['checks']['pass']} PASS; 0 REVIEW",
    ]
    (output_dir / "table1_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"[INFO] Table 1 generated: {output_dir / 'table1_formatted.csv'}")
    print(f"[INFO] Total N={len(df)}; no ischemia={int(df['ischemia_binary'].eq(0).sum())}; "
          f"ischemia={int(df['ischemia_binary'].eq(1).sum())}; indeterminate={int(df['ischemia_binary'].isna().sum())}")
    print(f"[INFO] Checks: {manifest['checks']['pass']} PASS; 0 REVIEW")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
