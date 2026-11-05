#!/usr/bin/env python3
"""
03_freeze_registry_vocabulary.py

Freeze the registry-derived predictor vocabulary for the revised CMR-MACE
analysis.

Input:
    data/processed/cmr_revision_analysis.csv

Outputs:
    config/registry_vocabulary.json
    results/audit/registry_vocabulary/registry_vocabulary.csv
    results/audit/registry_vocabulary/registry_vocabulary_checks.csv
    results/audit/registry_vocabulary/registry_vocabulary_manifest.json
    results/audit/registry_vocabulary/registry_vocabulary_summary.txt

Rules:
- Use pre-index code lists only.
- ICD-10 codes: uppercase, remove punctuation, retain first 3 alphanumeric
  characters (category level).
- KVA procedure codes: uppercase, remove punctuation, retain the full
  alphanumeric code.
- Count presence once per patient, irrespective of repeated appearances.
- Retain codes present in at least 10 MR patients.
- Use deterministic ordering: ICD first, then KVA; alphabetical within system.
- No outcome information is used in vocabulary construction.
- No patient-level feature matrix is written by this script.

Run from the repository root:
    python scripts/03_freeze_registry_vocabulary.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_VERSION = "1.0.0"
DEFAULT_INPUT = Path("data/processed/cmr_revision_analysis.csv")
DEFAULT_CONFIG_OUTPUT = Path("config/registry_vocabulary.json")
DEFAULT_AUDIT_DIR = Path("results/audit/registry_vocabulary")
EXPECTED_INPUT_SHA256 = "e94b8aebe4c4939b0b3def943879539f4f242ba7ef39fd7ac929407578f7db9c"
EXPECTED_N = 2159
DEFAULT_MIN_PATIENTS = 10
EXPECTED_ICD_FEATURES = 34
EXPECTED_KVA_FEATURES = 28
BLANK_STRINGS = {"", "nan", "none", "null", "na", "n/a", "nat", "<na>"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the registry predictor vocabulary."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--config-output", type=Path, default=DEFAULT_CONFIG_OUTPUT)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--min-patients", type=int, default=DEFAULT_MIN_PATIENTS)
    parser.add_argument("--expected-sha256", default=EXPECTED_INPUT_SHA256)
    parser.add_argument("--expected-n", type=int, default=EXPECTED_N)
    parser.add_argument("--allow-input-hash-mismatch", action="store_true")
    parser.add_argument(
        "--allow-feature-count-mismatch",
        action="store_true",
        help="Allow counts other than 34 ICD and 28 KVA after deliberate review.",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_canonical_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        sample = handle.read(131072)
    try:
        return csv.Sniffer().sniff(sample, delimiters=";,\t|").delimiter
    except csv.Error:
        first_line = sample.splitlines()[0] if sample else ""
        return max([";", ",", "\t", "|"], key=first_line.count)


def clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in BLANK_STRINGS else text


def split_code_list(value: Any) -> set[str]:
    text = clean_text(value).upper()
    if not text:
        return set()
    return {part for part in re.split(r"[;,\s]+", text) if part}


def normalize_icd(code: str) -> str:
    compact = re.sub(r"[^A-Z0-9]", "", code.upper())
    return compact[:3] if len(compact) >= 3 else ""


def normalize_kva(code: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", code.upper())


def patient_code_sets(series: pd.Series, code_system: str) -> list[set[str]]:
    normalizer = normalize_icd if code_system == "ICD10" else normalize_kva
    result: list[set[str]] = []
    for value in series:
        codes = {normalizer(code) for code in split_code_list(value)}
        codes.discard("")
        result.append(codes)
    return result


def count_patient_prevalence(code_sets: list[set[str]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for codes in code_sets:
        counts.update(codes)
    return counts


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8", float_format="%.12g")


def main() -> int:
    args = parse_args()
    source = args.input.resolve()
    config_output = args.config_output.resolve()
    audit_dir = args.audit_dir.resolve()

    if not source.exists():
        print(f"[ERROR] Input file not found: {source}", file=sys.stderr)
        return 2
    if args.min_patients < 1:
        print("[ERROR] --min-patients must be at least 1.", file=sys.stderr)
        return 2

    source_hash = sha256_file(source)
    hash_match = source_hash.lower() == args.expected_sha256.lower()
    if not hash_match and not args.allow_input_hash_mismatch:
        print("[ERROR] Processed dataset SHA-256 mismatch.", file=sys.stderr)
        print(f"[ERROR] Expected: {args.expected_sha256}", file=sys.stderr)
        print(f"[ERROR] Observed: {source_hash}", file=sys.stderr)
        return 3

    delimiter = detect_delimiter(source)
    df = pd.read_csv(
        source,
        sep=delimiter,
        dtype={"analysis_id": "Int64", "pre_icd_codes": "string", "pre_kva_codes": "string"},
        encoding="utf-8-sig",
        low_memory=False,
    )

    required = {"analysis_id", "pre_icd_codes", "pre_kva_codes"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if len(df) != args.expected_n:
        raise ValueError(f"Expected N={args.expected_n}; observed N={len(df)}")
    if df["analysis_id"].isna().any() or df["analysis_id"].duplicated().any():
        raise ValueError("analysis_id contains missing or duplicated values.")

    print(f"[INFO] Script version: {SCRIPT_VERSION}")
    print(f"[INFO] Input SHA-256 match: {hash_match}")
    print(f"[INFO] Cohort N: {len(df):,}")
    print(f"[INFO] Minimum patient count: {args.min_patients}")

    icd_sets = patient_code_sets(df["pre_icd_codes"], "ICD10")
    kva_sets = patient_code_sets(df["pre_kva_codes"], "KVA")
    icd_counts = count_patient_prevalence(icd_sets)
    kva_counts = count_patient_prevalence(kva_sets)

    icd_codes = sorted(code for code, count in icd_counts.items() if count >= args.min_patients)
    kva_codes = sorted(code for code, count in kva_counts.items() if count >= args.min_patients)

    entries: list[dict[str, Any]] = []
    position = 1
    for system, prefix, codes, counts in [
        ("ICD10", "pre_icd", icd_codes, icd_counts),
        ("KVA", "pre_kva", kva_codes, kva_counts),
    ]:
        for code in codes:
            count = int(counts[code])
            entries.append({
                "position": position,
                "feature_name": f"{prefix}__{code}",
                "code_system": system,
                "normalized_code": code,
                "patient_count": count,
                "patient_percent": round(count / len(df) * 100.0, 6),
            })
            position += 1

    # Hash only the substantive frozen specification, not paths or timestamps.
    frozen_specification = {
        "minimum_patient_count": args.min_patients,
        "cohort_n": len(df),
        "icd_rule": "uppercase; remove non-alphanumeric characters; first 3 characters",
        "kva_rule": "uppercase; remove non-alphanumeric characters; retain full code",
        "feature_order": "ICD10 alphabetical, then KVA alphabetical",
        "features": [
            {
                "feature_name": item["feature_name"],
                "code_system": item["code_system"],
                "normalized_code": item["normalized_code"],
                "patient_count": item["patient_count"],
            }
            for item in entries
        ],
    }
    vocabulary_hash = sha256_canonical_json(frozen_specification)

    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, observed: Any, expected: Any, detail: str) -> None:
        checks.append({
            "check": name,
            "status": "PASS" if passed else "REVIEW",
            "observed": observed,
            "expected": expected,
            "detail": detail,
        })

    check("Input SHA-256", hash_match, source_hash, args.expected_sha256,
          "Frozen processed-dataset identity.")
    check("Cohort size", len(df) == args.expected_n, len(df), args.expected_n,
          "Rows in processed MR dataset.")
    check("Unique analysis_id", df["analysis_id"].nunique() == len(df),
          int(df["analysis_id"].nunique()), len(df), "One row per analysis ID.")
    check("ICD threshold respected", all(icd_counts[c] >= args.min_patients for c in icd_codes),
          min((icd_counts[c] for c in icd_codes), default=None), f">={args.min_patients}",
          "Every retained ICD feature meets the threshold.")
    check("KVA threshold respected", all(kva_counts[c] >= args.min_patients for c in kva_codes),
          min((kva_counts[c] for c in kva_codes), default=None), f">={args.min_patients}",
          "Every retained KVA feature meets the threshold.")
    check("ICD feature count", len(icd_codes) == EXPECTED_ICD_FEATURES,
          len(icd_codes), EXPECTED_ICD_FEATURES, "Expected from prior aggregate audit.")
    check("KVA feature count", len(kva_codes) == EXPECTED_KVA_FEATURES,
          len(kva_codes), EXPECTED_KVA_FEATURES, "Expected from prior aggregate audit.")
    check("Total feature count", len(entries) == EXPECTED_ICD_FEATURES + EXPECTED_KVA_FEATURES,
          len(entries), EXPECTED_ICD_FEATURES + EXPECTED_KVA_FEATURES,
          "Frozen registry feature count.")
    check("Unique feature names", len({x["feature_name"] for x in entries}) == len(entries),
          len({x["feature_name"] for x in entries}), len(entries),
          "No duplicate feature names.")
    check("No post-index columns used", True, "pre_icd_codes; pre_kva_codes",
          "pre-index only", "Vocabulary construction does not inspect post-index codes.")
    check("No outcome columns used", True, "none", "none",
          "Vocabulary construction is independent of outcome status.")

    checks_df = pd.DataFrame(checks)
    count_reviews = checks_df[checks_df["check"].isin(
        ["ICD feature count", "KVA feature count", "Total feature count"]
    )]["status"].eq("REVIEW").sum()
    other_reviews = checks_df[~checks_df["check"].isin(
        ["ICD feature count", "KVA feature count", "Total feature count"]
    )]["status"].eq("REVIEW").sum()

    audit_dir.mkdir(parents=True, exist_ok=True)
    write_csv(checks_df, audit_dir / "registry_vocabulary_checks.csv")

    if other_reviews:
        raise RuntimeError(
            f"Vocabulary build stopped: {other_reviews} non-count validation check(s) require review."
        )
    if count_reviews and not args.allow_feature_count_mismatch:
        raise RuntimeError(
            f"Vocabulary build stopped: observed ICD={len(icd_codes)}, KVA={len(kva_codes)}; "
            "expected ICD=34, KVA=28. Inspect registry_vocabulary_checks.csv. "
            "Use --allow-feature-count-mismatch only after deliberate review."
        )

    vocabulary = {
        "schema_version": "1.0",
        "script_version": SCRIPT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_dataset": {
            "path": str(args.input),
            "sha256": source_hash,
            "cohort_n": len(df),
        },
        "rules": {
            "source_columns": ["pre_icd_codes", "pre_kva_codes"],
            "minimum_patient_count": args.min_patients,
            "patient_level_representation": "binary presence/absence",
            "icd_normalization": frozen_specification["icd_rule"],
            "kva_normalization": frozen_specification["kva_rule"],
            "ordering": frozen_specification["feature_order"],
            "outcome_information_used": False,
        },
        "counts": {
            "icd_features": len(icd_codes),
            "kva_features": len(kva_codes),
            "total_features": len(entries),
            "patients_with_any_pre_icd": int(sum(bool(x) for x in icd_sets)),
            "patients_with_any_pre_kva": int(sum(bool(x) for x in kva_sets)),
            "unique_icd_codes_before_threshold": len(icd_counts),
            "unique_kva_codes_before_threshold": len(kva_counts),
        },
        "vocabulary_sha256": vocabulary_hash,
        "features": entries,
    }

    config_output.parent.mkdir(parents=True, exist_ok=True)
    with config_output.open("w", encoding="utf-8") as handle:
        json.dump(vocabulary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    config_file_hash = sha256_file(config_output)

    vocabulary_df = pd.DataFrame(entries)
    write_csv(vocabulary_df, audit_dir / "registry_vocabulary.csv")

    # Aggregate threshold diagnostics without exporting rare code identities.
    threshold_rows = []
    for system, counts in [("ICD10", icd_counts), ("KVA", kva_counts)]:
        distribution = Counter(counts.values())
        threshold_rows.extend({
            "code_system": system,
            "patient_count": int(count),
            "number_of_codes": int(number),
            "retained_at_threshold": bool(count >= args.min_patients),
        } for count, number in sorted(distribution.items()))
    write_csv(pd.DataFrame(threshold_rows), audit_dir / "code_prevalence_distribution.csv")

    manifest = {
        "build": {
            "script": Path(__file__).name,
            "script_version": SCRIPT_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
        "input": {
            "path": str(args.input),
            "sha256": source_hash,
            "delimiter": delimiter,
            "rows": len(df),
        },
        "outputs": {
            "config_path": str(args.config_output),
            "config_file_sha256": config_file_hash,
            "vocabulary_specification_sha256": vocabulary_hash,
            "audit_csv": str(audit_dir / "registry_vocabulary.csv"),
        },
        "counts": vocabulary["counts"],
        "checks": {
            "pass": int(checks_df["status"].eq("PASS").sum()),
            "review": int(checks_df["status"].eq("REVIEW").sum()),
        },
        "privacy": (
            "Aggregate feature definitions only. Codes below the minimum patient threshold "
            "are not identified in outputs. No patient-level matrix is written."
        ),
    }
    with (audit_dir / "registry_vocabulary_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    summary = [
        "Frozen registry vocabulary",
        "=" * 26,
        f"Input dataset: {args.input}",
        f"Input SHA-256: {source_hash}",
        f"Input hash matched expected: {hash_match}",
        f"Cohort N: {len(df):,}",
        f"Minimum patient count: {args.min_patients}",
        "",
        f"ICD-10 features: {len(icd_codes)}",
        f"KVA features: {len(kva_codes)}",
        f"Total registry features: {len(entries)}",
        f"Patients with any pre-index ICD code: {vocabulary['counts']['patients_with_any_pre_icd']:,}",
        f"Patients with any pre-index KVA code: {vocabulary['counts']['patients_with_any_pre_kva']:,}",
        "",
        f"Vocabulary specification SHA-256: {vocabulary_hash}",
        f"Config file SHA-256: {config_file_hash}",
        f"Config file: {args.config_output}",
        "",
        f"Validation checks: {manifest['checks']['pass']} PASS; {manifest['checks']['review']} REVIEW",
        "No outcome information or post-index code lists were used.",
        "No patient-level feature matrix was written.",
    ]
    (audit_dir / "registry_vocabulary_summary.txt").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )

    print(f"[INFO] ICD features: {len(icd_codes)}")
    print(f"[INFO] KVA features: {len(kva_codes)}")
    print(f"[INFO] Total features: {len(entries)}")
    print(f"[INFO] Vocabulary SHA-256: {vocabulary_hash}")
    print(f"[INFO] Wrote: {config_output}")
    print(f"[INFO] Audit outputs: {audit_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
