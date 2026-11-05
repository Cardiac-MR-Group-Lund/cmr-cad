#!/usr/bin/env python3
"""Create Supplemental Table 6 from the frozen registry vocabulary.

The table is sorted by patient frequency, highest first. The frozen vocabulary
is used exactly as stored; no features are regenerated or filtered.

Input:
    config/registry_vocabulary.json

Outputs:
    results/tables/supplemental_table_6_registry_features.csv
    results/tables/supplemental_table_6_registry_features.xlsx

The CSV is semicolon-delimited with UTF-8 BOM for direct opening in Swedish
Excel. The displayed table omits the redundant technical feature name, while a
full archival CSV retaining that field is also produced.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

EXPECTED_FEATURES = 62
EXPECTED_ICD10 = 34
EXPECTED_KVA = 28


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("config/registry_vocabulary.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/tables"),
    )
    return parser.parse_args()


def find_feature_list(payload):
    """Recursively locate the list containing frozen feature records."""
    if isinstance(payload, list):
        if payload and isinstance(payload[0], dict) and "feature_name" in payload[0]:
            return payload
        for value in payload:
            result = find_feature_list(value)
            if result is not None:
                return result
    elif isinstance(payload, dict):
        for value in payload.values():
            result = find_feature_list(value)
            if result is not None:
                return result
    return None


def create_xlsx(table: pd.DataFrame, path: Path):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Supplemental Table 6"
    sheet.sheet_view.showGridLines = False

    title = "Supplemental Table 6. Registry-Derived Features Included in the Prediction Models"
    sheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(table.columns))
    title_cell = sheet.cell(1, 1, title)
    title_cell.font = Font(bold=True, size=12, color="FFFFFF")
    title_cell.fill = PatternFill("solid", fgColor="1F4E78")
    title_cell.alignment = Alignment(horizontal="left", vertical="center")
    sheet.row_dimensions[1].height = 24

    for column_index, heading in enumerate(table.columns, start=1):
        cell = sheet.cell(3, column_index, heading)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="5B9BD5")
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    for row_index, row in enumerate(table.itertuples(index=False), start=4):
        for column_index, value in enumerate(row, start=1):
            cell = sheet.cell(row_index, column_index, value)
            cell.alignment = Alignment(
                horizontal="right" if column_index in (1, 4, 5) else "left",
                vertical="center",
            )
            if row_index % 2 == 0:
                cell.fill = PatternFill("solid", fgColor="EAF2F8")

    for row_index in range(4, 4 + len(table)):
        sheet.cell(row_index, 1).number_format = "0"
        sheet.cell(row_index, 4).number_format = "0"
        sheet.cell(row_index, 5).number_format = "0.0"

    widths = [10, 16, 18, 18, 18]
    for index, width in enumerate(widths, start=1):
        sheet.column_dimensions[get_column_letter(index)].width = width

    sheet.freeze_panes = "A4"
    sheet.auto_filter.ref = f"A3:E{3 + len(table)}"

    footnote_row = 5 + len(table)
    footnote = (
        "Registry-derived features represent the presence of qualifying diagnosis or procedure "
        "codes during the 10-year period preceding index CMR. ICD-10 diagnosis codes were grouped "
        "at the three-character level. Only features recorded in at least 10 patients were included. "
        "Features are ordered by prevalence in the study cohort."
    )
    sheet.merge_cells(
        start_row=footnote_row,
        start_column=1,
        end_row=footnote_row,
        end_column=len(table.columns),
    )
    cell = sheet.cell(footnote_row, 1, footnote)
    cell.font = Font(italic=True, size=9, color="666666")
    cell.alignment = Alignment(wrap_text=True, vertical="top")
    sheet.row_dimensions[footnote_row].height = 48

    workbook.save(path)


def validate_xlsx(path: Path, expected_rows: int):
    workbook = load_workbook(path, read_only=True, data_only=True)
    sheet = workbook["Supplemental Table 6"]
    observed_rows = sum(
        1 for row in sheet.iter_rows(min_row=4, max_row=3 + expected_rows, values_only=True)
        if row[0] is not None
    )
    workbook.close()
    if observed_rows != expected_rows:
        raise ValueError(
            f"Workbook validation failed: expected {expected_rows} data rows, found {observed_rows}"
        )


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with args.input.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    features = find_feature_list(payload)
    if features is None:
        raise ValueError("Could not locate the feature list in the vocabulary JSON")
    if len(features) != EXPECTED_FEATURES:
        raise ValueError(f"Expected {EXPECTED_FEATURES} features; found {len(features)}")

    sorted_features = sorted(
        features,
        key=lambda item: (-int(item["patient_count"]), int(item["position"])),
    )

    display_rows = []
    archive_rows = []
    for rank, feature in enumerate(sorted_features, start=1):
        code_system = str(feature["code_system"])
        row = {
            "Rank": rank,
            "Code system": code_system,
            "Code": str(feature["normalized_code"]),
            "Patients, n": int(feature["patient_count"]),
            "Patients, %": float(feature["patient_percent"]),
        }
        display_rows.append(row)
        archive_rows.append({
            **row,
            "Frozen position": int(feature["position"]),
            "Technical feature name": str(feature["feature_name"]),
        })

    table = pd.DataFrame(display_rows)
    archive = pd.DataFrame(archive_rows)

    if table["Rank"].tolist() != list(range(1, EXPECTED_FEATURES + 1)):
        raise ValueError("Displayed ranks are incomplete")
    if archive["Technical feature name"].duplicated().any():
        raise ValueError("Duplicate frozen feature names found")
    if (table["Patients, n"] < 10).any():
        raise ValueError("A feature has fewer than 10 patients")
    if not table["Patients, n"].is_monotonic_decreasing:
        raise ValueError("Table is not sorted by decreasing patient count")

    icd_count = int(table["Code system"].eq("ICD10").sum())
    kva_count = int(table["Code system"].eq("KVA").sum())
    if icd_count != EXPECTED_ICD10 or kva_count != EXPECTED_KVA:
        raise ValueError(
            f"Unexpected code-system counts: ICD-10={icd_count}, KVA={kva_count}"
        )

    csv_path = output_dir / "supplemental_table_6_registry_features.csv"
    xlsx_path = output_dir / "supplemental_table_6_registry_features.xlsx"
    archive_path = output_dir / "supplemental_table_6_registry_features_archive.csv"

    table.to_csv(
        csv_path,
        index=False,
        sep=";",
        encoding="utf-8-sig",
        float_format="%.1f",
    )
    archive.to_csv(
        archive_path,
        index=False,
        sep=";",
        encoding="utf-8-sig",
        float_format="%.1f",
    )
    create_xlsx(table, xlsx_path)
    validate_xlsx(xlsx_path, EXPECTED_FEATURES)

    print(f"[DONE] {csv_path}")
    print(f"[DONE] {xlsx_path}")
    print(f"[DONE] {archive_path}")
    print(
        f"[INFO] Features: {len(table)} | ICD-10: {icd_count} | KVA: {kva_count} | "
        f"sorted by decreasing prevalence"
    )


if __name__ == "__main__":
    main()
