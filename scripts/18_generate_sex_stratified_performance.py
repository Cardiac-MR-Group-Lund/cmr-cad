#!/usr/bin/env python3
"""Create Supplemental Table 8: binary-model discrimination by sex.

Uses the frozen out-of-fold predictions from the principal full binary Random
Forest model. No model is refitted.

The analysis reports, separately for women and men:
- number of patients
- number and percentage with MACE
- AUROC with a 95% stratified-bootstrap confidence interval

It also estimates the direct difference:
    Delta AUROC = AUROC in men - AUROC in women
with a 95% bootstrap confidence interval and a two-sided null-centered bootstrap
P value.

Default inputs
--------------
- data/processed/cmr_revision_analysis.csv
- results/models/binary_rf/oof_predictions.csv

Outputs
-------
results/tables/
- supplemental_table_8_sex_stratified_performance.csv
- supplemental_table_8_sex_stratified_performance.xlsx
- supplemental_table_8_sex_comparison.csv
- sex_stratified_performance_run_summary.txt

CSV files are semicolon-delimited with a UTF-8 BOM for Swedish Excel.
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
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from sklearn.metrics import roc_auc_score

EXPECTED_N = 2159
BOOTSTRAP_SEED = 18042


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis-data",
        type=Path,
        default=Path("data/processed/cmr_revision_analysis.csv"),
    )
    parser.add_argument(
        "--oof-predictions",
        type=Path,
        default=Path("results/models/binary_rf/oof_predictions.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/tables"),
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument(
        "--prediction-column",
        type=str,
        default=None,
        help="Optional explicit OOF probability column for the full RF model.",
    )
    parser.add_argument(
        "--outcome-column",
        type=str,
        default=None,
        help="Optional explicit binary outcome column.",
    )
    parser.add_argument(
        "--sex-column",
        type=str,
        default=None,
        help="Optional explicit binary sex column coded 0=women, 1=men.",
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


def read_csv_auto(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=detect_delimiter(path),
        encoding="utf-8-sig",
        low_memory=False,
    )


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def resolve_exact_or_alias(
    frame: pd.DataFrame,
    explicit: str | None,
    aliases: list[str],
    label: str,
) -> str:
    if explicit is not None:
        if explicit not in frame.columns:
            raise ValueError(f"Requested {label} column not found: {explicit}")
        return explicit

    for alias in aliases:
        if alias in frame.columns:
            return alias
    lookup = {normalize(column): column for column in frame.columns}
    for alias in aliases:
        if normalize(alias) in lookup:
            return lookup[normalize(alias)]
    raise ValueError(
        f"Could not identify the {label} column. Available columns: {list(frame.columns)}"
    )


def resolve_prediction_column(frame: pd.DataFrame, explicit: str | None) -> str:
    if explicit is not None:
        if explicit not in frame.columns:
            raise ValueError(f"Requested prediction column not found: {explicit}")
        return explicit

    exact_candidates = [
        "probability__full",
        "oof_probability__full",
        "oof_probability_full",
        "oof_prob_full",
        "oof_rf_full",
        "prediction__full",
        "pred_full",
        "oof_RF_codes_plus_demo_cmr",
        "oof_probs_RF_codes_plus_demo_cmr",
        "oof_rf_codes_plus_demo_cmr",
    ]
    for candidate in exact_candidates:
        if candidate in frame.columns:
            return candidate

    # Conservative fallback: prediction-like column containing 'full', or the
    # historical name for registry + demographics + CMR.
    prediction_tokens = ("prob", "pred", "oof")
    matches = []
    for column in frame.columns:
        low = column.lower()
        is_prediction = any(token in low for token in prediction_tokens)
        is_full = (
            "full" in low
            or "codes_plus_demo_cmr" in low
            or "codes_demo_cmr" in low
            or ("registry" in low and "cmr" in low)
        )
        excluded = any(token in low for token in ("fold", "event", "label", "outcome"))
        if is_prediction and is_full and not excluded:
            matches.append(column)

    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(
            "Could not identify the full-model OOF probability column. "
            f"Available columns: {list(frame.columns)}"
        )
    raise ValueError(
        "Multiple possible full-model prediction columns were found: "
        f"{matches}. Rerun with --prediction-column COLUMN_NAME."
    )


def stratified_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    if len(positive) == 0 or len(negative) == 0:
        raise ValueError("Each sex subgroup must contain events and non-events")
    indices = np.concatenate([
        rng.choice(positive, size=len(positive), replace=True),
        rng.choice(negative, size=len(negative), replace=True),
    ])
    rng.shuffle(indices)
    return indices


def bootstrap_analysis(
    y_women: np.ndarray,
    p_women: np.ndarray,
    y_men: np.ndarray,
    p_men: np.ndarray,
    repeats: int,
):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    auroc_women = np.empty(repeats)
    auroc_men = np.empty(repeats)
    differences = np.empty(repeats)

    for repeat in range(repeats):
        women_indices = stratified_indices(y_women, rng)
        men_indices = stratified_indices(y_men, rng)
        auroc_women[repeat] = roc_auc_score(
            y_women[women_indices], p_women[women_indices]
        )
        auroc_men[repeat] = roc_auc_score(
            y_men[men_indices], p_men[men_indices]
        )
        differences[repeat] = auroc_men[repeat] - auroc_women[repeat]

    observed_women = float(roc_auc_score(y_women, p_women))
    observed_men = float(roc_auc_score(y_men, p_men))
    observed_difference = observed_men - observed_women

    centered = differences - observed_difference
    count = int(np.count_nonzero(np.abs(centered) >= abs(observed_difference)))
    p_value = float((count + 1) / (repeats + 1))

    return {
        "women": {
            "estimate": observed_women,
            "ci_lower": float(np.quantile(auroc_women, 0.025)),
            "ci_upper": float(np.quantile(auroc_women, 0.975)),
        },
        "men": {
            "estimate": observed_men,
            "ci_lower": float(np.quantile(auroc_men, 0.025)),
            "ci_upper": float(np.quantile(auroc_men, 0.975)),
        },
        "difference": {
            "estimate": observed_difference,
            "ci_lower": float(np.quantile(differences, 0.025)),
            "ci_upper": float(np.quantile(differences, 0.975)),
            "p_value": p_value,
        },
    }


def format_p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def create_xlsx(table: pd.DataFrame, comparison: pd.DataFrame, path: Path):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Supplemental Table 8"
    sheet.sheet_view.showGridLines = False

    title = "Supplemental Table 8. Binary-Model Discrimination Stratified by Sex"
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
                horizontal="right" if column_index in (2, 3, 4) else "left",
                vertical="center",
            )
            if row_index % 2 == 0:
                cell.fill = PatternFill("solid", fgColor="EAF2F8")

    widths = [15, 14, 14, 16, 25]
    for index, width in enumerate(widths, start=1):
        sheet.column_dimensions[get_column_letter(index)].width = width

    comparison_row = 7
    sheet.cell(comparison_row, 1, "Difference in AUROC (men - women)").font = Font(bold=True)
    sheet.cell(comparison_row, 2, comparison.iloc[0]["Delta AUROC (95% CI)"])
    sheet.cell(comparison_row, 3, "P value").font = Font(bold=True)
    sheet.cell(comparison_row, 4, comparison.iloc[0]["P value"])

    footnote_row = 9
    footnote = (
        "Performance was evaluated using held-out out-of-fold predictions from the full "
        "Random Forest model. Confidence intervals and the sex difference in AUROC were "
        "estimated using 10,000 outcome-stratified bootstrap repetitions. The subgroup "
        "analysis was exploratory."
    )
    sheet.merge_cells(
        start_row=footnote_row,
        start_column=1,
        end_row=footnote_row,
        end_column=len(table.columns),
    )
    footnote_cell = sheet.cell(footnote_row, 1, footnote)
    footnote_cell.font = Font(italic=True, size=9, color="666666")
    footnote_cell.alignment = Alignment(wrap_text=True, vertical="top")
    sheet.row_dimensions[footnote_row].height = 45

    workbook.save(path)


def validate_xlsx(path: Path):
    workbook = load_workbook(path, read_only=True, data_only=True)
    sheet = workbook["Supplemental Table 8"]
    if sheet["A4"].value != "Women" or sheet["A5"].value != "Men":
        raise ValueError("Excel table validation failed")
    workbook.close()


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.analysis_data.exists():
        raise FileNotFoundError(args.analysis_data)
    if not args.oof_predictions.exists():
        raise FileNotFoundError(args.oof_predictions)

    analysis = read_csv_auto(args.analysis_data)
    oof = read_csv_auto(args.oof_predictions)

    sex_column = resolve_exact_or_alias(
        analysis,
        args.sex_column,
        ["sex_male", "sex_numerical", "sex", "male"],
        "sex",
    )
    outcome_column_analysis = resolve_exact_or_alias(
        analysis,
        args.outcome_column,
        [
            "mace_0d_event",
            "mace_event",
            "binary_mace_event",
            "event",
            "y_mace",
            "y_mace_5y",
        ],
        "binary outcome",
    )
    prediction_column = resolve_prediction_column(oof, args.prediction_column)

    # Prefer ID-based alignment. If the OOF file also stores the outcome, the
    # frozen analysis dataset remains the source of truth for sex and outcome.
    if "analysis_id" in analysis.columns and "analysis_id" in oof.columns:
        combined = analysis[["analysis_id", sex_column, outcome_column_analysis]].merge(
            oof[["analysis_id", prediction_column]],
            on="analysis_id",
            how="inner",
            validate="one_to_one",
        )
    else:
        if len(analysis) != len(oof):
            raise ValueError(
                "The files lack a shared analysis_id and have different row counts; "
                "safe alignment is impossible."
            )
        combined = pd.DataFrame({
            sex_column: analysis[sex_column].to_numpy(),
            outcome_column_analysis: analysis[outcome_column_analysis].to_numpy(),
            prediction_column: oof[prediction_column].to_numpy(),
        })

    if len(combined) != EXPECTED_N:
        raise ValueError(f"Expected N={EXPECTED_N}; observed {len(combined)}")

    sex = pd.to_numeric(combined[sex_column], errors="coerce")
    outcome = pd.to_numeric(combined[outcome_column_analysis], errors="coerce")
    probability = pd.to_numeric(combined[prediction_column], errors="coerce")

    if sex.isna().any() or not sex.isin([0, 1]).all():
        raise ValueError(f"Sex column {sex_column} must be coded 0=women, 1=men")
    if outcome.isna().any() or not outcome.isin([0, 1]).all():
        raise ValueError(f"Outcome column {outcome_column_analysis} is not binary")
    if probability.isna().any() or ((probability < 0) | (probability > 1)).any():
        raise ValueError(f"Prediction column {prediction_column} is not a valid probability")

    women_mask = sex.eq(0).to_numpy()
    men_mask = sex.eq(1).to_numpy()
    y = outcome.astype(int).to_numpy()
    p = probability.to_numpy(float)

    results = bootstrap_analysis(
        y[women_mask], p[women_mask],
        y[men_mask], p[men_mask],
        args.bootstrap_repeats,
    )

    table_rows = []
    for label, mask, key in [
        ("Women", women_mask, "women"),
        ("Men", men_mask, "men"),
    ]:
        subgroup_y = y[mask]
        estimate = results[key]
        table_rows.append({
            "Sex": label,
            "Patients, n": int(mask.sum()),
            "Events, n": int(subgroup_y.sum()),
            "Event rate, %": float(subgroup_y.mean() * 100),
            "AUROC (95% CI)": (
                f"{estimate['estimate']:.3f} "
                f"({estimate['ci_lower']:.3f} to {estimate['ci_upper']:.3f})"
            ),
        })
    table = pd.DataFrame(table_rows)

    difference = results["difference"]
    comparison = pd.DataFrame([{
        "Comparison": "Men minus women",
        "Delta AUROC": difference["estimate"],
        "CI lower": difference["ci_lower"],
        "CI upper": difference["ci_upper"],
        "Delta AUROC (95% CI)": (
            f"{difference['estimate']:.3f} "
            f"({difference['ci_lower']:.3f} to {difference['ci_upper']:.3f})"
        ),
        "P value numeric": difference["p_value"],
        "P value": format_p(difference["p_value"]),
        "Bootstrap repetitions": args.bootstrap_repeats,
        "Bootstrap seed": BOOTSTRAP_SEED,
    }])

    table_path = output_dir / "supplemental_table_8_sex_stratified_performance.csv"
    comparison_path = output_dir / "supplemental_table_8_sex_comparison.csv"
    xlsx_path = output_dir / "supplemental_table_8_sex_stratified_performance.xlsx"

    table.to_csv(
        table_path,
        index=False,
        sep=";",
        encoding="utf-8-sig",
    )
    comparison.to_csv(
        comparison_path,
        index=False,
        sep=";",
        encoding="utf-8-sig",
        float_format="%.12g",
    )
    create_xlsx(table, comparison, xlsx_path)
    validate_xlsx(xlsx_path)

    summary_lines = [
        "Sex-stratified binary-model performance",
        "=" * 39,
        f"N={len(combined)}; events={int(y.sum())}",
        f"Sex column: {sex_column} (0=women, 1=men)",
        f"Outcome column: {outcome_column_analysis}",
        f"Prediction column: {prediction_column}",
        f"Bootstrap repetitions: {args.bootstrap_repeats}; seed={BOOTSTRAP_SEED}",
        "",
    ]
    for row in table.itertuples(index=False):
        summary_lines.append(
            f"{row[0]}: N={row[1]}, events={row[2]} ({row[3]:.1f}%); {row[4]}"
        )
    summary_lines.append(
        f"Men minus women: delta AUROC {difference['estimate']:.4f} "
        f"(95% CI {difference['ci_lower']:.4f} to {difference['ci_upper']:.4f}); "
        f"P={format_p(difference['p_value'])}"
    )
    (output_dir / "sex_stratified_performance_run_summary.txt").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "exploratory sex-stratified discrimination",
        "model": "full binary Random Forest",
        "n": len(combined),
        "events": int(y.sum()),
        "sex_column": sex_column,
        "outcome_column": outcome_column_analysis,
        "prediction_column": prediction_column,
        "bootstrap_repeats": args.bootstrap_repeats,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    (output_dir / "supplemental_table_8_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"[DONE] {table_path}")
    print(f"[DONE] {comparison_path}")
    print(f"[DONE] {xlsx_path}")
    print(
        f"[INFO] Women: N={int(women_mask.sum())}, events={int(y[women_mask].sum())}; "
        f"Men: N={int(men_mask.sum())}, events={int(y[men_mask].sum())}"
    )
    print(
        f"[INFO] Delta AUROC (men - women)={difference['estimate']:.4f}; "
        f"P={format_p(difference['p_value'])}"
    )


if __name__ == "__main__":
    main()
