#!/usr/bin/env python3
"""Merge frozen survival-model runs without refitting any model.

Reads OOF risk scores from:
- results/models/survival_comparison/                       (RSF)
- results/models/survival_comparison_ridge_extended/        (Ridge)
- results/models/survival_comparison_lasso_demo_cmr/        (LASSO reference)
- results/models/survival_comparison_lasso_full/            (LASSO full)
- results/models/survival_comparison_elastic_net_demo_cmr/  (EN reference)
- results/models/survival_comparison_elastic_net_full/      (EN full)

Recomputes pooled Harrell C-indices and paired bootstrap confidence intervals
from the saved, patient-level OOF risk scores. No model is fitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

SEED = 10042
EXPECTED_N = 2159
EXPECTED_EVENTS = 196
ENDPOINT = "primary_30d"
MODEL_ORDER = ["ridge", "lasso", "elastic_net", "rsf"]
FEATURE_ORDER = ["demo_cmr", "full"]
MODEL_LABELS = {
    "ridge": "Ridge Cox",
    "lasso": "LASSO Cox",
    "elastic_net": "Elastic-net Cox",
    "rsf": "Random Survival Forest",
}
FEATURE_LABELS = {
    "demo_cmr": "Demographics + CMR",
    "full": "Registry + demographics + CMR",
}
DEFAULT_RUNS = {
    ("rsf", "demo_cmr"): Path("results/models/survival_comparison"),
    ("rsf", "full"): Path("results/models/survival_comparison"),
    ("ridge", "demo_cmr"): Path("results/models/survival_comparison_ridge_extended"),
    ("ridge", "full"): Path("results/models/survival_comparison_ridge_extended"),
    ("lasso", "demo_cmr"): Path("results/models/survival_comparison_lasso_demo_cmr"),
    ("lasso", "full"): Path("results/models/survival_comparison_lasso_full"),
    ("elastic_net", "demo_cmr"): Path("results/models/survival_comparison_elastic_net_demo_cmr"),
    ("elastic_net", "full"): Path("results/models/survival_comparison_elastic_net_full"),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--analysis-data", type=Path,
                   default=Path("data/processed/cmr_revision_analysis.csv"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/models/survival_comparison_merged"))
    p.add_argument("--bootstrap-repeats", type=int, default=5000)
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv_auto(path: Path) -> pd.DataFrame:
    first = path.open("r", encoding="utf-8-sig").readline()
    sep = ";" if first.count(";") > first.count(",") else ","
    return pd.read_csv(path, sep=sep, encoding="utf-8-sig", low_memory=False)


def cindex(y, risk):
    return float(concordance_index_censored(y["event"], y["time"], risk)[0])


def load_risk(run_dir: Path, model: str, feature: str) -> pd.DataFrame:
    path = run_dir / "oof_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = read_csv_auto(path)
    col = f"risk__{ENDPOINT}__{model}__{feature}"
    required = {"analysis_id", "outer_fold", col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing {sorted(missing)}")
    out = df[["analysis_id", "outer_fold", col]].copy()
    out = out.rename(columns={col: "risk"})
    out["model"] = model
    out["feature_set"] = feature
    return out


def bootstrap(y, risks, repeats):
    rng = np.random.default_rng(SEED)
    event_idx = np.flatnonzero(y["event"])
    censor_idx = np.flatnonzero(~y["event"])
    draws = {k: np.empty(repeats) for k in risks}
    for i in range(repeats):
        idx = np.concatenate([
            rng.choice(event_idx, len(event_idx), replace=True),
            rng.choice(censor_idx, len(censor_idx), replace=True),
        ])
        rng.shuffle(idx)
        for key, risk in risks.items():
            draws[key][i] = cindex(y[idx], risk[idx])

    intervals = {
        key: tuple(float(x) for x in np.quantile(values, [0.025, 0.975]))
        for key, values in draws.items()
    }
    comparisons = []

    for model in MODEL_ORDER:
        ref = (model, "demo_cmr")
        full = (model, "full")
        delta = draws[full] - draws[ref]
        comparisons.append({
            "comparison_type": "registry_increment",
            "model": model,
            "comparison": f"{MODEL_LABELS[model]}: full minus demographics + CMR",
            "estimate": cindex(y, risks[full]) - cindex(y, risks[ref]),
            "ci_lower": float(np.quantile(delta, 0.025)),
            "ci_upper": float(np.quantile(delta, 0.975)),
        })

    for feature in FEATURE_ORDER:
        for cox in ["ridge", "lasso", "elastic_net"]:
            delta = draws[("rsf", feature)] - draws[(cox, feature)]
            comparisons.append({
                "comparison_type": "rsf_vs_cox",
                "model": f"rsf_minus_{cox}",
                "comparison": f"RSF minus {MODEL_LABELS[cox]} ({FEATURE_LABELS[feature]})",
                "estimate": cindex(y, risks[("rsf", feature)]) - cindex(y, risks[(cox, feature)]),
                "ci_lower": float(np.quantile(delta, 0.025)),
                "ci_upper": float(np.quantile(delta, 0.975)),
            })
    return intervals, pd.DataFrame(comparisons)


def plot(perf: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    colors = {"demo_cmr": "#1b9e77", "full": "#1f4e79"}
    y = 0
    ticks, labels = [], []
    for model in MODEL_ORDER:
        for feature in FEATURE_ORDER:
            r = perf[(perf.model == model) & (perf.feature_set == feature)].iloc[0]
            ax.errorbar(
                r.c_index, y,
                xerr=[[r.c_index-r.c_index_ci_lower], [r.c_index_ci_upper-r.c_index]],
                fmt="o", color=colors[feature], capsize=3,
            )
            ticks.append(y)
            labels.append(f"{MODEL_LABELS[model]} - {FEATURE_LABELS[feature]}")
            y += 1
        y += 0.35
    ax.axvline(0.5, color="#999999", linestyle=":", linewidth=1)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0.68, 0.85)
    ax.set_xlabel("Harrell C-index (95% CI)")
    ax.set_title("Primary Survival-Model Comparison")
    ax.invert_yaxis()
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out / "cindex_comparison_merged.png", dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(out / "cindex_comparison_merged.tiff", dpi=600, format="tiff",
                facecolor="white", bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(out / "cindex_comparison_merged.pdf", facecolor="white", bbox_inches="tight")
    fig.savefig(out / "cindex_comparison_merged.svg", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main():
    a = parse_args()
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    analysis = read_csv_auto(a.analysis_data)
    analysis = analysis.sort_values("analysis_id").reset_index(drop=True)
    if len(analysis) != EXPECTED_N:
        raise ValueError(f"Expected N={EXPECTED_N}; observed {len(analysis)}")
    event = pd.to_numeric(analysis["mace_30d_event"], errors="raise").astype(bool)
    time = pd.to_numeric(analysis["mace_30d_tte_years"], errors="raise").astype(float)
    if int(event.sum()) != EXPECTED_EVENTS:
        raise ValueError(f"Expected {EXPECTED_EVENTS} events; observed {int(event.sum())}")
    y = Surv.from_arrays(event=event, time=time)

    merged = None
    source_hashes = {}
    for key, directory in DEFAULT_RUNS.items():
        model, feature = key
        frame = load_risk(directory, model, feature)
        source_hashes[str(directory / "oof_predictions.csv")] = sha256(directory / "oof_predictions.csv")
        frame = frame.rename(columns={"risk": f"risk__{model}__{feature}"})
        frame = frame.drop(columns=["model", "feature_set"])
        merged = frame if merged is None else merged.merge(
            frame, on=["analysis_id", "outer_fold"], how="inner", validate="one_to_one"
        )

    merged = merged.sort_values("analysis_id").reset_index(drop=True)
    if len(merged) != EXPECTED_N or not np.array_equal(
        merged.analysis_id.to_numpy(), analysis.analysis_id.to_numpy()
    ):
        raise ValueError("OOF files do not align with the frozen analysis dataset")

    risks = {}
    for model in MODEL_ORDER:
        for feature in FEATURE_ORDER:
            col = f"risk__{model}__{feature}"
            risk = pd.to_numeric(merged[col], errors="coerce").to_numpy(float)
            if np.isnan(risk).any():
                raise ValueError(f"Missing risk values in {col}")
            risks[(model, feature)] = risk

    intervals, comparisons = bootstrap(y, risks, a.bootstrap_repeats)
    perf_rows = []
    for model in MODEL_ORDER:
        for feature in FEATURE_ORDER:
            risk = risks[(model, feature)]
            lo, hi = intervals[(model, feature)]
            perf_rows.append({
                "endpoint": ENDPOINT,
                "n": EXPECTED_N,
                "events": EXPECTED_EVENTS,
                "model": model,
                "model_label": MODEL_LABELS[model],
                "feature_set": feature,
                "feature_set_label": FEATURE_LABELS[feature],
                "c_index": cindex(y, risk),
                "c_index_ci_lower": lo,
                "c_index_ci_upper": hi,
            })
    perf = pd.DataFrame(perf_rows)

    delta = comparisons[comparisons.comparison_type == "registry_increment"].set_index("model")
    formatted = []
    for model in MODEL_ORDER:
        for feature in FEATURE_ORDER:
            r = perf[(perf.model == model) & (perf.feature_set == feature)].iloc[0]
            formatted.append({
                "Model": MODEL_LABELS[model],
                "Predictors": FEATURE_LABELS[feature],
                "C-index (95% CI)": f"{r.c_index:.2f} ({r.c_index_ci_lower:.2f}-{r.c_index_ci_upper:.2f})",
                "Delta C-index (95% CI)": "Reference" if feature == "demo_cmr" else
                    f"{delta.loc[model, 'estimate']:.3f} ({delta.loc[model, 'ci_lower']:.3f}-{delta.loc[model, 'ci_upper']:.3f})",
            })
    formatted = pd.DataFrame(formatted)

    perf.to_csv(out / "model_performance_merged.csv", index=False, float_format="%.12g")
    comparisons.to_csv(out / "paired_comparisons_merged.csv", index=False, float_format="%.12g")
    formatted.to_csv(out / "supplemental_table_survival_models_formatted.csv", index=False)
    merged.to_csv(out / "oof_predictions_merged.csv", index=False, sep=";", float_format="%.12g")
    plot(perf, out)

    checks = pd.DataFrame([
        {"check": "Cohort N", "status": "PASS", "observed": len(merged), "expected": EXPECTED_N},
        {"check": "Event count", "status": "PASS", "observed": int(event.sum()), "expected": EXPECTED_EVENTS},
        {"check": "Eight model outputs", "status": "PASS" if len(perf) == 8 else "REVIEW", "observed": len(perf), "expected": 8},
        {"check": "OOF completeness", "status": "PASS" if not merged.isna().any().any() else "REVIEW", "observed": int(merged.isna().sum().sum()), "expected": 0},
    ])
    checks.to_csv(out / "run_checks.csv", index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Merge frozen survival OOF predictions without refitting",
        "analysis_data_sha256": sha256(a.analysis_data),
        "source_oof_sha256": source_hashes,
        "bootstrap_seed": SEED,
        "bootstrap_repeats": a.bootstrap_repeats,
        "checks": {"pass": int((checks.status == "PASS").sum()),
                   "review": int((checks.status == "REVIEW").sum())},
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    lines = ["Merged frozen survival-model comparison", "=" * 39]
    for r in perf.itertuples(index=False):
        lines.append(f"{r.model_label} | {r.feature_set_label}: C-index {r.c_index:.4f} "
                     f"(95% CI {r.c_index_ci_lower:.4f}-{r.c_index_ci_upper:.4f})")
    lines.append("")
    for r in comparisons.itertuples(index=False):
        lines.append(f"{r.comparison}: {r.estimate:.4f} "
                     f"(95% CI {r.ci_lower:.4f}-{r.ci_upper:.4f})")
    lines.append(f"Checks: {int((checks.status == 'PASS').sum())} PASS; "
                 f"{int((checks.status == 'REVIEW').sum())} REVIEW")
    (out / "run_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[DONE] {out}")
    print(f"[INFO] Checks: {int((checks.status == 'PASS').sum())} PASS; "
          f"{int((checks.status == 'REVIEW').sum())} REVIEW")


if __name__ == "__main__":
    main()
