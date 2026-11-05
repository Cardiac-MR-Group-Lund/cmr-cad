#!/usr/bin/env python3
"""09_run_survival_model_comparison.py

Stepwise, deterministic nested-CV survival-model comparison.

Default models: Ridge Cox and Random Survival Forest.
Optional models: LASSO Cox and elastic-net Cox.

Examples
--------
Step 1 (recommended first run):
  python scripts/09_run_survival_model_comparison.py --models ridge rsf --rsf-trees 800

Full reviewer analysis:
  python scripts/09_run_survival_model_comparison.py --models ridge lasso elastic_net rsf --rsf-trees 800
"""
from __future__ import annotations

import argparse, csv, hashlib, importlib.util, json, os, random, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import warnings


os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored
    from sksurv.util import Surv
except ImportError as exc:
    raise SystemExit("Install scikit-survival first: pip install scikit-survival") from exc

warnings.filterwarnings(
    "ignore",
    message="all coefficients are zero, consider decreasing alpha",
    category=UserWarning,
)

SCRIPT_VERSION = "1.0.1"
SEED = 42
INNER_BASE_SEED = 2026
BOOTSTRAP_SEED = 9042
N_INNER = 5

DEFAULT_BASE = Path("scripts/07_run_binary_rf_nested_cv.py")
DEFAULT_DATA = Path("data/processed/cmr_revision_analysis.csv")
DEFAULT_FOLDS = Path("data/processed/outer_cv_folds.csv")
DEFAULT_VOCAB = Path("config/registry_vocabulary.json")
DEFAULT_PLAN = Path("config/cv_plan.json")
DEFAULT_OUT = Path("results/models/survival_comparison")

DATA_SHA = "e94b8aebe4c4939b0b3def943879539f4f242ba7ef39fd7ac929407578f7db9c"
VOCAB_SHA = "1250d92ca3f5d5da2049f3965837a7cfa0b70daaa9ffae2e09d71f8235eac023"
FOLD_SHA = "69c4a862c97d96a7c29e21dcc9c3a1bb7b74f87f1d0baee6c26b4f87fda219eb"
EXPECTED_N = 2159

ENDPOINTS = {
    "primary_30d": ("mace_30d_event", "mace_30d_tte_years", 196, "Primary MACE, 30-day definition"),
    "hard_mace": ("hard_mace_event", "hard_mace_tte_years", 83, "Hard MACE"),
}
MODEL_LABELS = {
    "ridge": "Ridge Cox", "lasso": "LASSO Cox",
    "elastic_net": "Elastic-net Cox", "rsf": "Random Survival Forest",
}
FEATURE_LABELS = {"demo_cmr": "Demographics + CMR", "full": "Registry + demographics + CMR"}
MODEL_ORDER = ["ridge", "lasso", "elastic_net", "rsf"]
FEATURE_ORDER = ["demo_cmr", "full"]

RIDGE_GRID = {"model__alpha": np.logspace(-4, 4, 17).tolist()}

LASSO_GRID = {
    "model__alphas": [
        [0.0031622776601683794],
        [0.01],
        [0.03162277660168379],
        [0.1],
        [0.31622776601683794],
    ]
}

ENET_GRID = {
    "model__alphas": [
        [0.01],
        [0.03162277660168379],
        [0.1],
    ],
    "model__l1_ratio": [0.25, 0.50, 0.75],
}
RSF_GRID = {
    "model__max_depth": [None, 10],
    "model__max_features": ["sqrt", "log2"],
    "model__min_samples_leaf": [5, 10, 20],
}


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-script", type=Path, default=DEFAULT_BASE)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    p.add_argument("--vocabulary", type=Path, default=DEFAULT_VOCAB)
    p.add_argument("--cv-plan", type=Path, default=DEFAULT_PLAN)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--endpoints", nargs="+", choices=list(ENDPOINTS), default=["primary_30d"])
    p.add_argument("--models", nargs="+", choices=MODEL_ORDER, default=["ridge", "rsf"])
    p.add_argument("--feature-sets", nargs="+", choices=FEATURE_ORDER, default=FEATURE_ORDER)
    p.add_argument("--rsf-trees", type=int, default=800)
    p.add_argument("--grid-jobs", type=int, default=-1)
    p.add_argument("--bootstrap-repeats", type=int, default=2000)
    p.add_argument("--allow-hash-mismatch", action="store_true")
    return p.parse_args()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def delimiter(path: Path) -> str:
    sample = path.read_text(encoding="utf-8-sig", errors="replace")[:131072]
    try:
        return csv.Sniffer().sniff(sample, delimiters=";,\t|").delimiter
    except csv.Error:
        first = sample.splitlines()[0] if sample else ""
        return max([";", ",", "\t", "|"], key=first.count)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_base(path: Path):
    spec = importlib.util.spec_from_file_location("rf_shared", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def surv(event, time):
    e = pd.to_numeric(event, errors="coerce")
    t = pd.to_numeric(time, errors="coerce")
    if e.isna().any() or not e.isin([0, 1]).all():
        raise ValueError("Invalid event indicator")
    if t.isna().any() or (t <= 0).any():
        raise ValueError("Invalid survival time")
    return Surv.from_arrays(e.astype(bool), t.astype(float))


def cindex(y, risk) -> float:
    return float(concordance_index_censored(y["event"], y["time"], risk)[0])


def scorer(estimator, X, y) -> float:
    return cindex(y, np.asarray(estimator.predict(X), dtype=float).reshape(-1))


def preprocessor(columns, base):
    cont = [c for c in columns if c in base.CONTINUOUS_CLINICAL]
    binary = [c for c in columns if c in base.BINARY_CLINICAL]
    registry = [c for c in columns if c.startswith("pre_icd__") or c.startswith("pre_kva__")]
    tr = []
    if cont:
        tr.append(("continuous", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), cont))
    if binary:
        tr.append(("binary", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("scaler", StandardScaler()),
        ]), binary))
    if registry:
        tr.append(("registry", StandardScaler(), registry))
    return ColumnTransformer(tr, remainder="drop", sparse_threshold=0.0)


def model_pipeline(key, columns, fold, base, trees):
    pre = preprocessor(columns, base)
    if key == "ridge":
        model = CoxPHSurvivalAnalysis(alpha=1.0, ties="breslow", n_iter=500)
        grid = RIDGE_GRID
    elif key == "lasso":
        model = CoxnetSurvivalAnalysis(l1_ratio=1.0, alphas=[0.01], fit_baseline_model=True,
                                       max_iter=50000, tol=1e-6)
        grid = LASSO_GRID
    elif key == "elastic_net":
        model = CoxnetSurvivalAnalysis(l1_ratio=0.5, alphas=[0.01], fit_baseline_model=True,
                                       max_iter=50000, tol=1e-6)
        grid = ENET_GRID
    else:
        model = RandomSurvivalForest(n_estimators=trees, min_samples_split=10,
                                     min_samples_leaf=10, max_features="sqrt",
                                     n_jobs=1, random_state=SEED + fold)
        grid = RSF_GRID
    return Pipeline([("preprocessor", pre), ("model", model)]), grid


def bootstrap(y, risks, repeats, seed):
    rng = np.random.default_rng(seed)
    event_idx = np.flatnonzero(y["event"])
    censor_idx = np.flatnonzero(~y["event"])
    samples = {k: np.empty(repeats) for k in risks}
    for i in range(repeats):
        idx = np.concatenate([
            rng.choice(event_idx, len(event_idx), replace=True),
            rng.choice(censor_idx, len(censor_idx), replace=True),
        ])
        rng.shuffle(idx)
        for k, risk in risks.items():
            samples[k][i] = cindex(y[idx], risk[idx])
    ci = {k: tuple(np.quantile(v, [0.025, 0.975])) for k, v in samples.items()}
    rows = []
    for model in MODEL_ORDER:
        a, b = (model, "demo_cmr"), (model, "full")
        if a in samples and b in samples:
            d = samples[b] - samples[a]
            rows.append({"comparison_type": "feature_increment", "model": model,
                         "comparison": "Full minus demographics + CMR",
                         "estimate": cindex(y, risks[b]) - cindex(y, risks[a]),
                         "ci_lower": float(np.quantile(d, .025)),
                         "ci_upper": float(np.quantile(d, .975))})
    for feature in FEATURE_ORDER:
        r = ("rsf", feature)
        if r in samples:
            for model in ["ridge", "lasso", "elastic_net"]:
                c = (model, feature)
                if c in samples:
                    d = samples[r] - samples[c]
                    rows.append({"comparison_type": "model_class", "model": f"rsf_minus_{model}",
                                 "comparison": f"RSF minus {MODEL_LABELS[model]} ({FEATURE_LABELS[feature]})",
                                 "estimate": cindex(y, risks[r]) - cindex(y, risks[c]),
                                 "ci_lower": float(np.quantile(d, .025)),
                                 "ci_upper": float(np.quantile(d, .975))})
    return ci, pd.DataFrame(rows)


def write(df, path, sep=","):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, sep=sep, encoding="utf-8", float_format="%.12g")


def plot_results(perf, out):
    endpoints = list(perf.endpoint.unique())
    fig, axes = plt.subplots(1, len(endpoints), figsize=(7 * len(endpoints), 5.8), squeeze=False)
    colors = {"demo_cmr": "#1b9e77", "full": "#1f4e79"}
    for ax, endpoint in zip(axes[0], endpoints):
        sub = perf[perf.endpoint == endpoint]
        pos, labels, y = [], [], 0
        for model in MODEL_ORDER:
            for feature in FEATURE_ORDER:
                row = sub[(sub.model == model) & (sub.feature_set == feature)]
                if row.empty: continue
                r = row.iloc[0]
                ax.errorbar(r.c_index, y,
                    xerr=[[r.c_index-r.c_index_ci_lower], [r.c_index_ci_upper-r.c_index]],
                    fmt="o", color=colors[feature], capsize=3)
                pos.append(y); labels.append(f"{MODEL_LABELS[model]} - {FEATURE_LABELS[feature]}")
                y += 1
            y += .3
        ax.axvline(.5, color="#999", linestyle=":")
        ax.set_yticks(pos); ax.set_yticklabels(labels, fontsize=8.5)
        ax.set_xlim(.45, .90); ax.set_xlabel("Harrell C-index (95% CI)")
        ax.set_title(ENDPOINTS[endpoint][3]); ax.invert_yaxis(); ax.grid(False)
    fig.tight_layout()
    for ext, dpi in [("png", 300), ("tiff", 600), ("pdf", None), ("svg", None)]:
        path = out / f"cindex_comparison.{ext}"
        kw = {"bbox_inches": "tight", "facecolor": "white"}
        if dpi: kw["dpi"] = dpi
        if ext == "tiff": kw["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **kw)
    plt.close(fig)


def main():
    a = args(); random.seed(SEED); np.random.seed(SEED)
    out = a.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    for p in [a.base_script, a.data, a.folds, a.vocabulary, a.cv_plan]:
        if not p.exists(): raise FileNotFoundError(p)
    base = load_base(a.base_script)
    vocab = read_json(a.vocabulary); plan = read_json(a.cv_plan)
    checks = {
        "dataset": sha(a.data) == DATA_SHA,
        "vocabulary": vocab.get("vocabulary_sha256") == VOCAB_SHA,
        "folds": plan["outer_cross_validation"].get("assignment_specification_sha256") == FOLD_SHA,
    }
    
    if not all(checks.values()) and not a.allow_hash_mismatch:
        raise RuntimeError(f"Frozen input mismatch: {checks}")

    df = pd.read_csv(a.data, sep=delimiter(a.data), encoding="utf-8-sig", low_memory=False)
    folds = pd.read_csv(a.folds, sep=delimiter(a.folds), encoding="utf-8-sig")
    df = df.merge(folds[["analysis_id", "outer_fold"]], on="analysis_id", validate="one_to_one")
    df = df.sort_values("analysis_id").reset_index(drop=True)
    if len(df) != EXPECTED_N: raise ValueError(f"Expected {EXPECTED_N} rows")
    reg = base.build_registry_matrix(df, vocab)
    if reg.shape[1] != 62: raise ValueError("Registry feature count changed")
    data = pd.concat([df, reg], axis=1)
    feature_sets = base.make_feature_sets(reg.columns.tolist())

    fold_rows, par_rows, oof = [], [], {"analysis_id": data.analysis_id.astype(int),
                                        "outer_fold": data.outer_fold.astype(int)}
    all_perf, all_comp = [], []
    print(f"[INFO] Script {SCRIPT_VERSION}; N={len(data)}; registry={reg.shape[1]}")

    for ep_i, ep in enumerate(a.endpoints):
        event_col, time_col, expected_events, label = ENDPOINTS[ep]
        y = surv(data[event_col], data[time_col])
        if int(y["event"].sum()) != expected_events: raise ValueError(f"{ep} event count changed")
        risks = {}
        print(f"[INFO] Endpoint {ep}: events={expected_events}")
        for model in [m for m in MODEL_ORDER if m in a.models]:
            for feature in [f for f in FEATURE_ORDER if f in a.feature_sets]:
                cols = feature_sets[feature].columns; X = data[cols]
                risk = np.full(len(data), np.nan)
                print(f"[INFO] Starting {MODEL_LABELS[model]} | {FEATURE_LABELS[feature]}")
                for fold in sorted(data.outer_fold.unique()):
                    test = data.outer_fold.eq(fold).to_numpy(); train = ~test
                    inner = StratifiedKFold(N_INNER, shuffle=True, random_state=INNER_BASE_SEED + int(fold))
                    splits = list(inner.split(np.zeros(train.sum()), y[train]["event"].astype(int)))
                    pipe, grid = model_pipeline(model, cols, int(fold), base, a.rsf_trees)
                    search = GridSearchCV(pipe, grid, scoring=scorer, cv=splits, refit=True,
                                          n_jobs=a.grid_jobs, error_score=np.nan)
                    search.fit(X.loc[train], y[train])
                    pred = np.asarray(search.predict(X.loc[test]), float).reshape(-1)
                    risk[test] = pred
                    fc = cindex(y[test], pred)
                    fold_rows.append({"endpoint": ep, "model": model, "feature_set": feature,
                                      "outer_fold": int(fold), "test_n": int(test.sum()),
                                      "test_events": int(y[test]["event"].sum()),
                                      "c_index": fc, "best_inner_c_index": float(search.best_score_)})
                    par_rows.append({"endpoint": ep, "model": model, "feature_set": feature,
                                     "outer_fold": int(fold), "best_inner_c_index": float(search.best_score_),
                                     "best_params": json.dumps(search.best_params_)})
                    print(f"[INFO] fold={int(fold)} C={fc:.4f} best={search.best_params_}")
                if np.isnan(risk).any(): raise RuntimeError("Missing OOF risks")
                risks[(model, feature)] = risk
                oof[f"risk__{ep}__{model}__{feature}"] = risk

        ci, comp = bootstrap(y, risks, a.bootstrap_repeats, BOOTSTRAP_SEED + ep_i)
        comp.insert(0, "endpoint", ep); all_comp.append(comp)
        for (model, feature), risk in risks.items():
            lo, hi = ci[(model, feature)]
            all_perf.append({"endpoint": ep, "endpoint_label": label, "n": len(y),
                             "events": expected_events, "model": model,
                             "model_label": MODEL_LABELS[model], "feature_set": feature,
                             "feature_set_label": FEATURE_LABELS[feature],
                             "n_original_predictors": len(feature_sets[feature].columns),
                             "c_index": cindex(y, risk), "c_index_ci_lower": float(lo),
                             "c_index_ci_upper": float(hi)})

    perf = pd.DataFrame(all_perf); comp = pd.concat(all_comp, ignore_index=True)
    write(perf, out / "model_performance.csv")
    write(pd.DataFrame(fold_rows), out / "fold_performance.csv")
    write(pd.DataFrame(par_rows), out / "selected_hyperparameters.csv")
    write(comp, out / "paired_model_comparisons.csv")
    write(pd.DataFrame(oof), out / "oof_predictions.csv", sep=";")
    plot_results(perf, out)

    check_df = pd.DataFrame([
        {"check": "Analysis dataset hash", "status": "PASS" if checks["dataset"] else "REVIEW"},
        {"check": "Vocabulary hash", "status": "PASS" if checks["vocabulary"] else "REVIEW"},
        {"check": "Fold assignment hash", "status": "PASS" if checks["folds"] else "REVIEW"},
        {"check": "Cohort N", "status": "PASS" if len(data) == EXPECTED_N else "REVIEW"},
        {"check": "Registry features", "status": "PASS" if reg.shape[1] == 62 else "REVIEW"},
        {"check": "OOF completeness", "status": "PASS" if not pd.DataFrame(oof).isna().any().any() else "REVIEW"},
    ])
    write(check_df, out / "run_checks.csv")
    reviews = int(check_df.status.eq("REVIEW").sum())

    manifest = {"script_version": SCRIPT_VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_sha256": sha(a.data), "vocabulary_sha256": vocab.get("vocabulary_sha256"),
                "fold_specification_sha256": plan["outer_cross_validation"].get("assignment_specification_sha256"),
                "endpoints": a.endpoints, "models": a.models, "feature_sets": a.feature_sets,
                "rsf_trees": a.rsf_trees, "bootstrap_repeats": a.bootstrap_repeats,
                "checks": {"pass": int(check_df.status.eq("PASS").sum()), "review": reviews}}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    lines = ["Nested-CV survival model comparison", "=" * 35]
    for r in perf.itertuples(index=False):
        lines.append(f"{r.endpoint_label} | {r.model_label} | {r.feature_set_label}: "
                     f"C-index {r.c_index:.4f} ({r.c_index_ci_lower:.4f}-{r.c_index_ci_upper:.4f})")
    lines.append(f"Checks: {int(check_df.status.eq('PASS').sum())} PASS; {reviews} REVIEW")
    (out / "run_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[DONE] {out}")
    print(f"[INFO] Checks: {int(check_df.status.eq('PASS').sum())} PASS; {reviews} REVIEW")

if __name__ == "__main__":
    main()
