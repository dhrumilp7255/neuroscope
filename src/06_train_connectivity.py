"""Functional phase, step 3: train and validate autism classifiers on connectivity features.

Loads the connectivity Parquet (from the bucket, or local with --local), then runs the same kind of
leakage safe nested cross validation used for the structural models, adapted for ~20,000 features:

  Dimensionality reduction inside every training fold (PCA or univariate selection), so the test
  rows never influence it. The 0.95 correlation filter is NOT used here, because a 20,000 by 20,000
  correlation matrix is too large; PCA/selection does that job instead.

Two validation schemes are reported:
  pooled_10fold   stratified 10 fold CV on all sites mixed. This is the scheme most ABIDE papers
                  report, so your numbers are comparable to the literature.
  site_held_out   whole scanner sites left out of training. The honest test of cross scanner
                  generalization, and always the lower, more trustworthy number.

Interpretation: for the linear model, the strongest connectivity edges (by standardized coefficient)
are written out, which is the connectivity analogue of the SHAP ranking.

Usage:
    python src/06_train_connectivity.py                 # from the bucket Parquet
    python src/06_train_connectivity.py --local         # from a local Parquet
    python src/06_train_connectivity.py --fast          # smaller grids, quick check
"""
import argparse
import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import FC_PARQUET, PROJECT, BUCKET
from common import clf_metrics

warnings.filterwarnings("ignore")
META = ["FILE_ID", "SUB_ID", "site", "site_group", "dx", "age", "sex_female", "fiq"]


def specs(seed, fast):
    pca_grid = {"pca__n_components": [50, 100]} if fast else {"pca__n_components": [50, 100, 200]}
    sel_grid = {"select__k": [500, 2000]} if fast else {"select__k": [500, 2000, 5000]}
    n_trees = 200 if fast else 400
    def pipe(steps):
        return Pipeline(steps)   # numpy throughout keeps memory low on wide data
    return {
        # age and sex only, the sanity baseline
        "demographics_only": dict(
            pipeline=pipe([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                           ("model", LogisticRegression(max_iter=5000, class_weight="balanced"))]),
            grid={"model__C": [0.1, 1.0]}, cols="demo", linear=False),
        # PCA then logistic regression, the standard strong connectivity baseline
        "pca_logreg": dict(
            pipeline=pipe([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                           ("pca", PCA(svd_solver="randomized", random_state=seed)),
                           ("model", LogisticRegression(max_iter=5000, class_weight="balanced"))]),
            grid={**pca_grid, "model__C": [0.01, 0.1, 1.0]}, cols="edges", linear=False),
        # univariate selection then L2 logistic regression, gives interpretable edge weights
        "select_logreg": dict(
            pipeline=pipe([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                           ("select", SelectKBest(f_classif)),
                           ("model", LogisticRegression(max_iter=5000, class_weight="balanced"))]),
            grid={**sel_grid, "model__C": [0.01, 0.1, 1.0]}, cols="edges", linear=True),
        # a tree model on a reduced set, for a nonlinear comparison
        "select_rf": dict(
            pipeline=pipe([("impute", SimpleImputer(strategy="median")),
                           ("select", SelectKBest(f_classif, k=2000)),
                           ("model", RandomForestClassifier(n_estimators=n_trees, min_samples_leaf=3,
                                                            class_weight="balanced", n_jobs=1, random_state=seed))]),
            grid={"model__max_depth": [None]} if fast else {"model__max_depth": [8, None]}, cols="edges", linear=False),
    }


def run(specs_dict, X_edges, X_demo, y, splits, seed, jobs=1):
    inner = StratifiedKFold(3, shuffle=True, random_state=seed)
    rows, oof_all, lin = [], {}, {}
    for name, spec in specs_dict.items():
        X = X_demo if spec["cols"] == "demo" else X_edges
        oof = np.full(len(y), np.nan)
        fold_auc, coefs = [], []
        for tr, te in splits:
            gs = GridSearchCV(spec["pipeline"], spec["grid"], cv=inner, scoring="roc_auc", n_jobs=jobs, pre_dispatch="n_jobs")
            gs.fit(X[tr], y[tr])
            oof[te] = gs.predict_proba(X[te])[:, 1]
            fold_auc.append(roc_safe(y[te], oof[te]))
            if spec["linear"]:
                est = gs.best_estimator_
                mask = est.named_steps["select"].get_support()
                w = np.zeros(X.shape[1]); w[mask] = est.named_steps["model"].coef_[0]
                coefs.append(w)
        pooled = clf_metrics(y, oof)
        rows.append({"model": name, **{f"pooled_{k}": v for k, v in pooled.items()},
                     "fold_auc_mean": np.nanmean(fold_auc), "fold_auc_std": np.nanstd(fold_auc)})
        oof_all[name] = oof
        if coefs:
            lin[name] = np.mean(coefs, axis=0)
        print(f"    {name:20s} " + "  ".join(f"{k}={v:.3f}" for k, v in pooled.items()), flush=True)
    return rows, oof_all, lin


def roc_safe(y, p):
    from sklearn.metrics import roc_auc_score
    try:
        return roc_auc_score(y, p)
    except ValueError:
        return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=FC_PARQUET)
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--bucket", default=BUCKET)
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--site_splits", type=int, default=5)
    ap.add_argument("--subgroup", default="all", choices=["all", "males", "females", "under18", "males_under18"])
    ap.add_argument("--jobs", type=int, default=1, help="parallel workers in the grid search; keep at 1 on a laptop (each worker copies the big matrix)")
    ap.add_argument("--max_rows", type=int, default=None)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.local:
        df = pd.read_parquet(args.parquet)
    else:
        from google.cloud import storage
        data = storage.Client(project=args.project).bucket(args.bucket).blob(args.parquet).download_as_bytes()
        df = pd.read_parquet(io.BytesIO(data))

    if args.max_rows:
        df = df.sample(n=min(args.max_rows, len(df)), random_state=args.seed).reset_index(drop=True)
    before = len(df)
    if args.subgroup in ("males", "males_under18"):
        df = df[df["sex_female"] == 0]
    if args.subgroup == "females":
        df = df[df["sex_female"] == 1]
    if args.subgroup in ("under18", "males_under18"):
        df = df[df["age"] < 18]
    df = df.reset_index(drop=True)
    if args.subgroup != "all":
        print(f"Subgroup '{args.subgroup}': {len(df)} of {before} subjects")

    file_ids = df["FILE_ID"].to_numpy()
    edge_cols = [c for c in df.columns if c.startswith("edge_")]
    X_edges = df[edge_cols].to_numpy(dtype=np.float32)   # float32 halves memory vs float64
    X_demo = df[["age", "sex_female"]].to_numpy(dtype=np.float64)
    y = df["dx"].astype(int).to_numpy()
    groups = df["site_group"].to_numpy()
    print(f"{len(df)} subjects, {len(edge_cols)} connectivity features, {df.site_group.nunique()} sites")

    all_rows = []
    sp = specs(args.seed, args.fast)

    print("  Scheme pooled_10fold (comparable to published ABIDE results)")
    pooled_splits = list(StratifiedKFold(10, shuffle=True, random_state=args.seed).split(X_edges, y))
    rp, oofp, lin = run(sp, X_edges, X_demo, y, pooled_splits, args.seed, args.jobs)

    print("  Scheme site_held_out (honest cross scanner test)")
    site_splits = list(GroupKFold(n_splits=args.site_splits).split(X_edges, y, groups))
    rs, oofs, _ = run(sp, X_edges, X_demo, y, site_splits, args.seed, args.jobs)

    for r in rp:
        all_rows.append({"scheme": "pooled_10fold", **r})
    for r in rs:
        all_rows.append({"scheme": "site_held_out", **r})

    # top connectivity edges from the interpretable linear model
    if "select_logreg" in lin:
        w = pd.Series(lin["select_logreg"], index=edge_cols)
        top = w.reindex(w.abs().sort_values(ascending=False).index).head(25)
        top.rename("mean_coef").to_csv(out / "connectivity_top_edges.csv")
        print("  Top connectivity edges (standardized logistic coefficient):")
        print("    " + "\n    ".join(f"{k}: {v:+.3f}" for k, v in top.head(10).items()))

    tag = f"fmri_{args.subgroup}"
    res = pd.DataFrame(all_rows); res["run"] = tag
    res.to_csv(out / f"model_results_{tag}.csv", index=False)
    pd.DataFrame({"FILE_ID": file_ids, "site_group": groups, "y_true": y,
                  **{f"pooled_{k}": v for k, v in oofp.items()},
                  **{f"site_{k}": v for k, v in oofs.items()}}).to_csv(out / f"oof_predictions_{tag}.csv", index=False)
    print("\n=== Summary ===")
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()