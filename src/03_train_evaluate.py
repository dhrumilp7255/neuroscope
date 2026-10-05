"""Phases 4 to 7: train, evaluate and explain models.

Phase 4  baselines:        regularized logistic regression / ridge, plus a demographics only model
Phase 5  stronger models:  random forest and gradient boosting
Phase 6  validation:       (a) stratified 5 fold CV, (b) site held out CV (whole scanner sites left out)
Phase 7  interpretation:   SHAP on the best tree model, computed only on held out folds

Hyperparameters are tuned by an inner 3 fold search that sees training rows only (nested CV).
Imputation, scaling and correlation filtering are fitted inside every training fold.

Usage:
    python src/03_train_evaluate.py            # full run, reads the table from BigQuery (names in src/config.py)
    python src/03_train_evaluate.py --local    # read data/processed/dataset.csv instead
    python src/03_train_evaluate.py --fast     # smaller grids and fewer trees, for a quick check
"""
import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, StratifiedKFold

from config import BQ_TABLE, PROJECT
from common import (META_COLS, DEMOGRAPHICS, SITE_COL, classification_specs, regression_specs,
                    clf_metrics, reg_metrics)

warnings.filterwarnings("ignore")


def run_models(specs, X, y, splits, task, seed, keep_tree_folds=False):
    """Nested CV for every model. Returns (summary rows, out of fold predictions, fold models)."""
    inner = StratifiedKFold(3, shuffle=True, random_state=seed) if task == "classification" \
        else KFold(3, shuffle=True, random_state=seed)
    scoring = "roc_auc" if task == "classification" else "neg_root_mean_squared_error"
    rows, oof_all, fold_models = [], {}, {}
    for name, spec in specs.items():
        Xm = X
        oof = np.full(len(y), np.nan)
        fold_scores, best_params, estimators = [], [], []
        for tr, te in splits:
            gs = GridSearchCV(spec["pipeline"], spec["grid"], cv=inner, scoring=scoring, n_jobs=-1)
            gs.fit(Xm.iloc[tr], y.iloc[tr])
            pred = gs.predict_proba(Xm.iloc[te])[:, 1] if task == "classification" else gs.predict(Xm.iloc[te])
            oof[te] = pred
            m = clf_metrics(y.iloc[te], pred) if task == "classification" else reg_metrics(y.iloc[te], pred)
            fold_scores.append(m)
            best_params.append(gs.best_params_)
            if keep_tree_folds and spec["tree"]:
                estimators.append((te, gs.best_estimator_))
        pooled = clf_metrics(y, oof) if task == "classification" else reg_metrics(y, oof)
        key = "auc" if task == "classification" else "rmse"
        row = {"model": name, **{f"pooled_{k}": v for k, v in pooled.items()},
               f"fold_{key}_mean": np.mean([f[key] for f in fold_scores]),
               f"fold_{key}_std": np.std([f[key] for f in fold_scores]),
               "n_features_input": X.shape[1] - 1, "best_params_per_fold": json.dumps(best_params, default=str)}
        rows.append(row)
        oof_all[name] = oof
        if estimators:
            fold_models[name] = estimators
        print(f"    {name:28s} " + "  ".join(f"{k}={v:.3f}" for k, v in pooled.items()), flush=True)
    return rows, oof_all, fold_models


def oof_shap(fold_estimators, X):
    """SHAP values for each row, computed by the model that did NOT train on that row."""
    parts = []
    for te, pipe in fold_estimators:
        Xt = pipe[:-1].transform(X.iloc[te])
        sv = shap.TreeExplainer(pipe.named_steps["model"]).shap_values(Xt)
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        parts.append(pd.DataFrame(sv, index=X.index[te], columns=Xt.columns))
    feat_cols = [c for c in X.columns if c != SITE_COL]
    S = pd.concat(parts).reindex(columns=feat_cols).fillna(0.0)  # a feature dropped in a fold gets 0
    return S.loc[X.index]


def shap_outputs(S, X, task, out_dir):
    fig_dir = out_dir / "figures"
    X = X[[c for c in X.columns if c in S.columns]]   # drop the carried site column so shapes match
    imp = S.abs().mean().sort_values(ascending=False)
    imp.rename("mean_abs_shap").to_csv(out_dir / f"shap_importance_{task}.csv")
    top = imp.index[:20]
    plt.figure()
    shap.summary_plot(S[top].values, X[top], feature_names=list(top), max_display=20, show=False)
    plt.tight_layout(); plt.savefig(fig_dir / f"shap_summary_{task}.png", dpi=200); plt.close()
    plt.figure()
    shap.summary_plot(S[top].values, X[top], feature_names=list(top), plot_type="bar", max_display=20, show=False)
    plt.tight_layout(); plt.savefig(fig_dir / f"shap_bar_{task}.png", dpi=200); plt.close()
    for f in imp.index[:3]:
        plt.figure()
        shap.dependence_plot(f, S.values, X, feature_names=list(X.columns), show=False)
        plt.tight_layout(); plt.savefig(fig_dir / f"shap_dependence_{task}_{f}.png", dpi=200); plt.close()
    return imp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/dataset.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--local", action="store_true", help="read the local CSV instead of BigQuery")
    ap.add_argument("--bq_table", default=BQ_TABLE)
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--site_splits", type=int, default=5)
    ap.add_argument("--max_rows", type=int, default=None, help="random subsample, for a quick smoke test only")
    ap.add_argument("--harmonize", action="store_true", help="apply ComBat scanner harmonization inside each training fold")
    ap.add_argument("--select_k", type=int, default=None, help="keep only the best K features, selected inside each training fold")
    ap.add_argument("--subgroup", default="all",
                    choices=["all", "males", "females", "under18", "males_under18"],
                    help="restrict to a more uniform subgroup before modeling")
    ap.add_argument("--tasks", default="classification,regression", help="comma list: classification, regression")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    if not args.local:
        from google.cloud import bigquery
        df = bigquery.Client(project=args.project).query(f"SELECT * FROM `{args.bq_table}`").to_dataframe()
        for c in df.columns:  # BigQuery returns nullable pandas types, sklearn wants plain floats
            if pd.api.types.is_extension_array_dtype(df[c]) and pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].astype("float64")
    else:
        df = pd.read_csv(args.data)
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

    feature_cols = [c for c in df.columns if c not in META_COLS]
    X_all = df[feature_cols + DEMOGRAPHICS].copy()   # MRI features plus age and sex
    X_all[SITE_COL] = df["site_group"].values        # carried for SiteHandler only, never a model feature
    print(f"{len(df)} subjects, {len(feature_cols)} MRI features + {len(DEMOGRAPHICS)} demographics"
          + (f"; harmonize={args.harmonize}" if True else "")
          + (f"; select_k={args.select_k}" if args.select_k else ""))

    all_rows = []
    for task in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        print(f"\n=== {task} ===")
        if task == "classification":
            mask = df["dx"].notna()
            y = df.loc[mask, "dx"].astype(int).reset_index(drop=True)
            specs = classification_specs(args.seed, args.fast, args.harmonize, args.select_k)
        else:
            mask = df["fiq"].notna()
            y = df.loc[mask, "fiq"].reset_index(drop=True)
            specs = regression_specs(args.seed, args.fast, args.harmonize, args.select_k)
        X = X_all[mask].reset_index(drop=True)
        groups = df.loc[mask, "site_group"].reset_index(drop=True)
        print(f"  n = {len(y)}")

        outer_a = StratifiedKFold(5, shuffle=True, random_state=args.seed) if task == "classification" \
            else KFold(5, shuffle=True, random_state=args.seed)
        splits_a = list(outer_a.split(X, y))
        splits_b = list(GroupKFold(n_splits=args.site_splits).split(X, y, groups))

        print("  Scheme A: stratified 5 fold CV")
        rows_a, oof_a, fold_models = run_models(specs, X, y, splits_a, task, args.seed, keep_tree_folds=True)
        print("  Scheme B: site held out CV (test sites never seen in training)")
        rows_b, oof_b, _ = run_models(specs, X, y, splits_b, task, args.seed)

        for r in rows_a:
            all_rows.append({"task": task, "scheme": "stratified_kfold", **r})
        for r in rows_b:
            all_rows.append({"task": task, "scheme": "site_held_out", **r})
        pd.DataFrame({"FILE_ID": df.loc[mask, "FILE_ID"].values, "site_group": groups.values, "y_true": y.values,
                      **{f"A_{k}": v for k, v in oof_a.items()},
                      **{f"B_{k}": v for k, v in oof_b.items()}}).to_csv(out / f"oof_predictions_{task}.csv", index=False)

        # Phase 7: SHAP for the better tree model under the stratified scheme
        key = "pooled_auc" if task == "classification" else "pooled_rmse"
        tree_rows = [r for r in rows_a if r["model"] in fold_models]
        best = (max if task == "classification" else min)(tree_rows, key=lambda r: r[key])["model"]
        print(f"  SHAP on out of fold predictions of: {best}")
        S = oof_shap(fold_models[best], X)
        imp = shap_outputs(S, X, task, out)
        print("  Top 10 features by mean |SHAP|:")
        print("    " + "\n    ".join(f"{k}: {v:.4f}" for k, v in imp.head(10).items()))

    res = pd.DataFrame(all_rows)
    tag = args.subgroup + ("_combat" if args.harmonize else "") + (f"_k{args.select_k}" if args.select_k else "")
    res["run"] = tag
    res.to_csv(out / f"model_results_{tag}.csv", index=False)
    res.to_csv(out / "model_results.csv", index=False)
    cols = [c for c in res.columns if c != "best_params_per_fold"]
    print("\n=== Summary ===")
    print(res[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()