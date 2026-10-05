"""Shared pieces: leakage safe preprocessing (including ComBat site harmonization),
model definitions per phase, and metrics.

Everything here is built to be fitted inside a training fold only. The whole chain
(site handling, imputation, harmonization, correlation filter, feature selection,
scaling, model) lives in one sklearn Pipeline, so cross validation refits all of it
on each fold and the test rows never influence any step.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (confusion_matrix, mean_absolute_error, mean_squared_error,
                             r2_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

META_COLS = ["FILE_ID", "SUB_ID", "site", "site_group", "dx", "age", "sex_female", "fiq"]
DEMOGRAPHICS = ["age", "sex_female"]
SITE_COL = "__site__"          # scanner site, carried in X only so SiteHandler can use it, never a model feature
COVARIATES = ["age", "sex_female"]   # biological signal ComBat preserves while removing site effects


def _aprior(d):
    m, s2 = d.mean(), d.var()
    return (2 * s2 + m ** 2) / s2 if s2 > 0 else np.inf


def _bprior(d):
    m, s2 = d.mean(), d.var()
    return (m * s2 + m ** 3) / s2 if s2 > 0 else np.inf


def _it_sol(Z, g_hat, d_hat, g_bar, t2, a, b, tol=1e-4, maxit=100):
    """Empirical Bayes fixed point for one batch, vectorised over features."""
    n = (~np.isnan(Z)).sum(axis=0)
    g_old, d_old = g_hat.copy(), d_hat.copy()
    for _ in range(maxit):
        g_new = (t2 * n * g_old + d_old * g_bar) / (t2 * n + d_old)
        sum2 = np.nansum((Z - g_new[None, :]) ** 2, axis=0)
        d_new = (0.5 * sum2 + b) / (n / 2.0 + a - 1.0)
        change = max(np.nanmax(np.abs(g_new - g_old) / (np.abs(g_old) + 1e-8)),
                     np.nanmax(np.abs(d_new - d_old) / (np.abs(d_old) + 1e-8)))
        g_old, d_old = g_new, d_new
        if change < tol:
            break
    return g_old, d_old


class ComBat(BaseEstimator, TransformerMixin):
    """Remove additive and multiplicative scanner site effects from feature columns,
    preserving the covariate signal. Fitted on training rows, applied to any rows.

    A site not seen during fit passes through unchanged (identity), the honest
    behaviour for leave one site out validation.
    """

    def __init__(self, n_features):
        self.n_features = n_features

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        Y, Cov = X[:, :self.n_features], X[:, self.n_features:]
        b = self._batch
        self.levels_ = list(pd.unique(b))
        onehot = np.column_stack([(b == lv).astype(float) for lv in self.levels_])
        design = np.column_stack([onehot, Cov]) if Cov.shape[1] else onehot
        B = len(self.levels_)
        coefs, *_ = np.linalg.lstsq(design, Y, rcond=None)
        batch_coef, cov_coef = coefs[:B], coefs[B:]
        sizes = onehot.sum(axis=0)
        self.grand_ = (sizes / sizes.sum()) @ batch_coef
        self.cov_coef_ = cov_coef
        resid = Y - design @ coefs
        self.var_ = (resid ** 2).mean(axis=0)
        self.var_[self.var_ < 1e-12] = 1e-12
        stand_mean = self.grand_[None, :] + (Cov @ cov_coef if Cov.shape[1] else 0.0)
        Z = (Y - stand_mean) / np.sqrt(self.var_)
        self.gamma_, self.delta_ = {}, {}
        for i, lv in enumerate(self.levels_):
            idx = onehot[:, i] == 1
            if idx.sum() < 2:
                self.gamma_[lv] = np.zeros(self.n_features)
                self.delta_[lv] = np.ones(self.n_features)
                continue
            g_hat = Z[idx].mean(axis=0)
            d_hat = Z[idx].var(axis=0)
            d_hat[d_hat < 1e-12] = 1e-12
            g_star, d_star = _it_sol(Z[idx], g_hat, d_hat, g_hat.mean(),
                                     g_hat.var() if g_hat.var() > 0 else 1e-8,
                                     _aprior(d_hat), _bprior(d_hat))
            self.gamma_[lv], self.delta_[lv] = g_star, d_star
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        Y, Cov = X[:, :self.n_features], X[:, self.n_features:]
        b = self._batch
        stand_mean = self.grand_[None, :] + (Cov @ self.cov_coef_ if Cov.shape[1] else 0.0)
        Z = (Y - stand_mean) / np.sqrt(self.var_)
        out = Y.copy()
        for lv in pd.unique(b):
            idx = b == lv
            g = self.gamma_.get(lv, np.zeros(self.n_features))
            d = self.delta_.get(lv, np.ones(self.n_features))
            out[idx] = (Z[idx] - g[None, :]) / np.sqrt(d)[None, :] * np.sqrt(self.var_)[None, :] + stand_mean[idx]
        X[:, :self.n_features] = out
        return X


class SiteHandler(BaseEstimator, TransformerMixin):
    """First pipeline step. Median imputes features (training medians only), optionally
    runs ComBat using the site column, then drops the site column.

    Input X columns: MRI features, then covariates (age, sex), then one site column
    named SITE_COL. Output drops the site column.
    """

    def __init__(self, harmonize=False):
        self.harmonize = harmonize

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.feat_cols_ = [c for c in X.columns if c not in COVARIATES + [SITE_COL]]
        self.cov_cols_ = [c for c in COVARIATES if c in X.columns]
        self.out_cols_ = self.feat_cols_ + self.cov_cols_
        self.medians_ = X[self.out_cols_].median(numeric_only=True)
        if self.harmonize and self.feat_cols_ and SITE_COL in X.columns:
            mat = X[self.out_cols_].fillna(self.medians_).to_numpy(dtype=float)
            self.combat_ = ComBat(n_features=len(self.feat_cols_))
            self.combat_._batch = X[SITE_COL].to_numpy()
            self.combat_.fit(mat)
        else:
            self.combat_ = None
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        out = X[self.out_cols_].fillna(self.medians_)
        if self.combat_ is not None:
            self.combat_._batch = X[SITE_COL].to_numpy()
            arr = self.combat_.transform(out.to_numpy(dtype=float))
            out = pd.DataFrame(arr, columns=self.out_cols_, index=X.index)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(self.out_cols_)


class ColumnKeeper(BaseEstimator, TransformerMixin):
    """Keep only named columns (used for the age and sex only baseline)."""

    def __init__(self, keep):
        self.keep = keep

    def fit(self, X, y=None):
        self.keep_ = [c for c in self.keep if c in pd.DataFrame(X).columns]
        return self

    def transform(self, X):
        return pd.DataFrame(X)[self.keep_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.keep_)


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop constant columns and one of every pair with |correlation| above threshold."""

    def __init__(self, threshold=0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        keep = [c for c in X.columns if X[c].nunique(dropna=True) > 1]
        corr = X[keep].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        drop = {c for c in upper.columns if (upper[c] > self.threshold).any()}
        self.keep_ = [c for c in keep if c not in drop]
        return self

    def transform(self, X):
        return pd.DataFrame(X)[self.keep_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.keep_)


def _full_pipe(model, scale, harmonize, select_k, task):
    steps = [("site", SiteHandler(harmonize=harmonize)), ("filter", CorrelationFilter(0.95))]
    if select_k:
        score = f_classif if task == "classification" else f_regression
        steps.append(("select", SelectKBest(score, k=select_k)))
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps).set_output(transform="pandas")


def _demographics_pipe(model):
    return Pipeline([("site", SiteHandler(harmonize=False)),
                     ("keep", ColumnKeeper(DEMOGRAPHICS)),
                     ("scale", StandardScaler()),
                     ("model", model)]).set_output(transform="pandas")


def classification_specs(seed, fast=False, harmonize=False, select_k=None):
    n_trees = 150 if fast else 300
    rf_grid = {"model__max_depth": [4, None], "model__max_features": ["sqrt"]} if fast else \
              {"model__max_depth": [4, 8, None], "model__max_features": ["sqrt", 0.3]}
    gb_grid = {"model__n_estimators": [100], "model__learning_rate": [0.05], "model__max_depth": [2, 3]} if fast else \
              {"model__n_estimators": [100, 200], "model__learning_rate": [0.03, 0.1], "model__max_depth": [2, 3]}
    lr = lambda: LogisticRegression(max_iter=5000, class_weight="balanced")
    full = lambda m, scale: _full_pipe(m, scale, harmonize, select_k, "classification")
    return {
        "demographics_only_logreg": dict(pipeline=_demographics_pipe(LogisticRegression(max_iter=5000, class_weight="balanced")),
                                         grid={"model__C": [0.1, 1.0]}, tree=False),
        "logreg_l2": dict(pipeline=full(lr(), True),
                          grid={"model__C": [0.0005, 0.002, 0.01, 0.05]}, tree=False),
        "random_forest": dict(pipeline=full(RandomForestClassifier(n_estimators=n_trees, min_samples_leaf=3,
                                                                   class_weight="balanced", n_jobs=1, random_state=seed), False),
                              grid=rf_grid, tree=True),
        "gradient_boosting": dict(pipeline=full(GradientBoostingClassifier(subsample=0.8, random_state=seed), False),
                                  grid=gb_grid, tree=True),
    }


def regression_specs(seed, fast=False, harmonize=False, select_k=None):
    n_trees = 150 if fast else 300
    rf_grid = {"model__max_depth": [4, None], "model__max_features": ["sqrt"]} if fast else \
              {"model__max_depth": [4, 8, None], "model__max_features": ["sqrt", 0.3]}
    gb_grid = {"model__n_estimators": [100], "model__learning_rate": [0.05], "model__max_depth": [2, 3]} if fast else \
              {"model__n_estimators": [100, 200], "model__learning_rate": [0.03, 0.1], "model__max_depth": [2, 3]}
    full = lambda m, scale: _full_pipe(m, scale, harmonize, select_k, "regression")
    return {
        "demographics_only_ridge": dict(pipeline=_demographics_pipe(Ridge()), grid={"model__alpha": [1.0, 10.0]}, tree=False),
        "ridge": dict(pipeline=full(Ridge(), True), grid={"model__alpha": [10, 100, 1000, 5000]}, tree=False),
        "random_forest": dict(pipeline=full(RandomForestRegressor(n_estimators=n_trees, min_samples_leaf=3,
                                                                  n_jobs=1, random_state=seed), False),
                              grid=rf_grid, tree=True),
        "gradient_boosting": dict(pipeline=full(GradientBoostingRegressor(subsample=0.8, random_state=seed), False),
                                  grid=gb_grid, tree=True),
    }


def clf_metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"auc": roc_auc_score(y, p), "accuracy": (tp + tn) / len(y),
            "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1)}


def reg_metrics(y, p):
    return {"rmse": float(np.sqrt(mean_squared_error(y, p))), "mae": mean_absolute_error(y, p),
            "r2": r2_score(y, p)}