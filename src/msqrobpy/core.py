from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import lstsq
from statsmodels.regression.linear_model import OLS

from .design import build_design_matrix, contrast_vector
from .moderation import moderate_variances
from .results import ContrastResult, FeatureModelResult, MsqrobFit


def _rlm_fit(
    X: np.ndarray,
    y: np.ndarray,
    maxiter: int = 5,
    k: float = 1.345,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """M-estimation IRLS matching R's MASS::rlm(method='M').

    Returns (coef, weights, residuals, rank).
    """
    # 1. Initial OLS fit
    coef, _, rank, _ = lstsq(X, y)
    resid = y - X @ coef

    # 2. IRLS loop (scale re-estimated each iteration via MAD)
    for _ in range(maxiter):
        scale = float(np.median(np.abs(resid)) / 0.6745)
        if scale < 1e-10:
            scale = 1e-10
        u = resid / scale
        # Huber weights: w = min(1, k / |u|)
        weights = np.minimum(1.0, k / np.maximum(np.abs(u), 1e-10))
        # Weighted LS
        sqrtw = np.sqrt(weights)
        Xw = X * sqrtw[:, None]
        yw = y * sqrtw
        coef, _, rank, _ = lstsq(Xw, yw)
        resid = y - X @ coef

    return coef, weights, resid, rank


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty_like(adj)
    out[order] = np.clip(adj, 0.0, 1.0)
    return out


class ProteinModelFit(MsqrobFit):
    """Extended fit object with contrast testing."""

    def test_contrast(self, contrast: str | pd.Series, name: str | None = None) -> ContrastResult:
        if isinstance(contrast, str):
            L = contrast_vector(self.design_columns, contrast)
            contrast_name = name or contrast
        else:
            L = contrast.reindex(self.design_columns).fillna(0.0)
            contrast_name = name or "contrast"

        rows = []
        for feature_id, model in self.models.items():
            coef = model.coef.reindex(self.design_columns).astype(float)
            estimate = float(np.dot(L.values, coef.values))
            vcov = model.vcov_unscaled.reindex(index=self.design_columns, columns=self.design_columns).astype(float)
            s2 = model.var_posterior if model.var_posterior is not None else model.var
            df = model.df_posterior if model.df_posterior is not None else model.df_residual
            se = float(np.sqrt(max(s2, 0.0) * float(L.values @ vcov.values @ L.values)))
            if not np.isfinite(se) or se <= 0:
                t_stat = np.nan
                p_value = np.nan
            else:
                t_stat = estimate / se
                if np.isinf(df):
                    p_value = 2 * stats.norm.sf(abs(t_stat))
                else:
                    p_value = 2 * stats.t.sf(abs(t_stat), df=max(df, 1e-6))
            rows.append(
                {
                    "feature_id": feature_id,
                    "contrast": contrast_name,
                    "estimate": estimate,
                    "std_error": se,
                    "t_stat": t_stat,
                    "df": df,
                    "p_value": p_value,
                    "sigma": model.sigma,
                    "sigma_posterior": model.sigma_posterior,
                    "method": model.fitted_method,
                }
            )
        table = pd.DataFrame(rows)
        table["adj_p_value"] = _benjamini_hochberg(table["p_value"].fillna(1.0).to_numpy())
        return ContrastResult(table=table)


def fit_protein_model(
    intensity_df: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    formula: str,
    robust: bool = True,
    empirical_bayes: bool = True,
    feature_axis: Literal[0, 1] = 0,
    maxiter: int = 5,
) -> ProteinModelFit:
    """Fit per-feature linear models for summarized protein abundances.

    Parameters
    ----------
    intensity_df:
        Matrix-like dataframe of quantitative abundances. By default, rows are
        features and columns are samples.
    sample_metadata:
        DataFrame indexed by sample names or containing rows in the same order as
        columns in `intensity_df`.
    formula:
        Fixed-effect model formula compatible with patsy.
    robust:
        Use robust linear regression instead of ordinary least squares.
    empirical_bayes:
        Apply variance moderation across fitted features.
    feature_axis:
        `0` if rows are features, `1` if columns are features.
    maxiter:
        Maximum IRLS iterations for robust regression. Default is 5,
        matching the ``msqrobLm`` default in the R msqrob2 package.
        Ignored when ``robust=False``.
    """
    y_df = intensity_df.copy()
    if feature_axis == 1:
        y_df = y_df.T
    if not y_df.columns.equals(sample_metadata.index):
        sample_metadata = sample_metadata.loc[y_df.columns]

    X, design_columns = build_design_matrix(formula, sample_metadata)

    models: dict[str, FeatureModelResult] = {}
    sample_vars = []
    dfs = []

    for feature_id, values in y_df.iterrows():
        y = values.to_numpy(dtype=float)
        obs = np.isfinite(y)
        if obs.sum() <= X.shape[1]:
            continue
        X_obs = X[obs, :]
        y_obs = y[obs]
        try:
            if robust:
                params, weights, resid, rank = _rlm_fit(
                    X_obs, y_obs, maxiter=maxiter
                )
                df_residual = max(float(weights.sum() - rank), 1.0)
                sigma = float(np.sqrt(np.sum(weights * resid ** 2) / df_residual))
                method = "rlm"
                coef = pd.Series(params, index=design_columns, dtype=float)
                # (X'WX)^{-1} – matches R's .vcovUnscaled()
                W = np.diag(weights)
                XtWX = X_obs.T @ W @ X_obs
                XtWX_inv = np.linalg.inv(XtWX)
                vcov_unscaled = pd.DataFrame(XtWX_inv, index=design_columns, columns=design_columns)
            else:
                fit = OLS(y_obs, X_obs).fit()
                sigma = float(np.sqrt(fit.scale))
                df_residual = float(fit.df_resid)
                method = "ols"
                weights = None
                coef = pd.Series(fit.params, index=design_columns, dtype=float)
                vcov_unscaled = pd.DataFrame(fit.normalized_cov_params, index=design_columns, columns=design_columns)
        except Exception:
            continue

        result = FeatureModelResult(
            feature_id=str(feature_id),
            coef=coef,
            vcov_unscaled=vcov_unscaled,
            sigma=sigma,
            df_residual=df_residual,
            fitted_method=method,
            weights=weights,
        )
        models[str(feature_id)] = result
        sample_vars.append(result.var)
        dfs.append(result.df_residual)

    if empirical_bayes and models:
        var_post, df_post, _, _ = moderate_variances(np.asarray(sample_vars), np.asarray(dfs))
        for result, vp, dp in zip(models.values(), var_post, df_post):
            result.var_posterior = float(vp)
            result.df_posterior = float(dp)

    return ProteinModelFit(
        models=models,
        design_columns=design_columns,
        formula=formula,
        sample_metadata=sample_metadata.copy(),
    )
