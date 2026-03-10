from __future__ import annotations

import numpy as np
from scipy import optimize, special


def _trigamma_inverse(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    # Reasonable starting point for positive x
    out[:] = np.where(x > 1e-6, 0.5 + 1.0 / x, 1e6)
    for _ in range(8):
        tri = special.polygamma(1, out)
        tetra = special.polygamma(2, out)
        out -= (tri - x) / tetra
        out = np.maximum(out, 1e-8)
    return out


def estimate_prior_df_var(sample_var: np.ndarray, df: np.ndarray) -> tuple[float, float]:
    """Estimate limma-style prior degrees of freedom and prior variance.

    Reimplements limma's ``fitFDist`` algorithm.  Under the scaled
    inverse-chi-squared model the log sample variances satisfy::

        Var[log(s_g^2)] = trigamma(d_g / 2) + trigamma(d0 / 2)

    so the prior trigamma contribution is isolated by subtracting the
    known per-feature ``trigamma(d_g / 2)`` from the observed variance
    of the centred log-variances.
    """
    sample_var = np.asarray(sample_var, dtype=float)
    df = np.asarray(df, dtype=float)
    valid = np.isfinite(sample_var) & np.isfinite(df) & (sample_var > 0) & (df > 0)
    n = int(valid.sum())
    if n < 3:
        return 0.0, float(np.nanmedian(sample_var[valid])) if valid.any() else np.nan

    sv = sample_var[valid]
    dv = df[valid]

    # Centred log-variances  (limma calls these ``e``)
    z = np.log(sv) - special.digamma(dv / 2.0) + np.log(dv / 2.0)
    emean = z.mean()
    evar = z.var(ddof=1)

    # Remove the known sampling-variance component  (key limma step)
    evar = evar - np.mean(special.polygamma(1, dv / 2.0))

    if evar > 0:
        # Invert trigamma to recover df0 / 2, then df0
        half_df0 = float(_trigamma_inverse(np.array([evar]))[0])
        df0 = 2.0 * half_df0
        s02 = float(np.exp(emean + special.digamma(half_df0) - np.log(half_df0)))
    else:
        # Variance is homogeneous across features → infinite prior df
        df0 = np.inf
        s02 = float(np.exp(emean))

    return df0, s02


def moderate_variances(sample_var: np.ndarray, df: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return posterior variances and posterior df for feature-wise residual variances."""
    df0, s02 = estimate_prior_df_var(sample_var, df)
    sample_var = np.asarray(sample_var, dtype=float)
    df = np.asarray(df, dtype=float)
    if np.isinf(df0):
        # All features share the same prior variance
        var_post = np.full_like(sample_var, s02)
        df_post = np.full_like(df, np.inf)
    else:
        var_post = (df * sample_var + df0 * s02) / np.maximum(df + df0, 1e-12)
        df_post = df + df0
    return var_post, df_post, df0, s02
