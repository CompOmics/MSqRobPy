from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.discrete.discrete_model import Logit

from .core import fit_protein_model
from .design import build_design_matrix, contrast_vector
from .results import ContrastResult


def fit_hurdle_model(
    intensity_df: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    formula: str,
    feature_axis: int = 0,
    robust: bool = True,
    empirical_bayes: bool = True,
) -> dict[str, object]:
    """Fit a simple hurdle workflow.

    The abundance component is modeled on observed intensities only, while the
    detection component models observation probability using logistic regression.
    Combined evidence is reported using Stouffer's method.
    """
    abundance_fit = fit_protein_model(
        intensity_df=intensity_df,
        sample_metadata=sample_metadata,
        formula=formula,
        robust=robust,
        empirical_bayes=empirical_bayes,
        feature_axis=feature_axis,
    )

    y_df = intensity_df if feature_axis == 0 else intensity_df.T
    if not y_df.columns.equals(sample_metadata.index):
        sample_metadata = sample_metadata.loc[y_df.columns]
    X, design_columns = build_design_matrix(formula, sample_metadata)

    detection_models = {}
    for feature_id, values in y_df.iterrows():
        detected = np.isfinite(values.to_numpy(dtype=float)).astype(float)
        if detected.min() == detected.max():
            continue
        try:
            mod = Logit(detected, X).fit(disp=False)
            detection_models[str(feature_id)] = mod
        except Exception:
            continue

    def test_contrast(contrast: str | pd.Series, name: str | None = None) -> ContrastResult:
        abundance = abundance_fit.test_contrast(contrast, name=name).table.set_index("feature_id")
        if isinstance(contrast, str):
            L = contrast_vector(design_columns, contrast)
            contrast_name = name or contrast
        else:
            L = contrast.reindex(design_columns).fillna(0.0)
            contrast_name = name or "contrast"

        rows = []
        for feature_id, row in abundance.iterrows():
            det_fit = detection_models.get(str(feature_id))
            det_est = np.nan
            det_se = np.nan
            det_z = np.nan
            det_p = np.nan
            if det_fit is not None:
                beta = np.asarray(det_fit.params)
                vcov = np.asarray(det_fit.cov_params())
                det_est = float(L.values @ beta)
                det_se = float(np.sqrt(max(L.values @ vcov @ L.values, 0.0)))
                if det_se > 0:
                    det_z = det_est / det_se
                    det_p = 2 * stats.norm.sf(abs(det_z))
            pvals = [p for p in [row["p_value"], det_p] if np.isfinite(p) and 0 < p <= 1]
            if pvals:
                z = np.mean([stats.norm.isf(p / 2.0) for p in pvals]) * np.sqrt(len(pvals))
                combined_p = 2 * stats.norm.sf(abs(z))
            else:
                z = np.nan
                combined_p = np.nan
            rows.append(
                {
                    "feature_id": feature_id,
                    "contrast": contrast_name,
                    "abundance_estimate": row["estimate"],
                    "abundance_p_value": row["p_value"],
                    "detection_estimate": det_est,
                    "detection_p_value": det_p,
                    "combined_z": z,
                    "combined_p_value": combined_p,
                }
            )
        table = pd.DataFrame(rows)
        if not table.empty:
            p = table["combined_p_value"].fillna(1.0).to_numpy()
            order = np.argsort(p)
            ranked = p[order]
            adj = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
            out = np.empty_like(adj)
            out[order] = np.clip(adj, 0.0, 1.0)
            table["adj_combined_p_value"] = out
        else:
            table["adj_combined_p_value"] = []
        return ContrastResult(table=table)

    return {
        "abundance_fit": abundance_fit,
        "detection_models": detection_models,
        "test_contrast": test_contrast,
    }
