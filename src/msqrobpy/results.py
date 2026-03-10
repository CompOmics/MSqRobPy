from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class FeatureModelResult:
    """Container holding model statistics for a single feature.

    Parameters
    ----------
    feature_id:
        Identifier of the modeled feature.
    coef:
        Estimated model coefficients.
    vcov_unscaled:
        Unscaled covariance matrix, i.e. the inverse of X'X or robust analogue.
    sigma:
        Residual standard deviation.
    df_residual:
        Residual degrees of freedom.
    fitted_method:
        Fitting backend used for this feature.
    weights:
        Optional robust regression weights.
    var_posterior:
        Empirical-Bayes posterior variance estimate.
    df_posterior:
        Empirical-Bayes posterior degrees of freedom.
    metadata:
        Optional free-form metadata.
    """

    feature_id: str
    coef: pd.Series
    vcov_unscaled: pd.DataFrame
    sigma: float
    df_residual: float
    fitted_method: str
    weights: Optional[np.ndarray] = None
    var_posterior: Optional[float] = None
    df_posterior: Optional[float] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def var(self) -> float:
        return float(self.sigma ** 2)

    @property
    def sigma_posterior(self) -> float:
        if self.var_posterior is None:
            return self.sigma
        return float(np.sqrt(self.var_posterior))


@dataclass
class ContrastResult:
    """Result table for one or multiple contrasts across features."""

    table: pd.DataFrame

    def top(self, n: int = 20, sort_by: str = "adj_p_value") -> pd.DataFrame:
        return self.table.sort_values(sort_by).head(n).copy()


@dataclass
class MsqrobFit:
    """Collection of per-feature model fits and moderated inference helpers."""

    models: Dict[str, FeatureModelResult]
    design_columns: list[str]
    formula: str
    sample_metadata: pd.DataFrame

    def coefficients(self) -> pd.DataFrame:
        return pd.DataFrame({k: v.coef for k, v in self.models.items()}).T
