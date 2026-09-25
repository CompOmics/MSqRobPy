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


#: Ranking columns tried in order when `ContrastResult.top` is called without
#: an explicit `sort_by`. The abundance model and the hurdle workflow report
#: significance under different column names.
_DEFAULT_SORT_COLUMNS = (
    "adj_p_value",
    "adj_combined_p_value",
    "p_value",
    "combined_p_value",
)


@dataclass
class ContrastResult:
    """Result table for one or multiple contrasts across features."""

    table: pd.DataFrame

    def top(self, n: int = 20, sort_by: Optional[str] = None) -> pd.DataFrame:
        """Return the `n` highest-ranked features.

        Parameters
        ----------
        n:
            Number of rows to return.
        sort_by:
            Column to sort on. When omitted, the first available column of
            `_DEFAULT_SORT_COLUMNS` is used, so the same call works for
            abundance and hurdle result tables.
        """
        if sort_by is None:
            sort_by = next(
                (c for c in _DEFAULT_SORT_COLUMNS if c in self.table.columns), None
            )
            if sort_by is None:
                return self.table.head(n).copy()
        elif sort_by not in self.table.columns:
            raise KeyError(
                f"Column {sort_by!r} is not in the result table. "
                f"Available columns: {sorted(self.table.columns)}"
            )
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
