from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.robust.scale import mad


def robust_summary(values: pd.Series | np.ndarray) -> float:
    """Return a robust summary statistic for peptide intensities.

    The current implementation uses the median, which is stable and easy to
    interpret. This function is intentionally simple so it can be swapped out in
    user code.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.median(arr))


def aggregate_peptides(
    peptide_df: pd.DataFrame,
    protein_col: str,
    peptide_col: str,
    sample_col: str,
    intensity_col: str,
    summary_func=robust_summary,
    min_peptides: int = 1,
) -> pd.DataFrame:
    """Aggregate peptide intensities into protein-level abundance values.

    Returns a feature x sample matrix with proteins as rows and samples as columns.
    Additional metadata columns are returned in long form if desired by merging
    with the originating table.
    """
    required = {protein_col, peptide_col, sample_col, intensity_col}
    missing = required.difference(peptide_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    counts = (
        peptide_df[[protein_col, peptide_col]]
        .drop_duplicates()
        .groupby(protein_col)[peptide_col]
        .nunique()
    )
    keep = counts[counts >= min_peptides].index
    sub = peptide_df[peptide_df[protein_col].isin(keep)].copy()

    agg = (
        sub.groupby([protein_col, sample_col])[intensity_col]
        .apply(summary_func)
        .unstack(sample_col)
        .sort_index()
    )
    return agg
