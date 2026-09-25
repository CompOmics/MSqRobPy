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


def _aggregate_by_feature(
    peptide_df: pd.DataFrame,
    feature_col: str,
    sample_col: str,
    intensity_col: str,
    summary_func=robust_summary,
    min_observations: int = 1,
) -> pd.DataFrame: 
    """Aggregate intensities by feature (internal helper).

    Collapse rows with same feature and sample to a single value via `summary_func`.
    Filter to features with at least `min_observations` entries (across all samples).

    Parameters
    ----------
    feature_col:
        Column name for the feature to aggregate by (protein, peptide, PTM site, etc).
    sample_col:
        Column name for samples.
    intensity_col:
        Column name for intensity values to aggregate.
    summary_func:
        Callable that takes a Series or array and returns a scalar.
    min_observations:
        Keep only features with at least this many observations across all samples.

    Returns
    -------
    Feature x sample matrix with features as rows and samples as columns.
    """
    required = {feature_col, sample_col, intensity_col}
    missing = required.difference(peptide_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    counts = peptide_df[[feature_col, sample_col]].drop_duplicates().groupby(feature_col).size()
    keep = counts[counts >= min_observations].index
    sub = peptide_df[peptide_df[feature_col].isin(keep)].copy()

    agg = (
        sub.groupby([feature_col, sample_col])[intensity_col]
        .apply(summary_func)
        .unstack(sample_col)
        .sort_index()
    )
    return agg


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

    Parameters
    ----------
    peptide_df:
        Long-format table with one row per peptide-sample pair.
    protein_col, peptide_col, sample_col, intensity_col:
        Column names.
    summary_func:
        Callable to summarize multiple peptide intensities per protein-sample.
    min_peptides:
        Keep only proteins observed in at least this many unique peptides.
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


def aggregate_features(
    peptide_df: pd.DataFrame,
    feature_col: str,
    sample_col: str,
    intensity_col: str,
    summary_func=robust_summary,
    min_observations: int = 1,
) -> pd.DataFrame:
    """Aggregate feature (peptide, PTM, etc.) intensities into a feature x sample matrix.

    Generalizes peptide and protein aggregation to any grouping column (protein,
    peptide ID, PTM site, etc.). Use this for PTM-level analysis or any custom
    feature grouping.

    Parameters
    ----------
    peptide_df:
        Long-format table with one row per feature-sample pair.
    feature_col:
        Column name for the feature to aggregate by (e.g., 'peptide_id', 'ptm_site').
    sample_col:
        Column name for samples.
    intensity_col:
        Column name for intensity values to aggregate.
    summary_func:
        Callable to summarize multiple intensities per feature-sample.
    min_observations:
        Keep only features with at least this many distinct samples.

    Returns
    -------
    Feature x sample matrix with features as rows and samples as columns.
    """
    return _aggregate_by_feature(
        peptide_df,
        feature_col=feature_col,
        sample_col=sample_col,
        intensity_col=intensity_col,
        summary_func=summary_func,
        min_observations=min_observations,
    )
