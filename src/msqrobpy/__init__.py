"""msqrobpy: Python-native differential abundance analysis for LC-MS proteomics."""

from .aggregation import aggregate_peptides, robust_summary
from .core import fit_protein_model
from .hurdle import fit_hurdle_model
from .results import FeatureModelResult, MsqrobFit, ContrastResult

__all__ = [
    "aggregate_peptides",
    "robust_summary",
    "fit_protein_model",
    "fit_hurdle_model",
    "FeatureModelResult",
    "MsqrobFit",
    "ContrastResult",
]
