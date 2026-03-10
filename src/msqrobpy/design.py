from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd
import patsy


_RANDOM_EFFECT_RE = re.compile(r"\([^)]*\|[^)]*\)")


def strip_random_effects(formula: str) -> str:
    """Remove lme4-style random effect terms from a formula.

    This allows a Python-native fixed-effect fallback API while clearly making
    mixed-effect support an explicit future extension.
    """
    cleaned = _RANDOM_EFFECT_RE.sub("", formula)
    cleaned = re.sub(r"\+\s*\+", "+", cleaned)
    cleaned = re.sub(r"~\s*\+", "~", cleaned)
    cleaned = re.sub(r"\+\s*$", "", cleaned)
    return cleaned.strip()


def build_design_matrix(formula: str, data: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Construct a fixed-effect design matrix from a formula and sample metadata."""
    clean = strip_random_effects(formula)
    X = patsy.dmatrix(clean, data=data, return_type="dataframe")
    return np.asarray(X, dtype=float), list(X.columns)


def contrast_vector(design_columns: Iterable[str], expression: str) -> pd.Series:
    """Create a contrast vector from a symbolic linear expression.

    The function supports direct references to coefficient names even when the
    names are not valid Python identifiers, for example
    ``condition[T.treated] - batch[T.B]``.
    """
    columns = list(design_columns)
    basis = {f"b{i}": np.eye(len(columns))[i] for i, _ in enumerate(columns)}
    rewritten = str(expression)
    # Replace longer names first to avoid partial matches.
    for i, col in sorted(enumerate(columns), key=lambda x: len(x[1]), reverse=True):
        rewritten = rewritten.replace(col, f"b{i}")
    # Exact coefficient name is a very common shorthand.
    if rewritten == expression and expression in columns:
        rewritten = f"b{columns.index(expression)}"
    try:
        value = eval(rewritten, {"__builtins__": {}}, basis)
    except Exception as exc:
        raise ValueError(f"Could not parse contrast expression: {expression}") from exc
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.shape[0] != len(columns):
        raise ValueError("Contrast length does not match design.")
    return pd.Series(arr, index=columns, dtype=float)
