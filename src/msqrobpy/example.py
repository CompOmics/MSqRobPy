from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_protein_data(
    n_features: int = 200,
    n_replicates_per_group: int = 4,
    effect_fraction: float = 0.1,
    effect_size: float = 1.0,
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a small synthetic protein-abundance dataset for testing and demos."""
    rng = np.random.default_rng(seed)
    n_samples = 2 * n_replicates_per_group
    samples = [f"S{i+1}" for i in range(n_samples)]
    condition = np.array(["control"] * n_replicates_per_group + ["treated"] * n_replicates_per_group)
    sample_metadata = pd.DataFrame({"condition": condition}, index=samples)

    baseline = rng.normal(25.0, 1.0, size=(n_features, 1))
    noise = rng.normal(0.0, 0.5, size=(n_features, n_samples))
    effects = np.zeros((n_features, n_samples))
    de_idx = rng.choice(n_features, int(n_features * effect_fraction), replace=False)
    effects[de_idx, n_replicates_per_group:] = effect_size
    mat = baseline + noise + effects
    intensity_df = pd.DataFrame(mat, index=[f"P{i+1}" for i in range(n_features)], columns=samples)
    return intensity_df, sample_metadata
