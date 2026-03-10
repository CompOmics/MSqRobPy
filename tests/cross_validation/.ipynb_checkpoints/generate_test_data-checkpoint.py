"""Generate a shared synthetic dataset for cross-validating R (msqrob2) vs Python (msqrobpy).

Writes CSV files that both R and Python can read.
"""

import os
import numpy as np
import pandas as pd

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def generate(
    n_features: int = 100,
    n_replicates_per_group: int = 4,
    effect_fraction: float = 0.15,
    effect_size: float = 1.5,
    seed: int = 42,
    missing_fraction: float = 0.05,
):
    rng = np.random.default_rng(seed)
    n_samples = 2 * n_replicates_per_group
    samples = [f"S{i+1}" for i in range(n_samples)]
    condition = (
        ["control"] * n_replicates_per_group
        + ["treated"] * n_replicates_per_group
    )
    sample_metadata = pd.DataFrame({"condition": condition}, index=samples)

    baseline = rng.normal(25.0, 1.5, size=(n_features, 1))
    noise = rng.normal(0.0, 0.4, size=(n_features, n_samples))
    effects = np.zeros((n_features, n_samples))
    n_de = int(n_features * effect_fraction)
    de_idx = rng.choice(n_features, n_de, replace=False)
    effects[de_idx, n_replicates_per_group:] = effect_size

    mat = baseline + noise + effects

    # Introduce some missing values (NaN)
    mask = rng.random(mat.shape) < missing_fraction
    mat[mask] = np.nan

    feature_ids = [f"P{i+1:04d}" for i in range(n_features)]
    intensity_df = pd.DataFrame(mat, index=feature_ids, columns=samples)

    # Save
    intensity_df.to_csv(os.path.join(OUT_DIR, "intensity_matrix.csv"))
    sample_metadata.to_csv(os.path.join(OUT_DIR, "sample_metadata.csv"))

    # Also save the DE ground truth
    de_labels = pd.DataFrame(
        {"feature_id": feature_ids, "is_de": [i in de_idx for i in range(n_features)]}
    )
    de_labels.to_csv(os.path.join(OUT_DIR, "de_ground_truth.csv"), index=False)

    print(f"Generated {n_features} features x {n_samples} samples")
    print(f"  DE features: {n_de} (effect_size={effect_size})")
    print(f"  Missing fraction: {missing_fraction}")
    print(f"  Files written to: {OUT_DIR}")


if __name__ == "__main__":
    generate()
