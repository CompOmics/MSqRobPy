import numpy as np
import pandas as pd

from msqrobpy.core import fit_protein_model
from msqrobpy.example import simulate_protein_data


def test_fit_and_contrast():
    intensity_df, sample_metadata = simulate_protein_data(n_features=60, n_replicates_per_group=4)
    fit = fit_protein_model(intensity_df, sample_metadata, "~ condition", robust=False, empirical_bayes=True)
    res = fit.test_contrast("condition[T.treated]")
    assert len(res.table) > 0
    assert "adj_p_value" in res.table.columns
    assert np.isfinite(res.table["estimate"]).any()
