from msqrobpy.core import fit_protein_model
from msqrobpy.example import simulate_protein_data

intensity_df, sample_metadata = simulate_protein_data()
fit = fit_protein_model(intensity_df, sample_metadata, "~ condition", robust=False)
res = fit.test_contrast("condition[T.treated]")
print(res.top(10))
