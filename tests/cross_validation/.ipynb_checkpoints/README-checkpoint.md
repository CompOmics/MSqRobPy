# Cross-Validation: msqrobpy (Python) vs msqrob2 (R)

This directory contains scripts to verify that the Python implementation
(`msqrobpy`) produces results that match the original R package (`msqrob2`).

## Prerequisites

### Python
```bash
pip install -e ../..          # install msqrobpy in editable mode
pip install numpy pandas scipy statsmodels patsy
```

### R
```r
install.packages("BiocManager")
BiocManager::install("msqrob2")
```

## Quick Start

Run everything from this directory:

```bash
# Step 1 – Generate shared synthetic data
python generate_test_data.py

# Step 2 – Run msqrob2 in R on the shared data
Rscript run_msqrob2_r.R .

# Step 3 – Run msqrobpy and compare with R results
python compare_r_python.py
```

Or use the all-in-one script:
```bash
python run_all.py
```

## What Is Compared

| Quantity | Tolerance | Notes |
|---|---|---|
| Regression coefficients (Intercept, conditionT) | 5% relative | Should match tightly |
| Residual sigma | 5% relative | Should match tightly |
| Residual df | 5% relative | Should match exactly |
| Unscaled vcov matrix | 5% relative | Should match tightly |
| Posterior variance (empirical Bayes) | 20% relative | Implementations differ |
| Posterior df (empirical Bayes) | 20% relative | Implementations differ |
| logFC (contrast estimate) | 5% relative | Should match tightly |
| Standard error, t-stat, p-values | 20% relative | Depend on EB moderation |
| Adjusted p-values (BH) | 20% relative | Depend on EB moderation |

## Expected Differences

The empirical Bayes variance moderation (`squeezeVar` in R / `moderate_variances`
in Python) uses different estimation strategies:

- **R (limma)**: ML/REML fitting of a scaled F-distribution model
- **Python (msqrobpy)**: moment-based approximation using digamma/trigamma

This means prior df and prior variance estimates may differ, propagating into
posterior variances, standard errors, t-statistics, and p-values. The core
regression (coefficients, sigma, df) should match very closely.
