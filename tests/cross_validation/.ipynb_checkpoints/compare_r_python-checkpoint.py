"""
Compare msqrobpy (Python) results against msqrob2 (R) reference outputs.

Usage
-----
1. python generate_test_data.py          # create shared CSV data
2. Rscript run_msqrob2_r.R  <this_dir>  # run msqrob2 in R (writes r_*.csv)
3. python compare_r_python.py            # this script

The script loads the same intensity/metadata CSVs, runs msqrobpy, reads the R
CSV exports, and compares every numerical quantity at multiple granularities.
"""

from __future__ import annotations

import os
import sys
import textwrap

import numpy as np
import pandas as pd

# ── locate package ──────────────────────────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from msqrobpy.core import fit_protein_model  # noqa: E402

# ── tolerances ──────────────────────────────────────────────────────────────
# absolute and relative tolerance for numerical comparisons
ABS_TOL = 1e-4   # values below this are treated as ~0
REL_TOL = 0.05   # 5 % relative difference allowed

# Relaxed tolerance for empirical Bayes quantities (implementation differs)
EB_REL_TOL = 0.20  # 20 % relative

# ── helpers ─────────────────────────────────────────────────────────────────

def _rel_diff(a: float, b: float) -> float:
    """Symmetric relative difference, safe for zeros and infinities."""
    if np.isinf(a) and np.isinf(b):
        return 0.0 if np.sign(a) == np.sign(b) else 2.0
    if np.isinf(a) or np.isinf(b):
        return 2.0
    denom = (abs(a) + abs(b)) / 2.0
    if denom < ABS_TOL:
        return 0.0
    return abs(a - b) / denom


def _compare_series(
    name: str,
    py_vals: pd.Series,
    r_vals: pd.Series,
    rel_tol: float = REL_TOL,
    label: str = "",
) -> dict:
    """Element-wise comparison between aligned pandas Series."""
    common = py_vals.dropna().index.intersection(r_vals.dropna().index)
    if len(common) == 0:
        return {"name": name, "n_compared": 0, "max_rel_diff": np.nan, "pass": False}

    py = py_vals.loc[common].astype(float)
    r = r_vals.loc[common].astype(float)
    rel = pd.Series([_rel_diff(a, b) for a, b in zip(py, r)], index=common)
    max_rd = rel.max()
    median_rd = rel.median()
    n_fail = (rel > rel_tol).sum()
    ok = n_fail == 0

    return {
        "name": f"{label}{name}" if label else name,
        "n_compared": len(common),
        "max_rel_diff": float(max_rd),
        "median_rel_diff": float(median_rd),
        "n_fail": int(n_fail),
        "pass": ok,
    }


def _section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def _report(result: dict):
    status = "PASS ✓" if result["pass"] else "FAIL ✗"
    print(
        f"  [{status}]  {result['name']:40s}  "
        f"n={result['n_compared']:4d}  "
        f"max_rel_diff={result.get('max_rel_diff', np.nan):.6f}  "
        f"median_rel_diff={result.get('median_rel_diff', np.nan):.6f}  "
        f"n_fail={result.get('n_fail', '?')}"
    )


# ── main ────────────────────────────────────────────────────────────────────

def main():
    # ------------------------------------------------------------------ data
    intensity_df = pd.read_csv(
        os.path.join(THIS_DIR, "intensity_matrix.csv"), index_col=0
    )
    sample_metadata = pd.read_csv(
        os.path.join(THIS_DIR, "sample_metadata.csv"), index_col=0
    )
    sample_metadata["condition"] = pd.Categorical(
        sample_metadata["condition"], categories=["control", "treated"]
    )

    all_results: list[dict] = []

    for robust, tag in [(True, "rlm"), (False, "ols")]:
        _section(f"{'Robust (RLM)' if robust else 'OLS'} – comparing model fit")

        # ---- check R reference files exist ----
        model_file = os.path.join(THIS_DIR, f"r_model_params_{tag}.csv")
        contrast_file = os.path.join(THIS_DIR, f"r_contrast_results_{tag}.csv")
        squeeze_file = os.path.join(THIS_DIR, f"r_squeezevar_{tag}.csv")
        vcov_file = os.path.join(THIS_DIR, f"r_vcov_unscaled_{tag}.csv")

        for f in [model_file, contrast_file, squeeze_file]:
            if not os.path.exists(f):
                print(f"  !! Missing R reference file: {f}")
                print(f"     Run 'Rscript run_msqrob2_r.R {THIS_DIR}' first.")
                sys.exit(1)

        # ---- Python fit ----
        fit = fit_protein_model(
            intensity_df,
            sample_metadata,
            "~ condition",
            robust=robust,
            empirical_bayes=True,
        )

        # ---- Read R references ----
        r_params = pd.read_csv(model_file).set_index("feature_id")
        r_contrast = pd.read_csv(contrast_file).set_index("feature_id")
        r_squeeze = pd.read_csv(squeeze_file).set_index("feature_id")

        # ---- Build Python param table ----
        py_rows = []
        for fid, m in fit.models.items():
            py_rows.append(
                {
                    "feature_id": fid,
                    "intercept": m.coef.get("Intercept", m.coef.get("(Intercept)", np.nan)),
                    "conditionT": m.coef.get(
                        "condition[T.treated]", np.nan
                    ),
                    "sigma": m.sigma,
                    "variance": m.var,
                    "df_residual": m.df_residual,
                    "var_posterior": m.var_posterior,
                    "df_posterior": m.df_posterior,
                    "sigma_posterior": m.sigma_posterior,
                }
            )
        py_params = pd.DataFrame(py_rows).set_index("feature_id")

        # ---- Compare coefficients --------------------------------------------------------
        common_features = py_params.index.intersection(r_params.index)
        print(f"\n  Features in common: {len(common_features)}  "
              f"(Python={len(py_params)}, R={len(r_params)})")

        # Intercept
        r = _compare_series("intercept", py_params["intercept"], r_params["intercept"])
        _report(r); all_results.append(r)

        # conditionT (log fold change coefficient)
        r = _compare_series("conditionT (coef)", py_params["conditionT"], r_params["conditionT"])
        _report(r); all_results.append(r)

        # sigma
        r = _compare_series("sigma", py_params["sigma"], r_params["sigma"])
        _report(r); all_results.append(r)

        # df_residual
        r = _compare_series("df_residual", py_params["df_residual"], r_params["df_residual"])
        _report(r); all_results.append(r)

        # ---- Compare vcov_unscaled (first 10 features) ----------------------------------
        if os.path.exists(vcov_file):
            r_vcov = pd.read_csv(vcov_file)
            n_vcov_match = 0
            n_vcov_total = 0
            max_vcov_rd = 0.0
            for fid in r_vcov["feature_id"].unique():
                if fid not in fit.models:
                    continue
                sub = r_vcov[r_vcov["feature_id"] == fid]
                py_vcov = fit.models[fid].vcov_unscaled
                for _, row in sub.iterrows():
                    rn, cn, rv = row["row_name"], row["col_name"], row["value"]
                    # Map R names to Python names
                    rn_py = rn.replace("(Intercept)", "Intercept").replace("conditiontreated", "condition[T.treated]")
                    cn_py = cn.replace("(Intercept)", "Intercept").replace("conditiontreated", "condition[T.treated]")
                    if rn_py in py_vcov.index and cn_py in py_vcov.columns:
                        pv = py_vcov.loc[rn_py, cn_py]
                        rd = _rel_diff(float(pv), float(rv))
                        max_vcov_rd = max(max_vcov_rd, rd)
                        n_vcov_total += 1
                        if rd <= REL_TOL:
                            n_vcov_match += 1
            ok = n_vcov_total > 0 and n_vcov_match == n_vcov_total
            result = {
                "name": f"vcov_unscaled ({tag})",
                "n_compared": n_vcov_total,
                "max_rel_diff": max_vcov_rd,
                "median_rel_diff": 0.0,
                "n_fail": n_vcov_total - n_vcov_match,
                "pass": ok,
            }
            _report(result); all_results.append(result)

        # ---- Compare empirical Bayes quantities (relaxed tolerance) ----------------------
        _section(f"{'Robust (RLM)' if robust else 'OLS'} – empirical Bayes (squeezeVar)")

        r = _compare_series(
            f"var_posterior ({tag})",
            py_params["var_posterior"],
            r_params["var_posterior"],
            rel_tol=EB_REL_TOL,
        )
        _report(r); all_results.append(r)

        r = _compare_series(
            f"df_posterior ({tag})",
            py_params["df_posterior"],
            r_params["df_posterior"],
            rel_tol=EB_REL_TOL,
        )
        _report(r); all_results.append(r)

        # Print prior df and prior var from R for information
        if len(r_squeeze) > 0:
            r_df_prior = r_squeeze["df_prior"].iloc[0]
            r_var_prior = r_squeeze["var_prior"].iloc[0]
            # Compute Python prior (re-run moderation to get these)
            from msqrobpy.moderation import estimate_prior_df_var
            py_vars = np.array([fit.models[f].var for f in common_features if f in fit.models])
            py_dfs = np.array([fit.models[f].df_residual for f in common_features if f in fit.models])
            py_df0, py_s02 = estimate_prior_df_var(py_vars, py_dfs)
            print(f"\n  Prior df  :  R = {r_df_prior:.4f}  |  Python = {py_df0:.4f}  |  rel_diff = {_rel_diff(r_df_prior, py_df0):.4f}")
            print(f"  Prior var :  R = {r_var_prior:.6f}  |  Python = {py_s02:.6f}  |  rel_diff = {_rel_diff(r_var_prior, py_s02):.4f}")

        # ---- Compare contrast results ---------------------------------------------------
        _section(f"{'Robust (RLM)' if robust else 'OLS'} – contrast test results")

        contrast_result = fit.test_contrast("condition[T.treated]")
        py_contrast = contrast_result.table.set_index("feature_id")

        for col_py, col_r in [
            ("estimate", "logFC"),
            ("std_error", "se"),
            ("df", "df"),
            ("t_stat", "t"),
            ("p_value", "pval"),
            ("adj_p_value", "adjPval"),
        ]:
            # Determine tolerance (relaxed for EB-dependent quantities)
            tol = REL_TOL if col_py in ("estimate",) else EB_REL_TOL
            r = _compare_series(
                f"{col_py} ({tag})",
                py_contrast[col_py],
                r_contrast[col_r],
                rel_tol=tol,
            )
            _report(r); all_results.append(r)

    # ── Summary ─────────────────────────────────────────────────────────────
    _section("SUMMARY")
    n_pass = sum(1 for r in all_results if r["pass"])
    n_total = len(all_results)
    print(f"\n  Total checks: {n_total}")
    print(f"  Passed:       {n_pass}")
    print(f"  Failed:       {n_total - n_pass}")

    if n_pass == n_total:
        print("\n  *** ALL CHECKS PASSED ***")
    else:
        print("\n  Failed checks:")
        for r in all_results:
            if not r["pass"]:
                print(f"    - {r['name']}  (max_rel_diff={r.get('max_rel_diff', '?'):.6f}, n_fail={r.get('n_fail', '?')})")
        print()
        print(textwrap.dedent("""\
            NOTE: Differences are expected in the empirical Bayes quantities
            because msqrobpy uses a moment-based approximation for squeezeVar
            while the R limma package uses a more sophisticated fitting algorithm.
            
            Tight agreement on coefficients, sigma, and df_residual (before
            moderation) confirms the core regression is equivalent. Moderate
            deviations in posterior variance / df and downstream p-values
            reflect the squeezeVar implementation difference.
        """))

    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
