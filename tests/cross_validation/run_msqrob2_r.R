#!/usr/bin/env Rscript
# -----------------------------------------------------------------------
# Run msqrob2 on the shared test data and export all intermediate & final
# results so we can compare them numerically with the Python implementation.
#
# Prerequisites:
#   install.packages("BiocManager")
#   BiocManager::install("msqrob2")     # pulls in limma, SummarizedExperiment, etc.
# -----------------------------------------------------------------------

library(msqrob2)
library(SummarizedExperiment)
library(limma)

cat("== msqrob2 cross-validation R script ==\n")

args <- commandArgs(trailingOnly = TRUE)
data_dir <- if (length(args) >= 1) args[1] else getwd()

cat("Data directory:", data_dir, "\n")

# ---------- 1. Read shared test data ------------------------------------

intensity <- read.csv(file.path(data_dir, "intensity_matrix.csv"),
                       row.names = 1, check.names = FALSE)
col_data <- read.csv(file.path(data_dir, "sample_metadata.csv"),
                      row.names = 1, check.names = FALSE)

# Make sure ordering matches
col_data <- col_data[colnames(intensity), , drop = FALSE]
col_data$condition <- factor(col_data$condition, levels = c("control", "treated"))

cat("Intensity matrix:", nrow(intensity), "features x", ncol(intensity), "samples\n")

# ---------- 2. Build SummarizedExperiment & fit -------------------------

se <- SummarizedExperiment(
    assays = list(intensity = as.matrix(intensity)),
    colData = DataFrame(col_data)
)

# ---- 2a. Robust (rlm) fit  (maxitRob=5, the msqrobLm default) ---------
se <- msqrob(se, formula = ~condition, robust = TRUE, maxitRob = 5,
             modelColumnName = "msqrobModels_rlm", overwrite = TRUE)

# ---- 2b. OLS fit -------------------------------------------------------
se <- msqrob(se, formula = ~condition, robust = FALSE,
             modelColumnName = "msqrobModels_ols", overwrite = TRUE)

# ---------- 3. Export per-feature model parameters -----------------------

export_model_params <- function(models, tag) {
    features <- names(models)
    rows <- list()
    for (feat in features) {
        m <- models[[feat]]
        tp <- m@type
        if (tp == "fitError") next
        coefs <- getCoef(m)
        rows[[feat]] <- data.frame(
            feature_id   = feat,
            method       = tp,
            intercept    = as.numeric(coefs["(Intercept)"]),
            conditionT   = as.numeric(coefs["conditiontreated"]),
            sigma        = getSigma(m),
            variance     = getVar(m),
            df_residual  = getDF(m),
            var_posterior = getVarPosterior(m),
            df_posterior  = getDfPosterior(m),
            sigma_posterior = getSigmaPosterior(m),
            stringsAsFactors = FALSE
        )
    }
    out <- do.call(rbind, rows)
    write.csv(out, file.path(data_dir, paste0("r_model_params_", tag, ".csv")),
              row.names = FALSE)
    cat("  Wrote", nrow(out), "model params for", tag, "\n")

    # Also export vcov_unscaled for the first 10 fitted features
    first10 <- head(out$feature_id, 10)
    vcov_rows <- list()
    for (feat in first10) {
        m <- models[[feat]]
        vu <- getVcovUnscaled(m)
        if (any(is.na(vu))) next
        for (rn in rownames(vu)) {
            for (cn in colnames(vu)) {
                vcov_rows[[length(vcov_rows) + 1]] <- data.frame(
                    feature_id = feat,
                    row_name   = rn,
                    col_name   = cn,
                    value      = vu[rn, cn],
                    stringsAsFactors = FALSE
                )
            }
        }
    }
    vcov_df <- do.call(rbind, vcov_rows)
    write.csv(vcov_df, file.path(data_dir, paste0("r_vcov_unscaled_", tag, ".csv")),
              row.names = FALSE)
    cat("  Wrote vcov_unscaled for", length(first10), "features (", tag, ")\n")
    invisible(out)
}

cat("Exporting model parameters...\n")
export_model_params(rowData(se)$msqrobModels_rlm, "rlm")
export_model_params(rowData(se)$msqrobModels_ols, "ols")

# ---------- 4. Contrast testing ------------------------------------------

run_contrast <- function(models, tag) {
    L <- makeContrast("conditiontreated=0", c("conditiontreated"))
    res <- topFeatures(models, L, sort = FALSE, alpha = 1)
    res$feature_id <- rownames(res)
    write.csv(res, file.path(data_dir, paste0("r_contrast_results_", tag, ".csv")),
              row.names = FALSE)
    cat("  Wrote", nrow(res), "contrast results for", tag, "\n")
    invisible(res)
}

cat("Running contrast tests...\n")
run_contrast(rowData(se)$msqrobModels_rlm, "rlm")
run_contrast(rowData(se)$msqrobModels_ols, "ols")

# ---------- 5. Also export squeezeVar details for debugging ---------------
export_squeezevar <- function(models, tag) {
    vars <- vapply(models, getVar, numeric(1))
    dfs  <- vapply(models, getDF, numeric(1))
    # Only include successfully fitted models
    ok <- is.finite(vars) & vars > 0 & is.finite(dfs) & dfs > 0
    sv <- squeezeVar(vars[ok], dfs[ok])
    out <- data.frame(
        feature_id   = names(vars[ok]),
        sample_var   = vars[ok],
        df_residual  = dfs[ok],
        var_post     = sv$var.post,
        df_prior     = sv$df.prior,
        var_prior    = sv$var.prior,
        stringsAsFactors = FALSE
    )
    write.csv(out, file.path(data_dir, paste0("r_squeezevar_", tag, ".csv")),
              row.names = FALSE)
    cat("  Wrote squeezeVar details for", nrow(out), "features (", tag, ")\n")
    cat("    df.prior =", sv$df.prior, "\n")
    cat("    var.prior =", sv$var.prior, "\n")
}

cat("Exporting squeezeVar details...\n")
export_squeezevar(rowData(se)$msqrobModels_rlm, "rlm")
export_squeezevar(rowData(se)$msqrobModels_ols, "ols")

cat("\n== R script complete ==\n")
cat("Output files in:", data_dir, "\n")
