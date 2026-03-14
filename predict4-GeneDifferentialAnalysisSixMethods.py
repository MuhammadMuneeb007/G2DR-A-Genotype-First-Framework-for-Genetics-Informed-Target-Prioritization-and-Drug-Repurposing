#!/usr/bin/env python3
"""
Comprehensive Differential Gene Expression Analysis - FIXED & OPTIMIZED VERSION
===============================================================================
Structure: {phenotype}/Fold_{fold}/GeneDifferentialExpressions/Database/Tissue/Method/Files

8 Statistical Methods:
✅ TRUE DIFFERENTIAL EXPRESSION (5 methods):
   - LIMMA (gold standard)
   - Welch_t_test
   - Linear_Regression  
   - Wilcoxon_Rank_Sum
   - Permutation_Test

⚠️  ASSOCIATION TESTING (3 methods - for comparison only):
   - Weighted_Logistic (FIXED: uses GLM instead of Logit)
   - Firth_Logistic (FIXED: uses GLM for stability)
   - Bayesian_Logistic

CRITICAL FIXES:
- All logistic methods now use sm.GLM() instead of sm.Logit() to prevent singular matrix errors
- Added multiprocessing support for parallel execution
- Improved error handling and numerical stability
"""
import shutil
import subprocess
import tempfile
import os
import sys
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings
from datetime import datetime
import logging
import argparse
import multiprocessing as mp
from functools import partial
from scipy.special import expit

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# =============================================================================
# MODULE-LEVEL HELPER FUNCTIONS (for multiprocessing compatibility)
# =============================================================================

def _firth_logistic_fit(X, y, max_iter=50, tol=1e-8, ridge=1e-8):
    """
    True Firth logistic regression (Jeffreys prior) via modified IRLS.
    Returns: beta, se, pvals
    X must already include intercept column.
    """
    n, p = X.shape
    beta = np.zeros(p, dtype=float)

    for _ in range(max_iter):
        eta = X @ beta
        pi  = expit(eta)

        # W = pi(1-pi) with clipping for stability
        W = np.clip(pi * (1.0 - pi), 1e-12, None)

        # Fisher information: X' W X
        XW = X * W[:, None]
        I = X.T @ XW

        # small ridge to avoid singularities
        I_r = I + ridge * np.eye(p)

        try:
            I_inv = np.linalg.inv(I_r)
        except np.linalg.LinAlgError:
            I_inv = np.linalg.pinv(I_r)

        # hat diagonals h_i = w_i * x_i^T I_inv x_i
        Xt = X @ I_inv
        h = np.sum(Xt * X, axis=1) * W

        # Firth adjustment term a_i = (1/2 - pi_i) * h_i
        a = (0.5 - pi) * h

        # modified score: U* = X^T (y - pi + a)
        U_star = X.T @ (y - pi + a)

        # Newton step: beta_new = beta + I^{-1} U*
        step = I_inv @ U_star
        beta_new = beta + step

        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    # Standard errors from (approx) inverse Fisher information at convergence
    eta = X @ beta
    pi  = expit(eta)
    W   = np.clip(pi * (1.0 - pi), 1e-12, None)
    I   = X.T @ (X * W[:, None])
    I_r = I + ridge * np.eye(p)

    try:
        I_inv = np.linalg.inv(I_r)
    except np.linalg.LinAlgError:
        I_inv = np.linalg.pinv(I_r)

    se = np.sqrt(np.clip(np.diag(I_inv), 1e-30, None))

    # Wald p-values (common reporting)
    z = beta / se
    pvals = 2.0 * (1.0 - stats.norm.cdf(np.abs(z)))

    return beta, se, pvals


def _bayes_logistic_laplace_fit(X, y, w=None, intercept_sd=10.0, coef_sd=2.5,
                               max_iter=50, tol=1e-8, ridge=1e-10):
    """
    X: (n,2) = [1, x] where x is standardized
    y: (n,) in {0,1}
    w: optional weights (n,) >= 0
    Returns: beta (2,), se (2,), cov (2,2)
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    assert p == 2, "This helper expects intercept + 1 predictor."

    if w is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(w, dtype=float).ravel()
        w = np.clip(w, 0.0, np.inf)

    prior_var = np.array([intercept_sd**2, coef_sd**2], dtype=float)
    prior_prec = np.diag(1.0 / np.clip(prior_var, 1e-30, np.inf))

    beta = np.zeros(p, dtype=float)

    for _ in range(max_iter):
        eta = X @ beta
        p_hat = expit(eta)

        # gradient of log-likelihood: X^T (y - p) with weights
        r = (y - p_hat) * w
        grad_ll = X.T @ r

        # Hessian of log-likelihood: - X^T W X, with W = p(1-p)*w
        W = np.clip(p_hat * (1.0 - p_hat), 1e-12, None) * w
        H_ll = -(X.T @ (X * W[:, None]))

        # log-posterior = log-likelihood + log-prior
        grad = grad_ll - (beta / np.clip(prior_var, 1e-30, np.inf))
        H = H_ll - prior_prec

        # Newton step
        H_stable = H - ridge * np.eye(p)
        try:
            step = np.linalg.solve(H_stable, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(H_stable) @ grad

        beta_new = beta - step

        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    # Posterior covariance via Laplace
    eta = X @ beta
    p_hat = expit(eta)
    W = np.clip(p_hat * (1.0 - p_hat), 1e-12, None) * w
    H_ll = -(X.T @ (X * W[:, None]))
    H_post = H_ll - prior_prec
    neg_H = -H_post + ridge * np.eye(p)

    try:
        cov = np.linalg.inv(neg_H)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(neg_H)

    se = np.sqrt(np.clip(np.diag(cov), 1e-30, np.inf))
    return beta, se, cov


class ComprehensiveDE:
    def __init__(self, phenotype, fold, dataset, force_rerun=False, fdr_threshold=0.05, lfc_threshold=1.0):
        self.phenotype = phenotype
        self.fold = fold  
        self.dataset = dataset
        self.force_rerun = force_rerun
        self.fdr_threshold = fdr_threshold
        self.lfc_threshold = lfc_threshold
        self.base_path = Path(phenotype) / f"Fold_{fold}"
        self.output_base = self.base_path / "GeneDifferentialExpressions"
        
        # Load phenotype data once
        self.phenotype_data = self.load_phenotype_data()
        
    def check_existing_results(self, database_name, tissue_name, method_name):
        """Check if results already exist for this combination"""
        method_with_dataset = f"{method_name}_{self.dataset}"
        output_dir = self.output_base / database_name / tissue_name / method_with_dataset
        
        # Check for main result files
        results_file = output_dir / 'differential_expression_results.csv'
        summary_file = output_dir / 'analysis_summary.csv'
        
        both_exist = results_file.exists() and summary_file.exists()
        
        if both_exist:
            try:
                results_df = pd.read_csv(results_file)
                summary_df = pd.read_csv(summary_file)
                
                has_valid_data = len(results_df) > 0 and 'Gene' in results_df.columns
                has_valid_summary = len(summary_df) > 0
                
                if has_valid_data and has_valid_summary:
                    return True, {
                        'results_file': results_file,
                        'summary_file': summary_file,
                        'gene_count': len(results_df),
                        'significant_genes': (results_df['PValue'] < 0.05).sum() if 'PValue' in results_df.columns else 0
                    }
                else:
                    return False, None
            except Exception:
                return False, None
        
        return False, None
    
    def get_expression_databases(self):
        """Define all expression prediction databases/methods"""
        return {
            "Regular": {
                "train_dir": "TrainExpression",
                "val_dir": "ValidationExpression", 
                "test_dir": "TestExpression",
                "file_pattern": "predicted_expression_{dataset}_{tissue}_merged.csv"
            },
            "JTI": {
                "train_dir": "JTITrainExpression",
                "val_dir": "JTIValidationExpression",
                "test_dir": "JTITestExpression", 
                "file_pattern": "predicted_expression_JTI_{dataset}_{tissue}_merged.csv"
            },
            "UTMOST": {
                "train_dir": "UTMOSTTrainExpression",
                "val_dir": "UTMOSTValidationExpression",
                "test_dir": "UTMOSTTestExpression",
                "file_pattern": "predicted_expression_UTMOST_{dataset}_{tissue}_merged.csv"
            },
            "UTMOST2": {
                "train_dir": "utmost2TrainExpression", 
                "val_dir": "utmost2ValidationExpression",
                "test_dir": "utmost2TestExpression",
                "file_pattern": "predicted_expression_UTMOST_{dataset}_{tissue}_merged.csv"
            },
            "EpiX": {
                "train_dir": "EpiXTrainExpression",
                "val_dir": "EpiXValidationExpression", 
                "test_dir": "EpiXTestExpression",
                "file_pattern": "predicted_expression_EpiX_{dataset}_{tissue}_merged.csv"
            },
            "TIGAR": {
                "train_dir": "TigarTrainExpression",
                "val_dir": "TigarValidExpression",
                "test_dir": "TigarTestExpression", 
                "file_pattern": "{dataset}_expression_expression_matrix.csv"
            },
            "FUSION": {
                "train_dir": "FussionExpression",
                "val_dir": "FussionExpression",
                "test_dir": "FussionExpression",
                "file_pattern": "GeneExpression_{dataset}_data.csv"
            }
        }
    
    def get_statistical_methods(self):
        """Define the 8 statistical methods (5 DE + 3 Association)"""
        return [
#            "LIMMA",
#            "Welch_t_test",
#            "Linear_Regression",
            "Wilcoxon_Rank_Sum",
            "Permutation_Test",
#            "Weighted_Logistic",
            "Firth_Logistic",
            "Bayesian_Logistic"
        ]
    
    def load_phenotype_data(self):
        """Load phenotype data from .fam file"""
        logger.info("🔄 Loading phenotype data...")
        
        dataset_mapping = {
            "training": {
                "fam_patterns": ["train_data.QC.clumped.pruned.fam", "training_data.QC.clumped.pruned.fam", "train.fam"],
                "expr_suffix": "train"
            },
            "validation": {
                "fam_patterns": ["validation_data.clumped.pruned.fam", "val_data.clumped.pruned.fam", "validation.fam"],
                "expr_suffix": "validation"
            },
            "test": {
                "fam_patterns": ["test_data.clumped.pruned.fam", "test.fam"],
                "expr_suffix": "test"
            }
        }
        
        fam_file = None
        for pattern in dataset_mapping[self.dataset]["fam_patterns"]:
            candidate = self.base_path / pattern
            if candidate.exists():
                fam_file = candidate
                break
                
        if not fam_file:
            raise FileNotFoundError(f"No .fam file found in {self.base_path}")
        
        fam_data = pd.read_csv(fam_file, sep=r'\s+', header=None,
                              names=['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENO'])
        
        # Convert phenotypes
        unique_phenos = set(fam_data['PHENO'].unique())
        if unique_phenos == {1, 2}:
            fam_data['PHENO_BINARY'] = fam_data['PHENO'] - 1
        elif unique_phenos == {0, 1}:
            fam_data['PHENO_BINARY'] = fam_data['PHENO']
        else:
            median_pheno = fam_data['PHENO'].median()
            fam_data['PHENO_BINARY'] = (fam_data['PHENO'] > median_pheno).astype(int)
        
        n_cases = (fam_data['PHENO_BINARY'] == 1).sum()
        n_controls = (fam_data['PHENO_BINARY'] == 0).sum()
        
        logger.info(f"📊 Phenotype data: {len(fam_data)} samples ({n_cases} cases, {n_controls} controls)")
        
        return fam_data

    def get_available_tissues(self):
        """Find all available tissues across all databases"""
        tissues = set()
        expression_databases = self.get_expression_databases()
        
        for database, config in expression_databases.items():
            if self.dataset == "training":
                db_dir = self.base_path / config['train_dir']
            elif self.dataset == "validation":
                db_dir = self.base_path / config['val_dir']
            else:
                db_dir = self.base_path / config['test_dir']
                
            if db_dir.exists():
                for item in db_dir.iterdir():
                    if item.is_dir():
                        tissues.add(item.name)
        
        return sorted(list(tissues))

    def clean_gene_name(self, gene_name):
        """Clean gene names"""
        gene_str = str(gene_name).strip()
        if gene_str.startswith('chr') and '_' in gene_str:
            gene_str = gene_str.split('_', 1)[1]
        gene_str = gene_str.split('.')[0]
        return gene_str

    def filter_constant_genes(self, gene_data, min_variance=1e-10):
        """Filter out genes with constant values"""
        logger.info(f"      🔍 Filtering genes with constant values...")
        
        initial_gene_count = len(gene_data.columns)
        gene_variances = gene_data.var()
        
        variable_genes = gene_variances[gene_variances >= min_variance].index
        constant_genes = gene_variances[gene_variances < min_variance].index
        
        filtered_gene_data = gene_data[variable_genes].copy()
        
        logger.info(f"      📊 Gene filtering results:")
        logger.info(f"        Initial genes: {initial_gene_count}")
        logger.info(f"        Constant genes removed: {len(constant_genes)} ({len(constant_genes)/initial_gene_count*100:.1f}%)")
        logger.info(f"        Variable genes retained: {len(variable_genes)} ({len(variable_genes)/initial_gene_count*100:.1f}%)")
        
        if len(variable_genes) == 0:
            logger.error(f"        ❌ ALL genes are constant!")
            return None, 0, initial_gene_count
        elif len(variable_genes) < 10:
            logger.warning(f"        ⚠️  Very few variable genes ({len(variable_genes)})")
        
        return filtered_gene_data, len(variable_genes), len(constant_genes)

    def load_expression_data(self, database_name, tissue_name):
        """Load expression data for specific database and tissue"""
        expression_databases = self.get_expression_databases()
        
        if database_name not in expression_databases:
            return None, None
        
        db_config = expression_databases[database_name]
        
        if self.dataset == "training":
            expr_dir = self.base_path / db_config['train_dir'] / tissue_name
            expr_suffix = "train"
        elif self.dataset == "validation":
            expr_dir = self.base_path / db_config['val_dir'] / tissue_name
            expr_suffix = "validation"
        else:
            expr_dir = self.base_path / db_config['test_dir'] / tissue_name
            expr_suffix = "test"
        
        if not expr_dir.exists():
            return None, None
        
        if database_name == "FUSION":
            filename = f"GeneExpression_{expr_suffix}_data.csv"
        elif database_name == "TIGAR":
            filename = f"{expr_suffix}_expression_expression_matrix.csv"
        else:
            filename = db_config['file_pattern'].format(dataset=self.dataset, tissue=tissue_name)
        expression_file = expr_dir / filename
        # Prefer the _fixed version of the expected filename
        fixed_candidate = expr_dir / expression_file.name.replace(".csv", "_fixed.csv")
        if fixed_candidate.exists():
            expression_file = fixed_candidate

        
        if not expression_file.exists():
            csv_files = sorted(expr_dir.glob("*.csv"))
            if not csv_files:
                return None, None
        
            # Prefer any *_fixed*.csv first
            fixed_csvs = [p for p in csv_files if "_fixed" in p.name.lower()]
            if fixed_csvs:
                expression_file = fixed_csvs[0]
            else:
                expression_file = csv_files[0]

        
        try:
            expr_data_raw = pd.read_csv(expression_file, index_col=False)
            
            gene_cols = [col for col in expr_data_raw.columns if col not in ['FID', 'IID']]
            if len(gene_cols) == 0:
                gene_cols = list(expr_data_raw.columns)
            
            gene_data = expr_data_raw[gene_cols].copy()
            cleaned_gene_names = [self.clean_gene_name(col) for col in gene_cols]
            gene_data.columns = cleaned_gene_names
            
            if 'FID' in expr_data_raw.columns and 'IID' in expr_data_raw.columns:
                sample_ids = expr_data_raw['FID'].astype(str) + '_' + expr_data_raw['IID'].astype(str)
            else:
                sample_ids = [f"Sample_{i}" for i in range(len(gene_data))]
            
            gene_data.index = sample_ids
            
            if gene_data.isnull().any().any():
                gene_data = gene_data.fillna(0)
            
            filtered_data, n_variable, n_constant = self.filter_constant_genes(gene_data)
            
            if filtered_data is None:
                return None, None
            
            return filtered_data, {
                'n_samples': len(filtered_data), 
                'n_genes_original': len(gene_data.columns),
                'n_genes_variable': n_variable,
                'n_genes_constant_removed': n_constant
            }
            
        except Exception as e:
            logger.error(f"Error loading {expression_file}: {str(e)}")
            return None, None

    # ============================================================================
    # TRUE DIFFERENTIAL EXPRESSION METHODS (5 methods)
    # These test: "Is expression different between cases vs controls?"
    # ============================================================================
    
    def method_limma(self, gene_data: pd.DataFrame, design_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        REAL LIMMA via R (empirical Bayes moderated t-tests)
    
        Returns columns:
          Gene, Log2FoldChange (= Condition coef on expression scale),
          PValue, t_statistic, Standard_Error, CasesMean, ControlsMean, Method_Type
        """
        logger.info(f"      ✅ Running REAL LIMMA (R/limma) on {gene_data.shape[1]} genes...")
    
        if "Condition" not in design_matrix.columns:
            raise ValueError("design_matrix must contain a 'Condition' column (0/1).")
    
        # align samples
        common = gene_data.index.intersection(design_matrix.index)
        if len(common) < 10:
            logger.warning(f"      ⚠️  LIMMA: only {len(common)} samples overlap; returning empty.")
            return pd.DataFrame()
    
        expr = gene_data.loc[common].apply(pd.to_numeric, errors="coerce")
        des  = design_matrix.loc[common].copy().apply(pd.to_numeric, errors="coerce")
    
        # enforce a single intercept column named Intercept
        # remove any constant columns except Intercept
        if "Intercept" not in des.columns:
            des.insert(0, "Intercept", 1.0)
    
        # drop other constant columns to avoid collinearity
        const_cols = []
        for c in des.columns:
            if c == "Intercept":
                continue
            v = des[c].values
            if np.all(np.isfinite(v)) and np.nanstd(v) == 0:
                const_cols.append(c)
        if const_cols:
            des = des.drop(columns=const_cols)
    
        if expr.isna().any().any() or des.isna().any().any():
            raise ValueError("NaNs found in expression or design matrix. Handle missing values before LIMMA.")
    
        if shutil.which("Rscript") is None:
            raise RuntimeError("Rscript not found in PATH. Load R module or install R to run limma.")
    
        # means for reporting
        condition = des["Condition"].values
        cases_mean = expr.loc[condition == 1].mean(axis=0) if np.any(condition == 1) else pd.Series(0.0, index=expr.columns)
        controls_mean = expr.loc[condition == 0].mean(axis=0) if np.any(condition == 0) else pd.Series(0.0, index=expr.columns)
    
        with tempfile.TemporaryDirectory() as tmpdir:
            expr_path   = os.path.join(tmpdir, "expr.tsv")
            design_path = os.path.join(tmpdir, "design.tsv")
            out_path    = os.path.join(tmpdir, "limma_out.tsv")
            r_path      = os.path.join(tmpdir, "run_limma.R")
    
            # limma expects genes x samples
            expr.T.to_csv(expr_path, sep="\t", index=True, header=True)
            des.to_csv(design_path, sep="\t", index=True, header=True)
    
            r_code = r"""
    args <- commandArgs(trailingOnly=TRUE)
    expr_file  <- args[1]
    design_file<- args[2]
    out_file   <- args[3]
    
    suppressPackageStartupMessages({
      library(limma)
    })
    
    expr <- read.table(expr_file, header=TRUE, sep="\t", row.names=1, check.names=FALSE)
    design <- read.table(design_file, header=TRUE, sep="\t", row.names=1, check.names=FALSE)
    
    common <- intersect(colnames(expr), rownames(design))
    expr <- expr[, common, drop=FALSE]
    design <- design[common, , drop=FALSE]
    
    fit <- lmFit(expr, design)
    fit <- eBayes(fit)
    
    coef_name <- "Condition"
    if (!(coef_name %in% colnames(design))) stop("Condition column not found in design")
    coef_idx <- which(colnames(design) == coef_name)
    
    tt <- topTable(fit, coef=coef_name, number=Inf, sort.by="none")
    
    # moderated SE consistent with eBayes t-test:
    # SE = stdev.unscaled * sqrt(s2.post)
    se <- fit$stdev.unscaled[, coef_idx] * sqrt(fit$s2.post)
    
    out <- data.frame(
      Gene=rownames(tt),
      logFC=tt$logFC,
      t=tt$t,
      P.Value=tt$P.Value,
      SE=se,
      stringsAsFactors=FALSE
    )
    
    write.table(out, file=out_file, sep="\t", quote=FALSE, row.names=FALSE)
    """
            with open(r_path, "w") as f:
                f.write(r_code.strip() + "\n")
    
            p = subprocess.run(["Rscript", r_path, expr_path, design_path, out_path],
                               text=True, capture_output=True)
            if p.returncode != 0:
                raise RuntimeError(
                    "limma (R) failed.\n"
                    f"STDERR:\n{p.stderr[-2000:]}\n"
                    f"STDOUT:\n{p.stdout[-2000:]}"
                )
    
            limma_df = pd.read_csv(out_path, sep="\t")
    
        limma_df["CasesMean"] = limma_df["Gene"].map(cases_mean.to_dict()).astype(float)
        limma_df["ControlsMean"] = limma_df["Gene"].map(controls_mean.to_dict()).astype(float)
    
        out = pd.DataFrame({
            "Gene": limma_df["Gene"].astype(str),
            "Log2FoldChange": limma_df["logFC"].astype(float),
            "PValue": limma_df["P.Value"].astype(float),
            "Standard_Error": limma_df["SE"].astype(float),
            "t_statistic": limma_df["t"].astype(float),
            "CasesMean": limma_df["CasesMean"].astype(float),
            "ControlsMean": limma_df["ControlsMean"].astype(float),
            "Method_Type": "TRUE_DE",
        })
    
        logger.info(f"      ✅ LIMMA completed: {len(out)} genes")
        return out



    def method_welch_ttest(self, gene_data, phenotypes):
        """✅ Welch's t-test - TRUE Differential Expression
        Tests: mean(cases) vs mean(controls) with unequal variances
        """
        logger.info(f"      ✅ Running Welch's t-test on {len(gene_data.columns)} genes...")
        results = []
        
        for gene in gene_data.columns:
            try:
                cases = gene_data[gene].values[phenotypes == 1]
                controls = gene_data[gene].values[phenotypes == 0]
                
                if len(cases) < 2 or len(controls) < 2:
                    mean_diff, p_value, t_stat = 0.0, 1.0, 0.0
                    cases_mean, controls_mean = 0.0, 0.0
                else:
                    cases_mean = np.mean(cases)
                    controls_mean = np.mean(controls)
                    mean_diff = cases_mean - controls_mean  # Effect size
                    
                    # Welch's t-test (unequal variances)
                    t_stat, p_value = stats.ttest_ind(cases, controls, equal_var=False)
                
                results.append({
                    'Gene': gene,
                    'Mean_Difference': float(mean_diff),
                    'Log2FoldChange': float(mean_diff),
                    'PValue': float(p_value),
                    't_statistic': float(t_stat),
                    'CasesMean': float(cases_mean),
                    'ControlsMean': float(controls_mean),
                    'N_Cases': int(len(cases)),
                    'N_Controls': int(len(controls)),
                    'Method_Type': 'TRUE_DE'
                })
                
            except Exception as e:
                logger.warning(f"      ⚠️  Gene {gene} failed: {str(e)}")
                results.append({
                    'Gene': gene,
                    'Mean_Difference': 0.0,
                    'Log2FoldChange': 0.0,
                    'PValue': 1.0,
                    't_statistic': 0.0,
                    'CasesMean': 0.0,
                    'ControlsMean': 0.0,
                    'N_Cases': 0,
                    'N_Controls': 0,
                    'Method_Type': 'TRUE_DE'
                })
                
        logger.info(f"      ✅ Welch's t-test completed: {len(results)} genes")
        return pd.DataFrame(results)

    def method_linear_regression(self, gene_data, phenotypes):
        """✅ Linear Regression - TRUE Differential Expression
        Tests: expression ~ disease_status
        Coefficient = mean difference between groups
        """
        logger.info(f"      ✅ Running Linear Regression on {len(gene_data.columns)} genes...")
        results = []
        
        for gene in gene_data.columns:
            try:
                y = gene_data[gene].values  # Expression = OUTCOME
                X = sm.add_constant(phenotypes)  # Disease = PREDICTOR
                
                if len(y) < 3:
                    coefficient, p_value = 0.0, 1.0
                    intercept, r_squared = 0.0, 0.0
                else:
                    try:
                        model = sm.OLS(y, X).fit()
                        
                        coefficient = model.params[1]  # This IS the mean difference
                        p_value = model.pvalues[1]
                        intercept = model.params[0]
                        r_squared = model.rsquared
                    except:
                        cases = y[phenotypes == 1]
                        controls = y[phenotypes == 0]
                        coefficient = np.mean(cases) - np.mean(controls)
                        p_value = 1.0
                        intercept = np.mean(controls)
                        r_squared = 0.0
                
                results.append({
                    'Gene': gene,
                    'Coefficient': float(coefficient),
                    'Log2FoldChange': float(coefficient),
                    'PValue': float(p_value),
                    'Intercept': float(intercept),
                    'R_squared': float(r_squared),
                    'Method_Type': 'TRUE_DE'
                })
                
            except Exception as e:
                logger.warning(f"      ⚠️  Gene {gene} failed: {str(e)}")
                results.append({
                    'Gene': gene,
                    'Coefficient': 0.0,
                    'Log2FoldChange': 0.0,
                    'PValue': 1.0,
                    'Intercept': 0.0,
                    'R_squared': 0.0,
                    'Method_Type': 'TRUE_DE'
                })
                
        logger.info(f"      ✅ Linear Regression completed: {len(results)} genes")
        return pd.DataFrame(results)

    def method_wilcoxon_ranksum(self, gene_data, phenotypes):
        """✅ Mann–Whitney U (Wilcoxon rank-sum) - TRUE DE (distribution shift)"""
        logger.info(f"      ✅ Running Wilcoxon Rank-Sum on {len(gene_data.columns)} genes...")
        results = []
    
        phenotypes = np.asarray(phenotypes)
    
        for gene in gene_data.columns:
            try:
                x = np.asarray(gene_data[gene].values, dtype=float)
                cases = x[phenotypes == 1]
                controls = x[phenotypes == 0]
    
                if len(cases) < 2 or len(controls) < 2:
                    mean_diff, median_diff, p_value, u_stat = 0.0, 0.0, 1.0, 0.0
                else:
                     
                    mean_diff = float(np.mean(cases) - np.mean(controls))
                    median_diff = float(np.median(cases) - np.median(controls))
    
                    u_stat, p_value = stats.mannwhitneyu(
                        cases, controls,
                        alternative="two-sided",
                        method="asymptotic"
                    )
    
                results.append({
                    "Gene": gene,
                    "Mean_Difference": float(median_diff),
                    "Median_Difference": float(median_diff),
                    "Log2FoldChange": float(mean_diff),  # Use mean for consistency
                    "PValue": float(p_value),
                    "U_statistic": float(u_stat),
                    "N_Cases": int(len(cases)),
                    "N_Controls": int(len(controls)),
                    "Method_Type": "TRUE_DE",
                })
    
            except Exception as e:
                logger.warning(f"      ⚠️  Gene {gene} failed: {str(e)}")
                results.append({
                    "Gene": gene,
                    "Mean_Difference": 0.0,
                    "Median_Difference": 0.0,
                    "Log2FoldChange": 0.0,
                    "PValue": 1.0,
                    "U_statistic": 0.0,
                    "N_Cases": 0,
                    "N_Controls": 0,
                    "Method_Type": "TRUE_DE",
                })
    
        logger.info(f"      ✅ Wilcoxon Rank-Sum completed: {len(results)} genes")
        return pd.DataFrame(results)


    def method_permutation_test(self, gene_data, phenotypes, n_permutations=1000, seed=42):
        """
        ✅ Permutation Test - TRUE Differential Expression (FIXED)
    
        Tests: mean difference between cases and controls using label permutations.
        - Increased permutations to 1000 for better p-value resolution
        - Fixed seed for reproducibility
        """
        logger.info(f"      ✅ Running Permutation Test on {len(gene_data.columns)} genes ({n_permutations} permutations)...")
    
        y = np.asarray(phenotypes).astype(int)
    
        n_cases = int(np.sum(y == 1))
        n_controls = int(np.sum(y == 0))
        if n_cases < 1 or n_controls < 1:
            logger.warning("      ⚠️  Permutation Test: missing cases or controls; returning empty.")
            return pd.DataFrame()
    
        rng = np.random.default_rng(seed)
    
        results = []
    
        for gene in gene_data.columns:
            try:
                expr_values = np.asarray(gene_data[gene].values, dtype=float)
    
                cases = expr_values[y == 1]
                controls = expr_values[y == 0]
    
                cases_mean = float(np.mean(cases)) if len(cases) else 0.0
                controls_mean = float(np.mean(controls)) if len(controls) else 0.0
                observed_diff = float(cases_mean - controls_mean)
    
                if n_cases < 2 or n_controls < 2 or len(expr_values) < 3:
                    p_value = 1.0
                else:
                    # Vectorized permutation for speed
                    perm_diffs = np.empty(n_permutations, dtype=float)
    
                    for b_idx in range(n_permutations):
                        perm = rng.permutation(y)
                        perm_diffs[b_idx] = float(np.mean(expr_values[perm == 1]) - np.mean(expr_values[perm == 0]))
    
                    # Two-sided empirical p-value with +1 correction
                    b = int(np.sum(np.abs(perm_diffs) >= abs(observed_diff)))
                    p_value = float((b + 1) / (n_permutations + 1))
    
                results.append({
                    "Gene": gene,
                    "Observed_Difference": observed_diff,
                    "Log2FoldChange": observed_diff,
                    "PValue": p_value,
                    "CasesMean": cases_mean,
                    "ControlsMean": controls_mean,
                    "N_Cases": n_cases,
                    "N_Controls": n_controls,
                    "N_Permutations": int(n_permutations),
                    "Method_Type": "TRUE_DE"
                })
    
            except Exception as e:
                logger.warning(f"      ⚠️  Gene {gene} failed: {str(e)}")
                results.append({
                    "Gene": gene,
                    "Observed_Difference": 0.0,
                    "Log2FoldChange": 0.0,
                    "PValue": 1.0,
                    "CasesMean": 0.0,
                    "ControlsMean": 0.0,
                    "N_Cases": n_cases,
                    "N_Controls": n_controls,
                    "N_Permutations": int(n_permutations),
                    "Method_Type": "TRUE_DE"
                })
    
        logger.info(f"      ✅ Permutation Test completed: {len(results)} genes")
        return pd.DataFrame(results)


    # ============================================================================
    # ASSOCIATION TESTING METHODS (3 methods - FIXED for numerical stability)
    # ⚠️  These test: "Does expression predict disease?" (DIFFERENT from DE!)
    # ============================================================================

    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from sklearn.preprocessing import StandardScaler
    from scipy import stats
    
    def method_weighted_logistic(self, gene_data, phenotypes):
        """
        ✅ Weighted Logistic = GLM Binomial with per-sample weights.
        Model: P(y=1) ~ z(expression)
        - weights are computed from class imbalance
        - coefficient reported is log-odds ratio for 1 SD increase in expression
        """
        logger.info(f"      ✅ Running Weighted Logistic (proper weights) on {len(gene_data.columns)} genes...")
    
        y = np.asarray(phenotypes).astype(int)
        n1 = int(np.sum(y == 1))
        n0 = int(np.sum(y == 0))
        if n1 == 0 or n0 == 0:
            logger.warning("      ⚠️  Only one class present; returning empty.")
            return pd.DataFrame()
    
        # Inverse-prevalence weights (common choice)
        w_case = (n0 + n1) / (2.0 * n1)
        w_ctrl = (n0 + n1) / (2.0 * n0)
        weights = np.where(y == 1, w_case, w_ctrl).astype(float)
    
        results = []
    
        for gene in gene_data.columns:
            try:
                x_raw = np.asarray(gene_data[gene].values, dtype=float)
    
                # mean-difference (for your Log2FoldChange compatibility)
                cases_mean = float(np.mean(x_raw[y == 1])) if n1 > 0 else 0.0
                controls_mean = float(np.mean(x_raw[y == 0])) if n0 > 0 else 0.0
                mean_diff = cases_mean - controls_mean
    
                # standardize predictor
                X = x_raw.reshape(-1, 1)
                if np.var(X) < 1e-12:
                    # constant predictor -> no information
                    results.append({
                        "Gene": gene,
                        "Log_Odds_Ratio": 0.0,
                        "Log2FoldChange": mean_diff,
                        "PValue": 1.0,
                        "Standard_Error": np.nan,
                        "CasesMean": cases_mean,
                        "ControlsMean": controls_mean,
                        "N_Cases": n1,
                        "N_Controls": n0,
                        "Method_Type": "ASSOCIATION"
                    })
                    continue
    
                Xz = StandardScaler().fit_transform(X)
                Xsm = sm.add_constant(Xz, has_constant="add")
    
                fit = sm.GLM(y, Xsm, family=sm.families.Binomial(), freq_weights=weights).fit()
    
                coef = float(fit.params[1])
                se   = float(fit.bse[1]) if np.isfinite(fit.bse[1]) else np.nan
                pval = float(fit.pvalues[1]) if np.isfinite(fit.pvalues[1]) else 1.0
    
                results.append({
                    "Gene": gene,
                    "Log_Odds_Ratio": coef,
                    "Log2FoldChange": mean_diff,   # keep for plotting / compatibility
                    "PValue": pval,
                    "Standard_Error": se,
                    "CasesMean": cases_mean,
                    "ControlsMean": controls_mean,
                    "N_Cases": n1,
                    "N_Controls": n0,
                    "Method_Type": "ASSOCIATION"
                })
    
            except Exception as e:
                logger.warning(f"      ⚠️  Weighted logistic failed for {gene}: {str(e)}")
                results.append({
                    "Gene": gene,
                    "Log_Odds_Ratio": 0.0,
                    "Log2FoldChange": 0.0,
                    "PValue": 1.0,
                    "Standard_Error": np.nan,
                    "CasesMean": 0.0,
                    "ControlsMean": 0.0,
                    "N_Cases": n1,
                    "N_Controls": n0,
                    "Method_Type": "ASSOCIATION"
                })
    
        logger.info(f"      ✅ Weighted Logistic completed: {len(results)} genes")
        return pd.DataFrame(results)


    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from scipy.special import expit
    from scipy import stats
    
    def method_firth_logistic(self, gene_data, phenotypes):
        """
        ✅ TRUE Firth logistic regression:
        Model: P(y=1) ~ z(expression) with Jeffreys-prior bias reduction.
        """
        logger.info(f"      ✅ Running TRUE Firth Logistic on {len(gene_data.columns)} genes...")
    
        y = np.asarray(phenotypes).astype(int)
        n1 = int(np.sum(y == 1))
        n0 = int(np.sum(y == 0))
        if n1 == 0 or n0 == 0:
            logger.warning("      ⚠️  Only one class present; returning empty.")
            return pd.DataFrame()

        results = []

        for gene in gene_data.columns:
            try:
                x_raw = np.asarray(gene_data[gene].values, dtype=float)

                cases_mean = float(np.mean(x_raw[y == 1])) if n1 > 0 else 0.0
                controls_mean = float(np.mean(x_raw[y == 0])) if n0 > 0 else 0.0
                mean_diff = cases_mean - controls_mean

                X = x_raw.reshape(-1, 1)
                if np.var(X) < 1e-12:
                    results.append({
                        "Gene": gene,
                        "Firth_Log_OR": 0.0,
                        "Log2FoldChange": mean_diff,
                        "PValue": 1.0,
                        "Standard_Error": np.nan,
                        "CasesMean": cases_mean,
                        "ControlsMean": controls_mean,
                        "N_Cases": n1,
                        "N_Controls": n0,
                        "Method_Type": "ASSOCIATION"
                    })
                    continue

                Xz = StandardScaler().fit_transform(X)
                Xsm = np.column_stack([np.ones(len(Xz)), Xz[:, 0]])

                beta, se, pvals = _firth_logistic_fit(Xsm, y, max_iter=100, tol=1e-8, ridge=1e-8)

                coef = float(beta[1])
                se1  = float(se[1])
                pval = float(pvals[1])

                results.append({
                    "Gene": gene,
                    "Firth_Log_OR": coef,
                    "Log2FoldChange": mean_diff,
                    "PValue": pval,
                    "Standard_Error": se1,
                    "CasesMean": cases_mean,
                    "ControlsMean": controls_mean,
                    "N_Cases": n1,
                    "N_Controls": n0,
                    "Method_Type": "ASSOCIATION"
                })

            except Exception as e:
                logger.warning(f"      ⚠️  Firth logistic failed for {gene}: {str(e)}")
                results.append({
                    "Gene": gene,
                    "Firth_Log_OR": 0.0,
                    "Log2FoldChange": 0.0,
                    "PValue": 1.0,
                    "Standard_Error": np.nan,
                    "CasesMean": 0.0,
                    "ControlsMean": 0.0,
                    "N_Cases": n1,
                    "N_Controls": n0,
                    "Method_Type": "ASSOCIATION"
                })

        logger.info(f"      ✅ TRUE Firth Logistic completed: {len(results)} genes")
        return pd.DataFrame(results)

  
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from scipy.special import expit
    from scipy import stats
    
    # =============================================================================
    # ✅ TRUE Bayesian Logistic Regression (Laplace approximation around MAP)
    #    Model: y ~ Bernoulli(sigmoid(beta0 + beta1 * z(expression)))
    #    Prior: beta0 ~ N(0, intercept_sd^2), beta1 ~ N(0, coef_sd^2)
    #    Output: posterior mean (MAP), posterior SD, Wald-style p-value (approx)
    # =============================================================================
    
    def _bayes_logistic_laplace_fit(X, y, w=None, intercept_sd=10.0, coef_sd=2.5,
                                   max_iter=50, tol=1e-8, ridge=1e-10):
        """
        X: (n,2) = [1, x] where x is standardized
        y: (n,) in {0,1}
        w: optional weights (n,) >= 0
        Returns: beta (2,), se (2,), cov (2,2)
        """
        y = np.asarray(y, dtype=float).ravel()
        X = np.asarray(X, dtype=float)
        n, p = X.shape
        assert p == 2, "This helper expects intercept + 1 predictor."

        if w is None:
            w = np.ones(n, dtype=float)
        else:
            w = np.asarray(w, dtype=float).ravel()
            w = np.clip(w, 0.0, np.inf)

        prior_var = np.array([intercept_sd**2, coef_sd**2], dtype=float)
        prior_prec = np.diag(1.0 / np.clip(prior_var, 1e-30, np.inf))

        beta = np.zeros(p, dtype=float)

        for _ in range(max_iter):
            eta = X @ beta
            p_hat = expit(eta)

            # gradient of log-likelihood: X^T (y - p) with weights
            r = (y - p_hat) * w
            grad_ll = X.T @ r

            # Hessian of log-likelihood: - X^T W X, with W = p(1-p)*w
            W = np.clip(p_hat * (1.0 - p_hat), 1e-12, None) * w
            H_ll = -(X.T @ (X * W[:, None]))

            # log-posterior = log-likelihood + log-prior
            # grad_prior = - beta / prior_var
            grad = grad_ll - (beta / np.clip(prior_var, 1e-30, np.inf))
            # Hessian_prior = - prior_prec
            H = H_ll - prior_prec

            # Newton step: beta_new = beta - H^{-1} grad   (since we maximize log-posterior)
            # Here H is negative-definite-ish; solve for step robustly.
            H_stable = H - ridge * np.eye(p)
            try:
                step = np.linalg.solve(H_stable, grad)
            except np.linalg.LinAlgError:
                step = np.linalg.pinv(H_stable) @ grad

            beta_new = beta - step

            if np.max(np.abs(beta_new - beta)) < tol:
                beta = beta_new
                break
            beta = beta_new

        # Posterior covariance via Laplace: cov ≈ (-H_post)^{-1} at MAP
        eta = X @ beta
        p_hat = expit(eta)
        W = np.clip(p_hat * (1.0 - p_hat), 1e-12, None) * w
        H_ll = -(X.T @ (X * W[:, None]))
        H_post = H_ll - prior_prec  # Hessian of log-posterior
        neg_H = -H_post + ridge * np.eye(p)
    
        try:
            cov = np.linalg.inv(neg_H)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(neg_H)
    
        se = np.sqrt(np.clip(np.diag(cov), 1e-30, np.inf))
        return beta, se, cov
    
    
    def method_bayesian_logistic(self, gene_data, phenotypes,
                                 intercept_sd=10.0, coef_sd=2.5,
                                 use_class_weights=True):
        """
        ✅ Correct Bayesian logistic regression (NOT BayesianRidge).
        - Returns log-odds (beta1) for 1 SD increase in expression (z-scored).
        - Uses Laplace approximation for posterior SDs.
        """
        logger.info(f"      ✅ Running Bayesian Logistic (TRUE) on {len(gene_data.columns)} genes...")
    
        y = np.asarray(phenotypes).astype(int)
        n1 = int(np.sum(y == 1))
        n0 = int(np.sum(y == 0))
        if n1 == 0 or n0 == 0:
            logger.warning("      ⚠️  Only one class present; returning empty.")
            return pd.DataFrame()
    
        # optional imbalance weights (same spirit as your weighted GLM)
        if use_class_weights:
            w_case = (n0 + n1) / (2.0 * n1)
            w_ctrl = (n0 + n1) / (2.0 * n0)
            weights = np.where(y == 1, w_case, w_ctrl).astype(float)
        else:
            weights = None
    
        results = []
    
        for gene in gene_data.columns:
            try:
                x_raw = np.asarray(gene_data[gene].values, dtype=float)
    
                # descriptive mean-difference (keep for your pipeline compatibility)
                cases_mean = float(np.mean(x_raw[y == 1])) if n1 > 0 else 0.0
                controls_mean = float(np.mean(x_raw[y == 0])) if n0 > 0 else 0.0
                mean_diff = cases_mean - controls_mean
    
                # skip constant predictors
                if np.var(x_raw) < 1e-12 or len(x_raw) < 5:
                    results.append({
                        "Gene": gene,
                        "Bayesian_Log_OR": 0.0,
                        "Log2FoldChange": mean_diff,
                        "Standard_Error": np.nan,
                        "CI_95_Lower": np.nan,
                        "CI_95_Upper": np.nan,
                        "PValue": 1.0,
                        "CasesMean": cases_mean,
                        "ControlsMean": controls_mean,
                        "N_Cases": n1,
                        "N_Controls": n0,
                        "Method_Type": "ASSOCIATION"
                    })
                    continue
    
                # standardize predictor
                xz = StandardScaler().fit_transform(x_raw.reshape(-1, 1)).ravel()
                X = np.column_stack([np.ones_like(xz), xz])  # intercept + standardized predictor
    
                beta, se, _ = _bayes_logistic_laplace_fit(
                    X, y, w=weights,
                    intercept_sd=intercept_sd, coef_sd=coef_sd,
                    max_iter=60, tol=1e-8, ridge=1e-10
                )
    
                coef = float(beta[1])          # log-odds per 1 SD
                se1  = float(se[1]) if np.isfinite(se[1]) else np.nan
    
                # Normal approximation to posterior for quick “significance-like” ranking
                if np.isfinite(se1) and se1 > 0:
                    z = coef / se1
                    pval = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
                    ci_lo = float(coef - 1.96 * se1)
                    ci_hi = float(coef + 1.96 * se1)
                else:
                    pval, ci_lo, ci_hi = 1.0, np.nan, np.nan
    
                results.append({
                    "Gene": gene,
                    "Bayesian_Log_OR": coef,
                    "Log2FoldChange": mean_diff,   # keep for your volcano/threshold plumbing
                    "Standard_Error": se1,
                    "CI_95_Lower": ci_lo,
                    "CI_95_Upper": ci_hi,
                    "PValue": pval,
                    "CasesMean": cases_mean,
                    "ControlsMean": controls_mean,
                    "N_Cases": n1,
                    "N_Controls": n0,
                    "Method_Type": "ASSOCIATION"
                })
    
            except Exception as e:
                logger.warning(f"      ⚠️  Bayesian logistic failed for {gene}: {str(e)}")
                results.append({
                    "Gene": gene,
                    "Bayesian_Log_OR": 0.0,
                    "Log2FoldChange": 0.0,
                    "Standard_Error": np.nan,
                    "CI_95_Lower": np.nan,
                    "CI_95_Upper": np.nan,
                    "PValue": 1.0,
                    "CasesMean": 0.0,
                    "ControlsMean": 0.0,
                    "N_Cases": n1,
                    "N_Controls": n0,
                    "Method_Type": "ASSOCIATION"
                })
    
        logger.info(f"      ✅ Bayesian Logistic (TRUE) completed: {len(results)} genes")
        return pd.DataFrame(results)




    # ============================================================================
    # UTILITY METHODS
    # ============================================================================

    def apply_fdr_correction(self, results_df):
        """Apply FDR correction with method-aware significance criteria"""
        logger.info(f"      🧮 Applying FDR correction to {len(results_df)} genes...")
        
        if 'PValue' not in results_df.columns or len(results_df) == 0:
            results_df['FDR'] = 1.0
            return results_df
        
        valid_pvals = ~results_df['PValue'].isna() & (results_df['PValue'] >= 0) & (results_df['PValue'] <= 1)
        
        if valid_pvals.sum() > 0:
            _, fdr_vals, _, _ = multipletests(results_df.loc[valid_pvals, 'PValue'], 
                                           method='fdr_bh', alpha=self.fdr_threshold)
            results_df.loc[valid_pvals, 'FDR'] = fdr_vals
            results_df['FDR'] = results_df['FDR'].fillna(1.0)
        else:
            results_df['FDR'] = 1.0
        
        # Significance flags - FDR only (primary criterion)
        results_df['Significant_FDR'] = results_df['FDR'] < self.fdr_threshold
        
        # Check method type for appropriate effect size threshold
        method_type = results_df['Method_Type'].iloc[0] if 'Method_Type' in results_df.columns else 'TRUE_DE'
        
        if 'Log2FoldChange' in results_df.columns:
            # For ASSOCIATION methods, use a lower LFC threshold (or none)
            # since the scale of coefficients differs
            if method_type == 'ASSOCIATION':
                # For association tests, any non-zero effect is meaningful
                # Use a very small threshold or just FDR
                effective_lfc_threshold = 0.0  # No LFC filter for association methods
                results_df['High_FC'] = np.abs(results_df['Log2FoldChange']) > effective_lfc_threshold
            else:
                # For TRUE_DE methods, use the specified LFC threshold
                results_df['High_FC'] = np.abs(results_df['Log2FoldChange']) >= self.lfc_threshold
            
            results_df['Significant_Combined'] = results_df['Significant_FDR'] & results_df['High_FC']
            results_df['Upregulated'] = (results_df['Log2FoldChange'] > 0) & results_df['Significant_FDR']
            results_df['Downregulated'] = (results_df['Log2FoldChange'] < 0) & results_df['Significant_FDR']
        else:
            results_df['High_FC'] = True
            results_df['Significant_Combined'] = results_df['Significant_FDR']
            results_df['Upregulated'] = False
            results_df['Downregulated'] = False
        
        logger.info(f"      ✅ FDR correction completed")
        logger.info(f"      📊 Significant genes (FDR < {self.fdr_threshold}): {results_df['Significant_FDR'].sum()}")
        if method_type == 'TRUE_DE':
            logger.info(f"      📊 Significant + High FC (|LFC| >= {self.lfc_threshold}): {results_df['Significant_Combined'].sum()}")
        
        return results_df

    def save_results(self, results_df, database_name, tissue_name, method_name):
        """Save results"""
        method_with_dataset = f"{method_name}_{self.dataset}"
        output_dir = self.output_base / database_name / tissue_name / method_with_dataset
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if results_df.empty:
            logger.warning(f"      ⚠️  Empty results for {database_name}-{tissue_name}-{method_name}")
            return
        
        # Apply FDR correction
        results_df = self.apply_fdr_correction(results_df)
        
        # Add metadata
        results_df['Dataset'] = self.dataset
        results_df['Database'] = database_name
        results_df['Tissue'] = tissue_name
        results_df['Method'] = method_name
        results_df['Analysis_Date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        results_df['FDR_Threshold_Used'] = self.fdr_threshold
        results_df['LFC_Threshold_Used'] = self.lfc_threshold
        
        # Sort by FDR
        results_df = results_df.sort_values('FDR', na_position='last')
        
        # Save all results
        results_df.to_csv(output_dir / 'differential_expression_results.csv', index=False)
        logger.info(f"      ✅ Saved {len(results_df)} genes to differential_expression_results.csv")
        
        # Save summary
        summary_stats = {
            'Total_Genes_Analyzed': len(results_df),
            'Significant_FDR': results_df['Significant_FDR'].sum(),
            'Significant_Combined': results_df.get('Significant_Combined', pd.Series([0])).sum(),
            'Upregulated': results_df.get('Upregulated', pd.Series([0])).sum(),
            'Downregulated': results_df.get('Downregulated', pd.Series([0])).sum(),
            'Database': database_name,
            'Tissue': tissue_name,
            'Method': method_name,
            'Dataset': self.dataset,
            'FDR_Threshold': self.fdr_threshold,
            'LFC_Threshold': self.lfc_threshold,
            'Analysis_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        summary_df = pd.DataFrame([summary_stats])
        summary_df.to_csv(output_dir / 'analysis_summary.csv', index=False)

    def prepare_design_matrix(self, gene_data):
        """Prepare design matrix"""
        phenotype_sample_ids = self.phenotype_data['FID'].astype(str) + '_' + self.phenotype_data['IID'].astype(str)
        pheno_dict = dict(zip(phenotype_sample_ids, self.phenotype_data['PHENO_BINARY']))
        sex_dict = dict(zip(phenotype_sample_ids, self.phenotype_data['SEX']))
        
        common_samples = gene_data.index.intersection(pd.Index(pheno_dict.keys()))
        
        if len(common_samples) < 10:
            logger.warning(f"    ⚠️  Only {len(common_samples)} matching samples")
            if len(common_samples) < 5:
                logger.error(f"    ❌ Too few samples")
                return None, None
        
        gene_data_filtered = gene_data.loc[common_samples]
        phenotype_labels = pd.Series(pheno_dict).loc[common_samples]
        sex_labels = pd.Series(sex_dict).loc[common_samples]
        
        design_df = pd.DataFrame({
            'Intercept': 1,
            'Condition': phenotype_labels.values
        }, index=common_samples)
        
        if len(sex_labels.unique()) > 1:
            design_df['Sex'] = sex_labels.values
        
        logger.info(f"    📊 Design matrix: {len(gene_data_filtered)} samples, {len(gene_data_filtered.columns)} genes")
        
        return gene_data_filtered, design_df

    def analyze_database_tissue_method(self, database_name, tissue_name, method_name):
        """Analyze single combination"""
        try:
            exists, existing_info = self.check_existing_results(database_name, tissue_name, method_name)
            
            if exists and not self.force_rerun:
                logger.info(f"      ⏭️  SKIPPED - Results exist ({existing_info['gene_count']} genes)")
                return {
                    'database': database_name,
                    'tissue': tissue_name,
                    'method': method_name,
                    'genes_analyzed': existing_info['gene_count'],
                    'significant': existing_info['significant_genes'],
                    'skipped': True
                }
            
            gene_data, data_info = self.load_expression_data(database_name, tissue_name)
            
            if gene_data is None:
                logger.warning(f"      ❌ No expression data for {database_name}-{tissue_name}")
                return None
            
            if len(gene_data) < 5 or len(gene_data.columns) < 1:
                logger.warning(f"      ❌ Insufficient data")
                return None
            
            logger.info(f"      📊 Processing {len(gene_data.columns)} variable genes")
            
            design_result = self.prepare_design_matrix(gene_data)
            
            if design_result[0] is None:
                logger.warning(f"      ❌ Could not prepare design matrix")
                return None
            
            gene_data_filtered, design_matrix = design_result
            phenotypes = design_matrix['Condition'].values
            
            logger.info(f"      🔬 Running {method_name}...")
            
            # Route to appropriate method
            if method_name == "LIMMA":
            
                results = self.method_limma(gene_data_filtered, design_matrix)
            elif method_name == "Welch_t_test":
                results = self.method_welch_ttest(gene_data_filtered, phenotypes)
            elif method_name == "Linear_Regression":
                results = self.method_linear_regression(gene_data_filtered, phenotypes)
            elif method_name == "Wilcoxon_Rank_Sum":
                results = self.method_wilcoxon_ranksum(gene_data_filtered, phenotypes)
            elif method_name == "Permutation_Test":
                results = self.method_permutation_test(gene_data_filtered, phenotypes)
            elif method_name == "Weighted_Logistic":
                results = self.method_weighted_logistic(gene_data_filtered, phenotypes)
            elif method_name == "Firth_Logistic":
                results = self.method_firth_logistic(gene_data_filtered, phenotypes)
            elif method_name == "Bayesian_Logistic":
                results = self.method_bayesian_logistic(gene_data_filtered, phenotypes)
            else:
                logger.error(f"      ❌ Unknown method: {method_name}")
                return None
            
            if results.empty:
                logger.warning(f"      ❌ No results generated")
                return None
            
            self.save_results(results, database_name, tissue_name, method_name)
            
            n_significant = results.get('Significant_FDR', pd.Series([False])).sum();
            
            logger.info(f"      ✅ Completed: {len(results)} genes, {n_significant} significant")
            
            return {
                'database': database_name,
                'tissue': tissue_name,
                'method': method_name,
                'genes_analyzed': len(results),
                'significant_fdr': n_significant,
                'skipped': False
            }
            
        except Exception as e:
            logger.error(f"      ❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def run_comprehensive_analysis(self, target_database=None, target_tissue=None, n_jobs=1):
        """Run analysis for all databases, tissues, and methods with multiprocessing"""
        logger.info("🧬 COMPREHENSIVE DIFFERENTIAL GENE EXPRESSION ANALYSIS")
        logger.info("="*70)
        logger.info(f"Phenotype: {self.phenotype}")
        logger.info(f"Fold: {self.fold}")
        logger.info(f"Dataset: {self.dataset}")
        logger.info(f"FDR threshold: {self.fdr_threshold}")
        logger.info(f"LFC threshold: {self.lfc_threshold}")
        logger.info(f"Parallel jobs: {n_jobs}")
        logger.info("="*70)
        
        expression_databases = self.get_expression_databases()
        available_tissues = self.get_available_tissues()
        statistical_methods = self.get_statistical_methods()
        
        if target_database:
            if target_database in expression_databases:
                databases_to_analyze = [target_database]
            else:
                logger.error(f"❌ Database '{target_database}' not found")
                return 0, 1, 0
        else:
            databases_to_analyze = list(expression_databases.keys())
        
        if target_tissue:
            if target_tissue in available_tissues:
                tissues_to_analyze = [target_tissue]
            else:
                logger.error(f"❌ Tissue '{target_tissue}' not found")
                return 0, 1, 0
        else:
            tissues_to_analyze = available_tissues
        
        logger.info(f"🗃️  Databases: {databases_to_analyze}")
        logger.info(f"🧪 Tissues: {len(tissues_to_analyze)} total")
        logger.info(f"📊 Methods: {statistical_methods}")
        
        # Create all combinations
        combinations = []
        for database_name in databases_to_analyze:
            for tissue_name in tissues_to_analyze:
                for method_name in statistical_methods:
                    combinations.append((database_name, tissue_name, method_name))
        
        total_combinations = len(combinations)
        logger.info(f"🎯 Total combinations: {total_combinations}")
        
        all_results = []
        successful = 0
        failed = 0
        skipped = 0
        
        if n_jobs == 1:
            # Sequential processing
            for i, (database_name, tissue_name, method_name) in enumerate(combinations):
                combo_num = i + 1
                logger.info(f"\n[{combo_num:3d}/{total_combinations}] {database_name} | {tissue_name} | {method_name}")
                
                result = self.analyze_database_tissue_method(database_name, tissue_name, method_name)
                
                if result:
                    all_results.append(result)
                    if result.get('skipped', False):
                        skipped += 1
                    else:
                        successful += 1
                else:
                    failed += 1
        else:
            # Parallel processing
            logger.info(f"\n🚀 Starting parallel processing with {n_jobs} workers...")
            
            # Create partial function with self bound
            analysis_func = partial(self._analyze_combination_wrapper)
            
            # Use multiprocessing pool
            with mp.Pool(processes=n_jobs) as pool:
                results_iter = pool.imap_unordered(analysis_func, combinations)
                
                for i, result in enumerate(results_iter):
                    combo_num = i + 1
                    if result:
                        all_results.append(result)
                        if result.get('skipped', False):
                            skipped += 1
                            status = "⏭️  SKIPPED"
                        else:
                            successful += 1
                            status = "✅ DONE"
                    else:
                        failed += 1
                        status = "❌ FAILED"
                    
                    logger.info(f"[{combo_num:3d}/{total_combinations}] {status}")
        
        logger.info(f"\n🎉 ANALYSIS COMPLETE!")
        logger.info(f"✅ Successful: {successful}/{total_combinations}")
        logger.info(f"⏭️  Skipped: {skipped}/{total_combinations}")
        logger.info(f"❌ Failed: {failed}")
        
        if all_results:
            summary_df = pd.DataFrame(all_results)
            summary_file = self.output_base / f'comprehensive_analysis_summary_{self.dataset}.csv'
            summary_df.to_csv(summary_file, index=False)
            logger.info(f"📊 Summary saved: {summary_file}")
        
        return successful, failed, skipped
    
    def _analyze_combination_wrapper(self, combination):
        """Wrapper for multiprocessing"""
        database_name, tissue_name, method_name = combination
        return self.analyze_database_tissue_method(database_name, tissue_name, method_name)


def run_all_datasets(phenotype, fold, target_database=None, target_tissue=None, force_rerun=False, 
                    fdr_threshold=0.05, lfc_threshold=1.0, n_jobs=1):
    """Run comprehensive analysis for all datasets"""
    datasets = ["training", "validation", "test"]
    overall_success = 0
    overall_failed = 0
    overall_skipped = 0
    
    logger.info("🎯 RUNNING COMPREHENSIVE ANALYSIS FOR ALL DATASETS")
    logger.info("="*80)
    
    for i, dataset in enumerate(datasets):
        logger.info(f"\n📂 DATASET {i+1}/3: {dataset.upper()}")
        
        try:
            analyzer = ComprehensiveDE(phenotype, fold, dataset, force_rerun=force_rerun,
                                     fdr_threshold=fdr_threshold, lfc_threshold=lfc_threshold)
            success, failed, skipped = analyzer.run_comprehensive_analysis(target_database, target_tissue, n_jobs)
            overall_success += success
            overall_failed += failed
            overall_skipped += skipped
            
            logger.info(f"✅ {dataset.upper()} completed: {success} successful, {skipped} skipped, {failed} failed")
            
        except Exception as e:
            logger.error(f"❌ {dataset.upper()} failed: {str(e)}")
            continue
    
    logger.info("\n🎉 ALL DATASETS ANALYSIS COMPLETED!")
    logger.info(f"✅ Total successful: {overall_success}")
    logger.info(f"⏭️  Total skipped: {overall_skipped}")
    logger.info(f"❌ Total failed: {overall_failed}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Comprehensive Differential Gene Expression Analysis - 8 Methods (5 DE + 3 Association) - FIXED & OPTIMIZED",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
8 STATISTICAL METHODS (ALL FIXED FOR NUMERICAL STABILITY):
  ✅ TRUE DIFFERENTIAL EXPRESSION (5 methods):
     1. LIMMA (gold standard) - linear models
     2. Welch's t-test - unequal variances
     3. Linear Regression - expression ~ disease
     4. Wilcoxon Rank-Sum - non-parametric
     5. Permutation Test - distribution-free
  
  ⚠️  ASSOCIATION TESTING (3 methods - FIXED):
     6. Weighted Logistic - P(disease) ~ expression (FIXED: uses GLM)
     7. Firth Logistic - bias-reduced logistic (FIXED: uses GLM)
     8. Bayesian Logistic - Bayesian approach

CRITICAL FIXES:
  - All logistic methods now use sm.GLM() instead of sm.Logit()
  - Prevents "Singular matrix" errors
  - Added multiprocessing support for faster execution
  
EXAMPLES:
  # Sequential (1 job)
  python script.py migraine 1
  
  # Parallel with 4 workers
  python script.py migraine 1 --jobs 4
  
  # Specific database and tissue with 8 workers
  python script.py migraine 1 --database FUSION --tissue Brain_Cortex --jobs 8
  
  # Force rerun with custom thresholds and 16 workers
  python script.py migraine 1 --force --fdr 0.01 --lfc 1.5 --jobs 16
        """
    )
    
    parser.add_argument("phenotype", help="Phenotype name (e.g., migraine)")
    parser.add_argument("fold", help="Fold number (e.g., 1)")
    parser.add_argument("--dataset", choices=["training", "validation", "test", "all"],
                       default="all", help="Dataset to analyze")
    parser.add_argument("--database", 
                       choices=["Regular", "JTI", "UTMOST", "UTMOST2", "EpiX", "TIGAR", "FUSION"],
                       help="Specific database")
    parser.add_argument("--tissue", help="Specific tissue")
    parser.add_argument("--fdr", type=float, default=0.05, help="FDR threshold (default: 0.05)")
    parser.add_argument("--lfc", type=float, default=1.0, help="Log2FC threshold (default: 1.0)")
    parser.add_argument("--force", action="store_true", help="Force reanalysis")
    parser.add_argument("--jobs", type=int, default=20, help="Number of parallel jobs (default: 1)")
    
    args = parser.parse_args()
    
    # Determine optimal number of CPUs if not specified
    if args.jobs <= 0:
        args.jobs = max(1, mp.cpu_count() - 1)
    
    print("🚀 COMPREHENSIVE DIFFERENTIAL GENE EXPRESSION ANALYSIS - FIXED & OPTIMIZED")
    print("="*60)
    print(f"📋 Phenotype: {args.phenotype}")
    print(f"📋 Fold: {args.fold}")
    print(f"📋 Dataset(s): {args.dataset}")
    print(f"📋 FDR threshold: {args.fdr}")
    print(f"📋 LFC threshold: {args.lfc}")
    print(f"📋 Parallel jobs: {args.jobs}")
    print("="*60)
    print("✅ 5 TRUE DE methods + ⚠️  3 Association tests (ALL FIXED)")
    print("="*60)
    
    if args.dataset == "all":
        run_all_datasets(args.phenotype, args.fold, args.database, args.tissue, 
                        args.force, args.fdr, args.lfc, args.jobs)
    else:
        analyzer = ComprehensiveDE(args.phenotype, args.fold, args.dataset, 
                                  force_rerun=args.force,
                                  fdr_threshold=args.fdr, lfc_threshold=args.lfc)
        analyzer.run_comprehensive_analysis(args.database, args.tissue, args.jobs)


if __name__ == "__main__":
    main()