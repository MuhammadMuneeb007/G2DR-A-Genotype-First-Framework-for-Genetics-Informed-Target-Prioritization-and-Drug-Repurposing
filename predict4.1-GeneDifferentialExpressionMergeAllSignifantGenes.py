#!/usr/bin/env python3
"""
🔥 OPTIMIZED Comprehensive Method-Database Differential Expression Analysis 🔥
Professional heatmaps for significant gene analysis only - FAST VERSION
USAGE: 
  python predict4.1-GeneDifferentialExpressionFindSignificantGenes.py phenotype_name
EXAMPLES:
  python predict4.1-GeneDifferentialExpressionFindSignificantGenes.py migraine
  python predict4.1-GeneDifferentialExpressionFindSignificantGenes.py diabetes
TO CHANGE DATABASES: Modify DATABASES_TO_ANALYZE list in the script
TO CHANGE THRESHOLDS: Modify FDR_THRESHOLD and EFFECT_THRESHOLD at top of script
Structure: {phenotype}/Fold_{fold}/GeneDifferentialExpressions/{Database}/{Tissue}/{Method}_{Dataset}/
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from collections import defaultdict
import argparse
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import lru_cache, partial
import multiprocessing as mp
from typing import Dict, List, Set, Tuple, Optional
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - MODIFY THESE SECTIONS TO CUSTOMIZE ANALYSIS
# =============================================================================

# GLOBAL THRESHOLDS - MODIFY THESE TO UPDATE ALL METHODS
FDR_THRESHOLD = 0.1           # False Discovery Rate threshold
DE_EFFECT_THRESHOLD = 0.5      # Effect size for DE methods
ASSOC_EFFECT_THRESHOLD = 0.5   # Effect size for association methods
EFFECT_THRESHOLD = 0.5

# DATABASES TO ANALYZE - MODIFY THIS LIST TO SELECT WHICH DATABASES TO RUN
DATABASES_TO_ANALYZE = ["Regular", "JTI", "UTMOST", "UTMOST2", "EpiX", "TIGAR", "FUSION"]
 
# ALL AVAILABLE DATABASES
ALL_DATABASES = ["Regular", "JTI", "UTMOST", "UTMOST2", "EpiX", "TIGAR", "FUSION"]

# PERFORMANCE SETTINGS
MAX_WORKERS = min(32, (os.cpu_count() or 1) * 2)  # Auto-detect optimal workers
CHUNK_SIZE = 100  # Process files in chunks
USE_PARALLEL = True  # Set to False to disable parallel processing for debugging

# =============================================================================
# SIGNIFICANCE THRESHOLDS - 8 METHODS INCLUDING NEW DE METHODS
# =============================================================================
def get_significance_thresholds():
    """Get significance thresholds using global FDR and Effect thresholds"""
    return {
        # ===== PROPER DIFFERENTIAL EXPRESSION METHODS =====
        "LIMMA": {
            "pvalue": 0.05,
            "fdr": FDR_THRESHOLD,
            "effect_size": DE_EFFECT_THRESHOLD,
            "effect_col": "Log2FoldChange",
            "description": "Linear models for microarray data (gold standard - TRUE DE)"
        },
        
        "Welch_t_test": {
            "pvalue": 0.05,
            "fdr": FDR_THRESHOLD,
            "effect_size": DE_EFFECT_THRESHOLD,
            "effect_col": "Log2FoldChange",
            "fallback_col": "Mean_Difference",
            "description": "Welch's t-test for unequal variances (TRUE DE)"
        },
        
        "Linear_Regression": {
            "pvalue": 0.05,
            "fdr": FDR_THRESHOLD,
            "effect_size": DE_EFFECT_THRESHOLD,
            "effect_col": "Log2FoldChange",
            "fallback_col": "Coefficient",
            "description": "Linear regression: expression ~ disease_status (TRUE DE)"
        },
        
        "Wilcoxon_Rank_Sum": {
            "pvalue": 0.05,
            "fdr": FDR_THRESHOLD,
            "effect_size": DE_EFFECT_THRESHOLD,
            "effect_col": "Log2FoldChange",
            "fallback_col": "Median_Difference",
            "description": "Non-parametric rank-sum test (TRUE DE)"
        },
        
        "Permutation_Test": {
            "pvalue": 0.05,
            "fdr": FDR_THRESHOLD,
            "effect_size": DE_EFFECT_THRESHOLD,
            "effect_col": "Log2FoldChange",
            "fallback_col": "Observed_Difference",
            "description": "Non-parametric permutation testing (TRUE DE)"
        },
        
        # ===== ASSOCIATION TESTING METHODS (KEPT FOR COMPARISON) =====
        "Weighted_Logistic": {
            "pvalue": 0.05,
            "fdr": FDR_THRESHOLD,
            "effect_size": ASSOC_EFFECT_THRESHOLD,
            "effect_col": "Log_Odds_Ratio",
            "fallback_col": "Log2FoldChange",
            "description": "⚠️ Association test - logistic regression with class weighting"
        },
        
        "Firth_Logistic": {
            "pvalue": 0.05,
            "fdr": FDR_THRESHOLD,
            "effect_size": ASSOC_EFFECT_THRESHOLD,
            "effect_col": "Firth_Log_OR",
            "fallback_col": "Log2FoldChange",
            "description": "⚠️ Association test - bias-reduced logistic regression"
        },
        
        "Bayesian_Logistic": {
            "pvalue": 0.05,
            "fdr": FDR_THRESHOLD,
            "effect_size": ASSOC_EFFECT_THRESHOLD,
            "effect_col": "Bayesian_Coefficient",
            "fallback_col": "Log2FoldChange",
            "description": "⚠️ Association test - Bayesian logistic with priors"
        }
    }

# =============================================================================
# OPTIMIZED FILE LOADING WITH PARALLEL PROCESSING
# =============================================================================

def load_single_file(file_info):
    """
    Load and filter a single file (for parallel processing)
    
    Parameters:
    -----------
    file_info : tuple
        (results_file, method_name, thresholds)
    
    Returns:
    --------
    tuple : (gene_set, gene_count)
    """
    results_file, method_name, thresholds = file_info
    
    try:
        # Fast CSV loading with specific columns only
        possible_cols = [
            'Gene', 'FDR', 'PValue', 'Log2FoldChange', 'Mean_Difference', 
            'Coefficient', 'Median_Difference', 'Observed_Difference',
            'Log_Odds_Ratio', 'Firth_Log_OR', 'Bayesian_Coefficient',
            'log2FoldChange', 'LogFC', 'log2FC', 'LFC',
            'fdr', 'adj.P.Val', 'padj', 'FDR_BH', 'q_value',
            'pvalue', 'P.Value', 'pval', 'p_value'
        ]
        
        # Read file and keep only relevant columns
        df = pd.read_csv(results_file, usecols=lambda x: x in possible_cols)
        
        if len(df) == 0:
            return set(), 0
        
        if method_name not in thresholds:
            return set(), 0
        
        method_thresholds = thresholds[method_name]
        filtered_df = df.copy()
        
        # PRIORITY 1: Apply FDR filter if available (already accounts for multiple testing)
        has_fdr = False
        if 'FDR' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['FDR'] <= method_thresholds['fdr']]
            has_fdr = True
        # FALLBACK: Use p-value only if FDR unavailable
        elif 'PValue' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['PValue'] <= method_thresholds['pvalue']]
        else:
            # No statistical significance column found
            return set(), 0
        
        if len(filtered_df) == 0:
            return set(), 0
        
        # PRIORITY 2: Apply effect size filter
        effect_col = method_thresholds['effect_col']
        fallback_col = method_thresholds.get('fallback_col', None)
        
        effect_size_applied = False
        
        # Try primary effect column first
        if effect_col in filtered_df.columns:
            valid_effects = filtered_df[effect_col].notna()
            filtered_df = filtered_df[valid_effects]
            
            if len(filtered_df) > 0:
                filtered_df = filtered_df[abs(filtered_df[effect_col]) >= method_thresholds['effect_size']]
                effect_size_applied = True
        
        # If primary column failed, try fallback
        if not effect_size_applied and fallback_col and fallback_col in df.columns:
            # Start fresh from original df
            filtered_df = df.copy()
            
            # Reapply statistical filter
            if has_fdr and 'FDR' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['FDR'] <= method_thresholds['fdr']]
            elif 'PValue' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['PValue'] <= method_thresholds['pvalue']]
            
            valid_effects = filtered_df[fallback_col].notna()
            filtered_df = filtered_df[valid_effects]
            
            if len(filtered_df) > 0:
                filtered_df = filtered_df[abs(filtered_df[fallback_col]) >= method_thresholds['effect_size']]
        
        # Extract gene names
        if 'Gene' in filtered_df.columns and len(filtered_df) > 0:
            gene_names = filtered_df['Gene'].astype(str).str.strip()
            gene_names = gene_names[~gene_names.isin(['', 'nan', 'NaN', 'None'])]
            return set(gene_names), len(gene_names)
        else:
            return set(), 0
        
    except Exception as e:
        # Silent failure for individual files
        return set(), 0

def load_volcano_file(file_info):
    """
    Load a single file for volcano plot (parallel processing)
    
    Parameters:
    -----------
    file_info : tuple
        (combo, thresholds)
    
    Returns:
    --------
    DataFrame or None
    """
    combo, thresholds = file_info
    
    try:
        df = pd.read_csv(combo['results_file'])
        
        if len(df) == 0:
            return None
        
        # Extract required columns
        available_cols = df.columns.tolist()
        
        # Find Log2FoldChange column
        log2fc_col = None
        for col in ['Log2FoldChange', 'log2FoldChange', 'LogFC', 'log2FC', 'LFC', 
                   'Mean_Difference', 'Coefficient', 'Median_Difference']:
            if col in available_cols:
                log2fc_col = col
                break
        
        # Find FDR column
        fdr_col = None
        for col in ['FDR', 'fdr', 'adj.P.Val', 'padj', 'FDR_BH', 'q_value']:
            if col in available_cols:
                fdr_col = col
                break
        
        # Find PValue column
        pval_col = None
        for col in ['PValue', 'pvalue', 'P.Value', 'pval', 'p_value']:
            if col in available_cols:
                pval_col = col
                break
        
        if not log2fc_col or not fdr_col:
            return None
        
        # Clean and prepare data
        plot_cols = ['Gene', log2fc_col, fdr_col]
        if pval_col:
            plot_cols.append(pval_col)
        
        plot_df = df[plot_cols].copy()
        plot_df = plot_df.dropna()
        
        if len(plot_df) == 0:
            return None
        
        # Remove extreme outliers
        plot_df = plot_df[abs(plot_df[log2fc_col]) <= 20]
        
        if len(plot_df) == 0:
            return None
        
        # Standardize column names
        plot_df = plot_df.rename(columns={
            log2fc_col: 'Log2FoldChange',
            fdr_col: 'FDR'
        })
        if pval_col:
            plot_df = plot_df.rename(columns={pval_col: 'PValue'})
        
        # Add metadata
        plot_df['Database'] = combo['database']
        plot_df['Tissue'] = combo['tissue']
        plot_df['Method'] = combo['method']
        plot_df['Fold'] = combo['fold']
        plot_df['Dataset'] = combo['dataset']
        plot_df['Unique_ID'] = plot_df['Gene'] + '_' + combo['tissue'] + '_' + combo['method'] + '_' + str(combo['fold']) + '_' + combo['dataset']
        
        # Calculate -log10(FDR)
        plot_df['FDR'] = plot_df['FDR'].clip(lower=1e-50, upper=1.0)
        plot_df['neg_log10_FDR'] = -np.log10(plot_df['FDR'])
        
        # Remove extreme -log10(FDR) values
        plot_df = plot_df[plot_df['neg_log10_FDR'] <= 100]
        
        if len(plot_df) == 0:
            return None
        
        # Determine significance using method-specific thresholds
        method_thresh = thresholds.get(combo['method'], {})
        fdr_thresh = method_thresh.get('fdr', FDR_THRESHOLD)
        effect_thresh = method_thresh.get('effect_size', EFFECT_THRESHOLD)
        
        plot_df['is_significant'] = (
            (plot_df['FDR'] <= fdr_thresh) & 
            (abs(plot_df['Log2FoldChange']) >= effect_thresh)
        )
        
        return plot_df
        
    except Exception as e:
        return None

class ComprehensiveMethodAnalyzer:
    def __init__(self, phenotype, target_databases=None):
        """
        Initialize the comprehensive analyzer for multiple databases
        
        Parameters:
        -----------
        phenotype : str
            Phenotype name (e.g., 'migraine')
        target_databases : list, optional
            List of specific databases to analyze. If None, uses DATABASES_TO_ANALYZE.
        """
        self.phenotype = phenotype
        self.databases = target_databases if target_databases else DATABASES_TO_ANALYZE
        self.datasets = ['training', 'validation', 'test']
        
        # Set up paths
        self.base_path = Path(phenotype)
        self.folds = self._discover_folds()
        
        # Output directory
        self.output_dir = self.base_path / "GeneDifferentialExpression" / "Files"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get current thresholds
        self.significance_thresholds = get_significance_thresholds()
        
        # Cache for combinations
        self._combinations_cache = None
        
        # Warning tracking (to avoid duplicate warnings)
        self._warnings_issued = set()
        
        print(f"🚀 OPTIMIZED COMPREHENSIVE METHOD-DATABASE ANALYSIS")
        print("=" * 80)
        print(f"📋 Phenotype: {phenotype}")
        print(f"📁 Base path: {self.base_path}")
        print(f"📁 Found folds: {', '.join(map(str, self.folds))}")
        print(f"🗃️  Databases: {', '.join(self.databases)} ({len(self.databases)} selected)")
        print(f"📊 Datasets: {', '.join(self.datasets)}")
        print(f"📊 Eight Methods:")
        for method, info in self.significance_thresholds.items():
            marker = "✅" if "TRUE DE" in info['description'] else "⚠️"
            print(f"   {marker} {method}: {info['description']}")
        print(f"💾 Output directory: {self.output_dir}")
        print("=" * 80)
        print(f"🎯 GLOBAL THRESHOLDS:")
        print(f"   FDR Threshold: {FDR_THRESHOLD}")
        print(f"   Effect Threshold: {EFFECT_THRESHOLD}")
        print("=" * 80)
        print(f"⚡ PERFORMANCE SETTINGS:")
        print(f"   Max Workers: {MAX_WORKERS}")
        print(f"   Parallel Processing: {'Enabled' if USE_PARALLEL else 'Disabled'}")
        print(f"   Chunk Size: {CHUNK_SIZE}")
        print("=" * 80)
    
    def _discover_folds(self):
        """Discover available folds"""
        folds = []
        if self.base_path.exists():
            for fold_dir in self.base_path.iterdir():
                if fold_dir.is_dir() and fold_dir.name.startswith('Fold_'):
                    try:
                        fold_num = fold_dir.name.replace('Fold_', '')
                        folds.append(int(fold_num))
                    except ValueError:
                        continue
        
        if not folds:
            print(f"⚠️  No folds found in {self.base_path}")
            folds = [1]
        
        return sorted(folds)
    
    def get_statistical_methods(self):
        """Get the 8 statistical methods"""
        return list(self.significance_thresholds.keys())
    
    from collections import defaultdict
    from pathlib import Path
    
    def discover_all_combinations(self):
        """
        Discover ALL combinations across specified databases, folds, datasets, methods.
        Robust to small directory/name differences and different result-file locations/names.
        """
        if self._combinations_cache is not None:
            return self._combinations_cache
    
        combinations = []
        statistical_methods = self.get_statistical_methods()
    
        print(f"🔍 Discovering all combinations...")
        print(f"🔬 Looking for these 8 methods: {statistical_methods}")
    
        # Normalize method names for matching
        def norm(s: str) -> str:
            return s.strip().lower().replace("-", "_")
    
        method_map = {norm(m): m for m in statistical_methods}  # normalized -> canonical
    
        # Dataset aliases (common mismatches)
        dataset_alias = {
            "train": "training",
            "training": "training",
            "val": "validation",
            "valid": "validation",
            "validation": "validation",
            "test": "test",
        }
    
        # If you have a "_fixed" tree, prefer it when present
        # (If you DON'T have it, it will fall back to the normal tree.)
        base_candidates = [
            "GeneDifferentialExpressions_fixed",
            "GeneDifferentialExpressions",
        ]
    
        # Helper: find the best results file inside a method_dataset_dir
        def pick_results_file(method_dataset_dir: Path) -> Path | None:
            # Common exact locations
            candidates = [
                method_dataset_dir / "differential_expression_results.csv",
                method_dataset_dir / "differential_expression_results_fixed.csv",
                method_dataset_dir / "differential_expression_results.csv.gz",
                method_dataset_dir / "differential_expression_results_fixed.csv.gz",
                method_dataset_dir / "Files" / "differential_expression_results.csv",
                method_dataset_dir / "Files" / "differential_expression_results_fixed.csv",
                method_dataset_dir / "Files" / "differential_expression_results.csv.gz",
                method_dataset_dir / "Files" / "differential_expression_results_fixed.csv.gz",
            ]
            existing = [p for p in candidates if p.exists()]
            if existing:
                # Prefer "fixed" if available
                existing.sort(key=lambda p: ("fixed" not in p.name.lower(), len(str(p))))
                return existing[0]
    
            # Fallback: any file that looks like a DE results file
            glob_hits = list(method_dataset_dir.glob("*differential*expression*results*.csv")) + \
                        list(method_dataset_dir.glob("*differential*expression*results*.csv.gz")) + \
                        list((method_dataset_dir / "Files").glob("*differential*expression*results*.csv")) + \
                        list((method_dataset_dir / "Files").glob("*differential*expression*results*.csv.gz"))
    
            glob_hits = [p for p in glob_hits if p.exists()]
            if glob_hits:
                glob_hits.sort(key=lambda p: ("fixed" not in p.name.lower(), len(str(p))))
                return glob_hits[0]
    
            return None
    
        # Track why things get skipped (helps debugging)
        skipped_no_dataset = 0
        skipped_bad_method = 0
        skipped_no_results = 0
    
        for fold in self.folds:
            for database in self.databases:
                # Choose base folder (prefer fixed tree if it exists)
                base_diff_expr_path = None
                for base_name in base_candidates:
                    candidate = self.base_path / f"Fold_{fold}" / base_name
                    if candidate.exists():
                        base_diff_expr_path = candidate
                        break
    
                if base_diff_expr_path is None:
                    continue
    
                database_path = base_diff_expr_path / database
                if not database_path.exists():
                    continue
    
                # Discover all tissues in this database
                for tissue_dir in database_path.iterdir():
                    if not tissue_dir.is_dir():
                        continue
    
                    tissue_name = tissue_dir.name
    
                    # Look for all method_dataset combinations in this tissue
                    for method_dataset_dir in tissue_dir.iterdir():
                        if not method_dataset_dir.is_dir():
                            continue
    
                        name = method_dataset_dir.name.strip()
    
                        # Parse dataset robustly:
                        # e.g. "Welch_t_test_training", "Welch_t_test_train", "Welch_t_test_training_fixed"
                        parts = name.split("_")
    
                        dataset_found = None
                        cut_idx = None
    
                        # scan from right to left to find a dataset token/alias
                        for i in range(len(parts) - 1, -1, -1):
                            token = parts[i].lower()
                            if token in dataset_alias:
                                dataset_found = dataset_alias[token]
                                cut_idx = i
                                break
    
                        if dataset_found is None or cut_idx is None:
                            skipped_no_dataset += 1
                            continue
    
                        method_raw = "_".join(parts[:cut_idx]).strip()
                        if not method_raw:
                            skipped_bad_method += 1
                            continue
    
                        method_key = norm(method_raw)
                        if method_key not in method_map:
                            skipped_bad_method += 1
                            continue
    
                        method_found = method_map[method_key]  # canonical method name
    
                        # Find results file (robust)
                        results_file = pick_results_file(method_dataset_dir)
                        if results_file is None:
                            skipped_no_results += 1
                            continue
    
                        combinations.append({
                            "fold": fold,
                            "tissue": tissue_name,
                            "database": database,
                            "method": method_found,
                            "dataset": dataset_found,
                            "results_file": results_file
                        })
    
        print(f"📊 Found {len(combinations)} total combinations")
    
        # Show method distribution
        method_counts = defaultdict(int)
        for combo in combinations:
            method_counts[combo["method"]] += 1
    
        print(f"📊 Method distribution:")
        for method in statistical_methods:
            count = method_counts[method]
            marker = "✅" if count > 0 else "❌"
            print(f"   {marker} {method}: {count} files")
    
        # Optional: show skip reasons (super useful when one method is under-counted)
        if skipped_no_dataset or skipped_bad_method or skipped_no_results:
            print("ℹ️  Skipped entries summary:")
            print(f"   - Could not parse dataset suffix : {skipped_no_dataset}")
            print(f"   - Method name not recognized     : {skipped_bad_method}")
            print(f"   - No results file found in dir   : {skipped_no_results}")
    
        # Cache and return
        self._combinations_cache = combinations
        return combinations

    
    def load_and_filter_results(self, results_file, method_name):
        """
        Load results and apply significance thresholds - returns gene set
        
        Parameters:
        -----------
        results_file : Path
            Path to the results CSV file
        method_name : str
            Name of the statistical method
        
        Returns:
        --------
        set : Set of significant gene names
        """
        try:
            df = pd.read_csv(results_file)
            
            if len(df) == 0:
                return set()
            
            if method_name not in self.significance_thresholds:
                return set()
            
            thresholds = self.significance_thresholds[method_name]
            filtered_df = df.copy()
            
            # PRIORITY 1: Apply FDR filter if available (already accounts for multiple testing)
            has_fdr = False
            if 'FDR' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['FDR'] <= thresholds['fdr']]
                has_fdr = True
            # FALLBACK: Use p-value only if FDR unavailable
            elif 'PValue' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['PValue'] <= thresholds['pvalue']]
                # Only warn once per method
                warning_key = f'_pvalue_warning_{method_name}'
                if warning_key not in self._warnings_issued:
                    print(f"   ⚠️  {method_name}: Using uncorrected p-values (FDR column not found)")
                    self._warnings_issued.add(warning_key)
            
            # PRIORITY 2: Apply effect size filter
            effect_col = thresholds['effect_col']
            fallback_col = thresholds.get('fallback_col', None)
            
            effect_size_applied = False
            
            # Try primary effect column first
            if effect_col in filtered_df.columns:
                valid_effects = filtered_df[effect_col].notna()
                filtered_df = filtered_df[valid_effects]
                
                if len(filtered_df) > 0:
                    filtered_df = filtered_df[abs(filtered_df[effect_col]) >= thresholds['effect_size']]
                    effect_size_applied = True
            
            # If primary column failed, try fallback
            if not effect_size_applied and fallback_col and fallback_col in df.columns:
                filtered_df = df.copy()
                
                # Reapply statistical filter
                if has_fdr and 'FDR' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['FDR'] <= thresholds['fdr']]
                elif 'PValue' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['PValue'] <= thresholds['pvalue']]
                
                valid_effects = filtered_df[fallback_col].notna()
                filtered_df = filtered_df[valid_effects]
                
                if len(filtered_df) > 0:
                    filtered_df = filtered_df[abs(filtered_df[fallback_col]) >= thresholds['effect_size']]
            
            # Return set of unique gene names
            if 'Gene' in filtered_df.columns and len(filtered_df) > 0:
                gene_names = filtered_df['Gene'].astype(str).str.strip()
                gene_names = gene_names[~gene_names.isin(['', 'nan', 'NaN', 'None'])]
                return set(gene_names)
            else:
                return set()
            
        except Exception as e:
            return set()
    
    def load_and_count_results(self, results_file, method_name):
        """
        Load results and return count of significant genes
        
        Parameters:
        -----------
        results_file : Path
            Path to the results CSV file
        method_name : str
            Name of the statistical method
        
        Returns:
        --------
        int : Count of significant genes
        """
        gene_set = self.load_and_filter_results(results_file, method_name)
        return len(gene_set)
    
    def load_and_filter_results_parallel(self, combinations_subset):
        """
        Load and filter multiple files in parallel
        
        Parameters:
        -----------
        combinations_subset : list
            List of combination dictionaries to process
        
        Returns:
        --------
        dict : Dictionary mapping combination keys to results
        """
        file_infos = [
            (combo['results_file'], combo['method'], self.significance_thresholds)
            for combo in combinations_subset
        ]
        
        results = {}
        
        if USE_PARALLEL and len(file_infos) > 10:
            # Use parallel processing for large batches
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {executor.submit(load_single_file, info): i 
                          for i, info in enumerate(file_infos)}
                
                for future in as_completed(futures):
                    idx = futures[future]
                    gene_set, count = future.result()
                    combo = combinations_subset[idx]
                    key = (combo['database'], combo['tissue'], combo['method'], 
                          combo['fold'], combo['dataset'])
                    results[key] = {'genes': gene_set, 'count': count}
        else:
            # Use serial processing for small batches or when parallel is disabled
            for i, info in enumerate(file_infos):
                gene_set, count = load_single_file(info)
                combo = combinations_subset[i]
                key = (combo['database'], combo['tissue'], combo['method'], 
                      combo['fold'], combo['dataset'])
                results[key] = {'genes': gene_set, 'count': count}
        
        return results
    
    def save_significant_genes_for_all_databases(self):
        """
        Save significant genes for each database-tissue-method combination
        OPTIMIZED with parallel processing
        
        Returns:
        --------
        tuple : (database_genes dict, all_database_genes dict)
        """
        print(f"\n🔥 SAVING SIGNIFICANT GENES FOR ALL DATABASES...")
        print("=" * 80)
        
        combinations = self.discover_all_combinations()
        
        # Create directory for significant genes
        sig_genes_dir = self.output_dir / "significant_genes"
        sig_genes_dir.mkdir(parents=True, exist_ok=True)
        
        # Load all files in parallel
        print(f"🚀 Loading {len(combinations)} files in parallel...")
        all_results = self.load_and_filter_results_parallel(combinations)
        
        # Organize by database
        database_genes = defaultdict(lambda: defaultdict(list))
        all_database_genes = defaultdict(set)
        
        for combo in combinations:
            database = combo['database']
            tissue = combo['tissue']
            method = combo['method']
            fold = combo['fold']
            dataset = combo['dataset']
            
            key = (database, tissue, method, fold, dataset)
            
            if key in all_results:
                significant_genes = all_results[key]['genes']
                
                if len(significant_genes) > 0:
                    # Store detailed info
                    for gene in significant_genes:
                        database_genes[database][(tissue, method, fold, dataset)].append(gene)
                        all_database_genes[database].add(gene)
        
        # Save individual database files
        for database in self.databases:
            if database in database_genes:
                # Create detailed file
                detailed_data = []
                for (tissue, method, fold, dataset), genes in database_genes[database].items():
                    for gene in genes:
                        detailed_data.append({
                            'Database': database,
                            'Tissue': tissue,
                            'Method': method,
                            'Fold': fold,
                            'Dataset': dataset,
                            'Gene': gene
                        })
                
                if detailed_data:
                    detailed_df = pd.DataFrame(detailed_data)
                    detailed_file = sig_genes_dir / f"{database}_detailed_significant_genes.csv"
                    detailed_df.to_csv(detailed_file, index=False)
                    
                    # Create unique genes file
                    unique_genes = sorted(list(all_database_genes[database]))
                    unique_df = pd.DataFrame({'Gene': unique_genes})
                    unique_file = sig_genes_dir / f"{database}_unique_significant_genes.csv"
                    unique_df.to_csv(unique_file, index=False)
                    
                    print(f"💾 {database}: {len(detailed_data)} entries, {len(unique_genes)} unique genes")
                    print(f"   📄 Detailed: {detailed_file}")
                    print(f"   📄 Unique: {unique_file}")
        
        # Save combined summary
        summary_data = []
        for database in self.databases:
            if database in all_database_genes:
                summary_data.append({
                    'Database': database,
                    'Unique_Genes': len(all_database_genes[database]),
                    'Total_Entries': sum(len(genes) for genes in database_genes[database].values())
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_file = sig_genes_dir / "database_summary.csv"
            summary_df.to_csv(summary_file, index=False)
            print(f"📊 Summary saved: {summary_file}")
        
        return database_genes, all_database_genes
    
    def create_merged_gene_matrix(self, target_databases=None):
        """
        Create matrix with UNIQUE genes merged from ALL methods per tissue-database
        OPTIMIZED with parallel processing
        
        Parameters:
        -----------
        target_databases : list, optional
            List of databases to include. If None, uses all databases.
        
        Returns:
        --------
        tuple : (matrix_df DataFrame, detailed_results dict)
        """
        if target_databases is None:
            target_databases = self.databases
            
        combinations = self.discover_all_combinations()
        
        # Filter to target databases
        combinations = [c for c in combinations if c['database'] in target_databases]
        
        if not combinations:
            print(f"⚠️  No combinations found for databases: {target_databases}")
            return pd.DataFrame(), {}
        
        print(f"🔥 Processing {len(combinations)} combinations for databases: {', '.join(target_databases)}")
        
        # Load all files in parallel
        print(f"🚀 Loading files in parallel...")
        all_results = self.load_and_filter_results_parallel(combinations)
        
        # Collect UNIQUE genes
        unique_gene_sets = defaultdict(set)
        all_tissues = set()
        
        for combo in combinations:
            tissue = combo['tissue']
            database = combo['database']
            method = combo['method']
            
            key = (database, tissue, method, combo['fold'], combo['dataset'])
            
            # Load significant genes for this specific combination
            if key in all_results:
                significant_genes = all_results[key]['genes']
                
                if len(significant_genes) > 0:
                    matrix_key = (tissue, database)
                    unique_gene_sets[matrix_key].update(significant_genes)
                    all_tissues.add(tissue)
        
        # Convert to sorted lists
        tissues_list = sorted(list(all_tissues))
        databases_list = sorted(target_databases)
        
        print(f"🎯 Final matrix: {len(tissues_list)} tissues × {len(databases_list)} databases")
        
        if not tissues_list or not databases_list:
            print(f"⚠️  Empty matrix - no tissues or databases found")
            return pd.DataFrame(), {}
        
        # Create count matrix
        count_matrix = np.zeros((len(tissues_list), len(databases_list)), dtype=int)
        detailed_results = {}
        
        for i, tissue in enumerate(tissues_list):
            for j, database in enumerate(databases_list):
                key = (tissue, database)
                unique_genes = unique_gene_sets.get(key, set())
                gene_count = len(unique_genes)
                count_matrix[i, j] = gene_count
                
                detailed_results[key] = {
                    'unique_genes': unique_genes,
                    'gene_count': gene_count
                }
        
        # Replace any NaN values in count matrix with 0
        count_matrix = np.nan_to_num(count_matrix, nan=0)
        
        # Ensure all tissue-database combinations exist in detailed_results (even if 0)
        for tissue in tissues_list:
            for database in databases_list:
                key = (tissue, database)
                if key not in detailed_results:
                    detailed_results[key] = {
                        'unique_genes': set(),
                        'gene_count': 0
                    }
        
        # Create DataFrame
        if len(databases_list) == 1:
            # Single database - create single column matrix
            matrix_df = pd.DataFrame(
                count_matrix,
                index=tissues_list,
                columns=[databases_list[0]]
            )
        else:
            # Multiple databases
            matrix_df = pd.DataFrame(
                count_matrix,
                index=tissues_list,
                columns=databases_list
            )
        
        # Replace any missing values with 0
        matrix_df = matrix_df.fillna(0)
        
        return matrix_df, detailed_results
    
    def create_tissue_method_matrix(self, target_database):
        """
        Create matrix with tissues (rows) vs methods (columns) for specific database
        FIXED: Now counts UNIQUE genes, not total entries
        """
        combinations = [c for c in self.discover_all_combinations() 
                       if c['database'] == target_database]
        
        if not combinations:
            print(f"⚠️  No combinations found for database: {target_database}")
            return pd.DataFrame(), {}
        
        print(f"🔥 Processing {len(combinations)} combinations for {target_database}")
        
        # Load all files in parallel
        print(f"🚀 Loading files in parallel...")
        all_results = self.load_and_filter_results_parallel(combinations)
        
        # Get expected methods
        expected_methods = self.get_statistical_methods()
        
        # FIXED: Collect UNIQUE genes per tissue-method combination
        tissue_method_unique_genes = defaultdict(lambda: defaultdict(set))
        all_tissues = set()
        detailed_results = defaultdict(list)
        
        for combo in combinations:
            tissue = combo['tissue']
            method = combo['method']
            fold = combo['fold']
            dataset = combo['dataset']
            
            key = (target_database, tissue, method, fold, dataset)
            
            # Get genes for this combination
            if key in all_results:
                genes = all_results[key]['genes']
                gene_count = all_results[key]['count']
            else:
                genes = set()
                gene_count = 0
            
            # Add UNIQUE genes to the tissue-method set
            tissue_method_unique_genes[tissue][method].update(genes)
            all_tissues.add(tissue)
            
            # Store detailed info
            detailed_results[(tissue, method)].append({
                'fold': fold,
                'dataset': dataset,
                'gene_count': gene_count,
                'results_file': str(combo['results_file'])
            })
        
        # Convert to sorted lists
        tissues_list = sorted(list(all_tissues))
        methods_list = expected_methods
        
        print(f"🎯 Creating matrix: {len(tissues_list)} tissues × {len(methods_list)} methods")
        
        if not tissues_list:
            print(f"⚠️  No tissues found")
            return pd.DataFrame(), {}
        
        # FIXED: Count UNIQUE genes per tissue-method
        count_matrix = np.zeros((len(tissues_list), len(methods_list)), dtype=int)
        
        for i, tissue in enumerate(tissues_list):
            for j, method in enumerate(methods_list):
                # Count UNIQUE genes for this tissue-method combination
                unique_genes = tissue_method_unique_genes[tissue][method]
                count_matrix[i, j] = len(unique_genes)
        
        # Create DataFrame
        matrix_df = pd.DataFrame(
            count_matrix,
            index=tissues_list,
            columns=methods_list
        ).fillna(0)
        
        return matrix_df, detailed_results
    
    def run_database_similarity_analysis(self):
        """
        Run database similarity analysis based on common genes
        OPTIMIZED with parallel processing
        
        Returns:
        --------
        DataFrame or None : Similarity matrix
        """
        print(f"\n🔥 ANALYZING DATABASE SIMILARITY...")
        print("=" * 80)
        
        if len(self.databases) < 2:
            print("⚠️  Need at least 2 databases for similarity analysis")
            return None
        
        try:
            # Get gene sets for each database
            combinations = self.discover_all_combinations()
            
            # Load all files in parallel
            print(f"🚀 Loading {len(combinations)} files in parallel...")
            all_results = self.load_and_filter_results_parallel(combinations)
            
            # Collect genes by database
            database_genes = defaultdict(set)
            
            for combo in combinations:
                database = combo['database']
                method = combo['method']
                
                key = (database, combo['tissue'], method, combo['fold'], combo['dataset'])
                
                if key in all_results:
                    significant_genes = all_results[key]['genes']
                    database_genes[database].update(significant_genes)
            
            # Calculate similarity matrix (Jaccard similarity)
            databases_list = sorted([db for db in self.databases if db in database_genes])
            
            if len(databases_list) < 2:
                print("⚠️  Insufficient databases with data for similarity analysis")
                return None
            
            similarity_matrix = np.zeros((len(databases_list), len(databases_list)))
            
            for i, db1 in enumerate(databases_list):
                for j, db2 in enumerate(databases_list):
                    genes1 = database_genes[db1]
                    genes2 = database_genes[db2]
                    
                    if len(genes1) == 0 and len(genes2) == 0:
                        similarity = 1.0  # Both empty
                    elif len(genes1) == 0 or len(genes2) == 0:
                        similarity = 0.0  # One empty
                    else:
                        # Jaccard similarity
                        intersection = len(genes1.intersection(genes2))
                        union = len(genes1.union(genes2))
                        similarity = intersection / union if union > 0 else 0.0
                    
                    similarity_matrix[i, j] = similarity
            
            # Replace any NaN values in similarity matrix with 0
            similarity_matrix = np.nan_to_num(similarity_matrix, nan=0.0)
            
            # Create similarity DataFrame
            similarity_df = pd.DataFrame(
                similarity_matrix,
                index=databases_list,
                columns=databases_list
            )
            
            # Ensure all values are filled with 0
            similarity_df = similarity_df.fillna(0)
            
            # Create professional similarity heatmap
            plt.figure(figsize=(10, 8))
            
            sns.heatmap(
                similarity_df,
                annot=True,
                fmt='.3f',
                cmap='RdYlBu_r',
                center=0.5,
                vmin=0,
                vmax=1,
                square=True,
                mask=None,
                cbar_kws={'label': 'Jaccard Similarity', 'shrink': 0.8},
                linewidths=0.5,
                annot_kws={'fontweight': 'normal'}
            )
            
            plt.title(f'{self.phenotype} - Database Similarity\n(Jaccard Similarity of Significant Genes)', 
                     fontsize=14, fontweight='normal', pad=20)
            plt.xlabel('Database', fontsize=12, fontweight='normal')
            plt.ylabel('Database', fontsize=12, fontweight='normal')
            plt.xticks(rotation=45, ha='right', fontweight='normal')
            plt.yticks(rotation=0, fontweight='normal')
            
            # Save similarity heatmap
            similarity_file = self.output_dir / f"database_similarity_heatmap.png"
            plt.savefig(similarity_file, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            # Save similarity matrix
            matrix_file = self.output_dir / f"database_similarity_matrix.csv"
            similarity_df.to_csv(matrix_file)
            
            print(f"🔥 Database similarity analysis completed")
            print(f"🔥 Similarity heatmap saved: {similarity_file}")
            print(f"🔥 Similarity matrix saved: {matrix_file}")
            
            return similarity_df
            
        except Exception as e:
            print(f"❌ Error in similarity analysis: {str(e)}")
            return None
    
    def create_professional_heatmap(self, matrix_df, title, save_path=None, show_zeros=True):
        """
        Create heatmap using original theme
        
        Parameters:
        -----------
        matrix_df : DataFrame
            Matrix to plot
        title : str
            Plot title
        save_path : Path, optional
            Where to save the plot
        show_zeros : bool
            Whether to show zero values
        
        Returns:
        --------
        Path or None : Path to saved file
        """
        if matrix_df.empty:
            print("❌ No data to plot")
            return None
        
        print(f"🔥 Creating heatmap: {title}")
        
        # Get dimensions for figure sizing
        n_tissues = len(matrix_df.index)
        n_methods = len(matrix_df.columns)
        
        # Original figure sizing
        width = max(12, n_methods * 2.5)
        height = max(8, n_tissues * 0.4)
        plt.figure(figsize=(width, height))
        
        # Ensure all values are properly filled with 0 for missing data
        matrix_df = matrix_df.fillna(0)
        
        # Create annotation matrix
        annotation_matrix = matrix_df.copy().astype(str)
        for i in range(len(matrix_df.index)):
            for j in range(len(matrix_df.columns)):
                value = matrix_df.iloc[i, j]
                if pd.notna(value) and value > 0:
                    if value >= 1000:
                        annotation_matrix.iloc[i, j] = f"{int(value/1000)}k"
                    elif value >= 100:
                        annotation_matrix.iloc[i, j] = f"{int(value)}"
                    else:
                        annotation_matrix.iloc[i, j] = f"{int(value)}"
                else:
                    annotation_matrix.iloc[i, j] = "0"
        
        # Normalization
        normalized_matrix = matrix_df.copy().astype(float)
        max_value = normalized_matrix.max().max()
        if max_value > 0:
            normalized_matrix = np.log1p(normalized_matrix) / np.log1p(max_value)
        
        # Create heatmap
        ax = sns.heatmap(
            normalized_matrix,
            annot=annotation_matrix,
            fmt='',
            cmap='RdYlBu_r',
            center=0.5,
            vmin=0,
            vmax=1,
            mask=None,
            cbar_kws={'label': 'Normalized Gene Count (Log Scale)', 'shrink': 0.8},
            linewidths=0.8,
            square=False,
            annot_kws={'fontsize': min(12, max(8, 120//max(n_tissues, n_methods))), 
                      'fontweight': 'normal'},
            xticklabels=True,
            yticklabels=True
        )
        
        # Title formatting
        plt.title(title, fontsize=min(16, max(12, 200//max(n_tissues, n_methods))), 
                 fontweight='normal', pad=25)
        plt.xlabel('Database' if n_methods > 1 else 'Expression Method', 
                  fontsize=14, fontweight='normal')
        plt.ylabel('Tissue (Alphabetical Order)', fontsize=14, fontweight='normal')
        
        # Axis label formatting
        ax.set_xticklabels(matrix_df.columns, rotation=45, ha='right', 
                          fontsize=11, fontweight='normal')
        plt.yticks(rotation=0, fontsize=min(11, max(8, 150//n_tissues)), fontweight='normal')
        
        # Add vertical lines to separate databases (if multiple)
        if n_methods > 1:
            for i in range(1, n_methods):
                plt.axvline(x=i, color='white', linewidth=2)
        
        plt.tight_layout()
        
        # Save the heatmap
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"🔥 Heatmap saved: {save_path}")
        
        plt.close()
        return save_path
    
    def create_combined_volcano_plot(self):
        """
        Create publication-quality volcano plot combining all databases
        OPTIMIZED with parallel processing
        
        Returns:
        --------
        Path or None : Path to combined volcano plot
        """
        print(f"\n🔥 CREATING VOLCANO PLOTS...")
        print("=" * 80)
        
        try:
            combinations = self.discover_all_combinations()
            
            if not combinations:
                print("⚠️  No combinations found for volcano plot")
                return None
            
            print(f"🔥 Processing {len(combinations)} combinations for volcano plot...")
            
            # Prepare file info for parallel loading
            file_infos = [(combo, self.significance_thresholds) for combo in combinations]
            
            # Collect and process data by database
            database_data = defaultdict(list)
            processed_files = 0
            
            if USE_PARALLEL and len(file_infos) > 10:
                # Use parallel processing
                print(f"🚀 Loading files in parallel with {MAX_WORKERS} workers...")
                with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    futures = [executor.submit(load_volcano_file, info) for info in file_infos]
                    
                    for future in as_completed(futures):
                        result = future.result()
                        if result is not None and len(result) > 0:
                            database = result['Database'].iloc[0]
                            database_data[database].append(result)
                            processed_files += 1
            else:
                # Use serial processing
                print(f"📊 Loading files sequentially...")
                for info in file_infos:
                    result = load_volcano_file(info)
                    if result is not None and len(result) > 0:
                        database = result['Database'].iloc[0]
                        database_data[database].append(result)
                        processed_files += 1
            
            if not database_data:
                print("❌ No valid data found for volcano plot")
                return None
            
            print(f"📊 Successfully processed {processed_files} files across {len(database_data)} Gene Expression Models")
            
            # Create individual volcano plots for each model
            individual_volcano_files = []
            for model_name, data_list in database_data.items():
                if data_list:
                    print(f"\n🔬 Creating volcano plot for {model_name} Gene Expression Model...")
                    combined_model_df = pd.concat(data_list, ignore_index=True)
                    volcano_file = self.create_individual_volcano_plot(combined_model_df, model_name)
                    if volcano_file:
                        individual_volcano_files.append(volcano_file)
            
            # Create combined volcano plot
            print(f"\n🔬 Creating combined volcano plot for all Gene Expression Models...")
            all_data = []
            for db_list in database_data.values():
                all_data.extend(db_list)
            
            if not all_data:
                print("❌ No data for combined volcano plot")
                return None
            
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_volcano_file = self.create_multi_database_volcano_plot(combined_df)
            
            # Save combined data
            data_file = self.output_dir / f"combined_volcano_data_all_models.csv"
            combined_df.to_csv(data_file, index=False)
            
            print(f"\n🔥 VOLCANO PLOT SUMMARY:")
            print(f"✅ Individual volcano plots created: {len(individual_volcano_files)}")
            print(f"✅ Combined volcano plot: {combined_volcano_file}")
            print(f"✅ Volcano data saved: {data_file}")
            
            return combined_volcano_file
            
        except Exception as e:
            print(f"❌ Error creating volcano plot: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_individual_volcano_plot(self, plot_df, database_name):
        """
        Create individual volcano plot for a single database
        FIXED: Deduplicate genes to show unique genes only
        """
        try:
            if len(plot_df) == 0:
                return None
            
            # ===== FIX: DEDUPLICATE GENES =====
            # Keep only the most significant entry per gene (lowest FDR)
            print(f"      📊 Before deduplication: {len(plot_df):,} entries, {plot_df['Gene'].nunique():,} unique genes")
            plot_df = plot_df.sort_values('FDR').drop_duplicates(subset=['Gene'], keep='first')
            print(f"      📊 After deduplication: {len(plot_df):,} unique genes")
            # ==================================
            
            # Remove extreme outliers
            reasonable_fc = plot_df[abs(plot_df['Log2FoldChange']) <= 15]
            if len(reasonable_fc) < len(plot_df) * 0.95:
                reasonable_fc = plot_df[abs(plot_df['Log2FoldChange']) <= 25]
            
            reasonable_data = reasonable_fc[reasonable_fc['neg_log10_FDR'] <= 50]
            if len(reasonable_data) < len(reasonable_fc) * 0.95:
                reasonable_data = reasonable_fc[reasonable_fc['neg_log10_FDR'] <= 100]
            
            if len(reasonable_data) == 0:
                return None
            
            plot_df = reasonable_data.copy()
            
            # Separate significant and non-significant genes
            sig_genes = plot_df[plot_df['is_significant']]
            non_sig_genes = plot_df[~plot_df['is_significant']]
            
            # Create plot
            plt.figure(figsize=(12, 10))
            
            # Calculate axis limits
            x_data = plot_df['Log2FoldChange']
            y_data = plot_df['neg_log10_FDR']
            
            x_min_data = x_data.min()
            x_max_data = x_data.max()
            y_min_data = 0
            y_max_data = y_data.max()
            
            x_min = min(x_min_data, -EFFECT_THRESHOLD * 1.5)
            x_max = max(x_max_data, EFFECT_THRESHOLD * 1.5)
            
            fdr_line_y = -np.log10(FDR_THRESHOLD)
            y_max = max(y_max_data, fdr_line_y * 1.3)
            
            x_range = x_max - x_min
            y_range = y_max - y_min_data
            x_pad = max(0.5, x_range * 0.15)
            y_pad = max(1.0, y_range * 0.1)
            
            final_x_min = x_min - x_pad
            final_x_max = x_max + x_pad
            final_y_min = y_min_data
            final_y_max = y_max + y_pad
            
            # Plot non-significant genes
            if len(non_sig_genes) > 0:
                plt.scatter(
                    non_sig_genes['Log2FoldChange'], 
                    non_sig_genes['neg_log10_FDR'],
                    c='lightgray',
                    alpha=0.6,
                    s=20,
                    label=f'Non-significant ({len(non_sig_genes):,})',
                    edgecolors='none'
                )
            
            # Plot significant genes
            if len(sig_genes) > 0:
                plt.scatter(
                    sig_genes['Log2FoldChange'], 
                    sig_genes['neg_log10_FDR'],
                    c='red',
                    alpha=0.8,
                    s=30,
                    label=f'Significant ({len(sig_genes):,})',
                    edgecolors='darkred',
                    linewidths=0.5
                )
            
            # Add threshold lines
            if fdr_line_y <= final_y_max:
                plt.axhline(y=fdr_line_y, color='red', linestyle='-', alpha=0.9, linewidth=4, 
                           label=f'FDR = {FDR_THRESHOLD}', zorder=10)
            
            if EFFECT_THRESHOLD <= final_x_max and EFFECT_THRESHOLD >= final_x_min:
                plt.axvline(x=EFFECT_THRESHOLD, color='blue', linestyle='-', alpha=0.9, linewidth=4,
                           label=f'Log₂FC = ±{EFFECT_THRESHOLD}', zorder=10)
            
            if -EFFECT_THRESHOLD <= final_x_max and -EFFECT_THRESHOLD >= final_x_min:
                plt.axvline(x=-EFFECT_THRESHOLD, color='blue', linestyle='-', alpha=0.9, linewidth=4, zorder=10)
            
            # Set axis limits
            plt.xlim(final_x_min, final_x_max)
            plt.ylim(final_y_min, final_y_max)
            
            # Labels
            plt.xlabel('Log₂ Fold Change', fontsize=14, fontweight='bold')
            plt.ylabel('-log₁₀(FDR)', fontsize=14, fontweight='bold')
            plt.title(f'{self.phenotype} - {database_name} Gene Expression Model\nVolcano Plot (Unique Genes)', 
                     fontsize=16, fontweight='bold', pad=20)
            
            # Add statistics
            total_genes = len(plot_df)
            sig_count = len(sig_genes)
            sig_percentage = (sig_count / total_genes * 100) if total_genes > 0 else 0
            
            plt.text(0.98, 0.98, f'Unique genes: {total_genes:,}\nSignificant: {sig_count:,} ({sig_percentage:.1f}%)', 
                    transform=plt.gca().transAxes, 
                    fontsize=11, fontweight='bold', verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))
            
            # Legend
            plt.legend(loc='upper left', fontsize=10, frameon=True, fancybox=True, shadow=True)
            
            # Grid
            plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            plt.tight_layout()
            
            # Save
            safe_name = database_name.replace(" ", "_").replace("/", "_")
            volcano_file = self.output_dir / f"volcano_plot_{safe_name.lower()}_model.png"
            plt.savefig(volcano_file, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return volcano_file
            
        except Exception as e:
            print(f"❌ Error creating {database_name} volcano plot: {str(e)}")
            return None
    
    
    def create_multi_database_volcano_plot(self, combined_df):
        """
        Create combined volcano plot with different colors for each database
        FIXED: Deduplicate genes to show unique genes only
        """
        try:
            if len(combined_df) == 0:
                return None
            
            # ===== FIX: DEDUPLICATE GENES =====
            # Keep only the most significant entry per gene (lowest FDR)
            print(f"      📊 Before deduplication: {len(combined_df):,} entries, {combined_df['Gene'].nunique():,} unique genes")
            combined_df = combined_df.sort_values('FDR').drop_duplicates(subset=['Gene'], keep='first')
            print(f"      📊 After deduplication: {len(combined_df):,} unique genes")
            # ==================================
            
            # Filter outliers
            reasonable_fc = combined_df[abs(combined_df['Log2FoldChange']) <= 15]
            reasonable_data = reasonable_fc[reasonable_fc['neg_log10_FDR'] <= 50]
            
            if len(reasonable_data) == 0:
                return None
            
            combined_df = reasonable_data.copy()
            
            # Define colors
            model_colors = {
                'Regular': {'sig': '#e74c3c', 'non_sig': '#f8d7da'},
                'JTI': {'sig': '#3498db', 'non_sig': '#d1ecf1'},
                'UTMOST': {'sig': '#f39c12', 'non_sig': '#fdeaa7'},
                'UTMOST2': {'sig': '#9b59b6', 'non_sig': '#e8daef'},
                'EpiX': {'sig': '#1abc9c', 'non_sig': '#d5f4e6'},
                'TIGAR': {'sig': '#34495e', 'non_sig': '#d5dbdb'},
                'FUSION': {'sig': '#27ae60', 'non_sig': '#d5f4e6'}
            }
            
            # Create plot
            plt.figure(figsize=(14, 12))
            
            # Calculate limits
            x_data = combined_df['Log2FoldChange']
            y_data = combined_df['neg_log10_FDR']
            
            x_min = min(x_data.min(), -EFFECT_THRESHOLD * 1.5)
            x_max = max(x_data.max(), EFFECT_THRESHOLD * 1.5)
            
            fdr_line_y = -np.log10(FDR_THRESHOLD)
            y_max = max(y_data.max(), fdr_line_y * 1.3)
            
            x_range = x_max - x_min
            y_range = y_max
            x_pad = max(0.5, x_range * 0.15)
            y_pad = max(1.0, y_range * 0.1)
            
            final_x_min = x_min - x_pad
            final_x_max = x_max + x_pad
            final_y_max = y_max + y_pad
            
            # Track all databases for legend
            all_databases = sorted(self.databases)
            plotted_databases = set()
            
            # Plot by database
            for database in combined_df['Database'].unique():
                db_data = combined_df[combined_df['Database'] == database]
                sig_genes = db_data[db_data['is_significant']]
                non_sig_genes = db_data[~db_data['is_significant']]
                
                colors = model_colors.get(database, {'sig': '#e74c3c', 'non_sig': '#f8d7da'})
                plotted_databases.add(database)
                
                # Plot non-significant
                if len(non_sig_genes) > 0:
                    plt.scatter(
                        non_sig_genes['Log2FoldChange'], 
                        non_sig_genes['neg_log10_FDR'],
                        c=colors['non_sig'],
                        alpha=0.6,
                        s=15,
                        edgecolors='none'
                    )
                
                # Plot significant
                plt.scatter(
                    sig_genes['Log2FoldChange'] if len(sig_genes) > 0 else [], 
                    sig_genes['neg_log10_FDR'] if len(sig_genes) > 0 else [],
                    c=colors['sig'],
                    alpha=0.8,
                    s=25,
                    edgecolors='black',
                    linewidths=0.5,
                    label=f'{database}: {len(sig_genes):,} genes'
                )
            
            # Add entries for databases with no data
            for database in all_databases:
                if database not in plotted_databases:
                    colors = model_colors.get(database, {'sig': '#e74c3c', 'non_sig': '#f8d7da'})
                    plt.scatter([], [], c=colors['sig'], alpha=0.8, s=25,
                              edgecolors='black', linewidths=0.5,
                              label=f'{database}: 0 genes')
            
            # Add threshold lines
            if fdr_line_y <= final_y_max:
                plt.axhline(y=fdr_line_y, color='red', linestyle='-', alpha=0.9, linewidth=4, 
                           label=f'FDR = {FDR_THRESHOLD}', zorder=10)
            
            plt.axvline(x=EFFECT_THRESHOLD, color='blue', linestyle='-', alpha=0.9, linewidth=4,
                       label=f'Log₂FC = ±{EFFECT_THRESHOLD}', zorder=10)
            plt.axvline(x=-EFFECT_THRESHOLD, color='blue', linestyle='-', alpha=0.9, linewidth=4, zorder=10)
            
            # Set limits
            plt.xlim(final_x_min, final_x_max)
            plt.ylim(0, final_y_max)
            
            # Labels
            plt.xlabel('Log₂ Fold Change', fontsize=14, fontweight='bold')
            plt.ylabel('-log₁₀(FDR)', fontsize=14, fontweight='bold')
            plt.title(f'{self.phenotype} - Combined Databases\nVolcano Plot (Unique Genes Only)', 
                     fontsize=16, fontweight='bold', pad=20)
            
            # Statistics
            total_genes = len(combined_df)
            total_sig = len(combined_df[combined_df['is_significant']])
            sig_percentage = (total_sig / total_genes * 100) if total_genes > 0 else 0
            
            plt.text(0.02, 0.98, f'Unique genes: {total_genes:,}\nSignificant: {total_sig:,} ({sig_percentage:.1f}%)', 
                    transform=plt.gca().transAxes, 
                    fontsize=11, fontweight='bold', verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9))
            
            # Legends
            handles, labels = plt.gca().get_legend_handles_labels()
            
            threshold_handles = []
            threshold_labels = []
            database_handles = []
            database_labels = []
            
            for handle, label in zip(handles, labels):
                if 'FDR' in label or 'Log₂FC' in label:
                    threshold_handles.append(handle)
                    threshold_labels.append(label)
                else:
                    database_handles.append(handle)
                    database_labels.append(label)
            
            # Sort database entries
            if database_handles:
                db_pairs = list(zip(database_handles, database_labels))
                db_pairs.sort(key=lambda x: x[1].split(':')[0])
                database_handles, database_labels = zip(*db_pairs)
            
            # Create legends
            if threshold_handles:
                threshold_legend = plt.legend(threshold_handles, threshold_labels, 
                                            loc='upper left', fontsize=10, 
                                            frameon=True, fancybox=True, shadow=True,
                                            title='Significance Thresholds', title_fontsize=11,
                                            bbox_to_anchor=(0.02, 0.85))
                plt.gca().add_artist(threshold_legend)
            
            if database_handles:
                plt.legend(database_handles, database_labels, 
                          loc='upper right', fontsize=10, 
                          frameon=True, fancybox=True, shadow=True,
                          title='Gene Expression Models', title_fontsize=11,
                          bbox_to_anchor=(0.98, 0.98))
            
            # Grid
            plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            plt.tight_layout()
            
            # Save
            volcano_file = self.output_dir / f"volcano_plot_combined_all_models.png"
            plt.savefig(volcano_file, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return volcano_file
            
        except Exception as e:
            print(f"❌ Error creating combined volcano plot: {str(e)}")
            return None
    
    def run_complete_analysis(self):
        """
        Run complete analysis for all specified databases
        
        Returns:
        --------
        tuple : (individual_results, combined_result, similarity_result, volcano_result)
        """
        print(f"\n🔥 RUNNING SIGNIFICANT GENE ANALYSIS (8 METHODS)...")
        print("=" * 80)
        
        individual_results = []
        combined_result = None
        similarity_result = None
        volcano_result = None
        saved_genes_result = None
        
        try:
            # 0. Save significant genes
            print(f"\n📊 STEP 0: Saving Significant Genes...")
            saved_genes_result = self.save_significant_genes_for_all_databases()
            
            # 1. Individual database analysis
            print(f"\n📊 STEP 1: Individual Database Analysis...")
            for database in self.databases:
                print(f"\n🔬 Analyzing {database} database...")
                matrix_df, detailed_results = self.create_tissue_method_matrix(database)
                
                if not matrix_df.empty:
                    title = f"{self.phenotype} - {database} Database\nTissues vs Methods (8 Methods, FDR<{FDR_THRESHOLD})"
                    save_path = self.output_dir / f"heatmap_{database.lower().replace(' ', '_')}_database.png"
                    
                    heatmap_file = self.create_professional_heatmap(matrix_df, title, save_path)
                    
                    if heatmap_file:
                        individual_results.append({
                            'database': database,
                            'heatmap_file': heatmap_file,
                            'matrix_df': matrix_df,
                            'detailed_results': detailed_results
                        })
                        
                        matrix_df.to_csv(self.output_dir / f"matrix_{database.lower().replace(' ', '_')}_database.csv")
                        print(f"🔥 {database} analysis completed")
            
            # 2. Combined database analysis
            print(f"\n📊 STEP 2: Combined Database Analysis...")
            if len(self.databases) > 1:
                combined_matrix_df, combined_detailed = self.create_merged_gene_matrix()
                
                if not combined_matrix_df.empty:
                    title = f"{self.phenotype} - All Databases Combined\nTissues vs Databases (FDR<{FDR_THRESHOLD})"
                    save_path = self.output_dir / f"heatmap_combined_all_databases.png"
                    
                    combined_heatmap = self.create_professional_heatmap(combined_matrix_df, title, save_path)
                    
                    if combined_heatmap:
                        combined_result = {
                            'heatmap_file': combined_heatmap,
                            'matrix_df': combined_matrix_df,
                            'detailed_results': combined_detailed
                        }
                        combined_matrix_df.to_csv(self.output_dir / f"matrix_combined_all_databases.csv")
            
            # 3. Database similarity
            print(f"\n📊 STEP 3: Database Similarity Analysis...")
            similarity_result = self.run_database_similarity_analysis()
            
            # 4. Volcano plots
            print(f"\n📊 STEP 4: Volcano Plot Analysis...")
            volcano_result = self.create_combined_volcano_plot()
            
            # Summary
            print(f"\n🔥 ANALYSIS COMPLETE - 8 STATISTICAL METHODS")
            print("=" * 80)
            print(f"✅ Methods analyzed:")
            for method in self.get_statistical_methods():
                info = self.significance_thresholds[method]
                marker = "✅" if "TRUE DE" in info['description'] else "⚠️"
                print(f"   {marker} {method}")
            print(f"✅ Thresholds: FDR<{FDR_THRESHOLD}, |Log2FC|≥{EFFECT_THRESHOLD}")
            print(f"✅ Individual databases: {len(individual_results)}/{len(self.databases)}")
            print(f"✅ Combined analysis: {'✓' if combined_result else '✗'}")
            print(f"✅ Database similarity: {'✓' if similarity_result is not None else '✗'}")
            print(f"✅ Volcano plots: {'✓' if volcano_result else '✗'}")
            print("=" * 80)
            
            return individual_results, combined_result, similarity_result, volcano_result
            
        except Exception as e:
            print(f"❌ Error in complete analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return individual_results, combined_result, similarity_result, volcano_result

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="🔥 OPTIMIZED Comprehensive Differential Expression Analysis - 8 Methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
SIMPLE USAGE:
  python predict4.1-GeneDifferentialExpressionFindSignificantGenes.py migraine

CONFIGURATION:
  FDR_THRESHOLD = {FDR_THRESHOLD}     # False Discovery Rate
  EFFECT_THRESHOLD = {EFFECT_THRESHOLD}   # Log2FoldChange threshold
  MAX_WORKERS = {MAX_WORKERS}          # Parallel workers (auto-detected)
  USE_PARALLEL = {USE_PARALLEL}         # Enable/disable parallel processing
  
8 STATISTICAL METHODS:
  ✅ TRUE DIFFERENTIAL EXPRESSION:
     - LIMMA (gold standard)
     - Welch's t-test
     - Linear Regression
     - Wilcoxon Rank-Sum
     - Permutation Test
  
  ⚠️  ASSOCIATION TESTING (for comparison):
     - Weighted Logistic
     - Firth Logistic
     - Bayesian Logistic

OUTPUT:
  - Individual database heatmaps (tissues × methods)
  - Combined database heatmap (tissues × databases)
  - Database similarity analysis
  - Individual and combined volcano plots
  - Significant gene lists

PERFORMANCE OPTIMIZATION:
  - Parallel file loading ({MAX_WORKERS} workers)
  - Cached filesystem operations
  - Vectorized data processing
  - Optimized memory usage
  - Expected speedup: 4-20x faster depending on CPU cores
        """
    )
    
    parser.add_argument("phenotype", help="Phenotype name (e.g., migraine)")
    args = parser.parse_args()
    
    print(f"🚀 OPTIMIZED DIFFERENTIAL EXPRESSION ANALYSIS - 8 METHODS")
    print("=" * 80)
    print(f"📋 Phenotype: {args.phenotype}")
    print(f"📋 Databases: {', '.join(DATABASES_TO_ANALYZE)}")
    print(f"📋 FDR Threshold: {FDR_THRESHOLD}")
    print(f"📋 Effect Threshold: {EFFECT_THRESHOLD}")
    print(f"⚡ Parallel Workers: {MAX_WORKERS}")
    print(f"⚡ Parallel Processing: {'Enabled' if USE_PARALLEL else 'Disabled'}")
    print("=" * 80)
    
    try:
        analyzer = ComprehensiveMethodAnalyzer(phenotype=args.phenotype)
        individual_results, combined_result, similarity_result, volcano_result = analyzer.run_complete_analysis()
        
        if len(individual_results) > 0 or combined_result or similarity_result or volcano_result:
            print(f"\n🔥 SUCCESS! Analysis completed for {args.phenotype}")
            print(f"🔥 Results saved to: {analyzer.output_dir}")
            print(f"🔥 8 statistical methods analyzed")
            print(f"🔥 Thresholds: FDR<{FDR_THRESHOLD}, |Log2FC|≥{EFFECT_THRESHOLD}")
            print(f"🔥 Performance: Optimized with {MAX_WORKERS} parallel workers")
        else:
            print(f"\n❌ No results generated")
            sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()