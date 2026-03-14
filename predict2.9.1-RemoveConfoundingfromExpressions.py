#!/usr/bin/env python3
"""
Gene Expression Covariate Adjustment Script (v2 - Corrected)
=============================================================
This script adjusts gene expression values by regressing out covariates
(Sex and PC1-PC10) from expression data.

For each expression file:
1. Load expression data
2. Load corresponding covariate file (COV_PCA)
3. Match samples between expression and covariates
4. Residualize all genes at once using vectorized linear algebra
5. Save adjusted file with '_fixed' suffix

Usage:
    python3 adjust_expression_covariates.py <phenotype>
    
Example:
    python3 adjust_expression_covariates.py BMI
"""

import os
import sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import argparse
from pathlib import Path

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_DIRS = {
    'Regular': {
        'train': 'TrainExpression', 
        'test': 'TestExpression', 
        'validation': 'ValidationExpression'
    },
    'JTI': {
        'train': 'JTITrainExpression', 
        'test': 'JTITestExpression', 
        'validation': 'JTIValidationExpression'
    },
    'UTMOST': {
        'train': 'UTMOSTTrainExpression', 
        'test': 'UTMOSTTestExpression', 
        'validation': 'UTMOSTValidationExpression'
    },
    'UTMOST2': {
        'train': 'utmost2TrainExpression', 
        'test': 'utmost2TestExpression', 
        'validation': 'utmost2ValidationExpression'
    },
    'EpiX': {
        'train': 'EpiXTrainExpression', 
        'test': 'EpiXTestExpression', 
        'validation': 'EpiXValidationExpression'
    },
    'TIGAR': {
        'train': 'TigarTrainExpression', 
        'test': 'TigarTestExpression', 
        'validation': 'TigarValidExpression'
    },
    'FUSION': {
        'train': 'FussionExpression', 
        'test': 'FussionExpression', 
        'validation': 'FussionExpression'
    }
}

# Special file naming for FUSION
FILE_NAMES = {
    'FUSION': {
        'train': 'GeneExpression_train_data.csv',
        'test': 'GeneExpression_test_data.csv',
        'validation': 'GeneExpression_validation_data.csv'
    }
}

# Covariate file mapping
COVARIATE_FILES = {
    'train': 'train_data.COV_PCA',
    'test': 'test_data.COV_PCA',
    'validation': 'validation_data.COV_PCA'
}

# Covariates to use for adjustment
COVARIATES_TO_USE = ['Sex', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10']

# Whether to add back mean after residualization (keeps similar scale)
ADD_BACK_MEAN = True


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_covariate_file(cov_path):
    """
    Load covariate file and return DataFrame with sample IDs as index.
    
    Parameters:
    -----------
    cov_path : str
        Path to the covariate file
        
    Returns:
    --------
    pd.DataFrame
        Covariate data with sample_id (FID_IID) as index
    """
    try:
        # Try tab-separated first (most common for these files)
        df = pd.read_csv(cov_path, sep='\t')
        
        # Check if we got expected columns - retry with whitespace if either is missing
        if ('IID' not in df.columns) or ('FID' not in df.columns):
            df = pd.read_csv(cov_path, sep=r'\s+')
        
        # Explicit check: both FID and IID must exist
        required = {'FID', 'IID'}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            raise ValueError(f"Covariate file missing required columns {missing}. Found: {list(df.columns)[:20]}")
        
        # Create sample ID
        df['sample_id'] = df['FID'].astype(str) + '_' + df['IID'].astype(str)
        
        # Check for duplicates
        if df['sample_id'].duplicated().any():
            n_dups = df['sample_id'].duplicated().sum()
            print(f"    ??  Found {n_dups} duplicate sample IDs in covariate file, keeping first")
            df = df.drop_duplicates(subset=['sample_id'], keep='first')
        
        df = df.set_index('sample_id')
        
        return df
        
    except Exception as e:
        print(f"  ??  Error loading covariate file {cov_path}: {e}")
        return None


def load_expression_file(expr_path):
    """
    Load expression file and return DataFrame.
    
    Parameters:
    -----------
    expr_path : str
        Path to expression file
        
    Returns:
    --------
    pd.DataFrame
        Expression data with FID, IID columns and gene columns
    """
    try:
        df = pd.read_csv(expr_path)
        return df
    except Exception as e:
        print(f"  ??  Error loading expression file {expr_path}: {e}")
        return None


def encode_sex_column(cov_df):
    """
    Robustly encode Sex column to numeric 0/1.
    
    Handles:
    - PLINK style: 1=male, 2=female -> 0, 1
    - String style: "M"/"F", "male"/"female" -> 0, 1
    - Already 0/1 -> unchanged
    
    Parameters:
    -----------
    cov_df : pd.DataFrame
        Covariate dataframe (modified in place)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with encoded Sex column
    """
    if 'Sex' not in cov_df.columns:
        return cov_df
    
    sex_col = cov_df['Sex'].copy()
    original_na_count = sex_col.isna().sum()
    
    # Check if string type
    if sex_col.dtype == object:
        # Map string values to numeric
        sex_map = {
            'M': 0, 'F': 1,
            'm': 0, 'f': 1,
            'Male': 0, 'Female': 1,
            'male': 0, 'female': 1,
            'MALE': 0, 'FEMALE': 1,
            '1': 0, '2': 1,  # String versions of PLINK codes
        }
        cov_df['Sex'] = sex_col.map(sex_map).astype(float)
    else:
        # Numeric: check if PLINK style (1/2)
        unique_vals = set(sex_col.dropna().unique())
        if unique_vals <= {1, 2}:
            # PLINK style: 1=male->0, 2=female->1
            cov_df['Sex'] = sex_col.replace({1: 0, 2: 1}).astype(float)
        elif unique_vals <= {0, 1}:
            # Already 0/1, keep as is
            cov_df['Sex'] = sex_col.astype(float)
        else:
            # Unknown coding, try to standardize
            print(f"    ??  Unknown Sex coding: {unique_vals}, treating as continuous")
            cov_df['Sex'] = sex_col.astype(float)
    
    # Check if encoding created many new NaNs (indicates mapping failures)
    new_na_count = cov_df['Sex'].isna().sum()
    na_from_encoding = new_na_count - original_na_count
    na_fraction = new_na_count / len(cov_df) if len(cov_df) > 0 else 0
    
    if na_from_encoding > 0:
        print(f"    ??  Sex encoding created {na_from_encoding} new NaN values")
    
    if na_fraction > 0.05:
        print(f"    ??  More than 5% ({na_fraction:.1%}) Sex values could not be encoded; check coding!")
        # Show unique unmapped values for debugging
        if sex_col.dtype == object:
            unmapped = sex_col[cov_df['Sex'].isna() & sex_col.notna()].unique()
            if len(unmapped) > 0:
                print(f"    ??  Unmapped Sex values: {list(unmapped)[:10]}")
    
    return cov_df


def residualize_vectorized(Y, X, add_back_mean=True):
    """
    Residualize Y against X using vectorized linear algebra.
    
    Computes: Y_res = Y - X @ (X^+ @ Y)
    
    Where X^+ is the Moore-Penrose pseudoinverse.
    
    Parameters:
    -----------
    Y : np.ndarray
        Expression matrix (n_samples x n_genes)
    X : np.ndarray
        Covariate matrix (n_samples x n_covariates)
    add_back_mean : bool
        If True, add back the column means to residuals
        
    Returns:
    --------
    np.ndarray
        Residualized expression matrix (n_samples x n_genes)
    """
    # Store original means if needed
    if add_back_mean:
        Y_means = np.nanmean(Y, axis=0)
    
    # Handle any remaining NaN in Y by temporarily replacing with column means
    Y_filled = Y.copy()
    col_means = np.nanmean(Y_filled, axis=0)
    nan_mask = np.isnan(Y_filled)
    
    # Fill NaN with column means (vectorized for performance with many genes)
    if nan_mask.any():
        idx = np.where(nan_mask)
        Y_filled[idx] = np.take(col_means, idx[1])
        # Handle columns that are entirely NaN (col_means will be NaN)
        Y_filled[np.isnan(Y_filled)] = 0.0
    
    # Add intercept to X
    X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
    
    # Compute pseudoinverse
    try:
        X_pinv = np.linalg.pinv(X_with_intercept)
    except np.linalg.LinAlgError:
        print("    ??  Pseudoinverse computation failed, returning original data")
        return Y
    
    # Compute residuals: Y - X @ (X^+ @ Y)
    # This is equivalent to: Y - X @ beta where beta = (X'X)^-1 X'Y
    Y_hat = X_with_intercept @ (X_pinv @ Y_filled)
    residuals = Y_filled - Y_hat
    
    # Add back means if requested
    if add_back_mean:
        residuals = residuals + Y_means
    
    # Restore NaN positions
    residuals[nan_mask] = np.nan
    
    return residuals


def adjust_expression_for_covariates(expr_df, cov_df, covariates, add_back_mean=True):
    """
    Adjust gene expression by regressing out covariates using vectorized operations.
    
    Parameters:
    -----------
    expr_df : pd.DataFrame
        Expression data with FID, IID and gene columns
    cov_df : pd.DataFrame
        Covariate data with sample_id as index
    covariates : list
        List of covariate column names to regress out
    add_back_mean : bool
        If True, add back gene means to residuals
        
    Returns:
    --------
    pd.DataFrame
        Adjusted expression data (same format as input)
    """
    # Create sample IDs and handle duplicates
    expr_df = expr_df.copy()
    expr_df = expr_df.drop_duplicates(subset=['FID', 'IID'], keep='first')
    expr_df['sample_id'] = expr_df['FID'].astype(str) + '_' + expr_df['IID'].astype(str)
    
    # Check for available covariates BEFORE trying to access them
    available_covs = [c for c in covariates if c in cov_df.columns]
    missing_covs = [c for c in covariates if c not in cov_df.columns]
    
    if missing_covs:
        print(f"    ??  Missing covariates in file: {missing_covs}")
    
    if not available_covs:
        print("    ??  No valid covariates found!")
        return None
    
    print(f"    Using covariates: {available_covs}")
    
    # Find common samples (preserving expression data order)
    common_mask = expr_df['sample_id'].isin(cov_df.index)
    
    if common_mask.sum() == 0:
        print("    ??  No common samples found between expression and covariate files!")
        return None
    
    if common_mask.sum() < len(expr_df):
        print(f"    ??  {len(expr_df) - common_mask.sum()} samples not found in covariate file, excluding them")
    
    # Filter expression to common samples
    expr_filtered = expr_df.loc[common_mask].copy()
    expr_filtered = expr_filtered.set_index('sample_id')
    
    # Get covariates for these samples (aligned by index)
    cov_filtered = cov_df.loc[expr_filtered.index, available_covs].copy()
    
    # Encode Sex column robustly
    cov_filtered = encode_sex_column(cov_filtered)
    
    # Handle missing values in covariates
    if cov_filtered.isnull().any().any():
        n_missing = cov_filtered.isnull().sum().sum()
        print(f"    ??  Found {n_missing} missing values in covariates, filling with column means")
        cov_filtered = cov_filtered.fillna(cov_filtered.mean())
    
    # Get gene columns (exclude FID, IID, sample_id)
    gene_cols = [col for col in expr_filtered.columns if col not in ['FID', 'IID', 'sample_id']]
    
    if len(gene_cols) == 0:
        print("    ??  No gene columns found!")
        return None
    
    # Prepare matrices
    X = cov_filtered.values.astype(np.float64)
    Y = expr_filtered[gene_cols].values.astype(np.float64)
    
    # Standardize covariates for numerical stability
    # (improves pseudoinverse computation, especially with mixed scales)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    
    # Handle constant columns (std=0)
    constant_cols = np.where(X_std == 0)[0]
    if len(constant_cols) > 0:
        print(f"    ??  Removing {len(constant_cols)} constant covariate columns")
        X = np.delete(X, constant_cols, axis=1)
        X_mean = np.delete(X_mean, constant_cols)
        X_std = np.delete(X_std, constant_cols)
    
    # Standardize: (X - mean) / std
    X_std[X_std == 0] = 1.0  # Safety check (shouldn't happen after removing constant cols)
    X = (X - X_mean) / X_std
    
    # Residualize using vectorized linear algebra
    Y_adjusted = residualize_vectorized(Y, X, add_back_mean=add_back_mean)
    
    # Create output DataFrame
    result_df = pd.DataFrame(Y_adjusted, columns=gene_cols, index=expr_filtered.index)
    result_df = result_df.reset_index()
    
    # Split sample_id back to FID and IID
    result_df[['FID', 'IID']] = result_df['sample_id'].str.split('_', n=1, expand=True)
    
    # Try to preserve original dtypes
    try:
        result_df['FID'] = result_df['FID'].astype(expr_df['FID'].dtype)
        result_df['IID'] = result_df['IID'].astype(expr_df['IID'].dtype)
    except:
        pass  # Keep as string if conversion fails
    
    # Reorder columns to match original format
    result_df = result_df[['FID', 'IID'] + gene_cols]
    
    print(f"    ? Adjusted {len(gene_cols)} genes for {len(result_df)} samples")
    
    return result_df


def get_output_filename(input_path):
    """
    Generate output filename by adding '_fixed' before extension.
    
    Example: expression.csv -> expression_fixed.csv
    """
    path = Path(input_path)
    return str(path.parent / f"{path.stem}_fixed{path.suffix}")


def process_single_file(task):
    """
    Process a single expression file (for parallel execution).
    
    Parameters:
    -----------
    task : dict
        Dictionary with 'expr_path', 'cov_path', 'covariates', 'add_back_mean'
        
    Returns:
    --------
    dict
        Result with 'success', 'output_path', 'message', 'task_info'
    """
    expr_path = task['expr_path']
    cov_path = task['cov_path']
    covariates = task['covariates']
    add_back_mean = task.get('add_back_mean', True)
    
    result = {
        'success': False,
        'output_path': None,
        'message': '',
        'task_info': task.get('info', {})
    }
    
    try:
        # Load expression data
        expr_df = load_expression_file(expr_path)
        if expr_df is None:
            result['message'] = "Failed to load expression file"
            return result
        
        # Load covariate data
        cov_df = load_covariate_file(cov_path)
        if cov_df is None:
            result['message'] = "Failed to load covariate file"
            return result
        
        # Adjust expression
        adjusted_df = adjust_expression_for_covariates(
            expr_df, cov_df, covariates, add_back_mean=add_back_mean
        )
        if adjusted_df is None:
            result['message'] = "Failed to adjust expression"
            return result
        
        # Generate output path
        output_path = get_output_filename(expr_path)
        
        # Save adjusted expression
        adjusted_df.to_csv(output_path, index=False)
        
        result['success'] = True
        result['output_path'] = output_path
        result['message'] = f"Saved {adjusted_df.shape[0]} samples, {adjusted_df.shape[1]-2} genes"
        
    except Exception as e:
        result['message'] = f"Error: {str(e)}"
    
    return result


# ============================================================================
# MAIN PROCESSING FUNCTIONS
# ============================================================================

def get_all_folds(phenotype):
    """Get all available folds for a phenotype."""
    folds = []
    if not os.path.exists(phenotype):
        return folds
    
    for item in os.listdir(phenotype):
        if os.path.isdir(os.path.join(phenotype, item)) and item.startswith('Fold_'):
            try:
                fold_num = int(item.split('_')[1])
                folds.append(fold_num)
            except:
                continue
    
    return sorted(folds)


def get_tissues_for_model(phenotype, fold, model, split_type):
    """Get all tissues available for a model/split combination."""
    fold_dir = f"{phenotype}/Fold_{fold}"
    dir_name = MODEL_DIRS[model][split_type]
    expr_path = os.path.join(fold_dir, dir_name)
    
    if not os.path.exists(expr_path):
        return []
    
    tissues = []
    for item in os.listdir(expr_path):
        if os.path.isdir(os.path.join(expr_path, item)):
            tissues.append(item)
    
    return sorted(tissues)


def find_expression_file(phenotype, fold, model, split_type, tissue):
    """Find the expression file for a given model/tissue/split combination."""
    fold_dir = f"{phenotype}/Fold_{fold}"
    dir_name = MODEL_DIRS[model][split_type]
    tissue_dir = os.path.join(fold_dir, dir_name, tissue)
    
    if not os.path.exists(tissue_dir):
        return None
    
    # FUSION has specific file names
    if model == 'FUSION':
        file_name = FILE_NAMES['FUSION'][split_type]
        file_path = os.path.join(tissue_dir, file_name)
        if os.path.exists(file_path):
            return file_path
        return None
    
    # Other models: find CSV file (exclude already processed _fixed files)
    csv_files = [f for f in os.listdir(tissue_dir) if f.endswith('.csv') and '_fixed' not in f]
    if csv_files:
        return os.path.join(tissue_dir, csv_files[0])
    
    return None


def process_phenotype(phenotype, n_workers=1, dry_run=False, covariates=None, add_back_mean=True):
    """
    Process all expression files for a phenotype.
    
    Parameters:
    -----------
    phenotype : str
        Phenotype directory name
    n_workers : int
        Number of parallel workers (default: 1 for sequential processing)
    dry_run : bool
        If True, only list files without processing
    covariates : list
        List of covariates to use (default: Sex + PC1-PC10)
    add_back_mean : bool
        If True, add back gene means to residuals
    """
    if covariates is None:
        covariates = COVARIATES_TO_USE
    
    print("\n" + "="*80)
    print("GENE EXPRESSION COVARIATE ADJUSTMENT (Vectorized)")
    print("="*80)
    print(f"\nPhenotype: {phenotype}")
    print(f"Covariates: {', '.join(covariates)}")
    print(f"Add back mean: {add_back_mean}")
    print(f"Workers: {n_workers}")
    
    # Get all folds
    folds = get_all_folds(phenotype)
    if not folds:
        print(f"\n? Error: No folds found for phenotype '{phenotype}'")
        return
    
    print(f"Folds found: {folds}")
    
    # Collect all tasks
    tasks = []
    
    for fold in folds:
        fold_dir = f"{phenotype}/Fold_{fold}"
        print(f"\n{'-'*40}")
        print(f"Scanning Fold {fold}...")
        
        for split_type in ['train', 'validation', 'test']:
            # Check for covariate file
            cov_file = os.path.join(fold_dir, COVARIATE_FILES[split_type])
            if not os.path.exists(cov_file):
                print(f"  ??  Covariate file not found: {cov_file}")
                continue
            
            for model in MODEL_DIRS.keys():
                tissues = get_tissues_for_model(phenotype, fold, model, split_type)
                
                for tissue in tissues:
                    expr_file = find_expression_file(phenotype, fold, model, split_type, tissue)
                    
                    if expr_file and os.path.exists(expr_file):
                        output_file = get_output_filename(expr_file)
                        
                        # Skip if already processed
                        if os.path.exists(output_file):
                            continue
                        
                        tasks.append({
                            'expr_path': expr_file,
                            'cov_path': cov_file,
                            'covariates': covariates,
                            'add_back_mean': add_back_mean,
                            'info': {
                                'fold': fold,
                                'model': model,
                                'split': split_type,
                                'tissue': tissue
                            }
                        })
    
    print(f"\n{'='*80}")
    print(f"Found {len(tasks)} expression files to process")
    print(f"{'='*80}")
    
    if dry_run:
        print("\n[DRY RUN] Files that would be processed:")
        for task in tasks[:20]:  # Show first 20
            info = task['info']
            print(f"  - Fold {info['fold']}/{info['model']}/{info['split']}/{info['tissue']}")
        if len(tasks) > 20:
            print(f"  ... and {len(tasks) - 20} more")
        return
    
    if not tasks:
        print("\n? All files have already been processed!")
        return
    
    # Process tasks
    success_count = 0
    fail_count = 0
    
    print("\nProcessing files...")
    
    if n_workers > 1:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_single_file, task): task for task in tasks}
            
            for i, future in enumerate(as_completed(futures), 1):
                task = futures[future]
                info = task['info']
                
                try:
                    result = future.result()
                    
                    print(f"\n[{i}/{len(tasks)}] Fold {info['fold']} | {info['model']} | {info['split']} | {info['tissue']}")
                    
                    if result['success']:
                        print(f"  ? {result['message']}")
                        success_count += 1
                    else:
                        print(f"  ? {result['message']}")
                        fail_count += 1
                        
                except Exception as e:
                    print(f"\n[{i}/{len(tasks)}] Fold {info['fold']} | {info['model']} | {info['split']} | {info['tissue']}")
                    print(f"  ? Exception: {e}")
                    fail_count += 1
    else:
        # Sequential processing (shows more detailed output)
        for i, task in enumerate(tasks, 1):
            info = task['info']
            print(f"\n[{i}/{len(tasks)}] Fold {info['fold']} | {info['model']} | {info['split']} | {info['tissue']}")
            print(f"  Input:  {task['expr_path']}")
            
            result = process_single_file(task)
            
            if result['success']:
                print(f"  Output: {result['output_path']}")
                print(f"  {result['message']}")
                success_count += 1
            else:
                print(f"  ? {result['message']}")
                fail_count += 1
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"  ? Successfully processed: {success_count}")
    print(f"  ? Failed: {fail_count}")
    print(f"  Total: {len(tasks)}")
    print("="*80 + "\n")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Adjust gene expression for covariates (Sex + PCs)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 adjust_expression_covariates.py BMI
    python3 adjust_expression_covariates.py BMI --dry-run
    python3 adjust_expression_covariates.py BMI --workers 4
    python3 adjust_expression_covariates.py BMI --no-add-mean
    python3 adjust_expression_covariates.py BMI --covariates Sex PC1 PC2 PC3 PC4 PC5
        """
    )
    
    parser.add_argument('phenotype', 
                        help='Phenotype directory name (e.g., BMI)')
    
    parser.add_argument('--dry-run', '-d', 
                        action='store_true',
                        help='List files without processing')
    
    parser.add_argument('--workers', '-w', 
                        type=int, 
                        default=1,
                        help='Number of parallel workers (default: 1)')
    
    parser.add_argument('--covariates', '-c',
                        nargs='+',
                        default=COVARIATES_TO_USE,
                        help='Covariates to regress out (default: Sex PC1-PC10)')
    
    parser.add_argument('--no-add-mean',
                        action='store_true',
                        help='Do not add back mean to residuals (default: add back mean)')
    
    args = parser.parse_args()
    
    # Run processing
    process_phenotype(
        args.phenotype,
        n_workers=args.workers,
        dry_run=args.dry_run,
        covariates=args.covariates,
        add_back_mean=not args.no_add_mean
    )


if __name__ == '__main__':
    main()