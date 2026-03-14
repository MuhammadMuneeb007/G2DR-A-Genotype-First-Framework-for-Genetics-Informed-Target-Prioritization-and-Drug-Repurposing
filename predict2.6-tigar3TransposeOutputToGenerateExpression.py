#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TIGAR Expression Matrix Processor
Processes TIGAR gene expression prediction output files to create transposed matrices
with samples as rows and genes as columns, properly ordered according to FAM files.

Usage: python3 tigar_matrix_processor.py <phenotype> <fold>
Example: python3 tigar_matrix_processor.py migraine 0
"""

import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import glob
import warnings
warnings.filterwarnings('ignore')

# ----------- CONFIGURATION -----------
if len(sys.argv) != 3:
    print("Usage: python3 tigar_matrix_processor.py <phenotype> <fold>")
    print("Example: python3 tigar_matrix_processor.py migraine 0")
    sys.exit(1)

Phenotype = sys.argv[1]
Fold = sys.argv[2]
Path = Phenotype + "/Fold_" + str(Fold) + "/"

# Define data types and their directories
datasets = {
    'train': {
        'expression_dir': Path + "TigarTrainExpression/",
        'fam_file': Path + "train_data.fam",
        'output_prefix': "train_expression"
    },
    'validation': {
        'expression_dir': Path + "TigarValidExpression/",
        'fam_file': Path + "validation_data.fam", 
        'output_prefix': "validation_expression"
    },
    'test': {
        'expression_dir': Path + "TigarTestExpression/",
        'fam_file': Path + "test_data.fam",
        'output_prefix': "test_expression"
    }
}

def load_fam_file(fam_file):
    """Load FAM file and return sample order"""
    if not os.path.exists(fam_file):
        print(f"[ERROR] FAM file not found: {fam_file}")
        return None
    
    # FAM format: FID IID PID MID SEX PHENOTYPE
    fam_df = pd.read_csv(fam_file, sep='\s+', header=None, 
                        names=['FID', 'IID', 'PID', 'MID', 'SEX', 'PHENOTYPE'])
    
    print(f"[INFO] Loaded {len(fam_df)} samples from FAM file")
    return fam_df[['FID', 'IID']]

def get_available_tissues(expression_dir):
    """Get list of available tissue directories"""
    if not os.path.exists(expression_dir):
        print(f"[ERROR] Expression directory not found: {expression_dir}")
        return []
    
    tissues = []
    for item in os.listdir(expression_dir):
        tissue_path = os.path.join(expression_dir, item)
        if os.path.isdir(tissue_path):
            tissues.append(item)
    
    tissues.sort()
    print(f"[INFO] Found {len(tissues)} tissues")
    return tissues

def get_chromosome_files(tissue_dir):
    """Find all chromosome prediction files in tissue directory"""
    chr_files = {}
    
    # Look for CHR*_Pred_GReX.txt files
    pattern = os.path.join(tissue_dir, "*", "CHR*_Pred_GReX.txt")
    files = glob.glob(pattern)
    
    for file_path in files:
        # Extract chromosome number from filename
        filename = os.path.basename(file_path)
        if filename.startswith("CHR") and "_Pred_GReX.txt" in filename:
            chr_num = filename.split("_")[0].replace("CHR", "")
            try:
                chr_int = int(chr_num)
                chr_files[chr_int] = file_path
            except ValueError:
                continue
    
    # Sort by chromosome number
    sorted_chromosomes = sorted(chr_files.keys())
    sorted_files = {chr_num: chr_files[chr_num] for chr_num in sorted_chromosomes}
    
    return sorted_files

def load_and_transpose_chromosome(chr_file, chromosome):
    """Load chromosome expression file and transpose it"""
    try:
        # Load the file with proper handling of scientific notation
        try:
            # Read with string type for gene columns to avoid type inference issues
            df = pd.read_csv(chr_file, sep='\t', low_memory=False, 
                           dtype={'TargetID': str, 'GeneName': str})
        except Exception as e:
            # Fallback to basic reading if dtype specification fails
            try:
                df = pd.read_csv(chr_file, sep='\t', low_memory=False)
            except Exception as e2:
                print(f"[ERROR] Failed to read file {chr_file}: {e2}")
                return None, None
        
        # Check if we have the expected columns
        required_cols = ['CHROM', 'GeneStart', 'GeneEnd', 'TargetID', 'GeneName']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"[ERROR] Missing required columns: {missing_cols}")
            return None, None
        
        # Get sample columns (everything after GeneName)
        gene_info_cols = ['CHROM', 'GeneStart', 'GeneEnd', 'TargetID', 'GeneName']
        sample_cols = [col for col in df.columns if col not in gene_info_cols]
        
        if len(sample_cols) == 0:
            print(f"[ERROR] No sample columns found in chromosome {chromosome}")
            return None, None
        
        # Create gene identifiers (use TargetID as primary, GeneName as backup)
        gene_ids = []
        for idx, row in df.iterrows():
            try:
                # Get TargetID and convert to string safely
                gene_id = row['TargetID']
                
                # Handle various data types including boolean
                if pd.isna(gene_id) or gene_id is None:
                    gene_id = ""
                elif isinstance(gene_id, bool):
                    gene_id = str(gene_id)  # Convert True/False to "True"/"False"
                else:
                    gene_id = str(gene_id)
                
                gene_id = gene_id.strip()
                
                # If TargetID is empty/invalid, try GeneName
                if gene_id == '' or gene_id.lower() in ['nan', 'none', 'true', 'false']:
                    gene_name = row['GeneName']
                    
                    # Same safe conversion for GeneName
                    if pd.isna(gene_name) or gene_name is None:
                        gene_id = f"UNKNOWN_GENE_{idx}"
                    elif isinstance(gene_name, bool):
                        gene_id = f"BOOL_GENE_{str(gene_name)}_{idx}"
                    else:
                        gene_id = str(gene_name).strip()
                        if gene_id == '' or gene_id.lower() in ['nan', 'none']:
                            gene_id = f"UNKNOWN_GENE_{idx}"
                
                # Final safety check
                if not isinstance(gene_id, str):
                    gene_id = f"ERROR_GENE_{idx}"
                
                # Add chromosome prefix to avoid duplicates across chromosomes
                gene_ids.append(f"chr{chromosome}_{gene_id}")
                
            except Exception as e:
                print(f"[WARNING] Error processing gene at row {idx}: {e}")
                # Use a fallback identifier
                gene_ids.append(f"chr{chromosome}_ERROR_GENE_{idx}")
        
        # Extract expression matrix (samples x genes)
        expression_matrix = df[sample_cols].T  # Transpose so samples are rows
        expression_matrix.columns = gene_ids
        
        # Convert sample IDs to FID_IID format if needed
        sample_ids = []
        for sample_id in sample_cols:
            # Ensure sample_id is a string
            sample_id = str(sample_id).strip()
            
            # Sample IDs might be in format like "1007725_1007725" 
            # We'll assume they're already in FID_IID format or convert them
            if '_' in sample_id and sample_id.count('_') == 1:
                # Already in FID_IID format
                sample_ids.append(sample_id)
            else:
                # Use sample_id as both FID and IID
                sample_ids.append(f"{sample_id}_{sample_id}")
        
        expression_matrix.index = sample_ids
        
        return expression_matrix, gene_ids
        
    except Exception as e:
        print(f"[ERROR] Failed to process chromosome {chromosome}: {e}")
        return None, None

def merge_chromosomes(chromosome_matrices, tissue_name):
    """Merge expression matrices from different chromosomes"""
    if not chromosome_matrices:
        print(f"[ERROR] No chromosome matrices to merge for tissue {tissue_name}")
        return None
    
    # Start with the first chromosome
    merged_matrix = chromosome_matrices[0].copy()
    
    # Merge with subsequent chromosomes
    for i in range(1, len(chromosome_matrices)):
        chr_matrix = chromosome_matrices[i]
        
        # Check if sample IDs match
        if not merged_matrix.index.equals(chr_matrix.index):
            # Use inner join to keep only common samples
            merged_matrix = merged_matrix.join(chr_matrix, how='inner')
        else:
            # Concatenate along columns (genes)
            merged_matrix = pd.concat([merged_matrix, chr_matrix], axis=1)
    
    return merged_matrix

def reorder_samples(expression_matrix, fam_df):
    fam_ids = fam_df['FID'].astype(str) + '_' + fam_df['IID'].astype(str)
    
    expr_samples = set(expression_matrix.index)
    fam_samples = set(fam_ids)
    common_samples = expr_samples.intersection(fam_samples)

    if not common_samples:
        print("[ERROR] No common samples between expression matrix and FAM file!")
        return None

    ordered_samples = [sid for sid in fam_ids if sid in common_samples]
    reordered_matrix = expression_matrix.loc[ordered_samples]

    # Corrected fam_subset selection
    fam_subset = fam_df[fam_ids.isin(common_samples)]

    # Build final matrix
    final_matrix = pd.DataFrame({
        'FID': fam_subset['FID'].values,
        'IID': fam_subset['IID'].values
    })
    for col in reordered_matrix.columns:
        final_matrix[col] = reordered_matrix[col].values

    return final_matrix

def save_expression_matrix(matrix, tissue_dir, dataset_prefix):
    """Save expression matrix to CSV file in the tissue directory"""
    # Create output file path in the tissue directory
    output_file = os.path.join(tissue_dir, f"{dataset_prefix}_expression_matrix.csv")
    
    # Save as CSV file
    try:
        matrix.to_csv(output_file, index=False)
        return output_file
    except Exception as e:
        print(f"[ERROR] Failed to save matrix to {output_file}: {e}")
        return None

def harmonize_genes_across_datasets(tissue_name, base_expression_dir):
    """
    Read all dataset files for a tissue and ensure they have the same genes.
    Keep only common genes across all datasets and save updated files.
    """
    print(f"[INFO] Harmonizing genes across datasets for tissue {tissue_name}...")
    
    # Define the dataset files to check
    dataset_files = {}
    for dataset_name, dataset_info in datasets.items():
        tissue_dir = os.path.join(dataset_info['expression_dir'], tissue_name)
        csv_file = os.path.join(tissue_dir, f"{dataset_info['output_prefix']}_expression_matrix.csv")
        
        if os.path.exists(csv_file):
            dataset_files[dataset_name] = csv_file
    
    if len(dataset_files) < 2:
        print(f"[WARNING] Less than 2 datasets found for tissue {tissue_name}. Skipping harmonization.")
        return dataset_files
    
    # Read all datasets and get their gene columns
    datasets_data = {}
    gene_sets = {}
    
    for dataset_name, file_path in dataset_files.items():
        try:
            df = pd.read_csv(file_path)
            datasets_data[dataset_name] = df
            
            # Get gene columns (exclude FID and IID)
            gene_columns = [col for col in df.columns if col not in ['FID', 'IID']]
            gene_sets[dataset_name] = set(gene_columns)
            
            print(f"[INFO] {dataset_name}: {len(df)} samples, {len(gene_columns)} genes")
            
        except Exception as e:
            print(f"[ERROR] Failed to read {file_path}: {e}")
            return {}
    
    # Find common genes across all datasets
    if gene_sets:
        common_genes = set.intersection(*gene_sets.values())
        common_genes = sorted(list(common_genes))  # Sort for consistency
        
        print(f"[INFO] Common genes across all datasets: {len(common_genes)}")
        
        if len(common_genes) == 0:
            print(f"[ERROR] No common genes found across datasets for tissue {tissue_name}")
            return {}
        
        # Update each dataset to keep only common genes
        updated_files = {}
        for dataset_name, df in datasets_data.items():
            # Keep FID, IID columns plus common genes
            columns_to_keep = ['FID', 'IID'] + common_genes
            harmonized_df = df[columns_to_keep].copy()
            
            # Save the harmonized dataset
            output_file = dataset_files[dataset_name]
            try:
                harmonized_df.to_csv(output_file, index=False)
                updated_files[dataset_name] = output_file
                
                num_samples = len(harmonized_df)
                num_genes = len(common_genes)
                print(f"[INFO] ✓ Updated {dataset_name}: {num_samples} samples, {num_genes} genes")
                
            except Exception as e:
                print(f"[ERROR] Failed to save harmonized {dataset_name}: {e}")
        
        # Verify all datasets now have the same number of genes
        print(f"[INFO] Harmonization complete for {tissue_name}:")
        for dataset_name in updated_files.keys():
            verify_df = pd.read_csv(updated_files[dataset_name])
            verify_genes = len([col for col in verify_df.columns if col not in ['FID', 'IID']])
            print(f"[INFO]   {dataset_name}: {verify_genes} genes")
        
        return updated_files
    
    return dataset_files

def process_tissue(tissue_name, dataset_info):
    """Process one tissue for one dataset"""
    tissue_dir = os.path.join(dataset_info['expression_dir'], tissue_name)
    
    if not os.path.exists(tissue_dir):
        print(f"[ERROR] Tissue directory not found: {tissue_dir}")
        return None
    
    # Get chromosome files
    chr_files = get_chromosome_files(tissue_dir)
    if not chr_files:
        print(f"[ERROR] No chromosome files found for tissue {tissue_name}")
        return None
    
    # Load and transpose each chromosome
    chromosome_matrices = []
    processed_chromosomes = []
    
    for chr_num in sorted(chr_files.keys()):
        chr_file = chr_files[chr_num]
        expr_matrix, gene_ids = load_and_transpose_chromosome(chr_file, chr_num)
        
        if expr_matrix is not None:
            chromosome_matrices.append(expr_matrix)
            processed_chromosomes.append(chr_num)
    
    if not chromosome_matrices:
        print(f"[ERROR] No chromosomes processed successfully for tissue {tissue_name}")
        return None
    
    # Merge chromosomes
    merged_matrix = merge_chromosomes(chromosome_matrices, tissue_name)
    if merged_matrix is None:
        return None
    
    # Load FAM file and reorder samples
    fam_df = load_fam_file(dataset_info['fam_file'])
    if fam_df is None:
        return None
    
    final_matrix = reorder_samples(merged_matrix, fam_df)
    if final_matrix is None:
        return None
    
    # Save the matrix in the tissue directory
    output_file = save_expression_matrix(final_matrix, tissue_dir, dataset_info['output_prefix'])
    
    return output_file

def process_tissue_across_all_datasets(tissue_name):
    """Process one tissue across all datasets and harmonize genes"""
    print(f"\n[INFO] Processing tissue {tissue_name} across all datasets...")
    print("=" * 60)
    
    tissue_results = {}
    
    # Process each dataset for this tissue
    for dataset_name, dataset_info in datasets.items():
        print(f"[INFO] Processing {dataset_name} dataset for {tissue_name}...")
        
        if not os.path.exists(dataset_info['expression_dir']):
            print(f"[WARNING] Expression directory not found: {dataset_info['expression_dir']}")
            continue
        
        output_file = process_tissue(tissue_name, dataset_info)
        
        if output_file is not None:
            tissue_results[dataset_name] = output_file
            
            # Get matrix info for reporting
            try:
                temp_df = pd.read_csv(output_file)
                num_samples = len(temp_df)
                num_genes = len(temp_df.columns) - 2  # Subtract FID and IID columns
                print(f"[INFO] ✓ {dataset_name}: {num_samples} samples, {num_genes} genes -> {output_file}")
            except:
                print(f"[INFO] ✓ {dataset_name}: Saved to {output_file}")
        else:
            print(f"[ERROR] Failed to process {dataset_name} for tissue {tissue_name}")
    
    # Harmonize genes across all datasets for this tissue
    if len(tissue_results) > 1:
        print(f"\n[INFO] Harmonizing genes across datasets for {tissue_name}...")
        harmonized_results = harmonize_genes_across_datasets(tissue_name, datasets['train']['expression_dir'])
        if harmonized_results:
            tissue_results.update(harmonized_results)
    else:
        print(f"[WARNING] Only {len(tissue_results)} dataset(s) processed for {tissue_name}. Skipping harmonization.")
    
    return tissue_results

def main():
    """Main function to process all datasets and tissues"""
    print("TIGAR EXPRESSION MATRIX PROCESSOR")
    print("=" * 50)
    print(f"Phenotype: {Phenotype}, Fold: {Fold}")
    
    # Check base path
    if not os.path.exists(Path):
        print(f"[ERROR] Base path not found: {Path}")
        sys.exit(1)
    
    # Get all available tissues from train dataset (assuming it has all tissues)
    train_tissues = get_available_tissues(datasets['train']['expression_dir'])
    if not train_tissues:
        print("[ERROR] No tissues found in training dataset!")
        sys.exit(1)
    
    print(f"[INFO] Found {len(train_tissues)} tissues to process")
    
    # Process each tissue across all datasets
    all_results = {}
    
    for i, tissue_name in enumerate(train_tissues, 1):
        print(f"\n{'='*80}")
        print(f"PROCESSING TISSUE {i}/{len(train_tissues)}: {tissue_name}")
        print(f"{'='*80}")
        
        tissue_results = process_tissue_across_all_datasets(tissue_name)
        if tissue_results:
            all_results[tissue_name] = tissue_results
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")
    
    if all_results:
        total_files = sum(len(results) for results in all_results.values())
        total_tissues = len(all_results)
        
        print(f"✓ Successfully processed {total_tissues} tissues")
        print(f"✓ Generated {total_files} harmonized CSV expression matrix files")
        
        # Detailed summary by tissue
        for tissue_name, tissue_results in all_results.items():
            print(f"\n{tissue_name}:")
            for dataset_name, output_file in tissue_results.items():
                # Verify gene counts
                try:
                    df = pd.read_csv(output_file)
                    num_samples = len(df)
                    num_genes = len([col for col in df.columns if col not in ['FID', 'IID']])
                    print(f"  - {dataset_name}: {num_samples} samples, {num_genes} genes")
                except:
                    print(f"  - {dataset_name}: {os.path.basename(output_file)}")
        
        print(f"\nAll CSV files have been harmonized to contain the same genes per tissue")
        print(f"Files saved in respective tissue directories")
        
    else:
        print("[ERROR] No expression matrices were generated!")
        sys.exit(1)
    
    print("\nProcessing completed successfully!")

if __name__ == "__main__":
    main()