#!/usr/bin/env python3

import os
import pandas as pd
import glob
import sys
from tqdm import tqdm

# Check command line arguments
if len(sys.argv) != 3:
    print("Usage: python script.py <phenotype> <fold>")
    print("Example: python script.py migraine 0")
    sys.exit(1)

PHENOTYPE = sys.argv[1]  # e.g., "migraine"
FOLD = sys.argv[2]       # e.g., "0"

# Configuration
INPUT_BASE = f"/data/ascher02/uqmmune1/ANNOVAR/{PHENOTYPE}/Fold_{FOLD}/Fussion"
OUTPUT_BASE = f"/data/ascher02/uqmmune1/ANNOVAR/{PHENOTYPE}/Fold_{FOLD}/FussionExpression"
DATA_TYPES = ["train_data", "validation_data", "test_data"]

def get_available_tissues():
    """Scan the input directory to find all available tissues"""
    if not os.path.exists(INPUT_BASE):
        print(f"❌ Error: Input base directory not found: {INPUT_BASE}")
        return []
    
    tissues = []
    for item in os.listdir(INPUT_BASE):
        tissue_path = os.path.join(INPUT_BASE, item)
        if os.path.isdir(tissue_path):
            # Check if this tissue directory contains score files
            score_files = glob.glob(f"{tissue_path}/*.score")
            if score_files:
                tissues.append(item)
    
    return sorted(tissues)

def read_gene_score_file(score_file):
    """Read a single gene score file and return gene name and data"""
    try:
        # Extract gene name from filename - remove data_type suffix
        basename = os.path.basename(score_file).replace('.score', '')
        # Remove data type suffix (e.g., _train_data, _validation_data, _test_data)
        for data_type in DATA_TYPES:
            if basename.endswith(f'_{data_type}'):
                gene_name = basename.replace(f'_{data_type}', '')
                break
        else:
            gene_name = basename
        
        # Read the score file
        df = pd.read_csv(score_file, delim_whitespace=True)
        
        # Check if file has data
        if len(df) == 0:
            return None, None
        
        # Check for expected columns and handle different PLINK output formats
        required_cols = ['FID', 'IID']
        score_col = None
        
        # Find the score column - PLINK can name it differently
        possible_score_cols = ['SCORE', 'SCORESUM', 'SCORE1_SUM', 'SCORE_SUM']
        for col in possible_score_cols:
            if col in df.columns:
                score_col = col
                break
        
        if score_col is None or not all(col in df.columns for col in required_cols):
            return None, None
            
        # Return gene name and the dataframe with FID, IID, SCORE
        result_df = df[required_cols + [score_col]].copy()
        result_df.rename(columns={score_col: 'SCORE'}, inplace=True)
        
        return gene_name, result_df
        
    except Exception as e:
        return None, None

def merge_gene_expressions_for_data_type(tissue, data_type):
    """Merge all gene expression files for a specific tissue and data type into a matrix"""
    
    # Find all score files for this tissue and data type
    score_dir = f"{INPUT_BASE}/{tissue}"
    score_files = glob.glob(f"{score_dir}/*_{data_type}.score")
    
    if not score_files:
        return None
        
    # Initialize the merged dataframe
    merged_df = None
    successful_genes = 0
    failed_genes = 0
    
    # Process each gene file
    for score_file in score_files:
        gene_name, gene_df = read_gene_score_file(score_file)
        
        if gene_name is None or gene_df is None:
            failed_genes += 1
            continue
            
        if merged_df is None:
            # First gene - initialize with FID, IID columns
            merged_df = gene_df[['FID', 'IID']].copy()
            merged_df[gene_name] = gene_df['SCORE']
        else:
            # Merge subsequent genes
            merged_df = merged_df.merge(
                gene_df[['FID', 'IID', 'SCORE']].rename(columns={'SCORE': gene_name}),
                on=['FID', 'IID'],
                how='outer'
            )
        
        successful_genes += 1
    
    if merged_df is None:
        return None
        
    # Fill missing values with 0 (for genes where individuals had no coverage)
    merged_df = merged_df.fillna(0)
    
    # Sort by FID and IID for consistency
    merged_df = merged_df.sort_values(['FID', 'IID'])
    
    return merged_df, successful_genes, failed_genes

def process_tissue(tissue):
    """Process all data types for a specific tissue"""
    print(f"\n🧪 Processing tissue: {tissue}")
    
    # Create tissue-specific output directory
    tissue_output_dir = f"{OUTPUT_BASE}/{tissue}"
    os.makedirs(tissue_output_dir, exist_ok=True)
    
    tissue_results = {}
    
    # Process each data type for this tissue
    for data_type in DATA_TYPES:
        result = merge_gene_expressions_for_data_type(tissue, data_type)
        
        if result is not None:
            merged_df, successful_genes, failed_genes = result
            
            # Create output filename
            output_file = f"{tissue_output_dir}/GeneExpression_{data_type}.csv"
            
            # Save the merged matrix
            merged_df.to_csv(output_file, index=False)
            
            tissue_results[data_type] = {
                'samples': len(merged_df),
                'genes': successful_genes,
                'failed': failed_genes,
                'success_rate': successful_genes / (successful_genes + failed_genes) * 100 if (successful_genes + failed_genes) > 0 else 0,
                'output_file': output_file
            }
        else:
            tissue_results[data_type] = None
    
    return tissue_results

def main():
    """Main function"""
    print(f"🧬 Creating gene expression matrices for {PHENOTYPE} Fold {FOLD}")
    print(f"📂 Input directory: {INPUT_BASE}")
    print(f"📂 Output directory: {OUTPUT_BASE}")
    
    # Check if input directory exists
    if not os.path.exists(INPUT_BASE):
        print(f"❌ Error: Input directory not found: {INPUT_BASE}")
        print("Make sure you've run the scoring script first!")
        sys.exit(1)
    
    # Get all available tissues
    tissues = get_available_tissues()
    
    if not tissues:
        print(f"❌ Error: No tissues with score files found in {INPUT_BASE}")
        print("Make sure your input directory contains tissue subdirectories with .score files!")
        sys.exit(1)
    
    print(f"🔍 Found {len(tissues)} tissues to process: {', '.join(tissues)}")
    print(f"📋 Data types: {', '.join(DATA_TYPES)}")
    
    # Create main output directory
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    
    # Process each tissue
    all_results = {}
    
    with tqdm(tissues, desc="Processing tissues", unit="tissue") as tissue_pbar:
        for tissue in tissue_pbar:
            tissue_pbar.set_postfix_str(f"Current: {tissue}")
            
            tissue_results = process_tissue(tissue)
            all_results[tissue] = tissue_results
            
            # Show progress for current tissue
            successful_data_types = sum(1 for result in tissue_results.values() if result is not None)
            tissue_pbar.write(f"  ✅ {tissue}: {successful_data_types}/{len(DATA_TYPES)} data types processed")
    
    # Final summary
    print(f"\n🎉 Gene expression matrix creation complete!")
    print(f"\n📊 Final Summary:")
    print(f"{'Tissue':<20} {'Data Type':<15} {'Samples':<8} {'Genes':<8} {'Failed':<8} {'Success Rate':<12}")
    print("-" * 85)
    
    total_files_created = 0
    
    for tissue in tissues:
        tissue_results = all_results[tissue]
        tissue_files = 0
        
        for data_type in DATA_TYPES:
            if tissue_results[data_type] is not None:
                summary = tissue_results[data_type]
                print(f"{tissue:<20} {data_type:<15} {summary['samples']:<8} {summary['genes']:<8} "
                      f"{summary['failed']:<8} {summary['success_rate']:<12.1f}%")
                tissue_files += 1
                total_files_created += 1
            else:
                print(f"{tissue:<20} {data_type:<15} {'No data':<8} {'No data':<8} {'No data':<8} {'No data':<12}")
        
        if tissue_files > 0:
            print("-" * 85)
    
    print(f"\n📁 Output Summary:")
    print(f"  🗂️  Total tissues processed: {len(tissues)}")
    print(f"  📄 Total files created: {total_files_created}")
    print(f"  📂 Output directory structure:")
    
    for tissue in tissues:
        tissue_results = all_results[tissue]
        tissue_files = [result for result in tissue_results.values() if result is not None]
        
        if tissue_files:
            print(f"    📁 {tissue}/")
            for data_type in DATA_TYPES:
                if tissue_results[data_type] is not None:
                    filename = f"GeneExpression_{data_type}.csv"
                    samples = tissue_results[data_type]['samples']
                    genes = tissue_results[data_type]['genes']
                    print(f"      📄 {filename} ({samples} samples × {genes} genes)")
    
    print(f"\n📝 File format:")
    print(f"  - Samples as rows, genes as columns")
    print(f"  - First two columns: FID, IID (sample identifiers)")
    print(f"  - Remaining columns: Gene expression prediction scores")
    print(f"  - Missing values filled with 0")
    
    # Show example of first tissue with data
    for tissue in tissues:
        tissue_results = all_results[tissue]
        first_successful = None
        for data_type in DATA_TYPES:
            if tissue_results[data_type] is not None:
                first_successful = (tissue, data_type)
                break
        if first_successful:
            tissue, data_type = first_successful
            output_file = tissue_results[data_type]['output_file']
            print(f"\n📋 Preview of {tissue} - {data_type} gene expression matrix:")
            try:
                preview_df = pd.read_csv(output_file, nrows=5)
                print(preview_df.iloc[:, :min(7, len(preview_df.columns))].to_string(index=False))
            except:
                print("  (Preview not available)")
            break

if __name__ == "__main__":
    main()