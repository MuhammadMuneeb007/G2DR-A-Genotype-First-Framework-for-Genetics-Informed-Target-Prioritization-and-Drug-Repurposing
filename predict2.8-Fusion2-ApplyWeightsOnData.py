#!/usr/bin/env python3

import os
import subprocess
import glob
import sys
from multiprocessing import Pool
from tqdm import tqdm
from pathlib import Path

# Check command line arguments
if len(sys.argv) != 3:
    print("Usage: python script.py <phenotype> <fold>")
    print("Example: python script.py migraine 0")
    sys.exit(1)

PHENOTYPE = sys.argv[1]  # e.g., "migraine"
FOLD = sys.argv[2]       # e.g., "0"

# Configuration
WEIGHTS_DIR = "/data/ascher02/uqmmune1/ANNOVAR/fusion_twas/models/GTExv8/weights/WEIGHTS"
BASE_DIR = f"/data/ascher02/uqmmune1/ANNOVAR/{PHENOTYPE}/Fold_{FOLD}"
OUTPUT_BASE = f"/data/ascher02/uqmmune1/ANNOVAR/{PHENOTYPE}/Fold_{FOLD}/Fussion"
DATA_TYPES = ["train_data", "validation_data", "test_data"]
N_CORES = 20

def get_all_tissues():
    """Find all tissue directories"""
    tissues = []
    for item in os.listdir(WEIGHTS_DIR):
        item_path = os.path.join(WEIGHTS_DIR, item)
        # Check if it's a directory and not a .hsq or .pos file
        if os.path.isdir(item_path) and not item.endswith('.hsq') and not item.endswith('.pos'):
            tissues.append(item)
    return sorted(tissues)

def process_gene_scoring(args):
    """Use existing score file to run PLINK scoring"""
    score_file, tissue_name, data_type = args
    
    try:
        # Set up genotype file path for this data type
        genotype_file = f"{BASE_DIR}/{data_type}"
        
        # Extract gene name from score file
        gene_name = os.path.basename(score_file).replace('.score', '')
        gene_clean = gene_name.replace(f'{tissue_name}.', '')  # Remove tissue prefix
        
        # Create output directory
        output_dir = f"{OUTPUT_BASE}/{tissue_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if final output already exists - SKIP if it does
        final_output = f"{output_dir}/{gene_clean}_{data_type}.score"
        if Path(final_output).is_file():
            return "skipped"
        
        # Set up PLINK scoring with more permissive settings
        temp_prefix = f"{output_dir}/{gene_clean}_{data_type}_temp"
        
        cmd = f"""plink --bfile {genotype_file} \
                  --score {score_file} 1 2 4 \
                  --out {temp_prefix} \
                  --allow-no-sex --silent \
                  --score-no-mean-imputation \
                  --allow-extra-chr"""
        
        # Run without capturing output - much faster!
        result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # PLINK will continue even with missing SNPs, so check for profile file
        profile_file = f"{temp_prefix}.profile"
        if os.path.exists(profile_file):
            # Check if we have reasonable number of SNPs
            with open(profile_file, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:  # Header + at least one data line
                    # Read first data line to check CNT2 (number of valid SNPs)
                    first_data = lines[1].split()
                    if len(first_data) >= 5:
                        valid_snps = int(first_data[4])  # CNT2 column
                        
                        # Only keep if we have at least 1 valid SNP
                        if valid_snps > 0:
                            os.rename(profile_file, final_output)
                            # Clean up other PLINK files
                            for ext in ['.log', '.nosex']:
                                temp_file = f"{temp_prefix}{ext}"
                                if os.path.exists(temp_file):
                                    os.remove(temp_file)
                            return final_output
            
            # If we reach here, gene had no valid SNPs, clean up
            os.remove(profile_file)
            for ext in ['.log', '.nosex']:
                temp_file = f"{temp_prefix}{ext}"
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            return "no_snps"
        else:
            return None
        
    except Exception as e:
        return None

def process_tissue_data_combination(tissue_name, data_type):
    """Process one tissue for one data type"""
    
    # Set up genotype file path
    genotype_file = f"{BASE_DIR}/{data_type}"
    
    # Check if genotype files exist
    genotype_bed = f"{genotype_file}.bed"
    if not os.path.exists(genotype_bed):
        tqdm.write(f"  ⚠️  Genotype file not found: {genotype_bed}")
        return 0, 0, 0, 0
    
    # Find all existing score files for the tissue
    tissue_dir = f"{WEIGHTS_DIR}/{tissue_name}"
    score_files = glob.glob(f"{tissue_dir}/*.score")
    
    if not score_files:
        tqdm.write(f"  ⚠️  No score files found in {tissue_dir}")
        return 0, 0, 0, 0
    
    # Pre-check which files already exist (fast batch check)
    output_dir = f"{OUTPUT_BASE}/{tissue_name}"
    existing_count = 0
    files_to_process = []
    
    for score_file in score_files:
        gene_name = os.path.basename(score_file).replace('.score', '')
        gene_clean = gene_name.replace(f'{tissue_name}.', '')
        final_output = f"{output_dir}/{gene_clean}_{data_type}.score"
        
        if Path(final_output).is_file():
            existing_count += 1
        else:
            files_to_process.append(score_file)
    
    if existing_count > 0:
        tqdm.write(f"  📋 {tissue_name} {data_type}: {existing_count} files already exist, {len(files_to_process)} to process")
    
    if not files_to_process:
        return 0, 0, 0, existing_count
    
    # Create argument tuples for parallel processing (only for files that need processing)
    args_list = [(score_file, tissue_name, data_type) for score_file in files_to_process]
    
    # Process genes in parallel with progress tracking
    print(f"    🚀 Starting parallel processing of {len(files_to_process)} files...")
    
    with Pool(N_CORES) as pool:
        # Use imap with chunk size for better progress updates
        results = []
        for i, result in enumerate(pool.imap(process_gene_scoring, args_list, chunksize=10)):
            results.append(result)
            # Print progress every 100 files
            if (i + 1) % 100 == 0:
                print(f"    📊 Processed {i + 1}/{len(files_to_process)} files...")
        
        print(f"    ✅ Completed processing {len(files_to_process)} files")
    
    # Count results
    successful = sum(1 for r in results if r is not None and r != "no_snps" and r != "skipped")
    no_snps = sum(1 for r in results if r == "no_snps")
    failed = sum(1 for r in results if r is None)
    skipped = sum(1 for r in results if r == "skipped")
    
    return successful, no_snps, failed, existing_count + skipped

def process_tissue_scoring(tissue_name):
    """Process scoring for one tissue across all data types"""
    
    # Find all existing score files for the tissue
    tissue_dir = f"{WEIGHTS_DIR}/{tissue_name}"
    score_files = glob.glob(f"{tissue_dir}/*.score")
    
    if not score_files:
        tqdm.write(f"  ⚠️  No score files found in {tissue_dir}")
        return 0, 0, 0, 0
    
    # Initialize counters for this tissue
    tissue_successful = 0
    tissue_no_snps = 0
    tissue_failed = 0
    tissue_skipped = 0
    
    # Process each data type for this tissue
    for data_type in tqdm(DATA_TYPES, desc=f"{tissue_name} data types", 
                         unit="dtype", leave=False):
        
        successful, no_snps, failed, skipped = process_tissue_data_combination(tissue_name, data_type)
        
        tissue_successful += successful
        tissue_no_snps += no_snps
        tissue_failed += failed
        tissue_skipped += skipped
        
        # Write progress for this combination
        if successful > 0 or no_snps > 0 or failed > 0 or skipped > 0:
            tqdm.write(f"  {tissue_name} {data_type}: {successful} success, {no_snps} no_snps, {failed} failed, {skipped} skipped")
    
    return tissue_successful, tissue_no_snps, tissue_failed, tissue_skipped

def main():
    """Main function"""
    print(f"Processing phenotype: {PHENOTYPE}")
    print(f"Processing fold: {FOLD}")
    print(f"Using {N_CORES} parallel processes")
    print(f"Base directory: {BASE_DIR}")
    print(f"Output directory: {OUTPUT_BASE}")
    
    # Get all tissues
    tissues = get_all_tissues()
    print(f"Found {len(tissues)} tissues to process")
    print(f"Data types: {', '.join(DATA_TYPES)}")
    
    # Create base output directory
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    
    # Check if base directory exists
    if not os.path.exists(BASE_DIR):
        print(f"Error: Base directory not found: {BASE_DIR}")
        print(f"Please check the phenotype and fold arguments.")
        sys.exit(1)
    
    # Initialize summary statistics
    total_successful = 0
    total_no_snps = 0
    total_failed = 0
    total_skipped = 0
    
    # Process each tissue with progress bar
    for tissue in tqdm(tissues, desc="Processing tissues", unit="tissue"):
        
        successful, no_snps, failed, skipped = process_tissue_scoring(tissue)
        
        # Update totals
        total_successful += successful
        total_no_snps += no_snps
        total_failed += failed
        total_skipped += skipped
        
        # Summary for this tissue
        if successful > 0:
            tqdm.write(f"✓ {tissue}: {successful} total successful genes")
    
    # Final summary
    print(f"\n🎉 Processing complete!")
    print(f"Phenotype: {PHENOTYPE}, Fold: {FOLD}")
    print(f"\nOverall Statistics:")
    print(f"  Successfully scored: {total_successful} gene-data combinations")
    print(f"  No valid SNPs: {total_no_snps} combinations")
    print(f"  Failed: {total_failed} combinations")
    print(f"  Skipped (already exist): {total_skipped} combinations")
    print(f"  Tissues processed: {len(tissues)}")
    print(f"  Data types per tissue: {len(DATA_TYPES)}")
    
    print(f"\nOutput structure:")
    print(f"  Location: {OUTPUT_BASE}/{{tissue}}/")
    print(f"  Files: {{gene}}_{{data_type}}.score")
    print(f"  Data types: train_data, validation_data, test_data")
    print(f"  Content: FID, IID, PHENO, CNT, CNT2, SCORE (predicted expression)")

if __name__ == "__main__":
    main()