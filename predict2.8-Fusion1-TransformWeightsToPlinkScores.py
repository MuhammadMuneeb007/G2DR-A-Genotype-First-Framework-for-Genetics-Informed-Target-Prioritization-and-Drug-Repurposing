#!/usr/bin/env python3

import os
import subprocess
import glob
from pathlib import Path
from tqdm import tqdm
import sys

def run_command(cmd):
    """Run shell command without hanging - fixed version"""
    try:
        # Don't capture output since we're redirecting to file with >
        result = subprocess.run(cmd, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        return False

def find_all_tissues_and_weights():
    """Find all tissues and their corresponding weight files"""
    base_path = "/data/ascher02/uqmmune1/ANNOVAR/fusion_twas/models/GTExv8/weights/WEIGHTS"
    
    # Find all weight files
    weight_pattern = os.path.join(base_path, "*", "*.wgt.RDat")
    weight_files = glob.glob(weight_pattern)
    
    if not weight_files:
        print(f"No weight files found in {base_path}")
        return {}
    
    # Group weight files by tissue
    tissues_dict = {}
    for weight_file in weight_files:
        tissue_name = os.path.basename(os.path.dirname(weight_file))
        if tissue_name not in tissues_dict:
            tissues_dict[tissue_name] = []
        tissues_dict[tissue_name].append(weight_file)
    
    return tissues_dict

def convert_weights_to_scores_for_tissue(tissue_name, weight_files, base_output_dir):
    """Convert weight files to score format for a specific tissue"""
    
    # Change to correct directory (where manual command worked)
    os.chdir("/data/ascher02/uqmmune1/ANNOVAR")
    
    successful_conversions = 0
    failed_conversions = 0
    skipped_conversions = 0
    
    # Pre-check all files to see which ones already exist (fast batch check)
    files_to_process = []
    for weight_file in weight_files:
        # Create score file in the SAME directory as weight file (not separate directory)
        score_filename = os.path.basename(weight_file).replace('.wgt.RDat', '.score')
        output_score_file = os.path.join(os.path.dirname(weight_file), score_filename)
        
        # Fast existence check - score file in same directory as weight file
        if Path(output_score_file).is_file():
            skipped_conversions += 1
        else:
            gene_id = score_filename.replace('.score', '')
            files_to_process.append((weight_file, output_score_file, gene_id))
    
    print(f"  Found {len(files_to_process)} files to convert, {skipped_conversions} already exist")
    
    # Process only the files that don't exist
    for weight_file, output_score_file, gene_id in tqdm(files_to_process, desc=f"Converting {tissue_name}", leave=False):
        
        # Use the exact path that worked manually
        score_cmd = f"Rscript fusion_twas/utils/make_score.R {weight_file} > {output_score_file}"
        
        # Run the command
        success = run_command(score_cmd)
        
        if success:
            successful_conversions += 1
        else:
            failed_conversions += 1
            print(f"Failed to convert {gene_id} in {tissue_name}")
    
    # Total successful includes both converted and skipped
    total_successful = successful_conversions + skipped_conversions
    return total_successful, failed_conversions

def convert_weights_to_scores():
    """Convert weight files to score format for all tissues"""
    
    # Configuration
    base_output_dir = "/data/ascher02/uqmmune1/ANNOVAR/fusion_twas/score_files"
    
    # Create base output directory
    os.makedirs(base_output_dir, exist_ok=True)
    
    print("Finding all tissues and weight files...")
    tissues_dict = find_all_tissues_and_weights()
    
    if not tissues_dict:
        print("No tissues found. Exiting.")
        return
    
    print(f"Found {len(tissues_dict)} tissues:")
    for tissue, weights in tissues_dict.items():
        print(f"  - {tissue}: {len(weights)} weight files")
    
    # Process each tissue
    total_successful = 0
    total_failed = 0
    
    for tissue_name in tqdm(tissues_dict.keys(), desc="Converting tissues"):
        weight_files = tissues_dict[tissue_name]
        
        print(f"\nConverting tissue: {tissue_name} ({len(weight_files)} genes)")
        
        successful, failed = convert_weights_to_scores_for_tissue(
            tissue_name, weight_files, base_output_dir
        )
        
        total_successful += successful
        total_failed += failed
        
        print(f"Tissue {tissue_name} completed: {successful} successful, {failed} failed")
    
    print(f"\n=== Final Summary ===")
    print(f"Total successful conversions: {total_successful}")
    print(f"Total failed conversions: {total_failed}")
    print(f"Score files saved in: {base_output_dir}")

if __name__ == "__main__":
    convert_weights_to_scores()