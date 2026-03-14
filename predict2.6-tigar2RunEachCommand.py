#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TIGAR Gene Expression Prediction Processor
Uses existing VCF files from TrainVCF/, ValidationVCF/, TestVCF/ directories
to run TIGAR gene expression prediction for all chromosomes

Requirements:
- VCF files already created in TrainVCF/, ValidationVCF/, TestVCF/ directories
- TIGAR installation with weights and gene annotation files
- bcftools for VCF processing

Output Structure:
- Expression results: TigarTrainExpression/, TigarValidationExpression/, TigarTestExpression/

Usage: python3 tigar_expression_processor.py <phenotype> <fold> [tissue] [chromosome]
"""

import os
import subprocess
import sys
import pandas as pd
import gzip
import numpy as np
from tqdm import tqdm
import glob
import shutil

# ----------- CONFIGURATION -----------
if len(sys.argv) < 3 or len(sys.argv) > 5:
    print("Usage: python3 tigar_expression_processor.py <phenotype> <fold> [tissue] [chromosome]")
    print("Example: python3 tigar_expression_processor.py migraine 0")
    print("Example: python3 tigar_expression_processor.py migraine 0 Whole_Blood")
    print("Example: python3 tigar_expression_processor.py migraine 0 Whole_Blood 1")
    print("Example: python3 tigar_expression_processor.py migraine 0 ALL 1-5")
    print("If no tissue is specified, all available tissues will be processed.")
    print("If no chromosome is specified, all chromosomes (1-22) will be processed.")
    print("Chromosome can be: single number (1), range (1-5), or comma-separated (1,3,5)")
    sys.exit(1)

Phenotype = sys.argv[1]
Fold = sys.argv[2]
Specified_Tissue = sys.argv[3] if len(sys.argv) >= 4 and sys.argv[3] != "ALL" else None
Specified_Chromosome = sys.argv[4] if len(sys.argv) == 5 else (sys.argv[3] if len(sys.argv) == 4 and sys.argv[3] not in ["ALL", "Whole_Blood", "Brain", "Liver", "Lung", "Muscle", "Heart"] else None)

# Parse chromosome specification
def parse_chromosome_spec(chrom_spec):
    """Parse chromosome specification into list of integers"""
    if not chrom_spec:
        return list(range(1, 23))  # Default: all autosomes
    
    chromosomes = []
    try:
        # Handle comma-separated list
        if ',' in chrom_spec:
            for part in chrom_spec.split(','):
                part = part.strip()
                if '-' in part:
                    # Handle range
                    start, end = map(int, part.split('-'))
                    chromosomes.extend(range(start, end + 1))
                else:
                    # Handle single chromosome
                    chromosomes.append(int(part))
        elif '-' in chrom_spec:
            # Handle range
            start, end = map(int, chrom_spec.split('-'))
            chromosomes.extend(range(start, end + 1))
        else:
            # Handle single chromosome
            chromosomes.append(int(chrom_spec))
        
        # Validate chromosome numbers
        valid_chromosomes = [c for c in chromosomes if 1 <= c <= 22]
        if len(valid_chromosomes) != len(chromosomes):
            invalid = [c for c in chromosomes if c not in valid_chromosomes]
            print(f"[WARNING] Invalid chromosome(s) ignored: {invalid}")
        
        return sorted(list(set(valid_chromosomes)))
        
    except ValueError as e:
        print(f"[ERROR] Invalid chromosome specification: {chrom_spec}")
        print("[ERROR] Use format like: 1, 1-5, or 1,3,5")
        sys.exit(1)

# Define chromosomes to process
CHROMOSOMES = parse_chromosome_spec(Specified_Chromosome)
Path = Phenotype + "/Fold_" + str(Fold) + "/"

# Define chromosomes to process (1-22 for autosomes)
# CHROMOSOMES = list(range(1, 23))

# Input file prefixes (for FAM files only)
train_prefix = Path + "train_data"
validation_prefix = Path + "validation_data"
test_prefix = Path + "test_data"

# VCF directories (should already exist with VCF files)
TrainVCF_dir = Path + "TrainVCF/"
ValidationVCF_dir = Path + "ValidationVCF/"
TestVCF_dir = Path + "TestVCF/"

# Expression directories
TrainExpression_dir = Path + "TigarTrainExpression/"
ValidationExpression_dir = Path + "TigarValidationExpression/"
TestExpression_dir = Path + "TigarTestExpression/"

# TIGAR configuration
TIGAR_dir = "/data/ascher02/uqmmune1/ANNOVAR/TIGAR"
weights_base_dir = "/data/ascher02/uqmmune1/ANNOVAR/TIGAR/weights/Weights"
gene_anno_file = "/data/ascher02/uqmmune1/ANNOVAR/TIGAR/gene_anno.csv"

def create_directory(directory_path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"[INFO] Created directory: {directory_path}")

def check_prerequisites():
    """Check if all required tools and files are available"""
    print("[INFO] Checking prerequisites...")
    
    # Check bcftools
    try:
        result = subprocess.run(['bcftools', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            print("[ERROR] bcftools not working properly")
            return False
        print(f"[INFO] ✓ bcftools found")
    except FileNotFoundError:
        print("[ERROR] bcftools not found")
        print("[INFO] Install with: conda install -c bioconda bcftools")
        return False
    
    # Check TIGAR directory
    if not os.path.exists(TIGAR_dir):
        print(f"[ERROR] TIGAR directory not found: {TIGAR_dir}")
        return False
    print(f"[INFO] ✓ TIGAR directory found: {TIGAR_dir}")
    
    # Check TIGAR script
    tigar_script = os.path.join(TIGAR_dir, "TIGAR_GReX_Pred.sh")
    if not os.path.exists(tigar_script):
        print(f"[ERROR] TIGAR script not found: {tigar_script}")
        return False
    print(f"[INFO] ✓ TIGAR script found")
    
    # Check gene annotation file
    if not os.path.exists(gene_anno_file):
        print(f"[ERROR] Gene annotation file not found: {gene_anno_file}")
        print("[INFO] You may need to create it: cp ExampleData/gene_anno.txt gene_annotation.txt")
        return False
    print(f"[INFO] ✓ Gene annotation file found")
    
    # Check weights directory
    if not os.path.exists(weights_base_dir):
        print(f"[ERROR] Weights directory not found: {weights_base_dir}")
        return False
    print(f"[INFO] ✓ Weights directory found")
    
    return True

def check_vcf_files():
    """Check if VCF directories and chrAll.vcf.gz files exist"""
    print("[INFO] Checking existing chrAll.vcf.gz files...")
    
    directories = [
        (TrainVCF_dir, "Training"),
        (ValidationVCF_dir, "Validation"), 
        (TestVCF_dir, "Test")
    ]
    
    vcf_status = {}
    
    for vcf_dir, dataset_name in directories:
        if not os.path.exists(vcf_dir):
            print(f"[ERROR] {dataset_name} VCF directory not found: {vcf_dir}")
            vcf_status[dataset_name.lower()] = False
            continue
        
        # Check for chrAll.vcf.gz file
        chrall_vcf_file = os.path.join(vcf_dir, "chrAll.vcf.gz")
        index_file_tbi = chrall_vcf_file + ".tbi"
        
        if os.path.exists(chrall_vcf_file) and os.path.exists(index_file_tbi):
            vcf_status[dataset_name.lower()] = chrall_vcf_file
            print(f"[INFO] ✓ {dataset_name} chrAll.vcf.gz file found: {chrall_vcf_file}")
        else:
            print(f"[ERROR] {dataset_name} chrAll.vcf.gz file or index missing")
            vcf_status[dataset_name.lower()] = False
    
    return vcf_status

def get_available_tissues(chromosome):
    """Get list of available tissues with weights for this chromosome"""
    tissues = []
    
    if not os.path.exists(weights_base_dir):
        print(f"[ERROR] Weights directory not found: {weights_base_dir}")
        return tissues
    
    try:
        for tissue_name in os.listdir(weights_base_dir):
            tissue_path = os.path.join(weights_base_dir, tissue_name)
            if os.path.isdir(tissue_path):
                # Check if this tissue has weights for our chromosome
                weight_file = get_weight_file_path(tissue_name, chromosome)
                if weight_file and os.path.exists(weight_file):
                    tissues.append(tissue_name)
        
        tissues.sort()
        
        # Filter by specified tissue if provided
        if Specified_Tissue:
            if Specified_Tissue in tissues:
                tissues = [Specified_Tissue]
                print(f"[INFO] Processing specified tissue: {Specified_Tissue}")
            else:
                print(f"[ERROR] Specified tissue '{Specified_Tissue}' not found for chr{chromosome}")
                print(f"[INFO] Available tissues: {', '.join(tissues)}")
                return []
        else:
            print(f"[INFO] Found {len(tissues)} tissues with weights for chr{chromosome}")
        
    except Exception as e:
        print(f"[ERROR] Failed to scan tissues: {e}")
    
    return tissues

def get_weight_file_path(tissue_name, chrom):
    """Get path to weight file for tissue and chromosome"""
    weight_dir = os.path.join(weights_base_dir, tissue_name, f"DPR_CHR{chrom}")
    weight_file = os.path.join(weight_dir, f"CHR{chrom}_DPR_train_eQTLweights.txt.gz")
    
    if os.path.exists(weight_file):
        return weight_file
    return None

def find_existing_sample_id_file(vcf_dir, dataset_name):
    """Find existing sample ID file in VCF directory"""
    # Try different possible sample ID file names
    possible_names = [
        f"{dataset_name}_sampleID.txt",
        f"training_sampleID.txt" if dataset_name == "training" else f"{dataset_name}_sampleID.txt",
        "sampleID.txt",
        f"{dataset_name}SampleID.txt"
    ]
    
    for filename in possible_names:
        sample_id_file = os.path.join(vcf_dir, filename)
        if os.path.exists(sample_id_file):
            print(f"[INFO] ✓ Found existing sample ID file: {sample_id_file}")
            return sample_id_file
    
    print(f"[ERROR] No sample ID file found in {vcf_dir}")
    print(f"[INFO] Looked for: {possible_names}")
    return None

def predict_expression_tigar(vcf_file, expression_dir, chrom, dataset_name, tissue_name):
    """
    Run TIGAR gene expression prediction for one chromosome and tissue using VCF
    """
    print(f"\n[INFO] === TIGAR Prediction: {dataset_name} chr{chrom} {tissue_name} ===")
    
    # Get weight file
    weight_file = get_weight_file_path(tissue_name, chrom)
    if not weight_file:
        print(f"[ERROR] Weight file not found for {tissue_name} chr{chrom}")
        return None
    
    # Create tissue output directory
    tissue_output_dir = os.path.join(expression_dir, tissue_name)
    create_directory(tissue_output_dir)
    
    # Find existing sample ID file in VCF directory
    sample_id_file = find_existing_sample_id_file(os.path.dirname(vcf_file), dataset_name)
    if not sample_id_file:
        return None
    
    # Verify VCF index exists
    index_file_tbi = vcf_file + '.tbi'
    if not os.path.exists(index_file_tbi):
        print(f"[ERROR] VCF index file not found for: {vcf_file}")
        return None
    
    try:
        print(f"[INFO] Running TIGAR gene expression prediction...")
        
        # TIGAR command for VCF format
        cmd = [
            "bash",
            os.path.join(TIGAR_dir, "TIGAR_GReX_Pred.sh"),
            "--format", "GT",  # Use GT for genotype format in VCF files
            "--gene_anno", gene_anno_file,
            "--test_sampleID", sample_id_file,
            "--chr", str(chrom),
            "--weight", weight_file,
            "--genofile", vcf_file,
            "--genofile_type", "vcf",  # Specify vcf format
            "--window", "1000000",  # 1Mb window
            "--missing_rate", "0.3",
            "--maf_diff", "0.3",
            "--thread", "10",
            "--out_dir", tissue_output_dir,
            "--TIGAR_dir", TIGAR_dir
        ]
        
        print(f"[INFO] Command: {' '.join(cmd)}")
        
        # Run TIGAR
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        stdout_lines = []
        with tqdm(desc=f"TIGAR {tissue_name[:15]}", unit="step") as pbar:
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    output_str = output.strip()
                    stdout_lines.append(output_str)
                    if output_str:
                        display_str = output_str[:40] + "..." if len(output_str) > 40 else output_str
                        pbar.set_description(f"TIGAR {tissue_name[:15]}: {display_str}")
                    pbar.update(1)
        
        stderr_output = process.stderr.read()
        rc = process.poll()
        
        if rc != 0:
            print(f"[ERROR] TIGAR failed with return code: {rc}")
            if stderr_output:
                print(f"[ERROR] STDERR: {stderr_output}")
            return None
        
        # Check output file
        output_file = os.path.join(tissue_output_dir, f"CHR{chrom}_Pred_GReX.txt")
        
        if os.path.exists(output_file):
            print(f"[INFO] ✓ TIGAR output created: {output_file}")
            
            # Read and display summary
            try:
                df = pd.read_csv(output_file, sep='\t')
                print(f"[INFO] Expression predicted for {len(df)} genes")
                print(f"[INFO] Sample count: {len([col for col in df.columns if col not in ['CHROM', 'GeneStart', 'GeneEnd', 'TargetID', 'GeneName']])}")
                
                # Convert to CSV
                csv_file = output_file.replace('.txt', '.csv')
                df.to_csv(csv_file, index=False)
                print(f"[INFO] ✓ CSV format saved: {csv_file}")
                
                return csv_file
                
            except Exception as e:
                print(f"[WARNING] Could not read output file: {e}")
                return output_file
        else:
            print(f"[ERROR] TIGAR output file not created")
            # List files in output directory for debugging
            if os.path.exists(tissue_output_dir):
                files = os.listdir(tissue_output_dir)
                print(f"[DEBUG] Files in {tissue_output_dir}: {files}")
            return None
        
    except Exception as e:
        print(f"[ERROR] TIGAR prediction failed: {e}")
        return None

def process_chromosome(chromosome, vcf_status):
    """
    Process one chromosome using chrAll.vcf.gz files:
    Run TIGAR for all tissues and datasets
    """
    print(f"\n[INFO] ========== PROCESSING CHROMOSOME {chromosome} ==========")
    
    # Define datasets with their chrAll VCF files
    datasets = []
    
    # Check which datasets have chrAll VCF files
    if vcf_status.get('training'):
        datasets.append((vcf_status['training'], TrainExpression_dir, "training"))
    
    if vcf_status.get('validation'):
        datasets.append((vcf_status['validation'], ValidationExpression_dir, "validation"))
    
    if vcf_status.get('test'):
        datasets.append((vcf_status['test'], TestExpression_dir, "test"))
    
    if not datasets:
        print(f"[ERROR] No chrAll.vcf.gz files found for processing")
        return {}
    
    # Get available tissues
    tissues = get_available_tissues(chromosome)
    if not tissues:
        print(f"[ERROR] No tissues with weights found for chromosome {chromosome}")
        return {}
    
    # Run TIGAR for all datasets and tissues
    print(f"\n[INFO] TIGAR GENE EXPRESSION PREDICTION - Chr{chromosome} (using chrAll.vcf.gz)")
    
    results = {}
    
    for vcf_file, expression_dir, dataset_name in datasets:
        dataset_results = {}
        
        print(f"\n[INFO] Processing {dataset_name} dataset with chrAll.vcf.gz...")
        
        for tissue_name in tissues:
            result_file = predict_expression_tigar(
                vcf_file, expression_dir, chromosome, dataset_name, tissue_name
            )
            
            if result_file:
                dataset_results[tissue_name] = result_file
        
        if dataset_results:
            results[dataset_name] = dataset_results
    
    return results

def main():
    """
    Main function to process all chromosomes using existing VCF files:
    Run TIGAR for all tissues, datasets, and chromosomes
    """
    print(f"[INFO] TIGAR Gene Expression Prediction Processor")
    print("=" * 70)
    print(f"[INFO] Phenotype: {Phenotype}")
    print(f"[INFO] Fold: {Fold}")
    if Specified_Tissue:
        print(f"[INFO] Target tissue: {Specified_Tissue}")
    else:
        print(f"[INFO] Processing all available tissues")
    
    if Specified_Chromosome:
        print(f"[INFO] Target chromosome(s): {Specified_Chromosome} -> {CHROMOSOMES}")
    else:
        print(f"[INFO] Processing all chromosomes: {CHROMOSOMES}")
    
    print(f"[INFO] Base path: {Path}")
    print(f"[INFO] Using existing VCF files from TrainVCF/, ValidationVCF/, TestVCF/")
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Check base path
    if not os.path.exists(Path):
        print(f"[ERROR] Base path not found: {Path}")
        sys.exit(1)
    
    # Check existing VCF files
    vcf_status = check_vcf_files()
    
    # Verify we have at least one dataset with chrAll VCF files
    has_vcf_files = any(vcf_status.values())
    if not has_vcf_files:
        print("[ERROR] No chrAll.vcf.gz files found in any dataset directory!")
        print("[INFO] Make sure chrAll.vcf.gz files exist in TrainVCF/, ValidationVCF/, TestVCF/ directories")
        print("[INFO] Run the VCF conversion script first")
        sys.exit(1)
    
    # Process all chromosomes
    all_results = {}
    total_chromosomes = len(CHROMOSOMES)
    
    for i, chromosome in enumerate(CHROMOSOMES, 1):
        print(f"\n[INFO] ========== CHROMOSOME {chromosome} ({i}/{total_chromosomes}) ==========")
        
        try:
            chr_results = process_chromosome(chromosome, vcf_status)
            if chr_results:
                all_results[chromosome] = chr_results
                print(f"[INFO] ✓ Chromosome {chromosome} completed successfully")
            else:
                print(f"[WARNING] Chromosome {chromosome} had no successful predictions")
        
        except Exception as e:
            print(f"[ERROR] Chromosome {chromosome} failed: {e}")
            continue
    
    # Final summary
    print(f"\n[INFO] ========== FINAL SUMMARY FOR ALL CHROMOSOMES ==========")
    
    if all_results:
        total_successful_chromosomes = len(all_results)
        print(f"[INFO] ✓ SUCCESS! Processed {total_successful_chromosomes}/{total_chromosomes} chromosomes")
        
        # Count total predictions
        total_predictions = 0
        for chr_results in all_results.values():
            for dataset_results in chr_results.values():
                total_predictions += len(dataset_results)
        
        print(f"[INFO] Total successful predictions: {total_predictions}")
        
        print(f"\n[INFO] Results summary by chromosome:")
        for chromosome, chr_results in all_results.items():
            print(f"  Chromosome {chromosome}:")
            for dataset_name, dataset_results in chr_results.items():
                print(f"    {dataset_name}: {len(dataset_results)} tissues")
        
        print(f"\n[INFO] Output directories:")
        print(f"  Expression results: {TrainExpression_dir}, {ValidationExpression_dir}, {TestExpression_dir}")
        
    else:
        print("[ERROR] No chromosomes were processed successfully!")
        sys.exit(1)
    
    print(f"\n[INFO] All chromosomes processing completed successfully!")

if __name__ == "__main__":
    main()