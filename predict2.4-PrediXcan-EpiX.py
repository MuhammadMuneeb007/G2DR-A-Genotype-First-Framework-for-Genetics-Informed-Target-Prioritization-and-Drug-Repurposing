#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import os
import subprocess
import sys
import pandas as pd
from tqdm import tqdm
import sys
# ----------- CONFIGURATION -----------
# File path setup
Phenotype = sys.argv[1]
Fold = sys.argv[2]
Path = Phenotype + "/Fold_" + str(Fold) + "/"

# Dosage input directories (where dosage files are located)
TrainDosage_dir = Path + "TrainDosage/"
ValidationDosage_dir = Path + "ValidationDosage/"
TestDosage_dir = Path + "TestDosage/"

# Expression output directories
TrainExpression_dir = Path + "EpiXTrainExpression/"
ValidationExpression_dir = Path + "EpiXValidationExpression/"
TestExpression_dir = Path + "EpiXTestExpression/"

# Gene expression prediction configuration
chromosomes = range(1, 23)  # Chromosomes 1-22
epix_models_dir = "/data/ascher02/uqmmune1/ANNOVAR/PrediXcan/episcan/models"

def get_available_tissues():
    """
    Get ALL EpiX tissue models - no filtering
    
    Returns:
        List of tissue names with their corresponding model file paths
    """
    tissues = []
    
    if not os.path.exists(epix_models_dir):
        print "[ERROR] EpiX models directory not found: {}".format(epix_models_dir)
        return tissues
    
    # ALL the EpiX files - no filtering, just process them all
    all_epix_files = [
        "CMC467_EpiX_alpha0.5_window1e6_filtered.db",
        "GTEx_Aor_EpiX_alpha0.5_window1e6_filtered.db",
        "GTEx_Blood_EpiX_alpha0.5_window1e6_filtered.db", 
        "GTEx_Liv_EpiX_alpha0.5_window1e6_filtered.db",
        "GTEx_SF_EpiX_alpha0.5_window1e6_filtered.db",
        "GTEx_SKLM_EpiX_alpha0.5_window1e6_filtered.db",
        "GTEx_VAF_EpiX_alpha0.5_window1e6_filtered.db",
        "STARNET_EpiX_AOR_alpha0.5_window1e6_filtered.db",
        "STARNET_EpiX_BLD_alpha0.5_window1e6_filtered.db",
        "STARNET_EpiX_LIV_alpha0.5_window1e6_filtered.db",
        "STARNET_EpiX_MAM_alpha0.5_window1e6_filtered.db",
        "STARNET_EpiX_SF_alpha0.5_window1e6_filtered.db",
        "STARNET_EpiX_SKLM_alpha0.5_window1e6_filtered.db",
        "STARNET_EpiX_VAF_alpha0.5_window1e6_filtered.db"
    ]
    
    print "[INFO] Processing {} EpiX model files".format(len(all_epix_files))
    
    for filename in all_epix_files:
        model_path = os.path.join(epix_models_dir, filename)
        
        if os.path.exists(model_path):
            tissue_name = None
            
            # Parse tissue names
            if filename.startswith("GTEx_"):
                parts = filename.split("_")
                tissue_name = "GTEx_" + parts[1]
            elif filename.startswith("STARNET_"):
                parts = filename.split("_")
                tissue_name = "STARNET_" + parts[2]
            elif filename.startswith("CMC467_"):
                tissue_name = "CMC467"
            
            if tissue_name:
                tissues.append((tissue_name, model_path))
                print "[INFO] Added: {} -> {}".format(tissue_name, filename)
            else:
                print "[WARNING] Could not parse tissue name from: {}".format(filename)
        else:
            print "[WARNING] File not found: {}".format(model_path)
    
    tissues.sort()  # Sort alphabetically
    print "[INFO] Total tissues loaded: {}".format(len(tissues))
    
    return tissues

def create_directory(directory_path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print "[INFO] Created directory: {}".format(directory_path)

def dosage_exists(dosage_dir, chrom):
    """Check if dosage files for this chromosome exist"""
    if not os.path.exists(dosage_dir):
        return False
    
    # Check if the main dosage file exists
    dosage_file = os.path.join(dosage_dir, "chr{}.txt.gz".format(chrom))
    return os.path.exists(dosage_file)

def predict_expression_for_tissue(dosage_dir, expression_dir, dataset_name, tissue_name, tissue_model_path):
    """
    Predict gene expression for a tissue using ALL chromosomes together using EpiX models
    
    Args:
        dosage_dir: Directory containing dosage files
        expression_dir: Base expression directory
        dataset_name: Name of dataset (for logging)
        tissue_name: Name of tissue
        tissue_model_path: Path to tissue-specific EpiX model file (.db.gz)
    
    Returns:
        Path to the output file or None if failed
    """
    print "[INFO] Predicting expression for {} - {} using EpiX model (using ALL chromosomes)".format(dataset_name, tissue_name)
    
    # Create tissue-specific subdirectory
    tissue_expression_dir = os.path.join(expression_dir, tissue_name)
    create_directory(tissue_expression_dir)
    
    # Define output file path with EpiX naming convention
    output_file = os.path.join(tissue_expression_dir, "predicted_expression_EpiX_{}_{}_merged.txt".format(dataset_name, tissue_name))
    
    # Check if dosage directory exists
    if not os.path.exists(dosage_dir):
        print "[WARNING] Dosage directory not found: {}".format(dosage_dir)
        return None
    
    # Check if any dosage files exist
    available_chroms = []
    for chrom in chromosomes:
        if dosage_exists(dosage_dir, chrom):
            available_chroms.append(chrom)
    
    if not available_chroms:
        print "[WARNING] No dosage files found in: {}".format(dosage_dir)
        return None
    
    print "[INFO] Found {} chromosomes: {}".format(len(available_chroms), ', '.join(map(str, available_chroms)))
    
    try:
        # PrediXcan command for EpiX models
        cmd = [
            "python2",
            os.path.join(predixcan_software_dir, "predict_gene_expression.py"),
            "--dosages", dosage_dir,
            "--dosages_prefix", "chr",  # For all chromosomes
            "--weights", tissue_model_path,
            "--output", output_file
        ]
        
        print "[INFO] Running EpiX PrediXcan: {}".format(' '.join(cmd))
        
        # Run with progress indication
        with tqdm(desc="Processing {} {}".format(dataset_name, tissue_name[:20]), unit="step") as pbar:
            pbar.set_description("Processing EpiX {} {}".format(dataset_name, tissue_name[:20]))
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    output_str = output.strip()
                    if len(output_str) > 0:
                        display_output = output_str[:40] + "..." if len(output_str) > 40 else output_str
                        pbar.set_description("EpiX {} {}: {}".format(dataset_name, tissue_name[:15], display_output))
                    pbar.update(1)
            
            rc = process.poll()
            if rc != 0:
                stderr_output = process.stderr.read()
                raise subprocess.CalledProcessError(rc, cmd, stderr_output)
            
            pbar.set_description("[OK] EpiX {} {} completed".format(dataset_name, tissue_name[:20]))
        
        if os.path.exists(output_file):
            print "[INFO] EpiX expression prediction completed: {}".format(output_file)
            
            # Also create CSV version
            csv_file = output_file.replace('.txt', '.csv')
            
            # Add sample IDs from FAM file
            final_file = add_sample_ids_to_expression(output_file, csv_file, dataset_name)
            
            return final_file if final_file else csv_file
        else:
            print "[WARNING] Expected output file not created: {}".format(output_file)
            return None
        
    except subprocess.CalledProcessError as e:
        print "[ERROR] Failed to predict expression for {} {} using EpiX: {}".format(dataset_name, tissue_name, e)
        if hasattr(e, 'stderr') and e.stderr:
            print "[ERROR] Error details: {}".format(e.stderr)
        return None
    except Exception as e:
        print "[ERROR] Unexpected error: {}".format(e)
        return None

def add_sample_ids_to_expression(txt_file, csv_file, dataset_name):
    """
    Add sample IDs from FAM file to expression predictions
    
    Args:
        txt_file: PrediXcan output file (tab-separated)
        csv_file: Output CSV file path
        dataset_name: Dataset name to determine FAM file
    
    Returns:
        Path to final CSV file with sample IDs or None if failed
    """
    try:
        # Determine FAM file path based on dataset name
        if "training" in dataset_name.lower():
            fam_file = os.path.join(Path, "train_data.fam")
        elif "validation" in dataset_name.lower():
            fam_file = os.path.join(Path, "validation_data.fam")
        elif "test" in dataset_name.lower():
            fam_file = os.path.join(Path, "test_data.fam")
        else:
            print "[WARNING] Cannot determine FAM file for dataset: {}".format(dataset_name)
            fam_file = None
        
        # Read expression data
        expr_df = pd.read_csv(txt_file, sep='\t')
        print "[INFO] Read expression data: {} samples x {} genes".format(len(expr_df), len(expr_df.columns))
        
        # Read sample IDs from FAM file
        if fam_file and os.path.exists(fam_file):
            print "[INFO] Reading sample IDs from FAM file: {}".format(fam_file)
            sample_ids_df = pd.read_csv(fam_file, sep=r'\s+', header=None, 
                                      names=['FID', 'IID', 'PID', 'MID', 'SEX', 'PHENOTYPE'])
            
            # Verify sample count matches
            if len(sample_ids_df) == len(expr_df):
                # Add sample IDs to expression data
                final_df = pd.concat([sample_ids_df[['FID', 'IID']], expr_df], axis=1)
                print "[INFO] Added sample IDs: {} samples matched".format(len(final_df))
            else:
                print "[WARNING] Sample count mismatch: FAM file has {} samples, expression file has {} samples".format(
                    len(sample_ids_df), len(expr_df))
                # Create dummy sample IDs
                final_df = pd.DataFrame({
                    'FID': ['FAM{}'.format(i+1) for i in range(len(expr_df))],
                    'IID': ['SAMPLE{}'.format(i+1) for i in range(len(expr_df))]
                })
                final_df = pd.concat([final_df, expr_df], axis=1)
        else:
            print "[WARNING] FAM file not found: {}".format(fam_file)
            # Create dummy sample IDs
            final_df = pd.DataFrame({
                'FID': ['FAM{}'.format(i+1) for i in range(len(expr_df))],
                'IID': ['SAMPLE{}'.format(i+1) for i in range(len(expr_df))]
            })
            final_df = pd.concat([final_df, expr_df], axis=1)
        
        # Save as CSV
        final_df.to_csv(csv_file, index=False)
        print "[INFO] Final expression file saved: {}".format(csv_file)
        print "[INFO] Final dimensions: {} samples x {} total columns".format(len(final_df), len(final_df.columns))
        
        return csv_file
        
    except Exception as e:
        print "[ERROR] Failed to add sample IDs: {}".format(e)
        return None

def predict_expression_for_dataset(dosage_dir, expression_dir, dataset_name, tissues):
    """
    Predict gene expression for all EpiX tissues in a dataset
    
    Args:
        dosage_dir: Directory containing dosage files
        expression_dir: Directory to save expression predictions
        dataset_name: Name of dataset (for logging)
        tissues: List of (tissue_name, tissue_model_path) tuples
    
    Returns:
        Dictionary of tissue_name -> expression_file paths
    """
    print "[INFO] Starting EpiX gene expression prediction for {} data".format(dataset_name)
    print "[INFO] Input directory: {}".format(dosage_dir)
    print "[INFO] Output directory: {}".format(expression_dir)
    print "[INFO] Processing {} EpiX tissues (using ALL chromosomes together)".format(len(tissues))
    
    # Check if dosage directory exists
    if not os.path.exists(dosage_dir):
        print "[ERROR] Dosage directory not found: {}".format(dosage_dir)
        return {}
    
    tissue_results = {}
    
    # Process each tissue
    for tissue_idx, (tissue_name, tissue_model_path) in enumerate(tissues, 1):
        print "\n[INFO] --- Processing EpiX tissue {}/{}: {} ---".format(tissue_idx, len(tissues), tissue_name)
        
        # Check if tissue model file exists
        if not os.path.exists(tissue_model_path):
            print "[WARNING] EpiX tissue model not found: {}".format(tissue_model_path)
            continue
        
        # Run PrediXcan for this tissue (using ALL chromosomes)
        output_file = predict_expression_for_tissue(
            dosage_dir, expression_dir, dataset_name, tissue_name, tissue_model_path
        )
        
        if output_file and os.path.exists(output_file):
            tissue_results[tissue_name] = output_file
            print "[INFO] [OK] {} EpiX {} tissue completed".format(dataset_name, tissue_name)
        else:
            print "[WARNING] {} EpiX {} tissue failed".format(dataset_name, tissue_name)
    
    return tissue_results

def check_dosage_files(dosage_dir, dataset_name):
    """Check what dosage files are available in a directory"""
    if not os.path.exists(dosage_dir):
        print "[WARNING] Directory not found: {}".format(dosage_dir)
        return []
    
    available_chroms = []
    for chrom in chromosomes:
        if dosage_exists(dosage_dir, chrom):
            available_chroms.append(chrom)
    
    print "[INFO] {} data: {} chromosomes available ({})".format(
        dataset_name, len(available_chroms), 
        ', '.join(map(str, available_chroms)) if available_chroms else "none"
    )
    
    return available_chroms

def main():
    """Main function to predict gene expression for all datasets and EpiX tissues"""
    
    # Get available EpiX tissues
    tissues = get_available_tissues()
    if not tissues:
        print "[ERROR] No EpiX tissue models found in: {}".format(epix_models_dir)
        sys.exit(1)
    
    # Define datasets to process
    datasets = [
        (TrainDosage_dir, TrainExpression_dir, "training"),
        (ValidationDosage_dir, ValidationExpression_dir, "validation"),
        (TestDosage_dir, TestExpression_dir, "test")
    ]
    
    # Display pipeline information
    print "[INFO] EpiX Gene Expression Prediction Pipeline"
    print "[INFO] " + "=" * 60
    print "[INFO] Working directory: {}".format(os.getcwd())
    print "[INFO] Base path: {}".format(Path)
    print "[INFO] EpiX models directory: {}".format(epix_models_dir)
    print "[INFO] EpiX tissues available: {}".format(len(tissues))
    print "[INFO] Chromosomes: {}".format(", ".join(map(str, chromosomes)))
    print "[INFO] Method: Run PrediXcan ONCE per EpiX tissue using ALL chromosomes"
    print "[INFO] Expected datasets:"
    for dosage_dir, expression_dir, dataset_name in datasets:
        print "  - {}: {}".format(dataset_name.capitalize(), dosage_dir)
        print "    Output: {}".format(expression_dir)
    
    total_predictions = len(datasets) * len(tissues)
    print "[INFO] Total EpiX predictions to perform: {} datasets x {} tissues = {}".format(
        len(datasets), len(tissues), total_predictions
    )
    
    # Check if base path exists
    if not os.path.exists(Path):
        print "[ERROR] Base path not found: {}".format(Path)
        print "[INFO] Please ensure the directory structure exists:"
        print "  {}".format(Path)
        print "  ├── TrainDosage/"
        print "  ├── ValidationDosage/"
        print "  ├── TestDosage/"
        print "  ├── EpiXTrainExpression/"
        print "  ├── EpiXValidationExpression/"
        print "  └── EpiXTestExpression/"
        sys.exit(1)
    
    print "[INFO] [OK] Base path exists: {}".format(Path)
    
    # Check if EpiX models directory exists
    if not os.path.exists(epix_models_dir):
        print "[ERROR] EpiX models directory not found: {}".format(epix_models_dir)
        sys.exit(1)
    print "[INFO] [OK] EpiX models directory verified"
    
    # Check available dosage files for each dataset
    print "\n[INFO] Checking available dosage files..."
    available_datasets = []
    
    for dosage_dir, expression_dir, dataset_name in datasets:
        available_chroms = check_dosage_files(dosage_dir, dataset_name)
        if available_chroms:
            available_datasets.append((dosage_dir, expression_dir, dataset_name))
        else:
            print "[WARNING] No dosage files found for {} data, skipping".format(dataset_name)
    
    if not available_datasets:
        print "[ERROR] No datasets with dosage files found!"
        print "[INFO] Please check that your dosage directories contain chr*.txt.gz files"
        sys.exit(1)
    
    print "\n[INFO] Processing {} datasets with available dosage files...".format(len(available_datasets))
    
    # Process each dataset
    all_successful_results = {}
    
    for i, (dosage_dir, expression_dir, dataset_name) in enumerate(available_datasets, 1):
        print "\n" + "=" * 80
        print "Processing dataset {}/{}: {} data".format(i, len(available_datasets), dataset_name)
        print "=" * 80
        
        # Predict gene expression for all tissues
        tissue_results = predict_expression_for_dataset(dosage_dir, expression_dir, dataset_name, tissues)
        
        if tissue_results:
            print "[INFO] [OK] Successfully completed {} data processing".format(dataset_name)
            print "[INFO] Tissues completed for {}: {}".format(dataset_name, len(tissue_results))
            all_successful_results[dataset_name] = tissue_results
            
            # Show tissue summary for this dataset
            for tissue_name, merged_file in tissue_results.items():
                print "  - {}: {}".format(tissue_name, os.path.basename(merged_file))
        else:
            print "[ERROR] Failed to complete {} data processing".format(dataset_name)
    
    # Final summary
    print "\n" + "=" * 80
    print "[INFO] EpiX Gene Expression Prediction Summary"
    print "=" * 80
    
    if all_successful_results:
        total_tissues_completed = sum(len(tissue_results) for tissue_results in all_successful_results.values())
        total_datasets_completed = len(all_successful_results)
        
        print "[INFO] SUCCESS! Completed {}/{} datasets using EpiX models".format(total_datasets_completed, len(available_datasets))
        print "[INFO] Total EpiX tissues processed: {}".format(total_tissues_completed)
        print "[INFO] Successfully processed:"
        
        for dataset_name, tissue_results in all_successful_results.items():
            print "  - {} data: {} EpiX tissues".format(dataset_name.capitalize(), len(tissue_results))
            for tissue_name in sorted(tissue_results.keys()):
                print "    * {}".format(tissue_name)
    else:
        print "[ERROR] No datasets were successfully processed with EpiX models!"
        sys.exit(1)
    
    # Count total files created
    print "\n[INFO] File summary:"
    total_expression_files = 0
    
    for _, expression_dir, name in datasets:
        if os.path.exists(expression_dir):
            # Count files in tissue subdirectories
            dataset_files = 0
            for tissue_name, _ in tissues:
                tissue_dir = os.path.join(expression_dir, tissue_name)
                if os.path.exists(tissue_dir):
                    tissue_files = [f for f in os.listdir(tissue_dir) if f.endswith((".csv", ".txt"))]
                    dataset_files += len(tissue_files)
            
            total_expression_files += dataset_files
            print "[INFO] {}: {} EpiX expression files created".format(name.capitalize(), dataset_files)
    
    print "[INFO] Total EpiX expression files created: {}".format(total_expression_files)
    print "[INFO] EpiX gene expression prediction pipeline completed!"
    
    # Display output structure
    print "\n[INFO] Output structure:"
    for dataset_name in all_successful_results.keys():
        if dataset_name == "training":
            exp_dir = TrainExpression_dir
        elif dataset_name == "validation":
            exp_dir = ValidationExpression_dir  
        elif dataset_name == "test":
            exp_dir = TestExpression_dir
        else:
            continue
        
        print "  {}".format(exp_dir)
        for tissue_name in sorted(all_successful_results[dataset_name].keys()):
            print "  ├── {}/".format(tissue_name)
            print "  │   └── predicted_expression_EpiX_{}_{}_merged.csv".format(dataset_name, tissue_name)

if __name__ == "__main__":
    # Set PrediXcan Software directory path
    home_dir = os.path.expanduser("./")
    predixcan_software_dir = os.path.join(home_dir, "PrediXcan", "Software")
    
    print "[INFO] Using PrediXcan Software directory: {}".format(predixcan_software_dir)
    
    # Check if predixcan software directory exists
    if not os.path.exists(predixcan_software_dir):
        print "[ERROR] predixcan_software_dir not found: {}".format(predixcan_software_dir)
        print "[ERROR] Please update the predixcan_software_dir variable with the correct path"
        sys.exit(1)
    
    # Check if required prediction script exists
    prediction_script = os.path.join(predixcan_software_dir, "predict_gene_expression.py")
    
    if not os.path.exists(prediction_script):
        print "[ERROR] Prediction script not found: {}".format(prediction_script)
        sys.exit(1)
    
    print "[INFO] [OK] Prediction script found"
    
    # Check if pandas is available
    try:
        import pandas as pd
        print "[INFO] [OK] Pandas is available"
    except ImportError:
        print "[ERROR] Pandas is required but not installed. Please install with: pip2 install pandas"
        sys.exit(1)
    
    # Check if EpiX models directory exists
    if not os.path.exists(epix_models_dir):
        print "[ERROR] EpiX models directory not found: {}".format(epix_models_dir)
        print "[ERROR] Please check the path: {}".format(epix_models_dir)
        sys.exit(1)
    
    print "[INFO] [OK] All dependencies verified"
    print "[INFO] Starting EpiX multi-tissue gene expression prediction pipeline...\n"
    
    main()