#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import os
import subprocess
import sys
import pandas as pd
from tqdm import tqdm
import gzip

def check_dosage_file_integrity(dosage_dir):
    """Check integrity of all dosage files and return list of valid ones"""
    chromosomes = range(1, 23)
    valid_chroms = []
    corrupted_chroms = []
    
    print "[INFO] Checking dosage file integrity..."
    
    for chrom in chromosomes:
        dosage_file = os.path.join(dosage_dir, "chr{}.txt.gz".format(chrom))
        
        if not os.path.exists(dosage_file):
            print "[WARNING] File not found: chr{}.txt.gz".format(chrom)
            continue
        
        try:
            # Test gzip file integrity
            with gzip.open(dosage_file, 'rb') as f:
                # Read a small portion to test
                f.read(1024)
            
            # If we get here, file is OK
            valid_chroms.append(chrom)
            print "[OK] chr{}.txt.gz is valid".format(chrom)
            
        except Exception as e:
            corrupted_chroms.append(chrom)
            print "[ERROR] chr{}.txt.gz is corrupted: {}".format(chrom, str(e))
    
    print "\n[INFO] File integrity check summary:"
    print "[INFO] Valid chromosomes: {}".format(', '.join(map(str, valid_chroms)))
    if corrupted_chroms:
        print "[WARNING] Corrupted chromosomes: {}".format(', '.join(map(str, corrupted_chroms)))
        print "[WARNING] These files need to be regenerated or fixed"
    
    return valid_chroms, corrupted_chroms

def create_clean_dosage_directory(source_dir, target_dir, valid_chroms):
    """Create a clean dosage directory with only valid files"""
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print "[INFO] Created clean dosage directory: {}".format(target_dir)
    
    for chrom in valid_chroms:
        source_file = os.path.join(source_dir, "chr{}.txt.gz".format(chrom))
        target_file = os.path.join(target_dir, "chr{}.txt.gz".format(chrom))
        
        if not os.path.exists(target_file):
            # Create symlink to save space
            os.symlink(os.path.abspath(source_file), target_file)
            print "[INFO] Linked chr{}.txt.gz to clean directory".format(chrom)

def run_predixcan_with_error_handling(dosage_dir, weights_file, output_file):
    """Run PrediXcan with better error handling"""
    
    # First check dosage file integrity
    valid_chroms, corrupted_chroms = check_dosage_file_integrity(dosage_dir)
    
    if not valid_chroms:
        print "[ERROR] No valid dosage files found!"
        return False
    
    if corrupted_chroms:
        print "[WARNING] Found {} corrupted files. Proceeding with {} valid chromosomes.".format(
            len(corrupted_chroms), len(valid_chroms))
        
        # Create clean directory with only valid files
        clean_dosage_dir = dosage_dir + "_clean"
        create_clean_dosage_directory(dosage_dir, clean_dosage_dir, valid_chroms)
        dosage_dir_to_use = clean_dosage_dir
    else:
        dosage_dir_to_use = dosage_dir
    
    # Run PrediXcan
    cmd = [
        "python2",
        "./PrediXcan/Software/predict_gene_expression.py",
        "--dosages", dosage_dir_to_use,
        "--dosages_prefix", "chr",
        "--weights", weights_file,
        "--output", output_file
    ]
    
    print "[INFO] Running PrediXcan with {} chromosomes...".format(len(valid_chroms))
    print "[INFO] Command: {}".format(' '.join(cmd))
    
    try:
        # Run with real-time output
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                 universal_newlines=True, bufsize=1)
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print output.strip()
        
        rc = process.poll()
        if rc == 0:
            print "[SUCCESS] PrediXcan completed successfully"
            return True
        else:
            print "[ERROR] PrediXcan failed with return code: {}".format(rc)
            return False
            
    except Exception as e:
        print "[ERROR] Failed to run PrediXcan: {}".format(e)
        return False

# Example usage
if __name__ == "__main__":
    dosage_dir = "migraine/Fold_0/TrainDosage/"
    weights_file = "/data/ascher02/uqmmune1/ANNOVAR/MR-JTI/models/JTI_models/UTMOST_Adipose_Subcutaneous.db"
    output_file = "migraine/Fold_0/UTMOSTTrainExpression/Adipose_Subcutaneous/predicted_expression_UTMOST_training_Adipose_Subcutaneous_merged.txt"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    success = run_predixcan_with_error_handling(dosage_dir, weights_file, output_file)
    
    if success:
        print "[INFO] Prediction completed successfully!"
    else:
        print "[ERROR] Prediction failed!"
        sys.exit(1)