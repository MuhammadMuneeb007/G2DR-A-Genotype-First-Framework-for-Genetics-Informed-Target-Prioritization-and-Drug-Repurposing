#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import os
import subprocess
import sys
from tqdm import tqdm
import sys

# File path setup
Phenotype = sys.argv[1]
Fold = sys.argv[2]

Path = Phenotype + "/Fold_" + str(Fold) + "/"

# Directory names for output
TrainDosage_dir = "TrainDosage/"
ValidationDosage_dir = "ValidationDosage/"
TestDosage_dir = "TestDosage/"

# Full output paths
train_output_path = Path + TrainDosage_dir
validation_output_path = Path + ValidationDosage_dir
test_output_path = Path + TestDosage_dir

# Input file prefixes (without .bed/.bim/.fam extensions)
train_prefix = Path + "train_data"
validation_prefix = Path + "validation_data"
test_prefix = Path + "test_data"

def create_directory(directory_path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print "[INFO] Created directory: {}".format(directory_path)

def convert_to_dosage(input_prefix, output_dir, dataset_name):
    """
    Convert plink files to dosage format
    
    Args:
        input_prefix: Path prefix for input .bed/.bim/.fam files
        output_dir: Directory where dosage files will be saved
        dataset_name: Name of dataset (for logging)
    """
    print "[INFO] Converting {} to dosage format".format(dataset_name)
    print "[INFO] Input prefix: {}".format(input_prefix)
    print "[INFO] Output directory: {}".format(output_dir)
    
    # Create output directory if it doesn't exist
    create_directory(output_dir)
    
    # Create the full output prefix path
    output_prefix = os.path.join(output_dir, "chr")
    
    try:
        # Run the conversion script with progress bar
        cmd = [
            "python2",  # Using python2 since the original script uses python2
            os.path.join(predixcan_software_dir, "convert_plink_to_dosage.py"),
            "-b", input_prefix,
            "-o", output_prefix,  # Full path including directory and prefix
            "-p", "plink"  # or "plink2" if you have plink2
        ]
        
        print "[INFO] Running command: {}".format(' '.join(cmd))
        
        # Use tqdm to show progress
        with tqdm(desc="Converting {}".format(dataset_name), unit="step") as pbar:
            pbar.set_description("Processing {}".format(dataset_name))
            
            # Run the subprocess
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Monitor the process
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    # Update progress bar description with any output
                    output_str = output.strip()
                    if len(output_str) > 0:
                        # Truncate long output for display
                        display_output = output_str[:50] + "..." if len(output_str) > 50 else output_str
                        pbar.set_description("{}: {}".format(dataset_name, display_output))
                    pbar.update(1)
            
            # Get the return code
            rc = process.poll()
            if rc != 0:
                stderr_output = process.stderr.read()
                raise subprocess.CalledProcessError(rc, cmd, stderr_output)
            
            pbar.set_description("[OK] {} conversion completed".format(dataset_name))
        
        print "[INFO] Successfully converted {} to dosage format".format(dataset_name)
        print "[INFO] Output files saved with prefix: {}".format(output_prefix)
        
        # List the created files
        if os.path.exists(output_dir):
            all_files = os.listdir(output_dir)
            dosage_files = [f for f in all_files if f.startswith("chr") and f.endswith(".txt.gz")]
            if dosage_files:
                print "[INFO] Created {} dosage files: {}".format(
                    len(dosage_files), ', '.join(sorted(dosage_files))
                )
        
    except subprocess.CalledProcessError as e:
        print "[ERROR] Failed to convert {}: {}".format(dataset_name, e)
        if hasattr(e, 'stderr') and e.stderr:
            print "[ERROR] Error details: {}".format(e.stderr)
        sys.exit(1)
    except Exception as e:
        print "[ERROR] Unexpected error converting {}: {}".format(dataset_name, e)
        sys.exit(1)

def main():
    """Main function to convert all datasets"""
    
    # Check if input files exist
    datasets = [
        (train_prefix, train_output_path, "training data"),
        (validation_prefix, validation_output_path, "validation data"),
        (test_prefix, test_output_path, "test data")
    ]
    
    print "[INFO] Verifying input files..."
    # Verify input files exist before processing
    for prefix, _, name in tqdm(datasets, desc="Checking input files"):
        for ext in ['.bed', '.bim', '.fam']:
            file_path = prefix + ext
            if not os.path.exists(file_path):
                print "[ERROR] Input file not found: {}".format(file_path)
                sys.exit(1)
        print "[INFO] [OK] Input files verified for {}".format(name)
    
    print "\n[INFO] Starting conversion of {} datasets...".format(len(datasets))
    
    # Convert each dataset with overall progress
    for i, (input_prefix, output_dir, dataset_name) in enumerate(tqdm(datasets, desc="Overall Progress"), 1):
        print "\n" + "=" * 60
        print "Processing dataset {}/{}: {}".format(i, len(datasets), dataset_name)
        print "=" * 60
        
        convert_to_dosage(input_prefix, output_dir, dataset_name)
        
        print "[INFO] [OK] Completed {} ({}/{})".format(dataset_name, i, len(datasets))
    
    print "\n" + "=" * 60
    print "[INFO] SUCCESS! All datasets successfully converted to dosage format!"
    print "=" * 60
    print "[INFO] Output directories:"
    print "  - Training: {}".format(train_output_path)
    print "  - Validation: {}".format(validation_output_path)
    print "  - Test: {}".format(test_output_path)
    
    # Summary of created files
    total_files = 0
    output_dirs = [
        (train_output_path, "Training"), 
        (validation_output_path, "Validation"), 
        (test_output_path, "Test")
    ]
    
    for output_dir, name in output_dirs:
        if os.path.exists(output_dir):
            all_files = os.listdir(output_dir)
            dosage_files = [f for f in all_files if f.startswith("chr") and f.endswith(".txt.gz")]
            total_files += len(dosage_files)
            print "  - {}: {} chromosome files".format(name, len(dosage_files))
    
    print "[INFO] Total dosage files created: {}".format(total_files)




if __name__ == "__main__":
    # Set this to your actual PrediXcan Software directory path
    # Based on your directory structure and user environment:
    
    # Option 1: Use home directory path (most likely for your setup)
    home_dir = os.path.expanduser("./")
    predixcan_software_dir = os.path.join(home_dir, "PrediXcan", "Software")
    
    # Option 2: If PrediXcan is in current working directory, use this instead:
    # predixcan_software_dir = os.path.join(os.getcwd(), "PrediXcan", "Software")
    
    # Option 3: If you know the exact absolute path, use it directly:
    # predixcan_software_dir = "/home/s4750311/PrediXcan/Software"
    
    print "[INFO] Using PrediXcan Software directory: {}".format(predixcan_software_dir)
    
    # Check if predixcan software directory exists
    if not os.path.exists(predixcan_software_dir):
        print "[ERROR] predixcan_software_dir not found: {}".format(predixcan_software_dir)
        print "[ERROR] Please update the predixcan_software_dir variable with the correct path"
        sys.exit(1)
    
    # Check if required conversion script exists
    conversion_script = os.path.join(predixcan_software_dir, "convert_plink_to_dosage.py")
    if not os.path.exists(conversion_script):
        print "[ERROR] Conversion script not found: {}".format(conversion_script)
        sys.exit(1)
    
    main()