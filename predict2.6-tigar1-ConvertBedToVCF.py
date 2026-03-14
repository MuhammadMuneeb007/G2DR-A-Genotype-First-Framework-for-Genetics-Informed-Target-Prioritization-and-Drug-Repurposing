#!/usr/bin/env python3

import os
import sys
import subprocess
import gzip

def run_cmd(cmd):
    """Run command and return success status"""
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {' '.join(cmd)}")
        print(f"Error: {e.stderr}")
        return False
    except Exception as e:
        print(f"Command failed: {' '.join(cmd)}")
        print(f"Error: {e}")
        return False

def bed_to_vcf(input_prefix, output_vcf, chromosome):
    """Convert BED to VCF for specific chromosome"""
    # Get output directory and create temp file path
    output_dir = os.path.dirname(output_vcf)
    temp_prefix = os.path.join(output_dir, f'temp_chr{chromosome}')
    temp_vcf = f'{temp_prefix}.vcf'
    
    print(f"Converting BED to VCF for chromosome {chromosome}...")
    
    # PLINK conversion with chromosome filter
    cmd = ['plink2', '--bfile', input_prefix, '--chr', str(chromosome), '--recode', 'vcf', '--out', temp_prefix]
    if not run_cmd(cmd):
        cmd = ['plink', '--bfile', input_prefix, '--chr', str(chromosome), '--recode', 'vcf', '--out', temp_prefix]
        if not run_cmd(cmd):
            print(f"Failed to run PLINK for chromosome {chromosome}")
            return False
    
    # Check if temp VCF was created
    if not os.path.exists(temp_vcf):
        print(f"PLINK did not create VCF file: {temp_vcf}")
        return False
    
    # Clean VCF (remove headers except #CHROM and filter invalid positions)
    print(f"Processing VCF file...")
    try:
        with open(temp_vcf, 'r') as infile, open(output_vcf, 'w') as outfile:
            data_lines = 0
            
            for line in infile:
                if line.startswith('#CHROM'):
                    outfile.write(line)
                elif not line.startswith('#'):
                    # Check if position is valid (> 0)
                    cols = line.strip().split('\t')
                    if len(cols) >= 2:
                        try:
                            pos = int(cols[1])
                            if pos > 0:  # Only keep positions > 0
                                outfile.write(line)
                                data_lines += 1
                        except ValueError:
                            continue
        
        print(f"Processed {data_lines} variants for chromosome {chromosome}")
        
        # Check if we have any data
        if data_lines == 0:
            print(f"Warning: No valid data lines for chromosome {chromosome}")
            return False
            
    except Exception as e:
        print(f"Error processing VCF: {e}")
        return False
    
    # Remove temp file
    if os.path.exists(temp_vcf):
        os.remove(temp_vcf)
    
    return True

def compress_and_index(filename):
    """Compress with bgzip and create tabix index"""
    # Compress
    cmd = ['bgzip', '-f', filename]
    if not run_cmd(cmd):
        return False
    
    # Index
    compressed = f"{filename}.gz"
    cmd = ['tabix', '-f', '-p', 'vcf', compressed]
    if not run_cmd(cmd):
        return False
    
    return compressed

def extract_sample_ids(vcf_gz, output_file):
    """Extract sample IDs from VCF and save to file"""
    try:
        print(f"Extracting sample IDs from {vcf_gz}...")
        with gzip.open(vcf_gz, 'rt') as infile:
            for line in infile:
                if line.startswith('#CHROM'):
                    cols = line.strip().split('\t')
                    if len(cols) > 9:
                        sample_ids = cols[9:]  # Sample IDs start from column 10
                        
                        print(f"Found {len(sample_ids)} samples")
                        
                        with open(output_file, 'w') as outfile:
                            for sample_id in sample_ids:
                                outfile.write(f"{sample_id}\n")
                        
                        print(f"Sample IDs written to {output_file}")
                        return True
                    else:
                        print(f"No sample columns found in VCF header")
                        return False
                    break
        
        print(f"No #CHROM line found in VCF file")
        return False
    except Exception as e:
        print(f"Error extracting sample IDs: {e}")
        return False

def create_chrall_vcf(input_prefix, output_dir):
    """Create chrAll.vcf directly from BED files (all autosomes)"""
    output_vcf = f"{output_dir}/chrAll.vcf"
    temp_prefix = os.path.join(output_dir, 'temp_chrAll')
    temp_vcf = f'{temp_prefix}.vcf'
    
    print(f"Creating chrAll.vcf from {input_prefix}...")
    
    # Convert entire BED file to VCF (autosomes only: 1-22)
    cmd = ['plink2', '--bfile', input_prefix, '--chr', '1-22', '--recode', 'vcf', '--out', temp_prefix]
    if not run_cmd(cmd):
        cmd = ['plink', '--bfile', input_prefix, '--chr', '1-22', '--recode', 'vcf', '--out', temp_prefix]
        if not run_cmd(cmd):
            print(f"Failed to create chrAll VCF from {input_prefix}")
            return False
    
    # Check if temp VCF was created
    if not os.path.exists(temp_vcf):
        print(f"PLINK did not create chrAll VCF file: {temp_vcf}")
        return False
    
    # Clean VCF (remove headers except #CHROM and filter invalid positions)
    print(f"Processing chrAll VCF file...")
    try:
        with open(temp_vcf, 'r') as infile, open(output_vcf, 'w') as outfile:
            data_lines = 0
            
            for line in infile:
                if line.startswith('#CHROM'):
                    outfile.write(line)
                elif not line.startswith('#'):
                    # Check if position is valid (> 0)
                    cols = line.strip().split('\t')
                    if len(cols) >= 2:
                        try:
                            pos = int(cols[1])
                            if pos > 0:  # Only keep positions > 0
                                outfile.write(line)
                                data_lines += 1
                        except ValueError:
                            continue
        
        print(f"Processed {data_lines} total variants for chrAll")
        
        # Check if we have any data
        if data_lines == 0:
            print(f"Warning: No valid data lines for chrAll")
            return False
            
    except Exception as e:
        print(f"Error processing chrAll VCF: {e}")
        return False
    
    # Remove temp file
    if os.path.exists(temp_vcf):
        os.remove(temp_vcf)
    
    # Compress and index chrAll
    if os.path.exists(output_vcf):
        chrall_vcf_gz = compress_and_index(output_vcf)
        if chrall_vcf_gz:
            print(f"chrAll VCF created and compressed: {chrall_vcf_gz}")
            return True
        else:
            print("Failed to compress chrAll VCF")
            return False
    
    return False

def check_bed_files(input_prefix):
    """Check if BED files exist and get basic info"""
    bed_file = f"{input_prefix}.bed"
    bim_file = f"{input_prefix}.bim"
    fam_file = f"{input_prefix}.fam"
    
    files = [bed_file, bim_file, fam_file]
    missing_files = [f for f in files if not os.path.exists(f)]
    
    if missing_files:
        print(f"Missing files: {missing_files}")
        return False
    
    try:
        # Count lines in BIM and FAM files
        with open(bim_file, 'r') as f:
            variant_count = sum(1 for line in f)
        
        with open(fam_file, 'r') as f:
            sample_count = sum(1 for line in f)
        
        print(f"Found {sample_count} samples and {variant_count} variants in {input_prefix}")
        return True
        
    except Exception as e:
        print(f"Error reading BED files: {e}")
        return False

def process_dataset(base_path, dataset_name, chromosomes):
    """Process a single dataset (train, validation, or test) for all chromosomes"""
    input_prefix = f"{base_path}/{dataset_name}_data"
    output_dir = f"{base_path}/{dataset_name.title()}VCF"
    
    print(f"\n=== Processing {dataset_name} dataset ===")
    
    # Check if BED files exist
    if not check_bed_files(input_prefix):
        print(f"BED files not found or invalid for {dataset_name}")
        return []
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Track if sample IDs have been extracted (only need to do once per dataset)
    sample_ids_extracted = False
    successful_chromosomes = []
    
    # Process individual chromosomes
    for chromosome in chromosomes:
        print(f"\nProcessing {dataset_name} data for chromosome {chromosome}...")
        
        output_vcf = f"{output_dir}/chr{chromosome}.vcf"
        
        # Convert BED to VCF for specific chromosome
        if bed_to_vcf(input_prefix, output_vcf, chromosome):
            # Check if VCF file has content
            if os.path.exists(output_vcf) and os.path.getsize(output_vcf) > 0:
                # Compress VCF
                vcf_gz = compress_and_index(output_vcf)
                if vcf_gz:
                    # Extract sample IDs for each dataset type (only once per dataset)
                    if not sample_ids_extracted:
                        # Create appropriate filename based on dataset type
                        if dataset_name == "train":
                            sample_id_file = f"{output_dir}/training_sampleID.txt"
                        elif dataset_name == "validation":
                            sample_id_file = f"{output_dir}/validation_sampleID.txt"
                        elif dataset_name == "test":
                            sample_id_file = f"{output_dir}/test_sampleID.txt"
                        else:
                            sample_id_file = f"{output_dir}/{dataset_name}_sampleID.txt"
                        
                        if extract_sample_ids(vcf_gz, sample_id_file):
                            sample_ids_extracted = True
                            print(f"Sample IDs extracted to {sample_id_file}")
                    
                    successful_chromosomes.append(chromosome)
                    print(f"Successfully processed chromosome {chromosome}")
                else:
                    print(f"Failed to compress VCF for chromosome {chromosome}")
            else:
                print(f"VCF file is empty or missing for chromosome {chromosome}")
        else:
            print(f"Failed to convert BED to VCF for chromosome {chromosome}")
    
    # Create chrAll VCF file directly from BED files
    print(f"\nCreating chrAll VCF file for {dataset_name} from original BED files...")
    if create_chrall_vcf(input_prefix, output_dir):
        print(f"Successfully created chrAll.vcf.gz for {dataset_name}")
    else:
        print(f"Failed to create chrAll.vcf.gz for {dataset_name}")
    
    if successful_chromosomes:
        print(f"Successfully processed {len(successful_chromosomes)} chromosomes for {dataset_name}")
    else:
        print(f"No successful chromosomes for {dataset_name}")
    
    return successful_chromosomes

def main():
    if len(sys.argv) != 3:
        print("Usage: python tigar1.py <phenotype> <fold>")
        print("Example: python tigar1.py migraine 0")
        return
    
    phenotype = sys.argv[1]
    fold = sys.argv[2]
    
    # Define chromosomes to process (AUTOSOMES ONLY: 1-22)
    chromosomes = list(range(1, 23))  # 1-22 only, NO X or Y
    
    print(f"PLINK BED to VCF Converter")
    print("=" * 50)
    print(f"Phenotype: {phenotype}")
    print(f"Fold: {fold}")
    print(f"Chromosomes to process: {chromosomes} (autosomes only)")
    
    # Build base path
    base_path = f"{phenotype}/Fold_{fold}"
    
    # Check if base path exists
    if not os.path.exists(base_path):
        print(f"Error: Base path does not exist: {base_path}")
        return
    
    print(f"Base path: {base_path}")
    
    # Process training data
    print("\n=== Processing Training Data ===")
    train_success = process_dataset(base_path, "train", chromosomes)
    
    # Process validation data
    print("\n=== Processing Validation Data ===")
    validation_success = process_dataset(base_path, "validation", chromosomes)
    
    # Process test data
    print("\n=== Processing Test Data ===")
    test_success = process_dataset(base_path, "test", chromosomes)
    
    # Final summary
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Training: {len(train_success)} chromosomes successful")
    print(f"Validation: {len(validation_success)} chromosomes successful")
    print(f"Test: {len(test_success)} chromosomes successful")
    
    total_successful = len(train_success) + len(validation_success) + len(test_success)
    total_possible = len(chromosomes) * 3  # 3 datasets
    
    print(f"Total: {total_successful}/{total_possible} conversions successful")
    
    if total_successful > 0:
        print(f"\nProcessing complete for {phenotype} fold {fold}")
        print("VCF files created in TrainVCF/, ValidationVCF/, TestVCF/ directories")
        print("chrAll.vcf.gz files created for each dataset")
        print("Sample ID files created for all datasets:")
        print("  - TrainVCF/training_sampleID.txt")
        print("  - ValidationVCF/validation_sampleID.txt") 
        print("  - TestVCF/test_sampleID.txt")
    else:
        print(f"\nNo successful conversions! Check your BED files.")
        return

if __name__ == "__main__":
    main()