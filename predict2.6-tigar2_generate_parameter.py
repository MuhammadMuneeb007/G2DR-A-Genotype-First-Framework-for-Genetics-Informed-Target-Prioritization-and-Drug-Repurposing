#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Parameter File Writer for TIGAR
Writes parameter combinations to files using hardcoded values.
All output dynamically reflects the actual parameter values used.
"""

import os
import itertools

def write_parameter_combinations():
    """Generate and write all parameter combinations to files"""
    print("[INFO] Using hardcoded parameter values...")
    
    # Define parameters directly
    phenotypes = ['migraine']  # TODO: Replace with your actual phenotypes
    folds = ['0']  # TODO: Uncomment full list if needed: ['0', '1', '2', '3', '4']
    
    # GTEx tissues (49 tissues)
    tissues = [
        'Adipose_Subcutaneous', 'Adipose_Visceral_Omentum', 'Adrenal_Gland',
        'Artery_Aorta', 'Artery_Coronary', 'Artery_Tibial', 'Brain_Amygdala',
        'Brain_Anterior_cingulate_cortex_BA24', 'Brain_Caudate_basal_ganglia',
        'Brain_Cerebellar_Hemisphere', 'Brain_Cerebellum', 'Brain_Cortex',
        'Brain_Frontal_Cortex_BA9', 'Brain_Hippocampus', 'Brain_Hypothalamus',
        'Brain_Nucleus_accumbens_basal_ganglia', 'Brain_Putamen_basal_ganglia',
        'Brain_Spinal_cord_cervical_c-1', 'Brain_Substantia_nigra',
        'Breast_Mammary_Tissue', 'Cells_Cultured_fibroblasts',
        'Cells_EBV-transformed_lymphocytes', 'Colon_Sigmoid', 'Colon_Transverse',
        'Esophagus_Gastroesophageal_Junction', 'Esophagus_Mucosa',
        'Esophagus_Muscularis', 'Heart_Atrial_Appendage', 'Heart_Left_Ventricle',
        'Kidney_Cortex', 'Liver', 'Lung', 'Minor_Salivary_Gland',
        'Muscle_Skeletal', 'Nerve_Tibial', 'Ovary', 'Pancreas', 'Pituitary',
        'Prostate', 'Skin_Not_Sun_Exposed_Suprapubic', 'Skin_Sun_Exposed_Lower_leg',
        'Small_Intestine_Terminal_Ileum', 'Spleen', 'Stomach', 'Testis',
        'Thyroid', 'Uterus', 'Vagina', 'Whole_Blood'
    ]
    
    # Chromosomes 1-22 (autosomes)
    chromosomes = [str(i) for i in range(1, 23)]
    
    # Calculate expected total
    expected_total = len(phenotypes) * len(folds) * len(tissues) * len(chromosomes)
    
    print(f"[INFO] Using {len(phenotypes)} phenotype(s): {', '.join(phenotypes)}")
    print(f"[INFO] Using {len(folds)} fold(s): {', '.join(folds)}")
    print(f"[INFO] Using {len(tissues)} GTEx tissues")
    print(f"[INFO] Using {len(chromosomes)} chromosomes (1-22)")
    print(f"[INFO] Expected combinations: {len(phenotypes)} × {len(folds)} × {len(tissues)} × {len(chromosomes)} = {expected_total:,}")
    
    # Generate all combinations
    combinations = list(itertools.product(phenotypes, folds, tissues, chromosomes))
    
    if len(combinations) == expected_total:
        print(f"[INFO] ✓ Generated {len(combinations):,} total combinations (as expected)")
    else:
        print(f"[WARNING] Generated {len(combinations):,} combinations, expected {expected_total:,}")
    
    # Write individual parameter files
    print("[INFO] Writing parameter files...")
    
    with open('all_phenotypes.txt', 'w') as f:
        for combo in combinations:
            f.write(f"{combo[0]}\n")
    
    with open('all_folds.txt', 'w') as f:
        for combo in combinations:
            f.write(f"{combo[1]}\n")
    
    with open('all_tissues.txt', 'w') as f:
        for combo in combinations:
            f.write(f"{combo[2]}\n")
    
    with open('all_chromosomes.txt', 'w') as f:
        for combo in combinations:
            f.write(f"{combo[3]}\n")
    
    # Write complete combinations file
    with open('all_combinations.txt', 'w') as f:
        f.write("phenotype\tfold\ttissue\tchromosome\n")  # header
        for combo in combinations:
            f.write(f"{combo[0]}\t{combo[1]}\t{combo[2]}\t{combo[3]}\n")
    
    print(f"[INFO] ✓ Created parameter files:")
    print(f"  - all_phenotypes.txt ({len(combinations):,} entries)")
    print(f"  - all_folds.txt ({len(combinations):,} entries)")  
    print(f"  - all_tissues.txt ({len(combinations):,} entries)")
    print(f"  - all_chromosomes.txt ({len(combinations):,} entries)")
    print(f"  - all_combinations.txt (tab-separated with header)")
    
    print(f"\n[INFO] Final Summary:")
    print(f"  - Parameters: {len(phenotypes)} phenotype(s), {len(folds)} fold(s), {len(tissues)} tissues, {len(chromosomes)} chromosomes")
    print(f"  - Calculation: {len(phenotypes)} × {len(folds)} × {len(tissues)} × {len(chromosomes)} = {len(combinations):,} combinations")
    
    # Show parameter details
    print(f"\n[INFO] Parameter Details:")
    print(f"  - Phenotypes: {phenotypes}")
    print(f"  - Folds: {folds}")
    print(f"  - Tissues: {len(tissues)} GTEx tissues (first 5: {tissues[:5]})")
    print(f"  - Chromosomes: 1-22")

def main():
    """Main function"""
    print("Dynamic Parameter File Writer for TIGAR")
    print("="*55)
    print("Generating parameter combinations based on actual values...")
    print()
    
    write_parameter_combinations()
    
    print(f"\n[INFO] Parameter files written successfully!")
    print(f"[INFO] Ready to use with TIGAR pipeline!")

if __name__ == "__main__":
    main()