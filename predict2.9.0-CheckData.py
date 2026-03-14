#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if len(sys.argv) != 2:
    print("Usage: python3 check_data.py <phenotype>")
    sys.exit(1)

phenotype = sys.argv[1]

# Model directory mappings
MODEL_DIRS = {
    'Regular': {'train': 'TrainExpression', 'test': 'TestExpression', 'validation': 'ValidationExpression'},
    'JTI': {'train': 'JTITrainExpression', 'test': 'JTITestExpression', 'validation': 'JTIValidationExpression'},
    'UTMOST': {'train': 'UTMOSTTrainExpression', 'test': 'UTMOSTTestExpression', 'validation': 'UTMOSTValidationExpression'},
    'UTMOST2': {'train': 'utmost2TrainExpression', 'test': 'utmost2TestExpression', 'validation': 'utmost2ValidationExpression'},
    'EpiX': {'train': 'EpiXTrainExpression', 'test': 'EpiXTestExpression', 'validation': 'EpiXValidationExpression'},
    'TIGAR': {'train': 'TigarTrainExpression', 'test': 'TigarTestExpression', 'validation': 'TigarValidExpression'},
    'FUSION': {'train': 'FussionExpression', 'test': 'FussionExpression', 'validation': 'FussionExpression'}
}

# File name mappings for FUSION (same directory, different files)
FILE_NAMES = {
    'FUSION': {'train': 'GeneExpression_train_data.csv', 'test': 'GeneExpression_test_data.csv', 'validation': 'GeneExpression_validation_data.csv'}
}

def get_gene_count(file_path):
    """Count genes in a file (columns except FID/IID)"""
    try:
        df = pd.read_csv(file_path, nrows=0)
        gene_cols = [col for col in df.columns if col not in ['FID', 'IID']]
        return len(gene_cols)
    except:
        return 0
def save_gene_counts_csv(gene_counts, tissues, models, split_type, phenotype):
    """Save gene count matrix as CSV"""

    df = pd.DataFrame(
        gene_counts,
        index=tissues,
        columns=models
    )

    df.index.name = "Tissue"

    output_file = f'predict-2.9.0-CheckData-{split_type.capitalize()}.csv'
    df.to_csv(output_file)

    print(f"? Saved CSV: {output_file}")

def get_all_tissues(phenotype):
    """Get all unique tissues across all models"""
    tissues = set()
    fold_dir = f"{phenotype}/Fold_1"
    
    if not os.path.exists(fold_dir):
        print(f"ERROR: {fold_dir} not found")
        return []
    
    for model, dirs in MODEL_DIRS.items():
        dir_name = dirs['train']
        expr_path = os.path.join(fold_dir, dir_name)
        if os.path.exists(expr_path):
            for item in os.listdir(expr_path):
                if os.path.isdir(os.path.join(expr_path, item)):
                    tissues.add(item)
    
    return sorted(list(tissues))

def create_gene_count_matrix(phenotype, split_type):
    """Create matrix of gene counts: tissues (rows) x models (columns)"""
    
    tissues = get_all_tissues(phenotype)
    models = list(MODEL_DIRS.keys())
    
    gene_counts = np.zeros((len(tissues), len(models)))
    
    fold_dir = f"{phenotype}/Fold_1"
    
    print(f"\n{'='*80}")
    print(f"CHECKING {split_type.upper()} DATA")
    print(f"{'='*80}\n")
    
    for i, tissue in enumerate(tissues):
        for j, model in enumerate(models):
            dir_name = MODEL_DIRS[model][split_type]
            tissue_dir = os.path.join(fold_dir, dir_name, tissue)
            
            print(f"Checking: {tissue_dir}")
            
            if os.path.exists(tissue_dir):
                # Handle FUSION (specific file names in same directory)
                if model == 'FUSION':
                    file_name = FILE_NAMES['FUSION'][split_type]
                    file_path = os.path.join(tissue_dir, file_name)
                    if os.path.exists(file_path):
                        gene_count = get_gene_count(file_path)
                        gene_counts[i, j] = gene_count
                        print(f"  -> Found: {file_name} ({gene_count} genes)")
                    else:
                        print(f"  -> File not found: {file_name}")
                else:
                    # Other models: find any CSV file
                    csv_files = [f for f in os.listdir(tissue_dir) if f.endswith('.csv')]
                    if csv_files:
                        csv_file = csv_files[0]
                        file_path = os.path.join(tissue_dir, csv_file)
                        gene_count = get_gene_count(file_path)
                        gene_counts[i, j] = gene_count
                        print(f"  -> Found: {csv_file} ({gene_count} genes)")
                    else:
                        print(f"  -> No CSV files")
            else:
                print(f"  -> Directory not found")
    
    return gene_counts, tissues, models

def plot_gene_counts(gene_counts, tissues, models, split_type, phenotype):
    """Create and save heatmap"""
    
    plt.figure(figsize=(12, max(8, len(tissues) * 0.3)))
    
    mask = gene_counts == 0
    
    sns.heatmap(gene_counts, 
                mask=mask,
                annot=True, 
                fmt='.0f',
                xticklabels=models,
                yticklabels=tissues,
                cmap='YlOrRd',
                cbar_kws={'label': 'Number of Genes'},
                linewidths=0.5,
                linecolor='gray')
    
    plt.title(f'Gene Counts: {split_type.capitalize()} Data - {phenotype.capitalize()}', 
              fontsize=14, pad=20)
    plt.xlabel('Gene Expression Models', fontsize=12)
    plt.ylabel('Tissue Types', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    output_file = f'predict-2.9.0-CheckData-{split_type.capitalize()}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n? Saved: {output_file}\n")
    plt.close()

# Main execution
for split_type in ['train', 'test', 'validation']:
    gene_counts, tissues, models = create_gene_count_matrix(phenotype, split_type)

    # Save CSV (NEW)
    save_gene_counts_csv(gene_counts, tissues, models, split_type, phenotype)

    # Save heatmap
    plot_gene_counts(gene_counts, tissues, models, split_type, phenotype)


print("="*80)
print("DONE!")
print("="*80)