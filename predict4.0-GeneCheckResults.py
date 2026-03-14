#!/usr/bin/env python3
"""
Simple script to generate a completion heatmap showing which database-tissue combinations
have ALL folds, datasets, and methods completed (8 methods).
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def check_completion_status(phenotype):
    """
    Check completion status for all database-tissue combinations
    Returns 1 if ALL folds, datasets, and methods are completed, 0 otherwise
    """
    
    # Define expected components - UPDATED TO 8 METHODS
    databases = ["Regular", "JTI", "UTMOST", "UTMOST2", "EpiX", "TIGAR", "FUSION"]
    methods = [
        "LIMMA",
        "Welch_t_test",
        "Linear_Regression",
        "Wilcoxon_Rank_Sum",
        "Permutation_Test",
        "Weighted_Logistic",
        "Firth_Logistic",
        "Bayesian_Logistic"
    ]
    datasets = ["training", "validation", "test"]
    
    phenotype_path = Path(phenotype)
    
    if not phenotype_path.exists():
        print(f"❌ Phenotype directory {phenotype} not found!")
        return None
    
    # Find all available folds
    fold_dirs = [d for d in phenotype_path.iterdir() if d.is_dir() and d.name.startswith("Fold_")]
    if not fold_dirs:
        print(f"❌ No Fold_ directories found in {phenotype}")
        return None
    
    folds = [d.name.replace("Fold_", "") for d in fold_dirs]
    print(f"📁 Found folds: {folds}")
    
    # Find all available tissues by scanning across all databases and folds
    all_tissues = set()
    available_databases = set()
    
    for fold_dir in fold_dirs:
        expr_dir = fold_dir / "GeneDifferentialExpressions"
        if expr_dir.exists():
            for db_dir in expr_dir.iterdir():
                if db_dir.is_dir() and db_dir.name in databases:
                    available_databases.add(db_dir.name)
                    for tissue_dir in db_dir.iterdir():
                        if tissue_dir.is_dir():
                            all_tissues.add(tissue_dir.name)
    
    all_tissues = sorted(list(all_tissues))
    available_databases = sorted(list(available_databases))
    
    print(f"🧪 Found tissues: {len(all_tissues)} tissues")
    print(f"🗃️  Found databases: {available_databases}")
    print(f"📊 Expected methods: {len(methods)} methods (8 total)")
    print(f"📂 Expected datasets: {len(datasets)} datasets")
    
    # List the 8 methods
    print(f"\n📋 8 Statistical Methods:")
    for i, method in enumerate(methods, 1):
        marker = "✅" if method in ["LIMMA", "Welch_t_test", "Linear_Regression", 
                                    "Wilcoxon_Rank_Sum", "Permutation_Test"] else "⚠️"
        print(f"   {i}. {marker} {method}")
    print()
    
    # Create completion matrix
    completion_matrix = []
    
    for tissue in all_tissues:
        tissue_row = []
        
        for database in databases:
            if database not in available_databases:
                # Database not available at all
                tissue_row.append(0)
                continue
                
            # Check if ALL folds, datasets, and methods are completed for this db-tissue combo
            total_expected = len(folds) * len(datasets) * len(methods)
            completed_count = 0
            
            for fold in folds:
                fold_path = phenotype_path / f"Fold_{fold}" / "GeneDifferentialExpressions" / database / tissue
                
                if not fold_path.exists():
                    continue  # This fold-db-tissue combo doesn't exist
                
                for dataset in datasets:
                    for method in methods:
                        method_dir = fold_path / f"{method}_{dataset}"
                        result_file = method_dir / "differential_expression_results.csv"
                        summary_file = method_dir / "analysis_summary.csv"
                        
                        # Check if both required files exist and are non-empty
                        if (result_file.exists() and summary_file.exists() and 
                            result_file.stat().st_size > 0 and summary_file.stat().st_size > 0):
                            completed_count += 1
            
            # Mark as complete (1) only if ALL expected combinations are done
            completion_status = 1 if completed_count == total_expected else 0
            tissue_row.append(completion_status)
            
            # Debug info for first few combinations
            if len(completion_matrix) < 3:  # Only for first 3 tissues
                print(f"  {database}-{tissue}: {completed_count}/{total_expected} completed -> {completion_status}")
        
        completion_matrix.append(tissue_row)
    
    # Convert to DataFrame
    completion_df = pd.DataFrame(completion_matrix, 
                                index=all_tissues, 
                                columns=databases)
    
    return completion_df

def create_completion_heatmap(completion_df, phenotype, output_dir=None):
    """Create and save completion heatmap"""
    
    if completion_df is None or completion_df.empty:
        print("❌ No data to plot")
        return
    
    # Calculate summary statistics
    total_combinations = completion_df.size
    completed_combinations = completion_df.sum().sum()
    completion_rate = (completed_combinations / total_combinations) * 100
    
    print(f"\n📊 COMPLETION SUMMARY (8 METHODS):")
    print(f"  Total combinations: {total_combinations}")
    print(f"  Completed combinations: {completed_combinations}")
    print(f"  Completion rate: {completion_rate:.1f}%")
    
    # Create heatmap
    plt.figure(figsize=(12, max(8, len(completion_df) * 0.4)))
    
    # Use a custom colormap: white for 0, green for 1
    colors = ['white', 'darkgreen']
    n_bins = 2
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    
    sns.heatmap(completion_df, 
                annot=True, 
                fmt='d',
                cmap=cmap,
                cbar_kws={'label': 'Completion Status (1=All Done, 0=Incomplete)'},
                linewidths=1,
                linecolor='gray',
                square=False,
                vmin=0,
                vmax=1)
    
    plt.title(f'Analysis Completion Status: {phenotype} (8 Methods)\n'
              f'Completed: {completed_combinations}/{total_combinations} combinations ({completion_rate:.1f}%)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Expression Prediction Database', fontsize=12)
    plt.ylabel('Tissue', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save heatmap
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        heatmap_file = output_path / f'completion_heatmap_{phenotype}_8methods.png'
    else:
        heatmap_file = f'completion_heatmap_{phenotype}_8methods.png'
    
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    print(f"🔥 Heatmap saved: {heatmap_file}")
    
    # Save completion matrix as CSV
    if output_dir:
        csv_file = output_path / f'completion_matrix_{phenotype}_8methods.csv'
    else:
        csv_file = f'completion_matrix_{phenotype}_8methods.csv'
    
    completion_df.to_csv(csv_file)
    print(f"📊 Completion matrix saved: {csv_file}")
    
    # Show plot
    plt.show()
    
    return heatmap_file, csv_file

def print_detailed_status(completion_df, phenotype):
    """Print detailed completion status"""
    
    print(f"\n📋 DETAILED COMPLETION STATUS FOR {phenotype.upper()} (8 METHODS)")
    print("="*80)
    
    # Database-wise completion
    db_completion = completion_df.sum(axis=0)
    total_tissues = len(completion_df)
    
    print("🗃️  DATABASE COMPLETION:")
    for db, completed_tissues in db_completion.items():
        completion_pct = (completed_tissues / total_tissues) * 100
        print(f"  {db}: {completed_tissues}/{total_tissues} tissues ({completion_pct:.1f}%)")
    
    # Tissue-wise completion
    tissue_completion = completion_df.sum(axis=1)
    total_databases = len(completion_df.columns)
    
    print(f"\n🧪 TISSUE COMPLETION (Top 10 most complete):")
    top_tissues = tissue_completion.sort_values(ascending=False).head(10)
    for tissue, completed_dbs in top_tissues.items():
        completion_pct = (completed_dbs / total_databases) * 100
        print(f"  {tissue}: {completed_dbs}/{total_databases} databases ({completion_pct:.1f}%)")
    
    # Incomplete combinations
    incomplete_combinations = []
    for tissue in completion_df.index:
        for database in completion_df.columns:
            if completion_df.loc[tissue, database] == 0:
                incomplete_combinations.append(f"{database}-{tissue}")
    
    if incomplete_combinations:
        print(f"\n❌ INCOMPLETE COMBINATIONS ({len(incomplete_combinations)} total):")
        for i, combo in enumerate(incomplete_combinations[:20]):  # Show first 20
            print(f"  {i+1:2d}. {combo}")
        if len(incomplete_combinations) > 20:
            print(f"  ... and {len(incomplete_combinations) - 20} more")
    else:
        print("\n✅ ALL COMBINATIONS COMPLETED! (ALL 8 METHODS)")

def main():
    parser = argparse.ArgumentParser(
        description="Generate completion heatmap for differential expression analysis (8 methods)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
8 STATISTICAL METHODS CHECKED:
  ✅ TRUE DIFFERENTIAL EXPRESSION:
     1. LIMMA (gold standard)
     2. Welch's t-test
     3. Linear Regression
     4. Wilcoxon Rank-Sum
     5. Permutation Test
  
  ⚠️  ASSOCIATION TESTING:
     6. Weighted Logistic
     7. Firth Logistic
     8. Bayesian Logistic

USAGE:
  python completion_check.py migraine
  python completion_check.py migraine --output results/ --detailed

OUTPUT:
  - Completion heatmap (PNG)
  - Completion matrix (CSV)
  - Optional detailed status report
        """
    )
    parser.add_argument("phenotype", help="Phenotype name (e.g., migraine)")
    parser.add_argument("--output", help="Output directory for saving files")
    parser.add_argument("--detailed", action="store_true", help="Show detailed completion status")
    
    args = parser.parse_args()
    
    print(f"🔍 CHECKING COMPLETION STATUS FOR: {args.phenotype} (8 METHODS)")
    print("="*80)
    
    # Check completion status
    completion_df = check_completion_status(args.phenotype)
    
    if completion_df is not None:
        # Create heatmap
        create_completion_heatmap(completion_df, args.phenotype, args.output)
        
        # Show detailed status if requested
        if args.detailed:
            print_detailed_status(completion_df, args.phenotype)
        
        print(f"\n✅ Completion check finished!")
        print(f"🔥 All 8 statistical methods were checked")
    
if __name__ == "__main__":
    main()