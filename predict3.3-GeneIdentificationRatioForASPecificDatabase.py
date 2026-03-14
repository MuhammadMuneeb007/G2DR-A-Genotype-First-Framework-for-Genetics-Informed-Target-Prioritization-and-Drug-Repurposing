#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Migraine Gene Count Analysis

This script provides a straightforward analysis:
- Load migraine gene list
- Load ML feature importance files (genes with non-zero weights)
- Count how many migraine genes each tissue identified
- Create simple heatmap visualization

Usage: python simple_migraine_count.py <phenotype> <expression_method>
Example: python simple_migraine_count.py migraine JTI
"""

import os
import sys
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Check command line arguments
if len(sys.argv) != 3:
    print("Usage: python simple_migraine_count.py <phenotype> <expression_methods>")
    print("Example: python simple_migraine_count.py migraine JTI")
    print("Example (multiple): python simple_migraine_count.py migraine Regular,JTI,UTMOST")
    print("\nAvailable expression methods:")
    print("  - Regular, JTI, UTMOST, UTMOST2, EpiX, TIGAR, FUSION")
    print("\n⚠️  Make sure EpiX data exists in EpiXTrainExpression directories")
    sys.exit(1)

PHENOTYPE = sys.argv[1]
EXPRESSION_METHODS = [method.strip() for method in sys.argv[2].split(',')]

print(f"🧬 Multi-Model Migraine Gene Count Analysis")
print(f"📊 Phenotype: {PHENOTYPE}")
print(f"🔬 Expression Methods: {', '.join(EXPRESSION_METHODS)}")
print("="*60)

def load_migraine_genes(migraine_file="migraine_genes.csv"):
    """Load the migraine gene list"""
    
    if not os.path.exists(migraine_file):
        print(f"❌ Migraine gene file not found: {migraine_file}")
        return None
    
    try:
        migraine_df = pd.read_csv(migraine_file)
        print(f"✅ Loaded {len(migraine_df)} migraine genes from {migraine_file}")
        
        # Create set of ensembl IDs for quick lookup
        migraine_genes = set(migraine_df['ensembl_gene_id'].tolist())
        
        return migraine_genes, migraine_df
        
    except Exception as e:
        print(f"❌ Error loading migraine genes: {e}")
        return None

def find_best_feature_files(expression_method):
    """Find all best feature files in the BestCombination_Features directory for a specific method"""
    
    features_dir = f"{PHENOTYPE}/Database/{expression_method}/BestCombination_Features/"
    
    if not os.path.exists(features_dir):
        print(f"❌ Best features directory not found: {features_dir}")
        if expression_method == "EpiX":
            print(f"   Note: EpiX directory should be named 'EpiX' (case sensitive)")
        print("   Please run the main merge script first to generate these files.")
        return []
    
    # Find all feature CSV files
    pattern = os.path.join(features_dir, "*_features.csv")
    feature_files = glob.glob(pattern)
    
    print(f"✅ Found {len(feature_files)} best feature files for {expression_method}")
    if expression_method == "EpiX" and len(feature_files) == 0:
        print(f"⚠️  No EpiX feature files found. Check directory structure:")
        print(f"     Expected: {features_dir}")
    
    return feature_files

def load_test_performance(expression_method):
    """Load test performance data from best performers table for a specific method"""
    
    best_file = f"{PHENOTYPE}/Database/{expression_method}/{PHENOTYPE}_{expression_method}_BestPerTissue_TestBased.csv"
    
    if not os.path.exists(best_file):
        print(f"⚠️  Test performance file not found: {best_file}")
        print("   Will proceed without test performance data")
        return None
    
    try:
        test_df = pd.read_csv(best_file)
        print(f"✅ Loaded test performance for {len(test_df)} tissues ({expression_method})")
        
        # Create dictionary for quick lookup: tissue -> test_auc
        test_performance = {}
        for _, row in test_df.iterrows():
            test_performance[row['Tissue']] = row['Test_AUC']
        
        return test_performance
        
    except Exception as e:
        print(f"⚠️  Error loading test performance: {e}")
        return None

def parse_tissue_info_from_filename(filename):
    """Extract tissue name, ML method, and feature count from filename"""
    
    basename = os.path.basename(filename)
    # Format: Tissue_MLMethod_FeatureCountF_features.csv
    
    if not basename.endswith('_features.csv'):
        return None, None, None
    
    # Remove '_features.csv'
    name_part = basename.replace('_features.csv', '')
    
    # Split by underscores
    parts = name_part.split('_')
    
    if len(parts) < 3:
        return None, None, None
    
    # Last part should be like "100F"
    feature_part = parts[-1]
    if not feature_part.endswith('F'):
        return None, None, None
    
    try:
        feature_count = int(feature_part[:-1])  # Remove 'F' and convert to int
    except:
        return None, None, None
    
    # Second to last should be ML method
    ml_method = parts[-2]
    
    # Everything else is tissue name
    tissue = '_'.join(parts[:-2])
    
    return tissue, ml_method, feature_count

def load_important_genes(feature_file):
    """Load genes identified as important (non-zero weights) from feature file"""
    
    try:
        df = pd.read_csv(feature_file)
        
        if 'Feature_Name' not in df.columns or 'Feature_Importance' not in df.columns:
            return None
        
        # Filter out genes with exactly zero weights (Feature_Importance == 0)
        important_df = df[df['Feature_Importance'] != 0.0].copy()
        
        print(f"   📊 Total genes in file: {len(df)}")
        print(f"   🎯 Genes with non-zero weights: {len(important_df)}")
        print(f"   ❌ Genes with zero weights: {len(df) - len(important_df)}")
        
        # Clean feature names - handle TIGAR format (chr*_ENSG*) and remove version numbers
        important_df['Feature_Name_Clean'] = important_df['Feature_Name'].str.replace(r'^chr\d+_', '', regex=True).str.replace(r'^chrX_', '', regex=True).str.replace(r'^chrY_', '', regex=True).str.split('.').str[0]
        
        # Return set of clean feature names (only non-zero weight genes)
        important_genes = set(important_df['Feature_Name_Clean'].tolist())
        
        return important_genes
        
    except Exception as e:
        print(f"   ❌ Error reading {os.path.basename(feature_file)}: {e}")
        return None

def analyze_migraine_gene_counts(feature_files, migraine_genes, test_performance, expression_method):
    """Count migraine genes identified by each tissue and calculate identification ratio"""
    
    print(f"\n🧪 Analyzing migraine gene identification ratios for {expression_method}...")
    print("="*60)
    
    results = []
    
    for feature_file in feature_files:
        filename = os.path.basename(feature_file)
        
        # Extract tissue name, method, and feature count
        tissue, ml_method, feature_count = parse_tissue_info_from_filename(feature_file)
        
        if tissue is None:
            print(f"❌ Could not parse tissue from filename: {filename}")
            continue
        
        print(f"🔬 Analyzing {tissue} ({ml_method}, {feature_count} features)...")
        
        # Load important genes (genes with non-zero weights)
        important_genes = load_important_genes(feature_file)
        
        if important_genes is None:
            print(f"   ❌ Could not load genes from {filename}")
            
            # Get test performance
            test_auc = test_performance.get(tissue, None) if test_performance else None
            
            results.append({
                'Tissue': tissue,
                'ML_Method': ml_method,
                'Feature_Count': feature_count,
                'Total_Important_Genes': 0,
                'Migraine_Genes_Found': 0,
                'Gene_Identification_Ratio': 0.0,  # migraine_genes / genes_passed_to_ML
                'Test_AUC': test_auc,
                'Expression_Method': expression_method,
                'Status': 'No Data'
            })
            continue
        
        # Count overlap with migraine genes
        migraine_overlap = important_genes.intersection(migraine_genes)
        migraine_count = len(migraine_overlap)
        total_important = len(important_genes)
        
        # Calculate Gene Identification Ratio: migraine_genes_found / genes_passed_to_ML
        gene_id_ratio = (migraine_count / feature_count * 100) if feature_count > 0 else 0
        
        # Get test performance
        test_auc = test_performance.get(tissue, None) if test_performance else None
        
        results.append({
            'Tissue': tissue,
            'ML_Method': ml_method,
            'Feature_Count': feature_count,
            'Total_Important_Genes': total_important,
            'Migraine_Genes_Found': migraine_count,
            'Gene_Identification_Ratio': gene_id_ratio,  # This is the key ratio!
            'Test_AUC': test_auc,
            'Expression_Method': expression_method,
            'Status': 'Success'
        })
        
        print(f"   ✅ Genes passed to ML: {feature_count}")
        print(f"   🎯 Important genes selected: {total_important}")
        print(f"   🧬 Migraine genes found: {migraine_count}")
        print(f"   📊 Gene ID Ratio: {migraine_count}/{feature_count} = {gene_id_ratio:.2f}%")
        if test_auc is not None:
            print(f"   🎯 Test AUC: {test_auc:.4f}")
        
        # Show some examples if any found
        if migraine_count > 0:
            example_genes = sorted(list(migraine_overlap))[:3]
            print(f"   📋 Examples: {', '.join(example_genes)}")
    
    return results

def create_results_dataframe(results):
    """Create and sort results DataFrame"""
    
    df = pd.DataFrame(results)
    
    if len(df) == 0:
        return df
    
    # Sort alphabetically by tissue name
    df = df.sort_values('Tissue')
    
    print(f"\n📊 SUMMARY:")
    print(f"   Total tissues analyzed: {len(df)}")
    print(f"   Tissues with migraine genes: {len(df[df['Migraine_Genes_Found'] > 0])}")
    print(f"   Average Gene ID Ratio: {df['Gene_Identification_Ratio'].mean():.2f}%")
    print(f"   Average Test AUC: {df['Test_AUC'].mean():.4f}")
    
    # Show best performer by Gene ID Ratio
    best_idx = df['Gene_Identification_Ratio'].idxmax()
    best_tissue = df.loc[best_idx, 'Tissue']
    best_ratio = df['Gene_Identification_Ratio'].max()
    
    print(f"   Best Gene ID Ratio: {best_tissue} ({best_ratio:.2f}%)")
    
    return df

def generate_simple_heatmap(results_df, migraine_genes, output_dir, expression_method):
    """Generate simple heatmap showing gene identification ratio and test AUC"""
    
    print(f"\n🔥 Generating gene identification ratio heatmap for {expression_method}...")
    
    if len(results_df) == 0:
        print("❌ No data for heatmap")
        return None
    
    # Sort tissues alphabetically
    tissue_order = sorted(results_df['Tissue'].tolist())
    
    # Create matrix with Gene ID Ratio and Test AUC
    heatmap_data = []
    for _, row in results_df.iterrows():
        heatmap_data.append({
            'Tissue': row['Tissue'],
            'Gene_ID_Ratio': row['Gene_Identification_Ratio'],
            'Test_AUC': row['Test_AUC'] if pd.notna(row['Test_AUC']) else 0.0
        })
    
    heatmap_df = pd.DataFrame(heatmap_data)
    heatmap_df = heatmap_df.set_index('Tissue')
    heatmap_df = heatmap_df.reindex(tissue_order)  # Sort alphabetically
    
    # Create custom annotations showing the actual numbers
    annot_df = pd.DataFrame(index=tissue_order, columns=['Gene_ID_Ratio', 'Test_AUC'])
    
    for tissue in tissue_order:
        row_data = results_df[results_df['Tissue'] == tissue]
        if len(row_data) > 0:
            row = row_data.iloc[0]
            # Show ratio as "migraine_found/total_passed"
            annot_df.loc[tissue, 'Gene_ID_Ratio'] = f"{row['Migraine_Genes_Found']}/{row['Feature_Count']}"
            # Show test AUC
            if pd.notna(row['Test_AUC']):
                annot_df.loc[tissue, 'Test_AUC'] = f"{row['Test_AUC']:.3f}"
            else:
                annot_df.loc[tissue, 'Test_AUC'] = "N/A"
        else:
            annot_df.loc[tissue, 'Gene_ID_Ratio'] = "0/0"
            annot_df.loc[tissue, 'Test_AUC'] = "N/A"
    
    # Create the heatmap
    plt.figure(figsize=(8, max(10, len(tissue_order) * 0.4)))
    
    # Generate heatmap
    ax = sns.heatmap(
        heatmap_df, 
        annot=annot_df,  # Use custom annotations
        fmt='',  # Don't format since we're using custom strings
        cmap='RdYlBu_r',
        center=None,
        cbar_kws={'label': 'Gene ID Ratio (%) / Test AUC'},
        linewidths=0.5,
        square=False,
        xticklabels=['Gene ID Ratio\n(Migraine/Total)', 'Test AUC\n(Performance)']
    )
    
    # Customize the plot
    plt.title(f'{PHENOTYPE} - Gene Identification & Test Performance\n({expression_method} Gene Expression Model - Alphabetical Order)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Metrics', fontsize=12, fontweight='bold')
    plt.ylabel('Tissue (Alphabetical Order)', fontsize=12, fontweight='bold')
    
    # Adjust text
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the heatmap
    heatmap_file = os.path.join(output_dir, f"{PHENOTYPE}_{expression_method}_GeneIDRatio_TestAUC_Heatmap.png")
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    
    print(f"   ✅ Gene ID ratio heatmap saved: {heatmap_file}")
    
    #plt.show()
    plt.close()
    
    return heatmap_file

def generate_detailed_heatmap(results_df, migraine_genes, output_dir):
    """Generate detailed heatmap with gene ID ratio, test AUC, and additional metrics"""
    
    print(f"\n🔥 Generating detailed metrics heatmap...")
    
    if len(results_df) == 0:
        print("❌ No data for heatmap")
        return None
    
    # Sort tissues alphabetically
    tissue_order = sorted(results_df['Tissue'].tolist())
    
    # Prepare detailed data - scale test AUC for better visualization
    detailed_data = []
    for _, row in results_df.iterrows():
        test_auc_scaled = (row['Test_AUC'] * 100) if pd.notna(row['Test_AUC']) else 0.0
        detailed_data.append({
            'Tissue': row['Tissue'],
            'Gene_ID_Ratio': row['Gene_Identification_Ratio'],
            'Test_AUC_Scaled': test_auc_scaled,  # Scale to percentage for better viz
            'Total_Important_Scaled': row['Total_Important_Genes'] / 100  # Scale for viz
        })
    
    detailed_df = pd.DataFrame(detailed_data)
    detailed_df = detailed_df.set_index('Tissue')
    detailed_df = detailed_df.reindex(tissue_order)
    
    # Final matrix for values (for coloring)
    final_matrix = detailed_df[['Gene_ID_Ratio', 'Test_AUC_Scaled', 'Total_Important_Scaled']].copy()
    
    # Create custom annotations
    annot_matrix = pd.DataFrame(index=tissue_order, columns=['Gene_ID_Ratio', 'Test_AUC_Scaled', 'Total_Important_Scaled'])
    
    for tissue in tissue_order:
        row_data = results_df[results_df['Tissue'] == tissue]
        if len(row_data) > 0:
            row = row_data.iloc[0]
            # Gene ID ratio as fraction
            annot_matrix.loc[tissue, 'Gene_ID_Ratio'] = f"{row['Migraine_Genes_Found']}/{row['Feature_Count']}"
            # Test AUC 
            if pd.notna(row['Test_AUC']):
                annot_matrix.loc[tissue, 'Test_AUC_Scaled'] = f"{row['Test_AUC']:.3f}"
            else:
                annot_matrix.loc[tissue, 'Test_AUC_Scaled'] = "N/A"
            # Total important (scaled)
            annot_matrix.loc[tissue, 'Total_Important_Scaled'] = f"{row['Total_Important_Genes']/100:.1f}"
        else:
            annot_matrix.loc[tissue, 'Gene_ID_Ratio'] = "0/0"
            annot_matrix.loc[tissue, 'Test_AUC_Scaled'] = "N/A"
            annot_matrix.loc[tissue, 'Total_Important_Scaled'] = "0.0"
    
    # Create the heatmap
    plt.figure(figsize=(12, max(10, len(tissue_order) * 0.4)))
    
    # Generate heatmap
    ax = sns.heatmap(
        final_matrix, 
        annot=annot_matrix,  # Use custom annotations
        fmt='', 
        cmap='RdYlBu_r',
        center=None,
        cbar_kws={'label': 'Gene ID Ratio (%) / Test AUC×100 / Important Genes÷100'},
        linewidths=0.5,
        square=False,
        xticklabels=['Gene ID Ratio\n(Migraine/Total)', 'Test AUC\n(Performance)', 'Important Genes\n(/100)']
    )
    
    # Customize the plot
    plt.title(f'{PHENOTYPE} - Detailed Gene Identification Analysis\n({EXPRESSION_METHOD} Gene Expression Model - Alphabetical Order)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Metrics', fontsize=12, fontweight='bold')
    plt.ylabel('Tissue (Alphabetical Order)', fontsize=12, fontweight='bold')
    
    # Rotate labels for readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the heatmap
    detailed_heatmap_file = os.path.join(output_dir, f"{PHENOTYPE}_{EXPRESSION_METHOD}_DetailedGeneID_Heatmap.png")
    plt.savefig(detailed_heatmap_file, dpi=300, bbox_inches='tight')
    
    print(f"   ✅ Detailed heatmap saved: {detailed_heatmap_file}")
    
    #plt.show()
    plt.close()
    
    return detailed_heatmap_file

def generate_cross_database_heatmap(all_results_df, migraine_genes, output_dir):
    """Generate cross-model heatmap with separate columns for Gene ID Ratio and Test AUC for each model"""
    
    print(f"\n{'🔥' * 50}")
    print("🔥 GENERATING CROSS-MODEL HEATMAP WITH SEPARATE METRIC COLUMNS...")
    print(f"{'🔥' * 50}")
    
    if len(all_results_df) == 0:
        print("❌ No data for cross-model heatmap")
        return None
    
    # Get unique tissues and expression methods
    all_tissues = sorted(all_results_df['Tissue'].unique())
    all_methods = sorted(all_results_df['Expression_Method'].unique())
    
    print(f"📊 Cross-model heatmap: {len(all_tissues)} tissues × {len(all_methods)} models × 2 metrics")
    
    # Create multi-level column structure: (Gene Expression Model, Metric)
    columns = []
    for method in all_methods:
        columns.append((method, 'Gene_ID_Ratio'))
        columns.append((method, 'Test_AUC'))
    
    # Create DataFrame with multi-level columns
    multi_columns = pd.MultiIndex.from_tuples(columns, names=['Gene Expression Model', 'Metric'])
    display_matrix = pd.DataFrame(index=all_tissues, columns=multi_columns)
    annotation_matrix = pd.DataFrame(index=all_tissues, columns=multi_columns)
    
    # Fill the matrices
    for tissue in all_tissues:
        for method in all_methods:
            tissue_method_data = all_results_df[
                (all_results_df['Tissue'] == tissue) & 
                (all_results_df['Expression_Method'] == method)
            ]
            
            if len(tissue_method_data) > 0:
                row = tissue_method_data.iloc[0]
                
                # Gene ID Ratio column
                gene_ratio = row['Gene_Identification_Ratio']
                display_matrix.loc[tissue, (method, 'Gene_ID_Ratio')] = gene_ratio
                annotation_matrix.loc[tissue, (method, 'Gene_ID_Ratio')] = f"{gene_ratio:.1f}%"
                
                # Test AUC column
                if pd.notna(row['Test_AUC']):
                    test_auc = row['Test_AUC']
                    display_matrix.loc[tissue, (method, 'Test_AUC')] = test_auc
                    annotation_matrix.loc[tissue, (method, 'Test_AUC')] = f"{test_auc:.3f}"
                else:
                    display_matrix.loc[tissue, (method, 'Test_AUC')] = np.nan
                    annotation_matrix.loc[tissue, (method, 'Test_AUC')] = "N/A"
            else:
                display_matrix.loc[tissue, (method, 'Gene_ID_Ratio')] = np.nan
                display_matrix.loc[tissue, (method, 'Test_AUC')] = np.nan
                annotation_matrix.loc[tissue, (method, 'Gene_ID_Ratio')] = "N/A"
                annotation_matrix.loc[tissue, (method, 'Test_AUC')] = "N/A"
    
    # Convert to numeric for proper coloring
    display_matrix = display_matrix.astype(float)
    
    # Create custom colormap for different metrics
    # Normalize values for better visualization
    normalized_matrix = display_matrix.copy()
    
    # Normalize Gene ID Ratios (0-100 scale) to 0-1
    for method in all_methods:
        gene_col = (method, 'Gene_ID_Ratio')
        if gene_col in normalized_matrix.columns:
            gene_values = normalized_matrix[gene_col].dropna()
            if len(gene_values) > 0:
                max_gene = max(gene_values.max(), 10)  # At least 10% for scaling
                normalized_matrix[gene_col] = normalized_matrix[gene_col] / max_gene
    
    # Normalize Test AUC (typically 0.5-0.7) to 0-1
    for method in all_methods:
        auc_col = (method, 'Test_AUC')
        if auc_col in normalized_matrix.columns:
            auc_values = normalized_matrix[auc_col].dropna()
            if len(auc_values) > 0:
                min_auc = max(auc_values.min(), 0.5)
                max_auc = min(auc_values.max(), 1.0)
                normalized_matrix[auc_col] = (normalized_matrix[auc_col] - min_auc) / (max_auc - min_auc)
    
    # Create the heatmap
    plt.figure(figsize=(max(20, len(all_methods) * 5), max(12, len(all_tissues) * 0.8)))
    
    mask = normalized_matrix.isna()
    ax = sns.heatmap(
        normalized_matrix,
        annot=annotation_matrix,
        fmt='',
        cmap='RdYlBu_r',
        center=0.5,
        vmin=0,
        vmax=1,
        mask=mask,
        cbar_kws={'label': 'Normalized Performance (0=Low, 1=High)', 'shrink': 0.8},
        linewidths=1.0,
        square=False,
        annot_kws={'fontsize': 12, 'fontweight': 'bold'}
    )
    
    # Customize the plot
    plt.title(f'{PHENOTYPE} - Gene ID Ratio & Test AUC Across Gene Expression Models\n(Separate columns for each metric per model)', 
              fontsize=16, fontweight='bold', pad=25)
    plt.xlabel('Gene Expression Model & Metric', fontsize=14, fontweight='bold')
    plt.ylabel('Tissue (Alphabetical Order)', fontsize=14, fontweight='bold')
    
    # Rotate x-axis labels and make them more readable
    ax.set_xticklabels([f"{model}\n{metric.replace('_', ' ')}" for model, metric in multi_columns], 
                       rotation=45, ha='right', fontsize=11, fontweight='bold')
    plt.yticks(rotation=0, fontsize=11, fontweight='bold')
    
    # Add vertical lines to separate models
    for i in range(1, len(all_methods)):
        plt.axvline(x=i*2, color='white', linewidth=3)
    
    plt.tight_layout()
    
    # Save the heatmap
    heatmap_file = os.path.join(output_dir, f"{PHENOTYPE}_CrossModel_SeparateColumns_GeneIDRatio_TestAUC_Heatmap.png")
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    
    print(f"🔥 Cross-model separate columns heatmap saved: {heatmap_file}")
    
    #plt.show()
    plt.close()
    
    # Save data matrix as CSV
    data_csv = os.path.join(output_dir, f"{PHENOTYPE}_CrossModel_SeparateColumns_Matrix.csv")
    display_matrix.to_csv(data_csv)
    
    print(f"📊 Data matrix saved: {data_csv}")
    
    return heatmap_file

def generate_cross_database_test_auc_heatmap(all_results_df, output_dir):
    """Generate cross-model heatmap comparing test AUC across models"""
    
    print(f"\n🔥 GENERATING CROSS-MODEL TEST AUC HEATMAP...")
    
    if len(all_results_df) == 0:
        print("❌ No data for cross-model test AUC heatmap")
        return None
    
    # Get unique tissues and expression methods
    all_tissues = sorted(all_results_df['Tissue'].unique())
    all_methods = sorted(all_results_df['Expression_Method'].unique())
    
    # Create matrix for test AUC
    auc_matrix = pd.DataFrame(index=all_tissues, columns=all_methods)
    
    for tissue in all_tissues:
        for method in all_methods:
            tissue_method_data = all_results_df[
                (all_results_df['Tissue'] == tissue) & 
                (all_results_df['Expression_Method'] == method)
            ]
            
            if len(tissue_method_data) > 0:
                row = tissue_method_data.iloc[0]
                if pd.notna(row['Test_AUC']):
                    auc_matrix.loc[tissue, method] = row['Test_AUC']
                else:
                    auc_matrix.loc[tissue, method] = np.nan
            else:
                auc_matrix.loc[tissue, method] = np.nan
    
    # Convert to numeric for proper coloring
    auc_matrix = auc_matrix.astype(float)
    
    # Create the heatmap
    plt.figure(figsize=(max(8, len(all_methods) * 2.5), max(10, len(all_tissues) * 0.4)))
    
    # Generate heatmap
    mask = auc_matrix.isna()
    ax = sns.heatmap(
        auc_matrix, 
        annot=True,
        fmt='.3f',
        cmap='RdYlBu_r',
        center=0.55,
        vmin=0.5,
        vmax=0.7,
        mask=mask,
        cbar_kws={'label': 'Test AUC'},
        linewidths=0.5,
        square=False
    )
    
    # Customize the plot
    plt.title(f'{PHENOTYPE} - Test AUC Performance Across Gene Expression Models\n(Best ML Method + Features per Tissue)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Gene Expression Model', fontsize=12, fontweight='bold')
    plt.ylabel('Tissue (Alphabetical Order)', fontsize=12, fontweight='bold')
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the heatmap
    auc_heatmap_file = os.path.join(output_dir, f"{PHENOTYPE}_CrossModel_TestAUC_Heatmap.png")
    plt.savefig(auc_heatmap_file, dpi=300, bbox_inches='tight')
    
    print(f"🔥 Cross-model Test AUC heatmap saved: {auc_heatmap_file}")
    
    #plt.show()
    plt.close()
    
    return auc_heatmap_file

def save_results(results_df, migraine_genes, expression_method, all_results_df=None):
    """Save results to CSV and generate visualizations"""
    
    print(f"\n💾 Saving results for {expression_method}...")
    
    # Create output directory
    output_dir = f"{PHENOTYPE}/Database/{expression_method}/GeneIDRatio/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save main results
    results_file = os.path.join(output_dir, f"{PHENOTYPE}_{expression_method}_GeneIdentificationRatio.csv")
    results_df.to_csv(results_file, index=False)
    
    print(f"   ✅ Results saved: {results_file}")
    
    # Generate heatmap for this expression method
    heatmap_file = generate_simple_heatmap(results_df, migraine_genes, output_dir, expression_method)
    
    # Save summary
    valid_auc = results_df[results_df['Test_AUC'].notna()]
    
    summary = {
        'total_tissues': int(len(results_df)),
        'total_migraine_genes': int(len(migraine_genes)),
        'tissues_with_migraine_genes': int(len(results_df[results_df['Migraine_Genes_Found'] > 0])),
        'max_gene_id_ratio': float(results_df['Gene_Identification_Ratio'].max()),
        'avg_gene_id_ratio': float(results_df['Gene_Identification_Ratio'].mean()),
        'avg_test_auc': float(valid_auc['Test_AUC'].mean()) if len(valid_auc) > 0 else None,
        'best_tissue_by_ratio': results_df.loc[results_df['Gene_Identification_Ratio'].idxmax(), 'Tissue'],
        'phenotype': PHENOTYPE,
        'expression_method': expression_method
    }
    
    import json
    summary_file = os.path.join(output_dir, f"{PHENOTYPE}_{expression_method}_GeneIDRatio_Summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"   ✅ Summary saved: {summary_file}")
    
    # Generate cross-model heatmaps if multiple expression methods and this is the final call
    cross_ratio_heatmap = None
    cross_auc_heatmap = None
    if all_results_df is not None and len(all_results_df['Expression_Method'].unique()) > 1:
        # Create cross-model output directory
        cross_db_output_dir = f"{PHENOTYPE}/Model/CrossModel_Comparison/"
        os.makedirs(cross_db_output_dir, exist_ok=True)
        
        cross_ratio_heatmap = generate_cross_database_heatmap(all_results_df, migraine_genes, cross_db_output_dir)
        cross_auc_heatmap = generate_cross_database_test_auc_heatmap(all_results_df, cross_db_output_dir)
        
        # Save all results to cross-model directory
        all_results_file = os.path.join(cross_db_output_dir, f"{PHENOTYPE}_AllModels_GeneIdentificationRatio.csv")
        all_results_df.to_csv(all_results_file, index=False)
        print(f"   ✅ All results saved: {all_results_file}")
    
    return results_file, summary_file, heatmap_file, cross_ratio_heatmap, cross_auc_heatmap

def display_top_performers(results_df, migraine_genes):
    """Display top performing tissues by Gene ID Ratio"""
    
    print(f"\n🏆 TOP 10 TISSUES BY GENE IDENTIFICATION RATIO:")
    print("-" * 70)
    
    # Sort by Gene ID Ratio (descending)
    top_tissues = results_df.sort_values('Gene_Identification_Ratio', ascending=False).head(10)
    
    for i, (_, row) in enumerate(top_tissues.iterrows(), 1):
        ratio_str = f"{row['Migraine_Genes_Found']}/{row['Feature_Count']}"
        test_auc_str = f"{row['Test_AUC']:.4f}" if pd.notna(row['Test_AUC']) else "N/A"
        
        print(f"{i:2d}. {row['Tissue']:30s}: {ratio_str:8s} = {row['Gene_Identification_Ratio']:5.2f}%, Test AUC: {test_auc_str}")
        print(f"    ML Method: {row['ML_Method']:15s}, Important genes: {row['Total_Important_Genes']:4d}")
    
    print(f"\n📈 GENE ID RATIO DISTRIBUTION:")
    for threshold in [0.5, 1.0, 2.0, 5.0]:
        count = len(results_df[results_df['Gene_Identification_Ratio'] >= threshold])
        percentage = (count / len(results_df) * 100) if len(results_df) > 0 else 0
        print(f"   Tissues with ≥{threshold:4.1f}% Gene ID Ratio: {count:2d} ({percentage:5.1f}%)")
    
    print(f"\n🎯 TEST AUC DISTRIBUTION:")
    valid_auc = results_df[results_df['Test_AUC'].notna()]
    if len(valid_auc) > 0:
        for threshold in [0.60, 0.65, 0.70, 0.75]:
            count = len(valid_auc[valid_auc['Test_AUC'] >= threshold])
            percentage = (count / len(valid_auc) * 100)
            print(f"   Tissues with ≥{threshold:4.2f} Test AUC: {count:2d} ({percentage:5.1f}%)")
    else:
        print("   No Test AUC data available")

def main():
    """Main analysis function for multiple expression methods"""
    
    print(f"\n{'🔥' * 60}")
    print("🔥 STARTING MULTI-MODEL GENE IDENTIFICATION RATIO ANALYSIS")
    print(f"{'🔥' * 60}")
    
    # Load migraine genes
    migraine_result = load_migraine_genes()
    if migraine_result is None:
        return False
    
    migraine_genes, migraine_df = migraine_result
    
    # Get expression methods from command line arguments
    expression_methods = EXPRESSION_METHODS
    print(f"🎯 Analyzing {len(expression_methods)} gene expression models: {', '.join(expression_methods)}")
    
    all_results = []
    all_results_dfs = []
    
    # Process each expression method
    for i, expression_method in enumerate(expression_methods, 1):
        print(f"\n{'='*80}")
        print(f"🔥 PROCESSING MODEL {i}/{len(expression_methods)}: {expression_method}")
        print(f"{'='*80}")
        
        # Find feature files for this expression method
        feature_files = find_best_feature_files(expression_method)
        if not feature_files:
            print(f"❌ No feature files found for {expression_method}")
            continue
        
        # Load test performance data for this expression method
        test_performance = load_test_performance(expression_method)
        
        # Analyze migraine gene counts and identification ratios
        results = analyze_migraine_gene_counts(feature_files, migraine_genes, test_performance, expression_method)
        
        if not results:
            print(f"❌ No results generated for {expression_method}")
            continue
        
        # Create results DataFrame
        results_df = create_results_dataframe(results)
        all_results_dfs.append(results_df)
        
        # Save results for this expression method
        results_file, summary_file, heatmap_file, _, _ = save_results(results_df, migraine_genes, expression_method)
        
        print(f"✅ {expression_method} analysis complete:")
        print(f"   📊 {len(results_df)} tissues analyzed")
        print(f"   🧬 {len(results_df[results_df['Migraine_Genes_Found'] > 0])} tissues with migraine genes")
        if len(results_df) > 0:
            print(f"   📈 Average gene ID ratio: {results_df['Gene_Identification_Ratio'].mean():.2f}%")
    
    # Combine all results for cross-model analysis
    if all_results_dfs:
        print(f"\n{'🔥' * 60}")
        print("🔥 GENERATING CROSS-MODEL COMPARISONS")
        print(f"{'🔥' * 60}")
        
        all_results_combined = pd.concat(all_results_dfs, ignore_index=True)
        
        # Generate cross-model heatmaps (save to the last expression method for consistency)
        if len(expression_methods) > 1:
            _, _, _, cross_ratio_heatmap, cross_auc_heatmap = save_results(
                all_results_dfs[-1], migraine_genes, expression_methods[-1], all_results_combined
            )
        
        # Display cross-model summary
        print(f"\n📊 CROSS-MODEL SUMMARY:")
        print(f"Total models: {len(expression_methods)}")
        print(f"Total tissue-model combinations: {len(all_results_combined)}")
        
        # Summary by model
        for method in expression_methods:
            method_data = all_results_combined[all_results_combined['Expression_Method'] == method]
            if len(method_data) > 0:
                avg_ratio = method_data['Gene_Identification_Ratio'].mean()
                tissues_with_genes = len(method_data[method_data['Migraine_Genes_Found'] > 0])
                print(f"   {method}: {avg_ratio:.2f}% avg ratio, {tissues_with_genes}/{len(method_data)} tissues with genes")
        
        # Best performers across all models
        if len(all_results_combined) > 0:
            best_overall = all_results_combined.loc[all_results_combined['Gene_Identification_Ratio'].idxmax()]
            print(f"\n🏆 Best overall performer:")
            print(f"   {best_overall['Tissue']} ({best_overall['Expression_Method']}): {best_overall['Gene_Identification_Ratio']:.2f}% ratio")
    
    print(f"\n🎉 MULTI-MODEL GENE IDENTIFICATION ANALYSIS COMPLETE!")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Analysis completed successfully!")
    else:
        print("\n❌ Analysis failed!")