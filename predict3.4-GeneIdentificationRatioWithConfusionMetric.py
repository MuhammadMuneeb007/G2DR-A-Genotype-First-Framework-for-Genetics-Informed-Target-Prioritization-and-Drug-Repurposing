#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common Genes Analysis for Best Performing Models

This script analyzes the best performing feature files for each tissue
and compares them with migraine genes.

Usage: python common_genes_analysis.py <phenotype> <expression_method>
Example: python common_genes_analysis.py migraine JTI
"""

import os
import sys
import pandas as pd
import numpy as np
import glob
from collections import defaultdict
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# Check command line arguments
if len(sys.argv) != 3:
    print("Usage: python common_genes_analysis.py <phenotype> <expression_method>")
    print("Example: python common_genes_analysis.py migraine JTI")
    sys.exit(1)

PHENOTYPE = sys.argv[1]
EXPRESSION_METHOD = sys.argv[2]

print(f"🧬 Analyzing best performing feature files")
print(f"📊 Phenotype: {PHENOTYPE}")
print(f"🔬 Expression Method: {EXPRESSION_METHOD}")
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
        
        print(f"📋 Migraine genes preview:")
        for i, (_, row) in enumerate(migraine_df.head(5).iterrows(), 1):
            print(f"   {i}. {row['gene']} ({row['ensembl_gene_id']})")
        
        return migraine_genes, migraine_df
        
    except Exception as e:
        print(f"❌ Error loading migraine genes: {e}")
        return None

def find_best_feature_files():
    """Find all best feature files in the BestCombination_Features directory"""
    
    features_dir = f"{PHENOTYPE}/Database/{EXPRESSION_METHOD}/BestCombination_Features/"
    
    if not os.path.exists(features_dir):
        print(f"❌ Best features directory not found: {features_dir}")
        print("   Please run the main merge script first to generate these files.")
        return []
    
    # Find all feature CSV files
    pattern = os.path.join(features_dir, "*_features.csv")
    feature_files = glob.glob(pattern)
    
    print(f"✅ Found {len(feature_files)} best feature files")
    
    return feature_files

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

def load_tissue_features(feature_file):
    """Load features for a tissue from its best feature file"""
    
    try:
        df = pd.read_csv(feature_file)
        
        if 'Feature_Name' not in df.columns:
            print(f"   ❌ No Feature_Name column in {os.path.basename(feature_file)}")
            return None
        
        # Clean feature names by removing version numbers (e.g., ENSG00000000001.1 -> ENSG00000000001)
        df['Feature_Name_Clean'] = df['Feature_Name'].str.split('.').str[0]
        
        # Return set of clean feature names
        features = set(df['Feature_Name_Clean'].tolist())
        
        return features
        
    except Exception as e:
        print(f"   ❌ Error loading {os.path.basename(feature_file)}: {e}")
        return None

def compare_with_migraine_genes(tissue_features, migraine_genes, feature_count_used=None):
    """Compare tissue features with migraine genes using confusion matrix approach"""
    
    if tissue_features is None:
        return {
            'total_features_passed': feature_count_used if feature_count_used else 0,
            'important_features': 0,
            'total_migraine_genes': len(migraine_genes),
            'true_positives': 0,  # TP: Important features that are migraine genes
            'false_negatives': len(migraine_genes),  # FN: Migraine genes not identified
            'false_positives': 0,  # FP: Important features that are not migraine genes
            'true_negatives': feature_count_used - len(migraine_genes) if feature_count_used else 'Unknown',  # TN
            'gene_identification_ratio': 0.0,  # TP / Total migraine genes
            'precision': 0.0,  # TP / (TP + FP)
            'recall': 0.0,  # TP / (TP + FN) = TP / Total migraine genes
            'specificity': 100.0 if feature_count_used and feature_count_used > len(migraine_genes) else 0.0,  # TN / (TN + FP)
            'accuracy': ((feature_count_used - len(migraine_genes)) / feature_count_used * 100) if feature_count_used else 0.0,  # (TP + TN) / Total
            'f1_score': 0.0,  # 2 * (precision * recall) / (precision + recall)
            'overlap_genes': set(),
            'missing_migraine_genes': migraine_genes.copy(),
            'non_migraine_important': set()
        }
    
    # Important Features (IF) - genes with non-zero weights
    important_features = tissue_features
    
    # True Positives (TP): Important features that are also migraine genes
    true_positives = important_features.intersection(migraine_genes)
    
    # False Negatives (FN): Migraine genes not identified as important
    false_negatives = migraine_genes - important_features
    
    # False Positives (FP): Important features that are not migraine genes
    false_positives = important_features - migraine_genes
    
    # True Negatives (TN): Genes that are neither important nor migraine genes
    # For genes passed to ML but given zero weights and are not migraine genes
    if feature_count_used:
        # Estimate TN as: total_features_passed - important_features - false_negatives
        # This assumes all migraine genes were in the feature set passed to ML
        estimated_tn = feature_count_used - len(important_features) - len(false_negatives)
        true_negatives = max(0, estimated_tn)  # Ensure non-negative
    else:
        true_negatives = 'Unknown'
    
    # Calculate metrics
    tp_count = len(true_positives)
    fn_count = len(false_negatives)
    fp_count = len(false_positives)
    tn_count = true_negatives if isinstance(true_negatives, int) else 0
    
    # Gene Identification Ratio: TP / Total Migraine Genes
    gene_identification_ratio = (tp_count / len(migraine_genes) * 100) if len(migraine_genes) > 0 else 0
    
    # Precision: TP / (TP + FP) - of all important features, how many are migraine genes
    precision = (tp_count / (tp_count + fp_count) * 100) if (tp_count + fp_count) > 0 else 0
    
    # Recall: TP / (TP + FN) = TP / Total Migraine Genes - what % of migraine genes were identified
    recall = (tp_count / (tp_count + fn_count) * 100) if (tp_count + fn_count) > 0 else 0
    
    # Specificity: TN / (TN + FP) - what % of non-migraine genes were correctly ignored
    if isinstance(true_negatives, int) and (tn_count + fp_count) > 0:
        specificity = (tn_count / (tn_count + fp_count) * 100)
    else:
        specificity = 0.0
    
    # Accuracy: (TP + TN) / (TP + TN + FP + FN) - overall correctness
    if isinstance(true_negatives, int):
        total_decisions = tp_count + tn_count + fp_count + fn_count
        if total_decisions > 0:
            accuracy = ((tp_count + tn_count) / total_decisions * 100)
        else:
            accuracy = 0.0
    else:
        accuracy = 0.0
    
    # F1 Score: Harmonic mean of precision and recall
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    
    comparison = {
        'total_features_passed': feature_count_used if feature_count_used else len(important_features),
        'important_features': len(important_features),
        'total_migraine_genes': len(migraine_genes),
        'true_positives': tp_count,
        'false_negatives': fn_count,
        'false_positives': fp_count,
        'true_negatives': true_negatives,
        'gene_identification_ratio': gene_identification_ratio,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'accuracy': accuracy,
        'f1_score': f1_score,
        'overlap_genes': true_positives,
        'missing_migraine_genes': false_negatives,
        'non_migraine_important': false_positives
    }
    
    return comparison

def analyze_all_tissues(feature_files, migraine_genes):
    """Analyze all tissue feature files and compare with migraine genes"""
    
    print(f"\n🧪 Analyzing all tissue feature files...")
    print("="*60)
    
    results = []
    
    for feature_file in feature_files:
        filename = os.path.basename(feature_file)
        
        # Parse tissue info from filename
        tissue, ml_method, feature_count = parse_tissue_info_from_filename(feature_file)
        
        if tissue is None:
            print(f"❌ Could not parse filename: {filename}")
            continue
        
        print(f"\n🔬 Analyzing {tissue}...")
        print(f"   Method: {ml_method}, Features: {feature_count}")
        
        # Load tissue features
        tissue_features = load_tissue_features(feature_file)
        
        if tissue_features is None:
            print(f"   ❌ Could not load features")
            # Add empty result with complete confusion matrix metrics
            tn_count = feature_count - len(migraine_genes) if feature_count else 0
            tn_count = max(0, tn_count)  # Ensure non-negative
            results.append({
                'Tissue': tissue,
                'ML_Method': ml_method,
                'Feature_Count': feature_count,
                'Total_Features_Passed': feature_count,
                'Important_Features': 0,
                'Total_Migraine_Genes': len(migraine_genes),
                'True_Positives': 0,
                'False_Negatives': len(migraine_genes),
                'False_Positives': 0,
                'True_Negatives': tn_count,
                'Gene_Identification_Ratio': 0.0,
                'Precision': 0.0,
                'Recall': 0.0,
                'Specificity': (tn_count / tn_count * 100) if tn_count > 0 else 0.0,
                'F1_Score': 0.0,
                'Accuracy': (tn_count / (tn_count + len(migraine_genes)) * 100) if (tn_count + len(migraine_genes)) > 0 else 0.0,
                'Feature_File': filename
            })
            continue
        
        print(f"   ✅ Loaded {len(tissue_features)} important features")
        
        # Compare with migraine genes using confusion matrix approach
        comparison = compare_with_migraine_genes(tissue_features, migraine_genes, feature_count)
        
        # Store results with complete confusion matrix metrics
        result = {
            'Tissue': tissue,
            'ML_Method': ml_method,
            'Feature_Count': feature_count,
            'Total_Features_Passed': comparison['total_features_passed'],
            'Important_Features': comparison['important_features'],
            'Total_Migraine_Genes': comparison['total_migraine_genes'],
            'True_Positives': comparison['true_positives'],
            'False_Negatives': comparison['false_negatives'],
            'False_Positives': comparison['false_positives'],
            'True_Negatives': comparison['true_negatives'],
            'Gene_Identification_Ratio': comparison['gene_identification_ratio'],
            'Precision': comparison['precision'],
            'Recall': comparison['recall'],
            'Specificity': comparison['specificity'],
            'F1_Score': comparison['f1_score'],
            'Accuracy': comparison['accuracy'],
            'Feature_File': filename
        }
        
        results.append(result)
        
        print(f"   📊 Complete Confusion Matrix Results:")
        print(f"      Total Features Passed (TF): {comparison['total_features_passed']}")
        print(f"      Important Features (IF): {comparison['important_features']} - genes with non-zero weights")
        print(f"      Genes with Zero Weights: {comparison['total_features_passed'] - comparison['important_features']}")
        print(f"   🎯 Confusion Matrix:")
        print(f"      True Positives (TP): {comparison['true_positives']} - Migraine genes correctly identified as important")
        print(f"      False Negatives (FN): {comparison['false_negatives']} - Migraine genes missed (zero weights)")  
        print(f"      False Positives (FP): {comparison['false_positives']} - Non-migraine genes incorrectly selected")
        print(f"      True Negatives (TN): {comparison['true_negatives']} - Non-migraine genes correctly ignored")
        print(f"   📈 Performance Metrics:")
        print(f"      Gene Identification Ratio: {comparison['gene_identification_ratio']:.1f}% - TP/Total Migraine Genes")
        print(f"      Precision (PPV): {comparison['precision']:.1f}% - TP/(TP+FP)")
        print(f"      Recall (Sensitivity): {comparison['recall']:.1f}% - TP/(TP+FN)")
        print(f"      Specificity (TNR): {comparison['specificity']:.1f}% - TN/(TN+FP)")
        print(f"      F1 Score: {comparison['f1_score']:.1f}% - Harmonic mean of precision & recall")
        print(f"      Accuracy: {comparison['accuracy']:.1f}% - (TP+TN)/(TP+TN+FP+FN)")
        
        # Show some True Positive genes if any
        if len(comparison['overlap_genes']) > 0:
            overlap_list = sorted(list(comparison['overlap_genes']))
            show_count = min(5, len(overlap_list))
            print(f"      Top {show_count} True Positive genes: {', '.join(overlap_list[:show_count])}")
    
    return results

def save_detailed_results(results, migraine_df, migraine_genes, feature_files):
    """Save detailed results including gene lists"""
    
    print(f"\n💾 Saving detailed results...")
    
    # Create output directory
    output_dir = f"{PHENOTYPE}/Database/{EXPRESSION_METHOD}/CommonGenesAnalysis/"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save main results table
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Gene_Identification_Ratio', ascending=False)
    
    results_file = os.path.join(output_dir, f"{PHENOTYPE}_{EXPRESSION_METHOD}_BestFeatures_ConfusionMatrix_Analysis.csv")
    results_df.to_csv(results_file, index=False, float_format='%.4f')
    
    print(f"   ✅ Main results: {results_file}")
    
    # 2. Save detailed gene overlap for each tissue (True Positives)
    for i, result in enumerate(results):
        tissue = result['Tissue']
        
        if result['True_Positives'] > 0:
            # Find the corresponding feature file and reload to get the overlapping genes
            feature_file = None
            for ff in feature_files:
                if result['Feature_File'] in ff:
                    feature_file = ff
                    break
            
            if feature_file:
                tissue_features = load_tissue_features(feature_file)
                if tissue_features:
                    comparison = compare_with_migraine_genes(tissue_features, migraine_genes, result['Feature_Count'])
                    
                    # Save True Positive genes (important features that are migraine genes)
                    if len(comparison['overlap_genes']) > 0:
                        overlap_data = []
                        
                        # Get gene names from migraine_df
                        for ensembl_id in comparison['overlap_genes']:
                            gene_info = migraine_df[migraine_df['ensembl_gene_id'] == ensembl_id]
                            if len(gene_info) > 0:
                                gene_name = gene_info.iloc[0]['gene']
                            else:
                                gene_name = 'Unknown'
                            
                            overlap_data.append({
                                'Gene_Name': gene_name,
                                'Ensembl_ID': ensembl_id,
                                'Category': 'True_Positive'
                            })
                        
                        if overlap_data:
                            overlap_df = pd.DataFrame(overlap_data)
                            overlap_file = os.path.join(output_dir, f"{tissue}_TruePositive_MigrainGenes.csv")
                            overlap_df.to_csv(overlap_file, index=False)
    
    # 3. Save summary statistics with safe type conversions and handle True_Negatives safely
    def safe_convert(value):
        """Safely convert numpy types to JSON-serializable Python types"""
        if pd.isna(value):
            return None
        elif value == 'Unknown':
            return 'Unknown'
        elif hasattr(value, 'item'):  # numpy scalar
            return value.item()
        elif isinstance(value, (np.integer, np.int64, np.int32)):
            return int(value)
        elif isinstance(value, (np.floating, np.float64, np.float32)):
            return float(value)
        else:
            return value
    
    # Handle True_Negatives column safely
    results_df['True_Negatives_Safe'] = pd.to_numeric(results_df['True_Negatives'], errors='coerce').fillna(0)
    
    # Calculate statistics with safe conversions
    try:
        summary_stats = {
            'total_tissues_analyzed': safe_convert(len(results)),
            'total_migraine_genes': safe_convert(len(migraine_genes)),
            'avg_gene_identification_ratio': safe_convert(results_df['Gene_Identification_Ratio'].mean()),
            'max_gene_identification_ratio': safe_convert(results_df['Gene_Identification_Ratio'].max()),
            'min_gene_identification_ratio': safe_convert(results_df['Gene_Identification_Ratio'].min()),
            'tissues_identifying_migraine_genes': safe_convert(len(results_df[results_df['True_Positives'] > 0])),
            'avg_important_features_per_tissue': safe_convert(results_df['Important_Features'].mean()),
            'total_true_positives': safe_convert(results_df['True_Positives'].sum()),
            'total_false_negatives': safe_convert(results_df['False_Negatives'].sum()),
            'total_false_positives': safe_convert(results_df['False_Positives'].sum()),
            'total_true_negatives': safe_convert(results_df['True_Negatives_Safe'].sum()),
            'avg_precision': safe_convert(results_df['Precision'].mean()),
            'avg_recall': safe_convert(results_df['Recall'].mean()),
            'avg_specificity': safe_convert(results_df['Specificity'].mean()),
            'avg_f1_score': safe_convert(results_df['F1_Score'].mean()),
            'avg_accuracy': safe_convert(results_df['Accuracy'].mean()),
            'phenotype': str(PHENOTYPE),
            'expression_method': str(EXPRESSION_METHOD),
            'feature_files_analyzed': safe_convert(len(feature_files))
        }
        
        print(f"   📊 Summary stats calculated successfully")
        
    except Exception as e:
        print(f"   ⚠️  Error calculating summary stats: {e}")
        # Fallback simple summary
        summary_stats = {
            'total_tissues_analyzed': len(results),
            'total_migraine_genes': len(migraine_genes),
            'phenotype': PHENOTYPE,
            'expression_method': EXPRESSION_METHOD,
            'error': str(e)
        }
    
    import json
    summary_file = os.path.join(output_dir, f"{PHENOTYPE}_{EXPRESSION_METHOD}_ConfusionMatrix_Summary.json")
    
    try:
        with open(summary_file, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        print(f"   ✅ Summary stats: {summary_file}")
    except Exception as json_error:
        print(f"   ⚠️  JSON save failed: {json_error}")
        # Try to save as text file instead
        txt_file = summary_file.replace('.json', '.txt')
        try:
            with open(txt_file, 'w') as f:
                for key, value in summary_stats.items():
                    f.write(f"{key}: {value}\n")
            print(f"   ✅ Summary stats saved as text: {txt_file}")
            summary_file = txt_file
        except Exception as txt_error:
            print(f"   ❌ Text save also failed: {txt_error}")
            summary_file = None
    
    return results_file, summary_file, results_df

def generate_migraine_overlap_heatmap(results_df, migraine_genes, output_dir):
    """Generate heatmap showing migraine gene identification metrics"""
    
    print(f"\n🔥 Generating migraine identification heatmap...")
    
    if len(results_df) == 0:
        print("❌ No data for heatmap")
        return None
    
    # Sort tissues alphabetically by name
    tissue_order = sorted(results_df['Tissue'].tolist())
    
    # Create the main heatmap matrix showing key metrics including TP, FN, FP
    viz_data = []
    for _, row in results_df.iterrows():
        viz_data.append({
            'Tissue': row['Tissue'],
            'Gene_ID_Ratio': row['Gene_Identification_Ratio'],
            'True_Positives': row['True_Positives'],
            'False_Negatives': row['False_Negatives'],
            'False_Positives': row['False_Positives'],
            'Precision': row['Precision'],
            'Recall': row['Recall']
        })
    
    viz_df = pd.DataFrame(viz_data)
    viz_df = viz_df.set_index('Tissue')
    viz_df = viz_df.reindex(tissue_order)  # Sort alphabetically by tissue name
    
    # Create the heatmap
    plt.figure(figsize=(14, max(10, len(tissue_order) * 0.4)))
    
    # Generate heatmap with custom colormap
    ax = sns.heatmap(
        viz_df, 
        annot=True, 
        fmt='.1f', 
        cmap='RdYlBu_r',
        center=None,  # Let it auto-center
        cbar_kws={'label': 'Count / Percentage'},
        linewidths=0.5,
        square=False,
        xticklabels=['Gene ID\nRatio (%)', 'True\nPositives', 'False\nNegatives', 'False\nPositives', 'Precision\n(%)', 'Recall\n(%)']
    )
    
    # Customize the plot
    plt.title(f'{PHENOTYPE} - Migraine Gene Identification Performance\n({EXPRESSION_METHOD} Expression Method - Alphabetical Order)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Performance Metrics', fontsize=12, fontweight='bold')
    plt.ylabel('Tissue (Alphabetical Order)', fontsize=12, fontweight='bold')
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the heatmap
    heatmap_file = os.path.join(output_dir, f"{PHENOTYPE}_{EXPRESSION_METHOD}_GeneIdentification_Heatmap.png")
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    
    print(f"   ✅ Gene identification heatmap saved: {heatmap_file}")
    
    plt.show()
    plt.close()
    
    return heatmap_file

def generate_detailed_comparison_heatmap(results_df, migraine_genes, output_dir):
    """Generate detailed heatmap showing complete confusion matrix metrics"""
    
    print(f"\n🔥 Generating complete confusion matrix heatmap...")
    
    if len(results_df) == 0:
        print("❌ No data for heatmap")
        return None
    
    # Sort tissues alphabetically by name
    tissue_order = sorted(results_df['Tissue'].tolist())
    results_sorted = results_df.set_index('Tissue').reindex(tissue_order).reset_index()
    
    # Handle True_Negatives that might be 'Unknown' strings
    results_sorted['True_Negatives_Safe'] = pd.to_numeric(results_sorted['True_Negatives'], errors='coerce').fillna(0)
    
    # Create matrix with complete confusion matrix metrics
    heatmap_matrix = results_sorted[['Tissue', 'True_Positives', 'False_Negatives', 
                                   'False_Positives', 'True_Negatives_Safe', 'Accuracy']].copy()
    
    # Scale True_Negatives for better visualization (they're usually much larger)
    heatmap_matrix['True_Negatives_Scaled'] = heatmap_matrix['True_Negatives_Safe'] / 10
    
    # Select final columns for heatmap
    final_matrix = heatmap_matrix[['True_Positives', 'False_Negatives', 
                                  'False_Positives', 'True_Negatives_Scaled', 'Accuracy']].copy()
    final_matrix.index = heatmap_matrix['Tissue']
    
    # Create the heatmap
    plt.figure(figsize=(14, max(12, len(results_sorted) * 0.35)))
    
    # Generate heatmap
    ax = sns.heatmap(
        final_matrix, 
        annot=True, 
        fmt='.1f', 
        cmap='RdYlBu_r',
        cbar_kws={'label': 'Count / Percentage'},
        linewidths=0.5,
        square=False,
        xticklabels=['True Positives\n(TP)', 'False Negatives\n(FN)', 'False Positives\n(FP)', 'True Negatives\n(/10)', 'Accuracy\n(%)']
    )
    
    # Customize the plot
    plt.title(f'{PHENOTYPE} - Complete Confusion Matrix Analysis\n({EXPRESSION_METHOD} Expression Method - Alphabetical Order)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Complete Confusion Matrix Metrics', fontsize=12, fontweight='bold')
    plt.ylabel('Tissue (Alphabetical Order)', fontsize=12, fontweight='bold')
    
    # Rotate labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the heatmap
    detailed_heatmap_file = os.path.join(output_dir, f"{PHENOTYPE}_{EXPRESSION_METHOD}_CompleteConfusionMatrix_Heatmap.png")
    plt.savefig(detailed_heatmap_file, dpi=300, bbox_inches='tight')
    
    print(f"   ✅ Complete confusion matrix heatmap saved: {detailed_heatmap_file}")
    
    plt.show()
    plt.close()
    
    return detailed_heatmap_file

def generate_method_performance_heatmap(results_df, output_dir):
    """Generate heatmap showing performance by ML method and feature count"""
    
    print(f"\n🔥 Generating method performance heatmap...")
    
    if len(results_df) == 0:
        print("❌ No data for heatmap")
        return None
    
    # Create pivot table: ML_Method vs Feature_Count showing average Gene ID Ratio
    method_pivot = results_df.pivot_table(
        values='Gene_Identification_Ratio', 
        index='ML_Method', 
        columns='Feature_Count', 
        aggfunc='mean',
        fill_value=0  # Fill missing values with 0
    )
    
    # Also create count matrix to show how many tissues use each combination
    count_pivot = results_df.pivot_table(
        values='Gene_Identification_Ratio', 
        index='ML_Method', 
        columns='Feature_Count', 
        aggfunc='count',
        fill_value=0  # Fill missing values with 0
    )
    
    # Convert count_pivot to integers to avoid format issues
    count_pivot = count_pivot.astype(int)
    
    # Create the heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Heatmap 1: Average Gene ID Ratio
    sns.heatmap(
        method_pivot, 
        annot=True, 
        fmt='.1f', 
        cmap='RdYlBu_r',
        cbar_kws={'label': 'Avg Gene ID Ratio %'},
        linewidths=0.5,
        square=False,
        ax=ax1
    )
    ax1.set_title('Average Gene Identification Ratio %\nby Method & Feature Count', fontweight='bold')
    ax1.set_xlabel('Feature Count', fontweight='bold')
    ax1.set_ylabel('ML Method', fontweight='bold')
    
    # Heatmap 2: Number of tissues using each combination
    sns.heatmap(
        count_pivot, 
        annot=True, 
        fmt='d', 
        cmap='Greens',
        cbar_kws={'label': 'Number of Tissues'},
        linewidths=0.5,
        square=False,
        ax=ax2
    )
    ax2.set_title('Number of Tissues Using\nEach Method & Feature Count', fontweight='bold')
    ax2.set_xlabel('Feature Count', fontweight='bold')
    ax2.set_ylabel('ML Method', fontweight='bold')
    
    # Overall title
    fig.suptitle(f'{PHENOTYPE} - Method Performance Analysis ({EXPRESSION_METHOD})', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the heatmap
    method_heatmap_file = os.path.join(output_dir, f"{PHENOTYPE}_{EXPRESSION_METHOD}_MethodPerformance_Heatmap.png")
    plt.savefig(method_heatmap_file, dpi=300, bbox_inches='tight')
    
    print(f"   ✅ Method performance heatmap saved: {method_heatmap_file}")
    
    plt.show()
    plt.close()
    
    return method_heatmap_file

def display_final_summary(results_df, migraine_genes):
    """Display final summary of the analysis with confusion matrix metrics"""
    
    print(f"\n🎉 MIGRAINE GENE IDENTIFICATION ANALYSIS COMPLETE!")
    print("="*50)
    
    # Handle True_Negatives safely for calculations
    results_df['True_Negatives_Safe'] = pd.to_numeric(results_df['True_Negatives'], errors='coerce').fillna(0)
    
    print(f"📊 SUMMARY STATISTICS:")
    print(f"   Total tissues analyzed: {len(results_df)}")
    print(f"   Total migraine genes (MG): {len(migraine_genes)}")
    print(f"   Average Gene ID Ratio: {results_df['Gene_Identification_Ratio'].mean():.1f}%")
    print(f"   Best Gene ID Ratio: {results_df['Gene_Identification_Ratio'].max():.1f}%")
    print(f"   Tissues identifying migraine genes: {len(results_df[results_df['True_Positives'] > 0])}")
    print(f"   Average Precision: {results_df['Precision'].mean():.1f}%")
    print(f"   Average Recall (Sensitivity): {results_df['Recall'].mean():.1f}%")
    print(f"   Average Specificity: {results_df['Specificity'].mean():.1f}%")
    print(f"   Average F1 Score: {results_df['F1_Score'].mean():.1f}%")
    print(f"   Average Accuracy: {results_df['Accuracy'].mean():.1f}%")
    
    print(f"\n🏆 TOP 10 TISSUES BY GENE IDENTIFICATION RATIO:")
    top_10 = results_df.head(10)
    for i, (_, row) in enumerate(top_10.iterrows(), 1):
        tn_display = row['True_Negatives'] if pd.notna(row['True_Negatives']) and row['True_Negatives'] != 'Unknown' else 'N/A'
        print(f"   {i:2d}. {row['Tissue'][:25]:25s}: {row['Gene_Identification_Ratio']:5.1f}% (TP:{row['True_Positives']:2d} FN:{row['False_Negatives']:2d} FP:{row['False_Positives']:2d} TN:{tn_display})")
        print(f"       Method: {row['ML_Method']:15s} Features: {row['Feature_Count']:4d}  Important: {row['Important_Features']:4d}")
        print(f"       Precision: {row['Precision']:5.1f}%  Recall: {row['Recall']:5.1f}%  Specificity: {row['Specificity']:5.1f}%  Accuracy: {row['Accuracy']:5.1f}%")
    
    print(f"\n📈 GENE IDENTIFICATION DISTRIBUTION:")
    for threshold in [0, 1, 5, 10, 20, 50]:
        count = len(results_df[results_df['Gene_Identification_Ratio'] >= threshold])
        percentage = (count / len(results_df) * 100) if len(results_df) > 0 else 0
        print(f"   ≥{threshold:2d}% Gene ID Ratio: {count:2d} tissues ({percentage:5.1f}%)")
    
    # Show method and feature count distributions
    print(f"\n🔬 ML METHOD DISTRIBUTION:")
    method_counts = results_df['ML_Method'].value_counts()
    for method, count in method_counts.items():
        avg_ratio = results_df[results_df['ML_Method'] == method]['Gene_Identification_Ratio'].mean()
        percentage = (count / len(results_df) * 100)
        print(f"   {method:20s}: {count:2d} tissues ({percentage:5.1f}%), Avg Gene ID: {avg_ratio:5.1f}%")
    
    print(f"\n🔢 FEATURE COUNT DISTRIBUTION:")
    feature_counts = results_df['Feature_Count'].value_counts().sort_index()
    for fc, count in feature_counts.items():
        avg_ratio = results_df[results_df['Feature_Count'] == fc]['Gene_Identification_Ratio'].mean()
        percentage = (count / len(results_df) * 100)
        print(f"   {fc:4d} features: {count:2d} tissues ({percentage:5.1f}%), Avg Gene ID: {avg_ratio:5.1f}%")
    
    # Show confusion matrix summary
    print(f"\n🎯 COMPLETE CONFUSION MATRIX SUMMARY (Totals across all tissues):")
    total_tp = results_df['True_Positives'].sum()
    total_fn = results_df['False_Negatives'].sum()
    total_fp = results_df['False_Positives'].sum()
    total_tn = results_df['True_Negatives_Safe'].sum()
    total_important = results_df['Important_Features'].sum()
    
    print(f"   True Positives (TP): {total_tp:4d} - Migraine genes correctly identified as important")
    print(f"   False Negatives (FN): {total_fn:4d} - Migraine genes missed (given zero weights)")
    print(f"   False Positives (FP): {total_fp:4d} - Non-migraine genes incorrectly identified as important")
    print(f"   True Negatives (TN): {total_tn:4.0f} - Non-migraine genes correctly given zero weights")
    print(f"   Total Important Features: {total_important:4d} - Genes with non-zero weights")
    print(f"   Total Genes Analyzed: {total_tp + total_fn + total_fp + total_tn:4.0f} - Across all tissues")
    
    # Calculate overall metrics
    overall_precision = (total_tp / (total_tp + total_fp) * 100) if (total_tp + total_fp) > 0 else 0
    overall_recall = (total_tp / (total_tp + total_fn) * 100) if (total_tp + total_fn) > 0 else 0
    overall_specificity = (total_tn / (total_tn + total_fp) * 100) if (total_tn + total_fp) > 0 else 0
    overall_accuracy = ((total_tp + total_tn) / (total_tp + total_fn + total_fp + total_tn) * 100) if (total_tp + total_fn + total_fp + total_tn) > 0 else 0
    overall_f1 = (2 * overall_precision * overall_recall / (overall_precision + overall_recall)) if (overall_precision + overall_recall) > 0 else 0
    
    print(f"\n📈 OVERALL PERFORMANCE METRICS:")
    print(f"   Overall Precision (PPV): {overall_precision:5.1f}% - TP/(TP+FP)")
    print(f"   Overall Recall (Sensitivity): {overall_recall:5.1f}% - TP/(TP+FN)")
    print(f"   Overall Specificity (TNR): {overall_specificity:5.1f}% - TN/(TN+FP)")
    print(f"   Overall Accuracy: {overall_accuracy:5.1f}% - (TP+TN)/(TP+FN+FP+TN)")
    print(f"   Overall F1 Score: {overall_f1:5.1f}% - Harmonic mean of precision & recall")
    
def main():
    """Main analysis function"""
    
    # Load migraine genes
    migraine_result = load_migraine_genes()
    if migraine_result is None:
        return False
    
    migraine_genes, migraine_df = migraine_result
    
    # Find best feature files
    feature_files = find_best_feature_files()
    if not feature_files:
        return False
    
    # Show some example files
    print(f"\n📁 Example feature files:")
    for i, ff in enumerate(feature_files[:3], 1):
        print(f"   {i}. {os.path.basename(ff)}")
    if len(feature_files) > 3:
        print(f"   ... and {len(feature_files)-3} more")
    
    # Analyze all tissues
    results = analyze_all_tissues(feature_files, migraine_genes)
    
    if not results:
        print("❌ No results generated")
        return False
    
    # Save detailed results
    results_file, summary_file, results_df = save_detailed_results(results, migraine_df, migraine_genes, feature_files)
    
    # Generate heatmaps
    print(f"\n{'🔥' * 40}")
    print("🔥 GENERATING VISUALIZATION HEATMAPS...")
    print(f"{'🔥' * 40}")
    
    output_dir = f"{PHENOTYPE}/Database/{EXPRESSION_METHOD}/CommonGenesAnalysis/"
    
    heatmap_files = []
    
    try:
        # Generate main overlap heatmap
        overlap_heatmap = generate_migraine_overlap_heatmap(results_df, migraine_genes, output_dir)
        if overlap_heatmap:
            heatmap_files.append(overlap_heatmap)
        
        # Generate detailed comparison heatmap
        detailed_heatmap = generate_detailed_comparison_heatmap(results_df, migraine_genes, output_dir)
        if detailed_heatmap:
            heatmap_files.append(detailed_heatmap)
        
        # Generate method performance heatmap
        method_heatmap = generate_method_performance_heatmap(results_df, output_dir)
        if method_heatmap:
            heatmap_files.append(method_heatmap)
            
        print(f"✅ Generated {len(heatmap_files)} heatmaps successfully!")
        
    except Exception as e:
        print(f"⚠️  Heatmap generation failed: {e}")
        import traceback
        traceback.print_exc()
        heatmap_files = []
    
    # Display final summary
    display_final_summary(results_df, migraine_genes)
    
    print(f"\n📁 OUTPUT FILES:")
    print(f"   📊 Main results: {results_file}")
    print(f"   📈 Summary: {summary_file}")
    print(f"   🧬 True Positive gene files: {len(results_df[results_df['True_Positives'] > 0])} files in same directory")
    
    if heatmap_files:
        print(f"   🔥 Heatmaps generated:")
        for i, hf in enumerate(heatmap_files, 1):
            print(f"      {i}. {os.path.basename(hf)}")
    else:
        print(f"   ⚠️  No heatmaps generated")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Analysis completed successfully!")
    else:
        print("\n❌ Analysis failed!")