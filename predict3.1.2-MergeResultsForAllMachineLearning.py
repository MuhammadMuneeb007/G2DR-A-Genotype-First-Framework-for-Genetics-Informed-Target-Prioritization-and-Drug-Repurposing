#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete ML Results Merger - Process Enhanced ML Pipeline Results

This script loads results from the enhanced ML pipeline and creates:
1. Merged tables per fold with all combinations
2. Cross-fold averages and best performers
3. Comprehensive analysis from all perspectives
4. Feature importance files for best combinations
5. Cross-database comparison heatmaps

Usage: python complete_merge_results.py <phenotype> <expression_method>
Example: python complete_merge_results.py migraine JTI
Example: python complete_merge_results.py migraine Regular,JTI,UTMOST
"""

import os
import sys
import pandas as pd
import numpy as np
import glob
import json
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from collections import defaultdict
import traceback

warnings.filterwarnings('ignore')

# Check command line arguments
if len(sys.argv) != 3:
    print("Usage: python complete_merge_results.py <phenotype> <expression_methods>")
    print("Example: python complete_merge_results.py migraine JTI")
    print("Example (multiple): python complete_merge_results.py migraine Regular,JTI,UTMOST")
    print("\nAvailable expression methods:")
    print("  - Regular, JTI, UTMOST, UTMOST2, EpiX, TIGAR, FUSION")
    sys.exit(1)

PHENOTYPE = sys.argv[1]
EXPRESSION_METHODS = [method.strip() for method in sys.argv[2].split(',')]

# Configuration
ML_METHODS = ['XGBoost', 'RandomForest', 'LogisticRegression']
FEATURE_COUNTS = [100, 500, 1000, 2000]
EXPECTED_FOLDS = [0, 1, 2, 3, 4]

print(f"🔬 Processing phenotype: {PHENOTYPE}")
print(f"🧬 Expression methods: {', '.join(EXPRESSION_METHODS)}")
print("="*60)

def find_fold_directories(expression_method):
    """Find all Fold_* directories for the given phenotype and expression method"""
    
    base_pattern = "{}/Fold_*".format(PHENOTYPE)
    fold_dirs = glob.glob(base_pattern)
    
    # Extract fold numbers and sort
    fold_info = []
    for fold_dir in fold_dirs:
        try:
            fold_num = int(fold_dir.split('_')[-1])
            results_dir = os.path.join(fold_dir, "EnhancedMLResults")
            method_dir = os.path.join(results_dir, expression_method)
            
            if os.path.exists(method_dir):
                fold_info.append((fold_num, fold_dir, method_dir))
        except ValueError:
            continue
    
    fold_info.sort(key=lambda x: x[0])  # Sort by fold number
    
    print("📁 Found {} folds with {} results:".format(len(fold_info), expression_method))
    for fold_num, fold_dir, method_dir in fold_info:
        print("  - Fold {}: {}".format(fold_num, method_dir))
    
    return fold_info

def get_available_tissues(fold_info):
    """Find all available tissues across all folds"""
    
    all_tissues = set()
    
    for fold_num, fold_dir, method_dir in fold_info:
        # Look in all feature count directories
        for feature_count in FEATURE_COUNTS:
            feature_dir = os.path.join(method_dir, "{}_features".format(feature_count))
            
            if os.path.exists(feature_dir):
                # Find all tissue files
                perf_files = glob.glob(os.path.join(feature_dir, "*_performance.csv"))
                
                for perf_file in perf_files:
                    # Extract tissue name from filename: tissue_method_performance.csv
                    filename = os.path.basename(perf_file)
                    # Remove "_performance.csv" and the method name
                    parts = filename.replace('_performance.csv', '').split('_')
                    if len(parts) >= 2:
                        # Tissue is everything except the last part (which is the method)
                        tissue = '_'.join(parts[:-1])
                        all_tissues.add(tissue)
    
    return sorted(list(all_tissues))

def load_single_performance_record(perf_file):
    """Load a single performance record from CSV file"""
    
    try:
        if not os.path.exists(perf_file):
            return None
            
        df = pd.read_csv(perf_file)
        
        if len(df) == 0:
            return None
        
        # Convert to dictionary format
        perf_dict = {}
        for _, row in df.iterrows():
            metric = row['Metric']
            value = row['Value']
            
            # Try to convert to numeric if possible
            try:
                if pd.notna(value) and value != 'N/A':
                    if isinstance(value, str) and ('.' in value or value.replace('-', '').isdigit()):
                        value = float(value)
                    elif isinstance(value, str) and value.isdigit():
                        value = int(value)
            except:
                pass
            
            perf_dict[metric] = value
        
        return perf_dict
        
    except Exception as e:
        return None

def load_single_feature_importance(fi_file, top_n=10):
    """Load feature importance from CSV file and return top N genes"""
    
    try:
        if not os.path.exists(fi_file):
            return {}
            
        df = pd.read_csv(fi_file)
        
        if len(df) == 0:
            return {}
        
        # Get top N genes
        top_genes = df.head(top_n)
        
        # Create dictionary with gene info
        gene_info = {}
        for i, (_, row) in enumerate(top_genes.iterrows(), 1):
            gene_info['Top_Gene_{}'.format(i)] = row['Feature_Name']
            gene_info['Top_Gene_{}_Importance'.format(i)] = row['Feature_Importance']
        
        # Also add summary statistics
        gene_info['Total_Genes'] = len(df)
        gene_info['Max_Importance'] = df['Feature_Importance'].max()
        gene_info['Mean_Importance'] = df['Feature_Importance'].mean()
        
        return gene_info
        
    except Exception as e:
        return {}

def create_fold_merged_table(fold_num, fold_dir, method_dir, tissues, expression_method):
    """Create merged table for a single fold"""
    
    print("  📊 Creating merged table for Fold {}...".format(fold_num))
    
    fold_records = []
    total_combinations = len(tissues) * len(ML_METHODS) * len(FEATURE_COUNTS)
    processed = 0
    
    for tissue in tissues:
        for feature_count in FEATURE_COUNTS:
            feature_dir = os.path.join(method_dir, "{}_features".format(feature_count))
            
            for ml_method in ML_METHODS:
                processed += 1
                
                if processed == 1 or processed % 100 == 0 or processed == total_combinations:
                    print("    📈 Processing: {}/{} - {}, {}, {} features".format(
                        processed, total_combinations, tissue, ml_method, feature_count
                    ))
                
                # Initialize record with metadata
                record = {
                    'Phenotype': PHENOTYPE,
                    'Expression_Method': expression_method,
                    'Fold': fold_num,
                    'Tissue': tissue,
                    'ML_Method': ml_method,
                    'Feature_Count': feature_count,
                    'Combination_ID': "{}_{}_{}_{}_{}_features".format(
                        PHENOTYPE, expression_method, tissue, ml_method, feature_count
                    )
                }
                
                # Load performance data
                perf_file = os.path.join(feature_dir, "{}_{}_performance.csv".format(tissue, ml_method))
                perf_data = load_single_performance_record(perf_file)
                
                if perf_data:
                    # Add performance metrics
                    record.update({
                        'Train_AUC': perf_data.get('Train_AUC', None),
                        'Val_AUC': perf_data.get('Val_AUC', None),
                        'Test_AUC': perf_data.get('Test_AUC', None),
                        'Features_Used': perf_data.get('Features_Used', None),
                        'Train_Samples': perf_data.get('Train_Samples', None),
                        'Val_Samples': perf_data.get('Val_Samples', None),
                        'Test_Samples': perf_data.get('Test_Samples', None),
                        'Best_Params': str(perf_data.get('Best_Params', 'N/A'))
                    })
                    record['Data_Available'] = True
                else:
                    # Fill with missing values
                    record.update({
                        'Train_AUC': None,
                        'Val_AUC': None,
                        'Test_AUC': None,
                        'Features_Used': None,
                        'Train_Samples': None,
                        'Val_Samples': None,
                        'Test_Samples': None,
                        'Best_Params': 'N/A'
                    })
                    record['Data_Available'] = False
                
                # Load feature importance data
                fi_file = os.path.join(feature_dir, "{}_{}_feature_importance.csv".format(tissue, ml_method))
                gene_info = load_single_feature_importance(fi_file, top_n=10)
                record.update(gene_info)
                
                # Add availability flags
                record['Performance_File_Exists'] = os.path.exists(perf_file)
                record['Feature_Importance_File_Exists'] = os.path.exists(fi_file)
                
                fold_records.append(record)
    
    print("    📋 Created {} records for Fold {}".format(len(fold_records), fold_num))
    
    # Convert to DataFrame
    fold_df = pd.DataFrame(fold_records)
    
    if len(fold_df) == 0:
        print("    ⚠️  No records created for Fold {}".format(fold_num))
        return pd.DataFrame(), {'fold': fold_num, 'total_combinations': 0, 'completed_combinations': 0, 'completion_rate': 0}
    
    # Add some computed columns
    if 'Test_AUC' in fold_df.columns and 'Train_AUC' in fold_df.columns:
        fold_df['AUC_Diff_Test_Train'] = fold_df['Test_AUC'] - fold_df['Train_AUC']
    if 'Test_AUC' in fold_df.columns and 'Val_AUC' in fold_df.columns:
        fold_df['AUC_Diff_Test_Val'] = fold_df['Test_AUC'] - fold_df['Val_AUC']
    
    # Sort by Test AUC (descending) for easier analysis
    try:
        fold_df = fold_df.sort_values(['Test_AUC'], ascending=False, na_position='last')
    except Exception as e:
        print("    ⚠️  Could not sort by Test_AUC: {}".format(e))
    
    print("    ✅ Created table with {} rows ({} combinations)".format(len(fold_df), total_combinations))
    
    # Basic statistics for this fold
    valid_data = fold_df[fold_df['Data_Available'] == True]
    completion_rate = len(valid_data) / len(fold_df) * 100 if len(fold_df) > 0 else 0
    
    fold_stats = {
        'fold': fold_num,
        'total_combinations': len(fold_df),
        'completed_combinations': len(valid_data),
        'completion_rate': completion_rate
    }
    
    if len(valid_data) > 0:
        fold_stats.update({
            'mean_test_auc': valid_data['Test_AUC'].mean(),
            'max_test_auc': valid_data['Test_AUC'].max(),
            'min_test_auc': valid_data['Test_AUC'].min(),
            'std_test_auc': valid_data['Test_AUC'].std()
        })
        
        # Best combination for this fold
        if valid_data['Test_AUC'].notna().any():
            best_idx = valid_data['Test_AUC'].idxmax()
            best_combo = valid_data.loc[best_idx]
            fold_stats['best_tissue'] = best_combo['Tissue']
            fold_stats['best_method'] = best_combo['ML_Method']
            fold_stats['best_features'] = best_combo['Feature_Count']
            fold_stats['best_test_auc'] = best_combo['Test_AUC']
    
    print("    📈 Fold {} stats: {:.1f}% complete, {} valid combinations".format(
        fold_num, completion_rate, len(valid_data)))
    
    return fold_df, fold_stats



def save_fold_table(fold_df, fold_stats, fold_num, expression_method):
    """Save the merged table for a single fold"""
    
    # Create output directory for this fold
    fold_output_dir = "{}/Database/{}/Fold_{}/".format(PHENOTYPE, expression_method, fold_num)
    os.makedirs(fold_output_dir, exist_ok=True)
    
    # Save main merged table for this fold
    main_file = os.path.join(fold_output_dir, "{}_{}_Fold{}_AllCombinations.csv".format(
        PHENOTYPE, expression_method, fold_num))
    fold_df.to_csv(main_file, index=False, float_format='%.6f')
    
    # Save fold statistics
    stats_file = os.path.join(fold_output_dir, "{}_{}_Fold{}_Statistics.json".format(
        PHENOTYPE, expression_method, fold_num))
    
    with open(stats_file, 'w') as f:
        json.dump(fold_stats, f, indent=2, default=str)
    
    print("    💾 Saved: {}".format(main_file))
    
    return main_file, stats_file

def load_all_fold_results_for_analysis(expression_method):
    """Load results from all fold tables for best performers analysis"""
    
    print("\n📂 Loading results from all fold tables for analysis...")
    
    # Find all fold result files
    base_dir = "{}/Database/{}/".format(PHENOTYPE, expression_method)
    
    if not os.path.exists(base_dir):
        print("❌ Results directory not found for analysis: {}".format(base_dir))
        return None
    
    # Look for fold tables
    fold_files = []
    for fold_num in EXPECTED_FOLDS:
        fold_file = os.path.join(base_dir, "Fold_{}/{}_{}_Fold{}_AllCombinations.csv".format(
            fold_num, PHENOTYPE, expression_method, fold_num))
        
        if os.path.exists(fold_file):
            fold_files.append((fold_num, fold_file))
        else:
            print("  ⚠️  Missing Fold {} for analysis: {}".format(fold_num, fold_file))
    
    if not fold_files:
        print("❌ No fold result files found for analysis!")
        return None
    
    # Load all fold data
    all_fold_data = []
    
    for fold_num, fold_file in fold_files:
        try:
            df = pd.read_csv(fold_file)
            print("    📊 Loaded Fold {} for analysis: {} combinations".format(fold_num, len(df)))
            all_fold_data.append(df)
            
        except Exception as e:
            print("    ❌ Error loading Fold {} for analysis: {}".format(fold_num, e))
    
    if not all_fold_data:
        print("❌ Failed to load any fold data for analysis!")
        return None
    
    # Combine all fold data
    print("  📋 Combining all fold data for analysis...")
    combined_df = pd.concat(all_fold_data, ignore_index=True)
    
    print("  ✅ Combined data for analysis: {} total combinations from {} folds".format(len(combined_df), len(all_fold_data)))
    
    return combined_df

def calculate_cross_fold_averages(combined_df, expression_method):
    """Calculate averages across folds for each tissue + method + feature combination"""
    
    print("\n🧮 Calculating cross-fold averages...")
    
    # Filter to only completed combinations (with valid data)
    valid_data = combined_df[combined_df['Data_Available'] == True].copy()
    
    if len(valid_data) == 0:
        print("❌ No valid data found for averaging!")
        return None
    
    print("  📊 Using {} valid combinations out of {} total for averaging".format(len(valid_data), len(combined_df)))
    
    # Group by tissue, ML method, and feature count
    grouping_cols = ['Tissue', 'ML_Method', 'Feature_Count']
    
    # Calculate statistics across folds
    fold_averages = valid_data.groupby(grouping_cols).agg({
        'Train_AUC': ['mean', 'std', 'count', 'min', 'max'],
        'Val_AUC': ['mean', 'std', 'count', 'min', 'max'],
        'Test_AUC': ['mean', 'std', 'count', 'min', 'max'],
        'Features_Used': 'mean',
        'Train_Samples': 'mean',
        'Val_Samples': 'mean',
        'Test_Samples': 'mean',
        'Fold': lambda x: list(sorted(x))  # Track which folds contributed
    }).round(6)
    
    # Flatten column names
    fold_averages.columns = ['_'.join(col).strip() if col[1] else col[0] 
                           for col in fold_averages.columns.values]
    
    # Clean up column names
    fold_averages = fold_averages.rename(columns={
        'Train_AUC_mean': 'Avg_Train_AUC',
        'Train_AUC_std': 'Std_Train_AUC',
        'Train_AUC_count': 'N_Folds_Train',
        'Train_AUC_min': 'Min_Train_AUC',
        'Train_AUC_max': 'Max_Train_AUC',
        'Val_AUC_mean': 'Avg_Val_AUC',
        'Val_AUC_std': 'Std_Val_AUC',
        'Val_AUC_count': 'N_Folds_Val',
        'Val_AUC_min': 'Min_Val_AUC',
        'Val_AUC_max': 'Max_Val_AUC',
        'Test_AUC_mean': 'Avg_Test_AUC',
        'Test_AUC_std': 'Std_Test_AUC',
        'Test_AUC_count': 'N_Folds_Test',
        'Test_AUC_min': 'Min_Test_AUC',
        'Test_AUC_max': 'Max_Test_AUC',
        'Features_Used_': 'Avg_Features_Used',
        'Train_Samples_': 'Avg_Train_Samples',
        'Val_Samples_': 'Avg_Val_Samples',
        'Test_Samples_': 'Avg_Test_Samples',
        'Fold_<lambda>': 'Contributing_Folds'
    })
    
    # Reset index to make grouping columns regular columns
    fold_averages = fold_averages.reset_index()
    
    # Add metadata
    fold_averages['Phenotype'] = PHENOTYPE
    fold_averages['Expression_Method'] = expression_method
    
    print("  ✅ Calculated averages for {} unique tissue+method+feature combinations".format(len(fold_averages)))
    
    return fold_averages

def find_best_performers_by_method_feature(fold_averages):
    """Find best performing tissue for each ML method + feature combination"""
    
    print("\n🏆 Finding best tissue for each ML method + feature combination...")
    
    best_performers = {}
    
    # Find best for each ML Method + Feature Count combination
    for ml_method in ML_METHODS:
        best_performers[ml_method] = {}
        
        for feature_count in FEATURE_COUNTS:
            # Filter to this specific method + feature combination
            method_feature_data = fold_averages[
                (fold_averages['ML_Method'] == ml_method) & 
                (fold_averages['Feature_Count'] == feature_count)
            ].copy()
            
            if len(method_feature_data) == 0:
                print("  ⚠️  No data for {} with {} features".format(ml_method, feature_count))
                continue
            
            # Sort by Test AUC (descending)
            method_feature_data = method_feature_data.sort_values('Avg_Test_AUC', ascending=False)
            
            # Get top performers
            top_10 = method_feature_data.head(10)
            
            best_performers[ml_method][feature_count] = {
                'top_10': top_10,
                'best_tissue': top_10.iloc[0]['Tissue'],
                'best_test_auc': top_10.iloc[0]['Avg_Test_AUC'],
                'best_val_auc': top_10.iloc[0]['Avg_Val_AUC'],
                'best_train_auc': top_10.iloc[0]['Avg_Train_AUC'],
                'best_test_std': top_10.iloc[0]['Std_Test_AUC'],
                'n_folds': top_10.iloc[0]['N_Folds_Test'],
                'total_tissues_tested': len(method_feature_data)
            }
            
            print("    ✅ {} + {} features: Best = {} (Test AUC: {:.4f} ± {:.4f})".format(
                ml_method, feature_count, 
                best_performers[ml_method][feature_count]['best_tissue'],
                best_performers[ml_method][feature_count]['best_test_auc'],
                best_performers[ml_method][feature_count]['best_test_std']
            ))
    
    return best_performers

def create_best_performers_summary(best_performers, expression_method):
    """Create summary table of best performers - best tissue per method+feature combination"""
    
    print("\n📋 Creating best performers summary table...")
    
    summary_records = []
    
    for ml_method in ML_METHODS:
        for feature_count in FEATURE_COUNTS:
            if feature_count in best_performers.get(ml_method, {}):
                bp = best_performers[ml_method][feature_count]
                
                summary_records.append({
                    'ML_Method': ml_method,
                    'Feature_Count': feature_count,
                    'Best_Tissue': bp['best_tissue'],
                    'Avg_Test_AUC': bp['best_test_auc'],
                    'Std_Test_AUC': bp['best_test_std'],
                    'Avg_Val_AUC': bp['best_val_auc'],
                    'Avg_Train_AUC': bp['best_train_auc'],
                    'N_Folds': bp.get('n_folds', 0),
                    'Total_Tissues_Tested': bp['total_tissues_tested'],
                    'Phenotype': PHENOTYPE,
                    'Expression_Method': expression_method
                })
    
    summary_df = pd.DataFrame(summary_records)
    
    # Sort by Test AUC (descending)
    summary_df = summary_df.sort_values('Avg_Test_AUC', ascending=False)
    
    print("  ✅ Created summary with {} method+feature combinations".format(len(summary_df)))
    
    return summary_df

def find_best_method_feature_per_tissue(fold_averages):
    """Find best method+feature combination for each tissue"""
    
    print("\n🧪 Finding best method+feature combination for each tissue...")
    
    best_per_tissue = {}
    
    for tissue in fold_averages['Tissue'].unique():
        # Filter to this tissue
        tissue_data = fold_averages[fold_averages['Tissue'] == tissue].copy()
        
        if len(tissue_data) == 0:
            continue
        
        # Sort by Test AUC (descending)
        tissue_data = tissue_data.sort_values('Avg_Test_AUC', ascending=False)
        tissue_data = tissue_data.reset_index(drop=True)
        tissue_data['Rank'] = tissue_data.index + 1
        
        # Get best combination
        best_combo = tissue_data.iloc[0]
        
        best_per_tissue[tissue] = {
            'best_combination': best_combo,
            'all_combinations': tissue_data,
            'best_ml_method': best_combo['ML_Method'],
            'best_feature_count': best_combo['Feature_Count'],
            'best_test_auc': best_combo['Avg_Test_AUC'],
            'best_val_auc': best_combo['Avg_Val_AUC'],
            'best_train_auc': best_combo['Avg_Train_AUC'],
            'test_std': best_combo['Std_Test_AUC'],
            'n_combinations_tested': len(tissue_data)
        }
        
        print("    ✅ {}: {} + {}F (Test AUC: {:.4f} ± {:.4f})".format(
            tissue, best_combo['ML_Method'], best_combo['Feature_Count'],
            best_combo['Avg_Test_AUC'], best_combo['Std_Test_AUC']
        ))
    
    return best_per_tissue

def create_tissue_summary_table(best_per_tissue, expression_method):
    """Create summary table of best method+feature per tissue"""
    
    print("\n📊 Creating tissue summary table...")
    
    tissue_records = []
    
    for tissue, tissue_info in best_per_tissue.items():
        tissue_records.append({
            'Tissue': tissue,
            'Best_ML_Method': tissue_info['best_ml_method'],
            'Best_Feature_Count': tissue_info['best_feature_count'],
            'Avg_Test_AUC': tissue_info['best_test_auc'],
            'Std_Test_AUC': tissue_info['test_std'],
            'Avg_Val_AUC': tissue_info['best_val_auc'],
            'Avg_Train_AUC': tissue_info['best_train_auc'],
            'N_Combinations_Tested': tissue_info['n_combinations_tested'],
            'Phenotype': PHENOTYPE,
            'Expression_Method': expression_method
        })
    
    tissue_summary_df = pd.DataFrame(tissue_records)
    
    # Sort by Test AUC (descending)
    tissue_summary_df = tissue_summary_df.sort_values('Avg_Test_AUC', ascending=False)
    
    print("  ✅ Created tissue summary with {} tissues".format(len(tissue_summary_df)))
    
    return tissue_summary_df

def create_detailed_rankings(fold_averages):
    """Create detailed rankings for each method+feature combination"""
    
    print("\n📊 Creating detailed rankings...")
    
    detailed_rankings = {}
    
    for ml_method in ML_METHODS:
        detailed_rankings[ml_method] = {}
        
        for feature_count in FEATURE_COUNTS:
            # Filter and sort by Test AUC
            method_data = fold_averages[
                (fold_averages['ML_Method'] == ml_method) & 
                (fold_averages['Feature_Count'] == feature_count)
            ].copy()
            
            if len(method_data) > 0:
                method_data = method_data.sort_values('Avg_Test_AUC', ascending=False)
                method_data = method_data.reset_index(drop=True)
                method_data['Rank'] = method_data.index + 1
                
                detailed_rankings[ml_method][feature_count] = method_data
    
    return detailed_rankings

def create_simple_best_per_tissue_table(fold_averages):
    """Create simple table with best method+feature combination per tissue based on TEST AUC"""
    
    print("\n🧪 Creating simple best performers table (one row per tissue, ranked by TEST AUC)...")
    
    simple_results = []
    unique_tissues = fold_averages['Tissue'].unique()
    
    for tissue in unique_tissues:
        # Filter to this specific tissue
        tissue_data = fold_averages[fold_averages['Tissue'] == tissue].copy()
        
        if len(tissue_data) == 0:
            continue
        
        # Sort by TEST AUC (descending) to find best combination
        tissue_data = tissue_data.sort_values('Avg_Test_AUC', ascending=False)
        
        # Get the best combination for this tissue based on TEST performance
        best_combo = tissue_data.iloc[0]
        
        # Create simple record with ONLY the 4 essential columns
        simple_results.append({
            'Tissue': tissue,
            'Test_AUC': best_combo['Avg_Test_AUC'],
            'Train_AUC': best_combo['Avg_Train_AUC'],
            'Val_AUC': best_combo['Avg_Val_AUC']
        })
    
    # Create DataFrame and sort by TEST AUC (descending)
    simple_df = pd.DataFrame(simple_results)
    simple_df = simple_df.sort_values('Test_AUC', ascending=False)
    
    print("  ✅ Created simple table for {} tissues (ranked by TEST AUC)".format(len(simple_df)))
    
    return simple_df

def load_best_combination_features(tissue, ml_method, feature_count, fold_averages, expression_method):
    """Load the actual feature list for the best combination of a tissue"""
    
    # Find a fold that has this combination
    best_combo_folds = fold_averages[
        (fold_averages['Tissue'] == tissue) & 
        (fold_averages['ML_Method'] == ml_method) & 
        (fold_averages['Feature_Count'] == feature_count)
    ]['Contributing_Folds'].iloc[0]
    
    # Try to load features from the first available fold
    for fold_num in best_combo_folds:
        feature_file_path = "{}/Fold_{}/EnhancedMLResults/{}/{}_features/{}_{}_feature_importance.csv".format(
            PHENOTYPE, fold_num, expression_method, feature_count, tissue, ml_method
        )
        
        if os.path.exists(feature_file_path):
            try:
                feature_df = pd.read_csv(feature_file_path)
                
                # Clean feature names by removing version numbers after the period
                if 'Feature_Name' in feature_df.columns:
                    feature_df['Feature_Name'] = feature_df['Feature_Name'].str.split('.').str[0]
                
                # Return sorted by importance (should already be sorted)
                feature_df = feature_df.sort_values('Feature_Importance', ascending=False)
                return feature_df
            except Exception as e:
                print("      ⚠️  Failed to load features from {}: {}".format(feature_file_path, e))
                continue
    
    return None

def find_best_test_auc_combination_with_available_features(tissue, fold_averages, expression_method):
    """Find the combination with best TEST AUC that actually has feature files available"""
    
    # Get all combinations for this tissue, sorted by TEST AUC
    tissue_data = fold_averages[fold_averages['Tissue'] == tissue].copy()
    tissue_data = tissue_data.sort_values('Avg_Test_AUC', ascending=False)
    
    # Try each combination in order of TEST AUC performance
    for i, (_, combo) in enumerate(tissue_data.iterrows()):
        ml_method = combo['ML_Method']
        feature_count = combo['Feature_Count']
        test_auc = combo['Avg_Test_AUC']
        
        # Check if feature files exist for this combination
        feature_df = load_best_combination_features(tissue, ml_method, feature_count, fold_averages, expression_method)
        
        if feature_df is not None:
            return combo, feature_df
    
    return None, None

def save_best_combination_features(simple_df, fold_averages, output_dir, expression_method):
    """Save the feature lists for all best combinations"""
    
    print("\n💾 Saving feature lists for best TEST AUC combinations...")
    
    # Create features directory
    features_dir = os.path.join(output_dir, "BestCombination_Features")
    os.makedirs(features_dir, exist_ok=True)
    
    features_saved = 0
    features_metadata = []
    
    for _, row in simple_df.iterrows():
        tissue = row['Tissue']
        
        # Find the best TEST AUC combination that actually has feature files
        best_combo, feature_df = find_best_test_auc_combination_with_available_features(tissue, fold_averages, expression_method)
        
        if best_combo is not None and feature_df is not None:
            ml_method = best_combo['ML_Method']
            feature_count = best_combo['Feature_Count']
            
            # Save the complete feature list
            feature_file = os.path.join(features_dir, "{}_{}_{}F_features.csv".format(
                tissue, ml_method, feature_count))
            feature_df.to_csv(feature_file, index=False, float_format='%.6f')
            
            features_saved += 1
            
            # Add metadata
            features_metadata.append({
                'Tissue': tissue,
                'ML_Method': ml_method,
                'Feature_Count': feature_count,
                'Test_AUC': best_combo['Avg_Test_AUC'],
                'Train_AUC': best_combo['Avg_Train_AUC'],
                'Val_AUC': best_combo['Avg_Val_AUC'],
                'N_Features_Saved': len(feature_df),
                'Top_Gene': feature_df.iloc[0]['Feature_Name'] if len(feature_df) > 0 else 'N/A',
                'Top_Gene_Importance': feature_df.iloc[0]['Feature_Importance'] if len(feature_df) > 0 else 'N/A',
                'Feature_File': feature_file,
                'Selection_Method': 'Best_Test_AUC_With_Available_Features'
            })
        else:
            # Add empty metadata to track missing features
            features_metadata.append({
                'Tissue': tissue,
                'ML_Method': 'N/A',
                'Feature_Count': 'N/A',
                'Test_AUC': 'N/A',
                'Train_AUC': 'N/A',
                'Val_AUC': 'N/A',
                'N_Features_Saved': 0,
                'Top_Gene': 'N/A',
                'Top_Gene_Importance': 'N/A',
                'Feature_File': 'N/A',
                'Selection_Method': 'No_Features_Available'
            })
    
    # Save features metadata summary
    if features_metadata:
        metadata_df = pd.DataFrame(features_metadata)
        metadata_file = os.path.join(features_dir, "BestCombinations_Features_Summary.csv")
        metadata_df.to_csv(metadata_file, index=False, float_format='%.6f')
        
        print("  ✅ Saved features metadata: {}".format(metadata_file))
        print("  📊 Features saved for {}/{} tissues".format(features_saved, len(simple_df)))
        
        return features_dir, features_saved, metadata_file
    else:
        print("  ❌ No features could be saved!")
        return features_dir, 0, None

def save_best_performers_analysis(fold_averages, best_performers_summary, detailed_rankings, best_performers, 
                                 best_per_tissue, tissue_summary_df, expression_method):
    """Save all best performers analysis results - COMPREHENSIVE ANALYSIS"""
    
    print("\n💾 Saving comprehensive best performers analysis results...")
    
    # Create output directory
    output_dir = "{}/Database/{}/BestPerformers/".format(PHENOTYPE, expression_method)
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = []
    
    # 1. Save complete cross-fold averages
    averages_file = os.path.join(output_dir, "{}_{}_CrossFoldAverages.csv".format(PHENOTYPE, expression_method))
    fold_averages.to_csv(averages_file, index=False, float_format='%.6f')
    saved_files.append(averages_file)
    print("    ✅ Saved cross-fold averages: {}".format(averages_file))
    
    # 2. Save best performers summary (best tissue per method+feature)
    summary_file = os.path.join(output_dir, "{}_{}_BestTissuePerMethodFeature.csv".format(PHENOTYPE, expression_method))
    best_performers_summary.to_csv(summary_file, index=False, float_format='%.6f')
    saved_files.append(summary_file)
    print("    ✅ Saved best tissue per method+feature: {}".format(summary_file))
    
    # 3. Save best method+feature per tissue summary
    tissue_summary_file = os.path.join(output_dir, "{}_{}_BestMethodFeaturePerTissue.csv".format(PHENOTYPE, expression_method))
    tissue_summary_df.to_csv(tissue_summary_file, index=False, float_format='%.6f')
    saved_files.append(tissue_summary_file)
    print("    ✅ Saved best method+feature per tissue: {}".format(tissue_summary_file))
    
    # 4. Save detailed rankings for each method+feature combination
    rankings_dir = os.path.join(output_dir, "DetailedRankings_ByMethodFeature")
    os.makedirs(rankings_dir, exist_ok=True)
    
    for ml_method in ML_METHODS:
        for feature_count in FEATURE_COUNTS:
            if feature_count in detailed_rankings.get(ml_method, {}):
                ranking_data = detailed_rankings[ml_method][feature_count]
                
                ranking_file = os.path.join(rankings_dir, "{}_{}_{}Features_TissueRankings.csv".format(
                    ml_method, expression_method, feature_count))
                ranking_data.to_csv(ranking_file, index=False, float_format='%.6f')
                saved_files.append(ranking_file)
    
    print("    ✅ Saved {} detailed ranking files (by method+feature)".format(len([f for f in saved_files if 'Rankings' in f])))
    
    # 5. Save detailed rankings for each tissue (all method+feature combinations per tissue)
    tissue_rankings_dir = os.path.join(output_dir, "DetailedRankings_ByTissue")
    os.makedirs(tissue_rankings_dir, exist_ok=True)
    
    for tissue in best_per_tissue.keys():
        tissue_data = best_per_tissue[tissue]['all_combinations']
        
        tissue_ranking_file = os.path.join(tissue_rankings_dir, "{}_{}_MethodFeatureRankings.csv".format(
            tissue, expression_method))
        tissue_data.to_csv(tissue_ranking_file, index=False, float_format='%.6f')
        saved_files.append(tissue_ranking_file)
    
    print("    ✅ Saved {} tissue ranking files (method+feature per tissue)".format(len(best_per_tissue)))
    
    # 6. Save top 10 performers for each method+feature
    top10_dir = os.path.join(output_dir, "Top10PerMethodFeature")
    os.makedirs(top10_dir, exist_ok=True)
    
    for ml_method in ML_METHODS:
        for feature_count in FEATURE_COUNTS:
            if feature_count in best_performers.get(ml_method, {}):
                top10_data = best_performers[ml_method][feature_count]['top_10']
                
                top10_file = os.path.join(top10_dir, "{}_{}_{}Features_Top10Tissues.csv".format(
                    ml_method, expression_method, feature_count))
                top10_data.to_csv(top10_file, index=False, float_format='%.6f')
                saved_files.append(top10_file)
    
    print("    ✅ Saved {} top-10 tissue files".format(len([f for f in saved_files if 'Top10' in f])))
    
    # 7. Save analysis metadata
    metadata = {
        'phenotype': PHENOTYPE,
        'expression_method': expression_method,
        'analyses_performed': [
            'best_tissue_per_method_feature',
            'best_method_feature_per_tissue',
            'detailed_rankings_by_method_feature',
            'detailed_rankings_by_tissue',
            'top10_performers_per_method_feature'
        ],
        'ml_methods': ML_METHODS,
        'feature_counts': FEATURE_COUNTS,
        'total_tissues_analyzed': len(best_per_tissue),
        'total_method_feature_combinations': len(best_performers_summary),
        'files_created': len(saved_files)
    }
    
    metadata_file = os.path.join(output_dir, "{}_{}_Analysis_Metadata.json".format(PHENOTYPE, expression_method))
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    saved_files.append(metadata_file)
    
    print("    ✅ Saved analysis metadata: {}".format(metadata_file))
    
    return saved_files, output_dir

def save_simple_best_performers_table(simple_df, expression_method):
    """Save the simple best performers table with 4 columns"""
    
    print("\n💾 Saving simple best performers table...")
    
    # Create output directory
    output_dir = "{}/Database/{}/".format(PHENOTYPE, expression_method)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the simple table (4 columns only)
    simple_file = os.path.join(output_dir, "{}_{}_BestPerTissue_TestBased.csv".format(PHENOTYPE, expression_method))
    simple_df.to_csv(simple_file, index=False, float_format='%.6f')
    
    print("  ✅ Saved simple table: {}".format(simple_file))
    print("  📄 File has {} columns: Tissue, Test_AUC, Train_AUC, Val_AUC".format(len(simple_df.columns)))
    print("  📄 File has {} tissues".format(len(simple_df)))
    
    return simple_file

def display_comprehensive_summary(best_performers_summary, tissue_summary_df):
    """Display comprehensive summary of all analyses"""
    
    print("\n🎉 COMPREHENSIVE ANALYSIS COMPLETE!")
    print("=" * 50)
    
    # === ANALYSIS 1: Best Tissue per Method+Feature ===
    print("\n📊 ANALYSIS 1: BEST TISSUE FOR EACH METHOD+FEATURE COMBINATION")
    print("-" * 50)
    
    if len(best_performers_summary) > 0:
        overall_best = best_performers_summary.iloc[0]
        print("🏆 OVERALL BEST COMBINATION:")
        print("   Method: {}".format(overall_best['ML_Method']))
        print("   Features: {}".format(overall_best['Feature_Count']))
        print("   Best Tissue: {}".format(overall_best['Best_Tissue']))
        print("   Test AUC: {:.4f} ± {:.4f}".format(overall_best['Avg_Test_AUC'], overall_best['Std_Test_AUC']))
        print("   Val AUC: {:.4f}".format(overall_best['Avg_Val_AUC']))
        
        print("\n📊 BEST TISSUE BY METHOD:")
        for method in ML_METHODS:
            method_best = best_performers_summary[best_performers_summary['ML_Method'] == method]
            if len(method_best) > 0:
                best_for_method = method_best.iloc[0]
                print("   {}: {} ({}F) - Test AUC: {:.4f}".format(
                    method, best_for_method['Best_Tissue'], 
                    best_for_method['Feature_Count'], best_for_method['Avg_Test_AUC']
                ))
    
    # === ANALYSIS 2: Best Method+Feature per Tissue ===
    print("\n🧪 ANALYSIS 2: BEST METHOD+FEATURE FOR EACH TISSUE")
    print("-" * 50)
    
    if len(tissue_summary_df) > 0:
        tissue_best = tissue_summary_df.iloc[0]
        print("🏆 OVERALL BEST TISSUE:")
        print("   Tissue: {}".format(tissue_best['Tissue']))
        print("   Best Method: {}".format(tissue_best['Best_ML_Method']))
        print("   Best Features: {}".format(tissue_best['Best_Feature_Count']))
        print("   Test AUC: {:.4f} ± {:.4f}".format(tissue_best['Avg_Test_AUC'], tissue_best['Std_Test_AUC']))
        print("   Val AUC: {:.4f}".format(tissue_best['Avg_Val_AUC']))
        
        print("\n🧪 TOP 5 TISSUES (by best Test AUC):")
        for i, (_, tissue_data) in enumerate(tissue_summary_df.head(5).iterrows(), 1):
            print("   {}. {}: {} + {}F - Test AUC: {:.4f}".format(
                i, tissue_data['Tissue'], tissue_data['Best_ML_Method'],
                tissue_data['Best_Feature_Count'], tissue_data['Avg_Test_AUC']
            ))
        
        # Method popularity analysis
        print("\n📈 METHOD POPULARITY AMONG TISSUES:")
        method_counts = tissue_summary_df['Best_ML_Method'].value_counts()
        for method, count in method_counts.items():
            percentage = (count / len(tissue_summary_df)) * 100
            print("   {}: {} tissues ({:.1f}%)".format(method, count, percentage))
        
        # Feature count popularity analysis
        print("\n🔢 FEATURE COUNT POPULARITY AMONG TISSUES:")
        feature_counts = tissue_summary_df['Best_Feature_Count'].value_counts().sort_index()
        for feature_count, count in feature_counts.items():
            percentage = (count / len(tissue_summary_df)) * 100
            print("   {} features: {} tissues ({:.1f}%)".format(feature_count, count, percentage))
        
        # Performance statistics
        print("\n📈 TISSUE TEST AUC STATISTICS:")
        test_aucs = tissue_summary_df['Avg_Test_AUC']
        print("   Mean: {:.4f}".format(test_aucs.mean()))
        print("   Std: {:.4f}".format(test_aucs.std()))
        print("   Range: {:.4f} - {:.4f}".format(test_aucs.min(), test_aucs.max()))
        print("   Tissues analyzed: {}".format(len(tissue_summary_df)))

def run_comprehensive_analysis(expression_method):
    """Run the comprehensive best performers analysis"""
    
    print("\n" + "=" * 80)
    print("🏆 RUNNING COMPREHENSIVE ANALYSIS for {}".format(expression_method))
    print("=" * 80)
    
    try:
        # Load all fold results for analysis
        combined_df = load_all_fold_results_for_analysis(expression_method)
        if combined_df is None:
            print("⚠️  Skipping analysis - no fold data available")
            return False
        
        # Calculate cross-fold averages
        fold_averages = calculate_cross_fold_averages(combined_df, expression_method)
        if fold_averages is None:
            print("⚠️  Skipping analysis - no valid data for averaging")
            return False
        
        # === COMPREHENSIVE ANALYSIS 1: Best tissue per method+feature ===
        best_performers = find_best_performers_by_method_feature(fold_averages)
        best_performers_summary = create_best_performers_summary(best_performers, expression_method)
        
        # === COMPREHENSIVE ANALYSIS 2: Best method+feature per tissue ===
        best_per_tissue = find_best_method_feature_per_tissue(fold_averages)
        tissue_summary_df = create_tissue_summary_table(best_per_tissue, expression_method)
        
        # === DETAILED RANKINGS ===
        detailed_rankings = create_detailed_rankings(fold_averages)
        
        # === SAVE ALL COMPREHENSIVE ANALYSIS RESULTS ===
        saved_files, analysis_dir = save_best_performers_analysis(
            fold_averages, best_performers_summary, detailed_rankings, best_performers, 
            best_per_tissue, tissue_summary_df, expression_method
        )
        
        # === CREATE AND SAVE SIMPLE TABLE ===
        simple_df = create_simple_best_per_tissue_table(fold_averages)
        simple_file = save_simple_best_performers_table(simple_df, expression_method)
        
        # === SAVE FEATURES FOR BEST COMBINATIONS ===
        output_dir = "{}/Database/{}/".format(PHENOTYPE, expression_method)
        features_dir, features_saved, metadata_file = save_best_combination_features(simple_df, fold_averages, output_dir, expression_method)
        
        # === DISPLAY COMPREHENSIVE SUMMARY ===
        display_comprehensive_summary(best_performers_summary, tissue_summary_df)
        
        print("\n📁 COMPREHENSIVE OUTPUT FILES:")
        print("  🎯 Simple table: {}".format(simple_file))
        print("  📊 Comprehensive analysis: {}".format(analysis_dir))
        print("  🧬 Features directory: {}".format(features_dir))
        print("  ✅ Total analysis files: {}".format(len(saved_files)))
        print("  ✅ Features saved for {}/{} tissues".format(features_saved, len(simple_df)))
        
        return True
        
    except Exception as e:
        print("❌ Comprehensive analysis failed: {}".format(e))
        traceback.print_exc()
        return False

def generate_cross_database_heatmap(expression_methods):
    """Generate a heatmap comparing test performance across multiple databases"""
    
    print(f"\n{'🔥' * 40}")
    print("🔥 GENERATING CROSS-DATABASE HEATMAP...")
    print(f"{'🔥' * 40}")
    
    # Collect data from all databases
    heatmap_data = {}
    all_tissues = set()
    
    for method in expression_methods:
        try:
            # Load the best per tissue test-based file for this method
            result_file = f"{PHENOTYPE}/Database/{method}/{PHENOTYPE}_{method}_BestPerTissue_TestBased.csv"
            
            if os.path.exists(result_file):
                df = pd.read_csv(result_file)
                print(f"✅ Loaded {len(df)} tissues from {method}")
                
                # Add to heatmap data
                heatmap_data[method] = df.set_index('Tissue')['Test_AUC'].to_dict()
                all_tissues.update(df['Tissue'].tolist())
            else:
                print(f"❌ File not found: {result_file}")
                heatmap_data[method] = {}
                
        except Exception as e:
            print(f"❌ Error loading {method}: {e}")
            heatmap_data[method] = {}
    
    if not heatmap_data or not all_tissues:
        print("❌ No data found for heatmap generation")
        return False
    
    # Create DataFrame for heatmap
    all_tissues = sorted(list(all_tissues))
    heatmap_df = pd.DataFrame(index=all_tissues, columns=expression_methods)
    
    for method in expression_methods:
        for tissue in all_tissues:
            heatmap_df.loc[tissue, method] = heatmap_data[method].get(tissue, np.nan)
    
    # Convert to numeric
    heatmap_df = heatmap_df.astype(float)
    
    print(f"📊 Heatmap dimensions: {heatmap_df.shape[0]} tissues × {heatmap_df.shape[1]} databases")
    print(f"📈 Test AUC range: {heatmap_df.min().min():.3f} - {heatmap_df.max().max():.3f}")
    
    # Create the heatmap
    plt.figure(figsize=(max(8, len(expression_methods) * 2), max(10, len(all_tissues) * 0.3)))
    
    # Generate heatmap
    mask = heatmap_df.isna()
    ax = sns.heatmap(
        heatmap_df, 
        annot=True, 
        fmt='.3f', 
        cmap='RdYlBu_r',
        center=0.55,  # Center around typical AUC values
        vmin=0.5, 
        vmax=0.7,
        mask=mask,
        cbar_kws={'label': 'Test AUC'},
        linewidths=0.5,
        square=False
    )
    
    # Customize the plot
    plt.title(f'{PHENOTYPE} - Test Performance Across Databases\n(Best ML Method)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Gene Expression Models', fontsize=12, fontweight='bold')
    plt.ylabel('Tissue', fontsize=12, fontweight='bold')
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the heatmap
    output_dir = f"{PHENOTYPE}/Database/"
    os.makedirs(output_dir, exist_ok=True)
    
    heatmap_file = f"{output_dir}{PHENOTYPE}_CrossDatabase_TestAUC_Heatmap.png"
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    
    # Also save as CSV
    csv_file = f"{output_dir}{PHENOTYPE}_CrossDatabase_TestAUC_Matrix.csv"
    heatmap_df.to_csv(csv_file)
    
    print(f"🔥 Heatmap saved: {heatmap_file}")
    print(f"📊 Data matrix saved: {csv_file}")
    
    plt.close()
    
    return True

def create_overall_summary(all_fold_stats, expression_method):
    """Create overall summary across all folds"""
    
    print("\n📊 Creating overall summary across folds...")
    
    summary = {
        'phenotype': PHENOTYPE,
        'expression_method': expression_method,
        'total_folds_processed': len(all_fold_stats),
        'fold_numbers': [stats['fold'] for stats in all_fold_stats]
    }
    
    # Calculate overall statistics
    if all_fold_stats:
        summary['overall_completion_rate'] = np.mean([stats['completion_rate'] for stats in all_fold_stats])
        summary['total_combinations_all_folds'] = sum([stats['total_combinations'] for stats in all_fold_stats])
        summary['total_completed_all_folds'] = sum([stats['completed_combinations'] for stats in all_fold_stats])
        
        # Performance statistics (only for folds with valid data)
        valid_fold_stats = [stats for stats in all_fold_stats if 'mean_test_auc' in stats]
        
        if valid_fold_stats:
            summary['mean_test_auc_across_folds'] = np.mean([stats['mean_test_auc'] for stats in valid_fold_stats])
            summary['std_test_auc_across_folds'] = np.std([stats['mean_test_auc'] for stats in valid_fold_stats])
            summary['best_test_auc_overall'] = max([stats['max_test_auc'] for stats in valid_fold_stats])
            summary['worst_test_auc_overall'] = min([stats['min_test_auc'] for stats in valid_fold_stats])
            
            # Find overall best combination
            best_fold_idx = np.argmax([stats['max_test_auc'] for stats in valid_fold_stats])
            best_fold_stats = valid_fold_stats[best_fold_idx]
            
            summary['overall_best_combination'] = {
                'fold': best_fold_stats['fold'],
                'tissue': best_fold_stats['best_tissue'],
                'ml_method': best_fold_stats['best_method'],
                'feature_count': best_fold_stats['best_features'],
                'test_auc': best_fold_stats['best_test_auc']
            }
    
    return summary

def process_single_expression_method(expression_method):
    """Process a single expression method"""
    
    # Find all fold directories for this method
    fold_info = find_fold_directories(expression_method)
    
    if not fold_info:
        print("❌ No fold directories with {} results found!".format(expression_method))
        print("Expected structure: {}/Fold_*/EnhancedMLResults/{}/".format(PHENOTYPE, expression_method))
        return False
    
    # Check for expected folds
    found_folds = [fold[0] for fold in fold_info]
    missing_folds = [i for i in EXPECTED_FOLDS if i not in found_folds]
    
    if missing_folds:
        print("⚠️  Missing folds: {}".format(missing_folds))
        print("   Found folds: {}".format(found_folds))
    else:
        print("✅ All 5 expected folds found: {}".format(found_folds))
    
    # Find available tissues
    tissues = get_available_tissues(fold_info)
    
    if not tissues:
        print("❌ No tissues found for {}!".format(expression_method))
        return False
    
    tissue_preview = ', '.join(tissues[:5])
    if len(tissues) > 5:
        tissue_preview += '...'
    print("\n🧪 Found {} tissues: {}".format(len(tissues), tissue_preview))
    
    # Calculate expected combinations per fold
    combinations_per_fold = len(tissues) * len(ML_METHODS) * len(FEATURE_COUNTS)
    
    print("\n📊 Per fold expectations:")
    print("  - {} tissues".format(len(tissues)))
    print("  - {} ML methods: {}".format(len(ML_METHODS), ', '.join(ML_METHODS)))
    print("  - {} feature counts: {}".format(len(FEATURE_COUNTS), FEATURE_COUNTS))
    print("  - {} combinations per fold".format(combinations_per_fold))
    
    # Process each fold
    all_saved_files = []
    all_fold_stats = []
    
    for fold_num, fold_dir, method_dir in fold_info:
        print("\n📁 Processing Fold {}...".format(fold_num))
        
        try:
            # Create merged table for this fold
            fold_df, fold_stats = create_fold_merged_table(fold_num, fold_dir, method_dir, tissues, expression_method)
            
            if len(fold_df) == 0:
                print("  ⚠️  No data found for Fold {}, skipping".format(fold_num))
                continue
            
            # Save fold table
            main_file, stats_file = save_fold_table(fold_df, fold_stats, fold_num, expression_method)
            
            all_saved_files.extend([main_file, stats_file])
            all_fold_stats.append(fold_stats)
            
            print("  ✅ Completed Fold {}: {} combinations saved".format(fold_num, len(fold_df)))
            
        except Exception as e:
            print("  ❌ Failed to process Fold {}: {}".format(fold_num, str(e)))
            traceback.print_exc()
            continue
    
    if not all_fold_stats:
        print("❌ No folds processed successfully!")
        return False
    
    # Create overall summary
    try:
        overall_summary = create_overall_summary(all_fold_stats, expression_method)
        
        # Save overall summary
        overall_summary_dir = "{}/Database/{}/".format(PHENOTYPE, expression_method)
        os.makedirs(overall_summary_dir, exist_ok=True)
        overall_summary_file = os.path.join(overall_summary_dir, "{}_{}_OverallSummary.json".format(
            PHENOTYPE, expression_method))
        
        with open(overall_summary_file, 'w') as f:
            json.dump(overall_summary, f, indent=2, default=str)
        
        all_saved_files.append(overall_summary_file)
        
    except Exception as e:
        print("⚠️  Failed to create overall summary: {}".format(e))
    
    # Final fold summary
    print("\n🎉 FOLD MERGING COMPLETE for {}!".format(expression_method))
    print("=" * 60)
    print("✅ Processed {} folds: {}".format(len(all_fold_stats), [stats['fold'] for stats in all_fold_stats]))
    print("✅ Created {} merged tables (one per fold)".format(len(all_fold_stats)))
    print("✅ Total files saved: {}".format(len(all_saved_files)))
    
    # Run comprehensive analysis
    print("\n" + "🔄" * 40)
    print("🚀 RUNNING COMPREHENSIVE ANALYSIS for {}...".format(expression_method))
    
    analysis_success = run_comprehensive_analysis(expression_method)
    
    if analysis_success:
        print("✅ Comprehensive analysis completed for {}".format(expression_method))
        return True
    else:
        print("⚠️  Fold merging completed but analysis failed for {}".format(expression_method))
        return False

def main():
    """Main function to create merged tables per fold for multiple expression methods"""
    
    print("🔄 Complete ML Results Merger Pipeline")
    print("📊 Phenotype: {}".format(PHENOTYPE))
    print("🧬 Expression Methods: {}".format(', '.join(EXPRESSION_METHODS)))
    print("🎯 Will process each method with comprehensive analysis and generate cross-database heatmap")
    print("=" * 80)
    
    successful_methods = []
    
    # Process each expression method
    for i, expression_method in enumerate(EXPRESSION_METHODS):
        print(f"\n{'🔬' * 20} METHOD {i+1}/{len(EXPRESSION_METHODS)} {'🔬' * 20}")
        print(f"🧬 Processing: {expression_method}")
        print("=" * 60)
        
        # Process this method
        success = process_single_expression_method(expression_method)
        
        if success:
            successful_methods.append(expression_method)
            print(f"✅ {expression_method} completed successfully")
        else:
            print(f"❌ {expression_method} failed")
    
    # Generate cross-database heatmap if we have multiple successful methods
    if len(successful_methods) >= 2:
        print(f"\n{'🔥' * 60}")
        print(f"🔥 GENERATING CROSS-DATABASE COMPARISON")
        print(f"🔥 Successful methods: {', '.join(successful_methods)}")
        print(f"{'🔥' * 60}")
        
        heatmap_success = generate_cross_database_heatmap(successful_methods)
        
        if heatmap_success:
            print(f"\n🎉 COMPLETE COMPREHENSIVE PIPELINE FINISHED!")
            print("=" * 80)
            print(f"✅ PROCESSED METHODS: {len(successful_methods)}/{len(EXPRESSION_METHODS)}")
            print(f"   📊 {', '.join(successful_methods)}")
            print("✅ COMPREHENSIVE ANALYSIS: Completed for all methods")
            print("   📊 Best tissue per method+feature combinations")
            print("   🧪 Best method+feature per tissue combinations") 
            print("   📈 Detailed rankings from all perspectives")
            print("   🔝 Top 10 performers analysis")
            print("   🧬 Feature importance for best combinations")
            print("✅ CROSS-DATABASE HEATMAP: Generated")
            print("   🔥 Tissues (Y-axis) vs Databases (X-axis)")
            print("   📈 Shows best test AUC per tissue per database")
            
            print(f"\n📁 FINAL OUTPUT STRUCTURE:")
            print(f"{PHENOTYPE}/Database/")
            for method in successful_methods:
                print(f"├── {method}/")
                print(f"│   ├── Fold_0/ to Fold_4/ (individual fold merged tables)")
                print(f"│   ├── {PHENOTYPE}_{method}_BestPerTissue_TestBased.csv")
                print(f"│   ├── {PHENOTYPE}_{method}_OverallSummary.json")
                print(f"│   ├── BestPerformers/ (comprehensive analysis)")
                print(f"│   │   ├── {PHENOTYPE}_{method}_CrossFoldAverages.csv")
                print(f"│   │   ├── {PHENOTYPE}_{method}_BestTissuePerMethodFeature.csv")
                print(f"│   │   ├── {PHENOTYPE}_{method}_BestMethodFeaturePerTissue.csv")
                print(f"│   │   ├── DetailedRankings_ByMethodFeature/")
                print(f"│   │   ├── DetailedRankings_ByTissue/")
                print(f"│   │   └── Top10PerMethodFeature/")
                print(f"│   └── BestCombination_Features/ (gene importance files)")
            print(f"├── {PHENOTYPE}_CrossDatabase_TestAUC_Heatmap.png")
            print(f"└── {PHENOTYPE}_CrossDatabase_TestAUC_Matrix.csv")
            
        else:
            print("⚠️  Methods processed but heatmap generation failed")
    else:
        print(f"\n⚠️  Only {len(successful_methods)} method(s) successful - need ≥2 for heatmap")
        if successful_methods:
            print(f"   Successful: {', '.join(successful_methods)}")

if __name__ == "__main__":
    main()