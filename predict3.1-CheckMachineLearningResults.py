#!/usr/bin/env python3
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

def check_ml_results_availability(phenotype, max_folds=10):
    """
    Check if ML model results exist for all folds, tissues, and databases
    
    Args:
        phenotype: Name of the phenotype (e.g., 'migraine')
        max_folds: Maximum number of folds to check (default: 10)
    
    Returns:
        tuple: (DataFrame with availability matrix, dict with tissues by method)
    """
    
    # All expression methods from your ML pipeline
    EXPRESSION_METHODS = [
        "Regular", "JTI", "UTMOST", "UTMOST2", 
        "EpiX", "TIGAR", "FUSION"
    ]
    
    # Find all available folds
    available_folds = []
    for fold in range(max_folds):
        fold_path = f"{phenotype}/Fold_{fold}/"
        if os.path.exists(fold_path):
            available_folds.append(fold)
    
    if not available_folds:
        print(f"❌ No folds found for phenotype: {phenotype}")
        return None, {}
    
    print(f"🔍 Found folds: {available_folds}")
    
    # Get all tissues and methods by checking ML results directories
    all_tissues = set()
    tissues_by_method = {}
    
    for fold in available_folds:
        fold_path = f"{phenotype}/Fold_{fold}/"
        results_dir = f"{fold_path}EnhancedMLResults/"
        
        if os.path.exists(results_dir):
            # Check each method subdirectory
            for method in EXPRESSION_METHODS:
                method_dir = os.path.join(results_dir, method)
                method_tissues = set()
                
                if os.path.exists(method_dir):
                    # Check feature subdirectories (100_features, 500_features, etc.)
                    feature_dirs = [d for d in os.listdir(method_dir) 
                                   if os.path.isdir(os.path.join(method_dir, d)) and d.endswith('_features')]
                    
                    for feature_dir in feature_dirs:
                        feature_path = os.path.join(method_dir, feature_dir)
                        # Look for result files to extract tissue names
                        result_files = glob(os.path.join(feature_path, "*_*_feature_importance.csv"))
                        
                        for file in result_files:
                            filename = os.path.basename(file)
                            # Extract tissue name (format: {tissue}_{model}_feature_importance.csv)
                            parts = filename.replace('_feature_importance.csv', '').split('_')
                            if len(parts) >= 2:
                                tissue = '_'.join(parts[:-1])  # Everything except the last part (model name)
                                method_tissues.add(tissue)
                                all_tissues.add(tissue)
                
                tissues_by_method[method] = sorted(list(method_tissues))
                if method_tissues:
                    print(f"  {method}: {len(method_tissues)} tissues found")
                else:
                    print(f"  {method}: No results found")
    
    all_tissues = sorted(list(all_tissues))
    print(f"🧪 Total unique tissues across all methods: {len(all_tissues)}")
    print(f"🧪 Tissues: {all_tissues}")
    
    # Check ML results availability
    availability_matrix = {}
    
    # Models that should be present (from your ML pipeline)
    expected_models = ['XGBoost', 'RandomForest', 'LogisticRegression']
    # Feature counts that should be present
    expected_features = [100, 500, 1000, 2000]
    
    for method in EXPRESSION_METHODS:
        availability_matrix[method] = {}
        
        # Get tissues available for this specific method
        method_tissues = tissues_by_method.get(method, [])
        
        for tissue in all_tissues:
            if tissue not in method_tissues:
                availability_matrix[method][tissue] = 0  # No results for this tissue-method combo
                continue
            
            # Check if ML results exist for ALL folds
            all_folds_complete = True
            
            for fold in available_folds:
                fold_path = f"{phenotype}/Fold_{fold}/"
                results_dir = f"{fold_path}EnhancedMLResults/"
                method_dir = os.path.join(results_dir, method)
                
                if not os.path.exists(method_dir):
                    all_folds_complete = False
                    break
                
                # Check if at least one feature count directory has complete results
                fold_has_results = False
                
                for n_features in expected_features:
                    feature_dir = os.path.join(method_dir, f"{n_features}_features")
                    
                    if os.path.exists(feature_dir):
                        # Check if this tissue has results for at least one model
                        tissue_has_model_results = False
                        
                        for model in expected_models:
                            # Check for the key result files
                            importance_file = os.path.join(feature_dir, f"{tissue}_{model}_feature_importance.csv")
                            performance_file = os.path.join(feature_dir, f"{tissue}_{model}_performance.csv")
                            model_file = os.path.join(feature_dir, f"{tissue}_{model}_model.pkl")
                            
                            if (os.path.exists(importance_file) and 
                                os.path.exists(performance_file) and 
                                os.path.exists(model_file)):
                                tissue_has_model_results = True
                                break
                        
                        if tissue_has_model_results:
                            fold_has_results = True
                            break  # Found results for this feature count
                
                if not fold_has_results:
                    all_folds_complete = False
                    break
            
            # Store result (1 if complete, 0 if missing)
            availability_matrix[method][tissue] = 1 if all_folds_complete else 0
    
    # Convert to DataFrame
    df = pd.DataFrame(availability_matrix)
    return df, tissues_by_method

def create_availability_heatmap(availability_df, phenotype, output_file=None):
    """
    Create heatmap showing ML results availability
    
    Args:
        availability_df: DataFrame with availability matrix
        phenotype: Phenotype name for title
        output_file: Optional output file path
    """
    
    if availability_df is None or availability_df.empty:
        print("❌ No data to plot")
        return
    
    # Create figure
    plt.figure(figsize=(12, max(8, len(availability_df) * 0.5)))
    
    # Create heatmap
    sns.heatmap(availability_df, 
                annot=True, 
                fmt='d', 
                cmap='RdYlGn',
                cbar_kws={'label': 'ML Results Available (1=Yes, 0=No)'},
                vmin=0, 
                vmax=1,
                linewidths=0.5)
    
    plt.title(f'ML Model Results Availability for {phenotype.upper()}\n(1=Complete across all folds, 0=Missing results)')
    plt.xlabel('Expression Methods (Databases)')
    plt.ylabel('Tissues')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"💾 Heatmap saved: {output_file}")
    else:
        plt.savefig(f'{phenotype}_ml_results_availability.png', dpi=300, bbox_inches='tight')
        print(f"💾 Heatmap saved: {phenotype}_ml_results_availability.png")
    
    plt.show()

def print_detailed_summary(availability_df, phenotype, tissues_by_method, available_folds):
    """Print detailed summary statistics for ML results"""
    
    if availability_df is None or availability_df.empty:
        return
    
    print(f"\n📊 ML RESULTS AVAILABILITY SUMMARY for {phenotype.upper()}")
    print("=" * 70)
    
    total_combinations = availability_df.size
    available_combinations = availability_df.sum().sum()
    
    print(f"Folds checked: {available_folds}")
    print(f"Total tissue-method combinations: {total_combinations}")
    print(f"Complete combinations: {available_combinations}")
    print(f"Missing combinations: {total_combinations - available_combinations}")
    print(f"Completion rate: {available_combinations/total_combinations*100:.1f}%")
    
    print(f"\n🧪 ML Results by Tissue:")
    for tissue in availability_df.index:
        tissue_complete = availability_df.loc[tissue].sum()
        tissue_total = len(availability_df.columns)
        print(f"  {tissue}: {tissue_complete}/{tissue_total} methods have results")
    
    print(f"\n🔬 ML Results by Method:")
    for method in availability_df.columns:
        method_complete = availability_df[method].sum()
        method_total = len(availability_df)
        tissues_found = len(tissues_by_method.get(method, []))
        print(f"  {method}: {method_complete}/{method_total} tissues complete ({tissues_found} tissues processed)")
    
    print(f"\n🔬 Method-specific tissues with ML results:")
    for method, tissues in tissues_by_method.items():
        if tissues:
            print(f"  {method}: {', '.join(tissues)}")
        else:
            print(f"  {method}: No ML results found")
    
    # Show missing combinations
    print(f"\n❌ Missing ML Results (Method-Tissue combinations):")
    missing_found = False
    for tissue in availability_df.index:
        for method in availability_df.columns:
            if availability_df.loc[tissue, method] == 0:
                if not missing_found:
                    missing_found = True
                print(f"  {method} - {tissue}")
    
    if not missing_found:
        print("  None! All combinations have ML results. 🎉")

def check_specific_fold_results(phenotype, fold, tissues_by_method):
    """Check detailed results for a specific fold"""
    
    fold_path = f"{phenotype}/Fold_{fold}/"
    results_dir = f"{fold_path}EnhancedMLResults/"
    
    if not os.path.exists(results_dir):
        print(f"❌ No EnhancedMLResults directory for Fold_{fold}")
        return
    
    print(f"\n🔍 Detailed check for Fold_{fold}:")
    
    for method, tissues in tissues_by_method.items():
        if not tissues:
            continue
            
        method_dir = os.path.join(results_dir, method)
        if not os.path.exists(method_dir):
            print(f"  ❌ {method}: Method directory missing")
            continue
        
        # Check feature directories
        feature_dirs = [d for d in os.listdir(method_dir) 
                       if os.path.isdir(os.path.join(method_dir, d)) and d.endswith('_features')]
        
        if not feature_dirs:
            print(f"  ❌ {method}: No feature directories found")
            continue
        
        print(f"  ✅ {method}: {len(feature_dirs)} feature directories, {len(tissues)} tissues")
        
        # Sample check: look at first feature directory
        sample_feature_dir = os.path.join(method_dir, feature_dirs[0])
        result_files = glob(os.path.join(sample_feature_dir, "*.csv"))
        model_files = glob(os.path.join(sample_feature_dir, "*.pkl"))
        
        print(f"    📁 {feature_dirs[0]}: {len(result_files)} CSV files, {len(model_files)} model files")

def main():
    """Main function"""
    
    if len(sys.argv) < 2:
        print("Usage: python check_ml_results.py <phenotype> [max_folds]")
        print("Example: python check_ml_results.py migraine 5")
        sys.exit(1)
    
    phenotype = sys.argv[1]
    max_folds = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    print(f"🔍 Checking ML model results availability for: {phenotype}")
    print(f"📁 Maximum folds to check: {max_folds}")
    print(f"🎯 Looking for: EnhancedMLResults directories with feature_importance, performance, and model files")
    print("=" * 70)
    
    # Check availability
    availability_df, tissues_by_method = check_ml_results_availability(phenotype, max_folds)
    
    if availability_df is not None and not availability_df.empty:
        available_folds = []
        for fold in range(max_folds):
            if os.path.exists(f"{phenotype}/Fold_{fold}/"):
                available_folds.append(fold)
        
        # Print summary
        print_detailed_summary(availability_df, phenotype, tissues_by_method, available_folds)
        
        # Create heatmap
        create_availability_heatmap(availability_df, phenotype)
        
        # Save results to CSV
        csv_file = f"{phenotype}_ml_results_availability.csv"
        availability_df.to_csv(csv_file)
        print(f"💾 Results saved: {csv_file}")
        
        # Check one fold in detail
        if available_folds:
            check_specific_fold_results(phenotype, available_folds[0], tissues_by_method)
        
    else:
        print("❌ No ML results found or failed to check availability")

if __name__ == "__main__":
    main()