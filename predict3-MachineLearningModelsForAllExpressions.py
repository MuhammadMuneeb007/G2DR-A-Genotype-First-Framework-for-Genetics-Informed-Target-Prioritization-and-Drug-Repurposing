#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Machine Learning Pipeline for All Gene Expression Methods

Usage: python enhanced_ml_pipeline.py <phenotype> <fold>
Example: python enhanced_ml_pipeline.py migraine 0

Features:
- Multiple ML models (XGBoost, Random Forest, Logistic Regression)
- Grid search hyperparameter tuning using validation set
- Multiple feature selection options (100, 500, 1000, 2000 features)
- Feature importance saving for EACH MODEL and EACH FEATURE COUNT
- Comprehensive evaluation across all expression methods
- Covariates (Sex, age, PCs) included as additional features
- Preprocessing (imputation/scaling) fit on TRAIN only, applied to val/test
- Uses RAW expression (not _fixed files)
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, classification_report, confusion_matrix, roc_curve
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available. Install with: pip install xgboost")

# Check command line arguments
if len(sys.argv) != 3:
    print("Usage: python enhanced_ml_pipeline.py <phenotype> <fold>")
    print("Example: python enhanced_ml_pipeline.py migraine 0")
    sys.exit(1)

PHENOTYPE = sys.argv[1]
FOLD = sys.argv[2]
BASE_PATH = "{}/Fold_{}/".format(PHENOTYPE, FOLD)

# Enhanced Configuration
N_FEATURES_TO_SELECT = [100, 500, 1000, 2000]  # Multiple feature selection options
RANDOM_STATE = 42

# Covariates to include as features
COVARIATE_COLUMNS = ['Sex', 'age_at_recruitment', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10']

# Covariate file mapping
COVARIATE_FILES = {
    'training': 'train_data.COV_PCA',
    'validation': 'validation_data.COV_PCA',
    'test': 'test_data.COV_PCA'
}

# All expression methods to test
EXPRESSION_METHODS = {
    "Regular": {
        "train_dir": "TrainExpression",
        "val_dir": "ValidationExpression", 
        "test_dir": "TestExpression",
        "file_pattern": "predicted_expression_{dataset}_{tissue}_merged.csv"
    },
    "JTI": {
        "train_dir": "JTITrainExpression",
        "val_dir": "JTIValidationExpression",
        "test_dir": "JTITestExpression", 
        "file_pattern": "predicted_expression_JTI_{dataset}_{tissue}_merged.csv"
    },

    "UTMOST": {
        "train_dir": "UTMOSTTrainExpression",
        "val_dir": "UTMOSTValidationExpression",
        "test_dir": "UTMOSTTestExpression",
        "file_pattern": "predicted_expression_UTMOST_{dataset}_{tissue}_merged.csv"
    }
    ,
    "UTMOST2": {
        "train_dir": "utmost2TrainExpression", 
        "val_dir": "utmost2ValidationExpression",
        "test_dir": "utmost2TestExpression",
        "file_pattern": "predicted_expression_UTMOST_{dataset}_{tissue}_merged.csv"
    },
    "EpiX": {
        "train_dir": "EpiXTrainExpression",
        "val_dir": "EpiXValidationExpression", 
        "test_dir": "EpiXTestExpression",
        "file_pattern": "predicted_expression_EpiX_{dataset}_{tissue}_merged.csv"
    },
    "TIGAR": {
        "train_dir": "TigarTrainExpression",
        "val_dir": "TigarValidExpression",
        "test_dir": "TigarTestExpression", 
        "file_pattern": "{dataset}_expression_expression_matrix.csv"
    },
    "FUSION": {
        "train_dir": "FussionExpression",
        "val_dir": "FussionExpression",
        "test_dir": "FussionExpression",
        "file_pattern": "GeneExpression_{dataset}_data.csv"
    }
}


def load_covariate_data():
    """
    Load covariate data for train, validation, and test sets.
    
    Returns:
        dict: Dictionary with 'training', 'validation', 'test' keys,
              each containing {'matrix': np.array, 'sample_ids': np.array, 'columns': list}
    """
    print("📋 Loading covariate data...")
    
    covariate_data = {}
    
    for dataset_name, cov_file in COVARIATE_FILES.items():
        cov_path = os.path.join(BASE_PATH, cov_file)
        
        if not os.path.exists(cov_path):
            print("  ⚠️  Covariate file not found: {}".format(cov_path))
            covariate_data[dataset_name] = None
            continue
        
        try:
            # Try tab-separated first
            df = pd.read_csv(cov_path, sep='\t')
            
            # Check if we got expected columns
            if ('IID' not in df.columns) or ('FID' not in df.columns):
                df = pd.read_csv(cov_path, sep=r'\s+')
            
            # Verify required columns exist
            required = {'FID', 'IID'}
            if not required.issubset(df.columns):
                missing = required - set(df.columns)
                print("  ❌ {} missing columns: {}".format(dataset_name, missing))
                covariate_data[dataset_name] = None
                continue
            
            # Create sample ID
            df['sample_id'] = df['FID'].astype(str) + '_' + df['IID'].astype(str)
            
            # Check which covariates are available
            available_covs = [c for c in COVARIATE_COLUMNS if c in df.columns]
            missing_covs = [c for c in COVARIATE_COLUMNS if c not in df.columns]
            
            if missing_covs:
                print("  ⚠️  {} missing covariates: {}".format(dataset_name, missing_covs))
            
            if not available_covs:
                print("  ❌ {} no valid covariates found".format(dataset_name))
                covariate_data[dataset_name] = None
                continue
            
            # Extract covariate matrix
            cov_matrix = df[available_covs].values.astype(np.float64)
            sample_ids = df['sample_id'].values
            
            # Encode Sex if present (PLINK style 1/2 -> 0/1)
            if 'Sex' in available_covs:
                sex_idx = available_covs.index('Sex')
                sex_col = cov_matrix[:, sex_idx]
                unique_vals = set(np.unique(sex_col[~np.isnan(sex_col)]))
                if unique_vals <= {1, 2}:
                    cov_matrix[:, sex_idx] = np.where(sex_col == 1, 0, np.where(sex_col == 2, 1, sex_col))
            
            covariate_data[dataset_name] = {
                'matrix': cov_matrix,
                'sample_ids': sample_ids,
                'columns': available_covs
            }
            
            print("  ✅ {}: {} samples, {} covariates".format(
                dataset_name, len(sample_ids), len(available_covs)))
            
        except Exception as e:
            print("  ❌ Failed to load {}: {}".format(cov_path, e))
            covariate_data[dataset_name] = None
 
    return covariate_data


def combine_expression_and_covariates(X_expr, covariate_data, dataset_name, expr_feature_names):
    """
    Combine expression data with covariate data.
    
    Args:
        X_expr: Expression matrix (n_samples x n_genes)
        covariate_data: Dictionary with covariate data
        dataset_name: 'training', 'validation', or 'test'
        expr_feature_names: List of gene names
    
    Returns:
        tuple: (combined_matrix, combined_feature_names, feature_types)
    """
    if covariate_data is None or covariate_data.get(dataset_name) is None:
        # No covariates available, return expression only
        feature_types = ['gene'] * X_expr.shape[1]
        return X_expr, expr_feature_names, feature_types
    
    cov_data = covariate_data[dataset_name]
    cov_matrix = cov_data['matrix']
    cov_columns = cov_data['columns']
    
    # Check sample counts match
    if X_expr.shape[0] != cov_matrix.shape[0]:
        print("      ⚠️  Sample count mismatch: expression={}, covariates={}".format(
            X_expr.shape[0], cov_matrix.shape[0]))
        # Truncate to minimum
        min_samples = min(X_expr.shape[0], cov_matrix.shape[0])
        X_expr = X_expr[:min_samples]
        cov_matrix = cov_matrix[:min_samples]
    
    # Combine matrices: [expression | covariates]
    X_combined = np.hstack([X_expr, cov_matrix])
    
    # Combine feature names
    combined_names = list(expr_feature_names) + ['COV_' + c for c in cov_columns]
    
    # Track feature types
    feature_types = ['gene'] * len(expr_feature_names) + ['covariate'] * len(cov_columns)
    
    return X_combined, combined_names, feature_types


def get_ml_models():
    """
    Define machine learning models and their hyperparameter grids
    Uses exact same hyperparameters as the original script with proper class weight handling
    
    Returns:
        dict: Dictionary of models and parameter grids
    """
    models = {}
    
    # XGBoost (if available) - with class weight handling
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = {
            'model': xgb.XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric='logloss',
                verbosity=0
            ),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'scale_pos_weight': [1.0, 'balanced']  # XGBoost class weight equivalent
            }
        }
    
    # Random Forest - with class weight in params
    models['RandomForest'] = {
        'model': RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2'],
            'class_weight': ['balanced', None]  # Class weight in hyperparameter grid
        }
    }
    
    # Logistic Regression - class weight already in params
    models['LogisticRegression'] = {
        'model': LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=1000,
            solver='liblinear'
        ),
        'params': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'class_weight': ['balanced', None]
        }
    }
    
    return models

def get_feature_importance(model, model_name):
    """
    Extract feature importance from trained model
    
    Args:
        model: Trained model
        model_name: Name of the model
    
    Returns:
        numpy.array: Feature importance scores
    """
    try:
        if hasattr(model, 'feature_importances_'):
            # Tree-based models (RandomForest, XGBoost)
            return model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Linear models (LogisticRegression)
            return np.abs(model.coef_[0])  # Use absolute values of coefficients
        else:
            print("    ⚠️  No feature importance available for {}".format(model_name))
            return None
    except Exception as e:
        print("    ⚠️  Failed to extract feature importance for {}: {}".format(model_name, e))
        return None

def train_model_with_grid_search(model_name, model_config, X_train, y_train, X_val, y_val, class_weight_dict):
    """
    Train a model with grid search using validation set with proper class weight handling
    
    Args:
        model_name: Name of the model
        model_config: Model configuration dict
        X_train, y_train: Training data
        X_val, y_val: Validation data for hyperparameter tuning (REQUIRED)
        class_weight_dict: Class weight dictionary for imbalanced data
    
    Returns:
        dict: Trained model and results with feature importance
    """
    print("      🔍 Training {} with grid search using validation data...".format(model_name))
    
    # Validation data is REQUIRED for proper grid search
    if X_val is None or y_val is None or len(X_val) == 0:
        print("      ❌ No validation data available for {} - grid search requires validation data".format(model_name))
        return None
    
    # Manual grid search using validation set
    best_score = -1
    best_params = {}
    best_model = None
    
    # Get parameter combinations
    param_grid = model_config['params']
    param_combinations = list(ParameterGrid(param_grid))
    print("      📊 Testing {} parameter combinations with validation AUC".format(len(param_combinations)))
    
    try:
        for i, params in enumerate(param_combinations):
            if (i + 1) % 20 == 0 or i == 0:
                print("      🔄 Testing combination {}/{}: {}".format(i+1, len(param_combinations), params))
            
            # Create model copy and set parameters
            temp_model = deepcopy(model_config['model'])
            
            # Handle class weights properly for each model type
            if model_name == 'XGBoost':
                # XGBoost uses scale_pos_weight instead of class_weight
                if 'scale_pos_weight' in params and params['scale_pos_weight'] == 'balanced':
                    # Calculate scale_pos_weight for XGBoost
                    neg_count = sum(y_train == 0)
                    pos_count = sum(y_train == 1)
                    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
                    params = params.copy()
                    params['scale_pos_weight'] = scale_pos_weight
                elif 'scale_pos_weight' in params and params['scale_pos_weight'] == 1.0:
                    # Use default scale_pos_weight
                    params = params.copy()
                    params['scale_pos_weight'] = 1.0
            
            # Set all parameters
            temp_model.set_params(**params)
            
            # Train on training set
            temp_model.fit(X_train, y_train)
            
            # Evaluate on validation set - THIS IS THE KEY GRID SEARCH STEP
            y_val_pred_proba = temp_model.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, y_val_pred_proba)
            
            # Keep best model based on validation AUC
            if val_auc > best_score:
                best_score = val_auc
                best_params = params.copy()
                best_model = deepcopy(temp_model)
        
        print("      ✅ {} best validation AUC: {:.4f} with params: {}".format(
            model_name, best_score, best_params))
        
        # Get feature importance from best model
        feature_importance = get_feature_importance(best_model, model_name)
        
        return {
            'model': best_model,
            'best_params': best_params,
            'best_val_score': best_score,
            'feature_importance': feature_importance
        }
        
    except Exception as e:
        print("      ❌ Grid search failed for {}: {}".format(model_name, e))
        return None

def evaluate_model(model, X, y, dataset_name, model_name):
    """
    Evaluate a trained model on a dataset
    
    Args:
        model: Trained model
        X, y: Data to evaluate on
        dataset_name: Name of dataset (train/val/test)
        model_name: Name of model
    
    Returns:
        dict: Evaluation metrics
    """
    if X is None or y is None:
        return None
    
    try:
        # Predictions
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        # Calculate metrics
        auc = roc_auc_score(y, y_pred_proba)
        precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary')
        
        results = {
            'dataset': dataset_name,
            'model': model_name,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'y_true': y.tolist(),
            'y_pred': y_pred.tolist(),
            'y_pred_proba': y_pred_proba.tolist()
        }
        
        return results
        
    except Exception as e:
        print("      ❌ Evaluation failed for {} on {}: {}".format(model_name, dataset_name, e))
        return None

def load_phenotype_data():
    """Load phenotype data from .fam files"""
    print("📋 Loading phenotype data from .fam files...")
    
    fam_files = {
        'training': "{}train_data.QC.clumped.pruned.fam".format(BASE_PATH),
        'validation': "{}validation_data.clumped.pruned.fam".format(BASE_PATH), 
        'test': "{}test_data.clumped.pruned.fam".format(BASE_PATH)
    }
    
    phenotype_data = {}
    
    for dataset_name, fam_file in fam_files.items():
        if os.path.exists(fam_file):
            try:
                # Read .fam file: FID IID PAT MAT SEX PHENO
                fam_df = pd.read_csv(fam_file, sep=r'\s+', header=None, 
                                   names=['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENO'])
                
                # Convert phenotype to 0/1 format
                if set(fam_df['PHENO'].unique()) == {1, 2}:
                    pheno_binary = fam_df['PHENO'] - 1  # 1,2 -> 0,1
                else:
                    pheno_binary = fam_df['PHENO']
                
                phenotype_data[dataset_name] = pheno_binary.values
                class_counts = dict(pd.Series(pheno_binary).value_counts())
                print("  ✅ {}: {} samples, classes: {}".format(dataset_name, len(pheno_binary), class_counts))
                
            except Exception as e:
                print("  ❌ Failed to load {}: {}".format(fam_file, e))
        else:
            print("  ⚠️  FAM file not found: {}".format(fam_file))
    
    return phenotype_data

def load_expression_data_for_method(method, tissue):
    """Load expression data for a specific method and tissue (RAW, not _fixed)"""
    
    method_config = EXPRESSION_METHODS[method]
    datasets = {}
    
    dataset_mapping = {
        'training': ('train_dir', 'train'),
        'validation': ('val_dir', 'validation'), 
        'test': ('test_dir', 'test')
    }
    
    for dataset_name, (dir_key, dataset_key) in dataset_mapping.items():
        
        # Build directory path
        expr_dir = "{}{}/{}/".format(BASE_PATH, method_config[dir_key], tissue)
        
        # Build filename based on method-specific patterns
        if method == "FUSION":
            filename = "GeneExpression_{}_data.csv".format(dataset_key)
        elif method == "TIGAR":
            filename = "{}_expression_expression_matrix.csv".format(dataset_key)
        else:
            # For other methods, use the standard pattern
            filename = method_config['file_pattern'].format(dataset=dataset_name, tissue=tissue)
        
        expr_file = os.path.join(expr_dir, filename)
        
        # Try to load the file (prefer non-fixed files)
        if os.path.exists(expr_file):
            try:
                df = pd.read_csv(expr_file, index_col=0)
                datasets[dataset_name] = (df.values, df.columns.tolist())
            except Exception as e:
                print("    ❌ Error reading {}: {}".format(expr_file, e))
                datasets[dataset_name] = None
        else:
            # Try alternative file patterns if the standard one fails
            if os.path.exists(expr_dir):
                # Look for CSV files, EXCLUDING _fixed files
                csv_files = [f for f in os.listdir(expr_dir) 
                            if f.endswith('.csv') and '_fixed' not in f]
                
                if csv_files:
                    try:
                        alt_file = os.path.join(expr_dir, csv_files[0])
                        df = pd.read_csv(alt_file, index_col=0)
                        datasets[dataset_name] = (df.values, df.columns.tolist())
                    except Exception as e:
                        print("    ❌ Error reading alternative file: {}".format(e))
                        datasets[dataset_name] = None
                else:
                    datasets[dataset_name] = None
            else:
                datasets[dataset_name] = None
    
    # Extract data and columns
    result_data = {}
    result_columns = {}
    
    for dataset_name in ['training', 'validation', 'test']:
        if datasets.get(dataset_name) is not None:
            if isinstance(datasets[dataset_name], tuple):
                result_data[dataset_name] = datasets[dataset_name][0]
                result_columns[dataset_name] = datasets[dataset_name][1]
            else:
                result_data[dataset_name] = datasets[dataset_name]
                result_columns[dataset_name] = None
        else:
            result_data[dataset_name] = None
            result_columns[dataset_name] = None
    
    return result_data.get('training'), result_data.get('validation'), result_data.get('test'), result_columns

def get_available_tissues():
    """Find all available tissues across all methods"""
    all_tissues = set()
    
    for method, config in EXPRESSION_METHODS.items():
        search_dir = "{}{}/".format(BASE_PATH, config['train_dir'])
        
        if os.path.exists(search_dir):
            tissues = [item for item in os.listdir(search_dir) 
                      if os.path.isdir(os.path.join(search_dir, item))]
            all_tissues.update(tissues)
    
    return sorted(list(all_tissues))

def save_model_results(result, method, tissue, n_features, output_dir):
    """
    Save detailed results for a single model-tissue-method-feature combination
    
    CRITICAL: This saves feature importance for EACH MODEL and EACH FEATURE COUNT
    """
    
    # Create method-specific subdirectory
    method_dir = os.path.join(output_dir, method)
    os.makedirs(method_dir, exist_ok=True)
    
    # Create feature-specific subdirectory  
    feature_dir = os.path.join(method_dir, "{}_features".format(n_features))
    os.makedirs(feature_dir, exist_ok=True)
    
    model_name = result['Model']
    
    # Save feature information with importance - FOR EACH MODEL AND EACH FEATURE COUNT
    if result.get('Selected_Features') and result.get('Feature_Importance'):
        
        # Verify we have feature importance data
        if len(result['Feature_Importance']) > 0:
            feature_data = {
                'Feature_Name': result['Selected_Features'],
                'Selection_Score': result.get('Feature_Scores', [0] * len(result['Selected_Features'])),
                'Feature_Importance': result['Feature_Importance'],
                'Feature_Type': result.get('Feature_Types', ['gene'] * len(result['Selected_Features']))
            }
            
            feature_df = pd.DataFrame(feature_data)
            feature_df = feature_df.sort_values('Feature_Importance', ascending=False)
            
            # Save with unique filename for each model and feature count
            feature_file = os.path.join(feature_dir, "{}_{}_feature_importance.csv".format(tissue, model_name))
            feature_df.to_csv(feature_file, index=False)
            
            # Count feature types
            n_covariates = sum(1 for t in result.get('Feature_Types', []) if t == 'covariate')
            n_genes = len(result['Selected_Features']) - n_covariates
            
            print("      💾 Saved feature importance: {} ({} genes + {} covariates, top: {})".format(
                feature_file, n_genes, n_covariates, feature_df.iloc[0]['Feature_Name']
            ))
        else:
            print("      ⚠️  No feature importance data for {}-{}-{}".format(method, tissue, model_name))
    
    # Save performance summary
    performance_data = {
        'Metric': ['Train_AUC', 'Val_AUC', 'Test_AUC', 'Features_Used', 'Covariates_Used', 
                   'Train_Samples', 'Val_Samples', 'Test_Samples', 'Best_Params'],
        'Value': [
            result.get('Train_AUC', 'N/A'),
            result.get('Val_AUC', 'N/A'), 
            result.get('Test_AUC', 'N/A'),
            result.get('Features_Used', 'N/A'),
            result.get('Covariates_Used', 'N/A'),
            result.get('Train_Samples', 'N/A'),
            result.get('Val_Samples', 'N/A'), 
            result.get('Test_Samples', 'N/A'),
            str(result.get('Best_Params', {}))
        ]
    }
    
    performance_df = pd.DataFrame(performance_data)
    performance_file = os.path.join(feature_dir, "{}_{}_performance.csv".format(tissue, model_name))
    performance_df.to_csv(performance_file, index=False)
    
    # Save trained model
    if result.get('Model_Object'):
        model_file = os.path.join(feature_dir, "{}_{}_model.pkl".format(tissue, model_name))
        with open(model_file, 'wb') as f:
            pickle.dump(result['Model_Object'], f)

def train_and_evaluate_enhanced(X_train, y_train, X_val, y_val, X_test, y_test, method, tissue, n_features, columns_info=None, feature_names=None, covariate_data=None):
    """Enhanced training and evaluation with multiple models, grid search, and covariates
    
    IMPORTANT: 
    - Uses RAW expression + covariates as features
    - Preprocessing (imputation/scaling) fit on TRAIN only, applied to val/test
    """
    
    if X_train is None or len(X_train) == 0:
        return []
    
    # Validation data is REQUIRED for proper grid search
    if X_val is None or y_val is None or len(X_val) == 0:
        print("    ⚠️  No validation data available for {}-{}, skipping grid search approach".format(method, tissue))
        return []
    
    print("    ✅ Validation data available: {} samples for grid search".format(len(X_val)))
    
    try:
        # Handle feature mismatch between datasets (common for FUSION)
        if columns_info and method in ["FUSION"]:
            train_cols = columns_info.get('training')
            val_cols = columns_info.get('validation') 
            test_cols = columns_info.get('test')
            
            # Find common features across all available datasets
            available_columns = [cols for cols in [train_cols, val_cols, test_cols] if cols is not None]
            if len(available_columns) > 1:
                common_features = set(available_columns[0])
                for cols in available_columns[1:]:
                    common_features = common_features.intersection(set(cols))
                common_features = list(common_features)
                
                if len(common_features) < 50:
                    print("    ⚠️  Only {} common features found, skipping".format(len(common_features)))
                    return []
                
                # Subset data to common features
                if train_cols:
                    train_indices = [train_cols.index(f) for f in common_features if f in train_cols]
                    X_train = X_train[:, train_indices]
                    feature_names = common_features
                
                if val_cols and X_val is not None:
                    val_indices = [val_cols.index(f) for f in common_features if f in val_cols]
                    X_val = X_val[:, val_indices]
                
                if test_cols and X_test is not None:
                    test_indices = [test_cols.index(f) for f in common_features if f in test_cols]
                    X_test = X_test[:, test_indices]
                
                print("    🔧 Using {} common features for {}".format(len(common_features), method))
        
        # ====================================================================
        # COMBINE EXPRESSION WITH COVARIATES
        # ====================================================================
        n_covariates = 0
        if covariate_data is not None:
            print("    📊 Adding covariates to feature matrix...")
            
            # Get feature names if not provided
            if feature_names is None:
                feature_names = ["Gene_{}".format(i) for i in range(X_train.shape[1])]
            
            # Combine for each split
            X_train, feature_names_combined, feature_types_train = combine_expression_and_covariates(
                X_train, covariate_data, 'training', feature_names
            )
            
            X_val, _, _ = combine_expression_and_covariates(
                X_val, covariate_data, 'validation', feature_names
            )
            
            if X_test is not None:
                X_test, _, _ = combine_expression_and_covariates(
                    X_test, covariate_data, 'test', feature_names
                )
            
            feature_names = feature_names_combined
            n_covariates = sum(1 for t in feature_types_train if t == 'covariate')
            
            print("    ✅ Combined features: {} genes + {} covariates = {} total".format(
                len(feature_names) - n_covariates, n_covariates, len(feature_names)))
        else:
            feature_types_train = ['gene'] * X_train.shape[1]
            if feature_names is None:
                feature_names = ["Gene_{}".format(i) for i in range(X_train.shape[1])]
        
        # ====================================================================
        # PREPROCESSING: Fit on TRAIN only, apply to val/test
        # ====================================================================
        print("    🔧 Preprocessing: imputation and scaling (fit on TRAIN only)...")
        
        # Imputer - fit on train
        imputer = SimpleImputer(strategy='median')
        X_train = imputer.fit_transform(X_train)
        X_val = imputer.transform(X_val)
        if X_test is not None:
            X_test = imputer.transform(X_test)
        
        # Scaler - fit on train
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        if X_test is not None:
            X_test = scaler.transform(X_test)
        
        # ====================================================================
        # FEATURE SELECTION
        # ====================================================================
        n_features_actual = min(n_features, X_train.shape[1])
        
        if n_features_actual < 10:
            return []
        
        print("    📊 Selecting {} features from {} total".format(n_features_actual, X_train.shape[1]))
        
        # Select top features using training data only
        selector = SelectKBest(score_func=f_classif, k=n_features_actual)
        X_train_selected = selector.fit_transform(X_train, y_train)
        
        # Get selected feature names, scores, and types
        selected_indices = selector.get_support(indices=True)
        selected_feature_names = [feature_names[i] for i in selected_indices]
        selected_feature_types = [feature_types_train[i] for i in selected_indices]
        feature_scores = selector.scores_[selected_indices]
        
        # Apply same selection to validation and test
        X_val_selected = selector.transform(X_val) if X_val is not None else None
        X_test_selected = selector.transform(X_test) if X_test is not None else None
        
        # Count selected feature types
        n_selected_covariates = sum(1 for t in selected_feature_types if t == 'covariate')
        n_selected_genes = len(selected_feature_names) - n_selected_covariates
        
        print("    ✅ Feature selection complete: {} genes + {} covariates selected".format(
            n_selected_genes, n_selected_covariates))
        
        # Calculate class weights for imbalanced data
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))
        print("    ⚖️  Class weights calculated: {}".format(class_weight_dict))
        
        # Get all models
        models = get_ml_models()
        
        # Train and evaluate each model with validation-based grid search
        all_results = []
        
        for model_name, model_config in models.items():
            print("    🤖 Training {} with {} features using validation grid search".format(model_name, n_features_actual))
            
            # Train model with grid search using validation data
            model_result = train_model_with_grid_search(
                model_name, model_config, 
                X_train_selected, y_train, 
                X_val_selected, y_val,
                class_weight_dict
            )
            
            if model_result is None:
                print("    ❌ Failed to train {}".format(model_name))
                continue
            
            # Evaluate on all datasets
            train_eval = evaluate_model(model_result['model'], X_train_selected, y_train, 'training', model_name)
            val_eval = evaluate_model(model_result['model'], X_val_selected, y_val, 'validation', model_name) if X_val_selected is not None else None
            test_eval = evaluate_model(model_result['model'], X_test_selected, y_test, 'test', model_name) if X_test_selected is not None else None
            
            # Compile results - INCLUDING FEATURE IMPORTANCE
            result = {
                'Method': method,
                'Tissue': tissue,
                'Model': model_name,
                'Features_Used': n_features_actual,
                'Covariates_Used': n_selected_covariates,
                'Train_Samples': len(X_train),
                'Val_Samples': len(X_val) if X_val is not None else 0,
                'Test_Samples': len(X_test) if X_test is not None else 0,
                'Selected_Features': selected_feature_names,
                'Feature_Types': selected_feature_types,
                'Feature_Scores': feature_scores.tolist(),
                'Feature_Importance': model_result['feature_importance'].tolist() if model_result['feature_importance'] is not None else [],
                'Best_Params': model_result['best_params'],
                'Best_Val_Score': model_result['best_val_score'],
                'Class_Weights_Used': class_weight_dict,
                'Model_Object': model_result['model'],
                'Train_AUC': train_eval['auc'] if train_eval else None,
                'Val_AUC': val_eval['auc'] if val_eval else None,
                'Test_AUC': test_eval['auc'] if test_eval else None
            }
            
            all_results.append(result)
            
            # Format output
            train_auc = "{:.3f}".format(result['Train_AUC']) if result['Train_AUC'] is not None else 'N/A'
            val_auc = "{:.3f}".format(result['Val_AUC']) if result['Val_AUC'] is not None else 'N/A'
            test_auc = "{:.3f}".format(result['Test_AUC']) if result['Test_AUC'] is not None else 'N/A'
            
            print("      ✅ {}: Train={}, Val={}, Test={} (FI: {} features, Best Val: {:.3f})".format(
                model_name, train_auc, val_auc, test_auc, len(result['Feature_Importance']), result['Best_Val_Score']
            ))
        
        return all_results
        
    except Exception as e:
        print("    ❌ Enhanced training failed for {}-{}: {}".format(method, tissue, e))
        import traceback
        traceback.print_exc()
        return []

def create_comprehensive_heatmaps(all_results, output_dir):
    """Create comprehensive heatmaps for all models and methods"""
    
    if not all_results:
        return
    
    results_df = pd.DataFrame(all_results)
    
    # Create separate heatmaps for each model
    models = results_df['Model'].unique()
    
    fig, axes = plt.subplots(len(models), 2, figsize=(20, 6 * len(models)))
    if len(models) == 1:
        axes = axes.reshape(1, -1)
    
    for i, model in enumerate(models):
        model_data = results_df[results_df['Model'] == model]
        
        # Validation AUC heatmap
        val_pivot = model_data.pivot_table(
            index='Method', 
            columns='Tissue', 
            values='Val_AUC', 
            aggfunc='first'
        )
        
        # Test AUC heatmap
        test_pivot = model_data.pivot_table(
            index='Method',
            columns='Tissue', 
            values='Test_AUC',
            aggfunc='first'
        )
        
        # Plot validation heatmap
        sns.heatmap(val_pivot, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                    cbar_kws={'label': 'Validation AUC'}, ax=axes[i, 0], 
                    vmin=0.4, vmax=1.0)
        axes[i, 0].set_title('{} - Validation AUC'.format(model))
        
        # Plot test heatmap
        sns.heatmap(test_pivot, annot=True, fmt='.2f', cmap='RdYlBu_r',
                    cbar_kws={'label': 'Test AUC'}, ax=axes[i, 1],
                    vmin=0.4, vmax=1.0)
        axes[i, 1].set_title('{} - Test AUC'.format(model))
    
    plt.tight_layout()
    
    # Save heatmaps
    heatmap_file = os.path.join(output_dir, 'comprehensive_performance_heatmaps.png')
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("🎨 Saved comprehensive performance heatmaps: {}".format(heatmap_file))

def check_existing_results(method, tissue, n_features, output_dir):
    """
    Check if all expected machine learning model results already exist
    
    Args:
        method: Expression method name
        tissue: Tissue name
        n_features: Number of features
        output_dir: Output directory path
    
    Returns:
        bool: True if all expected results exist, False otherwise
    """
    # Get expected models
    models = get_ml_models()
    model_names = list(models.keys())
    
    # Build paths for this combination
    method_dir = os.path.join(output_dir, method)
    feature_dir = os.path.join(method_dir, "{}_features".format(n_features))
    
    # Check if feature directory exists
    if not os.path.exists(feature_dir):
        return False
    
    # Check for all expected files for each model
    for model_name in model_names:
        # Expected files for each model
        expected_files = [
            "{}_{}_feature_importance.csv".format(tissue, model_name),
            "{}_{}_performance.csv".format(tissue, model_name),
            "{}_{}_model.pkl".format(tissue, model_name)
        ]
        
        # Check if all files exist
        for filename in expected_files:
            filepath = os.path.join(feature_dir, filename)
            if not os.path.exists(filepath):
                return False
    
    return True

def main():
    """Main function to run enhanced ML pipeline"""
    
    print("🧬 Enhanced ML Pipeline for All Expression Methods")
    print("📊 Phenotype: {}, Fold: {}".format(PHENOTYPE, FOLD))
    print("🎯 Using Multiple Models with Grid Search and Feature Importance")
    print("🔧 Feature selection options: {}".format(N_FEATURES_TO_SELECT))
    print("📋 Covariates: {} (included as features)".format(COVARIATE_COLUMNS))
    if XGBOOST_AVAILABLE:
        print("🚀 Models: XGBoost, Random Forest, Logistic Regression")
    else:
        print("🚀 Models: Random Forest, Logistic Regression")
    print("💾 Feature importance will be saved for EACH MODEL and EACH FEATURE COUNT")
    print("🔄 Will skip combinations where all results already exist")
    print("⚠️  Using RAW expression (not _fixed) + covariates as features")
    print("⚠️  Preprocessing (imputation/scaling) fit on TRAIN only")
    print("=" * 80)
    
    # Create output directory
    output_dir = "{}EnhancedMLResults/".format(BASE_PATH)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load phenotype data
    phenotype_data = load_phenotype_data()
    if not phenotype_data:
        print("❌ No phenotype data found!")
        sys.exit(1)
    
    # Load covariate data
    covariate_data = load_covariate_data()
    if covariate_data is None or all(v is None for v in covariate_data.values()):
        print("⚠️  No covariate data found, proceeding without covariates")
        covariate_data = None
    
    # Get available tissues
    tissues = get_available_tissues()
    if not tissues:
        print("❌ No tissues found!")
        sys.exit(1)
    
    tissue_preview = ', '.join(tissues[:3])
    if len(tissues) > 3:
        tissue_preview += '...'
    print("🧪 Found {} tissues: {}".format(len(tissues), tissue_preview))
    print("🔬 Testing {} methods: {}".format(len(EXPRESSION_METHODS), ', '.join(EXPRESSION_METHODS.keys())))
    print()
    
    # Store all results
    all_results = []
    
    # Calculate total combinations
    n_models = len(get_ml_models())
    total_models = len(EXPRESSION_METHODS) * len(tissues) * len(N_FEATURES_TO_SELECT) * n_models
    print("🎯 Total models to train: {} (Methods: {} × Tissues: {} × Features: {} × Models: {})".format(
        total_models, len(EXPRESSION_METHODS), len(tissues), len(N_FEATURES_TO_SELECT), n_models
    ))
    print()
    
    # Process each method, tissue, and feature count combination
    models_trained = 0
    models_skipped = 0
    
    for method in EXPRESSION_METHODS.keys():
        print("🔬 Processing method: {}".format(method))
        
        for tissue in tissues:
            print("  🧪 {} - ".format(tissue), end="")
            
            # Load expression data (RAW, not _fixed)
            X_train, X_val, X_test, columns_info = load_expression_data_for_method(method, tissue)
            
            if X_train is None:
                print("❌ No training data")
                continue
            
            # Get corresponding phenotype data
            y_train = phenotype_data.get('training')
            y_val = phenotype_data.get('validation') 
            y_test = phenotype_data.get('test')
            
            if y_train is None:
                print("❌ No training phenotypes")
                continue
            
            # Align data sizes
            min_train = min(len(X_train), len(y_train))
            X_train = X_train[:min_train]
            y_train = y_train[:min_train]
            
            if X_val is not None and y_val is not None:
                min_val = min(len(X_val), len(y_val))
                X_val = X_val[:min_val]
                y_val = y_val[:min_val]
            
            if X_test is not None and y_test is not None:
                min_test = min(len(X_test), len(y_test))
                X_test = X_test[:min_test]
                y_test = y_test[:min_test]
            
            # Get feature names
            feature_names = columns_info.get('training') if columns_info else None
            
            print("Testing {} feature counts".format(len(N_FEATURES_TO_SELECT)))
            
            # Test different feature counts - THIS IS WHERE FEATURE IMPORTANCE IS SAVED FOR EACH
            tissue_results = []
            for n_features in N_FEATURES_TO_SELECT:
                # Check if results already exist for this combination
                if check_existing_results(method, tissue, n_features, output_dir):
                    print("    ⏭️  {} features: SKIPPED (results exist)".format(n_features))
                    models_skipped += n_models
                    continue
                
                print("    📊 {} features:".format(n_features))
                
                # Train and evaluate with enhanced approach (including covariates)
                feature_results = train_and_evaluate_enhanced(
                    X_train, y_train, X_val, y_val, X_test, y_test, 
                    method, tissue, n_features, columns_info, feature_names,
                    covariate_data  # Pass covariate data
                )
                
                # Save individual model results - FEATURE IMPORTANCE SAVED HERE
                for result in feature_results:
                    save_model_results(result, method, tissue, n_features, output_dir)
                    tissue_results.append(result)
                    all_results.append(result)
                    models_trained += 1
                
                if feature_results:
                    print("    ✅ {} models trained for {} features".format(len(feature_results), n_features))
                else:
                    print("    ❌ No models trained for {} features".format(n_features))
            
            if tissue_results:
                print("  ✅ Completed tissue: {} ({} total models)".format(tissue, len(tissue_results)))
            else:
                print("  ⏭️  Skipped tissue: {} (all results exist)".format(tissue))
    
    # Save comprehensive results
    if all_results:
        # Create summary DataFrame
        summary_data = []
        for result in all_results:
            summary_data.append({
                'Method': result['Method'],
                'Tissue': result['Tissue'],
                'Model': result['Model'],
                'Features': result['Features_Used'],
                'Covariates': result.get('Covariates_Used', 0),
                'Train_Samples': result['Train_Samples'],
                'Val_Samples': result['Val_Samples'],
                'Test_Samples': result['Test_Samples'],
                'Train_AUC': round(result['Train_AUC'], 3) if result['Train_AUC'] is not None else None,
                'Val_AUC': round(result['Val_AUC'], 3) if result['Val_AUC'] is not None else None,
                'Test_AUC': round(result['Test_AUC'], 3) if result['Test_AUC'] is not None else None,
                'Best_Params': str(result['Best_Params']),
                'Feature_Importance_Length': len(result['Feature_Importance'])
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save main results file
        results_file = "{}enhanced_all_methods_all_models_results.csv".format(output_dir)
        summary_df.to_csv(results_file, index=False)
        
        # Create comprehensive heatmaps
        create_comprehensive_heatmaps(summary_data, output_dir)
        
        print("\n🎉 ENHANCED RESULTS SUMMARY")
        print("=" * 80)
        print("✅ Total models trained: {} / {} planned".format(models_trained, total_models))
        print("⏭️  Total models skipped: {} (results already existed)".format(models_skipped))
        print("✅ Methods tested: {}".format(len(summary_df['Method'].unique())))
        print("✅ Models tested: {}".format(len(summary_df['Model'].unique())))
        print("✅ Tissues tested: {}".format(len(summary_df['Tissue'].unique())))
        print("✅ Feature counts tested: {}".format(sorted(summary_df['Features'].unique())))
        print("✅ Covariates included: {}".format(COVARIATE_COLUMNS))
        print("💾 Results saved to: {}".format(results_file))
        
        # Verify feature importance was saved
        fi_saved = summary_df['Feature_Importance_Length'].sum()
        print("✅ Feature importance saved for {} models (total features tracked: {})".format(
            len(summary_df[summary_df['Feature_Importance_Length'] > 0]), fi_saved
        ))
        
        # Show best results per method-model combination
        print("\n🏆 Best Test AUC per Method-Model Combination:")
        for method in summary_df['Method'].unique():
            for model in summary_df['Model'].unique():
                method_model_data = summary_df[(summary_df['Method'] == method) & (summary_df['Model'] == model)]
                if len(method_model_data) > 0:
                    best_test_auc = method_model_data['Test_AUC'].max()
                    if not pd.isna(best_test_auc):
                        best_row = method_model_data[method_model_data['Test_AUC'] == best_test_auc].iloc[0]
                        print("  {:<12} {:<15}: {:.3f} (tissue: {}, features: {}, covs: {})".format(
                            method, model, best_test_auc, best_row['Tissue'], 
                            best_row['Features'], best_row['Covariates']
                        ))
        
        # Overall statistics
        valid_test_aucs = summary_df['Test_AUC'].dropna()
        if len(valid_test_aucs) > 0:
            print("\n📈 Overall Test AUC Statistics:")
            print("  Mean: {:.3f}".format(valid_test_aucs.mean()))
            print("  Std:  {:.3f}".format(valid_test_aucs.std()))
            print("  Max:  {:.3f}".format(valid_test_aucs.max()))
            print("  Min:  {:.3f}".format(valid_test_aucs.min()))
        
        print("\n📁 ENHANCED RESULTS STRUCTURE:")
        print("Main directory: {}".format(output_dir))
        print("├── enhanced_all_methods_all_models_results.csv (complete summary)")
        print("├── comprehensive_performance_heatmaps.png (visualization)")
        for method in summary_df['Method'].unique():
            print("├── {}/".format(method))
            method_data = summary_df[summary_df['Method'] == method]
            for n_features in sorted(method_data['Features'].unique()):
                print("│   ├── {}_features/".format(n_features))
                print("│   │   ├── *_*_feature_importance.csv ← GENE + COVARIATE IMPORTANCE")
                print("│   │   ├── *_*_performance.csv (detailed metrics)")
                print("│   │   └── *_*_model.pkl (trained models)")
        
        print("\n🔥 FEATURE IMPORTANCE CONFIRMATION:")
        print("✅ Feature importance saved for EVERY model and EVERY feature count")
        print("✅ Files named: {{tissue}}_{{model}}_feature_importance.csv")
        print("✅ Contains: Feature_Name, Selection_Score, Feature_Importance, Feature_Type")
        print("✅ Feature_Type: 'gene' for expression features, 'covariate' for Sex/age/PCs")
        print("✅ Each file shows which genes AND covariates are most predictive")
        
    else:
        print("❌ No new results generated (all may already exist)!")
        print("🔍 Check existing results in: {}".format(output_dir))
        
        # Check if we skipped everything
        if models_skipped > 0:
            print("⏭️  Skipped {} models because results already exist".format(models_skipped))
        
        # Don't exit with error if we just skipped everything
        if models_skipped == 0:
            sys.exit(1)

if __name__ == "__main__":
    main()