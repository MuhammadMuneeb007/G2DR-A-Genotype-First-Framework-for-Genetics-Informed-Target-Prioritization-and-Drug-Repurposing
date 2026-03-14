import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
import sys
import os
from datetime import datetime
warnings.filterwarnings('ignore')

def load_and_prepare_data(filepath):
    """Load the AUC matrix and prepare for analysis"""
    df = pd.read_csv(filepath, index_col=0)
    return df

def calculate_pairwise_correlation(df, col1, col2):
    """Calculate correlation between two databases using only common non-zero tissues"""
    data1 = df[col1]
    data2 = df[col2]
    
    # Find tissues where both databases have non-zero values
    mask = (data1 != 0) & (data2 != 0)
    common_tissues = df.index[mask]
    
    if len(common_tissues) < 2:
        return np.nan, common_tissues
    
    # Calculate correlation using only common tissues
    corr = data1[mask].corr(data2[mask])
    return corr, common_tissues

def calculate_correlations_proper(df):
    """Calculate correlation matrix properly handling missing data"""
    databases = df.columns
    n_db = len(databases)
    corr_matrix = pd.DataFrame(index=databases, columns=databases, dtype=float)
    
    # Fill diagonal with 1.0
    for i, db in enumerate(databases):
        corr_matrix.iloc[i, i] = 1.0
    
    # Calculate pairwise correlations
    for i in range(n_db):
        for j in range(i+1, n_db):
            db1, db2 = databases[i], databases[j]
            corr, common_tissues = calculate_pairwise_correlation(df, db1, db2)
            corr_matrix.iloc[i, j] = corr
            corr_matrix.iloc[j, i] = corr
    
    return corr_matrix

def get_anatomical_tissue_groups():
    """Define anatomical/physiological tissue groups for clustering"""
    tissue_groups = {
        'Central_Nervous_System': [
            'Brain_Anterior_cingulate_cortex_BA24', 'Brain_Caudate_basal_ganglia',
            'Brain_Cerebellar_Hemisphere', 'Brain_Cerebellum', 'Brain_Cortex',
            'Brain_Frontal_Cortex_BA9', 'Brain_Hippocampus', 'Brain_Hypothalamus',
            'Brain_Nucleus.accumbens_basal_ganglia', 'Brain_Putamen_basal_ganglia',
            'Brain_Spinal_cord_cervical_c-1', 'Brain_Substantia_nigra'
        ],
        'Digestive_System': [
            'Colon_Sigmoid', 'Colon_Transverse', 'Esophagus_Gastroesophageal_Junction',
            'Esophagus_Mucosa', 'Esophagus_Muscularis', 'Liver', 'Pancreas',
            'Small_Intestine_Terminal_Ileum', 'Stomach'
        ],
        'Cardiovascular_System': [
            'Artery_Aorta', 'Artery_Coronary', 'Artery_Tibial', 'Heart_Atrial_Appendage',
            'Heart_Left_Ventricle'
        ],
        'Respiratory_System': [
            'Lung'
        ],
        'Reproductive_System': [
            'Breast_Mammary_Tissue', 'Cervix_Ectocervix', 'Cervix_Endocervix',
            'Fallopian_Tube', 'Ovary', 'Prostate', 'Testis', 'Uterus', 'Vagina'
        ],
        'Musculoskeletal_System': [
            'Muscle_Skeletal'
        ],
        'Endocrine_System': [
            'Adrenal_Gland', 'Pituitary', 'Thyroid'
        ],
        'Immune_System': [
            'Spleen'
        ],
        'Urinary_System': [
            'Kidney_Cortex', 'Bladder'
        ],
        'Skin_Connective': [
            'Skin_Not_Sun_Exposed_Suprapubic', 'Skin_Sun_Exposed_Lower_leg',
            'Adipose_Subcutaneous', 'Adipose_Visceral_Omentum'
        ],
        'Secretory_Glands': [
            'Minor_Salivary_Gland'
        ]
    }
    return tissue_groups

def find_optimal_clusters(data, max_clusters=10, method='kmeans'):
    """Find optimal number of clusters using silhouette analysis"""
    if data.shape[0] < 4:
        return 2, []
    
    silhouette_scores = []
    cluster_range = range(2, min(max_clusters + 1, data.shape[0]))
    
    for n_clusters in cluster_range:
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif method == 'hierarchical':
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        
        cluster_labels = clusterer.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
    return optimal_clusters, silhouette_scores

def write_header(f, text, level=1):
    """Write formatted header to file"""
    if level == 1:
        f.write("\n" + "="*100 + "\n")
        f.write(text.center(100))
        f.write("\n" + "="*100 + "\n\n")
    elif level == 2:
        f.write("\n" + "-"*100 + "\n")
        f.write(text)
        f.write("\n" + "-"*100 + "\n\n")
    elif level == 3:
        f.write("\n" + text + "\n")
        f.write("-" * len(text) + "\n\n")

def analyze_and_report(phenotype):
    """Main analysis function that generates comprehensive text report"""
    
    # Set up file paths
    base_dir = f"/data/ascher02/uqmmune1/ANNOVAR/{phenotype}/Database"
    filepath = f"{base_dir}/{phenotype}_CrossDatabase_TestAUC_Matrix.csv"
    output_file = f"{base_dir}/{phenotype}_Performance_Analysis_Report.txt"
    
    # Check if input file exists
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from: {filepath}")
    df = load_and_prepare_data(filepath)
    
    # Handle missing values
    df_cleaned = df.dropna(how='all')
    df_filled = df_cleaned.fillna(0)
    df_filled = df_filled.replace([np.inf, -np.inf], 0)
    
    # Open output file
    with open(output_file, 'w') as f:
        
        # Write header
        f.write("="*100 + "\n")
        f.write(f"COMPREHENSIVE PERFORMANCE ANALYSIS REPORT".center(100) + "\n")
        f.write(f"Phenotype: {phenotype}".center(100) + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(100) + "\n")
        f.write("="*100 + "\n\n")
        
        # Data summary
        write_header(f, "DATA SUMMARY", level=1)
        f.write(f"Total Tissues: {df_filled.shape[0]}\n")
        f.write(f"Total Methods/Databases: {df_filled.shape[1]}\n")
        f.write(f"Data Range: {df_filled.min().min():.4f} to {df_filled.max().max():.4f}\n\n")
        
        f.write("Methods/Databases analyzed:\n")
        for i, method in enumerate(df_filled.columns, 1):
            f.write(f"  {i}. {method}\n")
        
        f.write(f"\nTissues analyzed:\n")
        for i, tissue in enumerate(df_filled.index, 1):
            f.write(f"  {i}. {tissue}\n")
        
        # ========================================
        # SECTION 1: BEST PERFORMING TISSUES
        # ========================================
        write_header(f, "SECTION 1: TISSUE PERFORMANCE ANALYSIS", level=1)
        
        # Calculate average performance per tissue
        tissue_avg_performance = []
        tissue_max_performance = []
        tissue_min_performance = []
        tissue_std_performance = []
        tissue_data_count = []
        
        for tissue in df_filled.index:
            tissue_values = df_filled.loc[tissue]
            non_zero_values = tissue_values[tissue_values != 0]
            
            if len(non_zero_values) > 0:
                tissue_avg_performance.append(non_zero_values.mean())
                tissue_max_performance.append(non_zero_values.max())
                tissue_min_performance.append(non_zero_values.min())
                tissue_std_performance.append(non_zero_values.std())
                tissue_data_count.append(len(non_zero_values))
            else:
                tissue_avg_performance.append(0.0)
                tissue_max_performance.append(0.0)
                tissue_min_performance.append(0.0)
                tissue_std_performance.append(0.0)
                tissue_data_count.append(0)
        
        # Create tissue performance dataframe
        tissue_perf_df = pd.DataFrame({
            'Tissue': df_filled.index,
            'Average_AUC': tissue_avg_performance,
            'Max_AUC': tissue_max_performance,
            'Min_AUC': tissue_min_performance,
            'Std_AUC': tissue_std_performance,
            'Methods_Count': tissue_data_count
        })
        
        # Sort by average performance
        tissue_perf_df = tissue_perf_df.sort_values('Average_AUC', ascending=False)
        
        write_header(f, "1.1 BEST PERFORMING TISSUES (Ranked by Average AUC)", level=2)
        f.write(f"{'Rank':<6} {'Tissue':<50} {'Avg AUC':<10} {'Max AUC':<10} {'Min AUC':<10} {'Std Dev':<10} {'Methods':<8}\n")
        f.write("-"*100 + "\n")
        
        for i, row in tissue_perf_df.iterrows():
            rank = list(tissue_perf_df.index).index(i) + 1
            f.write(f"{rank:<6} {row['Tissue']:<50} {row['Average_AUC']:<10.4f} {row['Max_AUC']:<10.4f} "
                   f"{row['Min_AUC']:<10.4f} {row['Std_AUC']:<10.4f} {int(row['Methods_Count']):<8}\n")
        
        # Highlight top performers
        write_header(f, "1.2 TOP 10 PERFORMING TISSUES", level=3)
        top_10_tissues = tissue_perf_df.head(10)
        for idx, row in top_10_tissues.iterrows():
            rank = list(tissue_perf_df.index).index(idx) + 1
            f.write(f"\n{rank}. {row['Tissue']}\n")
            f.write(f"   Average AUC: {row['Average_AUC']:.4f}\n")
            f.write(f"   Best Method AUC: {row['Max_AUC']:.4f}\n")
            f.write(f"   Consistency (Std Dev): {row['Std_AUC']:.4f}\n")
            f.write(f"   Methods with data: {int(row['Methods_Count'])}\n")
        
        # Highlight bottom performers
        write_header(f, "1.3 WORST 10 PERFORMING TISSUES", level=3)
        bottom_10_tissues = tissue_perf_df.tail(10)
        for idx, row in bottom_10_tissues.iterrows():
            rank = list(tissue_perf_df.index).index(idx) + 1
            f.write(f"\n{rank}. {row['Tissue']}\n")
            f.write(f"   Average AUC: {row['Average_AUC']:.4f}\n")
            f.write(f"   Best Method AUC: {row['Max_AUC']:.4f}\n")
            f.write(f"   Methods with data: {int(row['Methods_Count'])}\n")
        
        # Important tissues to explore
        write_header(f, "1.4 IMPORTANT TISSUES TO EXPLORE", level=3)
        f.write("Based on multiple criteria:\n\n")
        
        # High performers with consistency
        high_perf_consistent = tissue_perf_df[
            (tissue_perf_df['Average_AUC'] >= 0.7) & 
            (tissue_perf_df['Std_AUC'] <= 0.1)
        ]
        
        f.write("A. HIGH PERFORMING AND CONSISTENT TISSUES (Avg AUC >= 0.7, Std Dev <= 0.1):\n")
        if len(high_perf_consistent) > 0:
            for idx, row in high_perf_consistent.iterrows():
                f.write(f"   - {row['Tissue']}: Avg AUC = {row['Average_AUC']:.4f}, "
                       f"Std Dev = {row['Std_AUC']:.4f}\n")
        else:
            f.write("   None found with these criteria.\n")
        
        # High performers but variable
        high_perf_variable = tissue_perf_df[
            (tissue_perf_df['Average_AUC'] >= 0.7) & 
            (tissue_perf_df['Std_AUC'] > 0.1)
        ]
        
        f.write("\nB. HIGH PERFORMING BUT VARIABLE TISSUES (Avg AUC >= 0.7, Std Dev > 0.1):\n")
        f.write("   These tissues may benefit from method-specific optimization.\n")
        if len(high_perf_variable) > 0:
            for idx, row in high_perf_variable.iterrows():
                f.write(f"   - {row['Tissue']}: Avg AUC = {row['Average_AUC']:.4f}, "
                       f"Std Dev = {row['Std_AUC']:.4f}\n")
        else:
            f.write("   None found with these criteria.\n")
        
        # Tissues with potential (good max but lower average)
        tissue_perf_df['Potential'] = tissue_perf_df['Max_AUC'] - tissue_perf_df['Average_AUC']
        high_potential = tissue_perf_df[
            (tissue_perf_df['Potential'] >= 0.15) & 
            (tissue_perf_df['Max_AUC'] >= 0.7)
        ].sort_values('Potential', ascending=False)
        
        f.write("\nC. TISSUES WITH HIGH POTENTIAL (Max AUC >= 0.7, but Average much lower):\n")
        f.write("   These tissues perform well with specific methods and warrant further investigation.\n")
        if len(high_potential) > 0:
            for idx, row in high_potential.iterrows():
                f.write(f"   - {row['Tissue']}: Max AUC = {row['Max_AUC']:.4f}, "
                       f"Avg AUC = {row['Average_AUC']:.4f}, Gap = {row['Potential']:.4f}\n")
        else:
            f.write("   None found with these criteria.\n")
        
        # ========================================
        # SECTION 2: METHOD/DATABASE PERFORMANCE
        # ========================================
        write_header(f, "SECTION 2: METHOD/DATABASE PERFORMANCE ANALYSIS", level=1)
        
        # Calculate average performance per method
        method_avg_performance = []
        method_max_performance = []
        method_min_performance = []
        method_std_performance = []
        method_data_count = []
        
        for method in df_filled.columns:
            method_values = df_filled[method]
            non_zero_values = method_values[method_values != 0]
            
            if len(non_zero_values) > 0:
                method_avg_performance.append(non_zero_values.mean())
                method_max_performance.append(non_zero_values.max())
                method_min_performance.append(non_zero_values.min())
                method_std_performance.append(non_zero_values.std())
                method_data_count.append(len(non_zero_values))
            else:
                method_avg_performance.append(0.0)
                method_max_performance.append(0.0)
                method_min_performance.append(0.0)
                method_std_performance.append(0.0)
                method_data_count.append(0)
        
        # Create method performance dataframe
        method_perf_df = pd.DataFrame({
            'Method': df_filled.columns,
            'Average_AUC': method_avg_performance,
            'Max_AUC': method_max_performance,
            'Min_AUC': method_min_performance,
            'Std_AUC': method_std_performance,
            'Tissues_Count': method_data_count
        })
        
        # Sort by average performance
        method_perf_df = method_perf_df.sort_values('Average_AUC', ascending=False)
        
        write_header(f, "2.1 METHOD/DATABASE RANKINGS (Ranked by Average AUC)", level=2)
        f.write(f"{'Rank':<6} {'Method/Database':<30} {'Avg AUC':<10} {'Max AUC':<10} {'Min AUC':<10} {'Std Dev':<10} {'Tissues':<8}\n")
        f.write("-"*100 + "\n")
        
        for i, row in method_perf_df.iterrows():
            rank = list(method_perf_df.index).index(i) + 1
            f.write(f"{rank:<6} {row['Method']:<30} {row['Average_AUC']:<10.4f} {row['Max_AUC']:<10.4f} "
                   f"{row['Min_AUC']:<10.4f} {row['Std_AUC']:<10.4f} {int(row['Tissues_Count']):<8}\n")
        
        # Top methods
        write_header(f, "2.2 TOP PERFORMING METHODS", level=3)
        top_methods = method_perf_df.head(min(10, len(method_perf_df)))
        for idx, row in top_methods.iterrows():
            rank = list(method_perf_df.index).index(idx) + 1
            f.write(f"\n{rank}. {row['Method']}\n")
            f.write(f"   Average AUC: {row['Average_AUC']:.4f}\n")
            f.write(f"   Best Tissue AUC: {row['Max_AUC']:.4f}\n")
            f.write(f"   Consistency (Std Dev): {row['Std_AUC']:.4f}\n")
            f.write(f"   Tissues with data: {int(row['Tissues_Count'])}\n")
        
        # ========================================
        # SECTION 3: METHOD-TISSUE COMBINATIONS
        # ========================================
        write_header(f, "SECTION 3: BEST METHOD-TISSUE COMBINATIONS", level=1)
        
        # Create list of all method-tissue combinations with their AUC
        combinations = []
        for tissue in df_filled.index:
            for method in df_filled.columns:
                auc = df_filled.loc[tissue, method]
                if auc > 0:  # Only include non-zero values
                    combinations.append({
                        'Tissue': tissue,
                        'Method': method,
                        'AUC': auc
                    })
        
        combinations_df = pd.DataFrame(combinations)
        combinations_df = combinations_df.sort_values('AUC', ascending=False)
        
        write_header(f, "3.1 TOP 50 METHOD-TISSUE COMBINATIONS", level=2)
        f.write(f"{'Rank':<6} {'Tissue':<50} {'Method':<30} {'AUC':<10}\n")
        f.write("-"*100 + "\n")
        
        for i, row in combinations_df.head(50).iterrows():
            rank = list(combinations_df.index).index(i) + 1
            f.write(f"{rank:<6} {row['Tissue']:<50} {row['Method']:<30} {row['AUC']:<10.4f}\n")
        
        # Best combination per tissue
        write_header(f, "3.2 BEST METHOD FOR EACH TISSUE", level=2)
        best_method_per_tissue = df_filled.idxmax(axis=1)
        best_auc_per_tissue = df_filled.max(axis=1)
        
        # Create dataframe and sort by AUC
        best_tissue_method = pd.DataFrame({
            'Tissue': best_method_per_tissue.index,
            'Best_Method': best_method_per_tissue.values,
            'AUC': best_auc_per_tissue.values
        }).sort_values('AUC', ascending=False)
        
        for _, row in best_tissue_method.iterrows():
            f.write(f"{row['Tissue']:<50} -> {row['Best_Method']:<30} (AUC: {row['AUC']:.4f})\n")
        
        # Best tissue per method
        write_header(f, "3.3 BEST TISSUE FOR EACH METHOD", level=2)
        best_tissue_per_method = df_filled.idxmax(axis=0)
        best_auc_per_method = df_filled.max(axis=0)
        
        # Create dataframe and sort by AUC
        best_method_tissue = pd.DataFrame({
            'Method': best_tissue_per_method.index,
            'Best_Tissue': best_tissue_per_method.values,
            'AUC': best_auc_per_method.values
        }).sort_values('AUC', ascending=False)
        
        for _, row in best_method_tissue.iterrows():
            f.write(f"{row['Method']:<30} -> {row['Best_Tissue']:<50} (AUC: {row['AUC']:.4f})\n")
        
        # ========================================
        # SECTION 4: METHOD CORRELATIONS
        # ========================================
        write_header(f, "SECTION 4: METHOD PERFORMANCE CORRELATIONS", level=1)
        
        # Calculate method correlations
        method_corr = calculate_correlations_proper(df_filled)
        
        write_header(f, "4.1 COMPLETE METHOD CORRELATION MATRIX", level=2)
        # Save correlation matrix to CSV
        corr_csv_path = f"{base_dir}/{phenotype}_method_correlation_matrix.csv"
        method_corr.to_csv(corr_csv_path)
        f.write(f"Full correlation matrix saved to: {corr_csv_path}\n\n")
        
        # Print correlation matrix
        f.write("Correlation Matrix:\n")
        f.write(method_corr.to_string())
        f.write("\n\n")
        
        # Find highly correlated method pairs
        write_header(f, "4.2 HIGHLY CORRELATED METHODS (r >= 0.7)", level=3)
        
        high_corr_pairs = []
        for i in range(len(method_corr.columns)):
            for j in range(i+1, len(method_corr.columns)):
                corr_val = method_corr.iloc[i, j]
                if not np.isnan(corr_val) and corr_val >= 0.7:
                    high_corr_pairs.append({
                        'Method1': method_corr.columns[i],
                        'Method2': method_corr.columns[j],
                        'Correlation': corr_val
                    })
        
        if high_corr_pairs:
            high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('Correlation', ascending=False)
            f.write(f"Found {len(high_corr_df)} highly correlated method pairs:\n\n")
            for _, row in high_corr_df.iterrows():
                f.write(f"{row['Method1']:<30} <-> {row['Method2']:<30} (r = {row['Correlation']:.4f})\n")
        else:
            f.write("No method pairs with correlation >= 0.7 found.\n")
        
        # Moderately correlated methods
        write_header(f, "4.3 MODERATELY CORRELATED METHODS (0.5 <= r < 0.7)", level=3)
        
        mod_corr_pairs = []
        for i in range(len(method_corr.columns)):
            for j in range(i+1, len(method_corr.columns)):
                corr_val = method_corr.iloc[i, j]
                if not np.isnan(corr_val) and 0.5 <= corr_val < 0.7:
                    mod_corr_pairs.append({
                        'Method1': method_corr.columns[i],
                        'Method2': method_corr.columns[j],
                        'Correlation': corr_val
                    })
        
        if mod_corr_pairs:
            mod_corr_df = pd.DataFrame(mod_corr_pairs).sort_values('Correlation', ascending=False)
            f.write(f"Found {len(mod_corr_df)} moderately correlated method pairs:\n\n")
            for _, row in mod_corr_df.iterrows():
                f.write(f"{row['Method1']:<30} <-> {row['Method2']:<30} (r = {row['Correlation']:.4f})\n")
        else:
            f.write("No method pairs with correlation between 0.5 and 0.7 found.\n")
        
        # Low/negative correlations
        write_header(f, "4.4 WEAKLY OR NEGATIVELY CORRELATED METHODS (r < 0.3)", level=3)
        
        low_corr_pairs = []
        for i in range(len(method_corr.columns)):
            for j in range(i+1, len(method_corr.columns)):
                corr_val = method_corr.iloc[i, j]
                if not np.isnan(corr_val) and corr_val < 0.3:
                    low_corr_pairs.append({
                        'Method1': method_corr.columns[i],
                        'Method2': method_corr.columns[j],
                        'Correlation': corr_val
                    })
        
        if low_corr_pairs:
            low_corr_df = pd.DataFrame(low_corr_pairs).sort_values('Correlation')
            f.write(f"Found {len(low_corr_df)} weakly correlated method pairs:\n\n")
            for _, row in low_corr_df.iterrows():
                f.write(f"{row['Method1']:<30} <-> {row['Method2']:<30} (r = {row['Correlation']:.4f})\n")
        else:
            f.write("No method pairs with correlation < 0.3 found.\n")
        
        # ========================================
        # SECTION 5: TISSUE CORRELATIONS
        # ========================================
        write_header(f, "SECTION 5: TISSUE PERFORMANCE CORRELATIONS", level=1)
        
        # Calculate tissue correlations
        tissue_corr = df_filled.T.corr()
        
        write_header(f, "5.1 COMPLETE TISSUE CORRELATION MATRIX", level=2)
        # Save correlation matrix to CSV
        tissue_corr_csv_path = f"{base_dir}/{phenotype}_tissue_correlation_matrix.csv"
        tissue_corr.to_csv(tissue_corr_csv_path)
        f.write(f"Full correlation matrix saved to: {tissue_corr_csv_path}\n\n")
        
        f.write("Note: Due to large size, printing top correlations only. See CSV for full matrix.\n\n")
        
        # Find highly correlated tissue pairs
        write_header(f, "5.2 HIGHLY CORRELATED TISSUES (r >= 0.8)", level=3)
        
        high_tissue_corr = []
        for i in range(len(tissue_corr.columns)):
            for j in range(i+1, len(tissue_corr.columns)):
                corr_val = tissue_corr.iloc[i, j]
                if not np.isnan(corr_val) and corr_val >= 0.8:
                    high_tissue_corr.append({
                        'Tissue1': tissue_corr.columns[i],
                        'Tissue2': tissue_corr.columns[j],
                        'Correlation': corr_val
                    })
        
        if high_tissue_corr:
            high_tissue_corr_df = pd.DataFrame(high_tissue_corr).sort_values('Correlation', ascending=False)
            f.write(f"Found {len(high_tissue_corr_df)} highly correlated tissue pairs:\n\n")
            for _, row in high_tissue_corr_df.iterrows():
                f.write(f"{row['Tissue1']:<50} <-> {row['Tissue2']:<50} (r = {row['Correlation']:.4f})\n")
        else:
            f.write("No tissue pairs with correlation >= 0.8 found.\n")
        
        # Moderately correlated tissues
        write_header(f, "5.3 MODERATELY CORRELATED TISSUES (0.6 <= r < 0.8)", level=3)
        
        mod_tissue_corr = []
        for i in range(len(tissue_corr.columns)):
            for j in range(i+1, len(tissue_corr.columns)):
                corr_val = tissue_corr.iloc[i, j]
                if not np.isnan(corr_val) and 0.6 <= corr_val < 0.8:
                    mod_tissue_corr.append({
                        'Tissue1': tissue_corr.columns[i],
                        'Tissue2': tissue_corr.columns[j],
                        'Correlation': corr_val
                    })
        
        if mod_tissue_corr:
            mod_tissue_corr_df = pd.DataFrame(mod_tissue_corr).sort_values('Correlation', ascending=False)
            f.write(f"Found {len(mod_tissue_corr_df)} moderately correlated tissue pairs (showing top 20):\n\n")
            for _, row in mod_tissue_corr_df.head(20).iterrows():
                f.write(f"{row['Tissue1']:<50} <-> {row['Tissue2']:<50} (r = {row['Correlation']:.4f})\n")
            if len(mod_tissue_corr_df) > 20:
                f.write(f"\n... and {len(mod_tissue_corr_df) - 20} more. See CSV for complete data.\n")
        else:
            f.write("No tissue pairs with correlation between 0.6 and 0.8 found.\n")
        
        # Low correlations
        write_header(f, "5.4 WEAKLY OR NEGATIVELY CORRELATED TISSUES (r < 0.2)", level=3)
        
        low_tissue_corr = []
        for i in range(len(tissue_corr.columns)):
            for j in range(i+1, len(tissue_corr.columns)):
                corr_val = tissue_corr.iloc[i, j]
                if not np.isnan(corr_val) and corr_val < 0.2:
                    low_tissue_corr.append({
                        'Tissue1': tissue_corr.columns[i],
                        'Tissue2': tissue_corr.columns[j],
                        'Correlation': corr_val
                    })
        
        if low_tissue_corr:
            low_tissue_corr_df = pd.DataFrame(low_tissue_corr).sort_values('Correlation')
            f.write(f"Found {len(low_tissue_corr_df)} weakly correlated tissue pairs (showing top 20):\n\n")
            for _, row in low_tissue_corr_df.head(20).iterrows():
                f.write(f"{row['Tissue1']:<50} <-> {row['Tissue2']:<50} (r = {row['Correlation']:.4f})\n")
            if len(low_tissue_corr_df) > 20:
                f.write(f"\n... and {len(low_tissue_corr_df) - 20} more. See CSV for complete data.\n")
        else:
            f.write("No tissue pairs with correlation < 0.2 found.\n")
        
        # ========================================
        # SECTION 6: CLUSTERING ANALYSIS
        # ========================================
        write_header(f, "SECTION 6: CLUSTERING ANALYSIS", level=1)
        
        # Method clustering
        if df_filled.shape[1] >= 4:
            write_header(f, "6.1 METHOD CLUSTERING", level=2)
            
            scaler = StandardScaler()
            method_data_scaled = scaler.fit_transform(df_filled.T.values)
            
            optimal_k_methods, _ = find_optimal_clusters(method_data_scaled, method='kmeans')
            
            f.write(f"Optimal number of method clusters: {optimal_k_methods}\n\n")
            
            kmeans_methods = KMeans(n_clusters=optimal_k_methods, random_state=42, n_init=10)
            method_clusters = kmeans_methods.fit_predict(method_data_scaled)
            
            f.write("Method Cluster Assignments:\n\n")
            for cluster_id in range(optimal_k_methods):
                cluster_methods = [df_filled.columns[i] for i in range(len(df_filled.columns)) 
                                 if method_clusters[i] == cluster_id]
                avg_perf = np.mean([method_avg_performance[list(df_filled.columns).index(m)] 
                                   for m in cluster_methods])
                
                f.write(f"Cluster {cluster_id + 1} ({len(cluster_methods)} methods, Avg AUC: {avg_perf:.4f}):\n")
                for method in cluster_methods:
                    method_idx = list(df_filled.columns).index(method)
                    f.write(f"  - {method} (Avg AUC: {method_avg_performance[method_idx]:.4f})\n")
                f.write("\n")
        
        # Tissue clustering
        if df_filled.shape[0] >= 4:
            write_header(f, "6.2 TISSUE CLUSTERING", level=2)
            
            scaler = StandardScaler()
            tissue_data_scaled = scaler.fit_transform(df_filled.values)
            
            optimal_k_tissues, _ = find_optimal_clusters(tissue_data_scaled, method='kmeans')
            
            f.write(f"Optimal number of tissue clusters: {optimal_k_tissues}\n\n")
            
            kmeans_tissues = KMeans(n_clusters=optimal_k_tissues, random_state=42, n_init=10)
            tissue_clusters = kmeans_tissues.fit_predict(tissue_data_scaled)
            
            f.write("Tissue Cluster Assignments:\n\n")
            for cluster_id in range(optimal_k_tissues):
                cluster_tissues = [df_filled.index[i] for i in range(len(df_filled.index)) 
                                 if tissue_clusters[i] == cluster_id]
                avg_perf = np.mean([tissue_avg_performance[list(df_filled.index).index(t)] 
                                   for t in cluster_tissues])
                
                f.write(f"Cluster {cluster_id + 1} ({len(cluster_tissues)} tissues, Avg AUC: {avg_perf:.4f}):\n")
                for tissue in cluster_tissues[:20]:  # Show first 20
                    tissue_idx = list(df_filled.index).index(tissue)
                    f.write(f"  - {tissue} (Avg AUC: {tissue_avg_performance[tissue_idx]:.4f})\n")
                if len(cluster_tissues) > 20:
                    f.write(f"  ... and {len(cluster_tissues) - 20} more tissues\n")
                f.write("\n")
        
        # ========================================
        # SECTION 7: ANATOMICAL ANALYSIS
        # ========================================
        write_header(f, "SECTION 7: ANATOMICAL TISSUE GROUP ANALYSIS", level=1)
        
        tissue_groups = get_anatomical_tissue_groups()
        
        # Create mapping
        tissue_to_group = {}
        for group, tissues in tissue_groups.items():
            for tissue in tissues:
                tissue_to_group[tissue] = group
        
        # Analyze performance by anatomical group
        group_performance = {}
        for group in tissue_groups.keys():
            group_tissues = [t for t in tissue_groups[group] if t in df_filled.index]
            if group_tissues:
                group_aucs = [tissue_avg_performance[list(df_filled.index).index(t)] 
                             for t in group_tissues]
                group_performance[group] = {
                    'avg_auc': np.mean(group_aucs),
                    'max_auc': np.max(group_aucs),
                    'min_auc': np.min(group_aucs),
                    'std_auc': np.std(group_aucs),
                    'count': len(group_tissues),
                    'tissues': group_tissues
                }
        
        # Sort by average performance
        sorted_groups = sorted(group_performance.items(), 
                             key=lambda x: x[1]['avg_auc'], reverse=True)
        
        write_header(f, "7.1 PERFORMANCE BY ANATOMICAL SYSTEM", level=2)
        
        f.write(f"{'Rank':<6} {'Anatomical System':<35} {'Avg AUC':<10} {'Max AUC':<10} "
               f"{'Min AUC':<10} {'Tissues':<8}\n")
        f.write("-"*100 + "\n")
        
        for rank, (group, stats) in enumerate(sorted_groups, 1):
            f.write(f"{rank:<6} {group:<35} {stats['avg_auc']:<10.4f} {stats['max_auc']:<10.4f} "
                   f"{stats['min_auc']:<10.4f} {stats['count']:<8}\n")
        
        write_header(f, "7.2 DETAILED BREAKDOWN BY ANATOMICAL SYSTEM", level=2)
        
        for group, stats in sorted_groups:
            f.write(f"\n{group}:\n")
            f.write(f"  Average AUC: {stats['avg_auc']:.4f}\n")
            f.write(f"  Best performing tissue: ")
            best_tissue_idx = np.argmax([tissue_avg_performance[list(df_filled.index).index(t)] 
                                        for t in stats['tissues']])
            best_tissue = stats['tissues'][best_tissue_idx]
            f.write(f"{best_tissue} ({stats['max_auc']:.4f})\n")
            f.write(f"  Tissues in this group ({stats['count']}):\n")
            for tissue in stats['tissues']:
                tissue_idx = list(df_filled.index).index(tissue)
                f.write(f"    - {tissue}: {tissue_avg_performance[tissue_idx]:.4f}\n")
            f.write("\n")
        
        # ========================================
        # SECTION 8: KEY INSIGHTS AND RECOMMENDATIONS
        # ========================================
        write_header(f, "SECTION 8: KEY INSIGHTS AND RECOMMENDATIONS", level=1)
        
        write_header(f, "8.1 OVERALL SUMMARY", level=3)
        
        # Best overall tissue
        best_tissue_overall = tissue_perf_df.iloc[0]
        f.write(f"Best Performing Tissue:\n")
        f.write(f"  {best_tissue_overall['Tissue']}\n")
        f.write(f"  Average AUC: {best_tissue_overall['Average_AUC']:.4f}\n")
        f.write(f"  Max AUC: {best_tissue_overall['Max_AUC']:.4f}\n\n")
        
        # Best overall method
        best_method_overall = method_perf_df.iloc[0]
        f.write(f"Best Performing Method:\n")
        f.write(f"  {best_method_overall['Method']}\n")
        f.write(f"  Average AUC: {best_method_overall['Average_AUC']:.4f}\n")
        f.write(f"  Max AUC: {best_method_overall['Max_AUC']:.4f}\n\n")
        
        # Best combination
        best_combo = combinations_df.iloc[0]
        f.write(f"Best Method-Tissue Combination:\n")
        f.write(f"  Tissue: {best_combo['Tissue']}\n")
        f.write(f"  Method: {best_combo['Method']}\n")
        f.write(f"  AUC: {best_combo['AUC']:.4f}\n\n")
        
        # Overall statistics
        f.write(f"Overall Dataset Statistics:\n")
        f.write(f"  Mean AUC across all combinations: {df_filled[df_filled > 0].mean().mean():.4f}\n")
        f.write(f"  Median AUC: {df_filled[df_filled > 0].median().median():.4f}\n")
        f.write(f"  Standard Deviation: {df_filled[df_filled > 0].std().mean():.4f}\n")
        f.write(f"  Number of high-performing combinations (AUC >= 0.7): "
               f"{len(combinations_df[combinations_df['AUC'] >= 0.7])}\n")
        f.write(f"  Percentage of high performers: "
               f"{100 * len(combinations_df[combinations_df['AUC'] >= 0.7]) / len(combinations_df):.2f}%\n\n")
        
        write_header(f, "8.2 RECOMMENDATIONS FOR FURTHER INVESTIGATION", level=3)
        
        f.write("1. PRIORITY TISSUES TO EXPLORE:\n")
        priority_tissues = tissue_perf_df.head(5)
        for _, row in priority_tissues.iterrows():
            f.write(f"   - {row['Tissue']} (Avg AUC: {row['Average_AUC']:.4f})\n")
        
        f.write("\n2. PRIORITY METHODS TO USE:\n")
        priority_methods = method_perf_df.head(5)
        for _, row in priority_methods.iterrows():
            f.write(f"   - {row['Method']} (Avg AUC: {row['Average_AUC']:.4f})\n")
        
        f.write("\n3. RECOMMENDED METHOD-TISSUE COMBINATIONS:\n")
        top_combos = combinations_df.head(10)
        for i, row in top_combos.iterrows():
            rank = list(combinations_df.index).index(i) + 1
            f.write(f"   {rank}. {row['Tissue']} + {row['Method']} (AUC: {row['AUC']:.4f})\n")
        
        if len(high_corr_pairs) > 0:
            f.write("\n4. METHODS WITH SIMILAR PERFORMANCE (Consider using one):\n")
            for _, row in high_corr_df.head(5).iterrows():
                f.write(f"   - {row['Method1']} and {row['Method2']} (r = {row['Correlation']:.4f})\n")
        
        f.write("\n5. ANATOMICAL SYSTEMS TO PRIORITIZE:\n")
        for rank, (group, stats) in enumerate(sorted_groups[:5], 1):
            f.write(f"   {rank}. {group} (Avg AUC: {stats['avg_auc']:.4f}, {stats['count']} tissues)\n")
        
        # ========================================
        # FOOTER
        # ========================================
        f.write("\n" + "="*100 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*100 + "\n")
    
    print(f"\nComprehensive analysis report saved to: {output_file}")
    
    # Also save the detailed dataframes as CSV
    tissue_perf_df.to_csv(f"{base_dir}/{phenotype}_tissue_performance_detailed.csv", index=False)
    method_perf_df.to_csv(f"{base_dir}/{phenotype}_method_performance_detailed.csv", index=False)
    combinations_df.to_csv(f"{base_dir}/{phenotype}_all_combinations_ranked.csv", index=False)
    
    print(f"Additional CSV files saved:")
    print(f"  - {phenotype}_tissue_performance_detailed.csv")
    print(f"  - {phenotype}_method_performance_detailed.csv")
    print(f"  - {phenotype}_all_combinations_ranked.csv")
    print(f"  - {phenotype}_method_correlation_matrix.csv")
    print(f"  - {phenotype}_tissue_correlation_matrix.csv")

def main():
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python script.py <phenotype>")
        print("Example: python script.py migraine")
        sys.exit(1)
    
    phenotype = sys.argv[1]
    
    print(f"Starting comprehensive performance analysis for phenotype: {phenotype}")
    print("="*100)
    
    analyze_and_report(phenotype)
    
    print("\n" + "="*100)
    print("Analysis complete!")

if __name__ == "__main__":
    main()