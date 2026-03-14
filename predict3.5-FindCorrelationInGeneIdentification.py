import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import os

def get_input_directory(filepath):
    """Get the directory containing the input file."""
    return os.path.dirname(filepath)

def read_and_process_data(filepath):
    """Read and process the cross-database comparison matrix."""
    df = pd.read_csv(filepath)
    
    # Debug: print column structure
    print(f"Total columns: {len(df.columns)}")
    print(f"Column names: {list(df.columns)}")
    
    # Get the metric row (second row, index 0 since first row becomes header)
    metric_row = df.iloc[0]  # This contains Gene_ID_Ratio, Test_AUC, etc.
    print(f"Metric row: {list(metric_row)}")
    
    # Remove the metric row from the dataframe
    df = df.iloc[1:].reset_index(drop=True)
    
    # Get unique databases by removing pandas suffixes (.1, .2, etc.)
    database_columns = df.columns[1:]  # Skip 'Gene Expression Model' column
    unique_databases = []
    
    for col in database_columns:
        # Remove pandas suffix (.1, .2, etc.) to get base database name
        base_name = col.split('.')[0]
        if base_name not in unique_databases:
            unique_databases.append(base_name)
    
    print(f"Detected databases: {unique_databases}")
    
    processed_data = {}
    
    for db in unique_databases:
        # Find all columns for this database (including .1, .2 variants)
        db_columns = []
        for i, col in enumerate(df.columns):
            if col == db or col.startswith(f"{db}."):
                db_columns.append(i)
        
        if len(db_columns) < 2:
            print(f"Skipping {db} - insufficient columns ({len(db_columns)} found)")
            continue
        
        # Find which column is Gene_ID_Ratio and which is Test_AUC
        gene_ratio_col_idx = None
        test_auc_col_idx = None
        
        for col_idx in db_columns:
            metric_type = metric_row.iloc[col_idx]
            if metric_type == 'Gene_ID_Ratio':
                gene_ratio_col_idx = col_idx
            elif metric_type == 'Test_AUC':
                test_auc_col_idx = col_idx
        
        if gene_ratio_col_idx is None or test_auc_col_idx is None:
            print(f"Skipping {db} - could not find both Gene_ID_Ratio and Test_AUC columns")
            print(f"  Available columns: {[df.columns[i] for i in db_columns]}")
            print(f"  Corresponding metrics: {[metric_row.iloc[i] for i in db_columns]}")
            continue
        
        gene_ratio_col = df.columns[gene_ratio_col_idx]
        test_auc_col = df.columns[test_auc_col_idx]
        
        print(f"Processing {db}: Gene ratio col = {gene_ratio_col}, Test AUC col = {test_auc_col}")
        
        # Extract data for this database
        gene_ratios = pd.to_numeric(df[gene_ratio_col], errors='coerce')
        test_aucs = pd.to_numeric(df[test_auc_col], errors='coerce')
        models = df['Gene Expression Model']
        
        # Remove rows with missing data
        mask = ~(gene_ratios.isna() | test_aucs.isna())
        
        if mask.sum() > 0:  # Only add if we have valid data
            valid_gene_ratios = gene_ratios[mask].values
            valid_test_aucs = test_aucs[mask].values
            valid_models = models[mask].values
            
            # Debug: print data characteristics
            print(f"  -> {mask.sum()} valid data points for {db}")
            print(f"  -> Gene ratios: min={np.min(valid_gene_ratios):.3f}, max={np.max(valid_gene_ratios):.3f}, var={np.var(valid_gene_ratios):.6f}")
            print(f"  -> Test AUCs: min={np.min(valid_test_aucs):.3f}, max={np.max(valid_test_aucs):.3f}, var={np.var(valid_test_aucs):.6f}")
            print(f"  -> Unique gene ratio values: {len(np.unique(valid_gene_ratios))}")
            
            processed_data[db] = {
                'gene_ratios': valid_gene_ratios,
                'test_aucs': valid_test_aucs,
                'models': valid_models
            }
        else:
            print(f"  -> No valid data points for {db}")
    
    return processed_data

def calculate_correlations(data):
    """Calculate Pearson and Spearman correlations for each database."""
    correlations = {}
    
    for db, db_data in data.items():
        print(f"\nAnalyzing correlations for {db}:")
        
        if len(db_data['gene_ratios']) <= 2:  # Need at least 3 points for correlation
            print(f"  Skipping {db} - insufficient data points ({len(db_data['gene_ratios'])})")
            continue
        
        # Check for constant values (no variance)
        gene_var = np.var(db_data['gene_ratios'])
        auc_var = np.var(db_data['test_aucs'])
        
        print(f"  Gene ratio variance: {gene_var:.6f}")
        print(f"  Test AUC variance: {auc_var:.6f}")
        
        if gene_var == 0:
            print(f"  Skipping {db} - no variance in gene identification ratios (all values are constant)")
            continue
            
        if auc_var == 0:
            print(f"  Skipping {db} - no variance in test AUC values (all values are constant)")
            continue
        
        try:
            pearson_r, pearson_p = pearsonr(db_data['gene_ratios'], db_data['test_aucs'])
            spearman_r, spearman_p = spearmanr(db_data['gene_ratios'], db_data['test_aucs'])
            
            print(f"  Pearson correlation: r={pearson_r:.4f}, p={pearson_p:.4f}")
            print(f"  Spearman correlation: r={spearman_r:.4f}, p={spearman_p:.4f}")
            
            # Check for NaN values
            if np.isnan(pearson_r) or np.isnan(pearson_p):
                print(f"  Skipping {db} - correlation calculation resulted in NaN")
                continue
            
            correlations[db] = {
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'n_samples': len(db_data['gene_ratios'])
            }
            print(f"  Successfully calculated correlations for {db}")
            
        except Exception as e:
            print(f"  Error calculating correlation for {db}: {e}")
            continue
    
    return correlations

def create_correlation_plots(data, correlations, dataset_name, output_dir):
    """Create professional publication-quality correlation plots."""
    if not data:
        print("No data available for plotting.")
        return None
    
    # Set publication style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Calculate grid dimensions
    n_databases = len(data)
    n_cols = 3
    n_rows = (n_databases + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    databases = list(data.keys())
    
    for i, db in enumerate(databases):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        db_data = data[db]
        
        # Scatter plot
        ax.scatter(db_data['gene_ratios'], db_data['test_aucs'], 
                  alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
        
        # Add trend line if correlation exists and is valid
        if db in correlations:
            try:
                # Check if we have enough variance for trend line
                if np.var(db_data['gene_ratios']) > 1e-10 and len(np.unique(db_data['gene_ratios'])) > 1:
                    z = np.polyfit(db_data['gene_ratios'], db_data['test_aucs'], 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(db_data['gene_ratios'].min(), 
                                        db_data['gene_ratios'].max(), 100)
                    ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
            except (np.linalg.LinAlgError, ValueError) as e:
                print(f"Could not fit trend line for {db}: {e}")
            
            # Add correlation info
            corr_info = correlations[db]
            ax.text(0.05, 0.95, 
                   f"r = {corr_info['pearson_r']:.3f}\n"
                   f"p = {corr_info['pearson_p']:.3f}\n"
                   f"n = {corr_info['n_samples']}", 
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            # Add note for databases without valid correlations
            n_points = len(db_data['gene_ratios'])
            variance_gene = np.var(db_data['gene_ratios'])
            variance_auc = np.var(db_data['test_aucs'])
            
            ax.text(0.05, 0.95, 
                   f"No correlation\n"
                   f"n = {n_points}\n"
                   f"Gene var = {variance_gene:.3e}\n"
                   f"AUC var = {variance_auc:.3e}", 
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        ax.set_xlabel('Gene Identification Ratio', fontsize=12, fontweight='bold')
        ax.set_ylabel('Test AUC', fontsize=12, fontweight='bold')
        ax.set_title(f'{db}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)
    
    # Hide empty subplots
    for i in range(len(databases), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.suptitle(f'Gene Identification Ratio vs Test AUC Correlations - {dataset_name}', 
                fontsize=16, fontweight='bold', y=1.02)
    
    # Save plot
    output_path = os.path.join(output_dir, f'{dataset_name}_correlation_plots.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    return output_path

def create_summary_plot(correlations, dataset_name, output_dir):
    """Create a summary plot of correlations across databases."""
    if not correlations:
        print("No correlations to plot.")
        return None
    
    databases = list(correlations.keys())
    pearson_rs = [correlations[db]['pearson_r'] for db in databases]
    pearson_ps = [correlations[db]['pearson_p'] for db in databases]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Correlation coefficients bar plot
    colors = ['red' if p < 0.05 else 'lightblue' for p in pearson_ps]
    bars = ax1.bar(range(len(databases)), pearson_rs, color=colors, 
                   edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Database', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Pearson Correlation Coefficient', fontsize=12, fontweight='bold')
    ax1.set_title('Correlation Strength by Database', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(databases)))
    ax1.set_xticklabels(databases, rotation=45, ha='right')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.grid(True, alpha=0.3)
    
    # Add significance indicators
    for i, (r, p) in enumerate(zip(pearson_rs, pearson_ps)):
        if p < 0.001:
            ax1.text(i, r + 0.02, '***', ha='center', fontweight='bold')
        elif p < 0.01:
            ax1.text(i, r + 0.02, '**', ha='center', fontweight='bold')
        elif p < 0.05:
            ax1.text(i, r + 0.02, '*', ha='center', fontweight='bold')
    
    # P-values plot
    ax2.scatter(range(len(databases)), pearson_ps, s=100, 
               c=['red' if p < 0.05 else 'blue' for p in pearson_ps],
               edgecolors='black', linewidth=0.5)
    ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, 
               label='p = 0.05 threshold')
    ax2.set_xlabel('Database', fontsize=12, fontweight='bold')
    ax2.set_ylabel('P-value', fontsize=12, fontweight='bold')
    ax2.set_title('Statistical Significance by Database', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(databases)))
    ax2.set_xticklabels(databases, rotation=45, ha='right')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, f'{dataset_name}_correlation_summary.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    return output_path

def print_correlation_table(correlations, dataset_name):
    """Print a formatted table of correlations."""
    print(f"\n{dataset_name.upper()} - CORRELATION ANALYSIS RESULTS")
    print("=" * 80)
    print(f"{'Database':<20} {'Pearson r':<12} {'P-value':<12} {'Spearman r':<12} {'N':<8} {'Significance':<12}")
    print("-" * 80)
    
    if not correlations:
        print("No valid correlations found.")
        return
    
    for db, stats in correlations.items():
        significance = ""
        if stats['pearson_p'] < 0.001:
            significance = "***"
        elif stats['pearson_p'] < 0.01:
            significance = "**"
        elif stats['pearson_p'] < 0.05:
            significance = "*"
        else:
            significance = "ns"
        
        print(f"{db:<20} {stats['pearson_r']:<12.4f} {stats['pearson_p']:<12.4f} "
              f"{stats['spearman_r']:<12.4f} {stats['n_samples']:<8} {significance:<12}")
    
    print("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")

def main():
    if len(sys.argv) != 2:
        print("Usage: python predict3.3-FindCoorelationInGeneIdentification.py <dataset_name>")
        print("Example: python predict3.3-FindCoorelationInGeneIdentification.py migraine")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    
    # Construct file path /data/ascher02/uqmmune1/ANNOVAR/migraine/Model/CrossModel_Comparison/migraine_CrossModel_SeparateColumns_Matrix.csv
    filepath = f'/data/ascher02/uqmmune1/ANNOVAR/{dataset_name}/Model/CrossModel_Comparison/{dataset_name}_CrossModel_SeparateColumns_Matrix.csv'

    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    
    # Get output directory (same as input directory)
    output_dir = get_input_directory(filepath)
    
    print(f"Analyzing correlations for dataset: {dataset_name}")
    print(f"Reading data from: {filepath}")
    print(f"Output will be saved to: {output_dir}")
    
    # Process data
    data = read_and_process_data(filepath)
    
    if not data:
        print("No valid data found. Exiting.")
        sys.exit(1)
    
    # Calculate correlations
    correlations = calculate_correlations(data)
    
    # Print results table
    print_correlation_table(correlations, dataset_name)
    
    # Create plots only if we have data
    if data:
        plot_path1 = create_correlation_plots(data, correlations, dataset_name, output_dir)
        if plot_path1:
            print(f"\nPlots saved to:")
            print(f"  - {plot_path1}")
        
        if correlations:
            plot_path2 = create_summary_plot(correlations, dataset_name, output_dir)
            if plot_path2:
                print(f"  - {plot_path2}")

if __name__ == "__main__":
    main()
