#!/usr/bin/env python3
"""
🔥 Differential Expression Plotting Script 🔥
FINAL FIX: Aggressive dtype forcing for volcano plots with separate DE and ASSOC thresholds
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import argparse
from collections import defaultdict

warnings.filterwarnings('ignore')

# GLOBAL THRESHOLDS
FDR_THRESHOLD = 0.1
DE_EFFECT_THRESHOLD = 0.2      # For DE methods (Log2FoldChange)
ASSOC_EFFECT_THRESHOLD = 0.3   # For association methods (log-odds, coefficients)

# DATABASES
DATABASES_TO_ANALYZE = ["Regular", "JTI", "UTMOST", "UTMOST2", "EpiX", "TIGAR", "FUSION"]

# 8 STATISTICAL METHODS
DE_METHODS = ["LIMMA", "Welch_t_test", "Linear_Regression", "Wilcoxon_Rank_Sum", "Permutation_Test"]
ASSOC_METHODS = ["Weighted_Logistic", "Firth_Logistic", "Bayesian_Logistic"]
ALL_METHODS = DE_METHODS + ASSOC_METHODS

class DifferentialExpressionPlotter:
    def __init__(self, phenotype):
        """Initialize plotter"""
        self.phenotype = phenotype
        self.base_path = Path(phenotype)
        self.results_dir = self.base_path / "GeneDifferentialExpression" / "Files"
        self.plots_dir = self.results_dir / "plots_only"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🔥 DIFFERENTIAL EXPRESSION PLOTTER (8 METHODS)")
        print("=" * 80)
        print(f"📋 Phenotype: {phenotype}")
        print(f"📁 Results directory: {self.results_dir}")
        print(f"📁 Plots output: {self.plots_dir}")
        print(f"🎯 Thresholds:")
        print(f"   FDR < {FDR_THRESHOLD}")
        print(f"   DE Methods: |Log2FC| ≥ {DE_EFFECT_THRESHOLD}")
        print(f"   Association Methods: |Effect| ≥ {ASSOC_EFFECT_THRESHOLD}")
        print("=" * 80)
        self._validate_results()
    
    def _validate_results(self):
        """Check if required result files exist"""
        if not self.results_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {self.results_dir}")
        
        matrix_files = list(self.results_dir.glob("matrix_*_database.csv"))
        volcano_files = list(self.results_dir.glob("combined_volcano_data_*.csv"))
        
        print(f"📊 Found {len(matrix_files)} matrix files")
        print(f"📊 Found {len(volcano_files)} volcano data files")
        
        if len(matrix_files) == 0 and len(volcano_files) == 0:
            raise FileNotFoundError("No result files found.")
    
    def load_matrix_data(self, database_name):
        """Load matrix data"""
        safe_name = database_name.lower().replace(' ', '_')
        matrix_file = self.results_dir / f"matrix_{safe_name}_database.csv"
        
        if matrix_file.exists():
            df = pd.read_csv(matrix_file, index_col=0)
            print(f"📊 Loaded matrix for {database_name}: {df.shape}")
            return df
        return pd.DataFrame()
    
    def load_combined_matrix_data(self):
        """Load combined matrix"""
        matrix_file = self.results_dir / "matrix_combined_all_databases.csv"
        if matrix_file.exists():
            df = pd.read_csv(matrix_file, index_col=0)
            print(f"📊 Loaded combined matrix: {df.shape}")
            return df
        return pd.DataFrame()
    
    def load_similarity_data(self):
        """Load similarity matrix"""
        similarity_file = self.results_dir / "database_similarity_matrix.csv"
        if similarity_file.exists():
            df = pd.read_csv(similarity_file, index_col=0)
            print(f"📊 Loaded similarity matrix: {df.shape}")
            return df
        return pd.DataFrame()
    
    def load_volcano_data(self):
        """Load volcano data"""
        volcano_files = list(self.results_dir.glob("combined_volcano_data_*.csv"))
        if volcano_files:
            volcano_file = volcano_files[0]
            df = pd.read_csv(volcano_file)
            print(f"📊 Loaded volcano data: {df.shape} from {volcano_file.name}")
            return df
        return pd.DataFrame()
    
    def create_professional_heatmap(self, matrix_df, title, save_path=None):
        """Create professional heatmap"""
        if matrix_df.empty:
            return None
        
        print(f"🔥 Creating heatmap: {title}")
        
        n_tissues = len(matrix_df.index)
        n_methods = len(matrix_df.columns)
        
        width = max(12, n_methods * 2.5)
        height = max(8, n_tissues * 0.4)
        plt.figure(figsize=(width, height))
        
        matrix_df = matrix_df.fillna(0)
        
        # Create annotations
        annotation_matrix = matrix_df.copy().astype(str)
        for i in range(len(matrix_df.index)):
            for j in range(len(matrix_df.columns)):
                value = matrix_df.iloc[i, j]
                if pd.notna(value) and value > 0:
                    if value >= 1000:
                        annotation_matrix.iloc[i, j] = f"{int(value/1000)}k"
                    else:
                        annotation_matrix.iloc[i, j] = f"{int(value)}"
                else:
                    annotation_matrix.iloc[i, j] = "0"
        
        # Normalization
        normalized_matrix = matrix_df.copy().astype(float)
        max_value = normalized_matrix.max().max()
        if max_value > 0:
            normalized_matrix = np.log1p(normalized_matrix) / np.log1p(max_value)
        
        # Create heatmap
        ax = sns.heatmap(
            normalized_matrix, annot=annotation_matrix, fmt='', cmap='RdYlBu_r',
            center=0.5, vmin=0, vmax=1,
            cbar_kws={'label': 'Normalized Gene Count (Log Scale)', 'shrink': 0.8},
            linewidths=0.8, square=False,
            annot_kws={'fontsize': min(12, max(8, 120//max(n_tissues, n_methods))), 'fontweight': 'normal'},
            xticklabels=True, yticklabels=True
        )
        
        plt.title(title, fontsize=min(16, max(12, 200//max(n_tissues, n_methods))), 
                 fontweight='normal', pad=25)
        plt.xlabel('Database' if n_methods > 1 else 'Expression Method', fontsize=14, fontweight='normal')
        plt.ylabel('Tissue (Alphabetical Order)', fontsize=14, fontweight='normal')
        
        ax.set_xticklabels(matrix_df.columns, rotation=45, ha='right', fontsize=11, fontweight='normal')
        plt.yticks(rotation=0, fontsize=min(11, max(8, 150//n_tissues)), fontweight='normal')
        
        if n_methods > 1:
            for i in range(1, n_methods):
                plt.axvline(x=i, color='white', linewidth=2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ Heatmap saved: {save_path}")
        
        plt.close()
        return save_path
    
    def create_similarity_heatmap(self, similarity_df):
        """Create similarity heatmap"""
        if similarity_df.empty:
            return None
        
        print(f"🔥 Creating database similarity heatmap")
        
        plt.figure(figsize=(10, 8))
        
        sns.heatmap(
            similarity_df, annot=True, fmt='.3f', cmap='RdYlBu_r',
            center=0.5, vmin=0, vmax=1, square=True,
            cbar_kws={'label': 'Jaccard Similarity', 'shrink': 0.8},
            linewidths=0.5, annot_kws={'fontweight': 'normal'}
        )
        
        plt.title(f'{self.phenotype} - Database Similarity\n(Jaccard Similarity, FDR<{FDR_THRESHOLD})', 
                 fontsize=14, fontweight='normal', pad=20)
        plt.xlabel('Database', fontsize=12, fontweight='normal')
        plt.ylabel('Database', fontsize=12, fontweight='normal')
        plt.xticks(rotation=45, ha='right', fontweight='normal')
        plt.yticks(rotation=0, fontweight='normal')
        
        save_path = self.plots_dir / f"database_similarity_heatmap.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Similarity heatmap saved: {save_path}")
        return save_path
    
    def create_individual_volcano_plot(self, plot_df, database_name):
        """Create volcano plot - FINAL FIX with aggressive dtype forcing and dual thresholds"""
        try:
            if len(plot_df) == 0:
                print(f"⚠️  No data for {database_name}")
                return None
            
            print(f"🔥 Creating volcano plot for {database_name}")
            print(f"      📊 Initial: {len(plot_df):,} rows")
            
            # ===== AGGRESSIVE DTYPE FORCING =====
            plot_df = plot_df.copy()
            
            # Force numeric conversion and astype
            plot_df['Log2FoldChange'] = pd.to_numeric(plot_df['Log2FoldChange'], errors='coerce').astype(float)
            plot_df['FDR'] = pd.to_numeric(plot_df['FDR'], errors='coerce').astype(float)
            plot_df['neg_log10_FDR'] = pd.to_numeric(plot_df['neg_log10_FDR'], errors='coerce').astype(float)
            
            # Drop NaNs
            plot_df = plot_df.dropna(subset=['Log2FoldChange', 'FDR', 'neg_log10_FDR', 'Gene'])
            print(f"      📊 After cleaning: {len(plot_df):,} rows")
            
            if len(plot_df) == 0:
                print(f"⚠️  No valid data after cleaning")
                return None
            
            # Filter outliers
            plot_df = plot_df[(np.abs(plot_df['Log2FoldChange'].values) <= 15) & 
                            (plot_df['neg_log10_FDR'].values <= 50)]
            
            if len(plot_df) == 0:
                print(f"⚠️  No data after filtering")
                return None
            
            # DEDUPLICATE
            print(f"      📊 Before dedup: {len(plot_df):,} entries, {plot_df['Gene'].nunique():,} unique")
            plot_df = plot_df.sort_values('FDR').drop_duplicates(subset=['Gene'], keep='first')
            print(f"      📊 After dedup: {len(plot_df):,} unique genes")
            
            # Separate significant
            sig_genes = plot_df[plot_df['is_significant']]
            non_sig_genes = plot_df[~plot_df['is_significant']]
            
            # Create plot
            plt.figure(figsize=(12, 10))
            
            # Get arrays
            x_data = plot_df['Log2FoldChange'].values
            y_data = plot_df['neg_log10_FDR'].values
            
            x_min = min(x_data.min(), -DE_EFFECT_THRESHOLD * 1.5)
            x_max = max(x_data.max(), DE_EFFECT_THRESHOLD * 1.5)
            
            fdr_line_y = -np.log10(FDR_THRESHOLD)
            y_max = max(y_data.max(), fdr_line_y * 1.3)
            
            x_pad = max(0.5, (x_max - x_min) * 0.15)
            y_pad = max(1.0, y_max * 0.1)
            
            # Plot
            if len(non_sig_genes) > 0:
                plt.scatter(non_sig_genes['Log2FoldChange'].values, non_sig_genes['neg_log10_FDR'].values,
                          c='lightgray', alpha=0.6, s=20, label=f'Non-significant ({len(non_sig_genes):,})',
                          edgecolors='none')
            
            if len(sig_genes) > 0:
                plt.scatter(sig_genes['Log2FoldChange'].values, sig_genes['neg_log10_FDR'].values,
                          c='red', alpha=0.8, s=30, label=f'Significant ({len(sig_genes):,})',
                          edgecolors='darkred', linewidths=0.5)
            
            # Threshold lines - FDR
            plt.axhline(y=fdr_line_y, color='red', linestyle='-', alpha=0.9, linewidth=4, 
                       label=f'FDR = {FDR_THRESHOLD}', zorder=10)
            
            # Threshold lines - DE (solid blue)
            plt.axvline(x=DE_EFFECT_THRESHOLD, color='blue', linestyle='-', alpha=0.9, linewidth=4,
                       label=f'DE: Log₂FC = ±{DE_EFFECT_THRESHOLD}', zorder=10)
            plt.axvline(x=-DE_EFFECT_THRESHOLD, color='blue', linestyle='-', alpha=0.9, linewidth=4, zorder=10)
            
            # Threshold lines - Association (dashed green)
            plt.axvline(x=ASSOC_EFFECT_THRESHOLD, color='green', linestyle='--', alpha=0.7, linewidth=3,
                       label=f'Assoc: Effect = ±{ASSOC_EFFECT_THRESHOLD}', zorder=9)
            plt.axvline(x=-ASSOC_EFFECT_THRESHOLD, color='green', linestyle='--', alpha=0.7, linewidth=3, zorder=9)
            
            # Set limits
            plt.xlim(x_min - x_pad, x_max + x_pad)
            plt.ylim(0, y_max + y_pad)
            
            # Labels
            plt.xlabel('Log₂ Fold Change / Effect Size', fontsize=14, fontweight='bold')
            plt.ylabel('-log₁₀(FDR)', fontsize=14, fontweight='bold')
            plt.title(f'{self.phenotype} - {database_name}\nVolcano Plot (Unique Genes)', 
                     fontsize=16, fontweight='bold', pad=20)
            
            # Stats
            sig_pct = (len(sig_genes) / len(plot_df) * 100) if len(plot_df) > 0 else 0
            plt.text(0.98, 0.98, f'Unique: {len(plot_df):,}\nSignificant: {len(sig_genes):,} ({sig_pct:.1f}%)', 
                    transform=plt.gca().transAxes, fontsize=11, fontweight='bold',
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))
            
            plt.legend(loc='upper left', fontsize=9, frameon=True, fancybox=True, shadow=True)
            plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            plt.tight_layout()
            
            # Save
            safe_name = database_name.replace(" ", "_").replace("/", "_")
            save_path = self.plots_dir / f"volcano_plot_{safe_name.lower()}_model.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"✅ {database_name} volcano saved: {save_path}")
            return save_path
            
        except Exception as e:
            print(f"❌ Error creating {database_name} volcano plot: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_combined_volcano_plot(self, combined_df):
        """Create combined volcano - FINAL FIX with dual thresholds"""
        try:
            if len(combined_df) == 0:
                return None
            
            print(f"🔥 Creating combined volcano plot")
            print(f"      📊 Initial: {len(combined_df):,} rows")
            
            # AGGRESSIVE DTYPE FORCING
            combined_df = combined_df.copy()
            combined_df['Log2FoldChange'] = pd.to_numeric(combined_df['Log2FoldChange'], errors='coerce').astype(float)
            combined_df['FDR'] = pd.to_numeric(combined_df['FDR'], errors='coerce').astype(float)
            combined_df['neg_log10_FDR'] = pd.to_numeric(combined_df['neg_log10_FDR'], errors='coerce').astype(float)
            
            combined_df = combined_df.dropna(subset=['Log2FoldChange', 'FDR', 'neg_log10_FDR', 'Gene'])
            print(f"      📊 After cleaning: {len(combined_df):,} rows")
            
            if len(combined_df) == 0:
                return None
            
            # Filter
            combined_df = combined_df[(np.abs(combined_df['Log2FoldChange'].values) <= 15) & 
                                    (combined_df['neg_log10_FDR'].values <= 50)]
            
            if len(combined_df) == 0:
                return None
            
            # DEDUPLICATE
            print(f"      📊 Before dedup: {len(combined_df):,} entries, {combined_df['Gene'].nunique():,} unique")
            combined_df = combined_df.sort_values('FDR').drop_duplicates(subset=['Gene'], keep='first')
            print(f"      📊 After dedup: {len(combined_df):,} unique genes")
            
            # Colors
            model_colors = {
                'Regular': {'sig': '#e74c3c', 'non_sig': '#f8d7da'},
                'JTI': {'sig': '#3498db', 'non_sig': '#d1ecf1'},
                'UTMOST': {'sig': '#f39c12', 'non_sig': '#fdeaa7'},
                'UTMOST2': {'sig': '#9b59b6', 'non_sig': '#e8daef'},
                'EpiX': {'sig': '#1abc9c', 'non_sig': '#d5f4e6'},
                'TIGAR': {'sig': '#34495e', 'non_sig': '#d5dbdb'},
                'FUSION': {'sig': '#27ae60', 'non_sig': '#d5f4e6'}
            }
            
            plt.figure(figsize=(14, 12))
            
            # Calculate limits
            x_data = combined_df['Log2FoldChange'].values
            y_data = combined_df['neg_log10_FDR'].values
            
            x_min = min(x_data.min(), -DE_EFFECT_THRESHOLD * 1.5)
            x_max = max(x_data.max(), DE_EFFECT_THRESHOLD * 1.5)
            fdr_line_y = -np.log10(FDR_THRESHOLD)
            y_max = max(y_data.max(), fdr_line_y * 1.3)
            
            x_pad = max(0.5, (x_max - x_min) * 0.15)
            y_pad = max(1.0, y_max * 0.1)
            
            # Plot by database
            for database in combined_df['Database'].unique():
                db_data = combined_df[combined_df['Database'] == database]
                sig_genes = db_data[db_data['is_significant']]
                non_sig_genes = db_data[~db_data['is_significant']]
                
                colors = model_colors.get(database, {'sig': '#e74c3c', 'non_sig': '#f8d7da'})
                
                if len(non_sig_genes) > 0:
                    plt.scatter(non_sig_genes['Log2FoldChange'].values, non_sig_genes['neg_log10_FDR'].values,
                              c=colors['non_sig'], alpha=0.6, s=15, edgecolors='none')
                
                plt.scatter(sig_genes['Log2FoldChange'].values if len(sig_genes) > 0 else [], 
                          sig_genes['neg_log10_FDR'].values if len(sig_genes) > 0 else [],
                          c=colors['sig'], alpha=0.8, s=25, edgecolors='black', linewidths=0.5,
                          label=f'{database}: {len(sig_genes):,} genes')
            
            # Threshold lines - FDR
            plt.axhline(y=fdr_line_y, color='red', linestyle='-', alpha=0.9, linewidth=4, 
                       label=f'FDR = {FDR_THRESHOLD}', zorder=10)
            
            # Threshold lines - DE (solid blue)
            plt.axvline(x=DE_EFFECT_THRESHOLD, color='blue', linestyle='-', alpha=0.9, linewidth=4,
                       label=f'DE: Log₂FC = ±{DE_EFFECT_THRESHOLD}', zorder=10)
            plt.axvline(x=-DE_EFFECT_THRESHOLD, color='blue', linestyle='-', alpha=0.9, linewidth=4, zorder=10)
            
            # Threshold lines - Association (dashed green)
            plt.axvline(x=ASSOC_EFFECT_THRESHOLD, color='green', linestyle='--', alpha=0.7, linewidth=3,
                       label=f'Assoc: Effect = ±{ASSOC_EFFECT_THRESHOLD}', zorder=9)
            plt.axvline(x=-ASSOC_EFFECT_THRESHOLD, color='green', linestyle='--', alpha=0.7, linewidth=3, zorder=9)
            
            # Set limits
            plt.xlim(x_min - x_pad, x_max + x_pad)
            plt.ylim(0, y_max + y_pad)
            
            # Labels
            plt.xlabel('Log₂ Fold Change / Effect Size', fontsize=14, fontweight='bold')
            plt.ylabel('-log₁₀(FDR)', fontsize=14, fontweight='bold')
            plt.title(f'{self.phenotype} - Combined Databases\nVolcano Plot (Unique Genes)', 
                     fontsize=16, fontweight='bold', pad=20)
            
            # Stats
            total_sig = len(combined_df[combined_df['is_significant']])
            sig_pct = (total_sig / len(combined_df) * 100) if len(combined_df) > 0 else 0
            plt.text(0.02, 0.98, f'Unique: {len(combined_df):,}\nSignificant: {total_sig:,} ({sig_pct:.1f}%)', 
                    transform=plt.gca().transAxes, fontsize=11, fontweight='bold',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9))
            
            # Legends
            handles, labels = plt.gca().get_legend_handles_labels()
            
            # Separate threshold and database handles
            threshold_handles = []
            threshold_labels = []
            database_handles = []
            database_labels = []
            
            for handle, label in zip(handles, labels):
                if 'FDR' in label or 'Log₂FC' in label or 'DE:' in label or 'Assoc:' in label:
                    threshold_handles.append(handle)
                    threshold_labels.append(label)
                else:
                    database_handles.append(handle)
                    database_labels.append(label)
            
            # Sort database entries
            if database_handles:
                db_pairs = list(zip(database_handles, database_labels))
                db_pairs.sort(key=lambda x: x[1].split(':')[0])
                database_handles, database_labels = zip(*db_pairs)
            
            # Create legends
            if threshold_handles:
                threshold_legend = plt.legend(threshold_handles, threshold_labels, 
                                            loc='upper left', fontsize=9, 
                                            frameon=True, fancybox=True, shadow=True,
                                            title='Significance Thresholds', title_fontsize=10,
                                            bbox_to_anchor=(0.02, 0.85))
                plt.gca().add_artist(threshold_legend)
            
            if database_handles:
                plt.legend(database_handles, database_labels, 
                          loc='upper right', fontsize=9, 
                          frameon=True, fancybox=True, shadow=True,
                          title='Gene Expression Models', title_fontsize=10,
                          bbox_to_anchor=(0.98, 0.98))
            
            plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            plt.tight_layout()
            
            save_path = self.plots_dir / f"volcano_plot_combined_all_models.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"✅ Combined volcano saved: {save_path}")
            return save_path
            
        except Exception as e:
            print(f"❌ Error creating combined volcano: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def plot_all_heatmaps(self):
        """Create all heatmaps"""
        print(f"\n🔥 CREATING ALL HEATMAPS...")
        print("=" * 80)
        
        heatmap_files = []
        
        for database in DATABASES_TO_ANALYZE:
            matrix_df = self.load_matrix_data(database)
            if not matrix_df.empty:
                title = f"{self.phenotype} - {database}\nTissues vs Methods (FDR<{FDR_THRESHOLD})"
                save_path = self.plots_dir / f"heatmap_{database.lower().replace(' ', '_')}_database.png"
                heatmap_file = self.create_professional_heatmap(matrix_df, title, save_path)
                if heatmap_file:
                    heatmap_files.append(heatmap_file)
        
        combined_df = self.load_combined_matrix_data()
        if not combined_df.empty:
            title = f"{self.phenotype} - All Databases\nTissues vs Databases (FDR<{FDR_THRESHOLD})"
            save_path = self.plots_dir / f"heatmap_combined_all_databases.png"
            heatmap_file = self.create_professional_heatmap(combined_df, title, save_path)
            if heatmap_file:
                heatmap_files.append(heatmap_file)
        
        similarity_df = self.load_similarity_data()
        if not similarity_df.empty:
            similarity_file = self.create_similarity_heatmap(similarity_df)
            if similarity_file:
                heatmap_files.append(similarity_file)
        
        print(f"✅ Created {len(heatmap_files)} heatmaps")
        return heatmap_files
    
    def plot_all_volcano_plots(self):
        """Create all volcano plots"""
        print(f"\n🔥 CREATING ALL VOLCANO PLOTS...")
        print("=" * 80)
        
        volcano_files = []
        combined_volcano_df = self.load_volcano_data()
        
        if combined_volcano_df.empty:
            print("❌ No volcano data found")
            return volcano_files
        
        for database in combined_volcano_df['Database'].unique():
            db_data = combined_volcano_df[combined_volcano_df['Database'] == database]
            volcano_file = self.create_individual_volcano_plot(db_data, database)
            if volcano_file:
                volcano_files.append(volcano_file)
        
        combined_file = self.create_combined_volcano_plot(combined_volcano_df)
        if combined_file:
            volcano_files.append(combined_file)
        
        print(f"✅ Created {len(volcano_files)} volcano plots")
        return volcano_files
    
    def plot_all(self):
        """Create all plots"""
        print(f"\n🔥 CREATING ALL PLOTS...")
        print("=" * 80)
        
        try:
            heatmap_files = self.plot_all_heatmaps()
            volcano_files = self.plot_all_volcano_plots()
            
            print(f"\n🔥 PLOTTING COMPLETE")
            print("=" * 80)
            print(f"✅ Heatmaps: {len(heatmap_files)}")
            print(f"✅ Volcano plots: {len(volcano_files)}")
            print(f"✅ Total: {len(heatmap_files) + len(volcano_files)}")
            print(f"📁 Saved to: {self.plots_dir}")
            print("=" * 80)
            
            return heatmap_files, volcano_files
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return [], []

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Volcano Plotter - FINAL FIX with Dual Thresholds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
THRESHOLDS:
  FDR < {FDR_THRESHOLD}
  DE Methods: |Log2FC| ≥ {DE_EFFECT_THRESHOLD} (solid blue lines)
  Association Methods: |Effect| ≥ {ASSOC_EFFECT_THRESHOLD} (dashed green lines)

VISUALIZATION:
  - Blue solid lines: DE threshold ({DE_EFFECT_THRESHOLD})
  - Green dashed lines: Association threshold ({ASSOC_EFFECT_THRESHOLD})
  - Red horizontal line: FDR threshold ({FDR_THRESHOLD})
  - Red points: Significant genes
  - Gray points: Non-significant genes

FEATURES:
  ✅ Aggressive dtype forcing
  ✅ Gene deduplication (unique genes only)
  ✅ Dual threshold visualization
  ✅ Professional heatmaps
  ✅ Database similarity analysis
        """
    )
    parser.add_argument("phenotype", help="Phenotype name")
    args = parser.parse_args()
    
    print(f"🔥 VOLCANO PLOTTER - FINAL FIX WITH DUAL THRESHOLDS")
    print("=" * 80)
    print(f"📋 Phenotype: {args.phenotype}")
    print(f"🎯 Thresholds:")
    print(f"   FDR < {FDR_THRESHOLD}")
    print(f"   DE Methods: |Log2FC| ≥ {DE_EFFECT_THRESHOLD} (blue solid)")
    print(f"   Association Methods: |Effect| ≥ {ASSOC_EFFECT_THRESHOLD} (green dashed)")
    print(f"✅ Aggressive dtype forcing: ENABLED")
    print(f"✅ Gene deduplication: ENABLED")
    print(f"✅ Dual threshold visualization: ENABLED")
    print("=" * 80)
    
    try:
        plotter = DifferentialExpressionPlotter(args.phenotype)
        heatmap_files, volcano_files = plotter.plot_all()
        
        if len(heatmap_files) + len(volcano_files) > 0:
            print(f"\n✅ SUCCESS! Created {len(heatmap_files) + len(volcano_files)} plots")
            print(f"📁 Output: {plotter.plots_dir}")
        else:
            print(f"\n❌ No plots created")
            sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()