#!/usr/bin/env python3
"""
Top-K Gene Enrichment Analysis for Drug Repurposing
====================================================
Runs pathway/disease enrichment on ranked genes from comprehensive analysis.

Uses ranked genes from: {phenotype}/GeneDifferentialExpression/Files/UltimateCompleteRankingAnalysis/RANKED_composite.csv
Background: U_all genes from the analysis (all tested genes)

Performs enrichment for multiple Top-K sets:
- Top 50, 100, 200, 500, 1000 genes
- Tests robustness across different stringency levels

Output: {phenotype}/GeneDifferentialExpression/Files/UltimateCompleteRankingAnalysis/EnrichmentResults/

USAGE:
  python predict6-TopGeneEnrichment.py migraine
  python predict6-TopGeneEnrichment.py migraine --r-script /path/to/predict5-GeneEnrichment.R
"""

import argparse
import pandas as pd
import polars as pl
import numpy as np
from pathlib import Path
import subprocess
import sys
from datetime import datetime

class TopGeneEnrichmentAnalysis:
    """
    Performs enrichment analysis on top-K ranked genes for drug repurposing
    """
    
    def __init__(self, phenotype, r_script_path=None, top_k_values=None):
        self.phenotype = phenotype
        self.base_path = Path(phenotype)
        self.results_dir = self.base_path / "GeneDifferentialExpression" / "Files"
        self.ranking_dir = self.results_dir / "UltimateCompleteRankingAnalysis"
        self.output_dir = self.ranking_dir / "EnrichmentResults"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # R script path
        self.r_script = Path(r_script_path) if r_script_path else Path("predict5-GeneEnrichment.R")
        if not self.r_script.exists():
            raise FileNotFoundError(
                f"R script not found: {self.r_script}\n"
                f"Please ensure predict5-GeneEnrichment.R exists or specify --r-script path"
            )
        
        # Top-K values to analyze
        self.top_k_values = top_k_values or [50, 100, 200, 500, 1000,2000]
        
        print("=" * 100)
        print("🔥 TOP-K GENE ENRICHMENT ANALYSIS FOR DRUG REPURPOSING")
        print("=" * 100)
        print(f"📋 Phenotype: {phenotype}")
        print(f"📁 Ranking directory: {self.ranking_dir}")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🎯 Top-K values: {self.top_k_values}")
        print(f"📜 R script: {self.r_script}")
        print("=" * 100)
        
    def load_ranked_genes(self):
        """Load ranked genes from comprehensive analysis"""
        print("\n🔄 Loading ranked genes...")
        
        ranked_file = self.ranking_dir / "RANKED_composite.csv"
        if not ranked_file.exists():
            raise FileNotFoundError(
                f"Ranked genes file not found: {ranked_file}\n"
                f"Please run predict4_1_2-GeneDifferentialExpression-Analysis.py first"
            )
        
        self.ranked_df = pd.read_csv(ranked_file)
        
        if 'Gene' not in self.ranked_df.columns:
            raise ValueError("'Gene' column not found in ranked file")
        
        # Strip version numbers from ENSG IDs
        self.ranked_df['Gene'] = self.ranked_df['Gene'].astype(str).str.replace(r'\.\d+$', '', regex=True)
        
        print(f"✅ Loaded {len(self.ranked_df):,} ranked genes")
        print(f"   Top gene: {self.ranked_df.iloc[0]['Gene']}")
        if 'Symbol' in self.ranked_df.columns:
            print(f"   Symbol: {self.ranked_df.iloc[0].get('Symbol', 'N/A')}")
        print(f"   Score: {self.ranked_df.iloc[0].get('Importance_Score', 'N/A'):.4f}")
        
        return self.ranked_df
    
    def load_universe_genes(self):
        """Load universe (background) genes - all tested genes"""
        print("\n🔄 Loading universe genes (background)...")
        
        # Try to load from volcano data (most comprehensive)
        volcano_file = self.results_dir / "combined_volcano_data_all_models.csv"
        
        if volcano_file.exists():
            print(f"   Loading from: {volcano_file.name}")
            # Read only Gene column for efficiency using polars
            df = pl.read_csv(volcano_file, columns=['Gene'])
            # Strip version numbers from ENSG IDs
            df = df.with_columns(
                pl.col('Gene').cast(pl.Utf8).str.replace(r'\.\d+$', '', literal=False)
            )
            universe_genes = sorted(df['Gene'].unique().to_list())
            print(f"✅ Loaded {len(universe_genes):,} universe genes")
        else:
            # Fallback: use all genes from ranked file
            print("   ⚠️  Volcano file not found, using ranked genes as universe")
            universe_genes = sorted(self.ranked_df['Gene'].unique().tolist())
            print(f"✅ Using {len(universe_genes):,} genes as universe")
        
        self.universe_genes = universe_genes
        return universe_genes
    
    def create_gene_lists(self, k):
        """Create foreground and background gene lists for enrichment"""
        print(f"\n📊 Creating gene lists for Top-{k}...")
        
        # Get top-K genes
        top_k_genes = self.ranked_df.head(k)['Gene'].tolist()
        
        # Also check directionality if available
        up_genes = []
        down_genes = []
        
        if 'Direction' in self.ranked_df.columns:
            top_k_df = self.ranked_df.head(k)
            up_genes = top_k_df[top_k_df['Direction'] == 'Up']['Gene'].tolist()
            down_genes = top_k_df[top_k_df['Direction'] == 'Down']['Gene'].tolist()
            
            print(f"   Total genes: {len(top_k_genes)}")
            print(f"   Upregulated: {len(up_genes)}")
            print(f"   Downregulated: {len(down_genes)}")
            print(f"   Mixed/Unknown: {len(top_k_genes) - len(up_genes) - len(down_genes)}")
        else:
            print(f"   Total genes: {len(top_k_genes)}")
            print(f"   ⚠️  Direction information not available")
        
        # Background: all universe genes
        background_genes = self.universe_genes
        print(f"   Background genes: {len(background_genes):,}")
        
        return top_k_genes, background_genes, up_genes, down_genes
    
    def save_gene_lists(self, k, top_k_genes, background_genes, up_genes, down_genes):
        """Save gene lists to files for R script"""
        k_dir = self.output_dir / f"Top{k}"
        k_dir.mkdir(parents=True, exist_ok=True)
        
        # Save significant genes (top-K)
        sig_file = k_dir / "significant_genes.txt"
        with open(sig_file, 'w') as f:
            f.write('\n'.join(top_k_genes))
        
        # Save background
        bg_file = k_dir / "background_genes.txt"
        with open(bg_file, 'w') as f:
            f.write('\n'.join(background_genes))
        
        # Save directional genes if available
        up_file = k_dir / "upregulated_genes.txt"
        down_file = k_dir / "downregulated_genes.txt"
        
        if up_genes:
            with open(up_file, 'w') as f:
                f.write('\n'.join(up_genes))
        
        if down_genes:
            with open(down_file, 'w') as f:
                f.write('\n'.join(down_genes))
        
        # Save metadata
        metadata = {
            'Analysis_Type': 'Top-K Gene Enrichment',
            'K_Value': k,
            'Total_Genes': len(top_k_genes),
            'Upregulated': len(up_genes),
            'Downregulated': len(down_genes),
            'Background_Genes': len(background_genes),
            'Analysis_Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Source_File': 'RANKED_composite.csv',
            'Purpose': 'Drug repurposing pathway/disease enrichment'
        }
        
        pd.DataFrame([metadata]).to_csv(k_dir / "metadata.csv", index=False)
        
        return k_dir, sig_file, bg_file, up_file, down_file
    
    def run_enrichment(self, k_dir, sig_file, bg_file, up_file, down_file):
        """Run R enrichment script"""
        print(f"   🔄 Running enrichment analysis...")
        
        results_dir = k_dir / "enrichment_results"
        results_dir.mkdir(exist_ok=True)
        
        # Prepare command
        cmd = [
            "Rscript",
            str(self.r_script),
            str(sig_file),
            str(bg_file),
            str(up_file) if up_file.exists() else "NA",
            str(down_file) if down_file.exists() else "NA",
            str(results_dir)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )
            
            if result.returncode == 0:
                print(f"   ✅ Enrichment completed successfully")
                
                # Check what files were created
                enrichment_files = list(results_dir.glob("*.csv"))
                if enrichment_files:
                    print(f"   📁 Generated {len(enrichment_files)} enrichment files:")
                    for f in sorted(enrichment_files)[:5]:  # Show first 5
                        print(f"      - {f.name}")
                    if len(enrichment_files) > 5:
                        print(f"      ... and {len(enrichment_files) - 5} more")
                
                return True
            else:
                print(f"   ❌ Enrichment failed")
                print(f"   Error: {result.stderr[:500]}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"   ⏱️  Timeout: Enrichment exceeded 10 minutes")
            return False
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            return False
    
    def analyze_top_k(self, k):
        """Analyze a specific Top-K value"""
        print(f"\n{'=' * 100}")
        print(f"📊 ANALYZING TOP-{k} GENES")
        print(f"{'=' * 100}")
        
        if k > len(self.ranked_df):
            print(f"⚠️  Warning: K={k} exceeds available genes ({len(self.ranked_df)})")
            k = len(self.ranked_df)
            print(f"   Using K={k} instead")
        
        # Create gene lists
        top_k_genes, background_genes, up_genes, down_genes = self.create_gene_lists(k)
        
        # Save to files
        k_dir, sig_file, bg_file, up_file, down_file = self.save_gene_lists(
            k, top_k_genes, background_genes, up_genes, down_genes
        )
        
        # Run enrichment
        success = self.run_enrichment(k_dir, sig_file, bg_file, up_file, down_file)
        
        return success
    
    def create_summary_report(self, results):
        """Create summary report of all enrichment analyses"""
        print(f"\n{'=' * 100}")
        print(f"📋 CREATING SUMMARY REPORT")
        print(f"{'=' * 100}")
        
        report_lines = []
        report_lines.append("=" * 100)
        report_lines.append("TOP-K GENE ENRICHMENT ANALYSIS SUMMARY")
        report_lines.append("=" * 100)
        report_lines.append(f"Phenotype: {self.phenotype}")
        report_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total ranked genes: {len(self.ranked_df):,}")
        report_lines.append(f"Universe size: {len(self.universe_genes):,}")
        report_lines.append("=" * 100)
        
        report_lines.append("\nANALYSES PERFORMED:")
        report_lines.append("-" * 100)
        report_lines.append(f"{'K Value':<10} {'Status':<15} {'Genes':<10} {'Up':<10} {'Down':<10}")
        report_lines.append("-" * 100)
        
        for k, result in results.items():
            status = "✅ Success" if result['success'] else "❌ Failed"
            report_lines.append(
                f"{k:<10} {status:<15} {result['n_genes']:<10} "
                f"{result['n_up']:<10} {result['n_down']:<10}"
            )
        
        report_lines.append("\n" + "=" * 100)
        report_lines.append("OUTPUT STRUCTURE:")
        report_lines.append("=" * 100)
        report_lines.append(f"Base directory: {self.output_dir}")
        report_lines.append("\nFor each Top-K:")
        report_lines.append("  TopK/")
        report_lines.append("    ├── significant_genes.txt (top-K genes)")
        report_lines.append("    ├── background_genes.txt (universe)")
        report_lines.append("    ├── upregulated_genes.txt (if available)")
        report_lines.append("    ├── downregulated_genes.txt (if available)")
        report_lines.append("    ├── metadata.csv")
        report_lines.append("    └── enrichment_results/")
        report_lines.append("          ├── GO_BP_combined.csv")
        report_lines.append("          ├── GO_BP_upregulated.csv")
        report_lines.append("          ├── GO_BP_downregulated.csv")
        report_lines.append("          ├── GO_MF_*.csv")
        report_lines.append("          ├── GO_CC_*.csv")
        report_lines.append("          ├── KEGG_*.csv")
        report_lines.append("          ├── Reactome_*.csv")
        report_lines.append("          ├── Disease_Ontology_*.csv")
        report_lines.append("          └── enrichment_summary.csv")
        
        report_lines.append("\n" + "=" * 100)
        report_lines.append("INTERPRETATION FOR DRUG REPURPOSING:")
        report_lines.append("=" * 100)
        report_lines.append("1. PATHWAY ENRICHMENT (KEGG/Reactome):")
        report_lines.append("   - Identifies biological processes enriched in top genes")
        report_lines.append("   - Drugable pathways → potential therapeutic targets")
        report_lines.append("   - Compare across Top-K to validate robustness")
        report_lines.append("")
        report_lines.append("2. DISEASE ONTOLOGY:")
        report_lines.append("   - Shows diseases associated with your top genes")
        report_lines.append("   - Related diseases → repurposing opportunities")
        report_lines.append("   - Shared mechanisms across conditions")
        report_lines.append("")
        report_lines.append("3. GO ENRICHMENT:")
        report_lines.append("   - Biological processes (BP): What these genes do")
        report_lines.append("   - Molecular function (MF): Biochemical activities")
        report_lines.append("   - Cellular component (CC): Where they act")
        report_lines.append("")
        report_lines.append("4. NEXT STEPS:")
        report_lines.append("   - Query enriched pathways against DrugBank/LINCS")
        report_lines.append("   - Find drugs targeting enriched pathways")
        report_lines.append("   - Prioritize by pathway significance and drug availability")
        report_lines.append("=" * 100)
        
        # Save report
        report_file = self.output_dir / "ENRICHMENT_SUMMARY_REPORT.txt"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"\n✅ Summary report saved: {report_file}")
        
        # Also print to console
        print("\n" + "\n".join(report_lines))
        
        return report_file
    
    def run_all_analyses(self):
        """Run enrichment for all Top-K values"""
        print("\n🚀 STARTING TOP-K ENRICHMENT ANALYSES")
        print("=" * 100)
        
        # Load data
        self.load_ranked_genes()
        self.load_universe_genes()
        
        # Run analyses
        results = {}
        successful = 0
        failed = 0
        
        for k in self.top_k_values:
            try:
                # Get gene counts before analysis
                if k <= len(self.ranked_df):
                    top_k_df = self.ranked_df.head(k)
                    n_genes = len(top_k_df)
                    n_up = len(top_k_df[top_k_df['Direction'] == 'Up']) if 'Direction' in top_k_df.columns else 0
                    n_down = len(top_k_df[top_k_df['Direction'] == 'Down']) if 'Direction' in top_k_df.columns else 0
                else:
                    n_genes = len(self.ranked_df)
                    n_up = 0
                    n_down = 0
                
                success = self.analyze_top_k(k)
                
                results[k] = {
                    'success': success,
                    'n_genes': n_genes,
                    'n_up': n_up,
                    'n_down': n_down
                }
                
                if success:
                    successful += 1
                else:
                    failed += 1
                    
            except Exception as e:
                print(f"\n❌ Error analyzing Top-{k}: {str(e)}")
                results[k] = {
                    'success': False,
                    'n_genes': 0,
                    'n_up': 0,
                    'n_down': 0
                }
                failed += 1
        
        # Create summary report
        self.create_summary_report(results)
        
        # Final summary
        print(f"\n{'=' * 100}")
        print(f"🎉 ANALYSIS COMPLETE!")
        print(f"{'=' * 100}")
        print(f"✅ Successful analyses: {successful}/{len(self.top_k_values)}")
        print(f"❌ Failed analyses: {failed}/{len(self.top_k_values)}")
        print(f"\n📁 All results saved to: {self.output_dir}")
        print(f"{'=' * 100}")
        
        return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Top-K Gene Enrichment Analysis for Drug Repurposing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Run with default Top-K values (50, 100, 200, 500, 1000)
  python predict6-TopGeneEnrichment.py migraine
  
  # Custom Top-K values
  python predict6-TopGeneEnrichment.py migraine --topk 50,100,200
  
  # Specify R script location
  python predict6-TopGeneEnrichment.py migraine --r-script /path/to/predict5-GeneEnrichment.R

OUTPUT:
  {phenotype}/GeneDifferentialExpression/Files/UltimateCompleteRankingAnalysis/EnrichmentResults/
    ├── Top50/
    │   ├── enrichment_results/
    │   │   ├── GO_BP_combined.csv
    │   │   ├── KEGG_combined.csv
    │   │   ├── Reactome_combined.csv
    │   │   └── Disease_Ontology_combined.csv
    │   └── metadata.csv
    ├── Top100/
    ├── Top200/
    ├── Top500/
    ├── Top1000/
    └── ENRICHMENT_SUMMARY_REPORT.txt
        """
    )
    
    parser.add_argument("phenotype", help="Phenotype name (e.g., 'migraine')")
    parser.add_argument("--topk", type=str, default="50,100,200,500,1000,2000",
                       help="Comma-separated Top-K values (default: 50,100,200,500,1000)")
    parser.add_argument("--r-script", type=str, default=None,
                       help="Path to predict5-GeneEnrichment.R script")
    
    args = parser.parse_args()
    
    # Parse Top-K values
    try:
        top_k_values = [int(x.strip()) for x in args.topk.split(',')]
        top_k_values = sorted(set(top_k_values))  # Remove duplicates and sort
    except Exception as e:
        print(f"❌ Error parsing Top-K values: {e}")
        print("Using default: [50, 100, 200, 500, 1000, 2000]")
        top_k_values = [50, 100, 200, 500, 1000,2000]
    
    try:
        analyzer = TopGeneEnrichmentAnalysis(
            phenotype=args.phenotype,
            r_script_path=args.r_script,
            top_k_values=top_k_values
        )
        
        analyzer.run_all_analyses()
        return 0
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())