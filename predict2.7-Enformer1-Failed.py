#!/usr/bin/env python3
"""
Complete End-to-End Enformer Pipeline with Variation-Based Gene Selection

🎯 PIPELINE OVERVIEW:
1. Load training data and extract phenotypes
2. Rank ALL genes by genetic variation between cases/controls
3. Select top N genes with highest variation scores
4. Load Enformer model
5. Generate personalized sequences and expressions for all datasets
6. Save comprehensive results for ML analysis

Usage: python complete_enformer_pipeline.py phenotype fold [n_genes]
Example: python complete_enformer_pipeline.py migraine 0 20
"""

import sys
import pandas as pd
import torch
import numpy as np
from pandas_plink import read_plink
from pathlib import Path
import logging
from collections import defaultdict
from tqdm import tqdm
import warnings
import gc
import pysam

warnings.filterwarnings('ignore', category=FutureWarning, module='pandas_plink')

try:
    from enformer_pytorch import from_pretrained, seq_indices_to_one_hot
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "enformer-pytorch", "pandas-plink", "pysam", "scikit-learn"])
    from enformer_pytorch import from_pretrained, seq_indices_to_one_hot

# Configuration
BATCH_SIZE = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQUENCE_LENGTH = 196608
GENE_WINDOW = 100000  # 100kb window around each gene for SNP inclusion

# File paths - UPDATE THESE
REFERENCE_FASTA = "/data/ascher02/uqmmune1/ANNOVAR/Enformer/hg38.fa"
GENE_GTF = "/data/ascher02/uqmmune1/ANNOVAR/Enformer/gencode.v38.annotation.gtf"

class CompleteEnformerPipeline:
    """
    Complete end-to-end Enformer pipeline with variation-based gene selection
    """
    
    def __init__(self):
        self.setup_logging()
        self.gene_window = GENE_WINDOW
    
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def run_complete_pipeline(self, phenotype, fold, n_genes=20):
        """
        Main pipeline: variation-based gene selection + comprehensive Enformer analysis
        """
        self.logger.info("=" * 100)
        self.logger.info("🧬 COMPLETE ENFORMER PIPELINE WITH VARIATION-BASED GENE SELECTION")
        self.logger.info("=" * 100)
        self.logger.info(f"Phenotype: {phenotype}")
        self.logger.info(f"Fold: {fold}")
        self.logger.info(f"Target genes: {n_genes}")
        self.logger.info(f"Strategy: Select genes with highest genetic variation between cases/controls")
        
        # Setup paths
        base_path = f"{phenotype}/Fold_{fold}/"
        output_path = f"{phenotype}/Fold_{fold}/CompleteEnformerPipeline/"
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        try:
            # Step 1: Variation-based gene selection
            self.logger.info(f"\n{'='*80}")
            self.logger.info("STEP 1: VARIATION-BASED GENE SELECTION")
            self.logger.info(f"{'='*80}")
            
            selected_genes = self.select_genes_by_variation(
                base_path + "train_data", 
                GENE_GTF, 
                n_genes, 
                output_path,
                phenotype
            )
            
            if not selected_genes:
                self.logger.error("Gene selection failed!")
                return False
            
            # Step 2: Load Enformer model
            self.logger.info(f"\n{'='*80}")
            self.logger.info("STEP 2: LOADING ENFORMER MODEL")
            self.logger.info(f"{'='*80}")
            
            model = self.load_enformer_model()
            if model is None:
                return False
            
            # Step 3: Generate expressions for all datasets
            self.logger.info(f"\n{'='*80}")
            self.logger.info("STEP 3: GENERATING ENFORMER EXPRESSIONS FOR ALL DATASETS")
            self.logger.info(f"{'='*80}")
            
            datasets = {
                'train': base_path + "train_data",
                'validation': base_path + "validation_data",
                'test': base_path + "test_data"
            }
            
            all_results = {}
            
            for dataset_name, plink_prefix in datasets.items():
                if not Path(plink_prefix + ".bed").exists():
                    self.logger.warning(f"Skipping {dataset_name} - files not found: {plink_prefix}")
                    continue
                
                self.logger.info(f"\n--- Processing {dataset_name.upper()} dataset ---")
                
                # Generate expressions using Enformer
                expressions, sample_ids, feature_names = self.generate_gene_expressions(
                    plink_prefix, selected_genes, model
                )
                
                if expressions is not None:
                    all_results[dataset_name] = {
                        'expressions': expressions,
                        'sample_ids': sample_ids,
                        'feature_names': feature_names
                    }
                    
                    # Save results immediately  
                    dataset_info = self.save_dataset_results(
                        dataset_name, expressions, sample_ids, selected_genes, output_path, feature_names
                    )
                    
                    all_results[dataset_name]['files'] = dataset_info
            
            # Step 4: Final summary and analysis-ready outputs
            self.logger.info(f"\n{'='*80}")
            self.logger.info("STEP 4: PIPELINE COMPLETION & SUMMARY")
            self.logger.info(f"{'='*80}")
            
            self.generate_final_summary(all_results, selected_genes, output_path, phenotype)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Pipeline failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def select_genes_by_variation(self, train_plink_prefix, gtf_file, n_genes, output_path, phenotype):
        """
        Select genes based on genetic variation between cases and controls
        """
        self.logger.info(f"📊 SELECTING GENES BY GENETIC VARIATION")
        
        # Step 1: Load training data and phenotypes
        bim, fam, bed_array, phenotypes = self.load_training_data_with_phenotypes(train_plink_prefix)
        
        if phenotypes is None:
            self.logger.error("Could not load phenotypes!")
            return None
        
        # Step 2: Load all genes
        all_genes = self.load_all_genes_from_gtf(gtf_file)
        
        if not all_genes:
            self.logger.error("Could not load genes!")
            return None
        
        # Step 3: Calculate variation scores for all genes
        gene_variations = self.calculate_gene_variation_scores(
            all_genes, bim, bed_array, phenotypes, phenotype
        )
        
        # Step 4: Rank and select top genes
        top_genes = self.rank_and_select_top_genes(gene_variations, n_genes)
        
        # Step 5: Create Enformer regions for selected genes
        enformer_regions = self.create_enformer_regions_for_selected_genes(top_genes, gtf_file)
        
        # Step 6: Save selection results
        self.save_gene_selection_results(enformer_regions, gene_variations, output_path, phenotype)
        
        return enformer_regions
    
    def load_training_data_with_phenotypes(self, plink_prefix):
        """
        Load training PLINK data and extract phenotypes with robust handling
        """
        self.logger.info(f"Loading training data: {plink_prefix}")
        
        try:
            # Load PLINK data
            bim, fam, bed = read_plink(plink_prefix)
            bed_array = bed.compute()
            
            self.logger.info(f"Loaded {len(fam)} samples × {len(bim)} variants")
            
            # Extract phenotypes with string handling
            phenotypes = fam.iloc[:, 5].values
            
            self.logger.info(f"Original phenotype values: {np.unique(phenotypes[~pd.isna(phenotypes)])}")
            self.logger.info(f"Phenotype data type: {type(phenotypes[0])}")
            
            # Convert strings to numeric if needed
            if isinstance(phenotypes[0], str):
                self.logger.info("Converting string phenotypes to numeric...")
                phenotypes_numeric = []
                for val in phenotypes:
                    if pd.isna(val):
                        phenotypes_numeric.append(np.nan)
                    else:
                        try:
                            phenotypes_numeric.append(float(val))
                        except:
                            phenotypes_numeric.append(np.nan)
                phenotypes = np.array(phenotypes_numeric)
                self.logger.info(f"After conversion: {np.unique(phenotypes[~pd.isna(phenotypes)])}")
            
            # Convert to 0/1 format
            unique_vals = set(np.unique(phenotypes[~pd.isna(phenotypes)]))
            if unique_vals == {1.0, 2.0}:
                phenotypes_binary = phenotypes - 1  # 1->0 (control), 2->1 (case)
                self.logger.info("Converted from PLINK format (1,2) to binary (0,1)")
            elif unique_vals == {0.0, 1.0}:
                phenotypes_binary = phenotypes.copy()
                self.logger.info("Phenotypes already in binary format (0,1)")
            else:
                self.logger.error(f"Unexpected phenotype values: {unique_vals}")
                return None, None, None, None
            
            # Final validation
            n_controls = np.sum(phenotypes_binary == 0)
            n_cases = np.sum(phenotypes_binary == 1)
            n_missing = np.sum(pd.isna(phenotypes_binary))
            
            self.logger.info(f"Final phenotype distribution:")
            self.logger.info(f"  Controls (0): {n_controls}")
            self.logger.info(f"  Cases (1): {n_cases}")
            self.logger.info(f"  Missing: {n_missing}")
            
            if n_cases < 1 or n_controls < 1:
                self.logger.error("Need at least 1 case and 1 control for variation analysis!")
                return None, None, None, None
            
            self.logger.info("✅ Successfully loaded training data with phenotypes")
            return bim, fam, bed_array, phenotypes_binary.astype(float)
            
        except Exception as e:
            self.logger.error(f"Error loading training data: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None, None
    
    def load_all_genes_from_gtf(self, gtf_file):
        """
        Load ALL protein-coding genes from GTF file
        """
        self.logger.info(f"Loading all protein-coding genes from: {gtf_file}")
        
        genes = []
        
        try:
            with open(gtf_file, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                        
                    fields = line.strip().split('\t')
                    if len(fields) < 9 or fields[2] != 'gene':
                        continue
                    
                    chrom = fields[0]
                    start = int(fields[3])
                    end = int(fields[4])
                    strand = fields[6]
                    
                    # Only main chromosomes
                    if chrom not in [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY']:
                        continue
                    
                    # Parse gene info
                    attributes = fields[8]
                    gene_name = None
                    gene_type = None
                    
                    for attr in attributes.split(';'):
                        attr = attr.strip()
                        if attr.startswith('gene_name'):
                            gene_name = attr.split('"')[1]
                        elif attr.startswith('gene_type'):
                            gene_type = attr.split('"')[1]
                    
                    if gene_name and gene_type == 'protein_coding':
                        tss = start if strand == '+' else end
                        
                        # Define gene region for SNP inclusion (gene ± window)
                        region_start = max(1, min(start, end) - self.gene_window)
                        region_end = max(start, end) + self.gene_window
                        
                        gene_info = {
                            'gene_name': gene_name,
                            'chrom': chrom,
                            'gene_start': start,
                            'gene_end': end,
                            'strand': strand,
                            'tss': tss,
                            'region_start': region_start,
                            'region_end': region_end,
                            'region_length': region_end - region_start
                        }
                        
                        genes.append(gene_info)
            
            self.logger.info(f"Loaded {len(genes)} protein-coding genes for variation analysis")
            
            # Show chromosome distribution
            chrom_dist = defaultdict(int)
            for gene in genes:
                chrom_dist[gene['chrom']] += 1
            
            self.logger.info(f"Gene distribution across chromosomes:")
            for chrom in sorted(chrom_dist.keys())[:10]:  # Show first 10
                self.logger.info(f"  {chrom}: {chrom_dist[chrom]} genes")
            
            return genes
            
        except Exception as e:
            self.logger.error(f"Error loading GTF file: {e}")
            return []
    
    def calculate_gene_variation_scores(self, all_genes, bim, bed_array, phenotypes, phenotype_name):
        """
        Calculate genetic variation scores for each gene region
        """
        self.logger.info(f"📊 CALCULATING GENETIC VARIATION FOR {len(all_genes)} GENES")
        
        gene_variations = []
        total_snps_found = 0
        genes_with_snps = 0
        
        # Get case and control indices
        case_indices = np.where(phenotypes == 1)[0]
        control_indices = np.where(phenotypes == 0)[0]
        
        self.logger.info(f"Analysis groups: {len(case_indices)} cases, {len(control_indices)} controls")
        
        for gene_idx, gene in enumerate(tqdm(all_genes, desc="Calculating variations")):
            gene_name = gene['gene_name']
            chrom = gene['chrom']
            region_start = gene['region_start']
            region_end = gene['region_end']
            
            if gene_idx % 1000 == 0:
                self.logger.info(f"Processing gene {gene_idx+1}/{len(all_genes)}: {gene_name}")
            
            # Get SNPs in this gene region (with chromosome format fix)
            region_snps, snp_indices = self.get_snps_in_gene_region_fixed(bim, chrom, region_start, region_end)
            
            n_snps = len(region_snps)
            total_snps_found += n_snps
            
            if n_snps > 0:
                genes_with_snps += 1
                
                # Show details for first few genes with SNPs
                if genes_with_snps <= 10:
                    self.logger.info(f"  ✅ {gene_name} ({chrom}): {n_snps} SNPs")
            
            if n_snps == 0:
                # No SNPs in region
                gene_variation = {
                    'gene_name': gene_name,
                    'chrom': chrom,
                    'tss': gene['tss'],
                    'region_start': region_start,
                    'region_end': region_end,
                    'n_snps_in_region': 0,
                    'total_variation_score': 0.0,
                    'mean_variation_per_snp': 0.0,
                    'max_snp_variation': 0.0,
                    'variation_density': 0.0
                }
                gene_variations.append(gene_variation)
                continue
            
            # Calculate variation for all SNPs in this region
            snp_variation_scores = []
            
            for idx, (_, snp) in enumerate(region_snps.iterrows()):
                if idx >= len(snp_indices):
                    break
                
                snp_idx = snp_indices[idx]
                if snp_idx >= bed_array.shape[1]:
                    continue
                
                # Get genotypes for this SNP
                genotypes = bed_array[:, snp_idx]
                
                # Calculate variation between cases and controls
                variation_score = self.calculate_snp_variation(genotypes, case_indices, control_indices)
                snp_variation_scores.append(variation_score)
            
            # Aggregate SNP variations to gene-level metrics
            if snp_variation_scores:
                total_variation = sum(snp_variation_scores)
                mean_variation = np.mean(snp_variation_scores)
                max_variation = max(snp_variation_scores)
                variation_density = total_variation / (region_end - region_start) * 1000000  # Per Mb
            else:
                total_variation = mean_variation = max_variation = variation_density = 0.0
            
            gene_variation = {
                'gene_name': gene_name,
                'chrom': chrom,
                'tss': gene['tss'],
                'region_start': region_start,
                'region_end': region_end,
                'n_snps_in_region': n_snps,
                'total_variation_score': total_variation,
                'mean_variation_per_snp': mean_variation,
                'max_snp_variation': max_variation,
                'variation_density': variation_density
            }
            
            gene_variations.append(gene_variation)
        
        # Convert to DataFrame for easier handling
        gene_var_df = pd.DataFrame(gene_variations)
        
        self.logger.info(f"✅ Calculated variations for {len(gene_var_df)} genes")
        self.logger.info(f"Variation statistics:")
        self.logger.info(f"  Total SNPs found across all genes: {total_snps_found:,}")
        self.logger.info(f"  Genes with SNPs: {genes_with_snps}")
        self.logger.info(f"  Mean SNPs per gene: {gene_var_df['n_snps_in_region'].mean():.1f}")
        self.logger.info(f"  Max variation score: {gene_var_df['total_variation_score'].max():.3f}")
        
        return gene_var_df
    
    def get_snps_in_gene_region_fixed(self, bim, chrom, region_start, region_end):
        """
        Get all SNPs within a gene region - with chromosome format fix
        GTF uses: chr1, chr2, etc.
        BIM uses: 1, 2, etc.
        """
        # Convert GTF chromosome (chr1) to BIM format (1)
        bim_chrom = chrom.replace('chr', '') if chrom.startswith('chr') else chrom
        
        # Filter SNPs in region
        mask = (
            (bim['chrom'].astype(str) == bim_chrom) & 
            (bim['pos'] >= region_start) & 
            (bim['pos'] <= region_end)
        )
        
        region_snps = bim[mask].copy().reset_index()
        snp_indices = region_snps['index'].values if len(region_snps) > 0 else []
        
        return region_snps, snp_indices
    
    def calculate_snp_variation(self, genotypes, case_indices, control_indices):
        """
        Calculate variation score for a single SNP between cases and controls
        """
        try:
            # Remove missing genotypes
            valid_cases = case_indices[~pd.isna(genotypes[case_indices])]
            valid_controls = control_indices[~pd.isna(genotypes[control_indices])]
            
            if len(valid_cases) < 1 or len(valid_controls) < 1:
                return 0.0
            
            case_genotypes = genotypes[valid_cases]
            control_genotypes = genotypes[valid_controls]
            
            # Calculate genotype frequencies
            case_freq = {}
            control_freq = {}
            
            for geno in [0, 1, 2]:
                case_freq[geno] = np.sum(case_genotypes == geno) / len(case_genotypes)
                control_freq[geno] = np.sum(control_genotypes == geno) / len(control_genotypes)
            
            # Calculate variation as sum of absolute differences in frequencies
            variation_score = 0.0
            for geno in [0, 1, 2]:
                freq_diff = abs(case_freq[geno] - control_freq[geno])
                variation_score += freq_diff
            
            # Bonus for having variation in both groups
            case_has_variation = len(np.unique(case_genotypes)) > 1
            control_has_variation = len(np.unique(control_genotypes)) > 1
            
            if case_has_variation and control_has_variation:
                variation_score *= 1.5  # Boost score if both groups have genetic variation
            
            return variation_score
            
        except Exception:
            return 0.0
    
    def rank_and_select_top_genes(self, gene_variations_df, n_genes):
        """
        Rank genes by variation score and select top N
        """
        self.logger.info(f"🏆 RANKING GENES BY GENETIC VARIATION")
        
        # Sort by total variation score
        ranked_genes = gene_variations_df.sort_values('total_variation_score', ascending=False)
        
        # Select top N genes
        top_genes = ranked_genes.head(n_genes)
        
        self.logger.info(f"TOP {n_genes} GENES BY GENETIC VARIATION:")
        self.logger.info("-" * 100)
        self.logger.info(f"{'Rank':<4} {'Gene':<15} {'Chr':<5} {'SNPs':<5} {'Total Var':<12} {'Mean Var':<12} {'Max Var':<12}")
        self.logger.info("-" * 100)
        
        for i, (_, gene) in enumerate(top_genes.iterrows(), 1):
            self.logger.info(f"{i:<4} {gene['gene_name']:<15} {gene['chrom']:<5} "
                           f"{gene['n_snps_in_region']:<5} {gene['total_variation_score']:<12.3f} "
                           f"{gene['mean_variation_per_snp']:<12.3f} {gene['max_snp_variation']:<12.3f}")
        
        self.logger.info("-" * 100)
        self.logger.info(f"Selection summary:")
        self.logger.info(f"  Mean SNPs per selected gene: {top_genes['n_snps_in_region'].mean():.1f}")
        self.logger.info(f"  Mean variation score: {top_genes['total_variation_score'].mean():.3f}")
        self.logger.info(f"  Genes with SNPs: {(top_genes['n_snps_in_region'] > 0).sum()}")
        
        return top_genes
    
    def create_enformer_regions_for_selected_genes(self, selected_genes_df, gtf_file):
        """
        Create Enformer-compatible 196kb regions for selected genes
        """
        self.logger.info(f"Creating Enformer regions for {len(selected_genes_df)} selected genes...")
        
        enformer_regions = []
        
        for _, gene in selected_genes_df.iterrows():
            gene_name = gene['gene_name']
            chrom = gene['chrom']
            tss = gene['tss']
            
            # Create 196kb window centered on TSS
            half_window = SEQUENCE_LENGTH // 2
            enformer_start = max(1, tss - half_window)
            enformer_end = enformer_start + SEQUENCE_LENGTH - 1
            
            region = {
                'gene_name': gene_name,
                'region_id': gene_name,
                'chrom': chrom,
                'tss': tss,
                'start': enformer_start,
                'end': enformer_end,
                'region_type': 'variation_based',
                'n_snps_in_region': gene['n_snps_in_region'],
                'total_variation_score': gene['total_variation_score'],
                'mean_variation_per_snp': gene['mean_variation_per_snp'],
                'max_snp_variation': gene['max_snp_variation'],
                'variation_density': gene['variation_density'],
                'selection_method': 'genetic_variation_ranking'
            }
            
            enformer_regions.append(region)
        
        self.logger.info(f"Created {len(enformer_regions)} Enformer regions for top-variation genes")
        
        return enformer_regions
    
    def load_enformer_model(self):
        """Load Enformer model"""
        self.logger.info("Loading Enformer model...")
        
        try:
            model = from_pretrained('EleutherAI/enformer-official-rough')
            model.eval()
            
            if DEVICE == "cuda":
                model = model.cuda()
                self.logger.info("✅ Enformer model loaded on GPU")
            else:
                self.logger.info("✅ Enformer model loaded on CPU")
            
            return model
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load Enformer model: {e}")
            return None
    
    def generate_gene_expressions(self, plink_prefix, gene_regions, model):
        """
        Generate comprehensive gene expression predictions using Enformer
        """
        self.logger.info(f"🧬 GENERATING ENFORMER EXPRESSIONS")
        self.logger.info(f"Input: {plink_prefix}")
        self.logger.info(f"Genes: {len(gene_regions)} top-ranked by genetic variation")
        
        try:
            # Load PLINK data
            bim, fam, bed = read_plink(plink_prefix)
            bed_array = bed.compute()
            sample_ids = fam['iid'].tolist()
            
            self.logger.info(f"Processing {len(sample_ids)} samples × {len(gene_regions)} genes")
            
            # Initialize expression storage
            n_samples = len(sample_ids)
            n_genes = len(gene_regions)
            
            all_gene_expressions = []
            expression_feature_names = None
            
            # Process each top-variation gene with Enformer
            for gene_idx, region in enumerate(tqdm(gene_regions, desc="Enformer predictions")):
                gene_name = region['gene_name']
                chrom = region['chrom']
                start = region['start']
                end = region['end']
                variation_score = region.get('total_variation_score', 0)
                n_snps = region.get('n_snps_in_region', 0)
                
                self.logger.info(f"Gene {gene_idx+1}/{n_genes}: {gene_name} "
                               f"(Variation Score: {variation_score:.3f}, SNPs: {n_snps})")
                
                # Get reference sequence
                ref_seq = self.get_reference_sequence(chrom, start, end)
                if ref_seq is None:
                    self.logger.warning(f"⚠️  Skipping {gene_name} - no reference sequence")
                    if expression_feature_names is not None:
                        zero_expressions = np.zeros((n_samples, len(expression_feature_names)))
                        all_gene_expressions.append(zero_expressions)
                    continue
                
                # Get variants in Enformer region (196kb)
                region_variants, variant_indices = self.get_variants_in_region(bim, chrom, start, end)
                self.logger.info(f"   Found {len(region_variants)} variants in Enformer region")
                
                # Store expressions for this gene
                gene_sample_expressions = []
                
                # Process samples in batches
                for batch_start in range(0, n_samples, BATCH_SIZE):
                    batch_end = min(batch_start + BATCH_SIZE, n_samples)
                    batch_sequences = []
                    
                    # Create personalized sequences
                    for sample_idx in range(batch_start, batch_end):
                        # Get sample genotypes
                        if len(variant_indices) > 0:
                            valid_indices = [i for i in variant_indices if i < bed_array.shape[1]]
                            sample_genotypes = bed_array[sample_idx, valid_indices] if valid_indices else []
                        else:
                            sample_genotypes = []
                        
                        # Apply variants to reference
                        personalized_seq = self.apply_variants_to_sequence(
                            ref_seq, region_variants, sample_genotypes, start
                        )
                        
                        # Convert to tensor
                        seq_tensor = self.sequence_to_tensor(personalized_seq)
                        batch_sequences.append(seq_tensor)
                    
                    # Run Enformer prediction
                    if batch_sequences:
                        batch_tensor = torch.stack(batch_sequences).to(DEVICE)
                        
                        with torch.no_grad():
                            # 🔬 ENFORMER PREDICTION
                            outputs = model(batch_tensor)
                            human_preds = outputs['human']  # Shape: (batch_size, 896, 5313)
                            
                            # Extract expression features
                            batch_features, feature_names = self.extract_expression_features(human_preds.cpu().numpy())
                            
                            if expression_feature_names is None:
                                expression_feature_names = feature_names
                                self.logger.info(f"   Expression features: {len(feature_names)}")
                            
                            gene_sample_expressions.append(batch_features)
                        
                        if DEVICE == "cuda":
                            torch.cuda.empty_cache()
                
                # Combine batch results
                if gene_sample_expressions:
                    gene_expressions = np.concatenate(gene_sample_expressions, axis=0)
                    all_gene_expressions.append(gene_expressions)
                    
                    # Log statistics
                    primary_expr = gene_expressions[:, 0]
                    self.logger.info(f"   ✅ {gene_name}: range {primary_expr.min():.3f}-{primary_expr.max():.3f}")
                else:
                    if expression_feature_names is not None:
                        zero_expressions = np.zeros((n_samples, len(expression_feature_names)))
                        all_gene_expressions.append(zero_expressions)
            
            # Combine all expressions
            if all_gene_expressions:
                final_expressions = np.stack(all_gene_expressions, axis=1)
                self.logger.info(f"✅ Generated expression matrix: {final_expressions.shape}")
            else:
                self.logger.error("❌ No expressions generated!")
                return None, None, None
            
            return final_expressions, sample_ids, expression_feature_names
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate expressions: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def get_reference_sequence(self, chrom, start, end):
        """Get reference sequence from FASTA"""
        try:
            fasta = pysam.FastaFile(REFERENCE_FASTA)
            
            target_chrom = chrom
            if chrom not in fasta.references:
                alt_chrom = chrom.replace('chr', '') if chrom.startswith('chr') else f'chr{chrom}'
                if alt_chrom in fasta.references:
                    target_chrom = alt_chrom
                else:
                    fasta.close()
                    return None
            
            sequence = fasta.fetch(target_chrom, start-1, end)
            fasta.close()
            
            # Ensure exact length
            if len(sequence) < SEQUENCE_LENGTH:
                sequence = sequence + 'N' * (SEQUENCE_LENGTH - len(sequence))
            elif len(sequence) > SEQUENCE_LENGTH:
                sequence = sequence[:SEQUENCE_LENGTH]
            
            return sequence.upper()
            
        except Exception as e:
            self.logger.error(f"Error getting sequence: {e}")
            return None
    
    def get_variants_in_region(self, bim, chrom, start, end):
        """Get variants in genomic region"""
        bim_chrom = chrom.replace('chr', '') if chrom.startswith('chr') else chrom
        
        mask = (
            (bim['chrom'].astype(str) == bim_chrom) & 
            (bim['pos'] >= start) & 
            (bim['pos'] <= end)
        )
        
        region_variants = bim[mask].copy().reset_index()
        original_indices = region_variants['index'].values if len(region_variants) > 0 else []
        
        return region_variants, original_indices
    
    def apply_variants_to_sequence(self, ref_seq, variants_df, sample_genotypes, region_start):
        """Apply sample variants to reference sequence"""
        if len(variants_df) == 0 or len(sample_genotypes) == 0:
            return ref_seq
        
        seq_array = np.array(list(ref_seq))
        
        for idx, (_, variant) in enumerate(variants_df.iterrows()):
            if idx >= len(sample_genotypes):
                break
                
            genotype = sample_genotypes[idx]
            if np.isnan(genotype):
                continue
            
            pos_in_seq = variant['pos'] - region_start
            if not (0 <= pos_in_seq < len(seq_array)):
                continue
            
            # Simple SNPs only
            ref_allele = variant['a0']
            alt_allele = variant['a1']
            
            if (len(ref_allele) == 1 and len(alt_allele) == 1 and 
                ref_allele in 'ACGT' and alt_allele in 'ACGT'):
                
                if genotype == 1:  # Heterozygous
                    chosen_allele = np.random.choice([ref_allele, alt_allele])
                    seq_array[pos_in_seq] = chosen_allele
                elif genotype == 2:  # Homozygous alternate
                    seq_array[pos_in_seq] = alt_allele
        
        return ''.join(seq_array)
    
    def sequence_to_tensor(self, sequence):
        """Convert DNA sequence to tensor"""
        base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        indices = torch.tensor([base_to_idx.get(base, 4) for base in sequence])
        return seq_indices_to_one_hot(indices)
    
    def extract_expression_features(self, enformer_output):
        """Extract comprehensive expression features from Enformer output"""
        batch_size = enformer_output.shape[0]
        tss_bin = 448  # Center bin for TSS
        
        # Expression-relevant track groups
        track_groups = {
            'cage_tracks': list(range(0, 50)),
            'h3k4me3_tracks': list(range(100, 150)),
            'h3k27ac_tracks': list(range(200, 250)),
            'rna_seq_tracks': list(range(4800, 4850)),
            'pol2_tracks': list(range(500, 550)),
        }
        
        # Extract features
        expression_features = {}
        
        for group_name, tracks in track_groups.items():
            valid_tracks = [t for t in tracks if t < enformer_output.shape[2]]
            
            if len(valid_tracks) > 0:
                group_preds = enformer_output[:, tss_bin, valid_tracks]
                expression_features[f'{group_name}_mean'] = np.mean(group_preds, axis=1)
                expression_features[f'{group_name}_max'] = np.max(group_preds, axis=1)
        
        # Create primary expression score
        primary_score = np.zeros(batch_size)
        weights = {
            'cage_tracks_mean': 0.3,
            'rna_seq_tracks_mean': 0.3,
            'h3k4me3_tracks_mean': 0.2,
            'pol2_tracks_mean': 0.1,
            'h3k27ac_tracks_mean': 0.1
        }
        
        total_weight = 0
        for feature_name, weight in weights.items():
            if feature_name in expression_features:
                primary_score += expression_features[feature_name] * weight
                total_weight += weight
        
        if total_weight > 0:
            primary_score = primary_score / total_weight
        
        # Create final features
        final_features = [primary_score]
        feature_names = ['primary_expression_score']
        
        key_features = ['cage_tracks_mean', 'rna_seq_tracks_mean', 'h3k4me3_tracks_mean']
        for feat_name in key_features:
            if feat_name in expression_features:
                final_features.append(expression_features[feat_name])
                feature_names.append(feat_name)
        
        expression_matrix = np.column_stack(final_features)
        
        return expression_matrix, feature_names
    
    def save_gene_selection_results(self, enformer_regions, gene_variations_df, output_path, phenotype):
        """Save gene selection results"""
        self.logger.info("💾 Saving gene selection results...")
        
        # Save selected gene regions
        pd.DataFrame(enformer_regions).to_csv(output_path + "selected_gene_regions.csv", index=False)
        
        # Save all gene variations (for analysis)
        gene_variations_df.to_csv(output_path + "all_gene_variations.csv", index=False)
        
        # Save top genes summary
        top_genes = gene_variations_df.sort_values('total_variation_score', ascending=False).head(50)
        top_genes.to_csv(output_path + "top_50_variation_genes.csv", index=False)
        
        # Create selection summary
        summary = {
            'phenotype': phenotype,
            'total_genes_analyzed': len(gene_variations_df),
            'genes_selected': len(enformer_regions),
            'selection_method': 'genetic_variation_ranking',
            'mean_variation_score': float(gene_variations_df['total_variation_score'].mean()),
            'max_variation_score': float(gene_variations_df['total_variation_score'].max()),
            'genes_with_snps': int((gene_variations_df['n_snps_in_region'] > 0).sum())
        }
        
        pd.DataFrame([summary]).to_csv(output_path + "gene_selection_summary.csv", index=False)
        
        self.logger.info(f"✅ Gene selection results saved to: {output_path}")
    
    def save_dataset_results(self, dataset_name, expressions, sample_ids, gene_regions, output_path, feature_names):
        """Save comprehensive expression results"""
        self.logger.info(f"💾 SAVING {dataset_name.upper()} EXPRESSION RESULTS")
        
        n_samples, n_genes, n_features = expressions.shape
        gene_names = [region['gene_name'] for region in gene_regions]
        
        # Save primary expression matrix (main file for ML)
        primary_expressions = expressions[:, :, 0]
        primary_expr_df = pd.DataFrame(primary_expressions, index=sample_ids, columns=gene_names)
        primary_file = output_path + f"{dataset_name}_gene_expression.csv"
        primary_expr_df.to_csv(primary_file)
        
        # Save comprehensive expressions
        flat_expressions = expressions.reshape(n_samples, n_genes * n_features)
        flat_columns = [f"{gene}__{feat}" for gene in gene_names for feat in feature_names]
        comprehensive_df = pd.DataFrame(flat_expressions, index=sample_ids, columns=flat_columns)
        comprehensive_file = output_path + f"{dataset_name}_comprehensive_expressions.csv"
        comprehensive_df.to_csv(comprehensive_file)
        
        # Save raw predictions
        np.save(output_path + f"{dataset_name}_raw_predictions.npy", expressions)
        
        # Save metadata
        pd.DataFrame({'sample_id': sample_ids}).to_csv(output_path + f"{dataset_name}_samples.csv", index=False)
        
        self.logger.info(f"✅ Saved {dataset_name}: {n_samples} samples × {n_genes} genes × {n_features} features")
        
        return {
            'primary_file': primary_file,
            'comprehensive_file': comprehensive_file,
            'n_samples': n_samples,
            'n_genes': n_genes,
            'n_features': n_features
        }
    
    def generate_final_summary(self, results, gene_regions, output_path, phenotype):
        """Generate final pipeline summary"""
        self.logger.info("📋 Generating final summary...")
        
        # Save final gene metadata
        pd.DataFrame(gene_regions).to_csv(output_path + "final_selected_genes.csv", index=False)
        
        self.logger.info("\n" + "=" * 100)
        self.logger.info("🎉 COMPLETE ENFORMER PIPELINE FINISHED!")
        self.logger.info("=" * 100)
        
        self.logger.info(f"📁 Results saved to: {output_path}")
        self.logger.info(f"🎯 Selected {len(gene_regions)} genes with highest genetic variation")
        
        if results:
            self.logger.info(f"🔬 Generated Enformer expressions for {len(results)} datasets")
            for dataset_name, data in results.items():
                if 'expressions' in data:
                    shape = data['expressions'].shape
                    self.logger.info(f"   ✅ {dataset_name}: {shape[0]} samples × {shape[1]} genes × {shape[2]} features")
        
        self.logger.info("\n🧬 PIPELINE HIGHLIGHTS:")
        self.logger.info("   ✅ Variation-based gene selection (robust for small sample sizes)")
        self.logger.info("   ✅ Fixed chromosome format matching")
        self.logger.info("   ✅ Personalized genome sequences")
        self.logger.info("   ✅ Comprehensive Enformer expression prediction")
        self.logger.info("   ✅ Ready-to-use ML datasets")
        
        self.logger.info(f"\n🎯 GENES SELECTED BY GENETIC VARIATION")
        self.logger.info("   These genes showed the most genetic differences")
        self.logger.info("   between cases and controls in your training data")
        
        self.logger.info(f"\n📊 READY FOR MACHINE LEARNING:")
        self.logger.info("   📊 Use *_gene_expression.csv files for ML models")
        self.logger.info("   🔬 Train on train_gene_expression.csv")
        self.logger.info("   🧪 Validate on validation_gene_expression.csv")
        self.logger.info("   🎯 Test on test_gene_expression.csv")

def main():
    if len(sys.argv) < 3:
        print("Usage: python complete_enformer_pipeline.py phenotype fold [n_genes]")
        print("Example: python complete_enformer_pipeline.py migraine 0 20")
        print()
        print("🧬 Complete end-to-end Enformer pipeline:")
        print("  1. Rank genes by genetic variation between cases/controls")
        print("  2. Select top N genes with highest variation")
        print("  3. Generate Enformer expressions for all datasets")
        print("  4. Save ML-ready datasets")
        sys.exit(1)
    
    phenotype = sys.argv[1]
    fold = sys.argv[2]
    n_genes = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    
    # Run complete pipeline
    pipeline = CompleteEnformerPipeline()
    success = pipeline.run_complete_pipeline(phenotype, fold, n_genes)
    
    if success:
        print(f"\n🎉 SUCCESS! Complete Enformer pipeline finished!")
        print(f"📁 Results in: {phenotype}/Fold_{fold}/CompleteEnformerPipeline/")
        print(f"🎯 Generated expressions for {n_genes} top-variation genes")
        print(f"📊 Ready for machine learning analysis!")
    else:
        print(f"\n❌ Pipeline failed. Check logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()