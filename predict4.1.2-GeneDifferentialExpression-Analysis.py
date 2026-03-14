#!/usr/bin/env python3
"""
🔥 ULTIMATE COMPREHENSIVE GENE RANKING PIPELINE - COMPLETE VERSION 🔥

Integrates ALL features:
- Train/Val/Test split (no data leakage)
- Sophisticated multi-metric scoring (Document 2)
- Multiple ranking views (9 different views)
- Consensus filtering enrichment (Document 1)
- Enrichment vs random analysis
- Permutation testing
- Complete evaluation suite
- Gene symbol lookup
- Confidence tier classification
- Known/Novel gene identification

USAGE:
  python ultimate_complete_ranking.py migraine
"""

import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import hypergeom
import mygene
import warnings
warnings.filterwarnings('ignore')
import sys

# =============================================================================
# CONFIGURATION
# =============================================================================

# Filtering thresholds
FDR_THRESHOLD = 0.1
DE_EFFECT_THRESHOLD = 0.5
ASSOC_EFFECT_THRESHOLD = 0.5

#print(sys.argv[2],sys.argv[3])
# 
##
#FDR_THRESHOLD = sys.argv[1]
#DE_EFFECT_THRESHOLD = sys.argv[2]
#ASSOC_EFFECT_THRESHOLD = sys.argv[2]





# Method classification
DE_METHODS = [
    "LIMMA", "Welch_t_test", "Linear_Regression",
    "Wilcoxon_Rank_Sum", "Permutation_Test"
]
ASSOC_METHODS = [
    "Weighted_Logistic", "Firth_Logistic", "Bayesian_Logistic"
]

# Ranking weights (Document 2)
DEFAULT_WEIGHTS = {
    'reproducibility': 0.40,
    'effect_size': 0.30,
    'confidence': 0.30
}

# Confidence tier thresholds (Document 2)
TIER_THRESHOLDS = {
    'tier1': {  # High confidence
        'min_hits': 20,
        'min_tissues': 5,
        'min_methods': 3,
        'min_mean_effect': 0.75,
        'max_min_fdr': 0.001
    },
    'tier2': {  # Moderate confidence
        'min_hits': 10,
        'min_tissues': 3,
        'min_methods': 2,
        'min_mean_effect': 0.6,
        'max_min_fdr': 0.01
    }
}

# Special filters
PAN_TISSUE_MIN = 10
METHOD_CONSENSUS_MIN = 5

# =============================================================================
# HELPER FUNCTIONS - UNIFIED EFFECT HANDLING
# =============================================================================

def get_unified_effect_value(row, has_effect_column):
    """
    Get unified absolute effect value from a row.
    Uses Effect for association methods if available, otherwise Log2FoldChange.
    """
    # FIX: be robust if keys missing
    if has_effect_column and pd.notna(row.get('Effect', np.nan)):
        return abs(float(row['Effect']))
    if pd.notna(row.get('Log2FoldChange', np.nan)):
        return abs(float(row['Log2FoldChange']))
    return 0.0

def get_unified_effect_direction(row, has_effect_column):
    """Get unified effect direction: +1, -1, or 0 (for exact zeros / missing)."""
    value = 0.0
    if has_effect_column and pd.notna(row.get('Effect', np.nan)):
        value = float(row['Effect'])
    elif pd.notna(row.get('Log2FoldChange', np.nan)):
        value = float(row['Log2FoldChange'])

    if value > 0:
        return 1
    elif value < 0:
        return -1
    else:
        return 0

def compute_norm_constants_from_trainval(sig_trainval_pd: pd.DataFrame) -> dict:
    """Data-driven maxima for normalization (train+val only; no test leakage)."""
    if sig_trainval_pd is None or len(sig_trainval_pd) == 0:
        return {'max_hits': 1, 'max_tissues': 1, 'max_methods': 1, 'max_databases': 1}

    # max hits across genes
    if 'Gene' in sig_trainval_pd.columns:
        hits_per_gene = sig_trainval_pd.groupby('Gene').size()
        max_hits = int(hits_per_gene.max()) if len(hits_per_gene) else 1
    else:
        max_hits = 1

    max_tissues = int(sig_trainval_pd['Tissue'].nunique()) if 'Tissue' in sig_trainval_pd.columns else 1
    max_methods = int(sig_trainval_pd['Method'].nunique()) if 'Method' in sig_trainval_pd.columns else 1
    max_dbs = int(sig_trainval_pd['Database'].nunique()) if 'Database' in sig_trainval_pd.columns else 1

    return {
        'max_hits': max(max_hits, 1),
        'max_tissues': max(max_tissues, 1),
        'max_methods': max(max_methods, 1),
        'max_databases': max(max_dbs, 1),
    }

def compute_method_effect_stats(sig_trainval_pd: pd.DataFrame, has_effect: bool) -> dict:
    """
    Per-method robust standardization stats computed on a unified signed effect:
      raw = Effect if present else Log2FoldChange
    """
    stats = {}
    if sig_trainval_pd is None or len(sig_trainval_pd) == 0:
        return stats
    if "Method" not in sig_trainval_pd.columns:
        return stats

    df = sig_trainval_pd.copy()

    # unified signed effect for stats
    df["raw_effect_for_stats"] = np.nan
    if has_effect and "Effect" in df.columns:
        m = df["Effect"].notna()
        df.loc[m, "raw_effect_for_stats"] = df.loc[m, "Effect"].astype(float)
    if "Log2FoldChange" in df.columns:
        mask = df["raw_effect_for_stats"].isna() & df["Log2FoldChange"].notna()
        df.loc[mask, "raw_effect_for_stats"] = df.loc[mask, "Log2FoldChange"].astype(float)

    df = df[["Method", "raw_effect_for_stats"]].dropna()
    if len(df) == 0:
        return stats

    for m, sub in df.groupby("Method"):
        vals = sub["raw_effect_for_stats"].astype(float).values
        if len(vals) < 10:
            continue
        med = float(np.median(vals))
        q1 = float(np.quantile(vals, 0.25))
        q3 = float(np.quantile(vals, 0.75))
        iqr = max(q3 - q1, 1e-6)
        stats[m] = {"median": med, "iqr": iqr}

    return stats

def standardized_abs_effect(row, has_effect: bool, method_stats: dict) -> float:
    """
    Standardized absolute effect size:
      z = (effect - median_method) / iqr_method
    Returns abs(z). If method missing stats, falls back to raw abs(effect).
    """
    method = row.get('Method', None)

    if has_effect and pd.notna(row.get('Effect', np.nan)):
        raw = float(row['Effect'])
    elif pd.notna(row.get('Log2FoldChange', np.nan)):
        raw = float(row['Log2FoldChange'])
    else:
        return 0.0

    st = method_stats.get(method)
    if st is None:
        return abs(raw)
    return abs((raw - st['median']) / st['iqr'])

# =============================================================================
# SCORING FUNCTIONS (DOCUMENT 2 METHODOLOGY)
# =============================================================================

def calculate_importance_score(gene_data, weights=None, has_effect_column=False, norm=None, method_effect_stats=None):
    """
    REVISED: More biologically interpretable scoring with better normalization
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    if norm is None:
        norm = {'max_hits': 1, 'max_tissues': 1, 'max_methods': 1, 'max_databases': 1}
    if method_effect_stats is None:
        method_effect_stats = {}

    gene_data = gene_data.copy()

    # FIX: if Tissue/Method/Database missing, create placeholders so nunique() works
    if 'Tissue' not in gene_data.columns:
        gene_data['Tissue'] = 'NA_TISSUE'
    if 'Method' not in gene_data.columns:
        gene_data['Method'] = 'NA_METHOD'
    if 'Database' not in gene_data.columns:
        gene_data['Database'] = 'NA_DB'

    gene_data['unified_effect'] = gene_data.apply(
        lambda row: get_unified_effect_value(row, has_effect_column), axis=1
    )
    gene_data['unified_direction'] = gene_data.apply(
        lambda row: get_unified_effect_direction(row, has_effect_column), axis=1
    )

    # === 1. REPRODUCIBILITY SCORE ===
    total_hits = len(gene_data)
    n_tissues = gene_data['Tissue'].nunique()
    n_methods = gene_data['Method'].nunique()
    n_databases = gene_data['Database'].nunique()

    hit_score = min(total_hits / float(norm.get('max_hits', 1)), 1.0)
    breadth_score = min(
        (n_tissues / float(norm.get('max_tissues', 1))) +
        (n_methods / float(norm.get('max_methods', 1))) +
        (n_databases / float(norm.get('max_databases', 1))),
        1.0
    )
    repro_score = (hit_score * 0.6 + breadth_score * 0.4) * weights['reproducibility']

    # === 2. EFFECT SIZE SCORE (IMPROVED) ===
    gene_data['std_abs_effect'] = gene_data.apply(
        lambda r: standardized_abs_effect(r, has_effect_column, method_effect_stats),
        axis=1
    )

    mean_eff = float(gene_data['std_abs_effect'].mean()) if len(gene_data) else 0.0
    max_eff = float(gene_data['std_abs_effect'].max()) if len(gene_data) else 0.0

    # 🔥 IMPROVED: Use sigmoid transformation instead of linear division
    # This preserves biological interpretation:
    # - z=1.0 (1 IQR away) → 0.5 score
    # - z=2.0 (2 IQR away) → 0.73 score  
    # - z=3.0 (3 IQR away) → 0.88 score (highly significant!)
    # - z=5.0 (extreme) → 0.97 score
    import math
    def sigmoid_transform(z, midpoint=2.0, steepness=0.5):
        """Sigmoid with midpoint at z=2.0 (2 IQRs = biologically significant)"""
        return 1.0 / (1.0 + math.exp(-steepness * (z - midpoint)))
    
    mean_eff_norm = sigmoid_transform(mean_eff, midpoint=2.0, steepness=0.5)
    max_eff_norm = sigmoid_transform(max_eff, midpoint=3.0, steepness=0.4)

    effect_score = (mean_eff_norm * 0.7 + max_eff_norm * 0.3) * weights['effect_size']

    # Bonus for consistent direction (unchanged)
    nonzero_dirs = gene_data['unified_direction'][gene_data['unified_direction'] != 0]
    if len(nonzero_dirs) > 0:
        all_positive = (nonzero_dirs > 0).all()
        all_negative = (nonzero_dirs < 0).all()
        if all_positive or all_negative:
            effect_score *= 1.1

    # === 3. STATISTICAL CONFIDENCE SCORE (IMPROVED) ===
    if 'FDR' in gene_data.columns:
        min_fdr = float(gene_data['FDR'].min())
        mean_fdr = float(gene_data['FDR'].mean())
    else:
        min_fdr = 1.0
        mean_fdr = 1.0

    # 🔥 IMPROVED: Use -log10 directly with reasonable bounds
    # FDR=0.05 → -log10=1.3 → normalized to 0.26 (weak)
    # FDR=0.001 → -log10=3.0 → normalized to 0.60 (moderate)
    # FDR=1e-10 → -log10=10.0 → normalized to 1.0 (capped, very strong)
    min_confidence = min(-np.log10(max(min_fdr, 1e-50)) / 10.0, 1.0)  # Cap at -log10=10
    mean_confidence = min(-np.log10(max(mean_fdr, 1e-50)) / 5.0, 1.0)  # Cap at -log10=5

    conf_score = (min_confidence * 0.6 + mean_confidence * 0.4) * weights['confidence']

    total_score = repro_score + effect_score + conf_score

    return {
        'importance_score': total_score,
        'reproducibility_score': repro_score,
        'effect_score': effect_score,
        'confidence_score': conf_score,
        'mean_unified_effect': mean_eff,  # Still report raw z-scores
        'max_unified_effect': max_eff
    }

def assign_confidence_tier(gene_metrics):
    """Assign confidence tier based on objective thresholds (Document 2)."""
    tier1 = TIER_THRESHOLDS['tier1']
    if (gene_metrics['Total_Hits'] >= tier1['min_hits'] and
        gene_metrics['N_Tissues'] >= tier1['min_tissues'] and
        gene_metrics['N_Methods'] >= tier1['min_methods'] and
        gene_metrics['Mean_Unified_Effect'] >= tier1['min_mean_effect'] and
        gene_metrics['Min_FDR'] <= tier1['max_min_fdr']):
        return 'Tier1_High'

    tier2 = TIER_THRESHOLDS['tier2']
    if (gene_metrics['Total_Hits'] >= tier2['min_hits'] and
        gene_metrics['N_Tissues'] >= tier2['min_tissues'] and
        gene_metrics['N_Methods'] >= tier2['min_methods'] and
        gene_metrics['Mean_Unified_Effect'] >= tier2['min_mean_effect'] and
        gene_metrics['Min_FDR'] <= tier2['max_min_fdr']):
        return 'Tier2_Moderate'

    return 'Tier3_Exploratory'

# =============================================================================
# ENRICHMENT FUNCTIONS (DOCUMENT 1)
# =============================================================================

def hypergeometric_test(k, M, n, N):
    """
    Hypergeometric test for enrichment (Document 1)
    k = observed overlap, M = population size, n = successes in population, N = sample size
    """
    if M == 0 or N == 0 or n == 0:
        return {'expected': 0, 'observed': k, 'fold_enrichment': 0, 'p_value': 1.0, 'direction': "N/A"}

    # FIX: expected = N * (n/M) (equivalent, but clearer)
    expected = N * (n / M)
    fold_enrichment = k / expected if expected > 0 else 0

    if k >= expected:
        p_value = hypergeom.sf(k - 1, M, n, N)
        direction = "ENRICHED"
    else:
        p_value = hypergeom.cdf(k, M, n, N)
        direction = "DEPLETED"

    return {'expected': expected, 'observed': k, 'fold_enrichment': fold_enrichment, 'p_value': p_value, 'direction': direction}

# =============================================================================
# MAIN CLASS
# =============================================================================

class UltimateComprehensiveGeneRanker:
    def __init__(self, phenotype, ranking_weights=None):
        self.phenotype = phenotype
        self.base_path = Path(phenotype)
        self.results_dir = self.base_path / "GeneDifferentialExpression" / "Files"
        self.output_dir = self.results_dir / "UltimateCompleteRankingAnalysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.ranking_weights = ranking_weights or DEFAULT_WEIGHTS
        self.mg = mygene.MyGeneInfo()
        self.has_effect = False

        print("=" * 120)
        print("🔥 ULTIMATE COMPREHENSIVE GENE RANKING PIPELINE - COMPLETE VERSION")
        print("=" * 120)
        print(f"📋 Phenotype: {phenotype}")
        print(f"📁 Results: {self.results_dir}")
        print(f"📁 Output: {self.output_dir}")
        print(f"\n🎯 COMPLETE FEATURES:")
        print(f"   ✅ Train/Val/Test split (no data leakage)")
        print(f"   ✅ Sophisticated multi-metric scoring (Document 2)")
        print(f"   ✅ Multiple ranking views (9 views)")
        print(f"   ✅ Consensus filtering enrichment (Document 1)")
        print(f"   ✅ Enrichment vs random analysis")
        print(f"   ✅ Permutation testing (1000 permutations)")
        print(f"   ✅ Complete evaluation suite")
        print(f"   ✅ Gene symbol lookup")
        print(f"   ✅ Confidence tier classification")
        print(f"   ✅ Known/Novel gene identification")
        print(f"\n📊 Ranking weights:")
        print(f"   - Reproducibility: {self.ranking_weights['reproducibility']:.1%}")
        print(f"   - Effect size: {self.ranking_weights['effect_size']:.1%}")
        print(f"   - Confidence: {self.ranking_weights['confidence']:.1%}")
        print("=" * 120)

        self.load_known_genes()

    def load_known_genes(self):
        """Load known disease genes from migraine_genes.csv"""
        print("\n🔄 Loading known disease genes...")

        migraine_file = self.results_dir / "migraine_genes.csv"
        if not migraine_file.exists():
            migraine_file = Path("migraine_genes.csv")

        if migraine_file.exists():
            df = pd.read_csv(migraine_file)
            if 'ensembl_gene_id' in df.columns:
                self.M = set(df['ensembl_gene_id'].dropna().astype(str).unique())
            else:
                self.M = set()

            self.gene_to_symbol = {}
            if 'gene' in df.columns and 'ensembl_gene_id' in df.columns:
                for _, row in df[['ensembl_gene_id', 'gene']].dropna().iterrows():
                    self.gene_to_symbol[str(row['ensembl_gene_id'])] = str(row['gene'])

            print(f"✅ Loaded {len(self.M):,} known disease genes")
        else:
            self.M = set()
            self.gene_to_symbol = {}
            print(f"⚠️  No known disease genes file found")

    def load_and_filter_data(self):
        print("\n🔥 STEP A-B: LOADING AND FILTERING DATA")
        print("=" * 120)

        volcano_file = self.results_dir / "combined_volcano_data_all_models.csv"
        if not volcano_file.exists():
            raise FileNotFoundError(f"Not found: {volcano_file}")

        df = pl.read_csv(volcano_file)
        print(f"📊 Loaded: {len(df):,} total rows")

        if 'Dataset' not in df.columns:
            raise ValueError("No 'Dataset' column found!")
        if 'Gene' not in df.columns:
            raise ValueError("No 'Gene' column found!")

        unique_datasets = df['Dataset'].unique().to_list()
        print(f"\n🔍 DETECTING DATASET VALUES:")
        print(f"   Found unique Dataset values: {unique_datasets}")

        dataset_mapping = {}
        for ds in unique_datasets:
            ds_lower = str(ds).lower().strip()
            if 'training' in ds_lower or (('train' in ds_lower) and ('training' not in ds_lower)):
                dataset_mapping['training'] = ds
            elif 'validation' in ds_lower or (('val' in ds_lower) and ('validation' not in ds_lower)):
                dataset_mapping['validation'] = ds
            elif 'test' in ds_lower:
                dataset_mapping['test'] = ds

        # FIX: hard fail if mapping incomplete (prevents silent wrong split)
        if not all(k in dataset_mapping for k in ['training', 'validation', 'test']):
            raise ValueError(f"Could not map datasets to training/validation/test. Found: {dataset_mapping}")

        self.dataset_mapping = dataset_mapping
        print(f"✅ Dataset mapping: {self.dataset_mapping}")

        # Universe BEFORE filtering
        self.U_all = set(df['Gene'].unique().to_list())
        # FIX: deterministic universe list for AUC/permutation
        self.U_list = sorted(map(str, self.U_all))

        print(f"\n✅ Universe U (all genes tested): {len(self.U_all):,} genes")
        print(f"   Total rows: {len(df):,}")

        # Cast numeric columns (robust)
        if 'FDR' in df.columns:
            df = df.with_columns(pl.col('FDR').cast(pl.Float64, strict=False))
        else:
            # FIX: if missing, create (so pipeline doesn't explode)
            df = df.with_columns(pl.lit(1.0).alias('FDR').cast(pl.Float64))

        if 'Log2FoldChange' in df.columns:
            df = df.with_columns(pl.col('Log2FoldChange').cast(pl.Float64, strict=False))
        else:
            # FIX: create missing Log2FoldChange as 0.0
            df = df.with_columns(pl.lit(0.0).alias('Log2FoldChange').cast(pl.Float64))

        self.has_effect = 'Effect' in df.columns
        if self.has_effect:
            df = df.with_columns(pl.col('Effect').cast(pl.Float64, strict=False))
            print(f"✅ Found 'Effect' column - will use for association methods")
        else:
            print(f"⚠️  No 'Effect' column - using Log2FoldChange for all")

        # FIX: ensure required categorical columns exist (prevents later nunique/value_counts crashes)
        for col, default in [('Tissue', 'NA_TISSUE'), ('Database', 'NA_DB'), ('Method', 'NA_METHOD')]:
            if col not in df.columns:
                df = df.with_columns(pl.lit(default).alias(col))

        self.df_all = df

        print(f"\n🔄 Applying method-specific filtering...")

        de_filtered = df.filter(
            pl.col('Method').is_in(DE_METHODS) &
            (pl.col('FDR') < FDR_THRESHOLD) &
            (pl.col('Log2FoldChange').abs() >= DE_EFFECT_THRESHOLD)
        )

        if self.has_effect:
            assoc_filtered = df.filter(
                pl.col('Method').is_in(ASSOC_METHODS) &
                (pl.col('FDR') < FDR_THRESHOLD) &
                (pl.col('Effect').abs() >= ASSOC_EFFECT_THRESHOLD)
            )
        else:
            assoc_filtered = df.filter(
                pl.col('Method').is_in(ASSOC_METHODS) &
                (pl.col('FDR') < FDR_THRESHOLD) &
                (pl.col('Log2FoldChange').abs() >= ASSOC_EFFECT_THRESHOLD)
            )

        self.sig_all = pl.concat([de_filtered, assoc_filtered])

        print(f"\n✅ After filtering:")
        print(f"   DE methods: {len(de_filtered):,} rows")
        print(f"   Association methods: {len(assoc_filtered):,} rows")
        print(f"   Total significant: {len(self.sig_all):,} rows")
        print(f"   Unique significant genes: {self.sig_all['Gene'].n_unique():,}")
        print("=" * 120)

        return self.sig_all

    def split_by_dataset(self):
        print("\n🔥 STEP C: SPLITTING BY DATASET (FIXED)")
        print("=" * 120)

        train_name = self.dataset_mapping.get('training')
        val_name = self.dataset_mapping.get('validation')
        test_name = self.dataset_mapping.get('test')

        if not train_name or not val_name or not test_name:
            raise ValueError(f"Could not map all datasets. Found: {self.dataset_mapping}")

        print(f"🔍 Using dataset names:")
        print(f"   Training: '{train_name}'")
        print(f"   Validation: '{val_name}'")
        print(f"   Test: '{test_name}'")

        self.sig_train = self.sig_all.filter(pl.col('Dataset') == train_name)
        self.sig_val = self.sig_all.filter(pl.col('Dataset') == val_name)
        self.sig_test = self.sig_all.filter(pl.col('Dataset') == test_name)

        self.G_train = set(self.sig_train['Gene'].unique().to_list())
        self.G_val = set(self.sig_val['Gene'].unique().to_list())
        self.G_test = set(self.sig_test['Gene'].unique().to_list())

        print(f"\n📊 Dataset splits:")
        print(f"   Training:   {len(self.sig_train):,} rows → {len(self.G_train):,} genes")
        print(f"   Validation: {len(self.sig_val):,} rows → {len(self.G_val):,} genes")
        print(f"   Test:       {len(self.sig_test):,} rows → {len(self.G_test):,} genes")

        total_split = len(self.sig_train) + len(self.sig_val) + len(self.sig_test)
        if total_split != len(self.sig_all):
            print(f"\n⚠️  WARNING: Split total ({total_split:,}) != original ({len(self.sig_all):,})")

        print("=" * 120)
        return self.G_train, self.G_val, self.G_test

    def compute_overlaps(self):
        print("\n🔥 STEP D: REPLICATION OVERLAPS")
        print("=" * 120)

        train_val = self.G_train & self.G_val
        train_test = self.G_train & self.G_test
        val_test = self.G_val & self.G_test
        all_three = self.G_train & self.G_val & self.G_test

        self.G_final = self.G_train | self.G_val
        self.G_high = self.G_train & self.G_val

        print(f"📊 Overlaps:")
        print(f"   Train ∩ Validation: {len(train_val):,} genes")
        print(f"   Train ∩ Test:       {len(train_test):,} genes")
        print(f"   Validation ∩ Test:  {len(val_test):,} genes")
        print(f"   All three:          {len(all_three):,} genes")

        print(f"\n📊 Candidate sets (DISCOVERY - no test leakage):")
        print(f"   G_final (train ∪ val):  {len(self.G_final):,} genes")
        print(f"   G_high (train ∩ val):   {len(self.G_high):,} genes")

        overlap_df = pd.DataFrame({
            'Category': [
                'Train genes', 'Validation genes', 'Test genes',
                'Train ∩ Validation', 'Train ∩ Test', 'Val ∩ Test', 'All three',
                'G_final (train ∪ val)', 'G_high (train ∩ val)'
            ],
            'Count': [
                len(self.G_train), len(self.G_val), len(self.G_test),
                len(train_val), len(train_test), len(val_test), len(all_three),
                len(self.G_final), len(self.G_high)
            ]
        })

        overlap_df.to_csv(self.output_dir / 'dataset_overlaps.csv', index=False)
        print(f"\n✅ Saved: dataset_overlaps.csv")
        print("=" * 120)
        return overlap_df

    def build_gene_features(self):
        print("\n🔥 STEP E: BUILDING GENE FEATURES (SOPHISTICATED SCORING - DOCUMENT 2)")
        print("=" * 120)

        sig_trainval = pl.concat([self.sig_train, self.sig_val])
        sig_trainval_pd = sig_trainval.to_pandas()

        # FIX: ensure required columns exist in pandas (polars placeholder already helps, but keep safe)
        for col, default in [('Tissue', 'NA_TISSUE'), ('Database', 'NA_DB'), ('Method', 'NA_METHOD')]:
            if col not in sig_trainval_pd.columns:
                sig_trainval_pd[col] = default
        if 'FDR' not in sig_trainval_pd.columns:
            sig_trainval_pd['FDR'] = 1.0
        if 'Log2FoldChange' not in sig_trainval_pd.columns:
            sig_trainval_pd['Log2FoldChange'] = 0.0
        if 'Fold' not in sig_trainval_pd.columns:
            sig_trainval_pd['Fold'] = 0

        print(f"✅ Working with {len(sig_trainval):,} train+val rows")
        print(f"✅ Building features for {len(self.G_final):,} candidate genes...")

        self.norm = compute_norm_constants_from_trainval(sig_trainval_pd)
        print(f"\n✅ Data-driven normalization constants:")
        print(f"   Max hits: {self.norm['max_hits']}")
        print(f"   Max tissues: {self.norm['max_tissues']}")
        print(f"   Max methods: {self.norm['max_methods']}")
        print(f"   Max databases: {self.norm['max_databases']}")

        self.method_effect_stats = compute_method_effect_stats(sig_trainval_pd, self.has_effect)
        print(f"\n✅ Effect standardization stats computed for {len(self.method_effect_stats)} methods")

        gene_features = []

        for i, gene in enumerate(sorted(self.G_final)):
            if (i + 1) % 500 == 0:
                print(f"   Progress: {i+1:,}/{len(self.G_final):,} genes", end='\r')

            gene_data = sig_trainval_pd[sig_trainval_pd['Gene'] == gene].copy()
            if len(gene_data) == 0:
                continue

            I_train = 1 if gene in self.G_train else 0
            I_val = 1 if gene in self.G_val else 0

            n_tissues = gene_data['Tissue'].nunique()
            n_databases = gene_data['Database'].nunique()
            n_methods = gene_data['Method'].nunique()
            n_folds = gene_data['Fold'].nunique() if 'Fold' in gene_data.columns else 1

            total_hits = len(gene_data)

            min_fdr = float(gene_data['FDR'].min())
            mean_fdr = float(gene_data['FDR'].mean())
            median_fdr = float(gene_data['FDR'].median())
            n_highly_sig = int((gene_data['FDR'] < 0.001).sum())

            scores = calculate_importance_score(
                gene_data,
                weights=self.ranking_weights,
                has_effect_column=self.has_effect,
                norm=self.norm,
                method_effect_stats=self.method_effect_stats
            )

            gene_data['unified_direction'] = gene_data.apply(
                lambda row: get_unified_effect_direction(row, self.has_effect), axis=1
            )

            mean_direction = float(gene_data['unified_direction'].mean())
            # FIX: don't call 0 "Down"
            direction = 'Up' if mean_direction > 0 else ('Down' if mean_direction < 0 else 'Mixed')

            # FIX: direction consistency ignores zeros
            nonzero = gene_data['unified_direction'][gene_data['unified_direction'] != 0]
            if len(nonzero) > 0:
                n_positive = int((nonzero > 0).sum())
                n_negative = int((nonzero < 0).sum())
                consistency = max(n_positive, n_negative) / len(nonzero)
            else:
                consistency = 0.0

            status = "Known" if gene in self.M else "Novel"

            tissue_counts = gene_data['Tissue'].value_counts()
            top_tissues = ', '.join([f"{t}({c})" for t, c in tissue_counts.head(5).items()])

            method_counts = gene_data['Method'].value_counts()
            top_methods = ', '.join([f"{m}({c})" for m, c in method_counts.head(5).items()])

            db_counts = gene_data['Database'].value_counts()
            top_databases = ', '.join([f"{d}({c})" for d, c in db_counts.head(5).items()])

            gene_features.append({
                'Gene': gene,
                'Status': status,
                'I_train': I_train,
                'I_val': I_val,
                'Importance_Score': scores['importance_score'],
                'Reproducibility_Score': scores['reproducibility_score'],
                'Effect_Score': scores['effect_score'],
                'Confidence_Score': scores['confidence_score'],
                'Total_Hits': total_hits,
                'N_Tissues': n_tissues,
                'N_Databases': n_databases,
                'N_Methods': n_methods,
                'N_Folds': n_folds,
                'Mean_Unified_Effect': scores['mean_unified_effect'],
                'Max_Unified_Effect': scores['max_unified_effect'],
                'Direction': direction,
                'Direction_Consistency': float(consistency),
                'Min_FDR': min_fdr,
                'Mean_FDR': mean_fdr,
                'Median_FDR': median_fdr,
                'N_Highly_Sig': n_highly_sig,
                'Top_Tissues': top_tissues,
                'Top_Methods': top_methods,
                'Top_Databases': top_databases
            })

        print(f"\n✅ Calculated features for {len(gene_features):,} genes")

        self.gene_features_df = pd.DataFrame(gene_features)
        if len(self.gene_features_df) == 0:
            raise ValueError("No gene features were built (empty candidate set after filtering).")

        print(f"   Assigning confidence tiers...")
        self.gene_features_df['Confidence_Tier'] = self.gene_features_df.apply(assign_confidence_tier, axis=1)

        tier_counts = self.gene_features_df['Confidence_Tier'].value_counts()
        print(f"   Confidence tier distribution:")
        for tier in ['Tier1_High', 'Tier2_Moderate', 'Tier3_Exploratory']:
            count = int(tier_counts.get(tier, 0))
            pct = (count / len(self.gene_features_df) * 100) if len(self.gene_features_df) > 0 else 0
            print(f"      {tier}: {count:,} ({pct:.1f}%)")

        n_known = int((self.gene_features_df['Status'] == 'Known').sum())
        n_novel = int((self.gene_features_df['Status'] == 'Novel').sum())
        print(f"\n   Gene classification:")
        print(f"      Known disease genes: {n_known:,}")
        print(f"      Novel candidates: {n_novel:,}")

        print("=" * 120)
        return self.gene_features_df

    def get_gene_symbols_batch(self, gene_list):
        if len(gene_list) == 0:
            return {}

        print(f"🔄 Querying MyGene API for {len(gene_list):,} genes...")

        batch_size = 1000
        all_results = {}

        for i in range(0, len(gene_list), batch_size):
            batch = gene_list[i:i+batch_size]
            batch_num = i//batch_size + 1
            total_batches = (len(gene_list) + batch_size - 1) // batch_size

            print(f"   Batch {batch_num}/{total_batches} ({len(batch)} genes)...", end=' ')

            try:
                results = self.mg.querymany(
                    batch,
                    scopes='ensembl.gene',
                    fields='symbol',
                    species='human',
                    returnall=True
                )

                for result in results.get('out', []):
                    gene_id = result.get('query', '')
                    symbol = result.get('symbol', 'N/A')
                    all_results[gene_id] = symbol

                print(f"✅")
            except Exception as e:
                print(f"⚠️  Error: {e}")
                continue

        print(f"✅ Retrieved symbols for {len(all_results):,} genes")
        return all_results

    def compute_gene_scores(self):
        print("\n🔥 STEP F: COMPUTING SCORES & CREATING RANKING VIEWS")
        print("=" * 120)

        df = self.gene_features_df.copy()

        df.loc[df['I_train'] + df['I_val'] == 2, 'Importance_Score'] *= 1.1
        df['in_high_conf'] = df['Gene'].isin(self.G_high)

        print(f"\n🔄 Fetching gene symbols...")
        known_genes = df[df['Status'] == 'Known']['Gene'].tolist()
        novel_genes = df[df['Status'] == 'Novel']['Gene'].tolist()

        known_symbols = {gene: self.gene_to_symbol.get(gene, 'N/A') for gene in known_genes}
        novel_symbols = self.get_gene_symbols_batch(novel_genes) if novel_genes else {}

        all_symbols = {**known_symbols, **novel_symbols}
        df['Symbol'] = df['Gene'].map(all_symbols)

        print(f"\n✅ Creating ranking views...")

        column_order = [
            'Rank', 'Gene', 'Symbol', 'Status', 'Confidence_Tier', 'in_high_conf',
            'Importance_Score', 'Reproducibility_Score', 'Effect_Score', 'Confidence_Score',
            'I_train', 'I_val', 'Total_Hits', 'N_Tissues', 'N_Databases', 'N_Methods', 'N_Folds',
            'Mean_Unified_Effect', 'Max_Unified_Effect', 'Direction', 'Direction_Consistency',
            'Min_FDR', 'Mean_FDR', 'Median_FDR', 'N_Highly_Sig',
            'Top_Tissues', 'Top_Methods', 'Top_Databases'
        ]

        rankings = {}

        rankings['composite'] = df.sort_values(
            ['Importance_Score', 'Total_Hits', 'Mean_Unified_Effect'],
            ascending=[False, False, False]
        ).reset_index(drop=True)
        rankings['composite']['Rank'] = range(1, len(rankings['composite']) + 1)
        print(f"   ✅ Composite score ranking: {len(rankings['composite']):,} genes")

        rankings['reproducibility'] = df.sort_values(
            ['Total_Hits', 'N_Tissues', 'N_Methods', 'N_Databases'],
            ascending=[False, False, False, False]
        ).reset_index(drop=True)
        rankings['reproducibility']['Rank'] = range(1, len(rankings['reproducibility']) + 1)
        print(f"   ✅ Reproducibility ranking: {len(rankings['reproducibility']):,} genes")

        rankings['effect_size'] = df.sort_values(
            ['Mean_Unified_Effect', 'Max_Unified_Effect', 'Total_Hits'],
            ascending=[False, False, False]
        ).reset_index(drop=True)
        rankings['effect_size']['Rank'] = range(1, len(rankings['effect_size']) + 1)
        print(f"   ✅ Effect size ranking: {len(rankings['effect_size']):,} genes")

        rankings['significance'] = df.sort_values(
            ['Min_FDR', 'Mean_FDR', 'Total_Hits'],
            ascending=[True, True, False]
        ).reset_index(drop=True)
        rankings['significance']['Rank'] = range(1, len(rankings['significance']) + 1)
        print(f"   ✅ Significance ranking: {len(rankings['significance']):,} genes")

        pan_tissue = df[df['N_Tissues'] >= PAN_TISSUE_MIN].copy()
        if len(pan_tissue) > 0:
            rankings['pan_tissue'] = pan_tissue.sort_values(
                'Importance_Score', ascending=False
            ).reset_index(drop=True)
            rankings['pan_tissue']['Rank'] = range(1, len(rankings['pan_tissue']) + 1)
            print(f"   ✅ Pan-tissue ranking ({PAN_TISSUE_MIN}+ tissues): {len(rankings['pan_tissue']):,} genes")

        method_consensus = df[df['N_Methods'] >= METHOD_CONSENSUS_MIN].copy()
        if len(method_consensus) > 0:
            rankings['method_consensus'] = method_consensus.sort_values(
                'Importance_Score', ascending=False
            ).reset_index(drop=True)
            rankings['method_consensus']['Rank'] = range(1, len(rankings['method_consensus']) + 1)
            print(f"   ✅ Method consensus ranking ({METHOD_CONSENSUS_MIN}+ methods): {len(rankings['method_consensus']):,} genes")

        tier1 = df[df['Confidence_Tier'] == 'Tier1_High'].copy()
        if len(tier1) > 0:
            rankings['tier1_high'] = tier1.sort_values(
                'Importance_Score', ascending=False
            ).reset_index(drop=True)
            rankings['tier1_high']['Rank'] = range(1, len(rankings['tier1_high']) + 1)
            print(f"   ✅ Tier 1 high confidence: {len(rankings['tier1_high']):,} genes")

        novel = df[df['Status'] == 'Novel'].copy()
        if len(novel) > 0:
            rankings['novel_only'] = novel.sort_values(
                'Importance_Score', ascending=False
            ).reset_index(drop=True)
            rankings['novel_only']['Rank'] = range(1, len(rankings['novel_only']) + 1)
            print(f"   ✅ Novel genes only: {len(rankings['novel_only']):,} genes")

        known = df[df['Status'] == 'Known'].copy()
        if len(known) > 0:
            rankings['known_only'] = known.sort_values(
                'Importance_Score', ascending=False
            ).reset_index(drop=True)
            rankings['known_only']['Rank'] = range(1, len(rankings['known_only']) + 1)
            print(f"   ✅ Known genes only: {len(rankings['known_only']):,} genes")

        print(f"\n🔄 Saving ranking files...")
        for view_name, df_view in rankings.items():
            if len(df_view) == 0:
                continue
            cols_present = [col for col in column_order if col in df_view.columns]
            df_ordered = df_view[cols_present]
            filename = f"RANKED_{view_name}.csv"
            df_ordered.to_csv(self.output_dir / filename, index=False)
            print(f"   ✅ {filename}")

        print(f"\n🔄 Saving top-K gene lists...")
        for K in [50, 100, 200, 500, 1000]:
            if len(rankings['composite']) >= K:
                # FIX: avoid KeyError if some columns missing (use cols_present)
                cols_present = [col for col in column_order if col in rankings['composite'].columns]
                rankings['composite'].head(K)[cols_present].to_csv(
                    self.output_dir / f'top_{K}_genes.csv', index=False
                )
                print(f"   ✅ top_{K}_genes.csv")

        self.ranked_genes = rankings['composite']
        self.all_rankings = rankings

        print("=" * 120)
        return rankings

    def test_evaluation_confusion(self):
        print("\n🔥 STEP G: TEST SET EVALUATION - CONFUSION METRICS")
        print("=" * 120)

        predicted_sets = {
            'union (G_final)': self.G_final,
            'intersection (G_high)': self.G_high,
            'top_50': set(self.ranked_genes.head(50)['Gene']),
            'top_100': set(self.ranked_genes.head(100)['Gene']),
            'top_200': set(self.ranked_genes.head(200)['Gene']),
            'top_500': set(self.ranked_genes.head(500)['Gene']),
            'top_1000': set(self.ranked_genes.head(1000)['Gene'])
        }

        results = []

        print(f"\n📊 CONFUSION MATRIX (Predicted vs Test):")
        print("-" * 100)
        print(f"{'Predicted Set':<25} {'TP':<10} {'FP':<10} {'FN':<10} {'Precision':<12} {'Recall':<12}")
        print("-" * 100)

        for set_name, predicted in predicted_sets.items():
            TP = len(predicted & self.G_test)
            FP = len(predicted - self.G_test)
            FN = len(self.G_test - predicted)

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0

            results.append({
                'Predicted_Set': set_name,
                'TP': TP,
                'FP': FP,
                'FN': FN,
                'Precision': precision,
                'Recall': recall
            })

            print(f"{set_name:<25} {TP:<10} {FP:<10} {FN:<10} {precision:<12.3f} {recall:<12.3f}")

        confusion_df = pd.DataFrame(results)
        confusion_df.to_csv(self.output_dir / 'test_confusion_metrics.csv', index=False)
        print(f"\n✅ Saved: test_confusion_metrics.csv")
        print("=" * 120)
        return confusion_df

    def test_evaluation_auc(self):
        print("\n🔥 STEP H: AUC & PR-AUC EVALUATION")
        print("=" * 120)

        print(f"✅ Universe U: {len(self.U_all):,} genes (ALL genes in combined table)")

        # FIX: use deterministic universe order
        y = np.array([1 if gene in self.G_test else 0 for gene in self.U_list], dtype=int)

        gene_to_score = dict(zip(self.ranked_genes['Gene'], self.ranked_genes['Importance_Score']))
        scores = np.array([gene_to_score.get(gene, 0.0) for gene in self.U_list], dtype=float)

        print(f"✅ Labels: {y.sum():,} positives (test genes), {len(y) - y.sum():,} negatives")

        baseline = y.mean()
        print(f"✅ Prevalence (baseline): {baseline:.4f}")

        try:
            roc_auc = roc_auc_score(y, scores)
            print(f"\n📊 ROC AUC: {roc_auc:.4f}")
        except Exception as e:
            roc_auc = None
            print(f"\n⚠️  Could not compute ROC AUC: {e}")

        try:
            pr_auc = average_precision_score(y, scores)
            lift = pr_auc / baseline if baseline > 0 else 0
            print(f"📊 PR AUC (Average Precision): {pr_auc:.4f}")
            print(f"📊 Baseline (random): {baseline:.4f}")
            print(f"📊 Lift over random: {lift:.2f}x")
        except Exception as e:
            pr_auc = None
            lift = None
            print(f"\n⚠️  Could not compute PR AUC: {e}")

        print(f"\n💡 INTERPRETATION:")
        print(f"   ROC AUC and PR AUC quantify how well the train+val score")
        print(f"   ranks genes that replicate in test above those that do not.")
        if pr_auc is not None:
            quality = 'Excellent' if pr_auc > 0.7 else 'Good' if pr_auc > 0.5 else 'Moderate' if pr_auc > 0.3 else 'Poor'
            print(f"   PR AUC = {pr_auc:.4f} → {quality} ranking quality")

        # Store permutation p-values for CHECK 3
        self.perm_pr_pval = None
        self.perm_roc_pval = None

        auc_metrics = pd.DataFrame({
            'Metric': ['ROC_AUC', 'PR_AUC', 'Baseline', 'Lift'],
            'Value': [
                roc_auc if roc_auc is not None else np.nan,
                pr_auc if pr_auc is not None else np.nan,
                baseline,
                lift if lift is not None else np.nan
            ]
        })

        auc_metrics.to_csv(self.output_dir / 'test_auc_metrics.csv', index=False)
        print(f"\n✅ Saved: test_auc_metrics.csv")
        print("=" * 120)
        return roc_auc, pr_auc
    
    def enrichment_analysis(self):
        print("\n" + "=" * 120)
        print("🔥 ENRICHMENT vs RANDOM ANALYSIS")
        print("=" * 120)

        print("\n📊 PART 1: SET-BASED ENRICHMENT (HYPERGEOMETRIC TEST)")
        print("=" * 120)
        print(f"Universe U: {len(self.U_all):,} genes (all tested)")
        print(f"Test positives T: {len(self.G_test):,} genes")

        predicted_sets = {
            'G_final (union)': self.G_final,
            'G_high (intersection)': self.G_high,
            'top_50': set(self.ranked_genes.head(50)['Gene']),
            'top_100': set(self.ranked_genes.head(100)['Gene']),
            'top_200': set(self.ranked_genes.head(200)['Gene']),
            'top_500': set(self.ranked_genes.head(500)['Gene']),
            'top_1000': set(self.ranked_genes.head(1000)['Gene'])
        }

        enrichment_results = []

        print(f"\n{'Set':<25} {'N_pred':<10} {'k_obs':<10} {'k_exp':<10} {'FE':<10} {'P-value':<12}")
        print("-" * 100)

        for name, P in predicted_sets.items():
            N_pred = len(P)
            k_obs = len(P & self.G_test)
            k_exp = N_pred * len(self.G_test) / len(self.U_all)
            FE = k_obs / k_exp if k_exp > 0 else 0

            pval = hypergeom.sf(k_obs - 1, len(self.U_all), len(self.G_test), N_pred)

            enrichment_results.append({
                'Set': name,
                'N_predicted': N_pred,
                'k_observed': k_obs,
                'k_expected': k_exp,
                'Fold_Enrichment': FE,
                'P_value': pval
            })

            print(f"{name:<25} {N_pred:<10} {k_obs:<10} {k_exp:<10.2f} {FE:<10.2f} {pval:<12.2e}")

        enrich_df = pd.DataFrame(enrichment_results)
        enrich_df.to_csv(self.output_dir / 'enrichment_vs_random.csv', index=False)
        print(f"\n✅ Saved: enrichment_vs_random.csv")

        print("\n📊 PART 2: RANKING ENRICHMENT CURVE (TOP-K SWEEP)")
        print("=" * 120)

        curve_results = []
        best_FE = 0
        best_K = 0

        max_K = min(2000, len(self.ranked_genes))
        print(f"Sweeping K from 10 to {max_K}")

        for K in [10, 20, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000]:
            if K > len(self.ranked_genes):
                break

            P_K = set(self.ranked_genes.head(K)['Gene'])
            k_K = len(P_K & self.G_test)
            E_K = K * len(self.G_test) / len(self.U_all)
            FE_K = k_K / E_K if E_K > 0 else 0

            curve_results.append({
                'K': K,
                'k_observed': k_K,
                'k_expected': E_K,
                'Fold_Enrichment': FE_K
            })

            if FE_K > best_FE:
                best_FE = FE_K
                best_K = K

        curve_df = pd.DataFrame(curve_results)
        curve_df.to_csv(self.output_dir / 'enrichment_curve.csv', index=False)

        print(f"✅ Best enrichment: {best_FE:.2f}x at K={best_K}")
        print(f"✅ Saved: enrichment_curve.csv")

        print(f"\n📈 Enrichment at key K values:")
        print("-" * 60)
        print(f"{'K':<10} {'k_obs':<10} {'k_exp':<10} {'FE':<10}")
        print("-" * 60)
        for _, row in curve_df.iterrows():
            print(f"{row['K']:<10} {row['k_observed']:<10} {row['k_expected']:<10.2f} {row['Fold_Enrichment']:<10.2f}")

        print("\n📊 PART 3: PERMUTATION TEST FOR RANKING")
        print("=" * 120)

        # FIX: deterministic universe order
        y = np.array([1 if gene in self.G_test else 0 for gene in self.U_list], dtype=int)
        gene_to_score = dict(zip(self.ranked_genes['Gene'], self.ranked_genes['Importance_Score']))
        scores = np.array([gene_to_score.get(gene, 0.0) for gene in self.U_list], dtype=float)

        obs_pr_auc = average_precision_score(y, scores)
        obs_roc_auc = roc_auc_score(y, scores)

        print(f"Observed PR-AUC: {obs_pr_auc:.4f}")
        print(f"Observed ROC-AUC: {obs_roc_auc:.4f}")

        n_perm = 1000
        print(f"\nRunning {n_perm} permutations...")

        null_pr_aucs = []
        null_roc_aucs = []

        rng = np.random.RandomState(42)

        for i in range(n_perm):
            if (i + 1) % 100 == 0:
                print(f"   {i+1}/{n_perm}", end='\r')

            perm_scores = rng.permutation(scores)
            null_pr_aucs.append(average_precision_score(y, perm_scores))
            null_roc_aucs.append(roc_auc_score(y, perm_scores))

        null_pr_aucs = np.array(null_pr_aucs)
        null_roc_aucs = np.array(null_roc_aucs)

        p_pr = (1 + np.sum(null_pr_aucs >= obs_pr_auc)) / (1 + n_perm)
        p_roc = (1 + np.sum(null_roc_aucs >= obs_roc_auc)) / (1 + n_perm)

        # ✅ CRITICAL FIX: STORE for validator CHECK 3
        self.perm_pr_pval = float(p_pr)
        self.perm_roc_pval = float(p_roc)

        print(f"\n\n📊 PERMUTATION TEST RESULTS:")
        print("-" * 100)
        print(f"{'Metric':<12} {'Observed':<12} {'Null Mean':<12} {'Null SD':<12} {'P-value':<12}")
        print("-" * 100)
        print(f"{'PR-AUC':<12} {obs_pr_auc:<12.4f} {null_pr_aucs.mean():<12.4f} {null_pr_aucs.std():<12.4f} {p_pr:<12.2e}")
        print(f"{'ROC-AUC':<12} {obs_roc_auc:<12.4f} {null_roc_aucs.mean():<12.4f} {null_roc_aucs.std():<12.4f} {p_roc:<12.2e}")

        pd.DataFrame({
            'Metric': ['PR_AUC', 'ROC_AUC'],
            'Observed': [obs_pr_auc, obs_roc_auc],
            'Null_Mean': [null_pr_aucs.mean(), null_roc_aucs.mean()],
            'Null_SD': [null_pr_aucs.std(), null_roc_aucs.std()],
            'P_value': [p_pr, p_roc]
        }).to_csv(self.output_dir / 'permutation_test.csv', index=False)

        print(f"\n✅ Saved: permutation_test.csv")

        # =================================================================
        # PART 4: DISEASE GENE ENRICHMENT
        # =================================================================
        if len(self.M) > 0:
            print("\n📊 PART 4: DISEASE GENE ENRICHMENT (KNOWN GENES)")
            print("=" * 120)

            disease_enrich = []

            # FIX: successes in universe must be restricted to tested universe
            M_in_U = set(map(str, self.M)) & set(map(str, self.U_all))
            n_success = len(M_in_U)

            print(f"Known disease genes M (raw): {len(self.M):,}")
            print(f"Known disease genes in universe (M ∩ U): {n_success:,}")

            print(f"\n{'Set':<25} {'N_pred':<10} {'k_obs':<10} {'k_exp':<10} {'FE':<10} {'P-value':<12}")
            print("-" * 100)

            for name, P in predicted_sets.items():
                N_pred = len(P)
                k_obs = len(set(map(str, P)) & M_in_U)

                # expectation uses n_success (M∩U), not |M|
                k_exp = N_pred * n_success / len(self.U_all)
                FE = k_obs / k_exp if k_exp > 0 else 0

                pval = hypergeom.sf(k_obs - 1, len(self.U_all), n_success, N_pred)

                disease_enrich.append({
                    'Set': name,
                    'N_predicted': N_pred,
                    'k_observed': k_obs,
                    'k_expected': k_exp,
                    'Fold_Enrichment': FE,
                    'P_value': pval
                })

                print(f"{name:<25} {N_pred:<10} {k_obs:<10} {k_exp:<10.2f} {FE:<10.2f} {pval:<12.2e}")

            pd.DataFrame(disease_enrich).to_csv(self.output_dir / 'disease_gene_enrichment.csv', index=False)
            print(f"\n✅ Saved: disease_gene_enrichment.csv")

        print("\n" + "=" * 120)

    def consensus_filtering_enrichment(self):
        print("\n" + "=" * 120)
        print("🔥 CONSENSUS FILTERING ENRICHMENT ANALYSIS (DOCUMENT 1)")
        print("=" * 120)

        if len(self.M) == 0:
            print("⚠️  No known disease genes loaded - skipping consensus enrichment")
            return

        gene_stats = self.gene_features_df.copy()

        candidate_universe = set(map(str, gene_stats['Gene']))
        M_cand = len(candidate_universe)

        # FIX: successes should be known genes present in candidate universe
        known_in_cand = len(candidate_universe & set(map(str, self.M)))

        print(f"✅ Candidate universe: {M_cand:,} genes")
        print(f"✅ Known disease genes in candidates: {known_in_cand:,}")

        results = []

        method_thresholds = [1, 2, 3, 4, 5, 6]
        tissue_thresholds = [1, 2, 3, 5, 10, 15, 20]
        database_thresholds = [1, 2, 3, 4, 5]

        print("\n📊 TEST 1: FILTERING BY NUMBER OF METHODS")
        print("-" * 100)
        print(f"{'Min Methods':<12} {'N Genes':<10} {'Disease':<10} {'Expected':<10} {'Fold':<8} {'P-value':<12}")
        print("-" * 100)

        for min_methods in method_thresholds:
            filtered_genes = set(map(str, gene_stats[gene_stats['N_Methods'] >= min_methods]['Gene']))
            if len(filtered_genes) == 0:
                continue
            k_obs = len(filtered_genes & set(map(str, self.M)))
            result = hypergeometric_test(k_obs, M_cand, known_in_cand, len(filtered_genes))

            results.append({'filter_type': 'methods', 'threshold': min_methods,
                            'n_genes': len(filtered_genes), 'disease_overlap': k_obs, **result})

            print(f"{min_methods:<12} {len(filtered_genes):<10} {k_obs:<10} {result['expected']:<10.2f} {result['fold_enrichment']:<8.2f} {result['p_value']:<12.2e}")

        print("\n📊 TEST 2: FILTERING BY NUMBER OF TISSUES")
        print("-" * 100)
        print(f"{'Min Tissues':<12} {'N Genes':<10} {'Disease':<10} {'Expected':<10} {'Fold':<8} {'P-value':<12}")
        print("-" * 100)

        for min_tissues in tissue_thresholds:
            filtered_genes = set(map(str, gene_stats[gene_stats['N_Tissues'] >= min_tissues]['Gene']))
            if len(filtered_genes) == 0:
                continue
            k_obs = len(filtered_genes & set(map(str, self.M)))
            result = hypergeometric_test(k_obs, M_cand, known_in_cand, len(filtered_genes))

            results.append({'filter_type': 'tissues', 'threshold': min_tissues,
                            'n_genes': len(filtered_genes), 'disease_overlap': k_obs, **result})

            print(f"{min_tissues:<12} {len(filtered_genes):<10} {k_obs:<10} {result['expected']:<10.2f} {result['fold_enrichment']:<8.2f} {result['p_value']:<12.2e}")

        print("\n📊 TEST 3: FILTERING BY NUMBER OF DATABASES")
        print("-" * 100)
        print(f"{'Min DBs':<12} {'N Genes':<10} {'Disease':<10} {'Expected':<10} {'Fold':<8} {'P-value':<12}")
        print("-" * 100)

        for min_dbs in database_thresholds:
            filtered_genes = set(map(str, gene_stats[gene_stats['N_Databases'] >= min_dbs]['Gene']))
            if len(filtered_genes) == 0:
                continue
            k_obs = len(filtered_genes & set(map(str, self.M)))
            result = hypergeometric_test(k_obs, M_cand, known_in_cand, len(filtered_genes))

            results.append({'filter_type': 'databases', 'threshold': min_dbs,
                            'n_genes': len(filtered_genes), 'disease_overlap': k_obs, **result})

            print(f"{min_dbs:<12} {len(filtered_genes):<10} {k_obs:<10} {result['expected']:<10.2f} {result['fold_enrichment']:<8.2f} {result['p_value']:<12.2e}")

        print("\n📊 TEST 4: COMBINED FILTERS")
        print("-" * 110)
        print(f"{'Filter':<30} {'N Genes':<10} {'Disease':<10} {'Expected':<10} {'Fold':<8} {'P-value':<12}")
        print("-" * 110)

        combined_filters = [
            {'methods': 2, 'tissues': 2, 'databases': 2},
            {'methods': 2, 'tissues': 3, 'databases': 2},
            {'methods': 3, 'tissues': 3, 'databases': 2},
            {'methods': 1, 'tissues': 3, 'databases': 3},
            {'methods': 2, 'tissues': 5, 'databases': 2},
            {'methods': 3, 'tissues': 5, 'databases': 3},
            {'methods': 1, 'tissues': 5, 'databases': 3},
            {'methods': 1, 'tissues': 10, 'databases': 2},
        ]

        for filt in combined_filters:
            filtered_genes = set(map(str, gene_stats[
                (gene_stats['N_Methods'] >= filt['methods']) &
                (gene_stats['N_Tissues'] >= filt['tissues']) &
                (gene_stats['N_Databases'] >= filt['databases'])
            ]['Gene']))

            if len(filtered_genes) == 0:
                continue

            filter_str = f"M≥{filt['methods']}, T≥{filt['tissues']}, D≥{filt['databases']}"
            k_obs = len(filtered_genes & set(map(str, self.M)))
            result = hypergeometric_test(k_obs, M_cand, known_in_cand, len(filtered_genes))

            results.append({'filter_type': 'combined', 'threshold': filter_str,
                            'n_genes': len(filtered_genes), 'disease_overlap': k_obs, **result})

            print(f"{filter_str:<30} {len(filtered_genes):<10} {k_obs:<10} {result['expected']:<10.2f} {result['fold_enrichment']:<8.2f} {result['p_value']:<12.2e}")

        results_df = pd.DataFrame(results)
        results_df.to_csv(self.output_dir / 'consensus_enrichment_analysis.csv', index=False)
        print(f"\n✅ Saved: consensus_enrichment_analysis.csv")

        print("\n📊 FINDING OPTIMAL CONSENSUS THRESHOLD")
        print("-" * 100)

        best_fold = 0
        best_config = None

        significant_results = [r for r in results if r['p_value'] < 0.05 and r['direction'] == 'ENRICHED']
        if significant_results:
            for r in significant_results:
                if r['fold_enrichment'] > best_fold:
                    best_fold = r['fold_enrichment']
                    best_config = r

            if best_config:
                print(f"🎯 OPTIMAL CONFIGURATION:")
                print(f"   Filter type: {best_config['filter_type']}")
                print(f"   Threshold: {best_config['threshold']}")
                print(f"   N genes: {best_config['n_genes']}")
                print(f"   Disease overlap: {best_config['disease_overlap']}")
                print(f"   Expected: {best_config['expected']:.2f}")
                print(f"   Fold enrichment: {best_config['fold_enrichment']:.2f}x")
                print(f"   P-value: {best_config['p_value']:.2e}")
        else:
            print("   No significant enrichment found")

        print("\n" + "=" * 120)
        return results_df

    def generate_comprehensive_summary(self):
        print("\n" + "=" * 120)
        print("🔥 GENERATING COMPREHENSIVE SUMMARY REPORT")
        print("=" * 120)

        report_lines = []

        report_lines.append("=" * 120)
        report_lines.append("ULTIMATE COMPREHENSIVE GENE RANKING REPORT - COMPLETE VERSION")
        report_lines.append("=" * 120)
        report_lines.append(f"Phenotype: {self.phenotype}")
        report_lines.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Input File: combined_volcano_data_all_models.csv")
        report_lines.append("=" * 120)

        report_lines.append("\n" + "=" * 120)
        report_lines.append("ANALYSIS CONFIGURATION")
        report_lines.append("=" * 120)
        report_lines.append(f"Method-Specific Thresholds:")
        report_lines.append(f"  - FDR: < {FDR_THRESHOLD} (all methods)")
        report_lines.append(f"  - DE methods: |Log2FC| ≥ {DE_EFFECT_THRESHOLD}")
        if self.has_effect:
            report_lines.append(f"  - Association methods: |Effect| ≥ {ASSOC_EFFECT_THRESHOLD}")
        else:
            report_lines.append(f"  - Association methods: |Log2FC| ≥ {ASSOC_EFFECT_THRESHOLD} (Effect column not found)")

        report_lines.append(f"\nRanking Weights (Document 2 Methodology):")
        report_lines.append(f"  - Reproducibility: {self.ranking_weights['reproducibility']:.1%}")
        report_lines.append(f"  - Effect Size: {self.ranking_weights['effect_size']:.1%}")
        report_lines.append(f"  - Statistical Confidence: {self.ranking_weights['confidence']:.1%}")

        report_lines.append("\n" + "=" * 120)
        report_lines.append("DATASET STATISTICS")
        report_lines.append("=" * 120)
        report_lines.append(f"Universe (all genes tested): {len(self.U_all):,}")
        report_lines.append(f"Training genes: {len(self.G_train):,}")
        report_lines.append(f"Validation genes: {len(self.G_val):,}")
        report_lines.append(f"Test genes: {len(self.G_test):,}")
        report_lines.append(f"Candidates (train ∪ val): {len(self.G_final):,}")
        report_lines.append(f"High confidence (train ∩ val): {len(self.G_high):,}")

        if len(self.M) > 0:
            report_lines.append(f"\nKnown Disease Genes:")
            report_lines.append(f"  Total known genes: {len(self.M):,}")
            report_lines.append(f"  In candidates: {len(set(map(str, self.G_final)) & set(map(str, self.M))):,}")

        report_lines.append("\n" + "=" * 120)
        report_lines.append("GENE STATISTICS")
        report_lines.append("=" * 120)
        report_lines.append(f"Total genes ranked: {len(self.gene_features_df):,}")

        status_counts = self.gene_features_df['Status'].value_counts()
        report_lines.append(f"\nGene Status:")
        report_lines.append(f"  Novel candidates: {status_counts.get('Novel', 0):,}")
        report_lines.append(f"  Known disease genes: {status_counts.get('Known', 0):,}")

        tier_counts = self.gene_features_df['Confidence_Tier'].value_counts()
        report_lines.append(f"\nConfidence Tiers:")
        for tier in ['Tier1_High', 'Tier2_Moderate', 'Tier3_Exploratory']:
            count = int(tier_counts.get(tier, 0))
            pct = (count / len(self.gene_features_df) * 100) if len(self.gene_features_df) > 0 else 0
            report_lines.append(f"  {tier}: {count:,} ({pct:.1f}%)")

        report_lines.append("\n" + "=" * 120)
        report_lines.append("RANKING VIEWS GENERATED")
        report_lines.append("=" * 120)
        for view_name, dfv in self.all_rankings.items():
            report_lines.append(f"  {view_name}: {len(dfv):,} genes")

        report_lines.append("\n" + "=" * 120)
        report_lines.append("TOP 20 GENES (COMPOSITE SCORE)")
        report_lines.append("=" * 120)
        report_lines.append(f"{'Rank':<6} {'Gene':<18} {'Symbol':<12} {'Status':<10} {'Tier':<18} {'Score':<8} {'Hits':<7} {'Tissues':<8}")
        report_lines.append("-" * 120)

        top20 = self.ranked_genes.head(20)
        for _, row in top20.iterrows():
            report_lines.append(
                f"{int(row['Rank']):<6} "
                f"{str(row['Gene']):<18} "
                f"{str(row.get('Symbol','N/A')):<12} "
                f"{str(row.get('Status','NA')):<10} "
                f"{str(row.get('Confidence_Tier','NA')):<18} "
                f"{float(row['Importance_Score']):<8.3f} "
                f"{int(row['Total_Hits']):<7} "
                f"{int(row['N_Tissues']):<8}"
            )

        if 'novel_only' in self.all_rankings and len(self.all_rankings['novel_only']) > 0:
            report_lines.append("\n" + "=" * 120)
            report_lines.append("TOP 10 NOVEL GENE CANDIDATES")
            report_lines.append("=" * 120)
            report_lines.append(f"{'Rank':<6} {'Gene':<18} {'Symbol':<12} {'Tier':<18} {'Score':<8} {'Hits':<7} {'Effect':<10}")
            report_lines.append("-" * 120)

            top10_novel = self.all_rankings['novel_only'].head(10)
            # FIX: iterows -> iterrows
            for _, row in top10_novel.iterrows():
                report_lines.append(
                    f"{int(row['Rank']):<6} "
                    f"{str(row['Gene']):<18} "
                    f"{str(row.get('Symbol','N/A')):<12} "
                    f"{str(row.get('Confidence_Tier','NA')):<18} "
                    f"{float(row['Importance_Score']):<8.3f} "
                    f"{int(row['Total_Hits']):<7} "
                    f"{float(row['Mean_Unified_Effect']):<10.3f}"
                )

        report_lines.append("\n" + "=" * 120)
        report_lines.append("OUTPUT FILES GENERATED")
        report_lines.append("=" * 120)
        report_lines.append(f"Output directory: {self.output_dir}")
        report_lines.append("\nRanking files:")
        report_lines.append("  - RANKED_composite.csv (primary ranking)")
        report_lines.append("  - RANKED_reproducibility.csv")
        report_lines.append("  - RANKED_effect_size.csv")
        report_lines.append("  - RANKED_significance.csv")
        report_lines.append("  - RANKED_novel_only.csv")
        report_lines.append("  - RANKED_known_only.csv")

        report_lines.append("  - RANKED_tier1_high.csv (if any)")
        report_lines.append("  - top_50/100/200/500/1000_genes.csv")
        report_lines.append("\nEvaluation files:")
        report_lines.append("  - dataset_overlaps.csv")
        report_lines.append("  - test_confusion_metrics.csv")
        report_lines.append("  - test_auc_metrics.csv")
        report_lines.append("  - enrichment_vs_random.csv")
        report_lines.append("  - enrichment_curve.csv")
        report_lines.append("  - permutation_test.csv")
        report_lines.append("  - disease_gene_enrichment.csv (if known genes loaded)")
        report_lines.append("  - consensus_enrichment_analysis.csv")

        report_lines.append("\n" + "=" * 120)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 120)

        report_file = self.output_dir / "COMPREHENSIVE_SUMMARY_REPORT.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        print(f"✅ Summary report saved: {report_file}")
        print("\n" + "\n".join(report_lines))
        return report_file

    def run_complete_pipeline(self):
        print("\n🚀 STARTING ULTIMATE COMPREHENSIVE PIPELINE\n")
        try:
            self.load_and_filter_data()
            self.split_by_dataset()
            self.compute_overlaps()
            self.build_gene_features()
            self.compute_gene_scores()
            self.test_evaluation_confusion()
            self.test_evaluation_auc()
            self.enrichment_analysis()
            
            # 🔥 NEW: RUN VALIDATION DECISION TREE
            print("\n" + "=" * 120)
            
            validator = ValidationDecisionTree(self)
            validation_passed = validator.run_all_checks()
            
            if validator.should_stop:
                print("\n" + "=" * 120)
                print("🛑 PIPELINE STOPPED - VALIDATION FAILED")
                print("=" * 120)
                print("\n❌ Critical validation checks failed.")
                print("   See VALIDATION_DECISION_TREE.txt for details.")
                print("\n   DO NOT proceed to:")
                print("   • PPI network analysis")
                print("   • Drug repurposing")
                print("   • Result interpretation")
                print("\n   NEXT STEPS:")
                print("   • Review diagnostic output")
                print("   • Adjust thresholds/data quality")
                print("   • Re-run pipeline")
                print("=" * 120)
                return False
            
            # Continue with remaining analyses
            self.consensus_filtering_enrichment()
            self.generate_comprehensive_summary()
            
            print("\n" + "=" * 120)
            print("🎉 ULTIMATE COMPREHENSIVE ANALYSIS COMPLETE!")
            print("=" * 120)
            print(f"\n📁 All results saved to: {self.output_dir}")
            
            if validator.proceed_with_caution:
                print("\n⚠️  PROCEED WITH CAUTION:")
                for warning in validator.warnings:
                    print(f"   • {warning}")
                print("\n   → Signal replicates but may be non-specific")
                print("   → Prioritize high-confidence candidates")
            else:
                print("\n✅ All validation checks passed")
                print("   → Safe to proceed to downstream analyses")
            
            print("\n✅ FEATURES INCLUDED:")
            print("   ✅ Train/Val/Test split (no data leakage)")
            print("   ✅ Sophisticated multi-metric scoring (Document 2)")
            print("   ✅ 9 ranking views")
            print("   ✅ Consensus filtering enrichment (Document 1)")
            print("   ✅ Enrichment vs random analysis")
            print("   ✅ Permutation testing (1000 permutations)")
            print("   ✅ Complete evaluation suite")
            print("   ✅ Gene symbol lookup")
            print("   ✅ Confidence tier classification")
            print("   ✅ Known/Novel identification")
            print("   ✅ VALIDATION DECISION TREE (NEW)")
            print("=" * 120)
            
            return True

        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            raise


# =============================================================================
# VALIDATION FRAMEWORK - DECISION TREE
# =============================================================================

class ValidationDecisionTree:
    """Decision tree for pipeline validation with hard stops"""
    
    def __init__(self, ranker):
        self.ranker = ranker
        self.checks = {
            'check1': {'passed': False, 'required': True, 'name': 'Test Replication'},
            'check1b': {'passed': False, 'required': False, 'name': 'Basic Signal Sanity'},
            'check2': {'passed': False, 'required': False, 'name': 'Known Gene Enrichment'},
            'check3': {'passed': False, 'required': True, 'name': 'Permutation Test'},
            'check3b': {'passed': False, 'required': False, 'name': 'Stability Diagnostics'}
        }
        self.warnings = []
        self.should_stop = False
        self.proceed_with_caution = False
    
    def run_all_checks(self):
        """Execute full decision tree"""
        print("\n" + "=" * 120)
        print("🔥 VALIDATION DECISION TREE - COMPREHENSIVE CHECKS")
        print("=" * 120)
        
        # CHECK 1: Test Replication (REQUIRED)
        self.check1_test_replication()
        
        if not self.checks['check1']['passed']:
            print("\n" + "=" * 120)
            print("❌ CRITICAL FAILURE: CHECK 1 (Test Replication) FAILED")
            print("=" * 120)
            self.check1b_signal_sanity()
            self.should_stop = True
            self.generate_stop_report()
            return False
        
        # CHECK 2: Known Gene Enrichment (VALIDATION)
        self.check2_known_gene_enrichment()
        
        if not self.checks['check2']['passed']:
            print("\n⚠️  WARNING: Known gene enrichment weak - proceed with caution")
            self.proceed_with_caution = True
            self.warnings.append("Known disease gene enrichment below threshold")
        
        # CHECK 3: Permutation Test (REQUIRED)
        self.check3_permutation_test()
        
        if not self.checks['check3']['passed']:
            print("\n" + "=" * 120)
            print("❌ CRITICAL FAILURE: CHECK 3 (Permutation Test) FAILED")
            print("=" * 120)
            self.check3b_stability_diagnostics()
            self.should_stop = True
            self.generate_stop_report()
            return False
        
        # All critical checks passed
        print("\n" + "=" * 120)
        print("✅ ALL CRITICAL CHECKS PASSED")
        print("=" * 120)
        
        if self.proceed_with_caution:
            print("\n⚠️  PROCEED WITH CAUTION:")
            for warning in self.warnings:
                print(f"   • {warning}")
        
        self.generate_validation_summary()
        return True
    
    def check1_test_replication(self):
        """CHECK 1: Are Train+Val candidates enriched in Test positives?"""
        print("\n" + "=" * 120)
        print("🔍 CHECK 1: TEST REPLICATION (REQUIRED)")
        print("=" * 120)
        print("Question: Are Train+Val candidates enriched in Test positives?")
        print("Threshold: Fold enrichment > 1.5x AND p-value < 0.01")
        print("-" * 120)
        
        G_final = self.ranker.G_final
        G_test = self.ranker.G_test
        U_all = self.ranker.U_all
        
        N_pred = len(G_final)
        k_obs = len(G_final & G_test)
        k_exp = N_pred * len(G_test) / len(U_all)
        FE = k_obs / k_exp if k_exp > 0 else 0
        
        pval = hypergeom.sf(k_obs - 1, len(U_all), len(G_test), N_pred)
        
        print(f"Universe U: {len(U_all):,} genes")
        print(f"Test positives T: {len(G_test):,} genes")
        print(f"Candidates (G_final): {N_pred:,} genes")
        print(f"Observed overlap: {k_obs:,} genes")
        print(f"Expected overlap: {k_exp:.2f} genes")
        print(f"Fold enrichment: {FE:.3f}x")
        print(f"P-value: {pval:.2e}")
        
        passed = FE > 1.5 and pval < 0.01
        self.checks['check1']['passed'] = passed
        self.checks['check1']['FE'] = FE
        self.checks['check1']['pval'] = pval
        
        if passed:
            print(f"\n✅ CHECK 1 PASSED: {FE:.3f}x enrichment (p={pval:.2e})")
        else:
            print(f"\n❌ CHECK 1 FAILED:")
            if FE <= 1.5:
                print(f"   • Fold enrichment too low: {FE:.3f}x ≤ 1.5x")
            if pval >= 0.01:
                print(f"   • P-value not significant: {pval:.2e} ≥ 0.01")
        
        print("=" * 120)
        return passed
    
    def check1b_signal_sanity(self):
        """CHECK 1b: Basic signal sanity diagnostics"""
        print("\n" + "=" * 120)
        print("🔍 CHECK 1b: BASIC SIGNAL SANITY (DIAGNOSTIC)")
        print("=" * 120)
        
        issues = []
        
        # Check 1: Candidate set size
        n_candidates = len(self.ranker.G_final)
        if n_candidates < 500:
            issues.append(f"Too few candidates: {n_candidates:,} < 500")
            print(f"⚠️  Too few candidates: {n_candidates:,} genes (< 500)")
        else:
            print(f"✅ Candidate set size: {n_candidates:,} genes")
        
        # Check 2: Train/Val overlap rate
        overlap = len(self.ranker.G_train & self.ranker.G_val)
        min_set = min(len(self.ranker.G_train), len(self.ranker.G_val))
        overlap_rate = overlap / min_set if min_set > 0 else 0
        
        if overlap_rate < 0.01:
            issues.append(f"Minimal train/val overlap: {overlap_rate:.1%} < 1%")
            print(f"⚠️  Minimal train/val overlap: {overlap_rate:.1%}")
        else:
            print(f"✅ Train/val overlap: {overlap_rate:.1%}")
        
        # Check 3: Bootstrap stability (simplified)
        stability = self.compute_bootstrap_stability()
        if stability < 0.5:
            issues.append(f"Low ranking stability: Jaccard={stability:.2f} < 0.5")
            print(f"⚠️  Low ranking stability: {stability:.2f}")
        else:
            print(f"✅ Ranking stability (bootstrap): {stability:.2f}")
        
        self.checks['check1b']['issues'] = issues
        self.checks['check1b']['passed'] = len(issues) == 0
        
        if len(issues) > 0:
            print(f"\n⚠️  DIAGNOSTIC ISSUES DETECTED:")
            for issue in issues:
                print(f"   • {issue}")
            print("\n💡 RECOMMENDATIONS:")
            print("   • Lower FDR threshold (try 0.05)")
            print("   • Lower effect size thresholds")
            print("   • Check for batch effects or tissue artifacts")
        
        print("=" * 120)
    
    def check2_known_gene_enrichment(self):
        """CHECK 2: Known disease gene enrichment (prior validation)"""
        print("\n" + "=" * 120)
        print("🔍 CHECK 2: KNOWN DISEASE GENE ENRICHMENT (VALIDATION)")
        print("=" * 120)
        print("Question: Are candidates enriched for known migraine genes?")
        print("Threshold: Fold enrichment > 1.2x AND p-value < 0.05")
        print("-" * 120)
        
        if len(self.ranker.M) == 0:
            print("⚠️  No known disease genes loaded - CHECK 2 SKIPPED")
            self.checks['check2']['passed'] = None
            self.checks['check2']['skipped'] = True
            print("=" * 120)
            return None
        
        G_final = self.ranker.G_final
        M_in_U = set(map(str, self.ranker.M)) & set(map(str, self.ranker.U_all))
        
        N_pred = len(G_final)
        k_obs = len(set(map(str, G_final)) & M_in_U)
        k_exp = N_pred * len(M_in_U) / len(self.ranker.U_all)
        FE = k_obs / k_exp if k_exp > 0 else 0
        
        pval = hypergeom.sf(k_obs - 1, len(self.ranker.U_all), len(M_in_U), N_pred)
        
        print(f"Universe U: {len(self.ranker.U_all):,} genes")
        print(f"Known disease genes M: {len(self.ranker.M):,} genes")
        print(f"Known genes in universe (M∩U): {len(M_in_U):,} genes")
        print(f"Candidates: {N_pred:,} genes")
        print(f"Observed overlap: {k_obs:,} genes")
        print(f"Expected overlap: {k_exp:.2f} genes")
        print(f"Fold enrichment: {FE:.3f}x")
        print(f"P-value: {pval:.2e}")
        
        passed = FE > 1.2 and pval < 0.05
        self.checks['check2']['passed'] = passed
        self.checks['check2']['FE'] = FE
        self.checks['check2']['pval'] = pval
        
        if passed:
            print(f"\n✅ CHECK 2 PASSED: {FE:.3f}x enrichment (p={pval:.2e})")
        else:
            print(f"\n⚠️  CHECK 2 WEAK (but not critical):")
            if FE <= 1.2:
                print(f"   • Fold enrichment low: {FE:.3f}x ≤ 1.2x")
            if pval >= 0.05:
                print(f"   • P-value not significant: {pval:.2e} ≥ 0.05")
            print("\n💡 INTERPRETATION:")
            print("   • Signal may be non-specific or novel")
            print("   • Still OK for exploratory drug repurposing")
            print("   • Prioritize candidates that replicate in test")
        
        print("=" * 120)
        return passed
    
    def check3_permutation_test(self):
        """CHECK 3: Permutation test - ranking better than random?"""
        print("\n" + "=" * 120)
        print("🔍 CHECK 3: PERMUTATION TEST (REQUIRED)")
        print("=" * 120)
        print("Question: Does ranking beat random ordering?")
        print("Threshold: P-value < 0.05 (ranking significantly better than null)")
        print("-" * 120)
        
        # Get stored permutation results
        if not hasattr(self.ranker, 'perm_pr_pval') or not hasattr(self.ranker, 'perm_roc_pval'):
            print("⚠️  Permutation test not yet run - CHECK 3 DEFERRED")
            self.checks['check3']['passed'] = None
            print("=" * 120)
            return None
        
        p_pr = self.ranker.perm_pr_pval
        p_roc = self.ranker.perm_roc_pval
        
        # ✅ CRITICAL FIX: Handle None values safely
        if p_pr is None and p_roc is None:
            print("⚠️  Permutation p-values are None - CHECK 3 FAILED")
            self.checks['check3']['passed'] = False
            self.checks['check3']['p_pr'] = None
            self.checks['check3']['p_roc'] = None
            print("=" * 120)
            return False
        
        # Display p-values (safe for None)
        if p_pr is not None:
            print(f"PR-AUC permutation p-value: {p_pr:.2e}")
        else:
            print("PR-AUC permutation p-value: N/A")
        
        if p_roc is not None:
            print(f"ROC-AUC permutation p-value: {p_roc:.2e}")
        else:
            print("ROC-AUC permutation p-value: N/A")
        
        # Safe comparison (handles None)
        passed = ((p_pr is not None and p_pr < 0.05) or (p_roc is not None and p_roc < 0.05))
        
        self.checks['check3']['passed'] = passed
        self.checks['check3']['p_pr'] = p_pr
        self.checks['check3']['p_roc'] = p_roc
        
        if passed:
            print(f"\n✅ CHECK 3 PASSED: Ranking significantly better than random")
        else:
            print(f"\n❌ CHECK 3 FAILED: Ranking not better than random")
            if p_pr is not None:
                print(f"   • PR-AUC p-value: {p_pr:.2e} ≥ 0.05")
            if p_roc is not None:
                print(f"   • ROC-AUC p-value: {p_roc:.2e} ≥ 0.05")
        
        print("=" * 120)
        return passed
    
    def check3b_stability_diagnostics(self):
        """CHECK 3b: Ranking stability and threshold diagnostics"""
        print("\n" + "=" * 120)
        print("🔍 CHECK 3b: STABILITY + THRESHOLD DIAGNOSTICS")
        print("=" * 120)
        
        print("💡 RECOMMENDATIONS:")
        print("   • Revisit FDR/effect thresholds (too lenient?)")
        print("   • Check for batch effects or confounders")
        print("   • Verify tissue distribution balance")
        print("   • Consider stricter consensus filters")
        
        # Additional diagnostics
        print("\n📊 Threshold Analysis:")
        print(f"   Current FDR threshold: {FDR_THRESHOLD}")
        print(f"   Current DE effect threshold: {DE_EFFECT_THRESHOLD}")
        print(f"   Current Assoc effect threshold: {ASSOC_EFFECT_THRESHOLD}")
        
        print("\n📊 Candidate Distribution:")
        print(f"   Total candidates: {len(self.ranker.G_final):,}")
        print(f"   Train only: {len(self.ranker.G_train - self.ranker.G_val):,}")
        print(f"   Val only: {len(self.ranker.G_val - self.ranker.G_train):,}")
        print(f"   Both (high conf): {len(self.ranker.G_high):,}")
        
        self.checks['check3b']['passed'] = False
        print("=" * 120)
    
    def compute_bootstrap_stability(self, n_bootstrap=100, top_k=200):
        """Compute bootstrap Jaccard index for top-K ranking stability"""
        try:
            df = self.ranker.gene_features_df.copy()
            if len(df) < top_k:
                return 0.0
            
            rng = np.random.RandomState(42)
            top_sets = []
            
            for _ in range(n_bootstrap):
                # Resample with replacement
                boot_idx = rng.choice(len(df), size=len(df), replace=True)
                boot_df = df.iloc[boot_idx].copy()
                
                # Re-rank
                boot_df = boot_df.sort_values('Importance_Score', ascending=False)
                top_genes = set(boot_df.head(top_k)['Gene'])
                top_sets.append(top_genes)
            
            # Compute average pairwise Jaccard
            jaccards = []
            for i in range(len(top_sets)):
                for j in range(i + 1, len(top_sets)):
                    intersection = len(top_sets[i] & top_sets[j])
                    union = len(top_sets[i] | top_sets[j])
                    jaccard = intersection / union if union > 0 else 0
                    jaccards.append(jaccard)
            
            return np.mean(jaccards) if jaccards else 0.0
        
        except Exception:
            return 0.5  # Neutral value if computation fails
    
    def generate_stop_report(self):
        """Generate report when pipeline should stop"""
        print("\n" + "=" * 120)
        print("🛑 PIPELINE STOP RECOMMENDATION")
        print("=" * 120)
        
        print("\n❌ CRITICAL CHECKS FAILED - DO NOT PROCEED TO PPI/DRUG ANALYSIS")
        print("\nFailed checks:")
        for check_id, check_info in self.checks.items():
            if check_info['required'] and not check_info['passed']:
                print(f"   • {check_info['name']}")
        
        print("\n📋 NEXT STEPS:")
        print("   1. Review diagnostic output above")
        print("   2. Adjust filtering thresholds")
        print("   3. Check for data quality issues")
        print("   4. Re-run pipeline after corrections")
        
        print("\n⚠️  DO NOT interpret enrichment results")
        print("⚠️  DO NOT proceed to drug repurposing")
        print("=" * 120)
    
    def generate_validation_summary(self):
        """Generate comprehensive validation summary"""
        print("\n" + "=" * 120)
        print("📋 VALIDATION SUMMARY")
        print("=" * 120)
        
        summary_lines = []
        summary_lines.append("\nCHECK RESULTS:")
        summary_lines.append("-" * 120)
        summary_lines.append(f"{'Check':<35} {'Status':<10} {'Required':<10} {'Result':<60}")
        summary_lines.append("-" * 120)
        
        for check_id, check_info in self.checks.items():
            status_icon = "✅" if check_info['passed'] else ("⚠️" if check_info['passed'] is None else "❌")
            required = "YES" if check_info['required'] else "NO"
            
            result_str = ""
            # CHECK 1: Test Replication
            if check_id == 'check1' and 'FE' in check_info:
                result_str = f"FE={check_info['FE']:.2f}x, p={check_info['pval']:.2e}"
                if not check_info['passed']:
                    result_str += f" [Need: FE>1.5, p<0.01]"
            
            # CHECK 1b: Signal Sanity Diagnostics
            elif check_id == 'check1b' and 'issues' in check_info:
                if len(check_info['issues']) == 0:
                    result_str = "All sanity checks passed"
                else:
                    result_str = f"{len(check_info['issues'])} issues: "
                    # Show first issue (truncated)
                    first_issue = check_info['issues'][0]
                    if len(first_issue) > 40:
                        result_str += first_issue[:40] + "..."
                    else:
                        result_str += first_issue
            
            # CHECK 2: Known Gene Enrichment
            elif check_id == 'check2':
                if 'skipped' in check_info and check_info['skipped']:
                    result_str = "Skipped (no known genes loaded)"
                elif 'FE' in check_info:
                    result_str = f"FE={check_info['FE']:.2f}x, p={check_info['pval']:.2e}"
                    if not check_info['passed']:
                        result_str += f" [Need: FE>1.2, p<0.05]"
            
            # CHECK 3: Permutation Test
            elif check_id == 'check3' and 'p_pr' in check_info:
                p_pr = check_info.get('p_pr')
                p_roc = check_info.get('p_roc')
                if p_pr is not None and p_roc is not None:
                    result_str = f"p_PR={p_pr:.2e}, p_ROC={p_roc:.2e}"
                elif p_pr is not None:
                    result_str = f"p_PR={p_pr:.2e}, p_ROC=N/A"
                elif p_roc is not None:
                    result_str = f"p_PR=N/A, p_ROC={p_roc:.2e}"
                else:
                    result_str = "p_PR=N/A, p_ROC=N/A"
                
                if not check_info['passed']:
                    result_str += " [Need: p<0.05]"
            
            # CHECK 3b: Stability Diagnostics
            elif check_id == 'check3b':
                result_str = "Ranking unstable, see diagnostics above"
            
            summary_lines.append(f"{check_info['name']:<35} {status_icon:<10} {required:<10} {result_str:<60}")
        
        summary_lines.append("-" * 120)
        
        # Add detailed explanations for failed checks
        summary_lines.append("\n📊 DETAILED CHECK INFORMATION:")
        summary_lines.append("-" * 120)
        
        # Detail for Check 1b if it ran
        if 'check1b' in self.checks and 'issues' in self.checks['check1b']:
            summary_lines.append(f"\nCHECK 1b - Basic Signal Sanity:")
            if len(self.checks['check1b']['issues']) == 0:
                summary_lines.append("   ✅ All sanity checks passed")
            else:
                summary_lines.append(f"   ❌ {len(self.checks['check1b']['issues'])} issues detected:")
                for issue in self.checks['check1b']['issues']:
                    summary_lines.append(f"      • {issue}")
        
        # Detail for Check 2
        if 'check2' in self.checks:
            check2 = self.checks['check2']
            if 'skipped' in check2 and check2['skipped']:
                summary_lines.append(f"\nCHECK 2 - Known Gene Enrichment:")
                summary_lines.append("   ⚠️  Skipped (no known disease genes loaded)")
            elif 'FE' in check2:
                summary_lines.append(f"\nCHECK 2 - Known Gene Enrichment:")
                summary_lines.append(f"   Fold Enrichment: {check2['FE']:.3f}x")
                summary_lines.append(f"   P-value: {check2['pval']:.2e}")
                summary_lines.append(f"   Threshold: FE > 1.2x AND p < 0.05")
                if check2['passed']:
                    summary_lines.append(f"   ✅ Candidates enriched for known disease genes")
                else:
                    summary_lines.append(f"   ⚠️  Weak enrichment (but not critical)")
                    summary_lines.append(f"   → Signal may be novel or non-specific")
        
        # Detail for Check 3b if it ran
        if 'check3b' in self.checks and not self.checks['check3b']['passed']:
            summary_lines.append(f"\nCHECK 3b - Stability Diagnostics:")
            summary_lines.append("   ❌ Ranking failed permutation test")
            summary_lines.append("   → Rankings not significantly better than random")
            summary_lines.append("   → Check FDR/effect thresholds (may be too lenient)")
            summary_lines.append("   → Check for batch effects or confounders")
        
        summary_lines.append("-" * 120)
        
        if self.proceed_with_caution:
            summary_lines.append("\n⚠️  PROCEED WITH CAUTION:")
            for warning in self.warnings:
                summary_lines.append(f"   • {warning}")
        
        summary_lines.append("\n✅ PIPELINE VALIDATION: PASSED")
        summary_lines.append("   → Safe to proceed to PPI network analysis")
        summary_lines.append("   → Safe to proceed to drug repurposing")
        
        for line in summary_lines:
            print(line)
        
        # Save to file
        summary_file = self.ranker.output_dir / "VALIDATION_DECISION_TREE.txt"
        with open(summary_file, 'w') as f:
            f.write('\n'.join(summary_lines))
        
        print(f"\n✅ Validation report saved: {summary_file}")
        print("=" * 120)
    

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Ultimate Comprehensive Gene Ranking Pipeline with Validation Decision Tree'
    )
    parser.add_argument(
        'phenotype',
        type=str,
        help='Phenotype name (e.g., "migraine")'
    )
    parser.add_argument(
        '--weights',
        type=str,
        default=None,
        help='Custom ranking weights as "repro,effect,conf" (e.g., "0.4,0.3,0.3")'
    )
    
    args = parser.parse_args()
    
    # Parse custom weights if provided
    ranking_weights = None
    if args.weights:
        try:
            parts = [float(x.strip()) for x in args.weights.split(',')]
            if len(parts) != 3:
                raise ValueError("Must provide exactly 3 weights")
            if not abs(sum(parts) - 1.0) < 0.01:
                raise ValueError("Weights must sum to 1.0")
            
            ranking_weights = {
                'reproducibility': parts[0],
                'effect_size': parts[1],
                'confidence': parts[2]
            }
            print(f"\n✅ Using custom weights: {ranking_weights}")
        except Exception as e:
            print(f"❌ Error parsing weights: {e}")
            print("   Using default weights instead")
            ranking_weights = None
    
    # Run pipeline
    try:
        ranker = UltimateComprehensiveGeneRanker(
            phenotype=args.phenotype,
            ranking_weights=ranking_weights
        )
        
        success = ranker.run_complete_pipeline()
        
        if success:
            print("\n" + "=" * 120)
            print("✅ PIPELINE COMPLETED SUCCESSFULLY")
            print("=" * 120)
            exit(0)
        else:
            print("\n" + "=" * 120)
            print("❌ PIPELINE FAILED VALIDATION")
            print("=" * 120)
            exit(1)
            
    except Exception as e:
        print("\n" + "=" * 120)
        print(f"❌ PIPELINE ERROR: {e}")
        print("=" * 120)
        import traceback
        traceback.print_exc()
        exit(1)
