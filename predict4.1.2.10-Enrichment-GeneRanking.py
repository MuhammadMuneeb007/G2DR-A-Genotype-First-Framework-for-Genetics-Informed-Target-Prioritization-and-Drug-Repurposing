#!/usr/bin/env python3
"""
predict_final_FLEXIBLE.py
=========================
Complete analysis with FLEXIBLE target list size.
Specify how many top targets you want with --top-targets flag.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings
warnings.filterwarnings('ignore')


# -----------------------------
# Helpers
# -----------------------------
def strip_ensg_version(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.startswith("ENSG") and "." in s:
        return s.split(".")[0]
    return s


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def pct_rank(series: pd.Series) -> pd.Series:
    """
    Percentile rank normalization.
    CRITICAL: Returns 0 for missing/zero values to avoid giving credit to 'no evidence'.
    """
    s = pd.to_numeric(series, errors="coerce")
    ranked = s.rank(pct=True, method="average").fillna(0.0)
    # CRITICAL FIX: Set percentile to 0 where raw score is 0 or missing
    ranked[s <= 0] = 0.0
    return ranked


def _collect_compat(ldf: pl.LazyFrame) -> pl.DataFrame:
    try:
        return ldf.collect(engine="streaming")
    except TypeError:
        return ldf.collect()


def auc_pr(y: np.ndarray, score: np.ndarray):
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan, np.nan
    try:
        return roc_auc_score(y, score), average_precision_score(y, score)
    except:
        return np.nan, np.nan


def fmt_metric(val):
    if val != val:
        return "NA"
    return f"{val:.4f}"


def topk_stats(ranked_genes, positives: set, universe_n: int, k: int):
    """
    Calculate Top-K enrichment statistics.
    
    Formula:
    - Observed = number of positives in top K
    - Expected = K × (total_positives / universe_size)
    - Fold Enrichment = Observed / Expected
    - Precision = Observed / K
    - Recall = Observed / total_positives
    """
    k = min(int(k), len(ranked_genes))
    top = set(ranked_genes[:k])
    obs = len(top & positives)
    exp = k * (len(positives) / universe_n) if universe_n > 0 else np.nan
    fe = (obs / exp) if exp and exp > 0 else np.nan
    prec = obs / k if k > 0 else np.nan
    rec = obs / len(positives) if len(positives) > 0 else np.nan
    return obs, exp, fe, prec, rec


def print_block(title: str):
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)


def print_topk_table(ranked_genes, positives: set, universe_n: int, topk_list: list, label: str = ""):
    """Print enrichment table."""
    if label:
        print(f"\n  {label}:")
    hdr = f"{'TopK':>6}  {'Observed':>9}  {'Expected':>9}  {'FE':>7}  {'Precision':>10}  {'Recall':>10}"
    print(hdr)
    print("-" * len(hdr))
    for k in topk_list:
        obs, exp, fe, prec, rec = topk_stats(ranked_genes, positives, universe_n, k)
        exp_s = f"{exp:.2f}" if exp == exp else "NA"
        fe_s = f"{fe:.2f}" if fe == fe else "NA"
        print(f"{int(k):>6}  {obs:>9}  {exp_s:>9}  {fe_s:>7}  {prec:>10.4f}  {rec:>10.4f}")


def print_formulas():
    """Print all calculation formulas."""
    print("\n📐 CALCULATION FORMULAS:")
    print("\n  1. Percentile Normalization:")
    print("     Score_norm = percentile_rank(Score_raw)")
    print("     CRITICAL: If Score_raw = 0 (no evidence), then Score_norm = 0")
    print("     Range: [0, 1]")
    
    print("\n  2. Combined Score:")
    print("     Linear:         Combined = w_DE×DE + w_Path×Path + w_Drug×Drug + w_Hub×Hub")
    print("     Multiplicative: Combined = (w_DE×DE + w_Path×Path + w_Drug×Drug) × (1 + boost×Hub)")
    
    print("\n  3. Enrichment Statistics:")
    print("     CRITICAL: Universe = genes actually ranked (not full volcano)")
    print("     Observed  = count of positives in Top-K")
    print("     Expected  = K × (total_positives_in_universe / universe_size)")
    print("     Fold Enrichment = Observed / Expected")
    print("     Precision = Observed / K")
    print("     Recall    = Observed / total_positives")


# -----------------------------
# Volcano and genes
# -----------------------------
def volcano_universe_and_positives(volcano_path: Path, split_keywords: list, fdr_thr: float, effect_thr: float):
    print(f"\n📖 READING: {volcano_path}")
    ldf = pl.scan_csv(str(volcano_path), infer_schema_length=200)
    cols = ldf.collect_schema().names()
    print(f"   ✓ Columns found: {cols[:5]}..." if len(cols) > 5 else f"   ✓ Columns: {cols}")

    required = {"Gene", "Dataset", "FDR", "Log2FoldChange"}
    missing = required - set(cols)
    if missing:
        raise ValueError(f"Volcano missing columns: {sorted(missing)}")

    ldf2 = ldf.with_columns([
        pl.col("Gene").cast(pl.Utf8).str.strip_chars().str.replace(r"\.\d+$", "").alias("Gene_clean"),
        pl.col("Dataset").cast(pl.Utf8).str.to_lowercase().alias("Dataset_lc"),
        pl.col("FDR").cast(pl.Float64).alias("FDR_f"),
        pl.col("Log2FoldChange").cast(pl.Float64).alias("LFC_f"),
    ])

    universe_df = _collect_compat(
        ldf2.select(pl.col("Gene_clean").alias("Gene")).drop_nulls().unique()
    )
    universe = set(universe_df["Gene"].to_list())
    print(f"   ✓ Volcano universe: {len(universe):,} unique genes")

    split_mask = None
    for kw in split_keywords:
        m = pl.col("Dataset_lc").str.contains(str(kw).lower())
        split_mask = m if split_mask is None else (split_mask | m)

    pos_df = _collect_compat(
        ldf2.filter(split_mask)
            .filter((pl.col("FDR_f") < fdr_thr) & (pl.col("LFC_f").abs() >= effect_thr))
            .select(pl.col("Gene_clean").alias("Gene"))
            .drop_nulls()
            .unique()
    )
    positives = set(pos_df["Gene"].to_list())
    print(f"   ✓ Positives (keywords={split_keywords}): {len(positives):,} genes")

    return positives, universe


def load_migraine_genes(path: Path) -> set:
    print(f"\n📖 READING: {path}")
    mg = pd.read_csv(path)
    print(f"   ✓ Shape: {mg.shape}")
    if "ensembl_gene_id" not in mg.columns:
        raise ValueError("migraine_genes.csv must have 'ensembl_gene_id' column.")
    genes = set(mg["ensembl_gene_id"].astype(str).map(strip_ensg_version).tolist())
    genes = {g for g in genes if g.startswith("ENSG")}
    print(f"   ✓ Migraine genes loaded: {len(genes):,}")
    return genes


# -----------------------------
# Build master
# -----------------------------
def build_master(phenotype: str, base_dir: Path, use_directional_pathway: bool, apply_dircons_penalty: bool):
    base_path = base_dir / phenotype
    files_dir = base_path / "GeneDifferentialExpression" / "Files"
    ranking_dir = files_dir / "UltimateCompleteRankingAnalysis"

    de_file = ranking_dir / "RANKED_composite.csv"
    path_file = ranking_dir / "PathwayIntegration" / "GenePathwayScores.csv"
    hub_file = ranking_dir / "RANKED_WITH_HUB_composite.csv"
    drug_file = ranking_dir / "druggability_analysis" / f"{phenotype}_druggability_all.csv"
    if phenotype == "migraine" and not drug_file.exists():
        drug_file = ranking_dir / "druggability_analysis" / "migraine_druggability_all.csv"

    for p in [de_file, path_file]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    print(f"\n📖 READING: {de_file}")
    de = pd.read_csv(de_file)
    print(f"   ✓ Shape: {de.shape}")
    for c in ["Gene", "Importance_Score", "Direction"]:
        if c not in de.columns:
            raise ValueError(f"DE file missing '{c}': {de_file}")
    de["Gene"] = de["Gene"].astype(str).map(strip_ensg_version)
    de["Direction"] = de["Direction"].astype(str)

    print(f"\n📖 READING: {path_file}")
    ps = pd.read_csv(path_file)
    print(f"   ✓ Shape: {ps.shape}")
    if "Gene" not in ps.columns:
        raise ValueError(f"Pathway file missing 'Gene': {path_file}")
    ps["Gene"] = ps["Gene"].astype(str).map(strip_ensg_version)

    master = de.merge(ps, on="Gene", how="left", suffixes=("", "_path"))
    print(f"   ✓ Merged master shape: {master.shape}")

    for c in ["PathScore_Combined", "PathScore_Up", "PathScore_Down", "PathScore_Directional"]:
        if c in master.columns:
            master[c] = pd.to_numeric(master[c], errors="coerce").fillna(0.0)
        else:
            master[c] = 0.0

    if "DirectionConsistencyScore" in master.columns:
        master["DirectionConsistencyScore"] = pd.to_numeric(master["DirectionConsistencyScore"], errors="coerce")
    else:
        master["DirectionConsistencyScore"] = np.nan

    def choose_path(row):
        if use_directional_pathway:
            d = str(row.get("Direction", "Unknown")).strip().lower()
            if d == "up":
                return row.get("PathScore_Up", 0.0)
            if d == "down":
                return row.get("PathScore_Down", 0.0)
            return row.get("PathScore_Combined", 0.0)
        v = row.get("PathScore_Directional", 0.0)
        return v if v != 0 else row.get("PathScore_Combined", 0.0)

    master["PathScore_used_raw"] = master.apply(choose_path, axis=1).astype(float)
    if apply_dircons_penalty:
        dc = pd.to_numeric(master["DirectionConsistencyScore"], errors="coerce").fillna(0.0)
        master["PathScore_used_raw"] = master["PathScore_used_raw"] * dc

    master["Hub_Score_raw"] = 0.0
    if hub_file.exists():
        print(f"\n📖 READING: {hub_file}")
        hub = pd.read_csv(hub_file)
        print(f"   ✓ Shape: {hub.shape}")
        if "Gene" in hub.columns and "Hub_Score" in hub.columns:
            hub["Gene"] = hub["Gene"].astype(str).map(strip_ensg_version)
            hub_small = hub[["Gene", "Hub_Score"]].copy()
            hub_small["Hub_Score"] = pd.to_numeric(hub_small["Hub_Score"], errors="coerce").fillna(0.0)
            hub_small = hub_small.groupby("Gene", as_index=False)["Hub_Score"].max()
            master = master.merge(hub_small, on="Gene", how="left")
            master["Hub_Score_raw"] = pd.to_numeric(master.get("Hub_Score", 0), errors="coerce").fillna(0.0)
            print(f"   ✓ Hub scores loaded: {len(hub_small):,} genes")

    master["Drug_Score_raw"] = 0.0
    if drug_file.exists():
        print(f"\n📖 READING: {drug_file}")
        drug = pd.read_csv(drug_file)
        print(f"   ✓ Shape: {drug.shape}")
        if "Gene" not in drug.columns:
            raise ValueError(f"Drug file missing 'Gene': {drug_file}")
        drug["Gene"] = drug["Gene"].astype(str).map(strip_ensg_version)

        if "Druggability_Probability" in drug.columns:
            dsmall = drug[["Gene", "Druggability_Probability"]].copy()
            dsmall["Druggability_Probability"] = pd.to_numeric(dsmall["Druggability_Probability"], errors="coerce").fillna(0.0)
            dsmall = dsmall.groupby("Gene", as_index=False)["Druggability_Probability"].max()
            master = master.merge(dsmall, on="Gene", how="left")
            master["Drug_Score_raw"] = pd.to_numeric(master.get("Druggability_Probability", 0), errors="coerce").fillna(0.0)
            print(f"   ✓ Druggability scores loaded: {len(dsmall):,} genes")

    # CRITICAL: Use corrected pct_rank that sets 0 for missing evidence
    master["DE_Score_raw"] = pd.to_numeric(master["Importance_Score"], errors="coerce").fillna(0.0)
    master["DE_Score_norm"] = pct_rank(master["DE_Score_raw"])
    master["Path_Score_norm"] = pct_rank(master["PathScore_used_raw"])
    master["Drug_Score_norm"] = pct_rank(master["Drug_Score_raw"])
    master["Hub_Score_norm"] = pct_rank(master["Hub_Score_raw"])

    print(f"\n✓ Master dataset ready: {master.shape[0]:,} genes")
    print(f"✓ Percentile normalization applied (0 for missing/zero evidence)")
    
    return master, files_dir, ranking_dir


# -----------------------------
# Individual component evaluation
# -----------------------------
def evaluate_individual_components(eval_df: pd.DataFrame, test_pos: set, known: set,
                                    universe_n: int, topk_list: list, target_n: int):
    """Evaluate each component individually - TEST and MIGRAINE only."""
    
    components = [
        ("DE_Score_norm", "DE (Differential Expression)"),
        ("Path_Score_norm", "Pathway"),
        ("Drug_Score_norm", "Druggability"),
        ("Hub_Score_norm", "Hub (Network Centrality)"),
    ]
    
    individual_results = []
    
    for score_col, name in components:
        print_block(f"📊 INDIVIDUAL COMPONENT: {name}")
        
        sorted_df = eval_df.sort_values(score_col, ascending=False).reset_index(drop=True)
        ranked_genes = sorted_df["Gene"].tolist()
        scores = sorted_df[score_col].to_numpy()
        
        y_test = np.array([1 if g in test_pos else 0 for g in ranked_genes], dtype=int)
        y_mig = np.array([1 if g in known else 0 for g in ranked_genes], dtype=int)
        
        test_auc, test_pr = auc_pr(y_test, scores)
        mig_auc, mig_pr = auc_pr(y_mig, scores)
        
        print(f"\n  Overall Performance:")
        print(f"    TEST:       AUC={fmt_metric(test_auc)}, PR-AUC={fmt_metric(test_pr)}")
        print(f"    MIGRAINE:   AUC={fmt_metric(mig_auc)}, PR-AUC={fmt_metric(mig_pr)}")
        
        print_topk_table(ranked_genes, test_pos, universe_n, topk_list, "Enrichment vs TEST (Independent Replication)")
        print_topk_table(ranked_genes, known, universe_n, topk_list, "Enrichment vs MIGRAINE (Disease Genes)")
        
        # Store for summary at target_n
        obs_t, exp_t, fe_t, prec_t, _ = topk_stats(ranked_genes, test_pos, universe_n, target_n)
        obs_m, exp_m, fe_m, prec_m, _ = topk_stats(ranked_genes, known, universe_n, target_n)
        
        individual_results.append({
            'Component': name,
            'Test_PR': test_pr,
            'Mig_PR': mig_pr,
            f'Top{target_n}_Test_Obs': obs_t, f'Top{target_n}_Test_Exp': exp_t, f'Top{target_n}_Test_FE': fe_t, f'Top{target_n}_Test_Prec': prec_t,
            f'Top{target_n}_Mig_Obs': obs_m, f'Top{target_n}_Mig_Exp': exp_m, f'Top{target_n}_Mig_FE': fe_m, f'Top{target_n}_Mig_Prec': prec_m,
        })
    
    return pd.DataFrame(individual_results)


# -----------------------------
# Evaluate combined schemes
# -----------------------------
def evaluate_weighting_scheme(eval_df: pd.DataFrame, scheme_name: str, w_de: float, w_path: float, 
                              w_drug: float, w_hub: float, hub_boost: float,
                              test_pos: set, known: set, universe_n: int, topk_list: list):
    """Evaluate one weighting scheme."""
    
    core = (eval_df["DE_Score_norm"] * w_de + 
            eval_df["Path_Score_norm"] * w_path + 
            eval_df["Drug_Score_norm"] * w_drug)
    
    if hub_boost > 0:
        final_score = core * (1 + hub_boost * eval_df["Hub_Score_norm"])
        hub_method = f"boost={hub_boost}"
    else:
        final_score = core + eval_df["Hub_Score_norm"] * w_hub
        hub_method = f"linear={w_hub:.2f}"
    
    sorted_df = eval_df.copy()
    sorted_df["Combined_Score"] = final_score
    sorted_df = sorted_df.sort_values("Combined_Score", ascending=False).reset_index(drop=True)
    sorted_df["Rank"] = np.arange(1, len(sorted_df) + 1)
    
    ranked_genes = sorted_df["Gene"].tolist()
    scores = sorted_df["Combined_Score"].to_numpy()
    
    y_test = np.array([1 if g in test_pos else 0 for g in ranked_genes], dtype=int)
    y_mig = np.array([1 if g in known else 0 for g in ranked_genes], dtype=int)
    
    test_auc, test_pr = auc_pr(y_test, scores)
    mig_auc, mig_pr = auc_pr(y_mig, scores)
    
    topk_results = {}
    for k in topk_list:
        obs_t, exp_t, fe_t, prec_t, rec_t = topk_stats(ranked_genes, test_pos, universe_n, k)
        obs_m, exp_m, fe_m, prec_m, rec_m = topk_stats(ranked_genes, known, universe_n, k)
        
        topk_results[k] = {
            'test': (obs_t, exp_t, fe_t, prec_t, rec_t),
            'mig': (obs_m, exp_m, fe_m, prec_m, rec_m),
        }
    
    return {
        'scheme': scheme_name,
        'w_de': w_de, 'w_path': w_path, 'w_drug': w_drug, 'w_hub': w_hub,
        'hub_method': hub_method,
        'test_pr': test_pr, 'mig_pr': mig_pr,
        'topk_results': topk_results,
        'ranked_df': sorted_df,
        'ranked_genes': ranked_genes
    }


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="FLEXIBLE target prioritization - specify your target list size!")
    ap.add_argument("phenotype")
    ap.add_argument("--base-dir", default="/data/ascher02/uqmmune1/ANNOVAR")
    
    ap.add_argument("--fdr", type=float, default=0.10)
    ap.add_argument("--effect", type=float, default=0.50)
    
    ap.add_argument("--top-targets", type=int, default=200, 
                    help="Number of top targets for your final list (default: 200)")
    
    ap.add_argument("--use-directional-pathway", action="store_true")
    ap.add_argument("--apply-dircons-penalty", action="store_true")
    
    ap.add_argument("--val-keywords", default="validation,val")
    ap.add_argument("--test-keywords", default="test")
    
    args = ap.parse_args()
    
    phenotype = args.phenotype
    base_dir = Path(args.base_dir)
    target_n = args.top_targets
    
    # Build master
    print_block("📁 LOADING DATA FILES")
    master, files_dir, ranking_dir = build_master(
        phenotype=phenotype,
        base_dir=base_dir,
        use_directional_pathway=args.use_directional_pathway,
        apply_dircons_penalty=args.apply_dircons_penalty
    )
    
    # Load data
    volcano_file = files_dir / "combined_volcano_data_all_models.csv"
    migraine_file = files_dir / "migraine_genes.csv"
    
    val_kw = [s.strip() for s in args.val_keywords.split(",") if s.strip()]
    test_kw = [s.strip() for s in args.test_keywords.split(",") if s.strip()]
    
    val_pos, volcano_universe = volcano_universe_and_positives(volcano_file, val_kw, args.fdr, args.effect)
    test_pos, _ = volcano_universe_and_positives(volcano_file, test_kw, args.fdr, args.effect)
    
    known = load_migraine_genes(migraine_file)
    
    # Filter master to volcano universe
    master["Gene"] = master["Gene"].astype(str).map(strip_ensg_version)
    eval_df = master[master["Gene"].isin(volcano_universe)].copy().reset_index(drop=True)
    
    # CRITICAL FIX: Redefine universe to ACTUAL ranked genes
    eval_universe = set(eval_df["Gene"])
    test_pos_eval = test_pos & eval_universe
    known_eval = known & eval_universe
    universe_n_eval = len(eval_universe)
    
    # Dynamic topk_list based on target_n
    topk_list = sorted(list(set([50, 100, target_n, 500, 1000, 2000])))
    topk_list = [k for k in topk_list if k <= len(eval_df)]
    
    print_block("📋 DATA SUMMARY")
    print(f"Phenotype: {phenotype}")
    print(f"\n🎯 TARGET LIST SIZE: {target_n} genes")
    print(f"\n📊 UNIVERSE BREAKDOWN:")
    print(f"   Volcano universe: {len(volcano_universe):,} genes")
    print(f"   ✅ EVALUATION universe: {universe_n_eval:,} genes (genes actually ranked)")
    print(f"   → This is what we use for Expected calculations")
    print(f"\n📊 DATA SPLITS (in evaluation universe):")
    print(f"   TRAIN+VAL positives: {len(val_pos):,} total, {len(val_pos & eval_universe):,} in eval universe")
    print(f"   TEST positives:      {len(test_pos):,} total, {len(test_pos_eval):,} in eval universe ({100*len(test_pos_eval)/universe_n_eval:.1f}%)")
    print(f"   Migraine genes:      {len(known):,} total, {len(known_eval):,} in eval universe ({100*len(known_eval)/universe_n_eval:.1f}%)")
    print(f"\n⚠️  CRITICAL NOTES:")
    print(f"   • All scores built using TRAIN+VAL → no VAL enrichment reported")
    print(f"   • Expected = K × (positives_in_eval_universe / eval_universe_size)")
    print(f"   • Percentile normalization sets 0 for missing/zero evidence")
    
    print_formulas()
    
    # ==========================================
    # STEP 1: INDIVIDUAL COMPONENTS
    # ==========================================
    print_block("STEP 1: INDIVIDUAL COMPONENT ANALYSIS")
    individual_df = evaluate_individual_components(eval_df, test_pos_eval, known_eval, universe_n_eval, topk_list, target_n)
    
    print_block("📊 INDIVIDUAL COMPONENTS SUMMARY TABLE")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 160)
    print(individual_df.to_string(index=False))
    
    # ==========================================
    # STEP 2: COMBINED SCHEMES
    # ==========================================
    print_block("STEP 2: COMBINED WEIGHTING SCHEMES")
    
    schemes = [
        {
            'name': '1. Balanced Target Prioritization (RECOMMENDED)',
            'rationale': 'Equal emphasis on disease signal, biology, and actionability',
            'w_de': 0.45, 'w_path': 0.25, 'w_drug': 0.25, 'w_hub': 0.05, 'hub_boost': 0.0
        },
        {
            'name': '2. Balanced + Hub Boost',
            'rationale': 'Same as #1, but Hub as multiplicative boost (tie-breaker)',
            'w_de': 0.45, 'w_path': 0.25, 'w_drug': 0.25, 'w_hub': 0.0, 'hub_boost': 0.10
        },
        {
            'name': '3. Actionability-First',
            'rationale': 'Prioritizes druggable targets',
            'w_de': 0.30, 'w_path': 0.20, 'w_drug': 0.45, 'w_hub': 0.05, 'hub_boost': 0.0
        },
        {
            'name': '4. Biology-First',
            'rationale': 'Emphasizes pathway coherence',
            'w_de': 0.40, 'w_path': 0.45, 'w_drug': 0.10, 'w_hub': 0.05, 'hub_boost': 0.0
        },
        {
            'name': '5. DE-Heavy',
            'rationale': 'Strongest genetics/omics signal',
            'w_de': 0.60, 'w_path': 0.20, 'w_drug': 0.15, 'w_hub': 0.05, 'hub_boost': 0.0
        },
        {
            'name': '6. Equal Weights',
            'rationale': 'All components equal',
            'w_de': 0.25, 'w_path': 0.25, 'w_drug': 0.25, 'w_hub': 0.25, 'hub_boost': 0.0
        },
    ]
    
    print("\n🎯 PRE-SPECIFIED WEIGHTING SCHEMES:")
    for s in schemes:
        print(f"\n{s['name']}")
        print(f"   Rationale: {s['rationale']}")
        print(f"   Weights: DE={s['w_de']:.2f}, Path={s['w_path']:.2f}, Drug={s['w_drug']:.2f}, Hub={s['w_hub']:.2f}")
        if s['hub_boost'] > 0:
            print(f"   Hub method: Multiplicative boost ({s['hub_boost']}) - acts as tie-breaker")
    
    # Evaluate all schemes
    print_block("📊 EVALUATING ALL SCHEMES")
    
    results = []
    ranked_dfs = {}
    
    for scheme in schemes:
        result = evaluate_weighting_scheme(
            eval_df, scheme['name'], 
            scheme['w_de'], scheme['w_path'], scheme['w_drug'], scheme['w_hub'], scheme['hub_boost'],
            test_pos_eval, known_eval, universe_n_eval, topk_list
        )
        results.append(result)
        ranked_dfs[scheme['name']] = result['ranked_df']
    
    # Summary table at target_n
    summary_data = []
    for r in results:
        topn_test = r['topk_results'][target_n]['test']
        topn_mig = r['topk_results'][target_n]['mig']
        
        summary_data.append({
            'Scheme': r['scheme'],
            'wDE': r['w_de'], 'wPath': r['w_path'], 'wDrug': r['w_drug'], 
            'Hub': r['hub_method'],
            'TEST_PR': r['test_pr'],
            'MIG_PR': r['mig_pr'],
            f'Top{target_n}_Test_Obs': topn_test[0], f'Top{target_n}_Test_Exp': topn_test[1], f'Top{target_n}_Test_FE': topn_test[2],
            f'Top{target_n}_Mig_Obs': topn_mig[0], f'Top{target_n}_Mig_Exp': topn_mig[1], f'Top{target_n}_Mig_FE': topn_mig[2],
        })
    
    summary = pd.DataFrame(summary_data)
    
    print("\n" + "=" * 160)
    print(f"COMBINED SCHEMES SUMMARY (Top-{target_n} Focus)")
    print("=" * 160)
    print(summary.to_string(index=False))
    
    # Detailed results
    for result in results:
        print_block(f"📌 DETAILED: {result['scheme']}")
        
        print(f"\nWeights: DE={result['w_de']:.2f}, Path={result['w_path']:.2f}, "
              f"Drug={result['w_drug']:.2f}, Hub={result['hub_method']}")
        
        print(f"\n  Overall PR-AUC:")
        print(f"    TEST:       {fmt_metric(result['test_pr'])}")
        print(f"    MIGRAINE:   {fmt_metric(result['mig_pr'])}")
        
        print_topk_table(result['ranked_genes'], test_pos_eval, universe_n_eval, topk_list, "Enrichment vs TEST")
        print_topk_table(result['ranked_genes'], known_eval, universe_n_eval, topk_list, "Enrichment vs MIGRAINE")
    
    # ==========================================
    # SAVE OUTPUTS
    # ==========================================
    out_dir = ensure_dir(ranking_dir / "FinalIntegration")
    
    individual_df.to_csv(out_dir / "01_individual_components_summary.csv", index=False)
    summary.to_csv(out_dir / "02_combined_schemes_summary.csv", index=False)
    
    for scheme_name, ranked_df in ranked_dfs.items():
        safe_name = scheme_name.split('.')[0].strip().replace(' ', '_').replace('-', '_')
        
        # Top N targets
        topn = ranked_df.head(target_n)
        topn.to_csv(out_dir / f"TOP_{target_n}_{safe_name}.csv", index=False)
        
        # Full ranking
        view_cols = ["Rank", "Gene", "Combined_Score",
                     "DE_Score_raw", "DE_Score_norm",
                     "PathScore_used_raw", "Path_Score_norm",
                     "Drug_Score_raw", "Drug_Score_norm",
                     "Hub_Score_raw", "Hub_Score_norm", "Direction"]
        view_cols = [c for c in view_cols if c in ranked_df.columns]
        ranked_df[view_cols].to_csv(out_dir / f"RANKED_{safe_name}.csv", index=False)
    
    # ==========================================
    # FINAL RECOMMENDATION
    # ==========================================
    print_block("✅ FINAL RECOMMENDATION")
    
    recommended = results[0]
    topn_test = recommended['topk_results'][target_n]['test']
    topn_mig = recommended['topk_results'][target_n]['mig']
    
    print(f"\n🎯 RECOMMENDED: {recommended['scheme']}")
    print(f"\n   Formula: Combined = 0.45×DE + 0.25×Pathway + 0.25×Drug + 0.05×Hub")
    print(f"   (Hub at 5% acts as tie-breaker, not primary driver)")
    print(f"\n   Top-{target_n} Performance (CORRECTED Expected values):")
    print(f"   ┌─────────────┬──────────┬──────────┬────────────────┐")
    print(f"   │ Dataset     │ Observed │ Expected │ Fold Enrich.   │")
    print(f"   ├─────────────┼──────────┼──────────┼────────────────┤")
    print(f"   │ TEST        │  {topn_test[0]:3d}/{target_n:3d} │   {topn_test[1]:5.1f}  │  {topn_test[2]:4.2f}x ({topn_test[3]:5.1%}) │")
    print(f"   │ MIGRAINE    │  {topn_mig[0]:3d}/{target_n:3d} │   {topn_mig[1]:5.1f}  │  {topn_mig[2]:4.2f}x ({topn_mig[3]:5.1%}) │")
    print(f"   └─────────────┴──────────┴──────────┴────────────────┘")
    
    print(f"\n   Calculation Details (CORRECTED):")
    print(f"   • Evaluation universe = {universe_n_eval:,} genes (genes actually ranked)")
    print(f"   • Expected TEST = {target_n} × ({len(test_pos_eval):,}/{universe_n_eval:,}) = {topn_test[1]:.1f} genes")
    print(f"   • Expected MIG  = {target_n} × ({len(known_eval):,}/{universe_n_eval:,}) = {topn_mig[1]:.1f} genes")
    
    print(f"\n   ⚠️  KEY FIXES APPLIED:")
    print(f"   ✅ Expected uses actual ranked universe ({universe_n_eval:,}), not volcano ({len(volcano_universe):,})")
    print(f"   ✅ Percentile rank = 0 for genes with no evidence (not ~0.3-0.5)")
    print(f"   ✅ Hub kept at 5% as tie-breaker (appropriate for target selection)")
    
    print_block("✅ COMPLETE")
    print(f"\nOutput directory: {out_dir}")
    print(f"\n📄 Key files:")
    print(f"   • TOP_{target_n}_1.csv  <- Your prioritized target list (RECOMMENDED)")
    print(f"   • 01_individual_components_summary.csv")
    print(f"   • 02_combined_schemes_summary.csv")
    
    print(f"\n💡 To change target list size, run:")
    print(f"   python {Path(__file__).name} {phenotype} --top-targets 500")


if __name__ == "__main__":
    main()