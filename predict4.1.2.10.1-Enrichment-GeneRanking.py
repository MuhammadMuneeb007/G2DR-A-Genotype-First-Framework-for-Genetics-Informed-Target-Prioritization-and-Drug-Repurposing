#!/usr/bin/env python3
"""
predict4.1.2.10-Enrichment-GeneRanking.py
=========================================
Controlled baseline comparison at the gene-prioritization stage.

What this script does
---------------------
Compares five ranking strategies on the SAME ranked gene universe:
1. DE only
2. Pathway only
3. Hub only
4. Druggability only
5. Integrated score

Primary manuscript outputs:
- Gene ROC-AUC vs held-out test genes
- Gene PR-AUC vs held-out test genes
- Top-K migraine genes recovered
- Top-K expected
- Top-K fold enrichment
- Top-K precision

Important
---------
This script evaluates only the baseline-comparison step.
It does NOT try to decide a "recommended" weighting scheme automatically.
You should interpret the output manually in the manuscript.

Example
-------
python predict4.1.2.10-Enrichment-GeneRanking.py migraine --top-k 200 --fdr 0.10 --effect 0.50
"""

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
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


def fmt_metric(val) -> str:
    if pd.isna(val):
        return "NA"
    return f"{val:.4f}"


def pct_rank(series: pd.Series) -> pd.Series:
    """
    Percentile rank normalization.
    Missing values and zero-or-less evidence are assigned 0.
    """
    s = pd.to_numeric(series, errors="coerce")
    ranked = s.rank(pct=True, method="average").fillna(0.0)
    ranked[(s.isna()) | (s <= 0)] = 0.0
    return ranked.astype(float)


def _collect_compat(ldf: pl.LazyFrame) -> pl.DataFrame:
    try:
        return ldf.collect(engine="streaming")
    except TypeError:
        return ldf.collect()


def auc_pr(y_true: np.ndarray, score: np.ndarray):
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return np.nan, np.nan
    try:
        return roc_auc_score(y_true, score), average_precision_score(y_true, score)
    except Exception:
        return np.nan, np.nan


def print_block(title: str):
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)


def topk_stats(ranked_genes, positives: set, universe_n: int, k: int):
    """
    Top-K enrichment statistics.

    Observed = positives in Top-K
    Expected = K * (positives_in_universe / universe_size)
    FE       = Observed / Expected
    Precision = Observed / K
    Recall    = Observed / total_positives
    """
    k = min(int(k), len(ranked_genes))
    top = set(ranked_genes[:k])
    obs = len(top & positives)
    exp = k * (len(positives) / universe_n) if universe_n > 0 else np.nan
    fe = (obs / exp) if exp and exp > 0 else np.nan
    prec = obs / k if k > 0 else np.nan
    rec = obs / len(positives) if len(positives) > 0 else np.nan
    return obs, exp, fe, prec, rec


def print_topk_table(ranked_genes, positives: set, universe_n: int, topk_list: list, label: str):
    print(f"\n  {label}")
    hdr = f"{'TopK':>6}  {'Observed':>9}  {'Expected':>9}  {'FE':>7}  {'Precision':>10}  {'Recall':>10}"
    print(hdr)
    print("-" * len(hdr))
    for k in topk_list:
        obs, exp, fe, prec, rec = topk_stats(ranked_genes, positives, universe_n, k)
        exp_s = f"{exp:.2f}" if pd.notna(exp) else "NA"
        fe_s = f"{fe:.2f}" if pd.notna(fe) else "NA"
        print(f"{int(k):>6}  {obs:>9}  {exp_s:>9}  {fe_s:>7}  {prec:>10.4f}  {rec:>10.4f}")


# --------------------------------------------------------------------------------------
# Load positives / universe
# --------------------------------------------------------------------------------------
def volcano_universe_and_positives(volcano_path: Path, split_keywords: list, fdr_thr: float, effect_thr: float):
    print(f"\n📖 READING: {volcano_path}")
    ldf = pl.scan_csv(str(volcano_path), infer_schema_length=200)
    cols = ldf.collect_schema().names()
    print(f"   ✓ Columns found: {cols[:5]}..." if len(cols) > 5 else f"   ✓ Columns: {cols}")

    required = {"Gene", "Dataset", "FDR", "Log2FoldChange"}
    missing = required - set(cols)
    if missing:
        raise ValueError(f"Volcano file missing columns: {sorted(missing)}")

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
        raise ValueError("migraine_genes.csv must have column 'ensembl_gene_id'.")

    genes = set(mg["ensembl_gene_id"].astype(str).map(strip_ensg_version).tolist())
    genes = {g for g in genes if g.startswith("ENSG")}
    print(f"   ✓ Migraine genes loaded: {len(genes):,}")
    return genes


# --------------------------------------------------------------------------------------
# Build master table
# --------------------------------------------------------------------------------------
def build_master(
    phenotype: str,
    base_dir: Path,
    use_directional_pathway: bool = False,
    apply_dircons_penalty: bool = False,
):
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

    required_de_cols = {"Gene", "Importance_Score"}
    missing_de = required_de_cols - set(de.columns)
    if missing_de:
        raise ValueError(f"DE ranking file missing columns: {sorted(missing_de)}")

    de["Gene"] = de["Gene"].astype(str).map(strip_ensg_version)
    if "Direction" not in de.columns:
        de["Direction"] = "Unknown"

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

    if "DirectionConsistencyScore" not in master.columns:
        master["DirectionConsistencyScore"] = np.nan
    master["DirectionConsistencyScore"] = pd.to_numeric(master["DirectionConsistencyScore"], errors="coerce")

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

    # Hub
    master["Hub_Score_raw"] = 0.0
    if hub_file.exists():
        print(f"\n📖 READING: {hub_file}")
        hub = pd.read_csv(hub_file)
        print(f"   ✓ Shape: {hub.shape}")

        if {"Gene", "Hub_Score"}.issubset(hub.columns):
            hub["Gene"] = hub["Gene"].astype(str).map(strip_ensg_version)
            hub_small = hub[["Gene", "Hub_Score"]].copy()
            hub_small["Hub_Score"] = pd.to_numeric(hub_small["Hub_Score"], errors="coerce").fillna(0.0)
            hub_small = hub_small.groupby("Gene", as_index=False)["Hub_Score"].max()
            master = master.merge(hub_small, on="Gene", how="left")
            master["Hub_Score_raw"] = pd.to_numeric(master.get("Hub_Score", 0), errors="coerce").fillna(0.0)
            print(f"   ✓ Hub scores loaded: {len(hub_small):,} genes")

    # Druggability
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
            dsmall["Druggability_Probability"] = pd.to_numeric(
                dsmall["Druggability_Probability"], errors="coerce"
            ).fillna(0.0)
            dsmall = dsmall.groupby("Gene", as_index=False)["Druggability_Probability"].max()
            master = master.merge(dsmall, on="Gene", how="left")
            master["Drug_Score_raw"] = pd.to_numeric(
                master.get("Druggability_Probability", 0), errors="coerce"
            ).fillna(0.0)
            print(f"   ✓ Druggability scores loaded: {len(dsmall):,} genes")

    # Normalized scores
    master["DE_Score_raw"] = pd.to_numeric(master["Importance_Score"], errors="coerce").fillna(0.0)
    master["DE_Score_norm"] = pct_rank(master["DE_Score_raw"])
    master["Path_Score_norm"] = pct_rank(master["PathScore_used_raw"])
    master["Drug_Score_norm"] = pct_rank(master["Drug_Score_raw"])
    master["Hub_Score_norm"] = pct_rank(master["Hub_Score_raw"])

    print(f"\n✓ Master dataset ready: {master.shape[0]:,} genes")
    print("✓ Percentile normalization applied (0 for missing/zero evidence)")

    return master, files_dir, ranking_dir


# --------------------------------------------------------------------------------------
# Ranking evaluation
# --------------------------------------------------------------------------------------
def evaluate_one_ranking(
    df: pd.DataFrame,
    score_col: str,
    label: str,
    test_pos: set,
    known_pos: set,
    universe_n: int,
    topk_list: list,
    focus_k: int,
):
    sdf = df.sort_values(score_col, ascending=False).reset_index(drop=True).copy()
    ranked_genes = sdf["Gene"].tolist()
    scores = sdf[score_col].to_numpy()

    y_test = np.array([1 if g in test_pos else 0 for g in ranked_genes], dtype=int)
    y_mig = np.array([1 if g in known_pos else 0 for g in ranked_genes], dtype=int)

    test_auc, test_pr = auc_pr(y_test, scores)
    mig_auc, mig_pr = auc_pr(y_mig, scores)

    print_block(f"📊 RANKING: {label}")
    print("\n  Overall Performance:")
    print(f"    TEST:       ROC-AUC={fmt_metric(test_auc)}, PR-AUC={fmt_metric(test_pr)}")
    print(f"    MIGRAINE:   ROC-AUC={fmt_metric(mig_auc)}, PR-AUC={fmt_metric(mig_pr)}")

    print_topk_table(ranked_genes, test_pos, universe_n, topk_list, "Enrichment vs TEST (Held-out replication)")
    print_topk_table(ranked_genes, known_pos, universe_n, topk_list, "Enrichment vs MIGRAINE (Disease genes)")

    obs_t, exp_t, fe_t, prec_t, rec_t = topk_stats(ranked_genes, test_pos, universe_n, focus_k)
    obs_m, exp_m, fe_m, prec_m, rec_m = topk_stats(ranked_genes, known_pos, universe_n, focus_k)

    return {
        "Ranking": label,
        "Test_ROC_AUC": test_auc,
        "Test_PR_AUC": test_pr,
        "Migraine_ROC_AUC": mig_auc,
        "Migraine_PR_AUC": mig_pr,
        f"Top{focus_k}_Test_Obs": obs_t,
        f"Top{focus_k}_Test_Exp": exp_t,
        f"Top{focus_k}_Test_FE": fe_t,
        f"Top{focus_k}_Test_Precision": prec_t,
        f"Top{focus_k}_Mig_Obs": obs_m,
        f"Top{focus_k}_Mig_Exp": exp_m,
        f"Top{focus_k}_Mig_FE": fe_m,
        f"Top{focus_k}_Mig_Precision": prec_m,
        "_ranked_df": sdf,
    }


def add_integrated_score(df: pd.DataFrame, w_de: float, w_path: float, w_drug: float, w_hub: float):
    out = df.copy()
    out["Integrated_Score_norm"] = (
        w_de * out["DE_Score_norm"]
        + w_path * out["Path_Score_norm"]
        + w_drug * out["Drug_Score_norm"]
        + w_hub * out["Hub_Score_norm"]
    )
    return out


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Controlled baseline comparison of gene-ranking strategies.")
    ap.add_argument("phenotype")
    ap.add_argument("--base-dir", default="/data/ascher02/uqmmune1/ANNOVAR")

    ap.add_argument("--fdr", type=float, default=0.10)
    ap.add_argument("--effect", type=float, default=0.50)

    ap.add_argument("--top-k", type=int, default=200, help="Top-K for enrichment summary (default: 200)")
    ap.add_argument("--use-directional-pathway", action="store_true")
    ap.add_argument("--apply-dircons-penalty", action="store_true")
    ap.add_argument("--val-keywords", default="validation,val")
    ap.add_argument("--test-keywords", default="test")

    # Fixed manuscript-facing integrated score by default
    ap.add_argument("--w-de", type=float, default=0.45)
    ap.add_argument("--w-path", type=float, default=0.25)
    ap.add_argument("--w-drug", type=float, default=0.25)
    ap.add_argument("--w-hub", type=float, default=0.05)

    args = ap.parse_args()

    phenotype = args.phenotype
    base_dir = Path(args.base_dir)
    focus_k = args.top_k

    print_block("📁 LOADING DATA FILES")
    master, files_dir, ranking_dir = build_master(
        phenotype=phenotype,
        base_dir=base_dir,
        use_directional_pathway=args.use_directional_pathway,
        apply_dircons_penalty=args.apply_dircons_penalty,
    )

    volcano_file = files_dir / "combined_volcano_data_all_models.csv"
    migraine_file = files_dir / "migraine_genes.csv"

    val_kw = [s.strip() for s in args.val_keywords.split(",") if s.strip()]
    test_kw = [s.strip() for s in args.test_keywords.split(",") if s.strip()]

    val_pos, volcano_universe = volcano_universe_and_positives(volcano_file, val_kw, args.fdr, args.effect)
    test_pos, _ = volcano_universe_and_positives(volcano_file, test_kw, args.fdr, args.effect)
    known = load_migraine_genes(migraine_file)

    master["Gene"] = master["Gene"].astype(str).map(strip_ensg_version)
    eval_df = master[master["Gene"].isin(volcano_universe)].copy().reset_index(drop=True)

    eval_universe = set(eval_df["Gene"])
    val_pos_eval = val_pos & eval_universe
    test_pos_eval = test_pos & eval_universe
    known_eval = known & eval_universe
    universe_n_eval = len(eval_universe)

    # Integrated score
    eval_df = add_integrated_score(
        eval_df,
        w_de=args.w_de,
        w_path=args.w_path,
        w_drug=args.w_drug,
        w_hub=args.w_hub,
    )

    topk_list = sorted(set([50, 100, focus_k, 500, 1000, 2000]))
    topk_list = [k for k in topk_list if k <= len(eval_df)]

    print_block("📋 DATA SUMMARY")
    print(f"Phenotype: {phenotype}")
    print(f"\n🎯 Focus Top-K: {focus_k} genes")
    print(f"\n📊 UNIVERSE BREAKDOWN:")
    print(f"   Volcano universe: {len(volcano_universe):,} genes")
    print(f"   Evaluation universe: {universe_n_eval:,} genes (genes actually ranked)")
    print(f"\n📊 DATA SPLITS (in evaluation universe):")
    print(f"   TRAIN+VAL positives: {len(val_pos):,} total, {len(val_pos_eval):,} in evaluation universe")
    print(f"   TEST positives:      {len(test_pos):,} total, {len(test_pos_eval):,} in evaluation universe")
    print(f"   Migraine genes:      {len(known):,} total, {len(known_eval):,} in evaluation universe")
    print(f"\n📌 Integrated score weights:")
    print(f"   DE={args.w_de:.2f}, Pathway={args.w_path:.2f}, Druggability={args.w_drug:.2f}, Hub={args.w_hub:.2f}")
    print("\n⚠️ Expected counts are computed using the evaluation universe only.")

    rankings = [
        ("DE_Score_norm", "DE only"),
        ("Path_Score_norm", "Pathway only"),
        ("Hub_Score_norm", "Hub only"),
        ("Drug_Score_norm", "Druggability only"),
        ("Integrated_Score_norm", "Integrated"),
    ]

    print_block("STEP 1: BASELINE COMPARISON OF GENE-RANKING STRATEGIES")

    results = []
    ranked_outputs = {}
    for score_col, label in rankings:
        res = evaluate_one_ranking(
            eval_df,
            score_col=score_col,
            label=label,
            test_pos=test_pos_eval,
            known_pos=known_eval,
            universe_n=universe_n_eval,
            topk_list=topk_list,
            focus_k=focus_k,
        )
        ranked_outputs[label] = res.pop("_ranked_df")
        results.append(res)

    summary_df = pd.DataFrame(results)

    # Main manuscript-facing table
    manuscript_cols = [
        "Ranking",
        "Test_ROC_AUC",
        "Test_PR_AUC",
        f"Top{focus_k}_Mig_Obs",
        f"Top{focus_k}_Mig_Exp",
        f"Top{focus_k}_Mig_FE",
        f"Top{focus_k}_Mig_Precision",
        f"Top{focus_k}_Test_Obs",
        f"Top{focus_k}_Test_Exp",
        f"Top{focus_k}_Test_FE",
        f"Top{focus_k}_Test_Precision",
    ]
    manuscript_df = summary_df[manuscript_cols].copy()

    print_block("📊 MANUSCRIPT SUMMARY TABLE")
    print(manuscript_df.to_string(index=False))

    # Save outputs
    out_dir = ensure_dir(ranking_dir / "BaselineComparison")
    summary_df.to_csv(out_dir / "baseline_comparison_full.csv", index=False)
    manuscript_df.to_csv(out_dir / "baseline_comparison_manuscript_table.csv", index=False)

    for label, rdf in ranked_outputs.items():
        safe = label.lower().replace(" ", "_")
        rdf.to_csv(out_dir / f"ranked_{safe}.csv", index=False)
        rdf.head(focus_k).to_csv(out_dir / f"top_{focus_k}_{safe}.csv", index=False)

    # Simple interpretation block
    print_block("📌 INTERPRETATION GUIDE")
    print("Use the saved manuscript table for your Results section.")
    print("For the manuscript, compare:")
    print("  • held-out replication: Test ROC-AUC and Test PR-AUC")
    print(f"  • migraine-biology enrichment: Top-{focus_k} migraine genes, expected, FE, precision")
    print("\nDo not claim the integrated score is automatically optimal unless it clearly")
    print("balances held-out replication and migraine-gene enrichment better than simpler baselines.")

    print_block("✅ COMPLETE")
    print(f"Output directory: {out_dir}")
    print("\nSaved files:")
    print("  • baseline_comparison_full.csv")
    print("  • baseline_comparison_manuscript_table.csv")
    print(f"  • top_{focus_k}_de_only.csv")
    print(f"  • top_{focus_k}_pathway_only.csv")
    print(f"  • top_{focus_k}_hub_only.csv")
    print(f"  • top_{focus_k}_druggability_only.csv")
    print(f"  • top_{focus_k}_integrated.csv")


if __name__ == "__main__":
    main()
