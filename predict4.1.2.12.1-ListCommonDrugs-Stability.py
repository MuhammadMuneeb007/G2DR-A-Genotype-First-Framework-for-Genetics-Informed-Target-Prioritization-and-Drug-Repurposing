#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import math
import argparse
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


BASE_DIR_DEFAULT = Path("/data/ascher02/uqmmune1/ANNOVAR")
GENE_TOPK_DEFAULT = [50, 100, 200]
DRUG_TOPK_DEFAULT = [20, 50, 100]


# =============================================================================
# HELPERS
# =============================================================================
def safe_str(x):
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"", "nan", "none", "null"}:
        return ""
    return s


def strip_ensg_version(x):
    return re.sub(r"\.\d+$", "", safe_str(x))


def normalize_drug_name(x):
    s = safe_str(x).lower()
    if not s:
        return ""

    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"\[.*?\]", " ", s)
    s = s.replace("&", " and ")
    s = re.sub(r"[/,+;]", " ", s)
    s = re.sub(r"\b\d+(\.\d+)?\s*(mg|mcg|g|ml|iu|units?)\b", " ", s)

    salt_words = [
        "hydrochloride", "hcl", "sodium", "potassium", "calcium",
        "succinate", "tartrate", "maleate", "phosphate", "sulfate",
        "acetate", "chloride", "nitrate", "mesylate", "besylate",
        "benzoate", "bromide", "citrate", "lactate"
    ]
    s = re.sub(r"\b(" + "|".join(map(re.escape, salt_words)) + r")\b", " ", s)

    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_topk_list(text, default_vals):
    if text is None or str(text).strip() == "":
        return default_vals
    return [int(x.strip()) for x in str(text).split(",") if str(x).strip()]


def jaccard(a, b):
    a = set(a)
    b = set(b)
    if len(a) == 0 and len(b) == 0:
        return np.nan
    return len(a & b) / len(a | b)


def overlap_count(a, b):
    a = set(a)
    b = set(b)
    return len(a & b)


def overlap_fraction(a, b):
    a = set(a)
    b = set(b)
    if len(a) == 0:
        return np.nan
    return len(a & b) / len(a)


def mean_sd_ci(vals):
    vals = pd.Series(vals).dropna().astype(float)
    n = len(vals)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan
    mean = vals.mean()
    sd = vals.std(ddof=1) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    ci_low = mean - 1.96 * se
    ci_high = mean + 1.96 * se
    return mean, sd, ci_low, ci_high


def fmt(x):
    if pd.isna(x):
        return "NA"
    return f"{x:.3f}"


def print_block(title):
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)


# =============================================================================
# LOADING
# =============================================================================
def load_known_migraine_genes(migraine_genes_file: Path) -> set:
    if not migraine_genes_file.exists():
        print(f"WARNING: migraine gene file not found: {migraine_genes_file}")
        return set()

    df = pd.read_csv(migraine_genes_file)
    if "ensembl_gene_id" not in df.columns:
        print(f"WARNING: 'ensembl_gene_id' not found in {migraine_genes_file}")
        return set()

    genes = set(df["ensembl_gene_id"].dropna().astype(str).map(strip_ensg_version))
    return {g for g in genes if g}


def load_volcano_file(volcano_file: Path) -> pd.DataFrame:
    if not volcano_file.exists():
        raise FileNotFoundError(f"Volcano file not found: {volcano_file}")

    df = pd.read_csv(volcano_file)

    required = ["Gene", "Fold", "Dataset", "FDR", "PValue", "Log2FoldChange"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in volcano file: {missing}")

    df = df.copy()
    df["Gene"] = df["Gene"].astype(str).map(strip_ensg_version)
    df["Dataset"] = df["Dataset"].astype(str)
    df["Fold"] = pd.to_numeric(df["Fold"], errors="coerce")
    df["FDR"] = pd.to_numeric(df["FDR"], errors="coerce")
    df["PValue"] = pd.to_numeric(df["PValue"], errors="coerce")
    df["Log2FoldChange"] = pd.to_numeric(df["Log2FoldChange"], errors="coerce")

    eps = 1e-300
    df["neg_log10_FDR_calc"] = -np.log10(df["FDR"].clip(lower=eps))
    df["abs_log2FC"] = df["Log2FoldChange"].abs()
    return df


def load_drug_mapping(drug_table_file: Path) -> pd.DataFrame:
    if not drug_table_file.exists():
        raise FileNotFoundError(f"Drug directionality file not found: {drug_table_file}")

    df = pd.read_csv(drug_table_file)

    required = ["DrugName", "Gene"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in drug table: {missing}")

    df = df.copy()
    df["Gene"] = df["Gene"].astype(str).map(strip_ensg_version)
    df["DrugName"] = df["DrugName"].astype(str)
    df["DrugNorm"] = df["DrugName"].map(normalize_drug_name)

    for col, default in [
        ("DrugRank", np.nan),
        ("DrugScore", np.nan),
        ("Approved", ""),
        ("KnownMigraineDrug", "No"),
        ("DirectionMatch", ""),
        ("ActionType", ""),
        ("Symbol", ""),
    ]:
        if col not in df.columns:
            df[col] = default

    df = df[(df["DrugNorm"] != "") & (df["Gene"] != "")].copy()
    return df


# =============================================================================
# FOLD-WISE GENE RANKING
# =============================================================================
def build_fold_gene_rankings(volcano_df: pd.DataFrame, dataset_value="test") -> pd.DataFrame:
    df = volcano_df.copy()
    df = df[df["Dataset"].str.lower() == dataset_value.lower()].copy()

    if len(df) == 0:
        raise ValueError(f"No rows found for Dataset == {dataset_value}")

    agg_dict = {
        "Min_FDR": ("FDR", "min"),
        "Mean_FDR": ("FDR", "mean"),
        "Mean_neg_log10_FDR": ("neg_log10_FDR_calc", "mean"),
        "Max_neg_log10_FDR": ("neg_log10_FDR_calc", "max"),
        "Mean_abs_log2FC": ("abs_log2FC", "mean"),
        "Mean_log2FC": ("Log2FoldChange", "mean"),
        "N_support": ("Gene", "size"),
    }
    if "Tissue" in df.columns:
        agg_dict["N_tissues"] = ("Tissue", "nunique")
    else:
        agg_dict["N_tissues"] = ("Gene", "size")
    if "Method" in df.columns:
        agg_dict["N_methods"] = ("Method", "nunique")
    else:
        agg_dict["N_methods"] = ("Gene", "size")
    if "Database" in df.columns:
        agg_dict["N_databases"] = ("Database", "nunique")
    else:
        agg_dict["N_databases"] = ("Gene", "size")

    grouped = (
        df.groupby(["Fold", "Gene"], dropna=False)
        .agg(**agg_dict)
        .reset_index()
    )

    grouped["Direction"] = np.where(
        grouped["Mean_log2FC"] > 0,
        "higher in cases",
        np.where(grouped["Mean_log2FC"] < 0, "lower in cases", "unclear")
    )

    grouped["FoldScore"] = (
        grouped["Mean_neg_log10_FDR"] * grouped["N_support"]
        + 0.1 * grouped["Mean_abs_log2FC"]
    )

    grouped = grouped.sort_values(
        ["Fold", "FoldScore", "Min_FDR", "Mean_abs_log2FC"],
        ascending=[True, False, True, False]
    ).reset_index(drop=True)

    grouped["GeneRank"] = grouped.groupby("Fold").cumcount() + 1
    return grouped


# =============================================================================
# GENE STABILITY
# =============================================================================
def compute_gene_topk_stability(fold_rank_df: pd.DataFrame, topk_list):
    fold_to_genes = {}
    for fold, sub in fold_rank_df.groupby("Fold"):
        sub = sub.sort_values("GeneRank", ascending=True)
        fold_to_genes[int(fold)] = sub["Gene"].tolist()

    rows = []
    valid_folds = sorted(fold_to_genes.keys())

    for k in topk_list:
        pairwise_jaccards = []
        pairwise_overlap_fracs = []
        pairwise_overlap_counts = []

        for f1, f2 in combinations(valid_folds, 2):
            ids1 = fold_to_genes[f1][:k]
            ids2 = fold_to_genes[f2][:k]

            pairwise_jaccards.append(jaccard(ids1, ids2))
            pairwise_overlap_fracs.append(overlap_fraction(ids1, ids2))
            pairwise_overlap_counts.append(overlap_count(ids1, ids2))

        all_sets = [set(fold_to_genes[f][:k]) for f in valid_folds]
        if len(all_sets) >= 1:
            if len(all_sets) == 1:
                intersection_size = len(all_sets[0])
                union_size = len(all_sets[0])
            else:
                intersection_size = len(set.intersection(*all_sets))
                union_size = len(set.union(*all_sets))
        else:
            intersection_size = np.nan
            union_size = np.nan

        jac_mean, jac_sd, jac_ci_low, jac_ci_high = mean_sd_ci(pairwise_jaccards)
        of_mean, of_sd, of_ci_low, of_ci_high = mean_sd_ci(pairwise_overlap_fracs)
        oc_mean, oc_sd, oc_ci_low, oc_ci_high = mean_sd_ci(pairwise_overlap_counts)

        rows.append(
            {
                "TopK": k,
                "N_folds": len(valid_folds),

                "Mean_Pairwise_Jaccard": jac_mean,
                "SD_Pairwise_Jaccard": jac_sd,
                "CI95_Low_Jaccard": jac_ci_low,
                "CI95_High_Jaccard": jac_ci_high,

                "Mean_Pairwise_OverlapFraction": of_mean,
                "SD_Pairwise_OverlapFraction": of_sd,
                "CI95_Low_OverlapFraction": of_ci_low,
                "CI95_High_OverlapFraction": of_ci_high,

                "Mean_Pairwise_OverlapCount": oc_mean,
                "SD_Pairwise_OverlapCount": oc_sd,
                "CI95_Low_OverlapCount": oc_ci_low,
                "CI95_High_OverlapCount": oc_ci_high,

                "Intersection_Size_All_Folds": intersection_size,
                "Union_Size_All_Folds": union_size,
            }
        )

    return pd.DataFrame(rows)


def compute_fold_gene_performance(fold_rank_df: pd.DataFrame, known_genes: set) -> pd.DataFrame:
    rows = []

    for fold, sub in fold_rank_df.groupby("Fold"):
        sub = sub.copy().sort_values("GeneRank", ascending=True)
        sub["KnownMigraineGene"] = sub["Gene"].isin(known_genes).astype(int)
        sub["ScoreForEval"] = sub["FoldScore"].astype(float)

        y_true = sub["KnownMigraineGene"].values
        y_score = sub["ScoreForEval"].values

        roc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) >= 2 else np.nan
        pr = average_precision_score(y_true, y_score) if y_true.sum() > 0 else np.nan

        rows.append(
            {
                "Fold": int(fold),
                "N_genes": len(sub),
                "N_known_migraine_genes": int(y_true.sum()),
                "ROC_AUC": roc,
                "PR_AUC": pr,
            }
        )

    return pd.DataFrame(rows).sort_values("Fold")


# =============================================================================
# FOLD-WISE DRUG RANKING
# =============================================================================
def build_fold_drug_rankings(fold_gene_df: pd.DataFrame, drug_map_df: pd.DataFrame) -> pd.DataFrame:
    fold_gene_small = fold_gene_df[
        ["Fold", "Gene", "GeneRank", "FoldScore", "Direction"]
    ].copy()

    fold_gene_small = fold_gene_small.rename(
        columns={
            "GeneRank": "GeneRank_fold",
            "Direction": "GeneDirection_fold",
        }
    )

    keep_cols = ["Gene", "DrugNorm", "DrugName"]
    for c in ["Approved", "KnownMigraineDrug", "DirectionMatch", "Symbol"]:
        if c in drug_map_df.columns:
            keep_cols.append(c)

    drug_map_small = drug_map_df[keep_cols].copy()
    merged = fold_gene_small.merge(drug_map_small, on="Gene", how="inner")

    if len(merged) == 0:
        return pd.DataFrame()

    for c, default in [
        ("Approved", ""),
        ("KnownMigraineDrug", "No"),
        ("DirectionMatch", ""),
        ("Symbol", ""),
    ]:
        if c not in merged.columns:
            merged[c] = default
        merged[c] = merged[c].fillna(default)

    drug_fold = (
        merged.groupby(["Fold", "DrugNorm", "DrugName"], dropna=False)
        .agg(
            FoldDrugScore=("FoldScore", "sum"),
            N_target_genes=("Gene", "nunique"),
            Best_gene_rank=("GeneRank_fold", "min"),
            Mean_gene_rank=("GeneRank_fold", "mean"),
            N_consistent_pairs=("DirectionMatch", lambda s: (s.astype(str).str.lower() == "consistent").sum()),
            N_inconsistent_pairs=("DirectionMatch", lambda s: (s.astype(str).str.lower() == "inconsistent").sum()),
            N_unclear_pairs=("DirectionMatch", lambda s: (s.astype(str).str.lower() == "unclear").sum()),
            Approved_any=("Approved", lambda s: "Yes" if (s.astype(str).str.lower() == "yes").any() else "No"),
            KnownMigraineDrug_any=("KnownMigraineDrug", lambda s: "Yes" if (s.astype(str).str.lower() == "yes").any() else "No"),
            ExampleGene=("Symbol", lambda s: next((x for x in s if safe_str(x)), "")),
        )
        .reset_index()
    )

    drug_fold = drug_fold.sort_values(
        ["Fold", "FoldDrugScore", "N_target_genes", "Best_gene_rank"],
        ascending=[True, False, False, True]
    ).reset_index(drop=True)

    drug_fold["DrugRankWithinFold"] = drug_fold.groupby("Fold").cumcount() + 1
    return drug_fold


# =============================================================================
# DRUG STABILITY
# =============================================================================
def compute_drug_topk_stability(fold_rank_df: pd.DataFrame, topk_list):
    fold_to_drugs = {}
    for fold, sub in fold_rank_df.groupby("Fold"):
        sub = sub.sort_values("DrugRankWithinFold", ascending=True)
        fold_to_drugs[int(fold)] = sub["DrugNorm"].tolist()

    rows = []
    valid_folds = sorted(fold_to_drugs.keys())

    for k in topk_list:
        pairwise_jaccards = []
        pairwise_overlap_fracs = []
        pairwise_overlap_counts = []

        for f1, f2 in combinations(valid_folds, 2):
            ids1 = fold_to_drugs[f1][:k]
            ids2 = fold_to_drugs[f2][:k]

            pairwise_jaccards.append(jaccard(ids1, ids2))
            pairwise_overlap_fracs.append(overlap_fraction(ids1, ids2))
            pairwise_overlap_counts.append(overlap_count(ids1, ids2))

        all_sets = [set(fold_to_drugs[f][:k]) for f in valid_folds]
        if len(all_sets) >= 1:
            if len(all_sets) == 1:
                intersection_size = len(all_sets[0])
                union_size = len(all_sets[0])
            else:
                intersection_size = len(set.intersection(*all_sets))
                union_size = len(set.union(*all_sets))
        else:
            intersection_size = np.nan
            union_size = np.nan

        jac_mean, jac_sd, jac_ci_low, jac_ci_high = mean_sd_ci(pairwise_jaccards)
        of_mean, of_sd, of_ci_low, of_ci_high = mean_sd_ci(pairwise_overlap_fracs)
        oc_mean, oc_sd, oc_ci_low, oc_ci_high = mean_sd_ci(pairwise_overlap_counts)

        rows.append(
            {
                "TopK": k,
                "N_folds": len(valid_folds),

                "Mean_Pairwise_Jaccard": jac_mean,
                "SD_Pairwise_Jaccard": jac_sd,
                "CI95_Low_Jaccard": jac_ci_low,
                "CI95_High_Jaccard": jac_ci_high,

                "Mean_Pairwise_OverlapFraction": of_mean,
                "SD_Pairwise_OverlapFraction": of_sd,
                "CI95_Low_OverlapFraction": of_ci_low,
                "CI95_High_OverlapFraction": of_ci_high,

                "Mean_Pairwise_OverlapCount": oc_mean,
                "SD_Pairwise_OverlapCount": oc_sd,
                "CI95_Low_OverlapCount": oc_ci_low,
                "CI95_High_OverlapCount": oc_ci_high,

                "Intersection_Size_All_Folds": intersection_size,
                "Union_Size_All_Folds": union_size,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Combined Step 5 stability / uncertainty analysis for genes and drugs")
    parser.add_argument("phenotype", help="Phenotype name, e.g. migraine")
    parser.add_argument("--base-dir", default=str(BASE_DIR_DEFAULT))
    parser.add_argument("--dataset", default="test")
    parser.add_argument("--gene-topk", default="50,100,200")
    parser.add_argument("--drug-topk", default="20,50,100")
    args = parser.parse_args()

    phenotype = args.phenotype
    base_dir = Path(args.base_dir)
    gene_topk = parse_topk_list(args.gene_topk, GENE_TOPK_DEFAULT)
    drug_topk = parse_topk_list(args.drug_topk, DRUG_TOPK_DEFAULT)

    files_dir = base_dir / phenotype / "GeneDifferentialExpression" / "Files"
    ranking_dir = files_dir / "UltimateCompleteRankingAnalysis"
    drug_dir = ranking_dir / "DrugIntegration"
    output_dir = ranking_dir / "StabilityAnalysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    volcano_file = files_dir / "combined_volcano_data_all_models.csv"
    migraine_genes_file = files_dir / "migraine_genes.csv"
    if not migraine_genes_file.exists():
        migraine_genes_file = base_dir / "migraine_genes.csv"
    drug_table_file = drug_dir / "DrugDirectionalityTable_Rescued.csv"

    print_block("STEP 5 COMBINED: STABILITY AND UNCERTAINTY ANALYSIS")
    print(f"Phenotype:              {phenotype}")
    print(f"Volcano file:           {volcano_file}")
    print(f"Migraine genes file:    {migraine_genes_file}")
    print(f"Drug mapping file:      {drug_table_file}")
    print(f"Dataset filter:         {args.dataset}")
    print(f"Gene Top-K values:      {gene_topk}")
    print(f"Drug Top-K values:      {drug_topk}")
    print(f"Output dir:             {output_dir}")

    volcano_df = load_volcano_file(volcano_file)
    known_genes = load_known_migraine_genes(migraine_genes_file)
    drug_map_df = load_drug_mapping(drug_table_file)

    print_block("INPUT SUMMARY")
    print(f"Loaded volcano rows:              {len(volcano_df):,}")
    print(f"Known migraine genes available:   {len(known_genes):,}")
    print(f"Loaded drug-gene mapping rows:    {len(drug_map_df):,}")

    # ----------------------------- Gene analysis ------------------------------
    fold_gene_df = build_fold_gene_rankings(volcano_df, dataset_value=args.dataset)
    gene_stability_df = compute_gene_topk_stability(fold_gene_df, gene_topk)
    gene_perf_df = compute_fold_gene_performance(fold_gene_df, known_genes)

    roc_mean, roc_sd, roc_ci_l, roc_ci_u = mean_sd_ci(gene_perf_df["ROC_AUC"])
    pr_mean, pr_sd, pr_ci_l, pr_ci_u = mean_sd_ci(gene_perf_df["PR_AUC"])

    gene_uncertainty_df = pd.DataFrame(
        [
            {
                "Metric": "ROC-AUC",
                "N_folds": gene_perf_df["ROC_AUC"].notna().sum(),
                "Mean": roc_mean,
                "SD": roc_sd,
                "CI95_Lower": roc_ci_l,
                "CI95_Upper": roc_ci_u,
            },
            {
                "Metric": "PR-AUC",
                "N_folds": gene_perf_df["PR_AUC"].notna().sum(),
                "Mean": pr_mean,
                "SD": pr_sd,
                "CI95_Lower": pr_ci_l,
                "CI95_Upper": pr_ci_u,
            },
        ]
    )

    # ----------------------------- Drug analysis ------------------------------
    fold_drug_df = build_fold_drug_rankings(fold_gene_df, drug_map_df)
    if len(fold_drug_df) == 0:
        raise ValueError("No fold-specific drug rankings could be built from the provided files.")

    drug_stability_df = compute_drug_topk_stability(fold_drug_df, drug_topk)

    drug_fold_summary_df = (
        fold_drug_df.groupby("Fold", dropna=False)
        .agg(
            N_ranked_drugs=("DrugNorm", "nunique"),
            N_approved_drugs=("Approved_any", lambda s: (s.astype(str).str.lower() == "yes").sum()),
            N_known_migraine_drugs=("KnownMigraineDrug_any", lambda s: (s.astype(str).str.lower() == "yes").sum()),
        )
        .reset_index()
        .sort_values("Fold")
    )

    recurrent_rows = []
    for k in drug_topk:
        sub = fold_drug_df[fold_drug_df["DrugRankWithinFold"] <= k].copy()
        tmp = (
            sub.groupby(["DrugNorm", "DrugName"], dropna=False)
            .agg(
                FoldsPresent=("Fold", "nunique"),
                BestRank=("DrugRankWithinFold", "min"),
                MeanRank=("DrugRankWithinFold", "mean"),
                Approved=("Approved_any", lambda s: "Yes" if (s.astype(str).str.lower() == "yes").any() else "No"),
                KnownMigraineDrug=("KnownMigraineDrug_any", lambda s: "Yes" if (s.astype(str).str.lower() == "yes").any() else "No"),
                ExampleGene=("ExampleGene", lambda s: next((x for x in s if safe_str(x)), "")),
            )
            .reset_index()
            .sort_values(["FoldsPresent", "BestRank"], ascending=[False, True])
        )
        tmp["TopK"] = k
        recurrent_rows.append(tmp)

    recurrent_drugs_df = pd.concat(recurrent_rows, ignore_index=True)

    # ------------------------------- Save -------------------------------------
    fold_gene_df.to_csv(output_dir / "GeneRankings_ByFold_FromVolcano.csv", index=False)
    gene_stability_df.to_csv(output_dir / "Gene_Stability_Summary_FromVolcano.csv", index=False)
    gene_perf_df.to_csv(output_dir / "Gene_FoldPerformance_FromVolcano.csv", index=False)
    gene_uncertainty_df.to_csv(output_dir / "Gene_Uncertainty_Summary_FromVolcano.csv", index=False)

    for k in gene_topk:
        topk_df = fold_gene_df[fold_gene_df["GeneRank"] <= k].copy()
        topk_df.to_csv(output_dir / f"Top{k}_Genes_ByFold_FromVolcano.csv", index=False)

    fold_drug_df.to_csv(output_dir / "DrugRankings_ByFold_FromVolcano.csv", index=False)
    drug_stability_df.to_csv(output_dir / "Drug_Stability_Summary_FromVolcano.csv", index=False)
    drug_fold_summary_df.to_csv(output_dir / "Drug_FoldSummary_FromVolcano.csv", index=False)
    recurrent_drugs_df.to_csv(output_dir / "Drug_RecurrentAcrossFolds_FromVolcano.csv", index=False)

    notes = []
    notes.append("STEP 5 COMBINED MANUSCRIPT NOTES")
    notes.append("")
    notes.append(f"Dataset used for fold ranking: {args.dataset}")
    notes.append(f"Known migraine genes used for evaluation: {len(known_genes)}")
    notes.append("")

    notes.append("GENE STABILITY")
    for _, row in gene_stability_df.iterrows():
        notes.append(
            f"Top-{int(row['TopK'])} genes: mean pairwise Jaccard = {fmt(row['Mean_Pairwise_Jaccard'])} "
            f"(95% CI {fmt(row['CI95_Low_Jaccard'])}--{fmt(row['CI95_High_Jaccard'])}), "
            f"mean pairwise overlap count = {fmt(row['Mean_Pairwise_OverlapCount'])} "
            f"(95% CI {fmt(row['CI95_Low_OverlapCount'])}--{fmt(row['CI95_High_OverlapCount'])}), "
            f"intersection across all folds = "
            f"{int(row['Intersection_Size_All_Folds']) if pd.notna(row['Intersection_Size_All_Folds']) else 'NA'}."
        )
    notes.append(
        f"Fold-wise ROC-AUC mean +/- SD = {fmt(roc_mean)} +/- {fmt(roc_sd)} "
        f"(95% CI {fmt(roc_ci_l)}--{fmt(roc_ci_u)})."
    )
    notes.append(
        f"Fold-wise PR-AUC mean +/- SD = {fmt(pr_mean)} +/- {fmt(pr_sd)} "
        f"(95% CI {fmt(pr_ci_l)}--{fmt(pr_ci_u)})."
    )

    notes.append("")
    notes.append("DRUG STABILITY")
    for _, row in drug_stability_df.iterrows():
        notes.append(
            f"Top-{int(row['TopK'])} drugs: mean pairwise Jaccard = {fmt(row['Mean_Pairwise_Jaccard'])} "
            f"(95% CI {fmt(row['CI95_Low_Jaccard'])}--{fmt(row['CI95_High_Jaccard'])}), "
            f"mean pairwise overlap count = {fmt(row['Mean_Pairwise_OverlapCount'])} "
            f"(95% CI {fmt(row['CI95_Low_OverlapCount'])}--{fmt(row['CI95_High_OverlapCount'])}), "
            f"intersection across all folds = "
            f"{int(row['Intersection_Size_All_Folds']) if pd.notna(row['Intersection_Size_All_Folds']) else 'NA'}."
        )

    with open(output_dir / "Manuscript_Text_Combined_Stability.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(notes))

    # ------------------------------- Print ------------------------------------
    print_block("GENE STABILITY SUMMARY")
    print(gene_stability_df.to_string(index=False))

    print_block("GENE FOLD PERFORMANCE")
    print(gene_perf_df.to_string(index=False))

    print_block("GENE UNCERTAINTY SUMMARY")
    print(gene_uncertainty_df.to_string(index=False))

    print_block("DRUG STABILITY SUMMARY")
    print(drug_stability_df.to_string(index=False))

    print_block("DRUG FOLD SUMMARY")
    print(drug_fold_summary_df.to_string(index=False))

    print_block("MANUSCRIPT NOTES")
    print("\n".join(notes))

    print_block("OUTPUTS WRITTEN")
    print(f"  {output_dir / 'GeneRankings_ByFold_FromVolcano.csv'}")
    print(f"  {output_dir / 'Gene_Stability_Summary_FromVolcano.csv'}")
    print(f"  {output_dir / 'Gene_FoldPerformance_FromVolcano.csv'}")
    print(f"  {output_dir / 'Gene_Uncertainty_Summary_FromVolcano.csv'}")
    for k in gene_topk:
        print(f"  {output_dir / f'Top{k}_Genes_ByFold_FromVolcano.csv'}")
    print(f"  {output_dir / 'DrugRankings_ByFold_FromVolcano.csv'}")
    print(f"  {output_dir / 'Drug_Stability_Summary_FromVolcano.csv'}")
    print(f"  {output_dir / 'Drug_FoldSummary_FromVolcano.csv'}")
    print(f"  {output_dir / 'Drug_RecurrentAcrossFolds_FromVolcano.csv'}")
    print(f"  {output_dir / 'Manuscript_Text_Combined_Stability.txt'}")


if __name__ == "__main__":
    main()