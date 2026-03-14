#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import argparse
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd


BASE_DIR_DEFAULT = Path("/data/ascher02/uqmmune1/ANNOVAR")
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
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_topk_list(text, default_vals):
    if text is None or str(text).strip() == "":
        return default_vals
    return [int(x) for x in str(text).split(",")]


def jaccard(a, b):
    a = set(a)
    b = set(b)
    if len(a) == 0 and len(b) == 0:
        return np.nan
    return len(a & b) / len(a | b)


def overlap_fraction(a, b):
    a = set(a)
    b = set(b)
    if len(a) == 0:
        return np.nan
    return len(a & b) / len(a)


def fmt(x):
    if pd.isna(x):
        return "NA"
    return f"{x:.3f}"


# =============================================================================
# LOADING
# =============================================================================
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
    df["Fold"] = pd.to_numeric(df["Fold"], errors="coerce")
    df["Dataset"] = df["Dataset"].astype(str)
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
# FOLD GENE RANKING
# =============================================================================
def build_fold_gene_rankings(volcano_df: pd.DataFrame, dataset_value="test") -> pd.DataFrame:
    df = volcano_df.copy()
    df = df[df["Dataset"].str.lower() == dataset_value.lower()].copy()

    if len(df) == 0:
        raise ValueError(f"No rows found for Dataset == {dataset_value}")

    grouped = (
        df.groupby(["Fold", "Gene"], dropna=False)
        .agg(
            Min_FDR=("FDR", "min"),
            Mean_FDR=("FDR", "mean"),
            Mean_neg_log10_FDR=("neg_log10_FDR_calc", "mean"),
            Max_neg_log10_FDR=("neg_log10_FDR_calc", "max"),
            Mean_abs_log2FC=("abs_log2FC", "mean"),
            Mean_log2FC=("Log2FoldChange", "mean"),
            N_support=("Gene", "size"),
        )
        .reset_index()
    )

    grouped["Direction"] = np.where(
        grouped["Mean_log2FC"] > 0,
        "higher in cases",
        np.where(grouped["Mean_log2FC"] < 0, "lower in cases", "unclear")
    )

    grouped["FoldGeneScore"] = (
        grouped["Mean_neg_log10_FDR"] * grouped["N_support"]
        + 0.1 * grouped["Mean_abs_log2FC"]
    )

    grouped = grouped.sort_values(
        ["Fold", "FoldGeneScore", "Min_FDR", "Mean_abs_log2FC"],
        ascending=[True, False, True, False]
    ).reset_index(drop=True)

    grouped["GeneRank"] = grouped.groupby("Fold").cumcount() + 1
    return grouped


# =============================================================================
# FOLD DRUG RANKING
# =============================================================================
def build_fold_drug_rankings(fold_gene_df: pd.DataFrame, drug_map_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build fold-specific drug rankings by mapping fold-ranked genes to drugs.

    IMPORTANT:
    We rename the gene-side rank before merging to avoid pandas suffix issues.
    """

    fold_gene_small = fold_gene_df[
        ["Fold", "Gene", "GeneRank", "FoldGeneScore", "Direction"]
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
            FoldDrugScore=("FoldGeneScore", "sum"),
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
# STABILITY
# =============================================================================
def compute_topk_stability(fold_rank_df: pd.DataFrame, topk_list):
    fold_to_drugs = {}
    for fold, sub in fold_rank_df.groupby("Fold"):
        sub = sub.sort_values("DrugRankWithinFold", ascending=True)
        fold_to_drugs[int(fold)] = sub["DrugNorm"].tolist()

    rows = []
    valid_folds = sorted(fold_to_drugs.keys())

    for k in topk_list:
        pairwise_jaccards = []
        pairwise_overlaps = []

        for f1, f2 in combinations(valid_folds, 2):
            ids1 = fold_to_drugs[f1][:k]
            ids2 = fold_to_drugs[f2][:k]
            pairwise_jaccards.append(jaccard(ids1, ids2))
            pairwise_overlaps.append(overlap_fraction(ids1, ids2))

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

        rows.append(
            {
                "TopK": k,
                "N_folds": len(valid_folds),
                "Mean_Pairwise_Jaccard": np.nanmean(pairwise_jaccards) if pairwise_jaccards else np.nan,
                "SD_Pairwise_Jaccard": np.nanstd(pairwise_jaccards, ddof=1) if len(pairwise_jaccards) > 1 else 0.0 if len(pairwise_jaccards) == 1 else np.nan,
                "Mean_Pairwise_Overlap": np.nanmean(pairwise_overlaps) if pairwise_overlaps else np.nan,
                "SD_Pairwise_Overlap": np.nanstd(pairwise_overlaps, ddof=1) if len(pairwise_overlaps) > 1 else 0.0 if len(pairwise_overlaps) == 1 else np.nan,
                "Intersection_Size_All_Folds": intersection_size,
                "Union_Size_All_Folds": union_size,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Step 5B: drug stability from fold gene rankings and rescued drug mappings")
    parser.add_argument("phenotype", help="Phenotype name, e.g. migraine")
    parser.add_argument("--base-dir", default=str(BASE_DIR_DEFAULT))
    parser.add_argument("--dataset", default="test")
    parser.add_argument("--drug-topk", default="20,50,100")
    args = parser.parse_args()

    phenotype = args.phenotype
    base_dir = Path(args.base_dir)
    drug_topk = parse_topk_list(args.drug_topk, DRUG_TOPK_DEFAULT)

    files_dir = base_dir / phenotype / "GeneDifferentialExpression" / "Files"
    ranking_dir = files_dir / "UltimateCompleteRankingAnalysis"
    drug_dir = ranking_dir / "DrugIntegration"
    output_dir = ranking_dir / "StabilityAnalysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    volcano_file = files_dir / "combined_volcano_data_all_models.csv"
    drug_table_file = drug_dir / "DrugDirectionalityTable_Rescued.csv"

    print("=" * 120)
    print("STEP 5B: DRUG STABILITY ANALYSIS")
    print("=" * 120)
    print(f"Phenotype:              {phenotype}")
    print(f"Volcano file:           {volcano_file}")
    print(f"Drug mapping file:      {drug_table_file}")
    print(f"Dataset filter:         {args.dataset}")
    print(f"Drug Top-K values:      {drug_topk}")
    print(f"Output dir:             {output_dir}")
    print("=" * 120)

    volcano_df = load_volcano_file(volcano_file)
    drug_map_df = load_drug_mapping(drug_table_file)

    print(f"Loaded volcano rows:              {len(volcano_df):,}")
    print(f"Loaded drug-gene mapping rows:    {len(drug_map_df):,}")

    fold_gene_df = build_fold_gene_rankings(volcano_df, dataset_value=args.dataset)
    fold_drug_df = build_fold_drug_rankings(fold_gene_df, drug_map_df)

    if len(fold_drug_df) == 0:
        raise ValueError("No fold-specific drug rankings could be built from the provided files.")

    stability_df = compute_topk_stability(fold_drug_df, drug_topk)

    fold_summary = (
        fold_drug_df.groupby("Fold", dropna=False)
        .agg(
            N_ranked_drugs=("DrugNorm", "nunique"),
            N_approved_drugs=("Approved_any", lambda s: (s.astype(str).str.lower() == "yes").sum()),
            N_known_migraine_drugs=("KnownMigraineDrug_any", lambda s: (s.astype(str).str.lower() == "yes").sum()),
        )
        .reset_index()
        .sort_values("Fold")
    )

    fold_drug_df.to_csv(output_dir / "DrugRankings_ByFold_FromVolcano.csv", index=False)
    stability_df.to_csv(output_dir / "Drug_Stability_Summary_FromVolcano.csv", index=False)
    fold_summary.to_csv(output_dir / "Drug_FoldSummary_FromVolcano.csv", index=False)

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

    recurrent_df = pd.concat(recurrent_rows, ignore_index=True)
    recurrent_df.to_csv(output_dir / "Drug_RecurrentAcrossFolds_FromVolcano.csv", index=False)

    with open(output_dir / "Manuscript_Text_DrugStability_FromVolcano.txt", "w", encoding="utf-8") as f:
        lines = []
        lines.append("STEP 5B MANUSCRIPT NOTES")
        lines.append("")
        for _, row in stability_df.iterrows():
            lines.append(
                f"Top-{int(row['TopK'])} drugs: mean pairwise Jaccard = {fmt(row['Mean_Pairwise_Jaccard'])}, "
                f"mean pairwise overlap = {fmt(row['Mean_Pairwise_Overlap'])}, "
                f"intersection across all folds = {int(row['Intersection_Size_All_Folds']) if pd.notna(row['Intersection_Size_All_Folds']) else 'NA'}."
            )
        f.write("\n".join(lines))

    print("\nOUTPUTS WRITTEN")
    print(f"  {output_dir / 'DrugRankings_ByFold_FromVolcano.csv'}")
    print(f"  {output_dir / 'Drug_Stability_Summary_FromVolcano.csv'}")
    print(f"  {output_dir / 'Drug_FoldSummary_FromVolcano.csv'}")
    print(f"  {output_dir / 'Drug_RecurrentAcrossFolds_FromVolcano.csv'}")
    print(f"  {output_dir / 'Manuscript_Text_DrugStability_FromVolcano.txt'}")

    print("\nDRUG STABILITY SUMMARY")
    print(stability_df.to_string(index=False))

    print("\nDRUG FOLD SUMMARY")
    print(fold_summary.to_string(index=False))


if __name__ == "__main__":
    main()