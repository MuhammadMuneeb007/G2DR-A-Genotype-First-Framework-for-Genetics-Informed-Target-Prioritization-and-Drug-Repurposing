#!/usr/bin/env python3
"""
predict4.1.2.12.0-Diagnosis.py

Disease enrichment of drugs derived from TOP_N genes.

Key idea:
  1) Use TOP_{N}_1.csv to select the gene set
  2) Use your *already fetched* drug evidence table: GeneDrugTable_ALL.csv
     (produced by your DrugFinder pipeline; no API calls here)
  3) Build predicted drug set from those genes
  4) Compare against local merged drug–disease DB across ALL diseases
  5) Rank diseases by enrichment / p-value

Outputs:
  - DiseaseEnrichment_Top{N}.csv
  - DiseaseEnrichment_Top{N}_Hits.csv  (optional hits listing per disease)
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Optional, Tuple, List

import polars as pl

# -----------------------------
# Helpers
# -----------------------------
def pick_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    s = set(cols)
    for c in candidates:
        if c in s:
            return c
    # case-insensitive fallback
    lower_map = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def normalize_drug_expr(col: pl.Expr) -> pl.Expr:
    """
    Vectorized, conservative normalizer (Polars expressions only).
    IMPORTANT: does NOT drop drugs; only creates a key for matching.
    """
    salt_words = (
        "hydrochloride|hcl|sodium|potassium|calcium|succinate|tartrate|maleate|"
        "phosphate|sulfate|acetate|chloride|nitrate|mesylate|besylate|"
        "benzoate|bromide|citrate|lactate"
    )

    return (
        col.cast(pl.Utf8)
        .str.to_lowercase()
        .str.strip_chars()
        # remove bracketed parts
        .str.replace_all(r"\(.*?\)", " ")
        .str.replace_all(r"\[.*?\]", " ")
        # normalize separators
        .str.replace_all("&", " and ")
        .str.replace_all(r"[/,+;]", " ")
        # remove dosage tokens
        .str.replace_all(r"\b\d+(\.\d+)?\s*(mg|mcg|g|ml|iu|units?)\b", " ")
        # remove salts (conservative)
        .str.replace_all(rf"\b({salt_words})\b", " ")
        # keep alphanum
        .str.replace_all(r"[^a-z0-9]+", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def hypergeom_sf(x_minus_1: int, M: int, n: int, N: int) -> float:
    """
    Survival function: P(X >= x) where x_minus_1 = x-1 is provided.
    Using log-sum-exp exact computation (stable).
    M = universe size
    n = successes in population
    N = draws
    """
    # Need P(X >= x) => sf(x-1)
    x = x_minus_1 + 1
    if x <= 0:
        return 1.0
    if M <= 0 or n <= 0 or N <= 0:
        return 1.0

    if n > M:
        n = M
    if N > M:
        N = M

    max_x = min(n, N)
    if x > max_x:
        return 0.0

    # log C(a,b)
    def logC(a: int, b: int) -> float:
        if b < 0 or b > a:
            return float("-inf")
        return math.lgamma(a + 1) - math.lgamma(b + 1) - math.lgamma(a - b + 1)

    denom = logC(M, N)
    logs = []
    for k in range(x, max_x + 1):
        logs.append(logC(n, k) + logC(M - n, N - k) - denom)

    m = max(logs)
    s = sum(math.exp(li - m) for li in logs) * math.exp(m)
    # clamp
    if s < 0.0:
        return 0.0
    if s > 1.0:
        return 1.0
    return float(s)


# -----------------------------
# Core
# -----------------------------
def run(
    phenotype: str,
    base_dir: Path,
    top_genes: int,
    local_merged_db: Path,
    out_dir_override: Optional[Path],
    hits_max_diseases: int,
) -> None:
    base_dir = base_dir.resolve()

    # Paths
    genes_path = (
        base_dir / phenotype / "GeneDifferentialExpression" / "Files"
        / "UltimateCompleteRankingAnalysis" / "FinalIntegration"
        / f"TOP_{top_genes}_1.csv"
    )

    drug_integration_dir = (
        base_dir / phenotype / "GeneDifferentialExpression" / "Files"
        / "UltimateCompleteRankingAnalysis" / "DrugIntegration"
    )

    out_dir = out_dir_override if out_dir_override else drug_integration_dir
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    gene_drug_path = out_dir / "GeneDrugTable_ALL.csv"

    print("=" * 120)
    print(f"Phenotype: {phenotype}")
    print(f"Top genes: {top_genes}")
    print(f"Top genes file: {genes_path}")
    print(f"Gene→Drug cache table: {gene_drug_path}")
    print(f"Local merged DB: {local_merged_db}")
    print(f"Output dir: {out_dir}")
    print("=" * 120)

    if not genes_path.exists():
        raise FileNotFoundError(f"Missing TOP genes file: {genes_path}")
    if not gene_drug_path.exists():
        raise FileNotFoundError(
            f"Missing GeneDrugTable_ALL.csv at: {gene_drug_path}\n"
            f"Run your DrugFinder pipeline first to produce it for this phenotype."
        )
    if not local_merged_db.exists():
        raise FileNotFoundError(f"Missing local merged DB: {local_merged_db}")

    # -----------------------------
    # Load TOP genes
    # -----------------------------
    genes_df = pl.read_csv(genes_path, infer_schema_length=1000)
    gene_col = pick_col(genes_df.columns, ["Gene", "gene", "Ensembl_ID", "ensembl_gene_id", "ENSEMBL"])
    if gene_col is None:
        raise ValueError(f"Could not find gene column in {genes_path.name}. Columns={genes_df.columns}")

    top_genes_set = (
        genes_df.select(pl.col(gene_col).cast(pl.Utf8).str.strip_chars().alias("Gene"))
        .filter(pl.col("Gene").is_not_null() & (pl.col("Gene") != ""))
        .unique()
    )

    n_genes = top_genes_set.height
    print(f"✅ Loaded TOP genes: {n_genes:,}")

    # -----------------------------
    # Load Gene→Drug evidence table (your cached fetched drugs)
    # -----------------------------
    gd = pl.read_csv(gene_drug_path, infer_schema_length=2000, ignore_errors=True)
    # detect columns
    gd_gene_col = pick_col(gd.columns, ["Gene", "gene", "Ensembl_ID", "ensembl_gene_id"])
    gd_drugname_col = pick_col(gd.columns, ["DrugName", "drug_name", "DrugName_best", "name"])
    gd_drugnorm_col = pick_col(gd.columns, ["DrugNorm", "drug_norm", "DrugNorm_key"])

    if gd_gene_col is None:
        raise ValueError(f"GeneDrugTable_ALL.csv missing Gene column. Columns={gd.columns}")
    if gd_drugname_col is None and gd_drugnorm_col is None:
        raise ValueError(f"GeneDrugTable_ALL.csv missing DrugName/DrugNorm columns. Columns={gd.columns}")

    # Filter to selected genes
    gd2 = gd.join(top_genes_set, left_on=gd_gene_col, right_on="Gene", how="inner")

    # Create a drug matching key
    if gd_drugnorm_col:
        gd2 = gd2.with_columns(
            normalize_drug_expr(pl.col(gd_drugnorm_col)).alias("DrugNorm_key"),
        )
        # keep a display name too
        if gd_drugname_col:
            gd2 = gd2.with_columns(pl.col(gd_drugname_col).cast(pl.Utf8).alias("DrugName_disp"))
        else:
            gd2 = gd2.with_columns(pl.col(gd_drugnorm_col).cast(pl.Utf8).alias("DrugName_disp"))
    else:
        gd2 = gd2.with_columns(
            normalize_drug_expr(pl.col(gd_drugname_col)).alias("DrugNorm_key"),
            pl.col(gd_drugname_col).cast(pl.Utf8).alias("DrugName_disp"),
        )

    gd2 = gd2.filter(pl.col("DrugNorm_key").is_not_null() & (pl.col("DrugNorm_key") != ""))

    pred_drugs = (
        gd2.select(["DrugNorm_key", "DrugName_disp"])
        .group_by("DrugNorm_key")
        .agg(pl.col("DrugName_disp").drop_nulls().first().alias("DrugName_best"))
    )

    n_pred_total = pred_drugs.height
    print(f"✅ Predicted unique drugs (from TOP_{top_genes} genes): {n_pred_total:,}")

    # -----------------------------
    # Load local merged DB
    # -----------------------------
    db = pl.read_csv(local_merged_db, infer_schema_length=2000, ignore_errors=True)

    db_drug_col = pick_col(db.columns, ["drug_norm", "DrugNorm", "DrugNorm_key", "drug", "DrugName", "drug_name"])
    db_dis_col = pick_col(db.columns, ["disease_name", "DiseaseName", "Disease", "Disease_Description", "disease"])
    db_disid_col = pick_col(db.columns, ["disease_id", "DiseaseID", "DiseaseId", "diseaseId", "Disease_ID"])

    if db_drug_col is None or db_dis_col is None:
        raise ValueError(
            f"Local merged DB must have a drug column and disease name column.\n"
            f"Found columns: {db.columns}"
        )

    # Ensure normalized key exists in DB
    if db_drug_col.lower() in {"drug_norm", "drugnorm", "drugn_key", "drugnorm_key"} or "norm" in db_drug_col.lower():
        db2 = db.with_columns(normalize_drug_expr(pl.col(db_drug_col)).alias("DrugNorm_key"))
    else:
        db2 = db.with_columns(normalize_drug_expr(pl.col(db_drug_col)).alias("DrugNorm_key"))

    db2 = db2.filter(pl.col("DrugNorm_key").is_not_null() & (pl.col("DrugNorm_key") != ""))

    # standardize disease cols
    if db_disid_col:
        db2 = db2.with_columns(
            pl.col(db_dis_col).cast(pl.Utf8).alias("DiseaseName"),
            pl.col(db_disid_col).cast(pl.Utf8).alias("DiseaseId"),
        )
    else:
        db2 = db2.with_columns(
            pl.col(db_dis_col).cast(pl.Utf8).alias("DiseaseName"),
            pl.lit("").alias("DiseaseId"),
        )

    # Deduplicate to pairs (DrugNorm_key, DiseaseName, DiseaseId)
    pairs = db2.select(["DrugNorm_key", "DiseaseName", "DiseaseId"]).unique()
    print(f"✅ Local pairs (collapsed): {pairs.height:,}")

    # Universe size in DB
    universe_N = pairs.select(pl.col("DrugNorm_key").n_unique().alias("N")).item()
    print(f"✅ Universe drugs (normalized) in DB: {universe_N:,}")

    # -----------------------------
    # Map predicted drugs into DB universe (so your n is "mapped predicted")
    # -----------------------------
    # keep only predicted drugs that exist in DB universe
    pred_in_db = pred_drugs.join(
        pairs.select("DrugNorm_key").unique(),
        on="DrugNorm_key",
        how="inner",
    )
    n_mapped = pred_in_db.height
    print(f"✅ Predicted drugs that map into DB universe: {n_mapped:,} / {n_pred_total:,}")

    if n_mapped == 0:
        raise RuntimeError(
            "ZERO predicted drugs map into the local DB universe after normalization.\n"
            "This means your local merged DB drug naming doesn't match your predicted drugs.\n"
            "Fix: ensure your merged DB has a normalized drug column compatible with normalize_drug_expr."
        )

    # -----------------------------
    # Compute overlap per disease
    # -----------------------------
    # K = total unique drugs per disease in DB
    disease_K = (
        pairs.group_by(["DiseaseName", "DiseaseId"])
        .agg(pl.col("DrugNorm_key").n_unique().alias("K_disease_drugs"))
    )

    # x = overlap between predicted drugs and disease drugs
    overlap_pairs = pairs.join(pred_in_db.select("DrugNorm_key").unique(), on="DrugNorm_key", how="inner")

    disease_x = (
        overlap_pairs.group_by(["DiseaseName", "DiseaseId"])
        .agg(pl.col("DrugNorm_key").n_unique().alias("Overlap_x"))
    )

    # Merge + compute expectation and FE
    out = (
        disease_K.join(disease_x, on=["DiseaseName", "DiseaseId"], how="left")
        .with_columns(
            pl.col("Overlap_x").fill_null(0).cast(pl.Int64),
            pl.lit(int(universe_N)).alias("Universe_N"),
            pl.lit(int(n_mapped)).alias("Predicted_n"),
        )
        .with_columns(
            (pl.col("Predicted_n") * (pl.col("K_disease_drugs") / pl.col("Universe_N"))).alias("Expected"),
        )
        .with_columns(
            pl.when(pl.col("Expected") > 0)
            .then(pl.col("Overlap_x") / pl.col("Expected"))
            .otherwise(None)
            .alias("FoldEnrichment_FE")
        )
    )

    # Hypergeom p-values (python loop over collected columns; fast enough for ~10k diseases)
    # P(X >= x) with:
    #   M=Universe_N, n=K_disease_drugs, N=Predicted_n, x=Overlap_x
    cols = out.select(["Universe_N", "K_disease_drugs", "Predicted_n", "Overlap_x"]).to_dict(as_series=False)
    pvals = []
    for M, n, Ndraw, x in zip(cols["Universe_N"], cols["K_disease_drugs"], cols["Predicted_n"], cols["Overlap_x"]):
        # sf(x-1)
        p = hypergeom_sf(int(x) - 1, int(M), int(n), int(Ndraw)) if int(x) > 0 else 1.0
        pvals.append(p)

    out = out.with_columns(pl.Series("Hypergeom_p", pvals))

    # Rank diseases: by FE desc, then p asc, then overlap desc
    out = out.sort(
        by=["FoldEnrichment_FE", "Hypergeom_p", "Overlap_x", "K_disease_drugs"],
        descending=[True, False, True, False],
        nulls_last=True,
    )

    # Save main table
    out_file = out_dir / f"DiseaseEnrichment_Top{top_genes}.csv"
    out.write_csv(out_file)
    print(f"\n💾 Saved ranked disease enrichment: {out_file}")

    # -----------------------------
    # Optional: save hits (drug lists per disease) for top diseases
    # (avoid nested/list cols in CSV by writing one row per (disease, drug))
    # -----------------------------
    top_diseases = out.select(["DiseaseName", "DiseaseId"]).head(int(hits_max_diseases))

    hits = (
        overlap_pairs.join(top_diseases, on=["DiseaseName", "DiseaseId"], how="inner")
        .join(pred_in_db, on="DrugNorm_key", how="left")  # bring drug best name
        .select(["DiseaseName", "DiseaseId", "DrugNorm_key", "DrugName_best"])
        .sort(["DiseaseName", "DrugName_best"])
    )

    hits_file = out_dir / f"DiseaseEnrichment_Top{top_genes}_Hits.csv"
    hits.write_csv(hits_file)
    print(f"💾 Saved hits (top diseases only): {hits_file}")

    # Quick console summary
    print("\n" + "=" * 120)
    print("TOP 30 DISEASES BY ENRICHMENT")
    print("=" * 120)
    print(
        out.select(
            ["DiseaseName", "DiseaseId", "K_disease_drugs", "Overlap_x", "Expected", "FoldEnrichment_FE", "Hypergeom_p"]
        ).head(30).to_pandas().to_string(index=False)
    )

    print("\nNOTE (important): Expected overlap is NOT constant across diseases.")
    print("Expected = n * (K / N). K changes per disease, so expected changes per disease.")
    print("What is constant is: n (your predicted mapped drugs) and N (DB universe).")


def main():
    ap = argparse.ArgumentParser(description="Disease enrichment of predicted drugs from TOP_N genes (Polars)")
    ap.add_argument("phenotype", help="e.g. migraine")
    ap.add_argument("--base-dir", default="/data/ascher02/uqmmune1/ANNOVAR")
    ap.add_argument("--top-genes", type=int, required=True, help="Use TOP_{N}_1.csv gene list (N=top-genes)")

    ap.add_argument(
        "--local-merged-db",
        default="/data/ascher02/uqmmune1/ANNOVAR/AllDiseasesToDrugs/ALL_SOURCES_drug_disease_merged.csv",
        help="Merged drug–disease DB (local CSV)"
    )
    ap.add_argument("--outdir", default=None, help="Override output directory (default: DrugIntegration)")
    ap.add_argument("--hits-max-diseases", type=int, default=2000,
                    help="Write drug hits for top X diseases into Hits.csv (default: 2000)")

    args = ap.parse_args()

    run(
        phenotype=args.phenotype,
        base_dir=Path(args.base_dir),
        top_genes=int(args.top_genes),
        local_merged_db=Path(args.local_merged_db),
        out_dir_override=Path(args.outdir) if args.outdir else None,
        hits_max_diseases=int(args.hits_max_diseases),
    )


if __name__ == "__main__":
    main()
