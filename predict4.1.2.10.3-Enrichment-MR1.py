#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import math
import gzip
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# USER CONFIG
# =========================================================
BASE_DIR = Path("/data/ascher02/uqmmune1/ANNOVAR")
TOP_K_LIST = [50, 200, 500, 1000]

RUN_COLOC = True
COLOC_WINDOW_BP = 500_000
COLOC_PP4_THRESHOLD = 0.80

# Optional: set if you want proper cc coloc model
GWAS_CASE_FRACTION = None  # e.g. 53/733


# =========================================================
# BASIC HELPERS
# =========================================================
def strip_version(x):
    if pd.isna(x):
        return np.nan
    return str(x).split(".")[0].strip()


def norm_chr(x):
    s = str(x).replace("chr", "").strip()
    if s in {"X", "Y", "MT", "M"}:
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def parse_gtex_variant_id(variant_id):
    parts = str(variant_id).split("_")
    if len(parts) < 4:
        return None
    ch = norm_chr(parts[0])
    try:
        bp = int(parts[1])
    except Exception:
        return None
    ref = str(parts[2]).upper()
    alt = str(parts[3]).upper()
    return ch, bp, ref, alt


def normal_two_sided_p(z):
    return math.erfc(abs(z) / math.sqrt(2.0))


def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    out = np.full(len(p), np.nan)

    ok = np.isfinite(p)
    if ok.sum() == 0:
        return out

    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order]
    q = ranked * len(ranked) / (np.arange(1, len(ranked) + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1.0)

    temp = np.empty_like(q)
    temp[order] = q
    out[np.where(ok)[0]] = temp
    return out


def safe_mkdir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def is_palindromic(a1, a2):
    pair = {str(a1).upper(), str(a2).upper()}
    return pair == {"A", "T"} or pair == {"C", "G"}


# =========================================================
# PATHS
# =========================================================
def get_paths(phenotype):
    pheno_dir = BASE_DIR / phenotype

    ranked_file = pheno_dir / "GeneDifferentialExpression" / "Files" / "UltimateCompleteRankingAnalysis" / "RANKED_composite.csv"
    gwas_file = pheno_dir / f"{phenotype}.txt"
    gtex_dir = pheno_dir / "colocalization" / "qtl_data" / "gtex" / "GTEx_Analysis_v8_eQTL"
    clump_file = pheno_dir / "colocalization" / "clump" / f"{phenotype}.clumped"
    loci_dir = pheno_dir / "colocalization" / "loci"
    phenotype_genes_file = pheno_dir / "GeneDifferentialExpression" / "Files" / f"{phenotype}_genes.csv"

    out_dir = pheno_dir / "GeneDifferentialExpression" / "Files" / "UltimateCompleteRankingAnalysis" / "MR_COLOC_ABLATION"
    safe_mkdir(out_dir)

    return {
        "pheno_dir": pheno_dir,
        "ranked_file": ranked_file,
        "gwas_file": gwas_file,
        "gtex_dir": gtex_dir,
        "clump_file": clump_file,
        "loci_dir": loci_dir,
        "phenotype_genes_file": phenotype_genes_file,
        "out_dir": out_dir,
    }


# =========================================================
# RANKED FILE LOADING + DIRECTION
# =========================================================
def find_first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def infer_gene_direction(row, direction_col=None, beta_col=None):
    if direction_col is not None:
        val = row.get(direction_col, np.nan)
        if pd.notna(val):
            s = str(val).strip().lower()
            if s in {"up", "positive", "higher", "increase", "increased", "risk_up"}:
                return 1
            if s in {"down", "negative", "lower", "decrease", "decreased", "risk_down"}:
                return -1

    if beta_col is not None:
        val = pd.to_numeric(row.get(beta_col, np.nan), errors="coerce")
        if pd.notna(val):
            if val > 0:
                return 1
            if val < 0:
                return -1

    return 0


def direction_label(x):
    if x == 1:
        return "UP"
    if x == -1:
        return "DOWN"
    return "UNKNOWN"


def concordance_label(rank_dir, mr_dir):
    if rank_dir == 0 or mr_dir == 0:
        return "UNKNOWN"
    if rank_dir == mr_dir:
        return "CONCORDANT"
    return "DISCORDANT"


def load_ranked_genes_and_direction(ranked_file):
    df = pd.read_csv(ranked_file)

    gene_id_col = find_first_existing(df, ["Gene_ENSEMBL", "Gene", "gene_id", "ensembl_gene_id", "ENSEMBL_ID"])
    gene_symbol_col = find_first_existing(df, ["GeneSymbol", "Gene_Symbol", "SYMBOL", "symbol", "HGNC_Symbol", "GeneName", "Gene_Name"])
    score_col = find_first_existing(df, ["Importance_Score", "CoreScore", "Score", "Composite_Score", "FinalScore"])

    direction_col = find_first_existing(df, ["Direction", "Effect_Direction", "Gene_Direction", "Regulation", "Status"])
    beta_col = find_first_existing(df, ["Mean_Effect", "Average_Effect", "Effect", "EffectSize", "Beta", "logFC", "Mean_Beta", "Avg_Beta", "Zscore"])

    if gene_id_col is None:
        raise ValueError(f"Could not find gene column in ranked file: {ranked_file}")

    df = df.copy()
    df["Gene_ENSEMBL"] = df[gene_id_col].map(strip_version)

    if gene_symbol_col is not None:
        df["Gene_Symbol"] = df[gene_symbol_col].astype(str)
    else:
        df["Gene_Symbol"] = df["Gene_ENSEMBL"]

    if score_col is not None:
        df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
        df = df.sort_values(score_col, ascending=False)

    df["Rank_Direction_Num"] = df.apply(
        lambda r: infer_gene_direction(r, direction_col=direction_col, beta_col=beta_col),
        axis=1
    )
    df["Rank_Direction"] = df["Rank_Direction_Num"].map(direction_label)
    df = df.dropna(subset=["Gene_ENSEMBL"]).drop_duplicates(subset=["Gene_ENSEMBL"]).reset_index(drop=True)

    return df


def build_symbol_map(ranked_df, phenotype_genes_file=None):
    mp = {}
    if "Gene_ENSEMBL" in ranked_df.columns and "Gene_Symbol" in ranked_df.columns:
        for e, s in zip(ranked_df["Gene_ENSEMBL"], ranked_df["Gene_Symbol"]):
            e = strip_version(e)
            s = str(s).strip()
            if pd.notna(e) and s:
                mp[e] = s

    if phenotype_genes_file is not None and Path(phenotype_genes_file).exists():
        try:
            mg = pd.read_csv(phenotype_genes_file)

            if "ensembl_gene_id" in mg.columns:
                mg["Gene_ENSEMBL"] = mg["ensembl_gene_id"].map(strip_version)
            elif "Gene_ENSEMBL" in mg.columns:
                mg["Gene_ENSEMBL"] = mg["Gene_ENSEMBL"].map(strip_version)
            else:
                return mp

            sym_col = None
            for c in ["gene", "symbol", "Gene", "GeneSymbol"]:
                if c in mg.columns:
                    sym_col = c
                    break

            if sym_col is not None:
                for e, s in zip(mg["Gene_ENSEMBL"], mg[sym_col]):
                    e = strip_version(e)
                    s = str(s).strip()
                    if pd.notna(e) and s and e not in mp:
                        mp[e] = s
        except Exception:
            pass

    return mp


# =========================================================
# GWAS LOAD
# =========================================================
def load_gwas(gwas_file):
    df = pd.read_csv(gwas_file, sep=r"\s+|\t", engine="python")

    required = ["CHR", "BP", "SNP", "A1", "A2", "BETA", "SE"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"GWAS file missing columns: {missing}")

    keep = required.copy()
    for c in ["N", "MAF", "P"]:
        if c in df.columns:
            keep.append(c)

    df = df[keep].copy()
    df["CHR"] = df["CHR"].map(norm_chr)
    df["BP"] = pd.to_numeric(df["BP"], errors="coerce")
    df["BETA"] = pd.to_numeric(df["BETA"], errors="coerce")
    df["SE"] = pd.to_numeric(df["SE"], errors="coerce")
    df["A1"] = df["A1"].astype(str).str.upper()
    df["A2"] = df["A2"].astype(str).str.upper()

    if "N" in df.columns:
        df["N"] = pd.to_numeric(df["N"], errors="coerce")
    else:
        df["N"] = np.nan

    if "MAF" in df.columns:
        df["MAF"] = pd.to_numeric(df["MAF"], errors="coerce")
    else:
        df["MAF"] = np.nan

    if "P" in df.columns:
        df["P"] = pd.to_numeric(df["P"], errors="coerce")
    else:
        df["P"] = np.nan

    df = df.dropna(subset=["CHR", "BP", "BETA", "SE"])
    df["CHR"] = df["CHR"].astype(int)
    df["BP"] = df["BP"].astype(int)

    return df


# =========================================================
# GTEx ITERATION
# =========================================================
def iter_gtex_signif_rows(gz_file):
    with gzip.open(gz_file, "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
        idx = {c: i for i, c in enumerate(header)}

        needed = ["gene_id", "variant_id", "slope", "slope_se"]
        for col in needed:
            if col not in idx:
                raise ValueError(f"{gz_file.name} missing required column: {col}")

        has_p = "pval_nominal" in idx
        has_maf = "maf" in idx
        has_nsamp = "ma_samples" in idx

        for line in f:
            parts = line.rstrip("\n").split("\t")
            gene_id = strip_version(parts[idx["gene_id"]])
            variant_id = parts[idx["variant_id"]]
            slope = pd.to_numeric(parts[idx["slope"]], errors="coerce")
            slope_se = pd.to_numeric(parts[idx["slope_se"]], errors="coerce")
            pval = pd.to_numeric(parts[idx["pval_nominal"]], errors="coerce") if has_p else np.nan
            maf = pd.to_numeric(parts[idx["maf"]], errors="coerce") if has_maf else np.nan
            nsamp = pd.to_numeric(parts[idx["ma_samples"]], errors="coerce") if has_nsamp else np.nan

            yield gene_id, variant_id, slope, slope_se, pval, maf, nsamp


def collect_eqtls_for_top_genes(gtex_file, top_genes_set):
    tissue = gtex_file.name.replace(".v8.signif_variant_gene_pairs.txt.gz", "").replace(".signif_variant_gene_pairs.txt.gz", "")
    rows = []

    for gene_id, variant_id, slope, slope_se, pval, maf, nsamp in iter_gtex_signif_rows(gtex_file):
        if gene_id not in top_genes_set:
            continue

        parsed = parse_gtex_variant_id(variant_id)
        if parsed is None:
            continue

        ch, bp, ref, alt = parsed
        if ch is None or bp is None or not np.isfinite(slope) or not np.isfinite(slope_se):
            continue

        rows.append({
            "Gene_ENSEMBL": gene_id,
            "Tissue": tissue,
            "CHR": int(ch),
            "BP": int(bp),
            "REF": ref,
            "ALT": alt,
            "EQTL_BETA": float(slope),
            "EQTL_SE": float(slope_se),
            "EQTL_P": float(pval) if np.isfinite(pval) else np.nan,
            "EQTL_MAF": float(maf) if np.isfinite(maf) else np.nan,
            "EQTL_N": float(nsamp) if np.isfinite(nsamp) else np.nan,
        })

    if not rows:
        return pd.DataFrame(columns=[
            "Gene_ENSEMBL", "Tissue", "CHR", "BP", "REF", "ALT",
            "EQTL_BETA", "EQTL_SE", "EQTL_P", "EQTL_MAF", "EQTL_N"
        ])

    return pd.DataFrame(rows)


# =========================================================
# MR
# =========================================================
def harmonize(eqtl_df, gwas_df):
    merged = eqtl_df.merge(gwas_df, on=["CHR", "BP"], how="inner")
    if merged.empty:
        return merged

    same = (merged["A1"] == merged["REF"]) & (merged["A2"] == merged["ALT"])
    flip = (merged["A1"] == merged["ALT"]) & (merged["A2"] == merged["REF"])

    merged = merged[same | flip].copy()
    if merged.empty:
        return merged

    merged["EQTL_BETA_ALIGNED"] = np.where(
        flip[same | flip].values,
        -merged["EQTL_BETA"].values,
        merged["EQTL_BETA"].values
    )
    return merged


def run_wald_ratio(beta_gwas, se_gwas, beta_eqtl, se_eqtl):
    if not np.isfinite(beta_eqtl) or abs(beta_eqtl) < 1e-12:
        return np.nan, np.nan, np.nan, np.nan

    mr_beta = beta_gwas / beta_eqtl
    mr_se = math.sqrt((se_gwas ** 2) / (beta_eqtl ** 2) + ((beta_gwas ** 2) * (se_eqtl ** 2)) / (beta_eqtl ** 4))
    mr_z = mr_beta / mr_se if mr_se > 0 else np.nan
    mr_p = normal_two_sided_p(mr_z) if np.isfinite(mr_z) else np.nan
    return mr_beta, mr_se, mr_z, mr_p


def add_significance_labels(df):
    df = df.copy()
    df["MR_Significance"] = "NS"
    df.loc[df["MR_P"] < 0.05, "MR_Significance"] = "Nominal_P<0.05"
    df.loc[df["MR_FDR_BH"] < 0.10, "MR_Significance"] = "FDR<0.10"
    df.loc[df["MR_FDR_BH"] < 0.05, "MR_Significance"] = "FDR<0.05"
    return df


def run_mr_for_topk(top_df, gwas_df, gtex_files, symbol_map):
    top_gene_set = set(top_df["Gene_ENSEMBL"])
    all_results = []

    for gtex_file in gtex_files:
        eqtl_df = collect_eqtls_for_top_genes(gtex_file, top_gene_set)
        if eqtl_df.empty:
            continue

        merged = harmonize(eqtl_df, gwas_df)
        if merged.empty:
            continue

        merged["EQTL_ABS_Z"] = (merged["EQTL_BETA_ALIGNED"] / merged["EQTL_SE"]).abs()

        best_rows = []
        for (gene, tissue), sub in merged.groupby(["Gene_ENSEMBL", "Tissue"], sort=False):
            if sub["EQTL_P"].notna().any():
                sub = sub.sort_values(["EQTL_P", "EQTL_ABS_Z"], ascending=[True, False])
            else:
                sub = sub.sort_values("EQTL_ABS_Z", ascending=False)

            row = sub.iloc[0]
            mr_beta, mr_se, mr_z, mr_p = run_wald_ratio(
                beta_gwas=float(row["BETA"]),
                se_gwas=float(row["SE"]),
                beta_eqtl=float(row["EQTL_BETA_ALIGNED"]),
                se_eqtl=float(row["EQTL_SE"]),
            )

            mr_dir_num = 0
            if np.isfinite(mr_beta):
                if mr_beta > 0:
                    mr_dir_num = 1
                elif mr_beta < 0:
                    mr_dir_num = -1

            best_rows.append({
                "Gene_ENSEMBL": gene,
                "Gene_Symbol": symbol_map.get(gene, gene),
                "Tissue": tissue,
                "SNP": row["SNP"],
                "CHR": int(row["CHR"]),
                "BP": int(row["BP"]),
                "GWAS_BETA": float(row["BETA"]),
                "GWAS_SE": float(row["SE"]),
                "GWAS_P": float(row["P"]) if pd.notna(row.get("P", np.nan)) else np.nan,
                "EQTL_BETA": float(row["EQTL_BETA_ALIGNED"]),
                "EQTL_SE": float(row["EQTL_SE"]),
                "EQTL_P": float(row["EQTL_P"]) if np.isfinite(row["EQTL_P"]) else np.nan,
                "MR_BETA": mr_beta,
                "MR_SE": mr_se,
                "MR_Z": mr_z,
                "MR_P": mr_p,
                "MR_Direction_Num": mr_dir_num,
                "MR_Direction": direction_label(mr_dir_num),
            })

        all_results.extend(best_rows)

    if not all_results:
        return pd.DataFrame(), pd.DataFrame()

    mr_df = pd.DataFrame(all_results)
    mr_df["MR_FDR_BH"] = bh_fdr(mr_df["MR_P"].values)
    mr_df = add_significance_labels(mr_df)

    mr_df = mr_df.merge(
        top_df[["Gene_ENSEMBL", "Rank_Direction_Num", "Rank_Direction"]],
        on="Gene_ENSEMBL",
        how="left"
    )

    mr_df["Direction_Concordance"] = mr_df.apply(
        lambda r: concordance_label(r["Rank_Direction_Num"], r["MR_Direction_Num"]),
        axis=1
    )

    mr_df = mr_df.sort_values(["MR_FDR_BH", "MR_P", "EQTL_P"], ascending=[True, True, True]).reset_index(drop=True)

    gene_summary = (
        mr_df.groupby(["Gene_ENSEMBL", "Gene_Symbol"], as_index=False)
        .agg(
            Best_Tissue=("Tissue", "first"),
            Best_SNP=("SNP", "first"),
            Best_MR_BETA=("MR_BETA", "first"),
            Best_MR_P=("MR_P", "first"),
            Best_MR_FDR_BH=("MR_FDR_BH", "first"),
            Best_MR_Direction=("MR_Direction", "first"),
            Rank_Direction=("Rank_Direction", "first"),
            Best_Direction_Concordance=("Direction_Concordance", "first"),
            Num_Tissues=("Tissue", "nunique"),
            Num_Rows=("Tissue", "size"),
            Num_Nominal=("MR_P", lambda x: int((x < 0.05).sum())),
            Num_FDR_0_10=("MR_FDR_BH", lambda x: int((x < 0.10).sum())),
            Num_FDR_0_05=("MR_FDR_BH", lambda x: int((x < 0.05).sum())),
            Num_Concordant=("Direction_Concordance", lambda x: int((x == "CONCORDANT").sum())),
            Num_Discordant=("Direction_Concordance", lambda x: int((x == "DISCORDANT").sum())),
        )
        .sort_values(["Best_MR_FDR_BH", "Best_MR_P"], ascending=[True, True])
        .reset_index(drop=True)
    )

    return mr_df, gene_summary


# =========================================================
# COLOC SUPPORT
# =========================================================
def discover_tissues(gtex_dir):
    tissues = {}
    for f in sorted(Path(gtex_dir).glob("*.signif_variant_gene_pairs.txt.gz")):
        tissue_id = f.name.replace(".v8.signif_variant_gene_pairs.txt.gz", "").replace(".signif_variant_gene_pairs.txt.gz", "")
        tissues[tissue_id] = f
    return tissues


def load_clumped_loci(clump_file):
    if not Path(clump_file).exists():
        return pd.DataFrame(columns=["LOCUS_ID", "CHR", "BP"])

    loci = pd.read_csv(clump_file, sep=r"\s+", engine="python")
    if "CHR" not in loci.columns or "BP" not in loci.columns:
        return pd.DataFrame(columns=["LOCUS_ID", "CHR", "BP"])

    loci = loci.reset_index(drop=True).copy()
    loci["CHR"] = loci["CHR"].map(norm_chr)
    loci["BP"] = pd.to_numeric(loci["BP"], errors="coerce")
    loci = loci.dropna(subset=["CHR", "BP"]).copy()
    loci["CHR"] = loci["CHR"].astype(int)
    loci["BP"] = loci["BP"].astype(int)
    loci["LOCUS_ID"] = [f"locus_{i+1}" for i in range(len(loci))]
    return loci[["LOCUS_ID", "CHR", "BP"]]


def load_gwas_locus_file(gwas_locus_file):
    if not Path(gwas_locus_file).exists():
        return pd.DataFrame()

    df = pd.read_csv(gwas_locus_file, sep=r"\s+|\t", engine="python")
    needed = ["CHR", "BP", "SNP", "A1", "A2", "BETA", "SE"]
    for c in needed:
        if c not in df.columns:
            return pd.DataFrame()

    keep = needed.copy()
    for c in ["N", "MAF", "P"]:
        if c in df.columns:
            keep.append(c)

    df = df[keep].copy()
    df["CHR"] = df["CHR"].map(norm_chr)
    df["BP"] = pd.to_numeric(df["BP"], errors="coerce")
    df["BETA"] = pd.to_numeric(df["BETA"], errors="coerce")
    df["SE"] = pd.to_numeric(df["SE"], errors="coerce")
    df["A1"] = df["A1"].astype(str).str.upper()
    df["A2"] = df["A2"].astype(str).str.upper()

    if "N" in df.columns:
        df["N"] = pd.to_numeric(df["N"], errors="coerce")
    else:
        df["N"] = np.nan

    if "MAF" in df.columns:
        df["MAF"] = pd.to_numeric(df["MAF"], errors="coerce")
    else:
        df["MAF"] = np.nan

    if "P" in df.columns:
        df["P"] = pd.to_numeric(df["P"], errors="coerce")
    else:
        df["P"] = np.nan

    df = df.dropna(subset=["CHR", "BP", "BETA", "SE"])
    df["CHR"] = df["CHR"].astype(int)
    df["BP"] = df["BP"].astype(int)
    return df


def load_gtex_region_for_tissue(gtex_file, chrom, start, end, gene_set):
    rows = []
    for gene_id, variant_id, slope, slope_se, pval, maf, nsamp in iter_gtex_signif_rows(gtex_file):
        if gene_id not in gene_set:
            continue

        parsed = parse_gtex_variant_id(variant_id)
        if parsed is None:
            continue

        ch, bp, ref, alt = parsed
        if ch != chrom:
            continue
        if bp < start or bp > end:
            continue
        if not np.isfinite(slope) or not np.isfinite(slope_se):
            continue

        rows.append({
            "Gene_ENSEMBL": gene_id,
            "CHR": int(ch),
            "BP": int(bp),
            "REF_gtex": str(ref).upper(),
            "ALT_gtex": str(alt).upper(),
            "slope": float(slope),
            "slope_se": float(slope_se),
            "pval_nominal": float(pval) if np.isfinite(pval) else np.nan,
            "maf_eqtl": float(maf) if np.isfinite(maf) else np.nan,
            "ma_samples": float(nsamp) if np.isfinite(nsamp) else np.nan,
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def harmonize_for_coloc(eqtl_gene_df, gwas_reg_df):
    m = eqtl_gene_df.merge(gwas_reg_df, on=["CHR", "BP"], how="inner")
    if m.empty:
        return pd.DataFrame()

    same = (m["A1"] == m["REF_gtex"]) & (m["A2"] == m["ALT_gtex"])
    flip = (m["A1"] == m["ALT_gtex"]) & (m["A2"] == m["REF_gtex"])

    keep = m[same | flip].copy()
    if keep.empty:
        return pd.DataFrame()

    pal = keep.apply(lambda r: is_palindromic(r["A1"], r["A2"]), axis=1)
    high_maf = keep["MAF"].between(0.42, 0.58, inclusive="both")
    keep = keep[~(pal & high_maf)].copy()
    if keep.empty:
        return pd.DataFrame()

    keep["beta_eqtl_aligned"] = keep["slope"]
    keep.loc[flip[same | flip].values, "beta_eqtl_aligned"] = -keep.loc[flip[same | flip].values, "slope"]

    keep["varbeta_eqtl"] = keep["slope_se"] ** 2
    keep["varbeta_gwas"] = keep["SE"] ** 2

    if keep["MAF"].isna().all() and "maf_eqtl" in keep.columns and keep["maf_eqtl"].notna().any():
        keep["MAF"] = keep["maf_eqtl"]

    keep = keep.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["beta_eqtl_aligned", "varbeta_eqtl", "BETA", "varbeta_gwas"]
    )

    if keep.empty:
        return pd.DataFrame()

    return keep.copy()


def rscript_available():
    return shutil.which("Rscript") is not None


def check_coloc_package():
    if not rscript_available():
        return False
    cmd = ["Rscript", "-e", "suppressMessages(library(coloc)); cat('OK\\n')"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return p.returncode == 0
    except Exception:
        return False


def run_single_coloc_abf(harm_df, locus_id, tissue_id, gene_ensg):
    if harm_df.empty or len(harm_df) < 20:
        return None

    n_eqtl = pd.to_numeric(harm_df.get("ma_samples", np.nan), errors="coerce").dropna()
    n_eqtl = float(n_eqtl.median()) if len(n_eqtl) else 300.0

    n_gwas = pd.to_numeric(harm_df.get("N", np.nan), errors="coerce").dropna()
    n_gwas = float(n_gwas.median()) if len(n_gwas) else 100000.0

    if "MAF" not in harm_df.columns or harm_df["MAF"].isna().all():
        return None

    with tempfile.TemporaryDirectory(prefix="coloc_abf_") as tmpdir:
        tmpdir = Path(tmpdir)
        in_tsv = tmpdir / "harm.tsv"
        out_tsv = tmpdir / "out.tsv"

        tmp = harm_df.copy()
        tmp = tmp.sort_values("P" if "P" in tmp.columns else "BP", ascending=True).drop_duplicates("SNP", keep="first")
        tmp = tmp[["SNP", "BETA", "SE", "beta_eqtl_aligned", "slope_se", "MAF"]].copy()
        tmp.columns = ["SNP", "beta_gwas", "se_gwas", "beta_eqtl", "se_eqtl", "maf"]
        tmp.to_csv(in_tsv, sep="\t", index=False)

        if GWAS_CASE_FRACTION is None:
            gwas_block = f"type='quant', N={n_gwas}"
            warn_mode = "QUANT_APPROX"
        else:
            gwas_block = f"type='cc', N={n_gwas}, s={GWAS_CASE_FRACTION}"
            warn_mode = "CASE_CONTROL"

        r_code = f"""
        suppressMessages({{
          library(data.table)
          library(coloc)
        }})
        d <- fread("{str(in_tsv)}")
        d <- d[is.finite(beta_gwas) & is.finite(se_gwas) & is.finite(beta_eqtl) & is.finite(se_eqtl) & is.finite(maf)]
        if (nrow(d) < 20) {{
          fwrite(data.table(status="too_few_snps"), "{str(out_tsv)}", sep="\\t")
          quit(save="no", status=0)
        }}

        ds1 <- list(
          beta=d$beta_gwas,
          varbeta=(d$se_gwas^2),
          snp=d$SNP,
          MAF=d$maf,
          {gwas_block}
        )

        ds2 <- list(
          beta=d$beta_eqtl,
          varbeta=(d$se_eqtl^2),
          snp=d$SNP,
          MAF=d$maf,
          type='quant',
          N={n_eqtl}
        )

        res <- coloc.abf(dataset1=ds1, dataset2=ds2)
        s <- as.data.frame(as.list(res$summary))
        s$locus <- "{locus_id}"
        s$tissue_id <- "{tissue_id}"
        s$Gene_ENSEMBL <- "{gene_ensg}"
        s$mode <- "{warn_mode}"
        fwrite(s, "{str(out_tsv)}", sep="\\t")
        """

        try:
            p = subprocess.run(["Rscript", "-e", r_code], capture_output=True, text=True, timeout=300)
            if p.returncode != 0:
                return None
            if not out_tsv.exists():
                return None
            out = pd.read_csv(out_tsv, sep="\t")
            if out.empty:
                return None
            return out.iloc[0].to_dict()
        except Exception:
            return None


def run_coloc_for_topk(top_df, paths, symbol_map):
    out_rows = []

    if not RUN_COLOC:
        return pd.DataFrame()

    if not rscript_available():
        print("⚠️ Rscript not found. Skipping coloc.")
        return pd.DataFrame()

    if not check_coloc_package():
        print("⚠️ R package 'coloc' not available. Skipping coloc.")
        return pd.DataFrame()

    if not Path(paths["clump_file"]).exists():
        print("⚠️ Clump file missing. Skipping coloc.")
        return pd.DataFrame()

    if not Path(paths["loci_dir"]).exists():
        print("⚠️ Loci directory missing. Skipping coloc.")
        return pd.DataFrame()

    tissues = discover_tissues(paths["gtex_dir"])
    loci = load_clumped_loci(paths["clump_file"])
    if loci.empty:
        print("⚠️ No loci found in clump file. Skipping coloc.")
        return pd.DataFrame()

    gene_set = set(top_df["Gene_ENSEMBL"])

    for _, locus in loci.iterrows():
        locus_id = str(locus["LOCUS_ID"])
        chrom = int(locus["CHR"])
        lead_bp = int(locus["BP"])
        start = max(1, lead_bp - COLOC_WINDOW_BP)
        end = lead_bp + COLOC_WINDOW_BP

        gwas_locus_file = Path(paths["loci_dir"]) / locus_id / f"{locus_id}.full.hg38.z"
        gwas_reg = load_gwas_locus_file(gwas_locus_file)
        if gwas_reg.empty:
            continue

        gwas_reg = gwas_reg[(gwas_reg["CHR"] == chrom) & (gwas_reg["BP"] >= start) & (gwas_reg["BP"] <= end)].copy()
        if gwas_reg.empty:
            continue

        for tissue_id, tissue_file in tissues.items():
            eqtl_reg = load_gtex_region_for_tissue(tissue_file, chrom, start, end, gene_set)
            if eqtl_reg.empty:
                continue

            for gene_ensg, sub in eqtl_reg.groupby("Gene_ENSEMBL", sort=False):
                harm = harmonize_for_coloc(sub.copy(), gwas_reg.copy())
                if harm.empty or len(harm) < 20:
                    continue

                coloc_res = run_single_coloc_abf(harm, locus_id, tissue_id, gene_ensg)
                if coloc_res is None:
                    continue

                pp4 = np.nan
                for c in coloc_res.keys():
                    if str(c).replace(".", "").lower() in {"pph4abf", "pph4"}:
                        pp4 = pd.to_numeric(coloc_res[c], errors="coerce")
                        break

                out_rows.append({
                    "locus": locus_id,
                    "tissue_id": tissue_id,
                    "Gene_ENSEMBL": gene_ensg,
                    "Gene_Symbol": symbol_map.get(gene_ensg, gene_ensg),
                    "n_overlap_snps": int(harm["SNP"].nunique()) if "SNP" in harm.columns else len(harm),
                    "PP.H4": float(pp4) if pd.notna(pp4) else np.nan,
                    "mode": coloc_res.get("mode", ""),
                })

    if not out_rows:
        return pd.DataFrame()

    coloc_df = pd.DataFrame(out_rows).sort_values(["PP.H4"], ascending=[False]).reset_index(drop=True)
    return coloc_df


# =========================================================
# MAIN
# =========================================================
def main():
    if len(sys.argv) != 2:
        print("Usage:")
        print("  python predict4.1.2.10.3-Enrichment-MR1.py migraine")
        sys.exit(1)

    phenotype = sys.argv[1].strip()
    paths = get_paths(phenotype)

    print("=" * 120)
    print("MR + OPTIONAL COLOCALIZATION ABLATION")
    print("=" * 120)
    print(f"Phenotype      : {phenotype}")
    print(f"Ranked file    : {paths['ranked_file']}")
    print(f"GWAS file      : {paths['gwas_file']}")
    print(f"GTEx dir       : {paths['gtex_dir']}")
    print(f"Clump file     : {paths['clump_file']}")
    print(f"Loci dir       : {paths['loci_dir']}")
    print(f"Output dir     : {paths['out_dir']}")
    print(f"Top-K values   : {TOP_K_LIST}")
    print("=" * 120)

    for p in [paths["ranked_file"], paths["gwas_file"], paths["gtex_dir"]]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Missing required input:\n{p}")

    ranked_df = load_ranked_genes_and_direction(paths["ranked_file"])
    symbol_map = build_symbol_map(ranked_df, paths["phenotype_genes_file"])
    gwas_df = load_gwas(paths["gwas_file"])
    gtex_files = sorted(Path(paths["gtex_dir"]).glob("*.signif_variant_gene_pairs.txt.gz"))

    if not gtex_files:
        raise FileNotFoundError(f"No GTEx .signif_variant_gene_pairs.txt.gz files found in:\n{paths['gtex_dir']}")

    print(f"Total ranked genes       : {len(ranked_df):,}")
    print(f"GWAS rows loaded         : {len(gwas_df):,}")
    print(f"GTEx tissues discovered  : {len(gtex_files):,}")
    print(f"Known symbols mapped     : {len(symbol_map):,}")
    print("=" * 120)

    ablation_rows = []

    for topk in TOP_K_LIST:
        print("\n" + "#" * 120)
        print(f"RUNNING TOP-{topk}")
        print("#" * 120)

        top_df = ranked_df.head(topk).copy()

        mr_df, gene_summary = run_mr_for_topk(top_df, gwas_df, gtex_files, symbol_map)

        mr_out = Path(paths["out_dir"]) / f"MR_top{topk}_full.tsv"
        gene_out = Path(paths["out_dir"]) / f"MR_top{topk}_gene_summary.tsv"

        if mr_df.empty:
            print(f"TOP-{topk}: No MR results found.")
            pd.DataFrame().to_csv(mr_out, sep="\t", index=False)
            pd.DataFrame().to_csv(gene_out, sep="\t", index=False)

            ablation_rows.append({
                "TopK": topk,
                "MR_TotalRows": 0,
                "MR_UniqueGenes": 0,
                "MR_Nominal": 0,
                "MR_FDR_0_10": 0,
                "MR_FDR_0_05": 0,
                "MR_Concordant": 0,
                "MR_Discordant": 0,
                "COLOC_TotalRows": 0,
                "COLOC_PP4_0_80": 0,
                "COLOC_UniqueGenes_PP4_0_80": 0,
            })
            continue

        mr_df.to_csv(mr_out, sep="\t", index=False)
        gene_summary.to_csv(gene_out, sep="\t", index=False)

        print(f"TOP-{topk} MR total rows        : {len(mr_df):,}")
        print(f"TOP-{topk} MR unique genes      : {mr_df['Gene_ENSEMBL'].nunique():,}")
        print(f"TOP-{topk} MR nominal p<0.05    : {(mr_df['MR_P'] < 0.05).sum():,}")
        print(f"TOP-{topk} MR FDR<0.10          : {(mr_df['MR_FDR_BH'] < 0.10).sum():,}")
        print(f"TOP-{topk} MR FDR<0.05          : {(mr_df['MR_FDR_BH'] < 0.05).sum():,}")
        print(f"TOP-{topk} Concordant rows      : {(mr_df['Direction_Concordance'] == 'CONCORDANT').sum():,}")
        print(f"TOP-{topk} Discordant rows      : {(mr_df['Direction_Concordance'] == 'DISCORDANT').sum():,}")

        print("\nTOP MR GENE×TISSUE RESULTS")
        print(mr_df[[
            "Gene_ENSEMBL", "Gene_Symbol", "Tissue", "SNP",
            "MR_BETA", "MR_P", "MR_FDR_BH",
            "MR_Direction", "Rank_Direction", "Direction_Concordance", "MR_Significance"
        ]].head(20).to_string(index=False))

        print("\nTOP MR GENE SUMMARY")
        print(gene_summary[[
            "Gene_ENSEMBL", "Gene_Symbol", "Best_Tissue", "Best_SNP",
            "Best_MR_BETA", "Best_MR_P", "Best_MR_FDR_BH",
            "Best_MR_Direction", "Rank_Direction", "Best_Direction_Concordance",
            "Num_Tissues", "Num_Nominal", "Num_FDR_0_10", "Num_FDR_0_05"
        ]].head(20).to_string(index=False))

        coloc_df = run_coloc_for_topk(top_df, paths, symbol_map)
        coloc_out = Path(paths["out_dir"]) / f"COLOC_top{topk}_full.tsv"

        if coloc_df.empty:
            print(f"\nTOP-{topk}: Coloc skipped or no coloc results.")
            pd.DataFrame().to_csv(coloc_out, sep="\t", index=False)

            ablation_rows.append({
                "TopK": topk,
                "MR_TotalRows": len(mr_df),
                "MR_UniqueGenes": mr_df["Gene_ENSEMBL"].nunique(),
                "MR_Nominal": int((mr_df["MR_P"] < 0.05).sum()),
                "MR_FDR_0_10": int((mr_df["MR_FDR_BH"] < 0.10).sum()),
                "MR_FDR_0_05": int((mr_df["MR_FDR_BH"] < 0.05).sum()),
                "MR_Concordant": int((mr_df["Direction_Concordance"] == "CONCORDANT").sum()),
                "MR_Discordant": int((mr_df["Direction_Concordance"] == "DISCORDANT").sum()),
                "COLOC_TotalRows": 0,
                "COLOC_PP4_0_80": 0,
                "COLOC_UniqueGenes_PP4_0_80": 0,
            })
            continue

        coloc_df.to_csv(coloc_out, sep="\t", index=False)

        n_pp4 = int((coloc_df["PP.H4"] >= COLOC_PP4_THRESHOLD).sum())
        n_genes_pp4 = coloc_df.loc[coloc_df["PP.H4"] >= COLOC_PP4_THRESHOLD, "Gene_ENSEMBL"].nunique()

        print(f"\nTOP-{topk} COLOC total rows        : {len(coloc_df):,}")
        print(f"TOP-{topk} COLOC PP.H4>=0.80       : {n_pp4:,}")
        print(f"TOP-{topk} COLOC genes PP.H4>=0.80 : {n_genes_pp4:,}")

        print("\nTOP COLOC RESULTS")
        print(coloc_df[[
            "locus", "tissue_id", "Gene_ENSEMBL", "Gene_Symbol",
            "n_overlap_snps", "PP.H4", "mode"
        ]].head(20).to_string(index=False))

        ablation_rows.append({
            "TopK": topk,
            "MR_TotalRows": len(mr_df),
            "MR_UniqueGenes": mr_df["Gene_ENSEMBL"].nunique(),
            "MR_Nominal": int((mr_df["MR_P"] < 0.05).sum()),
            "MR_FDR_0_10": int((mr_df["MR_FDR_BH"] < 0.10).sum()),
            "MR_FDR_0_05": int((mr_df["MR_FDR_BH"] < 0.05).sum()),
            "MR_Concordant": int((mr_df["Direction_Concordance"] == "CONCORDANT").sum()),
            "MR_Discordant": int((mr_df["Direction_Concordance"] == "DISCORDANT").sum()),
            "COLOC_TotalRows": len(coloc_df),
            "COLOC_PP4_0_80": n_pp4,
            "COLOC_UniqueGenes_PP4_0_80": int(n_genes_pp4),
        })

    abl = pd.DataFrame(ablation_rows)
    abl_out = Path(paths["out_dir"]) / "MR_COLOC_ablation_summary.tsv"
    abl.to_csv(abl_out, sep="\t", index=False)

    print("\n" + "=" * 120)
    print("FINAL ABLATION SUMMARY")
    print("=" * 120)
    print(abl.to_string(index=False))
    print("=" * 120)
    print(f"Saved summary to: {abl_out}")
    print(f"All detailed files are in: {paths['out_dir']}")


if __name__ == "__main__":
    main()