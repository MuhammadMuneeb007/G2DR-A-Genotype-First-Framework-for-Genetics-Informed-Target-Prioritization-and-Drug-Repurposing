#!/usr/bin/env python3
"""
predict4.1.2.9-Enrichment-MasterTable.py
=======================================

PHASE-1 ONLY: Pathway-based gene score + direction-consistency score
for ALL genes used in enrichment (union of Top-K gene lists).

Inputs (expected):
  1) DE ranking file:
     {phenotype}/GeneDifferentialExpression/Files/UltimateCompleteRankingAnalysis/RANKED_composite.csv

  2) Enrichment outputs from predict4.1.2.8-Enrichment.py (clusterProfiler-style CSVs):
     {phenotype}/GeneDifferentialExpression/Files/UltimateCompleteRankingAnalysis/EnrichmentResults/Top{K}/enrichment_results/
         GO_BP_{combined|upregulated|downregulated}.csv
         GO_MF_{...}.csv
         GO_CC_{...}.csv
         KEGG_{...}.csv
         Reactome_{...}.csv
         Disease_Ontology_{...}.csv

Outputs:
  {phenotype}/GeneDifferentialExpression/Files/UltimateCompleteRankingAnalysis/PathwayIntegration/
     - PathwayEvidenceTable.csv
     - AllPathwayAssociations.csv
     - GenesUsedForEnrichment.csv
     - GenePathwayScores.csv           <-- SAME as before (keep name)
     - MASTER_Gene_Pathway_Table.csv   <-- NEW for PPI (Gene_Symbol, Pathway_Description)

Notes:
  - This script does NOT merge drug/hub scores and does NOT rank globally.
  - Direction-consistency is computed using Up vs Down pathway evidence vs gene Direction from DE.
"""

import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import re
import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

BASE_DIR = Path("/data/ascher02/uqmmune1/ANNOVAR")

PATHWAY_SOURCES = {
    "GO_BP": "GO Biological Process",
    "GO_MF": "GO Molecular Function",
    "GO_CC": "GO Cellular Component",
    "KEGG": "KEGG Pathway",
    "Reactome": "Reactome Pathway",
    "Disease_Ontology": "Disease Ontology",
}

DIRECTIONS = ["combined", "upregulated", "downregulated"]
DEFAULT_K_VALUES = [50, 100, 200, 500, 1000, 2000]

# -----------------------------------------------------------------------------
# UTILS
# -----------------------------------------------------------------------------

def strip_ensg_version(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.startswith("ENSG") and "." in s:
        return s.split(".")[0]
    return s

def clean_symbol(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return ""
    return s.upper()

def safe_float(x, default=np.nan):
    try:
        v = float(x)
        if np.isnan(v):
            return default
        return v
    except Exception:
        return default

def safe_log10_fdr(fdr) -> float:
    f = safe_float(fdr, default=np.nan)
    if np.isnan(f) or f <= 0:
        f = 1e-300
    return -np.log10(f)

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def read_lines_file(path: Path):
    genes = []
    with open(path, "r") as f:
        for line in f:
            g = line.strip()
            if g:
                genes.append(strip_ensg_version(g))
    return genes

def locate_enrichment_root(ranking_dir: Path) -> Path:
    """
    Support both layouts:
      - .../EnrichmentResults/Top{K}/enrichment_results/
      - .../EnrichmentResults/TopK/Top{K}/enrichment_results/   (older/alt)
    """
    cand1 = ranking_dir / "EnrichmentResults"
    cand2 = ranking_dir / "EnrichmentResults" / "TopK"
    for cand in [cand2, cand1]:
        if cand.exists() and any(p.is_dir() and p.name.startswith("Top") for p in cand.iterdir()):
            return cand
    return cand1

# -----------------------------------------------------------------------------
# STEP 1: LOAD ENRICHMENT EVIDENCE
# -----------------------------------------------------------------------------

class PathwayEvidenceAggregator:
    """
    Reads Top{K}/enrichment_results/*.csv across sources and directions,
    and standardizes clusterProfiler columns into a single evidence table.
    """

    def __init__(self, enrichment_root: Path, k_values):
        self.enrichment_root = enrichment_root
        self.k_values = [int(k) for k in k_values]

    def _topk_dir(self, k: int) -> Path:
        return self.enrichment_root / f"Top{k}"

    def load_one(self, k: int, source: str, direction: str):
        top_k_dir = self._topk_dir(k) / "enrichment_results"
        if not top_k_dir.exists():
            return None

        fp = top_k_dir / f"{source}_{direction}.csv"
        if not fp.exists():
            return None

        df = pd.read_csv(fp)

        colmap = {
            "ID": "term_id",
            "Description": "term_name",
            "p.adjust": "FDR",
            "pvalue": "p_value",
            "geneID": "overlap_genes",
            "Count": "overlap_count",
            "GeneRatio": "gene_ratio",
            "BgRatio": "bg_ratio",
        }
        df = df.rename(columns=colmap)

        if "term_id" not in df.columns or "term_name" not in df.columns:
            return None

        if "FDR" not in df.columns:
            df["FDR"] = df["p_value"] if "p_value" in df.columns else np.nan
        if "p_value" not in df.columns:
            df["p_value"] = np.nan

        # approximate term_size from GeneRatio "a/b" -> b
        if "gene_ratio" in df.columns:
            def _ratio_to_den(x):
                s = str(x)
                if "/" in s:
                    try:
                        return int(s.split("/")[1])
                    except Exception:
                        return 0
                return 0
            df["term_size"] = df["gene_ratio"].map(_ratio_to_den)
        else:
            df["term_size"] = 0

        df["K"] = int(k)
        df["Source"] = source
        df["Direction"] = direction

        keep = [
            "K", "Source", "Direction",
            "term_id", "term_name",
            "FDR", "p_value",
            "overlap_genes", "overlap_count",
            "term_size", "gene_ratio", "bg_ratio",
        ]
        return df[[c for c in keep if c in df.columns]]

    def aggregate(self) -> pd.DataFrame:
        parts = []
        for k in self.k_values:
            for src in PATHWAY_SOURCES.keys():
                for d in DIRECTIONS:
                    df = self.load_one(k, src, d)
                    if df is not None and len(df) > 0:
                        parts.append(df)

        if not parts:
            raise ValueError(
                f"No enrichment CSVs found under: {self.enrichment_root}\n"
                "Run predict4.1.2.8-Enrichment.py first, or check paths."
            )
        evidence = pd.concat(parts, ignore_index=True)
        if "overlap_genes" in evidence.columns:
            evidence["overlap_genes"] = evidence["overlap_genes"].astype(str)
        return evidence

# -----------------------------------------------------------------------------
# STEP 2: BUILD PATHWAY ASSOCIATION TABLE (ALL PATHWAYS; NO FILTER)
# -----------------------------------------------------------------------------

class PathwayAssociationBuilder:
    """
    Aggregate evidence across K:
      - pathway_weight = sum(-log10(FDR)) across all TopK where the term appears
      - support_K = number of distinct K values term appears in
      - all_genes = union of overlap genes across K (stored as "|" joined)
    """

    def __init__(self, evidence_df: pd.DataFrame):
        self.e = evidence_df

    def build_all(self) -> pd.DataFrame:
        groups = self.e.groupby(["term_id", "term_name", "Source", "Direction"], dropna=False)
        rows = []

        for (term_id, term_name, source, direction), g in groups:
            k_values = sorted(g["K"].dropna().unique().tolist())
            support_k = len(k_values)

            fdrs = g["FDR"].tolist()
            best_fdr = np.nanmin([safe_float(x, np.nan) for x in fdrs]) if fdrs else np.nan
            mean_fdr = np.nanmean([safe_float(x, np.nan) for x in fdrs]) if fdrs else np.nan

            pathway_weight = float(np.nansum([safe_log10_fdr(x) for x in fdrs]))

            all_genes = set()
            if "overlap_genes" in g.columns:
                for genes_str in g["overlap_genes"].dropna().tolist():
                    if pd.isna(genes_str) or str(genes_str).strip() == "":
                        continue
                    for tok in str(genes_str).split("/"):
                        tok = tok.strip()
                        if tok:
                            all_genes.add(tok)

            avg_term_size = float(np.nanmean(g["term_size"])) if "term_size" in g.columns else 0.0

            rows.append({
                "term_id": term_id,
                "term_name": term_name,
                "Source": source,
                "Direction": direction,
                "support_K": support_k,
                "K_values": ",".join(map(str, k_values)),
                "best_FDR": best_fdr,
                "mean_FDR": mean_fdr,
                "pathway_weight": pathway_weight,
                "avg_term_size": avg_term_size,
                "total_unique_genes": len(all_genes),
                "all_genes": "|".join(sorted(all_genes)),
            })

        df = pd.DataFrame(rows).sort_values("pathway_weight", ascending=False).reset_index(drop=True)
        return df

# -----------------------------------------------------------------------------
# STEP 3: GENE MASTER TABLE (GenePathwayScores.csv)
# -----------------------------------------------------------------------------

class GeneMasterTableBuilder:
    """
    Builds GenePathwayScores.csv (gene-level PathScore + direction consistency)
    """

    def __init__(self, pathway_df: pd.DataFrame, de_df: pd.DataFrame, genes_used: set):
        self.pathway_df = pathway_df
        self.de_df = de_df
        self.genes_used = set(genes_used)

        self.sym2ens = {}
        self.ens2sym = {}

        if "Symbol" in de_df.columns and "Gene" in de_df.columns:
            tmp = de_df[["Gene", "Symbol"]].dropna()
            for _, r in tmp.iterrows():
                ens = strip_ensg_version(r["Gene"])
                sym = clean_symbol(r["Symbol"])
                if ens and sym:
                    self.ens2sym[ens] = sym
                    self.sym2ens[sym] = ens

        self.gene_direction = {}
        if "Direction" in de_df.columns and "Gene" in de_df.columns:
            tmp = de_df[["Gene", "Direction"]].dropna()
            for _, r in tmp.iterrows():
                ens = strip_ensg_version(r["Gene"])
                d = str(r["Direction"]).strip()
                if ens:
                    self.gene_direction[ens] = d

    def _token_to_ensg(self, tok: str) -> str:
        tok = str(tok).strip()
        if not tok:
            return ""
        if tok.startswith("ENSG"):
            return strip_ensg_version(tok)
        tok = clean_symbol(tok)
        return self.sym2ens.get(tok, "")

    def build(self) -> pd.DataFrame:
        score = {
            "combined": defaultdict(float),
            "upregulated": defaultdict(float),
            "downregulated": defaultdict(float),
        }
        n_pathways = {
            "combined": defaultdict(int),
            "upregulated": defaultdict(int),
            "downregulated": defaultdict(int),
        }

        for _, r in self.pathway_df.iterrows():
            direction = str(r.get("Direction", "")).strip().lower()
            if direction not in score:
                continue

            pw = safe_float(r.get("pathway_weight", np.nan), default=np.nan)
            sup = safe_float(r.get("support_K", np.nan), default=np.nan)
            tsz = safe_float(r.get("avg_term_size", 0.0), default=0.0)

            if np.isnan(pw) or np.isnan(sup) or sup <= 0:
                continue

            denom = np.log(1.0 + max(tsz, 0.0))
            if denom <= 0:
                denom = 1.0

            contrib = (pw / sup) / denom

            genes_str = r.get("all_genes", "")
            if pd.isna(genes_str) or str(genes_str).strip() == "":
                continue

            for tok in str(genes_str).split("|"):
                ens = self._token_to_ensg(tok)
                if not ens:
                    continue
                if ens not in self.genes_used:
                    continue
                score[direction][ens] += contrib
                n_pathways[direction][ens] += 1

        rows = []
        for ens in sorted(self.genes_used):
            sym = self.ens2sym.get(ens, "")
            d = self.gene_direction.get(ens, "Unknown")

            ps_c = float(score["combined"].get(ens, 0.0))
            ps_u = float(score["upregulated"].get(ens, 0.0))
            ps_d = float(score["downregulated"].get(ens, 0.0))

            if str(d).lower() == "up":
                ps_dir = ps_u
            elif str(d).lower() == "down":
                ps_dir = ps_d
            else:
                ps_dir = ps_c

            denom_ud = ps_u + ps_d
            if str(d).lower() == "up" and denom_ud > 0:
                dir_cons = ps_u / denom_ud
            elif str(d).lower() == "down" and denom_ud > 0:
                dir_cons = ps_d / denom_ud
            else:
                dir_cons = np.nan

            if np.isnan(dir_cons):
                dir_label = "Unknown"
            elif dir_cons >= 0.60:
                dir_label = "Consistent"
            elif dir_cons <= 0.40:
                dir_label = "Opposite"
            else:
                dir_label = "Mixed"

            rows.append({
                "Gene": ens,
                "Symbol": sym,
                "DE_Direction": d,
                "PathScore_Combined": ps_c,
                "PathScore_Up": ps_u,
                "PathScore_Down": ps_d,
                "PathScore_Directional": ps_dir,
                "DirectionConsistencyScore": dir_cons,
                "DirectionConsistencyLabel": dir_label,
                "N_pathways_combined": int(n_pathways["combined"].get(ens, 0)),
                "N_pathways_up": int(n_pathways["upregulated"].get(ens, 0)),
                "N_pathways_down": int(n_pathways["downregulated"].get(ens, 0)),
            })

        out = pd.DataFrame(rows)
        for c in ["PathScore_Combined", "PathScore_Up", "PathScore_Down", "PathScore_Directional"]:
            if c in out.columns:
                out[c + "_pct"] = out[c].rank(pct=True, method="average").fillna(0.0)

        return out

# -----------------------------------------------------------------------------
# STEP 4 (NEW): MASTER gene→pathway mapping for PPI
# -----------------------------------------------------------------------------

_GENE_TOKEN_RE = re.compile(r"^[A-Z0-9][A-Z0-9\-\.]*$")

def load_symbol_maps(out_dir: Path, de_df: pd.DataFrame):
    """
    Use mapping in priority:
      1) PathwayIntegration/GeneMasterTable.csv if exists (Gene,Symbol)
      2) DE ranked file (Gene,Symbol)
    """
    ens2sym = {}
    sym2ens = {}

    gmt = out_dir / "GeneMasterTable.csv"
    if gmt.exists():
        try:
            df = pd.read_csv(gmt, usecols=["Gene", "Symbol"])
            df["Gene"] = df["Gene"].astype(str).map(strip_ensg_version)
            df["Symbol"] = df["Symbol"].astype(str).map(clean_symbol)
            for g, s in zip(df["Gene"], df["Symbol"]):
                if g and s:
                    ens2sym[g] = s
                    sym2ens[s] = g
        except Exception:
            pass

    if "Gene" in de_df.columns and "Symbol" in de_df.columns:
        tmp = de_df[["Gene", "Symbol"]].dropna()
        for _, r in tmp.iterrows():
            g = strip_ensg_version(r["Gene"])
            s = clean_symbol(r["Symbol"])
            if g and s:
                ens2sym.setdefault(g, s)
                sym2ens.setdefault(s, g)

    return ens2sym, sym2ens

def genes_used_to_symbols(genes_used: set, ens2sym: dict) -> set:
    out = set()
    for g in genes_used:
        g = str(g).strip()
        if not g:
            continue
        if g.startswith("ENSG"):
            s = ens2sym.get(strip_ensg_version(g), "")
            if s:
                out.add(s)
        else:
            out.add(clean_symbol(g))
    return {x for x in out if x}

def token_to_symbol(tok: str, ens2sym: dict) -> str:
    tok = str(tok).strip()
    if not tok:
        return ""
    if tok.startswith("ENSG"):
        return ens2sym.get(strip_ensg_version(tok), "")
    tok = clean_symbol(tok)
    if not tok:
        return ""
    if not _GENE_TOKEN_RE.match(tok):
        return ""
    return tok

def build_master_gene_pathway_table(out_dir: Path,
                                   pathways_df: pd.DataFrame,
                                   genes_used: set,
                                   de_df: pd.DataFrame):
    """
    Output columns required by your PPI module script:
      Gene_Symbol, Pathway_Description
    We also keep extra metadata for debugging/plotting.
    """
    ens2sym, sym2ens = load_symbol_maps(out_dir, de_df)
    allowed_symbols = genes_used_to_symbols(genes_used, ens2sym)

    # required columns
    needed = ["term_id", "term_name", "Source", "Direction", "all_genes"]
    for c in needed:
        if c not in pathways_df.columns:
            raise ValueError(f"AllPathwayAssociations missing required column: {c}")

    rows = []
    for r in pathways_df.itertuples(index=False):
        term_id = str(getattr(r, "term_id"))
        term_name = str(getattr(r, "term_name"))
        source = str(getattr(r, "Source"))
        direction = str(getattr(r, "Direction"))
        genes_str = getattr(r, "all_genes")

        if genes_str is None or (isinstance(genes_str, float) and np.isnan(genes_str)):
            continue

        pathway_desc = term_name  # IMPORTANT: keep simple for your printing (matches your expectation)

        toks = [t.strip() for t in str(genes_str).split("|") if str(t).strip()]
        for tok in toks:
            sym = token_to_symbol(tok, ens2sym)
            if not sym:
                continue
            if allowed_symbols and sym not in allowed_symbols:
                continue

            rows.append({
                "Gene_Symbol": sym,
                "Pathway_Description": pathway_desc,
                # helpful extras
                "term_id": term_id,
                "Source": source,
                "Direction": direction,
                "best_FDR": safe_float(getattr(r, "best_FDR", np.nan), np.nan),
                "mean_FDR": safe_float(getattr(r, "mean_FDR", np.nan), np.nan),
                "pathway_weight": safe_float(getattr(r, "pathway_weight", np.nan), np.nan),
                "support_K": safe_float(getattr(r, "support_K", np.nan), np.nan),
                "K_values": str(getattr(r, "K_values", "")),
            })

    mdf = pd.DataFrame(rows)
    if mdf.empty:
        raise ValueError(
            "MASTER_Gene_Pathway_Table.csv ended up empty.\n"
            "Likely cause: genes in AllPathwayAssociations are ENSG but Symbol mapping failed.\n"
            "Check that DE ranked file has Gene+Symbol, and/or PathwayIntegration/GeneMasterTable.csv exists."
        )

    mdf = mdf.drop_duplicates(subset=["Gene_Symbol", "Pathway_Description"]).copy()
    mdf = mdf.sort_values(["Pathway_Description", "Gene_Symbol"]).reset_index(drop=True)
    return mdf

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phenotype", help="e.g., migraine")
    ap.add_argument("--base-dir", default=str(BASE_DIR), help="Base directory containing phenotype folder")
    ap.add_argument("--k-values", default=",".join(map(str, DEFAULT_K_VALUES)),
                    help="Comma-separated TopK values to use (must exist if enrichment was run), e.g. 50,100,200,500,1000,2000")
    ap.add_argument("--min-existing-k", action="store_true",
                    help="If set, only use K where Top{K}/enrichment_results exists (auto-skip missing).")
    args = ap.parse_args()

    phenotype = args.phenotype
    base_dir = Path(args.base_dir)
    k_values = [int(x) for x in str(args.k_values).split(",") if str(x).strip()]

    base_path = base_dir / phenotype
    results_dir = base_path / "GeneDifferentialExpression" / "Files"
    ranking_dir = results_dir / "UltimateCompleteRankingAnalysis"
    ranked_file = ranking_dir / "RANKED_composite.csv"

    if not ranked_file.exists():
        raise FileNotFoundError(f"Missing DE ranked file: {ranked_file}")

    out_dir = ensure_dir(ranking_dir / "PathwayIntegration")

    # Load DE ranking (MUST include Symbol for PPI master table mapping)
    de_df = pd.read_csv(ranked_file)
    if "Gene" not in de_df.columns:
        raise ValueError("RANKED_composite.csv must contain a 'Gene' column.")
    if "Symbol" not in de_df.columns:
        raise ValueError("RANKED_composite.csv must contain a 'Symbol' column (needed for MASTER_Gene_Pathway_Table.csv).")

    de_df["Gene"] = de_df["Gene"].astype(str).map(strip_ensg_version)
    de_df["Symbol"] = de_df["Symbol"].astype(str).map(clean_symbol)

    # Locate enrichment root
    enr_root = locate_enrichment_root(ranking_dir)

    # Determine which K exist + build genes_used from significant_genes.txt (preferred)
    genes_used = set()
    k_used = []
    for k in k_values:
        topk_dir = enr_root / f"Top{k}"
        enr_dir = topk_dir / "enrichment_results"
        if args.min_existing_k and not enr_dir.exists():
            continue

        sig = topk_dir / "significant_genes.txt"
        if sig.exists():
            genes_k = read_lines_file(sig)
        else:
            genes_k = de_df.head(int(k))["Gene"].tolist()

        if genes_k:
            genes_used.update(map(strip_ensg_version, genes_k))
            k_used.append(int(k))

    if not genes_used:
        raise ValueError("Could not determine genes used for enrichment. Missing TopK folders and/or significant_genes.txt.")

    # 1) GenesUsedForEnrichment.csv  (SAME AS BEFORE)
    pd.DataFrame({"Gene": sorted(genes_used)}).to_csv(out_dir / "GenesUsedForEnrichment.csv", index=False)

    # 2) PathwayEvidenceTable.csv (SAME AS BEFORE)
    ev = PathwayEvidenceAggregator(enrichment_root=enr_root, k_values=k_used).aggregate()
    ev.to_csv(out_dir / "PathwayEvidenceTable.csv", index=False)

    # 3) AllPathwayAssociations.csv (SAME AS BEFORE)
    pathways = PathwayAssociationBuilder(ev).build_all()
    pathways.to_csv(out_dir / "AllPathwayAssociations.csv", index=False)

    # 4) GenePathwayScores.csv (SAME AS BEFORE)
    gene_scores = GeneMasterTableBuilder(pathway_df=pathways, de_df=de_df, genes_used=genes_used).build()
    gene_scores.to_csv(out_dir / "GenePathwayScores.csv", index=False)

    # 5) MASTER_Gene_Pathway_Table.csv (NEW FOR PPI)
    master_df = build_master_gene_pathway_table(out_dir=out_dir,
                                                pathways_df=pathways,
                                                genes_used=genes_used,
                                                de_df=de_df)
    master_df.to_csv(out_dir / "MASTER_Gene_Pathway_Table.csv", index=False)

    print("\n" + "=" * 120)
    print("✅ DONE: Original files saved + PPI master mapping saved.")
    print("=" * 120)
    print(f"Output dir: {out_dir}")
    print(f"- GenesUsedForEnrichment.csv:      {len(genes_used):,} genes")
    print(f"- PathwayEvidenceTable.csv:        {len(ev):,} rows")
    print(f"- AllPathwayAssociations.csv:      {len(pathways):,} rows")
    print(f"- GenePathwayScores.csv:           {len(gene_scores):,} genes")
    print(f"- MASTER_Gene_Pathway_Table.csv:   {len(master_df):,} gene-pathway pairs")

if __name__ == "__main__":
    main()
