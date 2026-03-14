#!/usr/bin/env python3
"""
predict4.1.2.14-PathwayDrugs_CORRECTED.py
========================================

Community-based drug enrichment analysis using the permutation-test cache,
with CONSISTENT drug normalization.

Key features:
  ? Normalizes drugs from BOTH cache + reference list using the SAME function
  ? Universe = drugs from ALL genes across the selected communities (and chosen gene-source)
  ? Robust symbol ? ensembl mapping (handles collisions by storing sets)
  ? Works with expanded communities file produced by:
        predict_PPI_Submodules_WITH_EXPANSION.py
  ? Option to use:
        --genes-source original   (ONLY original community genes)
        --genes-source expanded   (original + pathway-added genes)  [default]

Prereqs:
  - <BASE_DIR>/PermutationCache/<phenotype>_gene_drugs_incremental.json
  - <BASE_DIR>/PermutationCache/<phenotype>_ensembl_to_symbol.json
  - PPI expanded communities:
      <BASE_DIR>/<phenotype>/GeneDifferentialExpression/Files/UltimateCompleteRankingAnalysis/
          PPI_Submodules/Top{N}/{phenotype}_Top{N}_ExpandedModuleGenes.csv

Usage:
  python predict4.1.2.14-PathwayDrugs_CORRECTED.py migraine --top-genes 200 --top-communities 10 --genes-source original
  python predict4.1.2.14-PathwayDrugs_CORRECTED.py migraine --top-genes 200 --top-communities 10 --genes-source expanded

Optional:
  --ref-drugs <path/to/migraine_drugs.csv>
  --outdir   <output directory>

Outputs (written to outdir):
  - community_enrichment_Top{N}_{genes_source}.csv
  - module_<id>_drugs_Top{N}_{genes_source}.csv   (gene-drug pairs for each module)
  - module_<id>_overlap_drugs_Top{N}_{genes_source}.txt  (overlapping ref drugs per module)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

# =============================================================================
# CONFIG (YOUR CONVENTIONS)
# =============================================================================

BASE_DIR = Path("/data/ascher02/uqmmune1/ANNOVAR")
CACHE_DIR = BASE_DIR / "PermutationCache"

SUBMODULE_DIR_TEMPLATE = (
    "GeneDifferentialExpression/Files/UltimateCompleteRankingAnalysis/"
    "PPI_Submodules/Top{top_n}"
)

DEFAULT_COMMUNITY_DRUG_DIRNAME = "CommunityDrugDiscovery"


# =============================================================================
# NORMALIZATION UTILITIES
# =============================================================================

def normalize_drug(name: str) -> str:
    """Normalize drug names (salts/doses/parentheses/punct) -> stable token string."""
    if not name or pd.isna(name):
        return ""
    s = str(name).strip().lower()
    s = re.sub(r"\([^)]*\)", " ", s)  # remove parentheses content

    salts = [
        "hydrochloride", "hcl", "sodium", "potassium", "calcium",
        "succinate", "tartrate", "maleate", "phosphate", "sulfate",
        "acetate", "chloride", "mesylate", "citrate", "bromide",
        "fumarate", "nitrate", "besylate", "lactate", "benzoate"
    ]
    for salt in salts:
        s = re.sub(rf"\b{salt}\b", " ", s)

    # remove dose units
    s = re.sub(r"\b\d+(\.\d+)?\s*(mg|mcg|g|ml|iu|units?)\b", " ", s)

    # keep only alnum -> spaces
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def clean_sym(x: object) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    s = str(x).strip().upper()
    return s if s and s != "NAN" else ""


# =============================================================================
# DRUG CACHE
# =============================================================================

class DrugCache:
    """
    Read-only access to <phenotype>_gene_drugs_incremental.json
    Expected format: { ensembl_id: { source_name: [drug1, drug2, ...], ... }, ... }
    """

    def __init__(self, cache_file: Path):
        self.cache_file = cache_file
        self.data = self._load()

    def _load(self) -> Dict[str, Dict[str, List[str]]]:
        if not self.cache_file.exists():
            raise FileNotFoundError(
                f"Drug cache not found: {self.cache_file}\n\n"
                f"Run permutation test first (your cache builder):\n"
                f"  python predict_PermutationTest_COMPLETE.py <phenotype> --top-genes 500\n"
            )
        with open(self.cache_file, "r") as f:
            return json.load(f)

    def genes_in_cache(self) -> int:
        return len(self.data)

    def get_all_drugs_normalized(self, ensembl_id: str) -> Set[str]:
        """Return NORMALIZED drugs for an ensembl gene across all sources."""
        if ensembl_id not in self.data:
            return set()

        out: Set[str] = set()
        src_map = self.data.get(ensembl_id, {})
        if not isinstance(src_map, dict):
            return set()

        for _src, drug_list in src_map.items():
            if not isinstance(drug_list, list):
                continue
            for d in drug_list:
                dn = normalize_drug(d)
                if dn:
                    out.add(dn)
        return out


# =============================================================================
# SYMBOL MAPPING
# =============================================================================

def load_ensembl_to_symbol(phenotype: str) -> Dict[str, str]:
    symbol_cache = CACHE_DIR / f"{phenotype}_ensembl_to_symbol.json"
    if not symbol_cache.exists():
        raise FileNotFoundError(
            f"Symbol cache not found: {symbol_cache}\n\n"
            f"Run permutation test first:\n"
            f"  python predict_PermutationTest_COMPLETE.py {phenotype}\n"
        )
    with open(symbol_cache, "r") as f:
        d = json.load(f)

    # normalize symbols to upper
    out: Dict[str, str] = {}
    for eid, sym in d.items():
        eid = str(eid).strip()
        out[eid] = clean_sym(sym)
    print(f"   ? Loaded {len(out)} Ensembl?Symbol mappings")
    return out


def build_symbol_to_ensembl(ensembl_to_symbol: Dict[str, str]) -> Dict[str, Set[str]]:
    """
    Reverse map. IMPORTANT: symbol collisions exist, so store SET of ensembl IDs.
    """
    sym2ens: Dict[str, Set[str]] = {}
    for eid, sym in ensembl_to_symbol.items():
        if not sym:
            continue
        sym2ens.setdefault(sym, set()).add(eid)
    return sym2ens


# =============================================================================
# LOAD COMMUNITIES (EXPANDED FILE)
# =============================================================================

def load_expanded_communities(phenotype: str, top_n: int) -> Tuple[Dict[int, Dict], Path]:
    pheno_dir = BASE_DIR / phenotype
    submod_dir = pheno_dir / SUBMODULE_DIR_TEMPLATE.format(top_n=top_n)

    expanded_file = submod_dir / f"{phenotype}_Top{top_n}_ExpandedModuleGenes.csv"
    if not expanded_file.exists():
        raise FileNotFoundError(
            f"Expanded genes file not found: {expanded_file}\n"
            f"Run first:\n"
            f"  python predict_PPI_Submodules_WITH_EXPANSION.py {phenotype} --top-genes {top_n}\n"
        )

    print(f"?? Loading expanded communities: {expanded_file}")
    expanded_df = pd.read_csv(expanded_file, low_memory=False)

    # Pathway enrichment file (optional)
    enrich_file = submod_dir / f"{phenotype}_Top{top_n}_PathwayEnrichment.csv"
    enrich_df = None
    if enrich_file.exists():
        enrich_df = pd.read_csv(enrich_file, low_memory=False)
        print(f"?? Loading pathway enrichment: {enrich_file}")
    else:
        print(f"??  Pathway enrichment file not found: {enrich_file} (continuing)")

    # Validate
    if "Module_ID" not in expanded_df.columns:
        raise ValueError(f"{expanded_file} missing Module_ID column")
    if "Gene" not in expanded_df.columns:
        raise ValueError(f"{expanded_file} missing Gene column")

    communities: Dict[int, Dict] = {}
    for module_id in sorted(expanded_df["Module_ID"].dropna().unique()):
        mid = int(module_id)
        mdf = expanded_df[expanded_df["Module_ID"] == mid].copy()

        genes = [clean_sym(x) for x in mdf["Gene"].tolist()]
        genes = [g for g in genes if g]

        original_genes: List[str] = []
        added_genes: List[str] = []
        if "Source" in mdf.columns:
            original_genes = [clean_sym(x) for x in mdf.loc[mdf["Source"] == "Original_Module", "Gene"].tolist()]
            original_genes = [g for g in original_genes if g]
            added_genes = [clean_sym(x) for x in mdf.loc[mdf["Source"] == "Pathway_Expansion", "Gene"].tolist()]
            added_genes = [g for g in added_genes if g]

        top_pathways: List[str] = []
        if enrich_df is not None and "Module_ID" in enrich_df.columns and "Pathway" in enrich_df.columns:
            mod_enr = enrich_df[enrich_df["Module_ID"] == mid].copy()
            if len(mod_enr) > 0:
                if "FDR_BH" in mod_enr.columns:
                    mod_enr = mod_enr.sort_values("FDR_BH", ascending=True)
                top_pathways = mod_enr["Pathway"].astype(str).head(10).tolist()

        communities[mid] = {
            "genes": genes,                       # expanded (original + added) OR just all if Source missing
            "original_genes": original_genes,     # original-only if Source present
            "added_genes": added_genes,
            "size": len(genes),
            "original_size": len(original_genes),
            "pathways": top_pathways,
        }

    print(f"   ? Loaded {len(communities)} communities")
    print(f"   ? Total unique genes across all communities: {expanded_df['Gene'].astype(str).str.upper().nunique()}")
    return communities, submod_dir


def select_gene_list_for_module(d: Dict, genes_source: str) -> List[str]:
    """
    genes_source:
      - 'original' => use d['original_genes'] (fallback to d['genes'] if empty)
      - 'expanded' => use d['genes']
    """
    if genes_source == "original":
        g = d.get("original_genes", []) or []
        if len(g) == 0:
            # Fallback: if Source column missing upstream, we cannot separate original vs added
            g = d.get("genes", []) or []
        return g
    return d.get("genes", []) or []


# =============================================================================
# REFERENCE DRUGS
# =============================================================================

def load_reference_drugs(ref_path: Path) -> Set[str]:
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference drugs file not found: {ref_path}")

    ref_df = pd.read_csv(ref_path, low_memory=False)

    drug_col = None
    for col in ["drug_name", "DrugName", "drug", "Drug", "name", "intervention"]:
        if col in ref_df.columns:
            drug_col = col
            break
    if drug_col is None:
        raise ValueError(f"Could not find a drug column in {ref_path}. Columns: {list(ref_df.columns)}")

    norm = ref_df[drug_col].apply(normalize_drug)
    out = set(norm.dropna().astype(str).tolist())
    out.discard("")
    return out


# =============================================================================
# GET DRUGS FOR GENE SYMBOLS (via cache)
# =============================================================================

def get_drugs_for_gene_symbols(
    gene_symbols: List[str],
    sym2ens: Dict[str, Set[str]],
    drug_cache: DrugCache
) -> Tuple[Set[str], Dict[str, Set[str]]]:
    """
    Returns:
      all_drugs_norm: Set[str] normalized
      gene_to_drugs_norm: Dict[symbol -> Set[normalized drug]]
    """
    all_drugs: Set[str] = set()
    gene_to_drugs: Dict[str, Set[str]] = {}

    for sym in gene_symbols:
        s = clean_sym(sym)
        if not s:
            continue

        eids = sym2ens.get(s, set())
        if not eids:
            continue

        drugs_for_symbol: Set[str] = set()
        for eid in eids:
            drugs_for_symbol |= drug_cache.get_all_drugs_normalized(eid)

        if drugs_for_symbol:
            gene_to_drugs[s] = drugs_for_symbol
            all_drugs |= drugs_for_symbol

    return all_drugs, gene_to_drugs


# =============================================================================
# ENRICHMENT
# =============================================================================

def hypergeom_pvalue(k: int, M: int, n: int, N: int) -> float:
    """P(X >= k) with hypergeom SF."""
    if k <= 0:
        return 1.0
    if M <= 0 or N <= 0 or n <= 0:
        return 1.0
    return float(hypergeom.sf(k - 1, M, n, N))


def analyze_enrichment(
    module_id: int,
    community_drugs: Set[str],
    reference_drugs: Set[str],
    drug_universe: Set[str],
    n_genes: int,
    n_genes_with_drugs: int,
) -> Dict:
    overlap = community_drugs & reference_drugs

    M = len(drug_universe)      # universe size
    n = len(reference_drugs)    # reference size
    N = len(community_drugs)    # module drug set size
    k = len(overlap)            # observed overlap

    expected = (N * n) / M if M > 0 else 0.0
    fold = (k / expected) if expected > 0 else 0.0
    pval = hypergeom_pvalue(k, M, n, N)

    return {
        "module": module_id,
        "n_genes": n_genes,
        "n_genes_with_drugs": n_genes_with_drugs,
        "n_drugs": N,
        "overlap_observed": k,
        "overlap_expected": expected,
        "fold_enrichment": fold,
        "p_value": pval,
        "overlapping_drugs": sorted(list(overlap)),
        "universe_size": M,
    }


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_community_drug_analysis(
    phenotype: str,
    top_n: int,
    top_n_communities: int,
    ref_drugs_path: Path,
    outdir: Optional[Path] = None,
    genes_source: str = "expanded",
) -> pd.DataFrame:

    if genes_source not in {"original", "expanded"}:
        raise ValueError("genes_source must be 'original' or 'expanded'")

    if outdir is None:
        outdir = BASE_DIR / phenotype / "GeneDifferentialExpression" / "Files" / DEFAULT_COMMUNITY_DRUG_DIRNAME
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("?? COMMUNITY DRUG DISCOVERY (CONSISTENT NORMALIZATION)")
    print("=" * 100)
    print(f"Phenotype: {phenotype}")
    print(f"Communities from: TOP_{top_n} PPI analysis (expanded modules file)")
    print(f"Gene source: {genes_source}  (original=community-only, expanded=includes pathway-added)")
    print(f"Universe: drugs from ALL genes across selected communities (using gene source above)")
    print(f"Communities to analyze: {top_n_communities}")
    print(f"Reference drugs: {ref_drugs_path}")
    print(f"Output: {outdir}")
    print("=" * 100)

    # STEP 1: cache + mapping
    print("\n?? STEP 1: LOADING DRUG CACHE + MAPPINGS")
    print("-" * 100)

    cache_file = CACHE_DIR / f"{phenotype}_gene_drugs_incremental.json"
    drug_cache = DrugCache(cache_file)
    print(f"   ? Drug cache: {cache_file}")
    print(f"   ? Genes in cache: {drug_cache.genes_in_cache()}")

    ensembl_to_symbol = load_ensembl_to_symbol(phenotype)
    sym2ens = build_symbol_to_ensembl(ensembl_to_symbol)
    print(f"   ? Unique symbols in mapping: {len(sym2ens)}")

    # STEP 2: communities
    print("\n?? STEP 2: LOADING COMMUNITIES")
    print("-" * 100)
    communities, _submod_dir = load_expanded_communities(phenotype, top_n)

    # sort communities by size (expanded size, since that's what's available consistently)
    sorted_ids = sorted(communities.keys(), key=lambda cid: communities[cid]["size"], reverse=True)
    top_ids = sorted_ids[: int(top_n_communities)]

    print(f"\n?? Top {len(top_ids)} communities selected:")
    for cid in top_ids:
        d = communities[cid]
        print(f"   Module {cid}: {d['size']} genes (original: {d['original_size']}, added: {d['size'] - d['original_size']})")
        if d["pathways"]:
            print(f"      Top pathways: {', '.join(d['pathways'][:3])}")

    # STEP 3: universe from ALL genes across selected communities (respecting genes_source)
    print("\n?? STEP 3: BUILDING DRUG UNIVERSE FROM COMMUNITY GENES")
    print("-" * 100)

    all_genes: Set[str] = set()
    for cid in top_ids:
        d = communities[cid]
        all_genes.update(select_gene_list_for_module(d, genes_source))

    all_genes = {clean_sym(g) for g in all_genes if clean_sym(g)}
    print(f"   Total unique genes across selected communities ({genes_source}): {len(all_genes)}")

    drug_universe, universe_gene_to_drugs = get_drugs_for_gene_symbols(list(all_genes), sym2ens, drug_cache)
    genes_with_drugs = len(universe_gene_to_drugs)
    pct = (100.0 * genes_with_drugs / len(all_genes)) if len(all_genes) > 0 else 0.0

    print(f"   ? Genes with drugs: {genes_with_drugs}/{len(all_genes)} ({pct:.1f}%)")
    print(f"   ? DRUG UNIVERSE (normalized): {len(drug_universe)} unique drugs")

    # STEP 4: reference drugs (normalized)
    print("\n?? STEP 4: LOADING REFERENCE DRUGS (NORMALIZED)")
    print("-" * 100)
    reference_drugs = load_reference_drugs(ref_drugs_path)
    print(f"   ? Reference drugs (normalized): {len(reference_drugs)}")

    if len(drug_universe) == 0:
        print("\n??  Drug universe is EMPTY. None of the selected community genes had drugs in cache.")
        print("    Enrichment will be p=1.0; fix upstream: cache content, mapping, or select more/broader communities.\n")

    # STEP 5: analyze each community
    print("\n?? STEP 5: ANALYZING COMMUNITIES")
    print("-" * 100)
    print(f"   Drug universe (M):   {len(drug_universe)}")
    print(f"   Reference (n):       {len(reference_drugs)}")

    all_results: List[Dict] = []
    all_gene_drug_tables: Dict[int, Dict[str, Set[str]]] = {}
    per_module_drugsets: Dict[int, Set[str]] = {}

    for cid in top_ids:
        d = communities[cid]
        genes = select_gene_list_for_module(d, genes_source)

        print(f"\n   ?? Module {cid} ({len(genes)} genes; source={genes_source})")
        comm_drugs, gene_to_drugs = get_drugs_for_gene_symbols(genes, sym2ens, drug_cache)

        if gene_to_drugs:
            all_gene_drug_tables[cid] = gene_to_drugs
        per_module_drugsets[cid] = set(comm_drugs)

        result = analyze_enrichment(
            module_id=cid,
            community_drugs=comm_drugs,
            reference_drugs=reference_drugs,
            drug_universe=drug_universe,
            n_genes=len(genes),
            n_genes_with_drugs=len(gene_to_drugs),
        )

        # metadata (still useful even if genes_source=original)
        result["community_size"] = len(genes)
        result["original_size"] = d.get("original_size", 0)
        result["added_from_pathways"] = max(0, d.get("size", len(genes)) - d.get("original_size", 0))
        result["top_pathways"] = "; ".join(d.get("pathways", [])[:5])

        all_results.append(result)

        print(f"      Genes with drugs: {result['n_genes_with_drugs']}/{len(genes)}")
        print(f"      Unique drugs:     {result['n_drugs']}")
        print(f"      Overlap (k):      {result['overlap_observed']}")
        print(f"      Expected:         {result['overlap_expected']:.2f}")
        print(f"      Fold:             {result['fold_enrichment']:.3f}x")
        print(f"      P-value:          {result['p_value']:.4e}")

        if result["overlapping_drugs"]:
            print(f"      Overlapping ref drugs (first 15): {', '.join(result['overlapping_drugs'][:15])}")

    # STEP 6: save results
    print("\n?? STEP 6: SAVING RESULTS")
    print("-" * 100)

    summary_df = pd.DataFrame([{
        "Module": r["module"],
        "Community_Size": r["community_size"],
        "Genes_Source": genes_source,
        "Genes_With_Drugs": r["n_genes_with_drugs"],
        "N_Drugs": r["n_drugs"],
        "Universe_Size": r["universe_size"],
        "Reference_Size": len(reference_drugs),
        "Overlap_Observed": r["overlap_observed"],
        "Overlap_Expected": round(float(r["overlap_expected"]), 2),
        "Fold_Enrichment": round(float(r["fold_enrichment"]), 3),
        "P_Value": float(r["p_value"]),
        "Significant_p<0.05": "?" if float(r["p_value"]) < 0.05 else "?",
        "Top_Pathways": r.get("top_pathways", ""),
        "Overlapping_Drugs": "; ".join(r["overlapping_drugs"][:60]),
    } for r in all_results])

    summary_file = outdir / f"community_enrichment_Top{top_n}_{genes_source}.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"   ?? {summary_file}")

    # per-module gene-drug tables + overlap lists
    for cid, gene_to_drugs in all_gene_drug_tables.items():
        rows = []
        for gene, drugs in gene_to_drugs.items():
            for drug in sorted(drugs):
                rows.append({"Gene": gene, "Drug_Normalized": drug})
        if rows:
            drug_df = pd.DataFrame(rows)
            drug_file = outdir / f"module_{cid}_drugs_Top{top_n}_{genes_source}.csv"
            drug_df.to_csv(drug_file, index=False)

        # overlap text
        ov = next((r["overlapping_drugs"] for r in all_results if r["module"] == cid), [])
        overlap_file = outdir / f"module_{cid}_overlap_drugs_Top{top_n}_{genes_source}.txt"
        with open(overlap_file, "w") as f:
            for dname in ov:
                f.write(dname + "\n")

    print(f"   ?? Saved {len(all_gene_drug_tables)} gene-drug tables + overlap lists")

    # Combined enrichment across all analyzed communities (union of community drug sets)
    all_comm_drugs: Set[str] = set()
    for s in per_module_drugsets.values():
        all_comm_drugs |= set(s)

    if len(drug_universe) > 0 and len(all_comm_drugs) > 0:
        total_overlap = len(all_comm_drugs & reference_drugs)
        total_expected = (len(all_comm_drugs) * len(reference_drugs)) / len(drug_universe)
        total_fold = (total_overlap / total_expected) if total_expected > 0 else 0.0
        total_pval = hypergeom_pvalue(total_overlap, len(drug_universe), len(reference_drugs), len(all_comm_drugs))
    else:
        total_overlap, total_expected, total_fold, total_pval = 0, 0.0, 0.0, 1.0

    print("\n" + "=" * 100)
    print("?? COMBINED ENRICHMENT (Union of drugs across analyzed communities)")
    print("=" * 100)
    print(f"   Drug universe (M):    {len(drug_universe)}")
    print(f"   Community drugs (N):  {len(all_comm_drugs)}")
    print(f"   Reference drugs (n):  {len(reference_drugs)}")
    print(f"   Overlap observed (k): {total_overlap}")
    print(f"   Expected overlap:     {total_expected:.2f}")
    print(f"   Fold enrichment:      {total_fold:.3f}x")
    print(f"   P-value:              {total_pval:.4e}")
    print("=" * 100)
    print(f"? COMPLETE! Results: {outdir}")
    print("=" * 100)

    return summary_df


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("phenotype", help="Phenotype (e.g., migraine)")
    p.add_argument("--top-genes", type=int, default=200,
                   help="Use communities from PPI_Submodules/Top{N} (default: 200)")
    p.add_argument("--top-communities", type=int, default=10,
                   help="Number of communities to analyze (default: 10)")
    p.add_argument("--genes-source", choices=["original", "expanded"], default="expanded",
                   help="original=community-only genes; expanded=include pathway-added genes (default: expanded)")

    # Reference drugs file resolution
    p.add_argument("--ref-drugs", type=str, default="", help="Path to reference drugs CSV (optional)")
    p.add_argument("--outdir", type=str, default="", help="Output directory (optional)")

    args = p.parse_args()

    phenotype = args.phenotype.strip()
    if not phenotype:
        raise SystemExit("? Empty phenotype")

    # Resolve reference drugs file
    if args.ref_drugs:
        ref_path = Path(args.ref_drugs)
    else:
        cand = [
            BASE_DIR / f"{phenotype}_drugs.csv",
            BASE_DIR / "migraine_drugs.csv",
            Path(f"{phenotype}_drugs.csv"),
            Path("migraine_drugs.csv"),
        ]
        ref_path = next((c for c in cand if c.exists()), None)
        if ref_path is None:
            raise SystemExit(
                "? Could not find reference drugs CSV.\n"
                "Provide --ref-drugs <path/to/file.csv>\n"
            )

    outdir = Path(args.outdir) if args.outdir else None

    run_community_drug_analysis(
        phenotype=phenotype,
        top_n=int(args.top_genes),
        top_n_communities=int(args.top_communities),
        ref_drugs_path=ref_path,
        outdir=outdir,
        genes_source=str(args.genes_source),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
