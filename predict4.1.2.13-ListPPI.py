#!/usr/bin/env python3
"""
predict_PPI_Submodules_WITH_EXPANSION.py
=========================================

COMPLETE pipeline that:
1. Loads top N genes from TOP_{N}_1.csv
2. Builds PPI network communities
3. Finds enriched pathways per community
4. EXPANDS communities to include ALL genes from enriched pathways
5. Saves expanded gene lists for drug discovery

Usage:
  python predict_PPI_Submodules_WITH_EXPANSION.py migraine --top-genes 200
  python predict_PPI_Submodules_WITH_EXPANSION.py migraine --top-genes 500
"""

from __future__ import annotations

import sys
import math
import argparse
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

# =============================================================================
# CONFIG
# =============================================================================
BASE_DIR = Path("/data/ascher02/uqmmune1/ANNOVAR")

FINAL_INTEGRATION_DIR = "GeneDifferentialExpression/Files/UltimateCompleteRankingAnalysis/FinalIntegration"
PATHWAY_INTEGRATION_DIR = "GeneDifferentialExpression/Files/UltimateCompleteRankingAnalysis/PathwayIntegration"
EDGES_REL = "GeneDifferentialExpression/Files/UltimateCompleteRankingAnalysis/string_edges.csv"

PATHWAY_MAP_CANDIDATES_REL = [
    f"{PATHWAY_INTEGRATION_DIR}/MASTER_Gene_Pathway_Table.csv",
    f"{PATHWAY_INTEGRATION_DIR}/RobustPathways.csv",
    f"{PATHWAY_INTEGRATION_DIR}/AllPathwayAssociations.csv",
]

OUT_DIRNAME = "PPI_Submodules"

# Filters
MIN_COMMUNITY_SIZE = 4
MIN_GENES_PER_PATHWAY_HIT = 2
MIN_PATHWAY_SIZE_BG = 5

# Print settings
DEFAULT_TOP_PATHWAYS_PER_COMMUNITY = 10
DEFAULT_MAX_COMMUNITIES_TO_PRINT = 50

# =============================================================================
# UTILITIES
# =============================================================================
def die(msg: str) -> None:
    print(f"\n❌ {msg}\n", file=sys.stderr)
    raise SystemExit(1)

def clean_sym(x: object) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    s = str(x).strip()
    return s.upper() if s else ""

def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0)

def detect_edge_scale(scores: pd.Series) -> float:
    smax = float(scores.max())
    if smax <= 1.0 + 1e-9:
        return 1.0
    if smax <= 1000.0 + 1e-6:
        return 1000.0
    return smax

def strip_ensembl_version(g: str) -> str:
    return re.sub(r"\.\d+$", "", str(g).strip())

def split_genes_blob(blob: str) -> List[str]:
    if blob is None:
        return []
    s = str(blob).strip()
    if not s or s.lower() == "nan":
        return []
    parts = re.split(r"[;,\|\s/]+", s)
    parts = [clean_sym(p) for p in parts if clean_sym(p)]
    seen = set()
    out = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out

# =============================================================================
# LOAD GENES
# =============================================================================
def load_top_genes_from_file(phenotype: str, top_n: int) -> Tuple[List[str], Dict[str, float], str]:
    genes_dir = BASE_DIR / phenotype / FINAL_INTEGRATION_DIR
    genes_file = genes_dir / f"TOP_{top_n}_1.csv"
    
    if not genes_file.exists():
        available = sorted([f.name for f in genes_dir.glob("TOP_*.csv")])
        die(f"Gene file not found: {genes_file}\nAvailable: {available}")

    print(f"📂 Loading genes: {genes_file}")
    df = pd.read_csv(genes_file)

    if "Gene" not in df.columns:
        die(f"File must have 'Gene' column. Found: {list(df.columns)}")

    if "Symbol" in df.columns:
        df["Symbol"] = df["Symbol"].map(clean_sym)
    else:
        df["Symbol"] = df["Gene"].astype(str).apply(strip_ensembl_version).map(clean_sym)

    score_col = None
    for c in ["Combined_Score", "Score", "Importance_Score", "score"]:
        if c in df.columns:
            score_col = c
            break
    if score_col is None:
        df["_score_uniform"] = 1.0
        score_col = "_score_uniform"

    df[score_col] = safe_num(df[score_col])
    df = df.sort_values([score_col], ascending=[False]).reset_index(drop=True)

    symbols = [s for s in df["Symbol"].tolist() if s]
    if len(symbols) < 20:
        die(f"Too few valid symbols (n={len(symbols)})")

    score_map = {g: float(sc) for g, sc in zip(df["Symbol"].tolist(), df[score_col].tolist()) if g}

    print(f"   ✓ Loaded {len(symbols)} genes")
    print(f"   ✓ Score column: {score_col}")
    return symbols, score_map, score_col

# =============================================================================
# BUILD PPI
# =============================================================================
def build_induced_graph(edges_path: Path, node_list: List[str], min_edge_score01: float) -> Tuple[nx.Graph, int]:
    node_set = set(node_list)

    print(f"\n🔗 Building PPI network...")
    edf = pd.read_csv(edges_path, usecols=["protein1", "protein2", "score"], low_memory=False)
    edf["protein1"] = edf["protein1"].map(clean_sym)
    edf["protein2"] = edf["protein2"].map(clean_sym)
    edf["score"] = safe_num(edf["score"]).astype(float)

    scale = detect_edge_scale(edf["score"])
    thr_raw = float(min_edge_score01) * scale

    edf = edf[
        (edf["score"] >= thr_raw) &
        (edf["protein1"].isin(node_set)) &
        (edf["protein2"].isin(node_set)) &
        (edf["protein1"] != "") &
        (edf["protein2"] != "") &
        (edf["protein1"] != edf["protein2"])
    ].copy()

    print(f"   ✓ Edges after filtering: {len(edf):,}")

    G = nx.Graph()
    for r in edf.itertuples(index=False):
        w01 = float(r.score) / float(scale) if scale > 0 else 0.0
        w01 = max(0.0, min(1.0, w01))
        G.add_edge(r.protein1, r.protein2, weight=w01)

    for g in node_list:
        if g not in G:
            G.add_node(g)

    connected = sum(1 for n in G.nodes() if G.degree(n) > 0)
    print(f"   ✓ Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"   ✓ Connected nodes: {connected}")

    return G, len(edf)

# =============================================================================
# PATHWAY LOADING
# =============================================================================
def find_first_existing(pheno_dir: Path, rel_list: List[str]) -> Optional[Path]:
    for rel in rel_list:
        p = pheno_dir / rel
        if p.exists():
            return p
    return None

def load_pathway_mapping_from_file(path: Path) -> Tuple[Dict[str, Set[str]], Set[str], str]:
    df = pd.read_csv(path, low_memory=False)
    cols = set(df.columns)

    # Gene-pathway pairs
    gene_col = next((c for c in ["Gene_Symbol", "GeneSymbol", "Symbol", "gene_symbol", "Gene"] if c in cols), None)
    path_col = next((c for c in ["Pathway_Description", "Pathway", "term_name", "Description", "Term"] if c in cols), None)

    # Genes list
    genes_blob_col = next((c for c in ["Genes_in_Pathway", "Genes", "all_genes", "Members"] if c in cols), None)

    # List-style
    if path_col and genes_blob_col:
        if "overlap_count" in cols and genes_blob_col == "overlap_genes":
            return {}, set(), "UNUSABLE"

        pathway_to_genes: Dict[str, Set[str]] = {}
        bg: Set[str] = set()
        for pw, blob in zip(df[path_col].astype(str).values, df[genes_blob_col].values):
            pw = str(pw).strip()
            if not pw or pw.lower() == "nan":
                continue
            genes = split_genes_blob(blob)
            if not genes:
                continue
            gs = set(genes)
            pathway_to_genes.setdefault(pw, set()).update(gs)
            bg.update(gs)

        if len(pathway_to_genes) == 0 or len(bg) < 200:
            return {}, set(), "UNUSABLE"
        return pathway_to_genes, bg, "GENES_LIST"

    # Pairs
    if path_col and gene_col:
        pathway_to_genes: Dict[str, Set[str]] = {}
        bg: Set[str] = set()
        for g, pw in zip(df[gene_col].values, df[path_col].values):
            pw = str(pw).strip()
            if not pw or pw.lower() == "nan":
                continue
            gg = clean_sym(g)
            if not gg:
                continue
            pathway_to_genes.setdefault(pw, set()).add(gg)
            bg.add(gg)

        if len(pathway_to_genes) == 0 or len(bg) < 200:
            return {}, set(), "UNUSABLE"
        return pathway_to_genes, bg, "PAIRS"

    return {}, set(), "UNUSABLE"

# =============================================================================
# ENRICHMENT
# =============================================================================
def log_choose(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)

def hypergeom_tail_sf(M: int, K: int, n: int, k: int) -> float:
    if k <= 0:
        return 1.0
    hi = min(n, K)
    if k > hi:
        return 0.0

    logs = []
    for x in range(k, hi + 1):
        lp = log_choose(K, x) + log_choose(M - K, n - x) - log_choose(M, n)
        logs.append(lp)

    m = max(logs)
    s = sum(math.exp(li - m) for li in logs)
    p = math.exp(m) * s
    return float(min(max(p, 0.0), 1.0))

def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        q[i] = prev
    out = np.empty(n, dtype=float)
    out[order] = np.clip(q, 0.0, 1.0)
    return out

def enrich_modules(
    modules: List[Set[str]],
    pathway_to_genes: Dict[str, Set[str]],
    universe_genes: Set[str],
    score_map: Dict[str, float]
) -> pd.DataFrame:
    print(f"\n🧪 Testing pathway enrichment...")

    universe = set(universe_genes)
    M = len(universe)

    rows = []
    for mid, comm in enumerate(modules, start=1):
        comm_u = set(comm) & universe
        n = len(comm_u)
        if n < MIN_COMMUNITY_SIZE:
            continue

        for pathway, geneset in pathway_to_genes.items():
            geneset_u = set(geneset) & universe
            K = len(geneset_u)
            if K < MIN_PATHWAY_SIZE_BG:
                continue

            k = len(comm_u & geneset_u)
            if k < MIN_GENES_PER_PATHWAY_HIT:
                continue

            p = hypergeom_tail_sf(M, K, n, k)

            overlap_genes = sorted(
                list(comm_u & geneset_u),
                key=lambda x: score_map.get(x, 0.0),
                reverse=True
            )
            
            # ✅ NEW: Also store ALL pathway genes
            all_pathway_genes = sorted(
                list(geneset_u),
                key=lambda x: score_map.get(x, 0.0),
                reverse=True
            )

            rows.append({
                "Module_ID": mid,
                "Module_Size": len(comm),
                "Pathway": pathway,
                "k_in_module": k,
                "K_in_universe": K,
                "p_value": p,
                "Genes_Overlap": ";".join(overlap_genes),  # Genes in BOTH
                "Genes_All_In_Pathway": ";".join(all_pathway_genes),  # ALL pathway genes
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print("   ⚠️  No enrichments found")
        return df

    df["FDR_BH"] = bh_fdr(df["p_value"].to_numpy())
    df = df.sort_values(
        ["Module_ID", "FDR_BH", "p_value", "k_in_module"],
        ascending=[True, True, True, False]
    ).reset_index(drop=True)

    sig = int((df["FDR_BH"] < 0.05).sum())
    print(f"   ✓ Total enrichments: {len(df)}")
    print(f"   ✓ Significant (FDR < 0.05): {sig}")
    return df

# =============================================================================
# ✅ PRETTY PRINT COMMUNITIES
# =============================================================================
def pretty_print_communities(
    phenotype: str,
    modules: List[Set[str]],
    enrich_df: pd.DataFrame,
    G: nx.Graph,
    edges_in_induced: int,
    min_edge_score01: float,
    mapping_file: Path,
    top_paths_per_comm: int = DEFAULT_TOP_PATHWAYS_PER_COMMUNITY,
    max_comms_to_print: int = DEFAULT_MAX_COMMUNITIES_TO_PRINT
) -> None:
    """Print communities with their top enriched pathways in a nice format."""
    
    module_genes_count = sum(1 for n in G.nodes() if G.degree(n) > 0)
    
    print("\n" + "=" * 60)
    print("COMMUNITY SUMMARY")
    print("=" * 60)
    print(f"Phenotype: {phenotype}")
    print(f"Module genes (connected): {module_genes_count}")
    print(f"Edges in induced subgraph: {edges_in_induced}")
    print(f"Edge threshold used: {min_edge_score01:.3f}")
    print(f"Communities found: {len(modules)}")
    print(f"Pathway file: {mapping_file.name}")
    print("")

    # Build top pathways per module (by FDR then p-value then k)
    top_paths: Dict[int, List[Tuple[str, float]]] = {}
    if not enrich_df.empty:
        for mid, grp in enrich_df.groupby("Module_ID"):
            top_rows = grp.head(int(top_paths_per_comm))
            top_paths[int(mid)] = list(zip(top_rows["Pathway"].tolist(), top_rows["FDR_BH"].tolist()))

    shown = 0
    for mid, comm in enumerate(modules, start=1):
        if shown >= int(max_comms_to_print):
            print(f"... (truncated, {len(modules) - shown} more communities)")
            break
        if len(comm) < MIN_COMMUNITY_SIZE:
            continue
        shown += 1

        print(f"Community {mid} | size={len(comm)}")
        paths = top_paths.get(mid, [])
        if paths:
            for pathway, fdr in paths:
                sig_marker = "*" if fdr < 0.05 else ""
                print(f"  - {pathway} (FDR={fdr:.2e}){sig_marker}")
        else:
            print("  - (no pathways pass filters)")
        print("")

# =============================================================================
# ✅ PATHWAY EXPANSION
# =============================================================================
def expand_modules_with_pathways(
    modules: List[Set[str]],
    enrich_df: pd.DataFrame,
    pathway_to_genes: Dict[str, Set[str]],
    fdr_threshold: float = 0.05
) -> Tuple[Dict[int, Set[str]], pd.DataFrame]:
    """
    Expand each module to include ALL genes from significantly enriched pathways.
    
    Returns:
        - expanded_modules: Dict[module_id -> expanded gene set]
        - expansion_df: DataFrame with detailed expansion info
    """
    print(f"\n🔬 EXPANDING MODULES WITH PATHWAY GENES (FDR < {fdr_threshold})...")
    
    expanded_modules: Dict[int, Set[str]] = {}
    expansion_rows = []
    
    for mid in range(1, len(modules) + 1):
        original_genes = set(modules[mid - 1])
        expanded_genes = set(original_genes)
        
        # Get significant pathways
        sig_pathways = enrich_df[
            (enrich_df["Module_ID"] == mid) & 
            (enrich_df["FDR_BH"] < fdr_threshold)
        ].copy()
        
        added_from_pathways = set()
        pathway_details = []
        
        for _, row in sig_pathways.iterrows():
            pathway = row["Pathway"]
            if pathway in pathway_to_genes:
                pathway_genes = set(pathway_to_genes[pathway])
                new_genes = pathway_genes - expanded_genes
                
                if new_genes:
                    added_from_pathways.update(new_genes)
                    expanded_genes.update(new_genes)
                    pathway_details.append({
                        "pathway": pathway,
                        "fdr": row["FDR_BH"],
                        "genes_added": len(new_genes)
                    })
        
        expanded_modules[mid] = expanded_genes
        
        # Record expansion details
        for gene in expanded_genes:
            expansion_rows.append({
                "Module_ID": mid,
                "Gene": gene,
                "Source": "Original_Module" if gene in original_genes else "Pathway_Expansion",
                "N_Enriched_Pathways": len(sig_pathways),
                "Top_Pathways": ";".join(sig_pathways["Pathway"].head(5).tolist())
            })
        
        if added_from_pathways:
            print(f"   Module {mid}: {len(original_genes):>3} → {len(expanded_genes):>4} genes "
                  f"(+{len(added_from_pathways)} from {len(pathway_details)} pathways)")
    
    expansion_df = pd.DataFrame(expansion_rows).sort_values(["Module_ID", "Source"])
    
    total_original = sum(len(modules[i]) for i in range(len(modules)))
    total_expanded = sum(len(g) for g in expanded_modules.values())
    
    print(f"\n   ✅ EXPANSION SUMMARY:")
    print(f"      Original genes:  {total_original:,}")
    print(f"      Expanded genes:  {total_expanded:,}")
    print(f"      Genes added:     {total_expanded - total_original:,}")
    
    return expanded_modules, expansion_df

# =============================================================================
# OUTPUT
# =============================================================================
def write_outputs(
    phenotype: str,
    out_dir: Path,
    top_n: int,
    modules: List[Set[str]],
    expanded_modules: Dict[int, Set[str]],
    expansion_df: pd.DataFrame,
    enrich_df: pd.DataFrame,
    score_map: Dict[str, float]
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Module summary
    mod_rows = []
    for mid, original_genes in enumerate(modules, start=1):
        expanded_genes = expanded_modules.get(mid, original_genes)
        genes_sorted = sorted(list(original_genes), key=lambda x: score_map.get(x, 0.0), reverse=True)
        
        mod_rows.append({
            "Phenotype": phenotype,
            "Module_ID": mid,
            "Original_Size": len(original_genes),
            "Expanded_Size": len(expanded_genes),
            "Genes_Added": len(expanded_genes) - len(original_genes),
            "AvgScore_Original": float(np.mean([score_map.get(g, 0) for g in original_genes])),
            "TopGenes": ";".join(genes_sorted[:10]),
        })
    
    mods_df = pd.DataFrame(mod_rows).sort_values(["Expanded_Size"], ascending=[False])
    
    # Save files
    out_summary = out_dir / f"{phenotype}_Top{top_n}_ModuleSummary.csv"
    out_expansion = out_dir / f"{phenotype}_Top{top_n}_ExpandedModuleGenes.csv"
    out_enrich = out_dir / f"{phenotype}_Top{top_n}_PathwayEnrichment.csv"
    
    mods_df.to_csv(out_summary, index=False)
    expansion_df.to_csv(out_expansion, index=False)
    enrich_df.to_csv(out_enrich, index=False)
    
    print(f"\n💾 SAVED:")
    print(f"   {out_summary}")
    print(f"   {out_expansion}  ← USE THIS FOR DRUG DISCOVERY")
    print(f"   {out_enrich}")

# =============================================================================
# MAIN
# =============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phenotype", type=str)
    ap.add_argument("--top-genes", type=int, default=200)
    ap.add_argument("--min-edge", type=float, default=0.4)
    ap.add_argument("--fdr-threshold", type=float, default=0.05, help="FDR threshold for pathway expansion")
    ap.add_argument("--top-pathways", type=int, default=DEFAULT_TOP_PATHWAYS_PER_COMMUNITY, help="Top pathways per community to print")
    ap.add_argument("--max-comms", type=int, default=DEFAULT_MAX_COMMUNITIES_TO_PRINT, help="Max communities to print")
    args = ap.parse_args()

    phenotype = args.phenotype.strip()
    if not phenotype:
        die("Empty phenotype")

    pheno_dir = BASE_DIR / phenotype
    edges_path = pheno_dir / EDGES_REL
    if not edges_path.exists():
        die(f"STRING edges not found: {edges_path}")

    out_dir = pheno_dir / "GeneDifferentialExpression" / "Files" / "UltimateCompleteRankingAnalysis" / OUT_DIRNAME / f"Top{args.top_genes}"

    print("=" * 100)
    print("PPI SUBMODULE ANALYSIS WITH PATHWAY EXPANSION")
    print("=" * 100)

    # 1. Load genes
    top_syms, score_map, _ = load_top_genes_from_file(phenotype, args.top_genes)

    # 2. Build PPI
    G, edges_in_induced = build_induced_graph(edges_path, top_syms, args.min_edge)

    # 3. Communities
    print("\n📊 Detecting communities...")
    modules = list(greedy_modularity_communities(G, weight="weight"))
    modules = sorted(modules, key=lambda s: len(s), reverse=True)
    print(f"   ✓ Found {len(modules)} communities")

    # 4. Enrichment
    mapping_file = find_first_existing(pheno_dir, PATHWAY_MAP_CANDIDATES_REL)
    if mapping_file is None:
        die("No pathway mapping file found")
    
    print(f"\n📊 Using pathway file: {mapping_file.name}")
    pathway_to_genes, bg_genes, mode = load_pathway_mapping_from_file(mapping_file)
    
    if not pathway_to_genes:
        die(f"Pathway file not usable (mode={mode})")
    
    print(f"   ✓ Loaded {len(pathway_to_genes)} pathways, {len(bg_genes)} background genes")
    
    enrich_df = enrich_modules(modules, pathway_to_genes, bg_genes, score_map)

    # ✅ 5. PRINT COMMUNITIES WITH PATHWAYS
    pretty_print_communities(
        phenotype=phenotype,
        modules=modules,
        enrich_df=enrich_df,
        G=G,
        edges_in_induced=edges_in_induced,
        min_edge_score01=args.min_edge,
        mapping_file=mapping_file,
        top_paths_per_comm=args.top_pathways,
        max_comms_to_print=args.max_comms
    )

    # 6. ✅ EXPAND MODULES WITH PATHWAY GENES
    if not enrich_df.empty:
        expanded_modules, expansion_df = expand_modules_with_pathways(
            modules, enrich_df, pathway_to_genes, args.fdr_threshold
        )
    else:
        print("\n⚠️  No enrichments found - skipping expansion")
        expanded_modules = {mid: set(modules[mid-1]) for mid in range(1, len(modules)+1)}
        expansion_df = pd.DataFrame()

    # 7. Save
    write_outputs(phenotype, out_dir, args.top_genes, modules, expanded_modules, expansion_df, enrich_df, score_map)

    print("\n" + "=" * 100)
    print("✅ COMPLETE!")
    print("=" * 100)
    print(f"\n📁 Output directory: {out_dir}")
    print(f"\n💡 Next step: Use ExpandedModuleGenes.csv for drug discovery")
    print(f"   Example: python CommunityDrugDiscovery.py {phenotype} --input {out_dir}/...")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
 