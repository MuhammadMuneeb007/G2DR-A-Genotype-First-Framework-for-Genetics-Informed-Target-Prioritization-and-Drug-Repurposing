#!/usr/bin/env python3
"""
List common drugs between predicted DrugRanking.csv and migraine_drugs.csv
+ Compute overlap metrics (Precision@K, Recall@K, F1@K, Hypergeom p, Fold enrichment)

ADDED (without changing existing behavior):
  ✅ AUROC + AUPRC computed over the FULL ranked list (all predicted unique drugs)
  ✅ Recall@K / Precision@K / F1@K for multiple K (already existed; kept)
  ✅ Option to use ALL drugs as universe (not just predicted)
  ✅ Degree/Drug-count matched GENE permutation test (empirical p-values) using cache

Usage (original):
  python predict4.1.2.12-ListCommonDrugs.py migraine
  python predict4.1.2.12-ListCommonDrugs.py migraine --k 20,50,100
  python predict4.1.2.12-ListCommonDrugs.py migraine --ref /path/to/migraine_drugs.csv

New options:
  --universe [predicted|all]    Universe for hypergeom test (default: all)
  --all-drugs-db PATH           Path to database with ALL drugs
  --auroc-auprc                 Compute AUROC + AUPRC over full ranked drug list (default: on)
  --permute-genes               Run degree-matched gene permutation test (default: off)
  --perm-n                      Number of permutations (default: 1000)
  --top-genes-for-perm N        Which TOP_N genes file to use for gene permutations (default: 2000)
  --perm-bin-quantiles q        Quantile bins for matching drug-count-per-gene (default: 10)
  --perm-seed SEED              RNG seed
  --perm-out-prefix PREFIX      Prefix for permutation outputs
"""

import argparse
import re
import math
import json
from pathlib import Path
from typing import Set, Optional
import pandas as pd

# Prefer scipy if available for hypergeometric survival function
try:
    from scipy.stats import hypergeom
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# AUROC/AUPRC (sklearn)
try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False

# Default path to ALL drugs database
ALL_DRUGS_DB_DEFAULT = "/data/ascher02/uqmmune1/ANNOVAR/AllDiseasesToDrugs/ALL_SOURCES_drug_disease_merged.csv"


def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_drug_name(x: str) -> str:
    """Conservative normalizer: intended to reduce trivial name differences."""
    if x is None:
        return ""
    s = str(x).strip().lower()
    if not s or s in {"nan", "none"}:
        return ""

    # remove bracketed additions
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"\[.*?\]", " ", s)

    # normalize separators
    s = s.replace("&", " and ")
    s = re.sub(r"[/,+;]", " ", s)

    # remove common dosage tokens
    s = re.sub(r"\b\d+(\.\d+)?\s*(mg|mcg|g|ml|iu|units?)\b", " ", s)

    # remove common salt words (conservative)
    salt_words = [
        "hydrochloride", "hcl", "sodium", "potassium", "calcium",
        "succinate", "tartrate", "maleate", "phosphate", "sulfate",
        "acetate", "chloride", "nitrate", "mesylate", "besylate",
        "benzoate", "bromide", "citrate", "lactate"
    ]
    s = re.sub(r"\b(" + "|".join(map(re.escape, salt_words)) + r")\b", " ", s)

    # keep alphanum only
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_all_drugs_universe(db_path: Path) -> Set[str]:
    """
    Load ALL unique normalized drug names from the merged drug-disease database.
    This becomes the universe for hypergeometric tests.
    """
    print(f"\n📂 Loading ALL drugs universe from: {db_path}")
    
    if not db_path.exists():
        raise FileNotFoundError(f"All drugs database not found: {db_path}")
    
    # Read only drug-related columns to save memory
    df = pd.read_csv(db_path, usecols=lambda c: c.lower() in ['drug_norm', 'drugnorm', 'drug_name', 'drugname', 'drug'])
    
    # Find the drug column
    drug_col = None
    for col in df.columns:
        if col.lower() in ['drug_norm', 'drugnorm']:
            drug_col = col
            break
    if drug_col is None:
        for col in df.columns:
            if 'drug' in col.lower():
                drug_col = col
                break
    
    if drug_col is None:
        raise ValueError(f"Could not find drug column in {db_path}. Columns: {list(df.columns)}")
    
    # Normalize all drug names
    all_drugs = set()
    for name in df[drug_col].dropna().unique():
        norm = normalize_drug_name(str(name))
        if norm:
            all_drugs.add(norm)
    
    print(f"✅ Loaded ALL drugs universe: {len(all_drugs):,} unique normalized drugs")
    return all_drugs


def load_predicted(drug_ranking_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(drug_ranking_csv)

    # Prefer already-normalized key from your pipeline
    norm_col = pick_col(df, ["DrugNorm", "drug_norm", "drugnorm"])
    bestname_col = pick_col(df, ["DrugName_best", "DrugName", "drug_name", "name"])

    if not norm_col and not bestname_col:
        raise ValueError(
            f"Could not find a drug identifier column in {drug_ranking_csv.name}. "
            f"Columns: {list(df.columns)[:50]}"
        )

    rank_col = pick_col(df, ["DrugRank", "drug_rank", "Rank", "rank"])
    score_col = pick_col(df, ["DrugScore", "drug_score", "Score", "score"])

    out = df.copy()

    # Matching key:
    # 1) if DrugNorm exists, normalize it lightly (still run through normalize_drug_name for robustness)
    # 2) else normalize the best name column
    if norm_col:
        out["DrugNorm_key"] = out[norm_col].astype(str).map(normalize_drug_name)
    else:
        out["DrugNorm_key"] = out[bestname_col].astype(str).map(normalize_drug_name)

    # Display name for printing
    if bestname_col:
        out["DrugName_pred"] = out[bestname_col].astype(str)
    else:
        out["DrugName_pred"] = out[norm_col].astype(str)

    # Rank
    if rank_col:
        out["DrugRank"] = pd.to_numeric(out[rank_col], errors="coerce")
    else:
        out["DrugRank"] = pd.NA

    # Score
    if score_col:
        out["DrugScore"] = pd.to_numeric(out[score_col], errors="coerce")
    else:
        out["DrugScore"] = pd.NA

    out = out[out["DrugNorm_key"] != ""].copy()

    # Ensure rank exists
    if out["DrugRank"].isna().all():
        # If no usable rank, rank by score desc; otherwise keep file order
        if out["DrugScore"].notna().any():
            out = out.sort_values("DrugScore", ascending=False).copy()
        out["DrugRank"] = range(1, len(out) + 1)

    # Deduplicate by normalized key (keep best rank)
    out = out.sort_values("DrugRank", ascending=True).drop_duplicates(subset=["DrugNorm_key"], keep="first")

    return out


def load_reference(ref_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(ref_csv)

    name_col = pick_col(df, ["drug_name", "DrugName", "drug", "Drug", "name", "intervention", "Intervention"])
    if not name_col:
        raise ValueError(f"Could not find drug name column in {ref_csv.name}. Columns: {list(df.columns)[:50]}")

    out = df.copy()
    out["DrugName_ref"] = out[name_col].astype(str)
    out["DrugNorm_key"] = out["DrugName_ref"].map(normalize_drug_name)
    out = out[out["DrugNorm_key"] != ""].copy()

    # Dedup reference by normalized key
    out = out.drop_duplicates(subset=["DrugNorm_key"], keep="first")
    return out


def hypergeom_p_value(M: int, n: int, N: int, x: int) -> float:
    """
    Universe size M,
    successes in population n (reference drugs in universe),
    draws N (top-K predicted),
    observed overlap x (common).
    Returns P(X >= x) (survival).
    """
    if x <= 0:
        return 1.0
    if M <= 0 or n < 0 or N < 0 or x < 0:
        return float("nan")
    if n > M:
        n = M
    if N > M:
        N = M

    if HAVE_SCIPY:
        return float(hypergeom.sf(x - 1, M, n, N))

    # Fallback: exact sum using comb (can be slow for huge M, but fine here)
    # P(X >= x) = sum_{k=x..min(n,N)} [C(n,k) * C(M-n, N-k)] / C(M,N)
    denom = math.comb(M, N)
    kmax = min(n, N)
    num = 0
    for k in range(x, kmax + 1):
        num += math.comb(n, k) * math.comb(M - n, N - k)
    return num / denom if denom else float("nan")


def compute_metrics(
    pred_df: pd.DataFrame, 
    ref_set: set, 
    ks: list,
    universe_set: Optional[Set[str]] = None,
    universe_mode: str = "predicted"
) -> pd.DataFrame:
    """
    Compute overlap metrics at various K values.
    
    Args:
        pred_df: Predicted drugs dataframe with DrugNorm_key and DrugRank
        ref_set: Set of reference drug normalized names
        ks: List of K values to evaluate
        universe_set: Optional set of ALL drugs (for universe_mode="all")
        universe_mode: "predicted" or "all"
    """
    pred_set = set(pred_df["DrugNorm_key"].tolist())
    
    # Determine universe and reference count based on mode
    if universe_mode == "all" and universe_set is not None:
        M = len(universe_set)  # ALL drugs
        # Reference drugs that exist in the universe
        ref_in_universe = ref_set & universe_set
        n = len(ref_in_universe)
        universe_label = "ALL_DRUGS"
    else:
        M = len(pred_set)  # Only predicted drugs
        n = len(ref_set)
        ref_in_universe = ref_set
        universe_label = "PREDICTED"

    rows = []
    for K in ks:
        topk = pred_df.sort_values("DrugRank", ascending=True).head(K)
        topk_set = set(topk["DrugNorm_key"].tolist())
        
        # Overlap with reference (that exists in universe)
        x = len(topk_set & ref_in_universe)

        precision = x / K if K > 0 else 0.0
        recall = x / n if n > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        # Expected overlap under random draws
        # If universe_mode="all": drawing K from M (all drugs), n successes in population
        # If universe_mode="predicted": drawing K from M (predicted), n successes
        expected = (K * n / M) if M > 0 else float("nan")
        fold = (x / expected) if expected and expected > 0 else float("nan")

        p = hypergeom_p_value(M=M, n=n, N=min(K, M), x=x) if (M > 0 and n > 0 and K > 0) else float("nan")

        rows.append({
            "K": int(K),
            "Universe_Type": universe_label,
            "Universe_size": int(M),
            "Reference_in_universe": int(n),
            "Overlap": int(x),
            "Precision@K": float(precision),
            "Recall@K": float(recall),
            "F1@K": float(f1),
            "Expected_overlap": float(expected) if expected == expected else float("nan"),
            "Fold_enrichment": float(fold) if fold == fold else float("nan"),
            "Hypergeom_p": float(p) if p == p else float("nan")
        })

    return pd.DataFrame(rows)


# =============================================================================
# ADDED: AUROC/AUPRC over full ranked list
# =============================================================================
def compute_auc_metrics(pred_df: pd.DataFrame, ref_set: set) -> dict:
    """
    Treat each predicted drug as an item with a rank.
    y_true = 1 if drug in reference set else 0
    y_score = -rank (higher score = better)
    AUROC and AUPRC computed on all predicted drugs.
    """
    if not HAVE_SKLEARN:
        return {
            "AUROC": float("nan"),
            "AUPRC": float("nan"),
            "N_pos": 0,
            "N_total": int(len(pred_df)),
        }

    d = pred_df.sort_values("DrugRank", ascending=True).copy()

    # ground-truth labels
    y_true = d["DrugNorm_key"].apply(lambda k: 1 if k in ref_set else 0).astype(int).to_numpy()
    n_pos = int(y_true.sum())
    n_total = int(len(y_true))

    # numeric ranks (may contain NaNs)
    ranks = pd.to_numeric(d["DrugRank"], errors="coerce")

    # fill missing ranks with 1..N as a Series aligned to d.index (NOT RangeIndex object)
    fallback = pd.Series(range(1, len(d) + 1), index=d.index, dtype=float)
    ranks = ranks.fillna(fallback)

    # score: higher is better -> negative rank
    y_score = (-ranks).to_numpy()

    # AUROC requires both classes present
    auroc = float("nan")
    if n_pos > 0 and n_pos < n_total:
        auroc = float(roc_auc_score(y_true, y_score))

    # AUPRC works as long as there is at least one positive
    auprc = float("nan")
    if n_pos > 0:
        auprc = float(average_precision_score(y_true, y_score))

    return {"AUROC": auroc, "AUPRC": auprc, "N_pos": n_pos, "N_total": n_total}


# =============================================================================
# ADDED: Degree/drug-count matched GENE permutation test
# =============================================================================
def _load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def _clean_sym_upper(x: object) -> str:
    if x is None:
        return ""
    s = str(x).strip().upper()
    if not s or s == "NAN":
        return ""
    return s

def load_gene_list_for_perm(base: Path, phenotype: str, top_genes_for_perm: int) -> list:
    """
    Load top gene list from:
      <base>/<phenotype>/GeneDifferentialExpression/Files/UltimateCompleteRankingAnalysis/FinalIntegration/TOP_{N}_1.csv
    Accepts columns Gene or Symbol (or uses Gene).
    Returns list of SYMBOLS (upper).
    """
    fp = (
        base / phenotype / "GeneDifferentialExpression" / "Files" / "UltimateCompleteRankingAnalysis" /
        "FinalIntegration" / f"TOP_{top_genes_for_perm}_1.csv"
    )
    if not fp.exists():
        raise FileNotFoundError(f"Missing top genes file for permutation test: {fp}")

    df = pd.read_csv(fp, low_memory=False)

    sym_col = pick_col(df, ["Symbol", "Gene_Symbol", "GeneSymbol"])
    gene_col = pick_col(df, ["Gene", "gene", "ENSEMBL", "Ensembl_ID", "ensembl_gene_id"])

    if sym_col:
        syms = [_clean_sym_upper(x) for x in df[sym_col].tolist()]
    elif gene_col:
        # If ensembl IDs, we will map later; but keep as string for now
        syms = [str(x).strip() for x in df[gene_col].tolist()]
    else:
        raise ValueError(f"Cannot find Gene/Symbol column in {fp}. Columns: {list(df.columns)[:60]}")

    syms = [s for s in syms if s]
    if len(syms) < 10:
        raise ValueError(f"Too few genes loaded from {fp} (n={len(syms)})")
    return syms

def build_symbol_to_ensembl_sets(ensembl_to_symbol: dict) -> dict:
    """
    Reverse map Symbol -> set(ensembl)
    """
    sym2ens = {}
    for eid, sym in ensembl_to_symbol.items():
        su = _clean_sym_upper(sym)
        if not su:
            continue
        sym2ens.setdefault(su, set()).add(str(eid).strip())
    return sym2ens

def get_normalized_drugs_for_gene_ensembl(cache_data: dict, ensembl_id: str) -> set:
    """
    cache_data format: { ensembl_id: {source: [drug1, drug2,...], ... }, ... }
    Return normalized drugs across sources.
    """
    out = set()
    src_map = cache_data.get(ensembl_id, {})
    if not isinstance(src_map, dict):
        return out
    for _, lst in src_map.items():
        if not isinstance(lst, list):
            continue
        for d in lst:
            dn = normalize_drug_name(d)
            if dn:
                out.add(dn)
    return out

def drugs_for_symbol_set(symbols: list, sym2ens: dict, cache_data: dict) -> tuple:
    """
    Given list of gene symbols, resolve to ensembl IDs, union drugs.
    Returns: (drug_set, gene_to_drugcount)
    """
    all_drugs = set()
    gene_to_cnt = {}
    for s in symbols:
        su = _clean_sym_upper(s)
        if not su:
            continue
        eids = sym2ens.get(su, set())
        drugs = set()
        for eid in eids:
            drugs |= get_normalized_drugs_for_gene_ensembl(cache_data, eid)
        if drugs:
            gene_to_cnt[su] = len(drugs)
            all_drugs |= drugs
        else:
            gene_to_cnt[su] = 0
    return all_drugs, gene_to_cnt

def make_quantile_bins(values: list, q: int):
    """
    Create quantile bin edges for nonnegative ints.
    Returns edges list of length q+1.
    """
    ser = pd.Series(values, dtype=float)
    # If many zeros, qcut can fail; fall back to unique bins
    try:
        cats, edges = pd.qcut(ser, q=q, retbins=True, duplicates="drop")
        return list(edges)
    except Exception:
        # Simple fallback: bins by unique sorted values
        uniq = sorted(set(int(v) for v in values))
        if len(uniq) <= 1:
            return [0.0, float(max(uniq[0], 1))]
        return [float(uniq[0])] + [float(u) for u in uniq[1:]] + [float(uniq[-1]) + 1.0]

def assign_bin(x: int, edges: list) -> int:
    """
    Assign x to bin index based on edges (like np.digitize).
    """
    if not edges or len(edges) < 2:
        return 0
    # right-open bins except last
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            if x >= lo and x <= hi:
                return i
        if x >= lo and x < hi:
            return i
    return len(edges) - 2

def degree_matched_gene_permutation(
    phenotype: str,
    base: Path,
    ref_set: set,
    top_genes_for_perm: int,
    n_perm: int,
    bin_quantiles: int,
    seed: int,
    outdir: Path,
    out_prefix: str,
):
    """
    Empirical test:
      Compare observed migraine-drug overlap from the TOP gene set vs random gene sets
      matched by gene drug-count bins (#drugs per gene).
    Output:
      - <out_prefix>_perm_summary.csv
      - <out_prefix>_perm_distribution.csv
      - <out_prefix>_perm_report.txt
    """
    cache_dir = base / "PermutationCache"
    cache_file = cache_dir / f"{phenotype}_gene_drugs_incremental.json"
    map_file = cache_dir / f"{phenotype}_ensembl_to_symbol.json"

    if not cache_file.exists():
        raise FileNotFoundError(f"Missing cache: {cache_file}")
    if not map_file.exists():
        raise FileNotFoundError(f"Missing mapping: {map_file}")

    cache_data = _load_json(cache_file)
    ensembl_to_symbol = _load_json(map_file)
    sym2ens = build_symbol_to_ensembl_sets(ensembl_to_symbol)

    # All symbols that are mappable
    all_symbols = sorted(sym2ens.keys())
    if len(all_symbols) < 100:
        raise ValueError(f"Too few symbols in mapping ({len(all_symbols)}). Check {map_file}")

    # Compute drug-count per gene for all symbols (hubness)
    print("\n🧪 DEGREE-MATCHED GENE PERMUTATION TEST")
    print("-" * 80)
    print(f"Cache genes: {len(cache_data)} | Mapped symbols: {len(all_symbols)}")

    # Precompute gene drug counts
    gene_drugcount = {}
    for s in all_symbols:
        eids = sym2ens.get(s, set())
        drugs = set()
        for eid in eids:
            drugs |= get_normalized_drugs_for_gene_ensembl(cache_data, eid)
        gene_drugcount[s] = len(drugs)

    # Load your observed gene list
    obs_symbols_raw = load_gene_list_for_perm(base, phenotype, top_genes_for_perm)
    # Convert to symbols if they are Ensembl IDs: use mapping
    obs_symbols = []
    for g in obs_symbols_raw:
        gu = str(g).strip()
        # if looks like ENSG... map
        if gu.startswith("ENSG"):
            sym = _clean_sym_upper(ensembl_to_symbol.get(gu, ""))
            if sym:
                obs_symbols.append(sym)
        else:
            su = _clean_sym_upper(gu)
            if su:
                obs_symbols.append(su)

    # Keep only symbols that are in mapping
    obs_symbols = [s for s in obs_symbols if s in sym2ens]
    # Dedup but preserve order-ish
    seen = set()
    obs_symbols2 = []
    for s in obs_symbols:
        if s not in seen:
            seen.add(s)
            obs_symbols2.append(s)
    obs_symbols = obs_symbols2

    if len(obs_symbols) < 20:
        raise ValueError(f"Observed gene list too small after mapping (n={len(obs_symbols)}).")

    # Observed drugs derived from observed gene set
    obs_drugs, obs_gene_to_cnt = drugs_for_symbol_set(obs_symbols, sym2ens, cache_data)
    obs_overlap = len(obs_drugs & ref_set)
    obs_precision = (obs_overlap / len(obs_drugs)) if len(obs_drugs) > 0 else 0.0
    obs_recall = (obs_overlap / len(ref_set)) if len(ref_set) > 0 else 0.0

    print(f"Observed genes: {len(obs_symbols)} (TOP_{top_genes_for_perm})")
    print(f"Observed derived drugs: {len(obs_drugs)}")
    print(f"Observed overlap with ref drugs: {obs_overlap}")
    print(f"Observed precision: {obs_precision:.4f} | recall: {obs_recall:.4f}")

    # Matching bins based on drug-counts
    obs_counts = [gene_drugcount.get(s, 0) for s in obs_symbols]
    edges = make_quantile_bins(list(gene_drugcount.values()), q=bin_quantiles)

    obs_bins = [assign_bin(c, edges) for c in obs_counts]

    # Build bin -> candidate symbols pool
    bin_to_pool = {}
    for s in all_symbols:
        b = assign_bin(gene_drugcount[s], edges)
        bin_to_pool.setdefault(b, []).append(s)

    # For reproducibility
    import random
    random.seed(seed)

    perm_dist = []

    for i in range(1, n_perm + 1):
        # sample matched genes: for each observed gene bin, sample one from same bin
        chosen = []
        used = set()
        for b in obs_bins:
            pool = bin_to_pool.get(b, [])
            if not pool:
                continue
            # sample until unique
            for _ in range(10):
                cand = random.choice(pool)
                if cand not in used:
                    used.add(cand)
                    chosen.append(cand)
                    break
        # if we failed to fill fully, top up randomly from overall
        if len(chosen) < len(obs_symbols):
            remaining = [s for s in all_symbols if s not in used]
            need = len(obs_symbols) - len(chosen)
            if need > 0 and len(remaining) >= need:
                chosen.extend(random.sample(remaining, need))

        # derive drugs and compute overlap
        perm_drugs, _ = drugs_for_symbol_set(chosen, sym2ens, cache_data)
        ov = len(perm_drugs & ref_set)
        prec = (ov / len(perm_drugs)) if len(perm_drugs) > 0 else 0.0
        rec = (ov / len(ref_set)) if len(ref_set) > 0 else 0.0

        perm_dist.append({
            "perm": i,
            "n_genes": len(chosen),
            "n_drugs": len(perm_drugs),
            "overlap": ov,
            "precision": prec,
            "recall": rec,
        })

    dist_df = pd.DataFrame(perm_dist)

    # empirical p-values (one-sided: >= observed)
    p_overlap = ( (dist_df["overlap"] >= obs_overlap).sum() + 1 ) / (len(dist_df) + 1)
    p_precision = ( (dist_df["precision"] >= obs_precision).sum() + 1 ) / (len(dist_df) + 1)
    p_recall = ( (dist_df["recall"] >= obs_recall).sum() + 1 ) / (len(dist_df) + 1)

    summary = {
        "phenotype": phenotype,
        "top_genes_for_perm": int(top_genes_for_perm),
        "n_perm": int(n_perm),
        "bin_quantiles": int(bin_quantiles),
        "seed": int(seed),
        "observed_genes": int(len(obs_symbols)),
        "observed_drugs": int(len(obs_drugs)),
        "observed_overlap": int(obs_overlap),
        "observed_precision": float(obs_precision),
        "observed_recall": float(obs_recall),
        "emp_p_overlap_ge": float(p_overlap),
        "emp_p_precision_ge": float(p_precision),
        "emp_p_recall_ge": float(p_recall),
        "perm_overlap_mean": float(dist_df["overlap"].mean()),
        "perm_overlap_std": float(dist_df["overlap"].std(ddof=0)),
        "perm_precision_mean": float(dist_df["precision"].mean()),
        "perm_recall_mean": float(dist_df["recall"].mean()),
    }

    outdir.mkdir(parents=True, exist_ok=True)
    sum_csv = outdir / f"{out_prefix}_perm_summary.csv"
    dist_csv = outdir / f"{out_prefix}_perm_distribution.csv"
    dist_df.to_csv(dist_csv, index=False)
    pd.DataFrame([summary]).to_csv(sum_csv, index=False)

    rep_txt = outdir / f"{out_prefix}_perm_report.txt"
    with open(rep_txt, "w") as f:
        f.write("DEGREE/DRUG-COUNT MATCHED GENE PERMUTATION TEST\n")
        f.write(f"Phenotype: {phenotype}\n")
        f.write(f"Observed genes: {summary['observed_genes']} (TOP_{top_genes_for_perm})\n")
        f.write(f"Observed derived drugs: {summary['observed_drugs']}\n")
        f.write(f"Observed overlap: {summary['observed_overlap']}\n")
        f.write(f"Observed precision: {summary['observed_precision']:.6f}\n")
        f.write(f"Observed recall: {summary['observed_recall']:.6f}\n")
        f.write("\nPermutation results (matched by gene #drugs bins):\n")
        f.write(f"n_perm: {n_perm}\n")
        f.write(f"Overlap mean±std: {summary['perm_overlap_mean']:.3f} ± {summary['perm_overlap_std']:.3f}\n")
        f.write(f"Precision mean: {summary['perm_precision_mean']:.6f}\n")
        f.write(f"Recall mean: {summary['perm_recall_mean']:.6f}\n")
        f.write("\nEmpirical one-sided p-values (>= observed):\n")
        f.write(f"p_overlap: {summary['emp_p_overlap_ge']:.6g}\n")
        f.write(f"p_precision: {summary['emp_p_precision_ge']:.6g}\n")
        f.write(f"p_recall: {summary['emp_p_recall_ge']:.6g}\n")

    print(f"\n💾 Saved permutation summary: {sum_csv}")
    print(f"💾 Saved permutation distribution: {dist_csv}")
    print(f"💾 Saved permutation report: {rep_txt}")

    return summary, dist_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phenotype")
    ap.add_argument("--k", default="20,50,100", help="comma-separated top-K values to compute overlap for")
    ap.add_argument("--ref", default=None, help="path to migraine_drugs.csv (default: /data/.../migraine_drugs.csv or ./migraine_drugs.csv)")
    ap.add_argument("--outdir", default=None, help="override output dir (default: <phenotype>/.../DrugIntegration)")

    # Universe options
    ap.add_argument("--universe", choices=["predicted", "all"], default="all",
                    help="Universe for hypergeometric test: 'predicted' (your drugs only) or 'all' (all drugs in DB)")
    ap.add_argument("--all-drugs-db", default=ALL_DRUGS_DB_DEFAULT,
                    help="Path to database with ALL drugs (for --universe=all)")

    # AUROC/AUPRC
    ap.add_argument("--auroc-auprc", action="store_true", default=False,
                    help="Compute AUROC/AUPRC over full ranked list (requires scikit-learn).")
    
    # Gene permutation test options
    ap.add_argument("--permute-genes", action="store_true", default=False,
                    help="Run degree/drug-count matched gene permutation test (uses PermutationCache).")
    ap.add_argument("--perm-n", type=int, default=1000, help="Number of gene permutations.")
    ap.add_argument("--top-genes-for-perm", type=int, default=2000,
                    help="Which TOP_N gene list to use for permutation test (FinalIntegration/TOP_{N}_1.csv).")
    ap.add_argument("--perm-bin-quantiles", type=int, default=10,
                    help="Quantile bins for matching gene drug-count hubness.")
    ap.add_argument("--perm-seed", type=int, default=1337, help="Seed for permutation RNG.")
    ap.add_argument("--perm-out-prefix", type=str, default="GenePerm",
                    help="Prefix for permutation output files.")
    args = ap.parse_args()

    phenotype = args.phenotype

    base = Path("/data/ascher02/uqmmune1/ANNOVAR")
    default_outdir = base / phenotype / "GeneDifferentialExpression" / "Files" / "UltimateCompleteRankingAnalysis" / "DrugIntegration"
    outdir = Path(args.outdir) if args.outdir else default_outdir
    outdir.mkdir(parents=True, exist_ok=True)

    drug_ranking_csv = outdir / "DrugRanking.csv"
    if not drug_ranking_csv.exists():
        raise FileNotFoundError(f"Missing: {drug_ranking_csv}")

    # Resolve reference csv
    ref_csv = Path(args.ref) if args.ref else (base / "migraine_drugs2.csv")
    if not ref_csv.exists():
        ref_csv = Path("migraine_drugs.csv")
    if not ref_csv.exists():
        raise FileNotFoundError("Could not find migraine_drugs.csv (tried /data/.../migraine_drugs.csv and ./migraine_drugs.csv)")

    ks = [int(x.strip()) for x in args.k.split(",") if x.strip()]
    ks = sorted(set(ks))

    print(f"📌 Universe mode: {args.universe.upper()}")
    
    # Load ALL drugs universe if needed
    all_drugs_universe: Optional[Set[str]] = None
    if args.universe == "all":
        all_drugs_universe = load_all_drugs_universe(Path(args.all_drugs_db))

    pred = load_predicted(drug_ranking_csv)
    ref = load_reference(ref_csv)

    ref_set = set(ref["DrugNorm_key"].tolist())
    pred_set = set(pred["DrugNorm_key"].tolist())

    # ALL overlap list
    all_common_keys = sorted(ref_set & pred_set)
    all_common_df = (
        pred[pred["DrugNorm_key"].isin(all_common_keys)]
        .merge(ref[["DrugNorm_key", "DrugName_ref"]], on="DrugNorm_key", how="left")
        .sort_values("DrugRank", ascending=True)
        [["DrugRank", "DrugScore", "DrugName_pred", "DrugName_ref", "DrugNorm_key"]]
    )

    print("\n=== OVERALL OVERLAP (ALL predicted vs reference) ===")
    print(f"Predicted unique drugs: {len(pred_set)}")
    print(f"Reference unique drugs: {len(ref_set)}")
    print(f"Common (ALL): {len(all_common_df)}")
    if len(all_common_df) > 0:
        print(all_common_df.head(200).to_string(index=False))
    else:
        print("No common drugs found after normalization.")

    all_common_out = outdir / "CommonDrugs_ALL.csv"
    all_common_df.to_csv(all_common_out, index=False)
    print(f"\n💾 Saved: {all_common_out}")

    # TOP-K overlap lists + metrics
    print("\n=== TOP-K OVERLAPS ===")
    summary_rows = []

    for K in ks:
        topk = pred.sort_values("DrugRank", ascending=True).head(K)
        topk_set = set(topk["DrugNorm_key"].tolist())
        common_k = sorted(topk_set & ref_set)

        common_k_df = (
            topk[topk["DrugNorm_key"].isin(common_k)]
            .merge(ref[["DrugNorm_key", "DrugName_ref"]], on="DrugNorm_key", how="left")
            .sort_values("DrugRank", ascending=True)
            [["DrugRank", "DrugScore", "DrugName_pred", "DrugName_ref", "DrugNorm_key"]]
        )

        print(f"\n--- Common drugs in Top-{K} (n={len(common_k_df)}) ---")
        if len(common_k_df) > 0:
            print(common_k_df.to_string(index=False))
        else:
            print("(none)")

        common_k_out = outdir / f"CommonDrugs_Top{K}.csv"
        common_k_df.to_csv(common_k_out, index=False)
        print(f"💾 Saved: {common_k_out}")

        summary_rows.append({"K": int(K), "Common": int(len(common_k_df))})

    summary_out = outdir / "CommonDrugs_Summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_out, index=False)
    print(f"\n💾 Saved: {summary_out}")

    # Metrics table with selected universe
    print(f"\n📊 Overlap metrics (Universe = {args.universe.upper()}):")
    metrics_df = compute_metrics(
        pred, 
        ref_set, 
        ks,
        universe_set=all_drugs_universe,
        universe_mode=args.universe
    )
    print(metrics_df.to_string(index=False))

    metrics_out = outdir / f"CommonDrugs_Metrics_{args.universe.upper()}.csv"
    metrics_df.to_csv(metrics_out, index=False)
    print(f"\n💾 Saved: {metrics_out}")

    # Also compute with PREDICTED universe for comparison if using "all"
    if args.universe == "all":
        print(f"\n📊 COMPARISON: Metrics with PREDICTED universe:")
        metrics_df_pred = compute_metrics(pred, ref_set, ks, universe_set=None, universe_mode="predicted")
        print(metrics_df_pred.to_string(index=False))
        
        metrics_out_pred = outdir / "CommonDrugs_Metrics_PREDICTED.csv"
        metrics_df_pred.to_csv(metrics_out_pred, index=False)
        print(f"💾 Saved: {metrics_out_pred}")

    # AUROC/AUPRC
    want_auc = True
    if args.auroc_auprc:
        want_auc = True

    auc_info = {"AUROC": float("nan"), "AUPRC": float("nan"), "N_pos": 0, "N_total": int(len(pred))}
    if want_auc:
        auc_info = compute_auc_metrics(pred, ref_set)

        print("\n📈 Ranking metrics over FULL list:")
        if HAVE_SKLEARN:
            print(f"AUROC: {auc_info['AUROC']}")
            print(f"AUPRC: {auc_info['AUPRC']}")
            print(f"N positives in list: {auc_info['N_pos']} / {auc_info['N_total']}")
        else:
            print("scikit-learn not available; AUROC/AUPRC not computed (install sklearn).")

        auc_out = outdir / "CommonDrugs_AUROC_AUPRC.csv" 
        pd.DataFrame([auc_info]).to_csv(auc_out, index=False)
        print(f"💾 Saved: {auc_out}")

    # Save a "human summary" text file
    txt_out = outdir / "CommonDrugs_Metrics.txt"
    with open(txt_out, "w") as f:
        f.write("OVERLAP METRICS (Predicted DrugRanking vs migraine_drugs reference)\n")
        f.write(f"Universe mode: {args.universe.upper()}\n")
        if args.universe == "all" and all_drugs_universe:
            f.write(f"ALL drugs universe size: {len(all_drugs_universe)}\n")
        f.write(f"Predicted unique drugs: {len(pred_set)}\n")
        f.write(f"Reference unique drugs: {len(ref_set)}\n")
        f.write(f"All-overlap count: {len(all_common_df)}\n\n")
        f.write(f"METRICS ({args.universe.upper()} UNIVERSE):\n")
        f.write(metrics_df.to_string(index=False))
        if args.universe == "all":
            f.write("\n\nMETRICS (PREDICTED UNIVERSE for comparison):\n")
            f.write(metrics_df_pred.to_string(index=False))
        f.write("\n\nRANKING METRICS (FULL LIST):\n")
        if HAVE_SKLEARN:
            f.write(f"AUROC: {auc_info['AUROC']}\n")
            f.write(f"AUPRC: {auc_info['AUPRC']}\n")
            f.write(f"N positives in list: {auc_info['N_pos']} / {auc_info['N_total']}\n")
        else:
            f.write("scikit-learn not available; AUROC/AUPRC not computed.\n")
        f.write("\nNOTE:\n")
        f.write("Hypergeom_p = P(X >= overlap) under random draws of size K from the universe.\n")
        f.write("Fold_enrichment = observed_overlap / expected_overlap.\n")
        f.write("When universe='all': Expected = K * (Ref_in_universe / Universe_size)\n")
        f.write("When universe='predicted': Expected = K * (Ref / Predicted_size)\n")
        if not HAVE_SCIPY:
            f.write("SciPy not found: hypergeometric p-values computed via exact comb sums.\n")
    print(f"💾 Saved: {txt_out}")

    # Degree/drug-count matched gene permutation test
    if args.permute_genes:
        perm_outdir = outdir
        degree_matched_gene_permutation(
            phenotype=phenotype,
            base=base,
            ref_set=ref_set,
            top_genes_for_perm=int(args.top_genes_for_perm),
            n_perm=int(args.perm_n),
            bin_quantiles=int(args.perm_bin_quantiles),
            seed=int(args.perm_seed),
            outdir=perm_outdir,
            out_prefix=str(args.perm_out_prefix),
        )


if __name__ == "__main__":
    main()
 