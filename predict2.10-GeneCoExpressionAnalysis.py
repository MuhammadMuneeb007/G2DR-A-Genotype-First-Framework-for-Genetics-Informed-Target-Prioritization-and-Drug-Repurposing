#!/usr/bin/env python3
"""
Comprehensive Gene Co-Expression Analysis with Biological Annotations
Cross-Fold Consistency Summary (Meta-Analysis, NOT Data Merge)

Usage:
  python3 coexpression_analysis_complete.py <phenotype> [--use-original]

Defaults follow scientifically safer choices for p>>n:
  - Pearson correlations with PAIRWISE-complete observations (no mean imputation)
  - Hard gene filters (missingness + near-zero variance)
  - Optional Top-N variance selection, but CONSISTENT across folds (same gene list)
  - Empirical r-threshold per tissue×method×fold from null (e.g., 99.5th percentile)
  - Stability selection / bootstrap edges within fold
  - Modules (components) + module preservation across folds

Predictor-overlap filtering:
  - If you provide --predictor-map, edges with high predictor overlap are removed
  - If absent, reports label results as “architecture-influenced”

Notes:
  - "Co-expression" here means reproducible correlation structure of genetically
    predicted expression. With predicted expression, shared eQTL architecture can
    inflate correlations; overlap filtering + null thresholds + stability selection
    help control this.
"""

import os
import sys
import time
import argparse
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
from scipy.stats import t as student_t

warnings.filterwarnings("ignore")

# =============================================================================
# MULTIPLE TESTING HELPERS
# =============================================================================

def bh_fdr(pvals):
    """Benjamini–Hochberg FDR correction (returns q-values)."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * (m / (np.arange(1, m + 1)))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty_like(q)
    out[order] = q
    return out

def corr_pvals_from_r_and_n(r, n):
    """
    Two-sided correlation p-values from r with df=n-2.
    r, n can be vectors of same length.
    """
    r = np.asarray(r, dtype=float)
    n = np.asarray(n, dtype=float)
    df = np.maximum(n - 2.0, 1.0)
    denom = np.maximum(1.0 - r**2, 1e-15)
    t_stat = r * np.sqrt(df / denom)
    return 2.0 * student_t.sf(np.abs(t_stat), df=df)


# =============================================================================
# MODEL DIRECTORY MAPPINGS
# =============================================================================

MODEL_DIRS = {
    "Regular": "TrainExpression",
    "JTI": "JTITrainExpression",
    "UTMOST": "UTMOSTTrainExpression",
    "UTMOST2": "utmost2TrainExpression",
    "EpiX": "EpiXTrainExpression",
    "TIGAR": "TigarTrainExpression",
    "FUSION": "FusionExpression",
}

FILE_NAMES = {
    "FUSION": {
        "fixed": "GeneExpression_train_data_fixed.csv",
        "original": "GeneExpression_train_data.csv",
    }
}

# Strict: if _fixed missing (when required), skip that tissue/method/fold
REQUIRE_FIXED_STRICT = True


# =============================================================================
# PREDICTOR MAP (optional, but if provided it's ALWAYS applied)
# =============================================================================

def load_predictor_map(predictor_map_path):
    """
    Optional file describing prediction-model SNPs per gene.

    Supported formats:
      - TSV/CSV with columns: gene, snps   (snps = comma/space-separated list)
      - TSV/CSV with columns: gene, snp    (one row per SNP)

    Returns dict: gene_id -> set(snps)
    """
    if predictor_map_path is None:
        return None
    if not os.path.exists(predictor_map_path):
        print(f"WARNING: predictor map not found: {predictor_map_path}")
        return None

    try:
        df = pd.read_csv(predictor_map_path, sep=None, engine="python")
    except Exception as e:
        print(f"WARNING: could not read predictor map ({predictor_map_path}): {e}")
        return None

    df.columns = [c.lower() for c in df.columns]
    if "gene" not in df.columns:
        print("WARNING: predictor map must contain a 'gene' column.")
        return None

    gene_to_snps = defaultdict(set)

    if "snps" in df.columns:
        for _, row in df.iterrows():
            g = str(row["gene"]).strip()
            s = row["snps"]
            if pd.isna(s):
                continue
            if isinstance(s, str):
                parts = [x.strip() for x in (s.split(",") if "," in s else s.split()) if x.strip()]
            else:
                parts = [str(s).strip()]
            for snp in parts:
                if snp and snp.lower() != "nan":
                    gene_to_snps[g].add(snp)

    elif "snp" in df.columns:
        for _, row in df.iterrows():
            g = str(row["gene"]).strip()
            snp = str(row["snp"]).strip()
            if g and snp and snp.lower() != "nan":
                gene_to_snps[g].add(snp)
    else:
        print("WARNING: predictor map must contain either 'snps' or 'snp' column.")
        return None

    print(f"Loaded predictor SNP map for {len(gene_to_snps)} genes.")
    return dict(gene_to_snps)

def jaccard_overlap(a, b):
    if a is None or b is None:
        return 0.0
    if len(a) == 0 and len(b) == 0:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return (inter / union) if union > 0 else 0.0


# =============================================================================
# API FUNCTIONS (unchanged)
# =============================================================================

def query_mygene_batch(ensembl_ids, batch_size=200):
    """Query MyGene.info with Ensembl IDs; returns annotations and Ensembl→Symbol mapping."""
    print(f"  Querying MyGene.info for {len(ensembl_ids)} genes...")

    gene_annotations = {}
    ensembl_to_symbol = {}

    for i in range(0, len(ensembl_ids), batch_size):
        batch = ensembl_ids[i:i + batch_size]

        try:
            url = "https://mygene.info/v3/query"
            params = {
                "q": ",".join(batch),
                "scopes": "ensembl.gene",
                "fields": "symbol,name,summary,type_of_gene,pathway.kegg,pathway.reactome,go.BP,go.MF,entrezgene,MIM,generif,interpro",
                "species": "human",
                "size": len(batch),
            }

            response = requests.post(url, data=params, timeout=60)

            if response.status_code == 200:
                results = response.json()

                for result in results:
                    if "query" in result and not result.get("notfound", False):
                        ensembl_id = result["query"]
                        gene_symbol = result.get("symbol", ensembl_id)

                        ensembl_to_symbol[ensembl_id] = gene_symbol

                        gene_annotations[gene_symbol] = {
                            "ensembl_id": ensembl_id,
                            "name": result.get("name", ""),
                            "type": result.get("type_of_gene", ""),
                            "summary": result.get("summary", "")[:300] if result.get("summary") else "",
                            "entrez_id": str(result.get("entrezgene", "")),
                            "omim": result.get("MIM", ""),
                            "pathways": [],
                            "go_terms": [],
                            "diseases": [],
                            "is_tf": False,
                        }

                        # KEGG/Reactome pathways
                        if "pathway" in result:
                            pathway_data = result["pathway"]

                            if "kegg" in pathway_data:
                                kegg = pathway_data["kegg"]
                                if isinstance(kegg, list):
                                    for item in kegg[:3]:
                                        if isinstance(item, dict) and "name" in item:
                                            gene_annotations[gene_symbol]["pathways"].append(f"KEGG: {item['name']}")

                            if "reactome" in pathway_data:
                                reactome = pathway_data["reactome"]
                                if isinstance(reactome, list):
                                    for item in reactome[:2]:
                                        if isinstance(item, dict) and "name" in item:
                                            gene_annotations[gene_symbol]["pathways"].append(f"Reactome: {item['name']}")

                        # GO terms + TF inference
                        if "go" in result:
                            go_data = result["go"]
                            if isinstance(go_data, dict):
                                if "BP" in go_data:
                                    bp_terms = go_data["BP"]
                                    if isinstance(bp_terms, list):
                                        for term in bp_terms[:2]:
                                            if isinstance(term, dict) and "term" in term:
                                                gene_annotations[gene_symbol]["go_terms"].append(f"BP: {term['term']}")

                                if "MF" in go_data:
                                    mf_terms = go_data["MF"]
                                    if isinstance(mf_terms, list):
                                        for term in mf_terms[:2]:
                                            if isinstance(term, dict) and "term" in term:
                                                term_lower = term["term"].lower()
                                                gene_annotations[gene_symbol]["go_terms"].append(f"MF: {term['term']}")
                                                if any(kw in term_lower for kw in ["dna binding", "transcription factor", "dna-binding"]):
                                                    gene_annotations[gene_symbol]["is_tf"] = True

                        # InterPro TF domains
                        if "interpro" in result:
                            interpro = result["interpro"]
                            if isinstance(interpro, list):
                                for domain in interpro:
                                    if isinstance(domain, dict) and "desc" in domain:
                                        desc = domain["desc"].lower()
                                        if any(kw in desc for kw in ["transcription factor", "dna-binding", "homeobox", "zinc finger"]):
                                            gene_annotations[gene_symbol]["is_tf"] = True

                        # GeneRIF snippets (kept, not used for disease inference)
                        if "generif" in result:
                            generif = result["generif"]
                            if isinstance(generif, list):
                                for rif in generif[:3]:
                                    if isinstance(rif, dict) and "text" in rif:
                                        gene_annotations[gene_symbol]["diseases"].append({
                                            "evidence": rif["text"][:150],
                                            "source": "GeneRIF",
                                        })

            time.sleep(0.25)

        except Exception as e:
            print(f"    Warning: Batch {i // batch_size + 1} error: {str(e)[:80]}")
            continue

    print(f"    ✓ Retrieved annotations for {len(gene_annotations)} genes")
    return gene_annotations, ensembl_to_symbol


def query_string_interactions(gene_symbols, score_threshold=400):
    """Query STRING database for protein interactions."""
    print("  Querying STRING for protein interactions...")
    interactions = defaultdict(list)

    try:
        string_api_url = "https://string-db.org/api/json/network"
        batch_size = 30

        for i in range(0, min(len(gene_symbols), 100), batch_size):
            batch = gene_symbols[i:i + batch_size]

            params = {
                "identifiers": "%0d".join(batch),
                "species": 9606,
                "required_score": score_threshold,
                "limit": 500,
            }

            response = requests.post(string_api_url, data=params, timeout=30)

            if response.status_code == 200:
                results = response.json()
                for item in results:
                    gene1 = item.get("preferredName_A", "")
                    gene2 = item.get("preferredName_B", "")
                    score = item.get("score", 0)
                    if gene1 and gene2:
                        interactions[gene1].append({"partner": gene2, "score": score})
                        interactions[gene2].append({"partner": gene1, "score": score})

            time.sleep(0.5)

    except Exception as e:
        print(f"    Warning: {str(e)[:80]}")

    print(f"    ✓ Found interactions for {len(interactions)} genes")
    return dict(interactions)


def query_enrichr_pathways(gene_symbols, fdr_threshold=0.05):
    """
    Query Enrichr for pathway enrichment.
    Uses Enrichr adjusted p-values when present; else BH-FDR within each library.
    """
    print("  Querying Enrichr for pathway enrichment...")

    enrichment_results = {
        "pathways": [],
        "go_biological_process": [],
        "go_molecular_function": [],
    }

    if not gene_symbols:
        return enrichment_results

    try:
        genes_str = "\n".join(gene_symbols)
        add_list_url = "https://maayanlab.cloud/Enrichr/addList"

        payload = {
            "list": (None, genes_str),
            "description": (None, "gene_list"),
        }

        response = requests.post(add_list_url, files=payload, timeout=30)
        if response.status_code != 200:
            print("    Warning: Could not add gene list to Enrichr")
            return enrichment_results

        user_list_id = response.json().get("userListId")
        if not user_list_id:
            return enrichment_results

        databases = {
            "KEGG_2021_Human": "pathways",
            "WikiPathway_2021_Human": "pathways",
            "Reactome_2022": "pathways",
            "GO_Biological_Process_2023": "go_biological_process",
            "GO_Molecular_Function_2023": "go_molecular_function",
        }

        for db_name, result_category in databases.items():
            try:
                enrich_url = "https://maayanlab.cloud/Enrichr/enrich"
                params = {"userListId": user_list_id, "backgroundType": db_name}

                response = requests.get(enrich_url, params=params, timeout=30)
                if response.status_code != 200:
                    time.sleep(0.5)
                    continue

                data = response.json()
                if db_name not in data:
                    time.sleep(0.5)
                    continue

                items = data[db_name]
                if not items:
                    time.sleep(0.5)
                    continue

                rows = []
                for item in items[:200]:
                    if len(item) < 3:
                        continue
                    term_name = item[1]
                    p_value = float(item[2])
                    adj_p = None
                    if len(item) >= 7:
                        try:
                            adj_p = float(item[6])
                        except Exception:
                            adj_p = None
                    rows.append((term_name, p_value, adj_p))

                if not rows:
                    time.sleep(0.5)
                    continue

                if any(r[2] is None for r in rows):
                    pvals = np.array([r[1] for r in rows], dtype=float)
                    qvals = bh_fdr(pvals)
                    rows2 = [(rows[i][0], rows[i][1], float(qvals[i])) for i in range(len(rows))]
                else:
                    rows2 = [(t, p, float(a)) for (t, p, a) in rows]

                rows2 = [r for r in rows2 if r[2] <= fdr_threshold]
                rows2.sort(key=lambda x: x[2])

                for term_name, p_value, q_value in rows2[:20]:
                    enrichment_results[result_category].append({
                        "term": term_name,
                        "p_value": float(p_value),
                        "q_value": float(q_value),
                        "database": db_name,
                    })

                time.sleep(0.5)

            except Exception:
                continue

    except Exception as e:
        print(f"    Warning: {str(e)[:80]}")

    print(f"    ✓ Found {len(enrichment_results['pathways'])} pathways, {len(enrichment_results['go_biological_process'])} GO terms")
    return enrichment_results


OPEN_TARGETS_GRAPHQL = "https://api.platform.opentargets.org/api/v4/graphql"

def query_opentargets_diseases_for_genes(ensembl_ids, max_rows=30, score_threshold=0.15, sleep_s=0.2):
    """Query Open Targets gene–disease associations; returns dict ensembl_id -> disease names."""
    print("  Querying Open Targets for gene–disease associations...")

    query = """
    query targetDiseases($ensemblId: String!, $index: Int!, $size: Int!) {
      target(ensemblId: $ensemblId) {
        id
        approvedSymbol
        associatedDiseases(page: { index: $index, size: $size }) {
          count
          rows {
            score
            disease { id name }
          }
        }
      }
    }
    """

    out = {}
    for ensembl_id in ensembl_ids:
        try:
            variables = {"ensemblId": ensembl_id, "index": 0, "size": int(max_rows)}
            payload = {"query": query, "variables": variables}
            r = requests.post(OPEN_TARGETS_GRAPHQL, json=payload, timeout=45)

            if r.status_code != 200:
                out[ensembl_id] = []
                time.sleep(sleep_s)
                continue

            data = r.json()
            t = (data.get("data") or {}).get("target")
            if not t:
                out[ensembl_id] = []
                time.sleep(sleep_s)
                continue

            rows = ((t.get("associatedDiseases") or {}).get("rows") or [])
            diseases = []
            for row in rows:
                try:
                    score = float(row.get("score", 0.0))
                except Exception:
                    score = 0.0
                if score < score_threshold:
                    continue
                name = (row.get("disease") or {}).get("name")
                if name:
                    diseases.append(name)

            seen = set()
            uniq = []
            for d in diseases:
                if d not in seen:
                    seen.add(d)
                    uniq.append(d)
            out[ensembl_id] = uniq

            time.sleep(sleep_s)

        except Exception:
            out[ensembl_id] = []
            time.sleep(sleep_s)

    return out


# =============================================================================
# FILE / DIRECTORY HELPERS
# =============================================================================

def clean_gene_name(gene_name):
    """Remove version numbers from Ensembl IDs."""
    gene_str = str(gene_name).strip()
    if gene_str.startswith("chr") and "_" in gene_str:
        gene_str = gene_str.split("_", 1)[1]
    gene_str = gene_str.split(".")[0]
    return gene_str

def get_all_folds(phenotype):
    folds = []
    if not os.path.exists(phenotype):
        return folds
    for item in os.listdir(phenotype):
        if os.path.isdir(os.path.join(phenotype, item)) and item.startswith("Fold_"):
            try:
                folds.append(int(item.split("_")[1]))
            except Exception:
                continue
    return sorted(folds)

def get_all_tissues(phenotype, fold):
    tissues = set()
    fold_dir = f"{phenotype}/Fold_{fold}"
    if not os.path.exists(fold_dir):
        return []
    for _, dir_name in MODEL_DIRS.items():
        expr_path = os.path.join(fold_dir, dir_name)
        if os.path.exists(expr_path):
            for item in os.listdir(expr_path):
                if os.path.isdir(os.path.join(expr_path, item)):
                    tissues.add(item)
    return sorted(list(tissues))

def find_expression_file(tissue_dir, model, use_fixed=True):
    """
    Strict behavior:
      - use_fixed=True => must use *_fixed.csv (or FUSION fixed name) else skip
      - use_fixed=False => must use non-fixed file
    """
    if not os.path.exists(tissue_dir):
        return None

    if model == "FUSION":
        if use_fixed:
            fixed_path = os.path.join(tissue_dir, FILE_NAMES["FUSION"]["fixed"])
            return fixed_path if os.path.exists(fixed_path) else None
        else:
            original_path = os.path.join(tissue_dir, FILE_NAMES["FUSION"]["original"])
            return original_path if os.path.exists(original_path) else None

    csv_files = [f for f in os.listdir(tissue_dir) if f.endswith(".csv")]
    if not csv_files:
        return None

    if use_fixed:
        fixed_files = [f for f in csv_files if "_fixed.csv" in f]
        return os.path.join(tissue_dir, fixed_files[0]) if fixed_files else None
    else:
        non_fixed_files = [f for f in csv_files if "_fixed.csv" not in f]
        return os.path.join(tissue_dir, non_fixed_files[0]) if non_fixed_files else None

def load_expression_data(file_path):
    """Load expression data, clean gene names, keep numeric, keep NaNs (no imputation)."""
    try:
        df = pd.read_csv(file_path)
        gene_cols = [c for c in df.columns if c not in ["FID", "IID"]]
        if not gene_cols:
            return None

        cleaned = [clean_gene_name(c) for c in gene_cols]
        gene_df = df[gene_cols].copy()
        gene_df.columns = cleaned

        sample_ids = df["FID"].astype(str) + "_" + df["IID"].astype(str)
        gene_df.index = sample_ids

        gene_df = gene_df.apply(pd.to_numeric, errors="coerce")
        return gene_df
    except Exception:
        return None


# =============================================================================
# SCIENTIFIC FIXES: consistent cross-fold gene list + null thresholds + bootstrap
# =============================================================================

def gene_quality_stats(gene_df):
    """Return per-gene missing rate and variance (ignoring NaNs)."""
    miss = gene_df.isna().mean(axis=0)
    var = gene_df.var(axis=0, skipna=True)
    return miss, var

def build_consistent_gene_list_across_folds(
    fold_gene_dfs,
    max_missing=0.25,
    min_var=1e-8,
    top_n=2000,
    require_present_all_folds=True,
):
    """
    Choose ONE gene list per tissue×method, consistent across all folds.

    Steps:
      1) For each fold: compute missing rate + variance.
      2) Keep genes passing missing<=max_missing and var>=min_var.
      3) Combine across folds:
         - if require_present_all_folds: intersection of passing genes
         - else: genes passing in >= (some fraction) folds (not used by default)
      4) Optionally choose Top-N by median variance across folds (consistent ranking).
    """
    if not fold_gene_dfs:
        return []

    per_fold_miss = []
    per_fold_var = []
    per_fold_pass = []

    for df in fold_gene_dfs:
        miss, var = gene_quality_stats(df)
        per_fold_miss.append(miss)
        per_fold_var.append(var)
        pass_genes = set(var.index[(miss <= max_missing) & (var >= min_var)])
        per_fold_pass.append(pass_genes)

    if require_present_all_folds:
        genes_pass = set.intersection(*per_fold_pass) if per_fold_pass else set()
    else:
        # fallback: pass in at least half folds
        counts = defaultdict(int)
        for s in per_fold_pass:
            for g in s:
                counts[g] += 1
        need = max(1, len(per_fold_pass) // 2)
        genes_pass = {g for g, c in counts.items() if c >= need}

    if not genes_pass:
        return []

    # Rank by median variance across folds (only among genes_pass)
    vars_stack = []
    for v in per_fold_var:
        vars_stack.append(v.reindex(list(genes_pass)))
    V = pd.concat(vars_stack, axis=1)
    med_var = V.median(axis=1).sort_values(ascending=False)

    if top_n is not None and int(top_n) > 0:
        selected = list(med_var.head(int(top_n)).index)
    else:
        selected = list(med_var.index)

    return selected


def permute_preserve_missing(gene_df, rng):
    """
    Permute each gene's observed values among observed positions,
    keeping NaN positions fixed.
    """
    out = gene_df.copy()
    for col in out.columns:
        x = out[col].values
        mask = ~np.isnan(x)
        vals = x[mask].copy()
        rng.shuffle(vals)
        x2 = x.copy()
        x2[mask] = vals
        out[col] = x2
    return out


def empirical_abs_r_threshold(
    gene_df,
    quantile=0.995,
    n_perms=2,
    max_pairs=200000,
    min_pair_n=20,
    rng_seed=123,
):
    """
    Estimate per-fold empirical |r| threshold from a null where genes are independent:
      - Permute each gene's values among observed entries (missing pattern preserved)
      - Compute corr matrix (pairwise complete)
      - Sample |r| values from random gene-pairs to estimate quantile

    Returns threshold and diagnostics.
    """
    rng = np.random.default_rng(rng_seed)

    p = gene_df.shape[1]
    if p < 5:
        return None, {"reason": "too_few_genes"}

    # precompute pair counts on original missingness pattern
    mask = (~gene_df.isna()).astype(np.int8).values  # n x p
    N = mask.T @ mask  # p x p overlap counts

    # sample random pairs indices
    def sample_pairs(k):
        i = rng.integers(0, p, size=k, endpoint=False)
        j = rng.integers(0, p - 1, size=k, endpoint=False)
        j = np.where(j >= i, j + 1, j)
        ii = np.minimum(i, j)
        jj = np.maximum(i, j)
        return ii, jj

    vals = []
    pairs_per_perm = max_pairs // max(1, n_perms)

    for t in range(int(n_perms)):
        dfp = permute_preserve_missing(gene_df, rng)
        C = dfp.corr(method="pearson", min_periods=min_pair_n).values  # p x p
        ii, jj = sample_pairs(int(pairs_per_perm))
        ok = N[ii, jj] >= int(min_pair_n)
        sampled = np.abs(C[ii[ok], jj[ok]])
        sampled = sampled[np.isfinite(sampled)]
        if sampled.size:
            vals.append(sampled)

    if not vals:
        return None, {"reason": "null_failed_or_no_pairs"}

    vals = np.concatenate(vals)
    thr = float(np.quantile(vals, float(quantile)))
    return thr, {
        "null_quantile": float(quantile),
        "null_n": int(vals.size),
        "null_perms": int(n_perms),
        "min_pair_n": int(min_pair_n),
        "threshold": thr,
    }


def bootstrap_stability_edges(
    gene_df,
    abs_r_threshold,
    n_boot=50,
    stability_frac=0.6,
    sign_frac=0.9,
    min_pair_n=20,
    predictor_map=None,
    overlap_threshold=0.25,
    rng_seed=123,
):
    """
    Bootstrap individuals WITHIN a fold.
    Keep edges that appear in >= stability_frac of bootstraps and have sign consistency.

    Pearson is computed pairwise-complete in each bootstrap (pandas .corr).
    """
    rng = np.random.default_rng(rng_seed)

    genes = list(gene_df.columns)
    p = len(genes)
    n = gene_df.shape[0]

    # Precompute predictor sets lookup if provided
    pred = predictor_map if predictor_map is not None else None

    # For speed, store only edges that pass threshold in each bootstrap
    edge_counts = defaultdict(int)
    edge_sign = defaultdict(int)  # counts positive among times selected
    edge_r_values = defaultdict(list)
    edge_n_values = defaultdict(list)

    # For pairwise n per edge, we need overlap counts; compute from bootstrap each time is expensive.
    # We compute overlap counts using mask dot product per bootstrap (fast in numpy).
    for b in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)  # resample rows with replacement
        dfb = gene_df.iloc[idx]

        # corr matrix with pairwise complete
        C = dfb.corr(method="pearson", min_periods=min_pair_n)
        Cv = C.values

        # overlap counts
        mask = (~dfb.isna()).astype(np.int8).values
        N = mask.T @ mask

        # threshold edges
        iu, ju = np.triu_indices(p, k=1)
        r = Cv[iu, ju]
        nn = N[iu, ju]

        ok = np.isfinite(r) & (nn >= int(min_pair_n)) & (np.abs(r) >= float(abs_r_threshold))
        if not np.any(ok):
            continue

        # predictor overlap filter (mandatory if available)
        for idx_e in np.where(ok)[0]:
            i = int(iu[idx_e])
            j = int(ju[idx_e])
            a = genes[i]
            bname = genes[j]

            if pred is not None:
                jac = jaccard_overlap(pred.get(a, set()), pred.get(bname, set()))
                if jac >= float(overlap_threshold):
                    continue

            k = (a, bname) if a < bname else (bname, a)
            edge_counts[k] += 1
            if r[idx_e] > 0:
                edge_sign[k] += 1
            edge_r_values[k].append(float(r[idx_e]))
            edge_n_values[k].append(int(nn[idx_e]))

    # Select stable edges
    stable_edges = []
    min_count = int(np.ceil(float(stability_frac) * int(n_boot)))

    for (a, bname), c in edge_counts.items():
        if c < min_count:
            continue
        pos = edge_sign[(a, bname)]
        sign_cons = max(pos, c - pos) / max(c, 1)
        if sign_cons < float(sign_frac):
            continue

        rs = np.array(edge_r_values[(a, bname)], dtype=float)
        ns = np.array(edge_n_values[(a, bname)], dtype=int)

        stable_edges.append({
            "gene_a": a,
            "gene_b": bname,
            "boot_freq": int(c),
            "boot_freq_frac": float(c / max(1, int(n_boot))),
            "sign_consistency": float(sign_cons),
            "median_r": float(np.median(rs)) if rs.size else 0.0,
            "median_abs_r": float(np.median(np.abs(rs))) if rs.size else 0.0,
            "median_n": int(np.median(ns)) if ns.size else 0,
        })

    stable_edges.sort(key=lambda d: (-d["boot_freq"], -d["median_abs_r"]))
    return stable_edges


def edges_to_hubs(stable_edges):
    """Hub score as node strength sum(|median_r|) over stable edges."""
    strength = defaultdict(float)
    for e in stable_edges:
        w = abs(float(e.get("median_r", 0.0)))
        strength[e["gene_a"]] += w
        strength[e["gene_b"]] += w
    hubs = sorted(strength.items(), key=lambda x: x[1], reverse=True)
    return dict(hubs[:50])


def modules_from_edges(stable_edges, min_module_size=10):
    """
    Modules as connected components of the stable-edge graph.
    (Simple, robust; avoids fragile edge-level claims.)
    """
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    nodes = set()
    for e in stable_edges:
        a, b = e["gene_a"], e["gene_b"]
        nodes.add(a); nodes.add(b)
        union(a, b)

    comps = defaultdict(set)
    for n in nodes:
        comps[find(n)].add(n)

    modules = [sorted(list(s)) for s in comps.values() if len(s) >= int(min_module_size)]
    modules.sort(key=len, reverse=True)
    return modules


def module_preservation_jaccard(modules_by_fold):
    """
    Compute module preservation across folds using Jaccard overlap.

    For each fold, for each module, find best Jaccard in each other fold.
    Report mean best-Jaccard and proportion of modules preserved above thresholds.
    """
    folds = sorted(modules_by_fold.keys())
    if len(folds) < 2:
        return {"n_folds": len(folds), "note": "need >=2 folds"}

    # Convert to sets for speed
    mods = {f: [set(m) for m in modules_by_fold[f]] for f in folds}

    best_scores = []
    for f in folds:
        for m in mods[f]:
            if not m:
                continue
            # compare to all other folds, take best per fold, then average
            per_other = []
            for g in folds:
                if g == f:
                    continue
                best = 0.0
                for m2 in mods[g]:
                    inter = len(m & m2)
                    if inter == 0:
                        continue
                    union = len(m | m2)
                    j = inter / union if union > 0 else 0.0
                    if j > best:
                        best = j
                per_other.append(best)
            if per_other:
                best_scores.append(float(np.mean(per_other)))

    if not best_scores:
        return {"n_folds": len(folds), "note": "no modules to compare"}

    best_scores = np.array(best_scores, dtype=float)

    return {
        "n_folds": len(folds),
        "modules_compared": int(best_scores.size),
        "mean_best_jaccard": float(np.mean(best_scores)),
        "median_best_jaccard": float(np.median(best_scores)),
        "prop_best_jaccard_ge_0.3": float(np.mean(best_scores >= 0.3)),
        "prop_best_jaccard_ge_0.5": float(np.mean(best_scores >= 0.5)),
    }


# =============================================================================
# CORE ANALYSIS PER FOLD
# =============================================================================

def analyze_single_fold(
    gene_df,
    gene_list,
    min_pair_n=20,
    null_quantile=0.995,
    null_perms=2,
    null_pairs=200000,
    n_boot=50,
    stability_frac=0.6,
    sign_frac=0.9,
    predictor_map=None,
    overlap_threshold=0.25,
    min_module_size=10,
    rng_seed=123,
):
    """
    Analyze one fold with:
      - fixed gene list (same across folds)
      - null-derived abs(r) threshold
      - bootstrap stability edges
      - modules
    """

    df = gene_df.reindex(columns=gene_list)
    # Remove genes that are entirely missing in this fold (should be rare if intersection was used)
    all_missing = df.isna().all(axis=0)
    if all_missing.any():
        df = df.loc[:, ~all_missing]
    genes_used = list(df.columns)

    n = df.shape[0]
    p = df.shape[1]

    if n < max(5, int(min_pair_n)) or p < 10:
        return {
            "n_samples": int(n),
            "n_genes": int(p),
            "genes": genes_used,
            "null_threshold": None,
            "null_info": {"reason": "too_small"},
            "stable_edges": [],
            "n_edges": 0,
            "density": 0.0,
            "hub_genes": {},
            "modules": [],
            "module_stats": {},
            "architecture_label": "architecture-influenced" if predictor_map is None else "overlap-filtered",
        }

    # Null-derived threshold per fold
    thr, null_info = empirical_abs_r_threshold(
        df,
        quantile=null_quantile,
        n_perms=null_perms,
        max_pairs=null_pairs,
        min_pair_n=min_pair_n,
        rng_seed=rng_seed,
    )
    if thr is None:
        thr = 0.0  # fallback; but should be rare
        null_info = {"reason": "null_failed", **(null_info or {})}

    # Bootstrap stability edges
    stable_edges = bootstrap_stability_edges(
        df,
        abs_r_threshold=thr,
        n_boot=n_boot,
        stability_frac=stability_frac,
        sign_frac=sign_frac,
        min_pair_n=min_pair_n,
        predictor_map=predictor_map,             # mandatory if provided
        overlap_threshold=overlap_threshold,
        rng_seed=rng_seed + 7,
    )

    n_edges = len(stable_edges)
    max_edges = p * (p - 1) / 2
    density = float(n_edges / max_edges) if max_edges > 0 else 0.0

    # Hubs
    hubs = edges_to_hubs(stable_edges)

    # Modules
    modules = modules_from_edges(stable_edges, min_module_size=min_module_size)
    module_stats = {
        "n_modules": int(len(modules)),
        "largest_module": int(len(modules[0])) if modules else 0,
        "n_genes_in_modules": int(len(set().union(*[set(m) for m in modules])) ) if modules else 0,
    }

    # mean absolute correlation (descriptive) using pairwise complete
    C = df.corr(method="pearson", min_periods=min_pair_n).values
    np.fill_diagonal(C, np.nan)
    mean_abs = float(np.nanmean(np.abs(C)))

    return {
        "n_samples": int(n),
        "n_genes": int(p),
        "genes": genes_used,
        "null_threshold": float(thr),
        "null_info": null_info,
        "stable_edges": stable_edges,
        "n_edges": int(n_edges),
        "density": density,
        "mean_abs_corr": mean_abs,
        "hub_genes": hubs,
        "modules": modules,
        "module_stats": module_stats,
        "architecture_label": "architecture-influenced" if predictor_map is None else "overlap-filtered",
    }


def analyze_fold(
    phenotype,
    fold,
    model,
    tissue,
    use_fixed=True,
    gene_list=None,
    **kwargs
):
    fold_dir = f"{phenotype}/Fold_{fold}"
    dir_name = MODEL_DIRS[model]
    tissue_dir = os.path.join(fold_dir, dir_name, tissue)
    if not os.path.exists(tissue_dir):
        return None

    file_path = find_expression_file(tissue_dir, model, use_fixed=use_fixed)
    if file_path is None or not os.path.exists(file_path):
        return None

    gene_df = load_expression_data(file_path)
    if gene_df is None or gene_df.shape[1] < 10:
        return None

    if gene_list is None or len(gene_list) < 10:
        return None

    # ✅ FIX: avoid passing rng_seed twice
    base_seed = int(kwargs.pop("rng_seed", 123))
    fold_seed = base_seed + int(fold)

    fold_stats = analyze_single_fold(
        gene_df,
        gene_list,
        rng_seed=fold_seed,
        **kwargs
    )

    fold_stats["fold"] = fold
    fold_stats["file_used"] = os.path.basename(file_path)
    return fold_stats



# =============================================================================
# METHOD × TISSUE ANALYSIS ACROSS FOLDS
# =============================================================================

def analyze_method_tissue(
    phenotype,
    folds,
    model,
    tissue,
    use_fixed=True,
    max_missing=0.25,
    min_var=1e-8,
    top_n=2000,
    require_present_all_folds=True,
    predictor_map=None,
    overlap_threshold=0.25,
    min_pair_n=20,
    null_quantile=0.995,
    null_perms=2,
    null_pairs=200000,
    n_boot=50,
    stability_frac=0.6,
    sign_frac=0.9,
    min_module_size=10,
    rng_seed=123,
):
    print(f"    Analyzing {model} - {tissue}")

    # Load fold dataframes (for gene selection only)
    fold_dfs = []
    fold_files = []
    for fold in folds:
        fold_dir = f"{phenotype}/Fold_{fold}"
        tissue_dir = os.path.join(fold_dir, MODEL_DIRS[model], tissue)
        if not os.path.exists(tissue_dir):
            continue
        fp = find_expression_file(tissue_dir, model, use_fixed=use_fixed)
        if fp is None:
            continue
        df = load_expression_data(fp)
        if df is None or df.shape[1] < 10:
            continue
        fold_dfs.append(df)
        fold_files.append(os.path.basename(fp))

    if len(fold_dfs) == 0:
        return None

    # Build one consistent gene list across folds
    gene_list = build_consistent_gene_list_across_folds(
        fold_dfs,
        max_missing=max_missing,
        min_var=min_var,
        top_n=top_n,
        require_present_all_folds=require_present_all_folds,
    )

    if len(gene_list) < 10:
        print("      ✗ No genes passed consistent cross-fold filters.")
        return None

    print(f"      Consistent gene list size: {len(gene_list)} (max_missing={max_missing}, min_var={min_var}, top_n={top_n})")

    # Analyze each fold using SAME gene list
    fold_results = []
    files_used = set()

    for fold in folds:
        res = analyze_fold(
            phenotype,
            fold,
            model,
            tissue,
            use_fixed=use_fixed,
            gene_list=gene_list,
            predictor_map=predictor_map,
            overlap_threshold=overlap_threshold,
            min_pair_n=min_pair_n,
            null_quantile=null_quantile,
            null_perms=null_perms,
            null_pairs=null_pairs,
            n_boot=n_boot,
            stability_frac=stability_frac,
            sign_frac=sign_frac,
            min_module_size=min_module_size,
            rng_seed=rng_seed,
        )
        if res is not None:
            fold_results.append(res)
            files_used.add(res.get("file_used", "unknown"))

    if not fold_results:
        return None

    # Cross-fold modules preservation
    modules_by_fold = {fr["fold"]: fr.get("modules", []) for fr in fold_results}
    module_pres = module_preservation_jaccard(modules_by_fold)

    # Cross-fold consistency: hubs and stable edges
    hub_frequency = defaultdict(int)
    hub_ranks = defaultdict(list)
    hub_folds = defaultdict(set)

    edge_frequency = defaultdict(int)
    edge_folds = defaultdict(set)
    edge_abs = defaultdict(list)
    edge_sign_cons = defaultdict(list)

    stats_density = []
    stats_meanabs = []
    stats_thr = []

    for fr in fold_results:
        fid = fr["fold"]
        stats_density.append(fr.get("density", float("nan")))
        stats_meanabs.append(fr.get("mean_abs_corr", float("nan")))
        stats_thr.append(fr.get("null_threshold", float("nan")))

        # hubs: use hub score already computed
        hubs_sorted = sorted(fr.get("hub_genes", {}).items(), key=lambda x: x[1], reverse=True)
        for rank, (g, s) in enumerate(hubs_sorted, 1):
            hub_frequency[g] += 1
            hub_ranks[g].append(rank)
            hub_folds[g].add(fid)

        # edges: stable edges
        for e in fr.get("stable_edges", []):
            a, b = e["gene_a"], e["gene_b"]
            k = (a, b) if a < b else (b, a)
            edge_frequency[k] += 1
            edge_folds[k].add(fid)
            edge_abs[k].append(float(e.get("median_abs_r", 0.0)))
            edge_sign_cons[k].append(float(e.get("sign_consistency", 0.0)))

    # Assemble hub consistency
    consistent_hubs = []
    for g, c in hub_frequency.items():
        consistent_hubs.append({
            "gene": g,
            "fold_frequency": int(c),
            "folds_present": sorted(list(hub_folds[g])),
            "avg_rank": float(np.mean(hub_ranks[g])) if hub_ranks[g] else 999.0,
        })
    consistent_hubs.sort(key=lambda d: (-d["fold_frequency"], d["avg_rank"]))

    # Assemble edge consistency
    consistent_edges = []
    for (a, b), c in edge_frequency.items():
        consistent_edges.append({
            "gene_a": a,
            "gene_b": b,
            "fold_frequency": int(c),
            "folds_present": sorted(list(edge_folds[(a, b)])),
            "median_abs_r": float(np.median(edge_abs[(a, b)])) if edge_abs[(a, b)] else 0.0,
            "median_sign_consistency": float(np.median(edge_sign_cons[(a, b)])) if edge_sign_cons[(a, b)] else 0.0,
        })
    consistent_edges.sort(key=lambda d: (-d["fold_frequency"], -d["median_abs_r"]))

    # Biological annotations:
    # We still query MyGene with the consistent gene list for this tissue×method.
    print("      Querying biological databases...")
    gene_annotations, ensembl_to_symbol = query_mygene_batch(gene_list)

    # Convert hub gene IDs to symbols if available
    def to_symbol(g):
        return ensembl_to_symbol.get(g, g)

    top_hubs_symbols = [to_symbol(x["gene"]) for x in consistent_hubs[:50]]

    string_interactions = query_string_interactions(top_hubs_symbols[:30])
    enrichment_results = query_enrichr_pathways(top_hubs_symbols)

    tf_genes = {}
    for sym in top_hubs_symbols:
        a = gene_annotations.get(sym)
        if a and a.get("is_tf", False):
            tf_genes[sym] = {"evidence": a.get("go_terms", []), "name": a.get("name", "")}

    # Open Targets diseases (hub genes only)
    hub_ensembl_ids = []
    sym_to_eid = {}
    for sym in top_hubs_symbols:
        a = gene_annotations.get(sym)
        if a and a.get("ensembl_id"):
            eid = a["ensembl_id"]
            hub_ensembl_ids.append(eid)
            sym_to_eid[sym] = eid

    ot_map = query_opentargets_diseases_for_genes(hub_ensembl_ids, max_rows=30, score_threshold=0.15, sleep_s=0.2)
    disease_associations = {}
    for sym, eid in sym_to_eid.items():
        ds = ot_map.get(eid, [])
        if ds:
            disease_associations[sym] = ds

    return {
        "files_used": sorted(list(files_used)),
        "gene_list_size": int(len(gene_list)),
        "gene_list": gene_list,  # keep for reproducibility
        "fold_results": fold_results,
        "consistency": {
            "n_folds_analyzed": int(len(fold_results)),
            "n_total_folds": int(len(folds)),
            "network_stats": {
                "density_mean": float(np.nanmean(stats_density)),
                "density_std": float(np.nanstd(stats_density)),
                "mean_abs_corr_mean": float(np.nanmean(stats_meanabs)),
                "mean_abs_corr_std": float(np.nanstd(stats_meanabs)),
                "null_thr_mean": float(np.nanmean(stats_thr)),
                "null_thr_std": float(np.nanstd(stats_thr)),
            },
            "consistent_hubs": consistent_hubs,
            "consistent_edges": consistent_edges,
            "module_preservation": module_pres,
        },
        "ensembl_to_symbol": ensembl_to_symbol,
        "gene_annotations": gene_annotations,
        "transcription_factors": tf_genes,
        "disease_associations": disease_associations,
        "protein_interactions": string_interactions,
        "pathway_enrichment": enrichment_results,
        "architecture_label": "architecture-influenced" if predictor_map is None else "overlap-filtered",
    }


def analyze_all_methods_tissues(phenotype, folds, tissues, **kwargs):
    all_results = {}
    for method_idx, model in enumerate(MODEL_DIRS.keys(), 1):
        print(f"\n{'=' * 60}")
        print(f"[{method_idx}/{len(MODEL_DIRS)}] Processing {model}...")
        print(f"{'=' * 60}")

        tissue_results = {}
        for tissue_idx, tissue in enumerate(tissues, 1):
            print(f"\n  [{tissue_idx}/{len(tissues)}] Tissue: {tissue}")
            result = analyze_method_tissue(phenotype, folds, model, tissue, **kwargs)
            if result is not None:
                tissue_results[tissue] = result
                print("      ✓ Complete")
            else:
                print("      ✗ No data")

        all_results[model] = {"n_tissues": len(tissue_results), "tissue_results": tissue_results} if tissue_results else None

    return all_results


# =============================================================================
# AGGREGATION ACROSS METHODS × TISSUES (modules-forward reporting)
# =============================================================================

def aggregate_cross_method_tissue_consistency(all_results):
    gene_consistency = defaultdict(lambda: {
        "methods": set(),
        "tissues": set(),
        "folds_seen": set(),
        "appearances": 0,
        "avg_ranks": [],
    })

    module_pres_scores = []
    pathway_recurrence = defaultdict(lambda: {"count": 0, "tissues": set(), "methods": set()})
    go_recurrence = defaultdict(lambda: {"count": 0, "tissues": set()})
    tf_frequency = defaultdict(int)
    disease_count = defaultdict(int)
    tissue_stability = defaultdict(lambda: {"density": [], "mean_abs": [], "thr": [], "methods": []})

    for model, model_results in all_results.items():
        if model_results is None:
            continue
        for tissue, tr in model_results["tissue_results"].items():
            c = tr["consistency"]
            ns = c["network_stats"]

            tissue_stability[tissue]["density"].append(ns["density_mean"])
            tissue_stability[tissue]["mean_abs"].append(ns["mean_abs_corr_mean"])
            tissue_stability[tissue]["thr"].append(ns["null_thr_mean"])
            tissue_stability[tissue]["methods"].append(model)

            mp = c.get("module_preservation", {})
            if "mean_best_jaccard" in mp:
                module_pres_scores.append(mp["mean_best_jaccard"])

            # genes (top hubs)
            for g in c["consistent_hubs"][:30]:
                gene = g["gene"]
                gene_consistency[gene]["methods"].add(model)
                gene_consistency[gene]["tissues"].add(tissue)
                gene_consistency[gene]["folds_seen"].update(g.get("folds_present", []))
                gene_consistency[gene]["appearances"] += int(g.get("fold_frequency", 0))
                gene_consistency[gene]["avg_ranks"].append(float(g.get("avg_rank", 999)))

            for p in tr["pathway_enrichment"]["pathways"]:
                key = (p.get("database", "UNKNOWN_DB"), p.get("term", ""))
                pathway_recurrence[key]["count"] += 1
                pathway_recurrence[key]["tissues"].add(tissue)
                pathway_recurrence[key]["methods"].add(model)

            for go in tr["pathway_enrichment"]["go_biological_process"]:
                term = go.get("term", "")
                go_recurrence[term]["count"] += 1
                go_recurrence[term]["tissues"].add(tissue)

            for tf in tr["transcription_factors"].keys():
                tf_frequency[tf] += 1

            for _, diseases in tr["disease_associations"].items():
                for d in diseases:
                    disease_count[d] += 1

    scored_genes = []
    for gene, d in gene_consistency.items():
        n_methods = len(d["methods"])
        n_tissues = len(d["tissues"])
        avg_rank = float(np.mean(d["avg_ranks"])) if d["avg_ranks"] else 999.0
        consistency_score = (
            n_methods * 2
            + n_tissues * 1.5
            + d["appearances"] * 0.5
            + (50 - min(avg_rank, 50)) * 0.1
        )
        scored_genes.append({
            "gene": gene,
            "consistency_score": float(consistency_score),
            "n_methods": int(n_methods),
            "n_tissues": int(n_tissues),
            "analysis_appearances": int(d["appearances"]),
            "avg_rank": float(avg_rank),
        })
    scored_genes.sort(key=lambda x: x["consistency_score"], reverse=True)

    tissue_summary = {}
    for tissue, d in tissue_stability.items():
        tissue_summary[tissue] = {
            "mean_density": float(np.mean(d["density"])) if d["density"] else float("nan"),
            "std_density": float(np.std(d["density"])) if d["density"] else float("nan"),
            "mean_abs_corr": float(np.mean(d["mean_abs"])) if d["mean_abs"] else float("nan"),
            "std_abs_corr": float(np.std(d["mean_abs"])) if d["mean_abs"] else float("nan"),
            "mean_null_thr": float(np.mean(d["thr"])) if d["thr"] else float("nan"),
            "n_methods": int(len(d["methods"])),
        }

    return {
        "consistent_genes": scored_genes,
        "recurring_pathways": sorted(pathway_recurrence.items(), key=lambda x: x[1]["count"], reverse=True),
        "recurring_go_terms": sorted(go_recurrence.items(), key=lambda x: x[1]["count"], reverse=True),
        "transcription_factors": sorted(tf_frequency.items(), key=lambda x: x[1], reverse=True),
        "disease_associations": sorted(disease_count.items(), key=lambda x: x[1], reverse=True),
        "tissue_stability": tissue_summary,
        "module_preservation_overall_mean": float(np.mean(module_pres_scores)) if module_pres_scores else float("nan"),
        "module_preservation_n": int(len(module_pres_scores)),
    }


# =============================================================================
# REPORT WRITING (same filenames, stronger scientific reporting)
# =============================================================================

def write_narrative_report(output_file, all_results, aggregated, phenotype, folds, use_fixed, args):
    with open(output_file, "w") as f:
        f.write("=" * 100 + "\n")
        f.write("GENE CO-EXPRESSION ANALYSIS: CROSS-FOLD CONSISTENCY SUMMARY\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"Phenotype: {phenotype}\n")
        f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of Cross-Validation Folds: {len(folds)}\n")
        f.write(f"Expression Data: {'Covariate-adjusted (_fixed)' if use_fixed else 'Original (unadjusted)'}\n\n")

        f.write("SCIENTIFIC SETTINGS (key):\n")
        f.write("-" * 100 + "\n")
        f.write(f"- Pearson pairwise-complete correlations (min_pair_n={args.min_pair_n}); no mean-imputation.\n")
        f.write(f"- Gene filters: max_missing={args.max_missing}, min_var={args.min_var}.\n")
        f.write(f"- Top-N selection is CONSISTENT across folds (top_n={args.top_n}).\n")
        f.write(f"- Empirical threshold per fold: |r| >= null {args.null_quantile} quantile (perms={args.null_perms}).\n")
        f.write(f"- Bootstrap stability: n_boot={args.n_boot}, keep edges in >= {args.stability_frac} of bootstraps\n")
        f.write(f"  with sign consistency >= {args.sign_frac}.\n")
        if args.predictor_map:
            f.write(f"- Predictor overlap filter: ON (Jaccard >= {args.overlap_threshold} removed).\n")
        else:
            f.write("- Predictor overlap filter: OFF (no map) => results labeled architecture-influenced.\n")
        f.write(f"- Modules: connected components, min_module_size={args.min_module_size}; module preservation via Jaccard.\n\n")

        f.write("=" * 100 + "\n")
        f.write("EXECUTIVE SUMMARY\n")
        f.write("=" * 100 + "\n\n")

        n_methods = sum(1 for r in all_results.values() if r is not None)
        n_tissues = len(aggregated["tissue_stability"])
        n_consistent_genes = len([g for g in aggregated["consistent_genes"]
                                  if g["n_methods"] >= 2 or g["n_tissues"] >= 2])
        n_tfs = len(aggregated["transcription_factors"])
        n_pathways = len([p for p, d in aggregated["recurring_pathways"] if d["count"] >= 2])

        f.write(f"Methods analyzed: {n_methods}\n")
        f.write(f"Tissues analyzed: {n_tissues}\n")
        f.write(f"Consistent hub genes (≥2 methods/tissues): {n_consistent_genes}\n")
        f.write(f"Hub TFs identified: {n_tfs}\n")
        f.write(f"Recurring pathways (≥2 analyses): {n_pathways}\n")
        f.write(f"Overall module preservation (mean best-Jaccard): {aggregated['module_preservation_overall_mean']:.3f} ")
        f.write(f"(n={aggregated['module_preservation_n']})\n\n")

        if aggregated["consistent_genes"]:
            top = [g["gene"] for g in aggregated["consistent_genes"][:5]]
            f.write(f"Top consistent hub genes: {', '.join(top)}\n\n")

        f.write("=" * 100 + "\n")
        f.write("SECTION 1: CONSISTENT HUB GENES\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"{'Rank':<6}{'Gene':<18}{'Score':<10}{'Methods':<10}{'Tissues':<10}{'Appearances':<14}{'Avg Rank'}\n")
        f.write("-" * 90 + "\n")
        for i, g in enumerate(aggregated["consistent_genes"][:50], 1):
            f.write(f"{i:<6}{g['gene']:<18}{g['consistency_score']:<10.1f}{g['n_methods']:<10}{g['n_tissues']:<10}{g['analysis_appearances']:<14}{g['avg_rank']:.1f}\n")

        f.write("\n" + "=" * 100 + "\n")
        f.write("SECTION 2: RECURRING PATHWAYS\n")
        f.write("=" * 100 + "\n\n")
        for rank, ((db, term), d) in enumerate(aggregated["recurring_pathways"][:25], 1):
            f.write(f"{rank:>2}. {db}: {term}  (count={d['count']}, tissues={len(d['tissues'])}, methods={len(d['methods'])})\n")

        f.write("\n" + "=" * 100 + "\n")
        f.write("SECTION 3: TISSUE STABILITY\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"{'Tissue':<35}{'Density(mean±std)':<22}{'Mean|r|(mean±std)':<24}{'NullThr(mean)':<14}{'Methods'}\n")
        f.write("-" * 110 + "\n")
        for tissue, st in sorted(aggregated["tissue_stability"].items(), key=lambda x: x[1]["mean_abs_corr"], reverse=True):
            f.write(f"{tissue[:33]:<35}{st['mean_density']:.4f}±{st['std_density']:.4f}      ")
            f.write(f"{st['mean_abs_corr']:.4f}±{st['std_abs_corr']:.4f}        ")
            f.write(f"{st['mean_null_thr']:.3f}        {st['n_methods']}\n")

    print(f"  ✓ Narrative report saved: {output_file}")


def write_detailed_per_tissue_report(output_file, all_results, phenotype, use_fixed):
    with open(output_file, "w") as f:
        f.write("=" * 100 + "\n")
        f.write("DETAILED PER-TISSUE CO-EXPRESSION RESULTS (Supplementary)\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Phenotype: {phenotype}\n")
        f.write(f"Expression Data: {'Covariate-adjusted (_fixed)' if use_fixed else 'Original (unadjusted)'}\n\n")

        for model, model_results in all_results.items():
            f.write("\n" + "=" * 100 + "\n")
            f.write(f"METHOD: {model}\n")
            f.write("=" * 100 + "\n")

            if model_results is None:
                f.write("No data available.\n")
                continue

            for tissue, tr in sorted(model_results["tissue_results"].items()):
                c = tr["consistency"]
                ns = c["network_stats"]

                f.write(f"\n{'-' * 80}\n")
                f.write(f"Tissue: {tissue}\n")
                f.write(f"{'-' * 80}\n\n")
                f.write(f"Architecture label: {tr.get('architecture_label','')}\n")
                f.write(f"Consistent gene list size: {tr['gene_list_size']}\n")
                f.write(f"Files used: {', '.join(tr['files_used'])}\n")
                f.write(f"Folds analyzed: {c['n_folds_analyzed']}/{c['n_total_folds']}\n")
                f.write(f"Mean density: {ns['density_mean']:.4f} ± {ns['density_std']:.4f}\n")
                f.write(f"Mean |r|: {ns['mean_abs_corr_mean']:.4f} ± {ns['mean_abs_corr_std']:.4f}\n")
                f.write(f"Null |r| threshold (mean): {ns['null_thr_mean']:.3f} ± {ns['null_thr_std']:.3f}\n")
                mp = c.get("module_preservation", {})
                if "mean_best_jaccard" in mp:
                    f.write(f"Module preservation mean best-Jaccard: {mp['mean_best_jaccard']:.3f} (median={mp['median_best_jaccard']:.3f})\n")

                f.write("\nPer-fold summary:\n")
                f.write(f"{'Fold':<6}{'n':<8}{'p':<8}{'Thr':<10}{'Edges':<10}{'Density':<10}{'Mean|r|':<10}{'Modules':<10}{'LargestMod':<12}File\n")
                for fr in tr["fold_results"]:
                    ms = fr.get("module_stats", {})
                    f.write(f"{fr['fold']:<6}{fr['n_samples']:<8}{fr['n_genes']:<8}{fr.get('null_threshold',0):<10.3f}")
                    f.write(f"{fr['n_edges']:<10}{fr['density']:<10.5f}{fr.get('mean_abs_corr',float('nan')):<10.4f}")
                    f.write(f"{ms.get('n_modules',0):<10}{ms.get('largest_module',0):<12}{fr.get('file_used','')}\n")

                f.write("\nTop consistent hubs (within this tissue/method):\n")
                for g in c["consistent_hubs"][:20]:
                    f.write(f"  - {g['gene']}: fold_freq={g['fold_frequency']}, avg_rank={g['avg_rank']:.1f}\n")

                f.write("\nTop stable edges recurring across folds (within this tissue/method):\n")
                for e in c["consistent_edges"][:30]:
                    f.write(f"  - {e['gene_a']}--{e['gene_b']}: fold_freq={e['fold_frequency']}, med|r|={e['median_abs_r']:.3f}, sign_cons={e['median_sign_consistency']:.2f}\n")

                if tr["pathway_enrichment"]["pathways"]:
                    f.write("\nEnriched pathways (hub genes):\n")
                    for pth in tr["pathway_enrichment"]["pathways"][:10]:
                        f.write(f"  - {pth['database']}: {pth['term']} (q={pth.get('q_value', float('nan')):.2e})\n")

                if tr["transcription_factors"]:
                    f.write(f"\nTranscription factors among hubs: {len(tr['transcription_factors'])}\n")
                    f.write("  " + ", ".join(list(tr["transcription_factors"].keys())[:15]) + "\n")

                if tr["disease_associations"]:
                    f.write("\nOpen Targets disease names (hub genes):\n")
                    # show up to 15 distinct diseases
                    ds = []
                    for _, names in tr["disease_associations"].items():
                        ds.extend(names)
                    seen = set(); uniq = []
                    for d in ds:
                        if d not in seen:
                            seen.add(d); uniq.append(d)
                    for d in uniq[:15]:
                        f.write(f"  - {d}\n")

    print(f"  ✓ Detailed report saved: {output_file}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Gene co-expression analysis with cross-fold consistency (meta-analysis).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("phenotype", help="Phenotype directory name (e.g., BMI, migraine)")
    parser.add_argument("--use-original", action="store_true",
                        help="Use original (non-adjusted) expression files instead of _fixed files")

    # Gene filtering (consistent across folds)
    parser.add_argument("--max-missing", type=float, default=0.25,
                        help="Drop genes with missingness > this (per fold). Default 0.25")
    parser.add_argument("--min-var", type=float, default=1e-8,
                        help="Drop near-zero variance genes (per fold). Default 1e-8")
    parser.add_argument("--top-n", type=int, default=2000,
                        help="Keep top N genes by median variance across folds (consistent). 0 disables (keep all passing). Default 2000")
    parser.add_argument("--require-present-all-folds", action="store_true",
                        help="Require gene passes filters in ALL folds (intersection). Recommended. Default False (but we treat as True).")

    # Correlation settings
    parser.add_argument("--min-pair-n", type=int, default=20,
                        help="Minimum overlapping samples for a gene pair to compute correlation. Default 20")

    # Empirical threshold
    parser.add_argument("--null-quantile", type=float, default=0.995,
                        help="Empirical null quantile for abs(r) threshold per fold. Default 0.995")
    parser.add_argument("--null-perms", type=int, default=2,
                        help="Number of null permutations per fold. Default 2")
    parser.add_argument("--null-pairs", type=int, default=200000,
                        help="Random gene-pairs sampled from null correlations. Default 200000")

    # Bootstrap stability selection
    parser.add_argument("--n-boot", type=int, default=50,
                        help="Bootstrap resamples per fold. Default 50")
    parser.add_argument("--stability-frac", type=float, default=0.6,
                        help="Keep edges appearing in >= this fraction of bootstraps. Default 0.6")
    parser.add_argument("--sign-frac", type=float, default=0.9,
                        help="Require sign consistency >= this among selected bootstraps. Default 0.9")

    # Predictor overlap filtering (mandatory if provided)
    parser.add_argument("--predictor-map", type=str, default=None,
                        help="Optional predictor SNP map enabling overlap filtering.")
    parser.add_argument("--overlap-threshold", type=float, default=0.25,
                        help="Remove edges if predictor Jaccard overlap >= this. Default 0.25")

    # Modules
    parser.add_argument("--min-module-size", type=int, default=10,
                        help="Minimum module size (connected component size). Default 10")

    args = parser.parse_args()

    phenotype = args.phenotype
    use_fixed = not args.use_original

    predictor_map = load_predictor_map(args.predictor_map) if args.predictor_map else None

    # Important: treat require_present_all_folds as TRUE by default (your preference)
    require_present_all_folds = True if not args.require_present_all_folds else True

    top_n = args.top_n if args.top_n and args.top_n > 0 else None

    print(f"\n{'=' * 80}")
    print("GENE CO-EXPRESSION ANALYSIS (scientifically hardened)")
    print("Cross-Fold Consistency Summary (Meta-Analysis; modules-forward)")
    print(f"{'=' * 80}\n")

    print(f"Phenotype: {phenotype}")
    print(f"Expression files: {'Covariate-adjusted (_fixed)' if use_fixed else 'Original (unadjusted)'}")
    print("Correlation: Pearson pairwise-complete (NO mean-imputation)")
    print(f"Gene filters: max_missing={args.max_missing}, min_var={args.min_var}, top_n={top_n}")
    print(f"Consistent gene list across folds: {'YES (intersection)' if require_present_all_folds else 'NO'}")
    print(f"Empirical threshold: abs(r) >= null quantile {args.null_quantile} (perms={args.null_perms})")
    print(f"Bootstrap stability: n_boot={args.n_boot}, keep >= {args.stability_frac}, sign >= {args.sign_frac}")
    if predictor_map is not None:
        print(f"Predictor overlap filter: ON (Jaccard >= {args.overlap_threshold} removed)")
    else:
        print("Predictor overlap filter: OFF (no map) => results labeled architecture-influenced")
    print(f"Modules: connected components, min_module_size={args.min_module_size}\n")

    folds = get_all_folds(phenotype)
    if not folds:
        print(f"ERROR: No folds found for phenotype '{phenotype}'")
        sys.exit(1)

    first_fold = folds[0]
    tissues = get_all_tissues(phenotype, first_fold)
    if not tissues:
        print("ERROR: No tissues found")
        sys.exit(1)

    print(f"Found {len(folds)} folds: {folds}")
    print(f"Found {len(tissues)} tissues (from Fold_{first_fold}): {', '.join(tissues[:5])}...")
    print(f"Methods: {list(MODEL_DIRS.keys())}\n")

    all_results = analyze_all_methods_tissues(
        phenotype,
        folds,
        tissues,
        use_fixed=use_fixed,
        max_missing=float(args.max_missing),
        min_var=float(args.min_var),
        top_n=top_n,
        require_present_all_folds=require_present_all_folds,
        predictor_map=predictor_map,
        overlap_threshold=float(args.overlap_threshold),
        min_pair_n=int(args.min_pair_n),
        null_quantile=float(args.null_quantile),
        null_perms=int(args.null_perms),
        null_pairs=int(args.null_pairs),
        n_boot=int(args.n_boot),
        stability_frac=float(args.stability_frac),
        sign_frac=float(args.sign_frac),
        min_module_size=int(args.min_module_size),
        rng_seed=123,
    )

    aggregated = aggregate_cross_method_tissue_consistency(all_results)

    os.makedirs(phenotype, exist_ok=True)
    suffix = "_adjusted" if use_fixed else "_original"
    narrative_file = f"{phenotype}/coexpression_consistency_summary{suffix}.txt"
    detailed_file = f"{phenotype}/coexpression_detailed_results{suffix}.txt"

    write_narrative_report(narrative_file, all_results, aggregated, phenotype, folds, use_fixed, args)
    write_detailed_per_tissue_report(detailed_file, all_results, phenotype, use_fixed)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80 + "\n")
    print("Output files:")
    print(f"  1) {narrative_file}")
    print(f"  2) {detailed_file}\n")

    if aggregated["consistent_genes"]:
        top5 = [g["gene"] for g in aggregated["consistent_genes"][:5]]
        print(f"Top consistent hub genes: {', '.join(top5)}")
    print(f"Overall module preservation mean best-Jaccard: {aggregated['module_preservation_overall_mean']:.3f} (n={aggregated['module_preservation_n']})")


if __name__ == "__main__":
    main()
