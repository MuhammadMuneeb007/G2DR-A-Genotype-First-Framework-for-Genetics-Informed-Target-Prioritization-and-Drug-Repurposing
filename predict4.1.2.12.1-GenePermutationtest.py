#!/usr/bin/env python3
"""
predict_PermutationTest_COMPLETE.py
====================================

WHAT THIS TEST DOES:
--------------------
Tests whether your TOP-N ranked genes find more migraine drugs than random genes.

METHOD:
1. Take your top-N genes (e.g., top 200 from your ranking)
2. Find all drugs that target those genes (from OpenTargets + DGIdb + ChEMBL)
3. Count how many of those drugs are known migraine drugs
4. Repeat 1000× with random gene sets of same size
5. Compare: Do your top genes beat random?

BIOLOGICAL INTERPRETATION:
If p < 0.05: Your gene ranking captures migraine biology better than chance
If p >= 0.05: Your ranking doesn't specifically recover migraine drugs

Usage:
  # Use TOP_200_1.csv (default)
  python predict_PermutationTest_COMPLETE.py migraine
  
  # Use TOP_500_1.csv
  python predict_PermutationTest_COMPLETE.py migraine --top-genes 500
  
  # Custom options
  python predict_PermutationTest_COMPLETE.py migraine --top-genes 200 --n-perm 5000 --workers 20
"""

import argparse
import json
import re
import time
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
import mygene

# =============================================================================
# CONFIG
# =============================================================================
BASE_DIR = Path("/data/ascher02/uqmmune1/ANNOVAR")

# API endpoints
OPENTARGETS_GQL = "https://api.platform.opentargets.org/api/v4/graphql"
DGIDB_GQL = "https://dgidb.org/api/graphql"
CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"

# Cache
CACHE_DIR = BASE_DIR / "PermutationCache"

# Paths
FINAL_INTEGRATION_DIR = "GeneDifferentialExpression/Files/UltimateCompleteRankingAnalysis/FinalIntegration"
VOLCANO_REL = "GeneDifferentialExpression/Files/combined_volcano_data_all_models.csv"

# Batch sizes
OT_BATCH_SIZE = 25
DGIDB_BATCH_SIZE = 100
CHEMBL_MAX_MOLS = 30

# =============================================================================
# UTILITIES
# =============================================================================

def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def normalize_drug(name: str) -> str:
    """Normalize drug name for matching."""
    if not name or pd.isna(name):
        return ""
    s = str(name).strip().lower()
    s = re.sub(r"\([^)]*\)", " ", s)
    salts = ["hydrochloride", "hcl", "sodium", "potassium", "calcium",
             "succinate", "tartrate", "maleate", "phosphate", "sulfate",
             "acetate", "chloride", "mesylate", "citrate", "bromide",
             "fumarate", "nitrate", "besylate"]
    for salt in salts:
        s = re.sub(rf"\b{salt}\b", " ", s)
    s = re.sub(r"\b\d+(\.\d+)?\s*(mg|mcg|g|ml|iu|units?)\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return re.sub(r"\s+", " ", s)

def strip_version(g: str) -> str:
    """Remove version suffix from Ensembl ID."""
    if not g:
        return ""
    return re.sub(r"\.\d+$", "", str(g).strip())

def safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    s = str(x).strip()
    return "" if s.lower() in {"", "nan", "none", "null"} else s

def requests_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "PermutationTest/1.0",
        "Accept": "application/json",
    })
    return s

# =============================================================================
# ENSEMBL → SYMBOL CONVERSION
# =============================================================================

def convert_ensembl_to_symbols(ensembl_ids: List[str], cache_file: Path) -> Dict[str, str]:
    """Convert Ensembl IDs to gene symbols using MyGene with caching."""
    
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            cached = json.load(f)
    else:
        cached = {}
    
    to_query = [eid for eid in ensembl_ids if eid not in cached]
    
    if len(to_query) == 0:
        print(f"   ✓ All {len(ensembl_ids)} Ensembl IDs in symbol cache")
        return {eid: cached.get(eid, "") for eid in ensembl_ids}
    
    print(f"   Converting {len(to_query)} new Ensembl IDs to symbols (cached: {len(cached)})...")
    
    mg = mygene.MyGeneInfo()
    batch_size = 1000
    new_mappings = {}
    
    for i in tqdm(range(0, len(to_query), batch_size), desc="   MyGene"):
        batch = to_query[i:i+batch_size]
        
        try:
            results = mg.querymany(
                batch,
                scopes='ensembl.gene',
                fields='symbol',
                species='human',
                returnall=True,
                verbose=False
            )
            
            for result in results.get('out', []):
                query = result.get('query', '')
                symbol = result.get('symbol', '')
                if query and symbol:
                    new_mappings[query] = symbol.upper()
                elif query:
                    new_mappings[query] = ""
                    
        except Exception as e:
            print(f"   ⚠️ MyGene batch error: {e}")
            continue
    
    cached.update(new_mappings)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump(cached, f)
    
    found = sum(1 for v in new_mappings.values() if v)
    print(f"   ✓ Found symbols for {found}/{len(to_query)} new IDs")
    
    return {eid: cached.get(eid, "") for eid in ensembl_ids}

# =============================================================================
# LOAD DATA
# =============================================================================

def load_top_genes(phenotype: str, top_n: int) -> Tuple[List[str], pd.DataFrame]:
    """
    Load top N genes from FinalIntegration/TOP_{N}_1.csv
    
    Returns:
        (gene_list, full_dataframe)
    """
    # Construct path to TOP_N_1.csv file
    genes_dir = BASE_DIR / phenotype / FINAL_INTEGRATION_DIR
    genes_file = genes_dir / f"TOP_{top_n}_1.csv"
    
    if not genes_file.exists():
        raise FileNotFoundError(
            f"Gene file not found: {genes_file}\n"
            f"Available files in {genes_dir}:\n" + 
            "\n".join(f"  - {f.name}" for f in genes_dir.glob("TOP_*.csv"))
        )
    
    print(f"📂 Loading: {genes_file}")
    df = pd.read_csv(genes_file)
    
    if 'Gene' not in df.columns:
        raise ValueError(f"File must have 'Gene' column. Found: {list(df.columns)}")
    
    df['Gene'] = df['Gene'].astype(str).apply(strip_version)
    
    # Get rank column
    rank_col = None
    for c in ['Rank', 'Combined_Rank', 'GeneRank', 'rank']:
        if c in df.columns:
            rank_col = c
            break
    
    if rank_col:
        df = df.sort_values(rank_col, ascending=True).reset_index(drop=True)
    
    genes = df['Gene'].tolist()
    
    print(f"   ✓ Loaded {len(genes)} genes")
    print(f"   ✓ Top 10: {', '.join(genes[:10])}")
    
    return genes, df

def load_universe_genes(phenotype: str) -> List[str]:
    """Load all genes from volcano file as universe."""
    volcano_path = BASE_DIR / phenotype / VOLCANO_REL
    
    if volcano_path.exists():
        print(f"📂 Loading universe from: {volcano_path.name}")
        df = pd.read_csv(volcano_path, usecols=['Gene'])
    else:
        # Fallback: use all genes from ranking
        print(f"   ⚠️ Volcano file not found, using full ranking as universe")
        genes_dir = BASE_DIR / phenotype / FINAL_INTEGRATION_DIR
        ranked_file = genes_dir / "RANKED_1.csv"
        if not ranked_file.exists():
            raise FileNotFoundError(f"Cannot find universe genes: {volcano_path} or {ranked_file}")
        df = pd.read_csv(ranked_file, usecols=['Gene'])
    
    df['Gene'] = df['Gene'].astype(str).apply(strip_version)
    universe = df['Gene'].unique().tolist()
    
    print(f"   ✓ Universe: {len(universe)} genes")
    return universe

def load_reference_drugs(phenotype: str) -> Set[str]:
    """Load reference migraine drugs."""
    for path in [Path("migraine_drugs.csv"), BASE_DIR / "migraine_drugs.csv"]:
        if path.exists():
            print(f"📂 Loading reference drugs: {path}")
            df = pd.read_csv(path)
            drug_col = next((c for c in ['drug_name', 'DrugName', 'drug'] if c in df.columns), None)
            if drug_col:
                drugs = set(df[drug_col].apply(normalize_drug).unique())
                drugs.discard('')
                print(f"   ✓ {len(drugs)} reference migraine drugs")
                return drugs
    raise FileNotFoundError("migraine_drugs.csv not found")

# =============================================================================
# INCREMENTAL CACHE
# =============================================================================

class DrugCache:
    """Incremental cache for gene→drug lookups."""
    
    def __init__(self, cache_file: Path):
        self.cache_file = cache_file
        self.data = self._load()
    
    def _load(self) -> Dict[str, Dict[str, Any]]:
        if self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save(self):
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, 'w') as f:
            json.dump(self.data, f)
    
    def get_drugs(self, gene: str, source: str) -> Optional[List[str]]:
        if gene in self.data:
            return self.data[gene].get(source)
        return None
    
    def set_drugs(self, gene: str, source: str, drugs: List[str]):
        if gene not in self.data:
            self.data[gene] = {}
        self.data[gene][source] = drugs
    
    def has_source(self, gene: str, source: str) -> bool:
        return gene in self.data and source in self.data[gene]
    
    def get_all_drugs(self, gene: str) -> Set[str]:
        if gene not in self.data:
            return set()
        all_drugs = set()
        for source_drugs in self.data[gene].values():
            if isinstance(source_drugs, list):
                all_drugs.update(source_drugs)
        return all_drugs

# =============================================================================
# OPENTARGETS - BATCHED
# =============================================================================

def opentargets_batch_query(session: requests.Session, ensembl_ids: List[str]) -> Dict[str, List[str]]:
    """Query OpenTargets for multiple genes in ONE request."""
    if not ensembl_ids:
        return {}
    
    blocks = []
    for i, eid in enumerate(ensembl_ids):
        blocks.append(f'''
            t{i}: target(ensemblId: "{eid}") {{
              knownDrugs {{
                rows {{
                  drug {{ name }}
                }}
              }}
            }}
        ''')
    
    query = "query batchKnownDrugs {\n" + "\n".join(blocks) + "\n}"
    gene_to_drugs: Dict[str, List[str]] = {eid: [] for eid in ensembl_ids}
    
    try:
        r = session.post(OPENTARGETS_GQL, json={"query": query}, timeout=60)
        if r.status_code == 200:
            data = r.json()
            if "errors" not in data:
                payload = data.get("data", {}) or {}
                for i, eid in enumerate(ensembl_ids):
                    targ = payload.get(f"t{i}")
                    if targ and targ.get("knownDrugs"):
                        drugs = []
                        for row in targ["knownDrugs"].get("rows", []):
                            drug = row.get("drug", {})
                            name = safe_str(drug.get("name"))
                            if name:
                                norm = normalize_drug(name)
                                if norm:
                                    drugs.append(norm)
                        gene_to_drugs[eid] = list(set(drugs))
    except Exception:
        pass
    
    return gene_to_drugs

def query_opentargets_incremental(ensembl_ids: List[str], cache: DrugCache, batch_size: int = OT_BATCH_SIZE) -> int:
    """Query OpenTargets only for genes not already cached."""
    
    to_query = [eid for eid in ensembl_ids if not cache.has_source(eid, 'opentargets')]
    
    if len(to_query) == 0:
        print(f"   OpenTargets: All {len(ensembl_ids)} genes cached")
        return 0
    
    print(f"   OpenTargets: Querying {len(to_query)} new genes...")
    
    session = requests_session()
    total_drugs = 0
    
    for i in tqdm(range(0, len(to_query), batch_size), desc="   OpenTargets"):
        batch = to_query[i:i+batch_size]
        results = opentargets_batch_query(session, batch)
        
        for eid, drugs in results.items():
            cache.set_drugs(eid, 'opentargets', drugs)
            total_drugs += len(drugs)
        
        time.sleep(0.15)
        
        if (i // batch_size) % 50 == 0:
            cache.save()
    
    cache.save()
    print(f"         Found {total_drugs} drug annotations")
    return total_drugs

# =============================================================================
# DGIDB - BATCHED
# =============================================================================

def dgidb_batch_query(session: requests.Session, symbols: List[str]) -> Dict[str, List[str]]:
    """Query DGIdb for multiple genes in ONE request."""
    if not symbols:
        return {}
    
    safe_terms = [s.replace('"', '\\"') for s in symbols if s]
    if not safe_terms:
        return {}
    
    terms_str = ", ".join([f'"{t}"' for t in safe_terms])
    
    query = f"""
    query {{
      geneMatches(searchTerms: [{terms_str}]) {{
        directMatches {{
          matches {{
            name
            interactions {{
              drug {{ name }}
            }}
          }}
        }}
      }}
    }}
    """
    
    gene_to_drugs: Dict[str, List[str]] = {s.upper(): [] for s in symbols}
    
    try:
        r = session.post(DGIDB_GQL, json={"query": query}, timeout=90)
        if r.status_code == 200:
            data = r.json()
            if "errors" not in data:
                gm = (data.get("data") or {}).get("geneMatches") or {}
                for dm in gm.get("directMatches") or []:
                    for m in dm.get("matches") or []:
                        gene_name = safe_str(m.get("name")).upper()
                        drugs = []
                        for inter in m.get("interactions") or []:
                            drug = inter.get("drug") or {}
                            name = safe_str(drug.get("name"))
                            if name:
                                norm = normalize_drug(name)
                                if norm:
                                    drugs.append(norm)
                        if gene_name in gene_to_drugs:
                            gene_to_drugs[gene_name] = list(set(drugs))
    except Exception:
        pass
    
    return gene_to_drugs

def query_dgidb_incremental(symbols: List[str], ensembl_to_symbol: Dict[str, str], cache: DrugCache, batch_size: int = DGIDB_BATCH_SIZE) -> int:
    """Query DGIdb only for genes not already cached."""
    
    symbol_to_ensembl = {v: k for k, v in ensembl_to_symbol.items() if v}
    
    symbols_to_query = []
    for sym in symbols:
        if sym:
            eid = symbol_to_ensembl.get(sym.upper())
            if eid and not cache.has_source(eid, 'dgidb'):
                symbols_to_query.append(sym)
    
    if len(symbols_to_query) == 0:
        n_with_symbol = sum(1 for s in symbols if s)
        print(f"   DGIdb: All {n_with_symbol} genes cached")
        return 0
    
    print(f"   DGIdb: Querying {len(symbols_to_query)} new symbols...")
    
    session = requests_session()
    total_drugs = 0
    
    for i in tqdm(range(0, len(symbols_to_query), batch_size), desc="   DGIdb"):
        batch = symbols_to_query[i:i+batch_size]
        results = dgidb_batch_query(session, batch)
        
        for sym, drugs in results.items():
            eid = symbol_to_ensembl.get(sym.upper())
            if eid:
                cache.set_drugs(eid, 'dgidb', drugs)
                total_drugs += len(drugs)
        
        time.sleep(0.2)
        
        if (i // batch_size) % 20 == 0:
            cache.save()
    
    cache.save()
    print(f"         Found {total_drugs} drug annotations")
    return total_drugs

# =============================================================================
# CHEMBL - THREADED
# =============================================================================

def chembl_find_target(session: requests.Session, symbol: str) -> Optional[str]:
    if not symbol:
        return None
    try:
        r = session.get(f"{CHEMBL_API}/target/search.json", params={"q": symbol, "limit": 3}, timeout=20)
        if r.status_code == 200:
            targets = r.json().get("targets") or []
            if targets:
                return safe_str(targets[0].get("target_chembl_id"))
    except:
        pass
    return None

def chembl_target_drugs(session: requests.Session, target_id: str, max_mols: int) -> List[str]:
    if not target_id:
        return []
    
    drugs = []
    try:
        r = session.get(f"{CHEMBL_API}/activity.json", params={"target_chembl_id": target_id, "limit": 200}, timeout=30)
        if r.status_code == 200:
            seen = set()
            for a in r.json().get("activities") or []:
                mol_id = safe_str(a.get("molecule_chembl_id"))
                if mol_id and mol_id not in seen:
                    seen.add(mol_id)
                    name = safe_str(a.get("molecule_pref_name")) or mol_id
                    norm = normalize_drug(name)
                    if norm:
                        drugs.append(norm)
                    if len(drugs) >= max_mols:
                        break
    except:
        pass
    return list(set(drugs))

def query_chembl_single(args: Tuple[str, str]) -> Tuple[str, str, List[str]]:
    eid, symbol = args
    session = requests_session()
    target = chembl_find_target(session, symbol)
    if target:
        drugs = chembl_target_drugs(session, target, CHEMBL_MAX_MOLS)
        return (eid, symbol, drugs)
    return (eid, symbol, [])

def query_chembl_incremental(ensembl_to_symbol: Dict[str, str], cache: DrugCache, workers: int = 12) -> int:
    """Query ChEMBL only for genes not already cached."""
    
    to_query = []
    for eid, sym in ensembl_to_symbol.items():
        if sym and not cache.has_source(eid, 'chembl'):
            to_query.append((eid, sym))
    
    if len(to_query) == 0:
        n_with_symbol = sum(1 for s in ensembl_to_symbol.values() if s)
        print(f"   ChEMBL: All {n_with_symbol} genes cached")
        return 0
    
    print(f"   ChEMBL: Querying {len(to_query)} new genes with {workers} workers...")
    
    total_drugs = 0
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(query_chembl_single, args): args[0] for args in to_query}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="   ChEMBL"):
            try:
                eid, sym, drugs = future.result()
                cache.set_drugs(eid, 'chembl', drugs)
                total_drugs += len(drugs)
            except:
                pass
    
    cache.save()
    print(f"         Found {total_drugs} drug annotations")
    return total_drugs

# =============================================================================
# BUILD GENE→DRUG LOOKUP
# =============================================================================

def build_gene_drug_lookup(ensembl_ids: List[str], phenotype: str, workers: int = 12, use_chembl: bool = True) -> Dict[str, Set[str]]:
    """Build gene→drugs lookup using ALL databases with incremental caching."""
    
    symbol_cache_file = CACHE_DIR / f"{phenotype}_ensembl_to_symbol.json"
    drug_cache_file = CACHE_DIR / f"{phenotype}_gene_drugs_incremental.json"
    
    print(f"\n   [0/3] Converting Ensembl IDs to Symbols...")
    ensembl_to_symbol = convert_ensembl_to_symbols(ensembl_ids, symbol_cache_file)
    
    n_with_symbol = sum(1 for v in ensembl_to_symbol.values() if v)
    print(f"         {n_with_symbol}/{len(ensembl_ids)} genes have symbols")
    
    cache = DrugCache(drug_cache_file)
    print(f"   Loaded drug cache with {len(cache.data)} genes")
    
    print(f"\n   [1/3] OpenTargets...")
    query_opentargets_incremental(ensembl_ids, cache, OT_BATCH_SIZE)
    
    print(f"\n   [2/3] DGIdb...")
    symbols = [ensembl_to_symbol.get(eid, "") for eid in ensembl_ids]
    symbols = [s for s in symbols if s]
    query_dgidb_incremental(symbols, ensembl_to_symbol, cache, DGIDB_BATCH_SIZE)
    
    if use_chembl:
        print(f"\n   [3/3] ChEMBL...")
        query_chembl_incremental(ensembl_to_symbol, cache, workers)
    else:
        print(f"\n   [3/3] ChEMBL: Skipped")
    
    gene_to_drugs: Dict[str, Set[str]] = {}
    for eid in ensembl_ids:
        gene_to_drugs[eid] = cache.get_all_drugs(eid)
    
    genes_with_drugs = sum(1 for d in gene_to_drugs.values() if len(d) > 0)
    total_annotations = sum(len(d) for d in gene_to_drugs.values())
    
    print(f"\n   Summary:")
    print(f"   - Genes with drugs: {genes_with_drugs}/{len(ensembl_ids)} ({100*genes_with_drugs/len(ensembl_ids):.1f}%)")
    print(f"   - Total drug annotations: {total_annotations}")
    
    return gene_to_drugs

# =============================================================================
# PERMUTATION TEST
# =============================================================================

def compute_overlap(genes: List[str], gene_to_drugs: Dict[str, Set[str]], reference_drugs: Set[str]) -> Tuple[int, Set[str], int]:
    """Compute overlap between drugs and reference."""
    all_drugs = set()
    for gene in genes:
        if gene in gene_to_drugs:
            all_drugs.update(gene_to_drugs[gene])
    
    overlap = all_drugs & reference_drugs
    return len(overlap), overlap, len(all_drugs)

def run_permutation_test(
    observed_genes: List[str],
    universe_genes: List[str],
    gene_to_drugs: Dict[str, Set[str]],
    reference_drugs: Set[str],
    n_permutations: int = 1000,
    seed: int = 42
) -> Dict:
    """Run permutation test."""
    random.seed(seed)
    np.random.seed(seed)
    
    n_genes = len(observed_genes)
    
    # Observed
    obs_overlap, obs_drugs, obs_total = compute_overlap(observed_genes, gene_to_drugs, reference_drugs)
    
    print(f"\n📊 OBSERVED (your top {n_genes} genes):")
    print(f"   Total drugs found: {obs_total}")
    print(f"   Overlap with migraine drugs: {obs_overlap}")
    
    if obs_drugs:
        print(f"   Overlapping drugs: {', '.join(sorted(list(obs_drugs))[:10])}")
    
    # Valid universe
    valid_universe = [g for g in universe_genes if len(gene_to_drugs.get(g, set())) > 0]
    print(f"   Universe genes with drugs: {len(valid_universe)}")
    
    if len(valid_universe) < n_genes:
        print(f"   ⚠️ Warning: Universe smaller than sample")
        valid_universe = universe_genes
    
    # Permutations
    print(f"\n🔄 Running {n_permutations} permutations...")
    null_overlaps = []
    null_totals = []
    
    for _ in tqdm(range(n_permutations), desc="   Permuting"):
        random_genes = random.sample(valid_universe, min(n_genes, len(valid_universe)))
        rand_overlap, _, rand_total = compute_overlap(random_genes, gene_to_drugs, reference_drugs)
        null_overlaps.append(rand_overlap)
        null_totals.append(rand_total)
    
    null_overlaps = np.array(null_overlaps)
    
    # Statistics
    p_value = (np.sum(null_overlaps >= obs_overlap) + 1) / (n_permutations + 1)
    null_mean = np.mean(null_overlaps)
    null_std = np.std(null_overlaps)
    z_score = (obs_overlap - null_mean) / null_std if null_std > 0 else 0
    fold = obs_overlap / null_mean if null_mean > 0 else (float('inf') if obs_overlap > 0 else 0)
    
    return {
        'n_genes': n_genes,
        'observed_overlap': obs_overlap,
        'observed_total_drugs': obs_total,
        'observed_drugs': sorted(list(obs_drugs)),
        'null_mean': round(null_mean, 2),
        'null_std': round(null_std, 2),
        'null_median': round(np.median(null_overlaps), 2),
        'null_max': int(np.max(null_overlaps)),
        'null_min': int(np.min(null_overlaps)),
        'null_mean_total_drugs': round(np.mean(null_totals), 1),
        'p_value': p_value,
        'z_score': round(z_score, 2),
        'fold_vs_null': round(fold, 2),
        'n_permutations': n_permutations,
        'null_distribution': null_overlaps.tolist()
    }

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phenotype", help="Phenotype (e.g., migraine)")
    parser.add_argument("--top-genes", type=int, default=200, help="Load TOP_{N}_1.csv file (default: 200)")
    parser.add_argument("--n-perm", type=int, default=1000, help="Permutation iterations (default: 1000)")
    parser.add_argument("--workers", type=int, default=12, help="Thread workers for ChEMBL (default: 12)")
    parser.add_argument("--no-chembl", action="store_true", help="Skip ChEMBL (faster)")
    parser.add_argument("--rebuild-cache", action="store_true", help="Force rebuild ALL caches")
    args = parser.parse_args()
    
    phenotype = args.phenotype
    top_n = args.top_genes
    
    print("=" * 100)
    print("🧬 PERMUTATION TEST - DRUG ENRICHMENT ANALYSIS")
    print("=" * 100)
    print(f"\n📖 WHAT THIS TEST DOES:")
    print(f"   1. Takes your top-{top_n} ranked genes")
    print(f"   2. Finds drugs targeting those genes (OpenTargets + DGIdb + ChEMBL)")
    print(f"   3. Counts overlap with known migraine drugs")
    print(f"   4. Compares against {args.n_perm} random gene sets")
    print(f"   5. Tests: Do your genes beat random? (p < 0.05 = YES)")
    print()
    print(f"📋 Phenotype: {phenotype}")
    print(f"📋 Loading file: TOP_{top_n}_1.csv")
    print(f"📋 Permutations: {args.n_perm}")
    print(f"📋 Workers: {args.workers}")
    print(f"📋 Use ChEMBL: {not args.no_chembl}")
    print()
    
    if args.rebuild_cache:
        cache_files = list(CACHE_DIR.glob(f"{phenotype}_*.json"))
        for f in cache_files:
            f.unlink()
            print(f"   Deleted cache: {f.name}")
    
    output_dir = BASE_DIR / phenotype / "GeneDifferentialExpression" / "Files" / "UltimateCompleteRankingAnalysis" / "PermutationTest"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # STEP 1: Load genes
    # =========================================================================
    print("=" * 100)
    print("STEP 1: LOADING GENES")
    print("=" * 100)
    
    observed_genes, genes_df = load_top_genes(phenotype, top_n)
    universe_genes = load_universe_genes(phenotype)
    
    # =========================================================================
    # STEP 2: Build gene→drug lookup
    # =========================================================================
    print("\n" + "=" * 100)
    print("STEP 2: BUILDING GENE→DRUG LOOKUP (ALL DATABASES)")
    print("=" * 100)
    
    gene_to_drugs = build_gene_drug_lookup(
        universe_genes,
        phenotype,
        workers=args.workers,
        use_chembl=not args.no_chembl
    )
    
    # =========================================================================
    # STEP 3: Load reference drugs
    # =========================================================================
    print("\n" + "=" * 100)
    print("STEP 3: LOADING REFERENCE DRUGS")
    print("=" * 100)
    
    reference_drugs = load_reference_drugs(phenotype)
    
    # =========================================================================
    # STEP 4: Permutation test
    # =========================================================================
    print("\n" + "=" * 100)
    print("STEP 4: PERMUTATION TEST")
    print("=" * 100)
    
    results = run_permutation_test(
        observed_genes=observed_genes,
        universe_genes=universe_genes,
        gene_to_drugs=gene_to_drugs,
        reference_drugs=reference_drugs,
        n_permutations=args.n_perm
    )
    
    # =========================================================================
    # STEP 5: Results
    # =========================================================================
    print("\n" + "=" * 100)
    print("RESULTS")
    print("=" * 100)
    
    print(f"\n   YOUR TOP {results['n_genes']} GENES:")
    print(f"   Total drugs found: {results['observed_total_drugs']}")
    print(f"   Overlap with migraine drugs: {results['observed_overlap']}")
    
    print(f"\n   NULL DISTRIBUTION ({results['n_permutations']} random gene sets):")
    print(f"   Mean: {results['null_mean']:.2f} ± {results['null_std']:.2f}")
    print(f"   Median: {results['null_median']:.2f}")
    print(f"   Range: [{results['null_min']}, {results['null_max']}]")
    
    print(f"\n   STATISTICS:")
    print(f"   Fold enrichment: {results['fold_vs_null']:.2f}x")
    print(f"   Z-score: {results['z_score']:.2f}")
    print(f"   P-value: {results['p_value']:.4f}")
    
    # Interpretation
    print("\n" + "=" * 100)
    print("INTERPRETATION")
    print("=" * 100)
    if results['p_value'] < 0.01:
        print("✅ HIGHLY SIGNIFICANT (p < 0.01)")
        print(f"   Your top {top_n} genes retrieve {results['fold_vs_null']:.1f}x more migraine drugs than random!")
        print(f"   → Your gene ranking CAPTURES MIGRAINE BIOLOGY")
    elif results['p_value'] < 0.05:
        print("✅ SIGNIFICANT (p < 0.05)")
        print(f"   Your top {top_n} genes retrieve {results['fold_vs_null']:.1f}x more migraine drugs than random!")
        print(f"   → Your gene ranking captures migraine biology")
    elif results['p_value'] < 0.10:
        print("⚠️ TRENDING (p < 0.10)")
        print(f"   Trend observed ({results['fold_vs_null']:.1f}x) but not significant.")
    else:
        print("❌ NOT SIGNIFICANT")
        print(f"   Your genes don't beat random (observed={results['observed_overlap']}, expected={results['null_mean']:.1f})")
        print(f"   → Gene ranking may not capture migraine drug targets well")
    
    if results['observed_drugs']:
        print(f"\n🎯 OVERLAPPING DRUGS ({results['observed_overlap']}):")
        for drug in results['observed_drugs'][:30]:
            print(f"   - {drug}")
        if len(results['observed_drugs']) > 30:
            print(f"   ... and {len(results['observed_drugs']) - 30} more")
    
    # =========================================================================
    # STEP 6: Save
    # =========================================================================
    print("\n" + "=" * 100)
    print("STEP 6: SAVING RESULTS")
    print("=" * 100)
    
    summary = {k: v for k, v in results.items() if k != 'null_distribution'}
    summary_file = output_dir / f"permutation_summary_top{top_n}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"   💾 {summary_file}")
    
    null_file = output_dir / f"null_distribution_top{top_n}.csv"
    pd.DataFrame({'overlap': results['null_distribution']}).to_csv(null_file, index=False)
    print(f"   💾 {null_file}")
    
    drugs_file = output_dir / f"overlapping_drugs_top{top_n}.txt"
    with open(drugs_file, 'w') as f:
        f.write('\n'.join(results['observed_drugs']))
    print(f"   💾 {drugs_file}")
    
    txt_file = output_dir / f"permutation_results_top{top_n}.txt"
    with open(txt_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"PERMUTATION TEST - TOP {top_n} GENES\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"File: TOP_{top_n}_1.csv\n")
        f.write(f"Phenotype: {phenotype}\n")
        f.write(f"Databases: OpenTargets + DGIdb + ChEMBL\n")
        f.write(f"Permutations: {results['n_permutations']}\n\n")
        f.write("OBSERVED:\n")
        f.write(f"  Drugs found: {results['observed_total_drugs']}\n")
        f.write(f"  Migraine overlap: {results['observed_overlap']}\n\n")
        f.write("NULL:\n")
        f.write(f"  Mean: {results['null_mean']:.2f} ± {results['null_std']:.2f}\n")
        f.write(f"  Range: [{results['null_min']}, {results['null_max']}]\n\n")
        f.write("STATISTICS:\n")
        f.write(f"  Fold: {results['fold_vs_null']:.2f}x\n")
        f.write(f"  Z-score: {results['z_score']:.2f}\n")
        f.write(f"  P-value: {results['p_value']:.4f}\n\n")
        f.write("CONCLUSION: " + ("SIGNIFICANT" if results['p_value'] < 0.05 else "NOT SIGNIFICANT") + "\n")
        if results['observed_drugs']:
            f.write(f"\nOVERLAPPING DRUGS ({len(results['observed_drugs'])}):\n")
            for drug in results['observed_drugs']:
                f.write(f"  - {drug}\n")
    print(f"   💾 {txt_file}")
    
    print("\n" + "=" * 100)
    print(f"🎉 COMPLETE! Results in: {output_dir}")
    print("=" * 100)

if __name__ == "__main__":
    main()