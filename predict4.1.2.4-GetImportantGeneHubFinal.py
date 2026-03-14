#!/usr/bin/env python3
"""
?? HUB GENE ANALYSIS FOR RANKED GENES - FIXED (NO GENE LOSS) + PROPER SYMBOL INTEGRATION

Fixes the common issue:
- Ranking files may contain aliases / phenotype labels (e.g., FHM3) instead of true gene symbols
- STRING network nodes are returned as preferred gene symbols (e.g., SCN1A)
- Original code matched hub info ONLY by df['Symbol'] => many genes got Hub_Score=0

This version:
1) NEVER DROPS genes from any RANKED_* or RANKED_WITH_HUB_* outputs
2) Resolves symbol via:
   - MyGene.info: Ensembl gene id (df['Gene']) -> official symbol (df['Symbol_Resolved'])
   - STRING get_string_ids: resolved/original symbol -> preferredName (df['Symbol_Preferred'])
3) Calculates hub metrics on the FULL STRING network
4) Matches hub metrics using Symbol_Preferred first (best), then Symbol_Resolved, then Symbol

Outputs:
- RANKED_WITH_HUB_*.csv (same number of rows as input, always)
- HUB_GENES_* subsets (filtered views, optional)
- ALL_GENES_WITH_HUB_SCORES.csv (hub_score>0 subset)
- hub_cache.csv, string_edges.csv, symbol_resolution_cache.csv

Based on: predict4.4.5-GetImportantGeneHubFinal.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys
import requests
import time
import networkx as nx
import warnings
warnings.filterwarnings("ignore")


# -----------------------------
# Helpers
# -----------------------------
def _clean_sym(x: str) -> str:
    if x is None:
        return ""
    return str(x).strip().upper()


def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class HubGeneAnalyzer:
    def __init__(self, phenotype, force_string=False, hub_percentile=10,
                 confidence=0.4, resolve_with_mygene=True, resolve_with_string_ids=True):
        self.phenotype = phenotype
        self.base_path = Path(phenotype)

        # ? FIX 1: detect ranked output folder from previous step
        self.results_dir = self.base_path / "GeneDifferentialExpression" / "Files"
        self.analysis_dir = self._detect_previous_step_dir(self.results_dir)

        self.force_string = force_string
        self.hub_percentile = hub_percentile
        self.confidence = confidence

        # Resolution toggles
        self.resolve_with_mygene = resolve_with_mygene
        self.resolve_with_string_ids = resolve_with_string_ids

        # STRING API
        self.string_api_url = "https://string-db.org/api"
        self.species = "9606"
        self.max_genes_per_request = 2000

        # ? FIX 2: MyGene batch "querymany" is POST to /v3/query (NOT /v3/querymany)
        self.mygene_batch_url = "https://mygene.info/v3/query"

        # Network storage
        self.network_graph = None
        self.interactions_list = []

        # caches in memory
        self.symbol_resolved_map = {}    # original_symbol -> resolved_symbol (from Ensembl or original if ok)
        self.symbol_preferred_map = {}   # resolved_symbol -> STRING preferred symbol

        self.analysis_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 120)
        print("?? HUB GENE ANALYSIS (FIXED SYMBOL RESOLUTION, NO GENE LOSS)")
        print("=" * 120)
        print(f"?? Phenotype: {phenotype}")
        print(f"?? Ranked/analysis directory: {self.analysis_dir}")
        print(f"?? Force STRING query (ignore cache): {force_string}")
        print(f"?? STRING confidence threshold: {confidence}")
        print(f"?? Resolve via MyGene.info (Ensembl?Symbol): {resolve_with_mygene}")
        print(f"?? Resolve via STRING get_string_ids (Symbol?Preferred): {resolve_with_string_ids}")
        print(f"??? Hub classification: Top {hub_percentile}% = High Hubs, Top 25% = Moderate Hubs")
        print("=" * 120)

    def _detect_previous_step_dir(self, results_dir: Path) -> Path:
        """
        Prefer the output folder from the previous ranking pipeline step.
        Typical ranking output dir from your pipeline:
          .../Files/UltimateCompleteRankingAnalysis/
        Older scripts used:
          .../Files/Analysis/
        """
        candidates = [
            results_dir / "UltimateCompleteRankingAnalysis",
            results_dir / "Analysis",
        ]
        for c in candidates:
            if (c / "RANKED_composite.csv").exists():
                return c
        # fallback: keep Analysis (and create)
        return results_dir / "UltimateCompleteRankingAnalysis"

    # ---------------------------------------------------------------------------------
    # Load ranked files
    # ---------------------------------------------------------------------------------
    def load_ranked_files(self):
        print("\n?? Loading ranked gene files...")

        composite_file = self.analysis_dir / "RANKED_composite.csv"
        if not composite_file.exists():
            print("? Ranked gene files not found!")
            print(f"   Expected: {composite_file}")
            print("   Make sure you ran the previous ranking step first.")
            sys.exit(1)

        composite_df = pd.read_csv(composite_file)
        print(f"? Loaded composite ranking: {len(composite_df):,} rows")

        # ? FIX: support both known_only and existing_only naming
        ranking_files = {
            "composite": composite_file,
            "reproducibility": self.analysis_dir / "RANKED_reproducibility.csv",
            "effect_size": self.analysis_dir / "RANKED_effect_size.csv",
            "significance": self.analysis_dir / "RANKED_significance.csv",
            "pan_tissue": self.analysis_dir / "RANKED_pan_tissue.csv",
            "method_consensus": self.analysis_dir / "RANKED_method_consensus.csv",
            "tier1_high": self.analysis_dir / "RANKED_tier1_high.csv",
            "novel_only": self.analysis_dir / "RANKED_novel_only.csv",
            "known_only": self.analysis_dir / "RANKED_known_only.csv",
            "existing_only": self.analysis_dir / "RANKED_existing_only.csv",  # legacy
        }

        loaded = {}
        for name, path in ranking_files.items():
            if path.exists():
                df = pd.read_csv(path)
                loaded[name] = df
                print(f"   ? Loaded {name}: {len(df):,} rows")
            else:
                print(f"   ?? {name} not found (skipping)")

        # If existing_only exists but known_only doesn't, keep both keys consistent
        if "known_only" not in loaded and "existing_only" in loaded:
            loaded["known_only"] = loaded["existing_only"]

        return loaded

    # ---------------------------------------------------------------------------------
    # Cache I/O
    # ---------------------------------------------------------------------------------
    def _cache_paths(self):
        hub_cache = self.analysis_dir / "hub_cache.csv"
        edges_cache = self.analysis_dir / "string_edges.csv"
        sym_cache = self.analysis_dir / "symbol_resolution_cache.csv"
        return hub_cache, edges_cache, sym_cache

    def load_cache(self):
        hub_cache, edges_cache, sym_cache = self._cache_paths()

        if self.force_string:
            return None, None

        if not (hub_cache.exists() and edges_cache.exists() and sym_cache.exists()):
            return None, None

        print("\n??? Loading hub + edges + symbol resolution from cache...")

        try:
            hub_df = pd.read_csv(hub_cache)
            edges_df = pd.read_csv(edges_cache)
            sym_df = pd.read_csv(sym_cache)

            hub_info = {}
            for _, row in hub_df.iterrows():
                key = _clean_sym(row.get("Preferred_Symbol", ""))
                if not key:
                    continue
                hub_info[key] = {
                    "Hub_Score": float(row.get("Hub_Score", 0.0)),
                    "Hub_Rank": int(row.get("Hub_Rank", 0)),
                    "Hub_Degree": int(row.get("Hub_Degree", 0)),
                    "Hub_Betweenness": float(row.get("Hub_Betweenness", 0.0)),
                    "Hub_Closeness": float(row.get("Hub_Closeness", 0.0)),
                    "Hub_PageRank": float(row.get("Hub_PageRank", 0.0)),
                    "Hub_Class": str(row.get("Hub_Class", "Non-Hub")),
                }

            # Store maps
            if {"Symbol", "Symbol_Resolved", "Symbol_Preferred"}.issubset(set(sym_df.columns)):
                self.symbol_resolved_map = dict(zip(sym_df["Symbol"].astype(str), sym_df["Symbol_Resolved"].astype(str)))
                self.symbol_preferred_map = dict(zip(sym_df["Symbol_Resolved"].astype(str), sym_df["Symbol_Preferred"].astype(str)))

            # Rebuild network
            self.interactions_list = edges_df.to_dict("records")
            G = nx.Graph()
            for e in self.interactions_list:
                G.add_edge(e["protein1"], e["protein2"], weight=float(e["score"]))
            self.network_graph = G

            print(f"? Cache loaded: hub nodes={len(hub_info):,}, edges={len(edges_df):,}, graph={G.number_of_nodes():,} nodes")
            return hub_info, "cache"

        except Exception as e:
            print(f"?? Cache load failed: {e}")
            return None, None

    def save_cache(self, hub_info, edges_records):
        hub_cache, edges_cache, sym_cache = self._cache_paths()

        print("\n?? Saving caches...")

        hub_rows = []
        for pref_sym_u, data in hub_info.items():
            hub_rows.append({"Preferred_Symbol": pref_sym_u, **data})
        pd.DataFrame(hub_rows).to_csv(hub_cache, index=False)
        print(f"? Saved: {hub_cache.name}")

        pd.DataFrame(edges_records).to_csv(edges_cache, index=False)
        print(f"? Saved: {edges_cache.name}")

        sym_rows = []
        for orig_sym, resolved_sym in self.symbol_resolved_map.items():
            preferred = self.symbol_preferred_map.get(resolved_sym, resolved_sym)
            sym_rows.append({
                "Symbol": orig_sym,
                "Symbol_Resolved": resolved_sym,
                "Symbol_Preferred": preferred
            })
        pd.DataFrame(sym_rows).to_csv(sym_cache, index=False)
        print(f"? Saved: {sym_cache.name}")

    # ---------------------------------------------------------------------------------
    # Symbol resolution
    # ---------------------------------------------------------------------------------
    def resolve_symbols_for_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds:
          - Symbol_Resolved: best-effort official symbol (prefer Ensembl->symbol via MyGene)
          - Symbol_Preferred: STRING preferred symbol from get_string_ids
        Never drops rows.
        """
        df = df.copy()

        if "Symbol" not in df.columns:
            df["Symbol"] = "N/A"
        if "Gene" not in df.columns:
            df["Gene"] = "N/A"

        df["Symbol"] = df["Symbol"].astype(str)
        df["Gene"] = df["Gene"].astype(str)

        if self.resolve_with_mygene:
            df = self._resolve_ensembl_to_symbol_mygene(df)

        if self.resolve_with_string_ids:
            df = self._resolve_to_string_preferred(df)

        return df

    def _resolve_ensembl_to_symbol_mygene(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        If Symbol is NA/empty or "non-gene-like" (heuristic), try resolve from Ensembl Gene id.
        """
        bad_mask = (
            df["Symbol"].isna() |
            (df["Symbol"].str.strip() == "") |
            (df["Symbol"].str.upper().isin(["N/A", "NA", "NONE"])) |
            (df["Symbol"].str.upper().str.startswith("FHM")) |
            (df["Symbol"].str.contains(r"[ /\\]", regex=True))
        )

        df["Symbol_Resolved"] = df["Symbol"].map(lambda x: str(x).strip())

        ens_mask = bad_mask & df["Gene"].str.startswith("ENSG")
        ensembl_ids = df.loc[ens_mask, "Gene"].dropna().unique().tolist()

        if not ensembl_ids:
            for sym in df["Symbol"].unique():
                if sym not in self.symbol_resolved_map:
                    self.symbol_resolved_map[sym] = str(sym).strip()
            return df

        print(f"\n?? Resolving {len(ensembl_ids):,} Ensembl IDs ? symbols via MyGene.info (batch POST /v3/query)...")

        resolved = {}

        for chunk in chunk_list(ensembl_ids, 1000):
            try:
                # ? Correct MyGene batch (querymany-style): POST to /v3/query with q as LIST
                payload = {
                    "q": chunk,                     # list of ids
                    "scopes": "ensembl.gene",
                    "fields": "symbol",
                    "species": "human",
                }
                r = requests.post(self.mygene_batch_url, json=payload, timeout=120)
                r.raise_for_status()
                data = r.json()

                # Batch response is typically a list of objects: {query:..., symbol:..., notfound:...}
                if isinstance(data, list):
                    for item in data:
                        q = item.get("query")
                        sym = item.get("symbol")
                        if q and sym:
                            resolved[str(q)] = str(sym)

                time.sleep(0.2)

            except Exception as e:
                print(f"?? MyGene chunk failed: {e}")
                continue

        def pick_resolved(row):
            sym = str(row["Symbol"]).strip()
            gene = str(row["Gene"]).strip()
            if gene in resolved:
                return resolved[gene]
            return sym

        df.loc[ens_mask, "Symbol_Resolved"] = df.loc[ens_mask].apply(pick_resolved, axis=1)

        for _, row in df.iterrows():
            orig = str(row["Symbol"])
            res = str(row["Symbol_Resolved"])
            if orig not in self.symbol_resolved_map:
                self.symbol_resolved_map[orig] = res

        print(f"? MyGene resolution done. Resolved {len(resolved):,}/{len(ensembl_ids):,} Ensembl IDs.")
        return df

    def _resolve_to_string_preferred(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Use STRING /tsv/get_string_ids to map Symbol_Resolved to preferredName.
        """
        df["Symbol_Preferred"] = df["Symbol_Resolved"].map(lambda x: str(x).strip())

        syms = df["Symbol_Resolved"].dropna().astype(str).map(lambda x: x.strip()).unique().tolist()
        syms = [s for s in syms if s and s.upper() not in ["N/A", "NA", "NONE"]]

        if not syms:
            return df

        print(f"\n?? Mapping {len(syms):,} symbols ? STRING preferredName via get_string_ids...")

        preferred_map = {}
        url = f"{self.string_api_url}/tsv/get_string_ids"

        for batch in chunk_list(syms, 2000):
            identifiers = "\r".join(batch)
            params = {
                "identifiers": identifiers,
                "species": self.species,
                "limit": 1,
                "echo_query": 1
            }
            try:
                r = requests.post(url, data=params, timeout=120)
                r.raise_for_status()
                lines = r.text.strip().split("\n")
                if len(lines) <= 1:
                    continue
                header = lines[0].split("\t")
                col = {name: i for i, name in enumerate(header)}

                q_idx = col.get("queryItem")
                pref_idx = col.get("preferredName")
                score_idx = col.get("score")

                for line in lines[1:]:
                    parts = line.split("\t")
                    if q_idx is None or pref_idx is None or len(parts) <= max(q_idx, pref_idx):
                        continue
                    q = parts[q_idx].strip()
                    pref = parts[pref_idx].strip()
                    sc = float(parts[score_idx]) if score_idx is not None and len(parts) > score_idx else 0.0

                    if q and pref:
                        if (q not in preferred_map) or (sc > preferred_map[q][1]):
                            preferred_map[q] = (pref, sc)

                time.sleep(0.3)
            except Exception as e:
                print(f"?? STRING get_string_ids batch failed: {e}")
                continue

        for q, (pref, _sc) in preferred_map.items():
            preferred_map[q] = pref

        def map_pref(x):
            x = str(x).strip()
            return preferred_map.get(x, x)

        df["Symbol_Preferred"] = df["Symbol_Resolved"].map(map_pref)

        for res in df["Symbol_Resolved"].unique():
            rs = str(res).strip()
            if rs:
                self.symbol_preferred_map[rs] = map_pref(rs)

        mapped = sum(1 for s in syms if preferred_map.get(s, s) != s)
        print(f"? STRING preferredName mapping done. Mapped {mapped:,}/{len(syms):,} symbols.")
        return df

    # ---------------------------------------------------------------------------------
    # STRING network query
    # ---------------------------------------------------------------------------------
    def query_string_network(self, preferred_symbols):
        print(f"\n?? Querying STRING network for {len(preferred_symbols):,} preferred symbols...")
        print(f"   Confidence threshold: {self.confidence}")
        print(f"   Batch size: {self.max_genes_per_request}")

        all_edges = []
        url = f"{self.string_api_url}/tsv/network"
        required_score = int(self.confidence * 1000)

        for batch_i, batch in enumerate(chunk_list(preferred_symbols, self.max_genes_per_request), 1):
            print(f"\n   Batch {batch_i}: {len(batch):,} symbols")
            identifiers = "\r".join(batch)

            params = {
                "identifiers": identifiers,
                "species": self.species,
                "required_score": required_score
            }
            try:
                r = requests.post(url, data=params, timeout=120)
                if r.status_code != 200:
                    print(f"   ?? STRING network API error: {r.status_code}")
                    continue

                lines = r.text.strip().split("\n")
                if len(lines) <= 1:
                    print("   ?? No interactions returned in this batch")
                    continue

                header = lines[0].split("\t")
                col = {name: i for i, name in enumerate(header)}

                a_idx = col.get("preferredName_A", col.get("stringId_A", 2))
                b_idx = col.get("preferredName_B", col.get("stringId_B", 3))
                s_idx = col.get("score", 5)

                batch_edges = 0
                for line in lines[1:]:
                    parts = line.split("\t")
                    if len(parts) <= max(a_idx, b_idx, s_idx):
                        continue
                    p1 = parts[a_idx].strip()
                    p2 = parts[b_idx].strip()
                    sc = float(parts[s_idx])

                    if p1 and p2:
                        all_edges.append({"protein1": p1, "protein2": p2, "score": sc})
                        batch_edges += 1

                print(f"   ? Interactions found: {batch_edges:,}")
                time.sleep(1.0)

            except Exception as e:
                print(f"   ?? STRING network batch failed: {e}")
                continue

        if not all_edges:
            print("\n?? No interactions found overall.")
            return []

        print(f"\n? Total interactions found: {len(all_edges):,}")
        self.interactions_list = all_edges
        return all_edges

    # ---------------------------------------------------------------------------------
    # Hub metrics
    # ---------------------------------------------------------------------------------
    def calculate_hub_metrics(self, edges):
        print("\n?? Calculating hub metrics...")

        G = nx.Graph()
        for e in edges:
            G.add_edge(e["protein1"], e["protein2"], weight=float(e["score"]))
        self.network_graph = G

        print(f"   Network: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

        print("   Degree centrality...")
        degree_cent = nx.degree_centrality(G)

        print("   Betweenness centrality (sampled)...")
        k_sample = max(100, int(np.sqrt(G.number_of_nodes())))
        k_sample = min(k_sample, G.number_of_nodes())
        betweenness_cent = nx.betweenness_centrality(G, k=k_sample, seed=42)

        print("   Closeness centrality...")
        closeness_cent = nx.closeness_centrality(G)

        print("   PageRank...")
        pagerank = nx.pagerank(G)

        hub_data = {}
        for node in G.nodes():
            node_u = _clean_sym(node)
            hub_score = (
                degree_cent[node] * 0.35 +
                betweenness_cent[node] * 0.15 +
                closeness_cent[node] * 0.25 +
                pagerank[node] * 0.25
            )
            hub_data[node_u] = {
                "Hub_Score": float(hub_score),
                "Hub_Degree": int(G.degree(node)),
                "Hub_Betweenness": float(betweenness_cent[node]),
                "Hub_Closeness": float(closeness_cent[node]),
                "Hub_PageRank": float(pagerank[node]),
            }

        sorted_genes = sorted(hub_data.items(), key=lambda x: x[1]["Hub_Score"], reverse=True)
        total = len(sorted_genes)
        high_cutoff = max(1, int(total * (self.hub_percentile / 100.0)))
        mod_cutoff = max(2, int(total * 0.25))

        for rank, (gene, data) in enumerate(sorted_genes, 1):
            data["Hub_Rank"] = rank
            if rank <= high_cutoff:
                data["Hub_Class"] = "High_Hub"
            elif rank <= mod_cutoff:
                data["Hub_Class"] = "Moderate_Hub"
            else:
                data["Hub_Class"] = "Low_Hub"

        print(f"? Hub metrics computed for {total:,} nodes")
        print(f"? High hubs: {high_cutoff:,} | Moderate hubs: {mod_cutoff - high_cutoff:,} | Low hubs: {total - mod_cutoff:,}")
        return hub_data

    # ---------------------------------------------------------------------------------
    # Add hub columns without removing any rows
    # ---------------------------------------------------------------------------------
    def add_hub_columns(self, df, hub_info):
        df = df.copy()

        if "Symbol" not in df.columns:
            df["Symbol"] = "N/A"
        if "Gene" not in df.columns:
            df["Gene"] = "N/A"

        df = self.resolve_symbols_for_dataframe(df)

        df["Symbol_Matched"] = ""
        df["Hub_Class"] = "Non-Hub"
        df["Hub_Score"] = 0.0
        df["Hub_Rank"] = 0
        df["Hub_Degree"] = 0
        df["Hub_Betweenness"] = 0.0
        df["Hub_Closeness"] = 0.0
        df["Hub_PageRank"] = 0.0

        if not hub_info:
            return df

        matched = 0
        for i, row in df.iterrows():
            orig = _clean_sym(row.get("Symbol", ""))
            res = _clean_sym(row.get("Symbol_Resolved", ""))
            pref = _clean_sym(row.get("Symbol_Preferred", ""))

            key = None
            if pref and pref in hub_info:
                key = pref
            elif res and res in hub_info:
                key = res
            elif orig and orig in hub_info:
                key = orig

            if key is not None:
                info = hub_info[key]
                df.at[i, "Symbol_Matched"] = key
                df.at[i, "Hub_Class"] = info.get("Hub_Class", "Non-Hub")
                df.at[i, "Hub_Score"] = round(info.get("Hub_Score", 0.0), 6)
                df.at[i, "Hub_Rank"] = int(info.get("Hub_Rank", 0))
                df.at[i, "Hub_Degree"] = int(info.get("Hub_Degree", 0))
                df.at[i, "Hub_Betweenness"] = round(info.get("Hub_Betweenness", 0.0), 6)
                df.at[i, "Hub_Closeness"] = round(info.get("Hub_Closeness", 0.0), 6)
                df.at[i, "Hub_PageRank"] = round(info.get("Hub_PageRank", 0.0), 6)
                matched += 1

        print(f"? Hub matched: {matched:,}/{len(df):,}")
        return df

    # ---------------------------------------------------------------------------------
    # Save outputs (keeping full lists)
    # ---------------------------------------------------------------------------------
    def save_results(self, ranked_with_hub):
        saved = []
        for view, df in ranked_with_hub.items():
            out = self.analysis_dir / f"RANKED_WITH_HUB_{view}.csv"
            df.to_csv(out, index=False)
            saved.append(out)
            print(f"? Saved: {out.name} ({len(df):,} rows)")

        composite = ranked_with_hub.get("composite")
        if composite is not None:
            high = composite[composite["Hub_Class"] == "High_Hub"].copy()
            if len(high) > 0:
                high = high.sort_values("Hub_Score", ascending=False).reset_index(drop=True)
                out = self.analysis_dir / f"HUB_GENES_High_Top{self.hub_percentile}Percent.csv"
                high.to_csv(out, index=False)
                saved.append(out)
                print(f"? Saved: {out.name} ({len(high):,} rows)")

            mod = composite[composite["Hub_Class"] == "Moderate_Hub"].copy()
            if len(mod) > 0:
                mod = mod.sort_values("Hub_Score", ascending=False).reset_index(drop=True)
                out = self.analysis_dir / "HUB_GENES_Moderate_Top25Percent.csv"
                mod.to_csv(out, index=False)
                saved.append(out)
                print(f"? Saved: {out.name} ({len(mod):,} rows)")

            # ? FIX 3: do NOT overwrite the filtered subset
            with_hub = composite[composite["Hub_Score"] > 0].copy()
            with_hub = with_hub.sort_values("Hub_Score", ascending=False).reset_index(drop=True)
            out = self.analysis_dir / "ALL_GENES_WITH_HUB_SCORES.csv"
            with_hub.to_csv(out, index=False)
            saved.append(out)
            print(f"? Saved: {out.name} ({len(with_hub):,} rows)")

        return saved

    # ---------------------------------------------------------------------------------
    # Main run
    # ---------------------------------------------------------------------------------
    def run(self):
        ranked = self.load_ranked_files()
        composite = ranked.get("composite")
        if composite is None or len(composite) == 0:
            print("? Composite ranking missing/empty.")
            sys.exit(1)

        composite_resolved = self.resolve_symbols_for_dataframe(composite)

        pref_syms = composite_resolved["Symbol_Preferred"].astype(str).map(lambda x: x.strip()).tolist()
        pref_syms = [_clean_sym(x) for x in pref_syms if x and str(x).strip() and _clean_sym(x) not in ["N/A", "NA", "NONE"]]
        pref_syms = sorted(list(set(pref_syms)))

        print(f"\n?? Unique preferred symbols to query in STRING: {len(pref_syms):,}")

        hub_info, source = self.load_cache()
        if hub_info is None:
            edges = self.query_string_network(pref_syms)
            if not edges:
                print("?? No STRING edges ? hub metrics unavailable. Writing hub columns as zeros.")
                hub_info = {}
                edges = []
            else:
                hub_info = self.calculate_hub_metrics(edges)

            self.save_cache(hub_info, edges)
            source = "string_api"

        print(f"\n? Hub source: {source}")

        ranked_with_hub = {}
        for view, df in ranked.items():
            print(f"\n? Adding hub metrics to view: {view} ({len(df):,} rows)")
            ranked_with_hub[view] = self.add_hub_columns(df, hub_info)

        saved = self.save_results(ranked_with_hub)

        print("\n" + "=" * 120)
        print("?? HUB ANALYSIS COMPLETE (NO GENE LOSS)")
        print("=" * 120)
        for i, f in enumerate(saved, 1):
            print(f"  {i}. {f.name}")
        print("=" * 120)


def main():
    p = argparse.ArgumentParser(
        description="Hub Gene Analysis (Fixed symbol integration, no gene loss)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("phenotype", help="Phenotype name (folder)")
    p.add_argument("--force-string", action="store_true", help="Ignore cache and re-query STRING")
    p.add_argument("--hub-percentile", type=int, default=10, help="Percentile cutoff for High_Hub (default 10)")
    p.add_argument("--confidence", type=float, default=0.4, help="STRING confidence (0-1), default 0.4")
    p.add_argument("--no-mygene", action="store_true", help="Disable MyGene Ensembl->Symbol resolution")
    p.add_argument("--no-string-ids", action="store_true", help="Disable STRING get_string_ids preferred mapping")

    args = p.parse_args()

    if args.hub_percentile < 1 or args.hub_percentile > 50:
        print("? hub-percentile must be between 1 and 50")
        sys.exit(1)
    if args.confidence <= 0 or args.confidence > 1:
        print("? confidence must be in (0, 1]")
        sys.exit(1)

    analyzer = HubGeneAnalyzer(
        phenotype=args.phenotype,
        force_string=args.force_string,
        hub_percentile=args.hub_percentile,
        confidence=args.confidence,
        resolve_with_mygene=(not args.no_mygene),
        resolve_with_string_ids=(not args.no_string_ids),
    )
    analyzer.run()


if __name__ == "__main__":
    main()
