#!/usr/bin/env python3
"""
PROTEIN STRUCTURE DOWNLOADER (UPDATED - SYMBOL/ALIAS SAFE)
=========================================================
Fixes the "name issue" by:
  - Resolving Symbol -> Preferred symbol (Symbol_Preferred) if available
  - Optional fallback: Analysis/symbol_resolution_cache.csv (Symbol, Symbol_Resolved, Symbol_Preferred)
  - Using stable cache keys (ENSEMBL_ID preferred; else Preferred Symbol)
  - Using Preferred Symbol for UniProt lookup & folder naming

Priority:
  1) PDB (experimental) if available
  2) AlphaFold (predicted) if no PDB

Usage:
  python predict4.4.6-DownloadProteinsRelatedToSignifcantGenesFinal_FIXED.py migraine
  python predict4.4.6-DownloadProteinsRelatedToSignifcantGenesFinal_FIXED.py migraine --workers 8
  python predict4.4.6-DownloadProteinsRelatedToSignifcantGenesFinal_FIXED.py migraine --reset
"""

import requests
import pandas as pd
import time
import argparse
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Any
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import json
import sys
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

warnings.filterwarnings("ignore")


def clean_sym(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", ""}:
        return ""
    return s


def clean_ensembl(x: Any) -> str:
    s = clean_sym(x)
    if not s:
        return ""
    # Remove version suffix ENSG... .12
    return s.split(".")[0]


def safe_dirname(s: str) -> str:
    # Keep it filesystem-safe
    s = clean_sym(s)
    for ch in ["/", "\\", " ", "\t", ":", ";", "|"]:
        s = s.replace(ch, "_")
    return s


class ProteinStructureDownloader:
    """Protein structure downloader - PDB priority"""

    @staticmethod
    def get_session():
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    @staticmethod
    def get_uniprot_from_gene(gene_symbol: str, ensembl_id: str) -> Optional[str]:
        """
        Get UniProt accession.
        Uses:
          1) UniProt query gene:<symbol> + organism
          2) UniProt query xref:ensembl-<ENSG>
          3) MyGene fallback
        """
        session = ProteinStructureDownloader.get_session()
        gene_symbol = clean_sym(gene_symbol)
        ensembl_id = clean_ensembl(ensembl_id)

        # Strategy 1: Direct gene symbol
        if gene_symbol:
            try:
                url = (
                    "https://rest.uniprot.org/uniprotkb/search"
                    f"?query=gene:{gene_symbol}+AND+organism_id:9606"
                    "&format=json&size=1&fields=accession"
                )
                r = session.get(url, timeout=12)
                if r.status_code == 200:
                    data = r.json()
                    if data.get("results"):
                        return data["results"][0]["primaryAccession"]
            except:
                pass

        # Strategy 2: Ensembl mapping
        if ensembl_id:
            try:
                url = (
                    "https://rest.uniprot.org/uniprotkb/search"
                    f"?query=xref:ensembl-{ensembl_id}+AND+organism_id:9606"
                    "&format=json&size=1&fields=accession"
                )
                r = session.get(url, timeout=12)
                if r.status_code == 200:
                    data = r.json()
                    if data.get("results"):
                        return data["results"][0]["primaryAccession"]
            except:
                pass

        # Strategy 3: MyGene fallback
        if gene_symbol:
            try:
                url = (
                    "https://mygene.info/v3/query"
                    f"?q=symbol:{gene_symbol}+AND+taxid:9606&fields=uniprot&size=1"
                )
                r = session.get(url, timeout=12)
                if r.status_code == 200:
                    data = r.json()
                    hits = data.get("hits", [])
                    if hits:
                        hit = hits[0]
                        up = hit.get("uniprot")
                        if isinstance(up, dict):
                            if "Swiss-Prot" in up:
                                return up["Swiss-Prot"]
                            if "TrEMBL" in up:
                                tr = up["TrEMBL"]
                                if isinstance(tr, list) and tr:
                                    return tr[0]
                                if isinstance(tr, str):
                                    return tr
                        elif isinstance(up, str):
                            return up
            except:
                pass

        return None

    @staticmethod
    def search_pdb_fixed(uniprot_id: str) -> List[str]:
        """Search PDB with multiple strategies; returns up to 10 IDs."""
        if not uniprot_id or str(uniprot_id).lower() == "nan":
            return []

        session = ProteinStructureDownloader.get_session()
        pdb_ids: List[str] = []

        # Strategy 1: EBI Proteins API dbReferences
        try:
            url = f"https://www.ebi.ac.uk/proteins/api/proteins/{uniprot_id}"
            r = session.get(url, timeout=18)
            if r.status_code == 200:
                data = r.json()
                for ref in data.get("dbReferences", []):
                    if ref.get("type") == "PDB":
                        pid = ref.get("id")
                        if pid and pid not in pdb_ids:
                            pdb_ids.append(pid)
                if pdb_ids:
                    return pdb_ids[:10]
        except:
            pass

        # Strategy 2: RCSB GraphQL
        try:
            graphql_query = {
                "query": f"""
                {{
                  entries(input: {{
                    type: SIMPLE_SEARCH,
                    query: {{
                      type: TERMINAL,
                      service: TEXT,
                      parameters: {{
                        attribute: "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                        operator: EXACT_MATCH,
                        value: "{uniprot_id}"
                      }}
                    }}
                  }}) {{
                    identifiers
                  }}
                }}
                """
            }
            r = session.post(
                "https://data.rcsb.org/graphql",
                json=graphql_query,
                headers={"Content-Type": "application/json"},
                timeout=18
            )
            if r.status_code == 200:
                data = r.json()
                ids = (((data.get("data") or {}).get("entries")) or {}).get("identifiers", []) or []
                for pid in ids:
                    if pid and pid not in pdb_ids:
                        pdb_ids.append(pid)
                if pdb_ids:
                    return pdb_ids[:10]
        except:
            pass

        # Strategy 3: RCSB REST search fallback
        try:
            url = (
                "https://search.rcsb.org/rcsbsearch/v2/query?json="
                f"{{\"query\":{{\"type\":\"terminal\",\"service\":\"text\",\"parameters\":{{"
                f"\"attribute\":\"rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession\","
                f"\"operator\":\"exact_match\",\"value\":\"{uniprot_id}\"}}}},"
                "\"return_type\":\"entry\",\"request_options\":{\"paginate\":{\"start\":0,\"rows\":10}}}}"
            )
            r = session.get(url, timeout=18)
            if r.status_code == 200:
                data = r.json()
                for hit in data.get("result_set", []):
                    pid = hit.get("identifier")
                    if pid and pid not in pdb_ids:
                        pdb_ids.append(pid)
        except:
            pass

        return pdb_ids[:10] if pdb_ids else []

    @staticmethod
    def get_pdb_info_fast(pdb_id: str) -> Dict[str, Any]:
        info = {"resolution": None, "method": None}
        session = ProteinStructureDownloader.get_session()
        try:
            url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
            r = session.get(url, timeout=12)
            if r.status_code == 200:
                data = r.json()
                exptl = data.get("exptl") or []
                if exptl:
                    info["method"] = exptl[0].get("method", "Unknown")
                entry_info = data.get("rcsb_entry_info") or {}
                res_list = entry_info.get("resolution_combined") or []
                if res_list:
                    info["resolution"] = res_list[0]
        except:
            pass
        return info

    @staticmethod
    def download_pdb_fast(pdb_id: str, gene_dir: Path) -> Optional[str]:
        output_file = gene_dir / f"{pdb_id}.pdb"
        if output_file.exists() and output_file.stat().st_size > 100:
            return str(output_file)

        session = ProteinStructureDownloader.get_session()
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        try:
            r = session.get(url, timeout=35, stream=True)
            if r.status_code == 200:
                with open(output_file, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                if output_file.stat().st_size > 100:
                    return str(output_file)
                output_file.unlink(missing_ok=True)
        except:
            output_file.unlink(missing_ok=True)
        return None

    @staticmethod
    def download_alphafold_fast(uniprot_id: str, gene_dir: Path) -> Optional[Tuple[str, Optional[float]]]:
        session = ProteinStructureDownloader.get_session()
        for version in ["v6", "v4"]:
            output_file = gene_dir / f"AF-{uniprot_id}-F1-model_{version}.pdb"
            if output_file.exists() and output_file.stat().st_size > 100:
                plddt = ProteinStructureDownloader.get_alphafold_plddt_fast(uniprot_id, version)
                return str(output_file), plddt

            url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_{version}.pdb"
            try:
                r = session.get(url, timeout=35, stream=True)
                if r.status_code == 200:
                    with open(output_file, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                    if output_file.stat().st_size > 100:
                        plddt = ProteinStructureDownloader.get_alphafold_plddt_fast(uniprot_id, version)
                        return str(output_file), plddt
                    output_file.unlink(missing_ok=True)
            except:
                output_file.unlink(missing_ok=True)

        return None

    @staticmethod
    def get_alphafold_plddt_fast(uniprot_id: str, version: str = "v6") -> Optional[float]:
        session = ProteinStructureDownloader.get_session()
        try:
            url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-confidence_{version}.json"
            r = session.get(url, timeout=12)
            if r.status_code == 200:
                data = r.json()
                scores = data.get("confidenceScore") or []
                if scores:
                    return round(sum(scores) / len(scores), 2)
        except:
            pass
        return None


def load_progress(progress_file: Path) -> Dict[str, Any]:
    if not progress_file.exists():
        return {"processed": {}, "timestamp": None}
    try:
        with open(progress_file, "r") as f:
            return json.load(f)
    except:
        return {"processed": {}, "timestamp": None}


def save_progress(progress_file: Path, progress_data: Dict[str, Any]):
    try:
        progress_data["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(progress_file, "w") as f:
            json.dump(progress_data, f, indent=2)
    except Exception as e:
        print(f"⚠️  Warning: Could not save progress: {e}")


def load_symbol_resolution_cache(base_dir: Path) -> Dict[str, str]:
    """
    Optional: Analysis/symbol_resolution_cache.csv with columns:
      Symbol, Symbol_Resolved, Symbol_Preferred
    Returns mapping Symbol -> Symbol_Preferred
    """
    cache = base_dir / "symbol_resolution_cache.csv"
    mapping: Dict[str, str] = {}
    if not cache.exists():
        return mapping

    try:
        df = pd.read_csv(cache)
        if {"Symbol", "Symbol_Preferred"}.issubset(df.columns):
            for s, p in zip(df["Symbol"].astype(str), df["Symbol_Preferred"].astype(str)):
                s2 = clean_sym(s)
                p2 = clean_sym(p)
                if s2 and p2:
                    mapping[s2] = p2
    except:
        pass
    return mapping


def resolve_preferred_symbol(row: Dict[str, Any], sym_map: Dict[str, str]) -> Tuple[str, str]:
    """
    Return (original_symbol, preferred_symbol)
    Priority:
      1) row['Symbol_Preferred'] if present/non-empty
      2) sym_map[Symbol] from symbol_resolution_cache.csv
      3) row['Symbol']
    """
    orig = clean_sym(row.get("Symbol", row.get("Gene_Symbol", "UNKNOWN")))
    pref = clean_sym(row.get("Symbol_Preferred", ""))
    if not pref and orig and orig in sym_map:
        pref = sym_map[orig]
    if not pref:
        pref = orig
    return orig, pref


def stable_cache_key(row: Dict[str, Any], preferred_symbol: str) -> str:
    """
    Cache key:
      - prefer ENSG (stable)
      - else preferred symbol
    """
    ens = clean_ensembl(row.get("Gene", row.get("ENSEMBL_ID", "")))
    if ens:
        return f"ENSG:{ens}"
    return f"SYM:{clean_sym(preferred_symbol).upper()}"


def process_single_gene(args: Tuple) -> Dict[str, Any]:
    """
    Worker process:
      - Uses Preferred Symbol for UniProt lookup & directory naming
      - Uses stable cache key to avoid alias duplication
    """
    row_dict, structures_dir, idx, total, progress_data, sym_map = args

    orig_sym, pref_sym = resolve_preferred_symbol(row_dict, sym_map)
    ensembl = clean_ensembl(row_dict.get("Gene", row_dict.get("ENSEMBL_ID", "")))
    cache_key = stable_cache_key(row_dict, pref_sym)

    result: Dict[str, Any] = {
        "Cache_Key": cache_key,
        "Symbol_Original": orig_sym,
        "Symbol_Preferred": pref_sym,
        "ENSEMBL_ID": ensembl,
        "UniProt_ID": None,
        "Has_PDB": False,
        "Num_PDB": 0,
        "PDB_IDs": None,
        "Best_PDB_ID": None,
        "PDB_Resolution": None,
        "PDB_Method": None,
        "PDB_Files": None,
        "Has_AlphaFold": False,
        "AlphaFold_pLDDT": None,
        "AlphaFold_Quality": None,
        "AlphaFold_File": None,
        "Structure_Source": None,
        "Structure_Files": None,
        "Gene_Directory": None,
        "Processing_Status": "Not Processed",
    }

    # Cache check
    cached = progress_data["processed"].get(cache_key)
    if cached:
        # Only reuse cache if directory still exists or it had a terminal status
        gd = cached.get("Gene_Directory")
        if gd and Path(gd).exists():
            result.update(cached)
            print(f"[{idx+1}/{total}] {pref_sym:<15} ✓ Cached ({cached.get('Processing_Status','Cached')})")
            return result

    # UniProt lookup uses Preferred Symbol (alias-safe), with Ensembl fallback
    uniprot = ProteinStructureDownloader.get_uniprot_from_gene(pref_sym, ensembl)
    if not uniprot:
        # Try original symbol as fallback (sometimes preferred is weird)
        if orig_sym and orig_sym != pref_sym:
            uniprot = ProteinStructureDownloader.get_uniprot_from_gene(orig_sym, ensembl)

    if not uniprot:
        result["Processing_Status"] = "No UniProt"
        print(f"[{idx+1}/{total}] {pref_sym:<15} ✗ No UniProt")
        return result

    result["UniProt_ID"] = uniprot

    # Directory name based on preferred symbol; keep original in metadata
    dir_name = safe_dirname(pref_sym)
    gene_dir = Path(structures_dir) / dir_name
    gene_dir.mkdir(parents=True, exist_ok=True)

    pdb_ids = ProteinStructureDownloader.search_pdb_fixed(uniprot)
    pdb_files: List[str] = []

    if pdb_ids:
        result["Has_PDB"] = True
        result["Num_PDB"] = len(pdb_ids)
        result["PDB_IDs"] = "|".join(pdb_ids[:10])
        best_pdb = pdb_ids[0]
        result["Best_PDB_ID"] = best_pdb

        pdb_info = ProteinStructureDownloader.get_pdb_info_fast(best_pdb)
        result["PDB_Resolution"] = pdb_info.get("resolution")
        result["PDB_Method"] = pdb_info.get("method")

        with ThreadPoolExecutor(max_workers=3) as ex:
            futs = [ex.submit(ProteinStructureDownloader.download_pdb_fast, pid, gene_dir) for pid in pdb_ids[:3]]
            for fut in as_completed(futs):
                try:
                    fp = fut.result()
                    if fp:
                        pdb_files.append(fp)
                except:
                    pass

        if pdb_files:
            result["PDB_Files"] = "|".join(pdb_files)
            result["Structure_Source"] = "PDB"
            result["Structure_Files"] = result["PDB_Files"]
            result["Processing_Status"] = "Success"
        else:
            result["Processing_Status"] = "PDB Download Failed"

    else:
        af = ProteinStructureDownloader.download_alphafold_fast(uniprot, gene_dir)
        if af:
            af_file, plddt = af
            result["Has_AlphaFold"] = True
            result["AlphaFold_File"] = af_file
            result["AlphaFold_pLDDT"] = plddt

            if plddt is not None:
                if plddt >= 90:
                    result["AlphaFold_Quality"] = "Very High"
                elif plddt >= 70:
                    result["AlphaFold_Quality"] = "High"
                elif plddt >= 50:
                    result["AlphaFold_Quality"] = "Low"
                else:
                    result["AlphaFold_Quality"] = "Very Low"

            result["Structure_Source"] = "AlphaFold"
            result["Structure_Files"] = af_file
            result["Processing_Status"] = "Success"
        else:
            result["Processing_Status"] = "No Structures Available"

    # Keep directory only if there are files
    if gene_dir.exists():
        pdbs = list(gene_dir.glob("*.pdb"))
        if not pdbs:
            # leave folder removal attempt, but do NOT treat as crash
            try:
                gene_dir.rmdir()
            except:
                pass
            result["Gene_Directory"] = None
            if result["Processing_Status"] == "Success":
                result["Processing_Status"] = "No Structures Available"
        else:
            result["Gene_Directory"] = str(gene_dir)

    # Status line
    parts = [f"UP:{uniprot[:6]}"]
    if result["Has_PDB"]:
        res = f"{result['PDB_Resolution']:.2f}Å" if isinstance(result["PDB_Resolution"], (float, int)) else "N/A"
        parts.append(f"PDB:{len(pdb_ids)}({res})")
        parts.append("✅ PDB")
    elif result["Has_AlphaFold"]:
        p = result["AlphaFold_pLDDT"]
        ptxt = f"{p:.0f}" if isinstance(p, (float, int)) else "OK"
        parts.append(f"PDB:0 AF:✓({ptxt})")
        parts.append("✅ AlphaFold")
    else:
        parts.append("❌ None")

    tag = f"{pref_sym}"
    if orig_sym and orig_sym != pref_sym:
        tag += f" (orig:{orig_sym})"

    print(f"[{idx+1}/{total}] {tag:<28} {' '.join(parts)}")
    return result


def download_structures(phenotype: str, num_workers: Optional[int] = None, debug: bool = False):
    base_dir = Path(phenotype) / "GeneDifferentialExpression" / "Files" / "UltimateCompleteRankingAnalysis"
    structures_dir = base_dir / "protein_structures"
    output_file = base_dir / f"{phenotype}_genes_with_structures.csv"
    progress_file = base_dir / "structures_progress.json"

    print("=" * 120)
    print("🧬 PROTEIN STRUCTURE DOWNLOADER (ALIAS SAFE)")
    print("=" * 120)
    print(f"📋 Phenotype: {phenotype}")
    print(f"📁 Output folder: {structures_dir}")
    print(f"📄 Output table: {output_file.name}")
    print("⚡ Priority: PDB > AlphaFold")
    print("=" * 120)

    ranked_file = base_dir / "RANKED_WITH_HUB_composite.csv"
    if not ranked_file.exists():
        ranked_file = base_dir / "RANKED_composite.csv"
    if not ranked_file.exists():
        print("❌ Error: Gene ranking file not found!")
        return None

    genes_df = pd.read_csv(ranked_file)
    print(f"📊 Input rows: {len(genes_df):,}  (from {ranked_file.name})")

    # Load optional alias map
    sym_map = load_symbol_resolution_cache(base_dir)
    if sym_map:
        print(f"✅ Loaded symbol_resolution_cache.csv mappings: {len(sym_map):,}")
    else:
        print("ℹ️ No symbol_resolution_cache.csv found (still uses Symbol_Preferred if in CSV)")

    # Ensure Symbol column exists; do NOT drop rows, just handle invalid later
    if "Symbol" not in genes_df.columns:
        genes_df["Symbol"] = ""

    structures_dir.mkdir(parents=True, exist_ok=True)

    progress_data = load_progress(progress_file)
    cached_count = len(progress_data.get("processed", {}))
    if cached_count:
        print(f"📂 Cached entries: {cached_count:,}")

    if num_workers is None:
        num_workers = min(cpu_count(), 8)
    num_workers = max(1, int(num_workers))
    print(f"⚙️ Workers: {num_workers}")

    # Build args
    row_dicts = genes_df.to_dict("records")
    args_list = [
        (row_dict, structures_dir, idx, len(row_dicts), progress_data, sym_map)
        for idx, row_dict in enumerate(row_dicts)
    ]

    start = time.time()
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_single_gene, args_list)
        pool.close()
        pool.join()
    elapsed = time.time() - start

    # Update cache keyed by stable cache key
    for res in results:
        progress_data.setdefault("processed", {})
        progress_data["processed"][res["Cache_Key"]] = res
    save_progress(progress_file, progress_data)

    results_df = pd.DataFrame(results)

    # Merge back without losing rows
    merged_df = genes_df.copy()
    # Create preferred/original columns on the left side for visibility
    merged_df["Symbol_Original"] = merged_df["Symbol"].astype(str).map(lambda s: clean_sym(s))
    merged_df["Symbol_Preferred"] = merged_df.apply(
        lambda r: resolve_preferred_symbol(r.to_dict(), sym_map)[1], axis=1
    )
    merged_df["ENSEMBL_ID"] = merged_df.get("Gene", merged_df.get("ENSEMBL_ID", "")).astype(str).map(clean_ensembl)
    merged_df["Cache_Key"] = merged_df.apply(lambda r: stable_cache_key(r.to_dict(), r["Symbol_Preferred"]), axis=1)

    merged_df = merged_df.merge(
        results_df,
        on="Cache_Key",
        how="left",
        suffixes=("", "_struct")
    )

    merged_df.to_csv(output_file, index=False)

    # Summary
    total = len(merged_df)
    print("\n" + "=" * 120)
    print("📊 SUMMARY")
    print("=" * 120)
    print(f"⏱️ Time: {elapsed/60:.1f} min  ({elapsed/max(total,1):.2f} sec/row)")

    if "Processing_Status" in merged_df.columns:
        vc = merged_df["Processing_Status"].value_counts(dropna=False)
        print("\n📈 Status:")
        for status, count in vc.items():
            print(f"   {status}: {count:,} ({count/total*100:.1f}%)")

    has_uniprot = merged_df["UniProt_ID"].notna().sum() if "UniProt_ID" in merged_df.columns else 0
    has_pdb = (merged_df["Has_PDB"] == True).sum() if "Has_PDB" in merged_df.columns else 0
    has_af = (merged_df["Has_AlphaFold"] == True).sum() if "Has_AlphaFold" in merged_df.columns else 0
    has_any = merged_df["Structure_Source"].notna().sum() if "Structure_Source" in merged_df.columns else 0

    print("\n🧬 Structures:")
    print(f"   UniProt: {has_uniprot:,} ({has_uniprot/total*100:.1f}%)")
    print(f"   PDB:     {has_pdb:,} ({has_pdb/total*100:.1f}%)")
    print(f"   AF:      {has_af:,} ({has_af/total*100:.1f}%)")
    print(f"   Any:     {has_any:,} ({has_any/total*100:.1f}%)")

    print("\n" + "=" * 120)
    print("✅ COMPLETE!")
    print(f"📄 Saved: {output_file}")
    print(f"📁 Structures: {structures_dir}")
    print("=" * 120)

    return merged_df


def main():
    parser = argparse.ArgumentParser(
        description="🧬 Protein Structure Downloader (Alias Safe)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("phenotype", help="Phenotype name")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker processes (default: 8)")
    parser.add_argument("--reset", action="store_true", help="Clear structures_progress.json cache")
    parser.add_argument("--debug", action="store_true", help="Reserved (not used)")
    args = parser.parse_args()

    if args.reset:
        progress_file = Path(args.phenotype) / "GeneDifferentialExpression" / "Files" / "UltimateCompleteRankingAnalysis" / "structures_progress.json"
        if progress_file.exists():
            progress_file.unlink()
            print("✅ Cleared cache (structures_progress.json)")

    try:
        out = download_structures(args.phenotype, args.workers, args.debug)
        sys.exit(0 if out is not None else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
