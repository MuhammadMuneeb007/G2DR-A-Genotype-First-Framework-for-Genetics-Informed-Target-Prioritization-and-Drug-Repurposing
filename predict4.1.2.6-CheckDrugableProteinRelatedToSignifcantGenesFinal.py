#!/usr/bin/env python3
"""
COMPREHENSIVE DRUGGABILITY ANALYSIS - FIXED (ALL GENES + ALIAS-AWARE)
====================================================================
Fixes vs your current version:
  ? DOES NOT drop genes without structures (keeps ALL genes)
  ? Saves TWO outputs:
        1) <phenotype>_druggability_all.csv        (ALL rows, nothing removed)
        2) <phenotype>_druggability_complete.csv   (success-only, optional convenience)
  ? Drug querying checks OTHER FORMS (aliases) of the gene:
        - tries Symbol_Preferred / Symbol_Resolved / Symbol / Symbol_Original (if present)
        - DGIdb query uses MULTIPLE search terms at once
        - ChEMBL tries UniProt accession first (if available), then symbol search
  ? Fixes a major multiprocessing/resume bug:
        - progress_data is NOT mutated inside worker processes (that never persisted reliably)
        - fpocket results are merged back in the parent and then saved to progress JSON
  ? Adds clearer statuses:
        - DrugQuery_Status
        - Fpocket_Status
        - Processing_Status (always present; NEVER used to filter the ALL file)

Usage:
  python druggability_analyzer_complete_fixed.py migraine
  python druggability_analyzer_complete_fixed.py migraine --workers 4

Expected input:
  /data/ascher02/uqmmune1/ANNOVAR/<phenotype>/GeneDifferentialExpression/Files/UltimateCompleteRankingAnalysis/<phenotype>_genes_with_structures.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import subprocess
import requests
import time
import json
import sys
import re
import shutil
from typing import Dict, List, Optional, Tuple
from multiprocessing import Pool, cpu_count
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


# ----------------------------
# Helpers
# ----------------------------
def _safe_str(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", ""}:
        return ""
    return s


def _unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for it in items:
        it = _safe_str(it)
        if not it:
            continue
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


# ----------------------------
# Worker function (must be top-level for multiprocessing)
# ----------------------------
def fpocket_worker(payload: Dict) -> Dict:
    """
    Runs fpocket for ONE gene (if needed) and returns fpocket-derived metrics.
    IMPORTANT: does NOT write progress JSON directly (parent handles that).
    """
    analyzer_state = payload["analyzer_state"]
    row = payload["row"]
    idx = payload["idx"]
    total = payload["total"]

    phenotype = analyzer_state["phenotype"]
    fpocket_dir = Path(analyzer_state["fpocket_dir"])
    error_log_file = Path(analyzer_state["error_log_file"])

    gene = _safe_str(row.get("Symbol_Query", row.get("Symbol", "")))
    has_drugs = bool(row.get("Has_Drugs", False))
    gene_dir_path = _safe_str(row.get("Gene_Directory", ""))

    def log_error(step: str, error: str):
        try:
            with open(error_log_file, "a") as f:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{ts}] {gene} - {step}: {error}\n")
        except Exception:
            pass

    def get_all_isoforms(gene_dir: Path) -> List[Path]:
        if not gene_dir.exists():
            return []
        return list(gene_dir.glob("*.pdb"))

    def calculate_druggability_probability(pocket_data: Dict) -> float:
        score = 0.0
        weights_sum = 0.0

        if pocket_data.get("score") is not None:
            normalized = min(float(pocket_data["score"]), 1.0)
            score += normalized * 0.35
            weights_sum += 0.35

        if pocket_data.get("volume") is not None:
            vol = float(pocket_data["volume"])
            if 500 <= vol <= 2000:
                vol_score = 1.0
            elif vol < 500:
                vol_score = vol / 500.0
            else:
                vol_score = max(0.3, 2000.0 / vol)
            score += vol_score * 0.25
            weights_sum += 0.25

        if pocket_data.get("hydrophobicity") is not None:
            hydro = float(pocket_data["hydrophobicity"])
            if 0.3 <= hydro <= 0.7:
                hydro_score = 1.0
            elif hydro < 0.3:
                hydro_score = max(0.5, hydro / 0.3)
            else:
                hydro_score = max(0.3, (1.0 - hydro) / 0.5)
            score += hydro_score * 0.20
            weights_sum += 0.20

        if pocket_data.get("alpha_sphere_density") is not None:
            density = float(pocket_data["alpha_sphere_density"])
            density_score = min(density / 15.0, 1.0)
            score += density_score * 0.10
            weights_sum += 0.10

        if pocket_data.get("polarity") is not None:
            pol = float(pocket_data["polarity"])
            if 0.2 <= pol <= 0.6:
                pol_score = 1.0
            else:
                pol_score = 0.6
            score += pol_score * 0.10
            weights_sum += 0.10

        return (score / weights_sum) if weights_sum > 0 else 0.0

    def parse_fpocket_output(info_file: Path) -> Optional[Dict]:
        try:
            content = info_file.read_text()
            pockets = []
            pocket_sections = re.split(r"Pocket\s+\d+\s*:", content)

            for section in pocket_sections[1:]:
                pocket_data = {}

                m = re.search(r"Druggability Score\s*:\s*([\d.]+)", section)
                if m:
                    pocket_data["score"] = float(m.group(1))

                m = re.search(r"Volume\s*:\s*([\d.]+)", section)
                if m:
                    pocket_data["volume"] = float(m.group(1))

                m = re.search(r"Hydrophobicity score\s*:\s*([\d.]+)", section)
                if m:
                    pocket_data["hydrophobicity"] = float(m.group(1))

                m = re.search(r"Polarity score\s*:\s*([\d.]+)", section)
                if m:
                    pocket_data["polarity"] = float(m.group(1))

                m = re.search(r"Alpha sphere density\s*:\s*([\d.]+)", section)
                if m:
                    pocket_data["alpha_sphere_density"] = float(m.group(1))

                if pocket_data:
                    pocket_data["druggability_probability"] = calculate_druggability_probability(pocket_data)
                    pockets.append(pocket_data)

            if pockets:
                return {
                    "num_pockets": len(pockets),
                    "pockets": pockets,
                    "best_pocket": max(pockets, key=lambda x: x.get("druggability_probability", 0)),
                }
        except Exception as e:
            log_error("FpocketParse", str(e))
        return None

    def run_fpocket_with_logging(pdb_file: Path) -> Optional[Dict]:
        if not pdb_file.exists():
            log_error("Fpocket", f"PDB not found: {pdb_file}")
            return None

        output_dir = fpocket_dir / pdb_file.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        temp_pdb = output_dir / pdb_file.name
        try:
            shutil.copy(pdb_file, temp_pdb)
        except Exception as e:
            log_error("FpocketCopy", str(e))
            return None

        try:
            cmd = ["fpocket", "-f", temp_pdb.name]
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=120,
                cwd=str(output_dir),
            )

            if result.returncode != 0:
                err = result.stderr.decode(errors="ignore")[:400]
                log_error("Fpocket", f"Exit {result.returncode}: {err}")
                return None

            expected_output = output_dir / f"{temp_pdb.stem}_out"
            info_file = expected_output / f"{temp_pdb.stem}_info.txt"
            if not info_file.exists():
                log_error("Fpocket", "Info file not created")
                return None

            return parse_fpocket_output(info_file)

        except subprocess.TimeoutExpired:
            log_error("Fpocket", "Timeout (>120s)")
            return None
        except Exception as e:
            log_error("Fpocket", str(e))
            return None

    # ----------------------------
    # Return record template (always returned)
    # ----------------------------
    record = {
        "Symbol": gene,
        "Has_Drugs": has_drugs,
        "Total_Drug_Count": int(row.get("Total_Drug_Count", 0) or 0),
        "Isoforms_Analyzed": 0,
        "Isoforms_Druggable": 0,
        "Max_Druggability_Probability": 0.0,
        "Mean_Druggability_Probability": 0.0,
        "Best_Pocket_Volume": np.nan,
        "Best_Pocket_Hydrophobicity": np.nan,
        "Best_Pocket_Score": np.nan,
        "Total_Pockets": 0,
        "Total_Druggable_Pockets": 0,
        "Druggability_Class": "Unknown",
        "Druggability_Probability": 0.0,
        "Fpocket_Status": "Not Run",
    }

    # If has drugs, no fpocket needed
    if has_drugs:
        record["Druggability_Class"] = "Druggable (Experimental)"
        record["Druggability_Probability"] = 1.0
        record["Max_Druggability_Probability"] = 1.0
        record["Mean_Druggability_Probability"] = 1.0
        record["Fpocket_Status"] = "Not Needed (Has Drugs)"
        return record

    # No structure directory => cannot run fpocket
    if not gene_dir_path:
        record["Druggability_Class"] = "Unknown (No Structure)"
        record["Druggability_Probability"] = 0.0
        record["Fpocket_Status"] = "No Structure"
        return record

    gene_dir = Path(gene_dir_path)
    if not gene_dir.exists():
        record["Druggability_Class"] = "Unknown (No Structure)"
        record["Druggability_Probability"] = 0.0
        record["Fpocket_Status"] = "Directory Missing"
        return record

    isoforms = get_all_isoforms(gene_dir)
    if not isoforms:
        record["Druggability_Class"] = "Unknown (No Structure)"
        record["Druggability_Probability"] = 0.0
        record["Fpocket_Status"] = "No Isoforms"
        return record

    record["Isoforms_Analyzed"] = len(isoforms)

    # Run fpocket on isoforms
    isoform_results = []
    for pdb in isoforms:
        r = run_fpocket_with_logging(pdb)
        if r:
            isoform_results.append(r)

    if not isoform_results:
        record["Druggability_Class"] = "Not Druggable (No Pockets)"
        record["Druggability_Probability"] = 0.1
        record["Max_Druggability_Probability"] = 0.1
        record["Mean_Druggability_Probability"] = 0.1
        record["Fpocket_Status"] = "Failed (No Pockets)"
        return record

    # Aggregate results
    probs = []
    druggable_isoforms = 0
    total_pockets = 0
    best_pocket = None
    best_prob = -1.0

    for iso in isoform_results:
        pocket = iso["best_pocket"]
        prob = float(pocket.get("druggability_probability", 0.0))
        probs.append(prob)
        total_pockets += int(iso.get("num_pockets", 0))
        if prob >= 0.5:
            druggable_isoforms += 1
        if prob > best_prob:
            best_prob = prob
            best_pocket = pocket

    record["Isoforms_Druggable"] = druggable_isoforms
    record["Total_Pockets"] = total_pockets
    record["Total_Druggable_Pockets"] = int(sum(1 for p in probs if p >= 0.5))
    record["Max_Druggability_Probability"] = float(round(max(probs), 3))
    record["Mean_Druggability_Probability"] = float(round(np.mean(probs), 3))
    record["Druggability_Probability"] = record["Max_Druggability_Probability"]
    record["Fpocket_Status"] = "Success"

    if best_pocket:
        record["Best_Pocket_Volume"] = round(float(best_pocket.get("volume", 0.0)), 1)
        record["Best_Pocket_Hydrophobicity"] = round(float(best_pocket.get("hydrophobicity", 0.0)), 3)
        record["Best_Pocket_Score"] = round(float(best_pocket.get("score", 0.0)), 3)

    prob = record["Max_Druggability_Probability"]
    if prob >= 0.7:
        record["Druggability_Class"] = "Potentially Druggable (High)"
    elif prob >= 0.5:
        record["Druggability_Class"] = "Potentially Druggable (Moderate)"
    elif prob >= 0.3:
        record["Druggability_Class"] = "Potentially Druggable (Low)"
    else:
        record["Druggability_Class"] = "Not Druggable (Poor Pockets)"

    return record


class DruggabilityAnalyzer:
    """Complete druggability analysis with resume capability (FIXED)"""

    def __init__(self, phenotype: str):
        self.phenotype = phenotype

        # Paths
        self.base_dir = Path("/data/ascher02/uqmmune1/ANNOVAR")
        self.analysis_dir = self.base_dir / phenotype / "GeneDifferentialExpression" / "Files" / "UltimateCompleteRankingAnalysis"
        self.structures_dir = self.analysis_dir / "protein_structures"
        self.druggability_dir = self.analysis_dir / "druggability_analysis"
        self.druggability_dir.mkdir(parents=True, exist_ok=True)

        self.fpocket_dir = self.druggability_dir / "fpocket_results"
        self.fpocket_dir.mkdir(parents=True, exist_ok=True)

        # Progress tracking files
        self.progress_file = self.druggability_dir / "analysis_progress.json"
        self.drug_db_file = self.druggability_dir / f"{phenotype}_drug_gene_database.csv"
        self.error_log_file = self.druggability_dir / "error_log.txt"

        print("=" * 120)
        print("? COMPREHENSIVE DRUGGABILITY ANALYSIS - FIXED (ALL GENES + ALIAS AWARE)")
        print("=" * 120)
        print(f"Phenotype: {phenotype}")
        print(f"Output dir: {self.druggability_dir}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 120)

    # ----------------------------
    # Progress
    # ----------------------------
    def load_progress(self) -> Dict:
        if self.progress_file.exists():
            try:
                return json.loads(self.progress_file.read_text())
            except Exception:
                pass
        return {"drugs_checked": {}, "fpocket_completed": {}, "last_updated": None}

    def save_progress(self, progress_data: Dict):
        progress_data["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            self.progress_file.write_text(json.dumps(progress_data, indent=2))
        except Exception as e:
            print(f"?? Warning: Could not save progress: {e}")

    def log_error(self, gene: str, step: str, error: str):
        try:
            with open(self.error_log_file, "a") as f:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{ts}] {gene} - {step}: {error}\n")
        except Exception:
            pass

    # ----------------------------
    # Load input
    # ----------------------------
    def load_genes_with_structures(self) -> Optional[pd.DataFrame]:
        """
        FIX: Keep ALL genes. Do NOT filter Structure_Files.
        We only clean/standardize symbol columns.
        """
        structure_file = self.analysis_dir / f"{self.phenotype}_genes_with_structures.csv"
        if not structure_file.exists():
            print(f"? Structure file not found: {structure_file}")
            return None

        df = pd.read_csv(structure_file)

        # Ensure essential columns exist
        for col in ["Symbol"]:
            if col not in df.columns:
                df[col] = ""

        # Clean symbol-like columns if present
        for col in ["Symbol", "Symbol_Preferred", "Symbol_Resolved", "Symbol_Original"]:
            if col in df.columns:
                df[col] = df[col].apply(_safe_str)

        # Build a query symbol that prefers preferred/resolved, then Symbol
        def choose_symbol(row) -> str:
            cands = []
            for col in ["Symbol_Preferred", "Symbol_Resolved", "Symbol", "Symbol_Original"]:
                if col in row:
                    cands.append(row[col])
            cands = _unique_preserve_order(cands)
            return cands[0] if cands else ""

        df["Symbol_Query"] = df.apply(choose_symbol, axis=1)

        # Keep ALL rows, but if Symbol_Query empty, keep row and mark later
        print(f"\n?? Loaded input rows (ALL genes kept): {len(df):,}")
        return df

    # ----------------------------
    # Alias handling
    # ----------------------------
    def build_aliases_for_row(self, row: pd.Series) -> List[str]:
        """
        Returns a prioritized list of candidate gene symbols to query.
        """
        candidates = []
        for col in ["Symbol_Preferred", "Symbol_Resolved", "Symbol", "Symbol_Original", "Symbol_Query"]:
            if col in row.index:
                candidates.append(row.get(col, ""))

        # If you ever have a Synonyms column like "A|B|C", include it
        if "Synonyms" in row.index:
            syn = _safe_str(row.get("Synonyms", ""))
            if syn:
                candidates.extend([s.strip() for s in syn.split("|")])

        return _unique_preserve_order(candidates)

    # ----------------------------
    # DGIdb (GraphQL) - multi-term query
    # ----------------------------
    def query_dgidb_graphql_multi(self, gene_symbols: List[str]) -> Dict:
        """
        Query DGIdb with multiple candidate symbols at once.
        """
        if not gene_symbols:
            return {"success": True, "drugs": [], "error": None}

        url = "https://dgidb.org/api/graphql"
        # escape double quotes in symbols
        safe_terms = [s.replace('"', '\\"') for s in gene_symbols]
        terms_str = ", ".join([f"\"{t}\"" for t in safe_terms])

        query = f"""
        query {{
          geneMatches(searchTerms: [{terms_str}]) {{
            directMatches {{
              matches {{
                name
                interactions {{
                  drug {{
                    name
                    conceptId
                    approved
                  }}
                  interactionTypes {{
                    type
                    directionality
                  }}
                  publications {{
                    pmid
                  }}
                  sources {{
                    sourceDbName
                  }}
                  interactionScore
                }}
              }}
            }}
          }}
        }}
        """

        try:
            resp = requests.post(url, json={"query": query}, timeout=20)
            if resp.status_code != 200:
                return {"success": False, "drugs": [], "error": f"HTTP {resp.status_code}"}

            data = resp.json()
            if "errors" in data:
                return {"success": False, "drugs": [], "error": str(data["errors"])}

            drugs = []
            gm = data.get("data", {}).get("geneMatches", {})
            direct = gm.get("directMatches", []) if gm else []
            if direct:
                matches = direct[0].get("matches", [])
                for m in matches:
                    matched_name = m.get("name")
                    interactions = m.get("interactions", []) or []
                    for inter in interactions:
                        drug = inter.get("drug", {}) or {}
                        itypes = inter.get("interactionTypes", []) or []
                        pubs = inter.get("publications", []) or []
                        srcs = inter.get("sources", []) or []
                        drugs.append(
                            {
                                "query_terms": "|".join(gene_symbols)[:500],
                                "matched_gene": matched_name,
                                "drug_name": drug.get("name"),
                                "drug_id": drug.get("conceptId"),
                                "approved": drug.get("approved", False),
                                "interaction_type": itypes[0].get("type") if itypes else "Unknown",
                                "directionality": itypes[0].get("directionality") if itypes else None,
                                "score": inter.get("interactionScore"),
                                "pmids": "|".join([p.get("pmid", "") for p in pubs if p.get("pmid")])[:500],
                                "sources": "|".join([s.get("sourceDbName", "") for s in srcs if s.get("sourceDbName")])[
                                    :500
                                ],
                                "database": "DGIdb",
                            }
                        )

            return {"success": True, "drugs": drugs, "error": None}

        except Exception as e:
            return {"success": False, "drugs": [], "error": str(e)}

    # ----------------------------
    # ChEMBL - UniProt first, then symbol
    # ----------------------------
    def chembl_find_target_by_uniprot(self, uniprot_id: str) -> Optional[str]:
        """
        Attempts to find a ChEMBL target using UniProt accession.
        """
        uniprot_id = _safe_str(uniprot_id)
        if not uniprot_id:
            return None

        try:
            url = "https://www.ebi.ac.uk/chembl/api/data/target.json"
            params = {"target_components__accession": uniprot_id, "limit": 5}
            r = requests.get(url, params=params, timeout=20)
            if r.status_code != 200:
                return None
            data = r.json()
            targets = data.get("targets", []) or []
            if not targets:
                return None
            return targets[0].get("target_chembl_id")
        except Exception:
            return None

    def chembl_find_target_by_symbol(self, gene_symbol: str) -> Optional[str]:
        gene_symbol = _safe_str(gene_symbol)
        if not gene_symbol:
            return None

        try:
            url = "https://www.ebi.ac.uk/chembl/api/data/target/search.json"
            params = {"q": gene_symbol, "limit": 5}
            r = requests.get(url, params=params, timeout=20)
            if r.status_code != 200:
                return None
            data = r.json()
            targets = data.get("targets", []) or []
            if not targets:
                return None
            return targets[0].get("target_chembl_id")
        except Exception:
            return None

    def query_chembl_for_target_drugs(self, target_id: str, gene_context: str) -> List[Dict]:
        """
        Fetch activities for a ChEMBL target and return unique molecule hits.
        """
        if not target_id:
            return []

        drugs = []
        seen = set()
        try:
            activities_url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
            params = {"target_chembl_id": target_id, "limit": 200}
            r = requests.get(activities_url, params=params, timeout=25)
            if r.status_code != 200:
                return []

            act = r.json()
            for a in (act.get("activities", []) or [])[:200]:
                mol = a.get("molecule_chembl_id")
                if not mol or mol in seen:
                    continue
                seen.add(mol)

                drugs.append(
                    {
                        "query_terms": gene_context[:500],
                        "matched_gene": gene_context.split("|")[0] if gene_context else None,
                        "drug_name": a.get("molecule_pref_name", mol),
                        "drug_id": mol,
                        "approved": None,
                        "interaction_type": a.get("standard_type", "Unknown"),
                        "directionality": None,
                        "score": a.get("standard_value"),
                        "pmids": a.get("document_chembl_id", ""),
                        "sources": f"ChEMBL Target: {target_id}",
                        "database": "ChEMBL",
                    }
                )
        except Exception:
            return []

        return drugs

    def query_chembl_multi(self, gene_symbols: List[str], uniprot_id: str) -> Dict:
        """
        Tries UniProt first; if not found, tries symbol candidates in order.
        """
        try:
            # 1) UniProt -> target
            target = self.chembl_find_target_by_uniprot(uniprot_id)
            if target:
                drugs = self.query_chembl_for_target_drugs(target, gene_context="|".join(gene_symbols + [uniprot_id]))
                return {"success": True, "drugs": drugs, "error": None}

            # 2) fall back to symbol search
            for sym in gene_symbols:
                t = self.chembl_find_target_by_symbol(sym)
                if t:
                    drugs = self.query_chembl_for_target_drugs(t, gene_context="|".join(gene_symbols))
                    return {"success": True, "drugs": drugs, "error": None}

            return {"success": True, "drugs": [], "error": None}

        except Exception as e:
            return {"success": False, "drugs": [], "error": str(e)}

    # ----------------------------
    # Drug check for a gene (alias-aware)
    # ----------------------------
    def check_drugs_for_gene(self, aliases: List[str], uniprot_id: str) -> Dict:
        """
        Check both DGIdb and ChEMBL using multiple aliases.
        """
        all_drugs = []

        # DGIdb (multi-term)
        dgidb_result = self.query_dgidb_graphql_multi(aliases)
        if dgidb_result["success"]:
            all_drugs.extend(dgidb_result["drugs"])
        else:
            self.log_error("|".join(aliases), "DGIdb", dgidb_result["error"])

        time.sleep(0.2)

        # ChEMBL (UniProt first, then aliases)
        chembl_result = self.query_chembl_multi(aliases, uniprot_id)
        if chembl_result["success"]:
            all_drugs.extend(chembl_result["drugs"])
        else:
            self.log_error("|".join(aliases), "ChEMBL", chembl_result["error"])

        # Deduplicate drugs across both DBs (by drug_id + drug_name)
        dedup = {}
        for d in all_drugs:
            key = (d.get("drug_id"), d.get("drug_name"))
            if key not in dedup:
                dedup[key] = d
        all_drugs = list(dedup.values())

        return {
            "has_drugs": len(all_drugs) > 0,
            "drug_count": len(all_drugs),
            "dgidb_count": sum(1 for d in all_drugs if d.get("database") == "DGIdb"),
            "chembl_count": sum(1 for d in all_drugs if d.get("database") == "ChEMBL"),
            "drugs": all_drugs,
        }

    def check_all_drugs(self, genes_df: pd.DataFrame, progress_data: Dict) -> Tuple[pd.DataFrame, Dict]:
        """
        Step 1: check DGIdb + ChEMBL with resume caching.
        FIX: cache key uses ENSEMBL_ID if available else Symbol_Query.
        """
        print("\n" + "=" * 120)
        print("STEP 1: CHECKING DRUG DATABASES (DGIdb + ChEMBL) [ALIAS-AWARE]")
        print("=" * 120)

        all_drug_records = []
        gene_summaries = []

        # Choose stable cache key
        def cache_key(row: pd.Series) -> str:
            if "ENSEMBL_ID" in row.index:
                k = _safe_str(row.get("ENSEMBL_ID"))
                if k:
                    return k
            return _safe_str(row.get("Symbol_Query", row.get("Symbol", "")))

        for idx, row in genes_df.iterrows():
            key = cache_key(row)
            aliases = self.build_aliases_for_row(row)
            uniprot = _safe_str(row.get("UniProt_ID", ""))

            # If no aliases, still keep row
            if not aliases:
                gene_summaries.append(
                    {
                        "Cache_Key": key,
                        "Symbol_Query": _safe_str(row.get("Symbol_Query", "")),
                        "Has_Drugs": False,
                        "Total_Drug_Count": 0,
                        "DGIdb_Count": 0,
                        "ChEMBL_Count": 0,
                        "DrugQuery_Status": "No Symbol",
                    }
                )
                continue

            # Resume cache
            if key in progress_data["drugs_checked"]:
                cached = progress_data["drugs_checked"][key]
                gene_summaries.append(cached["summary"])
                all_drug_records.extend(cached["drugs"])
                if (idx + 1) % 50 == 0:
                    print(f"   Progress: {idx+1}/{len(genes_df)} (cache)", end="\r")
                continue

            # Query DBs
            result = self.check_drugs_for_gene(aliases, uniprot)

            summary = {
                "Cache_Key": key,
                "Symbol_Query": aliases[0],
                "Has_Drugs": bool(result["has_drugs"]),
                "Total_Drug_Count": int(result["drug_count"]),
                "DGIdb_Count": int(result["dgidb_count"]),
                "ChEMBL_Count": int(result["chembl_count"]),
                "DrugQuery_Status": "Success",
            }

            progress_data["drugs_checked"][key] = {"summary": summary, "drugs": result["drugs"]}
            gene_summaries.append(summary)
            all_drug_records.extend(result["drugs"])

            if (idx + 1) % 25 == 0:
                self.save_progress(progress_data)
                print(
                    f"   Progress: {idx+1}/{len(genes_df)} "
                    f"(drugs={summary['Total_Drug_Count']}, DGI={summary['DGIdb_Count']}, ChEMBL={summary['ChEMBL_Count']})",
                    end="\r",
                )

            time.sleep(0.25)

        print(f"\n   ? Drug check done for {len(genes_df):,} genes")

        # Save detailed drug DB
        if all_drug_records:
            drug_df = pd.DataFrame(all_drug_records)
            drug_df.to_csv(self.drug_db_file, index=False)
            print(f"   ?? Saved drug database: {self.drug_db_file.name} ({len(drug_df):,} drug-gene pairs)")

        summary_df = pd.DataFrame(gene_summaries)
        # Merge back without dropping rows
        out = genes_df.merge(summary_df, on="Cache_Key", how="left", suffixes=("", "_drug"))

        # Fill defaults
        for c in ["Has_Drugs", "Total_Drug_Count", "DGIdb_Count", "ChEMBL_Count"]:
            if c in out.columns:
                out[c] = out[c].fillna(0)
        if "Has_Drugs" in out.columns:
            out["Has_Drugs"] = out["Has_Drugs"].astype(bool)

        if "DrugQuery_Status" in out.columns:
            out["DrugQuery_Status"] = out["DrugQuery_Status"].fillna("Unknown")

        return out, progress_data

    # ----------------------------
    # Step 2: fpocket with resume (FIXED persistence)
    # ----------------------------
    def run_fpocket_all(self, genes_df: pd.DataFrame, progress_data: Dict, num_workers: int) -> Tuple[pd.DataFrame, Dict]:
        print("\n" + "=" * 120)
        print("STEP 2: FPOCKET ANALYSIS (genes without drugs) [RESUME-SAFE]")
        print("=" * 120)

        if num_workers <= 0:
            num_workers = 1

        # Choose stable cache key (same as drug check)
        def cache_key(row: pd.Series) -> str:
            if "ENSEMBL_ID" in row.index:
                k = _safe_str(row.get("ENSEMBL_ID"))
                if k:
                    return k
            return _safe_str(row.get("Symbol_Query", row.get("Symbol", "")))

        # Prepare results list, using cache for already done
        cached_records = []
        to_run_payloads = []

        # Minimal analyzer state sent to workers
        analyzer_state = {
            "phenotype": self.phenotype,
            "fpocket_dir": str(self.fpocket_dir),
            "error_log_file": str(self.error_log_file),
        }

        for idx, row in genes_df.iterrows():
            key = cache_key(row)
            if key in progress_data["fpocket_completed"]:
                rec = progress_data["fpocket_completed"][key]
                rec = dict(rec)
                rec["Cache_Key"] = key
                cached_records.append(rec)
            else:
                # Worker needs a dict
                row_dict = row.to_dict()
                row_dict["Cache_Key"] = key
                # Ensure we have a symbol to show / run
                row_dict["Symbol_Query"] = _safe_str(row_dict.get("Symbol_Query", row_dict.get("Symbol", "")))
                payload = {"analyzer_state": analyzer_state, "row": row_dict, "idx": idx, "total": len(genes_df)}
                to_run_payloads.append(payload)

        print(f"Workers: {num_workers}")
        print(f"Cached fpocket results: {len(cached_records):,}")
        print(f"Need fpocket evaluation: {len(to_run_payloads):,}")

        batch_size = 50
        new_records = []

        for start in range(0, len(to_run_payloads), batch_size):
            end = min(start + batch_size, len(to_run_payloads))
            batch = to_run_payloads[start:end]

            with Pool(processes=num_workers) as pool:
                batch_results = pool.map(fpocket_worker, batch)

            # Attach Cache_Key and save into progress in the PARENT
            for payload, rec in zip(batch, batch_results):
                key = payload["row"]["Cache_Key"]
                rec = dict(rec)
                rec["Cache_Key"] = key
                progress_data["fpocket_completed"][key] = rec
                new_records.append(rec)

            self.save_progress(progress_data)
            print(f"? Progress saved ({end}/{len(to_run_payloads)} new fpocket items)")

        # Build fpocket results DF
        fpocket_df = pd.DataFrame(cached_records + new_records)
        if fpocket_df.empty:
            # ensure required columns exist
            fpocket_df = pd.DataFrame(columns=["Cache_Key"])

        return fpocket_df, progress_data

    # ----------------------------
    # Main run
    # ----------------------------
    def run_analysis(self, num_workers: Optional[int] = None) -> Optional[pd.DataFrame]:
        progress_data = self.load_progress()

        if progress_data.get("last_updated"):
            print(f"\n?? Resuming from: {progress_data['last_updated']}")
            print(f"   Drugs cached: {len(progress_data.get('drugs_checked', {}))}")
            print(f"   Fpocket cached: {len(progress_data.get('fpocket_completed', {}))}")

        genes_df = self.load_genes_with_structures()
        if genes_df is None:
            return None

        # Ensure Cache_Key column exists for stable merges
        def make_cache_key(row: pd.Series) -> str:
            if "ENSEMBL_ID" in row.index:
                k = _safe_str(row.get("ENSEMBL_ID"))
                if k:
                    return k
            return _safe_str(row.get("Symbol_Query", row.get("Symbol", "")))

        genes_df["Cache_Key"] = genes_df.apply(make_cache_key, axis=1)

        # Step 1: drug DB checks
        genes_df, progress_data = self.check_all_drugs(genes_df, progress_data)
        self.save_progress(progress_data)

        # Step 2: fpocket
        if num_workers is None:
            num_workers = min(cpu_count(), 4)

        fpocket_df, progress_data = self.run_fpocket_all(genes_df, progress_data, num_workers)
        self.save_progress(progress_data)

        # Merge fpocket results back (LEFT join, keep ALL genes)
        # Use Cache_Key for stability
        if "Cache_Key" not in fpocket_df.columns:
            fpocket_df["Cache_Key"] = ""

        keep_cols = [
            "Cache_Key",
            "Isoforms_Analyzed",
            "Isoforms_Druggable",
            "Max_Druggability_Probability",
            "Mean_Druggability_Probability",
            "Best_Pocket_Volume",
            "Best_Pocket_Hydrophobicity",
            "Best_Pocket_Score",
            "Total_Pockets",
            "Total_Druggable_Pockets",
            "Druggability_Class",
            "Druggability_Probability",
            "Fpocket_Status",
        ]
        for c in keep_cols:
            if c not in fpocket_df.columns:
                fpocket_df[c] = np.nan

        final_df = genes_df.merge(fpocket_df[keep_cols], on="Cache_Key", how="left", suffixes=("", "_fp"))
        final_df = final_df.loc[:, ~final_df.columns.duplicated()]

        # Ensure these exist and fill defaults (so ALL rows are meaningful)
        for c in ["Has_Drugs"]:
            if c not in final_df.columns:
                final_df[c] = False
        final_df["Has_Drugs"] = final_df["Has_Drugs"].fillna(False).astype(bool)

        if "Total_Drug_Count" not in final_df.columns:
            final_df["Total_Drug_Count"] = 0
        final_df["Total_Drug_Count"] = final_df["Total_Drug_Count"].fillna(0).astype(int)

        if "DrugQuery_Status" not in final_df.columns:
            final_df["DrugQuery_Status"] = "Unknown"
        final_df["DrugQuery_Status"] = final_df["DrugQuery_Status"].fillna("Unknown")

        if "Fpocket_Status" not in final_df.columns:
            final_df["Fpocket_Status"] = "Not Run"
        final_df["Fpocket_Status"] = final_df["Fpocket_Status"].fillna("Not Run")

        if "Druggability_Class" not in final_df.columns:
            final_df["Druggability_Class"] = "Unknown"
        final_df["Druggability_Class"] = final_df["Druggability_Class"].fillna("Unknown")

        if "Druggability_Probability" not in final_df.columns:
            final_df["Druggability_Probability"] = 0.0
        final_df["Druggability_Probability"] = final_df["Druggability_Probability"].fillna(0.0).astype(float)

        # Build Processing_Status (DO NOT use it to filter ALL file)
        # Success means: drug query done AND either has drugs OR fpocket attempted OR no structure recognized
        def processing_status(row) -> str:
            symq = _safe_str(row.get("Symbol_Query", row.get("Symbol", "")))
            if not symq:
                return "No Symbol"
            dq = _safe_str(row.get("DrugQuery_Status", "Unknown"))
            fp = _safe_str(row.get("Fpocket_Status", "Not Run"))
            if dq != "Success":
                return f"DrugQuery:{dq}"
            # If has drugs, analysis is good
            if bool(row.get("Has_Drugs", False)):
                return "Success"
            # If no structure cases, still finished (cannot run fpocket)
            if fp in {"No Structure", "Directory Missing", "No Isoforms"}:
                return f"Success ({fp})"
            # If fpocket ran (success/fail), we have an outcome
            if fp.startswith("Success") or fp.startswith("Failed"):
                return "Success"
            # Otherwise unknown completion
            return "Partial"

        final_df["Processing_Status"] = final_df.apply(processing_status, axis=1)

        # SAVE outputs
        output_all = self.druggability_dir / f"{self.phenotype}_druggability_all.csv"
        final_df.to_csv(output_all, index=False)

        # Optional "complete" (success-only convenience)
        output_complete = self.druggability_dir / f"{self.phenotype}_druggability_complete.csv"
        complete_mask = final_df["Processing_Status"].astype(str).str.startswith("Success")
        final_df.loc[complete_mask].to_csv(output_complete, index=False)

        # Summary
        self.print_summary(final_df)

        print(f"\n{'='*120}")
        print("? ANALYSIS COMPLETE!")
        print(f"?? ALL genes:      {output_all}")
        print(f"?? SUCCESS-only:   {output_complete}")
        print(f"?? Drug database:  {self.drug_db_file}")
        print(f"?? Error log:      {self.error_log_file}")
        print(f"?? Resume cache:   {self.progress_file}")
        print("=" * 120)

        return final_df

    def print_summary(self, df: pd.DataFrame):
        print("\n" + "=" * 120)
        print("?? DRUGGABILITY ANALYSIS SUMMARY (ALL GENES)")
        print("=" * 120)

        total = len(df)
        has_drugs = int(df["Has_Drugs"].sum()) if "Has_Drugs" in df.columns else 0

        high = int((df["Druggability_Probability"] >= 0.7).sum())
        mod = int(((df["Druggability_Probability"] >= 0.5) & (df["Druggability_Probability"] < 0.7)).sum())
        low = int(((df["Druggability_Probability"] >= 0.3) & (df["Druggability_Probability"] < 0.5)).sum())

        print(f"Total genes (ALL): {total:,}")
        print(f"Known drugs:       {has_drugs:,} ({(has_drugs/total*100 if total else 0):.1f}%)")
        print(f"High prob (>=0.7): {high:,} ({(high/total*100 if total else 0):.1f}%)")
        print(f"Moderate (0.5-0.7):{mod:,} ({(mod/total*100 if total else 0):.1f}%)")
        print(f"Low (0.3-0.5):     {low:,} ({(low/total*100 if total else 0):.1f}%)")

        # Processing status distribution
        if "Processing_Status" in df.columns:
            print("\nProcessing_Status counts:")
            print(df["Processing_Status"].value_counts(dropna=False).head(20).to_string())

        # If Status column exists (Existing/Novel), show breakdown
        if "Status" in df.columns and "Has_Drugs" in df.columns:
            try:
                existing_drug = df[(df["Status"] == "Existing") & (df["Has_Drugs"] == True)]
                novel_drug = df[(df["Status"] == "Novel") & (df["Has_Drugs"] == True)]

                existing_pred = df[(df["Status"] == "Existing") & (df["Has_Drugs"] == False) & (df["Druggability_Probability"] >= 0.5)]
                novel_pred = df[(df["Status"] == "Novel") & (df["Has_Drugs"] == False) & (df["Druggability_Probability"] >= 0.5)]

                print("\nBy Gene Status:")
                print(f"Existing: {len(existing_drug)} with drugs, {len(existing_pred)} predicted (>=0.5)")
                print(f"Novel:    {len(novel_drug)} with drugs, {len(novel_pred)} predicted (>=0.5)")
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(
        description="? Complete Druggability Analysis - FIXED (ALL genes + alias-aware + resume-safe)",
        epilog="""
USAGE:
  python druggability_analyzer_complete_fixed.py migraine
  python druggability_analyzer_complete_fixed.py migraine --workers 4

OUTPUTS:
  .../druggability_analysis/<phenotype>_druggability_all.csv
  .../druggability_analysis/<phenotype>_druggability_complete.csv  (success-only convenience)
  .../druggability_analysis/<phenotype>_drug_gene_database.csv
  .../druggability_analysis/analysis_progress.json
  .../druggability_analysis/error_log.txt
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("phenotype", help="Phenotype name")
    parser.add_argument("--workers", type=int, default=None, help="Number of fpocket workers (default: min(cpu,4))")
    args = parser.parse_args()

    try:
        analyzer = DruggabilityAnalyzer(args.phenotype)
        analyzer.run_analysis(args.workers)
    except KeyboardInterrupt:
        print("\n?? Interrupted - progress will have been saved at batch boundaries.")
        sys.exit(1)
    except Exception as e:
        print(f"\n? Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
