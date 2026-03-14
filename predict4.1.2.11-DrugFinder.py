#!/usr/bin/env python3
"""
predict_DrugFinder_COMPLETE.py

Complete drug discovery pipeline with:
- Flexible --top-genes parameter
- Multi-K permutation tests (10, 50, 100, 200, 500, ALL)
- Multi-K overlap evaluation
- Symbol filling via MyGene.info
- Resume capability with caching
- MODIFIED: Option to use ALL drugs as universe (not just predicted)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import mygene

# -----------------------------
# Config
# -----------------------------
BASE_DIR_DEFAULT = Path("/data/ascher02/uqmmune1/ANNOVAR")

OPENTARGETS_GQL = "https://api.platform.opentargets.org/api/v4/graphql"
DGIDB_GQL = "https://dgidb.org/api/graphql"
CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"

# Default path to ALL drugs database
ALL_DRUGS_DB_DEFAULT = "/data/ascher02/uqmmune1/ANNOVAR/AllDiseasesToDrugs/ALL_SOURCES_drug_disease_merged.csv"

PHASE_WEIGHT = {
    "APPROVED": 5.0,
    "PHASE4": 4.0,
    "PHASE3": 3.0,
    "PHASE2": 2.0,
    "PHASE1": 1.0,
    "PHASE0": 0.5,
    "PRECLINICAL": 0.5,
    "NA": 0.5,
    "N/A": 0.5,
    "UNKNOWN": 0.5,
    "": 0.5,
}

SOURCE_WEIGHT = {
    "OpenTargets": 1.2,
    "DGIdb": 1.0,
    "ChEMBL": 0.8,
}

# -----------------------------
# Utilities
# -----------------------------
def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"", "nan", "none", "null"}:
        return ""
    return s


def strip_ensembl_version(g: str) -> str:
    g = safe_str(g)
    return re.sub(r"\.\d+$", "", g)


def normalize_drug_name(x: str) -> str:
    """Conservative normalizer."""
    if x is None:
        return ""
    s = str(x).strip().lower()
    if not s or s in {"nan", "none"}:
        return ""
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"\[.*?\]", " ", s)
    s = s.replace("&", " and ")
    s = re.sub(r"[/,+;]", " ", s)
    s = re.sub(r"\b\d+(\.\d+)?\s*(mg|mcg|g|ml|iu|units?)\b", " ", s)
    salt_words = [
        "hydrochloride", "hcl", "sodium", "potassium", "calcium",
        "succinate", "tartrate", "maleate", "phosphate", "sulfate",
        "acetate", "chloride", "nitrate", "mesylate", "besylate",
        "benzoate", "bromide", "citrate", "lactate"
    ]
    s = re.sub(r"\b(" + "|".join(map(re.escape, salt_words)) + r")\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def phase_to_weight(phase: Any, is_approved: Optional[bool] = None) -> Tuple[str, float]:
    if is_approved is True:
        return "APPROVED", PHASE_WEIGHT["APPROVED"]
    p = safe_str(phase).upper()
    if p.isdigit():
        p = f"PHASE{p}"
    if p in PHASE_WEIGHT:
        return p, float(PHASE_WEIGHT[p])
    p2 = p.replace(" ", "")
    if p2 in PHASE_WEIGHT:
        return p2, float(PHASE_WEIGHT[p2])
    return "NA", float(PHASE_WEIGHT["NA"])


def requests_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "DrugFinder/1.0 (research; contact: local)",
        "Accept": "application/json",
    })
    return s


def log_error(path: Path, msg: str):
    try:
        with open(path, "a") as f:
            f.write(f"[{now_ts()}] {msg}\n")
    except Exception:
        pass


# -----------------------------
# Hypergeom p-value
# -----------------------------
def log_choose(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def hypergeom_pval(N: int, K: int, n: int, x: int) -> float:
    if N <= 0 or K < 0 or n < 0:
        return 1.0
    x = max(0, x)
    max_x = min(K, n)
    if x > max_x:
        return 0.0
    denom = log_choose(N, n)
    logs = []
    for i in range(x, max_x + 1):
        logs.append(log_choose(K, i) + log_choose(N - K, n - i) - denom)
    m = max(logs)
    p = sum(math.exp(li - m) for li in logs) * math.exp(m)
    return min(max(p, 0.0), 1.0)


# -----------------------------
# Cache
# -----------------------------
@dataclass
class Cache:
    gene: Dict[str, Any]
    meta: Dict[str, Any]

    @staticmethod
    def load(path: Path) -> "Cache":
        if path.exists():
            try:
                data = json.loads(path.read_text())
                return Cache(gene=data.get("gene", {}), meta=data.get("meta", {}))
            except Exception:
                pass
        return Cache(gene={}, meta={})

    def save(self, path: Path):
        self.meta["last_saved"] = now_ts()
        path.write_text(json.dumps({"gene": self.gene, "meta": self.meta}, indent=2))


# -----------------------------
# Load ALL drugs universe from database
# -----------------------------
def load_all_drugs_universe(db_path: Path) -> Set[str]:
    """
    Load ALL unique normalized drug names from the merged drug-disease database.
    This becomes the universe for hypergeometric tests.
    """
    print(f"\n📂 Loading ALL drugs universe from: {db_path}")
    
    if not db_path.exists():
        raise FileNotFoundError(f"All drugs database not found: {db_path}")
    
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


# -----------------------------
# Load ranked genes
# -----------------------------
def fill_symbols(df: pd.DataFrame, cache_path: Path) -> pd.DataFrame:
    cache = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except Exception:
            pass

    mg = mygene.MyGeneInfo()
    missing = df.loc[df["Symbol"].isna() | (df["Symbol"] == ""), "Gene"].tolist()

    if missing:
        print(f"   Querying MyGene.info for {len(missing)} missing symbols...")
        try:
            res = mg.querymany(
                missing,
                scopes="ensembl.gene",
                fields="symbol",
                species="human",
                as_dataframe=False,
                verbose=False
            )
            for r in res:
                if "query" in r and "symbol" in r:
                    cache[r["query"]] = r["symbol"]
            cache_path.write_text(json.dumps(cache, indent=2))
        except Exception as e:
            print(f"   ⚠️  MyGene.info query failed: {e}")

    df["Symbol"] = df.apply(
        lambda r: r["Symbol"] if r["Symbol"] else cache.get(r["Gene"], ""),
        axis=1
    )
    return df


def load_ranked_genes(path: Path, top_genes: int, output_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "Gene" not in df.columns:
        raise ValueError("Ranked file must contain a 'Gene' column (ENSEMBL IDs).")

    df["Gene"] = df["Gene"].astype(str).apply(strip_ensembl_version)

    score_col = None
    for c in ["Combined_Score", "Importance_Score", "Score", "score"]:
        if c in df.columns:
            score_col = c
            break
    if score_col is None:
        df["__score_tmp"] = -np.arange(len(df), dtype=float)
        score_col = "__score_tmp"

    rank_col = None
    for c in ["Rank", "Combined_Rank", "GeneRank", "rank"]:
        if c in df.columns:
            rank_col = c
            break

    if rank_col is None:
        df["GeneRank"] = pd.Series(np.arange(1, len(df) + 1), index=df.index, dtype=int)
        rank_col = "GeneRank"
    else:
        tmp = pd.to_numeric(df[rank_col], errors="coerce")
        fallback = pd.Series(np.arange(1, len(df) + 1), index=df.index)
        df["GeneRank"] = tmp.fillna(fallback).astype(int)
        rank_col = "GeneRank"

    if "Symbol" not in df.columns:
        df["Symbol"] = ""

    df["Symbol"] = df["Symbol"].astype(str).apply(safe_str)
    df["GeneScore"] = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0).astype(float)

    df = df.sort_values(rank_col, ascending=True).reset_index(drop=True)
    if top_genes > 0:
        df = df.head(top_genes).copy()

    df = fill_symbols(df, output_dir / "ensembl_to_symbol.json")
    
    df["GeneWeight"] = (1.0 - df["GeneRank"].rank(pct=True)).astype(float)
    df["GeneWeight"] = (df["GeneWeight"] + 1e-6).astype(float)

    return df


# -----------------------------
# Load reference drugs
# -----------------------------
def load_reference_drugs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "drug_name" not in df.columns:
        raise ValueError("Reference migraine_drugs.csv must include 'drug_name' column.")
    df["drug_name"] = df["drug_name"].astype(str).apply(safe_str)
    df["drug_norm"] = df["drug_name"].apply(normalize_drug_name)
    df = df[df["drug_norm"] != ""].copy()
    return df


# -----------------------------
# API queries
# -----------------------------
def opentargets_batch_query(session: requests.Session, ensembl_ids: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    blocks = []
    for i, eid in enumerate(ensembl_ids):
        blocks.append(
            f'''
            t{i}: target(ensemblId: "{eid}") {{
              knownDrugs {{
                rows {{
                  drug {{
                    id
                    name
                    isApproved
                    hasBeenWithdrawn
                    yearOfFirstApproval
                  }}
                  disease {{
                    id
                    name
                  }}
                  phase
                  status
                  ctIds
                }}
              }}
            }}
            '''
        )

    query = "query batchKnownDrugs {\n" + "\n".join(blocks) + "\n}"

    r = session.post(OPENTARGETS_GQL, json={"query": query}, timeout=45)
    r.raise_for_status()
    data = r.json()
    if "errors" in data:
        raise RuntimeError(str(data["errors"])[:500])

    out: Dict[str, List[Dict[str, Any]]] = {eid: [] for eid in ensembl_ids}
    payload = data.get("data", {}) or {}
    for i, eid in enumerate(ensembl_ids):
        key = f"t{i}"
        targ = payload.get(key)
        if not targ:
            continue
        kd = (targ.get("knownDrugs") or {}).get("rows") or []
        rows = []
        for row in kd:
            drug = row.get("drug") or {}
            disease = row.get("disease") or {}
            rows.append({
                "DrugId": safe_str(drug.get("id")),
                "DrugName": safe_str(drug.get("name")),
                "IsApproved": bool(drug.get("isApproved")) if drug.get("isApproved") is not None else None,
                "Withdrawn": bool(drug.get("hasBeenWithdrawn")) if drug.get("hasBeenWithdrawn") is not None else None,
                "Phase": row.get("phase"),
                "Status": safe_str(row.get("status")),
                "DiseaseId": safe_str(disease.get("id")),
                "DiseaseName": safe_str(disease.get("name")),
                "CtIds": "|".join([safe_str(x) for x in (row.get("ctIds") or []) if safe_str(x)]),
                "Source": "OpenTargets",
            })
        out[eid] = rows
    return out


def dgidb_query(session: requests.Session, aliases: List[str]) -> List[Dict[str, Any]]:
    if not aliases:
        return []
    safe_terms = [a.replace('"', '\\"') for a in aliases if safe_str(a)]
    if not safe_terms:
        return []

    terms_str = ", ".join([f'"{t}"' for t in safe_terms])

    query = f"""
    query {{
      geneMatches(searchTerms: [{terms_str}]) {{
        directMatches {{
          searchTerm
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
              sources {{
                sourceDbName
              }}
              interactionScore
              publications {{
                pmid
              }}
            }}
          }}
        }}
      }}
    }}
    """

    r = session.post(DGIDB_GQL, json={"query": query}, timeout=30)
    if r.status_code != 200:
        return []
    data = r.json()
    if "errors" in data:
        return []

    out = []
    gm = (data.get("data") or {}).get("geneMatches") or {}
    direct = gm.get("directMatches") or []
    for dm in direct:
        matches = dm.get("matches") or []
        for m in matches:
            matched_gene = safe_str(m.get("name"))
            interactions = m.get("interactions") or []
            for inter in interactions:
                drug = inter.get("drug") or {}
                itypes = inter.get("interactionTypes") or []
                srcs = inter.get("sources") or []
                pubs = inter.get("publications") or []
                out.append({
                    "DrugId": safe_str(drug.get("conceptId")),
                    "DrugName": safe_str(drug.get("name")),
                    "IsApproved": bool(drug.get("approved")) if drug.get("approved") is not None else None,
                    "Phase": None,
                    "Status": "",
                    "DiseaseName": "",
                    "MatchedGene": matched_gene,
                    "InteractionType": safe_str(itypes[0].get("type")) if itypes else "",
                    "Directionality": safe_str(itypes[0].get("directionality")) if itypes else "",
                    "InteractionScore": inter.get("interactionScore"),
                    "SourcesDetail": "|".join([safe_str(s.get("sourceDbName")) for s in srcs if safe_str(s.get("sourceDbName"))])[:500],
                    "PMIDs": "|".join([safe_str(p.get("pmid")) for p in pubs if safe_str(p.get("pmid"))])[:500],
                    "Source": "DGIdb",
                })
    return out


def chembl_find_target(session: requests.Session, gene_symbol: str) -> Optional[str]:
    gene_symbol = safe_str(gene_symbol)
    if not gene_symbol:
        return None
    url = f"{CHEMBL_API}/target/search.json"
    r = session.get(url, params={"q": gene_symbol, "limit": 5}, timeout=25)
    if r.status_code != 200:
        return None
    data = r.json()
    targets = data.get("targets") or []
    if not targets:
        return None
    return safe_str(targets[0].get("target_chembl_id"))


def chembl_target_activities(session: requests.Session, target_id: str, max_molecules: int) -> List[Dict[str, Any]]:
    if not target_id:
        return []
    url = f"{CHEMBL_API}/activity.json"
    r = session.get(url, params={"target_chembl_id": target_id, "limit": 200}, timeout=35)
    if r.status_code != 200:
        return []
    data = r.json()
    acts = data.get("activities") or []
    out = []
    seen = set()
    for a in acts:
        mol_id = safe_str(a.get("molecule_chembl_id"))
        if not mol_id or mol_id in seen:
            continue
        seen.add(mol_id)
        out.append({
            "DrugId": mol_id,
            "DrugName": safe_str(a.get("molecule_pref_name")) or mol_id,
            "IsApproved": None,
            "Phase": None,
            "Status": "",
            "DiseaseName": "",
            "InteractionType": safe_str(a.get("standard_type")) or "",
            "InteractionScore": a.get("standard_value"),
            "Source": "ChEMBL",
            "ChEMBL_Target": target_id,
        })
        if len(out) >= max_molecules:
            break
    return out


# -----------------------------
# Build gene→drug table
# -----------------------------
def build_gene_drug_table(
    ranked: pd.DataFrame,
    output_dir: Path,
    cache: Cache,
    workers: int,
    ot_batch_size: int,
    chembl_max_mols: int,
    error_log: Path,
) -> pd.DataFrame:
    session_ot = requests_session()
    session_dgi = requests_session()
    session_chembl = requests_session()

    genes = ranked["Gene"].tolist()

    # OpenTargets
    ot_results: Dict[str, List[Dict[str, Any]]] = {}
    remaining = [g for g in genes if not cache.gene.get(g, {}).get("opentargets_done", False)]
    if remaining:
        print(f"\n🔎 OpenTargets: querying {len(remaining):,} genes in batches of {ot_batch_size} ...")
        for i in range(0, len(remaining), ot_batch_size):
            batch = remaining[i : i + ot_batch_size]
            try:
                res = opentargets_batch_query(session_ot, batch)
                for g in batch:
                    cache.gene.setdefault(g, {})
                    cache.gene[g]["opentargets"] = res.get(g, [])
                    cache.gene[g]["opentargets_done"] = True
                cache.save(output_dir / "progress_cache.json")
            except Exception as e:
                log_error(error_log, f"OpenTargets batch failed ({i}-{i+len(batch)}): {e}")
            if (i // ot_batch_size) % 5 == 0:
                print(f"   OpenTargets progress: {min(i+len(batch), len(remaining))}/{len(remaining)}")

    for g in genes:
        ot_results[g] = (cache.gene.get(g, {}).get("opentargets") or [])

    # DGIdb + ChEMBL
    def per_gene_queries(gene: str, symbol: str) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        cache.gene.setdefault(gene, {})
        dgi_rows = []
        chem_rows = []

        if not cache.gene[gene].get("dgidb_done", False):
            try:
                aliases = list({symbol, gene}) if symbol else [gene]
                dgi_rows = dgidb_query(session_dgi, aliases)
                cache.gene[gene]["dgidb"] = dgi_rows
                cache.gene[gene]["dgidb_done"] = True
            except Exception as e:
                log_error(error_log, f"DGIdb failed for {gene}/{symbol}: {e}")
        else:
            dgi_rows = cache.gene[gene].get("dgidb") or []

        if not cache.gene[gene].get("chembl_done", False):
            try:
                if symbol:
                    target = chembl_find_target(session_chembl, symbol)
                    if target:
                        chem_rows = chembl_target_activities(session_chembl, target, max_molecules=chembl_max_mols)
                cache.gene[gene]["chembl"] = chem_rows
                cache.gene[gene]["chembl_done"] = True
            except Exception as e:
                log_error(error_log, f"ChEMBL failed for {gene}/{symbol}: {e}")
        else:
            chem_rows = cache.gene[gene].get("chembl") or []

        return gene, dgi_rows, chem_rows

    todo = []
    for _, row in ranked.iterrows():
        gene = row["Gene"]
        sym = safe_str(row.get("Symbol", ""))
        if not cache.gene.get(gene, {}).get("dgidb_done", False) or not cache.gene.get(gene, {}).get("chembl_done", False):
            todo.append((gene, sym))

    if todo:
        print(f"\n🔎 DGIdb + ChEMBL: querying {len(todo):,} genes with {workers} workers ...")
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(per_gene_queries, g, s): (g, s) for g, s in todo}
            for fut in as_completed(futures):
                done += 1
                if done % 25 == 0 or done == len(todo):
                    print(f"   DGIdb/ChEMBL progress: {done}/{len(todo)}")
                    cache.save(output_dir / "progress_cache.json")

    # Build table
    rows = []
    for _, r in ranked.iterrows():
        gene = r["Gene"]
        sym = safe_str(r.get("Symbol", ""))
        gene_rank = int(r["GeneRank"])
        gene_score = float(r["GeneScore"])
        gene_weight = float(r["GeneWeight"])

        for ot in ot_results.get(gene, []):
            drug_id = safe_str(ot.get("DrugId"))
            drug_name = safe_str(ot.get("DrugName"))
            phase_label, phase_w = phase_to_weight(ot.get("Phase"), is_approved=ot.get("IsApproved"))
            rows.append({
                "Gene": gene,
                "Symbol": sym,
                "GeneRank": gene_rank,
                "GeneScore": gene_score,
                "GeneWeight": gene_weight,
                "DrugId": drug_id,
                "DrugName": drug_name,
                "DrugNorm": normalize_drug_name(drug_name),
                "Source": "OpenTargets",
                "EvidencePhase": phase_label,
                "EvidenceWeight": phase_w * SOURCE_WEIGHT["OpenTargets"],
                "IsApproved": ot.get("IsApproved"),
                "Status": safe_str(ot.get("Status")),
                "DiseaseIndications": safe_str(ot.get("DiseaseName")),
                "TargetEvidence": "",
                "MoA": "",
                "CtIds": safe_str(ot.get("CtIds")),
            })

        dgi = cache.gene.get(gene, {}).get("dgidb") or []
        for d in dgi:
            drug_id = safe_str(d.get("DrugId"))
            drug_name = safe_str(d.get("DrugName"))
            phase_label, phase_w = phase_to_weight(None, is_approved=d.get("IsApproved"))
            rows.append({
                "Gene": gene,
                "Symbol": sym,
                "GeneRank": gene_rank,
                "GeneScore": gene_score,
                "GeneWeight": gene_weight,
                "DrugId": drug_id,
                "DrugName": drug_name,
                "DrugNorm": normalize_drug_name(drug_name),
                "Source": "DGIdb",
                "EvidencePhase": phase_label,
                "EvidenceWeight": phase_w * SOURCE_WEIGHT["DGIdb"],
                "IsApproved": d.get("IsApproved"),
                "Status": "",
                "DiseaseIndications": "",
                "TargetEvidence": safe_str(d.get("SourcesDetail")),
                "MoA": safe_str(d.get("InteractionType")),
                "CtIds": "",
            })

        chem = cache.gene.get(gene, {}).get("chembl") or []
        for c in chem:
            drug_id = safe_str(c.get("DrugId"))
            drug_name = safe_str(c.get("DrugName"))
            phase_label, phase_w = phase_to_weight("PRECLINICAL", is_approved=None)
            rows.append({
                "Gene": gene,
                "Symbol": sym,
                "GeneRank": gene_rank,
                "GeneScore": gene_score,
                "GeneWeight": gene_weight,
                "DrugId": drug_id,
                "DrugName": drug_name,
                "DrugNorm": normalize_drug_name(drug_name),
                "Source": "ChEMBL",
                "EvidencePhase": phase_label,
                "EvidenceWeight": phase_w * SOURCE_WEIGHT["ChEMBL"],
                "IsApproved": None,
                "Status": "",
                "DiseaseIndications": "",
                "TargetEvidence": safe_str(c.get("ChEMBL_Target")),
                "MoA": safe_str(c.get("InteractionType")),
                "CtIds": "",
            })

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "Gene","Symbol","GeneRank","GeneScore","GeneWeight",
                "DrugId","DrugName","DrugNorm","Source","EvidencePhase","EvidenceWeight",
                "IsApproved","Status","DiseaseIndications","TargetEvidence","MoA","CtIds"
            ]
        )
    df = df[df["DrugNorm"].astype(str) != ""].copy()
    return df


# -----------------------------
# Rank drugs
# -----------------------------
def rank_drugs(gene_drug: pd.DataFrame) -> pd.DataFrame:
    if gene_drug.empty:
        return pd.DataFrame(columns=["DrugNorm","DrugName_best","DrugScore","N_target_genes","Best_gene_rank","Max_phase","Sources"])

    name_best = (
        gene_drug.groupby("DrugNorm")["DrugName"]
        .agg(lambda s: s.value_counts().index[0] if len(s.dropna()) else "")
        .rename("DrugName_best")
        .reset_index()
    )

    gene_drug["RowScore"] = gene_drug["GeneWeight"].astype(float) * gene_drug["EvidenceWeight"].astype(float)

    agg = gene_drug.groupby("DrugNorm").agg(
        DrugScore=("RowScore", "sum"),
        N_target_genes=("Gene", pd.Series.nunique),
        Best_gene_rank=("GeneRank", "min"),
        Sources=("Source", lambda s: "|".join(sorted(set([safe_str(x) for x in s if safe_str(x)])))),
    ).reset_index()

    phase_rank = {"APPROVED": 5, "PHASE4": 4, "PHASE3": 3, "PHASE2": 2, "PHASE1": 1, "PHASE0": 0, "PRECLINICAL": 0, "NA": 0, "UNKNOWN": 0, "N/A": 0, "": 0}
    gene_drug["PhaseRank"] = gene_drug["EvidencePhase"].astype(str).map(lambda x: phase_rank.get(x.upper(), 0))
    max_phase = gene_drug.groupby("DrugNorm")["PhaseRank"].max().reset_index()
    inv = {v: k for k, v in phase_rank.items()}
    max_phase["Max_phase"] = max_phase["PhaseRank"].map(lambda v: inv.get(int(v), "NA"))
    max_phase = max_phase.drop(columns=["PhaseRank"])

    out = agg.merge(name_best, on="DrugNorm", how="left").merge(max_phase, on="DrugNorm", how="left")
    out = out.sort_values(["DrugScore", "N_target_genes", "Best_gene_rank"], ascending=[False, False, True]).reset_index(drop=True)
    out["DrugRank"] = np.arange(1, len(out) + 1)
    
    if "N_target_genes" in out.columns:
        out["n_genes"] = out["N_target_genes"]
    
    return out


# -----------------------------
# Multi-K evaluation (MODIFIED: accepts external universe)
# -----------------------------
def evaluate_overlap_multi_k(
    drug_rank: pd.DataFrame, 
    ref: pd.DataFrame, 
    ks: List[int],
    universe_set: Optional[Set[str]] = None,  # NEW: external universe
    universe_mode: str = "predicted"  # NEW: "predicted" or "all"
) -> pd.DataFrame:
    """
    Evaluate overlap at multiple K values.
    
    Args:
        drug_rank: Ranked predicted drugs
        ref: Reference drugs dataframe
        ks: List of K values to evaluate
        universe_set: Optional set of ALL drugs (for universe_mode="all")
        universe_mode: "predicted" = use only predicted drugs as universe
                       "all" = use external universe_set as universe
    """
    ref_set = set(ref["drug_norm"].unique().tolist())
    pred_all = list(dict.fromkeys(drug_rank["DrugNorm"].astype(str)))
    pred_set = set(pred_all)
    
    # Determine universe based on mode
    if universe_mode == "all" and universe_set is not None:
        universe = len(universe_set)
        # Reference drugs that exist in universe
        ref_in_universe = ref_set & universe_set
        Kpop = len(ref_in_universe)
        universe_label = "ALL_DRUGS"
        print(f"\n📊 Using ALL DRUGS universe: {universe:,} drugs")
        print(f"   Reference drugs in universe: {Kpop:,} / {len(ref_set):,}")
    else:
        universe = len(pred_all)
        Kpop = len(ref_set)
        universe_label = "PREDICTED"
        print(f"\n📊 Using PREDICTED universe: {universe:,} drugs")

    rows = []
    for k in ks:
        k_actual = min(k, len(pred_all))
        topk = pred_all[:k_actual]
        topk_set = set(topk)
        
        # For "all" universe mode, we still look at overlap with ref_set
        # but expected is calculated against the full universe
        if universe_mode == "all" and universe_set is not None:
            inter = topk_set & ref_in_universe
        else:
            inter = topk_set & ref_set
        x = len(inter)

        precision = x / k_actual if k_actual > 0 else 0.0
        recall = x / Kpop if Kpop > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
        # Expected overlap calculation
        # For "all" mode: we're drawing k_actual from universe, Kpop are "successes"
        expected = k_actual * (Kpop / universe) if universe > 0 else 0.0
        fe = x / expected if expected > 0 else 0.0
        
        # Hypergeometric test
        # N = universe size
        # K = number of reference drugs (in universe)
        # n = number of predicted drugs at this cutoff (k_actual)
        # x = observed overlap
        pval = hypergeom_pval(N=universe, K=Kpop, n=k_actual, x=x) if universe > 0 else 1.0

        rows.append({
            "K": k,
            "K_actual": k_actual,
            "Universe": universe,
            "Universe_Type": universe_label,
            "Reference": Kpop,
            "Overlap": x,
            "Expected": expected,
            "FE": fe,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Hypergeom_p": pval,
        })

    return pd.DataFrame(rows)


# -----------------------------
# Multi-K permutation test
# -----------------------------
def permutation_test_multi_k(
    gene_drug: pd.DataFrame,
    drug_rank: pd.DataFrame,
    ref: pd.DataFrame,
    k_list: List[int],
    n_perm: int,
    seed: int = 13,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Permutation test for drug ranking.
    
    NOTE: This test shuffles gene weights among YOUR genes, so the universe
    is inherently your predicted drugs. This tests whether your gene RANKING
    matters, not whether your gene SET is good.
    """
    if gene_drug.empty or drug_rank.empty:
        null = pd.DataFrame({"k": [], "perm": [], "precision": [], "overlap": []})
        summary = pd.DataFrame({"k": [], "observed_prec": [], "null_mean": [], "null_std": [], "p_value": []})
        return null, summary

    rng = np.random.default_rng(seed)

    genes = gene_drug["Gene"].unique().tolist()
    drugs = gene_drug["DrugNorm"].unique().tolist()

    g2i = {g: i for i, g in enumerate(genes)}
    d2i = {d: i for i, d in enumerate(drugs)}

    gw = gene_drug.groupby("Gene")["GeneWeight"].max().reindex(genes).fillna(0.0).to_numpy(dtype=float)

    drug_idx = gene_drug["DrugNorm"].map(d2i).to_numpy(dtype=int)
    gene_idx = gene_drug["Gene"].map(g2i).to_numpy(dtype=int)
    evw = gene_drug["EvidenceWeight"].to_numpy(dtype=float)

    obs_scores = np.bincount(drug_idx, weights=evw * gw[gene_idx], minlength=len(drugs))

    ref_set = set(ref["drug_norm"].unique().tolist())

    # Observed for all K
    obs_results = {}
    for k in k_list:
        k_actual = min(k, len(drugs))
        topk_idx = np.argsort(-obs_scores)[:k_actual]
        pred_topk = {drugs[i] for i in topk_idx}
        obs_overlap = len(pred_topk & ref_set)
        obs_prec = obs_overlap / k_actual if k_actual > 0 else 0.0
        obs_results[k] = (obs_prec, obs_overlap, k_actual)

    # Permutations
    null_rows = []
    print(f"\n🧪 Running {n_perm} permutations for K={k_list} ...")
    print(f"   NOTE: Permutation test uses PREDICTED drugs as universe (tests ranking, not capture)")
    
    for p in range(n_perm):
        perm_gw = rng.permutation(gw)
        perm_scores = np.bincount(drug_idx, weights=evw * perm_gw[gene_idx], minlength=len(drugs))
        
        for k in k_list:
            k_actual = min(k, len(drugs))
            perm_top = np.argsort(-perm_scores)[:k_actual]
            perm_set = {drugs[i] for i in perm_top}
            ov = len(perm_set & ref_set)
            prec = ov / k_actual if k_actual > 0 else 0.0
            null_rows.append({"k": k, "perm": p + 1, "precision": prec, "overlap": ov})

        if (p + 1) % 100 == 0:
            print(f"   Progress: {p+1}/{n_perm}")

    null_df = pd.DataFrame(null_rows)

    # Compute summary statistics
    summary_rows = []
    for k in k_list:
        obs_prec, obs_ov, k_actual = obs_results[k]
        null_k = null_df[null_df["k"] == k]["precision"]
        null_mean = float(null_k.mean()) if len(null_k) > 0 else 0.0
        null_std = float(null_k.std(ddof=1)) if len(null_k) > 1 else 0.0
        p_value = (np.sum(null_k.to_numpy() >= obs_prec) + 1) / (len(null_k) + 1)
        
        summary_rows.append({
            "K": k,
            "K_actual": k_actual,
            "Observed_Prec": obs_prec,
            "Observed_Overlap": obs_ov,
            "Null_Mean_Prec": null_mean,
            "Null_Std_Prec": null_std,
            "P_value": float(p_value),
            "Significant": "✅" if obs_prec > null_mean and p_value < 0.05 else "❌"
        })

    summary_df = pd.DataFrame(summary_rows)
    
    return null_df, summary_df


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Complete drug discovery with multi-K permutation tests")
    ap.add_argument("phenotype", help="Phenotype name, e.g. migraine")
    ap.add_argument("--base-dir", default=str(BASE_DIR_DEFAULT))
    ap.add_argument("--top-genes", type=int, default=200, help="Top N genes to use (default: 200)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--ot-batch-size", type=int, default=20)
    ap.add_argument("--chembl-max-mols", type=int, default=40)
    ap.add_argument("--perm", type=int, default=1000, help="Permutation iterations (default: 1000)")
    
    # NEW: Universe options
    ap.add_argument("--universe", choices=["predicted", "all"], default="all",
                    help="Universe for hypergeometric test: 'predicted' (your drugs only) or 'all' (all drugs in DB)")
    ap.add_argument("--all-drugs-db", default=ALL_DRUGS_DB_DEFAULT,
                    help="Path to database with ALL drugs (for --universe=all)")
    
    args = ap.parse_args()

    phenotype = args.phenotype
    base_dir = Path(args.base_dir)

    # Construct path based on top-genes
    ranked_path = base_dir / phenotype / "GeneDifferentialExpression" / "Files" / "UltimateCompleteRankingAnalysis" / "FinalIntegration" / f"TOP_{args.top_genes}_1.csv"
    ref_path = Path("migraine_drugs2.csv")

    out_dir = base_dir / phenotype / "GeneDifferentialExpression" / "Files" / "UltimateCompleteRankingAnalysis" / "DrugIntegration"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"📌 Phenotype: {phenotype}")
    print(f"📌 Top genes: {args.top_genes}")
    print(f"📌 Ranked genes: {ranked_path}")
    print(f"📌 Reference drugs: {ref_path.resolve()}")
    print(f"📌 Output: {out_dir}")
    print(f"📌 Universe mode: {args.universe.upper()}")
    if args.universe == "all":
        print(f"📌 All drugs DB: {args.all_drugs_db}")

    if not ranked_path.exists():
        raise FileNotFoundError(f"Ranked genes not found: {ranked_path}")
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference migraine_drugs.csv not found: {ref_path.resolve()}")

    error_log = out_dir / "error_log.txt"
    cache_path = out_dir / "progress_cache.json"
    cache = Cache.load(cache_path)

    # Load ALL drugs universe if needed
    all_drugs_universe: Optional[Set[str]] = None
    if args.universe == "all":
        all_drugs_universe = load_all_drugs_universe(Path(args.all_drugs_db))

    ranked = load_ranked_genes(ranked_path, top_genes=args.top_genes, output_dir=out_dir)
    print(f"\n✅ Loaded ranked genes: {len(ranked):,}")
    
    symbols_available = (ranked["Symbol"] != "").sum()
    print(f"   Symbols available: {symbols_available}/{len(ranked)}")
    print(f"   Example: {ranked.iloc[0]['Gene']}  symbol={safe_str(ranked.iloc[0].get('Symbol',''))} rank={ranked.iloc[0]['GeneRank']}")

    ref = load_reference_drugs(ref_path)
    print(f"✅ Loaded reference drugs: {ref['drug_norm'].nunique():,} unique normalized names")

    # Build gene→drug table
    gene_drug = build_gene_drug_table(
        ranked=ranked,
        output_dir=out_dir,
        cache=cache,
        workers=max(1, args.workers),
        ot_batch_size=max(1, args.ot_batch_size),
        chembl_max_mols=max(1, args.chembl_max_mols),
        error_log=error_log,
    )

    gene_drug_file = out_dir / "GeneDrugTable_ALL.csv"
    gene_drug.to_csv(gene_drug_file, index=False)
    print(f"\n💾 Saved: {gene_drug_file} ({len(gene_drug):,} gene→drug evidence rows)")

    # Rank drugs
    drug_rank = rank_drugs(gene_drug)
    drug_rank_file = out_dir / "DrugRanking.csv"
    drug_rank.to_csv(drug_rank_file, index=False)
    print(f"💾 Saved: {drug_rank_file} ({len(drug_rank):,} unique predicted drugs)")
    
    # Hub bias diagnostic
    if "n_genes" in drug_rank.columns and len(drug_rank) > 1:
        hub_corr = drug_rank["DrugScore"].corr(drug_rank["n_genes"])
        print(f"\n🔬 Hub bias diagnostic: Corr(DrugScore, n_genes) = {hub_corr:.3f}")
        if hub_corr > 0.8:
            print("   ⚠️  Strong hub bias (score driven by target count)")
        elif hub_corr > 0.5:
            print("   ℹ️  Moderate hub bias")
        else:
            print("   ✅ Low hub bias (score reflects gene quality + evidence)")

    # Multi-K evaluation
    k_list = [10, 20, 50, 100, 200, 500, 1000, len(drug_rank)]
    k_list = sorted(list(set([k for k in k_list if k <= len(drug_rank)])))
    
    print(f"\n📊 MULTI-K OVERLAP EVALUATION")
    print(f"   Evaluating at K = {k_list}")
    print(f"   Universe mode: {args.universe.upper()}")
    
    # Run evaluation with selected universe
    eval_df = evaluate_overlap_multi_k(
        drug_rank, 
        ref, 
        k_list,
        universe_set=all_drugs_universe,
        universe_mode=args.universe
    )
    eval_file = out_dir / f"DrugOverlap_MultiK_{args.universe.upper()}.csv"
    eval_df.to_csv(eval_file, index=False)
    print(f"💾 Saved: {eval_file}")

    print("\n" + "="*120)
    print(f"OVERLAP METRICS (Universe = {args.universe.upper()})")
    print("="*120)
    print(eval_df.to_string(index=False))

    # Also run with predicted universe for comparison if using "all"
    if args.universe == "all":
        print("\n" + "="*120)
        print("COMPARISON: Same metrics with PREDICTED universe")
        print("="*120)
        eval_df_pred = evaluate_overlap_multi_k(drug_rank, ref, k_list, universe_set=None, universe_mode="predicted")
        eval_file_pred = out_dir / "DrugOverlap_MultiK_PREDICTED.csv"
        eval_df_pred.to_csv(eval_file_pred, index=False)
        print(eval_df_pred.to_string(index=False))
        print(f"💾 Saved: {eval_file_pred}")

    # Multi-K permutation test (always uses predicted universe)
    print(f"\n🧪 MULTI-K PERMUTATION TEST")
    print(f"   K values: {k_list}")
    print(f"   Permutations: {args.perm}")
    
    null_df, summary_df = permutation_test_multi_k(
        gene_drug=gene_drug,
        drug_rank=drug_rank,
        ref=ref,
        k_list=k_list,
        n_perm=args.perm,
        seed=13,
    )
    
    null_file = out_dir / "Permutation_Null_MultiK.csv"
    null_df.to_csv(null_file, index=False)
    
    summary_file = out_dir / "Permutation_Summary_MultiK.csv"
    summary_df.to_csv(summary_file, index=False)

    print(f"\n💾 Saved: {null_file}")
    print(f"💾 Saved: {summary_file}")

    # Display results
    print("\n" + "="*120)
    print("🎯 PERMUTATION TEST RESULTS (All K values)")
    print("="*120)
    print(summary_df.to_string(index=False))
    
    # Interpretation
    print("\n" + "="*120)
    print("📋 INTERPRETATION")
    print("="*120)
    
    sig_any = (summary_df["Significant"] == "✅").any()
    
    if sig_any:
        sig_k = summary_df[summary_df["Significant"] == "✅"]["K"].tolist()
        print(f"✅ SUCCESS: Significant enrichment at K = {sig_k}")
        print(f"   → Drug predictions show BIOLOGICAL SIGNAL")
    else:
        print(f"❌ NO SIGNIFICANCE: No K value shows significant enrichment (p < 0.05)")
        print(f"   → Drug predictions do not exceed random expectation")
    
    # Summary of universe comparison
    if args.universe == "all":
        print("\n" + "="*120)
        print("📋 UNIVERSE COMPARISON SUMMARY")
        print("="*120)
        print(f"""
   With PREDICTED universe ({len(drug_rank):,} drugs):
   - Tests: "Among my predictions, are migraine drugs ranked higher?"
   - This is what your permutation test evaluates
   
   With ALL drugs universe ({len(all_drugs_universe):,} drugs):
   - Tests: "Does my pipeline capture migraine drugs better than chance?"
   - This tests whether your gene set is good for drug discovery
   
   KEY INSIGHT:
   - If ALL universe shows enrichment but PREDICTED doesn't → Gene set is good, ranking needs work
   - If neither shows enrichment → Gene set doesn't capture migraine drug targets
   - If PREDICTED shows enrichment but ALL doesn't → Pipeline is biased toward certain drug types
""")
    
    # Show top drugs
    print("\n📋 Top 20 Predicted Drugs:")
    top20 = drug_rank.head(20)[["DrugRank", "DrugName_best", "DrugScore", "N_target_genes", "Best_gene_rank", "Max_phase"]]
    print(top20.to_string(index=False))

    print(f"\n✅ DONE — All outputs in: {out_dir}")


if __name__ == "__main__":
    main()
