#!/usr/bin/env python3
"""
predict4.1.2.12.1-DrugDirectionalityRescue.py

Build a unique drug-gene directionality table by combining:
1) Local pipeline outputs
2) Online rescue for missing mechanism/action info from:
   - ChEMBL
   - DGIdb

USAGE
-----
python predict4.1.2.12.1-DrugDirectionalityRescue.py migraine

OPTIONAL
--------
python predict4.1.2.12.1-DrugDirectionalityRescue.py migraine --top-drugs 100 --top-genes 300
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests


# =============================================================================
# CONFIG
# =============================================================================
BASE_DIR_DEFAULT = Path("/data/ascher02/uqmmune1/ANNOVAR")

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"
DGIDB_GQL = "https://dgidb.org/api/graphql"

REQUEST_TIMEOUT = 30
SLEEP_BETWEEN_REMOTE_CALLS = 0.05


# =============================================================================
# HELPERS
# =============================================================================
def safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"", "nan", "none", "null"}:
        return ""
    return s


def strip_ensg_version(x: Any) -> str:
    s = safe_str(x)
    return re.sub(r"\.\d+$", "", s)


def normalize_drug_name(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip().lower()
    if not s or s in {"nan", "none", "null"}:
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
        "benzoate", "bromide", "citrate", "lactate",
    ]
    s = re.sub(r"\b(" + "|".join(map(re.escape, salt_words)) + r")\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def requests_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": "DrugDirectionalityRescue/1.0",
            "Accept": "application/json",
        }
    )
    return s


def ensure_col(df: pd.DataFrame, col: str, default: Any = "") -> pd.DataFrame:
    if col not in df.columns:
        df[col] = default
    return df


def first_nonempty(values: List[Any], default: str = "") -> str:
    for v in values:
        s = safe_str(v)
        if s:
            return s
    return default


def unique_join(values: List[Any], sep: str = " | ") -> str:
    out = []
    seen = set()
    for v in values:
        s = safe_str(v)
        if s and s not in seen:
            out.append(s)
            seen.add(s)
    return sep.join(out)


# =============================================================================
# CLASSIFICATION
# =============================================================================
def normalize_gene_direction(x: Any) -> str:
    s = safe_str(x).lower()

    if s in {"up", "higher in cases", "high", "increased", "positive"}:
        return "higher in cases"
    if s in {"down", "lower in cases", "low", "decreased", "negative"}:
        return "lower in cases"
    return "unclear"


def classify_action_type(text: Any) -> str:
    """
    Collapse raw action/mechanism text into one of:
      inhibitor / antagonist / agonist / activator / modulator / unknown
    """
    t = safe_str(text).lower()

    if not t:
        return "unknown"

    inhibitor_terms = [
        "inhibitor", "inhibit", "blocking", "blocker", "block", "suppresses",
        "suppress", "downregulator", "down-regulator", "negative regulator"
    ]
    antagonist_terms = [
        "antagonist", "inverse agonist", "negative allosteric modulator"
    ]
    agonist_terms = [
        "agonist", "partial agonist", "full agonist", "positive allosteric modulator"
    ]
    activator_terms = [
        "activator", "activate", "activates", "inducer", "stimulator",
        "upregulator", "up-regulator"
    ]
    modulator_terms = [
        "modulator", "binder", "binding agent", "interacts with", "ligand"
    ]

    if any(k in t for k in inhibitor_terms):
        return "inhibitor"
    if any(k in t for k in antagonist_terms):
        return "antagonist"
    if any(k in t for k in agonist_terms):
        return "agonist"
    if any(k in t for k in activator_terms):
        return "activator"
    if any(k in t for k in modulator_terms):
        return "modulator"

    return "unknown"


def classify_direction_match(gene_direction: Any, action_type: Any) -> str:
    g = normalize_gene_direction(gene_direction)
    a = safe_str(action_type).lower()

    if g == "higher in cases":
        if a in {"inhibitor", "antagonist"}:
            return "consistent"
        if a in {"activator", "agonist"}:
            return "inconsistent"
        return "unclear"

    if g == "lower in cases":
        if a in {"activator", "agonist"}:
            return "consistent"
        if a in {"inhibitor", "antagonist"}:
            return "inconsistent"
        return "unclear"

    return "unclear"


def infer_approved_flag(row: pd.Series) -> str:
    for col in ["Approved", "IsApproved", "is_approved"]:
        if col in row.index:
            s = safe_str(row[col]).lower()
            if s in {"true", "1", "yes", "y"}:
                return "Yes"
            if s in {"false", "0", "no", "n"}:
                return "No"

    for col in ["Phase", "EvidencePhase", "Max_phase", "Status", "DiseaseIndications"]:
        if col in row.index:
            s = safe_str(row[col]).lower()
            if "approved" in s:
                return "Yes"

    return ""


# =============================================================================
# PATHS
# =============================================================================
def resolve_paths(phenotype: str, base_dir: Path) -> Dict[str, Path]:
    base_path = base_dir / phenotype
    files_dir = base_path / "GeneDifferentialExpression" / "Files"
    ranking_dir = files_dir / "UltimateCompleteRankingAnalysis"
    drug_dir = ranking_dir / "DrugIntegration"

    ranked_genes_file = ranking_dir / "RANKED_composite.csv"
    gene_drug_file = drug_dir / "GeneDrugTable_ALL.csv"
    drug_ranking_file = drug_dir / "DrugRanking.csv"

    migraine_genes_file = files_dir / "migraine_genes.csv"
    if not migraine_genes_file.exists():
        migraine_genes_file = base_dir / "migraine_genes.csv"

    phenotype_drugs_file = base_dir / f"{phenotype}_drugs.csv"
    if not phenotype_drugs_file.exists():
        phenotype_drugs_file = base_dir / "migraine_drugs.csv"

    return {
        "base_path": base_path,
        "files_dir": files_dir,
        "ranking_dir": ranking_dir,
        "drug_dir": drug_dir,
        "ranked_genes_file": ranked_genes_file,
        "gene_drug_file": gene_drug_file,
        "drug_ranking_file": drug_ranking_file,
        "migraine_genes_file": migraine_genes_file,
        "phenotype_drugs_file": phenotype_drugs_file,
    }


# =============================================================================
# LOAD LOCAL FILES
# =============================================================================
def load_ranked_genes(path: Path, top_genes: int = 0) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing ranked genes file: {path}")

    df = pd.read_csv(path)
    if "Gene" not in df.columns:
        raise ValueError(f"Missing 'Gene' in {path}")

    df["Gene"] = df["Gene"].astype(str).map(strip_ensg_version)

    if "Rank" not in df.columns:
        if "Importance_Score" in df.columns:
            df = df.sort_values("Importance_Score", ascending=False).reset_index(drop=True)
            df["Rank"] = np.arange(1, len(df) + 1)
        else:
            df["Rank"] = np.arange(1, len(df) + 1)

    for col, default in [
        ("Symbol", ""),
        ("Direction", "unclear"),
        ("Status", ""),
        ("Confidence_Tier", ""),
        ("Direction_Consistency", np.nan),
        ("Total_Hits", np.nan),
        ("N_Tissues", np.nan),
        ("N_Databases", np.nan),
        ("N_Methods", np.nan),
        ("Importance_Score", np.nan),
        ("Mean_Unified_Effect", np.nan),
        ("Max_Unified_Effect", np.nan),
        ("Min_FDR", np.nan),
        ("Mean_FDR", np.nan),
        ("Median_FDR", np.nan),
    ]:
        df = ensure_col(df, col, default)

    df["DiseaseDirection"] = df["Direction"].map(normalize_gene_direction)

    keep = [
        "Gene", "Symbol", "Rank", "Importance_Score", "Direction", "DiseaseDirection",
        "Status", "Confidence_Tier", "Direction_Consistency",
        "Total_Hits", "N_Tissues", "N_Databases", "N_Methods",
        "Mean_Unified_Effect", "Max_Unified_Effect",
        "Min_FDR", "Mean_FDR", "Median_FDR",
    ]
    df = df[keep].copy()
    df = df.sort_values("Rank", ascending=True).reset_index(drop=True)

    if top_genes and top_genes > 0:
        df = df.head(top_genes).copy()

    return df


def load_drug_ranking(path: Path, top_drugs: int = 0) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing drug ranking file: {path}")

    df = pd.read_csv(path)

    drug_name_col = pick_first_existing(df, ["DrugName_best", "DrugName", "drug_name", "Drug"])
    if drug_name_col is None:
        raise ValueError(f"Could not find drug name column in {path}")

    df["DrugName"] = df[drug_name_col].astype(str).map(safe_str)
    df["DrugNorm"] = df["DrugName"].map(normalize_drug_name)

    if "DrugRank" not in df.columns:
        if "DrugScore" in df.columns:
            df = df.sort_values("DrugScore", ascending=False).reset_index(drop=True)
        df["DrugRank"] = np.arange(1, len(df) + 1)

    df = ensure_col(df, "DrugScore", np.nan)
    df = ensure_col(df, "N_target_genes", np.nan)
    df = ensure_col(df, "Best_gene_rank", np.nan)
    df = ensure_col(df, "Max_phase", "")
    df = ensure_col(df, "Sources", "")

    keep = [
        "DrugRank", "DrugName", "DrugNorm", "DrugScore",
        "N_target_genes", "Best_gene_rank", "Max_phase", "Sources",
    ]
    df = df[keep].copy()
    df = df.sort_values("DrugRank", ascending=True).reset_index(drop=True)

    if top_drugs and top_drugs > 0:
        df = df.head(top_drugs).copy()

    return df


def load_gene_drug_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing gene-drug table: {path}")

    df = pd.read_csv(path)

    # required-ish
    if "Gene" not in df.columns:
        raise ValueError(f"Missing 'Gene' in {path}")

    drug_name_col = pick_first_existing(df, ["DrugName", "drug_name", "Drug", "drug"])
    if drug_name_col is None:
        raise ValueError(f"Could not find drug name column in {path}")

    df["Gene"] = df["Gene"].astype(str).map(strip_ensg_version)
    df["DrugName"] = df[drug_name_col].astype(str).map(safe_str)
    df["DrugNorm"] = df["DrugName"].map(normalize_drug_name)

    for col, default in [
        ("MatchedGene", ""),
        ("InteractionType", ""),
        ("Directionality", ""),
        ("MoA", ""),
        ("TargetEvidence", ""),
        ("Source", ""),
        ("Phase", ""),
        ("EvidencePhase", ""),
        ("Status", ""),
        ("CtIds", ""),
        ("PMIDs", ""),
        ("DiseaseName", ""),
        ("IsApproved", ""),
    ]:
        df = ensure_col(df, col, default)

    return df


def load_known_genes(path: Path) -> set:
    if not path.exists():
        return set()

    df = pd.read_csv(path)
    if "ensembl_gene_id" not in df.columns:
        return set()

    genes = set(df["ensembl_gene_id"].dropna().astype(str).map(strip_ensg_version))
    return {g for g in genes if g}


def load_known_drugs(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["DrugNorm", "KnownMigraineDrug", "KnownDrugEvidence"])

    df = pd.read_csv(path)
    drug_col = pick_first_existing(df, ["drug_name", "DrugName", "drug", "Drug"])
    if drug_col is None:
        return pd.DataFrame(columns=["DrugNorm", "KnownMigraineDrug", "KnownDrugEvidence"])

    df["DrugName_ref"] = df[drug_col].astype(str).map(safe_str)
    df["DrugNorm"] = df["DrugName_ref"].map(normalize_drug_name)
    df = df[df["DrugNorm"] != ""].copy()

    df["KnownMigraineDrug"] = "Yes"

    extra_cols = [c for c in df.columns if c not in {drug_col, "DrugName_ref", "DrugNorm"}]
    if extra_cols:
        def combine_row(row):
            parts = []
            for c in extra_cols:
                v = safe_str(row.get(c, ""))
                if v:
                    parts.append(f"{c}={v}")
            return " | ".join(parts)
        df["KnownDrugEvidence"] = df.apply(combine_row, axis=1)
    else:
        df["KnownDrugEvidence"] = ""

    return df[["DrugNorm", "KnownMigraineDrug", "KnownDrugEvidence"]].drop_duplicates("DrugNorm")


# =============================================================================
# ONLINE RESCUE CACHE
# =============================================================================
def load_json_cache(path: Path) -> Dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def save_json_cache(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2))


# =============================================================================
# ChEMBL RESCUE
# =============================================================================
def chembl_search_molecule(session: requests.Session, drug_name: str) -> List[Dict[str, Any]]:
    url = f"{CHEMBL_API}/molecule/search.json"
    r = session.get(url, params={"q": drug_name, "limit": 10}, timeout=REQUEST_TIMEOUT)
    if r.status_code != 200:
        return []
    data = r.json()
    return data.get("molecules", []) or []


def chembl_search_targets_for_gene(session: requests.Session, gene_symbol: str) -> List[Dict[str, Any]]:
    url = f"{CHEMBL_API}/target/search.json"
    r = session.get(url, params={"q": gene_symbol, "limit": 20}, timeout=REQUEST_TIMEOUT)
    if r.status_code != 200:
        return []
    data = r.json()
    return data.get("targets", []) or []


def chembl_mechanisms_for_molecule(session: requests.Session, molecule_chembl_id: str) -> List[Dict[str, Any]]:
    url = f"{CHEMBL_API}/mechanism.json"
    r = session.get(
        url,
        params={"molecule_chembl_id": molecule_chembl_id, "limit": 1000},
        timeout=REQUEST_TIMEOUT,
    )
    if r.status_code != 200:
        return []
    data = r.json()
    return data.get("mechanisms", []) or []


def chembl_pick_best_molecule(candidates: List[Dict[str, Any]], drug_name: str) -> Optional[Dict[str, Any]]:
    if not candidates:
        return None

    norm_query = normalize_drug_name(drug_name)

    scored = []
    for m in candidates:
        pref = safe_str(m.get("pref_name"))
        synonyms = " ".join([safe_str(x) for x in m.get("molecule_synonyms", [])]) if isinstance(m.get("molecule_synonyms"), list) else ""
        combined = f"{pref} {synonyms}"
        score = 0
        if normalize_drug_name(pref) == norm_query:
            score += 100
        if norm_query and norm_query in normalize_drug_name(combined):
            score += 20
        if safe_str(m.get("max_phase")):
            score += 2
        scored.append((score, m))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def chembl_rescue_pair(
    session: requests.Session,
    drug_name: str,
    gene_symbol: str,
    cache: Dict[str, Any],
) -> Dict[str, str]:
    """
    Return:
      ActionType
      ActionEvidenceSource
      ActionEvidenceText
      RescueStatus
    """
    cache_key = f"chembl::{normalize_drug_name(drug_name)}::{safe_str(gene_symbol).upper()}"
    if cache_key in cache:
        return cache[cache_key]

    out = {
        "ActionType": "unknown",
        "ActionEvidenceSource": "",
        "ActionEvidenceText": "",
        "RescueStatus": "Unresolved",
    }

    gene_symbol = safe_str(gene_symbol)
    if not drug_name or not gene_symbol:
        cache[cache_key] = out
        return out

    try:
        mol_candidates = chembl_search_molecule(session, drug_name)
        best_mol = chembl_pick_best_molecule(mol_candidates, drug_name)
        if best_mol is None:
            cache[cache_key] = out
            return out

        molecule_id = safe_str(best_mol.get("molecule_chembl_id"))
        if not molecule_id:
            cache[cache_key] = out
            return out

        mechs = chembl_mechanisms_for_molecule(session, molecule_id)
        if not mechs:
            cache[cache_key] = out
            return out

        target_hits = chembl_search_targets_for_gene(session, gene_symbol)
        target_ids = {safe_str(t.get("target_chembl_id")) for t in target_hits if safe_str(t.get("target_chembl_id"))}

        best_record = None
        # first prefer target-id match
        for mech in mechs:
            mech_tid = safe_str(mech.get("target_chembl_id"))
            if mech_tid and mech_tid in target_ids:
                best_record = mech
                break

        # fallback: try matching gene symbol in target name / mechanism text
        if best_record is None:
            gs = gene_symbol.lower()
            for mech in mechs:
                text_blob = " ".join(
                    [
                        safe_str(mech.get("action_type")),
                        safe_str(mech.get("mechanism_of_action")),
                        safe_str(mech.get("target_chembl_id")),
                        safe_str(mech.get("site_name")),
                    ]
                ).lower()
                if gs in text_blob:
                    best_record = mech
                    break

        if best_record is None:
            cache[cache_key] = out
            return out

        action_type_raw = safe_str(best_record.get("action_type"))
        moa_raw = safe_str(best_record.get("mechanism_of_action"))
        action_text = " | ".join([x for x in [action_type_raw, moa_raw] if x])

        action_type = classify_action_type(action_text)

        if action_type != "unknown":
            out = {
                "ActionType": action_type,
                "ActionEvidenceSource": "ChEMBL",
                "ActionEvidenceText": action_text,
                "RescueStatus": "Resolved_online_ChEMBL",
            }

    except Exception:
        pass

    cache[cache_key] = out
    return out


# =============================================================================
# DGIdb RESCUE
# =============================================================================
def dgidb_query_gene(session: requests.Session, gene_symbol: str) -> List[Dict[str, Any]]:
    gene_symbol = safe_str(gene_symbol)
    if not gene_symbol:
        return []

    query = f"""
    query {{
      geneMatches(searchTerms: ["{gene_symbol}"]) {{
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

    try:
        r = session.post(DGIDB_GQL, json={"query": query}, timeout=REQUEST_TIMEOUT)
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

                    itype = itypes[0] if itypes else {}
                    out.append(
                        {
                            "DrugName": safe_str(drug.get("name")),
                            "DrugNorm": normalize_drug_name(drug.get("name")),
                            "MatchedGene": matched_gene,
                            "InteractionType": safe_str(itype.get("type")),
                            "Directionality": safe_str(itype.get("directionality")),
                            "SourcesDetail": "|".join(
                                [safe_str(s.get("sourceDbName")) for s in srcs if safe_str(s.get("sourceDbName"))]
                            ),
                            "PMIDs": "|".join(
                                [safe_str(p.get("pmid")) for p in pubs if safe_str(p.get("pmid"))]
                            ),
                            "Approved": "Yes" if drug.get("approved") is True else "",
                        }
                    )
        return out
    except Exception:
        return []


def dgidb_rescue_pair(
    session: requests.Session,
    drug_name: str,
    gene_symbol: str,
    cache: Dict[str, Any],
) -> Dict[str, str]:
    cache_key = f"dgidb::{normalize_drug_name(drug_name)}::{safe_str(gene_symbol).upper()}"
    if cache_key in cache:
        return cache[cache_key]

    out = {
        "ActionType": "unknown",
        "ActionEvidenceSource": "",
        "ActionEvidenceText": "",
        "RescueStatus": "Unresolved",
    }

    gene_symbol = safe_str(gene_symbol)
    if not drug_name or not gene_symbol:
        cache[cache_key] = out
        return out

    try:
        rows = dgidb_query_gene(session, gene_symbol)
        dn = normalize_drug_name(drug_name)
        rows = [r for r in rows if r.get("DrugNorm") == dn]

        if rows:
            best = rows[0]
            text = " | ".join(
                [
                    safe_str(best.get("InteractionType")),
                    safe_str(best.get("Directionality")),
                    safe_str(best.get("SourcesDetail")),
                ]
            )
            action_type = classify_action_type(text)

            if action_type != "unknown":
                out = {
                    "ActionType": action_type,
                    "ActionEvidenceSource": "DGIdb",
                    "ActionEvidenceText": text,
                    "RescueStatus": "Resolved_online_DGIdb",
                }

    except Exception:
        pass

    cache[cache_key] = out
    return out


# =============================================================================
# BUILD UNIQUE PAIR TABLE
# =============================================================================
def build_pair_level_table(
    ranked: pd.DataFrame,
    gene_drug: pd.DataFrame,
    drug_rank: pd.DataFrame,
    known_genes: set,
    known_drugs_df: pd.DataFrame,
) -> pd.DataFrame:
    # merge drug rank + local gene-drug evidence + gene metadata
    merged = drug_rank.merge(gene_drug, on="DrugNorm", how="left", suffixes=("_rank", "_ev"))

    # use drug name from ranking if available
    merged["DrugName_final"] = merged.apply(
        lambda r: first_nonempty([r.get("DrugName_rank", ""), r.get("DrugName", ""), r.get("DrugName_ev", "")]),
        axis=1,
    )

    merged = merged.merge(ranked, on="Gene", how="left", suffixes=("", "_gene"))
    if not known_drugs_df.empty:
        merged = merged.merge(known_drugs_df, on="DrugNorm", how="left")
    else:
        merged["KnownMigraineDrug"] = "No"
        merged["KnownDrugEvidence"] = ""

    merged["KnownMigraineDrug"] = merged["KnownMigraineDrug"].fillna("No")
    merged["KnownDrugEvidence"] = merged["KnownDrugEvidence"].fillna("")
    merged["KnownMigraineGene"] = merged["Gene"].isin(known_genes).map({True: "Yes", False: "No"})

    # local action text
    merged["LocalActionText"] = merged.apply(
        lambda r: " | ".join(
            [
                x for x in [
                    safe_str(r.get("MoA", "")),
                    safe_str(r.get("InteractionType", "")),
                    safe_str(r.get("Directionality", "")),
                    safe_str(r.get("TargetEvidence", "")),
                ] if x
            ]
        ),
        axis=1,
    )
    merged["LocalActionType"] = merged["LocalActionText"].map(classify_action_type)

    # aggregate to unique DrugNorm-Gene pairs
    group_cols = ["DrugNorm", "Gene"]
    rows = []
    for (drug_norm, gene), sub in merged.groupby(group_cols, dropna=False):
        sub = sub.copy()

        drug_name = first_nonempty(sub["DrugName_final"].tolist())
        symbol = first_nonempty(sub["Symbol"].tolist())
        disease_direction = first_nonempty(sub["DiseaseDirection"].tolist(), "unclear")
        local_action_type = first_nonempty([x for x in sub["LocalActionType"].tolist() if safe_str(x) != "unknown"], "unknown")

        row = {
            "DrugRank": pd.to_numeric(sub["DrugRank"], errors="coerce").min(),
            "DrugName": drug_name,
            "DrugNorm": drug_norm,
            "DrugScore": pd.to_numeric(sub["DrugScore"], errors="coerce").max(),
            "Approved": first_nonempty(
                [infer_approved_flag(r) for _, r in sub.iterrows()]
            ),
            "KnownMigraineDrug": first_nonempty(sub["KnownMigraineDrug"].tolist(), "No"),
            "KnownDrugEvidence": unique_join(sub["KnownDrugEvidence"].tolist()),
            "Gene": gene,
            "Symbol": symbol,
            "GeneRank": pd.to_numeric(sub["Rank"], errors="coerce").min(),
            "Importance_Score": pd.to_numeric(sub["Importance_Score"], errors="coerce").max(),
            "KnownMigraineGene": first_nonempty(sub["KnownMigraineGene"].tolist(), "No"),
            "DiseaseDirection": disease_direction,
            "ActionType": local_action_type,
            "ActionEvidenceSource": "Local" if local_action_type != "unknown" else "",
            "ActionEvidenceText": unique_join(sub["LocalActionText"].tolist()),
            "DirectionMatch": classify_direction_match(disease_direction, local_action_type),
            "LocalSources": unique_join(sub["Source"].tolist()),
            "LocalInteractionType": unique_join(sub["InteractionType"].tolist()),
            "LocalDirectionality": unique_join(sub["Directionality"].tolist()),
            "LocalMoA": unique_join(sub["MoA"].tolist()),
            "TargetEvidence": unique_join(sub["TargetEvidence"].tolist()),
            "CtIds": unique_join(sub["CtIds"].tolist()),
            "PMIDs": unique_join(sub["PMIDs"].tolist()),
            "Status": first_nonempty(sub["Status"].tolist()),
            "Confidence_Tier": first_nonempty(sub["Confidence_Tier"].tolist()),
            "Direction_Consistency": pd.to_numeric(sub["Direction_Consistency"], errors="coerce").max(),
            "Total_Hits": pd.to_numeric(sub["Total_Hits"], errors="coerce").max(),
            "N_Tissues": pd.to_numeric(sub["N_Tissues"], errors="coerce").max(),
            "N_Databases": pd.to_numeric(sub["N_Databases"], errors="coerce").max(),
            "N_Methods": pd.to_numeric(sub["N_Methods"], errors="coerce").max(),
            "Mean_Unified_Effect": pd.to_numeric(sub["Mean_Unified_Effect"], errors="coerce").max(),
            "Max_Unified_Effect": pd.to_numeric(sub["Max_Unified_Effect"], errors="coerce").max(),
            "Min_FDR": pd.to_numeric(sub["Min_FDR"], errors="coerce").min(),
            "Mean_FDR": pd.to_numeric(sub["Mean_FDR"], errors="coerce").min(),
            "Median_FDR": pd.to_numeric(sub["Median_FDR"], errors="coerce").min(),
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values(["DrugRank", "GeneRank"], ascending=[True, True]).reset_index(drop=True)
    return out


# =============================================================================
# RESCUE MISSING PAIRS
# =============================================================================
def rescue_missing_pairs(
    df: pd.DataFrame,
    drug_dir: Path,
    enable_online: bool = True,
) -> pd.DataFrame:
    cache_file = drug_dir / "directionality_online_cache.json"
    cache = load_json_cache(cache_file)

    df = df.copy()
    df["RescueStatus"] = np.where(df["ActionType"] != "unknown", "Resolved_local", "Pending")

    if not enable_online:
        df.loc[df["ActionType"] == "unknown", "RescueStatus"] = "Unresolved"
        return df

    session = requests_session()

    n_total = len(df)
    unresolved_idx = df.index[df["ActionType"] == "unknown"].tolist()

    print(f"\n?? Online rescue for missing action info")
    print(f"   Total unique pairs: {n_total}")
    print(f"   Pairs needing rescue: {len(unresolved_idx)}")

    for i, idx in enumerate(unresolved_idx, start=1):
        drug_name = safe_str(df.at[idx, "DrugName"])
        gene_symbol = safe_str(df.at[idx, "Symbol"])

        # skip if no symbol
        if not gene_symbol:
            df.at[idx, "RescueStatus"] = "Unresolved_no_symbol"
            continue

        # try ChEMBL first
        chembl_hit = chembl_rescue_pair(session, drug_name, gene_symbol, cache)
        if chembl_hit["ActionType"] != "unknown":
            df.at[idx, "ActionType"] = chembl_hit["ActionType"]
            df.at[idx, "ActionEvidenceSource"] = chembl_hit["ActionEvidenceSource"]
            existing_text = safe_str(df.at[idx, "ActionEvidenceText"])
            df.at[idx, "ActionEvidenceText"] = unique_join([existing_text, chembl_hit["ActionEvidenceText"]])
            df.at[idx, "RescueStatus"] = chembl_hit["RescueStatus"]
            df.at[idx, "DirectionMatch"] = classify_direction_match(df.at[idx, "DiseaseDirection"], df.at[idx, "ActionType"])
            if i % 25 == 0:
                save_json_cache(cache_file, cache)
            time.sleep(SLEEP_BETWEEN_REMOTE_CALLS)
            continue

        # then DGIdb
        dgidb_hit = dgidb_rescue_pair(session, drug_name, gene_symbol, cache)
        if dgidb_hit["ActionType"] != "unknown":
            df.at[idx, "ActionType"] = dgidb_hit["ActionType"]
            df.at[idx, "ActionEvidenceSource"] = dgidb_hit["ActionEvidenceSource"]
            existing_text = safe_str(df.at[idx, "ActionEvidenceText"])
            df.at[idx, "ActionEvidenceText"] = unique_join([existing_text, dgidb_hit["ActionEvidenceText"]])
            df.at[idx, "RescueStatus"] = dgidb_hit["RescueStatus"]
            df.at[idx, "DirectionMatch"] = classify_direction_match(df.at[idx, "DiseaseDirection"], df.at[idx, "ActionType"])
        else:
            df.at[idx, "RescueStatus"] = "Unresolved"

        if i % 25 == 0:
            print(f"   processed {i}/{len(unresolved_idx)} unresolved pairs")
            save_json_cache(cache_file, cache)

        time.sleep(SLEEP_BETWEEN_REMOTE_CALLS)

    save_json_cache(cache_file, cache)
    return df


# =============================================================================
# SAVE OUTPUTS
# =============================================================================
def save_outputs(df: pd.DataFrame, drug_dir: Path) -> None:
    drug_dir.mkdir(parents=True, exist_ok=True)

    out_main = drug_dir / "DrugDirectionalityTable_Rescued.csv"
    out_summary = drug_dir / "DrugDirectionalitySummary_Rescued.csv"
    out_consistent = drug_dir / "DrugDirectionalityTopConsistent_Rescued.csv"
    out_unresolved = drug_dir / "DrugDirectionalityUnresolved_Rescued.csv"
    out_maintext = drug_dir / "DrugDirectionalityMainTextTable.csv"

    df = df.sort_values(["DrugRank", "GeneRank"], ascending=[True, True]).reset_index(drop=True)
    df.to_csv(out_main, index=False)

    summary = (
        df.groupby("DirectionMatch", dropna=False)
        .size()
        .reset_index(name="N_pairs")
        .sort_values("N_pairs", ascending=False)
        .reset_index(drop=True)
    )
    summary.to_csv(out_summary, index=False)

    top_consistent = df[df["DirectionMatch"] == "consistent"].copy()
    top_consistent = top_consistent.sort_values(
        ["DrugRank", "Approved", "KnownMigraineDrug", "GeneRank"],
        ascending=[True, True, True, True]
    )
    top_consistent.to_csv(out_consistent, index=False)

    unresolved = df[df["ActionType"] == "unknown"].copy()
    unresolved.to_csv(out_unresolved, index=False)

    maintext = df[
        [
            "DrugRank", "DrugName", "Gene", "Symbol", "GeneRank",
            "DiseaseDirection", "ActionType", "DirectionMatch",
            "Approved", "KnownMigraineDrug", "KnownMigraineGene",
            "ActionEvidenceSource", "ActionEvidenceText",
        ]
    ].copy()
    maintext = maintext.sort_values(["DrugRank", "GeneRank"], ascending=[True, True])
    maintext.to_csv(out_maintext, index=False)

    print("\n? OUTPUTS WRITTEN")
    print(f"   {out_main}")
    print(f"   {out_summary}")
    print(f"   {out_consistent}")
    print(f"   {out_unresolved}")
    print(f"   {out_maintext}")

    print("\n?? SUMMARY")
    if len(summary) > 0:
        print(summary.to_string(index=False))
    else:
        print("No rows found.")


# =============================================================================
# MAIN
# =============================================================================
def build_directionality_table(
    phenotype: str,
    base_dir: Path,
    top_drugs: int,
    top_genes: int,
    enable_online: bool,
) -> pd.DataFrame:
    paths = resolve_paths(phenotype, base_dir)

    print("=" * 120)
    print("DRUG DIRECTIONALITY RESCUE PIPELINE")
    print("=" * 120)
    print(f"Phenotype:              {phenotype}")
    print(f"Base directory:         {base_dir}")
    print(f"Ranked genes file:      {paths['ranked_genes_file']}")
    print(f"Gene-drug evidence:     {paths['gene_drug_file']}")
    print(f"Drug ranking file:      {paths['drug_ranking_file']}")
    print(f"Known genes file:       {paths['migraine_genes_file']}")
    print(f"Known drugs file:       {paths['phenotype_drugs_file']}")
    print("=" * 120)

    ranked = load_ranked_genes(paths["ranked_genes_file"], top_genes=top_genes)
    gene_drug = load_gene_drug_table(paths["gene_drug_file"])
    drug_rank = load_drug_ranking(paths["drug_ranking_file"], top_drugs=top_drugs)
    known_genes = load_known_genes(paths["migraine_genes_file"])
    known_drugs_df = load_known_drugs(paths["phenotype_drugs_file"])

    # Restrict gene_drug to selected top drugs and top genes if possible
    selected_drugs = set(drug_rank["DrugNorm"].dropna().tolist())
    selected_genes = set(ranked["Gene"].dropna().tolist())

    gene_drug = gene_drug[gene_drug["DrugNorm"].isin(selected_drugs)].copy()
    gene_drug = gene_drug[gene_drug["Gene"].isin(selected_genes)].copy()

    print(f"\n?? Loaded ranked genes: {len(ranked):,}")
    print(f"?? Loaded unique ranked drugs: {len(drug_rank):,}")
    print(f"?? Loaded gene-drug evidence rows after restriction: {len(gene_drug):,}")

    pair_df = build_pair_level_table(
        ranked=ranked,
        gene_drug=gene_drug,
        drug_rank=drug_rank,
        known_genes=known_genes,
        known_drugs_df=known_drugs_df,
    )

    print(f"?? Unique drug-gene pairs before rescue: {len(pair_df):,}")

    pair_df = rescue_missing_pairs(
        df=pair_df,
        drug_dir=paths["drug_dir"],
        enable_online=enable_online,
    )

    save_outputs(pair_df, paths["drug_dir"])
    return pair_df


def main():
    parser = argparse.ArgumentParser(description="Build rescued drug-gene directionality table.")
    parser.add_argument("phenotype", help="Phenotype name, e.g. migraine")
    parser.add_argument("--base-dir", default=str(BASE_DIR_DEFAULT), help="Base directory")
    parser.add_argument("--top-drugs", type=int, default=100, help="Top N drugs from DrugRanking.csv")
    parser.add_argument("--top-genes", type=int, default=500, help="Top N genes from RANKED_composite.csv")
    parser.add_argument("--no-online", action="store_true", help="Disable online rescue")
    args = parser.parse_args()

    build_directionality_table(
        phenotype=args.phenotype,
        base_dir=Path(args.base_dir),
        top_drugs=args.top_drugs,
        top_genes=args.top_genes,
        enable_online=not args.no_online,
    )


if __name__ == "__main__":
    main()