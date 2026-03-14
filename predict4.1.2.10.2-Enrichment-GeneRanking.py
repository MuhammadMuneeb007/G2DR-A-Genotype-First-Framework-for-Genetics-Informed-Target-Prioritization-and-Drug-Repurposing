#!/usr/bin/env python3
"""
predict4.1.2.10.3-Enrichment-GeneRanking-OpenTargetsAPI.py

Example:
python predict4.1.2.10.3-Enrichment-GeneRanking-OpenTargetsAPI.py migraine
python predict4.1.2.10.3-Enrichment-GeneRanking-OpenTargetsAPI.py migraine --use-ot-api
python predict4.1.2.10.3-Enrichment-GeneRanking-OpenTargetsAPI.py migraine --use-ot-api --include-alt-integrated
"""

import argparse
from pathlib import Path
import warnings
import time
import hashlib
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

OT_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def strip_ensg_version(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.startswith("ENSG") and "." in s:
        return s.split(".")[0]
    return s


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def fmt_metric(val) -> str:
    if pd.isna(val):
        return "NA"
    return f"{val:.4f}"


def safe_numeric(s):
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def pct_rank(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    ranked = s.rank(pct=True, method="average").fillna(0.0)
    ranked[(s.isna()) | (s <= 0)] = 0.0
    return ranked.astype(float)


def safe_neglog10(series: pd.Series, eps: float = 1e-300) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    s = s.clip(lower=eps)
    return -np.log10(s)


def _collect_compat(ldf: pl.LazyFrame) -> pd.DataFrame:
    try:
        return ldf.collect(engine="streaming").to_pandas()
    except TypeError:
        return ldf.collect().to_pandas()


def auc_pr(y_true: np.ndarray, score: np.ndarray):
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return np.nan, np.nan
    try:
        return roc_auc_score(y_true, score), average_precision_score(y_true, score)
    except Exception:
        return np.nan, np.nan


def print_block(title: str):
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)


def first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def existing_cols(df: pd.DataFrame, candidates: List[str]) -> List[str]:
    lower_map = {c.lower(): c for c in df.columns}
    out = []
    for cand in candidates:
        if cand in df.columns:
            out.append(cand)
        elif cand.lower() in lower_map:
            out.append(lower_map[cand.lower()])
    seen = set()
    final = []
    for x in out:
        if x not in seen:
            seen.add(x)
            final.append(x)
    return final


def first_existing_name(names: List[str], candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in names}
    for cand in candidates:
        if cand in names:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def normalize_colname(c: str) -> str:
    return str(c).strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")


def find_gene_col(df: pd.DataFrame) -> str:
    candidates = ["Gene", "gene", "ensembl_gene_id", "GeneID", "gene_id"]
    col = first_existing_col(df, candidates)
    if col is None:
        raise ValueError(f"Could not detect gene column. Found columns: {list(df.columns)}")
    return col


def detect_volcano_gene_column(cols: List[str]) -> str:
    preferred = ["Gene", "gene", "GeneID", "ensembl_gene_id"]
    found = first_existing_name(cols, preferred)
    if found is None:
        raise ValueError(f"Could not detect gene column in volcano file. Found columns: {cols}")
    return found


def detect_volcano_split_column(cols: List[str]) -> str:
    preferred = ["Dataset", "Database", "Tissue", "Method", "Fold"]
    found = first_existing_name(cols, preferred)
    if found is None:
        raise ValueError(
            f"Could not detect a split column in volcano file. "
            f"Tried one of {preferred}, but found columns: {cols}"
        )
    return found


def detect_drug_name_cols(df: pd.DataFrame) -> List[str]:
    exact = existing_cols(df, [
        "Drug", "drug", "DrugName", "drug_name", "compound_name",
        "molecule_name", "Compound", "compound", "druglabel",
        "drug_label", "approved_drug", "approved_drug_name"
    ])

    fuzzy = []
    for c in df.columns:
        n = normalize_colname(c)
        if any(k in n for k in [
            "drug", "compound", "molecule", "therapy", "treatment", "medication"
        ]):
            fuzzy.append(c)

    out = []
    seen = set()
    for c in exact + fuzzy:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def detect_evidence_cols(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Returns:
        direct_cols: evidence columns usable for direct target evidence
        ot_cols: columns specifically related to Open Targets
    """
    direct_exact = existing_cols(df, [
        "DGIdb_Evidence_Count", "dgidb_evidence_count",
        "ChEMBL_Evidence_Count", "chembl_evidence_count",
        "OpenTargets_Evidence_Count", "opentargets_evidence_count",
        "Known_Target_Evidence", "known_target_evidence",
        "Evidence_Count", "evidence_count",
        "n_interactions", "N_Interactions", "interaction_count",
        "ChEMBL_Interactions", "DGIdb_Interactions",
        "DGIdb_Score", "dgidb_score",
        "ChEMBL_Score", "chembl_score",
        "target_score", "Target_Score",
        "target_evidence", "Target_Evidence",
        "is_target", "Is_Target"
    ])

    ot_exact = existing_cols(df, [
        "OpenTargets_Evidence_Count", "opentargets_evidence_count",
        "OpenTargets_Score", "opentargets_score",
        "OT_Score", "ot_score",
        "OpenTargets_Association", "opentargets_association"
    ])

    direct_fuzzy = []
    ot_fuzzy = []

    for c in df.columns:
        n = normalize_colname(c)

        if any(k in n for k in [
            "evidence", "interaction", "target", "dgidb", "chembl"
        ]):
            if not any(bad in n for bad in ["type", "class", "source_name", "description", "name"]):
                direct_fuzzy.append(c)

        if "opentarget" in n or n.startswith("ot_") or "_ot_" in n:
            if not any(bad in n for bad in ["type", "class", "source_name", "description", "name"]):
                ot_fuzzy.append(c)

    def keep_numericish(cols):
        good = []
        for c in cols:
            vals = pd.to_numeric(df[c], errors="coerce")
            if vals.notna().sum() > 0:
                good.append(c)
        return good

    direct_cols = keep_numericish(direct_exact + direct_fuzzy)
    ot_cols = keep_numericish(ot_exact + ot_fuzzy)

    def uniq(lst):
        seen = set()
        out = []
        for x in lst:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    return uniq(direct_cols), uniq(ot_cols)


def build_fallback_binary_signal(df: pd.DataFrame, cols: List[str], new_col: str) -> pd.DataFrame:
    tmp = df[["Gene"] + cols].copy()
    score = np.zeros(len(tmp), dtype=float)

    for c in cols:
        s = tmp[c]
        num = pd.to_numeric(s, errors="coerce")
        if num.notna().sum() > 0:
            score += num.fillna(0).to_numpy()
            continue

        txt = s.astype(str).str.strip().str.lower()
        pos = txt.isin(["yes", "true", "1", "y", "t", "present", "linked", "known"])
        score += pos.astype(float).to_numpy()

    tmp[new_col] = score
    tmp = tmp.groupby("Gene", as_index=False)[new_col].max()
    return tmp


def topk_stats(ranked_genes, positives: set, universe_n: int, k: int):
    k = min(int(k), len(ranked_genes))
    top = set(ranked_genes[:k])
    obs = len(top & positives)
    exp = k * (len(positives) / universe_n) if universe_n > 0 else np.nan
    fe = (obs / exp) if exp and exp > 0 else np.nan
    prec = obs / k if k > 0 else np.nan
    rec = obs / len(positives) if len(positives) > 0 else np.nan
    return obs, exp, fe, prec, rec


def print_topk_table(ranked_genes, positives: set, universe_n: int, topk_list: list, label: str):
    print(f"\n  {label}")
    hdr = f"{'TopK':>6}  {'Observed':>9}  {'Expected':>9}  {'FE':>7}  {'Precision':>10}  {'Recall':>10}"
    print(hdr)
    print("-" * len(hdr))
    for k in topk_list:
        obs, exp, fe, prec, rec = topk_stats(ranked_genes, positives, universe_n, k)
        exp_s = f"{exp:.2f}" if pd.notna(exp) else "NA"
        fe_s = f"{fe:.2f}" if pd.notna(fe) else "NA"
        print(f"{int(k):>6}  {obs:>9}  {exp_s:>9}  {fe_s:>7}  {prec:>10.4f}  {rec:>10.4f}")


# --------------------------------------------------------------------------------------
# HTTP / Open Targets
# --------------------------------------------------------------------------------------
def make_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"],
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


SESSION = make_session()


def ot_post(query: str, variables: dict, timeout: int = 60) -> dict:
    r = SESSION.post(
        OT_GRAPHQL_URL,
        json={"query": query, "variables": variables},
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    if "errors" in data:
        raise RuntimeError(f"Open Targets GraphQL error: {data['errors']}")
    return data


def resolve_disease_id_from_name(disease_name: str) -> str:
    query = """
    query searchDisease($term: String!) {
      search(queryString: $term, entityNames: ["disease"], page: { index: 0, size: 20 }) {
        hits {
          id
          entity
          name
          description
        }
      }
    }
    """
    data = ot_post(query, {"term": disease_name})
    hits = data.get("data", {}).get("search", {}).get("hits", [])

    if not hits:
        raise RuntimeError(f"No Open Targets disease match found for phenotype: {disease_name}")

    disease_name_lc = disease_name.strip().lower()
    best = None
    for hit in hits:
        name = str(hit.get("name", "")).strip().lower()
        if name == disease_name_lc:
            best = hit
            break
    if best is None:
        best = hits[0]

    disease_id = best.get("id", "")
    disease_label = best.get("name", "")
    if not disease_id:
        raise RuntimeError(f"Open Targets returned hits for {disease_name}, but no disease ID was found")

    print(f"   ✓ Open Targets disease matched: {disease_label} ({disease_id})")
    return disease_id


def fetch_ot_association_for_gene(gene_id: str, disease_id: str, sleep_s: float = 0.03) -> float:
    query = """
    query targetAssociations($ensemblId: String!) {
      target(ensemblId: $ensemblId) {
        associatedDiseases(page: {index: 0, size: 500}) {
          rows {
            disease {
              id
              name
            }
            score
          }
        }
      }
    }
    """
    try:
        data = ot_post(query, {"ensemblId": gene_id})
        rows = data.get("data", {}).get("target", {}).get("associatedDiseases", {}).get("rows", [])
        for row in rows:
            disease = row.get("disease", {})
            if disease.get("id") == disease_id:
                time.sleep(sleep_s)
                return float(row.get("score", 0.0) or 0.0)
        time.sleep(sleep_s)
        return 0.0
    except Exception:
        time.sleep(sleep_s)
        return 0.0


def load_or_fetch_ot_scores(
    genes: List[str],
    disease_name: str,
    cache_file: Path,
    max_genes: Optional[int] = None,
) -> pd.DataFrame:
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    disease_id = resolve_disease_id_from_name(disease_name)

    if cache_file.exists():
        cached = pd.read_csv(cache_file)
        if {"Gene", "OpenTargets_API_raw"}.issubset(cached.columns):
            cached["Gene"] = cached["Gene"].astype(str).map(strip_ensg_version)
            existing = set(cached["Gene"])
        else:
            cached = pd.DataFrame(columns=["Gene", "OpenTargets_API_raw"])
            existing = set()
    else:
        cached = pd.DataFrame(columns=["Gene", "OpenTargets_API_raw"])
        existing = set()

    genes_to_fetch = [g for g in genes if g not in existing and str(g).startswith("ENSG")]
    if max_genes is not None:
        genes_to_fetch = genes_to_fetch[:max_genes]

    if genes_to_fetch:
        print(f"\n🌐 FETCHING OPEN TARGETS SCORES for disease='{disease_name}' on {len(genes_to_fetch):,} genes")
        rows = []
        for i, g in enumerate(genes_to_fetch, start=1):
            score = fetch_ot_association_for_gene(g, disease_id=disease_id)
            rows.append({"Gene": g, "OpenTargets_API_raw": score})
            if i % 100 == 0:
                print(f"   fetched {i:,}/{len(genes_to_fetch):,}")
        new_df = pd.DataFrame(rows)
        cached = pd.concat([cached, new_df], ignore_index=True)
        cached = cached.groupby("Gene", as_index=False)["OpenTargets_API_raw"].max()
        cached.to_csv(cache_file, index=False)
        print(f"   ✓ Saved Open Targets cache: {cache_file}")
    else:
        print("\n🌐 OPEN TARGETS CACHE already complete")

    cached["Gene"] = cached["Gene"].astype(str).map(strip_ensg_version)
    cached["OpenTargets_API_raw"] = safe_numeric(cached["OpenTargets_API_raw"])
    return cached


# --------------------------------------------------------------------------------------
# Positives / disease genes / volcano summary
# --------------------------------------------------------------------------------------
def volcano_universe_and_positives(
    volcano_path: Path,
    split_keywords: list,
    fdr_thr: float,
    effect_thr: float
):
    print(f"\n📖 READING: {volcano_path}")
    ldf = pl.scan_csv(str(volcano_path), infer_schema_length=200)
    cols = ldf.collect_schema().names()
    print(f"   ✓ Columns found: {cols[:8]}..." if len(cols) > 8 else f"   ✓ Columns: {cols}")

    gene_col = detect_volcano_gene_column(cols)
    split_col = detect_volcano_split_column(cols)

    required_like = [gene_col, split_col, "FDR", "Log2FoldChange"]
    missing = [c for c in required_like if c not in cols]
    if missing:
        raise ValueError(f"Volcano file missing required columns: {missing}")

    ldf2 = ldf.with_columns([
        pl.col(gene_col).cast(pl.Utf8).str.strip_chars().str.replace(r"\.\d+$", "").alias("Gene_clean"),
        pl.col(split_col).cast(pl.Utf8).str.to_lowercase().alias("Split_lc"),
        pl.col("FDR").cast(pl.Float64).alias("FDR_f"),
        pl.col("Log2FoldChange").cast(pl.Float64).alias("LFC_f"),
    ])

    universe_df = _collect_compat(
        ldf2.select(pl.col("Gene_clean").alias("Gene")).drop_nulls().unique()
    )
    universe = set(universe_df["Gene"].tolist())
    print(f"   ✓ Volcano universe: {len(universe):,} unique genes")
    print(f"   ✓ Split column used: {split_col}")

    split_mask = None
    for kw in split_keywords:
        m = pl.col("Split_lc").str.contains(str(kw).lower())
        split_mask = m if split_mask is None else (split_mask | m)

    pos_df = _collect_compat(
        ldf2.filter(split_mask)
            .filter((pl.col("FDR_f") < fdr_thr) & (pl.col("LFC_f").abs() >= effect_thr))
            .select(pl.col("Gene_clean").alias("Gene"))
            .drop_nulls()
            .unique()
    )
    positives = set(pos_df["Gene"].tolist())
    print(f"   ✓ Positives (keywords={split_keywords}): {len(positives):,} genes")
    return positives, universe


def load_disease_genes(path: Path, phenotype_label: str) -> set:
    print(f"\n📖 READING: {path}")
    mg = pd.read_csv(path)
    print(f"   ✓ Shape: {mg.shape}")

    gene_col = first_existing_col(mg, ["ensembl_gene_id", "Gene", "gene", "GeneID"])
    if gene_col is None:
        raise ValueError(f"{path.name} must contain one of: ensembl_gene_id, Gene, gene, GeneID")

    genes = set(mg[gene_col].astype(str).map(strip_ensg_version).tolist())
    genes = {g for g in genes if g.startswith("ENSG")}
    print(f"   ✓ {phenotype_label} genes loaded: {len(genes):,}")
    return genes


def build_volcano_gene_summary(volcano_path: Path) -> pd.DataFrame:
    print(f"\n📖 SUMMARIZING VOLCANO FILE: {volcano_path}")

    cols = pl.scan_csv(str(volcano_path), infer_schema_length=200).collect_schema().names()
    gene_col = detect_volcano_gene_column(cols)

    ldf = pl.scan_csv(str(volcano_path), infer_schema_length=200).with_columns([
        pl.col(gene_col).cast(pl.Utf8).str.strip_chars().str.replace(r"\.\d+$", "").alias("Gene_clean"),
        pl.col("FDR").cast(pl.Float64).alias("FDR_f"),
        pl.col("Log2FoldChange").cast(pl.Float64).alias("LFC_f"),
    ])

    summary = _collect_compat(
        ldf.group_by("Gene_clean").agg([
            pl.col("FDR_f").min().alias("Volcano_Min_FDR"),
            pl.col("LFC_f").abs().max().alias("Volcano_MaxAbsLFC"),
            pl.len().alias("Volcano_Nrows"),
        ]).rename({"Gene_clean": "Gene"})
    )

    summary["Gene"] = summary["Gene"].astype(str).map(strip_ensg_version)
    summary["Volcano_Min_FDR"] = safe_numeric(summary["Volcano_Min_FDR"]).replace(0, 1e-300)
    summary["Volcano_MaxAbsLFC"] = safe_numeric(summary["Volcano_MaxAbsLFC"])
    return summary


# --------------------------------------------------------------------------------------
# Build master
# --------------------------------------------------------------------------------------
def build_master(
    phenotype: str,
    base_dir: Path,
    use_ot_api: bool,
    ot_max_genes: Optional[int],
    use_directional_pathway: bool = False,
    apply_dircons_penalty: bool = False,
):
    base_path = base_dir / phenotype
    files_dir = base_path / "GeneDifferentialExpression" / "Files"
    ranking_dir = files_dir / "UltimateCompleteRankingAnalysis"

    de_file = ranking_dir / "RANKED_composite.csv"
    path_file = ranking_dir / "PathwayIntegration" / "GenePathwayScores.csv"
    hub_file = ranking_dir / "RANKED_WITH_HUB_composite.csv"
    drug_file = ranking_dir / "druggability_analysis" / f"{phenotype}_druggability_all.csv"
    if phenotype == "migraine" and not drug_file.exists():
        drug_file = ranking_dir / "druggability_analysis" / "migraine_druggability_all.csv"
    volcano_file = files_dir / "combined_volcano_data_all_models.csv"

    for p in [de_file, path_file, volcano_file]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    print(f"\n📖 READING: {de_file}")
    de = pd.read_csv(de_file)
    print(f"   ✓ Shape: {de.shape}")

    gene_col_de = first_existing_col(de, ["Gene", "gene", "ensembl_gene_id", "GeneID"])
    if gene_col_de is None:
        raise ValueError(f"Could not detect gene column in {de_file}")
    if gene_col_de != "Gene":
        de = de.rename(columns={gene_col_de: "Gene"})

    de["Gene"] = de["Gene"].astype(str).map(strip_ensg_version)
    if "Direction" not in de.columns:
        de["Direction"] = "Unknown"

    discovery_col = first_existing_col(de, [
        "Importance_Score", "Discovery_Score", "Composite_Score", "Score", "importance_score"
    ])
    if discovery_col is None:
        de["Importance_Score"] = 0.0
        discovery_col = "Importance_Score"

    print(f"\n📖 READING: {path_file}")
    ps = pd.read_csv(path_file)
    print(f"   ✓ Shape: {ps.shape}")

    gene_col_ps = first_existing_col(ps, ["Gene", "gene", "ensembl_gene_id", "GeneID"])
    if gene_col_ps is None:
        raise ValueError(f"Could not detect gene column in {path_file}")
    if gene_col_ps != "Gene":
        ps = ps.rename(columns={gene_col_ps: "Gene"})
    ps["Gene"] = ps["Gene"].astype(str).map(strip_ensg_version)

    master = de.merge(ps, on="Gene", how="left", suffixes=("", "_path"))
    print(f"   ✓ Merged master shape: {master.shape}")

    volsum = build_volcano_gene_summary(volcano_file)
    master = master.merge(volsum, on="Gene", how="left")

    for c in ["PathScore_Combined", "PathScore_Up", "PathScore_Down", "PathScore_Directional"]:
        if c in master.columns:
            master[c] = safe_numeric(master[c])
        else:
            master[c] = 0.0

    if "DirectionConsistencyScore" not in master.columns:
        master["DirectionConsistencyScore"] = np.nan
    master["DirectionConsistencyScore"] = pd.to_numeric(master["DirectionConsistencyScore"], errors="coerce")

    def choose_path(row):
        if use_directional_pathway:
            d = str(row.get("Direction", "Unknown")).strip().lower()
            if d == "up":
                return row.get("PathScore_Up", 0.0)
            if d == "down":
                return row.get("PathScore_Down", 0.0)
            return row.get("PathScore_Combined", 0.0)
        v = row.get("PathScore_Directional", 0.0)
        return v if v != 0 else row.get("PathScore_Combined", 0.0)

    master["PathScore_used_raw"] = master.apply(choose_path, axis=1).astype(float)
    if apply_dircons_penalty:
        master["PathScore_used_raw"] = master["PathScore_used_raw"] * safe_numeric(master["DirectionConsistencyScore"])

    master["Hub_Score_raw"] = 0.0
    if hub_file.exists():
        print(f"\n📖 READING: {hub_file}")
        hub = pd.read_csv(hub_file)
        print(f"   ✓ Shape: {hub.shape}")
        gene_col_hub = first_existing_col(hub, ["Gene", "gene", "ensembl_gene_id", "GeneID"])
        if gene_col_hub is not None and {"Hub_Score"}.issubset(hub.columns):
            if gene_col_hub != "Gene":
                hub = hub.rename(columns={gene_col_hub: "Gene"})
            hub["Gene"] = hub["Gene"].astype(str).map(strip_ensg_version)
            hub_small = hub[["Gene", "Hub_Score"]].copy()
            hub_small["Hub_Score"] = safe_numeric(hub_small["Hub_Score"])
            hub_small = hub_small.groupby("Gene", as_index=False)["Hub_Score"].max()
            master = master.merge(hub_small, on="Gene", how="left")
            master["Hub_Score_raw"] = safe_numeric(master["Hub_Score"])
            print(f"   ✓ Hub scores loaded: {len(hub_small):,} genes")

    master["Drug_Score_raw"] = 0.0
    master["DirectTarget_raw"] = 0.0
    master["DrugLinkCount_raw"] = 0.0
    master["OpenTargets_raw"] = 0.0

    if drug_file.exists():
        print(f"\n📖 READING: {drug_file}")
        drug = pd.read_csv(drug_file)
        print(f"   ✓ Shape: {drug.shape}")
        print(f"   ✓ Druggability columns: {list(drug.columns)}")

        gene_col = find_gene_col(drug)
        if gene_col != "Gene":
            drug = drug.rename(columns={gene_col: "Gene"})
        drug["Gene"] = drug["Gene"].astype(str).map(strip_ensg_version)

        # ------------------------------------------------------------------
        # 1. Druggability probability
        # ------------------------------------------------------------------
        prob_col = first_existing_col(drug, [
            "Druggability_Probability", "druggability_probability",
            "Probability", "probability",
            "DruggabilityScore", "druggability_score",
            "Score", "score"
        ])

        if prob_col:
            dsmall = drug[["Gene", prob_col]].copy()
            dsmall[prob_col] = safe_numeric(dsmall[prob_col])
            dsmall = dsmall.groupby("Gene", as_index=False)[prob_col].max()
            master = master.merge(dsmall, on="Gene", how="left")
            master["Drug_Score_raw"] = safe_numeric(master[prob_col])
            print(f"   ✓ Druggability probability loaded from: {prob_col}")
        else:
            print("   ⚠ No druggability probability column found")

        # ------------------------------------------------------------------
        # 2. Drug-link count (robust detection)
        # ------------------------------------------------------------------
        drug_name_cols = detect_drug_name_cols(drug)

        usable_drug_name_cols = []
        for c in drug_name_cols:
            if c == "Gene":
                continue
            s = drug[c].fillna("").astype(str).str.strip()
            nonempty = (s != "").sum()
            nunique = s[s != ""].nunique()
            if nonempty > 0 and nunique > 1:
                usable_drug_name_cols.append(c)

        if usable_drug_name_cols:
            tmp_parts = []
            for dcol in usable_drug_name_cols:
                t = drug[["Gene", dcol]].copy()
                t[dcol] = t[dcol].fillna("").astype(str).str.strip()
                t = t[t[dcol] != ""]
                if not t.empty:
                    t = t.rename(columns={dcol: "DrugName"})
                    tmp_parts.append(t)

            if tmp_parts:
                tmp = pd.concat(tmp_parts, ignore_index=True).drop_duplicates()
                link_count = tmp.groupby("Gene")["DrugName"].nunique().reset_index(name="UniqueDrugCount")
                master = master.merge(link_count, on="Gene", how="left")
                master["DrugLinkCount_raw"] = safe_numeric(master["UniqueDrugCount"])
                print(f"   ✓ Drug-link count computed from columns: {usable_drug_name_cols}")
            else:
                print("   ⚠ Drug-like columns found, but no usable non-empty drug names")
        else:
            print("   ⚠ No explicit drug-name column found; building weak fallback if possible")

            drug_flag_cols = []
            for c in drug.columns:
                n = normalize_colname(c)
                if any(k in n for k in ["drug", "compound", "molecule", "therapy", "treatment"]):
                    if c != "Gene" and c != prob_col:
                        drug_flag_cols.append(c)

            if drug_flag_cols:
                tmp = build_fallback_binary_signal(drug, drug_flag_cols, "WeakDrugLinkCount")
                master = master.merge(tmp, on="Gene", how="left")
                master["DrugLinkCount_raw"] = safe_numeric(master["WeakDrugLinkCount"])
                if master["DrugLinkCount_raw"].sum() > 0:
                    print(f"   ✓ Weak fallback drug-link count built from: {drug_flag_cols}")
                else:
                    print("   ⚠ Fallback drug-link signal still zero")
            else:
                print("   ⚠ No fallback drug-related columns available")

        # ------------------------------------------------------------------
        # 3. Direct target evidence (robust detection)
        # ------------------------------------------------------------------
        evidence_cols, ot_cols = detect_evidence_cols(drug)

        if evidence_cols:
            tmp = drug[["Gene"] + evidence_cols].copy()
            direct_score = np.zeros(len(tmp), dtype=float)

            for c in evidence_cols:
                num = pd.to_numeric(tmp[c], errors="coerce")
                if num.notna().sum() > 0:
                    direct_score += num.fillna(0).to_numpy()
                else:
                    txt = tmp[c].astype(str).str.strip().str.lower()
                    direct_score += txt.isin(
                        ["yes", "true", "1", "y", "t", "present", "linked", "known"]
                    ).astype(float).to_numpy()

            tmp["LocalDirectEvidence"] = direct_score
            tmp = tmp.groupby("Gene", as_index=False)["LocalDirectEvidence"].max()
            master = master.merge(tmp, on="Gene", how="left")
            master["DirectTarget_raw"] = safe_numeric(master["LocalDirectEvidence"])

            if master["DirectTarget_raw"].sum() > 0:
                print(f"   ✓ Local direct target evidence built from: {evidence_cols}")
            else:
                print("   ⚠ Evidence columns detected but produced zero signal")
        else:
            print("   ⚠ No local direct target-evidence columns found; local direct baseline will be weak/zero")

        # ------------------------------------------------------------------
        # 4. Local Open Targets-like signal from file, if present
        # ------------------------------------------------------------------
        if ot_cols:
            tmp = drug[["Gene"] + ot_cols].copy()
            ot_score = np.zeros(len(tmp), dtype=float)

            for c in ot_cols:
                num = pd.to_numeric(tmp[c], errors="coerce")
                if num.notna().sum() > 0:
                    ot_score += num.fillna(0).to_numpy()
                else:
                    txt = tmp[c].astype(str).str.strip().str.lower()
                    ot_score += txt.isin(
                        ["yes", "true", "1", "y", "t", "present", "linked", "known"]
                    ).astype(float).to_numpy()

            tmp["LocalOpenTargetsEvidence"] = ot_score
            tmp = tmp.groupby("Gene", as_index=False)["LocalOpenTargetsEvidence"].max()
            master = master.merge(tmp, on="Gene", how="left")
            master["OpenTargets_raw"] = safe_numeric(master["LocalOpenTargetsEvidence"])

            if master["OpenTargets_raw"].sum() > 0:
                print(f"   ✓ Local Open Targets-like evidence built from: {ot_cols}")
            else:
                print("   ⚠ Open Targets-like columns detected but produced zero signal")

    if use_ot_api:
        cache_file = ranking_dir / "ExpandedBaselineComparison" / f"opentargets_cache_{phenotype}.csv"
        ot_df = load_or_fetch_ot_scores(
            genes=master["Gene"].dropna().astype(str).unique().tolist(),
            disease_name=phenotype,
            cache_file=cache_file,
            max_genes=ot_max_genes,
        )
        master = master.merge(ot_df, on="Gene", how="left")
        master["OpenTargets_raw"] = np.maximum(
            safe_numeric(master.get("OpenTargets_raw", 0.0)),
            safe_numeric(master["OpenTargets_API_raw"])
        )
        master["DirectTarget_raw"] = safe_numeric(master["DirectTarget_raw"]) + safe_numeric(master["OpenTargets_API_raw"])
        print("   ✓ Open Targets API scores merged")

    if "Min_FDR" in master.columns:
        master["SigOnly_raw"] = safe_neglog10(master["Min_FDR"])
        print("   ✓ Significance-only baseline loaded from: Min_FDR")
    else:
        master["SigOnly_raw"] = safe_neglog10(master["Volcano_Min_FDR"])
        print("   ✓ Significance-only baseline loaded from volcano-derived minimum FDR")

    master["EffectOnly_raw"] = safe_numeric(master["Volcano_MaxAbsLFC"])
    print("   ✓ Effect-only baseline loaded from volcano-derived maximum absolute log2 fold-change")

    discovery_col_master = first_existing_col(master, [
        "Importance_Score", "Discovery_Score", "Composite_Score",
        "Score", "importance_score", "discovery_score"
    ])
    if discovery_col_master is None:
        master["Discovery_Score_raw"] = 0.0
        print("   ⚠ No discovery/importance score found; setting Discovery_Score_raw = 0")
    else:
        master["Discovery_Score_raw"] = safe_numeric(master[discovery_col_master])
        print(f"   ✓ Discovery score loaded from: {discovery_col_master}")

    raw_to_norm = {
        "SigOnly_raw": "SigOnly_norm",
        "EffectOnly_raw": "EffectOnly_norm",
        "Discovery_Score_raw": "Discovery_Score_norm",
        "PathScore_used_raw": "Path_Score_norm",
        "Hub_Score_raw": "Hub_Score_norm",
        "Drug_Score_raw": "Drug_Score_norm",
        "DirectTarget_raw": "DirectTarget_norm",
        "DrugLinkCount_raw": "DrugLinkCount_norm",
        "OpenTargets_raw": "OpenTargets_norm",
    }
    for raw_col, norm_col in raw_to_norm.items():
        master[norm_col] = pct_rank(master[raw_col])

    print(f"\n✓ Master dataset ready: {master.shape[0]:,} genes")
    print("✓ Expanded baselines created and percentile normalization applied")
    return master, files_dir, ranking_dir


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------
def evaluate_one_ranking(
    df: pd.DataFrame,
    score_col: str,
    label: str,
    test_pos: set,
    known_pos: set,
    universe_n: int,
    topk_list: list,
    focus_k: int,
    phenotype_label: str,
):
    sdf = df.sort_values(score_col, ascending=False).reset_index(drop=True).copy()
    ranked_genes = sdf["Gene"].tolist()
    scores = sdf[score_col].to_numpy()

    y_test = np.array([1 if g in test_pos else 0 for g in ranked_genes], dtype=int)
    y_known = np.array([1 if g in known_pos else 0 for g in ranked_genes], dtype=int)

    test_auc, test_pr = auc_pr(y_test, scores)
    known_auc, known_pr = auc_pr(y_known, scores)

    print_block(f"📊 RANKING: {label}")
    print(f"TEST:     ROC-AUC={fmt_metric(test_auc)}, PR-AUC={fmt_metric(test_pr)}")
    print(f"KNOWN:    ROC-AUC={fmt_metric(known_auc)}, PR-AUC={fmt_metric(known_pr)}")

    print_topk_table(ranked_genes, test_pos, universe_n, topk_list, "Enrichment vs TEST")
    print_topk_table(ranked_genes, known_pos, universe_n, topk_list, f"Enrichment vs {phenotype_label.upper()}")

    obs_t, exp_t, fe_t, prec_t, _ = topk_stats(ranked_genes, test_pos, universe_n, focus_k)
    obs_k, exp_k, fe_k, prec_k, _ = topk_stats(ranked_genes, known_pos, universe_n, focus_k)

    return {
        "Ranking": label,
        "Test_ROC_AUC": test_auc,
        "Test_PR_AUC": test_pr,
        f"{phenotype_label}_ROC_AUC": known_auc,
        f"{phenotype_label}_PR_AUC": known_pr,
        f"Top{focus_k}_Test_Obs": obs_t,
        f"Top{focus_k}_Test_Exp": exp_t,
        f"Top{focus_k}_Test_FE": fe_t,
        f"Top{focus_k}_Test_Precision": prec_t,
        f"Top{focus_k}_{phenotype_label}_Obs": obs_k,
        f"Top{focus_k}_{phenotype_label}_Exp": exp_k,
        f"Top{focus_k}_{phenotype_label}_FE": fe_k,
        f"Top{focus_k}_{phenotype_label}_Precision": prec_k,
        "_ranked_df": sdf,
    }


def add_integrated_scores(df: pd.DataFrame):
    out = df.copy()
    out["Integrated_Score_norm"] = (
        0.45 * out["Discovery_Score_norm"]
        + 0.25 * out["Path_Score_norm"]
        + 0.25 * out["Drug_Score_norm"]
        + 0.05 * out["Hub_Score_norm"]
    )
    out["Integrated_Balanced_norm"] = (
        0.30 * out["Discovery_Score_norm"]
        + 0.30 * out["Path_Score_norm"]
        + 0.20 * out["Hub_Score_norm"]
        + 0.20 * out["Drug_Score_norm"]
    )
    out["Integrated_BiologyFavoured_norm"] = (
        0.25 * out["Discovery_Score_norm"]
        + 0.35 * out["Path_Score_norm"]
        + 0.25 * out["Hub_Score_norm"]
        + 0.15 * out["Drug_Score_norm"]
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phenotype")
    ap.add_argument("--base-dir", default="/data/ascher02/uqmmune1/ANNOVAR")
    ap.add_argument("--fdr", type=float, default=0.10)
    ap.add_argument("--effect", type=float, default=0.50)
    ap.add_argument("--top-k", type=int, default=200)
    ap.add_argument("--use-ot-api", action="store_true")
    ap.add_argument("--ot-max-genes", type=int, default=None)
    ap.add_argument("--include-alt-integrated", action="store_true")
    ap.add_argument("--use-directional-pathway", action="store_true")
    ap.add_argument("--apply-dircons-penalty", action="store_true")
    ap.add_argument("--val-keywords", default="validation,val")
    ap.add_argument("--test-keywords", default="test")
    args = ap.parse_args()

    phenotype = args.phenotype
    base_dir = Path(args.base_dir)
    focus_k = args.top_k

    print_block("📁 LOADING DATA FILES")
    master, files_dir, ranking_dir = build_master(
        phenotype=phenotype,
        base_dir=base_dir,
        use_ot_api=args.use_ot_api,
        ot_max_genes=args.ot_max_genes,
        use_directional_pathway=args.use_directional_pathway,
        apply_dircons_penalty=args.apply_dircons_penalty,
    )

    volcano_file = files_dir / "combined_volcano_data_all_models.csv"
    disease_file = files_dir / f"{phenotype}_genes.csv"
    if not disease_file.exists() and phenotype == "migraine":
        disease_file = files_dir / "migraine_genes.csv"

    if not disease_file.exists():
        raise FileNotFoundError(f"Could not find phenotype gene file: {disease_file}")

    val_kw = [s.strip() for s in args.val_keywords.split(",") if s.strip()]
    test_kw = [s.strip() for s in args.test_keywords.split(",") if s.strip()]

    val_pos, volcano_universe = volcano_universe_and_positives(volcano_file, val_kw, args.fdr, args.effect)
    test_pos, _ = volcano_universe_and_positives(volcano_file, test_kw, args.fdr, args.effect)
    known = load_disease_genes(disease_file, phenotype)

    master["Gene"] = master["Gene"].astype(str).map(strip_ensg_version)
    eval_df = master[master["Gene"].isin(volcano_universe)].copy().reset_index(drop=True)

    eval_universe = set(eval_df["Gene"])
    test_pos_eval = test_pos & eval_universe
    known_eval = known & eval_universe
    universe_n_eval = len(eval_universe)

    eval_df = add_integrated_scores(eval_df)

    topk_list = sorted(set([50, 100, focus_k, 500, 1000, 2000]))
    topk_list = [k for k in topk_list if k <= len(eval_df)]

    rankings = [
        ("SigOnly_norm", "Significance only"),
        ("EffectOnly_norm", "Effect only"),
        ("Discovery_Score_norm", "Discovery score only"),
        ("Path_Score_norm", "Pathway only"),
        ("Hub_Score_norm", "Hub only"),
        ("Drug_Score_norm", "Druggability only"),
        ("DirectTarget_norm", "Direct target evidence only"),
        ("DrugLinkCount_norm", "Drug-link count only"),
        ("OpenTargets_norm", "Open Targets only"),
        ("Integrated_Score_norm", "Integrated"),
    ]

    if args.include_alt_integrated:
        rankings.extend([
            ("Integrated_Balanced_norm", "Integrated balanced"),
            ("Integrated_BiologyFavoured_norm", "Integrated biology-favoured"),
        ])

    usable_rankings = []
    seen_hashes = {}
    for score_col, label in rankings:
        if score_col not in eval_df.columns:
            continue
        vec = safe_numeric(eval_df[score_col])
        if vec.sum() <= 0:
            print(f"⚠ Skipping {label}: no usable signal found")
            continue
        h = hashlib.md5(np.round(vec.to_numpy(), 8).tobytes()).hexdigest()
        if h in seen_hashes:
            print(f"⚠ Skipping {label}: identical to {seen_hashes[h]}")
            continue
        seen_hashes[h] = label
        usable_rankings.append((score_col, label))

    print_block("STEP 1: EXPANDED BASELINE COMPARISON")

    results = []
    ranked_outputs = {}
    for score_col, label in usable_rankings:
        res = evaluate_one_ranking(
            eval_df,
            score_col=score_col,
            label=label,
            test_pos=test_pos_eval,
            known_pos=known_eval,
            universe_n=universe_n_eval,
            topk_list=topk_list,
            focus_k=focus_k,
            phenotype_label=phenotype,
        )
        ranked_outputs[label] = res.pop("_ranked_df")
        results.append(res)

    summary_df = pd.DataFrame(results)
    print_block("📊 MANUSCRIPT SUMMARY TABLE")
    print(summary_df.to_string(index=False))

    out_dir = ensure_dir(ranking_dir / "ExpandedBaselineComparison")
    summary_df.to_csv(out_dir / "expanded_baseline_comparison_full.csv", index=False)

    for label, rdf in ranked_outputs.items():
        safe = label.lower().replace(" ", "_").replace("-", "_")
        rdf.to_csv(out_dir / f"ranked_{safe}.csv", index=False)
        rdf.head(focus_k).to_csv(out_dir / f"top_{focus_k}_{safe}.csv", index=False)

    print_block("✅ COMPLETE")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()