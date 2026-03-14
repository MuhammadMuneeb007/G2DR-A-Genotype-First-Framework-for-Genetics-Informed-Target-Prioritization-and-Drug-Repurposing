#!/usr/bin/env python3
"""
predict4.1.2.11.2-DrugFinder.py
================================

Automatic migraine-evidence-tiered drug benchmark.

This script benchmarks the ranked drug list against a migraine-specific
evidence hierarchy closer to the reviewer request:

Tier1_MigraineSpecificApproved
    Drugs specifically approved / clearly migraine-specific therapeutic agents
    (e.g. triptans, ditans, gepants, CGRP monoclonals, ergot derivatives used for migraine)

Tier2_GuidelineSupported
    Guideline-supported acute/preventive migraine therapies that are not strictly
    migraine-specific branded classes (e.g. topiramate, propranolol, amitriptyline,
    NSAIDs, paracetamol, anti-emetics, onabotulinumtoxinA)

Tier3_EstablishedOffLabel
    Established off-label / recurring migraine-use drugs

Tier4_BroadLiteratureLinked
    Everything else in migraine_drugs.csv not captured above

Outputs
-------
- migraine_drugs_tiered_EVIDENCE_AUTO.csv
- TieredDrugBenchmark_Detailed.csv
- TieredDrugBenchmark_ManuscriptTable.csv
- TieredDrugBenchmark_Summary.csv
- TieredDrugBenchmark_ManuscriptText.txt

Optional
--------
You can supply an override CSV with columns:
    drug_name,evidence_tier,notes
or
    Drug_norm,evidence_tier,notes

This lets you correct any automatic assignments manually.

Example
-------
python predict4.1.2.11.2-DrugFinder.py migraine
python predict4.1.2.11.2-DrugFinder.py migraine --override-file migraine_drug_tier_overrides.csv
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd


VALID_TIERS = [
    "Tier1_MigraineSpecificApproved",
    "Tier2_GuidelineSupported",
    "Tier3_EstablishedOffLabel",
    "Tier4_BroadLiteratureLinked",
]

TIER_ORDER = {t: i + 1 for i, t in enumerate(VALID_TIERS)}


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def normalize_drug_name(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip().lower()
    if not s or s in {"nan", "none", "null"}:
        return ""

    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"\[.*?\]", " ", s)
    s = s.replace("&", " and ")
    s = s.replace("/", " ")
    s = s.replace("+", " ")
    s = s.replace(",", " ")
    s = s.replace(";", " ")
    s = s.replace("'", " ")

    # remove administration / trial clutter
    s = re.sub(r"\bduration of treatment\b.*", " ", s)
    s = re.sub(r"\boral\b", " ", s)
    s = re.sub(r"\bintravenous\b", " ", s)
    s = re.sub(r"\biv\b", " ", s)
    s = re.sub(r"\bsubcutaneous\b", " ", s)
    s = re.sub(r"\binjection\b", " ", s)
    s = re.sub(r"\bnasal\b", " ", s)
    s = re.sub(r"\btablet\b", " ", s)
    s = re.sub(r"\bcapsule\b", " ", s)
    s = re.sub(r"\bsolution\b", " ", s)
    s = re.sub(r"\bcream\b", " ", s)
    s = re.sub(r"\bpatch\b", " ", s)
    s = re.sub(r"\bcompound\b", " ", s)

    # remove dose text
    s = re.sub(r"\b\d+(\.\d+)?\s*(mg|mcg|g|ml|iu|units?|%)\b", " ", s)

    # remove common salt words
    salt_words = [
        "hydrochloride", "hcl", "sodium", "potassium", "calcium", "succinate",
        "tartrate", "maleate", "phosphate", "sulfate", "acetate", "chloride",
        "nitrate", "mesylate", "besylate", "benzoate", "bromide", "citrate",
        "lactate", "malate", "vfrm", "hydrobromide", "dihydrate", "monohydrate"
    ]
    s = re.sub(r"\b(" + "|".join(map(re.escape, salt_words)) + r")\b", " ", s)

    # remove protocol-like codes
    s = re.sub(r"\b[a-z]{1,6}[-]?\d{2,}\b", " ", s)

    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def detect_drug_column(df: pd.DataFrame, preferred: List[str]) -> str:
    cols_lower = {c.lower(): c for c in df.columns}
    for p in preferred:
        if p.lower() in cols_lower:
            return cols_lower[p.lower()]
    for c in df.columns:
        if "drug" in c.lower():
            return c
    raise ValueError(f"Could not detect drug column. Columns found: {list(df.columns)}")


def print_block(title: str):
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)


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

    if not logs:
        return 1.0

    m = max(logs)
    p = sum(math.exp(li - m) for li in logs) * math.exp(m)
    return min(max(p, 0.0), 1.0)


def topk_stats(ranked: List[str], positives: Set[str], universe_n: int, k: int):
    k = min(int(k), len(ranked))
    top = set(ranked[:k])
    obs = len(top & positives)
    exp = k * (len(positives) / universe_n) if universe_n > 0 else float("nan")
    fe = (obs / exp) if exp and exp > 0 else float("nan")
    prec = obs / k if k > 0 else float("nan")
    rec = obs / len(positives) if len(positives) > 0 else float("nan")
    return obs, exp, fe, prec, rec


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def load_ranked_drugs(drug_ranking_file: Path) -> pd.DataFrame:
    if not drug_ranking_file.exists():
        raise FileNotFoundError(f"Drug ranking file not found: {drug_ranking_file}")

    print(f"Reading ranked drugs: {drug_ranking_file}")
    df = pd.read_csv(drug_ranking_file)
    print(f"   Shape: {df.shape}")

    drug_col = detect_drug_column(df, ["DrugName_best", "DrugName", "Drug"])
    print(f"   Drug column: {drug_col}")

    if "DrugRank" in df.columns:
        df = df.sort_values("DrugRank", ascending=True).reset_index(drop=True)
    elif "DrugScore" in df.columns:
        df = df.sort_values("DrugScore", ascending=False).reset_index(drop=True)
        df["DrugRank"] = range(1, len(df) + 1)
    else:
        df["DrugRank"] = range(1, len(df) + 1)

    df["Drug_raw"] = df[drug_col].astype(str)
    df["Drug_norm"] = df["Drug_raw"].map(normalize_drug_name)
    df = df[df["Drug_norm"] != ""].copy()
    df = (
        df.sort_values("DrugRank", ascending=True)
        .drop_duplicates("Drug_norm", keep="first")
        .reset_index(drop=True)
    )

    print(f"   Unique normalized predicted drugs: {df.shape[0]:,}")
    return df


def load_migraine_drugs_csv(migraine_file: Path) -> pd.DataFrame:
    if not migraine_file.exists():
        raise FileNotFoundError(f"Migraine drug file not found: {migraine_file}")

    print(f"Reading migraine drug file: {migraine_file}")
    df = pd.read_csv(migraine_file)
    print(f"   Shape: {df.shape}")

    drug_col = detect_drug_column(df, ["drug_name", "DrugName", "Drug"])
    out = df.copy()
    out["drug_name_raw"] = out[drug_col].astype(str)
    out["Drug_norm"] = out["drug_name_raw"].map(normalize_drug_name)
    out = out[out["Drug_norm"] != ""].copy()

    for col in ["phase", "status", "source", "drug_type", "max_phase"]:
        if col not in out.columns:
            out[col] = ""

    print(f"   Unique non-empty normalized entries: {out['Drug_norm'].nunique():,}")
    return out


def load_all_drugs_universe(all_drugs_file: Path) -> Set[str]:
    if not all_drugs_file.exists():
        raise FileNotFoundError(f"ALL drugs universe file not found: {all_drugs_file}")

    print(f"Reading ALL-drugs universe: {all_drugs_file}")
    df = pd.read_csv(all_drugs_file, low_memory=False)

    drug_col = detect_drug_column(df, ["drug_name", "DrugName", "Drug", "drug_norm", "Drug_norm"])
    print(f"   Drug column: {drug_col}")

    universe = set(df[drug_col].dropna().astype(str).map(normalize_drug_name))
    universe.discard("")
    print(f"   Unique normalized ALL-drugs universe: {len(universe):,}")
    return universe


# --------------------------------------------------------------------------------------
# Tier rules
# --------------------------------------------------------------------------------------
# Tier 1: migraine-specific approved / clearly migraine-targeted classes
TIER1_EXACT = {
    "sumatriptan", "rizatriptan", "zolmitriptan", "naratriptan",
    "almotriptan", "eletriptan", "frovatriptan",
    "dihydroergotamine", "ergotamine",
    "lasmiditan",
    "ubrogepant", "rimegepant", "zavegepant", "atogepant",
    "erenumab", "fremanezumab", "galcanezumab", "eptinezumab",
}

TIER1_SUBSTRINGS = [
    "triptan",           # catches combinations / formulations
    "gepant",
    "dihydroergotamine",
    "ergotamine",
    "lasmiditan",
    "erenumab",
    "fremanezumab",
    "galcanezumab",
    "eptinezumab",
]

# Tier 2: guideline-supported acute or preventive migraine treatments
TIER2_EXACT = {
    "topiramate", "propranolol", "amitriptyline",
    "acetaminophen", "paracetamol", "aspirin",
    "ibuprofen", "naproxen", "diclofenac", "ketorolac",
    "metoclopramide", "prochlorperazine", "chlorpromazine",
    "ondansetron", "domperidone",
    "onabotulinumtoxina", "botulinum toxin type a", "botulinum toxin",
}

TIER2_SUBSTRINGS = [
    "onabotulinum",
    "botulinum",
    "metoclopramide",
    "prochlorperazine",
    "chlorpromazine",
    "acetaminophen",
    "paracetamol",
    "ibuprofen",
    "naproxen",
    "diclofenac",
    "ketorolac",
    "aspirin",
]

# Tier 3: established off-label / recurring migraine-use therapies
TIER3_EXACT = {
    "candesartan", "lisinopril", "verapamil",
    "gabapentin", "pregabalin",
    "nortriptyline", "venlafaxine", "duloxetine",
    "fluoxetine", "sertraline",
    "valproate", "divalproex", "sodium valproate",
    "memantine", "lamotrigine",
    "magnesium", "riboflavin", "coenzyme q10", "melatonin",
    "timolol", "atenolol", "metoprolol",
    "cyproheptadine",
}

TIER3_SUBSTRINGS = [
    "candesartan", "lisinopril", "verapamil",
    "gabapentin", "pregabalin",
    "nortriptyline", "venlafaxine", "duloxetine",
    "fluoxetine", "sertraline",
    "valpro", "divalpro", "lamotrigine", "memantine",
    "magnesium", "riboflavin", "melatonin",
    "timolol", "atenolol", "metoprolol",
    "cyproheptadine",
]


def text_blob(row: pd.Series) -> str:
    vals = [
        str(row.get("drug_name_raw", "")),
        str(row.get("Drug_norm", "")),
        str(row.get("phase", "")),
        str(row.get("status", "")),
        str(row.get("source", "")),
        str(row.get("drug_type", "")),
        str(row.get("max_phase", "")),
    ]
    return " | ".join(v for v in vals if v and v != "nan").lower()


def matches_exact_or_substring(norm_name: str, blob: str, exact_set: Set[str], substrings: List[str]) -> bool:
    if norm_name in exact_set:
        return True
    for s in substrings:
        if s in norm_name or s in blob:
            return True
    return False


def assign_migraine_evidence_tier(row: pd.Series) -> Tuple[str, str]:
    norm_name = str(row.get("Drug_norm", "")).strip().lower()
    blob = text_blob(row)

    # Tier 1
    if matches_exact_or_substring(norm_name, blob, TIER1_EXACT, TIER1_SUBSTRINGS):
        return "Tier1_MigraineSpecificApproved", "auto_tier1_rule"

    # Tier 2
    if matches_exact_or_substring(norm_name, blob, TIER2_EXACT, TIER2_SUBSTRINGS):
        return "Tier2_GuidelineSupported", "auto_tier2_rule"

    # Tier 3
    if matches_exact_or_substring(norm_name, blob, TIER3_EXACT, TIER3_SUBSTRINGS):
        return "Tier3_EstablishedOffLabel", "auto_tier3_rule"

    # fallback
    return "Tier4_BroadLiteratureLinked", "auto_fallback_broad"


# --------------------------------------------------------------------------------------
# Override handling
# --------------------------------------------------------------------------------------
def load_override_file(override_file: Path) -> pd.DataFrame:
    if not override_file.exists():
        raise FileNotFoundError(f"Override file not found: {override_file}")

    print(f"Reading override file: {override_file}")
    df = pd.read_csv(override_file)
    print(f"   Shape: {df.shape}")

    tier_col = None
    for cand in ["evidence_tier", "Tier", "tier"]:
        if cand in df.columns:
            tier_col = cand
            break
    if tier_col is None:
        raise ValueError("Override file must contain evidence_tier column")

    if "Drug_norm" in df.columns:
        df["Drug_norm"] = df["Drug_norm"].astype(str).str.strip().str.lower()
    else:
        drug_col = detect_drug_column(df, ["drug_name", "DrugName", "Drug"])
        df["Drug_norm"] = df[drug_col].astype(str).map(normalize_drug_name)

    df["evidence_tier"] = df[tier_col].astype(str).str.strip()
    bad = sorted(set(df["evidence_tier"]) - set(VALID_TIERS))
    if bad:
        raise ValueError(f"Invalid tiers in override file: {bad}")

    if "notes" not in df.columns:
        df["notes"] = ""

    df = df[df["Drug_norm"] != ""].copy()
    df = df[df["evidence_tier"] != ""].copy()
    df = df.drop_duplicates("Drug_norm", keep="first").reset_index(drop=True)
    return df[["Drug_norm", "evidence_tier", "notes"]]


# --------------------------------------------------------------------------------------
# Tier construction
# --------------------------------------------------------------------------------------
def collapse_metadata(group: pd.DataFrame) -> pd.Series:
    def join_unique(series):
        vals = sorted(set(str(x).strip() for x in series if str(x).strip() not in {"", "nan", "None"}))
        return "; ".join(vals)

    return pd.Series({
        "drug_name_example": group["drug_name_raw"].iloc[0],
        "phase_values": join_unique(group["phase"]),
        "status_values": join_unique(group["status"]),
        "source_values": join_unique(group["source"]),
        "drug_type_values": join_unique(group["drug_type"]),
        "max_phase_values": join_unique(group["max_phase"]),
        "n_records": group.shape[0],
    })


def build_auto_tiered_reference(migraine_df: pd.DataFrame, override_df: pd.DataFrame | None = None) -> pd.DataFrame:
    grouped = (
        migraine_df.groupby("Drug_norm", as_index=False)
        .apply(collapse_metadata, include_groups=False)
        .reset_index()
    )

    if "Drug_norm" not in grouped.columns:
        grouped = grouped.rename(columns={"index": "Drug_norm"})
    if "level_0" in grouped.columns:
        grouped = grouped.drop(columns=["level_0"])

    grouped["drug_name_raw"] = grouped["drug_name_example"]

    for c in ["phase_values", "status_values", "source_values", "drug_type_values", "max_phase_values"]:
        if c not in grouped.columns:
            grouped[c] = ""

    grouped["phase"] = grouped["phase_values"]
    grouped["status"] = grouped["status_values"]
    grouped["source"] = grouped["source_values"]
    grouped["drug_type"] = grouped["drug_type_values"]
    grouped["max_phase"] = grouped["max_phase_values"]

    tiers = grouped.apply(assign_migraine_evidence_tier, axis=1)
    grouped["Tier"] = [t[0] for t in tiers]
    grouped["AssignmentRule"] = [t[1] for t in tiers]
    grouped["notes"] = ""

    if override_df is not None and not override_df.empty:
        override_map = override_df.set_index("Drug_norm")["evidence_tier"].to_dict()
        notes_map = override_df.set_index("Drug_norm")["notes"].to_dict()

        hit_mask = grouped["Drug_norm"].isin(override_map)
        grouped.loc[hit_mask, "Tier"] = grouped.loc[hit_mask, "Drug_norm"].map(override_map)
        grouped.loc[hit_mask, "AssignmentRule"] = "manual_override"
        grouped.loc[hit_mask, "notes"] = grouped.loc[hit_mask, "Drug_norm"].map(notes_map).fillna("")

    grouped["Tier_Order"] = grouped["Tier"].map(TIER_ORDER)

    grouped = grouped[
        [
            "Drug_norm",
            "drug_name_example",
            "Tier",
            "AssignmentRule",
            "notes",
            "phase_values",
            "status_values",
            "source_values",
            "drug_type_values",
            "max_phase_values",
            "n_records",
            "Tier_Order",
        ]
    ].sort_values(["Tier_Order", "Drug_norm"]).reset_index(drop=True)

    return grouped


def tier_sets_from_df(tier_df: pd.DataFrame) -> Dict[str, Set[str]]:
    out = {}
    for tier in VALID_TIERS:
        out[tier] = set(tier_df.loc[tier_df["Tier"] == tier, "Drug_norm"].tolist())
    return out


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------
def evaluate_tiers(
    ranked_df: pd.DataFrame,
    tier_sets: Dict[str, Set[str]],
    universe_all: Set[str],
    topk_list: List[int],
):
    ranked_drugs = ranked_df["Drug_norm"].tolist()
    predicted_universe = set(ranked_drugs)

    detailed_rows = []
    manuscript_rows = []

    print_block("MIGRAINE-EVIDENCE TIERED DRUG BENCHMARK EVALUATION")

    for tier_name in VALID_TIERS:
        tier_raw = tier_sets[tier_name]
        tier_in_all = tier_raw & universe_all
        tier_in_pred = tier_raw & predicted_universe

        print(f"\n{tier_name}")
        print(f"  Raw reference drugs:         {len(tier_raw):,}")
        print(f"  In ALL-drugs universe:       {len(tier_in_all):,}")
        print(f"  In predicted drug universe:  {len(tier_in_pred):,}")

        hdr = f"{'K':>6}  {'Overlap':>8}  {'Expected':>9}  {'FE':>7}  {'Precision':>10}  {'Recall':>10}  {'Hypergeom_p':>12}"
        print(hdr)
        print("-" * len(hdr))

        row = {"Tier": tier_name}

        for k in topk_list:
            obs, exp, fe, prec, rec = topk_stats(ranked_drugs, tier_in_all, len(universe_all), k)
            pval = hypergeom_pval(len(universe_all), len(tier_in_all), min(k, len(ranked_drugs)), obs)

            exp_s = f"{exp:.2f}" if pd.notna(exp) else "NA"
            fe_s = f"{fe:.2f}" if pd.notna(fe) else "NA"
            print(f"{k:>6}  {obs:>8}  {exp_s:>9}  {fe_s:>7}  {prec:>10.4f}  {rec:>10.4f}  {pval:>12.3e}")

            detailed_rows.append({
                "Tier": tier_name,
                "K": k,
                "Universe_ALL": len(universe_all),
                "Reference_in_ALL": len(tier_in_all),
                "Reference_in_Predicted": len(tier_in_pred),
                "Overlap": obs,
                "Expected": exp,
                "FE": fe,
                "Precision": prec,
                "Recall": rec,
                "Hypergeom_p": pval,
            })

            if k == 20:
                row["Top20_Overlap"] = obs
            elif k == 50:
                row["Top50_Overlap"] = obs
            elif k == 100:
                row["Top100_Overlap"] = obs
                row["Precision@100"] = prec
                row["Recall@100"] = rec
                row["FE@100"] = fe
            elif k == 200:
                row["Top200_Overlap"] = obs

        manuscript_rows.append(row)

    detailed_df = pd.DataFrame(detailed_rows)
    manuscript_df = pd.DataFrame(manuscript_rows)

    summary_df = manuscript_df.copy()
    summary_df["Tier_Order"] = summary_df["Tier"].map(TIER_ORDER)
    summary_df = summary_df.sort_values("Tier_Order").drop(columns=["Tier_Order"]).reset_index(drop=True)

    return detailed_df, manuscript_df, summary_df


# --------------------------------------------------------------------------------------
# Manuscript text
# --------------------------------------------------------------------------------------
def build_manuscript_text(summary_df: pd.DataFrame) -> str:
    summary_df = summary_df.copy()
    row_map = {r["Tier"]: r for _, r in summary_df.iterrows()}

    def get_int(tier, col):
        v = row_map.get(tier, {}).get(col, 0)
        try:
            return int(v)
        except Exception:
            return 0

    def get_float(tier, col):
        v = row_map.get(tier, {}).get(col, 0.0)
        try:
            return float(v)
        except Exception:
            return 0.0

    t1_20 = get_int("Tier1_MigraineSpecificApproved", "Top20_Overlap")
    t1_50 = get_int("Tier1_MigraineSpecificApproved", "Top50_Overlap")
    t1_100 = get_int("Tier1_MigraineSpecificApproved", "Top100_Overlap")
    t1_200 = get_int("Tier1_MigraineSpecificApproved", "Top200_Overlap")
    t1_fe100 = get_float("Tier1_MigraineSpecificApproved", "FE@100")

    t2_100 = get_int("Tier2_GuidelineSupported", "Top100_Overlap")
    t2_200 = get_int("Tier2_GuidelineSupported", "Top200_Overlap")
    t2_fe100 = get_float("Tier2_GuidelineSupported", "FE@100")

    t3_100 = get_int("Tier3_EstablishedOffLabel", "Top100_Overlap")
    t3_200 = get_int("Tier3_EstablishedOffLabel", "Top200_Overlap")
    t3_fe100 = get_float("Tier3_EstablishedOffLabel", "FE@100")

    t4_100 = get_int("Tier4_BroadLiteratureLinked", "Top100_Overlap")
    t4_200 = get_int("Tier4_BroadLiteratureLinked", "Top200_Overlap")
    t4_fe100 = get_float("Tier4_BroadLiteratureLinked", "FE@100")

    text = f"""
Tiered migraine-evidence benchmark.
To provide a more clinically interpretable drug benchmark, we partitioned the curated migraine drug reference set into four evidence-oriented tiers: migraine-specific approved therapies, guideline-supported acute or preventive therapies, established off-label therapies, and broader literature-linked compounds. We then evaluated the ranked drug list against each tier separately. The Top-100 predicted drugs recovered {t1_100} Tier-1 migraine-specific approved therapies (FE={t1_fe100:.2f}), {t2_100} Tier-2 guideline-supported therapies (FE={t2_fe100:.2f}), {t3_100} Tier-3 established off-label therapies (FE={t3_fe100:.2f}), and {t4_100} Tier-4 broader literature-linked compounds (FE={t4_fe100:.2f}). At Top-200, the corresponding overlaps were {t1_200}, {t2_200}, {t3_200}, and {t4_200}, respectively. This tiered analysis provides a stricter assessment of translational specificity than a single pooled migraine-drug benchmark by distinguishing recovery of migraine-specific and clinically established therapies from recovery of broader literature-associated compounds.
""".strip()

    return text


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Automatic migraine-evidence-tiered drug benchmark.")
    ap.add_argument("phenotype")
    ap.add_argument("--base-dir", default="/data/ascher02/uqmmune1/ANNOVAR")
    ap.add_argument("--migraine-drugs-file", default="/data/ascher02/uqmmune1/ANNOVAR/migraine_drugs.csv")
    ap.add_argument("--override-file", default=None)
    ap.add_argument("--drug-ranking", default=None)
    ap.add_argument("--all-drugs-db", default="/data/ascher02/uqmmune1/ANNOVAR/AllDiseasesToDrugs/ALL_SOURCES_drug_disease_merged.csv")
    ap.add_argument("--topk", default="20,50,100,200")
    args = ap.parse_args()

    phenotype = args.phenotype
    base_dir = Path(args.base_dir)
    migraine_file = Path(args.migraine_drugs_file)
    all_drugs_file = Path(args.all_drugs_db)

    if args.drug_ranking is None:
        drug_ranking_file = (
            base_dir / phenotype / "GeneDifferentialExpression" / "Files" /
            "UltimateCompleteRankingAnalysis" / "DrugIntegration" / "DrugRanking.csv"
        )
    else:
        drug_ranking_file = Path(args.drug_ranking)

    topk_list = sorted(set(int(x.strip()) for x in args.topk.split(",") if x.strip()))

    out_dir = ensure_dir(
        base_dir / phenotype / "GeneDifferentialExpression" / "Files" /
        "UltimateCompleteRankingAnalysis" / "DrugIntegration" / "TieredBenchmarkEvidence"
    )

    print_block("INPUTS")
    print(f"Phenotype:            {phenotype}")
    print(f"Migraine drugs file:  {migraine_file}")
    print(f"Override file:        {args.override_file}")
    print(f"Drug ranking file:    {drug_ranking_file}")
    print(f"ALL drugs DB:         {all_drugs_file}")
    print(f"Top-K values:         {topk_list}")
    print(f"Output dir:           {out_dir}")

    ranked_df = load_ranked_drugs(drug_ranking_file)
    migraine_df = load_migraine_drugs_csv(migraine_file)
    universe_all = load_all_drugs_universe(all_drugs_file)

    override_df = None
    if args.override_file is not None:
        override_df = load_override_file(Path(args.override_file))

    print_block("BUILDING AUTOMATIC MIGRAINE-EVIDENCE TIERS")
    tier_df = build_auto_tiered_reference(migraine_df, override_df=override_df)
    tier_sets = tier_sets_from_df(tier_df)

    print("\nTier counts")
    for tier in VALID_TIERS:
        print(f"  {tier}: {len(tier_sets[tier]):,} drugs")

    tier_file = out_dir / "migraine_drugs_tiered_EVIDENCE_AUTO.csv"
    tier_df.to_csv(tier_file, index=False)

    detailed_df, manuscript_df, summary_df = evaluate_tiers(
        ranked_df=ranked_df,
        tier_sets=tier_sets,
        universe_all=universe_all,
        topk_list=topk_list,
    )

    detailed_path = out_dir / "TieredDrugBenchmark_Detailed.csv"
    manuscript_path = out_dir / "TieredDrugBenchmark_ManuscriptTable.csv"
    summary_path = out_dir / "TieredDrugBenchmark_Summary.csv"
    text_path = out_dir / "TieredDrugBenchmark_ManuscriptText.txt"

    detailed_df.to_csv(detailed_path, index=False)
    manuscript_df.to_csv(manuscript_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    manuscript_text = build_manuscript_text(summary_df)
    with open(text_path, "w") as f:
        f.write(manuscript_text + "\n")

    print_block("MANUSCRIPT TABLE")
    print(summary_df.to_string(index=False))

    print_block("MANUSCRIPT TEXT")
    print(manuscript_text)

    print_block("DONE")
    print("Saved:")
    print(f"  {tier_file}")
    print(f"  {detailed_path}")
    print(f"  {manuscript_path}")
    print(f"  {summary_path}")
    print(f"  {text_path}")


if __name__ == "__main__":
    main()