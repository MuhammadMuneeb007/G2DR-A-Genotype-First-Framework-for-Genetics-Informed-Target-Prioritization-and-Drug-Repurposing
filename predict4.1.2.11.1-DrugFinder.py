#!/usr/bin/env python3
"""
predict4.1.2.11.2-DrugFinder_PhaseTiered.py
===========================================

Build phase-based migraine drug tiers directly from migraine_drugs.csv,
then benchmark one ranked drug list against those tiers.

Tier definition
---------------
Tier1_ApprovedLate : APPROVED / PHASE4
Tier2_Phase3       : PHASE3 / PHASE2/PHASE3
Tier3_PhaseEarly   : PHASE2 / PHASE1/PHASE2 / PHASE1
Tier4_OtherBroad   : N/A / NA / UNKNOWN / missing / everything else

Outputs
-------
- migraine_drugs_tiered_PHASE.csv
- TieredDrugBenchmark_Detailed.csv
- TieredDrugBenchmark_ManuscriptTable.csv
- TieredDrugBenchmark_Summary.csv
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd


VALID_TIERS = [
    "Tier1_ApprovedLate",
    "Tier2_Phase3",
    "Tier3_PhaseEarly",
    "Tier4_OtherBroad",
]


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
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

    # remove obvious trial clutter
    s = re.sub(r"\bduration of treatment\b.*", " ", s)
    s = re.sub(r"\boral\b", " ", s)
    s = re.sub(r"\bintravenous\b", " ", s)
    s = re.sub(r"\biv\b", " ", s)

    # remove dose text
    s = re.sub(r"\b\d+(\.\d+)?\s*(mg|mcg|g|ml|iu|units?)\b", " ", s)

    # common salts / suffixes
    salt_words = [
        "hydrochloride", "hcl", "sodium", "potassium", "calcium", "succinate",
        "tartrate", "maleate", "phosphate", "sulfate", "acetate", "chloride",
        "nitrate", "mesylate", "besylate", "benzoate", "bromide", "citrate",
        "lactate", "malate", "vfrm"
    ]
    s = re.sub(r"\b(" + "|".join(map(re.escape, salt_words)) + r")\b", " ", s)

    # remove protocol codes like MK0462
    s = re.sub(r"\b[a-z]{1,5}\d{2,}\b", " ", s)

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


# ------------------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------------------
def load_ranked_drugs(drug_ranking_file: Path) -> pd.DataFrame:
    if not drug_ranking_file.exists():
        raise FileNotFoundError(f"Drug ranking file not found: {drug_ranking_file}")

    print(f"?? Reading ranked drugs: {drug_ranking_file}")
    df = pd.read_csv(drug_ranking_file)
    print(f"   ? Shape: {df.shape}")

    drug_col = detect_drug_column(df, ["DrugName_best", "DrugName", "Drug"])
    print(f"   ? Drug column: {drug_col}")

    if "DrugRank" in df.columns:
        df = df.sort_values("DrugRank", ascending=True).reset_index(drop=True)
    elif "DrugScore" in df.columns:
        df = df.sort_values("DrugScore", ascending=False).reset_index(drop=True)
        df["DrugRank"] = range(1, len(df) + 1)
    else:
        df["DrugRank"] = range(1, len(df) + 1)

    df["Drug_norm"] = df[drug_col].map(normalize_drug_name)
    df = df[df["Drug_norm"] != ""].copy()
    df = df.sort_values("DrugRank", ascending=True).drop_duplicates("Drug_norm", keep="first").reset_index(drop=True)

    print(f"   ? Unique normalized predicted drugs: {df.shape[0]:,}")
    return df


def load_migraine_drugs_csv(migraine_file: Path) -> pd.DataFrame:
    if not migraine_file.exists():
        raise FileNotFoundError(f"Migraine drug file not found: {migraine_file}")

    print(f"?? Reading migraine drug file: {migraine_file}")
    df = pd.read_csv(migraine_file)
    print(f"   ? Shape: {df.shape}")

    drug_col = detect_drug_column(df, ["drug_name", "DrugName", "Drug"])
    out = df.copy()
    out["drug_name_raw"] = out[drug_col].astype(str)
    out["Drug_norm"] = out["drug_name_raw"].map(normalize_drug_name)
    out = out[out["Drug_norm"] != ""].copy()

    for col in ["phase", "status", "source", "drug_type", "max_phase"]:
        if col not in out.columns:
            out[col] = ""

    out = out.reset_index(drop=True)
    print(f"   ? Unique non-empty normalized entries: {out['Drug_norm'].nunique():,}")
    return out


def load_all_drugs_universe(all_drugs_file: Path) -> Set[str]:
    if not all_drugs_file.exists():
        raise FileNotFoundError(f"ALL drugs universe file not found: {all_drugs_file}")

    print(f"?? Reading ALL-drugs universe: {all_drugs_file}")
    df = pd.read_csv(all_drugs_file, low_memory=False)

    drug_col = detect_drug_column(df, ["drug_name", "DrugName", "Drug", "drug_norm", "Drug_norm"])
    print(f"   ? Drug column: {drug_col}")

    universe = set(df[drug_col].dropna().astype(str).map(normalize_drug_name))
    universe.discard("")
    print(f"   ? Unique normalized ALL-drugs universe: {len(universe):,}")
    return universe


# ------------------------------------------------------------------------------
# Tier construction
# ------------------------------------------------------------------------------
def canonical_phase_text(row: pd.Series) -> str:
    vals = [
        str(row.get("phase", "")),
        str(row.get("max_phase", "")),
        str(row.get("status", "")),
        str(row.get("source", "")),
    ]
    txt = " | ".join(v for v in vals if v and v != "nan").upper()
    return txt


def assign_phase_tier(phase_text: str) -> str:
    pt = phase_text.upper()

    # Tier 1: approved / marketed / phase 4
    if ("APPROVED" in pt) or ("PHASE4" in pt) or ("PHASE 4" in pt):
        return "Tier1_ApprovedLate"

    # Tier 2: phase 3 or phase 2/3
    if ("PHASE3" in pt) or ("PHASE 3" in pt) or ("PHASE2/PHASE3" in pt) or ("PHASE 2/PHASE 3" in pt):
        return "Tier2_Phase3"

    # Tier 3: phase 2 / phase 1/2 / phase 1
    if (
        ("PHASE2" in pt) or ("PHASE 2" in pt) or
        ("PHASE1/PHASE2" in pt) or ("PHASE 1/PHASE 2" in pt) or
        ("PHASE1" in pt) or ("PHASE 1" in pt)
    ):
        return "Tier3_PhaseEarly"

    # Tier 4: everything else
    return "Tier4_OtherBroad"


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


def build_phase_tiered_reference(migraine_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        migraine_df.groupby("Drug_norm", as_index=False)
        .apply(collapse_metadata, include_groups=False)
        .reset_index()
    )

    if "Drug_norm" not in grouped.columns:
        grouped = grouped.rename(columns={"index": "Drug_norm"})
    if "level_0" in grouped.columns:
        grouped = grouped.drop(columns=["level_0"])

    grouped["phase_text"] = grouped.apply(
        lambda r: " | ".join([
            str(r.get("phase_values", "")),
            str(r.get("max_phase_values", "")),
            str(r.get("status_values", "")),
            str(r.get("source_values", "")),
        ]),
        axis=1
    )

    grouped["Tier"] = grouped["phase_text"].map(assign_phase_tier)

    grouped = grouped[
        [
            "Drug_norm",
            "drug_name_example",
            "Tier",
            "phase_values",
            "status_values",
            "source_values",
            "drug_type_values",
            "max_phase_values",
            "phase_text",
            "n_records",
        ]
    ].sort_values(["Tier", "Drug_norm"]).reset_index(drop=True)

    return grouped


def tier_sets_from_df(tier_df: pd.DataFrame) -> Dict[str, Set[str]]:
    out = {}
    for tier in VALID_TIERS:
        out[tier] = set(tier_df.loc[tier_df["Tier"] == tier, "Drug_norm"].tolist())
    return out


# ------------------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------------------
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

    print_block("TIERED DRUG BENCHMARK EVALUATION")

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

            print(f"{k:>6}  {obs:>8}  {exp:>9.2f}  {fe:>7.2f}  {prec:>10.4f}  {rec:>10.4f}  {pval:>12.3e}")

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
    summary_df["Tier_Order"] = [1, 2, 3, 4]
    summary_df = summary_df.sort_values("Tier_Order").drop(columns=["Tier_Order"]).reset_index(drop=True)

    return detailed_df, manuscript_df, summary_df


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Phase-tiered migraine drug benchmark.")
    ap.add_argument("phenotype")
    ap.add_argument("--base-dir", default="/data/ascher02/uqmmune1/ANNOVAR")
    ap.add_argument("--migraine-drugs-file", default="/data/ascher02/uqmmune1/ANNOVAR/migraine_drugs.csv")
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

    print_block("INPUTS")
    print(f"Phenotype:           {phenotype}")
    print(f"Migraine drugs file: {migraine_file}")
    print(f"Drug ranking file:   {drug_ranking_file}")
    print(f"ALL drugs DB:        {all_drugs_file}")
    print(f"Top-K values:        {topk_list}")

    ranked_df = load_ranked_drugs(drug_ranking_file)
    migraine_df = load_migraine_drugs_csv(migraine_file)
    universe_all = load_all_drugs_universe(all_drugs_file)

    print_block("AUTO-BUILDING PHASE-BASED TIERED REFERENCE")
    tier_df = build_phase_tiered_reference(migraine_df)
    tier_sets = tier_sets_from_df(tier_df)

    for tier in VALID_TIERS:
        print(f"{tier}: {len(tier_sets[tier]):,} drugs")

    out_dir = ensure_dir(
        base_dir / phenotype / "GeneDifferentialExpression" / "Files" /
        "UltimateCompleteRankingAnalysis" / "DrugIntegration" / "TieredBenchmarkPhase"
    )

    tier_file = out_dir / "migraine_drugs_tiered_PHASE.csv"
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

    detailed_df.to_csv(detailed_path, index=False)
    manuscript_df.to_csv(manuscript_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print_block("MANUSCRIPT TABLE")
    print(summary_df.to_string(index=False))

    print_block("DONE")
    print("Saved:")
    print(f"  {tier_file}")
    print(f"  {detailed_path}")
    print(f"  {manuscript_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()