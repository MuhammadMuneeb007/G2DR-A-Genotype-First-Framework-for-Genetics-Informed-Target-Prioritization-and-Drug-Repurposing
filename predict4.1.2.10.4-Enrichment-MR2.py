#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import pandas as pd
from pathlib import Path

# =============================================================================
# HARD-CODED SETTINGS
# =============================================================================
DISEASE_QUERY = "migraine"
OPEN_TARGETS_URL = "https://api.platform.opentargets.org/api/v4/graphql"

COMPOSITE_FILE = Path(
    "/data/ascher02/uqmmune1/ANNOVAR/migraine/"
    "GeneDifferentialExpression/Files/UltimateCompleteRankingAnalysis/"
    "FinalIntegration/RANKED_final_all_scores.csv"
)

REFERENCE_FILE = Path(
    "/data/ascher02/uqmmune1/ANNOVAR/migraine_genes.csv"
)

OUTDIR = Path(
    "/data/ascher02/uqmmune1/ANNOVAR/migraine/"
    "GeneDifferentialExpression/Files/UltimateCompleteRankingAnalysis/"
    "OpenTargetsComparison"
)

TOPK_LIST = [50, 100, 200, 500]


# =============================================================================
# HELPERS
# =============================================================================
def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def clean_ensg(x):
    s = safe_str(x)
    if not s:
        return ""
    if s.startswith("ENSG"):
        return s.split(".")[0]
    return ""


def norm_symbol(x):
    return safe_str(x).upper()


def print_block(title, width=120):
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def gql(query, variables=None):
    r = requests.post(
        OPEN_TARGETS_URL,
        json={"query": query, "variables": variables or {}},
        timeout=120
    )
    r.raise_for_status()
    data = r.json()
    if "errors" in data:
        raise RuntimeError(f"GraphQL error: {data['errors']}")
    return data["data"]


# =============================================================================
# OPEN TARGETS
# =============================================================================
def search_disease(query_string):
    query = """
    query SearchDisease($q: String!) {
      search(queryString: $q, entityNames: ["disease"], page: {index: 0, size: 20}) {
        total
        hits {
          id
          name
          entity
          score
        }
      }
    }
    """
    data = gql(query, {"q": query_string})
    hits = data["search"]["hits"]

    if not hits:
        raise RuntimeError(f"No disease hit found in Open Targets for query: {query_string}")

    print_block("OPEN TARGETS DISEASE SEARCH")
    for i, h in enumerate(hits[:10], 1):
        print(f"{i:>2}. {h['name']} | {h['id']} | score={h.get('score', 0):.4f}")

    best = hits[0]
    print(f"\nUsing best disease hit: {best['name']} ({best['id']})")
    return best["id"], best["name"]


def fetch_associated_targets(efo_id, page_size=500):
    query = """
    query DiseaseTargets($efoId: String!, $index: Int!, $size: Int!) {
      disease(efoId: $efoId) {
        id
        name
        associatedTargets(page: {index: $index, size: $size}) {
          count
          rows {
            score
            target {
              id
              approvedSymbol
              approvedName
            }
          }
        }
      }
    }
    """

    all_rows = []
    index = 0
    total_count = None

    while True:
        data = gql(query, {"efoId": efo_id, "index": index, "size": page_size})
        disease = data["disease"]
        assoc = disease["associatedTargets"]
        rows = assoc["rows"]

        if total_count is None:
            total_count = assoc["count"]
            print(f"\nTotal Open Targets associated targets reported: {total_count:,}")

        if not rows:
            break

        all_rows.extend(rows)
        print(f"Fetched page {index:>3} | cumulative rows = {len(all_rows):,}")

        if len(rows) < page_size:
            break

        index += 1

    out = []
    for r in all_rows:
        t = r.get("target") or {}
        out.append(
            {
                "ensembl_gene_id": clean_ensg(t.get("id")),
                "gene": norm_symbol(t.get("approvedSymbol")),
                "approved_name": safe_str(t.get("approvedName")),
                "ot_score": r.get("score", None),
            }
        )

    df = pd.DataFrame(out)
    df = df[df["ensembl_gene_id"] != ""].copy()
    df = df.drop_duplicates(subset=["ensembl_gene_id"]).copy()
    df = df.sort_values(["ot_score", "gene"], ascending=[False, True]).reset_index(drop=True)
    df["ot_rank"] = range(1, len(df) + 1)
    return df


# =============================================================================
# LOAD FILES
# =============================================================================
def load_reference_file(path):
    print_block("LOADING REFERENCE FILE")
    print(f"Reading: {path}")
    df = pd.read_csv(path)

    required = {"ensembl_gene_id", "gene"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Reference file missing columns: {sorted(missing)}")

    df = df.copy()
    df["ensembl_gene_id"] = df["ensembl_gene_id"].map(clean_ensg)
    df["gene"] = df["gene"].map(norm_symbol)

    df = df[df["ensembl_gene_id"] != ""].copy()
    df = df.drop_duplicates(subset=["ensembl_gene_id"]).reset_index(drop=True)

    print(f"Loaded reference genes with ENSG IDs: {len(df):,}")
    return df


def load_composite_file(path):
    print_block("LOADING COMPOSITE FILE")
    print(f"Reading: {path}")

    if not path.exists():
        raise FileNotFoundError(f"Composite file not found:\n{path}")

    df = pd.read_csv(path)

    if "Gene" not in df.columns:
        raise ValueError("Composite file must contain a 'Gene' column.")

    score_col = None
    for c in [
        "Final_Combined_Score",
        "Combined_Score",
        "Composite_Score",
        "Importance_Score",
        "Score"
    ]:
        if c in df.columns:
            score_col = c
            break

    if score_col is None:
        raise ValueError(
            "Composite file must contain one of: "
            "Final_Combined_Score, Combined_Score, Composite_Score, Importance_Score, Score"
        )

    out = pd.DataFrame()
    out["ensembl_gene_id"] = df["Gene"].map(clean_ensg)
    out["composite_score"] = pd.to_numeric(df[score_col], errors="coerce")

    if "Direction" in df.columns:
        out["Direction"] = df["Direction"].map(safe_str)
    else:
        out["Direction"] = ""

    out = out[out["ensembl_gene_id"] != ""].copy()
    out = out.sort_values("composite_score", ascending=False).reset_index(drop=True)
    out = out.drop_duplicates(subset=["ensembl_gene_id"]).reset_index(drop=True)
    out["my_rank"] = range(1, len(out) + 1)

    print(f"Loaded ranked genes with ENSG IDs: {len(out):,}")
    print(f"Using score column: {score_col}")
    return out


# =============================================================================
# LOOKUPS
# =============================================================================
def make_ensg_set(df):
    return set(df["ensembl_gene_id"].dropna().astype(str).str.strip()) - {""}


def build_lookup(df, source_name):
    cols = ["ensembl_gene_id"]
    for c in ["gene", "approved_name", "ot_score", "ot_rank", "composite_score", "my_rank", "Direction"]:
        if c in df.columns:
            cols.append(c)

    out = df[cols].copy()
    out["source"] = source_name
    out = out.drop_duplicates(subset=["ensembl_gene_id", "source"]).reset_index(drop=True)
    return out


def subset_lookup(ensg_set, *lookups):
    frames = []
    for lu in lookups:
        if lu is not None and not lu.empty:
            frames.append(lu[lu["ensembl_gene_id"].isin(ensg_set)])

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, axis=0, ignore_index=True)
    out = out.drop_duplicates(subset=["ensembl_gene_id", "source"]).reset_index(drop=True)
    return out.sort_values(["ensembl_gene_id", "source"])


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================
def topk_recovery(ranked_df, reference_set, k_list, label):
    ranked_ids = ranked_df["ensembl_gene_id"].tolist()
    ref_n = len(reference_set)

    rows = []
    print(f"\nTop-K recovery for {label}:")
    print(f"{'TopK':>6}  {'Recovered':>10}  {'%RefRecovered':>15}  {'Precision@K':>12}")
    print("-" * 55)

    for k in k_list:
        kk = min(k, len(ranked_ids))
        top_ids = set(ranked_ids[:kk])
        rec = len(top_ids & reference_set)
        pct = (100.0 * rec / ref_n) if ref_n else 0.0
        prec = (100.0 * rec / kk) if kk else 0.0

        print(f"{kk:>6}  {rec:>10}  {pct:>15.2f}  {prec:>12.2f}")

        rows.append(
            {
                "Method": label,
                "TopK": kk,
                "Recovered": rec,
                "Percent_reference_recovered": pct,
                "Precision_at_K": prec,
            }
        )

    return pd.DataFrame(rows)


def run_overlap_analysis(label, my_df, ref_df, ot_df, outdir):
    print_block(f"ANALYSIS: {label}")

    ref_set = make_ensg_set(ref_df)
    my_set = make_ensg_set(my_df)
    ot_set = make_ensg_set(ot_df)

    my_vs_ot = my_set & ot_set
    my_vs_ref = my_set & ref_set
    ot_vs_ref = ot_set & ref_set
    all_three = my_set & ot_set & ref_set

    g2dr_ref_not_in_ot = my_vs_ref - ot_set
    ot_ref_not_in_g2dr = ot_vs_ref - my_set

    ref_missing_in_ot = ref_set - ot_set
    ref_missing_in_my = ref_set - my_set
    my_only = my_set - ot_set
    ot_only = ot_set - my_set

    my_lu = build_lookup(my_df, "my_analysis")
    ot_lu = build_lookup(ot_df, "open_targets")
    ref_lu = build_lookup(ref_df, "reference")

    subset_lookup(my_vs_ot, my_lu, ot_lu, ref_lu).to_csv(outdir / f"{label}.common_my_vs_opentargets.csv", index=False)
    subset_lookup(my_vs_ref, my_lu, ot_lu, ref_lu).to_csv(outdir / f"{label}.common_my_vs_reference.csv", index=False)
    subset_lookup(ot_vs_ref, my_lu, ot_lu, ref_lu).to_csv(outdir / f"{label}.common_opentargets_vs_reference.csv", index=False)
    subset_lookup(all_three, my_lu, ot_lu, ref_lu).to_csv(outdir / f"{label}.common_all_three.csv", index=False)
    subset_lookup(g2dr_ref_not_in_ot, my_lu, ot_lu, ref_lu).to_csv(outdir / f"{label}.g2dr_reference_genes_not_in_opentargets.csv", index=False)
    subset_lookup(ot_ref_not_in_g2dr, my_lu, ot_lu, ref_lu).to_csv(outdir / f"{label}.opentargets_reference_genes_not_in_g2dr.csv", index=False)
    subset_lookup(ref_missing_in_ot, my_lu, ot_lu, ref_lu).to_csv(outdir / f"{label}.reference_missing_in_opentargets.csv", index=False)
    subset_lookup(ref_missing_in_my, my_lu, ot_lu, ref_lu).to_csv(outdir / f"{label}.reference_missing_in_my_analysis.csv", index=False)
    subset_lookup(my_only, my_lu, ot_lu, ref_lu).to_csv(outdir / f"{label}.my_analysis_not_in_opentargets.csv", index=False)
    subset_lookup(ot_only, my_lu, ot_lu, ref_lu).to_csv(outdir / f"{label}.opentargets_not_in_my_analysis.csv", index=False)

    ref_n = len(ref_set) if ref_set else 1
    my_rec = 100.0 * len(my_vs_ref) / ref_n
    ot_rec = 100.0 * len(ot_vs_ref) / ref_n
    all_rec = 100.0 * len(all_three) / ref_n
    g2dr_unique_ref_pct = 100.0 * len(g2dr_ref_not_in_ot) / ref_n

    print(f"Reference genes                    : {len(ref_set):,}")
    print(f"My analysis genes                 : {len(my_set):,}")
    print(f"Open Targets genes                : {len(ot_set):,}")
    print("-" * 120)
    print(f"Common genes: my analysis vs OT   : {len(my_vs_ot):,}")
    print(f"Common genes: my analysis vs ref  : {len(my_vs_ref):,}")
    print(f"Common genes: OT vs ref           : {len(ot_vs_ref):,}")
    print(f"Common genes in all three         : {len(all_three):,}")
    print(f"G2DR ref genes not in OT          : {len(g2dr_ref_not_in_ot):,}")
    print(f"OT ref genes not in G2DR          : {len(ot_ref_not_in_g2dr):,}")
    print("-" * 120)
    print(f"Reference recovered by G2DR       : {len(my_vs_ref):,} / {len(ref_set):,} ({my_rec:.2f}%)")
    print(f"Reference recovered by OT         : {len(ot_vs_ref):,} / {len(ref_set):,} ({ot_rec:.2f}%)")
    print(f"Reference common to all three     : {len(all_three):,} / {len(ref_set):,} ({all_rec:.2f}%)")
    print(f"G2DR-only recovered reference     : {len(g2dr_ref_not_in_ot):,} / {len(ref_set):,} ({g2dr_unique_ref_pct:.2f}%)")
    print("-" * 120)
    print(f"Reference missing in OT           : {len(ref_missing_in_ot):,}")
    print(f"Reference missing in G2DR         : {len(ref_missing_in_my):,}")
    print(f"G2DR genes not in OT              : {len(my_only):,}")
    print(f"OT genes not in G2DR              : {len(ot_only):,}")

    summary_df = pd.DataFrame(
        [
            ["Reference genes", len(ref_set)],
            ["My analysis genes", len(my_set)],
            ["Open Targets genes", len(ot_set)],
            ["My analysis ∩ Open Targets", len(my_vs_ot)],
            ["My analysis ∩ Reference", len(my_vs_ref)],
            ["Open Targets ∩ Reference", len(ot_vs_ref)],
            ["All three", len(all_three)],
            ["G2DR reference genes not in Open Targets", len(g2dr_ref_not_in_ot)],
            ["Open Targets reference genes not in G2DR", len(ot_ref_not_in_g2dr)],
            ["Reference missing in Open Targets", len(ref_missing_in_ot)],
            ["Reference missing in My analysis", len(ref_missing_in_my)],
            ["My analysis not in Open Targets", len(my_only)],
            ["Open Targets not in My analysis", len(ot_only)],
        ],
        columns=["Metric", "Count"]
    )
    summary_df.to_csv(outdir / f"{label}.summary_counts.csv", index=False)

    topk_my = topk_recovery(my_df, ref_set, TOPK_LIST, f"{label} | G2DR")
    topk_ot = topk_recovery(ot_df, ref_set, TOPK_LIST, f"{label} | Open Targets")
    topk_df = pd.concat([topk_my, topk_ot], ignore_index=True)
    topk_df.to_csv(outdir / f"{label}.topk_recovery_comparison.csv", index=False)

    return {
        "label": label,
        "reference_n": len(ref_set),
        "my_n": len(my_set),
        "ot_n": len(ot_set),
        "my_vs_ot": len(my_vs_ot),
        "my_vs_ref": len(my_vs_ref),
        "ot_vs_ref": len(ot_vs_ref),
        "all_three": len(all_three),
        "g2dr_ref_not_in_ot": len(g2dr_ref_not_in_ot),
        "ot_ref_not_in_g2dr": len(ot_ref_not_in_g2dr),
        "my_recovery_pct": my_rec,
        "ot_recovery_pct": ot_rec,
        "all_three_pct": all_rec,
        "g2dr_unique_ref_pct": g2dr_unique_ref_pct,
    }


# =============================================================================
# MAIN
# =============================================================================
def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print_block("INPUTS")
    print(f"Disease query   : {DISEASE_QUERY}")
    print(f"Composite file  : {COMPOSITE_FILE}")
    print(f"Reference file  : {REFERENCE_FILE}")
    print(f"Output dir      : {OUTDIR}")

    ref_df = load_reference_file(REFERENCE_FILE)
    my_df_all = load_composite_file(COMPOSITE_FILE)

    efo_id, disease_name = search_disease(DISEASE_QUERY)
    ot_df = fetch_associated_targets(efo_id)

    print_block("COUNTS")
    print(f"Reference genes loaded               : {len(ref_df):,}")
    print(f"My ranked genes loaded (all)         : {len(my_df_all):,}")
    print(f"Open Targets genes loaded            : {len(ot_df):,}")

    ot_target_set = make_ensg_set(ot_df)

    # Filtered G2DR: keep only genes that exist as Open Targets targets
    my_df_target_only = my_df_all[my_df_all["ensembl_gene_id"].isin(ot_target_set)].copy()
    my_df_target_only = my_df_target_only.sort_values("my_rank").reset_index(drop=True)
    my_df_target_only["my_rank"] = range(1, len(my_df_target_only) + 1)

    print(f"My ranked genes after OT-target filter: {len(my_df_target_only):,}")

    # Run both analyses
    results_all = run_overlap_analysis("ALL_GENES", my_df_all, ref_df, ot_df, OUTDIR)
    results_target = run_overlap_analysis("TARGET_FILTERED", my_df_target_only, ref_df, ot_df, OUTDIR)

    comparison_df = pd.DataFrame([
        {
            "Analysis": "ALL_GENES",
            "My_analysis_genes": results_all["my_n"],
            "OpenTargets_genes": results_all["ot_n"],
            "Reference_genes": results_all["reference_n"],
            "My_vs_Reference": results_all["my_vs_ref"],
            "OpenTargets_vs_Reference": results_all["ot_vs_ref"],
            "All_three": results_all["all_three"],
            "G2DR_ref_not_in_OT": results_all["g2dr_ref_not_in_ot"],
            "My_recovery_percent": results_all["my_recovery_pct"],
            "OT_recovery_percent": results_all["ot_recovery_pct"],
        },
        {
            "Analysis": "TARGET_FILTERED",
            "My_analysis_genes": results_target["my_n"],
            "OpenTargets_genes": results_target["ot_n"],
            "Reference_genes": results_target["reference_n"],
            "My_vs_Reference": results_target["my_vs_ref"],
            "OpenTargets_vs_Reference": results_target["ot_vs_ref"],
            "All_three": results_target["all_three"],
            "G2DR_ref_not_in_OT": results_target["g2dr_ref_not_in_ot"],
            "My_recovery_percent": results_target["my_recovery_pct"],
            "OT_recovery_percent": results_target["ot_recovery_pct"],
        }
    ])
    comparison_df.to_csv(OUTDIR / "comparison_all_vs_target_filtered.csv", index=False)

    print_block("SIDE-BY-SIDE COMPARISON")
    print(comparison_df.to_string(index=False))

    print_block("DROP-IN INTERPRETATION TEXT")
    print(
        f"In an external contextual comparison against Open Targets, we evaluated G2DR in two modes: "
        f"(i) across the full ranked gene universe and (ii) after restricting G2DR to genes that were "
        f"also represented in the Open Targets target space. In the full analysis, G2DR ranked "
        f"{results_all['my_n']:,} genes and recovered {results_all['my_vs_ref']:,} curated migraine "
        f"reference genes ({results_all['my_recovery_pct']:.2f}%), including "
        f"{results_all['g2dr_ref_not_in_ot']:,} curated reference genes absent from the Open Targets "
        f"migraine target list. After restricting G2DR to Open Targets-represented targets, "
        f"{results_target['my_n']:,} ranked genes remained for comparison. This target-filtered analysis "
        f"provides a fairer overlap benchmark against Open Targets, whereas the full analysis better reflects "
        f"the broader genotype-first design of G2DR, which retains coding and non-coding candidates to capture "
        f"more complete disease biology."
    )

    print_block("FILES WRITTEN")
    for p in sorted(OUTDIR.glob("*.csv")):
        print(p.name)


if __name__ == "__main__":
    main()
