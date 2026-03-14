#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR_DEFAULT = Path("/data/ascher02/uqmmune1/ANNOVAR")


# =============================================================================
# HELPERS
# =============================================================================
def safe_str(x):
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"", "nan", "none", "null"}:
        return ""
    return s


def strip_ensg_version(x):
    return re.sub(r"\.\d+$", "", safe_str(x))


def normalize_drug_name(x):
    s = safe_str(x).lower()
    if not s:
        return ""
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"\[.*?\]", " ", s)
    s = s.replace("&", " and ")
    s = re.sub(r"[/,+;]", " ", s)
    s = re.sub(r"\b\d+(\.\d+)?\s*(mg|mcg|g|ml|iu|units?)\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def yes_no(x):
    s = safe_str(x).lower()
    if s in {"yes", "true", "1", "y"}:
        return "Yes"
    if s in {"no", "false", "0", "n"}:
        return "No"
    return ""


def first_nonempty(values, default=""):
    for v in values:
        s = safe_str(v)
        if s:
            return s
    return default


def unique_join(values, sep=" | "):
    seen = set()
    out = []
    for v in values:
        s = safe_str(v)
        if s and s not in seen:
            out.append(s)
            seen.add(s)
    return sep.join(out)


# =============================================================================
# LOADERS
# =============================================================================
def load_ranked_genes(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing ranked genes file: {path}")

    df = pd.read_csv(path)
    if "Gene" not in df.columns:
        raise ValueError(f"'Gene' column missing in {path}")

    df = df.copy()
    df["Gene"] = df["Gene"].astype(str).map(strip_ensg_version)

    if "Rank" not in df.columns:
        if "Importance_Score" in df.columns:
            df = df.sort_values("Importance_Score", ascending=False).reset_index(drop=True)
            df["Rank"] = np.arange(1, len(df) + 1)
        else:
            df["Rank"] = np.arange(1, len(df) + 1)

    for col, default in [
        ("Symbol", ""),
        ("Importance_Score", np.nan),
        ("Direction", ""),
        ("Status", ""),
        ("Confidence_Tier", ""),
        ("Direction_Consistency", np.nan),
        ("Total_Hits", np.nan),
        ("N_Tissues", np.nan),
        ("N_Databases", np.nan),
        ("N_Methods", np.nan),
        ("Mean_Unified_Effect", np.nan),
        ("Max_Unified_Effect", np.nan),
        ("Min_FDR", np.nan),
        ("Mean_FDR", np.nan),
        ("Median_FDR", np.nan),
    ]:
        if col not in df.columns:
            df[col] = default

    df["DiseaseDirection"] = df["Direction"].replace(
        {
            "Up": "higher in cases",
            "Down": "lower in cases",
            "Mixed": "unclear",
            "up": "higher in cases",
            "down": "lower in cases",
            "mixed": "unclear",
        }
    )
    df["DiseaseDirection"] = df["DiseaseDirection"].fillna("unclear")
    return df


def load_migraine_genes(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["ensembl_gene_id"])

    df = pd.read_csv(path)
    if "ensembl_gene_id" in df.columns:
        df = df.copy()
        df["ensembl_gene_id"] = df["ensembl_gene_id"].astype(str).map(strip_ensg_version)
    return df


def load_drug_directionality(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing drug directionality table: {path}")

    df = pd.read_csv(path)

    required = ["DrugName", "Gene"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"'{c}' column missing in {path}")

    df = df.copy()
    df["Gene"] = df["Gene"].astype(str).map(strip_ensg_version)
    df["DrugNorm"] = df["DrugName"].astype(str).map(normalize_drug_name)

    for col, default in [
        ("DrugRank", np.nan),
        ("DrugScore", np.nan),
        ("Approved", ""),
        ("KnownMigraineDrug", "No"),
        ("DirectionMatch", ""),
        ("ActionType", ""),
        ("Symbol", ""),
        ("KnownMigraineGene", "No"),
        ("ActionEvidenceSource", ""),
        ("ActionEvidenceText", ""),
        ("Confidence_Tier", ""),
    ]:
        if col not in df.columns:
            df[col] = default

    return df


# =============================================================================
# BUILD TOP GENE PANEL
# =============================================================================
def build_gene_panel(ranked_genes: pd.DataFrame,
                     migraine_genes_df: pd.DataFrame,
                     drug_df: pd.DataFrame,
                     top_n: int = 20) -> pd.DataFrame:

    known_gene_set = set()
    if "ensembl_gene_id" in migraine_genes_df.columns:
        known_gene_set = set(migraine_genes_df["ensembl_gene_id"].dropna().astype(str))

    drug_by_gene = (
        drug_df.groupby("Gene", dropna=False)
        .agg(
            LinkedDrugs=("DrugName", lambda s: unique_join(s.tolist(), sep=", ")),
            N_LinkedDrugs=("DrugName", "nunique"),
            AnyApproved=("Approved", lambda s: "Yes" if (s.astype(str).str.lower() == "yes").any() else "No"),
            AnyKnownMigraineDrug=("KnownMigraineDrug", lambda s: "Yes" if (s.astype(str).str.lower() == "yes").any() else "No"),
            BestDirectionality=("DirectionMatch", lambda s: first_nonempty(
                [x for x in s if safe_str(x).lower() == "consistent"],
                default=first_nonempty(s.tolist(), default="")
            )),
        )
        .reset_index()
    )

    top = ranked_genes.sort_values("Rank", ascending=True).head(top_n).copy()
    top["KnownMigraineGene"] = top["Gene"].isin(known_gene_set).map({True: "Yes", False: "No"})

    top = top.merge(drug_by_gene, on="Gene", how="left")

    top["LinkedDrugs"] = top["LinkedDrugs"].fillna("")
    top["N_LinkedDrugs"] = top["N_LinkedDrugs"].fillna(0).astype(int)
    top["AnyApproved"] = top["AnyApproved"].fillna("No")
    top["AnyKnownMigraineDrug"] = top["AnyKnownMigraineDrug"].fillna("No")
    top["BestDirectionality"] = top["BestDirectionality"].fillna("")

    top["WhyPrioritized"] = (
        "Rank " + top["Rank"].astype(int).astype(str)
        + "; score=" + top["Importance_Score"].round(3).astype(str)
        + "; hits=" + top["Total_Hits"].fillna(0).astype(int).astype(str)
    )

    top["EvidenceLayers"] = (
        "tissues=" + top["N_Tissues"].fillna(0).astype(int).astype(str)
        + "; databases=" + top["N_Databases"].fillna(0).astype(int).astype(str)
        + "; methods=" + top["N_Methods"].fillna(0).astype(int).astype(str)
    )

    out = top[
        [
            "Rank",
            "Gene",
            "Symbol",
            "WhyPrioritized",
            "EvidenceLayers",
            "DiseaseDirection",
            "KnownMigraineGene",
            "Confidence_Tier",
            "N_LinkedDrugs",
            "AnyApproved",
            "AnyKnownMigraineDrug",
            "BestDirectionality",
            "LinkedDrugs",
        ]
    ].copy()

    out = out.rename(
        columns={
            "Rank": "GeneRank",
            "AnyApproved": "HasApprovedDrug",
            "AnyKnownMigraineDrug": "LinkedKnownMigraineDrug",
        }
    )

    return out


# =============================================================================
# BUILD TOP DRUG PANEL
# =============================================================================
def build_drug_panel(drug_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    df = drug_df.copy()

    # collapse repeated drug-gene rows to drug-level summary
    drug_panel = (
        df.groupby(["DrugNorm", "DrugName"], dropna=False)
        .agg(
            DrugRank=("DrugRank", "min"),
            DrugScore=("DrugScore", "max"),
            Approved=("Approved", lambda s: "Yes" if (s.astype(str).str.lower() == "yes").any() else "No"),
            KnownMigraineDrug=("KnownMigraineDrug", lambda s: "Yes" if (s.astype(str).str.lower() == "yes").any() else "No"),
            TargetGenes=("Symbol", lambda s: unique_join([x for x in s if safe_str(x)], sep=", ")),
            N_TargetGenes=("Gene", "nunique"),
            BestDirectionality=("DirectionMatch", lambda s: first_nonempty(
                [x for x in s if safe_str(x).lower() == "consistent"],
                default=first_nonempty(s.tolist(), default="")
            )),
            ActionTypes=("ActionType", lambda s: unique_join(s.tolist(), sep=", ")),
            EvidenceSources=("ActionEvidenceSource", lambda s: unique_join(s.tolist(), sep=", ")),
            BestGeneRank=("Gene", lambda s: np.nan),
        )
        .reset_index()
    )

    # compute best linked gene rank separately
    gene_rank_lookup = (
        df.groupby(["DrugNorm", "DrugName"], dropna=False)["Gene"]
        .count()
        .reset_index(name="tmp")
    )
    _ = gene_rank_lookup  # keeps linter quiet if not used

    # manual best gene rank from original df if available
    if "GeneRank" in df.columns:
        best_rank_df = (
            df.groupby(["DrugNorm", "DrugName"], dropna=False)["GeneRank"]
            .min()
            .reset_index(name="BestGeneRank")
        )
        drug_panel = drug_panel.drop(columns=["BestGeneRank"]).merge(
            best_rank_df, on=["DrugNorm", "DrugName"], how="left"
        )

    drug_panel = drug_panel.sort_values(["DrugRank", "DrugScore"], ascending=[True, False]).reset_index(drop=True)
    drug_panel = drug_panel.head(top_n).copy()

    drug_panel["TierCategory"] = np.where(
        drug_panel["Approved"] == "Yes",
        "Approved/Repurposable",
        "Investigational/Other"
    )

    out = drug_panel[
        [
            "DrugRank",
            "DrugName",
            "DrugScore",
            "TargetGenes",
            "N_TargetGenes",
            "BestGeneRank",
            "TierCategory",
            "BestDirectionality",
            "KnownMigraineDrug",
            "Approved",
            "ActionTypes",
            "EvidenceSources",
        ]
    ].copy()

    return out


# =============================================================================
# LATEX WRITER
# =============================================================================
def latex_escape(x):
    s = safe_str(x)
    repl = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s


def dataframe_to_latex_table(df: pd.DataFrame, caption: str, label: str, max_rows: int = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows).copy()

    cols = list(df.columns)
    colspec = "p{1.0cm}" + "p{2.0cm}" * (len(cols) - 1) if len(cols) > 1 else "p{3cm}"

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(rf"\begin{{tabular}}{{{'l' * len(cols)}}}")
    lines.append(r"\hline")
    lines.append(" & ".join([rf"\textbf{{{latex_escape(c)}}}" for c in cols]) + r" \\")
    lines.append(r"\hline")

    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                if np.isnan(v):
                    vals.append("")
                else:
                    vals.append(latex_escape(f"{v:.3f}"))
            else:
                vals.append(latex_escape(v))
        lines.append(" & ".join(vals) + r" \\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Step 6: top-gene / top-drug validation panel")
    parser.add_argument("phenotype", help="Phenotype name, e.g. migraine")
    parser.add_argument("--base-dir", default=str(BASE_DIR_DEFAULT))
    parser.add_argument("--top-genes", type=int, default=15)
    parser.add_argument("--top-drugs", type=int, default=15)
    args = parser.parse_args()

    phenotype = args.phenotype
    base_dir = Path(args.base_dir)

    files_dir = base_dir / phenotype / "GeneDifferentialExpression" / "Files"
    ranking_dir = files_dir / "UltimateCompleteRankingAnalysis"
    drug_dir = ranking_dir / "DrugIntegration"
    out_dir = ranking_dir / "ValidationPanel"
    out_dir.mkdir(parents=True, exist_ok=True)

    ranked_genes_file = ranking_dir / "RANKED_composite.csv"
    migraine_genes_file = files_dir / "migraine_genes.csv"
    if not migraine_genes_file.exists():
        migraine_genes_file = base_dir / "migraine_genes.csv"

    drug_directionality_file = drug_dir / "DrugDirectionalityTable_Rescued.csv"

    print("=" * 120)
    print("STEP 6: TOP-GENE / TOP-DRUG VALIDATION PANEL")
    print("=" * 120)
    print(f"Phenotype:                {phenotype}")
    print(f"Ranked genes file:        {ranked_genes_file}")
    print(f"Migraine genes file:      {migraine_genes_file}")
    print(f"Drug directionality file: {drug_directionality_file}")
    print(f"Output dir:               {out_dir}")
    print("=" * 120)

    ranked_genes = load_ranked_genes(ranked_genes_file)
    migraine_genes = load_migraine_genes(migraine_genes_file)
    drug_df = load_drug_directionality(drug_directionality_file)

    gene_panel = build_gene_panel(ranked_genes, migraine_genes, drug_df, top_n=args.top_genes)
    drug_panel = build_drug_panel(drug_df, top_n=args.top_drugs)

    gene_panel_file = out_dir / "TopGeneValidationPanel.csv"
    drug_panel_file = out_dir / "TopDrugValidationPanel.csv"
    gene_panel.to_csv(gene_panel_file, index=False)
    drug_panel.to_csv(drug_panel_file, index=False)

    gene_latex = dataframe_to_latex_table(
        gene_panel,
        caption="Top prioritized genes from the final migraine ranking, summarizing evidence support, disease direction, migraine relevance, and linked druggability information.",
        label="tab:top_gene_validation_panel",
        max_rows=args.top_genes,
    )

    drug_latex = dataframe_to_latex_table(
        drug_panel,
        caption="Top prioritized drugs from the final migraine repurposing analysis, summarizing linked target genes, ranking position, directionality support, approval status, and migraine relevance.",
        label="tab:top_drug_validation_panel",
        max_rows=args.top_drugs,
    )

    with open(out_dir / "TopGeneValidationPanel.tex", "w", encoding="utf-8") as f:
        f.write(gene_latex)

    with open(out_dir / "TopDrugValidationPanel.tex", "w", encoding="utf-8") as f:
        f.write(drug_latex)

    # manuscript helper text
    lines = []
    lines.append("STEP 6 MANUSCRIPT NOTES")
    lines.append("")
    lines.append(f"Top gene panel rows: {len(gene_panel)}")
    lines.append(f"Top drug panel rows: {len(drug_panel)}")
    lines.append("")
    lines.append("Top gene panel highlights:")
    for _, row in gene_panel.head(min(10, len(gene_panel))).iterrows():
        lines.append(
            f"{row['Symbol']} (rank {row['GeneRank']}): direction={row['DiseaseDirection']}, "
            f"known migraine gene={row['KnownMigraineGene']}, linked drugs={row['N_LinkedDrugs']}."
        )

    lines.append("")
    lines.append("Top drug panel highlights:")
    for _, row in drug_panel.head(min(10, len(drug_panel))).iterrows():
        lines.append(
            f"{row['DrugName']} (rank {row['DrugRank']}): targets={row['TargetGenes']}, "
            f"directionality={row['BestDirectionality']}, approved={row['Approved']}."
        )

    with open(out_dir / "Manuscript_Text_ValidationPanel.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\nOUTPUTS WRITTEN")
    print(f"  {gene_panel_file}")
    print(f"  {drug_panel_file}")
    print(f"  {out_dir / 'TopGeneValidationPanel.tex'}")
    print(f"  {out_dir / 'TopDrugValidationPanel.tex'}")
    print(f"  {out_dir / 'Manuscript_Text_ValidationPanel.txt'}")

    print("\nTOP GENE PANEL")
    print(gene_panel.head(15).to_string(index=False))

    print("\nTOP DRUG PANEL")
    print(drug_panel.head(15).to_string(index=False))


if __name__ == "__main__":
    main()