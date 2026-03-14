#!/usr/bin/env python3
"""
Database Importance — Discovery Ranking + Test Replication + Disease Enrichment
=============================================================================

Hard-coded thresholds (as requested):
- FDR < 0.10
- |effect| >= 0.50   where effect := coalesce(Effect, Log2FoldChange)

Signal definition (used everywhere):
  significant := (FDR < 0.10) AND (|effect| >= 0.50)

Evidence definition for DATABASE unit:
  Evidence triplet = unique (Gene, Tissue, Method)

Outputs (under <phenotype>/GeneDifferentialExpression/Files/DatabaseImportance):
A) Discovery ranking (train+val only)
 - database_importance_discovery_gene_level.csv
 - database_importance_discovery_evidence_level.csv
 - database_importance_discovery_combined.csv
 - database_importance_discovery_top10_table.tex
 - database_importance_discovery_report.txt

B) Disease enrichment (separate)
 - database_disease_enrichment_discovery.csv
 - database_disease_enrichment_test.csv

C) Test replication (separate)
 - database_test_replication_gene_level.csv
 - database_test_replication_evidence_level.csv

Run:
  python predict4.4.9-GetImportantDatabaseFinal.py migraine
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl

# -----------------------------
# HARD-CODED THRESHOLDS
# -----------------------------
FDR_THR: float = 0.10
EFFECT_THR: float = 0.50


def normalize_gene_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.strip()
         .str.lower()
         .str.split(".")
         .str[0]
    )


def load_known_genes(phenotype: str, base_path: Path, results_dir: Path) -> tuple[set[str], str]:
    candidates = [
        base_path / f"{phenotype}_genes.csv",
        Path(f"{phenotype}_genes.csv"),
        base_path / "migraine_genes.csv",
        Path("migraine_genes.csv"),
        base_path / "reference_genes.csv",
        Path("reference_genes.csv"),
        results_dir / f"{phenotype}_genes.csv",
        results_dir / "migraine_genes.csv",
        results_dir / "reference_genes.csv",
    ]
    ref_file = None
    for fp in candidates:
        if fp.exists():
            ref_file = fp
            break
    if ref_file is None:
        raise FileNotFoundError(
            "Reference genes file not found. Looked for: "
            f"{phenotype}_genes.csv / migraine_genes.csv / reference_genes.csv"
        )

    ref = pd.read_csv(ref_file)
    if "ensembl_gene_id" not in ref.columns:
        raise ValueError(f"{ref_file} missing required column: ensembl_gene_id")

    known = normalize_gene_series(ref["ensembl_gene_id"].dropna())
    known = set([g for g in known if g not in {"", "nan", "none"}])
    return known, ref_file.name


def detect_train_val_test_labels(dataset_values: list[str]) -> tuple[str, str, str]:
    train_label = None
    val_label = None
    test_label = None
    for ds in dataset_values:
        low = str(ds).lower()
        if "train" in low:
            train_label = ds
        elif "val" in low:
            val_label = ds
        elif "test" in low:
            test_label = ds
    if train_label is None or val_label is None or test_label is None:
        raise ValueError(f"Could not detect train/val/test labels from Dataset values: {dataset_values}")
    return train_label, val_label, test_label


def safe_neglog10(p: float) -> float:
    return float(-np.log10(max(float(p), 1e-300)))


def latex_escape(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
         .replace("&", "\\&")
         .replace("%", "\\%")
         .replace("$", "\\$")
         .replace("#", "\\#")
         .replace("_", "\\_")
         .replace("{", "\\{")
         .replace("}", "\\}")
         .replace("~", "\\textasciitilde{}")
         .replace("^", "\\textasciicircum{}")
    )


def write_top10_latex_discovery(df: pd.DataFrame, outpath: Path) -> None:
    top = df.head(10).copy()
    lines: list[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Top databases ranked by discovery (train+validation) signal enrichment of known disease genes.}")
    lines.append("\\label{tab:database_importance_discovery}")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.15}")
    lines.append("\\begin{tabular}{r p{4.6cm} r r r r}")
    lines.append("\\hline")
    lines.append("\\textbf{Rank} & \\textbf{Database} & \\textbf{$N$} & \\textbf{$k$} & \\textbf{FE} & \\textbf{Dir.Cons.} \\\\")
    lines.append("\\hline")
    for _, r in top.iterrows():
        lines.append(
            f"{int(r['Rank'])} & {latex_escape(str(r['Database']))} & "
            f"{int(r['N_genes'])} & {int(r['k_known_genes'])} & "
            f"{float(r['FE_genes']):.2f} & {float(r['DirectionConsistency']):.3f} \\\\"
        )
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    outpath.write_text("\n".join(lines), encoding="utf-8")


def empirical_p_hypergeom_geq(rng: np.random.Generator, M: int, K: int, N: int, k_obs: int, n_perm: int) -> float:
    if N <= 0 or M <= 0 or K <= 0 or K > M:
        return 1.0
    draws = rng.hypergeometric(ngood=K, nbad=M - K, nsample=N, size=n_perm)
    return (1.0 + float(np.sum(draws >= k_obs))) / (1.0 + n_perm)


def _as_set_of_str(x) -> set[str]:
    if x is None:
        return set()
    if isinstance(x, (list, tuple, set)):
        return set(x)
    if isinstance(x, np.ndarray):
        return set(x.tolist())
    return set()


def _as_set_of_evidence_tuples(x, cols: list[str]) -> set[tuple]:
    """
    Evidence from polars Struct often becomes list[dict] in pandas,
    sometimes list[tuple], sometimes ndarray. We normalize all to set[tuple(values...)].
    """
    if x is None:
        return set()
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if not isinstance(x, list):
        return set()

    out = set()
    for item in x:
        if item is None:
            continue
        if isinstance(item, dict):
            out.add(tuple(item.get(c) for c in cols))
        elif hasattr(item, "values") and not isinstance(item, (str, bytes)):
            # polars StructRow-like
            try:
                vals = list(item.values())
                out.add(tuple(vals))
            except Exception:
                pass
        elif isinstance(item, (list, tuple)):
            out.add(tuple(item))
        else:
            # unknown scalar
            out.add((item,))
    return out


def compute_gene_level_enrichment(
    df_sig: pl.DataFrame,
    all_units: list[str],
    unit_col: str,
    gene_col: str,
    known_set_in_universe: set[str],
    M_universe: int,
    n_perm: int,
    seed: int
) -> pd.DataFrame:
    units_df = pl.DataFrame({unit_col: all_units})

    df_sig = df_sig.with_columns(
        pl.col(gene_col).is_in(sorted(known_set_in_universe)).alias("is_known")
    )

    stats = (
        df_sig.group_by(unit_col)
        .agg([
            pl.col(gene_col).n_unique().alias("N_genes"),
            pl.col(gene_col).filter(pl.col("is_known")).n_unique().alias("k_known_genes"),
        ])
    )

    stats = units_df.join(stats, on=unit_col, how="left").fill_null(0)
    out = stats.to_pandas()

    K_size = len(known_set_in_universe)
    out["expected_known_genes"] = out["N_genes"].astype(float) * (K_size / M_universe if M_universe > 0 else 0.0)
    out["FE_genes"] = np.where(out["expected_known_genes"] > 0, out["k_known_genes"] / out["expected_known_genes"], 0.0)

    rng = np.random.default_rng(seed)
    out["p_emp_genes"] = [
        empirical_p_hypergeom_geq(rng, M_universe, K_size, int(N), int(k), n_perm)
        for N, k in zip(out["N_genes"], out["k_known_genes"])
    ]
    return out


def compute_evidence_level_enrichment(
    df_sig: pl.DataFrame,
    df_universe: pl.DataFrame,
    all_units: list[str],
    unit_col: str,
    evidence_cols: list[str],
    gene_col: str,
    known_set_in_universe: set[str],
    n_perm: int,
    seed: int
) -> tuple[pd.DataFrame, int, int]:
    units_df = pl.DataFrame({unit_col: all_units})

    df_univ = df_universe.with_columns(
        pl.col(gene_col).is_in(sorted(known_set_in_universe)).alias("is_known")
    )
    evU_size = df_univ.select(pl.struct(evidence_cols).alias("ev")).unique().height
    evK_size = (
        df_univ.filter(pl.col("is_known"))
               .select(pl.struct(evidence_cols).alias("ev"))
               .unique()
               .height
    )

    df_sig2 = df_sig.with_columns(
        pl.col(gene_col).is_in(sorted(known_set_in_universe)).alias("is_known")
    )

    ev_stats = (
        df_sig2.group_by(unit_col)
        .agg([
            pl.struct(evidence_cols).n_unique().alias("N_evidence"),
            pl.struct(evidence_cols).filter(pl.col("is_known")).n_unique().alias("k_known_evidence"),
        ])
    )

    ev_stats = units_df.join(ev_stats, on=unit_col, how="left").fill_null(0)
    out = ev_stats.to_pandas()

    out["expected_known_evidence"] = out["N_evidence"].astype(float) * (evK_size / evU_size if evU_size > 0 else 0.0)
    out["FE_evidence"] = np.where(out["expected_known_evidence"] > 0, out["k_known_evidence"] / out["expected_known_evidence"], 0.0)

    rng = np.random.default_rng(seed)
    out["p_emp_evidence"] = [
        empirical_p_hypergeom_geq(rng, evU_size, evK_size, int(N), int(k), n_perm)
        for N, k in zip(out["N_evidence"], out["k_known_evidence"])
    ]
    return out, evU_size, evK_size


def compute_direction_consistency(
    df_sig: pl.DataFrame,
    all_units: list[str],
    unit_col: str,
    gene_col: str,
    effect_expr: pl.Expr
) -> pd.DataFrame:
    units_df = pl.DataFrame({unit_col: all_units})

    df_sig = df_sig.with_columns(
        pl.when(effect_expr > 0).then(1)
          .when(effect_expr < 0).then(-1)
          .otherwise(0)
          .alias("sign")
    )

    gene_cons = (
        df_sig.filter(pl.col("sign") != 0)
        .group_by([unit_col, gene_col])
        .agg([
            pl.col("sign").mean().abs().alias("gene_consistency"),
            pl.len().alias("n_obs"),
        ])
        .with_columns(pl.col("n_obs").cast(pl.Float64).log1p().alias("w"))
    )

    unit_cons = (
        gene_cons.group_by(unit_col)
        .agg([((pl.col("w") * pl.col("gene_consistency")).sum() / pl.col("w").sum()).alias("DirectionConsistency")])
    )

    return units_df.join(unit_cons, on=unit_col, how="left").with_columns(
        pl.col("DirectionConsistency").fill_null(0.0)
    ).to_pandas()


def compute_replication_gene_level(
    df_tv_sig: pl.DataFrame,
    df_test_sig: pl.DataFrame,
    df_test_universe: pl.DataFrame,
    all_units: list[str],
    unit_col: str,
    gene_col: str,
    n_perm: int,
    seed: int
) -> pd.DataFrame:
    units_df = pl.DataFrame({unit_col: all_units})

    U_test = set(df_test_universe.select(pl.col(gene_col).unique()).to_series().to_list())
    M_test = len(U_test)

    disc_sets = df_tv_sig.group_by(unit_col).agg([pl.col(gene_col).unique().alias("disc_genes")])
    test_sets = df_test_sig.group_by(unit_col).agg([pl.col(gene_col).unique().alias("test_genes")])

    merged = units_df.join(disc_sets, on=unit_col, how="left").join(test_sets, on=unit_col, how="left")
    out = merged.to_pandas()

    rng = np.random.default_rng(seed)
    rows = []
    for _, r in out.iterrows():
        unit = r[unit_col]
        disc = _as_set_of_str(r.get("disc_genes"))
        test = _as_set_of_str(r.get("test_genes"))

        n_disc = len(disc)
        n_test = len(test)
        n_overlap = len(disc & test)

        rep_rate_from_disc = n_overlap / max(n_disc, 1)
        rep_rate_in_test = n_overlap / max(n_test, 1)

        expected = n_disc * (n_test / M_test) if M_test > 0 else 0.0
        fe_rep = (n_overlap / expected) if expected > 0 else 0.0

        p_rep = empirical_p_hypergeom_geq(rng, M_test, n_test, n_disc, n_overlap, n_perm)

        rows.append({
            unit_col: unit,
            "N_disc_sig_genes": n_disc,
            "N_test_sig_genes": n_test,
            "N_overlap_sig_genes": n_overlap,
            "ReplicationRate_from_disc": rep_rate_from_disc,
            "ReplicationRate_in_test": rep_rate_in_test,
            "ExpectedOverlap_in_test": expected,
            "FE_replication": fe_rep,
            "p_emp_replication": p_rep,
            "TestUniverseGenes_M": M_test,
        })
    return pd.DataFrame(rows)


def compute_replication_evidence_level(
    df_tv_sig: pl.DataFrame,
    df_test_sig: pl.DataFrame,
    df_test_universe: pl.DataFrame,
    all_units: list[str],
    unit_col: str,
    evidence_cols: list[str],
    n_perm: int,
    seed: int
) -> pd.DataFrame:
    units_df = pl.DataFrame({unit_col: all_units})
    evU_test_size = df_test_universe.select(pl.struct(evidence_cols).alias("ev")).unique().height

    disc_sets = df_tv_sig.group_by(unit_col).agg([pl.struct(evidence_cols).unique().alias("disc_ev")])
    test_sets = df_test_sig.group_by(unit_col).agg([pl.struct(evidence_cols).unique().alias("test_ev")])

    merged = units_df.join(disc_sets, on=unit_col, how="left").join(test_sets, on=unit_col, how="left")
    out = merged.to_pandas()

    rng = np.random.default_rng(seed)
    rows = []
    for _, r in out.iterrows():
        unit = r[unit_col]
        disc = _as_set_of_evidence_tuples(r.get("disc_ev"), evidence_cols)
        test = _as_set_of_evidence_tuples(r.get("test_ev"), evidence_cols)

        n_disc = len(disc)
        n_test = len(test)
        n_overlap = len(disc & test)

        rep_rate_from_disc = n_overlap / max(n_disc, 1)
        rep_rate_in_test = n_overlap / max(n_test, 1)

        expected = n_disc * (n_test / evU_test_size) if evU_test_size > 0 else 0.0
        fe_rep = (n_overlap / expected) if expected > 0 else 0.0

        p_rep = empirical_p_hypergeom_geq(rng, evU_test_size, n_test, n_disc, n_overlap, n_perm)

        rows.append({
            unit_col: unit,
            "N_disc_sig_evidence": n_disc,
            "N_test_sig_evidence": n_test,
            "N_overlap_sig_evidence": n_overlap,
            "ReplicationRate_from_disc": rep_rate_from_disc,
            "ReplicationRate_in_test": rep_rate_in_test,
            "ExpectedOverlap_in_test": expected,
            "FE_replication": fe_rep,
            "p_emp_replication": p_rep,
            "TestUniverseEvidence_M": evU_test_size,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phenotype", help="Phenotype/disease name (folder prefix)")
    ap.add_argument("--n_perm", type=int, default=10000, help="Random trials for empirical p-values (default: 10000)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = ap.parse_args()

    phenotype = args.phenotype
    base_path = Path(phenotype)
    results_dir = base_path / "GeneDifferentialExpression" / "Files"
    volcano_file = results_dir / "combined_volcano_data_all_models.csv"
    if not volcano_file.exists():
        raise FileNotFoundError(f"Missing file: {volcano_file}")

    outdir = results_dir / "DatabaseImportance"
    outdir.mkdir(parents=True, exist_ok=True)

    known_set, known_src = load_known_genes(phenotype, base_path, results_dir)

    header_cols = pd.read_csv(volcano_file, nrows=0).columns.tolist()
    required = {"Dataset", "Gene", "Tissue", "Method", "Database", "FDR"}
    missing = sorted(list(required - set(header_cols)))
    if missing:
        raise ValueError(f"Input file missing required columns: {missing}")

    has_effect = "Effect" in header_cols
    has_l2fc = "Log2FoldChange" in header_cols
    if not has_effect and not has_l2fc:
        raise ValueError("Need Effect and/or Log2FoldChange columns for effect thresholding.")

    use_cols = ["Dataset", "Gene", "Tissue", "Method", "Database", "FDR"]
    if has_effect:
        use_cols.append("Effect")
    if has_l2fc:
        use_cols.append("Log2FoldChange")

    df = pl.read_csv(volcano_file, columns=use_cols)

    df = df.with_columns(
        pl.col("Gene").cast(pl.Utf8).str.to_lowercase().str.split(".").list.first().alias("Gene_norm")
    )

    dataset_values = df.select(pl.col("Dataset").unique()).to_series().to_list()
    train_label, val_label, test_label = detect_train_val_test_labels(dataset_values)

    df_tv = df.filter(pl.col("Dataset").is_in([train_label, val_label]))
    df_test = df.filter(pl.col("Dataset") == test_label)

    all_dbs = sorted(df.select(pl.col("Database").unique()).to_series().to_list())

    # effect := coalesce(Effect, Log2FoldChange)
    effect_expr = pl.coalesce([
        pl.col("Effect").cast(pl.Float64, strict=False) if has_effect else pl.lit(None, dtype=pl.Float64),
        pl.col("Log2FoldChange").cast(pl.Float64, strict=False) if has_l2fc else pl.lit(None, dtype=pl.Float64),
    ])

    U_tv = set(df_tv.select(pl.col("Gene_norm").unique()).to_series().to_list())
    K_tv = set(known_set) & U_tv
    M_tv = len(U_tv)
    K_tv_size = len(K_tv)

    U_test = set(df_test.select(pl.col("Gene_norm").unique()).to_series().to_list())
    K_test = set(known_set) & U_test
    M_test = len(U_test)
    K_test_size = len(K_test)

    if M_tv == 0:
        raise ValueError("Empty discovery universe after restricting to train+val.")
    if M_test == 0:
        raise ValueError("Empty test universe after restricting to test.")

    df_tv_sig = df_tv.filter(
        (pl.col("FDR").cast(pl.Float64, strict=False) < FDR_THR) & (effect_expr.abs() >= EFFECT_THR)
    )
    df_test_sig = df_test.filter(
        (pl.col("FDR").cast(pl.Float64, strict=False) < FDR_THR) & (effect_expr.abs() >= EFFECT_THR)
    )

    # A) Discovery ranking
    gl_disc = compute_gene_level_enrichment(df_tv_sig, all_dbs, "Database", "Gene_norm", K_tv, M_tv, args.n_perm, args.seed)
    el_disc, evU_tv_size, evK_tv_size = compute_evidence_level_enrichment(
        df_tv_sig, df_tv, all_dbs, "Database", ["Gene_norm", "Tissue", "Method"], "Gene_norm", K_tv, args.n_perm, args.seed + 7
    )
    extra_counts = (
        df_tv_sig.group_by("Database").agg([
            pl.col("Tissue").n_unique().alias("N_tissues"),
            pl.col("Method").n_unique().alias("N_methods"),
        ])
    )
    extra_counts = pl.DataFrame({"Database": all_dbs}).join(extra_counts, on="Database", how="left").fill_null(0).to_pandas()

    dc_disc = compute_direction_consistency(df_tv_sig, all_dbs, "Database", "Gene_norm", effect_expr)

    disc = gl_disc.merge(el_disc, on="Database", how="left").merge(dc_disc, on="Database", how="left").merge(extra_counts, on="Database", how="left")
    disc["neglog10_p_gene"] = disc["p_emp_genes"].apply(safe_neglog10)
    disc["neglog10_p_evd"] = disc["p_emp_evidence"].apply(safe_neglog10)
    disc["CombinedScore"] = (
        np.log1p(disc["FE_genes"].astype(float)) * disc["neglog10_p_gene"]
        + np.log1p(disc["FE_evidence"].astype(float)) * disc["neglog10_p_evd"]
        + 5.0 * disc["DirectionConsistency"].astype(float)
        + 0.25 * np.log1p(disc["N_evidence"].astype(float))
    )
    disc = disc.sort_values("CombinedScore", ascending=False).reset_index(drop=True)
    disc["Rank"] = np.arange(1, len(disc) + 1)

    # B) Disease enrichment (discovery + test)
    disease_disc = disc[["Rank", "Database", "N_genes", "k_known_genes", "expected_known_genes", "FE_genes", "p_emp_genes"]].copy()

    gl_test = compute_gene_level_enrichment(df_test_sig, all_dbs, "Database", "Gene_norm", K_test, M_test, args.n_perm, args.seed + 101)
    disease_test = gl_test.sort_values(["FE_genes", "p_emp_genes"], ascending=[False, True]).reset_index(drop=True)
    disease_test["Rank"] = np.arange(1, len(disease_test) + 1)
    disease_test = disease_test[["Rank", "Database", "N_genes", "k_known_genes", "expected_known_genes", "FE_genes", "p_emp_genes"]]

    # C) Test replication
    rep_gene = compute_replication_gene_level(df_tv_sig, df_test_sig, df_test, all_dbs, "Database", "Gene_norm", args.n_perm, args.seed + 202)
    rep_gene = rep_gene.sort_values(["FE_replication", "p_emp_replication"], ascending=[False, True]).reset_index(drop=True)

    rep_evd = compute_replication_evidence_level(
        df_tv_sig, df_test_sig, df_test, all_dbs, "Database", ["Gene_norm", "Tissue", "Method"], args.n_perm, args.seed + 303
    )
    rep_evd = rep_evd.sort_values(["FE_replication", "p_emp_replication"], ascending=[False, True]).reset_index(drop=True)

    # Save outputs
    disc_gene_out = disc[["Rank", "Database", "N_genes", "k_known_genes", "expected_known_genes", "FE_genes", "p_emp_genes"]].copy()
    disc_evd_out = disc[["Rank", "Database", "N_evidence", "k_known_evidence", "expected_known_evidence", "FE_evidence", "p_emp_evidence", "N_tissues", "N_methods"]].copy()

    disc.to_csv(outdir / "database_importance_discovery_combined.csv", index=False)
    disc_gene_out.to_csv(outdir / "database_importance_discovery_gene_level.csv", index=False)
    disc_evd_out.to_csv(outdir / "database_importance_discovery_evidence_level.csv", index=False)
    write_top10_latex_discovery(disc, outdir / "database_importance_discovery_top10_table.tex")

    disease_disc.to_csv(outdir / "database_disease_enrichment_discovery.csv", index=False)
    disease_test.to_csv(outdir / "database_disease_enrichment_test.csv", index=False)

    rep_gene.to_csv(outdir / "database_test_replication_gene_level.csv", index=False)
    rep_evd.to_csv(outdir / "database_test_replication_evidence_level.csv", index=False)

    report_path = outdir / "database_importance_discovery_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("DATABASE IMPORTANCE — DISCOVERY + TEST REPLICATION + DISEASE ENRICHMENT\n")
        f.write("=" * 120 + "\n")
        f.write(f"Phenotype: {phenotype}\n")
        f.write(f"Input: {volcano_file}\n")
        f.write(f"Known gene source: {known_src}\n\n")
        f.write("Thresholds (HARD-CODED):\n")
        f.write(f"  FDR < {FDR_THR}\n")
        f.write(f"  |effect| >= {EFFECT_THR}   (effect := coalesce(Effect, Log2FoldChange))\n\n")
        f.write("Dataset labels:\n")
        f.write(f"  Train: {train_label}\n  Validation: {val_label}\n  Test: {test_label}\n\n")
        f.write("Universes:\n")
        f.write(f"  Discovery genes: M_tv={M_tv}, Known in-universe: K_tv={K_tv_size}\n")
        f.write(f"  Test genes:      M_test={M_test}, Known in-universe: K_test={K_test_size}\n")
        f.write(f"  Discovery evidence universe: evU_tv={evU_tv_size}, Known evidence: evK_tv={evK_tv_size}\n\n")
        f.write("Discovery combined ranking:\n\n")
        show_cols = [
            "Rank", "Database",
            "N_genes", "k_known_genes", "expected_known_genes", "FE_genes", "p_emp_genes",
            "DirectionConsistency",
            "N_evidence", "k_known_evidence", "expected_known_evidence", "FE_evidence", "p_emp_evidence",
            "N_tissues", "N_methods",
            "CombinedScore",
        ]
        f.write(disc[show_cols].to_string(index=False))
        f.write("\n")

    print("\n✅ Completed DATABASE importance with discovery ranking + test replication + disease enrichment.")
    print(f"📁 Output directory: {outdir}\n")

    print("Discovery ranking (top 10):")
    print(disc[["Rank", "Database", "N_genes", "k_known_genes", "FE_genes", "p_emp_genes", "DirectionConsistency"]].head(10).to_string(index=False))

    print("\nDisease enrichment (test) top 10:")
    print(disease_test[["Rank", "Database", "N_genes", "k_known_genes", "FE_genes", "p_emp_genes"]].head(10).to_string(index=False))

    print("\nTest replication (gene-level) top 10:")
    print(rep_gene[["Database", "N_disc_sig_genes", "N_test_sig_genes", "N_overlap_sig_genes", "FE_replication", "p_emp_replication"]].head(10).to_string(index=False))

    print("\nFiles written:")
    print(" - database_importance_discovery_gene_level.csv")
    print(" - database_importance_discovery_evidence_level.csv")
    print(" - database_importance_discovery_combined.csv")
    print(" - database_importance_discovery_top10_table.tex")
    print(" - database_importance_discovery_report.txt")
    print(" - database_disease_enrichment_discovery.csv")
    print(" - database_disease_enrichment_test.csv")
    print(" - database_test_replication_gene_level.csv")
    print(" - database_test_replication_evidence_level.csv")


if __name__ == "__main__":
    main()
