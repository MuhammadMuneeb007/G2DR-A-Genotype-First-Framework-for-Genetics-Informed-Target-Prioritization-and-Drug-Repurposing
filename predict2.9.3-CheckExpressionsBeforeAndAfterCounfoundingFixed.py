#!/usr/bin/env python3
"""
BEFORE vs AFTER confounding adjustment: correlations + expression change
======================================================================

What this does (exactly):

A) Correlations (your simple logic, equal weights everywhere, NO Fisher-z):
   For each split (train/validation/test):
     For each fold:
       For each tissue:
         For each model pair:
           - Find common genes
           - For each gene: Pearson r across matched samples
           - Tissue correlation = mean(gene r)
       Fold correlation  = mean(tissue correlations)
     Split correlation = mean(fold correlations)

We compute this twice:
  1) RAW files  (non "_fixed")
  2) FIXED files (with "_fixed")
Then we report:
  - raw_r, fixed_r, delta = fixed_r - raw_r
  - overall summary + per split + per model-pair

B) Expression change (how much expression values changed):
   For each model, split, fold, tissue:
     - Load RAW and FIXED matrices
     - Match samples and common genes
     - Compute:
         mean_abs_delta  = mean(|fixed - raw|)
         rmse_delta      = sqrt(mean((fixed - raw)^2))
         mean_delta      = mean(fixed - raw)
         corr_raw_fixed  = correlation of flattened values (raw vs fixed)
   Aggregate these metrics per model and also overall.

Outputs in <phenotype>/:
  - before_after_pairwise.csv
  - expression_change_by_model.csv
  - expression_change_overall.csv
  - heatmap_delta_<split>.png  (fixed - raw correlation deltas)

Usage:
  python3 before_after_confounding_report.py <phenotype>

Notes:
- For FUSION, filenames are fixed as in your folders.
- For other models, we pick ONE csv per tissue folder:
    RAW   = first .csv that does NOT contain "_fixed"
    FIXED = first .csv that DOES contain "_fixed"
"""

import os
import sys
from itertools import combinations
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

if len(sys.argv) != 2:
    print("Usage: python3 before_after_confounding_report.py <phenotype>")
    sys.exit(1)

phenotype = sys.argv[1]

MODEL_DIRS = {
    "Regular": {"train": "TrainExpression", "test": "TestExpression", "validation": "ValidationExpression"},
    "JTI": {"train": "JTITrainExpression", "test": "JTITestExpression", "validation": "JTIValidationExpression"},
    "UTMOST": {"train": "UTMOSTTrainExpression", "test": "UTMOSTTestExpression", "validation": "UTMOSTValidationExpression"},
    "UTMOST2": {"train": "utmost2TrainExpression", "test": "utmost2TestExpression", "validation": "utmost2ValidationExpression"},
    "EpiX": {"train": "EpiXTrainExpression", "test": "EpiXTestExpression", "validation": "EpiXValidationExpression"},
    "TIGAR": {"train": "TigarTrainExpression", "test": "TigarTestExpression", "validation": "TigarValidExpression"},
    "FUSION": {"train": "FussionExpression", "test": "FussionExpression", "validation": "FussionExpression"},
}

FUSION_RAW = {
    "train": "GeneExpression_train_data.csv",
    "test": "GeneExpression_test_data.csv",
    "validation": "GeneExpression_validation_data.csv",
}
FUSION_FIXED = {
    "train": "GeneExpression_train_data_fixed.csv",
    "test": "GeneExpression_test_data_fixed.csv",
    "validation": "GeneExpression_validation_data_fixed.csv",
}


# -------------------------
# helpers: files, loading
# -------------------------
def clean_gene(g: object) -> str:
    g = str(g).strip()
    if g.startswith("chr") and "_" in g:
        g = g.split("_", 1)[1]
    return g.split(".")[0]


def load_data(path: str):
    """Return (gene_df, sample_ids, gene_set) or (None, None, None)."""
    try:
        df = pl.read_csv(path)
        gene_cols = [c for c in df.columns if c not in ["FID", "IID"]]
        if not gene_cols:
            return None, None, None
        sample_ids = (df["FID"].cast(pl.Utf8) + "_" + df["IID"].cast(pl.Utf8)).to_list()
        data = df.select(gene_cols)
        data.columns = [clean_gene(c) for c in gene_cols]
        return data, sample_ids, set(data.columns)
    except Exception:
        return None, None, None


def get_folds(phenotype_dir: str):
    if not os.path.exists(phenotype_dir):
        return []
    folds = []
    for d in os.listdir(phenotype_dir):
        if d.startswith("Fold_") and os.path.isdir(os.path.join(phenotype_dir, d)):
            try:
                folds.append(int(d.split("_")[1]))
            except Exception:
                pass
    return sorted(folds)


def get_tissues(phenotype_dir: str, fold: int):
    tissues = set()
    fold_dir = os.path.join(phenotype_dir, f"Fold_{fold}")
    if not os.path.exists(fold_dir):
        return []
    for _, dirs in MODEL_DIRS.items():
        p = os.path.join(fold_dir, dirs["train"])
        if os.path.exists(p):
            for t in os.listdir(p):
                if os.path.isdir(os.path.join(p, t)):
                    tissues.add(t)
    return sorted(tissues)


def find_raw_and_fixed_files(tissue_dir: str, model: str, split: str):
    """
    Returns (raw_path, fixed_path) possibly None.
    For non-FUSION:
      raw  = first *.csv without "_fixed"
      fixed= first *.csv with "_fixed"
    """
    if not os.path.exists(tissue_dir):
        return None, None

    if model == "FUSION":
        raw_p = os.path.join(tissue_dir, FUSION_RAW[split])
        fix_p = os.path.join(tissue_dir, FUSION_FIXED[split])
        return (raw_p if os.path.exists(raw_p) else None), (fix_p if os.path.exists(fix_p) else None)

    csvs = [f for f in os.listdir(tissue_dir) if f.lower().endswith(".csv")]
    raw_candidates = sorted([f for f in csvs if "_fixed" not in f.lower()], key=len)
    fix_candidates = sorted([f for f in csvs if "_fixed" in f.lower()], key=len)

    raw_path = os.path.join(tissue_dir, raw_candidates[0]) if raw_candidates else None
    fix_path = os.path.join(tissue_dir, fix_candidates[0]) if fix_candidates else None

    if raw_path and not os.path.exists(raw_path):
        raw_path = None
    if fix_path and not os.path.exists(fix_path):
        fix_path = None
    return raw_path, fix_path


def load_tissue_version(phenotype_dir: str, fold: int, model: str, split: str, tissue: str, version: str):
    """
    version: "raw" or "fixed"
    """
    tissue_dir = os.path.join(phenotype_dir, f"Fold_{fold}", MODEL_DIRS[model][split], tissue)
    raw_p, fix_p = find_raw_and_fixed_files(tissue_dir, model, split)
    path = raw_p if version == "raw" else fix_p
    if not path:
        return None, None, None
    return load_data(path)


# -------------------------
# math: correlation pieces
# -------------------------
def safe_corr(v1: np.ndarray, v2: np.ndarray):
    valid = ~(np.isnan(v1) | np.isnan(v2))
    if valid.sum() < 2:
        return np.nan
    a = v1[valid]
    b = v2[valid]
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    r = np.corrcoef(a, b)[0, 1]
    if np.isnan(r) or np.isinf(r):
        return np.nan
    return float(r)


def tissue_corr_mean_gene_r(d1: pl.DataFrame, d2: pl.DataFrame, genes, s1, s2):
    """
    One tissue number:
      gene r across samples, then tissue r = mean(gene r)
    """
    common_samples = sorted(set(s1) & set(s2))
    if len(common_samples) < 2:
        return np.nan, 0, 0

    pos1 = {sid: i for i, sid in enumerate(s1)}
    pos2 = {sid: i for i, sid in enumerate(s2)}
    i1 = [pos1[sid] for sid in common_samples]
    i2 = [pos2[sid] for sid in common_samples]

    sub1 = d1.select(list(genes))[i1, :]
    sub2 = d2.select(list(genes))[i2, :]

    rs = []
    usable = 0
    for g in genes:
        r = safe_corr(sub1[g].to_numpy(), sub2[g].to_numpy())
        if not np.isnan(r):
            rs.append(r)
            usable += 1

    if not rs:
        return np.nan, len(common_samples), 0
    return float(np.mean(rs)), len(common_samples), usable


def compute_correlations_equalweights(phenotype_dir: str, folds, split: str, version: str):
    """
    version: "raw" or "fixed"
    returns:
      results[pair] = split-level r
      details: list of per (fold,tissue,pair) records to support debug
    """
    models = list(MODEL_DIRS.keys())
    pairs = list(combinations(models, 2))

    fold_rs = {f"{a}_vs_{b}": [] for a, b in pairs}
    details = []

    for fold in folds:
        tissues = get_tissues(phenotype_dir, fold)
        if not tissues:
            continue

        tissue_rs = {f"{a}_vs_{b}": [] for a, b in pairs}

        for tissue in tissues:
            # load per model for this tissue/version
            tdata = {}
            for m in models:
                d, s, g = load_tissue_version(phenotype_dir, fold, m, split, tissue, version)
                if d is not None:
                    tdata[m] = (d, s, g)

            for m1, m2 in pairs:
                if m1 not in tdata or m2 not in tdata:
                    continue
                d1, s1, g1 = tdata[m1]
                d2, s2, g2 = tdata[m2]

                common_genes = (g1 & g2)
                if not common_genes:
                    continue

                r_tissue, n_samp, n_usable = tissue_corr_mean_gene_r(d1, d2, common_genes, s1, s2)
                if np.isnan(r_tissue):
                    continue

                pk = f"{m1}_vs_{m2}"
                tissue_rs[pk].append(r_tissue)

                details.append(
                    {
                        "version": version,
                        "split": split,
                        "fold": fold,
                        "tissue": tissue,
                        "pair": pk,
                        "tissue_r": r_tissue,
                        "n_common_samples": n_samp,
                        "n_common_genes": len(common_genes),
                        "n_usable_genes": n_usable,
                    }
                )

        # fold r = mean(tissue r)
        for pk, rs in tissue_rs.items():
            if rs:
                fold_rs[pk].append(float(np.mean(rs)))

    # split r = mean(fold r)
    results = {}
    for pk, rs in fold_rs.items():
        if rs:
            results[pk] = float(np.mean(rs))
    return results, details, models


# -------------------------
# expression change metrics
# -------------------------
def expression_change_metrics(raw_d: pl.DataFrame, fix_d: pl.DataFrame, raw_s, fix_s, raw_genes, fix_genes):
    """
    Compare raw vs fixed for one (model, split, fold, tissue).
    Returns dict metrics or None.
    """
    common_samples = sorted(set(raw_s) & set(fix_s))
    if len(common_samples) < 2:
        return None

    common_genes = sorted(list(raw_genes & fix_genes))
    if not common_genes:
        return None

    pos_raw = {sid: i for i, sid in enumerate(raw_s)}
    pos_fix = {sid: i for i, sid in enumerate(fix_s)}
    i_raw = [pos_raw[sid] for sid in common_samples]
    i_fix = [pos_fix[sid] for sid in common_samples]

    A = raw_d.select(common_genes)[i_raw, :].to_numpy()
    B = fix_d.select(common_genes)[i_fix, :].to_numpy()

    # align NaNs
    mask = ~(np.isnan(A) | np.isnan(B))
    if mask.sum() < 2:
        return None

    delta = (B - A)

    mean_abs = float(np.nanmean(np.abs(delta)))
    rmse = float(np.sqrt(np.nanmean(delta ** 2)))
    mean_delta = float(np.nanmean(delta))

    # correlation of flattened values (raw vs fixed)
    a_flat = A[mask].ravel()
    b_flat = B[mask].ravel()
    if a_flat.size < 2 or np.std(a_flat) == 0 or np.std(b_flat) == 0:
        corr_raw_fixed = np.nan
    else:
        corr_raw_fixed = float(np.corrcoef(a_flat, b_flat)[0, 1])

    return {
        "n_samples": len(common_samples),
        "n_genes": len(common_genes),
        "mean_abs_delta": mean_abs,
        "rmse_delta": rmse,
        "mean_delta": mean_delta,
        "corr_raw_fixed": corr_raw_fixed,
    }


# -------------------------
# reporting + outputs
# -------------------------
def save_pairwise_csv(out_dir: str, rows):
    os.makedirs(out_dir, exist_ok=True)
    pl.DataFrame(rows).write_csv(os.path.join(out_dir, "before_after_pairwise.csv"))
    print(f"? {out_dir}/before_after_pairwise.csv")


def save_expr_change_csv(out_dir: str, by_model_rows, overall_rows):
    os.makedirs(out_dir, exist_ok=True)
    pl.DataFrame(by_model_rows).write_csv(os.path.join(out_dir, "expression_change_by_model.csv"))
    pl.DataFrame(overall_rows).write_csv(os.path.join(out_dir, "expression_change_overall.csv"))
    print(f"? {out_dir}/expression_change_by_model.csv")
    print(f"? {out_dir}/expression_change_overall.csv")


def plot_delta_heatmap(out_dir: str, models, split: str, pair_delta_map):
    """
    pair_delta_map: dict pair_key -> delta (fixed - raw)
    """
    os.makedirs(out_dir, exist_ok=True)
    n = len(models)
    mat = np.zeros((n, n), dtype=float)
    mat[:] = 0.0
    np.fill_diagonal(mat, 0.0)
    idx = {m: i for i, m in enumerate(models)}

    for pk, d in pair_delta_map.items():
        a, b = pk.split("_vs_")
        i, j = idx[a], idx[b]
        mat[i, j] = mat[j, i] = d

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(mat, dtype=bool), k=1)
    sns.heatmap(
        mat,
        mask=mask,
        annot=True,
        fmt=".3f",
        xticklabels=models,
        yticklabels=models,
        center=0,
        cmap="RdBu_r",
        square=True,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(f"Correlation delta (fixed - raw) | {split} | {out_dir}", fontsize=14, weight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, f"heatmap_delta_{split}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"? {path}")


def summarize_print(pair_rows, expr_overall_rows, expr_model_rows):
    # quick high-signal console summary
    print("\n" + "=" * 95)
    print("SUMMARY: CORRELATION CHANGE (fixed - raw)")
    print("=" * 95)

    if pair_rows:
        df = pl.DataFrame(pair_rows)
        # overall across all splits/pairs (simple mean of deltas)
        overall = df.select(pl.col("delta").mean()).item()
        print(f"Overall mean delta across all splits+pairs: {overall:+.4f}")

        for sp in ["train", "validation", "test"]:
            sub = df.filter(pl.col("split") == sp)
            if sub.height > 0:
                m = sub.select(pl.col("delta").mean()).item()
                print(f"  {sp:10s}: mean delta = {m:+.4f}")

        # top 5 increases / decreases
        print("\nTop +delta pairs (biggest increases):")
        top_up = df.sort("delta", descending=True).head(5)
        for r in top_up.iter_rows(named=True):
            print(f"  {r['split']:10s} {r['pair']:30s}  raw={r['raw_r']:+.3f}  fixed={r['fixed_r']:+.3f}  ?={r['delta']:+.3f}")

        print("\nTop -delta pairs (biggest decreases):")
        top_dn = df.sort("delta", descending=False).head(5)
        for r in top_dn.iter_rows(named=True):
            print(f"  {r['split']:10s} {r['pair']:30s}  raw={r['raw_r']:+.3f}  fixed={r['fixed_r']:+.3f}  ?={r['delta']:+.3f}")
    else:
        print("No pairwise results computed (missing files or no overlaps).")

    print("\n" + "=" * 95)
    print("SUMMARY: EXPRESSION VALUE CHANGE (raw -> fixed)")
    print("=" * 95)

    if expr_overall_rows:
        r = expr_overall_rows[0]
        print(
            f"Overall: mean_abs_delta={r['mean_abs_delta']:.6g} | rmse={r['rmse_delta']:.6g} | "
            f"mean_delta={r['mean_delta']:.6g} | corr(raw,fixed)={r['corr_raw_fixed']:.4f} | "
            f"avg_samples={r['avg_n_samples']:.1f} | avg_genes={r['avg_n_genes']:.1f} | blocks={r['n_blocks']}"
        )

    if expr_model_rows:
        print("\nPer model (averaged over split/fold/tissue blocks):")
        # print compact table-like
        for r in expr_model_rows:
            print(
                f"  {r['model']:8s}  mean_abs={r['mean_abs_delta']:.6g}  rmse={r['rmse_delta']:.6g}  "
                f"mean?={r['mean_delta']:.6g}  corr={r['corr_raw_fixed']:.4f}  blocks={r['n_blocks']}"
            )


# -------------------------
# main
# -------------------------
print("\n" + "=" * 80)
print("BEFORE vs AFTER CONFOUNDING ADJUSTMENT REPORT")
print("=" * 80)
print("Correlation logic: tissue(mean gene r) -> fold(mean tissue r) -> split(mean fold r)")
print("Also computes expression-value deltas raw vs fixed (overall and per model).")
print("=" * 80)

folds = get_folds(phenotype)
if not folds:
    print(f"? No folds found for '{phenotype}'")
    sys.exit(1)

splits = ["train", "validation", "test"]
models = list(MODEL_DIRS.keys())

# --- A) correlations before/after ---
pair_rows = []
all_models_for_plot = None

for split in splits:
    raw_res, raw_details, models_used = compute_correlations_equalweights(phenotype, folds, split, "raw")
    fix_res, fix_details, _ = compute_correlations_equalweights(phenotype, folds, split, "fixed")
    all_models_for_plot = models_used

    # unify keys
    keys = sorted(set(raw_res.keys()) | set(fix_res.keys()))
    pair_delta_map = {}

    for pk in keys:
        raw_r = raw_res.get(pk, np.nan)
        fix_r = fix_res.get(pk, np.nan)
        if np.isnan(raw_r) or np.isnan(fix_r):
            continue
        delta = fix_r - raw_r
        pair_delta_map[pk] = delta
        pair_rows.append(
            {
                "phenotype": phenotype,
                "split": split,
                "pair": pk,
                "raw_r": float(raw_r),
                "fixed_r": float(fix_r),
                "delta": float(delta),
            }
        )

    # delta heatmap per split
    if pair_delta_map:
        plot_delta_heatmap(phenotype, all_models_for_plot, split, pair_delta_map)

# save pairwise table
save_pairwise_csv(phenotype, pair_rows)

# --- B) expression change raw vs fixed ---
expr_block_rows = []  # per (model, split, fold, tissue)
for split in splits:
    for fold in folds:
        tissues = get_tissues(phenotype, fold)
        if not tissues:
            continue
        for tissue in tissues:
            for model in models:
                raw_d, raw_s, raw_g = load_tissue_version(phenotype, fold, model, split, tissue, "raw")
                fix_d, fix_s, fix_g = load_tissue_version(phenotype, fold, model, split, tissue, "fixed")
                if raw_d is None or fix_d is None:
                    continue

                m = expression_change_metrics(raw_d, fix_d, raw_s, fix_s, raw_g, fix_g)
                if m is None:
                    continue

                expr_block_rows.append(
                    {
                        "phenotype": phenotype,
                        "model": model,
                        "split": split,
                        "fold": fold,
                        "tissue": tissue,
                        **m,
                    }
                )

# aggregate per model
expr_model_rows = []
if expr_block_rows:
    df_expr = pl.DataFrame(expr_block_rows)

    for model in models:
        sub = df_expr.filter(pl.col("model") == model)
        if sub.height == 0:
            continue
        expr_model_rows.append(
            {
                "phenotype": phenotype,
                "model": model,
                "mean_abs_delta": sub.select(pl.col("mean_abs_delta").mean()).item(),
                "rmse_delta": sub.select(pl.col("rmse_delta").mean()).item(),
                "mean_delta": sub.select(pl.col("mean_delta").mean()).item(),
                "corr_raw_fixed": sub.select(pl.col("corr_raw_fixed").mean()).item(),
                "avg_n_samples": sub.select(pl.col("n_samples").mean()).item(),
                "avg_n_genes": sub.select(pl.col("n_genes").mean()).item(),
                "n_blocks": sub.height,
            }
        )

    # overall
    expr_overall_rows = [
        {
            "phenotype": phenotype,
            "mean_abs_delta": df_expr.select(pl.col("mean_abs_delta").mean()).item(),
            "rmse_delta": df_expr.select(pl.col("rmse_delta").mean()).item(),
            "mean_delta": df_expr.select(pl.col("mean_delta").mean()).item(),
            "corr_raw_fixed": df_expr.select(pl.col("corr_raw_fixed").mean()).item(),
            "avg_n_samples": df_expr.select(pl.col("n_samples").mean()).item(),
            "avg_n_genes": df_expr.select(pl.col("n_genes").mean()).item(),
            "n_blocks": df_expr.height,
        }
    ]
else:
    expr_model_rows = []
    expr_overall_rows = [
        {
            "phenotype": phenotype,
            "mean_abs_delta": np.nan,
            "rmse_delta": np.nan,
            "mean_delta": np.nan,
            "corr_raw_fixed": np.nan,
            "avg_n_samples": np.nan,
            "avg_n_genes": np.nan,
            "n_blocks": 0,
        }
    ]

save_expr_change_csv(phenotype, expr_model_rows, expr_overall_rows)

# print to console
summarize_print(pair_rows, expr_overall_rows, expr_model_rows)

print("\n? DONE")
print(f"Outputs saved under: {phenotype}/")
print("  - before_after_pairwise.csv")
print("  - expression_change_by_model.csv")
print("  - expression_change_overall.csv")
print("  - heatmap_delta_<split>.png")
