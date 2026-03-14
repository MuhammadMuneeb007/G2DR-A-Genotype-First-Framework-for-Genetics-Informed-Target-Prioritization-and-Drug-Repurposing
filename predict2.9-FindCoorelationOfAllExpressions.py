#!/usr/bin/env python3
"""
Simple correlation summary (EQUAL WEIGHTS IN r-SPACE) - NON-FIXED FILES
=====================================================================

Your requested logic:

For each split (train/validation/test):
  For each fold:
    For each tissue:
      - Find common genes between two models in that tissue
      - Compute ONE tissue correlation = mean over genes of per-gene Pearson r
        (gene-wise r computed across matched samples)
    - Fold correlation = mean of tissue correlations (equal weight per tissue)
  - Split correlation = mean of fold correlations (equal weight per fold)

IMPORTANT:
- Uses ONLY NON-fixed expression files (no "_fixed" filtering).
- FUSION uses original names: GeneExpression_<split>_data.csv
- No Fisher z (atanh/tanh) anywhere.

Outputs (saved under <phenotype>/):
  - correlation_<split>_raw.png
  - correlation_results_raw.csv
  - correlation_table_raw.tex
"""

import os
import sys
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

if len(sys.argv) != 2:
    print("Usage: python3 correlation_raw.py <phenotype>")
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

# FUSION non-fixed naming
FILE_NAMES_FUSION = {
    "train": "GeneExpression_train_data.csv",
    "test": "GeneExpression_test_data.csv",
    "validation": "GeneExpression_validation_data.csv",
}

# For non-FUSION models:
# - We select a CSV that does NOT contain "_fixed" (case-insensitive).
AVOID_FIXED = True


def clean_gene(g: object) -> str:
    g = str(g).strip()
    if g.startswith("chr") and "_" in g:
        g = g.split("_", 1)[1]
    return g.split(".")[0]


def load_data(path: str):
    """Return (data_frame_of_genes, sample_id_list) or (None, None)."""
    try:
        df = pl.read_csv(path)
        gene_cols = [c for c in df.columns if c not in ["FID", "IID"]]
        if not gene_cols:
            return None, None
        samples = (df["FID"].cast(pl.Utf8) + "_" + df["IID"].cast(pl.Utf8)).to_list()
        data = df.select(gene_cols)
        data.columns = [clean_gene(c) for c in gene_cols]
        return data, samples
    except Exception:
        return None, None


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
    """Union tissues seen under TrainExpression dirs (same as your earlier logic)."""
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


def find_file(tissue_dir: str, model: str, split: str):
    if not os.path.exists(tissue_dir):
        return None

    if model == "FUSION":
        p = os.path.join(tissue_dir, FILE_NAMES_FUSION[split])
        return p if os.path.exists(p) else None

    csvs = [f for f in os.listdir(tissue_dir) if f.lower().endswith(".csv")]
    if AVOID_FIXED:
        csvs = [f for f in csvs if "_fixed" not in f.lower()]
    # pick something stable: shortest name first
    if not csvs:
        return None
    csvs = sorted(csvs, key=len)
    return os.path.join(tissue_dir, csvs[0])


def load_tissue(phenotype_dir: str, fold: int, model: str, split: str, tissue: str):
    base = os.path.join(phenotype_dir, f"Fold_{fold}", MODEL_DIRS[model][split], tissue)
    path = find_file(base, model, split)
    if not path:
        return None, None, None
    data, samples = load_data(path)
    if data is None:
        return None, None, None
    return data, samples, set(data.columns)


def safe_corr(v1: np.ndarray, v2: np.ndarray):
    """Pearson r with NaN handling; returns np.nan if not computable."""
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


def tissue_correlation_mean_r(d1: pl.DataFrame, d2: pl.DataFrame, genes, s1, s2):
    """
    ONE number per tissue:
      - match common samples
      - for each gene compute Pearson r across samples
      - tissue r = mean(gene r)  [equal weight per gene]
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


def analyze_split_equalweights(phenotype_dir: str, folds, split: str):
    """
    Implements:
      tissue r = mean gene r
      fold r   = mean tissue r
      split r  = mean fold r
    """
    models = list(MODEL_DIRS.keys())
    pairs = list(combinations(models, 2))

    fold_rs = {f"{a}_vs_{b}": [] for a, b in pairs}
    fold_counts = {f"{a}_vs_{b}": [] for a, b in pairs}  # list of (n_tissues_used, sum_usable_genes)

    for fold in folds:
        tissues = get_tissues(phenotype_dir, fold)
        if not tissues:
            continue

        tissue_rs = {f"{a}_vs_{b}": [] for a, b in pairs}
        tissue_gene_usable = {f"{a}_vs_{b}": 0 for a, b in pairs}

        for tissue in tissues:
            tdata = {}
            for m in models:
                d, s, g = load_tissue(phenotype_dir, fold, m, split, tissue)
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

                r_tissue, _, usable_genes = tissue_correlation_mean_r(d1, d2, common_genes, s1, s2)
                if not np.isnan(r_tissue):
                    pk = f"{m1}_vs_{m2}"
                    tissue_rs[pk].append(r_tissue)
                    tissue_gene_usable[pk] += usable_genes

        for pk, rs in tissue_rs.items():
            if rs:
                fold_rs[pk].append(float(np.mean(rs)))
                fold_counts[pk].append((len(rs), tissue_gene_usable[pk]))

    results = {}
    for pk, rs in fold_rs.items():
        if not rs:
            continue

        rs_arr = np.array(rs, dtype=float)
        mean_r = float(np.mean(rs_arr))

        if len(rs_arr) > 1:
            sd = float(np.std(rs_arr, ddof=1))
            se = sd / np.sqrt(len(rs_arr))
            ci_l = mean_r - 1.96 * se
            ci_u = mean_r + 1.96 * se
        else:
            sd = 0.0
            se = 0.0
            ci_l = mean_r
            ci_u = mean_r

        ci_l = float(np.clip(ci_l, -1.0, 1.0))
        ci_u = float(np.clip(ci_u, -1.0, 1.0))

        counts = fold_counts[pk]
        tissues_per_fold = [c[0] for c in counts] if counts else []
        usable_genes_per_fold = [c[1] for c in counts] if counts else []

        results[pk] = {
            "r": mean_r,
            "ci_l": ci_l,
            "ci_u": ci_u,
            "n_folds": len(rs_arr),
            "fold_sd": sd,
            "avg_tissues_per_fold": float(np.mean(tissues_per_fold)) if tissues_per_fold else 0.0,
            "avg_usable_genes_per_fold": float(np.mean(usable_genes_per_fold)) if usable_genes_per_fold else 0.0,
        }

    return results, models


def plot_matrix(results, models, split, phenotype_dir):
    n = len(models)
    mat = np.eye(n)
    idx = {m: i for i, m in enumerate(models)}

    for pk, s in results.items():
        a, b = pk.split("_vs_")
        i, j = idx[a], idx[b]
        mat[i, j] = mat[j, i] = s["r"]

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(mat, dtype=bool), k=1)
    sns.heatmap(
        mat,
        mask=mask,
        annot=True,
        fmt=".3f",
        xticklabels=models,
        yticklabels=models,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(f"{split.capitalize()} - {phenotype_dir} (raw files, equal weights)", fontsize=14, weight="bold")
    plt.tight_layout()

    os.makedirs(phenotype_dir, exist_ok=True)
    out = os.path.join(phenotype_dir, f"correlation_{split}_raw.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ {out}")


def save_csv(all_results, phenotype_dir):
    rows = []
    for split, res in all_results.items():
        for pk, s in res.items():
            a, b = pk.split("_vs_")
            rows.append(
                {
                    "Split": split,
                    "Model_1": a,
                    "Model_2": b,
                    "r": s["r"],
                    "CI_Lower_desc": s["ci_l"],
                    "CI_Upper_desc": s["ci_u"],
                    "N_Folds": s["n_folds"],
                    "Fold_SD": s["fold_sd"],
                    "Avg_Tissues_per_Fold": s["avg_tissues_per_fold"],
                    "Avg_Usable_Genes_per_Fold": s["avg_usable_genes_per_fold"],
                }
            )

    os.makedirs(phenotype_dir, exist_ok=True)
    out = os.path.join(phenotype_dir, "correlation_results_raw.csv")
    pl.DataFrame(rows).write_csv(out)
    print(f"✅ {out}")


def save_latex(all_results, phenotype_dir):
    os.makedirs(phenotype_dir, exist_ok=True)
    out = os.path.join(phenotype_dir, "correlation_table_raw.tex")

    pairs = sorted(all_results.get("train", {}).keys())

    with open(out, "w") as f:
        f.write("\\begin{table*}[htbp]\n\\centering\n")
        f.write(f"\\caption{{Cross-model correlations ({phenotype_dir}, raw files; equal weights in $r$-space). ")
        f.write("Tissue $r$ is the mean of gene-wise Pearson correlations; fold $r$ is the mean across tissues; ")
        f.write("split $r$ is the mean across folds.}}\n")
        f.write("\\begin{tabular}{lccc}\n\\hline\n")
        f.write("\\textbf{Model Pair} & \\textbf{Train} & \\textbf{Validation} & \\textbf{Test} \\\\\n\\hline\n")

        for pk in pairs:
            name = pk.replace("_vs_", " vs ")
            vals = []
            for sp in ["train", "validation", "test"]:
                if sp in all_results and pk in all_results[sp]:
                    s = all_results[sp][pk]
                    vals.append(f"{s['r']:.2f}")
                else:
                    vals.append("---")
            f.write(f"{name} & {vals[0]} & {vals[1]} & {vals[2]} \\\\\n")

        f.write("\\hline\n\\end{tabular}\n\\end{table*}\n")

    print(f"✅ {out}")


def print_results(all_results):
    print("\n" + "=" * 90)
    print("RESULTS (raw files, equal weights in r-space)")
    print("=" * 90)
    for split in ["train", "validation", "test"]:
        if split not in all_results:
            continue
        print(f"\n{split.upper()}:")
        print("-" * 90)
        items = sorted(all_results[split].items(), key=lambda x: x[1]["r"], reverse=True)
        for pk, s in items:
            print(
                f"  {pk:30s}: r = {s['r']:+.3f} "
                f"[{s['ci_l']:+.3f}, {s['ci_u']:+.3f}]  "
                f"folds={s['n_folds']}, avg_tissues/fold={s['avg_tissues_per_fold']:.1f}, "
                f"avg_usable_genes/fold={s['avg_usable_genes_per_fold']:.0f}"
            )


# ===========================
# MAIN
# ===========================

print("\n" + "=" * 70)
print("CORRELATION ANALYSIS (RAW FILES, EQUAL WEIGHTS IN r-SPACE)")
print("=" * 70)
print("Logic:")
print("  Tissue  -> mean over genes of Pearson r (one tissue number)")
print("  Fold    -> mean over tissues (equal weight per tissue)")
print("  Split   -> mean over folds (equal weight per fold)")
print("=" * 70)

folds = get_folds(phenotype)
if not folds:
    print(f"❌ No folds found for '{phenotype}'")
    sys.exit(1)

print(f"\nPhenotype: {phenotype}")
print(f"Folds: {folds}")
print(f"Using NON-fixed expression files (skipping any '*_fixed*.csv')\n")

all_results = {}

for split in ["train", "validation", "test"]:
    res, models = analyze_split_equalweights(phenotype, folds, split)
    all_results[split] = res
    plot_matrix(res, models, split, phenotype)

print_results(all_results)
save_csv(all_results, phenotype)
save_latex(all_results, phenotype)

print("\n✅ DONE\n")
