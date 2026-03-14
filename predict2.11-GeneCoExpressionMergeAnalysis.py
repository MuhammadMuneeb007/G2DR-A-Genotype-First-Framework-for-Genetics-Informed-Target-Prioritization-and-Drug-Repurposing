#!/usr/bin/env python3
"""
Robust Coexpression Analysis Summary Generator
==============================================
Reads coexpression analysis outputs and generates a concise, LLM-ready summary.

Fixes vs your current script:
- Robust section detection (doesn't depend on exact "SECTION X:" strings)
- Flexible table parsing (handles column changes without silently failing)
- Safer pathway parsing (won't break on numbers inside pathway names)
- Tissue stability parsing without requiring the "±" symbol
- Clear warnings to STDERR when parsing looks suspicious / incomplete
- Optional JSON sidecar support (if a .json exists, it will be used)

Usage:
    python3 summarize_coexpression_robust.py <phenotype> [--adjusted|--original]
    python3 summarize_coexpression_robust.py <phenotype> --file <path>
    python3 summarize_coexpression_robust.py <phenotype> --compact

Example:
    python3 summarize_coexpression_robust.py migraine
    python3 summarize_coexpression_robust.py migraine --original
"""

import os
import sys
import re
import json
import argparse
from typing import Dict, List, Optional, Any, Tuple


# -------------------------------
# Utilities
# -------------------------------

def eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def safe_int(x: str) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def safe_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def wrap_text(text: str, width: int = 78) -> List[str]:
    words = text.split()
    out, line = [], ""
    for w in words:
        if len(line) + len(w) + (1 if line else 0) > width:
            out.append(line)
            line = w
        else:
            line = (line + " " + w) if line else w
    if line:
        out.append(line)
    return out


def normalize_heading(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return s


def split_csv_like(line: str) -> List[str]:
    # Split on commas, but also allow " , " spacing; strip blanks
    parts = [p.strip() for p in line.split(",")]
    return [p for p in parts if p]


# -------------------------------
# Robust parsing logic
# -------------------------------

def try_load_json_sidecar(txt_path: str) -> Optional[Dict[str, Any]]:
    """
    If a JSON sidecar exists (same basename + .json), load it.
    """
    base, _ = os.path.splitext(txt_path)
    cand = base + ".json"
    if os.path.exists(cand):
        try:
            with open(cand, "r") as f:
                return json.load(f)
        except Exception as ex:
            eprint(f"[WARN] Found JSON sidecar but failed to read it: {cand} ({ex})")
    return None


def extract_kv(lines: List[str]) -> Dict[str, str]:
    """
    Extract key-value pairs from lines like:
      Phenotype: migraine
      Folds: [0,1,2,3,4]
      Expression: _fixed ...
    """
    kv = {}
    for ln in lines:
        m = re.match(r"^\s*([A-Za-z][A-Za-z0-9 _/\-]+)\s*:\s*(.+?)\s*$", ln)
        if m:
            k = normalize_heading(m.group(1))
            v = m.group(2).strip()
            kv[k] = v
    return kv


def detect_sections(lines: List[str]) -> Dict[str, List[str]]:
    """
    Flexible section detection.
    Recognizes:
      - 'SECTION X: ...'
      - lines like 'TOP CONSISTENT HUB GENES'
      - blocks separated by ===== lines
    """
    sections: Dict[str, List[str]] = {}
    current = "preamble"
    sections[current] = []

    def start_section(name: str):
        nonlocal current
        key = normalize_heading(name)
        current = key if key else "unknown"
        sections.setdefault(current, [])

    for ln in lines:
        raw = ln.rstrip("\n")

        # SECTION header
        m = re.match(r"^\s*SECTION\s*\d+\s*:\s*(.+?)\s*$", raw, flags=re.I)
        if m:
            start_section(m.group(1))
            continue

        # Narrative
        if re.search(r"\bNARRATIVE\s+SUMMARY\b", raw, flags=re.I):
            start_section("Narrative summary")
            continue

        # Common prominent headings
        if re.match(r"^\s*[A-Z0-9][A-Z0-9 \-/]{8,}\s*$", raw) and not raw.strip().startswith(("HTTP", "WWW")):
            # Avoid treating separators as headings
            if set(raw.strip()) <= set("= -"):
                sections[current].append(raw)
                continue
            # Treat as heading if it contains meaningful keywords
            if any(k in raw.upper() for k in ["HUB", "PATHWAY", "GO", "DISEASE", "TISSUE", "ENRICH", "SUMMARY"]):
                start_section(raw.strip())
                continue

        sections[current].append(raw)

    return sections


def parse_consistent_genes(section_lines: List[str]) -> List[Dict[str, Any]]:
    """
    Parse consistent hub genes from table-like content.
    Supports multiple formats, e.g.:
      rank gene score n_methods n_tissues fold_appearances avg_rank
      gene frequency avg_rank
      rank gene frequency avg_rank
    """
    genes = []

    patterns = [
        # rank gene score methods tissues foldapp avgrank
        re.compile(r"^\s*(\d+)\s+(\S+)\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+(\d+)\s+(\d+)\s+(\d+)\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$"),
        # rank gene freq avgrank
        re.compile(r"^\s*(\d+)\s+(\S+)\s+(\d+)\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$"),
        # gene freq avgrank
        re.compile(r"^\s*(\S+)\s+(\d+)\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$"),
    ]

    # Skip header-ish lines
    for ln in section_lines:
        if not ln.strip():
            continue
        if any(h in ln.lower() for h in ["rank", "gene", "score", "method", "tissue", "avg", "frequency"]):
            # header line: skip
            continue
        if ln.strip().startswith(("-", "•")):
            continue

        for pi, pat in enumerate(patterns):
            m = pat.match(ln)
            if not m:
                continue

            if pi == 0:
                rank = safe_int(m.group(1))
                gene = m.group(2)
                score = safe_float(m.group(3))
                n_methods = safe_int(m.group(4))
                n_tissues = safe_int(m.group(5))
                fold_app = safe_int(m.group(6))
                avg_rank = safe_float(m.group(7))
                genes.append({
                    "rank": rank,
                    "gene": gene,
                    "score": score,
                    "n_methods": n_methods,
                    "n_tissues": n_tissues,
                    "fold_appearances": fold_app,
                    "avg_rank": avg_rank,
                })
            elif pi == 1:
                rank = safe_int(m.group(1))
                gene = m.group(2)
                freq = safe_int(m.group(3))
                avg_rank = safe_float(m.group(4))
                genes.append({
                    "rank": rank,
                    "gene": gene,
                    "score": None,
                    "n_methods": None,
                    "n_tissues": None,
                    "fold_appearances": freq,
                    "avg_rank": avg_rank,
                })
            else:
                gene = m.group(1)
                freq = safe_int(m.group(2))
                avg_rank = safe_float(m.group(3))
                genes.append({
                    "rank": None,
                    "gene": gene,
                    "score": None,
                    "n_methods": None,
                    "n_tissues": None,
                    "fold_appearances": freq,
                    "avg_rank": avg_rank,
                })
            break

    # Clean duplicates by best rank/frequency
    seen = {}
    for g in genes:
        key = g["gene"]
        if key not in seen:
            seen[key] = g
        else:
            # keep entry with higher fold_appearances; tie-break by avg_rank
            a = seen[key]
            fa = a.get("fold_appearances") or -1
            fb = g.get("fold_appearances") or -1
            if fb > fa:
                seen[key] = g
            elif fb == fa:
                ar = a.get("avg_rank") or 1e9
                br = g.get("avg_rank") or 1e9
                if br < ar:
                    seen[key] = g

    out = list(seen.values())
    # sort
    def sort_key(x):
        r = x.get("rank")
        fa = x.get("fold_appearances") or 0
        ar = x.get("avg_rank") or 1e9
        return (r if r is not None else 1e9, -fa, ar)

    out.sort(key=sort_key)
    return out


def parse_tiers(lines: List[str]) -> Dict[str, List[str]]:
    """
    Parse tier gene lists from text blocks.
    Works with:
      - 'TIER 1 ...' followed by comma-separated lists across wrapped lines
      - bullet lists
    """
    tiers = {"tier1": [], "tier2": [], "tier3": []}
    current = None

    tier_map = [
        ("tier 1", "tier1"),
        ("tier1", "tier1"),
        ("tier 2", "tier2"),
        ("tier2", "tier2"),
        ("tier 3", "tier3"),
        ("tier3", "tier3"),
    ]

    for ln in lines:
        low = ln.lower()

        found = None
        for k, v in tier_map:
            if k in low and "tier" in low:
                found = v
                break
        if found:
            current = found
            continue

        if current:
            # stop on major section change
            if re.match(r"^\s*SECTION\b", ln, flags=re.I) or (ln.strip() and set(ln.strip()) <= set("= -")):
                current = None
                continue

            # collect genes from comma-separated lists or bullet lines
            stripped = ln.strip()
            if not stripped:
                continue

            if stripped.startswith(("-", "•")):
                stripped = stripped.lstrip("-•").strip()

            # If line looks like "→ GENE1, GENE2, ..."
            stripped = stripped.replace("→", "").strip()

            # Extract gene-like tokens (allow letters, numbers, dash, dot)
            if "," in stripped:
                candidates = split_csv_like(stripped)
            else:
                # if space-separated list, still try
                candidates = re.split(r"\s+", stripped)

            # keep plausible gene symbols
            genes = []
            for c in candidates:
                c = c.strip().strip(",;")
                if not c:
                    continue
                if len(c) > 40:
                    continue
                if re.match(r"^[A-Za-z0-9][A-Za-z0-9\.\-_]*$", c):
                    # avoid generic words
                    if c.lower() in {"genes", "none", "confidence", "highest", "high", "cross", "tissue"}:
                        continue
                    genes.append(c)

            tiers[current].extend(genes)

    # de-dup preserve order
    for k in tiers:
        seen = set()
        out = []
        for g in tiers[k]:
            if g and g not in seen:
                seen.add(g)
                out.append(g)
        tiers[k] = out
    return tiers


def parse_terms_generic(lines: List[str], max_items: int = 200) -> List[Dict[str, Any]]:
    """
    Parse enriched terms / pathways / GO lines in a robust way.

    Supports:
      - "- TERM [q=..., p=...] (DB)"
      - "• TERM (recurrence: X)"
      - "1 TERM ... X"
    """
    out = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue

        # ignore headers
        if any(h in s.lower() for h in ["rank", "term", "pathway", "count", "recurrence", "database", "q=", "p="]) and len(s.split()) <= 6:
            continue

        # strip bullet
        s2 = s.lstrip("-•").strip()

        # Parse q/p if present
        qv = pv = None
        mqp = re.search(r"\bq\s*=\s*([0-9eE\.\-+]+)", s2)
        if mqp:
            qv = safe_float(mqp.group(1))
        mp = re.search(r"\bp\s*=\s*([0-9eE\.\-+]+)", s2)
        if mp:
            pv = safe_float(mp.group(1))

        # Parse DB in parentheses at end
        db = None
        mdb = re.search(r"\(([^)]+)\)\s*$", s2)
        if mdb and len(mdb.group(1)) <= 60:
            db = mdb.group(1).strip()

        # Parse recurrence/count
        cnt = None
        mc = re.search(r"(recurrence|count)\s*[:=]\s*(\d+)", s2, flags=re.I)
        if mc:
            cnt = safe_int(mc.group(2))
        else:
            # trailing integer often used for count
            mt = re.search(r"\b(\d+)\s*$", s2)
            if mt:
                cnt = safe_int(mt.group(1))

        # Remove bracketed metadata for cleaner term
        term = re.sub(r"\[[^\]]*\]", "", s2).strip()
        term = re.sub(r"\([^)]*\)\s*$", "", term).strip() if db else term
        # Remove leading rank number
        term = re.sub(r"^\d+\s+", "", term).strip()

        # If term still ends with a count, drop it (only if it looks like a count token)
        term = re.sub(r"\s+\d+\s*$", "", term).strip()

        if term and len(term) > 2 and len(out) < max_items:
            out.append({"term": term, "count": cnt, "q": qv, "p": pv, "database": db})

    # de-dup by term
    seen = set()
    dedup = []
    for x in out:
        t = x["term"]
        if t not in seen:
            seen.add(t)
            dedup.append(x)
    return dedup


def parse_tfs(lines: List[str]) -> List[Dict[str, Any]]:
    out = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        s = s.lstrip("-•").strip()
        # patterns:
        # TF (freq=12)
        m = re.match(r"^([A-Za-z0-9\.\-_]+)\s*\(freq\s*=\s*(\d+)\)\s*$", s, flags=re.I)
        if m:
            out.append({"tf": m.group(1), "frequency": safe_int(m.group(2))})
            continue
        # TF 12
        m2 = re.match(r"^([A-Za-z0-9\.\-_]+)\s+(\d+)\s*$", s)
        if m2:
            tf = m2.group(1)
            if tf.lower() in {"tf", "transcription", "factor", "factors"}:
                continue
            out.append({"tf": tf, "frequency": safe_int(m2.group(2))})
    # de-dup
    seen = set()
    dedup = []
    for x in out:
        if x["tf"] not in seen:
            seen.add(x["tf"])
            dedup.append(x)
    # sort by frequency desc if available
    dedup.sort(key=lambda x: (-(x["frequency"] or 0), x["tf"]))
    return dedup


def parse_tissue_stability(lines: List[str]) -> List[Dict[str, Any]]:
    """
    Parse tissue stability lines without requiring ±.
    Accepts formats like:
      TissueName  mean_density=0.123  mean_abs_r=0.456 ...
      TissueName  0.1234 0.5678 0.0123
      • TissueName ...
    """
    out = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if any(h in s.lower() for h in ["tissue", "mean", "density", "stability"]) and len(s.split()) < 3:
            continue

        s = s.lstrip("-•").strip()

        # if line has key=val tokens
        if "=" in s:
            parts = s.split()
            tissue = parts[0]
            vals = {}
            for p in parts[1:]:
                if "=" in p:
                    k, v = p.split("=", 1)
                    vals[k.strip()] = v.strip().strip(",;")
            out.append({"tissue": tissue, "raw_line": s, "metrics": vals})
            continue

        # if table-ish numeric columns after tissue
        parts = s.split()
        if len(parts) >= 2 and re.match(r"^[A-Za-z0-9\-_]+$", parts[0]):
            tissue = parts[0]
            nums = [safe_float(x) for x in parts[1:]]
            if any(x is not None for x in nums):
                out.append({"tissue": tissue, "raw_line": s, "metrics": None})
                continue

    # de-dup by tissue
    seen = set()
    dedup = []
    for x in out:
        t = x["tissue"]
        if t not in seen:
            seen.add(t)
            dedup.append(x)
    return dedup


def parse_summary_file(txt_path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(txt_path):
        return None

    # Prefer JSON sidecar if present
    js = try_load_json_sidecar(txt_path)
    if js is not None and isinstance(js, dict):
        # Normalize minimal keys into expected schema if possible
        return {
            "phenotype": js.get("phenotype", ""),
            "expression_type": js.get("expression_type", js.get("expression", "")),
            "n_folds": js.get("n_folds", js.get("folds", 0)),
            "narrative": js.get("narrative", ""),
            "consistent_genes": js.get("consistent_genes", []),
            "tier1_genes": js.get("tier1_genes", []),
            "tier2_genes": js.get("tier2_genes", []),
            "tier3_genes": js.get("tier3_genes", []),
            "pathways": js.get("pathways", []),
            "go_terms": js.get("go_terms", []),
            "transcription_factors": js.get("transcription_factors", []),
            "disease_associations": js.get("disease_associations", []),
            "tissue_stability": js.get("tissue_stability", []),
            "_source": "json",
        }

    with open(txt_path, "r") as f:
        content = f.read()
    lines = content.splitlines()

    kv = extract_kv(lines)
    sections = detect_sections(lines)

    phenotype = kv.get("phenotype", "")
    # Some files might have "phenotype name"
    if not phenotype:
        phenotype = kv.get("phenotype name", "")

    n_folds = 0
    if "number of cross validation folds" in kv:
        n_folds = safe_int(kv["number of cross validation folds"]) or 0
    elif "folds" in kv:
        # try parse [0,1,2,3,4]
        m = re.search(r"\d+", kv["folds"])
        if m:
            # might be list; approximate as count of ints
            n_folds = len(re.findall(r"\d+", kv["folds"]))
    elif "cross validation folds" in kv:
        n_folds = safe_int(kv["cross validation folds"]) or 0

    expr_type = kv.get("expression data", "") or kv.get("expression", "")

    # narrative
    narrative_lines = sections.get("narrative summary", [])
    narrative = " ".join([ln.strip() for ln in narrative_lines if ln.strip() and not ln.strip().startswith(("-", "•"))])

    # consistent genes: search best candidate section(s)
    consistent = []
    cand_keys = [k for k in sections.keys() if "hub" in k and "gene" in k] + \
               [k for k in sections.keys() if "consistent" in k and "gene" in k]
    if not cand_keys:
        # fallback: any section containing many gene-like rows
        cand_keys = list(sections.keys())

    for ck in cand_keys:
        parsed = parse_consistent_genes(sections.get(ck, []))
        if len(parsed) > len(consistent):
            consistent = parsed

    # tiers: scan whole file (robust)
    tiers = parse_tiers(lines)

    # pathways/go/tf/disease/tissue: find likely sections, fallback to whole file
    def best_section_contains(*needles: str) -> List[str]:
        for k, v in sections.items():
            if all(n in k for n in needles):
                return v
        # fallback by partial
        for k, v in sections.items():
            if any(n in k for n in needles):
                return v
        return []

    pathways_lines = best_section_contains("pathway") or best_section_contains("enrich", "path") or []
    go_lines = best_section_contains("go") or best_section_contains("biological process") or []
    tf_lines = best_section_contains("transcription") or best_section_contains("tf") or []
    disease_lines = best_section_contains("disease") or best_section_contains("association") or []
    tissue_lines = best_section_contains("tissue") and best_section_contains("stability") or best_section_contains("tissue") or []

    pathways = parse_terms_generic(pathways_lines if pathways_lines else lines)
    go_terms = parse_terms_generic(go_lines if go_lines else [])
    tfs = parse_tfs(tf_lines if tf_lines else [])
    diseases = parse_terms_generic(disease_lines if disease_lines else [])
    tissue = parse_tissue_stability(tissue_lines if tissue_lines else [])

    # Convert diseases generic terms into disease_associations-like objects where possible
    disease_associations = []
    for d in diseases:
        disease_associations.append({
            "disease": d["term"],
            "count": d.get("count"),
            "q": d.get("q"),
            "p": d.get("p"),
        })

    # Sanity warnings
    if "phenotype" not in kv and not phenotype:
        eprint("[WARN] Could not find phenotype line (e.g., 'Phenotype: ...').")
    if consistent and len(consistent) < 3:
        eprint("[WARN] Parsed very few hub genes; report format may have changed.")
    if not consistent and any("hub" in k for k in sections.keys()):
        eprint("[WARN] Found hub-related section headings but could not parse hub table rows.")

    return {
        "phenotype": phenotype,
        "expression_type": expr_type,
        "n_folds": n_folds,
        "narrative": narrative,
        "consistent_genes": consistent,
        "tier1_genes": tiers["tier1"],
        "tier2_genes": tiers["tier2"],
        "tier3_genes": tiers["tier3"],
        "pathways": pathways,
        "go_terms": go_terms,
        "transcription_factors": tfs,
        "disease_associations": disease_associations,
        "tissue_stability": tissue,
        "_source": "text",
    }


# -------------------------------
# Output formatting
# -------------------------------

def generate_summary(results: Dict[str, Any]) -> str:
    if not results:
        return "ERROR: No results parsed."

    summary = []
    summary.append("=" * 80)
    summary.append("GENE CO-EXPRESSION ANALYSIS SUMMARY")
    summary.append("=" * 80)
    summary.append("")
    summary.append(f"PHENOTYPE: {results.get('phenotype','')}")
    summary.append(f"EXPRESSION DATA: {results.get('expression_type','')}")
    summary.append(f"CROSS-VALIDATION FOLDS: {results.get('n_folds',0)}")
    summary.append(f"PARSE SOURCE: {results.get('_source','')}")
    summary.append("")

    if results.get("narrative"):
        summary.append("OVERVIEW:")
        summary.append("-" * 40)
        summary.extend(wrap_text(results["narrative"], width=78))
        summary.append("")

    # Hub genes
    summary.append("TOP CONSISTENT HUB GENES:")
    summary.append("-" * 40)
    hubs = results.get("consistent_genes", [])
    if hubs:
        summary.append(f"{'Gene':<12} {'FoldFreq':<9} {'AvgRank':<8} {'Notes'}")
        summary.append("-" * 60)
        for g in hubs[:20]:
            gene = g.get("gene", "")
            freq = g.get("fold_appearances") or g.get("fold_frequency") or ""
            avg_rank = g.get("avg_rank")
            ar = f"{avg_rank:.2f}" if isinstance(avg_rank, (int, float)) and avg_rank is not None else ""
            notes = ""
            nm = g.get("n_methods")
            nt = g.get("n_tissues")
            if isinstance(nm, int) and isinstance(nt, int):
                if nm >= 5 and nt >= 4:
                    notes = "VERY HIGH confidence"
                elif nm >= 4 or nt >= 3:
                    notes = "HIGH confidence"
                elif nm >= 3 or nt >= 2:
                    notes = "MODERATE confidence"
                else:
                    notes = "Standard"
            summary.append(f"{gene:<12} {str(freq):<9} {ar:<8} {notes}".rstrip())
    else:
        summary.append("No hub genes parsed (or report format mismatch).")
    summary.append("")

    # Tiers
    summary.append("HUB GENE TIERS (if present):")
    summary.append("-" * 40)
    t1 = results.get("tier1_genes", [])
    t2 = results.get("tier2_genes", [])
    t3 = results.get("tier3_genes", [])
    summary.append(f"TIER 1: {len(t1)} genes")
    if t1:
        summary.append(f"  → {', '.join(t1[:12])}")
    summary.append(f"TIER 2: {len(t2)} genes")
    if t2:
        summary.append(f"  → {', '.join(t2[:12])}")
    summary.append(f"TIER 3: {len(t3)} genes")
    if t3:
        summary.append(f"  → {', '.join(t3[:12])}")
    summary.append("")

    # TFs
    summary.append("TRANSCRIPTION FACTORS:")
    summary.append("-" * 40)
    tfs = results.get("transcription_factors", [])
    if tfs:
        tf_str = ", ".join([f"{x['tf']} (freq={x.get('frequency','')})" for x in tfs[:15]])
        summary.extend(wrap_text(tf_str, width=78))
    else:
        summary.append("None parsed.")
    summary.append("")

    # Pathways
    summary.append("TOP PATHWAYS / ENRICHED TERMS:")
    summary.append("-" * 40)
    pws = results.get("pathways", [])
    if pws:
        for x in pws[:10]:
            tail = []
            if x.get("count") is not None:
                tail.append(f"recurrence={x['count']}")
            if x.get("q") is not None:
                tail.append(f"q={x['q']:.2e}")
            if x.get("database"):
                tail.append(x["database"])
            meta = f" ({', '.join(tail)})" if tail else ""
            summary.append(f"  • {x['term']}{meta}")
    else:
        summary.append("None parsed.")
    summary.append("")

    # GO terms
    summary.append("TOP GO TERMS:")
    summary.append("-" * 40)
    gos = results.get("go_terms", [])
    if gos:
        for x in gos[:10]:
            tail = []
            if x.get("count") is not None:
                tail.append(f"recurrence={x['count']}")
            if x.get("q") is not None:
                tail.append(f"q={x['q']:.2e}")
            meta = f" ({', '.join(tail)})" if tail else ""
            summary.append(f"  • {x['term']}{meta}")
    else:
        summary.append("None parsed.")
    summary.append("")

    # Diseases
    summary.append("DISEASE ASSOCIATIONS:")
    summary.append("-" * 40)
    dis = results.get("disease_associations", [])
    if dis:
        for x in dis[:10]:
            meta = []
            if x.get("count") is not None:
                meta.append(f"count={x['count']}")
            if x.get("q") is not None:
                meta.append(f"q={x['q']:.2e}")
            meta_s = f" ({', '.join(meta)})" if meta else ""
            summary.append(f"  • {x['disease']}{meta_s}")
    else:
        summary.append("None parsed.")
    summary.append("")

    # Tissue stability
    summary.append("TISSUE STABILITY (if present):")
    summary.append("-" * 40)
    ts = results.get("tissue_stability", [])
    if ts:
        for x in ts[:10]:
            summary.append(f"  • {x.get('tissue','')}")
    else:
        summary.append("None parsed.")
    summary.append("")

    # Questions for LLM
    summary.append("=" * 80)
    summary.append("QUESTIONS FOR BIOLOGICAL INTERPRETATION:")
    summary.append("=" * 80)
    summary.append("")
    if hubs:
        top_genes = [x.get("gene", "") for x in hubs[:10] if x.get("gene")]
        if top_genes:
            summary.append(f"1. What are the known functions of these hub genes: {', '.join(top_genes)}?")
            summary.append("")
            summary.append(f"2. How might these genes relate to {results.get('phenotype','this phenotype')}?")
            summary.append("")
    if pws:
        top_pws = [x["term"] for x in pws[:3]]
        summary.append(f"3. How are these pathways relevant: {'; '.join(top_pws)}?")
        summary.append("")
    summary.append("4. Which hubs are druggable / known targets, and what evidence supports them?")
    summary.append("5. What orthogonal validation would you recommend (e.g., TWAS colocalization, MR, replication)?")
    summary.append("")
    summary.append("=" * 80)
    summary.append("END OF SUMMARY")
    summary.append("=" * 80)

    return "\n".join(summary)


def generate_compact_prompt(results: Dict[str, Any]) -> str:
    if not results:
        return "ERROR: No results parsed."

    phenotype = results.get("phenotype", "")
    prompt = []
    prompt.append(f"Please interpret the following genetically-imputed gene co-expression results for {phenotype}:")
    prompt.append("")

    hubs = results.get("consistent_genes", [])
    if hubs:
        top_genes = [x.get("gene", "") for x in hubs[:15] if x.get("gene")]
        if top_genes:
            prompt.append(f"TOP HUB GENES: {', '.join(top_genes)}")
            prompt.append("")

    tier1 = results.get("tier1_genes", [])
    if tier1:
        prompt.append(f"HIGHEST CONFIDENCE HUBS: {', '.join(tier1[:12])}")
        prompt.append("")

    tfs = results.get("transcription_factors", [])
    if tfs:
        tf_list = [x["tf"] for x in tfs[:10] if x.get("tf")]
        if tf_list:
            prompt.append(f"TRANSCRIPTION FACTORS: {', '.join(tf_list)}")
            prompt.append("")

    pws = results.get("pathways", [])
    if pws:
        prompt.append("ENRICHED TERMS / PATHWAYS:")
        prompt.append("; ".join([x["term"] for x in pws[:8]]))
        prompt.append("")

    gos = results.get("go_terms", [])
    if gos:
        prompt.append("GO BIOLOGICAL PROCESSES:")
        prompt.append("; ".join([x["term"] for x in gos[:8]]))
        prompt.append("")

    dis = results.get("disease_associations", [])
    if dis:
        prompt.append("DISEASE ASSOCIATIONS:")
        prompt.append("; ".join([x["disease"] for x in dis[:6]]))
        prompt.append("")

    prompt.append("QUESTIONS:")
    prompt.append(f"1. What mechanisms link these hubs to {phenotype}?")
    prompt.append("2. Which hubs are the best therapeutic targets and why?")
    prompt.append("3. Are any hubs known drug targets or supported by Open Targets / GWAS?")
    prompt.append("4. What follow-up validation would you recommend (coloc/MR/replication)?")

    return "\n".join(prompt)


# -------------------------------
# Main
# -------------------------------

def resolve_results_file(phenotype: str, original: bool, file_arg: Optional[str]) -> str:
    if file_arg:
        return file_arg

    suffix = "_original" if original else "_adjusted"
    cand = [
        os.path.join(phenotype, f"coexpression_consistency_summary{suffix}.txt"),
        os.path.join(phenotype, f"coexpression_consistency_summary{suffix}.report.txt"),
        os.path.join(phenotype, "coexpression_consistency_summary.txt"),
        os.path.join(phenotype, "coexpression_consistency_summary.report.txt"),
    ]
    for p in cand:
        if os.path.exists(p):
            return p
    return cand[0]  # default (will error later)


def main():
    parser = argparse.ArgumentParser(
        description="Robust summarizer for coexpression analysis output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("phenotype", help="Phenotype directory name (e.g., migraine)")
    parser.add_argument("--original", action="store_true", help="Use original (non-adjusted) results instead of adjusted")
    parser.add_argument("--compact", "-c", action="store_true", help="Generate compact prompt format")
    parser.add_argument("--file", "-f", type=str, default=None, help="Directly specify the results file path")
    args = parser.parse_args()

    file_path = resolve_results_file(args.phenotype, args.original, args.file)

    if not os.path.exists(file_path):
        eprint(f"ERROR: Results file not found: {file_path}")
        eprint("\nTried looking for:")
        eprint(f"  - {args.phenotype}/coexpression_consistency_summary_adjusted.txt")
        eprint(f"  - {args.phenotype}/coexpression_consistency_summary_original.txt")
        eprint(f"  - {args.phenotype}/coexpression_consistency_summary.txt")
        sys.exit(1)

    eprint(f"[INFO] Reading results from: {file_path}")
    results = parse_summary_file(file_path)
    if not results:
        eprint("ERROR: Could not parse results file.")
        sys.exit(1)

    out = generate_compact_prompt(results) if args.compact else generate_summary(results)
    print(out)


if __name__ == "__main__":
    main()
