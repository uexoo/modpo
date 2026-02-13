"""Analyze HH-RLHF Option1/2: shared-output conflict map and conflict-stratified tradeoff."""

from dataclasses import dataclass, field
import csv
import glob
import hashlib
import json
import math
import os
import re
from itertools import combinations
from typing import Optional

import numpy as np
import tyro
from scipy.stats import kendalltau, spearmanr


def _sign(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    s = np.zeros_like(x, dtype=np.int8)
    s[x > eps] = 1
    s[x < -eps] = -1
    return s


def _safe_corr(a: np.ndarray, b: np.ndarray):
    pearson = np.nan
    spearman = np.nan
    kendall = np.nan
    if len(a) >= 3 and np.std(a) > 0 and np.std(b) > 0:
        pearson = float(np.corrcoef(a, b)[0, 1])
    if len(a) >= 3:
        sp = spearmanr(a, b, nan_policy="omit").correlation
        kd = kendalltau(a, b, nan_policy="omit").correlation
        if sp is not None and np.isfinite(sp):
            spearman = float(sp)
        if kd is not None and np.isfinite(kd):
            kendall = float(kd)
    return pearson, spearman, kendall


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_bootstrap: int, stat_fn):
    if len(values) == 0:
        return (np.nan, np.nan)
    stats = []
    n = len(values)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        val = stat_fn(values[idx])
        if np.isfinite(val):
            stats.append(val)
    if not stats:
        return (np.nan, np.nan)
    lo, hi = np.quantile(np.asarray(stats), [0.025, 0.975])
    return float(lo), float(hi)


def _bootstrap_ci_pair(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, n_bootstrap: int, stat_fn):
    if len(a) == 0:
        return (np.nan, np.nan)
    stats = []
    n = len(a)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        val = stat_fn(a[idx], b[idx])
        if np.isfinite(val):
            stats.append(val)
    if not stats:
        return (np.nan, np.nan)
    lo, hi = np.quantile(np.asarray(stats), [0.025, 0.975])
    return float(lo), float(hi)


def _extract_prompt_text(obj: dict) -> Optional[str]:
    rp = obj.get("raw_prompt")
    if isinstance(rp, str) and rp.strip():
        return rp
    p = obj.get("prompt")
    if isinstance(p, str) and p.strip():
        return p
    return None


@dataclass
class ScriptArguments:
    output_dir: str = field(metadata={"help": "Output directory for analysis artifacts"})
    scores_dir: list[str] = field(default_factory=list, metadata={"help": "Directory with scores_ray2333.jsonl. Repeat."})
    label: Optional[list[str]] = field(default=None, metadata={"help": "Label for each --scores_dir. Repeat."})
    helpful_key: str = field(default="ray2333_helpful")
    harmless_key: str = field(default="ray2333_harmless")
    bootstrap: int = field(default=1000)
    seed: int = field(default=42)
    max_examples: Optional[int] = field(default=None)


def _load_scores(scores_dir: str, helpful_key: str, harmless_key: str, max_examples: Optional[int]):
    files = sorted(glob.glob(os.path.join(scores_dir, "*.jsonl")))
    if not files:
        raise FileNotFoundError(f"No jsonl files in {scores_dir}")

    hashes = []
    helpful = []
    harmless = []

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                prompt = _extract_prompt_text(obj)
                scores = obj.get("scores", {})
                h = scores.get(helpful_key)
                s = scores.get(harmless_key)
                if not isinstance(prompt, str) or h is None or s is None:
                    continue
                h = float(h)
                s = float(s)
                if not np.isfinite(h) or not np.isfinite(s):
                    continue

                prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
                hashes.append(prompt_hash)
                helpful.append(h)
                harmless.append(s)

                if max_examples is not None and len(hashes) >= max_examples:
                    return hashes, np.asarray(helpful, dtype=float), np.asarray(harmless, dtype=float)

    return hashes, np.asarray(helpful, dtype=float), np.asarray(harmless, dtype=float)


def _check_prompt_alignment(loaded: dict[str, tuple[list[str], np.ndarray, np.ndarray]]):
    labels = list(loaded.keys())
    ref_label = labels[0]
    ref_hashes = loaded[ref_label][0]

    for label in labels[1:]:
        hashes = loaded[label][0]
        if len(hashes) != len(ref_hashes):
            raise ValueError(f"Prompt count mismatch: {label}={len(hashes)} vs {ref_label}={len(ref_hashes)}")
        for i, (a, b) in enumerate(zip(ref_hashes, hashes)):
            if a != b:
                raise ValueError(
                    f"Prompt order/content mismatch at index {i}: {ref_label}={a} vs {label}={b}. "
                    "Run validate_eval_set.py first."
                )


def _conflict_ratio(dh: np.ndarray, ds: np.ndarray) -> float:
    sh = _sign(dh)
    ss = _sign(ds)
    mask = (sh != 0) & (ss != 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(sh[mask] != ss[mask]))


def _pearson(dh: np.ndarray, ds: np.ndarray) -> float:
    if len(dh) < 3 or np.std(dh) == 0 or np.std(ds) == 0:
        return np.nan
    return float(np.corrcoef(dh, ds)[0, 1])


def _parse_weight(label: str) -> Optional[float]:
    m = re.search(r"w([0-9]+(?:\.[0-9]+)?)", label)
    if not m:
        return None
    return float(m.group(1))


def _pareto_mask(points: list[tuple[float, float]]) -> list[bool]:
    n = len(points)
    out = [True] * n
    for i in range(n):
        hi, si = points[i]
        for j in range(n):
            if i == j:
                continue
            hj, sj = points[j]
            if (hj >= hi and sj >= si) and (hj > hi or sj > si):
                out[i] = False
                break
    return out


def main():
    args = tyro.cli(ScriptArguments)

    if len(args.scores_dir) < 2:
        raise ValueError("Need at least two --scores_dir entries.")

    labels = args.label
    if labels is not None and len(labels) != len(args.scores_dir):
        raise ValueError("--label count must match --scores_dir count.")
    if labels is None:
        labels = [os.path.basename(d.rstrip("/")) for d in args.scores_dir]

    loaded = {}
    for label, scores_dir in zip(labels, args.scores_dir):
        hashes, helpful, harmless = _load_scores(
            scores_dir=scores_dir,
            helpful_key=args.helpful_key,
            harmless_key=args.harmless_key,
            max_examples=args.max_examples,
        )
        loaded[label] = (hashes, helpful, harmless)

    _check_prompt_alignment(loaded)

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    n_prompts = len(next(iter(loaded.values()))[0])
    label_list = list(loaded.keys())

    # Option 1: model-pair conflict/correlation map.
    opt1_rows = []
    all_dh = []
    all_ds = []
    for a, b in combinations(label_list, 2):
        h_a = loaded[a][1]
        s_a = loaded[a][2]
        h_b = loaded[b][1]
        s_b = loaded[b][2]

        dh = h_a - h_b
        ds = s_a - s_b
        all_dh.append(dh)
        all_ds.append(ds)

        conflict = _conflict_ratio(dh, ds)
        pearson, spearman, kendall = _safe_corr(dh, ds)

        conflict_ci_low, conflict_ci_high = _bootstrap_ci_pair(
            dh, ds, rng, args.bootstrap, lambda x, y: _conflict_ratio(x, y)
        )
        pearson_ci_low, pearson_ci_high = _bootstrap_ci_pair(
            dh, ds, rng, args.bootstrap, lambda x, y: _pearson(x, y)
        )

        opt1_rows.append(
            {
                "model_a": a,
                "model_b": b,
                "n": len(dh),
                "conflict_ratio": None if not np.isfinite(conflict) else float(conflict),
                "conflict_ci_low": None if not np.isfinite(conflict_ci_low) else float(conflict_ci_low),
                "conflict_ci_high": None if not np.isfinite(conflict_ci_high) else float(conflict_ci_high),
                "pearson_r": None if not np.isfinite(pearson) else float(pearson),
                "pearson_ci_low": None if not np.isfinite(pearson_ci_low) else float(pearson_ci_low),
                "pearson_ci_high": None if not np.isfinite(pearson_ci_high) else float(pearson_ci_high),
                "spearman_rho": None if not np.isfinite(spearman) else float(spearman),
                "kendall_tau": None if not np.isfinite(kendall) else float(kendall),
            }
        )

    if all_dh:
        dh = np.concatenate(all_dh)
        ds = np.concatenate(all_ds)
        conflict = _conflict_ratio(dh, ds)
        pearson, spearman, kendall = _safe_corr(dh, ds)
        opt1_rows.append(
            {
                "model_a": "ALL",
                "model_b": "ALL",
                "n": len(dh),
                "conflict_ratio": None if not np.isfinite(conflict) else float(conflict),
                "conflict_ci_low": None,
                "conflict_ci_high": None,
                "pearson_r": None if not np.isfinite(pearson) else float(pearson),
                "pearson_ci_low": None,
                "pearson_ci_high": None,
                "spearman_rho": None if not np.isfinite(spearman) else float(spearman),
                "kendall_tau": None if not np.isfinite(kendall) else float(kendall),
            }
        )

    opt1_csv = os.path.join(args.output_dir, "option1_model_pair_conflict_correlation.csv")
    with open(opt1_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_a",
                "model_b",
                "n",
                "conflict_ratio",
                "conflict_ci_low",
                "conflict_ci_high",
                "pearson_r",
                "pearson_ci_low",
                "pearson_ci_high",
                "spearman_rho",
                "kendall_tau",
            ],
        )
        writer.writeheader()
        writer.writerows(opt1_rows)

    # Option 2: prompt-level conflict bins and per-bin tradeoff geometry.
    pair_indices = list(combinations(range(len(label_list)), 2))
    prompt_conflict = np.full(n_prompts, np.nan, dtype=float)

    helpful_matrix = np.vstack([loaded[label][1] for label in label_list])
    harmless_matrix = np.vstack([loaded[label][2] for label in label_list])

    for i in range(n_prompts):
        indicators = []
        for ai, bi in pair_indices:
            dh = helpful_matrix[ai, i] - helpful_matrix[bi, i]
            ds = harmless_matrix[ai, i] - harmless_matrix[bi, i]
            sh = _sign(np.asarray([dh]))[0]
            ss = _sign(np.asarray([ds]))[0]
            if sh == 0 or ss == 0:
                continue
            indicators.append(1.0 if sh != ss else 0.0)
        if indicators:
            prompt_conflict[i] = float(np.mean(indicators))

    finite = np.isfinite(prompt_conflict)
    if finite.sum() == 0:
        raise ValueError("All prompt-level conflicts are NaN; cannot build bins.")

    q1, q2 = np.quantile(prompt_conflict[finite], [1 / 3, 2 / 3])
    bins = np.full(n_prompts, "unknown", dtype=object)
    for i in range(n_prompts):
        x = prompt_conflict[i]
        if not np.isfinite(x):
            continue
        if x <= q1:
            bins[i] = "low"
        elif x <= q2:
            bins[i] = "mid"
        else:
            bins[i] = "high"

    opt2_stats_rows = []
    for bin_name in ["low", "mid", "high"]:
        idx = np.where(bins == bin_name)[0]
        if len(idx) == 0:
            continue
        for li, label in enumerate(label_list):
            h = helpful_matrix[li, idx]
            s = harmless_matrix[li, idx]

            h_mean = float(np.mean(h))
            s_mean = float(np.mean(s))
            h_lo, h_hi = _bootstrap_ci(h, rng, args.bootstrap, lambda x: float(np.mean(x)))
            s_lo, s_hi = _bootstrap_ci(s, rng, args.bootstrap, lambda x: float(np.mean(x)))

            opt2_stats_rows.append(
                {
                    "bin": bin_name,
                    "label": label,
                    "weight": _parse_weight(label),
                    "n_prompts": len(idx),
                    "mean_helpful": h_mean,
                    "helpful_ci_low": None if not np.isfinite(h_lo) else float(h_lo),
                    "helpful_ci_high": None if not np.isfinite(h_hi) else float(h_hi),
                    "mean_harmless": s_mean,
                    "harmless_ci_low": None if not np.isfinite(s_lo) else float(s_lo),
                    "harmless_ci_high": None if not np.isfinite(s_hi) else float(s_hi),
                }
            )

    # Pareto per bin using mean helpful/harmless.
    opt2_pareto_rows = []
    sft_candidates = [l for l in label_list if "sft" in l.lower()]
    sft_label = sft_candidates[0] if sft_candidates else label_list[0]

    for bin_name in ["low", "mid", "high"]:
        rows = [r for r in opt2_stats_rows if r["bin"] == bin_name]
        if not rows:
            continue

        points = [(r["mean_helpful"], r["mean_harmless"]) for r in rows]
        mask = _pareto_mask(points)

        sft_row = next((r for r in rows if r["label"] == sft_label), None)
        for r, is_pareto in zip(rows, mask):
            both_above_sft = None
            if sft_row is not None:
                both_above_sft = bool(
                    (r["mean_helpful"] > sft_row["mean_helpful"]) and
                    (r["mean_harmless"] > sft_row["mean_harmless"]) 
                )
            out = dict(r)
            out["is_pareto"] = is_pareto
            out["both_above_sft"] = both_above_sft
            out["sft_label"] = sft_label
            opt2_pareto_rows.append(out)

    opt2_stats_csv = os.path.join(args.output_dir, "option2_bin_model_stats.csv")
    with open(opt2_stats_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "bin",
                "label",
                "weight",
                "n_prompts",
                "mean_helpful",
                "helpful_ci_low",
                "helpful_ci_high",
                "mean_harmless",
                "harmless_ci_low",
                "harmless_ci_high",
            ],
        )
        writer.writeheader()
        writer.writerows(opt2_stats_rows)

    opt2_pareto_csv = os.path.join(args.output_dir, "option2_bin_pareto.csv")
    with open(opt2_pareto_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "bin",
                "label",
                "weight",
                "n_prompts",
                "mean_helpful",
                "mean_harmless",
                "is_pareto",
                "both_above_sft",
                "sft_label",
            ],
        )
        writer.writeheader()
        writer.writerows(opt2_pareto_rows)

    bins_json = os.path.join(args.output_dir, "option2_prompt_conflict_bins.json")
    with open(bins_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_prompts": n_prompts,
                "q1": float(q1),
                "q2": float(q2),
                "counts": {
                    "low": int(np.sum(bins == "low")),
                    "mid": int(np.sum(bins == "mid")),
                    "high": int(np.sum(bins == "high")),
                    "unknown": int(np.sum(bins == "unknown")),
                },
                "sft_label": sft_label,
            },
            f,
            indent=2,
        )

    summary_md = os.path.join(args.output_dir, "summary_opt12.md")
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# HH-RLHF Option1/Option2 Summary\n\n")
        f.write(f"- n_prompts: {n_prompts}\n")
        f.write(f"- labels: {', '.join(label_list)}\n")
        f.write(f"- sft_label: {sft_label}\n")
        f.write(f"- prompt_conflict_quantiles: q1={q1:.6f}, q2={q2:.6f}\n")
        f.write("\n## Option 1 (model-pair conflict/correlation)\n")
        for row in opt1_rows:
            f.write(
                f"- {row['model_a']} vs {row['model_b']}: conflict={row['conflict_ratio']} "
                f"pearson={row['pearson_r']} n={row['n']}\n"
            )
        f.write("\n## Option 2 (bin-wise Pareto)\n")
        for bin_name in ["low", "mid", "high"]:
            rows = [r for r in opt2_pareto_rows if r["bin"] == bin_name and r["is_pareto"]]
            if not rows:
                continue
            labels_str = ", ".join(r["label"] for r in rows)
            f.write(f"- {bin_name}: Pareto labels = {labels_str}\n")

    print(f"Saved: {opt1_csv}")
    print(f"Saved: {opt2_stats_csv}")
    print(f"Saved: {opt2_pareto_csv}")
    print(f"Saved: {bins_json}")
    print(f"Saved: {summary_md}")


if __name__ == "__main__":
    main()
