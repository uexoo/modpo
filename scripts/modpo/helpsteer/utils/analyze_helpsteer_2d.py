#!/usr/bin/env python3
"""
Analyze a 2D HelpSteer objective pair from generation + ArmoRM score artifacts.

This script computes:
- prompt alignment across labels
- prompt truncation + response capping diagnostics
- full-set and uncapped-intersection mean points
- Pareto sets under max/max or max/min objective modes
- paired bootstrap deltas vs a reference label (default: sft)

Expected eval structure under --eval_root:
- gens_sft/*.jsonl
- gens_modpo_w*/.jsonl (or other label naming compatible with gens_<label>)
- scores_armorm/<label>/scores_armorm.jsonl
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import random
import statistics
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from transformers import AutoTokenizer


def _iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _single_jsonl_file(dir_path: str) -> str:
    files = sorted(glob.glob(os.path.join(dir_path, "*.jsonl")))
    if not files:
        raise FileNotFoundError(f"No .jsonl files found in {dir_path}")
    return files[0]


def _prompt_hash(obj: dict) -> str:
    raw_prompt = obj.get("raw_prompt")
    if not isinstance(raw_prompt, str) or not raw_prompt:
        raw_prompt = obj.get("prompt", "")
    return hashlib.sha256(raw_prompt.encode("utf-8")).hexdigest()[:16]


def _bootstrap_ci_mean(values: Sequence[float], seed: int, n_boot: int, alpha: float) -> tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    mean = float(statistics.mean(values))
    if len(values) == 1:
        return mean, mean, mean

    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.mean(sample))
    means.sort()
    lo_idx = int((alpha / 2.0) * n_boot)
    hi_idx = int((1.0 - alpha / 2.0) * n_boot) - 1
    lo_idx = max(0, min(lo_idx, n_boot - 1))
    hi_idx = max(0, min(hi_idx, n_boot - 1))
    return mean, float(means[lo_idx]), float(means[hi_idx])


def _pareto_labels(labels: Sequence[str], points: Sequence[tuple[float, float]], y_mode: str) -> list[str]:
    out = []
    for i, (x_i, y_i) in enumerate(points):
        dominated = False
        for j, (x_j, y_j) in enumerate(points):
            if i == j:
                continue
            if y_mode == "min":
                no_worse = (x_j >= x_i) and (y_j <= y_i)
                strictly_better = (x_j > x_i) or (y_j < y_i)
            else:
                no_worse = (x_j >= x_i) and (y_j >= y_i)
                strictly_better = (x_j > x_i) or (y_j > y_i)
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            out.append(labels[i])
    return out


@dataclass
class LabelData:
    label: str
    score_rows: list[dict]
    gen_rows: list[dict]
    prompt_hashes: list[str]
    prompt_tok_lens: list[int]
    response_tok_lens: list[int]


def _load_label_data(
    eval_root: str,
    label: str,
    tokenizer,
) -> LabelData:
    score_file = _single_jsonl_file(os.path.join(eval_root, "scores_armorm", label))
    score_rows = list(_iter_jsonl(score_file))

    gens_dir = os.path.join(eval_root, "gens_sft" if label == "sft" else f"gens_{label}")
    gen_file = _single_jsonl_file(gens_dir)
    gen_rows = list(_iter_jsonl(gen_file))

    if len(score_rows) != len(gen_rows):
        raise ValueError(
            f"Length mismatch for label={label}: scores={len(score_rows)} gens={len(gen_rows)}"
        )

    prompt_hashes = []
    prompt_tok_lens = []
    response_tok_lens = []
    for row in gen_rows:
        prompt_hashes.append(_prompt_hash(row))
        prompt = row.get("prompt", "") or ""
        response = row.get("response", "") or ""
        prompt_tok_lens.append(len(tokenizer(prompt, add_special_tokens=False)["input_ids"]))
        response_tok_lens.append(len(tokenizer(response, add_special_tokens=False)["input_ids"]))

    return LabelData(
        label=label,
        score_rows=score_rows,
        gen_rows=gen_rows,
        prompt_hashes=prompt_hashes,
        prompt_tok_lens=prompt_tok_lens,
        response_tok_lens=response_tok_lens,
    )


def _default_labels(eval_root: str) -> list[str]:
    score_root = os.path.join(eval_root, "scores_armorm")
    labels = [d for d in sorted(os.listdir(score_root)) if os.path.isdir(os.path.join(score_root, d))]
    preferred = ["sft", "modpo_w0.1", "modpo_w0.2", "modpo_w0.4", "modpo_w0.6", "modpo_w0.8", "modpo_w0.9"]
    return [l for l in preferred if l in labels] + [l for l in labels if l not in preferred]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze HelpSteer 2D objective geometry from eval artifacts.")
    parser.add_argument("--eval_root", required=True, help="Eval root containing gens_* and scores_armorm/.")
    parser.add_argument("--tokenizer_path", required=True, help="Tokenizer path used for token-length diagnostics.")
    parser.add_argument("--x_key", required=True, help="Score key for x objective (maximize).")
    parser.add_argument(
        "--y_key",
        required=True,
        help="Score key for y objective (maximize if --y_mode=max, minimize if --y_mode=min).",
    )
    parser.add_argument("--y_mode", choices=["max", "min"], required=True)
    parser.add_argument("--max_input_length", type=int, default=1536)
    parser.add_argument("--max_new_tokens", type=int, default=2560)
    parser.add_argument("--labels", nargs="*", default=None, help="Optional explicit labels.")
    parser.add_argument("--reference_label", default="sft", help="Reference label for paired deltas.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mean_bootstrap", type=int, default=1000)
    parser.add_argument("--paired_bootstrap", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output_dir", required=True, help="Directory for JSON/CSV/MD outputs.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    labels = args.labels if args.labels else _default_labels(args.eval_root)
    if args.reference_label not in labels:
        raise ValueError(f"reference_label={args.reference_label!r} not in labels={labels}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    label_data: Dict[str, LabelData] = {label: _load_label_data(args.eval_root, label, tokenizer) for label in labels}

    ref_hashes = label_data[args.reference_label].prompt_hashes
    alignment_ok = True
    mismatches: dict[str, bool] = {}
    for label in labels:
        same = label_data[label].prompt_hashes == ref_hashes
        alignment_ok = alignment_ok and same
        if not same:
            mismatches[label] = True

    n_total = len(ref_hashes)
    uncapped_indices = []
    prompt_truncated_flags = []
    for i in range(n_total):
        prompt_truncated = any(label_data[label].prompt_tok_lens[i] > args.max_input_length for label in labels)
        capped_any = any(label_data[label].response_tok_lens[i] >= args.max_new_tokens for label in labels)
        prompt_truncated_flags.append(prompt_truncated)
        if (not prompt_truncated) and (not capped_any):
            uncapped_indices.append(i)

    capping = {}
    for label in labels:
        resp_lens = label_data[label].response_tok_lens
        cap_rate = sum(v >= args.max_new_tokens for v in resp_lens) / len(resp_lens)
        prompt_trunc_rate = sum(prompt_truncated_flags) / len(prompt_truncated_flags)
        capping[label] = {
            "n": len(resp_lens),
            "prompt_trunc_rate": prompt_trunc_rate,
            "cap_rate_toklen_ge_max_new_tokens": cap_rate,
            "mean_response_tok": float(statistics.mean(resp_lens)),
            "max_response_tok": max(resp_lens),
        }

    means_full: dict[str, dict[str, float]] = {}
    means_uncapped: dict[str, dict[str, float]] = {}
    ci_full: dict[str, dict[str, dict[str, float]]] = {}
    all_keys = sorted(
        {
            k
            for label in labels
            for row in label_data[label].score_rows[:1]
            for k in row.get("scores", {}).keys()
        }
    )
    for label_i, label in enumerate(labels):
        rows = label_data[label].score_rows
        means_full[label] = {}
        means_uncapped[label] = {}
        ci_full[label] = {}
        for key_i, key in enumerate(all_keys):
            vals = [float(row["scores"][key]) for row in rows if key in row.get("scores", {})]
            if vals:
                m, lo, hi = _bootstrap_ci_mean(
                    vals,
                    seed=args.seed + label_i * 101 + key_i,
                    n_boot=args.mean_bootstrap,
                    alpha=args.alpha,
                )
                means_full[label][key] = m
                ci_full[label][key] = {"mean": m, "ci_low": lo, "ci_high": hi, "n": len(vals)}
            vals_unc = [
                float(rows[idx]["scores"][key])
                for idx in uncapped_indices
                if key in rows[idx].get("scores", {})
            ]
            means_uncapped[label][key] = float(statistics.mean(vals_unc)) if vals_unc else float("nan")

    points_full = [(means_full[label][args.x_key], means_full[label][args.y_key]) for label in labels]
    points_uncapped = [(means_uncapped[label][args.x_key], means_uncapped[label][args.y_key]) for label in labels]
    pareto_full = _pareto_labels(labels, points_full, args.y_mode)
    pareto_uncapped = _pareto_labels(labels, points_uncapped, args.y_mode)

    reference_rows = label_data[args.reference_label].score_rows
    paired_vs_reference = {}
    for label_i, label in enumerate(labels):
        if label == args.reference_label:
            continue
        rows = label_data[label].score_rows
        dx = [
            float(rows[idx]["scores"][args.x_key]) - float(reference_rows[idx]["scores"][args.x_key])
            for idx in uncapped_indices
        ]
        dy = [
            float(rows[idx]["scores"][args.y_key]) - float(reference_rows[idx]["scores"][args.y_key])
            for idx in uncapped_indices
        ]
        dx_m, dx_lo, dx_hi = _bootstrap_ci_mean(
            dx,
            seed=args.seed + 10000 + label_i,
            n_boot=args.paired_bootstrap,
            alpha=args.alpha,
        )
        dy_m, dy_lo, dy_hi = _bootstrap_ci_mean(
            dy,
            seed=args.seed + 20000 + label_i,
            n_boot=args.paired_bootstrap,
            alpha=args.alpha,
        )
        paired_vs_reference[label] = {
            "d_x_mean": dx_m,
            "d_x_ci_low": dx_lo,
            "d_x_ci_high": dx_hi,
            "d_y_mean": dy_m,
            "d_y_ci_low": dy_lo,
            "d_y_ci_high": dy_hi,
        }

    result = {
        "config": {
            "eval_root": args.eval_root,
            "tokenizer_path": args.tokenizer_path,
            "x_key": args.x_key,
            "y_key": args.y_key,
            "y_mode": args.y_mode,
            "max_input_length": args.max_input_length,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "mean_bootstrap": args.mean_bootstrap,
            "paired_bootstrap": args.paired_bootstrap,
            "alpha": args.alpha,
        },
        "labels": labels,
        "alignment": {
            "ok": alignment_ok,
            "reference": args.reference_label,
            "n_reference": n_total,
            "mismatches": mismatches,
        },
        "n_total": n_total,
        "uncapped_intersection_n": len(uncapped_indices),
        "capping": capping,
        "means_full": means_full,
        "means_uncapped_intersection": means_uncapped,
        "ci_full": ci_full,
        "pareto_full": pareto_full,
        "pareto_uncapped_intersection": pareto_uncapped,
        "paired_vs_reference_uncapped": paired_vs_reference,
    }

    safe_x = args.x_key.replace("armorm_", "").replace("-", "_")
    safe_y = args.y_key.replace("armorm_", "").replace("-", "_")
    json_path = os.path.join(args.output_dir, f"analysis_2d_{safe_x}_vs_{safe_y}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    pareto_json_path = os.path.join(args.output_dir, f"pareto_2d_{safe_x}_vs_{safe_y}.json")
    with open(pareto_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "labels": labels,
                "x_key": args.x_key,
                "y_key": args.y_key,
                "y_mode": args.y_mode,
                "pareto_full": pareto_full,
                "pareto_uncapped_intersection": pareto_uncapped,
            },
            f,
            indent=2,
        )

    pareto_csv_path = os.path.join(args.output_dir, f"pareto_2d_{safe_x}_vs_{safe_y}.csv")
    with open(pareto_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "x_full", "y_full", "x_uncapped", "y_uncapped", "is_pareto_full", "is_pareto_uncapped"])
        for label in labels:
            writer.writerow(
                [
                    label,
                    means_full[label].get(args.x_key),
                    means_full[label].get(args.y_key),
                    means_uncapped[label].get(args.x_key),
                    means_uncapped[label].get(args.y_key),
                    label in pareto_full,
                    label in pareto_uncapped,
                ]
            )

    md_path = os.path.join(args.output_dir, f"analysis_2d_report_{safe_x}_vs_{safe_y}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# 2D Analysis Report: {args.x_key} vs {args.y_key}\n\n")
        f.write(f"- labels: {labels}\n")
        f.write(f"- n_total: {n_total}\n")
        f.write(f"- uncapped_intersection_n: {len(uncapped_indices)}\n")
        f.write(f"- alignment_ok: {alignment_ok}\n")
        f.write(f"- pareto_full: {pareto_full}\n")
        f.write(f"- pareto_uncapped_intersection: {pareto_uncapped}\n\n")
        f.write("## Full-set means\n\n")
        f.write("| label | x | y |\n|---|---:|---:|\n")
        for label in labels:
            f.write(
                f"| {label} | {means_full[label].get(args.x_key, float('nan')):.6f} | "
                f"{means_full[label].get(args.y_key, float('nan')):.6f} |\n"
            )
        f.write("\n## Uncapped-intersection means\n\n")
        f.write("| label | x | y |\n|---|---:|---:|\n")
        for label in labels:
            f.write(
                f"| {label} | {means_uncapped[label].get(args.x_key, float('nan')):.6f} | "
                f"{means_uncapped[label].get(args.y_key, float('nan')):.6f} |\n"
            )

    print(f"Wrote {json_path}")
    print(f"Wrote {pareto_json_path}")
    print(f"Wrote {pareto_csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()

