#!/usr/bin/env python3
"""
Compare two HelpSteer eval roots with paired bootstrap deltas.

Outputs:
- JSON summary with per-label paired deltas (new-old) and within-new deltas vs SFT.
- CSV tables for quick inspection.
- Optional thesis-freeze markdown summary.
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
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


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


@dataclass(frozen=True)
class LabelScores:
    label: str
    keys_in_order: list[str]
    values: Dict[str, tuple[float, float]]


def _default_labels(eval_root: str) -> list[str]:
    score_root = os.path.join(eval_root, "scores_armorm")
    labels = [d for d in sorted(os.listdir(score_root)) if os.path.isdir(os.path.join(score_root, d))]
    preferred = ["sft", "modpo_w0.1", "modpo_w0.2", "modpo_w0.4", "modpo_w0.6", "modpo_w0.8", "modpo_w0.9"]
    return [l for l in preferred if l in labels] + [l for l in labels if l not in preferred]


def _load_label_scores(eval_root: str, label: str, x_key: str, y_key: str) -> LabelScores:
    score_file = _single_jsonl_file(os.path.join(eval_root, "scores_armorm", label))
    gen_file = _single_jsonl_file(os.path.join(eval_root, "gens_sft" if label == "sft" else f"gens_{label}"))
    score_rows = list(_iter_jsonl(score_file))
    gen_rows = list(_iter_jsonl(gen_file))
    if len(score_rows) != len(gen_rows):
        raise ValueError(
            f"Length mismatch for label={label}: scores={len(score_rows)} gens={len(gen_rows)}"
        )

    values: Dict[str, tuple[float, float]] = {}
    keys_in_order: list[str] = []
    seen_hash_counts: dict[str, int] = defaultdict(int)

    for g, s in zip(gen_rows, score_rows):
        h = _prompt_hash(g)
        idx = seen_hash_counts[h]
        seen_hash_counts[h] += 1
        pair_key = f"{h}:{idx}"

        scores = s.get("scores", {})
        if x_key not in scores or y_key not in scores:
            raise KeyError(f"Missing keys for label={label}: need {x_key} and {y_key}")
        values[pair_key] = (float(scores[x_key]), float(scores[y_key]))
        keys_in_order.append(pair_key)

    return LabelScores(label=label, keys_in_order=keys_in_order, values=values)


def _paired_deltas(
    left: LabelScores,
    right: LabelScores,
    seed: int,
    n_boot: int,
    alpha: float,
) -> dict:
    common = [k for k in left.keys_in_order if k in right.values]
    dx = [left.values[k][0] - right.values[k][0] for k in common]
    dy = [left.values[k][1] - right.values[k][1] for k in common]

    dx_m, dx_lo, dx_hi = _bootstrap_ci_mean(dx, seed=seed, n_boot=n_boot, alpha=alpha)
    dy_m, dy_lo, dy_hi = _bootstrap_ci_mean(dy, seed=seed + 10000, n_boot=n_boot, alpha=alpha)

    left_x = [left.values[k][0] for k in common]
    left_y = [left.values[k][1] for k in common]
    right_x = [right.values[k][0] for k in common]
    right_y = [right.values[k][1] for k in common]

    return {
        "n_common": len(common),
        "left_mean_x": float(statistics.mean(left_x)) if left_x else float("nan"),
        "left_mean_y": float(statistics.mean(left_y)) if left_y else float("nan"),
        "right_mean_x": float(statistics.mean(right_x)) if right_x else float("nan"),
        "right_mean_y": float(statistics.mean(right_y)) if right_y else float("nan"),
        "delta_x_mean": dx_m,
        "delta_x_ci_low": dx_lo,
        "delta_x_ci_high": dx_hi,
        "delta_y_mean": dy_m,
        "delta_y_ci_low": dy_lo,
        "delta_y_ci_high": dy_hi,
    }


def _load_json(path: str | None) -> dict | None:
    if not path:
        return None
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired bootstrap comparison for HelpSteer eval roots.")
    parser.add_argument("--new_eval_root", required=True, help="New run eval root containing gens_* and scores_armorm/.")
    parser.add_argument("--old_eval_root", required=True, help="Baseline/previous run eval root.")
    parser.add_argument("--x_key", default="armorm_helpsteer-helpfulness")
    parser.add_argument("--y_key", default="armorm_helpsteer-verbosity")
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--reference_label", default="sft")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--new_tag", default="new")
    parser.add_argument("--old_tag", default="old")
    parser.add_argument("--new_analysis_json", default=None)
    parser.add_argument("--old_analysis_json", default=None)
    parser.add_argument("--freeze_md_name", default="thesis_freeze_helpsteer_seed_comparison.md")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    labels = args.labels if args.labels else _default_labels(args.new_eval_root)
    if args.reference_label not in labels:
        raise ValueError(f"reference_label={args.reference_label!r} not in labels={labels}")

    new_scores = {
        label: _load_label_scores(args.new_eval_root, label, args.x_key, args.y_key)
        for label in labels
    }
    old_scores = {
        label: _load_label_scores(args.old_eval_root, label, args.x_key, args.y_key)
        for label in labels
    }

    cross_seed = {}
    for i, label in enumerate(labels):
        cross_seed[label] = _paired_deltas(
            left=new_scores[label],
            right=old_scores[label],
            seed=args.seed + i * 137,
            n_boot=args.bootstrap,
            alpha=args.alpha,
        )

    within_new_vs_ref = {}
    ref = new_scores[args.reference_label]
    for i, label in enumerate(labels):
        if label == args.reference_label:
            continue
        within_new_vs_ref[label] = _paired_deltas(
            left=new_scores[label],
            right=ref,
            seed=args.seed + 50000 + i * 173,
            n_boot=args.bootstrap,
            alpha=args.alpha,
        )

    out = {
        "config": {
            "new_eval_root": args.new_eval_root,
            "old_eval_root": args.old_eval_root,
            "x_key": args.x_key,
            "y_key": args.y_key,
            "labels": labels,
            "reference_label": args.reference_label,
            "bootstrap": args.bootstrap,
            "alpha": args.alpha,
            "seed": args.seed,
            "new_tag": args.new_tag,
            "old_tag": args.old_tag,
        },
        "cross_seed_new_minus_old": cross_seed,
        "within_new_vs_reference": within_new_vs_ref,
    }

    json_path = os.path.join(args.output_dir, "paired_bootstrap_helpsteer_seed_comparison.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    cross_csv = os.path.join(args.output_dir, "paired_bootstrap_helpsteer_seed_comparison.csv")
    with open(cross_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "label",
                "n_common",
                f"{args.new_tag}_mean_x",
                f"{args.new_tag}_mean_y",
                f"{args.old_tag}_mean_x",
                f"{args.old_tag}_mean_y",
                "delta_x_mean",
                "delta_x_ci_low",
                "delta_x_ci_high",
                "delta_y_mean",
                "delta_y_ci_low",
                "delta_y_ci_high",
            ]
        )
        for label in labels:
            d = cross_seed[label]
            w.writerow(
                [
                    label,
                    d["n_common"],
                    d["left_mean_x"],
                    d["left_mean_y"],
                    d["right_mean_x"],
                    d["right_mean_y"],
                    d["delta_x_mean"],
                    d["delta_x_ci_low"],
                    d["delta_x_ci_high"],
                    d["delta_y_mean"],
                    d["delta_y_ci_low"],
                    d["delta_y_ci_high"],
                ]
            )

    within_csv = os.path.join(args.output_dir, "paired_bootstrap_helpsteer_within_new_vs_sft.csv")
    with open(within_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "label",
                "n_common",
                "label_mean_x",
                "label_mean_y",
                "reference_mean_x",
                "reference_mean_y",
                "delta_x_mean",
                "delta_x_ci_low",
                "delta_x_ci_high",
                "delta_y_mean",
                "delta_y_ci_low",
                "delta_y_ci_high",
            ]
        )
        for label in labels:
            if label == args.reference_label:
                continue
            d = within_new_vs_ref[label]
            w.writerow(
                [
                    label,
                    d["n_common"],
                    d["left_mean_x"],
                    d["left_mean_y"],
                    d["right_mean_x"],
                    d["right_mean_y"],
                    d["delta_x_mean"],
                    d["delta_x_ci_low"],
                    d["delta_x_ci_high"],
                    d["delta_y_mean"],
                    d["delta_y_ci_low"],
                    d["delta_y_ci_high"],
                ]
            )

    new_analysis = _load_json(args.new_analysis_json)
    old_analysis = _load_json(args.old_analysis_json)
    freeze_md = os.path.join(args.output_dir, args.freeze_md_name)
    with open(freeze_md, "w", encoding="utf-8") as f:
        f.write("# HelpSteer Thesis Freeze: Seed Comparison\n\n")
        f.write(f"- x_key: `{args.x_key}` (maximize)\n")
        f.write(f"- y_key: `{args.y_key}` (minimize)\n")
        f.write(f"- new_eval_root: `{args.new_eval_root}`\n")
        f.write(f"- old_eval_root: `{args.old_eval_root}`\n")
        f.write("\n## Primary Claim (Full-Set Pareto)\n\n")
        if new_analysis and "pareto_full" in new_analysis:
            f.write(f"- `{args.new_tag}` pareto_full: {new_analysis['pareto_full']}\n")
        else:
            f.write("- Unknown from current context: missing new analysis JSON for pareto_full.\n")
        if old_analysis and "pareto_full" in old_analysis:
            f.write(f"- `{args.old_tag}` pareto_full: {old_analysis['pareto_full']}\n")
        else:
            f.write("- Unknown from current context: missing old analysis JSON for pareto_full.\n")

        f.write("\n## Diagnostic (Uncapped Intersection)\n\n")
        if new_analysis:
            f.write(f"- `{args.new_tag}` n_total: {new_analysis.get('n_total')}\n")
            f.write(f"- `{args.new_tag}` uncapped_intersection_n: {new_analysis.get('uncapped_intersection_n')}\n")
            f.write(
                f"- `{args.new_tag}` pareto_uncapped_intersection: "
                f"{new_analysis.get('pareto_uncapped_intersection')}\n"
            )
        else:
            f.write("- Unknown from current context: missing new analysis JSON for uncapped diagnostics.\n")

        f.write("\n## Cross-Seed Paired Deltas (new - old)\n\n")
        f.write("| label | n | delta_x_mean [CI] | delta_y_mean [CI] |\n")
        f.write("|---|---:|---:|---:|\n")
        for label in labels:
            d = cross_seed[label]
            f.write(
                f"| {label} | {d['n_common']} | "
                f"{d['delta_x_mean']:.6f} [{d['delta_x_ci_low']:.6f}, {d['delta_x_ci_high']:.6f}] | "
                f"{d['delta_y_mean']:.6f} [{d['delta_y_ci_low']:.6f}, {d['delta_y_ci_high']:.6f}] |\n"
            )

        f.write("\n## Within-New Paired Deltas vs SFT\n\n")
        f.write("| label | n | delta_x_mean [CI] | delta_y_mean [CI] |\n")
        f.write("|---|---:|---:|---:|\n")
        for label in labels:
            if label == args.reference_label:
                continue
            d = within_new_vs_ref[label]
            f.write(
                f"| {label} | {d['n_common']} | "
                f"{d['delta_x_mean']:.6f} [{d['delta_x_ci_low']:.6f}, {d['delta_x_ci_high']:.6f}] | "
                f"{d['delta_y_mean']:.6f} [{d['delta_y_ci_low']:.6f}, {d['delta_y_ci_high']:.6f}] |\n"
            )

    print(f"Wrote {json_path}")
    print(f"Wrote {cross_csv}")
    print(f"Wrote {within_csv}")
    print(f"Wrote {freeze_md}")


if __name__ == "__main__":
    main()
