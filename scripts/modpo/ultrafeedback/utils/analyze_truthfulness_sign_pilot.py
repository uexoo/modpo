"""Analyze UltraFeedback truthfulness sign-ablation pilot outputs.

This script compares sign variants (e.g., +0.1 vs -0.1) at matched w values
using paired prompt-level deltas and bootstrap CIs over ArmoRM scores.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple


TRUTH_KEY = "armorm_ultrafeedback-truthfulness"
HELP_KEY = "armorm_ultrafeedback-helpfulness"


def _sign_tag(sign: float) -> str:
    prefix = "pos" if sign >= 0 else "neg"
    mag = str(abs(sign)).replace(".", "p")
    return f"{prefix}{mag}"


def _label_for(sign: float, w: str) -> str:
    return f"modpo_sign{_sign_tag(sign)}_w{w}"


def _load_scores(path: str, score_key: str) -> List[float]:
    values: List[float] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            scores = obj.get("scores", {})
            if score_key not in scores:
                raise KeyError(f"Missing score key '{score_key}' in {path}")
            values.append(float(scores[score_key]))
    if not values:
        raise ValueError(f"No rows found in {path}")
    return values


def _bootstrap_ci(
    deltas: List[float],
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    if not deltas:
        raise ValueError("Cannot bootstrap an empty delta list.")
    rng = random.Random(seed)
    n = len(deltas)
    means: List[float] = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += deltas[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo_idx = max(0, int(math.floor((alpha / 2.0) * (n_boot - 1))))
    hi_idx = min(n_boot - 1, int(math.floor((1.0 - alpha / 2.0) * (n_boot - 1))))
    mean = sum(deltas) / n
    return mean, means[lo_idx], means[hi_idx]


@dataclass
class PerWResult:
    w: str
    n: int
    truth_delta_mean: float
    truth_delta_ci_lo: float
    truth_delta_ci_hi: float
    helpful_delta_mean: float
    helpful_delta_ci_lo: float
    helpful_delta_ci_hi: float


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze sign-ablation pilot scores for UF truthfulness.")
    parser.add_argument("--scores_root", required=True, help="Root dir containing per-label score dirs.")
    parser.add_argument("--w_values", nargs="+", required=True, help="List of w values used in pilot.")
    parser.add_argument("--pos_sign", type=float, default=0.1, help="Positive sign value used in pilot.")
    parser.add_argument("--neg_sign", type=float, default=-0.1, help="Negative sign value used in pilot.")
    parser.add_argument("--truth_key", default=TRUTH_KEY, help="Score key for truthfulness.")
    parser.add_argument("--help_key", default=HELP_KEY, help="Score key for helpfulness.")
    parser.add_argument("--bootstrap_iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    results: List[PerWResult] = []
    all_truth_pos = True
    all_truth_neg = True

    for wi, w in enumerate(args.w_values):
        pos_label = _label_for(args.pos_sign, w)
        neg_label = _label_for(args.neg_sign, w)
        pos_path = os.path.join(args.scores_root, pos_label, "scores_armorm.jsonl")
        neg_path = os.path.join(args.scores_root, neg_label, "scores_armorm.jsonl")
        if not os.path.exists(pos_path):
            raise FileNotFoundError(f"Missing positive-sign file: {pos_path}")
        if not os.path.exists(neg_path):
            raise FileNotFoundError(f"Missing negative-sign file: {neg_path}")

        pos_truth = _load_scores(pos_path, args.truth_key)
        neg_truth = _load_scores(neg_path, args.truth_key)
        pos_help = _load_scores(pos_path, args.help_key)
        neg_help = _load_scores(neg_path, args.help_key)

        if len(pos_truth) != len(neg_truth):
            raise RuntimeError(
                f"Length mismatch truth scores for w={w}: pos={len(pos_truth)} neg={len(neg_truth)}"
            )
        if len(pos_help) != len(neg_help):
            raise RuntimeError(
                f"Length mismatch helpful scores for w={w}: pos={len(pos_help)} neg={len(neg_help)}"
            )

        truth_deltas = [p - n for p, n in zip(pos_truth, neg_truth)]
        help_deltas = [p - n for p, n in zip(pos_help, neg_help)]

        truth_mean, truth_lo, truth_hi = _bootstrap_ci(
            truth_deltas, n_boot=args.bootstrap_iters, seed=args.seed + wi
        )
        help_mean, help_lo, help_hi = _bootstrap_ci(
            help_deltas, n_boot=args.bootstrap_iters, seed=args.seed + 100000 + wi
        )

        if truth_mean <= 0:
            all_truth_pos = False
        if truth_mean >= 0:
            all_truth_neg = False

        results.append(
            PerWResult(
                w=w,
                n=len(truth_deltas),
                truth_delta_mean=truth_mean,
                truth_delta_ci_lo=truth_lo,
                truth_delta_ci_hi=truth_hi,
                helpful_delta_mean=help_mean,
                helpful_delta_ci_lo=help_lo,
                helpful_delta_ci_hi=help_hi,
            )
        )

    if all_truth_pos:
        verdict = "positive_sign_improves_truthfulness_consistently"
        recommended_margin_beta_sign = "positive"
    elif all_truth_neg:
        verdict = "negative_sign_improves_truthfulness_consistently"
        recommended_margin_beta_sign = "negative"
    else:
        verdict = "inconclusive_or_mixed_sign_effect_stop_and_investigate"
        recommended_margin_beta_sign = "undetermined"

    payload: Dict[str, object] = {
        "scores_root": args.scores_root,
        "truth_key": args.truth_key,
        "help_key": args.help_key,
        "pos_sign": args.pos_sign,
        "neg_sign": args.neg_sign,
        "w_values": args.w_values,
        "bootstrap_iters": args.bootstrap_iters,
        "seed": args.seed,
        "results": [r.__dict__ for r in results],
        "verdict": verdict,
        "recommended_margin_beta_sign": recommended_margin_beta_sign,
    }

    print("=== Sign Ablation Summary (pos - neg) ===")
    for r in results:
        print(
            f"w={r.w} n={r.n} "
            f"delta_truth={r.truth_delta_mean:+.4f} "
            f"ci=[{r.truth_delta_ci_lo:+.4f},{r.truth_delta_ci_hi:+.4f}] "
            f"delta_help={r.helpful_delta_mean:+.4f} "
            f"ci=[{r.helpful_delta_ci_lo:+.4f},{r.helpful_delta_ci_hi:+.4f}]"
        )
    print(f"VERDICT: {verdict}")

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote_json={args.output_json}")

    if recommended_margin_beta_sign == "undetermined":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
