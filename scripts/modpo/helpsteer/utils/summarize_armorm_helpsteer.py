import argparse
import glob
import json
import os
import random
import statistics
from dataclasses import dataclass
from typing import Iterable, Optional


HELPSTEER_KEYS = [
    "armorm_helpsteer-helpfulness",
    "armorm_helpsteer-correctness",
    "armorm_helpsteer-coherence",
    "armorm_helpsteer-complexity",
    "armorm_helpsteer-verbosity",
]


@dataclass(frozen=True)
class Summary:
    n: int
    mean: float
    std: float
    ci_low: float
    ci_high: float


def _iter_jsonl(path_or_dir: str) -> Iterable[dict]:
    if os.path.isdir(path_or_dir):
        for path in sorted(glob.glob(os.path.join(path_or_dir, "*.jsonl"))):
            yield from _iter_jsonl(path)
        return
    with open(path_or_dir, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _bootstrap_ci_mean(values: list[float], seed: int, n_boot: int, alpha: float) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.mean(sample))
    means.sort()
    lo_idx = int((alpha / 2) * n_boot)
    hi_idx = int((1 - alpha / 2) * n_boot) - 1
    lo_idx = max(0, min(lo_idx, n_boot - 1))
    hi_idx = max(0, min(hi_idx, n_boot - 1))
    return float(means[lo_idx]), float(means[hi_idx])


def _summarize(values: list[float], seed: int, n_boot: int, alpha: float) -> Summary:
    mean = float(statistics.mean(values)) if values else float("nan")
    std = float(statistics.stdev(values)) if len(values) > 1 else 0.0
    ci_low, ci_high = _bootstrap_ci_mean(values, seed=seed, n_boot=n_boot, alpha=alpha)
    return Summary(n=len(values), mean=mean, std=std, ci_low=ci_low, ci_high=ci_high)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize ArmoRM HelpSteer dimension scores from score_armorm.py outputs "
        "(expects per-record 'scores' dict containing armorm_helpsteer-* keys)."
    )
    parser.add_argument(
        "--scores_path",
        action="append",
        required=True,
        help="Path to scores_armorm.jsonl OR a directory containing it. Repeat for multiple models.",
    )
    parser.add_argument("--label", action="append", help="Optional label(s) corresponding to --scores_path.")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap samples for CI on the mean.")
    parser.add_argument("--alpha", type=float, default=0.05, help="CI alpha (0.05 => 95%% CI).")
    args = parser.parse_args()

    labels: Optional[list[str]] = args.label
    if labels is not None and len(labels) != len(args.scores_path):
        raise ValueError("If provided, --label must be repeated exactly as many times as --scores_path.")
    if labels is None:
        labels = [os.path.basename(p.rstrip("/")) for p in args.scores_path]

    print("=== ArmoRM HelpSteer summary ===")
    print(f"keys={','.join(k.replace('armorm_helpsteer-','') for k in HELPSTEER_KEYS)}")
    print(f"bootstrap={args.bootstrap} alpha={args.alpha}")

    for label, path in zip(labels, args.scores_path):
        values_by_key = {k: [] for k in HELPSTEER_KEYS}
        missing = {k: 0 for k in HELPSTEER_KEYS}
        n_total = 0
        for obj in _iter_jsonl(path):
            n_total += 1
            scores = obj.get("scores", {})
            for k in HELPSTEER_KEYS:
                if k not in scores:
                    missing[k] += 1
                    continue
                values_by_key[k].append(float(scores[k]))
            if args.max_examples is not None and n_total >= args.max_examples:
                break

        print(f"\n[{label}] n_total={n_total}")
        for k in HELPSTEER_KEYS:
            vals = values_by_key[k]
            summ = _summarize(vals, seed=args.seed, n_boot=args.bootstrap, alpha=args.alpha)
            miss = missing[k]
            name = k.replace("armorm_helpsteer-", "")
            print(
                f"  {name:<12} n={summ.n:<4} missing={miss:<4} "
                f"mean={summ.mean:.4f} std={summ.std:.4f} "
                f"ci=[{summ.ci_low:.4f},{summ.ci_high:.4f}]"
            )


if __name__ == "__main__":
    main()

