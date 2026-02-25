"""RQ3 evaluation: fine-grid vs coarse-grid vs LoRA interpolation.

Three-way comparison of inference-time model selection strategies:
  1. Coarse-grid selection (nearest from {0.0, 0.5, 1.0})
  2. Fine-grid   selection (nearest from {0.0, 0.2, 0.4, 0.6, 0.8, 1.0})
  3. LoRA interpolation  (weighted blend of two nearest fine-grid adapters)

For each target w* and each dimension (reward, cost), computes per-prompt
win-rates with Wilson 95% CIs.

Usage:
    PYTHONPATH=. python scripts/eval/eval_rq3.py \
        --eval_root outputs/rq3/eval \
        --output_path outputs/rq3/eval/rq3_tier1_results.json
"""

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import tyro


# ---------------------------------------------------------------------------
# Grid definitions (must match training + plan_rq3_merged_2026-02-23.md)
# ---------------------------------------------------------------------------
FINE_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
COARSE_GRID = [0.0, 0.5, 1.0]

# All candidate targets where at least two methods differ
ALL_TARGETS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Interpolation bracket definitions: target -> (lower_w, upper_w)
# Only for off-fine-grid targets; on-grid targets use the exact model.
INTERP_BRACKETS = {
    0.1: (0.0, 0.2),
    0.3: (0.2, 0.4),
    0.5: (0.4, 0.6),
    0.7: (0.6, 0.8),
    0.9: (0.8, 1.0),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def nearest_selection(target: float, grid: List[float], tie_break: str = "lower") -> float:
    """Select the nearest grid point. Lower wins ties (pre-registered)."""
    candidates = sorted(grid)
    best = candidates[0]
    best_dist = abs(target - best)
    for w in candidates[1:]:
        dist = abs(target - w)
        if dist < best_dist:
            best = w
            best_dist = dist
        elif dist == best_dist and tie_break == "upper":
            best = w  # override to upper on tie
    return best


def wilson_ci(wins: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score 95% confidence interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p_hat = wins / n
    denom = 1 + z ** 2 / n
    center = (p_hat + z ** 2 / (2 * n)) / denom
    spread = z * math.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2)) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))


def weight_to_dir(w: float) -> str:
    """Map a weight value to the eval directory name."""
    return f"modpo_w{w:.1f}"


def interp_to_dir(target: float) -> str:
    """Map an interpolation target to the eval directory name."""
    return f"interp_w{target:.1f}"


def load_scores(eval_root: str, subdir: str) -> List[Dict]:
    """Load raw.jsonl from eval_root/subdir/score/raw.jsonl."""
    jsonl_path = os.path.join(eval_root, subdir, "score", "raw.jsonl")
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Score file not found: {jsonl_path}")
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_pairwise(
    scores_a: List[Dict],
    scores_b: List[Dict],
    label_a: str,
    label_b: str,
) -> Dict:
    """Compute per-prompt pairwise win-rates for reward (higher=better) and cost (lower=better)."""
    n = min(len(scores_a), len(scores_b))
    if len(scores_a) != len(scores_b):
        print(f"  WARNING: sample count mismatch ({len(scores_a)} vs {len(scores_b)}), using min={n}")

    reward_wins, cost_wins = 0, 0
    reward_ties, cost_ties = 0, 0
    for i in range(n):
        ra, rb = scores_a[i]["reward"], scores_b[i]["reward"]
        ca, cb = scores_a[i]["cost"], scores_b[i]["cost"]
        # Reward: higher is better for model A
        if ra > rb:
            reward_wins += 1
        elif ra == rb:
            reward_ties += 1
        # Cost: lower is better for model A
        if ca < cb:
            cost_wins += 1
        elif ca == cb:
            cost_ties += 1

    r_wr = reward_wins / n if n > 0 else 0.0
    c_wr = cost_wins / n if n > 0 else 0.0
    r_ci = wilson_ci(reward_wins, n)
    c_ci = wilson_ci(cost_wins, n)

    return {
        "model_a": label_a,
        "model_b": label_b,
        "n": n,
        "reward": {
            "wins_a": reward_wins,
            "ties": reward_ties,
            "win_rate": round(r_wr, 4),
            "ci_lower": round(r_ci[0], 4),
            "ci_upper": round(r_ci[1], 4),
            "significant": r_ci[0] > 0.5 or r_ci[1] < 0.5,
            "favors": label_a if r_ci[0] > 0.5 else (label_b if r_ci[1] < 0.5 else "neither"),
        },
        "cost": {
            "wins_a": cost_wins,
            "ties": cost_ties,
            "win_rate": round(c_wr, 4),
            "ci_lower": round(c_ci[0], 4),
            "ci_upper": round(c_ci[1], 4),
            "significant": c_ci[0] > 0.5 or c_ci[1] < 0.5,
            "favors": label_a if c_ci[0] > 0.5 else (label_b if c_ci[1] < 0.5 else "neither"),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@dataclass
class Args:
    eval_root: str = field(metadata={"help": "root eval directory (contains modpo_w*/score/ and interp_w*/score/)"})
    output_path: str = field(default="rq3_tier1_results.json", metadata={"help": "output JSON path"})
    tie_break: str = field(default="lower", metadata={"help": "tie-break rule: lower or upper"})


if __name__ == "__main__":
    args = tyro.cli(Args)

    results = {"metadata": {}, "tier0": [], "tier1": [], "summary": {}}
    results["metadata"] = {
        "fine_grid": FINE_GRID,
        "coarse_grid": COARSE_GRID,
        "tie_break": args.tie_break,
        "eval_root": args.eval_root,
    }

    # ------------------------------------------------------------------
    # Selection map
    # ------------------------------------------------------------------
    selection_map = []
    for t in ALL_TARGETS:
        fine_sel = nearest_selection(t, FINE_GRID, args.tie_break)
        coarse_sel = nearest_selection(t, COARSE_GRID, args.tie_break)
        has_interp = t in INTERP_BRACKETS
        entry = {
            "target": t,
            "fine_selection": fine_sel,
            "coarse_selection": coarse_sel,
            "grids_differ": abs(fine_sel - coarse_sel) > 1e-9,
            "has_interp": has_interp,
        }
        if has_interp:
            entry["interp_lower"], entry["interp_upper"] = INTERP_BRACKETS[t]
        selection_map.append(entry)
    results["metadata"]["selection_map"] = selection_map

    print("=" * 70)
    print("RQ3 Evaluation — Three-Way Comparison")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Tier 0: fine-grid vs coarse-grid (8 informative targets)
    # ------------------------------------------------------------------
    print("\n--- Tier 0: Fine-grid vs Coarse-grid ---")
    tier0_sig_reward, tier0_sig_cost = 0, 0
    for entry in selection_map:
        t = entry["target"]
        if not entry["grids_differ"]:
            print(f"  w*={t:.1f}: same selection ({entry['fine_selection']:.1f}), skipping")
            continue

        fine_dir = weight_to_dir(entry["fine_selection"])
        coarse_dir = weight_to_dir(entry["coarse_selection"])

        try:
            fine_scores = load_scores(args.eval_root, fine_dir)
            coarse_scores = load_scores(args.eval_root, coarse_dir)
        except FileNotFoundError as e:
            print(f"  w*={t:.1f}: SKIP — {e}")
            continue

        pw = compute_pairwise(fine_scores, coarse_scores, f"fine({entry['fine_selection']:.1f})", f"coarse({entry['coarse_selection']:.1f})")
        pw["target"] = t
        results["tier0"].append(pw)

        sig_r = "***" if pw["reward"]["significant"] else ""
        sig_c = "***" if pw["cost"]["significant"] else ""
        if pw["reward"]["significant"]:
            tier0_sig_reward += 1
        if pw["cost"]["significant"]:
            tier0_sig_cost += 1
        print(f"  w*={t:.1f}  fine={entry['fine_selection']:.1f} vs coarse={entry['coarse_selection']:.1f}"
              f"  reward WR={pw['reward']['win_rate']:.3f} [{pw['reward']['ci_lower']:.3f},{pw['reward']['ci_upper']:.3f}]{sig_r}"
              f"  cost WR={pw['cost']['win_rate']:.3f} [{pw['cost']['ci_lower']:.3f},{pw['cost']['ci_upper']:.3f}]{sig_c}")

    results["summary"]["tier0_sig_reward"] = tier0_sig_reward
    results["summary"]["tier0_sig_cost"] = tier0_sig_cost
    results["summary"]["tier0_total_targets"] = len(results["tier0"])

    # ------------------------------------------------------------------
    # Tier 1: interpolation comparisons
    # ------------------------------------------------------------------
    print("\n--- Tier 1: Interpolation Comparisons ---")
    tier1_interp_vs_fine = []
    tier1_interp_vs_coarse = []
    n_interp_beats_fine_reward, n_interp_beats_fine_cost = 0, 0

    for entry in selection_map:
        t = entry["target"]
        if not entry["has_interp"]:
            continue

        interp_dir = interp_to_dir(t)
        fine_dir = weight_to_dir(entry["fine_selection"])
        coarse_dir = weight_to_dir(entry["coarse_selection"])

        # Check if interpolated scores exist
        interp_score_path = os.path.join(args.eval_root, interp_dir, "score", "raw.jsonl")
        if not os.path.exists(interp_score_path):
            print(f"  w*={t:.1f}: SKIP — interpolated scores not found at {interp_score_path}")
            continue

        interp_scores = load_scores(args.eval_root, interp_dir)
        fine_scores = load_scores(args.eval_root, fine_dir)

        # Interp vs Fine
        pw_if = compute_pairwise(interp_scores, fine_scores, f"interp({t:.1f})", f"fine({entry['fine_selection']:.1f})")
        pw_if["target"] = t
        tier1_interp_vs_fine.append(pw_if)

        sig_r = "***" if pw_if["reward"]["significant"] else ""
        sig_c = "***" if pw_if["cost"]["significant"] else ""
        if pw_if["reward"]["significant"] and pw_if["reward"]["favors"].startswith("interp"):
            n_interp_beats_fine_reward += 1
        if pw_if["cost"]["significant"] and pw_if["cost"]["favors"].startswith("interp"):
            n_interp_beats_fine_cost += 1
        print(f"  w*={t:.1f}  interp vs fine({entry['fine_selection']:.1f})"
              f"  reward WR={pw_if['reward']['win_rate']:.3f} [{pw_if['reward']['ci_lower']:.3f},{pw_if['reward']['ci_upper']:.3f}]{sig_r}"
              f"  cost WR={pw_if['cost']['win_rate']:.3f} [{pw_if['cost']['ci_lower']:.3f},{pw_if['cost']['ci_upper']:.3f}]{sig_c}")

        # Interp vs Coarse (only if grids differ)
        if entry["grids_differ"]:
            coarse_scores = load_scores(args.eval_root, coarse_dir)
            pw_ic = compute_pairwise(interp_scores, coarse_scores, f"interp({t:.1f})", f"coarse({entry['coarse_selection']:.1f})")
            pw_ic["target"] = t
            tier1_interp_vs_coarse.append(pw_ic)

            sig_r = "***" if pw_ic["reward"]["significant"] else ""
            sig_c = "***" if pw_ic["cost"]["significant"] else ""
            print(f"  w*={t:.1f}  interp vs coarse({entry['coarse_selection']:.1f})"
                  f"  reward WR={pw_ic['reward']['win_rate']:.3f} [{pw_ic['reward']['ci_lower']:.3f},{pw_ic['reward']['ci_upper']:.3f}]{sig_r}"
                  f"  cost WR={pw_ic['cost']['win_rate']:.3f} [{pw_ic['cost']['ci_lower']:.3f},{pw_ic['cost']['ci_upper']:.3f}]{sig_c}")

    results["tier1"] = {
        "interp_vs_fine": tier1_interp_vs_fine,
        "interp_vs_coarse": tier1_interp_vs_coarse,
    }
    results["summary"]["tier1_interp_beats_fine_reward"] = n_interp_beats_fine_reward
    results["summary"]["tier1_interp_beats_fine_cost"] = n_interp_beats_fine_cost
    results["summary"]["tier1_interp_targets_evaluated"] = len(tier1_interp_vs_fine)

    # ------------------------------------------------------------------
    # Compute mean scores per model for sanity checking
    # ------------------------------------------------------------------
    print("\n--- Mean Scores (sanity check) ---")
    all_dirs = set()
    for entry in selection_map:
        all_dirs.add(weight_to_dir(entry["fine_selection"]))
        all_dirs.add(weight_to_dir(entry["coarse_selection"]))
        if entry["has_interp"]:
            all_dirs.add(interp_to_dir(entry["target"]))

    mean_scores = {}
    for d in sorted(all_dirs):
        try:
            scores = load_scores(args.eval_root, d)
            mean_r = sum(s["reward"] for s in scores) / len(scores)
            mean_c = sum(s["cost"] for s in scores) / len(scores)
            mean_scores[d] = {"mean_reward": round(mean_r, 4), "mean_cost": round(mean_c, 4), "n": len(scores)}
            print(f"  {d:20s}  reward={mean_r:+.4f}  cost={mean_c:+.4f}  n={len(scores)}")
        except FileNotFoundError:
            print(f"  {d:20s}  NOT FOUND")
    results["mean_scores"] = mean_scores

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_path}")

    # ------------------------------------------------------------------
    # Print markdown summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nTier 0 (fine vs coarse):")
    print(f"  Reward: fine-grid significant on {tier0_sig_reward}/{len(results['tier0'])} targets")
    print(f"  Cost:   fine-grid significant on {tier0_sig_cost}/{len(results['tier0'])} targets")
    print(f"\nTier 1 (interpolation):")
    print(f"  Interp beats fine-grid (reward): {n_interp_beats_fine_reward}/{len(tier1_interp_vs_fine)} targets")
    print(f"  Interp beats fine-grid (cost):   {n_interp_beats_fine_cost}/{len(tier1_interp_vs_fine)} targets")
    if tier1_interp_vs_fine:
        mean_wr_r = sum(p["reward"]["win_rate"] for p in tier1_interp_vs_fine) / len(tier1_interp_vs_fine)
        mean_wr_c = sum(p["cost"]["win_rate"] for p in tier1_interp_vs_fine) / len(tier1_interp_vs_fine)
        print(f"  Mean interp-vs-fine reward WR: {mean_wr_r:.3f}")
        print(f"  Mean interp-vs-fine cost WR:   {mean_wr_c:.3f}")
