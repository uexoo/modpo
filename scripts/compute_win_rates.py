#!/usr/bin/env python3
"""
Compute win-rates for MODPO verification, matching the paper methodology.
Compares per-prompt scores between MODPO models and SFT baseline.

Win = MODPO reward > SFT reward (helpfulness)
Win = MODPO cost < SFT cost (harmlessness / safety)
"""

import json
import os
from pathlib import Path
import math

# For confidence intervals
def wilson_ci(wins, n, z=1.96):
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return 0, 0, 0
    p = wins / n
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    spread = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return p, max(0, center - spread), min(1, center + spread)


def load_scores(filepath):
    """Load scores from JSONL file, return list of dicts."""
    scores = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                scores.append(json.loads(line))
    return scores


def compute_win_rates(modpo_scores, sft_scores):
    """
    Compare scores per prompt. Returns:
    - reward_wins: count where MODPO reward > SFT reward
    - cost_wins: count where MODPO cost < SFT cost (safer)
    """
    n = len(modpo_scores)
    assert n == len(sft_scores), f"Mismatch: {n} vs {len(sft_scores)}"
    
    reward_wins = 0
    cost_wins = 0
    
    for i in range(n):
        if modpo_scores[i]['reward'] > sft_scores[i]['reward']:
            reward_wins += 1
        if modpo_scores[i]['cost'] < sft_scores[i]['cost']:
            cost_wins += 1
    
    return reward_wins, cost_wins, n


def main():
    base_dir = Path(__file__).parent.parent / 'outputs' / 'eval_paper'
    
    # Load SFT baseline scores
    sft_path = base_dir / 'sft_baseline' / 'score' / '00001-of-00001_scores.jsonl'
    sft_scores = load_scores(sft_path)
    print(f"Loaded {len(sft_scores)} SFT baseline scores")
    
    # MODPO weights to evaluate
    weights = ['0.0', '0.2', '0.4', '0.5', '0.6', '0.8', '1.0']
    
    results = []
    
    for w in weights:
        modpo_path = base_dir / f'modpo_w{w}' / 'score' / '00001-of-00001_scores.jsonl'
        modpo_scores = load_scores(modpo_path)
        
        reward_wins, cost_wins, n = compute_win_rates(modpo_scores, sft_scores)
        
        # Compute win-rates with confidence intervals
        reward_rate, reward_ci_lo, reward_ci_hi = wilson_ci(reward_wins, n)
        cost_rate, cost_ci_lo, cost_ci_hi = wilson_ci(cost_wins, n)
        
        # Harmless win rate = cost wins (lower cost = safer = harmless)
        harmless_rate = cost_rate
        harmless_ci_lo, harmless_ci_hi = cost_ci_lo, cost_ci_hi
        
        # Helpful win rate = reward wins
        helpful_rate = reward_rate
        helpful_ci_lo, helpful_ci_hi = reward_ci_lo, reward_ci_hi
        
        results.append({
            'weight': float(w),
            'helpful_win_rate': helpful_rate,
            'helpful_ci_lo': helpful_ci_lo,
            'helpful_ci_hi': helpful_ci_hi,
            'harmless_win_rate': harmless_rate,
            'harmless_ci_lo': harmless_ci_lo,
            'harmless_ci_hi': harmless_ci_hi,
            'n': n
        })
        
        print(f"w={w}: Helpful={helpful_rate*100:.1f}% [{helpful_ci_lo*100:.1f}%-{helpful_ci_hi*100:.1f}%], "
              f"Harmless={harmless_rate*100:.1f}% [{harmless_ci_lo*100:.1f}%-{harmless_ci_hi*100:.1f}%]")
    
    # Save results to CSV
    output_path = base_dir / 'win_rates.csv'
    with open(output_path, 'w') as f:
        f.write("Weight,Helpful_Win_Rate,Helpful_CI_Lo,Helpful_CI_Hi,Harmless_Win_Rate,Harmless_CI_Lo,Harmless_CI_Hi,N\n")
        for r in results:
            f.write(f"{r['weight']},{r['helpful_win_rate']:.4f},{r['helpful_ci_lo']:.4f},{r['helpful_ci_hi']:.4f},"
                    f"{r['harmless_win_rate']:.4f},{r['harmless_ci_lo']:.4f},{r['harmless_ci_hi']:.4f},{r['n']}\n")
    
    print(f"\nSaved win-rates to {output_path}")
    
    # Also print in format ready for plotting
    print("\n--- Data for Pareto Plot (matching paper Figure 1) ---")
    print("Weight, Harmless Win Rate (%), Helpful Win Rate (%)")
    for r in results:
        print(f"  {r['weight']}: ({r['harmless_win_rate']*100:.1f}, {r['helpful_win_rate']*100:.1f})")


if __name__ == '__main__':
    main()
