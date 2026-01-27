"""
Compute pairwise preference conflict ratios across UltraFeedback dimensions.

This script analyzes whether different dimensions in UltraFeedback actually
conflict with each other (i.e., prefer different responses for the same prompt).

Conflict ratio = # examples where dimensions disagree / # total examples

A low conflict ratio suggests the dimensions are correlated (no trade-off needed).
A high conflict ratio suggests genuine conflict (MODPO trade-off is meaningful).
"""

import json
from collections import defaultdict
from itertools import combinations
from datasets import load_dataset
import numpy as np

# UltraFeedback dimensions
DIMENSIONS = ["instruction_following", "honesty", "truthfulness", "helpfulness"]


def get_dimension_preference(completions: list, dimension: str) -> tuple[int, int] | None:
    """
    For a given dimension, return (best_idx, worst_idx) based on scores.
    Returns None if scores are tied or missing.
    """
    scores = []
    for i, comp in enumerate(completions):
        annotations = comp.get("annotations", {})
        dim_data = annotations.get(dimension)
        if dim_data is not None:
            # Rating is stored as string in a dict
            rating = dim_data.get("Rating") if isinstance(dim_data, dict) else None
            if rating is not None:
                try:
                    scores.append((i, int(rating)))
                except (ValueError, TypeError):
                    continue
    
    if len(scores) < 2:
        return None
    
    # Find best and worst
    scores.sort(key=lambda x: x[1], reverse=True)
    best_idx = scores[0][0]
    worst_idx = scores[-1][0]
    best_score = scores[0][1]
    worst_score = scores[-1][1]
    
    # If tied, no clear preference
    if best_score == worst_score:
        return None
    
    return (best_idx, worst_idx)


def compute_conflict_matrix(dataset, max_samples: int = 5000):
    """
    Compute pairwise conflict ratios between all dimension pairs.
    """
    conflict_counts = defaultdict(int)
    agree_counts = defaultdict(int)
    total_counts = defaultdict(int)
    
    samples_processed = 0
    
    for example in dataset:
        if samples_processed >= max_samples:
            break
            
        completions = example.get("completions", [])
        if len(completions) < 2:
            continue
        
        # Get preferences for each dimension
        preferences = {}
        for dim in DIMENSIONS:
            pref = get_dimension_preference(completions, dim)
            if pref is not None:
                preferences[dim] = pref
        
        # Compare all pairs
        for dim1, dim2 in combinations(DIMENSIONS, 2):
            if dim1 in preferences and dim2 in preferences:
                pair = tuple(sorted([dim1, dim2]))
                total_counts[pair] += 1
                
                # Check if they agree on best response
                if preferences[dim1][0] == preferences[dim2][0]:
                    agree_counts[pair] += 1
                else:
                    conflict_counts[pair] += 1
        
        samples_processed += 1
    
    # Compute conflict ratios
    results = {}
    for pair in combinations(DIMENSIONS, 2):
        pair = tuple(sorted(pair))
        total = total_counts[pair]
        if total > 0:
            conflict_ratio = conflict_counts[pair] / total
            results[pair] = {
                "conflict_ratio": conflict_ratio,
                "conflicts": conflict_counts[pair],
                "agrees": agree_counts[pair],
                "total": total,
            }
    
    return results


def main():
    print("Loading UltraFeedback dataset...")
    dataset = load_dataset("openbmb/UltraFeedback", split="train")
    print(f"Dataset size: {len(dataset)}")
    
    print("\nComputing conflict ratios (this may take a moment)...")
    results = compute_conflict_matrix(dataset, max_samples=10000)
    
    print("\n" + "=" * 70)
    print("PAIRWISE PREFERENCE CONFLICT RATIOS")
    print("=" * 70)
    print(f"{'Dimension Pair':<45} {'Conflict %':>10} {'n':>8}")
    print("-" * 70)
    
    # Sort by conflict ratio
    sorted_results = sorted(results.items(), key=lambda x: x[1]["conflict_ratio"], reverse=True)
    
    for pair, data in sorted_results:
        pair_name = f"{pair[0]} vs {pair[1]}"
        conflict_pct = data["conflict_ratio"] * 100
        print(f"{pair_name:<45} {conflict_pct:>9.1f}% {data['total']:>8}")
    
    print("-" * 70)
    
    # Highlight the pair we used
    helpfulness_honesty = tuple(sorted(["helpfulness", "honesty"]))
    if helpfulness_honesty in results:
        data = results[helpfulness_honesty]
        print(f"\n>>> HELPFULNESS vs HONESTY (used in RQ1): {data['conflict_ratio']*100:.1f}% conflict")
        print(f"    Agrees: {data['agrees']}, Conflicts: {data['conflicts']}, Total: {data['total']}")
    
    # Find highest conflict pair
    if sorted_results:
        highest_pair, highest_data = sorted_results[0]
        print(f"\n>>> HIGHEST CONFLICT PAIR: {highest_pair[0]} vs {highest_pair[1]}")
        print(f"    Conflict ratio: {highest_data['conflict_ratio']*100:.1f}%")
        print(f"    This pair would be best for demonstrating MODPO trade-offs!")
    
    # Save detailed results
    output_path = "outputs/analysis/conflict_ratios.json"
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    serializable_results = {
        f"{k[0]}_vs_{k[1]}": v for k, v in results.items()
    }
    with open(output_path, "w") as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    main()
