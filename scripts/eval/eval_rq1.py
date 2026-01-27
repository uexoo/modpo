"""
RQ1 Evaluation Orchestration Script.

Runs LLM-as-judge evaluation across all MODPO weights and dimensions,
then aggregates results into a CSV summary.
"""
import os
import json
import argparse
import csv
from pathlib import Path
from typing import List

from openai import OpenAI

from llm_judge import load_generations, evaluate_pairwise, wilson_confidence_interval


def find_generation_files(eval_dir: Path, model_name: str) -> List[Path]:
    """Find all JSONL generation files for a model."""
    # Try 'generations' first (new structure), then 'gen' (old structure)
    for subdir in ["generations", "gen"]:
        gen_dir = eval_dir / model_name / subdir
        if gen_dir.exists():
            return list(gen_dir.glob("*.jsonl"))
    return []


def aggregate_generations(files: List[Path]) -> list[dict]:
    """Aggregate generations from multiple JSONL files."""
    all_gens = []
    for f in sorted(files):
        all_gens.extend(load_generations(f))
    return all_gens


def main():
    parser = argparse.ArgumentParser(description="RQ1 Evaluation Orchestration")
    parser.add_argument("--eval_dir", type=str, required=True, help="Directory with eval outputs")
    parser.add_argument("--dimensions", type=str, nargs="+", required=True, 
                        choices=["helpfulness", "harmlessness", "honesty"],
                        help="Dimensions to evaluate")
    parser.add_argument("--weights", type=str, nargs="+", default=["0.0", "0.5", "1.0"],
                        help="MODPO weights to evaluate")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model")
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI API key")
    parser.add_argument("--output_csv", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()
    
    eval_dir = Path(args.eval_dir)
    
    # Initialize OpenAI client
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key required")
    client = OpenAI(api_key=api_key)
    
    # Load SFT baseline generations
    sft_files = None
    for sft_name in ["sft_baseline", "sft", "sft_helpfulness"]:
        sft_files = find_generation_files(eval_dir, sft_name)
        if sft_files:
            break
    if not sft_files:
        raise ValueError(f"No SFT baseline generations found in {eval_dir}")
    sft_gens = aggregate_generations(sft_files)
    print(f"Loaded {len(sft_gens)} SFT baseline generations")
    
    # Results storage
    all_results = []
    
    # Evaluate each weight
    for w in args.weights:
        model_name = f"modpo_w{w}"
        modpo_files = find_generation_files(eval_dir, model_name)
        if not modpo_files:
            print(f"Warning: No generations found for {model_name}, skipping")
            continue
        
        modpo_gens = aggregate_generations(modpo_files)
        print(f"\nEvaluating {model_name} ({len(modpo_gens)} generations)")
        
        # Ensure same length
        min_len = min(len(modpo_gens), len(sft_gens))
        modpo_gens = modpo_gens[:min_len]
        sft_subset = sft_gens[:min_len]
        
        # Evaluate each dimension
        for dim in args.dimensions:
            print(f"  Evaluating {dim}...")
            
            results = evaluate_pairwise(
                modpo_gens=modpo_gens,
                sft_gens=sft_subset,
                dimension=dim,
                client=client,
                model=args.model,
            )
            
            # Save detailed results
            detail_path = eval_dir / model_name / f"judge_{dim}.json"
            with open(detail_path, "w") as f:
                json.dump(results, f, indent=2)
            
            # Add to summary
            all_results.append({
                "weight": w,
                "dimension": dim,
                "win_rate": results["win_rate"],
                "ci_lower": results["ci_lower"],
                "ci_upper": results["ci_upper"],
                "wins": results["wins"],
                "losses": results["losses"],
                "errors": results["errors"],
            })
            
            print(f"    {dim}: {results['win_rate']:.1%} [{results['ci_lower']:.1%}, {results['ci_upper']:.1%}]")
    
    # Save CSV summary
    output_csv = args.output_csv or str(eval_dir / "win_rates_llm_judge.csv")
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["weight", "dimension", "win_rate", "ci_lower", "ci_upper", "wins", "losses", "errors"])
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"\nResults saved to {output_csv}")
    
    # Print summary table
    print("\n=== SUMMARY ===")
    print(f"{'Weight':<10} {'Dimension':<15} {'Win Rate':<15} {'95% CI':<20}")
    print("-" * 60)
    for r in all_results:
        ci = f"[{r['ci_lower']:.1%}, {r['ci_upper']:.1%}]"
        print(f"{r['weight']:<10} {r['dimension']:<15} {r['win_rate']:.1%}         {ci:<20}")


if __name__ == "__main__":
    main()
