#!/bin/bash
# Quick evaluation: Compare fixed w=0.0 (positive beta) with original models
# Expected runtime: ~15-20 minutes

set -e
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0

OUTPUT_ROOT="./outputs/rq1/ultrafeedback"
TEST_DIR="./outputs/rq1/ultrafeedback/test_fix"
NUM_SAMPLES=50

echo "=== Quick Evaluation: Fixed vs Original w=0.0 ==="
echo ""

# Step 1: Generate from fixed model
echo "Step 1: Generating from fixed w=0.0 model..."
python scripts/modpo/ultrafeedback/gen.py \
    --model_path $TEST_DIR/modpo_w0.0_posbeta/best_checkpoint \
    --output_path $TEST_DIR/generations_w0.0_posbeta.jsonl \
    --num_samples $NUM_SAMPLES \
    --max_length 512

echo ""
echo "Step 2: Running LLM-as-judge evaluation..."

# Step 2: Evaluate both dimensions
python << 'EOF'
import json
import os
from pathlib import Path

# Use the existing llm_judge module
import sys
sys.path.insert(0, '.')
from scripts.eval.llm_judge import evaluate_pairwise, DIMENSION_PROMPTS

TEST_DIR = Path("outputs/rq1/ultrafeedback/test_fix")
ORIG_DIR = Path("outputs/rq1/ultrafeedback")

# Load generations
def load_generations(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

# Load fixed model generations
fixed_gens = load_generations(TEST_DIR / "generations_w0.0_posbeta.jsonl")
print(f"Loaded {len(fixed_gens)} generations from fixed w=0.0")

# Load original w=0.0 and w=1.0 for comparison
orig_w00_path = ORIG_DIR / "generations" / "modpo_w0.0.jsonl"
orig_w10_path = ORIG_DIR / "generations" / "modpo_w1.0.jsonl"
sft_path = ORIG_DIR / "generations" / "sft_helpfulness.jsonl"

comparisons = []

if orig_w00_path.exists():
    orig_w00 = load_generations(orig_w00_path)
    print(f"Loaded {len(orig_w00)} generations from original w=0.0")
    comparisons.append(("Fixed w=0.0 vs Original w=0.0", fixed_gens, orig_w00))

if orig_w10_path.exists():
    orig_w10 = load_generations(orig_w10_path)
    print(f"Loaded {len(orig_w10)} generations from original w=1.0")
    comparisons.append(("Fixed w=0.0 vs Original w=1.0", fixed_gens, orig_w10))

if sft_path.exists():
    sft_gens = load_generations(sft_path)
    print(f"Loaded {len(sft_gens)} generations from SFT baseline")
    comparisons.append(("Fixed w=0.0 vs SFT", fixed_gens, sft_gens))

if not comparisons:
    print("No original generations found for comparison!")
    print("Run the original evaluation first to generate baseline responses.")
    exit(1)

print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)

results = []
for comparison_name, gens_a, gens_b in comparisons:
    print(f"\n{comparison_name}")
    print("-" * len(comparison_name))
    
    for dimension in ["helpfulness", "honesty"]:
        wins_a = 0
        wins_b = 0
        ties = 0
        n_samples = min(len(gens_a), len(gens_b), 50)
        
        for i in range(n_samples):
            prompt = gens_a[i].get("prompt", gens_a[i].get("instruction", ""))
            resp_a = gens_a[i].get("response", gens_a[i].get("generation", ""))
            resp_b = gens_b[i].get("response", gens_b[i].get("generation", ""))
            
            try:
                result = evaluate_pairwise(prompt, resp_a, resp_b, dimension)
                if result == "A":
                    wins_a += 1
                elif result == "B":
                    wins_b += 1
                else:
                    ties += 1
            except Exception as e:
                print(f"  Error evaluating sample {i}: {e}")
                ties += 1
        
        win_rate_a = 100 * wins_a / n_samples
        win_rate_b = 100 * wins_b / n_samples
        tie_rate = 100 * ties / n_samples
        
        print(f"  {dimension.capitalize():12}: Fixed wins {win_rate_a:.1f}% | Orig wins {win_rate_b:.1f}% | Ties {tie_rate:.1f}%")
        
        results.append({
            "comparison": comparison_name,
            "dimension": dimension,
            "fixed_wins": wins_a,
            "orig_wins": wins_b,
            "ties": ties,
            "fixed_win_rate": win_rate_a,
        })

# Save results
results_path = TEST_DIR / "evaluation_results.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {results_path}")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)
print("""
If the fix works correctly:
- Fixed w=0.0 should BEAT original w=0.0 on HONESTY
- Fixed w=0.0 may LOSE to original w=1.0 on HELPFULNESS
- This would show the expected trade-off behavior!
""")
EOF

echo ""
echo "=== Evaluation Complete ==="
