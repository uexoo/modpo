#!/bin/bash
# Quick evaluation: Compare fixed w=0.0 (positive beta) with original models
# Expected runtime: ~15-20 minutes

set -e
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0

OUTPUT_ROOT="./outputs/rq1/ultrafeedback"
TEST_DIR="./outputs/rq1/ultrafeedback/test_fix"
EVAL_SIZE=50

echo "=== Quick Evaluation: Fixed vs Original w=0.0 ==="
echo ""

# Step 1: Generate from fixed model
echo "Step 1: Generating from fixed w=0.0 model..."
python scripts/modpo/ultrafeedback/utils/gen.py \
    --sft_model_name $OUTPUT_ROOT/sft_helpfulness/best_checkpoint \
    --adapter_model_name $TEST_DIR/modpo_w0.0_posbeta/best_checkpoint \
    --output_dir $TEST_DIR/generations_posbeta \
    --eval_size $EVAL_SIZE \
    --max_length 512 \
    --batch_size 4

echo ""
echo "Step 2: Running LLM-as-judge evaluation..."

# Step 2: Evaluate both dimensions
python << 'EOF'
import json
import os
from pathlib import Path
import glob

# Use the existing llm_judge module
import sys
sys.path.insert(0, '.')
from scripts.eval.llm_judge import evaluate_pairwise

TEST_DIR = Path("outputs/rq1/ultrafeedback/test_fix")
ORIG_DIR = Path("outputs/rq1/ultrafeedback")

def load_generations(path):
    """Load generations from directory or file."""
    if path.is_dir():
        files = sorted(glob.glob(str(path / "*.jsonl")))
        results = []
        for f in files:
            with open(f) as fp:
                for line in fp:
                    results.append(json.loads(line))
        return results
    elif path.exists():
        with open(path) as f:
            return [json.loads(line) for line in f]
    return None

# Load fixed model generations
fixed_gens = load_generations(TEST_DIR / "generations_posbeta")
if not fixed_gens:
    print("ERROR: Could not load fixed model generations!")
    exit(1)
print(f"Loaded {len(fixed_gens)} generations from fixed w=0.0")

# Try different paths for original generations
comparisons = []

# Correct path structure: modpo_w0.0/generations/
orig_paths = [
    ("Original w=0.0", ORIG_DIR / "modpo_w0.0" / "generations"),
    ("Original w=1.0", ORIG_DIR / "modpo_w1.0" / "generations"),
    ("SFT baseline", ORIG_DIR / "sft_helpfulness" / "generations"),
]

for name, path in orig_paths:
    gens = load_generations(path)
    if gens:
        print(f"Loaded {len(gens)} generations from {name}")
        comparisons.append((f"Fixed w=0.0 vs {name}", fixed_gens, gens))

if not comparisons:
    print("\nNo original generations found. Comparing against SFT directly...")
    # Generate from SFT for comparison
    print("You need to first run generation on the original models.")
    print("Checking if original modpo models exist...")
    for w in ["0.0", "0.5", "1.0"]:
        model_path = ORIG_DIR / f"modpo_w{w}" / "best_checkpoint"
        if model_path.exists():
            print(f"  modpo_w{w}: EXISTS")
        else:
            print(f"  modpo_w{w}: NOT FOUND")
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
        
        print(f"  {dimension.capitalize():12}: Fixed wins {win_rate_a:.1f}% | Baseline wins {win_rate_b:.1f}% | Ties {tie_rate:.1f}%")
        
        results.append({
            "comparison": comparison_name,
            "dimension": dimension,
            "fixed_wins": wins_a,
            "baseline_wins": wins_b,
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
