#!/bin/bash
# Quick evaluation: Compare fixed w=0.0 (positive beta) with original models
# Uses the proper eval_rq1.py script for methodologically sound comparison
# Expected runtime: ~10-15 minutes per model comparison

set -e
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0

OUTPUT_ROOT="./outputs/rq1/ultrafeedback"
TEST_DIR="./outputs/rq1/ultrafeedback/test_fix"
EVAL_SIZE=50

echo "=== Quick Evaluation: Fixed vs Original w=0.0 ==="
echo ""

# Step 1: Generate from fixed model (skip if already done)
if [ ! -f "$TEST_DIR/generations_posbeta/00001-of-00001.jsonl" ]; then
    echo "Step 1: Generating from fixed w=0.0 model..."
    python scripts/modpo/ultrafeedback/utils/gen.py \
        --sft_model_name $OUTPUT_ROOT/sft_helpfulness/best_checkpoint \
        --adapter_model_name $TEST_DIR/modpo_w0.0_posbeta/best_checkpoint \
        --output_dir $TEST_DIR/generations_posbeta \
        --eval_size $EVAL_SIZE \
        --max_length 512 \
        --batch_size 4
else
    echo "Step 1: Generations already exist, skipping..."
fi

echo ""
echo "Step 2: Running LLM-as-judge evaluation (using proper eval_rq1.py methodology)..."

# Step 2: Use proper evaluation methodology
# Match sample counts for valid comparison
python << 'EOF'
import json
import os
import glob
from pathlib import Path
from openai import OpenAI
import random

# Load the existing llm_judge module for proper methodology
import sys
sys.path.insert(0, 'scripts/eval')
from llm_judge import call_judge, JUDGE_PROMPTS

TEST_DIR = Path("outputs/rq1/ultrafeedback/test_fix")
ORIG_DIR = Path("outputs/rq1/ultrafeedback")

def load_gens_from_dir(dir_path):
    """Load all generations from a directory."""
    files = sorted(glob.glob(str(dir_path / "*.jsonl")))
    gens = []
    for f in files:
        with open(f) as fp:
            for line in fp:
                gens.append(json.loads(line))
    return gens

# Initialize OpenAI client
client = OpenAI()

# Load generations
fixed_gens = load_gens_from_dir(TEST_DIR / "generations_posbeta")
orig_w00_gens = load_gens_from_dir(ORIG_DIR / "modpo_w0.0" / "generations")
orig_w10_gens = load_gens_from_dir(ORIG_DIR / "modpo_w1.0" / "generations")
sft_gens = load_gens_from_dir(ORIG_DIR / "sft_helpfulness" / "generations")

print(f"Fixed (pos beta): {len(fixed_gens)} generations")
print(f"Original w=0.0:   {len(orig_w00_gens)} generations")
print(f"Original w=1.0:   {len(orig_w10_gens)} generations")  
print(f"SFT baseline:     {len(sft_gens)} generations")

# For quick test, use first 50 samples that match across all
N = min(50, len(fixed_gens), len(orig_w00_gens), len(orig_w10_gens), len(sft_gens))

def evaluate_pair(gens_a, gens_b, dimension, n_samples):
    """Evaluate using proper pairwise methodology with position randomization."""
    wins_a, wins_b, ties = 0, 0, 0
    
    for i in range(n_samples):
        prompt_a = gens_a[i].get("prompt", "")
        resp_a = gens_a[i].get("response", gens_a[i].get("prompt_response", ""))
        resp_b = gens_b[i].get("response", gens_b[i].get("prompt_response", ""))
        
        # Randomize order (proper methodology to avoid position bias)
        if random.random() < 0.5:
            resp_first, resp_second = resp_a, resp_b
            a_is_first = True
        else:
            resp_first, resp_second = resp_b, resp_a
            a_is_first = False
        
        try:
            verdict = call_judge(
                client=client,
                prompt=prompt_a[:1000],  # Truncate for API limits
                response_a=resp_first[:2000],
                response_b=resp_second[:2000],
                dimension=dimension,
                model="gpt-4o-mini"
            )
            
            # Map verdict back accounting for randomization
            if verdict == "A":
                if a_is_first:
                    wins_a += 1
                else:
                    wins_b += 1
            elif verdict == "B":
                if a_is_first:
                    wins_b += 1
                else:
                    wins_a += 1
            else:
                ties += 1
        except Exception as e:
            print(f"  Error sample {i}: {e}")
            ties += 1
    
    return wins_a, wins_b, ties

print("\n" + "="*70)
print("LLM-AS-JUDGE EVALUATION (with position randomization)")
print("="*70)

comparisons = [
    ("Fixed w=0.0 (pos beta) vs Original w=0.0 (neg beta)", fixed_gens, orig_w00_gens),
    ("Fixed w=0.0 (pos beta) vs Original w=1.0", fixed_gens, orig_w10_gens),
    ("Fixed w=0.0 (pos beta) vs SFT baseline", fixed_gens, sft_gens),
]

all_results = []
for name, gens_a, gens_b in comparisons:
    print(f"\n{name}")
    print("-" * len(name))
    
    for dimension in ["helpfulness", "honesty"]:
        wins_a, wins_b, ties = evaluate_pair(gens_a, gens_b, dimension, N)
        win_rate_a = 100 * wins_a / N
        win_rate_b = 100 * wins_b / N
        tie_rate = 100 * ties / N
        
        print(f"  {dimension:12}: Model A wins {wins_a:2d} ({win_rate_a:5.1f}%) | Model B wins {wins_b:2d} ({win_rate_b:5.1f}%) | Ties {ties:2d} ({tie_rate:5.1f}%)")
        
        all_results.append({
            "comparison": name,
            "dimension": dimension,
            "wins_a": wins_a,
            "wins_b": wins_b,
            "ties": ties,
            "n_samples": N,
        })

# Save results
with open(TEST_DIR / "quick_eval_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)
print(f"""
Key question: Does fixing the beta sign create the expected trade-off?

If the fix works:
  • Fixed w=0.0 (maximizing honesty) should WIN on HONESTY vs Original w=0.0
  • Fixed w=0.0 should be competitive or WIN on HELPFULNESS vs Original w=0.0
    (because orig w=0.0 with negative beta was MINIMIZING honesty, not maximizing it)

This would confirm the beta sign bug was inverting the objective.

Results saved to: {TEST_DIR / "quick_eval_results.json"}
""")
EOF

echo ""
echo "=== Evaluation Complete ==="
