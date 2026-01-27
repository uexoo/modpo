"""
Quick check: Are the RM adapter weights actually non-zero?
"""
import torch
from pathlib import Path
from peft import PeftConfig
from peft.utils import load_peft_weights

RM_PATH = Path("outputs/rq1/ultrafeedback/rm_honesty/best_checkpoint")

print("Loading RM adapter weights...")
weights = load_peft_weights(str(RM_PATH), device="cpu")

print(f"\nAdapter has {len(weights)} weight tensors")

total_params = 0
nonzero_params = 0
weight_stats = []

for name, tensor in weights.items():
    n_params = tensor.numel()
    n_nonzero = (tensor != 0).sum().item()
    total_params += n_params
    nonzero_params += n_nonzero
    
    weight_stats.append({
        "name": name,
        "shape": tuple(tensor.shape),
        "mean": tensor.float().mean().item(),
        "std": tensor.float().std().item(),
        "min": tensor.float().min().item(),
        "max": tensor.float().max().item(),
        "nonzero_pct": 100 * n_nonzero / n_params,
    })

print(f"\nTotal parameters: {total_params:,}")
print(f"Non-zero parameters: {nonzero_params:,} ({100*nonzero_params/total_params:.1f}%)")

print(f"\n{'Layer':<50} {'Shape':<20} {'Mean':>10} {'Std':>10} {'Non-zero%':>10}")
print("-" * 100)
for s in weight_stats[:10]:  # Show first 10
    print(f"{s['name']:<50} {str(s['shape']):<20} {s['mean']:>10.6f} {s['std']:>10.6f} {s['nonzero_pct']:>10.1f}%")

if len(weight_stats) > 10:
    print(f"... and {len(weight_stats) - 10} more layers")

# Summary
avg_std = sum(s['std'] for s in weight_stats) / len(weight_stats)
print(f"\nAverage weight std across layers: {avg_std:.6f}")

if avg_std < 1e-5:
    print("\n⚠️  ADAPTER WEIGHTS ARE NEAR-ZERO!")
    print("   The RM training may have failed or not converged.")
else:
    print("\n✅ Adapter weights are non-trivial.")
    print("   The RM learned something - issue is elsewhere.")
