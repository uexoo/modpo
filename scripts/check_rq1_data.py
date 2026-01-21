
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

from src.data.configs import DATASET_CONFIGS

def check_dataset(name, split="train"):
    print(f"\nChecking {name} [{split}]...")
    try:
        rdp = DATASET_CONFIGS[name](sanity_check=True)
        ds = rdp.get_preference_dataset(split=split)
        print(f"  Success! Size: {len(ds)}")
        print(f"  Sample: {ds[0]}")
    except Exception as e:
        print(f"  FAILED: {e}")

if __name__ == "__main__":
    check_dataset("Anthropic/hh-rlhf-helpful")
    check_dataset("Anthropic/hh-rlhf-harmless")
    check_dataset("OpenBMB/UltraFeedback-honesty")
