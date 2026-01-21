
from datasets import get_dataset_config_names, load_dataset

try:
    configs = get_dataset_config_names("Anthropic/hh-rlhf")
    print(f"Available configs: {configs}")
except Exception as e:
    print(f"Could not get configs: {e}")

try:
    # Load first few examples of default to see what's in there
    ds = load_dataset("Anthropic/hh-rlhf", split="train", streaming=True)
    print("Default config first example:", next(iter(ds)))
except Exception as e:
    print(f"Could not load default: {e}")
