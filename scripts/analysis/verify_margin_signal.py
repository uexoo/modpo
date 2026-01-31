"""
Proper diagnostic for MODPO margin reward signal.

This script uses the EXACT same loading pattern as modpo.py training script
to verify that the margin_reward adapter produces different outputs than the base.

Run on server:
    PYTHONPATH=. python scripts/analysis/verify_margin_signal.py
"""
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
from accelerate import Accelerator
from tqdm import tqdm

# Import the exact same utilities used in training
from src.utils import prepare_model_for_peft, PeftAsPreTrained
from src.utils.reward import ImplicitRewardWrapper, RewardWrapperInput

# Paths (same as run_full_experiment.sh)
OUTPUT_ROOT = Path("outputs/rq1/ultrafeedback")
SFT_PATH = OUTPUT_ROOT / "sft_helpfulness/best_checkpoint"
RM_PATH = OUTPUT_ROOT / "rm_honesty/best_checkpoint"

PROMPT_TEMPLATE = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{raw_prompt}\n\n### Response:\n"
BETA = 0.1
N_SAMPLES = 50


def main():
    print("=" * 60)
    print("MARGIN SIGNAL VERIFICATION")
    print("Using EXACT same loading pattern as modpo.py")
    print("=" * 60)
    
    # Check paths
    if not SFT_PATH.exists():
        print(f"ERROR: SFT not found at {SFT_PATH}")
        return
    if not RM_PATH.exists():
        print(f"ERROR: RM not found at {RM_PATH}")
        return
    
    print(f"\n1. Loading SFT model from {SFT_PATH}")
    sft_model = AutoModelForCausalLM.from_pretrained(
        str(SFT_PATH),
        torch_dtype=torch.bfloat16,
        device_map={"": Accelerator().local_process_index},
    )
    sft_model.config.update({"use_cache": False})
    
    print("\n2. Adding trainable LoRA adapter (simulating training setup)")
    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(sft_model, peft_config)
    
    print("\n3. Loading margin_reward adapter from RM checkpoint")
    try:
        model.load_adapter(str(RM_PATH), adapter_name="margin_reward")
        print("   ✅ Loaded via load_adapter()")
    except Exception as e:
        print(f"   ❌ load_adapter failed: {e}")
        print("   Attempting manual loading...")
        from peft import PeftConfig
        from peft.utils import load_peft_weights, set_peft_model_state_dict
        
        peft_config_rm = PeftConfig.from_pretrained(str(RM_PATH))
        model.peft_config["margin_reward"] = peft_config_rm
        if hasattr(model.base_model, "peft_config"):
            model.base_model.peft_config["margin_reward"] = peft_config_rm
        model.base_model.inject_adapter(model, "margin_reward")
        adapters_weights = load_peft_weights(str(RM_PATH), device=model.device)
        set_peft_model_state_dict(model, adapters_weights, adapter_name="margin_reward")
        print("   ✅ Loaded manually")
    
    print(f"\n4. Available adapters: {list(model.peft_config.keys())}")
    print(f"   Active adapter: {model.active_adapter}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(SFT_PATH))
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create reward wrapper EXACTLY as in modpo.py
    print("\n5. Creating ImplicitRewardWrapper (same as modpo.py)")
    reward_wrapper = ImplicitRewardWrapper(
        model=PeftAsPreTrained(model, "margin_reward"),  # Uses RM adapter
        ref_model=PeftAsPreTrained(model),               # Disables adapter (base SFT)
        tokenizer=tokenizer,
        beta=BETA,
        prompt_template=PROMPT_TEMPLATE,
    )
    
    # Load test data
    print(f"\n6. Loading {N_SAMPLES} test samples")
    dataset = load_dataset("openbmb/UltraFeedback", split="train")
    dataset = dataset.select(range(min(len(dataset), N_SAMPLES)))
    
    # Compute rewards
    print("\n7. Computing implicit rewards...\n")
    rewards = []
    
    for example in tqdm(dataset):
        instruction = example["instruction"]
        completions = example["completions"]
        if len(completions) < 1:
            continue
        
        response = completions[0]["response"]
        
        # Use the reward wrapper
        with torch.no_grad():
            reward = reward_wrapper(RewardWrapperInput(
                raw_prompt=[instruction],
                response=[response]
            ))
            rewards.append(reward.item())
    
    rewards = torch.tensor(rewards)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nImplicit reward statistics (beta={BETA}):")
    print(f"  Mean:   {rewards.mean():.6f}")
    print(f"  Std:    {rewards.std():.6f}")
    print(f"  Min:    {rewards.min():.6f}")
    print(f"  Max:    {rewards.max():.6f}")
    
    if rewards.std() < 0.001:
        print("\n⚠️  REWARDS ARE CONSTANT!")
        print("   The margin_reward adapter is NOT producing different outputs.")
        print("   Possible causes:")
        print("   - Adapter is not being activated during forward pass")
        print("   - LoRA weights are near-zero despite appearing non-zero")
        print("   - Bug in PeftAsPreTrained context switching")
    elif rewards.std() < 0.1:
        print("\n⚠️  LOW VARIANCE in rewards")
        print("   The margin_reward adapter has minimal effect on outputs.")
        print("   Trade-off signal is weak.")
    else:
        print("\n✅ REWARDS HAVE VARIANCE")
        print("   The margin_reward adapter IS producing differentiated outputs.")
        print("   The constant training metric might be a logging issue.")
    
    # Test adapter switching explicitly
    print("\n" + "=" * 60)
    print("ADAPTER SWITCHING TEST")
    print("=" * 60)
    
    test_input = tokenizer("Hello, how are you?", return_tensors="pt").to(model.device)
    
    # With margin_reward adapter
    model.set_adapter("margin_reward")
    with torch.no_grad():
        logits_rm = model(**test_input).logits
    
    # With default adapter disabled
    with model.disable_adapter():
        with torch.no_grad():
            logits_base = model(**test_input).logits
    
    diff = (logits_rm - logits_base).abs().mean().item()
    print(f"\nMean absolute difference in logits:")
    print(f"  |logits_rm - logits_base| = {diff:.6f}")
    
    if diff < 1e-5:
        print("\n⚠️  ADAPTERS PRODUCE IDENTICAL OUTPUTS")
        print("   This confirms the margin_reward adapter is not being used.")
    else:
        print("\n✅ ADAPTERS PRODUCE DIFFERENT OUTPUTS")
        print(f"   Difference is {diff:.6f}, which is significant.")


if __name__ == "__main__":
    main()
