"""
Diagnostic script to compare implicit rewards between helpfulness SFT and honesty RM.

This tests whether the honesty margin RM learned distinct signals from helpfulness.
If they're highly correlated, the margin RM isn't providing useful conflict signal.

Run on server:
    PYTHONPATH=. python scripts/analysis/diagnose_margin_rm.py
"""
import os
import torch
import numpy as np
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

# Paths (adjust if needed)
OUTPUT_ROOT = Path("outputs/rq1/ultrafeedback")
SFT_PATH = OUTPUT_ROOT / "sft_helpfulness/best_checkpoint"
RM_PATH = OUTPUT_ROOT / "rm_honesty/best_checkpoint"

PROMPT_TEMPLATE = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{raw_prompt}\n\n### Response:\n"
BETA = 0.1
N_SAMPLES = 100


def compute_implicit_reward(model, ref_model, tokenizer, prompt, response, beta=0.1):
    """Compute implicit DPO reward: beta * (log pi(y|x) - log pi_ref(y|x))"""
    full_text = prompt + response
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    prompt_inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    prompt_len = prompt_inputs["input_ids"].shape[1]
    
    with torch.no_grad():
        # Model log probs
        outputs = model(**inputs)
        logits = outputs.logits[:, prompt_len-1:-1, :]
        labels = inputs["input_ids"][:, prompt_len:]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)
        model_score = token_log_probs.sum().item()
        
        # Ref model log probs
        ref_outputs = ref_model(**inputs)
        ref_logits = ref_outputs.logits[:, prompt_len-1:-1, :]
        ref_log_probs = torch.nn.functional.log_softmax(ref_logits, dim=-1)
        ref_token_log_probs = torch.gather(ref_log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)
        ref_score = ref_token_log_probs.sum().item()
    
    return beta * (model_score - ref_score)


def main():
    print("=" * 60)
    print("MARGIN RM DIAGNOSTIC")
    print("=" * 60)
    
    # Check paths exist
    if not SFT_PATH.exists():
        print(f"ERROR: SFT model not found at {SFT_PATH}")
        return
    if not RM_PATH.exists():
        print(f"ERROR: RM model not found at {RM_PATH}")
        return
    
    print(f"\nLoading models...")
    print(f"  SFT: {SFT_PATH}")
    print(f"  RM:  {RM_PATH}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(SFT_PATH))
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load SFT (reference model)
    sft_model = AutoModelForCausalLM.from_pretrained(
        str(SFT_PATH),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    sft_model.eval()
    
    # Load RM (SFT + honesty adapter)
    rm_model = PeftModel.from_pretrained(
        sft_model,
        str(RM_PATH),
        torch_dtype=torch.bfloat16,
    )
    rm_model.eval()
    
    print(f"\nLoading UltraFeedback test samples...")
    dataset = load_dataset("openbmb/UltraFeedback", split="train")
    # Take first N samples
    dataset = dataset.select(range(min(len(dataset), N_SAMPLES)))
    
    print(f"\nComputing implicit rewards on {len(dataset)} samples...")
    
    helpfulness_rewards = []
    honesty_rewards = []
    
    for example in tqdm(dataset):
        instruction = example["instruction"]
        completions = example["completions"]
        
        if len(completions) < 2:
            continue
        
        prompt = PROMPT_TEMPLATE.format(raw_prompt=instruction)
        
        # Get scores for first completion
        response = completions[0]["response"]
        
        # Honesty RM reward (using the adapter)
        honesty_r = compute_implicit_reward(
            rm_model, sft_model, tokenizer, prompt, response, beta=BETA
        )
        honesty_rewards.append(honesty_r)
        
        # Get ground truth annotations
        annotations = completions[0].get("annotations", {})
        helpfulness_score = annotations.get("helpfulness", {}).get("Rating", "0")
        honesty_score = annotations.get("honesty", {}).get("Rating", "0")
        
        try:
            helpfulness_rewards.append(float(helpfulness_score))
        except:
            helpfulness_rewards.append(0)
    
    # Compute correlation
    helpfulness_rewards = np.array(helpfulness_rewards)
    honesty_rewards = np.array(honesty_rewards)
    
    # Correlation between model's implicit reward and ground truth helpfulness
    corr_with_helpfulness = np.corrcoef(honesty_rewards, helpfulness_rewards)[0, 1]
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nHonesty RM implicit rewards statistics:")
    print(f"  Mean:   {np.mean(honesty_rewards):.4f}")
    print(f"  Std:    {np.std(honesty_rewards):.4f}")
    print(f"  Min:    {np.min(honesty_rewards):.4f}")
    print(f"  Max:    {np.max(honesty_rewards):.4f}")
    
    print(f"\nCorrelation between Honesty RM reward and ground-truth helpfulness rating:")
    print(f"  r = {corr_with_helpfulness:.4f}")
    
    if abs(corr_with_helpfulness) > 0.5:
        print("\n⚠️  HIGH CORRELATION with helpfulness!")
        print("   The honesty RM is likely learning helpfulness signals, not honesty-specific ones.")
        print("   This explains why both dimensions improve together - there's no real conflict.")
    else:
        print("\n✅ LOW CORRELATION with helpfulness.")
        print("   The honesty RM learned distinct signals.")
        print("   The issue is likely the negative beta sign or weight interpretation.")
    
    # Save results
    output_file = "outputs/analysis/margin_rm_diagnostic.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        f.write(f"Honesty RM Correlation with Helpfulness: {corr_with_helpfulness:.4f}\n")
        f.write(f"Mean Honesty RM reward: {np.mean(honesty_rewards):.4f}\n")
        f.write(f"Std Honesty RM reward: {np.std(honesty_rewards):.4f}\n")
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
