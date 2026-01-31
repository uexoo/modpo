"""Score generations with the honesty RM using implicit rewards.

This script loads generated outputs and scores them using the honesty RM
to bypass LLM judge and directly measure RM effectiveness.

Usage:
    cd /mount/arbeitsdaten/tc/t/d/t/t/modpo
    PYTHONPATH=. python scripts/modpo/ultrafeedback/utils/score.py \
        --input_dir outputs/rq1/ultrafeedback/generations/w0.3 \
        --output_dir outputs/rq1/ultrafeedback/scores/w0.3 \
        --sft_model_name outputs/rq1/ultrafeedback/sft_helpfulness/best_checkpoint \
        --rm_model_name outputs/rq1/ultrafeedback/rm_honesty/best_checkpoint
"""
from dataclasses import dataclass, field
from typing import Optional
import os
import json

import torch
import tyro
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_dataset
from peft import get_peft_model, LoraConfig, PeftConfig
from peft.utils import load_peft_weights, set_peft_model_state_dict
from accelerate import Accelerator

from src.utils import (
    disable_progress_bar_non_local_main, PeftAsPreTrained
)
from src.utils.reward import ImplicitRewardWrapper, RewardWrapperInput

disable_progress_bar_non_local_main()

BETA = 0.1  # Same as used in MODPO training
PROMPT_TEMPLATE = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{raw_prompt}\n\n### Response:\n"


@dataclass
class ScriptArguments:

    input_dir: str = field(metadata={"help": "path to generations (jsonl files)"})
    output_dir: str = field(metadata={"help": "output path for scores"})
    sft_model_name: str = field(metadata={"help": "path to SFT model"})
    rm_model_name: str = field(metadata={"help": "path to honesty RM adapter"})
    beta: float = field(default=BETA, metadata={"help": "reward scaling factor"})


def load_generations(input_dir: str):
    """Load all jsonl files from input directory."""
    results = []
    for filename in os.listdir(input_dir):
        if filename.endswith('.jsonl'):
            filepath = os.path.join(input_dir, filename)
            with open(filepath, 'r') as f:
                for line in f:
                    results.append(json.loads(line))
    return results


if __name__ == "__main__":

    script_args = tyro.cli(ScriptArguments)

    print("=" * 60)
    print("SCORING GENERATIONS WITH HONESTY RM")
    print("=" * 60)

    # Load SFT model
    print(f"\n1. Loading SFT model from {script_args.sft_model_name}")
    sft_model = AutoModelForCausalLM.from_pretrained(
        script_args.sft_model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": Accelerator().local_process_index},
    )
    sft_model.config.update({"use_cache": False})

    # Add dummy LoRA to enable adapter loading
    print("\n2. Adding trainable LoRA adapter")
    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(sft_model, peft_config)

    # Load honesty RM adapter
    print(f"\n3. Loading honesty RM adapter from {script_args.rm_model_name}")
    try:
        model.load_adapter(script_args.rm_model_name, adapter_name="margin_reward")
        print("   ✅ Loaded via load_adapter()")
    except Exception as e:
        print(f"   ❌ load_adapter failed: {e}")
        print("   Attempting manual loading...")
        peft_config_rm = PeftConfig.from_pretrained(script_args.rm_model_name)
        model.peft_config["margin_reward"] = peft_config_rm
        if hasattr(model.base_model, "peft_config"):
            model.base_model.peft_config["margin_reward"] = peft_config_rm
        model.base_model.inject_adapter(model, "margin_reward")
        adapters_weights = load_peft_weights(script_args.rm_model_name, device=model.device)
        set_peft_model_state_dict(model, adapters_weights, adapter_name="margin_reward")
        print("   ✅ Loaded manually")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.sft_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Create reward wrapper
    print(f"\n4. Creating ImplicitRewardWrapper (beta={script_args.beta})")
    reward_wrapper = ImplicitRewardWrapper(
        model=PeftAsPreTrained(model, "margin_reward"),
        ref_model=PeftAsPreTrained(model),
        tokenizer=tokenizer,
        beta=script_args.beta,
        prompt_template=PROMPT_TEMPLATE,
    )

    # Load generations
    print(f"\n5. Loading generations from {script_args.input_dir}")
    generations = load_generations(script_args.input_dir)
    print(f"   Loaded {len(generations)} samples")

    # Score each generation
    print("\n6. Computing implicit honesty scores...")
    results = []
    with torch.no_grad():
        for sample in tqdm.tqdm(generations):
            # Extract raw prompt from formatted prompt
            prompt = sample.get('prompt', '')
            response = sample.get('response', '')
            
            # The prompt in gen.py is already formatted, extract raw_prompt
            if "### Instruction:" in prompt and "### Response:" in prompt:
                raw_prompt = prompt.split("### Instruction:")[1].split("### Response:")[0].strip()
            else:
                raw_prompt = prompt
            
            reward = reward_wrapper(RewardWrapperInput(
                raw_prompt=[raw_prompt],
                response=[response]
            ))
            
            results.append({
                "prompt": prompt,
                "response": response,
                "honesty_score": reward.item(),
            })

    # Save results
    os.makedirs(script_args.output_dir, exist_ok=True)
    
    # Raw scores
    raw_path = os.path.join(script_args.output_dir, "scores.jsonl")
    dataset = Dataset.from_list(results)
    dataset.to_json(raw_path)
    print(f"\n7. Saved raw scores to {raw_path}")

    # Summary statistics
    scores = [r["honesty_score"] for r in results]
    mean_score = sum(scores) / len(scores)
    std_score = (sum((s - mean_score) ** 2 for s in scores) / len(scores)) ** 0.5
    min_score = min(scores)
    max_score = max(scores)

    summary_path = os.path.join(script_args.output_dir, "summary.csv")
    with open(summary_path, "w") as f:
        f.write("metric,value\n")
        f.write(f"mean,{mean_score:.6f}\n")
        f.write(f"std,{std_score:.6f}\n")
        f.write(f"min,{min_score:.6f}\n")
        f.write(f"max,{max_score:.6f}\n")
        f.write(f"n_samples,{len(scores)}\n")

    print(f"   Saved summary to {summary_path}")
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Mean honesty score:  {mean_score:.6f}")
    print(f"  Std:                 {std_score:.6f}")
    print(f"  Min:                 {min_score:.6f}")
    print(f"  Max:                 {max_score:.6f}")
    print(f"  N samples:           {len(scores)}")
