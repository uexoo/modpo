"""
Diagnostic script to compare ArmoRM (external evaluator) scores with implicit RM (internal training) rewards.

This script:
1. Loads the SFT baseline model (reference).
2. Loads the Honesty RM adapter (trained signal).
3. Reads existing `scores_armorm.jsonl` (containing ArmoRM scores).
4. Computes the implicit reward: beta * (log_prob_rm(y|x) - log_prob_sft(y|x)).
5. Calculates the Pearson correlation between `armorm_honesty` and `implicit_reward`.

Usage (on server):
    python scripts/analysis/compare_armorm_vs_implicit.py \
        --sft_path outputs/rq1/ultrafeedback/sft_helpfulness/best_checkpoint \
        --rm_adapter_path outputs/rq1/ultrafeedback/rm_honesty/best_checkpoint \
        --input_file outputs/rq1/ultrafeedback/final_run/modpo_w0.8/scores_armorm/scores_armorm.jsonl
"""

import os
import json
import torch
import tyro
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import Optional

@dataclass
class ScriptArguments:
    sft_path: str
    rm_adapter_path: str
    input_file: str
    beta: float = 0.1
    max_samples: Optional[int] = None
    batch_size: int = 4

def compute_implicit_rewards_batch(model, ref_model, tokenizer, prompts, responses, beta):
    """Compute implicit rewards for a batch of prompt-response pairs."""
    
    # Format inputs: PROMPT + RESPONSE
    # Note: We assume the model expects standard text concatenation or chat template content
    # For UltraFeedback models, we usually concatenate prompt + response.
    # We need to being careful about masking the prompt in loss calculation if we used standard training,
    # but for simple log prob gathering of the response tokens, we can just feed the whole sequence.
    
    full_texts = [p + r for p, r in zip(prompts, responses)]
    
    # Tokenize
    inputs = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # We need to find where the prompt ends to only sum logprobs of the response
    # This is a bit tricky with batching/padding. 
    # A robust way is: forward pass, gather log probs, mask out prompt tokens.
    
    # Quick fix: compute lengths of prompts
    prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    # This might differ if padding is involved in full_texts differently.
    # Ideally we use data collator logic, but let's try a simpler approach since we are in inference mode.
    # We can assume left-padding for generation usually, but here we can use right padding for scoring.
    
    # Actually, simpler approach for correctness: calculate one by one if batching is risky with masking.
    # But for speed let's try batching with careful indexing.
    
    # Get prompt lengths (number of tokens)
    # Note: tokenizer adds BOS token usually.
    prompt_lens = prompt_inputs["attention_mask"].sum(dim=1) 
    
    with torch.no_grad():
        # 1. Policy Model (RM) Forward
        outputs = model(**inputs)
        logits = outputs.logits[:, :-1, :] # standard causal shifting
        labels = inputs["input_ids"][:, 1:]
        
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        # Gather log probs of the actual tokens
        token_log_probs = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)
        
        # 2. Ref Model (SFT) Forward
        ref_outputs = ref_model(**inputs)
        ref_logits = ref_outputs.logits[:, :-1, :]
        ref_log_probs = torch.nn.functional.log_softmax(ref_logits, dim=-1)
        ref_token_log_probs = torch.gather(ref_log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)
        
        # 3. Sum over response tokens only
        rewards = []
        for i in range(len(prompts)):
            # Valid tokens mask (ignore padding)
            # attention_mask in inputs marks real tokens [1, 1, ..., 1, 0, 0]
            # labels are shifted by 1 compared to input_ids.
            # input_ids: [BOS, P1, P2, R1, R2, PAD]
            # labels:    [P1,  P2, R1, R2, PAD, ???]
            # We want R1, R2.
            
            # The prompt length `pl` tells us how to skip P1, P2.
            # Note: prompt_inputs['length'] might include BOS. 
            # If full_inputs includes BOS, prompt len match.
            
            # Let's rely on simple slicing: start at prompt_len - 1 (because labels are shifted)
            # Actually, `prompt_lens[i]` is count of tokens. 
            # In `token_log_probs` (seq_len-1), index `k` corresponds to predicting `input_ids[k+1]`.
            # We want to predict response tokens.
            # First response token is at index `prompt_len`.
            # In `input_ids`, it is at `prompt_len`.
            # In `token_log_probs`, we predict it using logits at `prompt_len-1`.
            
            # So start index in `token_log_probs` is `prompt_len[i] - 1`.
            # End index is determined by attention mask sum - 1.
            
            start_idx = prompt_lens[i] - 1
            valid_len = inputs["attention_mask"][i].sum() - 1 # -1 because logits are one shorter
            
            if start_idx >= valid_len:
                # Fallback or weird case (empty response?)
                rewards.append(0.0)
                continue

            policy_sum = token_log_probs[i, start_idx:valid_len].sum()
            ref_sum = ref_token_log_probs[i, start_idx:valid_len].sum()
            
            # Implicit Reward
            r = beta * (policy_sum - ref_sum)
            rewards.append(r.item())
            
    return rewards

def main():
    args = tyro.cli(ScriptArguments)
    
    print(f"Loading SFT model: {args.sft_path}")
    sft_model = AutoModelForCausalLM.from_pretrained(
        args.sft_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    sft_model.eval()
    
    print(f"Loading SFT tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.sft_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Determine padding side? usually right is fine for causal LM batching if using attention mask correctly
    tokenizer.padding_side = "right" 

    print(f"Loading RM Adapter: {args.rm_adapter_path}")
    # Load adapter on top of SFT model (acting as the Policy/RM now)
    # Use load_adapter if we want to keep one model instance and disable/enable,
    # or just load it as a separate PeftModel.
    # PeftModel wraps the base model.
    # For efficiency we could use one model and context managers, but let's be safe and load two if memoery permits.
    # SFT is 7B/8B. Loading twice might OOM on 48GB if not careful.
    # Better strategy: Load SFT. Load Adapter. 
    # To get REF logprobs: disable adapter.
    # To get POL logprobs: enable adapter.
    
    rm_model = PeftModel.from_pretrained(sft_model, args.rm_adapter_path, adapter_name="honesty_rm")
    rm_model.eval()
    
    print(f"Reading ArmoRM scores: {args.input_file}")
    data = []
    with open(args.input_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
            
    if args.max_samples:
        data = data[:args.max_samples]
        print(f"Truncated to {len(data)} samples.")
        
    print("Computing implicit rewards...")
    
    implicit_rewards = []
    armorm_scores = []
    
    # Process in batches
    batched_data = [data[i:i + args.batch_size] for i in range(0, len(data), args.batch_size)]
    
    for batch in tqdm(batched_data):
        prompts = []
        responses = []
        batch_armorm = []
        
        for idx, item in enumerate(batch):
            if "prompt" not in item or "response" not in item:
                if "messages" in item:
                     # Fallback logic
                     prompts.append(item["messages"][0]["content"])
                     responses.append(item["messages"][1]["content"])
                else:
                    continue
            else:
                prompts.append(item["prompt"])
                responses.append(item["response"])
                
            if "scores" in item:
                # Handle old/new format
                if "armorm_honesty" in item["scores"]:
                    batch_armorm.append(item["scores"]["armorm_honesty"])
                elif "honesty" in item["scores"]: # fallback
                     batch_armorm.append(item["scores"]["honesty"])
                else:
                     batch_armorm.append(0.0) # padding/placeholder if missing
            else:
                batch_armorm.append(0.0)
        
        if not prompts:
            continue
            
        # Compute implicit rewards
        # 1. Provide Chat Template formatted inputs
        # We need to construct messages for apply_chat_template
        
        full_inputs_ids = []
        prompt_lens = []
        
        for p, r in zip(prompts, responses):
            # 1. Format full conversation
            messages = [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
            # tokenize=True returns list of ints
            full_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
            
            # 2. Format prompt only to find split point
            prompt_msgs = [{"role": "user", "content": p}]
            prompt_ids = tokenizer.apply_chat_template(prompt_msgs, tokenize=True, add_generation_prompt=True)
            
            full_inputs_ids.append(torch.tensor(full_ids))
            prompt_lens.append(len(prompt_ids))
            
        # Pad inputs manually since they are lists of tensors of varying length
        # Or use tokenizer.pad if we can pass lists.. tokenizer usually takes list of strings or list of list of ints.
        # Let's use tokenizer.pad with "input_ids": [list of ints]
        
        # We need to convert list of tensors/lists to padded batch
        # Let's use torch.nn.utils.rnn.pad_sequence or re-tokenize?
        # Re-tokenizing is hard because we already have IDs.
        
        # Simple padding logic
        max_len = max(len(ids) for ids in full_inputs_ids)
        # Cap at 1024 or higher if needed
        max_len = min(max_len, 2048) # Allow more context
        
        padded_input_ids = []
        attention_masks = []
        
        for ids in full_inputs_ids:
            # Truncate
            if len(ids) > max_len:
                ids = ids[:max_len]
            
            # Pad (left or right? Causal LM usually left for generation, right for training/scoring works if masked)
            # We used right padding in previous attempts. Let's stick to Right Padding for scoring.
            pad_len = max_len - len(ids)
            padded = torch.cat([ids, torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long)])
            mask = torch.cat([torch.ones(len(ids), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])
            
            padded_input_ids.append(padded)
            attention_masks.append(mask)
            
        input_ids = torch.stack(padded_input_ids).to(rm_model.device)
        attention_mask = torch.stack(attention_masks).to(rm_model.device)
        
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        
        # Enable adapter
        rm_model.set_adapter("honesty_rm")
        
        with torch.no_grad():
            # A. Wrapped Forward (RM)
            outputs = rm_model(**inputs)
            logits = outputs.logits[:, :-1, :]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            token_log_probs = torch.gather(log_probs, 2, inputs["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1)
            
            # B. Disable Adapter (SFT Ref)
            with rm_model.disable_adapter():
                ref_outputs = rm_model(**inputs)
                ref_logits = ref_outputs.logits[:, :-1, :]
                ref_log_probs = torch.nn.functional.log_softmax(ref_logits, dim=-1)
                ref_token_log_probs = torch.gather(ref_log_probs, 2, inputs["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1)
            
            # C. Aggregate
            for i in range(len(prompts)):
                # prompt_lens[i] is where response starts
                # In logits (shifted), index is prompt_lens[i] - 1
                start_idx = prompt_lens[i] - 1
                
                # Valid length: sum of mask - 1
                valid_len = inputs["attention_mask"][i].sum() - 1
                
                if start_idx >= valid_len:
                    implicit_rewards.append(0.0)
                    armorm_scores.append(batch_armorm[i])
                    continue

                policy_sum = token_log_probs[i, start_idx:valid_len].sum()
                ref_sum = ref_token_log_probs[i, start_idx:valid_len].sum()
                
                r = args.beta * (policy_sum - ref_sum)
                implicit_rewards.append(r.item())
                armorm_scores.append(batch_armorm[i])

    # Analysis
    implicit_rewards = np.array(implicit_rewards)
    armorm_scores = np.array(armorm_scores)
    
    correlation = np.corrcoef(implicit_rewards, armorm_scores)[0, 1]
    
    print("\n" + "="*30)
    print("RESULTS: ArmoRM vs Implicit RM")
    print("="*30)
    print(f"Num samples: {len(implicit_rewards)}")
    print(f"Correlation: {correlation:.4f}")
    
    print(f"\nArmoRM Stats:")
    print(f"  Mean: {np.mean(armorm_scores):.4f}")
    print(f"  Std:  {np.std(armorm_scores):.4f}")
    
    print(f"\nImplicit RM Stats:")
    print(f"  Mean: {np.mean(implicit_rewards):.4f}")
    print(f"  Std:  {np.std(implicit_rewards):.4f}")
    
    print("\nInterpretation:")
    if correlation < 0.5:
        print("❌ LOW correlation. The evaluators are measuring different things.")
    elif correlation > 0.7:
        print("✅ HIGH correlation. The evaluators agree.")
    else:
        print("⚠️ MODERATE correlation.")

if __name__ == "__main__":
    main()
