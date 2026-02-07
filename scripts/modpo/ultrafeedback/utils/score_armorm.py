import os
import torch
import tyro
import json
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Fix for "huggingface_hub.utils._validators. HFValidationError" when offline/remote
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Authoritative reference list from RLHFlow/ArmoRM-Llama3-8B-v0.1 documentation
# https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1
REF_ATTRIBUTES = {
    0: "helpsteer-helpfulness",
    1: "helpsteer-correctness",
    2: "helpsteer-coherence",
    3: "helpsteer-complexity",
    4: "helpsteer-verbosity",
    5: "ultrafeedback-overall_score",
    6: "ultrafeedback-instruction_following",
    7: "ultrafeedback-truthfulness",
    8: "ultrafeedback-honesty",
    9: "ultrafeedback-helpfulness",
    10: "beavertails-is_safe",
    11: "prometheus-score",
    12: "argilla-overall_quality",
    13: "argilla-judge_lm",
    14: "code-complexity",
    15: "code-style",
    16: "code-explanation",
    17: "code-instruction-following",
    18: "code-readability",
}

REQUIRED_ATTRS = {
    "helpsteer-helpfulness",
    "helpsteer-correctness",
    "helpsteer-coherence",
    "helpsteer-complexity",
    "helpsteer-verbosity",
    "ultrafeedback-honesty",
    "ultrafeedback-helpfulness",
}


def _canonicalize_attributes_map(raw_map):
    out = {}
    if raw_map is None:
        return out
    if isinstance(raw_map, dict):
        items = raw_map.items()
    else:
        items = enumerate(raw_map)
    for k, v in items:
        try:
            idx = int(k)
        except (TypeError, ValueError):
            continue
        out[idx] = str(v)
    return dict(sorted(out.items()))


def _is_valid_attributes_map(attributes_map: dict[int, str]) -> bool:
    if not attributes_map:
        return False
    labels = set(attributes_map.values())
    if not REQUIRED_ATTRS.issubset(labels):
        return False
    # ArmoRM v0.1 should expose at least the known 19 dimensions.
    return len(attributes_map) >= len(REF_ATTRIBUTES)

@dataclass
class ScriptArguments:
    input_dir: str = field(metadata={"help": "Path to input generations directory containing .jsonl files"})
    output_dir: str = field(metadata={"help": "Path to save output scores"})
    model_path: str = field(default="RLHFlow/ArmoRM-Llama3-8B-v0.1", metadata={"help": "HuggingFace model path"})
    batch_size: int = field(default=8, metadata={"help": "Batch size for inference"})
    debug_max_samples: Optional[int] = field(default=None, metadata={"help": "Limit samples for debugging"})

def main():
    args = tyro.cli(ScriptArguments)
    
    print(f"Loading model: {args.model_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model and tokenizer
    # Trust remote code is required for ArmoRM
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    print(f"[DEBUG] Moving model to {device}...")
    model.to(device)
    print(f"[DEBUG] Model moved.")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError(
            "Tokenizer does not support apply_chat_template(). "
            "ArmoRM scoring requires a newer Transformers version. "
            "Try running in a dedicated eval environment (see packages/modpo/setup_armorm_env.sh)."
        )
    # Fix for Llama 3 tokenizer: set pad_token and padding_side for batch inference
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Sync pad_token_id in model config
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Resolve ArmoRM attribute labels deterministically and fail if required dims are absent.
    candidates = []
    if hasattr(model, "score") and hasattr(model.score, "attributes"):
        candidates.append(
            ("model.score.attributes", _canonicalize_attributes_map(model.score.attributes))
        )
    if hasattr(model.config, "attributes"):
        candidates.append(
            ("model.config.attributes", _canonicalize_attributes_map(model.config.attributes))
        )
    if hasattr(model.config, "id2label") and model.config.id2label:
        candidates.append(
            ("model.config.id2label", _canonicalize_attributes_map(model.config.id2label))
        )

    attributes_map = {}
    source = "unresolved"
    for candidate_source, candidate_map in candidates:
        if _is_valid_attributes_map(candidate_map):
            attributes_map = candidate_map
            source = candidate_source
            break
        if candidate_map:
            print(
                f"WARNING: Ignoring invalid attributes from {candidate_source}. "
                f"Sample={list(candidate_map.values())[:5]}"
            )

    if not attributes_map:
        print(
            "WARNING: Could not resolve valid attributes from model metadata. "
            "Falling back to documented RLHFlow mapping."
        )
        attributes_map = _canonicalize_attributes_map(REF_ATTRIBUTES)
        source = "documentation_fallback"

    if not _is_valid_attributes_map(attributes_map):
        missing = sorted(REQUIRED_ATTRS - set(attributes_map.values()))
        raise ValueError(
            "Resolved attributes map is invalid. "
            f"missing={missing} size={len(attributes_map)} source={source}"
        )

    print(f"\n{'='*40}")
    print(f"ATTRIBUTE MAPPING (Source: {source})")
    print(f"{'='*40}")
    
    # Print all attributes (explicit for auditability)
    for idx in sorted(attributes_map.keys()):
        print(f"  [{idx}] {attributes_map[idx]}")

    val2id = {v: int(k) for k, v in attributes_map.items()}
    helpsteer_helpfulness_idx = val2id["helpsteer-helpfulness"]
    helpsteer_correctness_idx = val2id["helpsteer-correctness"]
    helpsteer_coherence_idx = val2id["helpsteer-coherence"]
    helpsteer_complexity_idx = val2id["helpsteer-complexity"]
    helpsteer_verbosity_idx = val2id["helpsteer-verbosity"]
    ultrafeedback_honesty_idx = val2id["ultrafeedback-honesty"]
    ultrafeedback_helpfulness_idx = val2id["ultrafeedback-helpfulness"]

    print(f"{'='*40}")
    print("✅ Indices Resolved:")
    print(f"   HelpSteer helpfulness: {helpsteer_helpfulness_idx}")
    print(f"   HelpSteer correctness: {helpsteer_correctness_idx}")
    print(f"   HelpSteer coherence:   {helpsteer_coherence_idx}")
    print(f"   HelpSteer complexity:  {helpsteer_complexity_idx}")
    print(f"   HelpSteer verbosity:   {helpsteer_verbosity_idx}")
    print(f"   UltraFeedback honesty: {ultrafeedback_honesty_idx}")
    print(f"   UltraFeedback helpf.:  {ultrafeedback_helpfulness_idx}")
    print(f"{'='*40}\n")

    # Setup directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find input file
    input_file = os.path.join(args.input_dir, "00001-of-00001.jsonl") # Assuming typical structure
    if not os.path.exists(input_file):
        # Try finding any jsonl
        files = [f for f in os.listdir(args.input_dir) if f.endswith('.jsonl')]
        if not files:
            raise FileNotFoundError(f"No .jsonl files found in {args.input_dir}")
        input_file = os.path.join(args.input_dir, files[0])
    
    print(f"Processing file: {input_file}")
    
    # Load data
    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f]
        
    if args.debug_max_samples:
        data = data[:args.debug_max_samples]
        print(f"Debug mode: limited to {len(data)} samples")

    results = []
    
    # Batch processing
    for i in tqdm(range(0, len(data), args.batch_size)):
        print(f"[DEBUG] Processing batch {i // args.batch_size}...")
        batch = data[i:i + args.batch_size]
        
        # Prepare inputs
        # ArmoRM expects chat template format
        chat_inputs = []
        for item in batch:
            # Handle prompt/response format
            # Typical format in these files: "prompt", "response" (or "messages")
            # Prefer raw_prompt if present (generations may store both raw_prompt and a formatted prompt template).
            prompt = item.get("raw_prompt", None)
            if not isinstance(prompt, str) or not prompt.strip():
                prompt = item.get("prompt", "")
            response = item.get("response", "")
            if not prompt or not response:
                 # Try to extract from messages if available
                 if "messages" in item:
                     msgs = item.get("messages")
                     if isinstance(msgs, list) and msgs:
                         user_msg = next((m for m in msgs if m.get("role") in ("user", "human")), None)
                         assistant_msg = next((m for m in reversed(msgs) if m.get("role") in ("assistant", "ai")), None)
                         if user_msg and assistant_msg:
                             prompt = user_msg.get("content", prompt)
                             response = assistant_msg.get("content", response)
            
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
            chat_inputs.append(messages)
            
        # Tokenize
        # Use chat template for formatting, then tokenize normally to ensure we have attention_mask for padded batches.
        chat_texts = [
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            for messages in chat_inputs
        ]
        encodings = tokenizer(
            chat_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        )
        encodings = {k: v.to(device) for k, v in encodings.items()}
        
        # Inference
        with torch.no_grad():
            output = model(**encodings)
            # ArmoRM output.rewards is the multi-objective score tensor
            rewards = output.rewards.cpu().float()

        max_required_idx = max(
            helpsteer_helpfulness_idx,
            helpsteer_correctness_idx,
            helpsteer_coherence_idx,
            helpsteer_complexity_idx,
            helpsteer_verbosity_idx,
            ultrafeedback_honesty_idx,
            ultrafeedback_helpfulness_idx,
        )
        if rewards.shape[1] <= max_required_idx:
            raise ValueError(
                "Rewards tensor shape is smaller than required indices. "
                f"shape={tuple(rewards.shape)} max_required_idx={max_required_idx}"
            )
        
        # Extract scores
        for j, item in enumerate(batch):
            result_item = item.copy()
            if "scores" not in result_item:
                result_item["scores"] = {}
                
            # Extract ALL attributes
            # We iterate over the verified attributes_map
            for attr_idx, attr_name in attributes_map.items():
                # Convert index to int just in case
                idx = int(attr_idx)
                if idx < rewards.shape[1]:
                    score_val = rewards[j, idx].item()
                    # Clean attribute name for json key (remove prefix if desired? keeping full for now)
                    key_name = f"armorm_{attr_name}"
                    result_item["scores"][key_name] = score_val
            
            # Explicit aliases to avoid ambiguity about which benchmark each key belongs to.
            result_item["scores"]["armorm_ultrafeedback-honesty"] = rewards[j, ultrafeedback_honesty_idx].item()
            result_item["scores"]["armorm_ultrafeedback-helpfulness"] = rewards[j, ultrafeedback_helpfulness_idx].item()
            result_item["scores"]["armorm_helpsteer_helpfulness"] = rewards[j, helpsteer_helpfulness_idx].item()

            # Backward compatibility for old analysis scripts.
            result_item["scores"]["armorm_honesty"] = rewards[j, ultrafeedback_honesty_idx].item()
            result_item["scores"]["armorm_helpfulness"] = rewards[j, ultrafeedback_helpfulness_idx].item()
            
            results.append(result_item)

    # Save results
    output_file = os.path.join(args.output_dir, "scores_armorm.jsonl")
    print(f"Saving {len(results)} results to {output_file}")
    with open(output_file, 'w') as f:
        for item in results:
            f.write(json.dumps(item) + "\n")
            
    # Calculate and print summary statistics for ALL attributes
    print("\n" + "="*60)
    print(f"SUMMARY RESULTS (N={len(results)})")
    print(f"{'Attribute':<40} | {'Average':<10}")
    print("-" * 53)
    
    # Calculate averages
    # We use the attributes_map to ensure data order
    for attr_idx in sorted(attributes_map.keys()):
        attr_name = attributes_map[attr_idx]
        key_name = f"armorm_{attr_name}"
        
        # safely compute average
        values = [r["scores"].get(key_name, 0.0) for r in results]
        if values:
            avg_val = sum(values) / len(values)
            print(f"{attr_name:<40} | {avg_val:.4f}")
            
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
