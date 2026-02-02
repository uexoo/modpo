import os
import torch
import tyro
import json
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset

# Fix for "huggingface_hub.utils._validators. HFValidationError" when offline/remote
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

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
    # Fix for Llama 3 tokenizer: set pad_token and padding_side for batch inference
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Sync pad_token_id in model config
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # ArmoRM attributes mapping (from model card/attributes)
    # We verify the indices for Honesty and Helpfulness
    # Expected: 
    # 'ultrafeedback-honesty' (index 8?)
    # 'ultrafeedback-helpfulness' (index 9?)
    # We will locate them dynamically to be safe
    
    # Try to find attributes mapping
    # 1. Check model.config.id2label (Standard for SequenceClassification)
    # 2. Check model.score.attributes (Custom ArmoRM)
    # 3. Check model.config.attributes (Custom ArmoRM)
    
    attributes_map = None
    source = "Unknown"
    
    if hasattr(model.config, 'id2label') and model.config.id2label:
        attributes_map = model.config.id2label
        source = "model.config.id2label"
    elif hasattr(model, 'score') and hasattr(model.score, 'attributes'):
        attributes_map = {i: attr for i, attr in enumerate(model.score.attributes)}
        source = "model.score.attributes"
    elif hasattr(model.config, 'attributes'):
        attributes_map = {i: attr for i, attr in enumerate(model.config.attributes)}
        source = "model.config.attributes"
        
    # Authoritative reference list from RLHFlow/ArmoRM-Llama3-8B-v0.1 documentation
    # https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1
    REF_ATTRIBUTES = {
        0: 'helpsteer-helpfulness',
        1: 'helpsteer-correctness',
        2: 'helpsteer-coherence',
        3: 'helpsteer-complexity',
        4: 'helpsteer-verbosity',
        5: 'ultrafeedback-overall_score',
        6: 'ultrafeedback-instruction_following',
        7: 'ultrafeedback-truthfulness',
        8: 'ultrafeedback-honesty',
        9: 'ultrafeedback-helpfulness',
        10: 'beavertails-is_safe',
        11: 'prometheus-score',
        12: 'argilla-overall_quality',
        13: 'argilla-judge_lm',
        14: 'code-complexity',
        15: 'code-style',
        16: 'code-explanation',
        17: 'code-instruction-following',
        18: 'code-readability'
    }

    if not attributes_map:
        print(f"WARNING: Could not find embedded attributes in model config.")
        print(f"Using authoritative reference list from RLHFlow documentation.")
        attributes_map = REF_ATTRIBUTES
        source = "Documentation (Hardcoded)"

    print(f"\n{'='*40}")
    print(f"ATTRIBUTE MAPPING (Source: {source})")
    print(f"{'='*40}")
    
    # Print all attributes
    for idx in sorted(attributes_map.keys()):
        try:
            i = int(idx)
            attr_name = attributes_map[idx]
            print(f"  [{i}] {attr_name}")
        except:
             pass 

    # Find indices
    honesty_idx = -1
    helpfulness_idx = -1
    
    # Invert for searching
    val2id = {str(v): k for k, v in attributes_map.items()}
    
    # Search logic
    if 'ultrafeedback-honesty' in val2id:
        honesty_idx = int(val2id['ultrafeedback-honesty'])
    if 'ultrafeedback-helpfulness' in val2id:
        helpfulness_idx = int(val2id['ultrafeedback-helpfulness'])
        
    print(f"{'='*40}")
    if honesty_idx != -1 and helpfulness_idx != -1:
         print(f"✅ Indices Resolved:")
         print(f"   Shape: (Batch, 19)")
         print(f"   Honesty Index:     {honesty_idx} ('ultrafeedback-honesty')")
         print(f"   Helpfulness Index: {helpfulness_idx} ('ultrafeedback-helpfulness')")
    else:
         print(f"❌ Failed to resolve one or both indices.")
         raise ValueError("Could not decisively find required labels.")
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
            prompt = item.get("prompt", "")
            response = item.get("response", "")
            if not prompt or not response:
                 # Try to extract from messages if available
                 if "messages" in item:
                     # Assume last message is assistant, second to last is user
                     pass # Logic depends on exact format, sticking to prompt/response for now
            
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
            chat_inputs.append(messages)
            
        # Tokenize
        inputs = tokenizer.apply_chat_template(
            chat_inputs, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=4096 
        ).to(device)
        
        # Inference
        with torch.no_grad():
            output = model(inputs)
            # ArmoRM output.rewards is the multi-objective score tensor
            rewards = output.rewards.cpu().float()
        
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
            
            # Explicitly ensure our main ones are there with the standard names expected by other scripts
            result_item["scores"]["armorm_honesty"] = rewards[j, honesty_idx].item()
            result_item["scores"]["armorm_helpfulness"] = rewards[j, helpfulness_idx].item()
            
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
