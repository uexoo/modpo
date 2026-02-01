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
    
    if hasattr(model, 'score') and hasattr(model.score, 'attributes'):
        attributes = model.score.attributes
        print(f"Model attributes found: {attributes}")
        try:
            honesty_idx = attributes.index('ultrafeedback-honesty')
            helpfulness_idx = attributes.index('ultrafeedback-helpfulness')
            print(f"Found indices -> Honesty: {honesty_idx}, Helpfulness: {helpfulness_idx}")
        except ValueError as e:
            print(f"ERROR: Could not find required attributes in model. Available: {attributes}")
            raise e
    else:
        # Fallback to hardcoded indices from verified documentation if attribute access fails
        print("WARNING: Could not access model.score.attributes. Using hardcoded indices [Honesty=8, Helpfulness=9]")
        honesty_idx = 8
        helpfulness_idx = 9

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
        print("[DEBUG] Applying chat template...")
        inputs = tokenizer.apply_chat_template(
            chat_inputs, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=4096 
        ).to(device)
        print("[DEBUG] Tokenization complete. Input shape:", inputs.shape)
        
        # Inference
        print("[DEBUG] Running model inference...")
        with torch.no_grad():
            output = model(inputs)
            print("[DEBUG] Inference complete.")
            # ArmoRM output.rewards is the multi-objective score tensor
            # Shape: (batch_size, num_objectives)
            rewards = output.rewards.cpu().float()
        
        # Extract scores
        for j, item in enumerate(batch):
            honesty_score = rewards[j, honesty_idx].item()
            helpfulness_score = rewards[j, helpfulness_idx].item()
            
            result_item = item.copy()
            if "scores" not in result_item:
                result_item["scores"] = {}
            
            result_item["scores"]["armorm_honesty"] = honesty_score
            result_item["scores"]["armorm_helpfulness"] = helpfulness_score
            
            results.append(result_item)

    # Save results
    output_file = os.path.join(args.output_dir, "scores_armorm.jsonl")
    print(f"Saving {len(results)} results to {output_file}")
    with open(output_file, 'w') as f:
        for item in results:
            f.write(json.dumps(item) + "\n")
            
    # Calculate and print summary statistics
    avg_honest = sum(r["scores"]["armorm_honesty"] for r in results) / len(results)
    avg_helpful = sum(r["scores"]["armorm_helpfulness"] for r in results) / len(results)
    
    print("\n" + "="*30)
    print(f"SUMMARY RESULTS")
    print(f"Samples: {len(results)}")
    print(f"Avg Honesty: {avg_honest:.4f}")
    print(f"Avg Helpfulness: {avg_helpful:.4f}")
    print("="*30 + "\n")

if __name__ == "__main__":
    main()
