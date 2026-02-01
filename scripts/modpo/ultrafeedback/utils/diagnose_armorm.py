import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
import time

def diagnose():
    model_path = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # 1. Environment Info
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    
    # 2. Test Loading with SDPA (more stable than Flash Attn in some envs)
    print("\n--- Testing Model Loading (SDPA) ---")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa" # Try SDPA first
        )
        model.to(device)
        print("Model loaded and moved to GPU.")
    except Exception as e:
        print(f"Failed to load with SDPA: {e}")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # 3. Test simple inference (No batching, No padding)
    print("\n--- Testing Single Sample Inference ---")
    prompt = "Hello, how are you?"
    response = "I am doing well, thank you!"
    messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
    
    # Use tokenizer directly to get attention_mask
    tokenized = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=False)
    input_ids = tokenized.to(device)
    
    print(f"Input shape: {input_ids.shape}")
    print("Running forward pass...")
    
    start_time = time.time()
    try:
        with torch.no_grad():
            # Test with explicit keyword arguments
            output = model(input_ids=input_ids)
        print(f"Inference successful! Time: {time.time() - start_time:.2f}s")
        print(f"Rewards shape: {output.rewards.shape}")
    except Exception as e:
        print(f"Forward pass failed: {e}")

    # 4. Test with Attention Mask (Crucial for padded batches)
    print("\n--- Testing Sample with Attention Mask ---")
    # Manual tokenization to get both ids and mask
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    encodings = tokenizer(text, return_tensors="pt", padding=True)
    encodings = {k: v.to(device) for k, v in encodings.items()}
    
    print(f"Encodings keys: {encodings.keys()}")
    print("Running forward pass with mask...")
    try:
        with torch.no_grad():
            output = model(**encodings)
        print("Inference with mask successful!")
    except Exception as e:
        print(f"Forward pass with mask failed: {e}")

if __name__ == "__main__":
    diagnose()
