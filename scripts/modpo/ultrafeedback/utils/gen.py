"""Generate responses for UltraFeedback evaluation."""
from dataclasses import dataclass, field
from typing import Optional
import os
import math

import torch
import tyro
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from peft import PeftModel

from src.data.configs import DATASET_CONFIGS, DEFAULT_PROMPT_TEMPLATE
from src.utils import (
    print_local_main, disable_progress_bar_non_local_main, set_seeds
)

disable_progress_bar_non_local_main()


@dataclass
class ScriptArguments:

    sft_model_name: str = field(metadata={"help": "the sft model name"})
    adapter_model_name: str = field(default=None, metadata={"help": "lora name"})
    use_flash_attention_2: Optional[bool] = field(default=False, metadata={"help": "whether to use flash attention 2"})
    prompt_template: Optional[str] = field(default=DEFAULT_PROMPT_TEMPLATE, metadata={"help": "the prompt template"})
    dataset_name: Optional[str] = field(default="OpenBMB/UltraFeedback-helpfulness", metadata={"help": "the dataset name"})
    dataset_caching: Optional[bool] = field(default=False, metadata={"help": "used cached dataset"})

    output_dir: Optional[str] = field(default=None, metadata={"help": "output path for generations"})
    eval_size: Optional[int] = field(default=700, metadata={"help": "number of prompts for generations"})
    max_length: Optional[int] = field(default=512, metadata={"help": "the maximum sequence length"})
    batch_size: Optional[int] = field(default=8)
    rank: Optional[int] = field(default=0)
    world_size: Optional[int] = field(default=1)
    seed: Optional[int] = field(default=0)


if __name__ == "__main__":

    script_args = tyro.cli(ScriptArguments)
    set_seeds(script_args.seed)

    # base model
    print_local_main("loading model...")
    sft_model = AutoModelForCausalLM.from_pretrained(
        script_args.sft_model_name,
        use_flash_attention_2=script_args.use_flash_attention_2,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if script_args.adapter_model_name:
        model = PeftModel.from_pretrained(sft_model, script_args.adapter_model_name)
    else:
        model = sft_model

    # tokenizer: left padding for generation
    tokenizer = AutoTokenizer.from_pretrained(script_args.sft_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # dataset - UltraFeedback only has validation split
    if not script_args.dataset_caching:
        from datasets import disable_caching
        disable_caching()
    rdp = DATASET_CONFIGS[script_args.dataset_name](prompt_template=script_args.prompt_template)
    eval_dataset = rdp.get_sft_dataset(split="validation")
    
    if len(eval_dataset) > script_args.eval_size:
        eval_dataset = eval_dataset.select(range(script_args.eval_size))

    split_size = math.ceil(len(eval_dataset) / script_args.world_size)
    eval_dataset = eval_dataset.select(range(
        script_args.rank * split_size, 
        min((script_args.rank + 1) * split_size, len(eval_dataset))
    ))
    
    os.makedirs(script_args.output_dir, exist_ok=True)
    output_path = os.path.join(
        script_args.output_dir, 
        f"{str(script_args.rank + 1).zfill(5)}-of-{str(script_args.world_size).zfill(5)}.jsonl"
    )

    results = []
    max_input_length = script_args.max_length - 256  # Reserve 256 tokens for generation
    for idx in tqdm.tqdm(range(0, len(eval_dataset), script_args.batch_size)):
        batch = eval_dataset[idx: idx + script_args.batch_size]
        prompt_tokenized = tokenizer(
            batch["prompt"], 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=max_input_length,
        )
        output_tokenized = model.generate(
            input_ids=prompt_tokenized["input_ids"].cuda(),
            attention_mask=prompt_tokenized["attention_mask"].cuda(),
            max_new_tokens=256,
            do_sample=False,
        )
        output = tokenizer.batch_decode(output_tokenized, skip_special_tokens=True)
        for i, sample in enumerate(output):
            prompt = batch["prompt"][i] if isinstance(batch["prompt"], list) else batch["prompt"]
            results.append({
                'prompt': prompt,
                'response': sample[len(prompt):] if sample.startswith(prompt) else sample,
                'prompt_response': sample
            })

    dataset = Dataset.from_list(results)
    dataset.to_json(output_path)
    print(f"Saved {len(results)} generations to {output_path}")
