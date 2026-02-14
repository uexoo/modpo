"""Generate responses for UltraFeedback evaluation."""

import math
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import tqdm
import tyro
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.configs import DATASET_CONFIGS, DEFAULT_PROMPT_TEMPLATE
from src.utils import disable_progress_bar_non_local_main, print_local_main, set_seeds

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
    max_new_tokens: Optional[int] = field(default=256, metadata={"help": "maximum number of new tokens to generate"})
    max_input_length: Optional[int] = field(
        default=None,
        metadata={"help": "maximum prompt token length; defaults to max_length - max_new_tokens"},
    )
    do_sample: Optional[bool] = field(default=False, metadata={"help": "sampling toggle for generation"})
    temperature: Optional[float] = field(default=1.0, metadata={"help": "sampling temperature (used when do_sample=True)"})
    top_p: Optional[float] = field(default=1.0, metadata={"help": "nucleus sampling p (used when do_sample=True)"})
    repetition_penalty: Optional[float] = field(default=1.0, metadata={"help": "repetition penalty (1.0 disables)"})
    no_repeat_ngram_size: Optional[int] = field(default=0, metadata={"help": "block repeated ngrams (0 disables)"})
    batch_size: Optional[int] = field(default=8)
    rank: Optional[int] = field(default=0)
    world_size: Optional[int] = field(default=1)
    seed: Optional[int] = field(default=0)


if __name__ == "__main__":
    script_args = tyro.cli(ScriptArguments)
    set_seeds(script_args.seed)

    if script_args.output_dir is None:
        raise ValueError("--output_dir is required.")
    if script_args.eval_size is None or int(script_args.eval_size) <= 0:
        raise ValueError("--eval_size must be a positive integer.")
    if script_args.batch_size is None or int(script_args.batch_size) <= 0:
        raise ValueError("--batch_size must be a positive integer.")
    if script_args.world_size is None or int(script_args.world_size) <= 0:
        raise ValueError("--world_size must be a positive integer.")
    if script_args.rank is None or int(script_args.rank) < 0:
        raise ValueError("--rank must be >= 0.")
    if int(script_args.rank) >= int(script_args.world_size):
        raise ValueError("--rank must be < --world_size.")
    if script_args.repetition_penalty is None or float(script_args.repetition_penalty) <= 0:
        raise ValueError("--repetition_penalty must be > 0.")
    if script_args.no_repeat_ngram_size is None or int(script_args.no_repeat_ngram_size) < 0:
        raise ValueError("--no_repeat_ngram_size must be >= 0.")
    if bool(script_args.do_sample):
        if script_args.temperature is None or float(script_args.temperature) <= 0:
            raise ValueError("--temperature must be > 0 when --do_sample=True.")
        if script_args.top_p is None or not (0 < float(script_args.top_p) <= 1):
            raise ValueError("--top_p must be in (0, 1] when --do_sample=True.")

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
    model.eval()

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
    eval_dataset = eval_dataset.select(
        range(script_args.rank * split_size, min((script_args.rank + 1) * split_size, len(eval_dataset)))
    )

    os.makedirs(script_args.output_dir, exist_ok=True)
    output_path = os.path.join(
        script_args.output_dir,
        f"{str(script_args.rank + 1).zfill(5)}-of-{str(script_args.world_size).zfill(5)}.jsonl",
    )

    if script_args.max_input_length is None:
        max_input_length = int(script_args.max_length) - int(script_args.max_new_tokens)
    else:
        max_input_length = int(script_args.max_input_length)
    if max_input_length <= 0:
        raise ValueError(
            f"Invalid max_input_length={max_input_length}. "
            f"Choose max_length ({script_args.max_length}) and max_new_tokens ({script_args.max_new_tokens}) "
            "so that max_length - max_new_tokens > 0, or set --max_input_length explicitly."
        )

    if (not bool(script_args.do_sample)) and (
        float(script_args.temperature) != 1.0 or float(script_args.top_p) != 1.0
    ):
        print_local_main("do_sample=False; temperature/top_p are ignored.")
    print_local_main(
        "decode config: "
        f"do_sample={bool(script_args.do_sample)} "
        f"temperature={float(script_args.temperature)} "
        f"top_p={float(script_args.top_p)} "
        f"repetition_penalty={float(script_args.repetition_penalty)} "
        f"no_repeat_ngram_size={int(script_args.no_repeat_ngram_size)}"
    )

    results = []
    for idx in tqdm.tqdm(range(0, len(eval_dataset), script_args.batch_size)):
        batch = eval_dataset[idx : idx + script_args.batch_size]
        prompt_tokenized = tokenizer(
            batch["prompt"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        )
        input_len_padded = prompt_tokenized["input_ids"].shape[1]

        generate_kwargs = {
            "input_ids": prompt_tokenized["input_ids"].cuda(),
            "attention_mask": prompt_tokenized["attention_mask"].cuda(),
            "max_new_tokens": int(script_args.max_new_tokens),
            "do_sample": bool(script_args.do_sample),
            "repetition_penalty": float(script_args.repetition_penalty),
            "no_repeat_ngram_size": int(script_args.no_repeat_ngram_size),
        }
        if bool(script_args.do_sample):
            generate_kwargs["temperature"] = float(script_args.temperature)
            generate_kwargs["top_p"] = float(script_args.top_p)

        output_tokenized = model.generate(**generate_kwargs)
        output_full = tokenizer.batch_decode(output_tokenized, skip_special_tokens=True)
        output_completion = tokenizer.batch_decode(output_tokenized[:, input_len_padded:], skip_special_tokens=True)
        for i, sample in enumerate(output_full):
            prompt = batch["prompt"][i] if isinstance(batch["prompt"], list) else batch["prompt"]
            raw_prompt = None
            if "raw_prompt" in batch:
                raw_prompt = batch["raw_prompt"][i] if isinstance(batch["raw_prompt"], list) else batch["raw_prompt"]
            completion = output_completion[i] if isinstance(output_completion, list) else output_completion
            results.append(
                {
                    "raw_prompt": raw_prompt,
                    "prompt": prompt,
                    "response": completion,
                    "prompt_response": sample,
                    "response_had_prompt_prefix_match": sample.startswith(prompt),
                }
            )

    dataset = Dataset.from_list(results)
    dataset.to_json(output_path)
    print(f"Saved {len(results)} generations to {output_path}")
