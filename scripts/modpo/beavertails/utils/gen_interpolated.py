"""Generate responses from a linearly interpolated LoRA adapter (Rewarded Soups).

RQ3 Tier 1: interpolate adapter parameters between two trained MODPO
models bracketing a target preference w* and generate responses.

Interpolation convention:
    theta = (1 - alpha) * theta_lower + alpha * theta_upper
    alpha = 0 -> pure lower adapter, alpha = 1 -> pure upper adapter.
    For midpoint targets on a 0.2-step grid, alpha = 0.5.

Reference: Ramé et al. (2023), "Rewarded Soups", NeurIPS 2023.
"""

from dataclasses import dataclass, field
from typing import Optional
import os
import tempfile
import shutil

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


def load_adapter_weights(adapter_path):
    """Load adapter state dict, supporting both .bin and .safetensors formats."""
    bin_path = os.path.join(adapter_path, "adapter_model.bin")
    st_path = os.path.join(adapter_path, "adapter_model.safetensors")
    if os.path.exists(bin_path):
        return torch.load(bin_path, map_location="cpu")
    elif os.path.exists(st_path):
        try:
            from safetensors.torch import load_file
            return load_file(st_path)
        except ImportError:
            raise ImportError(
                f"Found {st_path} but safetensors package is not installed. "
                "Install with: pip install safetensors"
            )
    else:
        raise FileNotFoundError(
            f"No adapter_model.bin or adapter_model.safetensors in {adapter_path}"
        )


@dataclass
class ScriptArguments:

    sft_model_name: str = field(metadata={"help": "the sft (base) model name"})
    adapter_lower: str = field(metadata={"help": "path to the lower-weight LoRA adapter checkpoint"})
    adapter_upper: str = field(metadata={"help": "path to the upper-weight LoRA adapter checkpoint"})
    alpha: float = field(default=0.5, metadata={"help": "interpolation coefficient: 0=lower, 1=upper"})

    use_flash_attention_2: Optional[bool] = field(default=False, metadata={"help": "whether to use flash attention 2"})
    prompt_template: Optional[str] = field(default=DEFAULT_PROMPT_TEMPLATE, metadata={"help": "the prompt template"})
    dataset_name: Optional[str] = field(default="PKU-Alignment/PKU-SafeRLHF-10K", metadata={"help": "the dataset name"})
    dataset_caching: Optional[bool] = field(default=False, metadata={"help": "use cached dataset"})

    output_dir: Optional[str] = field(default=None, metadata={"help": "output path for generations"})
    eval_size: Optional[int] = field(default=200, metadata={"help": "number of prompts for generation"})
    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    batch_size: Optional[int] = field(default=8, metadata={"help": "generation batch size"})
    seed: Optional[int] = field(default=0, metadata={"help": "random seed"})


if __name__ == "__main__":

    script_args = tyro.cli(ScriptArguments)
    set_seeds(script_args.seed)

    assert 0.0 <= script_args.alpha <= 1.0, f"alpha must be in [0, 1], got {script_args.alpha}"
    assert script_args.output_dir is not None, "must specify --output_dir"

    # ------------------------------------------------------------------
    # 1) Interpolate adapter weights
    # ------------------------------------------------------------------
    print_local_main(f"=== LoRA interpolation (alpha={script_args.alpha}) ===")
    print_local_main(f"  lower adapter: {script_args.adapter_lower}")
    print_local_main(f"  upper adapter: {script_args.adapter_upper}")

    sd_lower = load_adapter_weights(script_args.adapter_lower)
    sd_upper = load_adapter_weights(script_args.adapter_upper)

    # Verify structural compatibility
    if set(sd_lower.keys()) != set(sd_upper.keys()):
        diff = set(sd_lower.keys()) ^ set(sd_upper.keys())
        raise ValueError(f"Adapter key mismatch — cannot interpolate. Diff keys: {diff}")

    # theta_interp = (1 - alpha) * theta_lower + alpha * theta_upper
    sd_interp = {}
    for key in sd_lower:
        sd_interp[key] = (1.0 - script_args.alpha) * sd_lower[key] + script_args.alpha * sd_upper[key]

    n_params = sum(p.numel() for p in sd_interp.values())
    print_local_main(f"  interpolated {len(sd_interp)} tensors ({n_params:,} parameters)")

    # Save interpolated adapter to a temp directory with the lower adapter's config
    tmp_dir = tempfile.mkdtemp(prefix="lora_interp_")
    bin_path = os.path.join(script_args.adapter_lower, "adapter_model.bin")
    if os.path.exists(bin_path):
        torch.save(sd_interp, os.path.join(tmp_dir, "adapter_model.bin"))
    else:
        # fall back to .bin even if source was safetensors (PeftModel loads both)
        torch.save(sd_interp, os.path.join(tmp_dir, "adapter_model.bin"))
    shutil.copy(
        os.path.join(script_args.adapter_lower, "adapter_config.json"),
        os.path.join(tmp_dir, "adapter_config.json")
    )

    # ------------------------------------------------------------------
    # 2) Load base model + interpolated adapter
    # ------------------------------------------------------------------
    print_local_main("loading base model...")
    sft_model = AutoModelForCausalLM.from_pretrained(
        script_args.sft_model_name,
        use_flash_attention_2=script_args.use_flash_attention_2,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(sft_model, tmp_dir)
    shutil.rmtree(tmp_dir)
    print_local_main("interpolated model loaded")

    # tokenizer: left padding for generation (matches gen.py)
    tokenizer = AutoTokenizer.from_pretrained(script_args.sft_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ------------------------------------------------------------------
    # 3) Load dataset
    # ------------------------------------------------------------------
    if not script_args.dataset_caching:
        from datasets import disable_caching
        disable_caching()
    rdp = DATASET_CONFIGS[script_args.dataset_name](prompt_template=script_args.prompt_template)
    eval_dataset = rdp.get_sft_dataset(split="validation").select(range(script_args.eval_size))

    os.makedirs(script_args.output_dir, exist_ok=True)
    output_path = os.path.join(script_args.output_dir, "00001-of-00001.jsonl")

    # ------------------------------------------------------------------
    # 4) Generate
    # ------------------------------------------------------------------
    print_local_main(f"generating {len(eval_dataset)} responses (max_length={script_args.max_length})...")
    results = []
    for idx in tqdm.tqdm(range(0, len(eval_dataset), script_args.batch_size)):
        batch = eval_dataset[idx: idx + script_args.batch_size]
        prompt_tokenized = tokenizer(
            batch["prompt"],
            return_tensors="pt",
            padding=True,
        )
        output_tokenized = model.generate(
            input_ids=prompt_tokenized["input_ids"].cuda(),
            attention_mask=prompt_tokenized["attention_mask"].cuda(),
            max_length=script_args.max_length,
        )
        output = tokenizer.batch_decode(output_tokenized, skip_special_tokens=True)
        for sample in output:
            results.append({"prompt_response": sample})

    dataset = Dataset.from_list(results)
    dataset.to_json(output_path)
    print_local_main(f"saved {len(results)} generations -> {output_path}")
