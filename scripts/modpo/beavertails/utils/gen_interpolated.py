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


def resolve_adapter_param_key(adapter_key, model_param_keys, adapter_name):
    """Resolve adapter checkpoint keys to live PeftModel parameter keys."""
    if adapter_key in model_param_keys:
        return adapter_key

    candidates = [adapter_key]
    for token in ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B"):
        suffix = f".{token}.weight"
        if adapter_key.endswith(suffix):
            # PEFT modules usually include adapter namespace, e.g. `.default.weight`.
            candidates.append(adapter_key[:-len(".weight")] + f".{adapter_name}.weight")
            break

    for candidate in candidates[1:]:
        if candidate in model_param_keys:
            return candidate
    return None


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
    dataset_name: Optional[str] = field(default="PKU-Alignment/PKU-SafeRLHF-10K-safer", metadata={"help": "the dataset name"})
    dataset_caching: Optional[bool] = field(default=False, metadata={"help": "use cached dataset"})
    dataset_num_proc: Optional[int] = field(default=1, metadata={"help": "num_proc for dataset preprocessing"})

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

    # ------------------------------------------------------------------
    # 2) Load base model + adapter, then replace weights in-place
    # ------------------------------------------------------------------
    # Strategy: load PeftModel from the lower adapter (real checkpoint path,
    # device_map="auto" dispatches correctly — same as gen.py), then
    # overwrite LoRA parameter tensors with interpolated values on the
    # correct device.  This avoids the temp-dir approach which fails to
    # propagate device_map to the adapter.
    print_local_main("loading base model...")
    sft_model = AutoModelForCausalLM.from_pretrained(
        script_args.sft_model_name,
        use_flash_attention_2=script_args.use_flash_attention_2,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(sft_model, script_args.adapter_lower)
    print_local_main("adapter loaded, injecting interpolated weights...")

    # Replace adapter weights in-place.
    active_adapter = getattr(model, "active_adapter", "default")
    if isinstance(active_adapter, list):
        active_adapter = active_adapter[0]

    replaced = 0
    unresolved = []
    model_sd = dict(model.named_parameters())
    model_param_keys = set(model_sd.keys())
    for key, interp_val in sd_interp.items():
        resolved_key = resolve_adapter_param_key(key, model_param_keys, active_adapter)
        if resolved_key is None:
            unresolved.append(key)
            continue
        model_sd[resolved_key].data.copy_(interp_val.to(model_sd[resolved_key].device, model_sd[resolved_key].dtype))
        replaced += 1

    print_local_main(f"  replaced {replaced}/{len(sd_interp)} adapter parameter tensors")
    if unresolved:
        preview = ", ".join(unresolved[:5])
        raise ValueError(
            f"Could not map {len(unresolved)} interpolated adapter keys to model parameters. "
            f"First unresolved keys: {preview}"
        )

    # tokenizer: left padding for generation (matches gen.py)
    tokenizer = AutoTokenizer.from_pretrained(script_args.sft_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()

    try:
        generation_device = model.get_input_embeddings().weight.device
    except Exception:
        generation_device = next(model.parameters()).device
    print_local_main(f"generation input device: {generation_device}")

    # ------------------------------------------------------------------
    # 3) Load dataset
    # ------------------------------------------------------------------
    if not script_args.dataset_caching:
        from datasets import disable_caching
        disable_caching()
    rdp = DATASET_CONFIGS[script_args.dataset_name](
        prompt_template=script_args.prompt_template,
        num_proc=script_args.dataset_num_proc,
    )
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
            input_ids=prompt_tokenized["input_ids"].to(generation_device),
            attention_mask=prompt_tokenized["attention_mask"].to(generation_device),
            max_length=script_args.max_length,
        )
        output = tokenizer.batch_decode(output_tokenized, skip_special_tokens=True)
        for sample in output:
            results.append({"prompt_response": sample})

    dataset = Dataset.from_list(results)
    dataset.to_json(output_path)
    print_local_main(f"saved {len(results)} generations -> {output_path}")
