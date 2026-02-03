import argparse
import glob
import json
import os
import statistics
from typing import Iterable, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.configs import DEFAULT_PROMPT_TEMPLATE
from src.utils import PeftAsPreTrained
from src.utils.reward import ImplicitRewardWrapper, RewardWrapperInput


def _single_adapter_name(adapter_name) -> str:
    if isinstance(adapter_name, str):
        return adapter_name
    if isinstance(adapter_name, (list, tuple)) and len(adapter_name) == 1:
        return adapter_name[0]
    raise ValueError(f"Unexpected active_adapter value: {adapter_name!r}")


def _extract_raw_prompt(formatted_prompt: str, prompt_template: str) -> str:
    if "{raw_prompt}" not in prompt_template:
        return formatted_prompt
    prefix, suffix = prompt_template.split("{raw_prompt}", 1)
    if formatted_prompt.startswith(prefix) and formatted_prompt.endswith(suffix):
        start = len(prefix)
        end = len(formatted_prompt) - len(suffix) if suffix else len(formatted_prompt)
        return formatted_prompt[start:end]
    return formatted_prompt


def _iter_jsonl(dir_path: str) -> Iterable[dict]:
    for path in sorted(glob.glob(os.path.join(dir_path, "*.jsonl"))):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def _mean_std(values) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.stdev(values))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score generation JSONL directories with an implicit reward adapter "
        "(logp_adapter - logp_base), and print basic length stats."
    )
    parser.add_argument("--sft_model_name", required=True, help="Merged SFT model (full causal LM).")
    parser.add_argument(
        "--adapter_model_name",
        required=True,
        help="LoRA adapter checkpoint directory (e.g., .../best_checkpoint).",
    )
    parser.add_argument("--prompt_template", default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--beta", type=float, default=0.1, help="Scaling for implicit reward wrapper.")
    parser.add_argument(
        "--average_log_prob",
        action="store_true",
        help="Use average log-prob per token instead of summed log-prob.",
    )
    parser.add_argument("--gens_dir", action="append", required=True, help="Directory of *.jsonl generations. Repeat.")
    parser.add_argument("--label", action="append", help="Optional label(s) corresponding to --gens_dir.")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--use_flash_attention_2", action="store_true")
    args = parser.parse_args()

    labels: Optional[list[str]] = args.label
    if labels is not None and len(labels) != len(args.gens_dir):
        raise ValueError("If provided, --label must be repeated exactly as many times as --gens_dir.")
    if labels is None:
        labels = [os.path.basename(d.rstrip("/")) for d in args.gens_dir]

    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base = AutoModelForCausalLM.from_pretrained(
        args.sft_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_flash_attention_2=args.use_flash_attention_2,
    )
    model = PeftModel.from_pretrained(base, args.adapter_model_name)
    adapter_name = _single_adapter_name(model.active_adapter)

    wrapper = ImplicitRewardWrapper(
        model=PeftAsPreTrained(model, adapter_name),
        ref_model=PeftAsPreTrained(model),
        tokenizer=tokenizer,
        beta=args.beta,
        prompt_template=args.prompt_template,
        average_log_prob=args.average_log_prob,
    )

    print("=== Implicit reward scoring ===")
    print(f"adapter={args.adapter_model_name}")
    print(f"beta={args.beta} average_log_prob={args.average_log_prob}")

    for label, gens_dir in zip(labels, args.gens_dir):
        scores = []
        word_lens = []
        tok_lens = []
        for obj in _iter_jsonl(gens_dir):
            raw_prompt = obj.get("raw_prompt")
            if raw_prompt is None:
                prompt = obj.get("prompt")
                if prompt is None:
                    raise KeyError(
                        f"Missing both 'raw_prompt' and 'prompt' in generation record from {gens_dir}."
                    )
                raw_prompt = _extract_raw_prompt(prompt, args.prompt_template)
            response = obj.get("response")
            if response is None:
                raise KeyError(f"Missing 'response' in generation record from {gens_dir}.")

            with torch.no_grad():
                r = wrapper(RewardWrapperInput(raw_prompt=[raw_prompt], response=[response])).detach()
            scores.append(float(r.float().cpu().item()))
            word_lens.append(len(response.split()))
            tok_lens.append(len(tokenizer(response, add_special_tokens=False)["input_ids"]))

            if args.max_examples is not None and len(scores) >= args.max_examples:
                break

        mean_score, std_score = _mean_std(scores)
        mean_words, std_words = _mean_std(word_lens)
        mean_toks, std_toks = _mean_std(tok_lens)
        print(
            f"{label}: n={len(scores)} "
            f"mean_reward={mean_score:.4f} std_reward={std_score:.4f} "
            f"mean_words={mean_words:.1f} std_words={std_words:.1f} "
            f"mean_toks={mean_toks:.1f} std_toks={std_toks:.1f}"
        )


if __name__ == "__main__":
    main()

