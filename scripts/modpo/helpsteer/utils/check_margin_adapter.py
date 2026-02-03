import argparse
import textwrap
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.configs import DATASET_CONFIGS, DEFAULT_PROMPT_TEMPLATE
from src.utils import PeftAsPreTrained
from src.utils.reward import ImplicitRewardWrapper, RewardWrapperInput


def _truncate(text: str, max_chars: int = 200) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _single_adapter_name(adapter_name) -> str:
    if isinstance(adapter_name, str):
        return adapter_name
    if isinstance(adapter_name, (list, tuple)) and len(adapter_name) == 1:
        return adapter_name[0]
    raise ValueError(f"Unexpected active_adapter value: {adapter_name!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sanity-check a DPO-trained margin adapter by verifying that chosen > rejected "
        "under the implicit reward (logp_adapter - logp_base)."
    )
    parser.add_argument("--sft_model_name", required=True, help="Merged SFT model (full causal LM).")
    parser.add_argument(
        "--margin_adapter_model_name",
        required=True,
        help="LoRA adapter checkpoint directory (e.g., .../best_checkpoint).",
    )
    parser.add_argument(
        "--dataset_name",
        default="nvidia/HelpSteer-pairwise-verbosity",
        help="Preference dataset used to train the margin adapter.",
    )
    parser.add_argument("--split", default="validation", choices=["train", "validation"])
    parser.add_argument("--prompt_template", default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--beta", type=float, default=0.1, help="Scaling for implicit reward wrapper.")
    parser.add_argument(
        "--average_log_prob",
        action="store_true",
        help="Use average log-prob per token instead of summed log-prob.",
    )
    parser.add_argument("--max_examples", type=int, default=256)
    parser.add_argument("--use_flash_attention_2", action="store_true")
    args = parser.parse_args()

    rdp = DATASET_CONFIGS[args.dataset_name](
        prompt_template=args.prompt_template,
        sanity_check=False,
    )
    dataset = rdp.get_preference_dataset(split=args.split)
    if args.max_examples is not None and len(dataset) > args.max_examples:
        dataset = dataset.select(range(args.max_examples))

    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base = AutoModelForCausalLM.from_pretrained(
        args.sft_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_flash_attention_2=args.use_flash_attention_2,
    )
    model = PeftModel.from_pretrained(base, args.margin_adapter_model_name)
    adapter_name = _single_adapter_name(model.active_adapter)

    wrapper = ImplicitRewardWrapper(
        model=PeftAsPreTrained(model, adapter_name),
        ref_model=PeftAsPreTrained(model),
        tokenizer=tokenizer,
        beta=args.beta,
        prompt_template=args.prompt_template,
        average_log_prob=args.average_log_prob,
    )

    margins = []
    failures = []
    for i, ex in enumerate(dataset):
        raw_prompt = ex["raw_prompt"]
        chosen = ex["chosen"]
        rejected = ex["rejected"]
        with torch.no_grad():
            r = wrapper(
                RewardWrapperInput(
                    raw_prompt=[raw_prompt, raw_prompt],
                    response=[chosen, rejected],
                )
            ).detach()
        chosen_r, rejected_r = r.float().cpu().tolist()
        margin = chosen_r - rejected_r
        margins.append(margin)
        if margin <= 0:
            failures.append((margin, raw_prompt, chosen, rejected))

    mean_margin = float(torch.tensor(margins).mean().item()) if margins else float("nan")
    acc = 1.0 - (len(failures) / len(margins)) if margins else 0.0

    print("=== Margin adapter sanity check ===")
    print(f"dataset={args.dataset_name} split={args.split} n={len(margins)}")
    print(f"implicit_reward_beta={args.beta} average_log_prob={args.average_log_prob}")
    print(f"accuracy(chosen>rejected)={acc:.3f} mean_margin={mean_margin:.4f}")

    if failures:
        failures.sort(key=lambda x: x[0])  # most negative first
        worst = failures[0]
        print("\n--- Worst failure (most negative margin) ---")
        print(f"margin={worst[0]:.4f}")
        print("prompt:")
        print(textwrap.indent(_truncate(worst[1], 400), "  "))
        print("chosen (truncated):")
        print(textwrap.indent(_truncate(worst[2], 400), "  "))
        print("rejected (truncated):")
        print(textwrap.indent(_truncate(worst[3], 400), "  "))
        print(f"\nfailures={len(failures)}")


if __name__ == "__main__":
    main()

