"""Score HH-RLHF generations with Ray2333 helpful/harmless reward models."""

from dataclasses import dataclass, field
import glob
import json
import os
from typing import Optional

import torch
import tyro
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class ScriptArguments:
    input_dir: str = field(metadata={"help": "Directory containing generation JSONL files"})
    output_dir: str = field(metadata={"help": "Directory to write scores_ray2333.jsonl"})
    helpful_model_name: str = field(
        default="Ray2333/gpt2-large-helpful-reward_model",
        metadata={"help": "HF model name for helpful reward model"},
    )
    harmless_model_name: str = field(
        default="Ray2333/gpt2-large-harmless-reward_model",
        metadata={"help": "HF model name for harmless reward model"},
    )
    batch_size: int = field(default=16, metadata={"help": "Batch size for scoring"})
    max_length: int = field(default=1024, metadata={"help": "Max token length for scoring inputs"})
    debug_max_samples: Optional[int] = field(default=None, metadata={"help": "Optional cap for quick debugging"})


def _iter_jsonl(dir_path: str):
    for path in sorted(glob.glob(os.path.join(dir_path, "*.jsonl"))):
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                yield path, i, json.loads(line)


def _extract_question(obj: dict) -> Optional[str]:
    raw_prompt = obj.get("raw_prompt")
    if isinstance(raw_prompt, str) and raw_prompt.strip():
        return raw_prompt

    prompt = obj.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt

    return None


def _extract_answer(obj: dict) -> Optional[str]:
    response = obj.get("response")
    if isinstance(response, str) and response.strip():
        return response

    prompt_response = obj.get("prompt_response")
    prompt = obj.get("prompt")
    if isinstance(prompt_response, str) and isinstance(prompt, str) and prompt_response.startswith(prompt):
        answer = prompt_response[len(prompt):].strip()
        return answer if answer else None

    return None


def _score_batch(model, tokenizer, questions, answers, device, max_length):
    enc = tokenizer(
        questions,
        answers,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits

    if logits.ndim != 2:
        raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")

    if logits.shape[1] == 1:
        return logits[:, 0].float().cpu().tolist()

    # Some checkpoints expose two logits. Use positive-minus-negative as scalar.
    return (logits[:, -1] - logits[:, 0]).float().cpu().tolist()


def main():
    args = tyro.cli(ScriptArguments)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading helpful RM: {args.helpful_model_name}")
    helpful_tok = AutoTokenizer.from_pretrained(args.helpful_model_name, use_fast=True)
    helpful_rm = AutoModelForSequenceClassification.from_pretrained(args.helpful_model_name)
    if helpful_tok.pad_token is None:
        helpful_tok.pad_token = helpful_tok.eos_token
    helpful_rm.to(device).eval()

    print(f"Loading harmless RM: {args.harmless_model_name}")
    harmless_tok = AutoTokenizer.from_pretrained(args.harmless_model_name, use_fast=True)
    harmless_rm = AutoModelForSequenceClassification.from_pretrained(args.harmless_model_name)
    if harmless_tok.pad_token is None:
        harmless_tok.pad_token = harmless_tok.eos_token
    harmless_rm.to(device).eval()

    rows = []
    for path, line_no, obj in _iter_jsonl(args.input_dir):
        q = _extract_question(obj)
        a = _extract_answer(obj)
        if not q or not a:
            continue
        rows.append((path, line_no, obj, q, a))
        if args.debug_max_samples is not None and len(rows) >= args.debug_max_samples:
            break

    if not rows:
        raise ValueError(f"No valid prompt/response records found in {args.input_dir}")

    print(f"Scoring {len(rows)} samples from {args.input_dir}")

    helpful_scores = []
    harmless_scores = []

    for i in range(0, len(rows), args.batch_size):
        batch = rows[i:i + args.batch_size]
        questions = [r[3] for r in batch]
        answers = [r[4] for r in batch]

        helpful_scores.extend(
            _score_batch(
                model=helpful_rm,
                tokenizer=helpful_tok,
                questions=questions,
                answers=answers,
                device=device,
                max_length=args.max_length,
            )
        )
        harmless_scores.extend(
            _score_batch(
                model=harmless_rm,
                tokenizer=harmless_tok,
                questions=questions,
                answers=answers,
                device=device,
                max_length=args.max_length,
            )
        )

    if len(helpful_scores) != len(rows) or len(harmless_scores) != len(rows):
        raise RuntimeError("Score length mismatch.")

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "scores_ray2333.jsonl")

    with open(output_file, "w", encoding="utf-8") as f:
        for idx, (_, _, obj, _, _) in enumerate(rows):
            out = obj.copy()
            if "scores" not in out or not isinstance(out["scores"], dict):
                out["scores"] = {}
            out["scores"]["ray2333_helpful"] = float(helpful_scores[idx])
            out["scores"]["ray2333_harmless"] = float(harmless_scores[idx])
            f.write(json.dumps(out, ensure_ascii=True) + "\n")

    mean_helpful = sum(helpful_scores) / len(helpful_scores)
    mean_harmless = sum(harmless_scores) / len(harmless_scores)

    print(f"Saved {len(rows)} scored samples to {output_file}")
    print(f"Mean helpful score: {mean_helpful:.6f}")
    print(f"Mean harmless score: {mean_harmless:.6f}")


if __name__ == "__main__":
    main()
