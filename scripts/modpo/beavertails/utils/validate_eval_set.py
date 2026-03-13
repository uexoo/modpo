import argparse
import glob
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Iterable, Optional

from datasets import load_dataset


DEFAULT_PROMPT_TEMPLATE = "BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT:"


@dataclass(frozen=True)
class ExpectedPrompt:
    raw_prompt: str
    formatted_prompt: str
    prompt_hash: str


@dataclass(frozen=True)
class Record:
    kind: str
    content: str
    path: str
    line_no: int


def _iter_jsonl(dir_path: str) -> Iterable[tuple[str, int, dict]]:
    for path in sorted(glob.glob(os.path.join(dir_path, "*.jsonl"))):
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                yield path, i, json.loads(line)


def _record_from_obj(obj: dict) -> Optional[Record]:
    if isinstance(obj.get("raw_prompt"), str) and obj["raw_prompt"].strip():
        return Record(kind="raw_prompt", content=obj["raw_prompt"], path="", line_no=0)
    if isinstance(obj.get("prompt"), str) and obj["prompt"].strip():
        return Record(kind="prompt", content=obj["prompt"], path="", line_no=0)
    if isinstance(obj.get("prompt_response"), str) and obj["prompt_response"].strip():
        return Record(kind="prompt_response", content=obj["prompt_response"], path="", line_no=0)
    return None


def _load_dir(dir_path: str, max_examples: Optional[int]) -> list[Record]:
    records: list[Record] = []
    for path, line_no, obj in _iter_jsonl(dir_path):
        rec = _record_from_obj(obj)
        if rec is None:
            raise KeyError(
                f"Missing one of 'raw_prompt', 'prompt', or 'prompt_response' at {path}:{line_no}"
            )
        records.append(Record(kind=rec.kind, content=rec.content, path=path, line_no=line_no))
        if max_examples is not None and len(records) >= max_examples:
            break
    return records


def _base_dataset_name(dataset_name: str) -> str:
    if dataset_name.startswith("PKU-Alignment/PKU-SafeRLHF-10K-"):
        return "PKU-Alignment/PKU-SafeRLHF-10K"
    if dataset_name.startswith("PKU-Alignment/PKU-SafeRLHF-"):
        return "PKU-Alignment/PKU-SafeRLHF"
    if dataset_name in {"PKU-Alignment/PKU-SafeRLHF", "PKU-Alignment/PKU-SafeRLHF-10K"}:
        return dataset_name
    raise ValueError(
        "Unsupported dataset_name. Expected one of the PKU-SafeRLHF variants, "
        f"got: {dataset_name}"
    )


def _load_expected_prompts(
    dataset_name: str,
    prompt_template: str,
    n_expected: int,
) -> list[ExpectedPrompt]:
    base_name = _base_dataset_name(dataset_name)
    dataset = load_dataset(base_name, split="train").train_test_split(test_size=0.1, seed=0)["test"]
    if n_expected > len(dataset):
        raise ValueError(
            f"Requested {n_expected} prompts but validation split for {base_name} only has {len(dataset)} rows."
        )
    prompts = dataset.select(range(n_expected))["prompt"]
    expected = []
    for raw_prompt in prompts:
        prompt_hash = hashlib.sha256(raw_prompt.encode("utf-8")).hexdigest()[:16]
        expected.append(
            ExpectedPrompt(
                raw_prompt=raw_prompt,
                formatted_prompt=prompt_template.format(raw_prompt=raw_prompt),
                prompt_hash=prompt_hash,
            )
        )
    return expected


def _validate_record(record: Record, expected: ExpectedPrompt) -> tuple[bool, dict]:
    if record.kind == "raw_prompt":
        ok = record.content == expected.raw_prompt
        observed_hash = hashlib.sha256(record.content.encode("utf-8")).hexdigest()[:16]
        details = {
            "record_kind": record.kind,
            "observed_hash": observed_hash,
            "expected_hash": expected.prompt_hash,
            "observed_preview": record.content[:200],
            "expected_preview": expected.raw_prompt[:200],
        }
        return ok, details

    if record.kind == "prompt":
        ok = record.content == expected.formatted_prompt
        observed_hash = hashlib.sha256(record.content.encode("utf-8")).hexdigest()[:16]
        expected_hash = hashlib.sha256(expected.formatted_prompt.encode("utf-8")).hexdigest()[:16]
        details = {
            "record_kind": record.kind,
            "observed_hash": observed_hash,
            "expected_hash": expected_hash,
            "observed_preview": record.content[:200],
            "expected_preview": expected.formatted_prompt[:200],
        }
        return ok, details

    ok = record.content.startswith(expected.formatted_prompt)
    details = {
        "record_kind": record.kind,
        "expected_hash": expected.prompt_hash,
        "observed_preview": record.content[:200],
        "expected_preview": expected.formatted_prompt[:200],
    }
    return ok, details


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate that BeaverTails generation directories match the expected PKU-SafeRLHF "
            "validation prompt set and prompt order. This is intended for RQ3 auditability."
        )
    )
    parser.add_argument(
        "--dataset_name",
        default="PKU-Alignment/PKU-SafeRLHF-10K-safer",
        help=(
            "Dataset variant used for generation. Prompts are loaded from the corresponding "
            "PKU-SafeRLHF train split with the same validation split convention as the training code."
        ),
    )
    parser.add_argument(
        "--prompt_template",
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Prompt template used during generation.",
    )
    parser.add_argument(
        "--gens_dir",
        action="append",
        required=True,
        help="Directory containing one or more *.jsonl generation files. Repeat to compare multiple dirs.",
    )
    parser.add_argument("--label", action="append", help="Optional label(s) corresponding to --gens_dir.")
    parser.add_argument(
        "--eval_size",
        type=int,
        default=None,
        help=(
            "Optional number of prompts to validate. If omitted, uses the number of records in the "
            "reference generation directory."
        ),
    )
    parser.add_argument("--max_examples", type=int, default=None, help="Only read the first N examples per dir.")
    parser.add_argument(
        "--write_report",
        default=None,
        help="Optional path to write a JSON report of mismatches.",
    )
    args = parser.parse_args()

    labels = args.label
    if labels is not None and len(labels) != len(args.gens_dir):
        raise ValueError("If provided, --label must be repeated exactly as many times as --gens_dir.")
    if labels is None:
        labels = [os.path.basename(d.rstrip("/")) for d in args.gens_dir]

    loaded: dict[str, list[Record]] = {}
    for label, dir_path in zip(labels, args.gens_dir):
        if not os.path.isdir(dir_path):
            raise FileNotFoundError(f"Not a directory: {dir_path}")
        recs = _load_dir(dir_path, args.max_examples)
        if not recs:
            raise ValueError(f"No records found in {dir_path} (*.jsonl empty?)")
        loaded[label] = recs

    ref_label = labels[0]
    ref_records = loaded[ref_label]
    n_expected = args.eval_size if args.eval_size is not None else len(ref_records)
    expected_prompts = _load_expected_prompts(
        dataset_name=args.dataset_name,
        prompt_template=args.prompt_template,
        n_expected=n_expected,
    )

    report = {
        "dataset_name": args.dataset_name,
        "prompt_template": args.prompt_template,
        "reference": ref_label,
        "n_reference": len(ref_records),
        "n_expected": len(expected_prompts),
        "comparisons": [],
    }

    print("=== BeaverTails eval-set validation ===")
    print(f"dataset_name={args.dataset_name}")
    print(f"reference={ref_label} n={len(ref_records)} expected={len(expected_prompts)}")

    ok = True
    for label in labels:
        records = loaded[label]
        n = min(len(records), len(expected_prompts))
        mismatches = []
        for i in range(n):
            record_ok, details = _validate_record(records[i], expected_prompts[i])
            if not record_ok:
                mismatches.append(
                    {
                        "index": i,
                        "path": records[i].path,
                        "line_no": records[i].line_no,
                        **details,
                    }
                )
                if len(mismatches) >= 5:
                    break

        same_len = len(records) == len(expected_prompts)
        is_ok = same_len and len(mismatches) == 0
        ok = ok and is_ok

        print(f"- {label}: n={len(records)} same_len={same_len} first_mismatches={len(mismatches)}")
        if mismatches:
            first = mismatches[0]
            print(f"  first mismatch @ index={first['index']}")
            print(f"    kind={first['record_kind']} path={first['path']}:{first['line_no']}")
            print(f"    expected_hash={first['expected_hash']}")

        report["comparisons"].append(
            {
                "label": label,
                "n": len(records),
                "same_len": same_len,
                "first_mismatches": mismatches,
            }
        )

    if args.write_report:
        os.makedirs(os.path.dirname(args.write_report) or ".", exist_ok=True)
        with open(args.write_report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"wrote_report={args.write_report}")

    if not ok:
        raise SystemExit(
            "FAILED: generation dirs do not match the expected BeaverTails eval prompt set "
            "(see mismatch report above)."
        )
    print("OK: all generation dirs match the expected BeaverTails eval prompt set.")


if __name__ == "__main__":
    main()
