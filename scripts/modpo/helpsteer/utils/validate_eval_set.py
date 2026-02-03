import argparse
import glob
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Iterable, Optional

from src.data.configs import DEFAULT_PROMPT_TEMPLATE


@dataclass(frozen=True)
class Record:
    prompt_text: str
    prompt_hash: str
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


def _extract_raw_prompt(formatted_prompt: str, prompt_template: str) -> str:
    if "{raw_prompt}" not in prompt_template:
        return formatted_prompt
    prefix, suffix = prompt_template.split("{raw_prompt}", 1)
    if formatted_prompt.startswith(prefix) and formatted_prompt.endswith(suffix):
        start = len(prefix)
        end = len(formatted_prompt) - len(suffix) if suffix else len(formatted_prompt)
        return formatted_prompt[start:end]
    return formatted_prompt


def _record_from_obj(obj: dict, prompt_template: str) -> Optional[Record]:
    raw_prompt = obj.get("raw_prompt")
    if isinstance(raw_prompt, str) and raw_prompt.strip():
        prompt_text = raw_prompt
    else:
        prompt = obj.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            return None
        prompt_text = _extract_raw_prompt(prompt, prompt_template)
    prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:16]
    return Record(prompt_text=prompt_text, prompt_hash=prompt_hash, path="", line_no=0)


def _load_dir(dir_path: str, prompt_template: str, max_examples: Optional[int]) -> list[Record]:
    records: list[Record] = []
    for path, line_no, obj in _iter_jsonl(dir_path):
        rec = _record_from_obj(obj, prompt_template)
        if rec is None:
            raise KeyError(f"Missing 'raw_prompt' and 'prompt' at {path}:{line_no}")
        records.append(Record(prompt_text=rec.prompt_text, prompt_hash=rec.prompt_hash, path=path, line_no=line_no))
        if max_examples is not None and len(records) >= max_examples:
            break
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate that multiple generation directories contain the same eval prompt set "
        "(same order, same content), and report mismatches for debugging."
    )
    parser.add_argument(
        "--prompt_template",
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Template used to format prompts (for extracting raw_prompt from 'prompt' when needed).",
    )
    parser.add_argument(
        "--gens_dir",
        action="append",
        required=True,
        help="Directory containing one or more *.jsonl generation files. Repeat to compare multiple dirs.",
    )
    parser.add_argument("--label", action="append", help="Optional label(s) corresponding to --gens_dir.")
    parser.add_argument("--max_examples", type=int, default=None, help="Only read the first N examples per dir.")
    parser.add_argument(
        "--write_report",
        default=None,
        help="Optional path to write a JSON report of mismatches (useful for debugging).",
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
        recs = _load_dir(dir_path, args.prompt_template, args.max_examples)
        if not recs:
            raise ValueError(f"No records found in {dir_path} (*.jsonl empty?)")
        loaded[label] = recs

    ref_label = labels[0]
    ref = loaded[ref_label]
    report = {
        "reference": ref_label,
        "n_reference": len(ref),
        "comparisons": [],
    }

    print("=== Eval-set validation ===")
    print(f"reference={ref_label} n={len(ref)}")
    ok = True
    for label in labels[1:]:
        other = loaded[label]
        n = min(len(ref), len(other))
        mismatches = []
        for i in range(n):
            if ref[i].prompt_hash != other[i].prompt_hash:
                mismatches.append(
                    {
                        "index": i,
                        "ref": {
                            "hash": ref[i].prompt_hash,
                            "path": ref[i].path,
                            "line_no": ref[i].line_no,
                            "prompt_preview": ref[i].prompt_text[:200],
                        },
                        "other": {
                            "hash": other[i].prompt_hash,
                            "path": other[i].path,
                            "line_no": other[i].line_no,
                            "prompt_preview": other[i].prompt_text[:200],
                        },
                    }
                )
                if len(mismatches) >= 5:
                    break

        same_len = len(ref) == len(other)
        same_prefix = len(mismatches) == 0 and n == min(len(ref), len(other))
        is_ok = same_len and same_prefix
        ok = ok and is_ok

        print(f"- {label}: n={len(other)} same_len={same_len} first_mismatches={len(mismatches)}")
        if mismatches:
            first = mismatches[0]
            print(f"  first mismatch @ index={first['index']}")
            print(f"    ref  ({first['ref']['hash']}): {first['ref']['path']}:{first['ref']['line_no']}")
            print(f"    other({first['other']['hash']}): {first['other']['path']}:{first['other']['line_no']}")

        report["comparisons"].append(
            {
                "label": label,
                "n": len(other),
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
        raise SystemExit("FAILED: generation dirs do not share the same eval prompt set (see mismatch report above).")
    print("OK: all generation dirs share the same eval prompt set.")


if __name__ == "__main__":
    main()

