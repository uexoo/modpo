#!/usr/bin/env python3
"""Build a CI-class robustness matrix from UF summary JSON files.

Each input summary is expected to contain ``pairwise_vs_sft`` rows with:
  - label
  - delta_help_vs_sft_mean / ci_lo / ci_hi
  - delta_truth_vs_sft_mean / ci_lo / ci_hi

The script classifies each row and aggregates class counts by ``w``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable


CLASS_ORDER = [
    "strict_tradeoff",
    "help_up_truth_flat",
    "help_flat_truth_down",
    "strict_reverse",
    "other",
]


@dataclass(frozen=True)
class Condition:
    seed: str
    decode: str
    summary_path: str


def _parse_condition(raw: str) -> Condition:
    parts = raw.split("::", 2)
    if len(parts) != 3:
        raise ValueError(
            f"Invalid --condition value: {raw!r}. Expected format: seed::decode::/abs/path/to/summary.json"
        )
    seed, decode, summary_path = (x.strip() for x in parts)
    if not seed or not decode or not summary_path:
        raise ValueError(f"Invalid --condition value (empty field): {raw!r}")
    return Condition(seed=seed, decode=decode, summary_path=summary_path)


def _extract_w(label: str) -> float:
    match = re.search(r"_w([0-9]+(?:\.[0-9]+)?)$", label)
    if not match:
        match = re.search(r"w([0-9]+(?:\.[0-9]+)?)$", label)
    if not match:
        raise ValueError(f"Could not extract w from label: {label!r}")
    return float(match.group(1))


def _classify(
    help_ci_lo: float,
    help_ci_hi: float,
    truth_ci_lo: float,
    truth_ci_hi: float,
) -> str:
    help_up = help_ci_lo > 0
    help_down = help_ci_hi < 0
    help_flat = help_ci_lo <= 0 <= help_ci_hi

    truth_up = truth_ci_lo > 0
    truth_down = truth_ci_hi < 0
    truth_flat = truth_ci_lo <= 0 <= truth_ci_hi

    if help_up and truth_down:
        return "strict_tradeoff"
    if help_up and truth_flat:
        return "help_up_truth_flat"
    if help_flat and truth_down:
        return "help_flat_truth_down"
    if help_down and truth_up:
        return "strict_reverse"
    return "other"


def _iter_pair_rows(payload: dict) -> Iterable[dict]:
    rows = payload.get("pairwise_vs_sft", [])
    if isinstance(rows, dict):
        for label, row in rows.items():
            if not isinstance(row, dict):
                continue
            item = dict(row)
            item.setdefault("label", label)
            yield item
        return
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict):
                yield row


def _load_records(condition: Condition) -> list[dict]:
    with open(condition.summary_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    records: list[dict] = []
    for row in _iter_pair_rows(payload):
        label = str(row["label"])
        w = _extract_w(label)

        d_help_mean = float(row["delta_help_vs_sft_mean"])
        d_help_ci_lo = float(row["delta_help_vs_sft_ci_lo"])
        d_help_ci_hi = float(row["delta_help_vs_sft_ci_hi"])
        d_truth_mean = float(row["delta_truth_vs_sft_mean"])
        d_truth_ci_lo = float(row["delta_truth_vs_sft_ci_lo"])
        d_truth_ci_hi = float(row["delta_truth_vs_sft_ci_hi"])

        cls = _classify(
            help_ci_lo=d_help_ci_lo,
            help_ci_hi=d_help_ci_hi,
            truth_ci_lo=d_truth_ci_lo,
            truth_ci_hi=d_truth_ci_hi,
        )
        records.append(
            {
                "seed": condition.seed,
                "decode": condition.decode,
                "label": label,
                "w": w,
                "d_help_mean": d_help_mean,
                "d_help_ci_lo": d_help_ci_lo,
                "d_help_ci_hi": d_help_ci_hi,
                "d_truth_mean": d_truth_mean,
                "d_truth_ci_lo": d_truth_ci_lo,
                "d_truth_ci_hi": d_truth_ci_hi,
                "class": cls,
                "summary_path": condition.summary_path,
            }
        )
    return records


def _sort_records(records: list[dict]) -> list[dict]:
    def keyfn(r: dict) -> tuple[float, str, str, str]:
        return (float(r["w"]), str(r["seed"]), str(r["decode"]), str(r["label"]))

    return sorted(records, key=keyfn)


def _build_summary_by_w(records: list[dict]) -> dict[str, dict[str, int]]:
    by_w: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        w_key = f"{float(r['w']):.1f}"
        by_w[w_key][str(r["class"])] += 1

    out: dict[str, dict[str, int]] = {}
    for w_key in sorted(by_w, key=float):
        counts = by_w[w_key]
        row = {c: int(counts.get(c, 0)) for c in CLASS_ORDER}
        row["total"] = int(sum(row[c] for c in CLASS_ORDER))
        out[w_key] = row
    return out


def _fmt4(value: float) -> str:
    return f"{value:.4f}"


def _write_markdown(path: str, summary_by_w: dict[str, dict[str, int]], records: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# CI Robustness Matrix\n\n")
        f.write("| w | strict_tradeoff | help_up_truth_flat | help_flat_truth_down | strict_reverse | other | total |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for w_key in sorted(summary_by_w, key=float):
            row = summary_by_w[w_key]
            f.write(
                f"| {w_key} | {row['strict_tradeoff']} | {row['help_up_truth_flat']} | "
                f"{row['help_flat_truth_down']} | {row['strict_reverse']} | {row['other']} | {row['total']} |\n"
            )

        f.write("\n## Per-condition rows\n\n")
        f.write("| seed | decode | label | w | d_help [lo,hi] | d_truth [lo,hi] | class |\n")
        f.write("|---|---|---|---:|---|---|---|\n")
        for r in records:
            f.write(
                f"| {r['seed']} | {r['decode']} | {r['label']} | {r['w']:.1f} | "
                f"{_fmt4(r['d_help_mean'])} [{_fmt4(r['d_help_ci_lo'])},{_fmt4(r['d_help_ci_hi'])}] | "
                f"{_fmt4(r['d_truth_mean'])} [{_fmt4(r['d_truth_ci_lo'])},{_fmt4(r['d_truth_ci_hi'])}] | "
                f"{r['class']} |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build UF CI robustness matrix from summary JSON files.")
    parser.add_argument(
        "--condition",
        action="append",
        required=True,
        help="Repeated condition spec: seed::decode::/abs/path/to/summary.json",
    )
    parser.add_argument("--output_json", required=True, help="Path to output matrix JSON.")
    parser.add_argument("--output_md", default=None, help="Path to output markdown table. Optional.")
    args = parser.parse_args()

    conditions = [_parse_condition(c) for c in args.condition]
    all_records: list[dict] = []
    for condition in conditions:
        all_records.extend(_load_records(condition))

    records = _sort_records(all_records)
    summary_by_w = _build_summary_by_w(records)
    payload = {
        "records": records,
        "summary_by_w": summary_by_w,
    }

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    output_md = args.output_md
    if output_md is None:
        root, _ = os.path.splitext(args.output_json)
        output_md = f"{root}.md"
    _write_markdown(output_md, summary_by_w=summary_by_w, records=records)

    print(f"wrote_json={args.output_json}")
    print(f"wrote_md={output_md}")


if __name__ == "__main__":
    main()
