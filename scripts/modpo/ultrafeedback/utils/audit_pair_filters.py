import argparse
import json
import os
from typing import Optional

from src.data.raw_data.ultrafeedback import UltraFeedbackRDP, ultrafeedback_transform_to_preference


def _load_pair_dataset(path: str, split: str, num_proc: int, max_prompts: Optional[int]):
    rdp = UltraFeedbackRDP(path=path, num_proc=num_proc, sanity_check=False)
    raw = rdp._get_raw_dataset(split=split)  # noqa: SLF001 - intentional internal use for diagnostics
    if max_prompts is not None and max_prompts > 0:
        raw = raw.select(range(min(len(raw), max_prompts)))
    pairs = raw.map(
        ultrafeedback_transform_to_preference,
        batched=True,
        num_proc=num_proc,
        remove_columns=raw.column_names,
    )
    return pairs


def _count(dataset, fn):
    return int(sum(1 for x in dataset if fn(x)))


def main():
    parser = argparse.ArgumentParser(
        description="Audit UltraFeedback pair-filter settings before long training runs."
    )
    parser.add_argument("--path", default="OpenBMB/UltraFeedback")
    parser.add_argument("--split", default="train", choices=["train", "validation"])
    parser.add_argument("--dimension", default="truthfulness")
    parser.add_argument("--anchor_dimension", default="helpfulness")
    parser.add_argument("--min_gap", type=float, default=2.0)
    parser.add_argument("--min_anchor_gap", type=float, default=2.0)
    parser.add_argument("--require_disagreement", action="store_true")
    parser.add_argument("--require_agreement", action="store_true")
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument("--max_prompts", type=int, default=0)
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    if args.min_gap < 0:
        raise ValueError("--min_gap must be >= 0.")
    if args.min_anchor_gap < 0:
        raise ValueError("--min_anchor_gap must be >= 0.")
    if args.require_disagreement and args.require_agreement:
        raise ValueError("--require_disagreement and --require_agreement cannot both be set.")

    max_prompts = args.max_prompts if args.max_prompts > 0 else None
    ds = _load_pair_dataset(
        path=args.path,
        split=args.split,
        num_proc=args.num_proc,
        max_prompts=max_prompts,
    )

    dim_cid = f"{args.dimension}_chosen_id"
    dim_gap = f"{args.dimension}_gap"
    anc_cid = f"{args.anchor_dimension}_chosen_id"
    anc_gap = f"{args.anchor_dimension}_gap"

    total_pairs = len(ds)
    dim_non_tie = _count(ds, lambda x: x[dim_cid] != -1)
    dim_gap_ok = _count(ds, lambda x: x[dim_cid] != -1 and x[dim_gap] >= float(args.min_gap))
    anchor_non_tie = _count(ds, lambda x: x[anc_cid] != -1)
    anchor_gap_ok = _count(ds, lambda x: x[anc_cid] != -1 and x[anc_gap] >= float(args.min_anchor_gap))
    joint_gap_ok = _count(
        ds,
        lambda x: x[dim_cid] != -1
        and x[dim_gap] >= float(args.min_gap)
        and x[anc_cid] != -1
        and x[anc_gap] >= float(args.min_anchor_gap),
    )
    disagreement = _count(
        ds,
        lambda x: x[dim_cid] != -1
        and x[dim_gap] >= float(args.min_gap)
        and x[anc_cid] != -1
        and x[anc_gap] >= float(args.min_anchor_gap)
        and x[dim_cid] != x[anc_cid],
    )
    agreement = max(0, joint_gap_ok - disagreement)

    disagreement_rate = (float(disagreement) / float(joint_gap_ok)) if joint_gap_ok else 0.0
    agreement_rate = (float(agreement) / float(joint_gap_ok)) if joint_gap_ok else 0.0
    if args.require_disagreement:
        final_pairs = disagreement
    elif args.require_agreement:
        final_pairs = agreement
    else:
        final_pairs = joint_gap_ok
    final_rate = (float(final_pairs) / float(total_pairs)) if total_pairs else 0.0

    out = {
        "path": args.path,
        "split": args.split,
        "dimension": args.dimension,
        "anchor_dimension": args.anchor_dimension,
        "min_gap": args.min_gap,
        "min_anchor_gap": args.min_anchor_gap,
        "require_disagreement": bool(args.require_disagreement),
        "require_agreement": bool(args.require_agreement),
        "num_proc": args.num_proc,
        "max_prompts": args.max_prompts,
        "total_pairs": total_pairs,
        "dim_non_tie_pairs": dim_non_tie,
        "dim_gap_ok_pairs": dim_gap_ok,
        "anchor_non_tie_pairs": anchor_non_tie,
        "anchor_gap_ok_pairs": anchor_gap_ok,
        "joint_gap_ok_pairs": joint_gap_ok,
        "joint_disagreement_pairs": disagreement,
        "joint_disagreement_rate": disagreement_rate,
        "joint_agreement_pairs": agreement,
        "joint_agreement_rate": agreement_rate,
        "final_pairs": final_pairs,
        "final_pair_rate": final_rate,
    }

    print("=== UltraFeedback Pair Filter Audit ===")
    for k, v in out.items():
        print(f"{k}={v}")

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"wrote_json={args.output_json}")


if __name__ == "__main__":
    main()
