# HelpSteer MODPO Pipeline (SFT → DPO margin adapter → MODPO)

This directory provides a **fully reproducible** and **resumable** pipeline for running MODPO on the
[nvidia/HelpSteer](https://huggingface.co/datasets/nvidia/HelpSteer) dataset.

The pipeline follows the repo’s intended pattern:

1. **SFT** (reference / preference objective model)
2. **DPO** on a *margin dataset* to produce a **LoRA adapter** that is treated as an **implicit reward model**
3. **MODPO** with a `w` grid (trade-off between preference vs. margin objective)
4. **Generate** outputs
5. **Score** with the implicit reward adapter (and basic length stats)

## Important: HelpSteer “verbosity” semantics

HelpSteer attribute ratings are on a 0–4 scale, where **higher means better** for each attribute (including
`verbosity` = “amount of detail relative to what is asked”). If you use a DPO-trained verbosity adapter as a
*cost* in MODPO, you are explicitly pushing the policy **away from what HelpSteer defines as “better verbosity”**.

If your actual goal is “shorter/less verbose outputs”, consider using a direct **length penalty** margin objective
(see `scripts/modpo/summarize_w_length_penalty/`) instead of HelpSteer’s `verbosity` attribute.

## Choosing `MARGIN_BETA` (direction)

MODPO subtracts the margin term in the loss. With an implicit reward adapter, the wrapper returns:

`reward(x,y) = beta * (logp_adapter(y|x) - logp_base(y|x))`

That means:

- `MARGIN_BETA > 0`: treat the margin adapter’s preference as a **cost** (penalize what the adapter prefers)
- `MARGIN_BETA < 0`: treat the margin adapter’s preference as a **reward** (encourage what the adapter prefers)

The pipeline runs `scripts/modpo/helpsteer/utils/check_margin_adapter.py` after training the margin adapter to
verify that, on its training dimension, **chosen scores higher than rejected** under the implicit reward.

## Quick start (server)

From the repo root (server):

```bash
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=0
bash scripts/modpo/helpsteer/run_pipeline_resumable.sh
```

The script is designed to be safe to rerun: it will **skip** completed stages and **resume** from the last
checkpoint when possible.

## Files

- `scripts/modpo/helpsteer/run_pipeline_resumable.sh`: end-to-end resumable pipeline
- `scripts/modpo/helpsteer/utils/check_margin_adapter.py`: sanity-checks that the DPO margin adapter prefers
  chosen over rejected on its training dimension
- `scripts/modpo/helpsteer/utils/score_implicit_reward.py`: scores generations with the implicit reward adapter
  and prints summary stats
