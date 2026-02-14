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

## Empirical Sign Result (HelpSteer, 2026-02-14)

Controlled A/B runs were executed with identical recipe and opposite sign (`+0.1` vs `-0.1`) at `w=0.2` and `w=0.8`.

- In this HelpSteer pipeline, `MARGIN_BETA=+0.1` increased verbosity signal.
- In this HelpSteer pipeline, `MARGIN_BETA=-0.1` reduced verbosity signal.
- Therefore, for the objective "maximize helpfulness, minimize verbosity", use **negative** `MARGIN_BETA`.

The default in `run_pipeline_resumable.sh` is set accordingly to `MARGIN_BETA=-0.1`.

## Quick start (server)

From the repo root (server):

Goal: launch a PRD-aligned full run with explicit protocol-critical variables.

```bash
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=0
export PRECISION=bf16  # bf16|fp16|fp32
export OUTPUT_ROOT=./outputs/helpsteer/rq1_hs_tradeoff_2026-02-14_full
export RUN_TAG=rq1_hs_tradeoff_2026-02-14_full
export W_VALUES="0.1 0.2 0.4 0.6 0.8 0.9"
export TRAIN_MAX_LENGTH=512
export MAX_LENGTH=4096
export MAX_INPUT_LENGTH=1536
export MAX_NEW_TOKENS=2560
export EVAL_SIZE=300
export BETA=0.1
export MARGIN_BETA=-0.1
export REQUIRE_EXPLICIT_CRITICALS=1
export ENFORCE_NEGATIVE_MARGIN_BETA=1
export ENFORCE_PRD_W_GRID=1
bash scripts/modpo/helpsteer/run_pipeline_resumable.sh
```

By default, the pipeline uses `meta-llama/Llama-2-7b-hf` as the base model. This repo is gated on Hugging Face,
so make sure you have access and are logged in (e.g., `huggingface-cli login`) before running.

The script is designed to be safe to rerun: it will **skip** completed stages and **resume** from the last
checkpoint when possible.

## Protocol-critical preflight behavior

- `run_pipeline_resumable.sh` now checks critical vars before launch.
- With `REQUIRE_EXPLICIT_CRITICALS=1` (default), these vars must be explicitly set by caller:
  - `OUTPUT_ROOT`
  - `RUN_TAG`
  - `W_VALUES`
  - `MAX_LENGTH`
  - `MAX_NEW_TOKENS`
  - `EVAL_SIZE`
  - `BETA`
  - `MARGIN_BETA`
- Optional safety gates:
  - `ENFORCE_NEGATIVE_MARGIN_BETA=1` fails fast if `MARGIN_BETA >= 0`.
  - `ENFORCE_PRD_W_GRID=1` fails fast if `W_VALUES` differs from PRD grid.
- Optional dry run:
  - `PREFLIGHT_ONLY=1` runs validation + variable echo, then exits before training.

## Training vs generation length controls

- `TRAIN_MAX_LENGTH` controls SFT/DPO/MODPO training sequence length.
- `MAX_LENGTH`, `MAX_INPUT_LENGTH`, and `MAX_NEW_TOKENS` control generation only.
- This decoupling avoids accidentally changing training recipe while raising generation budget.

## Generation decode controls (tracked)

`scripts/modpo/ultrafeedback/utils/gen.py` now exposes decode knobs in CLI and logs effective values:

- `--do_sample`
- `--temperature`
- `--top_p`
- `--repetition_penalty`
- `--no_repeat_ngram_size`

Goal: antidegen-style deterministic decode (recommended pilot before full run freeze).

```bash
export GEN_DO_SAMPLE=False
export GEN_REPETITION_PENALTY=1.10
export GEN_NO_REPEAT_NGRAM_SIZE=4
```

Practical recommendation for corrected-sign HelpSteer runs:

- Run a short pilot (`EVAL_SIZE=64`) with and without antidegen knobs.
- Compare capping/truncation and prompt alignment.
- Freeze one decode protocol before the full run.

## ASHA tuning (MODPO-only, fixed `w=0.7`)

Use this when you want to tune shared MODPO hyperparameters first, then freeze them for a paper-style
multi-weight run.

- Scope: MODPO stage only (reuses existing SFT + margin adapter checkpoints)
- Fixed weight: `w=0.7`
- Pruning objective: `eval_rewards/margins` (maximize)
- Sanity metric also logged: `eval_loss`
- Tuned params by default: `learning_rate`, `weight_decay`, `warmup_ratio`, `beta`, `margin_beta` (signed; for verbosity-reduction studies, keep it negative)
- Runtime dependency: `optuna` with Journal storage support (`JournalStorage` / `JournalFileBackend`)

Run workers in `tmux` so SSH disconnects do not kill jobs. Launch one worker process per GPU:

```bash
STUDY=hs_modpo_w07_asha_v1
OUT=./outputs/helpsteer/asha
SFT=./outputs/helpsteer/rq1_hs_tradeoff_2026-02-06/sft_helpfulness/merged_checkpoint
MARGIN=./outputs/helpsteer/rq1_hs_tradeoff_2026-02-06/margin_verbosity_dpo/best_checkpoint
RUNG_STEPS=300,600,1000
export PYTHONPATH=.

echo "STUDY=$STUDY"
echo "OUT=$OUT"
echo "SFT=$SFT"
echo "MARGIN=$MARGIN"
echo "RUNG_STEPS=$RUNG_STEPS"

python scripts/modpo/helpsteer/optuna_asha_modpo.py --study_name "$STUDY" --output_root "$OUT" --rung_steps "$RUNG_STEPS" --sft_model_name "$SFT" --margin_reward_model_name "$MARGIN" --worker_id gpu0 --gpu_id 0 &
python scripts/modpo/helpsteer/optuna_asha_modpo.py --study_name "$STUDY" --output_root "$OUT" --rung_steps "$RUNG_STEPS" --sft_model_name "$SFT" --margin_reward_model_name "$MARGIN" --worker_id gpu1 --gpu_id 1 &
python scripts/modpo/helpsteer/optuna_asha_modpo.py --study_name "$STUDY" --output_root "$OUT" --rung_steps "$RUNG_STEPS" --sft_model_name "$SFT" --margin_reward_model_name "$MARGIN" --worker_id gpu2 --gpu_id 2 &
wait
python scripts/modpo/helpsteer/optuna_asha_modpo.py --study_name "$STUDY" --output_root "$OUT" --rung_steps "$RUNG_STEPS" --sft_model_name "$SFT" --margin_reward_model_name "$MARGIN" --summarize_only
```

Default ASHA budget is `24` total trials with rung steps `300,600,1000`.

## Paper-style evaluation (recommended)

The training pipeline only scores generations with the **implicit reward adapter** (logp_adapter − logp_base),
which is useful for debugging but is **not** an independent oracle metric.

For a paper-style evaluation on **HelpSteer native dimensions**, score the generated responses with an external
RM that exposes the HelpSteer heads (e.g., ArmoRM), and then compare models in the resulting objective space.

After generation is complete:

```bash
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=0
export OUTPUT_ROOT=./outputs/helpsteer/v2  # or your run directory

# Optional: quick smoke-test with fewer samples
export ARMORM_DEBUG_MAX_SAMPLES=64

bash scripts/modpo/helpsteer/run_armorm_eval.sh
```

Note: ArmoRM often requires a newer Transformers/Accelerate stack than this repo’s pinned training deps; keep
evaluation in a dedicated environment if needed (see `packages/modpo/setup_armorm_env.sh`).

## Files

- `scripts/modpo/helpsteer/run_pipeline_resumable.sh`: end-to-end resumable pipeline
- `scripts/modpo/helpsteer/run_armorm_eval.sh`: evaluate generations with ArmoRM + print HelpSteer-head summary
- `scripts/modpo/helpsteer/utils/check_margin_adapter.py`: sanity-checks that the DPO margin adapter prefers
  chosen over rejected on its training dimension
- `scripts/modpo/helpsteer/utils/score_implicit_reward.py`: scores generations with the implicit reward adapter
  and prints summary stats
- `scripts/modpo/helpsteer/utils/validate_eval_set.py`: verifies that multiple generation dirs share the same prompt set
- `scripts/modpo/helpsteer/utils/summarize_armorm_helpsteer.py`: prints a compact table for ArmoRM HelpSteer heads
- `scripts/modpo/helpsteer/optuna_asha_modpo.py`: ASHA-style Optuna tuner for MODPO-only hyperparameter search
