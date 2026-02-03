#!/bin/bash
set -euo pipefail

# Resumable, single-GPU HelpSteer pipeline:
#   SFT (helpfulness) -> merge SFT LoRA -> DPO margin adapter (verbosity) -> MODPO (w-grid) -> gen -> score
#
# Run from the modpo repo root (the directory that contains `scripts/` and `src/`):
#   export PYTHONPATH=. CUDA_VISIBLE_DEVICES=0
#   bash scripts/modpo/helpsteer/run_pipeline_resumable.sh

# Optional conda activation (safe no-op if conda is unavailable)
if ! command -v conda >/dev/null 2>&1; then
  if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
  fi
fi
conda activate thesis >/dev/null 2>&1 || true

export PYTHONPATH=${PYTHONPATH:-.}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# -----------------------
# Configuration (override via env vars)
# -----------------------

BASE_MODEL=${BASE_MODEL:-"meta-llama/Llama-2-7b-hf"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"./outputs/helpsteer/v2"}

RUN_TAG=${RUN_TAG:-"helpsteer_v2"}

MAX_LENGTH=${MAX_LENGTH:-512}
GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-4}
EVAL_SIZE=${EVAL_SIZE:-300}

# Core objective scaling
BETA=${BETA:-0.1}
# MODPO treats the margin term as a "cost" (it is subtracted in the loss).
# - If you want the margin adapter to behave as a cost: keep this positive.
# - If you want to *reward* what the margin adapter prefers: set this negative.
MARGIN_BETA=${MARGIN_BETA:-0.1}

# Trade-off values (must be strictly between 0 and 1 for scripts that use w=(w, 1-w))
W_VALUES=${W_VALUES:-"0.1 0.9"}

# Training knobs (single-GPU safe defaults)
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}
GRAD_ACCUM=${GRAD_ACCUM:-8}
SFT_MAX_STEPS=${SFT_MAX_STEPS:-1000}
DPO_MAX_STEPS=${DPO_MAX_STEPS:-1000}
MODPO_MAX_STEPS=${MODPO_MAX_STEPS:-1000}
SAVE_STEPS=${SAVE_STEPS:-200}
EVAL_STEPS=${EVAL_STEPS:-200}
LOGGING_STEPS=${LOGGING_STEPS:-10}

# W&B metadata (optional; training scripts default to report_to="wandb")
export WANDB_PROJECT=${WANDB_PROJECT:-"modpo-helpsteer"}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-"${RUN_TAG}"}
export WANDB_TAGS=${WANDB_TAGS:-"helpsteer,modpo,resumable"}

LOGDIR="${OUTPUT_ROOT}/logs"
mkdir -p "${LOGDIR}"

last_ckpt() {
  ls -d "$1"/checkpoint-* 2>/dev/null | sort -V | tail -n1 || true
}

run_resumable() {
  local outdir="$1"; shift
  local ckpt

  if [ -d "$outdir/best_checkpoint" ]; then
    echo "[SKIP] best_checkpoint exists: $outdir/best_checkpoint"
    return 0
  fi

  ckpt=$(last_ckpt "$outdir")
  if [ -n "$ckpt" ]; then
    echo "[RESUME] $outdir from $ckpt"
    "$@" --resume-from-checkpoint "$ckpt"
    return 0
  fi

  if [ -d "$outdir" ] && [ "$(ls -A "$outdir" 2>/dev/null)" ]; then
    echo "[ERROR] $outdir exists but no checkpoint-* found; refusing to overwrite."
    exit 1
  fi

  echo "[RUN] $outdir fresh"
  "$@"
}

echo "=== HelpSteer MODPO resumable pipeline ==="
echo "BASE_MODEL=$BASE_MODEL"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "W_VALUES=$W_VALUES"
echo "BETA=$BETA MARGIN_BETA=$MARGIN_BETA"

# -----------------------
# 1) SFT (helpfulness)
# -----------------------

SFT_OUT="${OUTPUT_ROOT}/sft_helpfulness"
WANDB_JOB_TYPE="sft_helpfulness" \
run_resumable "${SFT_OUT}" \
accelerate launch scripts/examples/sft/sft.py \
  --base_model_name "${BASE_MODEL}" \
  --dataset_name "nvidia/HelpSteer-pairwise-helpfulness" \
  --generate-during-eval False \
  --max_length "${MAX_LENGTH}" \
  --training_args.output_dir "${SFT_OUT}" \
  --training_args.run_name "${RUN_TAG}_sft_helpfulness" \
  --training_args.max_steps "${SFT_MAX_STEPS}" \
  --training_args.per_device_train_batch_size "${TRAIN_BATCH_SIZE}" \
  --training_args.per_device_eval_batch_size "${EVAL_BATCH_SIZE}" \
  --training_args.gradient_accumulation_steps "${GRAD_ACCUM}" \
  --training_args.gradient_checkpointing \
  --training_args.logging_steps "${LOGGING_STEPS}" \
  --training_args.save_strategy steps \
  --training_args.save_steps "${SAVE_STEPS}" \
  --training_args.save_total_limit 5 \
  --training_args.evaluation_strategy steps \
  --training_args.eval_steps "${EVAL_STEPS}" \
  --training_args.load_best_model_at_end True \
  | tee -a "${LOGDIR}/sft_helpfulness.log"

SFT_CHECKPOINT="${SFT_OUT}/best_checkpoint"
SFT_MODEL="${SFT_CHECKPOINT}"

# If SFT produced a PEFT adapter, merge it into a full model for downstream scripts.
if [ -f "${SFT_CHECKPOINT}/adapter_config.json" ]; then
  SFT_MERGED="${SFT_OUT}/merged_checkpoint"
  if [ -f "${SFT_MERGED}/config.json" ]; then
    echo "[SKIP] merged SFT exists: ${SFT_MERGED}"
  else
    echo "[MERGE] SFT adapter -> full model: ${SFT_MERGED}"
    python src/tools/merge_peft_adapter.py \
      --adapter_model_name "${SFT_CHECKPOINT}" \
      --base_model_name "${BASE_MODEL}" \
      --dtype bf16 \
      --output_name "${SFT_MERGED}" \
      | tee -a "${LOGDIR}/merge_sft.log"
  fi
  SFT_MODEL="${SFT_MERGED}"
fi

echo "Using SFT model: ${SFT_MODEL}"

# -----------------------
# 2) Train margin adapter via DPO (verbosity)
# -----------------------

MARGIN_OUT="${OUTPUT_ROOT}/margin_verbosity_dpo"
WANDB_JOB_TYPE="margin_verbosity_dpo" \
run_resumable "${MARGIN_OUT}" \
accelerate launch scripts/examples/dpo/dpo.py \
  --sft_model_name "${SFT_MODEL}" \
  --dataset_name "nvidia/HelpSteer-pairwise-verbosity" \
  --beta "${BETA}" \
  --generate-during-eval False \
  --max_length "${MAX_LENGTH}" \
  --training_args.output_dir "${MARGIN_OUT}" \
  --training_args.run_name "${RUN_TAG}_margin_verbosity_dpo" \
  --training_args.max_steps "${DPO_MAX_STEPS}" \
  --training_args.per_device_train_batch_size "${TRAIN_BATCH_SIZE}" \
  --training_args.per_device_eval_batch_size "${EVAL_BATCH_SIZE}" \
  --training_args.gradient_accumulation_steps "${GRAD_ACCUM}" \
  --training_args.gradient_checkpointing \
  --training_args.logging_steps "${LOGGING_STEPS}" \
  --training_args.save_strategy steps \
  --training_args.save_steps "${SAVE_STEPS}" \
  --training_args.save_total_limit 5 \
  --training_args.evaluation_strategy steps \
  --training_args.eval_steps "${EVAL_STEPS}" \
  --training_args.load_best_model_at_end True \
  | tee -a "${LOGDIR}/margin_verbosity_dpo.log"

MARGIN_ADAPTER="${MARGIN_OUT}/best_checkpoint"

echo "=== Sanity check: margin adapter prefers chosen over rejected ==="
python scripts/modpo/helpsteer/utils/check_margin_adapter.py \
  --sft_model_name "${SFT_MODEL}" \
  --margin_adapter_model_name "${MARGIN_ADAPTER}" \
  --dataset_name "nvidia/HelpSteer-pairwise-verbosity" \
  --split validation \
  --beta "${BETA}" \
  --max_examples 256 \
  | tee -a "${LOGDIR}/check_margin_adapter.log"

# -----------------------
# 3) MODPO (helpfulness vs verbosity-margin)
# -----------------------

for w in ${W_VALUES}; do
  OUT="${OUTPUT_ROOT}/modpo_w${w}"
  WANDB_JOB_TYPE="modpo" \
  run_resumable "${OUT}" \
  accelerate launch scripts/modpo/ultrafeedback/modpo.py \
    --sft_model_name "${SFT_MODEL}" \
    --margin_reward_model_name "${MARGIN_ADAPTER}" \
    --dataset_name "nvidia/HelpSteer-pairwise-helpfulness" \
    --w "${w}" \
    --beta "${BETA}" \
    --margin_beta "${MARGIN_BETA}" \
    --generate-during-eval False \
    --max_length "${MAX_LENGTH}" \
    --training_args.output_dir "${OUT}" \
    --training_args.run_name "${RUN_TAG}_modpo_w${w}" \
    --training_args.max_steps "${MODPO_MAX_STEPS}" \
    --training_args.per_device_train_batch_size "${TRAIN_BATCH_SIZE}" \
    --training_args.per_device_eval_batch_size "${EVAL_BATCH_SIZE}" \
    --training_args.gradient_accumulation_steps "${GRAD_ACCUM}" \
    --training_args.gradient_checkpointing \
    --training_args.logging_steps "${LOGGING_STEPS}" \
    --training_args.save_strategy steps \
    --training_args.save_steps "${SAVE_STEPS}" \
    --training_args.save_total_limit 5 \
    --training_args.evaluation_strategy steps \
    --training_args.eval_steps "${EVAL_STEPS}" \
    --training_args.load_best_model_at_end True \
    | tee -a "${LOGDIR}/modpo_w${w}.log"
done

# -----------------------
# 4) Generate outputs
# -----------------------

EVAL_DIR="${OUTPUT_ROOT}/eval"

echo "=== Generating SFT baseline ==="
python scripts/modpo/ultrafeedback/utils/gen.py \
  --sft_model_name "${SFT_MODEL}" \
  --dataset_name "nvidia/HelpSteer" \
  --eval_size "${EVAL_SIZE}" \
  --max_length "${MAX_LENGTH}" \
  --batch_size "${GEN_BATCH_SIZE}" \
  --output_dir "${EVAL_DIR}/gens_sft" \
  | tee -a "${LOGDIR}/gen_sft.log"

for w in ${W_VALUES}; do
  echo "=== Generating MODPO w=${w} ==="
  python scripts/modpo/ultrafeedback/utils/gen.py \
    --sft_model_name "${SFT_MODEL}" \
    --adapter_model_name "${OUTPUT_ROOT}/modpo_w${w}/best_checkpoint" \
    --dataset_name "nvidia/HelpSteer" \
    --eval_size "${EVAL_SIZE}" \
    --max_length "${MAX_LENGTH}" \
    --batch_size "${GEN_BATCH_SIZE}" \
    --output_dir "${EVAL_DIR}/gens_modpo_w${w}" \
    | tee -a "${LOGDIR}/gen_modpo_w${w}.log"
done

# -----------------------
# 5) Score outputs with implicit reward adapter
# -----------------------

echo "=== Scoring generations with the margin (verbosity) implicit reward adapter ==="
python scripts/modpo/helpsteer/utils/score_implicit_reward.py \
  --sft_model_name "${SFT_MODEL}" \
  --adapter_model_name "${MARGIN_ADAPTER}" \
  --beta "${BETA}" \
  --gens_dir "${EVAL_DIR}/gens_sft" \
  --label "sft" \
  $(for w in ${W_VALUES}; do echo --gens_dir "${EVAL_DIR}/gens_modpo_w${w}" --label "modpo_w${w}"; done) \
  | tee -a "${LOGDIR}/score_margin_adapter.log"

echo "=== Pipeline complete ==="
echo "Outputs: ${OUTPUT_ROOT}"
