#!/bin/bash
set -euo pipefail

# HH-RLHF Option1/Option2 pipeline (resumable):
#   SFT(helpful) -> DPO(harmless) -> MODPO(w-grid) -> generation (greedy)
#   -> Ray2333 scoring -> Option1/2 analysis
#
# Defaults to a fast sanity profile. Set RUN_PROFILE=full for the full grid run.

export PYTHONPATH=${PYTHONPATH:-.}

BASE_MODEL=${BASE_MODEL:-"PKU-Alignment/alpaca-7b-reproduced"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"./outputs/rq2/hh_opt12"}
RUN_TAG=${RUN_TAG:-"hh_opt12"}
RUN_PROFILE=${RUN_PROFILE:-"sanity"}  # sanity|full
SEED=${SEED:-42}
PRECISION=${PRECISION:-"bf16"}        # bf16|fp16|fp32

BETA=${BETA:-0.1}
MARGIN_BETA=${MARGIN_BETA:--0.1}

MAX_LENGTH=${MAX_LENGTH:-1024}
GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-8}

if [ -n "${W_VALUES:-}" ]; then
  if [ -z "${W_VALUES}" ]; then
    echo "[ERROR] W_VALUES is set but empty."
    exit 1
  fi
  read -r -a W_GRID <<< "${W_VALUES}"
else
  if [ "${RUN_PROFILE}" = "sanity" ]; then
    W_GRID=(0.2 0.8)
  else
    W_GRID=(0.1 0.2 0.4 0.6 0.8 0.9)
  fi
fi

if [ "${RUN_PROFILE}" = "sanity" ]; then
  EVAL_SIZE=${EVAL_SIZE:-200}
  SFT_MAX_STEPS=${SFT_MAX_STEPS:-150}
  DPO_MAX_STEPS=${DPO_MAX_STEPS:-150}
  MODPO_MAX_STEPS=${MODPO_MAX_STEPS:-150}
else
  EVAL_SIZE=${EVAL_SIZE:-700}
  SFT_MAX_STEPS=${SFT_MAX_STEPS:-1000}
  DPO_MAX_STEPS=${DPO_MAX_STEPS:-1000}
  MODPO_MAX_STEPS=${MODPO_MAX_STEPS:-1000}
fi

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-2}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-2}
GRAD_ACCUM=${GRAD_ACCUM:-4}
SAVE_STEPS=${SAVE_STEPS:-200}
EVAL_STEPS=${EVAL_STEPS:-200}
LOGGING_STEPS=${LOGGING_STEPS:-10}

export WANDB_PROJECT=${WANDB_PROJECT:-"thesis-modpo"}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-"hh_opt12_${RUN_TAG}_${RUN_PROFILE}"}
export WANDB_TAGS=${WANDB_TAGS:-"thesis,rq2,hh-rlhf,modpo,opt1,opt2,shared-output-conflict,greedy"}

LOGDIR="${OUTPUT_ROOT}/logs"
EVAL_DIR="${OUTPUT_ROOT}/eval"
SCORES_DIR="${EVAL_DIR}/scores_ray2333"
ANALYSIS_DIR="${EVAL_DIR}/analysis_opt12"
mkdir -p "${LOGDIR}" "${EVAL_DIR}" "${SCORES_DIR}" "${ANALYSIS_DIR}"

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
    echo "[ERROR] $outdir exists but has no checkpoint-* and no best_checkpoint; refusing overwrite."
    exit 1
  fi

  echo "[RUN] $outdir fresh"
  "$@"
}

echo "=== HH-RLHF Option1/2 Pipeline ==="
echo "BASE_MODEL=$BASE_MODEL"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "RUN_TAG=$RUN_TAG"
echo "RUN_PROFILE=$RUN_PROFILE"
echo "W_GRID=${W_GRID[*]}"
echo "EVAL_SIZE=$EVAL_SIZE"
echo "BETA=$BETA"
echo "MARGIN_BETA=$MARGIN_BETA"
echo "MAX_LENGTH=$MAX_LENGTH"
echo "PRECISION=$PRECISION"
echo "SEED=$SEED"
echo "WANDB_PROJECT=$WANDB_PROJECT"
echo "WANDB_RUN_GROUP=$WANDB_RUN_GROUP"
echo "WANDB_TAGS=$WANDB_TAGS"

# 1) SFT on helpful
SFT_OUT="${OUTPUT_ROOT}/sft_helpful"
WANDB_JOB_TYPE="sft_helpful" \
run_resumable "${SFT_OUT}" \
accelerate launch scripts/examples/sft/sft.py \
  --base-model-name "${BASE_MODEL}" \
  --dataset-name "Anthropic/hh-rlhf-helpful" \
  --precision "${PRECISION}" \
  --max-length "${MAX_LENGTH}" \
  --generate-during-eval False \
  --training_args.output_dir "${SFT_OUT}" \
  --training_args.run_name "${RUN_TAG}__sft_helpful__seed${SEED}" \
  --training_args.seed "${SEED}" \
  --training_args.max_steps "${SFT_MAX_STEPS}" \
  --training_args.per_device_train_batch_size "${TRAIN_BATCH_SIZE}" \
  --training_args.per_device_eval_batch_size "${EVAL_BATCH_SIZE}" \
  --training_args.gradient_accumulation_steps "${GRAD_ACCUM}" \
  --training_args.logging_steps "${LOGGING_STEPS}" \
  --training_args.save_strategy steps \
  --training_args.save_steps "${SAVE_STEPS}" \
  --training_args.save_total_limit 3 \
  --training_args.evaluation_strategy steps \
  --training_args.eval_steps "${EVAL_STEPS}" \
  --training_args.load_best_model_at_end True \
  | tee -a "${LOGDIR}/sft_helpful.log"

SFT_MODEL="${SFT_OUT}/best_checkpoint"
if [ -f "${SFT_MODEL}/adapter_config.json" ]; then
  SFT_MERGED="${SFT_OUT}/merged_checkpoint"
  if [ ! -f "${SFT_MERGED}/config.json" ]; then
    echo "[MERGE] SFT adapter -> full model: ${SFT_MERGED}"
    python src/tools/merge_peft_adapter.py \
      --adapter_model_name "${SFT_MODEL}" \
      --base_model_name "${BASE_MODEL}" \
      --dtype bf16 \
      --output_name "${SFT_MERGED}" \
      | tee -a "${LOGDIR}/merge_sft.log"
  fi
  SFT_MODEL="${SFT_MERGED}"
fi

echo "Using SFT model: ${SFT_MODEL}"

# 2) DPO margin adapter on harmless
RM_OUT="${OUTPUT_ROOT}/rm_harmless"
WANDB_JOB_TYPE="dpo_harmless" \
run_resumable "${RM_OUT}" \
accelerate launch scripts/examples/dpo/dpo.py \
  --sft-model-name "${SFT_MODEL}" \
  --dataset-name "Anthropic/hh-rlhf-harmless" \
  --precision "${PRECISION}" \
  --beta "${BETA}" \
  --max-length "${MAX_LENGTH}" \
  --generate-during-eval False \
  --training_args.output_dir "${RM_OUT}" \
  --training_args.run_name "${RUN_TAG}__dpo_harmless__seed${SEED}__beta${BETA}" \
  --training_args.seed "${SEED}" \
  --training_args.max_steps "${DPO_MAX_STEPS}" \
  --training_args.per_device_train_batch_size "${TRAIN_BATCH_SIZE}" \
  --training_args.per_device_eval_batch_size "${EVAL_BATCH_SIZE}" \
  --training_args.gradient_accumulation_steps "${GRAD_ACCUM}" \
  --training_args.logging_steps "${LOGGING_STEPS}" \
  --training_args.save_strategy steps \
  --training_args.save_steps "${SAVE_STEPS}" \
  --training_args.save_total_limit 3 \
  --training_args.evaluation_strategy steps \
  --training_args.eval_steps "${EVAL_STEPS}" \
  --training_args.load_best_model_at_end True \
  | tee -a "${LOGDIR}/dpo_harmless.log"

MARGIN_ADAPTER="${RM_OUT}/best_checkpoint"

# 3) MODPO w-grid
for w in "${W_GRID[@]}"; do
  OUT="${OUTPUT_ROOT}/modpo_w${w}"
  WANDB_JOB_TYPE="modpo" \
  run_resumable "${OUT}" \
  accelerate launch scripts/modpo/hh_rlhf/modpo.py \
    --sft-model-name "${SFT_MODEL}" \
    --margin-reward-model-name "${MARGIN_ADAPTER}" \
    --dataset-name "Anthropic/hh-rlhf-helpful" \
    --w "${w}" \
    --beta "${BETA}" \
    --margin-beta "${MARGIN_BETA}" \
    --precision "${PRECISION}" \
    --max-length "${MAX_LENGTH}" \
    --generate-during-eval False \
    --training_args.output_dir "${OUT}" \
    --training_args.run_name "${RUN_TAG}__modpo__w${w}__seed${SEED}__beta${BETA}__mb${MARGIN_BETA}" \
    --training_args.seed "${SEED}" \
    --training_args.max_steps "${MODPO_MAX_STEPS}" \
    --training_args.per_device_train_batch_size "${TRAIN_BATCH_SIZE}" \
    --training_args.per_device_eval_batch_size "${EVAL_BATCH_SIZE}" \
    --training_args.gradient_accumulation_steps "${GRAD_ACCUM}" \
    --training_args.logging_steps "${LOGGING_STEPS}" \
    --training_args.save_strategy steps \
    --training_args.save_steps "${SAVE_STEPS}" \
    --training_args.save_total_limit 3 \
    --training_args.evaluation_strategy steps \
    --training_args.eval_steps "${EVAL_STEPS}" \
    --training_args.load_best_model_at_end True \
    | tee -a "${LOGDIR}/modpo_w${w}.log"
done

# 4) Generation (6A: greedy defaults)
python scripts/modpo/hh_rlhf/utils/gen.py \
  --sft-model-name "${SFT_MODEL}" \
  --dataset-name "Anthropic/hh-rlhf-helpful" \
  --eval-size "${EVAL_SIZE}" \
  --max-length "${MAX_LENGTH}" \
  --batch-size "${GEN_BATCH_SIZE}" \
  --seed "${SEED}" \
  --output-dir "${EVAL_DIR}/gens_sft" \
  | tee -a "${LOGDIR}/gen_sft.log"

for w in "${W_GRID[@]}"; do
  python scripts/modpo/hh_rlhf/utils/gen.py \
    --sft-model-name "${SFT_MODEL}" \
    --adapter-model-name "${OUTPUT_ROOT}/modpo_w${w}/best_checkpoint" \
    --dataset-name "Anthropic/hh-rlhf-helpful" \
    --eval-size "${EVAL_SIZE}" \
    --max-length "${MAX_LENGTH}" \
    --batch-size "${GEN_BATCH_SIZE}" \
    --seed "${SEED}" \
    --output-dir "${EVAL_DIR}/gens_modpo_w${w}" \
    | tee -a "${LOGDIR}/gen_modpo_w${w}.log"
done

# 5) Prompt alignment validation across models
python scripts/modpo/helpsteer/utils/validate_eval_set.py \
  --gens_dir "${EVAL_DIR}/gens_sft" --label sft \
  $(for w in "${W_GRID[@]}"; do echo --gens_dir "${EVAL_DIR}/gens_modpo_w${w}" --label "modpo_w${w}"; done) \
  | tee -a "${LOGDIR}/validate_eval_set.log"

# 6) Score with Ray2333 helpful+harmless RMs
python scripts/modpo/hh_rlhf/utils/score_ray2333.py \
  --input-dir "${EVAL_DIR}/gens_sft" \
  --output-dir "${SCORES_DIR}/sft" \
  | tee -a "${LOGDIR}/score_sft.log"

for w in "${W_GRID[@]}"; do
  python scripts/modpo/hh_rlhf/utils/score_ray2333.py \
    --input-dir "${EVAL_DIR}/gens_modpo_w${w}" \
    --output-dir "${SCORES_DIR}/modpo_w${w}" \
    | tee -a "${LOGDIR}/score_modpo_w${w}.log"
done

# 7) Option1 + Option2 analysis
python scripts/modpo/hh_rlhf/utils/analyze_opt12_conflict_tradeoff.py \
  --scores-dir "${SCORES_DIR}/sft" --label sft \
  $(for w in "${W_GRID[@]}"; do echo --scores-dir "${SCORES_DIR}/modpo_w${w}" --label "modpo_w${w}"; done) \
  --output-dir "${ANALYSIS_DIR}" \
  | tee -a "${LOGDIR}/analysis_opt12.log"

echo "=== Pipeline complete ==="
echo "Scores: ${SCORES_DIR}"
echo "Analysis: ${ANALYSIS_DIR}"
