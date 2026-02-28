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

# Track whether protocol-critical vars were explicitly set by caller.
OUTPUT_ROOT_SET="${OUTPUT_ROOT+x}"
RUN_TAG_SET="${RUN_TAG+x}"
W_VALUES_SET="${W_VALUES+x}"
MAX_LENGTH_SET="${MAX_LENGTH+x}"
MAX_NEW_TOKENS_SET="${MAX_NEW_TOKENS+x}"
EVAL_SIZE_SET="${EVAL_SIZE+x}"
BETA_SET="${BETA+x}"
MARGIN_BETA_SET="${MARGIN_BETA+x}"

REQUIRE_EXPLICIT_CRITICALS=${REQUIRE_EXPLICIT_CRITICALS:-1}
ENFORCE_NEGATIVE_MARGIN_BETA=${ENFORCE_NEGATIVE_MARGIN_BETA:-0}
ENFORCE_PRD_W_GRID=${ENFORCE_PRD_W_GRID:-0}
PRD_W_VALUES=${PRD_W_VALUES:-"0.1 0.2 0.4 0.6 0.8 0.9"}

BASE_MODEL=${BASE_MODEL:-"meta-llama/Llama-2-7b-hf"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"./outputs/helpsteer/v2"}

RUN_TAG=${RUN_TAG:-"helpsteer_v2"}

TRAIN_MAX_LENGTH=${TRAIN_MAX_LENGTH:-512}
TRAIN_SEED=${TRAIN_SEED:-42}
MAX_LENGTH=${MAX_LENGTH:-4096}
MAX_INPUT_LENGTH=${MAX_INPUT_LENGTH:-1536}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2560}
GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-4}
EVAL_SIZE=${EVAL_SIZE:-300}
GEN_DO_SAMPLE=${GEN_DO_SAMPLE:-False}
GEN_TEMPERATURE=${GEN_TEMPERATURE:-1.0}
GEN_TOP_P=${GEN_TOP_P:-1.0}
GEN_REPETITION_PENALTY=${GEN_REPETITION_PENALTY:-1.0}
GEN_NO_REPEAT_NGRAM_SIZE=${GEN_NO_REPEAT_NGRAM_SIZE:-0}
DATA_NUM_PROC=${DATA_NUM_PROC:-1}
PRECISION=${PRECISION:-"bf16"}  # bf16|fp16|fp32 (forwarded to training scripts)

# Core objective scaling
BETA=${BETA:-0.1}
# Empirical sign lock for HelpSteer verbosity-control (2026-02-14 sign ablation):
# - In this pipeline, MARGIN_BETA > 0 increased verbosity.
# - In this pipeline, MARGIN_BETA < 0 reduced verbosity.
# Keep MARGIN_BETA negative when the objective is lower verbosity.
MARGIN_BETA=${MARGIN_BETA:--0.1}

# Trade-off values (must be strictly between 0 and 1 for scripts that use w=(w, 1-w))
W_VALUES=${W_VALUES:-"0.1 0.2 0.4 0.6 0.8 0.9"}

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

critical_nonempty=(OUTPUT_ROOT RUN_TAG W_VALUES MAX_LENGTH MAX_NEW_TOKENS EVAL_SIZE BETA MARGIN_BETA)
missing_nonempty=()
for v in "${critical_nonempty[@]}"; do
  if [ -z "${!v}" ]; then
    missing_nonempty+=("${v}")
  fi
done
if [ "${#missing_nonempty[@]}" -gt 0 ]; then
  echo "[ERROR] Critical vars must be non-empty: ${missing_nonempty[*]}"
  exit 1
fi

if [ "${REQUIRE_EXPLICIT_CRITICALS}" = "1" ]; then
  missing_explicit=()
  [ -z "${OUTPUT_ROOT_SET}" ] && missing_explicit+=(OUTPUT_ROOT)
  [ -z "${RUN_TAG_SET}" ] && missing_explicit+=(RUN_TAG)
  [ -z "${W_VALUES_SET}" ] && missing_explicit+=(W_VALUES)
  [ -z "${MAX_LENGTH_SET}" ] && missing_explicit+=(MAX_LENGTH)
  [ -z "${MAX_NEW_TOKENS_SET}" ] && missing_explicit+=(MAX_NEW_TOKENS)
  [ -z "${EVAL_SIZE_SET}" ] && missing_explicit+=(EVAL_SIZE)
  [ -z "${BETA_SET}" ] && missing_explicit+=(BETA)
  [ -z "${MARGIN_BETA_SET}" ] && missing_explicit+=(MARGIN_BETA)
  if [ "${#missing_explicit[@]}" -gt 0 ]; then
    echo "[ERROR] Missing explicit critical vars: ${missing_explicit[*]}"
    echo "Set each var in the same shell before launch. Example:"
    echo "  OUTPUT_ROOT=... RUN_TAG=... W_VALUES='0.1 0.2 0.4 0.6 0.8 0.9' MAX_LENGTH=4096 MAX_NEW_TOKENS=2560 EVAL_SIZE=300 BETA=0.1 MARGIN_BETA=-0.1 bash scripts/modpo/helpsteer/run_pipeline_resumable.sh"
    exit 1
  fi
fi

if ! [[ "${EVAL_SIZE}" =~ ^[0-9]+$ ]] || [ "${EVAL_SIZE}" -le 0 ]; then
  echo "[ERROR] EVAL_SIZE must be a positive integer. Got: ${EVAL_SIZE}"
  exit 1
fi

if ! [[ "${MAX_LENGTH}" =~ ^[0-9]+$ ]] || [ "${MAX_LENGTH}" -le 0 ]; then
  echo "[ERROR] MAX_LENGTH must be a positive integer. Got: ${MAX_LENGTH}"
  exit 1
fi
if ! [[ "${MAX_NEW_TOKENS}" =~ ^[0-9]+$ ]] || [ "${MAX_NEW_TOKENS}" -le 0 ]; then
  echo "[ERROR] MAX_NEW_TOKENS must be a positive integer. Got: ${MAX_NEW_TOKENS}"
  exit 1
fi
if ! [[ "${TRAIN_MAX_LENGTH}" =~ ^[0-9]+$ ]] || [ "${TRAIN_MAX_LENGTH}" -le 0 ]; then
  echo "[ERROR] TRAIN_MAX_LENGTH must be a positive integer. Got: ${TRAIN_MAX_LENGTH}"
  exit 1
fi
if ! [[ "${TRAIN_SEED}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] TRAIN_SEED must be a non-negative integer. Got: ${TRAIN_SEED}"
  exit 1
fi
if [ -n "${MAX_INPUT_LENGTH}" ]; then
  if ! [[ "${MAX_INPUT_LENGTH}" =~ ^[0-9]+$ ]] || [ "${MAX_INPUT_LENGTH}" -le 0 ]; then
    echo "[ERROR] MAX_INPUT_LENGTH must be empty or a positive integer. Got: ${MAX_INPUT_LENGTH}"
    exit 1
  fi
  if [ "${MAX_INPUT_LENGTH}" -gt "${MAX_LENGTH}" ]; then
    echo "[ERROR] MAX_INPUT_LENGTH (${MAX_INPUT_LENGTH}) cannot exceed MAX_LENGTH (${MAX_LENGTH})."
    exit 1
  fi
elif [ "${MAX_LENGTH}" -le "${MAX_NEW_TOKENS}" ]; then
  echo "[ERROR] MAX_LENGTH (${MAX_LENGTH}) must be > MAX_NEW_TOKENS (${MAX_NEW_TOKENS}) when MAX_INPUT_LENGTH is not set."
  exit 1
fi

if ! [[ "${GEN_DO_SAMPLE}" =~ ^(True|False)$ ]]; then
  echo "[ERROR] GEN_DO_SAMPLE must be True or False. Got: ${GEN_DO_SAMPLE}"
  exit 1
fi
if ! [[ "${GEN_NO_REPEAT_NGRAM_SIZE}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] GEN_NO_REPEAT_NGRAM_SIZE must be an integer >= 0. Got: ${GEN_NO_REPEAT_NGRAM_SIZE}"
  exit 1
fi
if ! [[ "${DATA_NUM_PROC}" =~ ^[0-9]+$ ]] || [ "${DATA_NUM_PROC}" -le 0 ]; then
  echo "[ERROR] DATA_NUM_PROC must be a positive integer. Got: ${DATA_NUM_PROC}"
  exit 1
fi

read -r -a W_GRID <<< "${W_VALUES}"
if [ "${#W_GRID[@]}" -eq 0 ]; then
  echo "[ERROR] W_VALUES produced an empty grid."
  exit 1
fi
for w in "${W_GRID[@]}"; do
  awk "BEGIN { exit !(${w} > 0 && ${w} < 1) }" || {
    echo "[ERROR] Each w must be strictly between 0 and 1. Invalid: ${w}"
    exit 1
  }
done

if [ "${ENFORCE_PRD_W_GRID}" = "1" ] && [ "${W_VALUES}" != "${PRD_W_VALUES}" ]; then
  echo "[ERROR] W_VALUES must match PRD grid when ENFORCE_PRD_W_GRID=1."
  echo "Expected: ${PRD_W_VALUES}"
  echo "Got:      ${W_VALUES}"
  exit 1
fi

if [ "${ENFORCE_NEGATIVE_MARGIN_BETA}" = "1" ]; then
  awk "BEGIN { exit !(${MARGIN_BETA} < 0) }" || {
    echo "[ERROR] MARGIN_BETA must be negative when ENFORCE_NEGATIVE_MARGIN_BETA=1. Got: ${MARGIN_BETA}"
    exit 1
  }
fi

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
echo "TRAIN_MAX_LENGTH=$TRAIN_MAX_LENGTH"
echo "TRAIN_SEED=$TRAIN_SEED"
echo "MAX_LENGTH=$MAX_LENGTH"
echo "MAX_INPUT_LENGTH=$MAX_INPUT_LENGTH"
echo "MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
echo "EVAL_SIZE=$EVAL_SIZE"
echo "BETA=$BETA MARGIN_BETA=$MARGIN_BETA"
echo "PRECISION=$PRECISION"
echo "GEN_DO_SAMPLE=$GEN_DO_SAMPLE GEN_TEMPERATURE=$GEN_TEMPERATURE GEN_TOP_P=$GEN_TOP_P GEN_REPETITION_PENALTY=$GEN_REPETITION_PENALTY GEN_NO_REPEAT_NGRAM_SIZE=$GEN_NO_REPEAT_NGRAM_SIZE"
echo "DATA_NUM_PROC=$DATA_NUM_PROC"
echo "REQUIRE_EXPLICIT_CRITICALS=$REQUIRE_EXPLICIT_CRITICALS ENFORCE_NEGATIVE_MARGIN_BETA=$ENFORCE_NEGATIVE_MARGIN_BETA ENFORCE_PRD_W_GRID=$ENFORCE_PRD_W_GRID"
if [ "${PREFLIGHT_ONLY:-0}" = "1" ]; then
  echo "PREFLIGHT_ONLY=1 -> exiting before training/generation."
  exit 0
fi

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
  --max_length "${TRAIN_MAX_LENGTH}" \
  --num_proc "${DATA_NUM_PROC}" \
  --precision "${PRECISION}" \
  --training_args.output_dir "${SFT_OUT}" \
  --training_args.run_name "${RUN_TAG}_sft_helpfulness" \
  --training_args.seed "${TRAIN_SEED}" \
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
  --max_length "${TRAIN_MAX_LENGTH}" \
  --num_proc "${DATA_NUM_PROC}" \
  --precision "${PRECISION}" \
  --training_args.output_dir "${MARGIN_OUT}" \
  --training_args.run_name "${RUN_TAG}_margin_verbosity_dpo" \
  --training_args.seed "${TRAIN_SEED}" \
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

for w in "${W_GRID[@]}"; do
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
    --max_length "${TRAIN_MAX_LENGTH}" \
    --num_proc "${DATA_NUM_PROC}" \
    --precision "${PRECISION}" \
    --training_args.output_dir "${OUT}" \
    --training_args.run_name "${RUN_TAG}_modpo_w${w}" \
    --training_args.seed "${TRAIN_SEED}" \
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
GEN_COMMON_ARGS=(
  --dataset_name "nvidia/HelpSteer"
  --eval_size "${EVAL_SIZE}"
  --max_length "${MAX_LENGTH}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --batch_size "${GEN_BATCH_SIZE}"
  --do_sample "${GEN_DO_SAMPLE}"
  --temperature "${GEN_TEMPERATURE}"
  --top_p "${GEN_TOP_P}"
  --repetition_penalty "${GEN_REPETITION_PENALTY}"
  --no_repeat_ngram_size "${GEN_NO_REPEAT_NGRAM_SIZE}"
)
if [ -n "${MAX_INPUT_LENGTH}" ]; then
  GEN_COMMON_ARGS+=(--max_input_length "${MAX_INPUT_LENGTH}")
fi

echo "=== Generating SFT baseline ==="
python scripts/modpo/ultrafeedback/utils/gen.py \
  --sft_model_name "${SFT_MODEL}" \
  "${GEN_COMMON_ARGS[@]}" \
  --output_dir "${EVAL_DIR}/gens_sft" \
  | tee -a "${LOGDIR}/gen_sft.log"

for w in "${W_GRID[@]}"; do
  echo "=== Generating MODPO w=${w} ==="
  python scripts/modpo/ultrafeedback/utils/gen.py \
    --sft_model_name "${SFT_MODEL}" \
    --adapter_model_name "${OUTPUT_ROOT}/modpo_w${w}/best_checkpoint" \
    "${GEN_COMMON_ARGS[@]}" \
    --output_dir "${EVAL_DIR}/gens_modpo_w${w}" \
    | tee -a "${LOGDIR}/gen_modpo_w${w}.log"
done

# -----------------------
# 5) Score outputs with implicit reward adapter
# -----------------------

echo "=== Scoring generations with the margin (verbosity) implicit reward adapter ==="
SCORE_ARGS=(--gens_dir "${EVAL_DIR}/gens_sft" --label "sft")
for w in "${W_GRID[@]}"; do
  SCORE_ARGS+=(--gens_dir "${EVAL_DIR}/gens_modpo_w${w}" --label "modpo_w${w}")
done
python scripts/modpo/helpsteer/utils/score_implicit_reward.py \
  --sft_model_name "${SFT_MODEL}" \
  --adapter_model_name "${MARGIN_ADAPTER}" \
  --beta "${BETA}" \
  "${SCORE_ARGS[@]}" \
  | tee -a "${LOGDIR}/score_margin_adapter.log"

echo "=== Pipeline complete ==="
echo "Outputs: ${OUTPUT_ROOT}"
