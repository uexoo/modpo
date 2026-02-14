#!/bin/bash
set -euo pipefail

# Resumable UltraFeedback truthfulness pipeline:
#   SFT (helpfulness) -> merge SFT LoRA -> DPO margin adapter (truthfulness)
#   -> MODPO sweep -> generation -> prompt alignment -> ArmoRM scoring
#   -> implicit reward sanity scoring -> optional sign-ablation verdict
#
# Design notes:
# - This script follows the strict, resumable style used in the HelpSteer pipeline.
# - HelpSteer sign findings (2026-02-14) are objective-specific to verbosity reduction.
#   For UltraFeedback truthfulness, historical runs used positive margin sign.
# - We still support a pilot sign-ablation mode for empirical sign lock before full runs.

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

OUTPUT_ROOT_SET="${OUTPUT_ROOT+x}"
RUN_TAG_SET="${RUN_TAG+x}"
W_VALUES_SET="${W_VALUES+x}"
MAX_LENGTH_SET="${MAX_LENGTH+x}"
MAX_NEW_TOKENS_SET="${MAX_NEW_TOKENS+x}"
EVAL_SIZE_SET="${EVAL_SIZE+x}"
BETA_SET="${BETA+x}"
MARGIN_BETA_SET="${MARGIN_BETA+x}"

RUN_PROFILE=${RUN_PROFILE:-pilot} # smoke|pilot|full
REQUIRE_EXPLICIT_CRITICALS=${REQUIRE_EXPLICIT_CRITICALS:-1}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}

RUN_SIGN_ABLATION_SET="${RUN_SIGN_ABLATION+x}"
if [ -z "${RUN_SIGN_ABLATION_SET}" ]; then
  if [ "${RUN_PROFILE}" = "full" ]; then
    RUN_SIGN_ABLATION=0
  else
    RUN_SIGN_ABLATION=1
  fi
else
  RUN_SIGN_ABLATION=${RUN_SIGN_ABLATION}
fi

case "${RUN_PROFILE}" in
  smoke)
    PROFILE_W_VALUES="0.2 0.8"
    PROFILE_SFT_STEPS=40
    PROFILE_DPO_STEPS=40
    PROFILE_MODPO_STEPS=80
    PROFILE_EVAL_SIZE=64
    PROFILE_MAX_LENGTH=2048
    PROFILE_MAX_INPUT_LENGTH=1024
    PROFILE_MAX_NEW_TOKENS=1024
    ;;
  pilot)
    PROFILE_W_VALUES="0.2 0.8"
    PROFILE_SFT_STEPS=300
    PROFILE_DPO_STEPS=300
    PROFILE_MODPO_STEPS=400
    PROFILE_EVAL_SIZE=128
    PROFILE_MAX_LENGTH=4096
    PROFILE_MAX_INPUT_LENGTH=1536
    PROFILE_MAX_NEW_TOKENS=2560
    ;;
  full)
    PROFILE_W_VALUES="0.1 0.2 0.4 0.6 0.8 0.9"
    PROFILE_SFT_STEPS=1000
    PROFILE_DPO_STEPS=1000
    PROFILE_MODPO_STEPS=1000
    PROFILE_EVAL_SIZE=300
    PROFILE_MAX_LENGTH=4096
    PROFILE_MAX_INPUT_LENGTH=1536
    PROFILE_MAX_NEW_TOKENS=2560
    ;;
  *)
    echo "[ERROR] RUN_PROFILE must be one of: smoke, pilot, full. Got: ${RUN_PROFILE}"
    exit 1
    ;;
esac

BASE_MODEL=${BASE_MODEL:-"PKU-Alignment/alpaca-7b-reproduced"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"./outputs/ultrafeedback/rq1_truthfulness_${RUN_PROFILE}"}
RUN_TAG=${RUN_TAG:-"uf_truthfulness_${RUN_PROFILE}"}

TRAIN_MAX_LENGTH=${TRAIN_MAX_LENGTH:-512}
MAX_LENGTH=${MAX_LENGTH:-${PROFILE_MAX_LENGTH}}
MAX_INPUT_LENGTH=${MAX_INPUT_LENGTH:-${PROFILE_MAX_INPUT_LENGTH}}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-${PROFILE_MAX_NEW_TOKENS}}
GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-4}
EVAL_SIZE=${EVAL_SIZE:-${PROFILE_EVAL_SIZE}}
GEN_DO_SAMPLE=${GEN_DO_SAMPLE:-False}
GEN_TEMPERATURE=${GEN_TEMPERATURE:-1.0}
GEN_TOP_P=${GEN_TOP_P:-1.0}
GEN_REPETITION_PENALTY=${GEN_REPETITION_PENALTY:-1.10}
GEN_NO_REPEAT_NGRAM_SIZE=${GEN_NO_REPEAT_NGRAM_SIZE:-4}
PRECISION=${PRECISION:-"bf16"} # bf16|fp16|fp32

# Core objective scaling:
# - Primary objective: UltraFeedback helpfulness (preference dataset).
# - Margin objective: UltraFeedback truthfulness (margin adapter).
# Historical UF runs used positive sign for truth-related margin objectives.
BETA=${BETA:-0.1}
MARGIN_BETA=${MARGIN_BETA:-0.1}

W_VALUES=${W_VALUES:-${PROFILE_W_VALUES}}
SIGN_VALUES=${SIGN_VALUES:-"0.1 -0.1"}
SIGN_W_VALUES=${SIGN_W_VALUES:-${W_VALUES}}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}
GRAD_ACCUM=${GRAD_ACCUM:-8}
SFT_MAX_STEPS=${SFT_MAX_STEPS:-${PROFILE_SFT_STEPS}}
DPO_MAX_STEPS=${DPO_MAX_STEPS:-${PROFILE_DPO_STEPS}}
MODPO_MAX_STEPS=${MODPO_MAX_STEPS:-${PROFILE_MODPO_STEPS}}
SAVE_STEPS=${SAVE_STEPS:-200}
EVAL_STEPS=${EVAL_STEPS:-200}
LOGGING_STEPS=${LOGGING_STEPS:-10}

ARMORM_MODEL_PATH=${ARMORM_MODEL_PATH:-"RLHFlow/ArmoRM-Llama3-8B-v0.1"}
ARMORM_BATCH_SIZE=${ARMORM_BATCH_SIZE:-8}
ARMORM_DEBUG_MAX_SAMPLES=${ARMORM_DEBUG_MAX_SAMPLES:-}
IMPLICIT_MAX_EXAMPLES=${IMPLICIT_MAX_EXAMPLES:-}

SMOKE_BOOTSTRAP_ITERS=${SMOKE_BOOTSTRAP_ITERS:-2000}
SMOKE_BOOTSTRAP_SEED=${SMOKE_BOOTSTRAP_SEED:-42}
SMOKE_MAX_CAP_RATE=${SMOKE_MAX_CAP_RATE:-0.20}
ENFORCE_CAP_RATE_GATE=${ENFORCE_CAP_RATE_GATE:-1}

export WANDB_PROJECT=${WANDB_PROJECT:-"modpo-ultrafeedback"}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-"${RUN_TAG}"}
export WANDB_TAGS=${WANDB_TAGS:-"ultrafeedback,truthfulness,modpo,${RUN_PROFILE}"}

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
    echo "Set each var in the same shell before launch."
    exit 1
  fi
fi

for intv in EVAL_SIZE MAX_LENGTH MAX_NEW_TOKENS TRAIN_MAX_LENGTH TRAIN_BATCH_SIZE EVAL_BATCH_SIZE GRAD_ACCUM GEN_BATCH_SIZE; do
  if ! [[ "${!intv}" =~ ^[0-9]+$ ]] || [ "${!intv}" -le 0 ]; then
    echo "[ERROR] ${intv} must be a positive integer. Got: ${!intv}"
    exit 1
  fi
done
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
  echo "[ERROR] MAX_LENGTH (${MAX_LENGTH}) must be > MAX_NEW_TOKENS (${MAX_NEW_TOKENS}) when MAX_INPUT_LENGTH is empty."
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
if ! [[ "${RUN_SIGN_ABLATION}" =~ ^(0|1)$ ]]; then
  echo "[ERROR] RUN_SIGN_ABLATION must be 0 or 1. Got: ${RUN_SIGN_ABLATION}"
  exit 1
fi
if ! [[ "${ENFORCE_CAP_RATE_GATE}" =~ ^(0|1)$ ]]; then
  echo "[ERROR] ENFORCE_CAP_RATE_GATE must be 0 or 1. Got: ${ENFORCE_CAP_RATE_GATE}"
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

SIGN_GRID=()
SIGN_W_GRID=()
if [ "${RUN_SIGN_ABLATION}" = "1" ]; then
  read -r -a SIGN_GRID <<< "${SIGN_VALUES}"
  read -r -a SIGN_W_GRID <<< "${SIGN_W_VALUES}"
  if [ "${#SIGN_GRID[@]}" -lt 2 ]; then
    echo "[ERROR] RUN_SIGN_ABLATION=1 requires at least two SIGN_VALUES. Got: ${SIGN_VALUES}"
    exit 1
  fi
  if [ "${#SIGN_W_GRID[@]}" -eq 0 ]; then
    echo "[ERROR] SIGN_W_VALUES produced an empty grid."
    exit 1
  fi
  for w in "${SIGN_W_GRID[@]}"; do
    awk "BEGIN { exit !(${w} > 0 && ${w} < 1) }" || {
      echo "[ERROR] Each sign-ablation w must be in (0,1). Invalid: ${w}"
      exit 1
    }
  done
fi

sign_to_tag() {
  local sign="$1"
  local prefix="pos"
  if awk "BEGIN { exit !(${sign} < 0) }"; then
    prefix="neg"
  fi
  local mag="${sign#-}"
  mag="${mag#+}"
  mag="${mag//./p}"
  echo "${prefix}${mag}"
}

last_ckpt() {
  ls -d "$1"/checkpoint-* 2>/dev/null | sort -V | tail -n1 || true
}

run_resumable() {
  local outdir="$1"; shift
  local logfile="$1"; shift
  local ckpt
  mkdir -p "$(dirname "${logfile}")"
  if [ -d "${outdir}/best_checkpoint" ]; then
    echo "[SKIP] best_checkpoint exists: ${outdir}/best_checkpoint" | tee -a "${logfile}"
    return 0
  fi
  ckpt="$(last_ckpt "${outdir}")"
  if [ -n "${ckpt}" ]; then
    echo "[RESUME] ${outdir} from ${ckpt}" | tee -a "${logfile}"
    "$@" --resume-from-checkpoint "${ckpt}" | tee -a "${logfile}"
    return 0
  fi
  if [ -d "${outdir}" ] && [ "$(ls -A "${outdir}" 2>/dev/null)" ]; then
    echo "[ERROR] ${outdir} exists but no checkpoint-* found; refusing to overwrite." | tee -a "${logfile}"
    exit 1
  fi
  echo "[RUN] ${outdir} fresh" | tee -a "${logfile}"
  "$@" | tee -a "${logfile}"
}

run_smoke_preflight() {
  echo "=== Smoke preflight: import/version/schema checks ==="
  python - <<'PY'
import importlib
from src.data.configs import DATASET_CONFIGS

mods = ["torch", "transformers", "datasets", "peft", "trl", "accelerate", "tyro"]
print("package_versions:")
for name in mods:
    m = importlib.import_module(name)
    print(f"  {name}={getattr(m, '__version__', 'unknown')}")

required = [
    "OpenBMB/UltraFeedback-helpfulness",
    "OpenBMB/UltraFeedback-truthfulness",
]
for key in required:
    if key not in DATASET_CONFIGS:
        raise KeyError(f"Missing dataset key: {key}")
    rdp = DATASET_CONFIGS[key](sanity_check=True, num_proc=1)
    train = rdp.get_preference_dataset(split="train")
    val = rdp.get_preference_dataset(split="validation")
    if len(train) == 0 or len(val) == 0:
        raise RuntimeError(f"Empty dataset after preprocessing for {key}.")
    print(f"  dataset_ok {key}: train={len(train)} val={len(val)}")
print("smoke_preflight=ok")
PY

  python -m py_compile \
    scripts/modpo/ultrafeedback/modpo.py \
    scripts/modpo/ultrafeedback/utils/gen.py \
    scripts/modpo/ultrafeedback/utils/score_armorm.py \
    scripts/modpo/helpsteer/utils/validate_eval_set.py \
    scripts/modpo/helpsteer/utils/score_implicit_reward.py \
    scripts/modpo/ultrafeedback/utils/analyze_truthfulness_sign_pilot.py
}

echo "=== UltraFeedback truthfulness resumable pipeline ==="
echo "RUN_PROFILE=${RUN_PROFILE}"
echo "BASE_MODEL=${BASE_MODEL}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "RUN_TAG=${RUN_TAG}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "W_VALUES=${W_VALUES}"
echo "RUN_SIGN_ABLATION=${RUN_SIGN_ABLATION}"
echo "SIGN_VALUES=${SIGN_VALUES}"
echo "SIGN_W_VALUES=${SIGN_W_VALUES}"
echo "TRAIN_MAX_LENGTH=${TRAIN_MAX_LENGTH}"
echo "MAX_LENGTH=${MAX_LENGTH}"
echo "MAX_INPUT_LENGTH=${MAX_INPUT_LENGTH}"
echo "MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "EVAL_SIZE=${EVAL_SIZE}"
echo "BETA=${BETA} MARGIN_BETA=${MARGIN_BETA}"
echo "PRECISION=${PRECISION}"
echo "GEN_DO_SAMPLE=${GEN_DO_SAMPLE} GEN_TEMPERATURE=${GEN_TEMPERATURE} GEN_TOP_P=${GEN_TOP_P} GEN_REPETITION_PENALTY=${GEN_REPETITION_PENALTY} GEN_NO_REPEAT_NGRAM_SIZE=${GEN_NO_REPEAT_NGRAM_SIZE}"
echo "ARMORM_MODEL_PATH=${ARMORM_MODEL_PATH} ARMORM_BATCH_SIZE=${ARMORM_BATCH_SIZE}"
echo "SMOKE_MAX_CAP_RATE=${SMOKE_MAX_CAP_RATE} ENFORCE_CAP_RATE_GATE=${ENFORCE_CAP_RATE_GATE}"
echo "REQUIRE_EXPLICIT_CRITICALS=${REQUIRE_EXPLICIT_CRITICALS}"

if [ "${PREFLIGHT_ONLY}" = "1" ]; then
  run_smoke_preflight
  echo "PREFLIGHT_ONLY=1 -> exiting before training/generation."
  exit 0
fi

run_smoke_preflight | tee -a "${LOGDIR}/smoke_preflight.log"

# 1) SFT helpfulness
SFT_OUT="${OUTPUT_ROOT}/sft_helpfulness"
run_resumable "${SFT_OUT}" "${LOGDIR}/sft_helpfulness.log" \
  accelerate launch scripts/examples/sft/sft.py \
    --base_model_name "${BASE_MODEL}" \
    --dataset_name "OpenBMB/UltraFeedback-helpfulness" \
    --generate-during-eval False \
    --max_length "${TRAIN_MAX_LENGTH}" \
    --precision "${PRECISION}" \
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
    --training_args.load_best_model_at_end True

SFT_CHECKPOINT="${SFT_OUT}/best_checkpoint"
SFT_MODEL="${SFT_CHECKPOINT}"
if [ -f "${SFT_CHECKPOINT}/adapter_config.json" ]; then
  SFT_MERGED="${SFT_OUT}/merged_checkpoint"
  if [ -f "${SFT_MERGED}/config.json" ]; then
    echo "[SKIP] merged SFT exists: ${SFT_MERGED}" | tee -a "${LOGDIR}/merge_sft.log"
  else
    echo "[MERGE] SFT adapter -> full model: ${SFT_MERGED}" | tee -a "${LOGDIR}/merge_sft.log"
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

# 2) Margin adapter via DPO truthfulness
MARGIN_OUT="${OUTPUT_ROOT}/margin_truthfulness_dpo"
run_resumable "${MARGIN_OUT}" "${LOGDIR}/margin_truthfulness_dpo.log" \
  accelerate launch scripts/examples/dpo/dpo.py \
    --sft_model_name "${SFT_MODEL}" \
    --dataset_name "OpenBMB/UltraFeedback-truthfulness" \
    --beta "${BETA}" \
    --generate-during-eval False \
    --max_length "${TRAIN_MAX_LENGTH}" \
    --precision "${PRECISION}" \
    --training_args.output_dir "${MARGIN_OUT}" \
    --training_args.run_name "${RUN_TAG}_margin_truthfulness_dpo" \
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
    --training_args.load_best_model_at_end True

MARGIN_ADAPTER="${MARGIN_OUT}/best_checkpoint"

# 3) MODPO sweep
MODEL_LABELS=()
if [ "${RUN_SIGN_ABLATION}" = "1" ]; then
  echo "=== MODPO sign-ablation mode enabled ==="
  for sign in "${SIGN_GRID[@]}"; do
    for w in "${SIGN_W_GRID[@]}"; do
      sign_tag="$(sign_to_tag "${sign}")"
      label="modpo_sign${sign_tag}_w${w}"
      out="${OUTPUT_ROOT}/${label}"
      run_resumable "${out}" "${LOGDIR}/${label}.log" \
        accelerate launch scripts/modpo/ultrafeedback/modpo.py \
          --sft_model_name "${SFT_MODEL}" \
          --margin_reward_model_name "${MARGIN_ADAPTER}" \
          --dataset_name "OpenBMB/UltraFeedback-helpfulness" \
          --w "${w}" \
          --beta "${BETA}" \
          --margin_beta "${sign}" \
          --generate-during-eval False \
          --max_length "${TRAIN_MAX_LENGTH}" \
          --precision "${PRECISION}" \
          --training_args.output_dir "${out}" \
          --training_args.run_name "${RUN_TAG}_${label}" \
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
          --training_args.load_best_model_at_end True
      MODEL_LABELS+=("${label}")
    done
  done
else
  echo "=== MODPO single-sign mode: MARGIN_BETA=${MARGIN_BETA} ==="
  for w in "${W_GRID[@]}"; do
    label="modpo_w${w}"
    out="${OUTPUT_ROOT}/${label}"
    run_resumable "${out}" "${LOGDIR}/${label}.log" \
      accelerate launch scripts/modpo/ultrafeedback/modpo.py \
        --sft_model_name "${SFT_MODEL}" \
        --margin_reward_model_name "${MARGIN_ADAPTER}" \
        --dataset_name "OpenBMB/UltraFeedback-helpfulness" \
        --w "${w}" \
        --beta "${BETA}" \
        --margin_beta "${MARGIN_BETA}" \
        --generate-during-eval False \
        --max_length "${TRAIN_MAX_LENGTH}" \
        --precision "${PRECISION}" \
        --training_args.output_dir "${out}" \
        --training_args.run_name "${RUN_TAG}_${label}" \
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
        --training_args.load_best_model_at_end True
    MODEL_LABELS+=("${label}")
  done
fi

# 4) Generate outputs
EVAL_DIR="${OUTPUT_ROOT}/eval"
GEN_COMMON_ARGS=(
  --dataset_name "OpenBMB/UltraFeedback-helpfulness"
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

if [ ! -s "${EVAL_DIR}/gens_sft/00001-of-00001.jsonl" ]; then
  echo "=== Generating SFT baseline ==="
  python scripts/modpo/ultrafeedback/utils/gen.py \
    --sft_model_name "${SFT_MODEL}" \
    "${GEN_COMMON_ARGS[@]}" \
    --output_dir "${EVAL_DIR}/gens_sft" \
    | tee -a "${LOGDIR}/gen_sft.log"
else
  echo "[SKIP] generation exists: ${EVAL_DIR}/gens_sft/00001-of-00001.jsonl"
fi

for label in "${MODEL_LABELS[@]}"; do
  out_file="${EVAL_DIR}/gens_${label}/00001-of-00001.jsonl"
  if [ -s "${out_file}" ]; then
    echo "[SKIP] generation exists: ${out_file}"
    continue
  fi
  echo "=== Generating ${label} ==="
  python scripts/modpo/ultrafeedback/utils/gen.py \
    --sft_model_name "${SFT_MODEL}" \
    --adapter_model_name "${OUTPUT_ROOT}/${label}/best_checkpoint" \
    "${GEN_COMMON_ARGS[@]}" \
    --output_dir "${EVAL_DIR}/gens_${label}" \
    | tee -a "${LOGDIR}/gen_${label}.log"
done

# 5) Validate prompt alignment before scoring
VAL_ARGS=(
  --gens_dir "${EVAL_DIR}/gens_sft" --label sft
)
for label in "${MODEL_LABELS[@]}"; do
  VAL_ARGS+=(--gens_dir "${EVAL_DIR}/gens_${label}" --label "${label}")
done
python scripts/modpo/helpsteer/utils/validate_eval_set.py "${VAL_ARGS[@]}" \
  --write_report "${EVAL_DIR}/eval_set_validation.json" \
  | tee -a "${LOGDIR}/validate_eval_set.log"

# 6) ArmoRM scoring (per model dir)
mkdir -p "${EVAL_DIR}/scores_armorm"
ALL_LABELS=(sft "${MODEL_LABELS[@]}")
for label in "${ALL_LABELS[@]}"; do
  input_dir="${EVAL_DIR}/gens_${label}"
  output_dir="${EVAL_DIR}/scores_armorm/${label}"
  score_args=(
    --input_dir "${input_dir}"
    --output_dir "${output_dir}"
    --model_path "${ARMORM_MODEL_PATH}"
    --batch_size "${ARMORM_BATCH_SIZE}"
  )
  if [ -n "${ARMORM_DEBUG_MAX_SAMPLES}" ]; then
    score_args+=(--debug_max_samples "${ARMORM_DEBUG_MAX_SAMPLES}")
  fi
  python scripts/modpo/ultrafeedback/utils/score_armorm.py "${score_args[@]}" \
    | tee -a "${LOGDIR}/score_armorm_${label}.log"
done

# 7) Implicit reward sanity scoring
implicit_args=(
  --sft_model_name "${SFT_MODEL}"
  --adapter_model_name "${MARGIN_ADAPTER}"
  --beta "${BETA}"
  --gens_dir "${EVAL_DIR}/gens_sft" --label sft
)
for label in "${MODEL_LABELS[@]}"; do
  implicit_args+=(--gens_dir "${EVAL_DIR}/gens_${label}" --label "${label}")
done
if [ -n "${IMPLICIT_MAX_EXAMPLES}" ]; then
  implicit_args+=(--max_examples "${IMPLICIT_MAX_EXAMPLES}")
fi
python scripts/modpo/helpsteer/utils/score_implicit_reward.py "${implicit_args[@]}" \
  | tee -a "${LOGDIR}/score_implicit_reward.log"

# 8) Generation cap diagnostics gate
UF_DIAG_EVAL_DIR="${EVAL_DIR}" \
UF_DIAG_LABELS="$(IFS=,; echo "${ALL_LABELS[*]}")" \
UF_DIAG_SFT_MODEL="${SFT_MODEL}" \
UF_DIAG_MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
UF_DIAG_CAP_THRESH="${SMOKE_MAX_CAP_RATE}" \
UF_DIAG_ENFORCE="${ENFORCE_CAP_RATE_GATE}" \
python - <<'PY'
import glob
import json
import os
import statistics
from transformers import AutoTokenizer

eval_dir = os.environ["UF_DIAG_EVAL_DIR"]
labels = [x for x in os.environ["UF_DIAG_LABELS"].split(",") if x]
sft_model = os.environ["UF_DIAG_SFT_MODEL"]
max_new = int(os.environ["UF_DIAG_MAX_NEW_TOKENS"])
cap_thresh = float(os.environ["UF_DIAG_CAP_THRESH"])
enforce = os.environ["UF_DIAG_ENFORCE"] == "1"

tokenizer = AutoTokenizer.from_pretrained(sft_model, trust_remote_code=True)
rows = []
cap_fail = []
for label in labels:
    path_glob = os.path.join(eval_dir, f"gens_{label}", "*.jsonl")
    files = sorted(glob.glob(path_glob))
    if not files:
        raise FileNotFoundError(f"No generation files for label={label} ({path_glob})")
    tok_lens = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                resp = obj.get("response", "")
                tok_lens.append(len(tokenizer(resp, add_special_tokens=False)["input_ids"]))
    if not tok_lens:
        raise RuntimeError(f"No generations found for label={label}")
    cap_rate = sum(1 for x in tok_lens if x >= (max_new - 1)) / len(tok_lens)
    rows.append(
        {
            "label": label,
            "n": len(tok_lens),
            "mean_toks": statistics.mean(tok_lens),
            "p95_toks": sorted(tok_lens)[int(0.95 * (len(tok_lens) - 1))],
            "cap_rate": cap_rate,
        }
    )
    if cap_rate > cap_thresh:
        cap_fail.append((label, cap_rate))

print("=== Generation Length Diagnostics ===")
for r in rows:
    print(
        f"{r['label']}: n={r['n']} mean_toks={r['mean_toks']:.1f} "
        f"p95_toks={r['p95_toks']} cap_rate={r['cap_rate']:.4f}"
    )

if cap_fail and enforce:
    msg = ", ".join([f"{label}={rate:.4f}" for label, rate in cap_fail])
    raise SystemExit(f"FAILED cap-rate gate (>{cap_thresh}): {msg}")
PY

# 9) Pilot sign-ablation verdict
if [ "${RUN_SIGN_ABLATION}" = "1" ]; then
  pos_sign=""
  neg_sign=""
  for sign in "${SIGN_GRID[@]}"; do
    if awk "BEGIN { exit !(${sign} < 0) }"; then
      neg_sign="${sign}"
    else
      pos_sign="${sign}"
    fi
  done
  if [ -n "${pos_sign}" ] && [ -n "${neg_sign}" ]; then
    python scripts/modpo/ultrafeedback/utils/analyze_truthfulness_sign_pilot.py \
      --scores_root "${EVAL_DIR}/scores_armorm" \
      --w_values "${SIGN_W_GRID[@]}" \
      --pos_sign "${pos_sign}" \
      --neg_sign "${neg_sign}" \
      --bootstrap_iters "${SMOKE_BOOTSTRAP_ITERS}" \
      --seed "${SMOKE_BOOTSTRAP_SEED}" \
      --output_json "${EVAL_DIR}/sign_ablation_summary.json" \
      | tee -a "${LOGDIR}/sign_ablation_summary.log"
  else
    echo "[WARN] Could not find both a positive and negative sign in SIGN_VALUES; skipping sign verdict."
  fi
fi

echo "=== Pipeline complete ==="
echo "Outputs: ${OUTPUT_ROOT}"

