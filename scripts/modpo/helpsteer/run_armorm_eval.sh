#!/bin/bash
set -euo pipefail

# Score HelpSteer generations with ArmoRM (HelpSteer heads) and print a compact summary.
#
# Intended to run AFTER `run_pipeline_resumable.sh` finished generation.
# Run from the modpo repo root (the directory that contains `scripts/` and `src/`):
#   export PYTHONPATH=. CUDA_VISIBLE_DEVICES=0
#   export OUTPUT_ROOT=./outputs/helpsteer/v2_nanfix_2026-02-03
#   bash scripts/modpo/helpsteer/run_armorm_eval.sh

export PYTHONPATH=${PYTHONPATH:-.}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

OUTPUT_ROOT=${OUTPUT_ROOT:-"./outputs/helpsteer/v2"}
EVAL_DIR="${OUTPUT_ROOT}/eval"

ARMORM_MODEL_PATH=${ARMORM_MODEL_PATH:-"RLHFlow/ArmoRM-Llama3-8B-v0.1"}
ARMORM_BATCH_SIZE=${ARMORM_BATCH_SIZE:-8}
ARMORM_DEBUG_MAX_SAMPLES=${ARMORM_DEBUG_MAX_SAMPLES:-""} # empty => full

OUT_DIR="${EVAL_DIR}/scores_armorm"
mkdir -p "${OUT_DIR}"

echo "=== ArmoRM eval (HelpSteer heads) ==="
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "EVAL_DIR=${EVAL_DIR}"
echo "ARMORM_MODEL_PATH=${ARMORM_MODEL_PATH}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

if [ ! -d "${EVAL_DIR}" ]; then
  echo "[ERROR] Missing eval dir: ${EVAL_DIR}"
  echo "Run generation first (via the pipeline), or set OUTPUT_ROOT to the correct run output."
  exit 1
fi

gens_dirs=()
labels=()

if [ -d "${EVAL_DIR}/gens_sft" ]; then
  gens_dirs+=("${EVAL_DIR}/gens_sft")
  labels+=("sft")
fi

for d in "${EVAL_DIR}"/gens_modpo_w*; do
  if [ -d "${d}" ]; then
    w="$(basename "${d}" | sed 's/^gens_//')"
    gens_dirs+=("${d}")
    labels+=("${w}")
  fi
done

if [ "${#gens_dirs[@]}" -lt 2 ]; then
  echo "[ERROR] Expected at least 2 generation dirs (sft + >=1 modpo). Found: ${#gens_dirs[@]}"
  echo "Looked for: ${EVAL_DIR}/gens_sft and ${EVAL_DIR}/gens_modpo_w*"
  exit 1
fi

echo "Found generation dirs:"
for i in "${!gens_dirs[@]}"; do
  echo "  ${labels[$i]} => ${gens_dirs[$i]}"
done

echo
echo "=== Validating eval prompt set alignment ==="
python scripts/modpo/helpsteer/utils/validate_eval_set.py \
  $(for i in "${!gens_dirs[@]}"; do echo --gens_dir "${gens_dirs[$i]}" --label "${labels[$i]}"; done)

echo
echo "=== Scoring with ArmoRM ==="
for i in "${!gens_dirs[@]}"; do
  label="${labels[$i]}"
  in_dir="${gens_dirs[$i]}"
  out="${OUT_DIR}/${label}"
  mkdir -p "${out}"
  echo "[SCORE] ${label} -> ${out}"
  extra=()
  if [ -n "${ARMORM_DEBUG_MAX_SAMPLES}" ]; then
    extra+=(--debug_max_samples "${ARMORM_DEBUG_MAX_SAMPLES}")
  fi
  python scripts/modpo/ultrafeedback/utils/score_armorm.py \
    --input_dir "${in_dir}" \
    --output_dir "${out}" \
    --model_path "${ARMORM_MODEL_PATH}" \
    --batch_size "${ARMORM_BATCH_SIZE}" \
    "${extra[@]}"
done

echo
echo "=== Summary (HelpSteer heads only) ==="
python scripts/modpo/helpsteer/utils/summarize_armorm_helpsteer.py \
  $(for i in "${!labels[@]}"; do echo --scores_path "${OUT_DIR}/${labels[$i]}/scores_armorm.jsonl" --label "${labels[$i]}"; done)

echo "=== Done ==="
echo "Outputs: ${OUT_DIR}"

