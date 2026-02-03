#!/bin/bash
set -e  # Exit on first error

# Configuration
# Explicitly set CUDA_VISIBLE_DEVICES to 0 to avoid multi-GPU launch issues
# unless configured otherwise.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Base model used for SFT (needed to merge LoRA adapters, if present)
BASE_MODEL=${BASE_MODEL:-"PKU-Alignment/alpaca-7b-reproduced"}

OUTPUT_ROOT="./outputs/rq1/ultrafeedback"
EVAL_DIR="$OUTPUT_ROOT/eval"
GEN_SCRIPT="scripts/modpo/ultrafeedback/utils/gen.py"
EVAL_SCRIPT="scripts/eval/eval_rq1.py"

# Standardizing python path
export PYTHONPATH=.

# Memory-safe settings (Batch size 1, Grad accum 8 = Effective 8)
BATCH_ARGS="--training_args.per_device_train_batch_size 1 --training_args.per_device_eval_batch_size 1 --training_args.gradient_accumulation_steps 8"
MAX_LEN_ARGS="--max_length 512"

echo "=== Resuming RQ1 Pipeline from Step 3 ==="
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
echo "Output Root: $OUTPUT_ROOT"

# Resolve SFT model path (merge LoRA adapter if needed)
SFT_MODEL="$OUTPUT_ROOT/sft_helpfulness/best_checkpoint"
if [ -f "$SFT_MODEL/adapter_config.json" ]; then
    SFT_MERGED="$OUTPUT_ROOT/sft_helpfulness/merged_checkpoint"
    if [ ! -f "$SFT_MERGED/config.json" ]; then
        echo "Merging SFT LoRA adapter -> $SFT_MERGED"
        python src/tools/merge_peft_adapter.py \
            --adapter_model_name "$SFT_MODEL" \
            --base_model_name "$BASE_MODEL" \
            --dtype bf16 \
            --output_name "$SFT_MERGED"
    fi
    SFT_MODEL="$SFT_MERGED"
fi

# 3. Train MODPO (Helpfulness vs Honesty)
echo "=== Step 3: Training MODPO models ==="
for w in 0.1 0.5 0.9; do
    echo "----------------------------------------------------------------"
    echo "Training MODPO w=$w..."
    echo "----------------------------------------------------------------"
    
    # Check if already done (simple check: if directory exists)
    # You might want to remove this check if you want to force overwrite, 
    # but for safety let's just log it.
    if [ -d "$OUTPUT_ROOT/modpo_w${w}" ]; then
        echo "WARNING: $OUTPUT_ROOT/modpo_w${w} already exists. Overwriting/Resuming..."
    fi

    accelerate launch scripts/modpo/ultrafeedback/modpo.py \
        --sft_model_name $SFT_MODEL \
        --margin_reward_model_name $OUTPUT_ROOT/rm_honesty/best_checkpoint \
        --dataset_name OpenBMB/UltraFeedback-helpfulness \
        --w $w \
        --training_args.output_dir $OUTPUT_ROOT/modpo_w${w} \
        --training_args.run_name rq1_uf_modpo_w${w} \
        --training_args.num_train_epochs 1 \
        $BATCH_ARGS $MAX_LEN_ARGS
done

# 4. Generate responses (SFT baseline + all MODPO models)
echo "=== Step 4: Generating responses ==="

# SFT baseline
echo "Generating SFT baseline responses..."
python $GEN_SCRIPT \
    --sft_model_name $SFT_MODEL \
    --dataset_name OpenBMB/UltraFeedback-helpfulness \
    --output_dir $EVAL_DIR/sft_baseline/gen \
    --eval_size 700

# MODPO models
for w in 0.1 0.5 0.9; do
    echo "Generating responses for MODPO w=$w..."
    python $GEN_SCRIPT \
        --sft_model_name $SFT_MODEL \
        --adapter_model_name $OUTPUT_ROOT/modpo_w${w}/best_checkpoint \
        --dataset_name OpenBMB/UltraFeedback-helpfulness \
        --output_dir $EVAL_DIR/modpo_w${w}/gen \
        --eval_size 700
done

# 5. LLM-as-judge evaluation
echo "=== Step 5: Running LLM-as-judge evaluation ==="
# NOTE: Requires OPENAI_API_KEY to be set in environment
python $EVAL_SCRIPT \
    --eval_dir $EVAL_DIR \
    --dimensions helpfulness honesty \
    --weights 0.1 0.5 0.9 \
    --output_csv $EVAL_DIR/win_rates.csv

echo "=== Pipeline complete! ==="
echo "Results saved to $EVAL_DIR/win_rates.csv"
