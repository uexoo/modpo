#!/bin/bash
set -e  # Exit on first error (prevents cascading failures)

# Configuration
MODEL_SIZE="7b"
BASE_MODEL="PKU-Alignment/alpaca-7b-reproduced"  # Same as verification for controlled experiment
OUTPUT_ROOT="./outputs/rq1/ultrafeedback"
EVAL_DIR="$OUTPUT_ROOT/eval"
GEN_SCRIPT="scripts/modpo/ultrafeedback/utils/gen.py"
EVAL_SCRIPT="scripts/eval/eval_rq1.py"

# Fix for 'No module named src' - required since no setup.py
export PYTHONPATH=.

# Memory-safe settings (MODPO paper uses batch_size=1, grad_accum=8)
# max_length=512 to avoid OOM on long UltraFeedback sequences
BATCH_ARGS="--training_args.per_device_train_batch_size 1 --training_args.per_device_eval_batch_size 1 --training_args.gradient_accumulation_steps 8"
MAX_LEN_ARGS="--max_length 512"


# 1. Train SFT Reference (Helpfulness)
echo "=== Step 1: Training SFT Reference on UltraFeedback Helpfulness ==="
PYTHONPATH=. accelerate launch scripts/examples/sft/sft.py \
    --base_model_name $BASE_MODEL \
    --dataset_name OpenBMB/UltraFeedback-helpfulness \
    --training_args.output_dir $OUTPUT_ROOT/sft_helpfulness \
    --training_args.run_name rq1_uf_sft_helpfulness \
    --training_args.num_train_epochs 1 \
    $BATCH_ARGS $MAX_LEN_ARGS

# Merge SFT LoRA adapter if needed (downstream scripts expect a full causal LM checkpoint)
SFT_MODEL="$OUTPUT_ROOT/sft_helpfulness/best_checkpoint"
if [ -f "$SFT_MODEL/adapter_config.json" ]; then
    echo "=== Step 1b: Merging SFT LoRA adapter ==="
    SFT_MERGED="$OUTPUT_ROOT/sft_helpfulness/merged_checkpoint"
    if [ ! -f "$SFT_MERGED/config.json" ]; then
        PYTHONPATH=. python src/tools/merge_peft_adapter.py \
            --adapter_model_name "$SFT_MODEL" \
            --base_model_name "$BASE_MODEL" \
            --dtype bf16 \
            --output_name "$SFT_MERGED"
    fi
    SFT_MODEL="$SFT_MERGED"
fi


# 2. Train Margin Reward Model (Honesty)
echo "=== Step 2: Training Margin Reward Model on UltraFeedback Honesty ==="
PYTHONPATH=. accelerate launch scripts/examples/dpo/dpo.py \
    --sft_model_name $SFT_MODEL \
    --dataset_name OpenBMB/UltraFeedback-honesty \
    --training_args.output_dir $OUTPUT_ROOT/rm_honesty \
    --training_args.run_name rq1_uf_rm_honesty \
    --training_args.num_train_epochs 1 \
    $BATCH_ARGS $MAX_LEN_ARGS

# 3. Train MODPO (Helpfulness vs Honesty)
echo "=== Step 3: Training MODPO models ==="
for w in 0.1 0.5 0.9; do
    echo "Training MODPO w=$w..."
    PYTHONPATH=. accelerate launch scripts/modpo/ultrafeedback/modpo.py \
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
PYTHONPATH=. python $GEN_SCRIPT \
    --sft_model_name $SFT_MODEL \
    --dataset_name OpenBMB/UltraFeedback-helpfulness \
    --output_dir $EVAL_DIR/sft_baseline/gen \
    --eval_size 700

# MODPO models
for w in 0.1 0.5 0.9; do
    echo "Generating responses for MODPO w=$w..."
    PYTHONPATH=. python $GEN_SCRIPT \
        --sft_model_name $SFT_MODEL \
        --adapter_model_name $OUTPUT_ROOT/modpo_w${w}/best_checkpoint \
        --dataset_name OpenBMB/UltraFeedback-helpfulness \
        --output_dir $EVAL_DIR/modpo_w${w}/gen \
        --eval_size 700
done

# 5. LLM-as-judge evaluation
echo "=== Step 5: Running LLM-as-judge evaluation ==="
PYTHONPATH=. python $EVAL_SCRIPT \
    --eval_dir $EVAL_DIR \
    --dimensions helpfulness honesty \
    --weights 0.1 0.5 0.9 \
    --output_csv $EVAL_DIR/win_rates.csv

echo "=== Pipeline complete! ==="
echo "Results saved to $EVAL_DIR/win_rates.csv"

