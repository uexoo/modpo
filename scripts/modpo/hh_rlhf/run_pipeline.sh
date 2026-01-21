#!/bin/bash

# Configuration
MODEL_SIZE="7b"
BASE_MODEL="PKU-Alignment/alpaca-7b-reproduced"  # Same as verification for consistency
OUTPUT_ROOT="./outputs/rq1/hh_rlhf"
EVAL_DIR="$OUTPUT_ROOT/eval"

# 1. Train SFT Reference Policy (Helpful)
echo "=== Step 1: Training SFT Reference on HH-RLHF Helpful ==="
accelerate launch scripts/examples/sft/sft.py \
    --base_model_name $BASE_MODEL \
    --dataset_name Anthropic/hh-rlhf-helpful \
    --training_args.output_dir $OUTPUT_ROOT/sft_helpful \
    --training_args.run_name rq1_hh_sft_helpful

# 2. Train Margin Reward Model (Harmless)
echo "=== Step 2: Training Margin Reward Model on HH-RLHF Harmless ==="
accelerate launch scripts/examples/dpo/dpo.py \
    --sft_model_name $OUTPUT_ROOT/sft_helpful/best_checkpoint \
    --dataset_name Anthropic/hh-rlhf-harmless \
    --training_args.output_dir $OUTPUT_ROOT/rm_harmless \
    --training_args.run_name rq1_hh_rm_harmless

# 3. Train MODPO (Helpful Base + Harmless Margin)
echo "=== Step 3: Training MODPO models ==="
for w in 0.0 0.5 1.0; do
    echo "Training MODPO w=$w..."
    accelerate launch scripts/modpo/hh_rlhf/modpo.py \
        --sft_model_name $OUTPUT_ROOT/sft_helpful/best_checkpoint \
        --margin_reward_model_name $OUTPUT_ROOT/rm_harmless/best_checkpoint \
        --dataset_name Anthropic/hh-rlhf-helpful \
        --w $w \
        --training_args.output_dir $OUTPUT_ROOT/modpo_w${w} \
        --training_args.run_name rq1_hh_modpo_w${w}
done

# 4. Generate responses (SFT baseline + all MODPO models)
echo "=== Step 4: Generating responses ==="

# SFT baseline
echo "Generating SFT baseline responses..."
python scripts/modpo/hh_rlhf/utils/gen.py \
    --sft_model_name $OUTPUT_ROOT/sft_helpful/best_checkpoint \
    --dataset_name Anthropic/hh-rlhf-helpful \
    --output_dir $EVAL_DIR/sft_baseline/gen \
    --eval_size 700

# MODPO models
for w in 0.0 0.5 1.0; do
    echo "Generating responses for MODPO w=$w..."
    python scripts/modpo/hh_rlhf/utils/gen.py \
        --sft_model_name $OUTPUT_ROOT/sft_helpful/best_checkpoint \
        --adapter_model_name $OUTPUT_ROOT/modpo_w${w}/best_checkpoint \
        --dataset_name Anthropic/hh-rlhf-helpful \
        --output_dir $EVAL_DIR/modpo_w${w}/gen \
        --eval_size 700
done

# 5. LLM-as-judge evaluation
echo "=== Step 5: Running LLM-as-judge evaluation ==="
python scripts/eval/eval_rq1.py \
    --eval_dir $EVAL_DIR \
    --dimensions helpfulness harmlessness \
    --weights 0.0 0.5 1.0 \
    --output_csv $EVAL_DIR/win_rates.csv

echo "=== Pipeline complete! ==="
echo "Results saved to $EVAL_DIR/win_rates.csv"

