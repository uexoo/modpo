#!/bin/bash
# Negative Beta Experiment (Diagnostic)
# Hypothesis: The original math was correct, but magnitudes were off.
# This run uses the NEW instrumentation to log raw margin values.
#
# This script runs MODPO with w=0.0 and w=1.0 using NEGATIVE beta for margin.
#
# Constraints: Single GPU usage.

set -e
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0

OUTPUT_ROOT="./outputs/rq1/ultrafeedback_debug/neg_beta"
SFT_MODEL="./outputs/rq1/ultrafeedback/sft_helpfulness/best_checkpoint"
RM_MODEL="./outputs/rq1/ultrafeedback/rm_honesty/best_checkpoint"

mkdir -p $OUTPUT_ROOT

echo "=== MODPO Debug: Negative Beta Diagnostic ==="
echo "Output: $OUTPUT_ROOT"
echo ""

# Train for w=0.0 and w=1.0
for W in 0.0 1.0; do
    OUTPUT_DIR="$OUTPUT_ROOT/modpo_w${W}"
    
    echo "Training w=$W with NEGATIVE beta..."
    
    # Using explicit --margin_beta -0.1 (Negative - The Original Math)
    accelerate launch --num_processes 1 --mixed_precision fp16 \
        scripts/modpo/ultrafeedback/modpo.py \
        --sft_model_name $SFT_MODEL \
        --margin_reward_model_name $RM_MODEL \
        --dataset_name OpenBMB/UltraFeedback-helpfulness \
        --w $W \
        --beta 0.1 \
        --margin_beta -0.1 \
        --max_length 512 \
        --training_args.output_dir $OUTPUT_DIR \
        --training_args.run_name debug_neg_beta_w${W} \
        --training_args.num_train_epochs 1 \
        --training_args.max_steps 200 \
        --training_args.per_device_train_batch_size 1 \
        --training_args.gradient_accumulation_steps 8 \
        --training_args.learning_rate 1e-4 \
        --training_args.report_to wandb \
        --training_args.logging_steps 1
    
    echo "Completed w=$W"
    echo ""
done
