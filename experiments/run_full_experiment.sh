#!/bin/bash
# Full MODPO Experiment: UltraFeedback Helpfulness vs Honesty
# Runs the complete sweep for the thesis RQ1.
# - Weights: 0.8 (High Helpfulness), 0.6, 0.4, 0.2 (High Honesty)
# - Beta: 0.1
# - Margin Beta: 0.1 (Correct Positive Sign for Trade-off)
# - Precision: fp16
# - Features:
#   - Evaluation during training (every 25% of steps)
#   - Best checkpoint saved based on eval_loss
#   - 3 checkpoints retained for safe resumption
#   - Batch size = 1 with grad_accum = 8 (OOM fix)
#   - Max length = 512 (OOM fix)

set -e

# Configuration
OUTPUT_ROOT="./outputs/rq1/ultrafeedback/final_run"
SFT_MODEL="./outputs/rq1/ultrafeedback/sft_helpfulness/best_checkpoint"
RM_MODEL="./outputs/rq1/ultrafeedback/rm_honesty/best_checkpoint"

# Ensure output directory exists
mkdir -p $OUTPUT_ROOT

echo "=== Starting Full MODPO Experiment Run ==="
echo "Output: $OUTPUT_ROOT"
echo "Weights: 0.2"
echo "Features: Eval during training, best checkpoint, resumable"
echo "----------------------------------------"

# Loop through weights (high helpfulness to high honesty)
for W in 0.2; do
    OUTPUT_DIR="$OUTPUT_ROOT/modpo_w${W}"
    RUN_NAME="modpo_uf_w${W}_final"
    
    echo "Processing w=${W}..."
    
    # Check if already done
    if [ -d "$OUTPUT_DIR/best_checkpoint" ]; then
        echo "Found existing checkpoint at $OUTPUT_DIR. Skipping."
        continue
    fi
    
    echo "Launching training..."
    
    # Run training with evaluation and robust checkpointing
    accelerate launch --num_processes 1 --mixed_precision fp16 \
        scripts/modpo/ultrafeedback/modpo.py \
        --sft_model_name $SFT_MODEL \
        --margin_reward_model_name $RM_MODEL \
        --dataset_name OpenBMB/UltraFeedback-helpfulness \
        --w $W \
        --beta 0.1 \
        --margin_beta 0.1 \
        --max_length 512 \
        --training_args.output_dir $OUTPUT_DIR \
        --training_args.run_name $RUN_NAME \
        --training_args.num_train_epochs 1 \
        --training_args.per_device_train_batch_size 1 \
        --training_args.gradient_accumulation_steps 8 \
        --training_args.learning_rate 1e-4 \
        --training_args.report_to wandb \
        --training_args.logging_steps 10 \
        --training_args.evaluation_strategy steps \
        --training_args.eval_steps 0.25 \
        --training_args.save_strategy steps \
        --training_args.save_steps 0.25 \
        --training_args.save_total_limit 3 \
        --training_args.load_best_model_at_end True \
        --training_args.metric_for_best_model eval_loss \
        --training_args.greater_is_better False

    echo "Finished w=${W}"
    echo "----------------------------------------"

done

echo "All experiments completed successfully!"
