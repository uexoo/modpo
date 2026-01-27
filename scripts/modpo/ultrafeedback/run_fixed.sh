#!/bin/bash
# Full MODPO training with FIXED positive beta
# This reruns RQ1 experiment with the corrected objective direction
#
# BUG FIXED: The original script used margin_beta = -script_args.beta
# which inverted the honesty objective. Changed to positive beta
# to match BeaverTails implementation.
#
# Expected runtime: ~24 hours per weight
# Total: ~72 hours for all 3 weights

set -e
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0

OUTPUT_ROOT="./outputs/rq1/ultrafeedback_fixed"
SFT_MODEL="./outputs/rq1/ultrafeedback/sft_helpfulness/best_checkpoint"
RM_MODEL="./outputs/rq1/ultrafeedback/rm_honesty/best_checkpoint"

mkdir -p $OUTPUT_ROOT

echo "=== MODPO Training with Fixed Positive Beta ==="
echo "Output: $OUTPUT_ROOT"
echo ""

# Train for each weight
for W in 0.0 0.5 1.0; do
    OUTPUT_DIR="$OUTPUT_ROOT/modpo_w${W}"
    
    if [ -d "$OUTPUT_DIR/best_checkpoint" ]; then
        echo "Skipping w=$W (already exists)"
        continue
    fi
    
    echo "Training w=$W..."
    echo "Started at: $(date)"
    
    accelerate launch --num_processes 1 --mixed_precision fp16 \
        scripts/modpo/ultrafeedback/modpo.py \
        --sft_model_name $SFT_MODEL \
        --margin_reward_model_name $RM_MODEL \
        --dataset_name OpenBMB/UltraFeedback-helpfulness \
        --w $W \
        --beta 0.1 \
        --max_length 1024 \
        --training_args.output_dir $OUTPUT_DIR \
        --training_args.run_name modpo_fixed_w${W} \
        --training_args.num_train_epochs 3 \
        --training_args.per_device_train_batch_size 4 \
        --training_args.gradient_accumulation_steps 2 \
        --training_args.learning_rate 1e-4 \
        --training_args.report_to wandb
    
    echo "Completed w=$W at: $(date)"
    echo ""
done

echo "=== All Training Complete ==="
echo "Next: Run generation and evaluation"
echo ""
echo "  # Generate responses:"
echo "  for W in 0.0 0.5 1.0; do"
echo "      python scripts/modpo/ultrafeedback/utils/gen.py \\"
echo "          --sft_model_name $SFT_MODEL \\"
echo "          --adapter_model_name $OUTPUT_ROOT/modpo_w\${W}/best_checkpoint \\"
echo "          --output_dir $OUTPUT_ROOT/modpo_w\${W}/generations \\"
echo "          --eval_size 700"
echo "  done"
echo ""
echo "  # Run LLM-as-judge evaluation:"
echo "  python scripts/eval/eval_rq1.py \\"
echo "      --eval_dir $OUTPUT_ROOT \\"
echo "      --dimensions helpfulness honesty"
