#!/bin/bash
# Quick test: Train with POSITIVE beta (the fix) on sanity check mode
# Expected runtime: 30-60 minutes
#
# Parameters matched to BeaverTails (paper) except:
# - max_length: 512 (paper: 1024) - reduced for memory
# - epochs: 1 (paper: 3) - reduced for speed
# - sanity_check: True - only 100 samples

set -e
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0  # Single GPU only

OUTPUT_ROOT="./outputs/rq1/ultrafeedback"
TEST_DIR="./outputs/rq1/ultrafeedback/test_fix"

echo "=== Quick Fix Test: Positive Beta ==="
echo "GPU: $CUDA_VISIBLE_DEVICES (single GPU)"
echo "This trains w=0.0 with POSITIVE beta on 100 samples (~30-60 min)"
echo ""

# Create patched modpo.py with positive beta (matching BeaverTails)
MODPO_SCRIPT="scripts/modpo/ultrafeedback/modpo.py"
PATCHED_SCRIPT="/tmp/modpo_positive_beta.py"

sed 's/margin_beta = -script_args.beta/margin_beta = script_args.beta  # FIXED: positive beta like BeaverTails/' \
    $MODPO_SCRIPT > $PATCHED_SCRIPT

echo "Created patched script at $PATCHED_SCRIPT"
echo ""

# Train w=0.0 with positive beta (sanity check mode)
# All parameters match paper defaults except noted overrides
echo "Training w=0.0 with positive beta..."
accelerate launch --num_processes 1 --mixed_precision fp16 $PATCHED_SCRIPT \
    --sft_model_name $OUTPUT_ROOT/sft_helpfulness/best_checkpoint \
    --margin_reward_model_name $OUTPUT_ROOT/rm_honesty/best_checkpoint \
    --dataset_name OpenBMB/UltraFeedback-helpfulness \
    --w 0.0 \
    --beta 0.1 \
    --max_length 512 \
    --sanity_check True \
    --training_args.output_dir $TEST_DIR/modpo_w0.0_posbeta \
    --training_args.run_name test_posbeta_w0.0 \
    --training_args.num_train_epochs 1 \
    --training_args.per_device_train_batch_size 1 \
    --training_args.gradient_accumulation_steps 8 \
    --training_args.learning_rate 1e-4 \
    --training_args.fp16 True \
    --training_args.report_to none

echo ""
echo "=== Training Complete ==="
echo "Model saved to: $TEST_DIR/modpo_w0.0_posbeta"
echo ""
echo "Expected result with positive beta:"
echo "  - rewards/margins should trend NEGATIVE (opposite of original -125 not +89)"
echo "  - accuracy should be LOW initially (model learning opposite direction)"
echo "This proves the beta sign controls objective direction."

