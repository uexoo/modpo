#!/bin/bash
# Quick test: Train with POSITIVE beta (the fix) on sanity check mode
# Expected runtime: 30-60 minutes

set -e
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

OUTPUT_ROOT="./outputs/rq1/ultrafeedback"
TEST_DIR="./outputs/rq1/ultrafeedback/test_fix"

echo "=== Quick Fix Test: Positive Beta ==="
echo "This trains w=0.0 with POSITIVE beta on 100 samples (~30-60 min)"
echo ""

# We need a modified modpo.py that uses positive beta
# Create it inline by patching the original

MODPO_SCRIPT="scripts/modpo/ultrafeedback/modpo.py"
PATCHED_SCRIPT="/tmp/modpo_positive_beta.py"

# Create patched version with positive beta
sed 's/margin_beta = -script_args.beta/margin_beta = script_args.beta  # FIXED: positive beta/' \
    $MODPO_SCRIPT > $PATCHED_SCRIPT

echo "Created patched script at $PATCHED_SCRIPT"
echo ""

# Train w=0.0 with positive beta (sanity check mode)
echo "Training w=0.0 with positive beta..."
accelerate launch $PATCHED_SCRIPT \
    --sft_model_name $OUTPUT_ROOT/sft_helpfulness/best_checkpoint \
    --margin_reward_model_name $OUTPUT_ROOT/rm_honesty/best_checkpoint \
    --dataset_name OpenBMB/UltraFeedback-helpfulness \
    --w 0.0 \
    --sanity_check \
    --training_args.output_dir $TEST_DIR/modpo_w0.0_posbeta \
    --training_args.run_name test_posbeta_w0.0 \
    --training_args.num_train_epochs 1 \
    --training_args.per_device_train_batch_size 1 \
    --training_args.gradient_accumulation_steps 8 \
    --training_args.report_to none

echo ""
echo "=== Training Complete ==="
echo "Model saved to: $TEST_DIR/modpo_w0.0_posbeta"
echo ""
echo "Next: Generate responses and compare with original w=0.0"
echo "If the fix works, w=0.0 should show HIGHER honesty than w=1.0 (opposite of current)"
