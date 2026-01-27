#!/bin/bash
# Master Debug Server Script
# Runs both Positive Beta (Fix) and Negative Beta (Diagnostic) experiments sequentially.
# Designed for execution on the server with background logging.

set -e

# Define log files
LOG_POS="debug_pos_beta.log"
LOG_NEG="debug_neg_beta.log"

echo "============================================="
echo "   MODPO Debug: Master Execution Sequence"
echo "============================================="
echo "Started at: $(date)"
echo ""

# 1. Run Positive Beta Experiment (The Requested "Fix")
echo ">>> STEP 1: Running Positive Beta Experiment..."
echo "    Logging to: $LOG_POS"
echo "    (Tail this file to monitor progress: tail -f $LOG_POS)"
echo ""

# Run in background to decouple from shell if needed, but we wait for it
bash experiments/positive_beta.sh > "$LOG_POS" 2>&1

echo ">>> [DONE] Positive Beta Experiment finished."
echo "---------------------------------------------"

# 2. Run Negative Beta Experiment (The Diagnostic)
echo ">>> STEP 2: Running Negative Beta Diagnostic..."
echo "    Logging to: $LOG_NEG"
echo "    (Tail this file to monitor progress: tail -f $LOG_NEG)"
echo ""

bash experiments/negative_beta_diagnostic.sh > "$LOG_NEG" 2>&1

echo ">>> [DONE] Negative Beta Experiment finished."
echo "============================================="
echo "All experiments completed at: $(date)"
echo "Please check $LOG_POS and $LOG_NEG for results."
