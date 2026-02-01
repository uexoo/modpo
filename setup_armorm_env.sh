#!/bin/bash
# setup_armorm_env.sh
# Creates a dedicated Conda environment for ArmoRM scoring
# Usage: bash setup_armorm_env.sh

ENV_NAME="armorm_eval"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Conda could not be found. Please ensure conda is initializing correctly."
    exit 1
fi

echo "Creating Conda environment: $ENV_NAME"
conda create -n $ENV_NAME python=3.10 -y

# IMPORTANT: We cannot rely on 'conda activate' inside a script easily without init.
# Instead, we tell the user to activate it, OR we use 'conda run' for installation.
# Using 'conda run' is safer to avoid polluting the current environment.

echo "Installing dependencies into $ENV_NAME..."

# Install Pytorch
conda run -n $ENV_NAME conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install Transformers >= 4.40
conda run -n $ENV_NAME pip install "transformers>=4.40" "accelerate>=0.30" "tyro" "datasets" "protobuf" "sentencepiece"

echo "---------------------------------------"
echo "Environment $ENV_NAME setup complete."
echo "To use it, run:"
echo "  conda activate $ENV_NAME"
echo "---------------------------------------"
