#!/bin/bash
# setup_armorm_env.sh
# Creates a dedicated Conda environment for ArmoRM scoring
# Usage: bash setup_armorm_env.sh

ENV_NAME="armorm_eval"

echo "Creating Conda environment: $ENV_NAME"
conda create -n $ENV_NAME python=3.10 -y

echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo "Installing dependencies..."
# Install Pytorch (adjust cuda version if needed, assuming 12.1 for A6000)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install Transformers >= 4.40 for Llama 3 support
pip install "transformers>=4.40" "accelerate>=0.30" "tyro" "datasets" "protobuf" "sentencepiece"

echo "Environment $ENV_NAME setup complete."
echo "Activate with: conda activate $ENV_NAME"
