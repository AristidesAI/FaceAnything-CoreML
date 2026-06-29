#!/usr/bin/env bash
# Face Anything installation script.
#
# Creates a conda environment, installs PyTorch (CUDA 12.8 build we tested),
# the Python dependencies, and the package, then downloads the checkpoint.
#
# Usage:
#   bash install.sh                 # env name "faceanything"
#   bash install.sh my_env_name     # custom env name
#
# Requires a CUDA GPU and conda. Tested with Python 3.11 / PyTorch 2.9 (CUDA 12.8).
set -e

ENV_NAME="${1:-faceanything}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

echo ">>> Creating conda env '$ENV_NAME' (python 3.11)"
conda create -n "$ENV_NAME" python=3.11 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo ">>> Installing PyTorch (CUDA 12.8)"
pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128

echo ">>> Installing dependencies"
pip install -r requirements.txt

echo ">>> Installing the faceanything package"
pip install -e .

echo ">>> Downloading the checkpoint (~15 GB) into checkpoints/checkpoint.pt"
mkdir -p checkpoints
if [ -f checkpoints/checkpoint.pt ]; then
    echo "    checkpoints/checkpoint.pt already exists, skipping download."
else
    gdown --fuzzy "https://drive.google.com/file/d/1PdQQxzm-tU50RmJhgeoMCYVRlEiW3f8p/view?usp=sharing" \
        -O checkpoints/checkpoint.pt
fi

echo ""
echo ">>> Done. Activate the environment with:"
echo "      conda activate $ENV_NAME"
echo ">>> Then run, e.g.:"
echo "      python run_inference.py --input path/to/images --output output/demo"
