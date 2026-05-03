#!/bin/bash
set -e

MINICONDA_DIR="$HOME/miniconda3"
CONDA_ENV_NAME="tunix"
CONDA_ENV_DIR="$MINICONDA_DIR/envs/$CONDA_ENV_NAME"
# CACHE_DIR="/home/gs1693/bucket/cache"

# # Route pip/conda caches and build temp dir to the mounted bucket
# mkdir -p "$CACHE_DIR/pip" "$CACHE_DIR/conda" "$CACHE_DIR/tmp"
# export PIP_CACHE_DIR="$CACHE_DIR/pip"
# export CONDA_PKGS_DIRS="$CACHE_DIR/conda"
# export TMPDIR="$CACHE_DIR/tmp"  # pip source builds (e.g. vLLM) use TMPDIR, not PIP_CACHE_DIR

# Download and install Miniconda if not already installed
if [ ! -d "$MINICONDA_DIR" ]; then
    echo "Downloading Miniconda..."
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$MINICONDA_DIR"
    rm /tmp/miniconda.sh
fi

# Initialize conda for this shell session
source "$MINICONDA_DIR/etc/profile.d/conda.sh"

# Accept Anaconda Terms of Service for default channels
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create the conda environment if it does not exist
if [ ! -d "$CONDA_ENV_DIR" ]; then
    echo "Creating conda environment '${CONDA_ENV_NAME}'..."
    conda create -y --name "$CONDA_ENV_NAME" python=3.11
fi

conda activate "$CONDA_ENV_NAME"

pip install uv
uv pip install -e ".[prod]"
VLLM_TARGET_DEVICE=tpu uv pip install -r https://github.com/google/tunix/raw/main/requirements/requirements.txt
uv pip install -r https://github.com/google/tunix/raw/main/requirements/special_requirements.txt
