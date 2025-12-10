#!/bin/bash

# Set up cache directory
# HF_CACHE_DIR="$(pwd)/containers/download_hf_cache/hf_cache"
# hf auth login

# Export cache environment variables
# export HF_HOME="${HF_CACHE_DIR}"

# Run download in container
apptainer run \
    --nv \
    --bind "${HF_HOME}:${HF_HOME}" \
    --env HF_HOME="${HF_HOME}" \
    "${POST_TRAIN_BENCH_CONTAINERS_DIR}/${POST_TRAIN_BENCH_CONTAINER_NAME}.sif" containers/download_hf_cache/download_resources.py

echo "Downloads complete! Cache at: ${HF_HOME}"