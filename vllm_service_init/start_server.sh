#!/bin/bash

set -e

MODEL_PATH="your model path"

echo "Model Server Start: ${MODEL_PATH}"

CUDA_VISIBLE_DEVICES=0 python start_vllm_server.py \
  --model "${MODEL_PATH}" \
  --port 5000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9