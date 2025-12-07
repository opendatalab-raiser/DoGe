#!/bin/bash

set -e

MODEL_PATH="your model path"

echo "正在使用模型启动服务器: ${MODEL_PATH}"

CUDA_VISIBLE_DEVICES=0 python vllm_service_init/start_vllm_server.py \
  --model "${MODEL_PATH}" \
  --port 5000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9