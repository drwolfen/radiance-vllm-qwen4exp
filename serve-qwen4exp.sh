#!/bin/bash
# serve-qwen4exp.sh -- Launch radiance-vllm-qwen4exp staging container on port 8085
# Dual AMD Radeon AI PRO R9700 (gfx1201) with Tiered PCIe 3.0 Offload

set -euo pipefail

PORT="${PORT:-8085}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-qwen4exp}"
IMAGE="${IMAGE:-drwolfen/radiance-vllm-qwen4exp:0.1.0}"
MODELS="${MODELS:-/home/ydj/LLM-Models}"
MODEL_NAME="${MODEL_NAME:-Qwen3.8-Flash-Next}"
MODEL_DIR="${MODEL_DIR:-/models/Qwen3.8-Flash-Next-UD-IQ4_XS}"

echo "Starting ${CONTAINER_NAME} on http://0.0.0.0:${PORT}/v1..."

docker run -d \
  --name "${CONTAINER_NAME}" \
  --restart unless-stopped \
  --ipc host \
  --shm-size 32gb \
  --network bridge \
  -p "${PORT}:8000" \
  --device /dev/kfd:/dev/kfd \
  --device /dev/dri:/dev/dri \
  --security-opt seccomp=unconfined \
  --group-add video \
  --group-add render \
  -v "${MODELS}:/models:ro" \
  -v "$(pwd)/qwen4exp:/app/qwen4exp:ro" \
  -e HSA_ENABLE_SDMA=1 \
  -e GPU_MAX_HW_QUEUES=1 \
  -e HIP_VISIBLE_DEVICES=0,1 \
  -e ROCR_VISIBLE_DEVICES=0,1 \
  -e PYTORCH_ROCM_ARCH=gfx1201 \
  -e VLLM_TARGET_DEVICE=rocm \
  -e VLLM_LOGGING_LEVEL=INFO \
  "${IMAGE}" \
  python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_DIR}" \
    --served-model-name "${MODEL_NAME}" \
    --tensor-parallel-size 2 \
    --max-model-len 262144 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.92 \
    --port 8000 \
    --host 0.0.0.0

echo "Container ${CONTAINER_NAME} started on port ${PORT}."
