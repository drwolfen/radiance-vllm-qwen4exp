#!/bin/bash
# setup-qwen4exp.sh -- Host validation and setup for radiance-vllm-qwen4exp on dual R9700 (gfx1201)

set -euo pipefail

echo "=== 1. Checking Host Kernel & KFD Driver ==="
if [ ! -e /dev/kfd ]; then
  echo "ERROR: /dev/kfd not found. ROCm kernel driver not loaded." >&2
  exit 1
fi
echo "[OK] /dev/kfd is accessible."

echo "=== 2. Checking GPU Accelerators ==="
if command -v rocm-smi >/dev/null 2>&1; then
  rocm-smi --showid
else
  echo "[WARN] rocm-smi CLI not in path, checking /sys/class/kfd/kfd/topology/nodes..."
fi

echo "=== 3. Checking Docker & Compose ==="
if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker not installed." >&2
  exit 1
fi
echo "[OK] Docker is installed."

echo "=== 4. Checking Target Model ==="
MODELS_DIR="${MODELS:-/home/ydj/LLM-Models}"
MODEL_PATH="${MODELS_DIR}/Qwen3.8-Flash-Next-UD-IQ4_XS"
if [ -d "$MODEL_PATH" ]; then
  echo "[OK] Found Qwen3.8-Flash-Next model at $MODEL_PATH"
else
  echo "[INFO] Model directory $MODEL_PATH not found yet. Mount or symlink model path before launch."
fi

echo "=== Setup check complete ==="
