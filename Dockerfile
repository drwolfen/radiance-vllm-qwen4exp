# radiance-vllm-qwen4exp: Dedicated vLLM Engine for Qwen3.8-Flash-Next on RDNA4 (gfx1201)
# Multi-stage build targeting ROCm 7.14 / 7.2.4 with custom qwen4exp architectural extensions

ARG ROCM_BASE=rocm/dev-ubuntu-24.04:7.14.0-full
ARG GFX_ARCH=gfx1201
ARG RELEASE_BASE=ubuntu:24.04

ARG TORCH_VERSION=2.12.1
ARG VLLM_VERSION=0.28.0
ARG TRANSFORMERS_VERSION=5.14.1
ARG NUMPY_VERSION=2.3.5

# =====================================================================================
# STAGE 1: Builder
# =====================================================================================
FROM ${ROCM_BASE} AS builder
ARG GFX_ARCH
ARG TORCH_VERSION
ARG VLLM_VERSION

ENV DEBIAN_FRONTEND=noninteractive \
    PYTORCH_ROCM_ARCH=${GFX_ARCH} \
    ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev python3-pip python3-venv git ninja-build cmake numactl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/rocm6.2 || true

# =====================================================================================
# STAGE 2: Final Runtime Image
# =====================================================================================
FROM ${RELEASE_BASE} AS final
ARG GFX_ARCH

ENV DEBIAN_FRONTEND=noninteractive \
    PYTORCH_ROCM_ARCH=${GFX_ARCH} \
    ROCM_PATH=/opt/rocm \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv libnuma1 numactl curl jq \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY . /app

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=5 \
    CMD curl -f http://127.0.0.1:8000/health || exit 1

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
