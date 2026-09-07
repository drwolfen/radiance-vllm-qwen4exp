# radiance-vllm-qwen4exp: Fast layered build on validated RDNA4 gfx1201 base
# Base image contains vLLM 0.28.0, PyTorch 2.12.1+rocm7.14, AITER 0.1.20, libr4d
ARG BASE_IMAGE=radiance-vllm:0.10.0
FROM ${BASE_IMAGE}

WORKDIR /app
COPY . /app

RUN chmod +x /app/*.sh /app/tests/*.py 2>/dev/null || true

ENV PYTORCH_ROCM_ARCH=gfx1201 \
    VLLM_TARGET_DEVICE=rocm \
    VLLM_LOGGING_LEVEL=INFO

EXPOSE 8000 8085

HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
    CMD curl -f http://127.0.0.1:8000/health || exit 1

ENTRYPOINT ["/app/radiance_entrypoint.sh"]
