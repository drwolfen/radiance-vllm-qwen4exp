# R9V gfx1201 kernels

Shape-specialized ROCm kernels used by the R9V Qwen3.8 Flash Next inference
profile on Radeon R9700 (`gfx1201`). This is the kernel component of the R9V
release; it is independent of the unlicensed Radiance launcher source.

## Production kernel families

- `dense_mmvq_hip`: MTP multi-row GGUF GEMV reuse, exact Q8 attention M=3,
  HyperConnection down, and fused HyperConnection up/gate-mix paths.
- `tiered_iq_moe_hip`: mixed VRAM/UVA expert GEMV, three-row expert reuse, and
  graph-safe LRU expert caching.
- `fused_gdn_mtp_hip`: Qwen3.8 speculative GDN core specialized for TP2.

The checked-in `.cu` files are canonical. PyTorch's ROCm extension builder
generates HIP-translated sources and build artifacts locally; those generated
files are intentionally excluded.

## Requirements

- Linux with ROCm and a PyTorch ROCm build.
- `gfx1201` hardware for the production paths.
- The R9V vLLM GGUF-plugin fork installed or checked out. The dense and tiered
  kernels include its GGUF quant-format and dot-product headers.

If the plugin is not importable, point the builders at its source tree:

```bash
export VLLM_GGUF_PLUGIN_CSRC=/path/to/vllm_gguf_plugin/csrc
```

Build into disposable candidate directories before promotion:

```bash
QWEN38_DENSE_MMVQ_BUILD_DIR="$PWD/dense_mmvq_hip/build-candidate" \
  python dense_mmvq_hip/build_extension.py

python tiered_iq_moe_hip/build_extension.py \
  --build-directory tiered_iq_moe_hip/build-candidate \
  --variants auto,u2,u5,u10,reuse3,reuse3v2

python fused_gdn_mtp_hip/build_extension.py
```

## Scope and portability

These are deliberately narrow kernels, not generic ROCm replacements. Their
host APIs fail closed on incompatible dtype, shape, qtype, device, or layout.
Unsupported inputs remain on the vLLM/plugin fallback path. Porting to another
GPU or model means revalidating wave layout, LDS usage, quant formats, exact
rounding boundaries, graph replay, and real-weight parity.

## Safety

The R9V release does not include or enable the experimental R4D collective or
attention paths used during early research. Run bounded tests on one explicitly
selected non-display GPU before a full model launch. The GDN helper requires
`R9V_HEADLESS_BDF`, `R9V_HEADLESS_UUID`, `R9V_HEADLESS_RENDER`,
`R9V_HEADLESS_CARD`, `R9V_DISPLAY_BDF`, and `R9V_DEV_IMAGE`; it will not guess
which GPU is safe to use.

## Licensing

R9V-owned code is Apache-2.0. GGUF quant primitives are consumed from the
Apache-2.0 vLLM GGUF plugin, which in turn contains MIT-licensed llama.cpp/ggml
material. See `THIRD_PARTY_NOTICES.md` for the retained attribution.
