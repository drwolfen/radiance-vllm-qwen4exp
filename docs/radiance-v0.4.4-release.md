# Radiance-vLLM Qwen4exp v0.4.4 Release & Qualification Note

## Overview
Radiance-vLLM Qwen4exp v0.4.4 synchronizes the complete upstream Dyluhn/R9V v0.4.4 runtime stack, introducing:
1. **CED Dynamic VRAM-Vision Swapping**: The vision encoder weights (0.42 GiB per GPU) and CED prefill projector (1.76 GiB per GPU) dynamically share memory on dual AMD Radeon AI PRO R9700 GPUs.
2. **Context Embedding Distillation (CED) Acceleration**: 1.5× to 1.82× faster prefill on long contexts (>=8,192 tokens).
3. **Full Mutable Expert Deduplication**: Reduced host RAM requirement from 71.4 GiB to 56.3 GiB (60.8 GiB with CED enabled).
4. **Verified Performance Parity**:
   - Single-stream sustained decode: **45.8 tok/s** (TPOT 40.9 ms)
   - Multi-stream sustained generation: **42–45 tok/s**
   - Tool calling & reasoning verification: **100% PASS**
   - APC prefix caching speedup: **1.14x**
   - Full regression suite: **752 test cases PASSED**

## Local deployment verification (2026-09-26)
- Installed `qwen38-mtp4-uncensored` on the v0.4.4 runtime over the previous standard `qwen38` service; 20 model artifacts (92.39 GiB) and the `sha256:2dac17a2` runtime bundle verified (size + sha256).
- Setup host doctor PASS=27 / FAIL=0; first-start qualification passed (130,941-token prompt); runtime doctor PASS=33 / FAIL=0.
- CED prefill **1.74x** at 14,420 tokens (6.25 s with CED on vs 10.86 s opted out via `r9v_ced: false`).
- MTP acceptance 50.5% (54 drafts / 109 accepted, mean emitted length 3.019).
- Previous production image (`r9v-qwen38-flash-next:latest`, `sha256:36237d4034b0`) retained for rollback.

## 192K context (2026-09-26)
- Model is natively 256K (no RoPE scaling); envelope raised **128K -> 192K** (`R9V_MAX_MODEL_LEN=196608`, KV `3672113152`, runtime `context_tokens=196608`).
- GPU KV cache **137,196 -> 213,995 tokens**; concurrency at full envelope 1.05x -> 1.09x (KV 18,676 B/token per rank).
- Re-qualified at the new envelope: passed, `context_limit=196608`, headroom min free 2.06 / 1.52 GiB.
- **Decode unchanged**: 21.60 -> 21.66 ms/step (fixed prompt, MTP4); CED prefill 1.57-1.74x; tool calling OK.
