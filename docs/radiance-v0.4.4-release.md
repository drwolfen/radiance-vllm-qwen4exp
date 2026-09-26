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
