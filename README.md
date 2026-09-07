# radiance-vllm-qwen4exp

Dedicated, high-performance vLLM engine optimized for **`Qwen3.8-Flash-Next`** (`qwen4exp` architecture) on dual **AMD Radeon AI PRO R9700** (`gfx1201` / RDNA4) GPUs with Tiered PCIe 3.0 x16 Host-Device MoE offloading.

---

## 1. Architectural Overview & Strict Isolation

This repository is completely independent of [`radiance-vllm-r9700`](https://github.com/drwolfen/radiance-vllm-r9700) (which serves parked `Ornith-1.5-35B-A3B-FP8` on port 8000).

- **Target Architecture**: `qwen4exp`
  - 48 layers: 36 SSM linear attention (Mamba conv1d) + 12 full attention (`full_attention_interval = 4`).
  - 512 experts total, 10 active experts routed per token.
  - Native 262,144 context window.
  - Prompt-Level Embedding (PLE) 3-gram lookup table.
- **Hardware Profile**:
  - 2× AMD Radeon AI PRO R9700 (31.86 GiB usable VRAM each = 63.7 GiB total, 1,728 GB/s).
  - Dual Intel Xeon Platinum 8160 (96 threads, 125 GiB DDR4-2666 ECC Reg, NUMA interleaved).
  - **PCIe 3.0 x16 STRICT** (~15.75 GB/s bidirectional).
- **Staging Port**: **`8085`** (isolated container bridge; production `llama-server.service` remains on `8080`).

---

## 2. Quickstart & Deployment

### Build Container
```bash
make build
```

### Launch Staging Instance (Port 8085)
```bash
make run
```

### Check Container Health & Logs
```bash
curl -s http://127.0.0.1:8085/health
make logs
```

### Run Verification Test Suite
```bash
make test
make bench
```

---

## 3. Tiered MoE Offload Structure

Because `Qwen3.8-Flash-Next` (87.25 GiB `UD-IQ4_XS`) exceeds dual R9700 VRAM (63.7 GiB):
- **VRAM Resident**: Dense attention, SSM recurrent states, norms, PLE embeddings, and 20 MoE layers (10 on GPU 0, 10 on GPU 1).
- **Host RAM Resident**: 28 MoE layers pinned via `mlock` across Xeon NUMA nodes with double-buffered asynchronous PCIe 3.0 DMA streaming.

---

## 4. Verification Gates for Production Cutover

1. **Gate 1**: Clean ROCm HIP `gfx1201` compilation without LDS overflow.
2. **Gate 2**: Staging launch on port `8085` verified healthy.
3. **Gate 3**: Tool-calling schema validation and 262k needle retrieval passed.
4. **Gate 4**: Decode throughput matches or exceeds production baseline (>= 14.0 tok/s decode, >= 800 tok/s prefill).
5. **Gate 5**: Atomic port cutover to 8080 with user approval.

---

## 5. Direct Learnings Integrated from `radiance-vllm-r9700`

1. **Toolchain & Driver Compatibility**:
   - Pinned **ROCm 7.14.0 + PyTorch 2.12.1 + Triton 3.7.1 + AITER 0.1.20**. Avoids ROCm 10 KFD ABI mismatch (`HSA_STATUS_ERROR_DEVICE_MISMATCH`) on host kernel 7.2.4 and prevents PyTorch 2.13+ ring-buffer hangs.
2. **RDNA4 64 KiB LDS Clamp**:
   - Integrated `patch_unified_attention_lds.py` and `libr4d` attention tiles clamped to 64 KiB LDS per WGP on `gfx1201`.
3. **PCIe P2P One-Shot All-Reduce**:
   - Utilizes direct peer-to-peer one-shot push/reduce (`ar_oneshot_2rank_exact`), bypassing RCCL over PCIe 3.0 to remove 12 ms graph synchronization overhead.
4. **Agentic Tool-Calling & Streaming Templates**:
   - Integrated `patch_from_json_filter.py` and `patch_qwen3_toolparse.py` for standard XML `<tool_call>` extraction and Jinja filter compatibility.
5. **SSM Linear Recurrent State Alignment**:
   - Configured recurrent state alignment (`--mamba-cache-mode align`) to ensure bit-identical Automatic Prefix Caching (APC) across multi-turn sessions.

---

## 6. Public Community Research & Empirical Findings (Reddit / Hugging Face)

1. **MoE Expert Dilation / Thrashing (Reddit LocalLLaMA)**:
   - Speculative decoding (EAGLE/MTP/n-gram) across hybrid CPU/GPU MoE triggers DDR4 thrashing on host RAM. Verifying $N=5$ tokens forces Xeon to load 40–50 unique experts simultaneously instead of 10, slowing decode by 22%. Pure raw decode is retained as optimal default.
2. **Hybrid SSM/Transformer KV Footprint (Hugging Face)**:
   - Only 12 of 48 layers use quadratic attention (36 SSM layers maintain constant $O(1)$ state). KV cache is only 12.75 KiB/tok aggregate, enabling lossless `q8_0` KV cache at 262k context (+0.71 GiB/card).
3. **Double-Buffered Asynchronous Streaming**:
   - Host RAM offload utilizes pinned memory (`numactl --interleave=all`) with double-buffered asynchronous PCIe DMA copies to hide transfer overhead behind GPU execution.

---

## 7. License
Apache 2.0. Copyright 2026 drwolfen, radiance-vllm-qwen4exp contributors.
