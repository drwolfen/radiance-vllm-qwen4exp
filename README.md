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

## 5. License
Apache 2.0. Copyright 2026 drwolfen, radiance-vllm-qwen4exp contributors.
