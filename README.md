# Radiance-vLLM Qwen4exp (Dual AMD Radeon AI PRO R9700)

High-throughput, production-grade **vLLM** inference engine for **`Qwen3.8-Flash-Next`** (`UD-IQ4_XS`, `qwen4exp` SSM + QSA + PLE + 512-MoE) on dual AMD Radeon AI PRO R9700 accelerators (`gfx1201`).

Based on the qualified **`Dyluhn/R9V`** architecture, incorporating Tensor Parallelism (TP=2), custom RDNA4 WMMA kernels, an asymmetric 16-slot LRU dynamic expert cache, and a dedicated **FP8 MTP-2 draft head** achieving **~35.4 tok/s decode** and **~1,727 tok/s prefill**.

---

## Key Architectural Features

- **No `llama.cpp` Dependencies**: 100% native vLLM engine via `vllm-gguf-plugin` (PR #53899 fork).
- **Dedicated FP8 MTP Drafter**: `Qwen3.8-Flash-Next-mtp-drafter` (`mtp/model.safetensors`, 2.69 GB) executed via `qwen38_fused_gdn_mtp_hip.so` with depth=2 speculative decoding.
- **Custom `gfx1201` HIP Kernels**:
  - `qwen38_dense_mmvq_hip.so`: Fast MMVQ dense attention projection.
  - `qwen38_tiered_iq_moe_hip.so`: Tiered IQ MoE kernel with dynamic LRU caching on Rank 1.
  - `qwen38_fused_gdn_mtp_hip.so`: Fused GDN linear recurrent state update & MTP verification.
- **SSD-Backed PLE Table**: 28.80 GB extracted packed IQ4_NL Prompt-Level Embedding (`per_layer_token_embd.iq4_nl.bin`).
- **Complete Packaging**: Standalone repository containing full runtime source, kernel builds, profile catalogs, and docker compose orchestration.

---

## Hardware Requirements

| Component | Specification |
|---|---|
| **GPUs** | 2× AMD Radeon AI PRO R9700 (32 GiB each, `gfx1201`) |
| **Driver / ROCm** | ROCm 7.14.0+ host driver, `/dev/kfd` and `/dev/dri` |
| **Host Subsystem** | Dual Intel Xeon / AMD EPYC, $\ge 128\text{ GiB}$ DDR4/DDR5 |
| **Interconnect** | PCIe 3.0 x16 or PCIe 4.0/5.0 with P2P DMA |
| **Fast Storage** | NVMe SSD with $\ge 150\text{ GiB}$ free space |

---

## Quick Start

### 1. Preflight Verification
```bash
make doctor
```

### 2. Prepare PLE Embedding Table
```bash
make ple
```

### 3. Build Runtime Image
```bash
make build
```

### 4. Launch Staging Service (Port 8085)
```bash
make run-staging
```

Verify service readiness:
```bash
curl http://127.0.0.1:8085/v1/models
```

---

## Environment Configuration

Key environment variables (managed in `profiles/qwen38-flash-next/dual-r9700/profile.env`):

| Variable | Description | Production Default |
|---|---|---|
| `R9V_VISIBLE_DEVICES` | TP rank GPU index order | `0,1` |
| `R9V_MTP_SPEC_TOKENS` | Speculative draft token count | `2` |
| `R9V_MTP_QUANTIZATION` | MTP draft head precision | `fp8` |
| `R9V_TIERED_EXPERT_CACHE_SLOTS` | LRU dynamic expert cache slots | `16` |
| `R9V_TIERED_EXPERT_CACHE_RANKS` | Ranks hosting dynamic expert cache | `1` |
| `R9V_PLE_RESIDENCY_MODE` | PLE embedding table storage mode | `ssd` |
| `R9V_HOST_PORT` | Exposed OpenAI-compatible API port | `8085` (staging) / `8080` (prod) |

---

## License & Attribution

- **Engine & Kernels:** Apache-2.0 License.
- **Model Weights & Draft Head:** Qwen Community License 1.0.
- Based on the [R9V Project](https://github.com/Dyluhn/R9V) by Dyluhn.
