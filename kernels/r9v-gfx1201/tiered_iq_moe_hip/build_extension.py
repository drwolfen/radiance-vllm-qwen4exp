# SPDX-License-Identifier: Apache-2.0
import argparse
import importlib.util
import os
from pathlib import Path

from torch.utils.cpp_extension import load

root = Path(__file__).resolve().parent


def resolve_plugin_csrc() -> Path:
    configured = os.environ.get("VLLM_GGUF_PLUGIN_CSRC")
    if configured:
        candidate = Path(configured).expanduser().resolve()
    else:
        spec = importlib.util.find_spec("vllm_gguf_plugin")
        locations = () if spec is None else spec.submodule_search_locations or ()
        if not locations:
            raise RuntimeError(
                "vllm_gguf_plugin is not importable; set VLLM_GGUF_PLUGIN_CSRC "
                "to its csrc directory"
            )
        candidate = Path(next(iter(locations))) / "csrc"
    required = candidate / "gguf" / "ggml-common_hip.h"
    if not required.is_file():
        raise RuntimeError(f"missing GGUF HIP headers under {candidate}")
    return candidate


plugin_headers = resolve_plugin_csrc()

VARIANT_BITS = {
    "auto": 1,
    "u2": 2,
    "u5": 4,
    "u10": 8,
    "reuse3": 16,
    "reuse3v2": 32,
}


def parse_variants(value: str) -> int:
    names = [name.strip().lower() for name in value.split(",") if name.strip()]
    unknown = sorted(set(names) - VARIANT_BITS.keys())
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown variants {', '.join(unknown)}; choose from "
            + ", ".join(VARIANT_BITS)
        )
    mask = 0
    for name in names:
        mask |= VARIANT_BITS[name]
    return mask


parser = argparse.ArgumentParser(
    description="Build the mixed VRAM/UVA Q4 expert GEMV extension"
)
parser.add_argument(
    "--variants",
    type=parse_variants,
    default=os.environ.get(
        "QWEN38_TIERED_IQ_MOE_BUILD_VARIANTS",
        "auto,u2,u5,u10,reuse3,reuse3v2",
    ),
    help=(
        "comma-separated exact-shape variants to compile "
        "(auto,u2,u5,u10,reuse3,reuse3v2)"
    ),
)
parser.add_argument(
    "--build-directory",
    type=Path,
    default=root / "build",
    help="artifact directory (use a candidate directory before promotion)",
)
args = parser.parse_args()
variant_mask = args.variants
build_directory = args.build_directory.resolve()

build_directory.mkdir(parents=True, exist_ok=True)
load(
    name="qwen38_tiered_iq_moe_hip",
    sources=[str(root / "tiered_iq_moe_hip.cu")],
    extra_include_paths=[str(plugin_headers), str(plugin_headers / "gguf")],
    extra_cflags=["-O3", "-std=c++17"],
    extra_cuda_cflags=[
        "-O3",
        "-std=c++17",
        "-DUSE_ROCM",
        f"-DQWEN38_TIERED_IQ_MOE_SPECIALIZED_VARIANTS={variant_mask}",
    ],
    build_directory=str(build_directory),
    verbose=True,
)
