# SPDX-License-Identifier: Apache-2.0
import importlib.util
import os
from pathlib import Path

from torch.utils.cpp_extension import load

root = Path(__file__).resolve().parent
build_directory = Path(os.environ.get("QWEN38_DENSE_MMVQ_BUILD_DIR", root / "build"))
build_directory.mkdir(parents=True, exist_ok=True)


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


headers = resolve_plugin_csrc()
load(
    name="qwen38_dense_mmvq_hip",
    sources=[str(root / "dense_mmvq_hip.cu")],
    extra_include_paths=[str(headers), str(headers / "gguf")],
    extra_cflags=["-O3", "-std=c++17"],
    extra_cuda_cflags=[
        "-O3",
        "-std=c++17",
        "-DUSE_ROCM",
        "--offload-arch=gfx1201",
    ],
    build_directory=str(build_directory),
    verbose=True,
)
