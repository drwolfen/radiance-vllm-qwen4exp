# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path

from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parent
BUILD_DIRECTORY = Path(
    os.environ.get("QWEN38_GDN_BUILD_DIR", ROOT / "build")
).resolve()
BUILD_DIRECTORY.mkdir(parents=True, exist_ok=True)

load(
    name="qwen38_fused_gdn_mtp_hip",
    sources=[str(ROOT / "fused_gdn_mtp_hip.cu")],
    build_directory=str(BUILD_DIRECTORY),
    extra_cuda_cflags=["-O3", "-ffast-math", "--offload-arch=gfx1201"],
    verbose=True,
)

print(BUILD_DIRECTORY / "qwen38_fused_gdn_mtp_hip.so")
