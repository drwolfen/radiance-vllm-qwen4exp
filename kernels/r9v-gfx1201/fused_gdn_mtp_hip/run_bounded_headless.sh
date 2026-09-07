#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Isolated parity + graph-replay benchmark for the non-display R9700.  This
# loads no model, exposes no display render node, and the Python test caps the
# PyTorch allocator at 5% before its first allocation.
readonly headless_bdf=${R9V_HEADLESS_BDF:?Set R9V_HEADLESS_BDF, for example 0000:13:00.0}
readonly headless_uuid=${R9V_HEADLESS_UUID:?Set R9V_HEADLESS_UUID to the GPU UUID}
readonly headless_render=${R9V_HEADLESS_RENDER:?Set R9V_HEADLESS_RENDER, for example /dev/dri/renderD129}
readonly headless_card=${R9V_HEADLESS_CARD:?Set R9V_HEADLESS_CARD, for example /dev/dri/card0}
readonly display_bdf=${R9V_DISPLAY_BDF:?Set R9V_DISPLAY_BDF to the boot/display GPU BDF}
readonly image=${R9V_DEV_IMAGE:?Set R9V_DEV_IMAGE to the vLLM ROCm development image}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

read_sysfs() {
    local path=$1
    [[ -r "$path" ]] || {
        printf 'Required sysfs identity is unavailable: %s\n' "$path" >&2
        exit 1
    }
    tr -d '\n' < "$path"
}

[[ "$(read_sysfs "/sys/bus/pci/devices/$headless_bdf/boot_vga")" == 0 ]] || {
    printf 'Refusing: %s is marked as the boot/display GPU.\n' "$headless_bdf" >&2
    exit 1
}
[[ "$(read_sysfs "/sys/bus/pci/devices/$display_bdf/boot_vga")" == 1 ]] || {
    printf 'Refusing: expected display GPU %s is not boot_vga.\n' "$display_bdf" >&2
    exit 1
}
[[ "GPU-$(read_sysfs "/sys/bus/pci/devices/$headless_bdf/unique_id")" == "$headless_uuid" ]] || {
    printf 'Refusing: the headless GPU UUID/BDF mapping changed.\n' >&2
    exit 1
}
[[ -e "$headless_render" && -e "$headless_card" ]] || {
    printf 'Refusing: headless DRM nodes are unavailable.\n' >&2
    exit 1
}
[[ -f "$script_dir/build/qwen38_fused_gdn_mtp_hip.so" ]] || {
    printf 'Fused GDN extension is missing; build it before testing.\n' >&2
    exit 1
}

if docker ps --format '{{.Names}}' | grep -Eq '^qwen38'; then
    printf 'Refusing while a Qwen3.8 container is already running.\n' >&2
    exit 1
fi

render_gid=$(getent group render | cut -d: -f3)
video_gid=$(getent group video | cut -d: -f3)

# ROCm/KFD needs the DRM directory for topology discovery on this host.
# ROCR_VISIBLE_DEVICES below still exposes only the verified headless UUID to
# HIP, matching the already-validated smoke_headless_gpu.sh pattern.
exec docker run --rm \
    --network=none \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add="$render_gid" \
    --group-add="$video_gid" \
    --security-opt=seccomp=unconfined \
    --security-opt=label=disable \
    --env=ROCR_VISIBLE_DEVICES="$headless_uuid" \
    --env=RADIANCE_USE_R4D=0 \
    --env=RADIANCE_USE_R4D_GDN=0 \
    --volume="$script_dir:/work:ro" \
    --entrypoint=/opt/vllm/bin/python \
    "$image" \
    /work/test_and_bench.py \
    --tokens=3 \
    --iterations=200 \
    --graph-replays=32 \
    --compare-fla
