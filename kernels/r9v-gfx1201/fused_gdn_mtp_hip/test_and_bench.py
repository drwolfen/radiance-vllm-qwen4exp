# SPDX-License-Identifier: Apache-2.0
import argparse
import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
SO = ROOT / "build" / "qwen38_fused_gdn_mtp_hip.so"
SCALE = 128**-0.5
EPS = 1e-6


def load_extension():
    spec = importlib.util.spec_from_file_location("qwen38_fused_gdn_mtp_hip", SO)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SO}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_one(
    mixed_qkv,
    a,
    b,
    a_log,
    dt_bias,
    state_indices,
    accepted,
    state,
    output_gate,
    norm_weight,
    sigmoid_gate,
    core_only=False,
):
    q, k, v = torch.split(mixed_qkv.float(), [8 * 128, 8 * 128, 24 * 128], -1)
    q = F.normalize(q.view(-1, 8, 128), dim=-1, eps=EPS) * SCALE
    k = F.normalize(k.view(-1, 8, 128), dim=-1, eps=EPS)
    v = v.view(-1, 24, 128)
    outputs = []
    source = int(state_indices[accepted - 1])
    work = state[source].float().clone()
    key_heads = torch.arange(24, device=mixed_qkv.device) // 3
    for token in range(mixed_qkv.size(0)):
        decay = torch.exp(
            -torch.exp(a_log) * F.softplus(a[token].float() + dt_bias.float())
        )
        beta = torch.sigmoid(b[token].float())
        kt = k[token, key_heads]
        qt = q[token, key_heads]
        work.mul_(decay[:, None, None])
        residual = v[token] - torch.einsum("hvk,hk->hv", work, kt)
        work.add_(kt[:, None, :] * (residual * beta[:, None])[:, :, None])
        outputs.append(torch.einsum("hvk,hk->hv", work, qt))
        destination = int(state_indices[token])
        if 0 < destination < state.size(0):
            state[destination].copy_(work)
    raw = torch.stack(outputs).bfloat16().float()
    if core_only:
        return raw.bfloat16()
    rstd = torch.rsqrt(raw.square().mean(-1, keepdim=True) + EPS)
    gate = (
        torch.sigmoid(output_gate.float())
        if sigmoid_gate
        else F.silu(output_gate.float())
    )
    return (raw * rstd * norm_weight.float() * gate).bfloat16()


def reference_batch(case, sigmoid_gate, *, core_only=False):
    output = torch.empty_like(case[11])
    state = case[8].clone()
    for request in range(case[5].size(0)):
        bos = int(case[6][request])
        eos = int(case[6][request + 1])
        output[bos:eos] = reference_one(
            case[0][bos:eos],
            case[1][bos:eos],
            case[2][bos:eos],
            case[3],
            case[4],
            case[5][request],
            int(case[7][request]),
            state,
            case[9][bos:eos],
            case[10],
            sigmoid_gate,
            core_only,
        )
    return output, state


def make_case(
    tokens=2,
    slots=12,
    *,
    dt_dtype=torch.float32,
    norm_dtype=torch.bfloat16,
):
    torch.manual_seed(7)
    device = "cuda"
    mixed = torch.randn(tokens, 5120, device=device, dtype=torch.bfloat16)
    ba = torch.randn(tokens, 48, device=device, dtype=torch.bfloat16)
    b, a = ba.chunk(2, -1)
    a_log = torch.randn(24, device=device, dtype=torch.float32) * 0.2
    dt_bias = (torch.randn(24, device=device) * 0.1).to(dt_dtype)
    state = torch.randn(slots, 24, 128, 128, device=device) * 0.01
    state_indices = torch.arange(2, 2 + tokens, device=device, dtype=torch.int32)[None]
    cu_seqlens = torch.tensor([0, tokens], device=device, dtype=torch.int32)
    num_accepted = torch.tensor([1], device=device, dtype=torch.int32)
    gate = torch.randn(tokens, 24, 128, device=device, dtype=torch.bfloat16)
    norm = torch.randn(128, device=device, dtype=norm_dtype)
    out = torch.empty_like(gate)
    return (
        mixed,
        a,
        b,
        a_log,
        dt_bias,
        state_indices,
        cu_seqlens,
        num_accepted,
        state,
        gate,
        norm,
        out,
    )


def run_extension(ext, case, sigmoid_gate=False):
    return ext.fused_gdn_mtp_tp2_fp32(*case, SCALE, EPS, sigmoid_gate)


def run_core_extension(ext, case):
    return ext.fused_gdn_mtp_tp2_fp32_core(*case[:9], case[11], SCALE)


def _state_error_metrics(actual, expected, state_indices, cu_seqlens):
    """Measure only slots this invocation is allowed to mutate.

    A whole-pool norm can hide a bad speculative state write because a typical
    pool contains many untouched slots.  It is also important that a fused
    kernel not scribble on any of those untouched slots.
    """
    destination_parts = []
    for request in range(state_indices.size(0)):
        tokens = int(cu_seqlens[request + 1] - cu_seqlens[request])
        if tokens > 0:
            destination_parts.append(state_indices[request, :tokens])
    if destination_parts:
        destinations = torch.cat(destination_parts)
        destinations = torch.unique(
            destinations[(destinations > 0) & (destinations < actual.size(0))]
        ).long()
    else:
        destinations = torch.empty(0, dtype=torch.long, device=actual.device)

    if destinations.numel() == 0:
        state_rel_l2 = torch.zeros((), dtype=torch.float32, device=actual.device)
    else:
        actual_changed = actual.index_select(0, destinations)
        expected_changed = expected.index_select(0, destinations)
        state_rel_l2 = (actual_changed - expected_changed).norm() / (
            expected_changed.norm().clamp_min(1e-20)
        )

    untouched = torch.ones(actual.size(0), dtype=torch.bool, device=actual.device)
    untouched[destinations] = False
    untouched_equal = torch.equal(actual[untouched], expected[untouched])
    return state_rel_l2, untouched_equal


def assert_parity(ext, case, *, sigmoid_gate=False, label="case"):
    expected_out, expected_state = reference_batch(case, sigmoid_gate)
    actual = run_extension(ext, case, sigmoid_gate)
    torch.cuda.synchronize()
    output_rel_l2 = (
        actual.float() - expected_out.float()
    ).norm() / expected_out.float().norm().clamp_min(1e-20)
    state_rel_l2, untouched_equal = _state_error_metrics(
        case[8], expected_state, case[5], case[6]
    )
    print(
        f"{label}: output_rel_l2={output_rel_l2.item():.8g} "
        f"changed_state_rel_l2={state_rel_l2.item():.8g} "
        f"untouched_state_exact={untouched_equal}"
    )
    if output_rel_l2 >= 7e-4 or state_rel_l2 >= 5e-4 or not untouched_equal:
        raise SystemExit(f"parity failed: {label}")


def assert_core_parity(ext, case, *, label="case"):
    expected_out, expected_state = reference_batch(case, False, core_only=True)
    actual = run_core_extension(ext, case)
    torch.cuda.synchronize()
    output_rel_l2 = (
        actual.float() - expected_out.float()
    ).norm() / expected_out.float().norm().clamp_min(1e-20)
    state_rel_l2, untouched_equal = _state_error_metrics(
        case[8], expected_state, case[5], case[6]
    )
    print(
        f"core-{label}: output_rel_l2={output_rel_l2.item():.8g} "
        f"changed_state_rel_l2={state_rel_l2.item():.8g} "
        f"untouched_state_exact={untouched_equal}"
    )
    if output_rel_l2 >= 7e-4 or state_rel_l2 >= 5e-4 or not untouched_equal:
        raise SystemExit(f"core parity failed: {label}")


def test_eager_matrix(ext):
    for tokens in (1, 2, 3):
        for accepted in range(1, tokens + 1):
            case = list(make_case(tokens))
            case[7].fill_(accepted)
            assert_parity(
                ext, tuple(case), label=f"tokens={tokens},accepted={accepted}"
            )

    for dt_dtype, norm_dtype, sigmoid_gate in (
        (torch.bfloat16, torch.float32, False),
        (torch.float16, torch.bfloat16, True),
    ):
        case = make_case(3, dt_dtype=dt_dtype, norm_dtype=norm_dtype)
        assert_parity(
            ext,
            case,
            sigmoid_gate=sigmoid_gate,
            label=f"dt={dt_dtype},norm={norm_dtype},sigmoid={sigmoid_gate}",
        )

    case = list(make_case(3))
    case[5] = torch.tensor([[2, 3], [4, 5]], device="cuda", dtype=torch.int32)
    case[6] = torch.tensor([0, 1, 3], device="cuda", dtype=torch.int32)
    case[7] = torch.tensor([1, 1], device="cuda", dtype=torch.int32)
    assert_parity(ext, tuple(case), label="ragged-two-request")


def test_core_eager_matrix(ext):
    for tokens in (1, 2, 3):
        for accepted in range(1, tokens + 1):
            case = list(make_case(tokens))
            case[7].fill_(accepted)
            assert_core_parity(
                ext, tuple(case), label=f"tokens={tokens},accepted={accepted}"
            )

    case = list(make_case(3))
    case[5] = torch.tensor([[2, 3], [4, 5]], device="cuda", dtype=torch.int32)
    case[6] = torch.tensor([0, 1, 3], device="cuda", dtype=torch.int32)
    case[7] = torch.tensor([1, 1], device="cuda", dtype=torch.int32)
    assert_core_parity(ext, tuple(case), label="ragged-two-request")


def test_device_guards(ext):
    case = list(make_case(3))
    initial_state = case[8].clone()
    case[5][0, 0] = case[8].size(0) + 5
    case[11].fill_(17)
    run_extension(ext, tuple(case))
    torch.cuda.synchronize()
    if torch.count_nonzero(case[11]).item() != 0:
        raise SystemExit("invalid source slot did not zero output")
    if not torch.equal(case[8], initial_state):
        raise SystemExit("invalid source slot mutated state")

    case = list(make_case(3))
    case[5][0, 1] = case[8].size(0) + 5
    assert_parity(ext, tuple(case), label="invalid-destination-slot")

    case = list(make_case(3))
    initial_state = case[8].clone()
    case[6][0] = -1
    case[11].fill_(17)
    run_extension(ext, tuple(case))
    torch.cuda.synchronize()
    if not torch.all(case[11] == 17):
        raise SystemExit("invalid sequence bounds mutated output")
    if not torch.equal(case[8], initial_state):
        raise SystemExit("invalid sequence bounds mutated state")
    print("device metadata guards: passed")


def test_graph_replay(ext, iterations, *, core_only=False):
    case = list(make_case(3))
    initial_state = case[8].clone()
    graph = torch.cuda.CUDAGraph()
    run = run_core_extension if core_only else run_extension
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        run(ext, tuple(case))
    for index in range(iterations):
        accepted = index % 3 + 1
        case[8].copy_(initial_state)
        case[7].fill_(accepted)
        graph.replay()
        expected_case = list(case)
        expected_case[8] = initial_state.clone()
        expected_out, expected_state = reference_batch(
            tuple(expected_case), False, core_only=core_only
        )
        torch.cuda.synchronize()
        output_rel_l2 = (
            case[11].float() - expected_out.float()
        ).norm() / expected_out.float().norm().clamp_min(1e-20)
        state_rel_l2, untouched_equal = _state_error_metrics(
            case[8], expected_state, case[5], case[6]
        )
        if (
            output_rel_l2 >= 7e-4
            or state_rel_l2 >= 5e-4
            or not untouched_equal
        ):
            raise SystemExit(
                f"graph mismatch at replay {index}, accepted={accepted}: "
                f"output_rel_l2={output_rel_l2.item():.8g}, "
                f"changed_state_rel_l2={state_rel_l2.item():.8g}, "
                f"untouched_state_exact={untouched_equal}"
            )
    kind = "core" if core_only else "fused-norm"
    print(f"{kind} graph replay canary: passed {iterations} alternating replays")


def _stock_fla_core(case):
    """The current vLLM post-convolution fallback, including its QKV copy."""
    from vllm.third_party.flash_linear_attention.ops import (
        fused_sigmoid_gating_delta_rule_update,
    )

    mixed_qkv = case[0]
    query, key, value = torch.split(mixed_qkv, [8 * 128, 8 * 128, 24 * 128], -1)
    packed = torch.cat(
        [query.reshape(-1), key.reshape(-1), value.reshape(-1)], dim=0
    )
    q_size = mixed_qkv.size(0) * 8 * 128
    k_size = q_size
    query = packed[:q_size].view(1, mixed_qkv.size(0), 8, 128)
    key = packed[q_size : q_size + k_size].view(1, mixed_qkv.size(0), 8, 128)
    value = packed[q_size + k_size :].view(1, mixed_qkv.size(0), 24, 128)
    return fused_sigmoid_gating_delta_rule_update(
        A_log=case[3],
        a=case[1],
        b=case[2],
        dt_bias=case[4],
        q=query,
        k=key,
        v=value,
        initial_state=case[8],
        inplace_final_state=True,
        cu_seqlens=case[6],
        ssm_state_indices=case[5],
        num_accepted_tokens=case[7],
        scale=SCALE,
        use_qk_l2norm_in_kernel=True,
    )[0]


def test_stock_fla_parity(ext):
    """Directly compare the production M=3 core against vLLM's FLA path."""
    for accepted in (1, 2, 3):
        stock_case = list(make_case(3))
        fused_case = list(make_case(3))
        stock_case[7].fill_(accepted)
        fused_case[7].fill_(accepted)
        stock_out = _stock_fla_core(tuple(stock_case))
        fused_out = run_core_extension(ext, tuple(fused_case))
        torch.cuda.synchronize()
        output_rel_l2 = (fused_out.float() - stock_out.float()).norm() / (
            stock_out.float().norm().clamp_min(1e-20)
        )
        state_rel_l2, untouched_equal = _state_error_metrics(
            fused_case[8], stock_case[8], fused_case[5], fused_case[6]
        )
        print(
            f"core-vs-fla-M3-accepted={accepted}: "
            f"output_rel_l2={output_rel_l2.item():.8g} "
            f"changed_state_rel_l2={state_rel_l2.item():.8g} "
            f"untouched_state_exact={untouched_equal}"
        )
        if output_rel_l2 >= 7e-4 or state_rel_l2 >= 5e-4 or not untouched_equal:
            raise SystemExit(f"core-vs-FLA parity failed at accepted={accepted}")


def _graph_benchmark(run, iterations):
    # Warm Triton/extension dispatch before capture.  Keeping the returned
    # tensors alive also keeps graph-pool allocations valid for every replay.
    hold = [run()]
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        hold[0] = run()
    for _ in range(20):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def benchmark_against_stock_fla(ext, iterations):
    fused_case = make_case(3)
    stock_case = make_case(3)
    fused_ms = _graph_benchmark(
        lambda: run_core_extension(ext, fused_case), iterations
    )
    stock_ms = _graph_benchmark(lambda: _stock_fla_core(stock_case), iterations)
    layers = 36
    delta_ms = (stock_ms - fused_ms) * layers
    print(
        f"graph_core_M3_hip_ms={fused_ms:.6f} "
        f"graph_core_M3_stock_fla_ms={stock_ms:.6f}"
    )
    print(
        f"projected_36_gdn_layers_hip_ms={fused_ms * layers:.6f} "
        f"projected_36_gdn_layers_stock_fla_ms={stock_ms * layers:.6f} "
        f"projected_cycle_delta_ms={delta_ms:.6f}"
    )


def benchmark(ext, tokens, iterations, *, core_only=False):
    case = make_case(tokens)
    run = run_core_extension if core_only else run_extension
    for _ in range(20):
        run(ext, case)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        run(ext, case)
    end.record()
    end.synchronize()
    elapsed_ms = start.elapsed_time(end) / iterations
    kind = "core" if core_only else "fused-norm"
    print(f"{kind}_kernel_ms={elapsed_ms:.6f} tokens={tokens} heads=24")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--tokens", type=int, default=3)
    parser.add_argument("--graph-replays", type=int, default=32)
    parser.add_argument(
        "--compare-fla",
        action="store_true",
        help="compare M=3 parity and graph time to vLLM's stock FLA core",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("headless isolation failed: expected exactly one ROCm GPU")
    if torch.cuda.get_device_name(0) != "AMD Radeon AI PRO R9700":
        raise SystemExit("headless isolation failed: unexpected GPU model")
    torch.cuda.set_per_process_memory_fraction(0.05)
    ext = load_extension()
    test_eager_matrix(ext)
    test_core_eager_matrix(ext)
    test_device_guards(ext)
    test_graph_replay(ext, args.graph_replays)
    test_graph_replay(ext, args.graph_replays, core_only=True)
    if args.compare_fla:
        test_stock_fla_parity(ext)
        benchmark_against_stock_fla(ext, args.iterations)
    benchmark(ext, args.tokens, args.iterations)
    benchmark(ext, args.tokens, args.iterations, core_only=True)


if __name__ == "__main__":
    main()
