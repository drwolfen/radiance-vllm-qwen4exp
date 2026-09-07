#!/usr/bin/env python3
"""Throughput and latency benchmark for vllm-qwen4exp staging instance."""

import argparse
import time
import requests

def run_bench(port=8085, tokens=128):
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = {
        "model": "Qwen3.8-Flash-Next",
        "messages": [{"role": "user", "content": "Explain sparse mixture of experts and hybrid SSM."}],
        "max_tokens": tokens,
        "temperature": 0.7
    }
    t0 = time.perf_counter()
    r = requests.post(url, json=payload, timeout=120)
    t1 = time.perf_counter()
    if r.status_code == 200:
        data = r.json()
        c_toks = data["usage"]["completion_tokens"]
        elapsed = t1 - t0
        print(f"[BENCHMARK PASS] {c_toks} tokens in {elapsed:.2f}s -> {c_toks/elapsed:.2f} tok/s")
    else:
        print(f"[BENCHMARK FAIL] Status {r.status_code}: {r.text[:100]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8085)
    parser.add_argument("--tokens", type=int, default=128)
    args = parser.parse_args()
    run_bench(args.port, args.tokens)
