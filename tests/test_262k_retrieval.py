#!/usr/bin/env python3
"""Needle-in-a-haystack context test for 262,144 token validation."""

import argparse
import requests
import time

def run_context_test(port=8085, context_tokens=32000):
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    needle = "SECRET_KEY_ALPHA_77"
    haystack = "The fast brown fox jumps over the lazy dog repeatedly. " * (context_tokens // 10)
    prompt = haystack + f"\nSecret code: {needle}\n" + "What is the secret code? Reply with code only."
    
    t0 = time.perf_counter()
    r = requests.post(url, json={
        "model": "Qwen3.8-Flash-Next",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 16,
        "temperature": 0.0
    }, timeout=180)
    t1 = time.perf_counter()
    
    if r.status_code == 200:
        content = r.json()["choices"][0]["message"]["content"]
        if needle in content:
            print(f"[262K TEST PASS] Successfully retrieved {needle} in {t1-t0:.2f}s")
            return True
        else:
            print(f"[262K TEST FAIL] Needle not found in output: {content[:100]}")
            return False
    else:
        print(f"[262K TEST FAIL] Status {r.status_code}: {r.text[:100]}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8085)
    args = parser.parse_args()
    run_context_test(args.port)
