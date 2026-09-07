#!/usr/bin/env python3
"""Tool-calling regression test for vllm-qwen4exp staging instance."""

import argparse
import requests

def run_tool_test(port=8085):
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = {
        "model": "Qwen3.8-Flash-Next",
        "messages": [{"role": "user", "content": "Check the system GPU status."}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "system_info",
                "description": "Get GPU memory usage and temperatures",
                "parameters": {"type": "object", "properties": {"device": {"type": "string"}}}
            }
        }],
        "temperature": 0.1
    }
    r = requests.post(url, json=payload, timeout=60)
    if r.status_code == 200:
        msg = r.json()["choices"][0]["message"]
        tool_calls = msg.get("tool_calls", [])
        if tool_calls and tool_calls[0]["function"]["name"] == "system_info":
            print("[TOOL TEST PASS] Function called correctly:", tool_calls[0]["function"]["name"])
            return True
        else:
            print("[TOOL TEST FAIL] Missing or invalid tool_calls:", msg)
            return False
    else:
        print(f"[TOOL TEST FAIL] Status {r.status_code}: {r.text[:100]}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8085)
    args = parser.parse_args()
    run_tool_test(args.port)
