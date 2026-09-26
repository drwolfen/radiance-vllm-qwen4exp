#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bounded protocol checks with informational text/tool/vision observations."""

import argparse
import base64
import hashlib
import json
import os
import struct
import threading
import time
import urllib.request
import zlib
from pathlib import Path


def call(url, endpoint, body, timeout=1200):
    request = urllib.request.Request(
        url.rstrip("/") + endpoint,
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read(16 * 1024 * 1024 + 1)
        if len(data) > 16 * 1024 * 1024:
            raise ValueError("response exceeded 16 MiB")
        return json.loads(data)


def red_image(width=512, height=512):
    def chunk(kind, data):
        return (
            struct.pack("!I", len(data))
            + kind
            + data
            + struct.pack("!I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress((b"\0" + b"\xff\0\0" * width) * height))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode()


def _run_workload(url, model, output, context, request_timeout=1200):
    results = {}
    runtime_failures = []
    semantic_observations = []

    def ask(name, content, **extra):
        body = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": 128,
            "chat_template_kwargs": {"enable_thinking": False},
            **extra,
        }
        started = time.monotonic()
        request_evidence = {
            "timeout_seconds": request_timeout,
            "started_at": time.time(),
            "max_tokens": body["max_tokens"],
            "request_sha256": hashlib.sha256(
                json.dumps(body, sort_keys=True).encode()
            ).hexdigest(),
        }
        pending = output / (name + "-request.json")
        pending.write_text(json.dumps(request_evidence, indent=2))
        try:
            response = call(url, "/v1/chat/completions", body, timeout=request_timeout)
        except Exception as error:
            request_evidence.update(
                error=str(error), seconds=time.monotonic() - started
            )
            pending.write_text(json.dumps(request_evidence, indent=2))
            raise
        result = {"seconds": time.monotonic() - started, "response": response}
        (output / (name + ".json")).write_text(json.dumps(result, indent=2))
        choice = response["choices"][0]
        if choice.get("finish_reason") not in ("stop", "length", "tool_calls"):
            raise ValueError(f"{name}: incomplete response")
        results[name] = result
        return choice["message"]

    if (
        ask("text", "Compute 17 + 25. Reply with only the integer result.")
        .get("content", "")
        .strip()
        != "42"
    ):
        semantic_observations.append("text arithmetic mismatch")
    tool = {
        "type": "function",
        "function": {
            "name": "record_code",
            "description": "Record the requested code.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
    }
    message = ask(
        "tool",
        "Use the record_code tool to record exactly R9V-731. Do not answer in prose.",
        tools=[tool],
        tool_choice="auto",
        max_tokens=256,
    )
    calls = message.get("tool_calls") or []
    try:
        tool_ok = (
            len(calls) == 1
            and calls[0]["function"]["name"] == "record_code"
            and json.loads(calls[0]["function"]["arguments"]) == {"code": "R9V-731"}
        )
    except (KeyError, TypeError, ValueError):
        tool_ok = False
    if not tool_ok:
        # A malformed tool call is a protocol failure; a well-formed call with
        # an unexpected code remains an answer-quality observation.
        try:
            function = calls[0]["function"]
            arguments = json.loads(function["arguments"])
            schema_ok = (
                len(calls) == 1
                and function["name"] == "record_code"
                and isinstance(arguments, dict)
                and isinstance(arguments.get("code"), str)
            )
        except (KeyError, TypeError, ValueError, IndexError):
            schema_ok = False
        (runtime_failures if not schema_ok else semantic_observations).append(
            "tool call schema failed" if not schema_ok else "tool code mismatch"
        )
    for name, width, height in [
        ("vision", 512, 512),
        ("vision-wide", 1024, 256),
        ("vision-tall", 256, 1024),
    ]:
        message = ask(
            name,
            [
                {
                    "type": "text",
                    "text": "What single color fills this image? Reply with only the color name.",
                },
                {"type": "image_url", "image_url": {"url": red_image(width, height)}},
            ],
        )
        if message.get("content", "").strip().lower().rstrip(".") != "red":
            semantic_observations.append(f"{name} color mismatch")
    if context:

        def prompt(n):
            return (
                "Remember the code R9V-731. "
                + "The following passage describes trees and their leaves. " * n
                + "What code appeared at the beginning? Reply with only that code."
            )

        low, high = 1, context
        while low < high:
            middle = (low + high + 1) // 2
            count = call(
                url,
                "/tokenize",
                {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt(middle)}],
                    "add_generation_prompt": True,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )["count"]
            if count <= context - 128:
                low = middle
            else:
                high = middle - 1
        message = ask("context", prompt(low))
        actual = results["context"]["response"]["usage"]["prompt_tokens"]
        if not context - 160 <= actual <= context - 128:
            runtime_failures.append(
                f"Context probe did not reach advertised envelope: {actual}/{context}"
            )
        if "R9V-731" not in message.get("content", ""):
            semantic_observations.append("long-context retrieval mismatch")
    time.sleep(10)
    if (
        ask("idle_resume", "Compute 17 + 25. Reply with only the integer result.")
        .get("content", "")
        .strip()
        != "42"
    ):
        semantic_observations.append("idle-resume arithmetic mismatch")
    (output / "result.json").write_text(
        json.dumps(
            {
                "passed": not runtime_failures,
                "context_limit": context,
                "checks": list(results),
                "semantic_observations": semantic_observations,
                "runtime_failures": runtime_failures,
            },
            indent=2,
        )
    )
    if runtime_failures:
        raise ValueError("; ".join(runtime_failures))


def run(url, model, output, context, request_timeout=1200):
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    bdfs = [v for v in os.environ.get("R9V_EXPECTED_GPU_BDFS", "").split(",") if v]
    minimum = [2**64] * len(bdfs)
    errors = []
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            try:
                for rank, bdf in enumerate(bdfs):
                    base = Path("/sys/bus/pci/devices") / bdf
                    free = int((base / "mem_info_vram_total").read_text()) - int(
                        (base / "mem_info_vram_used").read_text()
                    )
                    minimum[rank] = min(minimum[rank], free)
            except (OSError, ValueError) as error:
                errors.append(str(error))
                return
            stop.wait(0.5)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    failure = None
    try:
        _run_workload(url, model, output, context, request_timeout)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        failure = error
    finally:
        stop.set()
        thread.join(timeout=2)
    result_path = output / "result.json"
    result = (
        json.loads(result_path.read_text())
        if result_path.exists()
        else {
            "passed": False,
            "context_limit": context,
            "error": str(failure),
            "response_files": sorted(
                p.name
                for p in output.glob("*.json")
                if not p.name.endswith("-request.json")
            ),
        }
    )
    if failure is not None:
        result.update(passed=False, error=str(failure))
    result["minimum_physical_free_bytes"] = minimum
    if errors:
        result.update(passed=False, telemetry_errors=errors)
    if bdfs:
        try:
            from tools.expert_budget import headroom_bytes
        except ModuleNotFoundError:
            from expert_budget import headroom_bytes
        targets = headroom_bytes(
            os.environ.get("R9V_MIN_FREE_VRAM_GIB_BY_RANK", "3,3"), len(bdfs)
        )
        result["headroom_passed"] = all(
            free >= target for free, target in zip(minimum, targets)
        )
        if (
            os.environ.get("R9V_QUALIFY_HEADROOM", "1") == "1"
            and not result["headroom_passed"]
        ):
            result["passed"] = False
    result_path.write_text(json.dumps(result, indent=2))
    if failure is not None:
        raise failure
    if not result["passed"]:
        raise ValueError(
            "workload memory headroom or telemetry check failed; see result.json"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8004")
    parser.add_argument("--model", default="qwen3.8-flash-next")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--context", type=int, default=131072)
    parser.add_argument("--request-timeout", type=int, default=1200)
    args = parser.parse_args()
    if not 5 <= args.request_timeout <= 1200:
        parser.error("--request-timeout must be 5..1200 seconds")
    if not 0 <= args.context <= 262144:
        parser.error("--context must be 0..262144")
    try:
        run(args.url, args.model, args.output, args.context, args.request_timeout)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Runtime workload failed: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
